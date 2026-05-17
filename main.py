import json
import pandas as pd
import argparse
import os
from config import get_api_key, get_column_descriptions
from api_clients import analyze_columns_with_deepseek
from data_processing import apply_column_filtering, restore_dataframe


def lowercase_analysis_results(analysis_results):
    """
    Convert all string values in the analysis results dictionary to lowercase.

    Args:
        analysis_results (dict): Dictionary containing hierarchical, mathematical,
                               and temporal analysis results.

    Returns:
        dict: New dictionary with all string values converted to lowercase.
    """
    lowercased_results = {
        "hierarchical": {},
        "mathematical": {},
        "temporal": {}
    }

    # Process hierarchical analysis
    for col, categories in analysis_results["hierarchical"].items():
        lowercased_results["hierarchical"][col] = {
            k: v.lower() if isinstance(v, str) else v
            for k, v in categories.items()
        }

    # Process mathematical analysis
    for col, operations in analysis_results["mathematical"].items():
        lowercased_results["mathematical"][col] = [
            op.lower() if isinstance(op, str) else op
            for op in operations
        ]

    # Process temporal analysis
    for col, patterns in analysis_results["temporal"].items():
        lowercased_results["temporal"][col] = [
            pattern.lower() if isinstance(pattern, str) else pattern
            for pattern in patterns
        ]

    return lowercased_results


# ---------------------------------------------------------------------------
# Legacy pipeline (original 3-prompt approach)
# ---------------------------------------------------------------------------

def run_legacy_pipeline(args):
    """Run the original 3-prompt LLM analysis pipeline."""
    # Get API key and column descriptions
    API_KEY = get_api_key(args.model)
    COLUMN_DESCRIPTIONS = get_column_descriptions(args.data)

    if not COLUMN_DESCRIPTIONS:
        print(f"No column descriptions found for dataset: {args.data}")
        return

    # Load dataset
    df = pd.read_csv(f"data/{args.data}.csv")

    # Perform the analysis
    analysis_results = analyze_columns_with_deepseek(API_KEY, COLUMN_DESCRIPTIONS, args)

    # Print results
    print("=== Hierarchical Analysis ===")
    print(json.dumps(analysis_results["hierarchical"], indent=2))

    print("\n=== Mathematical Analysis ===")
    print(json.dumps(analysis_results["mathematical"], indent=2))

    print("\n=== Temporal Analysis ===")
    print(json.dumps(analysis_results["temporal"], indent=2))

    # Convert to lowercase
    analysis_results = lowercase_analysis_results(analysis_results)

    # Apply filtering
    df_transformed, backup = apply_column_filtering(
        df,
        analysis_results["hierarchical"],
        analysis_results["mathematical"],
        analysis_results["temporal"],
        args
    )

    # Create results directory if it doesn't exist
    results_dir = f"results/{args.data}"
    os.makedirs(results_dir, exist_ok=True)

    # Save filtered output
    filtered_output_path = os.path.join(results_dir, "FilteredOutput.csv")
    df_transformed.to_csv(filtered_output_path, index=False)
    print(f"Filtered output saved to: {filtered_output_path}")

    # Restore when needed
    df_restored = restore_dataframe(df_transformed, backup, args)
    restored_output_path = os.path.join(results_dir, "Restored.csv")
    df_restored.to_csv(restored_output_path, index=False)
    print(f"Restored output saved to: {restored_output_path}")


# ---------------------------------------------------------------------------
# CR-KG pipeline (new graph-based approach)
# ---------------------------------------------------------------------------

def run_crkg_pipeline(args):
    """Run the Column Relationship Knowledge Graph pipeline.

    Stages:
      1. LLM ensemble  -> candidate graph
      2. Data validation -> validated graph (hallucinations pruned)
      3. Compression plan -> columns to keep / compress + reconstruction map
      4. Apply compression, save results + stats
    """
    from reasoning.llm_ensemble import run_ensemble, call_single_model
    from reasoning.graph_validator import validate_graph
    from reasoning.graph_compressor import generate_compression_plan

    COLUMN_DESCRIPTIONS = get_column_descriptions(args.data)
    if not COLUMN_DESCRIPTIONS:
        print(f"No column descriptions found for dataset: {args.data}")
        return

    # Load dataset
    data_path = f"data/{args.data}.csv"
    if not os.path.exists(data_path):
        print(f"Dataset file not found: {data_path}")
        return
    df = pd.read_csv(data_path)
    print(f"Loaded dataset '{args.data}' with shape {df.shape}")

    # ------------------------------------------------------------------
    # Stage 1: LLM-driven candidate graph construction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 1: LLM-Driven Candidate Graph Construction")
    print("=" * 60)

    # Determine which models to use for the ensemble
    models = _get_ensemble_models(args)

    # Parse --temp_range into a list of floats (if provided)
    temperatures = None
    temp_range_str = getattr(args, "temp_range", None)
    if temp_range_str:
        temperatures = [float(t.strip()) for t in temp_range_str.split(",") if t.strip()]

    if len(models) == 1 and temperatures is None:
        # Single model -- no majority voting, just use the graph directly
        candidate_graph = call_single_model(models[0], COLUMN_DESCRIPTIONS, args)
        if candidate_graph is None:
            print("LLM call failed -- aborting CR-KG pipeline.")
            return
        ensemble_stats = {"n_models": 1, "n_success": 1}
        candidate_result = {
            "nodes": candidate_graph.get("nodes", []),
            "edges": candidate_graph.get("edges", []),
            "stats": ensemble_stats,
        }
    else:
        # Multi-model ensemble with majority voting
        candidate_result = run_ensemble(
            COLUMN_DESCRIPTIONS, models, args, temperatures=temperatures,
        )
        if not candidate_result["edges"]:
            print("Ensemble produced no edges -- aborting CR-KG pipeline.")
            return

    print(f"\nCandidate graph: {len(candidate_result['nodes'])} nodes, "
          f"{len(candidate_result['edges'])} edges")

    # ------------------------------------------------------------------
    # Stage 2: Data-driven validation & pruning
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 2: Data-Driven Validation & Pruning")
    print("=" * 60)

    validation_threshold = getattr(args, "validation_threshold", 0.90)
    validated_result = validate_graph(df, candidate_result, threshold=validation_threshold)

    print(f"\nValidated graph: {len(validated_result['edges'])} edges kept, "
          f"{len(validated_result['removed_edges'])} pruned")

    # ------------------------------------------------------------------
    # Stage 3: Graph-based compression planning
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 3: Graph-Based Compression Planning")
    print("=" * 60)

    compression_plan = generate_compression_plan(validated_result)

    # ------------------------------------------------------------------
    # Stage 4: Apply compression and save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 4: Apply Compression & Save Results")
    print("=" * 60)

    results_dir = f"results/{args.data}"
    os.makedirs(results_dir, exist_ok=True)

    # Apply compression: drop the columns identified for compression
    df_compressed = df.copy()
    cols_to_drop = [
        c for c in compression_plan["columns_to_compress"]
        if c in df_compressed.columns
    ]
    if cols_to_drop:
        df_compressed.drop(columns=cols_to_drop, inplace=True)
        print(f"Dropped {len(cols_to_drop)} columns: {cols_to_drop}")
    else:
        print("No columns to drop (all columns are independent).")

    # Save compressed output
    compressed_path = os.path.join(results_dir, "CRKG_FilteredOutput.csv")
    df_compressed.to_csv(compressed_path, index=False)
    print(f"Compressed output saved to: {compressed_path}")

    # Save the full CR-KG result (graph + validation + compression plan)
    crkg_output = {
        "candidate_graph": {
            "nodes": candidate_result.get("nodes", []),
            "edges": candidate_result.get("edges", []),
            "stats": candidate_result.get("stats", {}),
        },
        "validated_graph": {
            "nodes": validated_result.get("nodes", []),
            "edges": validated_result.get("edges", []),
            "removed_edges": validated_result.get("removed_edges", []),
            "stats": validated_result.get("stats", {}),
        },
        "compression_plan": {
            "columns_to_keep": compression_plan["columns_to_keep"],
            "columns_to_compress": compression_plan["columns_to_compress"],
            "reconstruction_map": compression_plan["reconstruction_map"],
            "stats": compression_plan["stats"],
        },
    }

    crkg_json_path = os.path.join(results_dir, "CRKG_graph.json")
    with open(crkg_json_path, "w") as f:
        json.dump(crkg_output, f, indent=2, default=str)
    print(f"CR-KG graph + stats saved to: {crkg_json_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("CR-KG PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Dataset           : {args.data} ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"  Models used       : {models}")
    print(f"  Candidate edges   : {len(candidate_result['edges'])}")
    print(f"  Validated edges   : {len(validated_result['edges'])}")
    print(f"  Hallucination rate: {validated_result['stats'].get('hallucination_rate', 0):.1%}")
    print(f"  Columns kept      : {len(compression_plan['columns_to_keep'])}")
    print(f"  Columns compressed: {len(compression_plan['columns_to_compress'])}")
    print(f"  Compression ratio : {compression_plan['stats'].get('compression_ratio', 0):.1%}")


def _get_ensemble_models(args) -> list:
    """Determine which models to use based on --model and --ensemble flags.

    If --ensemble is provided, it is a comma-separated list of model names.
    Otherwise, the single --model value is used.
    """
    ensemble_str = getattr(args, "ensemble", None)
    if ensemble_str:
        return [m.strip() for m in ensemble_str.split(",") if m.strip()]
    return [args.model]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TabKG: Knowledge Graph-guided tabular data compression and synthesis"
    )
    parser.add_argument("--data", help="Dataset name", type=str, default="icustays")
    parser.add_argument("--model", help="Generative model (gpt, deepseek, claude, llama)",
                        type=str, default="deepseek")
    parser.add_argument("--temp", type=float, default=0.1,
                        help="LLM sampling temperature")
    parser.add_argument("--max_tok", type=int, default=2000,
                        help="Max tokens for LLM response")
    parser.add_argument("--method", type=str, default="legacy",
                        choices=["legacy", "crkg"],
                        help="Pipeline method: 'legacy' (3-prompt) or 'crkg' (CR-KG)")
    parser.add_argument("--ensemble", type=str, default=None,
                        help="Comma-separated model list for CR-KG ensemble "
                             "(e.g. 'deepseek,gpt-4,claude'). Overrides --model "
                             "for the ensemble stage.")
    parser.add_argument("--temp_range", type=str, default=None,
                        help="Comma-separated temperatures for same-model ensemble "
                             "voting (e.g. '0.1,0.3,0.5,0.7,0.9'). Used with "
                             "--ensemble when all models are the same, e.g.: "
                             "--ensemble 'deepseek,deepseek,deepseek,deepseek,deepseek' "
                             "--temp_range '0.1,0.3,0.5,0.7,0.9'")
    parser.add_argument("--validation_threshold", type=float, default=0.90,
                        help="Minimum validation score to keep a CR-KG edge")
    args = parser.parse_args()

    if args.method == "crkg":
        run_crkg_pipeline(args)
    else:
        run_legacy_pipeline(args)


if __name__ == "__main__":
    main()
