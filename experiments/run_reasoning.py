"""
Run CR-KG reasoning experiments across multiple datasets and LLMs.
Only requires column descriptions (no CSV data files needed).

Usage:
    # Single model across all datasets
    python experiments/run_reasoning.py --model deepseek

    # Multiple models for cross-model comparison
    python experiments/run_reasoning.py --model deepseek,gpt-4

    # Same-model temperature sweep
    python experiments/run_reasoning.py --model deepseek --temp_sweep
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_column_descriptions
from reasoning.llm_ensemble import call_single_model, run_ensemble


# Datasets to evaluate (grouped by domain)
DATASETS = {
    # Supply Chain
    "dataco": "Supply Chain (Retail)",
    "adult": "Demographics (Adult Income)",
    # MIMIC-III Clinical
    "icustays": "Clinical (ICU Stays)",
    "purchasing": "Supply Chain (Purchasing)",
}

# Ground truth relationships for F1 evaluation (key datasets)
GROUND_TRUTH = {
    "dataco": {
        "hierarchical": [
            # Geographical (Orders): city→state→country→region→market
            ("order_city", "order_state"), ("order_state", "order_country"),
            ("order_country", "order_region"), ("order_region", "market"),
            # Geographical (Customers): zipcode→city→state→country
            ("customer_zipcode", "customer_city"),
            ("customer_city", "customer_state"), ("customer_state", "customer_country"),
            # Product: card→name, card→category, category_id→name, dept_id→name
            ("category_id", "category_name"), ("department_id", "department_name"),
            ("product_card_id", "product_name"), ("product_card_id", "product_price"),
            ("product_card_id", "product_category_id"),
            ("product_category_id", "category_id"),
            ("order_item_cardprod_id", "product_card_id"),
            # IDs
            ("order_customer_id", "customer_id"),
            ("order_id", "order_item_id"),
            # Geo linkage
            ("latitude", "customer_city"), ("longitude", "customer_city"),
            ("latitude", "longitude"),
        ],
        "mathematical": [
            # total_amount = quantity × price
            ("order_item_quantity", "order_item_total_amount"),
            ("order_item_product_price", "order_item_total_amount"),
            # discount = discount_rate × price
            ("order_item_discount_rate", "order_item_discount"),
            ("order_item_product_price", "order_item_discount"),
            ("order_item_discount", "order_item_total_amount"),
            # sales related
            ("order_item_total_amount", "sales"),
            ("sales", "sales_per_customer"),
            # profit
            ("order_item_total_amount", "profit_per_order"),
            ("order_item_total_amount", "order_profit_per_order"),
            ("order_item_profit_ratio", "profit_per_order"),
            ("order_item_profit_ratio", "order_profit_per_order"),
            ("sales", "order_profit_per_order"),
            ("sales", "profit_per_order"),
            ("order_profit_per_order", "profit_per_order"),
        ],
        "temporal": [
            ("order_date", "shipping_date"),
        ],
        "semantic": [
            ("order_status", "label"),
            ("shipping_mode", "shipping_date"),
        ],
    },
    "purchasing": {
        "hierarchical": [
            ("supplier_number", "supplier_name"),
            ("material_number", "material_description"),
        ],
        "mathematical": [
            ("quantity", "net_amount"),
            ("unit_price", "net_amount"),
            ("net_amount", "gross_amount"),
            ("discount_rate", "gross_amount"),
        ],
        "temporal": [
            ("order_created_date", "planned_delivery_date"),
            ("order_created_date", "actual_delivery_date"),
            ("planned_delivery_date", "actual_delivery_date"),
            ("actual_delivery_date", "goods_receipt_date"),
            ("goods_receipt_date", "invoice_date"),
        ],
        "semantic": [
            ("planned_delivery_date", "delivery_indicator"),
            ("actual_delivery_date", "delivery_indicator"),
            ("order_status", "actual_delivery_date"),
        ],
    },
}


def compute_f1(predicted_edges, ground_truth, dataset_name):
    """Compute precision, recall, F1 against ground truth."""
    if dataset_name not in GROUND_TRUTH:
        return {"precision": None, "recall": None, "f1": None, "note": "No ground truth"}

    gt = GROUND_TRUTH[dataset_name]

    # Flatten ground truth into (source, target, type) tuples
    gt_set = set()
    for edge_type in ["hierarchical", "mathematical", "temporal", "semantic"]:
        for item in gt.get(edge_type, []):
            if len(item) >= 2:
                gt_set.add((item[0].lower(), item[1].lower(), edge_type))

    if not gt_set:
        return {"precision": None, "recall": None, "f1": None, "note": "Empty ground truth"}

    # Flatten predicted edges
    pred_set = set()
    for edge in predicted_edges:
        pred_set.add((
            edge["source"].strip().lower(),
            edge["target"].strip().lower(),
            edge["type"].strip().lower(),
        ))

    # Strict match (type must match)
    tp_strict = len(pred_set & gt_set)

    # Relaxed match (ignore edge type — only check source→target pair)
    gt_pairs = {(s, t) for s, t, _ in gt_set}
    pred_pairs = {(s, t) for s, t, _ in pred_set}
    tp = len(pred_pairs & gt_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "gt_size": len(gt_set),
        "pred_size": len(pred_set),
    }


def run_single_experiment(dataset_name, model_name, temperature, max_tok=2000):
    """Run CR-KG reasoning on one dataset with one model at one temperature."""
    col_desc = get_column_descriptions(dataset_name)
    if not col_desc:
        return None

    # Create args-like object
    class Args:
        pass
    args = Args()
    args.temp = temperature
    args.max_tok = max_tok

    start_time = time.time()
    graph = call_single_model(model_name, col_desc, args)
    elapsed = time.time() - start_time

    if graph is None:
        return {
            "dataset": dataset_name,
            "model": model_name,
            "temperature": temperature,
            "status": "failed",
            "time_seconds": round(elapsed, 2),
        }

    edges = graph.get("edges", [])
    nodes = graph.get("nodes", [])

    # Compute F1 if ground truth available
    f1_result = compute_f1(edges, GROUND_TRUTH, dataset_name)

    # Count edge types
    type_counts = {}
    for e in edges:
        t = e.get("type", "unknown").lower()
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        "dataset": dataset_name,
        "domain": DATASETS.get(dataset_name, "Unknown"),
        "model": model_name,
        "temperature": temperature,
        "status": "success",
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "edge_types": type_counts,
        "f1_metrics": f1_result,
        "time_seconds": round(elapsed, 2),
        "edges": edges,  # Full edges for later analysis
    }


def main():
    parser = argparse.ArgumentParser(description="Run CR-KG reasoning experiments")
    parser.add_argument("--model", type=str, default="deepseek",
                        help="Comma-separated model names (e.g. 'deepseek,gpt-4')")
    parser.add_argument("--datasets", type=str, default=None,
                        help="Comma-separated dataset names. Default: all")
    parser.add_argument("--temp", type=float, default=0.1,
                        help="Temperature for single-temp runs")
    parser.add_argument("--temp_sweep", action="store_true",
                        help="Run temperature sweep (0.1, 0.3, 0.5, 0.7, 0.9)")
    parser.add_argument("--max_tok", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="results/reasoning")
    args = parser.parse_args()

    models = [m.strip() for m in args.model.split(",")]
    datasets = (
        [d.strip() for d in args.datasets.split(",")]
        if args.datasets
        else list(DATASETS.keys())
    )
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9] if args.temp_sweep else [args.temp]

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []
    total = len(models) * len(datasets) * len(temperatures)
    count = 0

    print(f"\n{'='*70}")
    print(f"CR-KG Reasoning Experiments")
    print(f"Models: {models}")
    print(f"Datasets: {datasets}")
    print(f"Temperatures: {temperatures}")
    print(f"Total runs: {total}")
    print(f"{'='*70}\n")

    for model in models:
        for dataset in datasets:
            for temp in temperatures:
                count += 1
                print(f"\n[{count}/{total}] {model} | {dataset} | temp={temp}")
                print("-" * 50)

                result = run_single_experiment(dataset, model, temp, args.max_tok)
                if result:
                    # Print summary
                    if result["status"] == "success":
                        f1 = result["f1_metrics"].get("f1")
                        f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
                        print(f"  Nodes: {result['n_nodes']}, Edges: {result['n_edges']}, "
                              f"F1: {f1_str}, Time: {result['time_seconds']}s")
                        print(f"  Edge types: {result['edge_types']}")
                    else:
                        print(f"  FAILED ({result['time_seconds']}s)")

                    # Don't save full edges in summary (too large)
                    result_summary = {k: v for k, v in result.items() if k != "edges"}
                    all_results.append(result_summary)

                    # Save individual result with edges
                    individual_path = os.path.join(
                        args.output_dir,
                        f"{dataset}_{model}_{temp}.json"
                    )
                    with open(individual_path, "w") as f:
                        json.dump(result, f, indent=2, default=str)

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print results table
    print(f"\n\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Dataset':<20} {'Model':<12} {'Temp':<6} {'Nodes':<7} {'Edges':<7} {'F1':<8} {'Time':<6}")
    print("-" * 70)
    for r in all_results:
        if r["status"] == "success":
            f1 = r["f1_metrics"].get("f1")
            f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
            print(f"{r['dataset']:<20} {r['model']:<12} {r['temperature']:<6} "
                  f"{r['n_nodes']:<7} {r['n_edges']:<7} {f1_str:<8} {r['time_seconds']:<6}")
        else:
            print(f"{r['dataset']:<20} {r['model']:<12} {r['temperature']:<6} "
                  f"{'FAILED':<7} {'':<7} {'':<8} {r['time_seconds']:<6}")

    print(f"\nResults saved to: {args.output_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
