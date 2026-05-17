"""
Graph-based compression planning via DAG construction and tree decomposition.

Takes a validated CR-KG and produces a compression plan that identifies which
columns to keep (root / independent), which to compress away (dependent), and
how to reconstruct them.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Set, Tuple


# ---------------------------------------------------------------------------
# DAG construction
# ---------------------------------------------------------------------------

def build_dependency_dag(validated_graph: dict) -> dict:
    """Convert a validated CR-KG into a Directed Acyclic Graph.

    If cycles are detected they are broken by removing the edge with the
    lowest validation score in each cycle.

    Args:
        validated_graph: Dict with ``"nodes"`` and ``"edges"`` (each edge has
            ``source``, ``target``, ``type``, and optionally
            ``validation_score``).

    Returns:
        A dict with:
          - ``nodes``: list of node names
          - ``adjacency``: ``{source: [target, ...]}``
          - ``edges``: list of edges in the DAG (may be fewer than input
            if cycles were broken)
          - ``topological_order``: nodes in topological order
          - ``removed_cycle_edges``: edges removed to break cycles
    """
    nodes: List[str] = list(validated_graph.get("nodes", []))
    edges: List[dict] = list(validated_graph.get("edges", []))

    # Sort edges by validation_score descending so we preferentially keep
    # high-confidence edges when breaking cycles.
    edges_sorted = sorted(
        edges,
        key=lambda e: e.get("validation_score", e.get("confidence", 0)),
        reverse=True,
    )

    adjacency: Dict[str, List[str]] = defaultdict(list)
    kept_edges: List[dict] = []
    removed_edges: List[dict] = []

    # Incrementally add edges, skipping any that would create a cycle
    node_set: Set[str] = set(nodes)

    for edge in edges_sorted:
        src = edge["source"]
        tgt = edge["target"]

        # Ensure both nodes exist
        node_set.add(src)
        node_set.add(tgt)

        # Tentatively add edge
        adjacency[src].append(tgt)

        if _has_cycle(adjacency, node_set):
            # Remove it -- this edge would create a cycle
            adjacency[src].remove(tgt)
            removed_edges.append(edge)
            print(
                f"[compressor] Removed cycle edge: {src} -> {tgt} "
                f"(score={edge.get('validation_score', '?')})"
            )
        else:
            kept_edges.append(edge)

    # Compute topological order
    topo_order = _topological_sort(adjacency, node_set)

    return {
        "nodes": sorted(node_set),
        "adjacency": dict(adjacency),
        "edges": kept_edges,
        "topological_order": topo_order,
        "removed_cycle_edges": removed_edges,
    }


def _has_cycle(adjacency: Dict[str, List[str]], nodes: Set[str]) -> bool:
    """Check for cycles using iterative DFS with coloring."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}

    for start in nodes:
        if color[start] != WHITE:
            continue
        stack = [(start, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                color[node] = BLACK
                continue
            if color[node] == GRAY:
                return True
            if color[node] == BLACK:
                continue
            color[node] = GRAY
            stack.append((node, True))
            for nbr in adjacency.get(node, []):
                if color.get(nbr, WHITE) == GRAY:
                    return True
                if color.get(nbr, WHITE) == WHITE:
                    stack.append((nbr, False))
    return False


def _topological_sort(adjacency: Dict[str, List[str]], nodes: Set[str]) -> List[str]:
    """Kahn's algorithm for topological sort."""
    in_degree: Dict[str, int] = defaultdict(int)
    for n in nodes:
        in_degree.setdefault(n, 0)
    for src, targets in adjacency.items():
        for tgt in targets:
            in_degree[tgt] += 1

    queue = deque(n for n in nodes if in_degree[n] == 0)
    order: List[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for nbr in adjacency.get(node, []):
            in_degree[nbr] -= 1
            if in_degree[nbr] == 0:
                queue.append(nbr)

    return order


# ---------------------------------------------------------------------------
# Minimal independent set
# ---------------------------------------------------------------------------

def find_minimal_independent_set(dag: dict) -> Tuple[List[str], List[str], Dict[str, dict]]:
    """Identify root (keep) and dependent (compress) columns from the DAG.

    Root nodes are those with in-degree 0 in the DAG.  For each dependent
    (non-root) node, a reconstruction descriptor is produced from its
    incoming edges.

    Args:
        dag: Output of :func:`build_dependency_dag`.

    Returns:
        A tuple of:
          - ``keep_columns``: columns to retain (root nodes)
          - ``compress_columns``: columns to drop / compress
          - ``reconstruction_map``: ``{col: {"sources": [...], "type": ...,
            "rule": ...}}``
    """
    adjacency = dag.get("adjacency", {})
    nodes = set(dag.get("nodes", []))
    edges = dag.get("edges", [])

    # Compute in-degree
    in_degree: Dict[str, int] = {n: 0 for n in nodes}
    for src, targets in adjacency.items():
        for tgt in targets:
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

    keep_columns = sorted(n for n in nodes if in_degree.get(n, 0) == 0)
    compress_columns = sorted(n for n in nodes if in_degree.get(n, 0) > 0)

    # Build reconstruction map from edges
    reconstruction_map: Dict[str, dict] = {}
    for edge in edges:
        tgt = edge["target"]
        if tgt in compress_columns:
            if tgt not in reconstruction_map:
                reconstruction_map[tgt] = {
                    "sources": [],
                    "type": edge.get("type", "unknown"),
                    "rule": edge.get("rule", ""),
                }
            src = edge["source"]
            if src not in reconstruction_map[tgt]["sources"]:
                reconstruction_map[tgt]["sources"].append(src)

    return keep_columns, compress_columns, reconstruction_map


# ---------------------------------------------------------------------------
# Public API: full compression plan
# ---------------------------------------------------------------------------

def generate_compression_plan(validated_graph: dict) -> dict:
    """Generate a complete compression plan from a validated CR-KG.

    Pipeline:
      1. Build DAG (break cycles if needed)
      2. Find minimal independent set (roots vs dependents)
      3. Attach reconstruction functions for each compressed column

    Args:
        validated_graph: Dict with ``"nodes"`` and ``"edges"`` from the
            validator.

    Returns:
        A dict with:
          - ``columns_to_keep``: list of column names to retain
          - ``columns_to_compress``: list of column names to drop
          - ``reconstruction_map``: per-column reconstruction descriptors
          - ``dag``: the full DAG structure (for visualization)
          - ``stats``: summary statistics
    """
    # Step 1
    dag = build_dependency_dag(validated_graph)

    # Step 2
    keep_cols, compress_cols, recon_map = find_minimal_independent_set(dag)

    stats = {
        "total_columns": len(dag["nodes"]),
        "columns_kept": len(keep_cols),
        "columns_compressed": len(compress_cols),
        "compression_ratio": (
            round(len(compress_cols) / len(dag["nodes"]), 4)
            if dag["nodes"]
            else 0.0
        ),
        "cycle_edges_removed": len(dag.get("removed_cycle_edges", [])),
    }

    print(f"\n[compressor] === Compression Plan ===")
    print(f"  Total columns   : {stats['total_columns']}")
    print(f"  Columns to keep : {stats['columns_kept']}  {keep_cols}")
    print(f"  Columns to drop : {stats['columns_compressed']}  {compress_cols}")
    print(f"  Compression ratio: {stats['compression_ratio']:.1%}")

    return {
        "columns_to_keep": keep_cols,
        "columns_to_compress": compress_cols,
        "reconstruction_map": recon_map,
        "dag": dag,
        "stats": stats,
    }
