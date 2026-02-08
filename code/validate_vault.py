"""
Validate computed GrokMirror graphs against the real vault mesh audit data.
Confirms our edge predicates reproduce the actual v2.0 results from Jan 28.
"""

import json
import sys
import os
import time
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from grokmirror_core import EdgeRules, build_graph, graph_metrics


def load_vault_mesh(path: str) -> nx.Graph:
    """Load the real vault mesh from vault_mesh_audit.json."""
    print(f"  Loading vault mesh from {path}...")
    t0 = time.time()
    with open(path) as f:
        data = json.load(f)

    meta = data['metadata']
    mesh = data['mesh']
    n = meta['vault_file_count']

    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    for node_str, neighbors in mesh.items():
        node = int(node_str)
        for nb in neighbors:
            if nb > node:  # avoid double-counting
                G.add_edge(node, nb)

    elapsed = time.time() - t0
    print(f"  Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges():,} edges ({elapsed:.2f}s)")
    return G, meta


def validate_against_vault(vault_path: str, output_path: str = None) -> dict:
    """
    Full validation: build v2.0 graph from scratch and compare
    edge-by-edge against the stored vault mesh.
    """
    print(f"\n{'='*60}")
    print(f"  VAULT VALIDATION")
    print(f"{'='*60}\n")

    # Load real data
    G_real, meta = load_vault_mesh(vault_path)
    n = meta['vault_file_count']

    # Build from scratch
    print(f"\n  Rebuilding v2.0 graph from scratch (n={n})...")
    rules = EdgeRules(n)
    t0 = time.time()
    G_computed = build_graph(n, rules.v2_strategic_or, label='v2.0_recomputed')
    build_time = time.time() - t0
    print(f"  Built: {G_computed.number_of_edges():,} edges ({build_time:.2f}s)")

    # Compare
    real_edges = set(G_real.edges())
    computed_edges = set(G_computed.edges())

    # Normalize edge direction for comparison
    real_norm = {(min(u,v), max(u,v)) for u,v in real_edges}
    comp_norm = {(min(u,v), max(u,v)) for u,v in computed_edges}

    in_real_not_computed = real_norm - comp_norm
    in_computed_not_real = comp_norm - real_norm
    intersection = real_norm & comp_norm

    result = {
        'n': n,
        'real_edges': len(real_norm),
        'computed_edges': len(comp_norm),
        'intersection': len(intersection),
        'in_real_not_computed': len(in_real_not_computed),
        'in_computed_not_real': len(in_computed_not_real),
        'match_pct': len(intersection) / max(len(real_norm), 1) * 100,
        'build_time_s': build_time,
    }

    print(f"\n  EDGE COMPARISON:")
    print(f"    Real edges:           {result['real_edges']:>12,}")
    print(f"    Computed edges:       {result['computed_edges']:>12,}")
    print(f"    Intersection:         {result['intersection']:>12,}")
    print(f"    In real, not computed: {result['in_real_not_computed']:>12,}")
    print(f"    In computed, not real: {result['in_computed_not_real']:>12,}")
    print(f"    Match:                {result['match_pct']:>11.2f}%")

    if result['match_pct'] == 100.0 and result['in_computed_not_real'] == 0:
        print(f"\n  ✓ PERFECT MATCH — computed graph reproduces vault exactly.")
    elif result['match_pct'] > 99.9:
        print(f"\n  ~ NEAR MATCH — minor discrepancies (likely boundary effects).")
    else:
        print(f"\n  ✗ MISMATCH — investigating...")
        if in_real_not_computed:
            samples = list(in_real_not_computed)[:5]
            print(f"    Sample edges in real but not computed: {samples}")
            for u, v in samples:
                s = u + v
                d = abs(u - v)
                print(f"      ({u},{v}): sum={s} prime={rules.sieve[s]} diff={d} pow2={d in rules.pow2s}")

    # Compute metrics on the real graph
    print(f"\n  Computing metrics on real vault graph...")
    t0 = time.time()
    vault_metrics = graph_metrics(G_real, sample_paths=500)
    metric_time = time.time() - t0
    result['vault_metrics'] = vault_metrics
    result['metric_time_s'] = metric_time

    print(f"    Nodes:           {vault_metrics['n']}")
    print(f"    Edges:           {vault_metrics['edges']:,}")
    print(f"    Density:         {vault_metrics['density']:.4f}")
    print(f"    Components:      {vault_metrics['num_components']}")
    print(f"    Giant component: {vault_metrics['giant_component_size']} ({vault_metrics['giant_component_frac']:.4f})")
    print(f"    Diameter:        {vault_metrics['diameter_sampled']}")
    print(f"    Avg path length: {vault_metrics['avg_path_length_sampled']:.4f}")
    print(f"    Clustering:      {vault_metrics['clustering_coefficient']:.4f}")
    print(f"    Transitivity:    {vault_metrics['transitivity']:.4f}")
    print(f"    Degree mean:     {vault_metrics['degree_mean']:.1f}")
    print(f"    Degree std:      {vault_metrics['degree_std']:.1f}")
    print(f"    Connected:       {vault_metrics['is_connected']}")
    print(f"    Time:            {metric_time:.2f}s")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")

    return result


if __name__ == '__main__':
    vault_path = os.path.join(os.path.dirname(__file__), '..', 'vault_mesh_audit.json')
    outdir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(outdir, exist_ok=True)
    validate_against_vault(vault_path, output_path=os.path.join(outdir, 'vault_validation.json'))
