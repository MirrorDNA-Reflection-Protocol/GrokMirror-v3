#!/usr/bin/env python3
"""
Multi-seed Erdős-Rényi baselines for n=5090.

Runs 10 seeds per graph, computes mean ± std for all metrics.
Uses gnm_random_graph (exact edge count, no enumeration overhead).
"""

import json
import time
import sys
import os
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from grokmirror_core import graph_metrics


def er_exact_edges(n: int, m: int, seed: int) -> nx.Graph:
    """Erdős-Rényi with exactly m edges using nx.gnm_random_graph."""
    G = nx.gnm_random_graph(n, m, seed=seed)
    mapping = {i: i + 1 for i in range(n)}
    return nx.relabel_nodes(G, mapping)


def multiseed_baseline(n: int, m: int, seeds: int = 10, sample_paths: int = 300) -> dict:
    """Run ER baseline across multiple seeds, return mean ± std for each metric."""
    all_metrics = []
    for s in range(seeds):
        G = er_exact_edges(n, m, seed=s + 100)
        metrics = graph_metrics(G, sample_paths=sample_paths)
        all_metrics.append(metrics)

    numeric_keys = [
        'density', 'giant_component_frac', 'diameter_sampled',
        'avg_path_length_sampled', 'clustering_coefficient',
        'transitivity', 'degree_mean', 'degree_std',
    ]
    result = {'seeds': seeds, 'target_edges': m}
    for key in numeric_keys:
        vals = [m_dict[key] for m_dict in all_metrics if key in m_dict]
        if vals:
            result[key] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
            }
    return result


def run_multiseed(n: int = 5090, seeds: int = 10):
    """Run multi-seed baselines for all 4 primary GrokMirror graphs."""

    # Load edge counts from our existing phase diagram
    phase_path = os.path.join(os.path.dirname(__file__), 'results_n5090', 'phase_diagram_n5090.json')
    with open(phase_path) as f:
        phase = json.load(f)

    targets = {
        'v1.0_strict_AND': None,
        'prime_sum_only': None,
        'pow2_diff_only': None,
        'v2.0_strategic_OR': None,
    }
    for label, data in phase['rules'].items():
        if label in targets:
            targets[label] = data['edges']

    print(f"\n{'='*60}", flush=True)
    print(f"  MULTI-SEED BASELINES — n={n}, {seeds} seeds", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = {'n': n, 'seeds': seeds, 'baselines': {}}

    for label, m in targets.items():
        print(f"  {label}: {m:,} target edges, {seeds} seeds...", flush=True)
        t0 = time.time()
        baseline = multiseed_baseline(n, m, seeds=seeds)
        elapsed = time.time() - t0
        results['baselines'][label] = baseline
        print(f"    Done ({elapsed:.1f}s)", flush=True)

        # Print key comparisons
        for key in ['clustering_coefficient', 'diameter_sampled', 'avg_path_length_sampled', 'degree_std']:
            if key in baseline:
                b = baseline[key]
                print(f"    ER {key}: {b['mean']:.4f} ± {b['std']:.4f}", flush=True)

    # Load our GM metrics for side-by-side
    print(f"\n  COMPARISON TABLE (GM vs ER mean ± std):", flush=True)
    print(f"  {'Graph':<22} {'Metric':<25} {'GrokMirror':>12} {'ER mean':>12} {'ER std':>10} {'Ratio':>8}", flush=True)
    print(f"  {'─'*90}", flush=True)

    gm_metrics = {}
    for label in targets:
        gm_metrics[label] = phase['rules'][label]

    for label in targets:
        gm = gm_metrics[label]
        bl = results['baselines'][label]
        for key in ['clustering_coefficient', 'transitivity', 'diameter_sampled', 'avg_path_length_sampled', 'degree_std']:
            if key in bl and key in gm:
                gm_val = gm[key]
                er_mean = bl[key]['mean']
                er_std = bl[key]['std']
                ratio = gm_val / er_mean if er_mean != 0 else float('inf')
                print(f"  {label:<22} {key:<25} {gm_val:>12.4f} {er_mean:>12.4f} {er_std:>10.4f} {ratio:>8.3f}", flush=True)

    # Save
    outdir = os.path.join(os.path.dirname(__file__), 'results_n5090')
    outpath = os.path.join(outdir, 'multiseed_baselines_n5090.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {outpath}", flush=True)

    return results


if __name__ == '__main__':
    run_multiseed()
