#!/usr/bin/env python3
"""
Asymptotic Conjecture — Does k*/log₂n converge?

Find the exact percolation threshold k* at multiple scales
and track the ratio k*/log₂n as n grows.
"""

import json
import math
import time
import sys
import os
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from grokmirror_core import EdgeRules, build_graph


def find_critical_k(n: int) -> dict:
    """Binary-search for the exact k* where G_n^k first becomes connected."""
    rules = EdgeRules(n)
    lo, hi = 1, n

    # First: quick scan to bracket
    for k in [2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256]:
        if k >= n:
            break
        G = build_graph(n, rules.diff_leq_k(k))
        if nx.is_connected(G):
            hi = k
            break
        lo = k

    # Binary search within bracket
    while hi - lo > 1:
        mid = (lo + hi) // 2
        G = build_graph(n, rules.diff_leq_k(mid))
        if nx.is_connected(G):
            hi = mid
        else:
            lo = mid

    # hi is now the smallest k where connected
    k_star = hi
    log2n = math.log2(n)
    ratio = k_star / log2n

    # Also get giant component fraction just below threshold
    G_below = build_graph(n, rules.diff_leq_k(k_star - 1))
    comps = list(nx.connected_components(G_below))
    giant_below = max(len(c) for c in comps) / n

    # And metrics at threshold
    G_at = build_graph(n, rules.diff_leq_k(k_star))
    edges_at = G_at.number_of_edges()
    density_at = edges_at / (n * (n - 1) / 2)

    return {
        'n': n,
        'k_star': k_star,
        'log2_n': log2n,
        'ratio': ratio,
        'edges_at_threshold': edges_at,
        'density_at_threshold': density_at,
        'giant_frac_below': giant_below,
    }


def run_asymptotic(sizes=None, output_path=None):
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000, 5090]

    print(f"\n{'='*60}", flush=True)
    print(f"  ASYMPTOTIC CONJECTURE: k*/log₂n CONVERGENCE", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = []

    print(f"  {'n':>6}  {'k*':>4}  {'log₂n':>7}  {'k*/log₂n':>10}  {'edges':>10}  {'density':>10}  {'giant below':>12}", flush=True)
    print(f"  {'─'*70}", flush=True)

    for n in sizes:
        t0 = time.time()
        r = find_critical_k(n)
        elapsed = time.time() - t0
        results.append(r)

        print(f"  {r['n']:>6}  {r['k_star']:>4}  {r['log2_n']:>7.2f}  {r['ratio']:>10.4f}  "
              f"{r['edges_at_threshold']:>10,}  {r['density_at_threshold']:>10.6f}  "
              f"{r['giant_frac_below']:>12.4f}  ({elapsed:.1f}s)", flush=True)

    # Analyze convergence
    ratios = [r['ratio'] for r in results]
    print(f"\n  CONVERGENCE ANALYSIS:", flush=True)
    print(f"  Ratios: {[f'{r:.4f}' for r in ratios]}", flush=True)

    if len(ratios) >= 3:
        # Check if ratio is monotonically decreasing/increasing or stabilizing
        diffs = [ratios[i+1] - ratios[i] for i in range(len(ratios)-1)]
        print(f"  Deltas: {[f'{d:+.4f}' for d in diffs]}", flush=True)

        # Last 3 ratios — compute variance to assess convergence
        tail = ratios[-3:]
        import numpy as np
        tail_std = float(np.std(tail))
        tail_mean = float(np.mean(tail))
        print(f"  Last 3 ratios: mean={tail_mean:.4f}, std={tail_std:.4f}", flush=True)

        if tail_std < 0.05:
            print(f"\n  CONJECTURE SUPPORTED: k*/log₂n appears to converge to ~{tail_mean:.2f}", flush=True)
        else:
            print(f"\n  CONJECTURE UNCERTAIN: ratio still drifting (std={tail_std:.4f})", flush=True)

    output = {
        'description': 'Asymptotic analysis of percolation threshold k* vs log2(n)',
        'results': results,
        'ratios': ratios,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved to {output_path}", flush=True)

    return output


if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(outdir, exist_ok=True)
    run_asymptotic(output_path=os.path.join(outdir, 'asymptotic_conjecture.json'))
