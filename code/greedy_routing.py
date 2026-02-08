#!/usr/bin/env python3
"""
Greedy Routing Experiment on D_n.

D_n has a natural coordinate system: each node IS its coordinate (integer on [1,n]).
Greedy routing: at each hop, forward to the neighbor closest to the target.

This tests whether D_n is not just small-world but *navigable* —
i.e., whether decentralized routing achieves near-optimal paths
without global knowledge.
"""

import json
import math
import time
import sys
import os
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from grokmirror_core import EdgeRules, build_graph


def greedy_route(G, source, target, max_hops=None):
    """
    Greedy routing: at each step, move to the neighbor of current node
    that is closest to target (by |current - target|).
    Returns (path, success).
    """
    if max_hops is None:
        max_hops = G.number_of_nodes()

    path = [source]
    current = source
    visited = {source}

    while current != target and len(path) <= max_hops:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            return path, False

        # Pick neighbor closest to target
        best = None
        best_dist = abs(current - target)
        for nb in neighbors:
            d = abs(nb - target)
            if d < best_dist:
                best_dist = d
                best = nb

        if best is None or best in visited:
            # Stuck — no neighbor is closer than current
            # Try any unvisited neighbor (fallback)
            unvisited = [nb for nb in neighbors if nb not in visited]
            if not unvisited:
                return path, False
            # Pick the unvisited neighbor closest to target
            best = min(unvisited, key=lambda nb: abs(nb - target))
            if abs(best - target) >= abs(current - target):
                return path, False  # truly stuck

        visited.add(best)
        path.append(best)
        current = best

    return path, (current == target)


def routing_experiment(n, num_pairs=5000, seed=42, output_path=None):
    """
    Run greedy routing on D_n for random source-target pairs.
    Compare greedy path length to BFS shortest path.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"  GREEDY ROUTING EXPERIMENT — D_n, n={n}", flush=True)
    print(f"{'='*60}\n", flush=True)

    rules = EdgeRules(n)
    print(f"  Building D_n...", flush=True)
    G = build_graph(n, rules.pow2_diff_only)
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges():,}", flush=True)

    rng = np.random.RandomState(seed)
    sources = rng.randint(1, n + 1, size=num_pairs)
    targets = rng.randint(1, n + 1, size=num_pairs)
    # Remove self-pairs
    mask = sources != targets
    sources = sources[mask]
    targets = targets[mask]
    num_pairs = len(sources)
    print(f"  Testing {num_pairs} random pairs...\n", flush=True)

    greedy_lengths = []
    shortest_lengths = []
    successes = 0
    stretch_ratios = []

    t0 = time.time()
    for i in range(num_pairs):
        s, t = int(sources[i]), int(targets[i])

        # Greedy route
        path, success = greedy_route(G, s, t)

        if success:
            successes += 1
            greedy_len = len(path) - 1

            # BFS shortest path
            sp = nx.shortest_path_length(G, s, t)

            greedy_lengths.append(greedy_len)
            shortest_lengths.append(sp)
            stretch_ratios.append(greedy_len / sp if sp > 0 else 1.0)

    elapsed = time.time() - t0

    greedy_arr = np.array(greedy_lengths)
    shortest_arr = np.array(shortest_lengths)
    stretch_arr = np.array(stretch_ratios)

    results = {
        'n': n,
        'num_pairs': num_pairs,
        'success_rate': successes / num_pairs,
        'successes': successes,
        'failures': num_pairs - successes,
        'greedy_path_length': {
            'mean': float(greedy_arr.mean()),
            'std': float(greedy_arr.std()),
            'median': float(np.median(greedy_arr)),
            'max': int(greedy_arr.max()),
            'p95': float(np.percentile(greedy_arr, 95)),
            'p99': float(np.percentile(greedy_arr, 99)),
        },
        'shortest_path_length': {
            'mean': float(shortest_arr.mean()),
            'std': float(shortest_arr.std()),
            'median': float(np.median(shortest_arr)),
            'max': int(shortest_arr.max()),
        },
        'stretch': {
            'mean': float(stretch_arr.mean()),
            'std': float(stretch_arr.std()),
            'median': float(np.median(stretch_arr)),
            'max': float(stretch_arr.max()),
            'p95': float(np.percentile(stretch_arr, 95)),
            'pct_optimal': float(np.mean(stretch_arr == 1.0) * 100),
            'pct_within_2x': float(np.mean(stretch_arr <= 2.0) * 100),
        },
        'theoretical_diameter': math.ceil(math.log2(n)),
        'runtime_s': elapsed,
    }

    print(f"  RESULTS:", flush=True)
    print(f"  {'─'*50}", flush=True)
    print(f"  Success rate:         {results['success_rate']*100:.2f}% ({successes}/{num_pairs})", flush=True)
    print(f"  Greedy path length:   {results['greedy_path_length']['mean']:.2f} ± {results['greedy_path_length']['std']:.2f} "
          f"(median {results['greedy_path_length']['median']:.0f}, max {results['greedy_path_length']['max']})", flush=True)
    print(f"  Shortest path length: {results['shortest_path_length']['mean']:.2f} ± {results['shortest_path_length']['std']:.2f} "
          f"(median {results['shortest_path_length']['median']:.0f}, max {results['shortest_path_length']['max']})", flush=True)
    print(f"  Stretch (greedy/BFS): {results['stretch']['mean']:.4f} ± {results['stretch']['std']:.4f}", flush=True)
    print(f"  Optimal (stretch=1):  {results['stretch']['pct_optimal']:.1f}%", flush=True)
    print(f"  Within 2x optimal:    {results['stretch']['pct_within_2x']:.1f}%", flush=True)
    print(f"  Theoretical diameter: {results['theoretical_diameter']} (ceil(log₂n))", flush=True)
    print(f"  Runtime:              {elapsed:.1f}s", flush=True)

    # Stretch histogram
    bins = [1.0, 1.01, 1.5, 2.0, 3.0, 5.0, float('inf')]
    labels = ['=1.0', '1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0-5.0', '>5.0']
    hist, _ = np.histogram(stretch_arr, bins=bins)
    results['stretch_histogram'] = {labels[i]: int(hist[i]) for i in range(len(labels))}

    print(f"\n  STRETCH DISTRIBUTION:", flush=True)
    for label, count in results['stretch_histogram'].items():
        pct = count / len(stretch_arr) * 100
        bar = '█' * int(pct / 2)
        print(f"    {label:>8}: {count:>5} ({pct:>5.1f}%) {bar}", flush=True)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {output_path}", flush=True)

    return results


def routing_scaling(sizes=None, num_pairs=2000, output_path=None):
    """Test how greedy routing scales with n."""
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000, 5090]

    print(f"\n{'='*60}", flush=True)
    print(f"  ROUTING SCALING EXPERIMENT", flush=True)
    print(f"{'='*60}\n", flush=True)

    results = []

    print(f"  {'n':>6}  {'success':>8}  {'greedy':>8}  {'BFS':>6}  {'stretch':>8}  {'%optimal':>9}  {'log₂n':>6}", flush=True)
    print(f"  {'─'*60}", flush=True)

    for n in sizes:
        rules = EdgeRules(n)
        G = build_graph(n, rules.pow2_diff_only)
        rng = np.random.RandomState(42)

        pairs = min(num_pairs, n * (n - 1) // 2)
        src = rng.randint(1, n + 1, size=pairs)
        tgt = rng.randint(1, n + 1, size=pairs)
        mask = src != tgt
        src, tgt = src[mask], tgt[mask]

        g_lens = []
        s_lens = []
        succ = 0
        for i in range(len(src)):
            path, ok = greedy_route(G, int(src[i]), int(tgt[i]))
            if ok:
                succ += 1
                gl = len(path) - 1
                sl = nx.shortest_path_length(G, int(src[i]), int(tgt[i]))
                g_lens.append(gl)
                s_lens.append(sl)

        g_arr = np.array(g_lens) if g_lens else np.array([0])
        s_arr = np.array(s_lens) if s_lens else np.array([0])
        stretch = g_arr / np.maximum(s_arr, 1)

        r = {
            'n': n,
            'success_rate': succ / len(src) if len(src) > 0 else 0,
            'greedy_mean': float(g_arr.mean()),
            'bfs_mean': float(s_arr.mean()),
            'stretch_mean': float(stretch.mean()),
            'pct_optimal': float(np.mean(stretch == 1.0) * 100),
            'log2_n': math.log2(n),
        }
        results.append(r)

        print(f"  {n:>6}  {r['success_rate']*100:>7.1f}%  {r['greedy_mean']:>8.2f}  "
              f"{r['bfs_mean']:>6.2f}  {r['stretch_mean']:>8.4f}  "
              f"{r['pct_optimal']:>8.1f}%  {r['log2_n']:>6.2f}", flush=True)

    output = {'description': 'Greedy routing scaling on D_n', 'results': results}

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved to {output_path}", flush=True)

    return output


if __name__ == '__main__':
    outdir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(outdir, exist_ok=True)

    routing_experiment(5090, num_pairs=5000,
                       output_path=os.path.join(outdir, 'greedy_routing_n5090.json'))
    routing_scaling(output_path=os.path.join(outdir, 'routing_scaling.json'))
