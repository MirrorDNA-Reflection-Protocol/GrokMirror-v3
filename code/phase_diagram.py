"""
Phase Diagram Analysis — Map the percolation transition between v1.0 and v2.0.

Tests a spectrum of rule variants from strict (v1.0) to loose (v2.0)
and measures connectivity at each point. This is the core scientific
contribution: finding the critical threshold where connectivity emerges.
"""

import json
import time
import sys
import os
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from grokmirror_core import EdgeRules, build_graph, graph_metrics


def run_phase_diagram(n: int, output_path: str = None) -> dict:
    """
    Run the full phase diagram analysis for graph size n.
    Tests all rule variants and returns structured results.
    """
    print(f"\n{'='*60}")
    print(f"  PHASE DIAGRAM ANALYSIS — n = {n}")
    print(f"{'='*60}\n")

    rules = EdgeRules(n)
    results = {'n': n, 'rules': {}}

    # ── Define all rule variants ──────────────────────────────────

    rule_variants = [
        # (label, rule_function, description)
        ("v1.0_strict_AND", rules.v1_strict_and,
         "Sum=Prime AND Diff=Pow2"),

        ("prime_sum_only", rules.prime_sum_only,
         "Sum=Prime (no diff constraint)"),

        ("pow2_diff_only", rules.pow2_diff_only,
         "Diff=Pow2 (no sum constraint)"),

        ("v2.0_strategic_OR", rules.v2_strategic_or,
         "Sum=Prime OR Diff=Pow2"),
    ]

    # Intermediate: prime AND diff <= k
    for k in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        r = rules.diff_leq_k(k)
        rule_variants.append((
            f"prime_AND_diff_leq_{k}",
            r,
            f"Sum=Prime AND |i-j| <= {k}"
        ))

    # Intermediate: prime AND diff <= c * log2(n)
    for mult in [0.5, 1.0, 2.0, 4.0, 8.0]:
        r = rules.diff_leq_logn_mult(mult)
        rule_variants.append((
            f"prime_AND_diff_leq_{mult}log2n",
            r,
            f"Sum=Prime AND |i-j| <= {mult}*log2(n) (={r.threshold})"
        ))

    # Near-prime relaxation
    for tol in [1, 2, 4, 8]:
        r = rules.near_prime_sum(tol)
        rule_variants.append((
            f"near_prime_tol_{tol}",
            r,
            f"Near-prime sum (tolerance={tol})"
        ))

    # Relaxed pow2
    for gap in [1, 2, 4, 8]:
        r = rules.prime_and_diff_pow2_relaxed(gap)
        rule_variants.append((
            f"prime_AND_near_pow2_gap_{gap}",
            r,
            f"Sum=Prime AND |i-j| within {gap} of pow2"
        ))

    # ── Build and analyze each variant ────────────────────────────

    for label, rule_func, description in rule_variants:
        print(f"  [{label}] {description}")
        t0 = time.time()
        G = build_graph(n, rule_func, label=label)
        build_time = time.time() - t0

        t0 = time.time()
        metrics = graph_metrics(G, sample_paths=min(200, n))
        metric_time = time.time() - t0

        metrics['build_time_s'] = build_time
        metrics['metric_time_s'] = metric_time
        metrics['description'] = description
        metrics['label'] = label

        results['rules'][label] = metrics
        print(f"    edges={metrics['edges']:,}  density={metrics['density']:.4f}  "
              f"giant={metrics['giant_component_frac']:.3f}  "
              f"diameter={metrics['diameter_sampled']}  "
              f"time={build_time:.2f}s")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")

    return results


def run_growth_dynamics(sizes: list = None, output_path: str = None) -> dict:
    """
    Test how key rules behave as n grows.
    Critical for understanding if results are intrinsic or just density artifacts.
    """
    if sizes is None:
        sizes = [50, 100, 200, 500, 1000, 2000]

    print(f"\n{'='*60}")
    print(f"  GROWTH DYNAMICS — sizes: {sizes}")
    print(f"{'='*60}\n")

    key_rules = ['v1_strict_and', 'v2_strategic_or', 'prime_sum_only', 'pow2_diff_only']
    results = {'sizes': sizes, 'dynamics': {rule: [] for rule in key_rules}}

    for n in sizes:
        print(f"\n  n = {n}")
        rules = EdgeRules(n)

        for rule_name in key_rules:
            rule_func = getattr(rules, rule_name)
            t0 = time.time()
            G = build_graph(n, rule_func, label=rule_name)
            build_time = time.time() - t0

            metrics = graph_metrics(G, sample_paths=min(100, n))
            metrics['build_time_s'] = build_time

            results['dynamics'][rule_name].append(metrics)
            print(f"    {rule_name}: edges={metrics['edges']:,} density={metrics['density']:.4f} "
                  f"giant_frac={metrics['giant_component_frac']:.3f}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GrokMirror Phase Diagram')
    parser.add_argument('--n', type=int, default=500,
                        help='Graph size for phase diagram (default: 500)')
    parser.add_argument('--growth', action='store_true',
                        help='Also run growth dynamics')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory')
    args = parser.parse_args()

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    run_phase_diagram(args.n, output_path=os.path.join(outdir, f'phase_diagram_n{args.n}.json'))

    if args.growth:
        run_growth_dynamics(output_path=os.path.join(outdir, 'growth_dynamics.json'))
