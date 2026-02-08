"""
Baseline Comparisons — Erdos-Renyi random graphs vs GrokMirror.

The key question: does the number-theoretic structure of GrokMirror
produce meaningfully different topology than a random graph with the
same density? If not, the mathematical framing needs rethinking.
"""

import json
import time
import sys
import os
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from grokmirror_core import (
    EdgeRules, build_graph, build_erdos_renyi_matched,
    graph_metrics, spectral_metrics, small_world_sigma, robustness_test
)


def run_baselines(n: int, output_path: str = None) -> dict:
    """
    Build GrokMirror graphs and their Erdos-Renyi counterparts.
    Compare all metrics side by side.
    """
    print(f"\n{'='*60}")
    print(f"  BASELINE COMPARISONS — n = {n}")
    print(f"{'='*60}\n")

    rules = EdgeRules(n)
    results = {'n': n, 'comparisons': {}}

    key_rules = [
        ('v1.0_strict_AND', rules.v1_strict_and),
        ('prime_sum_only', rules.prime_sum_only),
        ('pow2_diff_only', rules.pow2_diff_only),
        ('v2.0_strategic_OR', rules.v2_strategic_or),
    ]

    for label, rule_func in key_rules:
        print(f"\n  Building {label}...")
        t0 = time.time()
        G = build_graph(n, rule_func, label=label)
        gm_build = time.time() - t0

        m = G.number_of_edges()
        print(f"    GrokMirror: {m:,} edges ({gm_build:.2f}s)")

        # Build matched random graph
        print(f"    Building Erdos-Renyi baseline ({m:,} target edges)...")
        G_rand = build_erdos_renyi_matched(n, m, seed=42)
        m_rand = G_rand.number_of_edges()
        print(f"    Erdos-Renyi: {m_rand:,} edges")

        # Compute metrics for both
        print(f"    Computing GrokMirror metrics...")
        gm_metrics = graph_metrics(G, sample_paths=min(200, n))
        print(f"    Computing Erdos-Renyi metrics...")
        er_metrics = graph_metrics(G_rand, sample_paths=min(200, n))

        comparison = {
            'grokmirror': gm_metrics,
            'erdos_renyi': er_metrics,
            'delta': {}
        }

        # Compute deltas for key metrics
        for key in ['density', 'giant_component_frac', 'diameter_sampled',
                     'avg_path_length_sampled', 'clustering_coefficient',
                     'transitivity', 'degree_std']:
            gm_val = gm_metrics.get(key, 0)
            er_val = er_metrics.get(key, 0)
            if er_val != 0 and er_val != float('inf'):
                comparison['delta'][key] = {
                    'grokmirror': gm_val,
                    'erdos_renyi': er_val,
                    'ratio': gm_val / er_val if er_val != 0 else float('inf'),
                    'diff': gm_val - er_val,
                }

        results['comparisons'][label] = comparison

        print(f"    Comparison:")
        for key, delta in comparison['delta'].items():
            print(f"      {key}: GM={delta['grokmirror']:.4f} ER={delta['erdos_renyi']:.4f} "
                  f"ratio={delta['ratio']:.3f}")

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")

    return results


def run_advanced_analysis(n: int, output_path: str = None) -> dict:
    """
    Deep analysis on the most interesting graphs:
    spectral properties, robustness, small-world coefficient.
    """
    print(f"\n{'='*60}")
    print(f"  ADVANCED ANALYSIS — n = {n}")
    print(f"{'='*60}\n")

    rules = EdgeRules(n)
    results = {'n': n, 'advanced': {}}

    targets = [
        ('v1.0_strict_AND', rules.v1_strict_and),
        ('pow2_diff_only', rules.pow2_diff_only),
        ('v2.0_strategic_OR', rules.v2_strategic_or),
    ]

    for label, rule_func in targets:
        print(f"\n  Analyzing {label}...")
        G = build_graph(n, rule_func, label=label)
        entry = {'edges': G.number_of_edges()}

        # Spectral
        print(f"    Spectral analysis...")
        try:
            spec = spectral_metrics(G)
            entry['spectral'] = spec
            print(f"    λ₂ (algebraic connectivity) = {spec['lambda_2']:.6f}")
        except Exception as e:
            entry['spectral'] = {'error': str(e)}
            print(f"    Spectral failed: {e}")

        # Small-world
        if nx.is_connected(G) or (not nx.is_connected(G) and
                len(max(nx.connected_components(G), key=len)) > n * 0.5):
            print(f"    Small-world sigma...")
            try:
                sw = small_world_sigma(G, nrand=3)
                entry['small_world'] = sw
                print(f"    σ = {sw['sigma']:.4f} (>1 = small-world)")
            except Exception as e:
                entry['small_world'] = {'error': str(e)}
                print(f"    Small-world failed: {e}")
        else:
            entry['small_world'] = {'skipped': 'graph too disconnected'}

        # Robustness
        print(f"    Robustness testing...")
        try:
            rob = robustness_test(G, removal_fracs=[0.05, 0.10, 0.20, 0.30, 0.50])
            entry['robustness'] = rob
            for frac, data in rob['random_removal'].items():
                print(f"    Random {float(frac)*100:.0f}% removal: giant_frac={data['mean']:.3f}")
            for frac, val in rob['targeted_removal'].items():
                print(f"    Targeted {float(frac)*100:.0f}% removal: giant_frac={val:.3f}")
        except Exception as e:
            entry['robustness'] = {'error': str(e)}
            print(f"    Robustness failed: {e}")

        # Degree distribution shape
        degrees = sorted([d for _, d in G.degree()])
        entry['degree_distribution'] = {
            'histogram': np.histogram(degrees, bins=50)[0].tolist(),
            'bin_edges': np.histogram(degrees, bins=50)[1].tolist(),
            'skewness': float(np.mean(((np.array(degrees) - np.mean(degrees)) / max(np.std(degrees), 1e-10))**3)),
            'kurtosis': float(np.mean(((np.array(degrees) - np.mean(degrees)) / max(np.std(degrees), 1e-10))**4) - 3),
        }

        # Assortativity
        if G.number_of_edges() > 0:
            entry['assortativity'] = nx.degree_assortativity_coefficient(G)
            print(f"    Assortativity: {entry['assortativity']:.4f}")

        results['advanced'][label] = entry

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GrokMirror Baselines')
    parser.add_argument('--n', type=int, default=500,
                        help='Graph size (default: 500)')
    parser.add_argument('--advanced', action='store_true',
                        help='Also run advanced analysis')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory')
    args = parser.parse_args()

    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    run_baselines(args.n, output_path=os.path.join(outdir, f'baselines_n{args.n}.json'))

    if args.advanced:
        run_advanced_analysis(args.n, output_path=os.path.join(outdir, f'advanced_n{args.n}.json'))
