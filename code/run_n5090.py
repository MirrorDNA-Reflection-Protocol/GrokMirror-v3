#!/usr/bin/env python3
"""
GrokMirror Full Analysis at n=5090 (real vault scale).

Optimized version: skips expensive near-prime variants,
focuses on the core rules that matter for the paper.
"""

import json
import time
import sys
import os
import math
import platform
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from grokmirror_core import (
    EdgeRules, build_graph, build_erdos_renyi_matched,
    graph_metrics, spectral_metrics, small_world_sigma, robustness_test
)


def system_info():
    import psutil
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
    }


def build_graph_report(n, rule_func, label, desc, sample_paths=500):
    """Build graph and compute metrics, returning both."""
    print(f"  [{label}] {desc}", flush=True)
    t0 = time.time()
    G = build_graph(n, rule_func, label=label)
    build_time = time.time() - t0
    print(f"    Built: {G.number_of_edges():,} edges ({build_time:.1f}s)", flush=True)

    t0 = time.time()
    metrics = graph_metrics(G, sample_paths=sample_paths)
    metric_time = time.time() - t0
    metrics['build_time_s'] = build_time
    metrics['metric_time_s'] = metric_time
    metrics['description'] = desc
    metrics['label'] = label

    print(f"    density={metrics['density']:.4f}  giant={metrics['giant_component_frac']:.3f}  "
          f"diam={metrics['diameter_sampled']}  clustering={metrics.get('clustering_coefficient', 0):.4f}  "
          f"({metric_time:.1f}s)", flush=True)

    return G, metrics


def run_n5090():
    n = 5090
    outdir = os.path.join(os.path.dirname(__file__), 'results_n5090')
    os.makedirs(outdir, exist_ok=True)

    print(f"\n{'#'*60}", flush=True)
    print(f"  GROKMIRROR FULL ANALYSIS — n = {n} (REAL VAULT)", flush=True)
    print(f"{'#'*60}\n", flush=True)

    master = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'system': system_info(),
            'n': n,
        }
    }

    total_start = time.time()
    rules = EdgeRules(n)

    # ═══════════════════════════════════════════════════════════════
    #  STAGE 1: PHASE DIAGRAM (core rules only)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print(f"  STAGE 1: PHASE DIAGRAM", flush=True)
    print(f"{'='*60}\n", flush=True)

    phase_results = {}

    # The four primary rules
    primary = [
        ("v1.0_strict_AND", rules.v1_strict_and, "Sum=Prime AND Diff=Pow2"),
        ("prime_sum_only", rules.prime_sum_only, "Sum=Prime only"),
        ("pow2_diff_only", rules.pow2_diff_only, "Diff=Pow2 only"),
        ("v2.0_strategic_OR", rules.v2_strategic_or, "Sum=Prime OR Diff=Pow2"),
    ]

    graphs = {}
    for label, func, desc in primary:
        G, m = build_graph_report(n, func, label, desc)
        phase_results[label] = m
        graphs[label] = G

    # Locality-bounded variants (the percolation sweep)
    for k in [2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256, 512, 1024]:
        label = f"prime_AND_diff_leq_{k}"
        desc = f"Sum=Prime AND |i-j| <= {k}"
        func = rules.diff_leq_k(k)
        G, m = build_graph_report(n, func, label, desc, sample_paths=300)
        phase_results[label] = m
        # Keep graphs near percolation threshold for deeper analysis
        if 0.2 < m['giant_component_frac'] < 1.0 or k in [16, 24, 32]:
            graphs[label] = G

    # Log-scaled locality
    for mult in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]:
        func = rules.diff_leq_logn_mult(mult)
        label = f"prime_AND_diff_leq_{mult}log2n"
        desc = f"Sum=Prime AND |i-j| <= {mult}*log2(n) (={func.threshold})"
        _, m = build_graph_report(n, func, label, desc, sample_paths=300)
        phase_results[label] = m

    # Relaxed pow2 (fast — same cost as v1)
    for gap in [1, 2, 4, 8, 16]:
        func = rules.prime_and_diff_pow2_relaxed(gap)
        label = f"prime_AND_near_pow2_gap_{gap}"
        desc = f"Sum=Prime AND |i-j| within {gap} of pow2"
        _, m = build_graph_report(n, func, label, desc, sample_paths=300)
        phase_results[label] = m

    master['phase_diagram'] = {'n': n, 'rules': phase_results}
    with open(os.path.join(outdir, 'phase_diagram_n5090.json'), 'w') as f:
        json.dump(master['phase_diagram'], f, indent=2, default=str)
    print(f"\n  Phase diagram saved.", flush=True)

    # ═══════════════════════════════════════════════════════════════
    #  STAGE 2: BASELINE COMPARISONS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print(f"  STAGE 2: BASELINE COMPARISONS (Erdős-Rényi)", flush=True)
    print(f"{'='*60}\n", flush=True)

    baseline_results = {}
    for label in ['v1.0_strict_AND', 'prime_sum_only', 'pow2_diff_only', 'v2.0_strategic_OR']:
        G = graphs[label]
        m = G.number_of_edges()
        print(f"\n  {label}: building ER baseline ({m:,} target edges)...", flush=True)

        G_er = build_erdos_renyi_matched(n, m, seed=42)
        er_metrics = graph_metrics(G_er, sample_paths=300)
        gm_metrics = phase_results[label]

        comparison = {
            'grokmirror': gm_metrics,
            'erdos_renyi': er_metrics,
            'delta': {}
        }

        for key in ['density', 'giant_component_frac', 'diameter_sampled',
                     'avg_path_length_sampled', 'clustering_coefficient',
                     'transitivity', 'degree_std', 'degree_mean']:
            gm_val = gm_metrics.get(key, 0)
            er_val = er_metrics.get(key, 0)
            if isinstance(er_val, (int, float)) and er_val != 0 and er_val != float('inf'):
                comparison['delta'][key] = {
                    'grokmirror': gm_val,
                    'erdos_renyi': er_val,
                    'ratio': gm_val / er_val,
                    'diff': gm_val - er_val,
                }
                print(f"    {key}: GM={gm_val:.4f}  ER={er_val:.4f}  ratio={gm_val/er_val:.3f}", flush=True)

        baseline_results[label] = comparison

    master['baselines'] = {'n': n, 'comparisons': baseline_results}
    with open(os.path.join(outdir, 'baselines_n5090.json'), 'w') as f:
        json.dump(master['baselines'], f, indent=2, default=str)
    print(f"\n  Baselines saved.", flush=True)

    # ═══════════════════════════════════════════════════════════════
    #  STAGE 3: ADVANCED ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print(f"  STAGE 3: ADVANCED ANALYSIS", flush=True)
    print(f"{'='*60}\n", flush=True)

    advanced_results = {}

    for label in ['pow2_diff_only', 'v2.0_strategic_OR']:
        G = graphs[label]
        entry = {'edges': G.number_of_edges()}
        print(f"\n  --- {label} ---", flush=True)

        # Spectral
        print(f"    Spectral analysis...", flush=True)
        try:
            spec = spectral_metrics(G)
            entry['spectral'] = spec
            print(f"    λ₂ = {spec['lambda_2']:.6f}", flush=True)
        except Exception as e:
            entry['spectral'] = {'error': str(e)}
            print(f"    Spectral error: {e}", flush=True)

        # Small-world sigma
        print(f"    Small-world sigma...", flush=True)
        try:
            sw = small_world_sigma(G, nrand=3)
            entry['small_world'] = sw
            print(f"    σ = {sw['sigma']:.4f}  (γ={sw['gamma']:.4f}  λ={sw['lambda']:.4f})", flush=True)
        except Exception as e:
            entry['small_world'] = {'error': str(e)}
            print(f"    Small-world error: {e}", flush=True)

        # Robustness
        print(f"    Robustness testing...", flush=True)
        try:
            rob = robustness_test(G, removal_fracs=[0.05, 0.10, 0.20, 0.30, 0.50], trials=3)
            entry['robustness'] = rob
            for frac, data in rob['random_removal'].items():
                print(f"      Random {float(frac)*100:.0f}%: giant={data['mean']:.4f}", flush=True)
            for frac, val in rob['targeted_removal'].items():
                print(f"      Targeted {float(frac)*100:.0f}%: giant={val:.4f}", flush=True)
        except Exception as e:
            entry['robustness'] = {'error': str(e)}
            print(f"    Robustness error: {e}", flush=True)

        # Degree distribution
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
            print(f"    Assortativity: {entry['assortativity']:.4f}", flush=True)

        advanced_results[label] = entry

    # Also do v1.0 spectral/assortativity (cheap since it's tiny)
    G_v1 = graphs['v1.0_strict_AND']
    v1_entry = {'edges': G_v1.number_of_edges()}
    try:
        v1_entry['spectral'] = spectral_metrics(G_v1)
    except:
        pass
    if G_v1.number_of_edges() > 0:
        v1_entry['assortativity'] = nx.degree_assortativity_coefficient(G_v1)
    advanced_results['v1.0_strict_AND'] = v1_entry

    master['advanced'] = {'n': n, 'advanced': advanced_results}
    with open(os.path.join(outdir, 'advanced_n5090.json'), 'w') as f:
        json.dump(master['advanced'], f, indent=2, default=str)
    print(f"\n  Advanced analysis saved.", flush=True)

    # ═══════════════════════════════════════════════════════════════
    #  STAGE 4: PERCOLATION THRESHOLD ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}", flush=True)
    print(f"  STAGE 4: PERCOLATION THRESHOLD DETAIL", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Fine-grained sweep near the critical point
    perc_results = {}
    for k in range(8, 33):
        func = rules.diff_leq_k(k)
        label = f"diff_leq_{k}"
        G = build_graph(n, func, label=label)
        components = list(nx.connected_components(G))
        giant = max(components, key=len) if components else set()
        giant_frac = len(giant) / n
        m = G.number_of_edges()
        density = m / (n * (n-1) / 2)
        is_conn = nx.is_connected(G)

        perc_results[k] = {
            'k': k,
            'k_over_log2n': k / math.log2(n),
            'edges': m,
            'density': density,
            'giant_frac': giant_frac,
            'num_components': len(components),
            'is_connected': is_conn,
        }

        marker = "◆" if is_conn else "○"
        print(f"    {marker} k={k:>3} ({k/math.log2(n):.2f}·log₂n)  "
              f"edges={m:>8,}  density={density:.4f}  "
              f"giant={giant_frac:.4f}  components={len(components)}", flush=True)

    master['percolation_detail'] = perc_results
    with open(os.path.join(outdir, 'percolation_detail_n5090.json'), 'w') as f:
        json.dump(perc_results, f, indent=2, default=str)
    print(f"\n  Percolation detail saved.", flush=True)

    # ═══════════════════════════════════════════════════════════════
    #  FINALIZE
    # ═══════════════════════════════════════════════════════════════
    total_elapsed = time.time() - total_start
    master['metadata']['total_runtime_s'] = total_elapsed

    master_path = os.path.join(outdir, 'master_results_n5090.json')
    with open(master_path, 'w') as f:
        json.dump(master, f, indent=2, default=str)

    print(f"\n{'#'*60}", flush=True)
    print(f"  COMPLETE — {total_elapsed:.1f}s total", flush=True)
    print(f"  Master: {master_path}", flush=True)
    print(f"{'#'*60}", flush=True)

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n  SUMMARY TABLE (n={n}):", flush=True)
    print(f"  {'Rule':<35} {'Edges':>10} {'Density':>8} {'Giant':>7} {'Diam':>5} {'Clust':>7}", flush=True)
    print(f"  {'─'*75}", flush=True)
    for label, m in sorted(phase_results.items(), key=lambda x: x[1]['density']):
        print(f"  {label:<35} {m['edges']:>10,} {m['density']:>8.4f} "
              f"{m['giant_component_frac']:>7.3f} {m['diameter_sampled']:>5} "
              f"{m.get('clustering_coefficient', 0):>7.4f}", flush=True)

    # ── Percolation summary ───────────────────────────────────────
    print(f"\n  PERCOLATION THRESHOLD:", flush=True)
    for k_val, data in sorted(perc_results.items()):
        if data['giant_frac'] > 0.99 and not perc_results.get(k_val - 1, {}).get('is_connected', True):
            print(f"    CRITICAL k* = {k_val} ({data['k_over_log2n']:.2f}·log₂n)", flush=True)
            print(f"    Edges at threshold: {data['edges']:,}", flush=True)
            print(f"    Density at threshold: {data['density']:.4f}", flush=True)
            break

    return master


if __name__ == '__main__':
    run_n5090()
