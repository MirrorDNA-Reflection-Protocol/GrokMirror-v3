#!/usr/bin/env python3
"""
GrokMirror Full Analysis Pipeline

Runs all analyses at multiple scales and aggregates results into
a single structured output suitable for paper inclusion.

Usage:
    python3 run_all.py                    # Default: n=500 (fast, ~2 min)
    python3 run_all.py --n 1000           # Medium (~8 min)
    python3 run_all.py --n 500 --full     # Full suite with growth dynamics
"""

import json
import time
import sys
import os
import platform
import psutil

sys.path.insert(0, os.path.dirname(__file__))

from phase_diagram import run_phase_diagram, run_growth_dynamics
from baselines import run_baselines, run_advanced_analysis


def system_info() -> dict:
    """Capture system metadata for reproducibility."""
    return {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
        'python_version': platform.python_version(),
        'hostname': platform.node(),
    }


def run_all(n: int = 500, full: bool = False, output_dir: str = None):
    """Run the complete analysis pipeline."""

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*60}")
    print(f"  GROKMIRROR EMPIRICAL ANALYSIS PIPELINE")
    print(f"  n = {n}  |  full = {full}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    master = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'system': system_info(),
            'parameters': {'n': n, 'full': full},
        }
    }

    total_start = time.time()

    # ── 1. Phase Diagram ──────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  STAGE 1: PHASE DIAGRAM")
    print("="*60)
    phase_path = os.path.join(output_dir, f'phase_diagram_n{n}.json')
    phase = run_phase_diagram(n, output_path=phase_path)
    master['phase_diagram'] = phase

    # ── 2. Baseline Comparisons ───────────────────────────────────
    print("\n\n" + "="*60)
    print("  STAGE 2: BASELINE COMPARISONS")
    print("="*60)
    baseline_path = os.path.join(output_dir, f'baselines_n{n}.json')
    baselines = run_baselines(n, output_path=baseline_path)
    master['baselines'] = baselines

    # ── 3. Advanced Analysis ──────────────────────────────────────
    print("\n\n" + "="*60)
    print("  STAGE 3: ADVANCED METRICS")
    print("="*60)
    adv_path = os.path.join(output_dir, f'advanced_n{n}.json')
    advanced = run_advanced_analysis(n, output_path=adv_path)
    master['advanced'] = advanced

    # ── 4. Growth Dynamics (optional) ─────────────────────────────
    if full:
        print("\n\n" + "="*60)
        print("  STAGE 4: GROWTH DYNAMICS")
        print("="*60)
        growth_sizes = [50, 100, 200, 500]
        if n >= 1000:
            growth_sizes.append(1000)
        growth_path = os.path.join(output_dir, 'growth_dynamics.json')
        growth = run_growth_dynamics(sizes=growth_sizes, output_path=growth_path)
        master['growth_dynamics'] = growth

    total_elapsed = time.time() - total_start
    master['metadata']['total_runtime_s'] = total_elapsed

    # ── Save master results ───────────────────────────────────────
    master_path = os.path.join(output_dir, 'master_results.json')
    with open(master_path, 'w') as f:
        json.dump(master, f, indent=2, default=str)

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n\n{'#'*60}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Master results: {master_path}")
    print(f"{'#'*60}")

    # ── Print key findings ────────────────────────────────────────
    print(f"\n  KEY FINDINGS (n={n}):")
    print(f"  {'─'*50}")

    phase_rules = phase['rules']
    if 'v1.0_strict_AND' in phase_rules:
        v1 = phase_rules['v1.0_strict_AND']
        print(f"  v1.0 Strict AND:  edges={v1['edges']:>8,}  density={v1['density']:.4f}  "
              f"giant={v1['giant_component_frac']:.3f}")
    if 'v2.0_strategic_OR' in phase_rules:
        v2 = phase_rules['v2.0_strategic_OR']
        print(f"  v2.0 Strategic OR: edges={v2['edges']:>8,}  density={v2['density']:.4f}  "
              f"giant={v2['giant_component_frac']:.3f}")

    # Find percolation threshold
    print(f"\n  PERCOLATION THRESHOLD CANDIDATES:")
    for label, data in sorted(phase_rules.items(), key=lambda x: x[1]['density']):
        frac = data['giant_component_frac']
        if 0.3 < frac < 0.95:
            print(f"    {label}: density={data['density']:.4f} giant={frac:.3f}")

    # Baseline comparison summary
    if 'comparisons' in baselines:
        print(f"\n  BASELINE COMPARISON SUMMARY:")
        for label, comp in baselines['comparisons'].items():
            if 'delta' in comp:
                clust = comp['delta'].get('clustering_coefficient', {})
                if clust:
                    print(f"    {label}: clustering ratio (GM/ER) = {clust.get('ratio', 'N/A'):.3f}")

    return master


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GrokMirror Full Analysis')
    parser.add_argument('--n', type=int, default=500,
                        help='Primary graph size (default: 500)')
    parser.add_argument('--full', action='store_true',
                        help='Include growth dynamics (slower)')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    outdir = args.output_dir or os.path.join(os.path.dirname(__file__), 'results')
    run_all(n=args.n, full=args.full, output_dir=outdir)
