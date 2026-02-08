"""
GrokMirror Core — Graph construction and edge predicates.

Builds the number-theoretic graphs used in the Grok-Mirror analysis.
All rule variants are defined here for consistency across analysis and tests.
"""

import math
import numpy as np
import networkx as nx
from functools import lru_cache
from typing import Callable


# ── Primality ──────────────────────────────────────────────────────────

def prime_sieve(limit: int) -> np.ndarray:
    """Sieve of Eratosthenes. Returns boolean array where sieve[i] = True if i is prime."""
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False
    return sieve


def pow2_set(limit: int) -> set:
    """Returns set of all powers of 2 up to limit."""
    result = set()
    p = 1
    while p <= limit:
        result.add(p)
        p *= 2
    return result


# ── Edge Predicates ────────────────────────────────────────────────────

class EdgeRules:
    """All GrokMirror edge rule variants."""

    def __init__(self, n: int):
        self.n = n
        self.sieve = prime_sieve(2 * n + 1)
        self.pow2s = pow2_set(n)

    def v1_strict_and(self, i: int, j: int) -> bool:
        """v1.0: Edge if (i+j is prime) AND (|i-j| is power of 2)."""
        return bool(self.sieve[i + j]) and (abs(i - j) in self.pow2s)

    def v2_strategic_or(self, i: int, j: int) -> bool:
        """v2.0: Edge if (i+j is prime) OR (|i-j| is power of 2)."""
        return bool(self.sieve[i + j]) or (abs(i - j) in self.pow2s)

    def prime_sum_only(self, i: int, j: int) -> bool:
        """Edge if (i+j is prime)."""
        return bool(self.sieve[i + j])

    def pow2_diff_only(self, i: int, j: int) -> bool:
        """Edge if (|i-j| is power of 2)."""
        return abs(i - j) in self.pow2s

    def diff_leq_k(self, k: int):
        """Factory: Edge if (i+j is prime) AND (|i-j| <= k)."""
        def rule(i: int, j: int) -> bool:
            return bool(self.sieve[i + j]) and abs(i - j) <= k
        rule.__name__ = f"prime_and_diff_leq_{k}"
        return rule

    def diff_leq_logn_mult(self, multiplier: float):
        """Factory: Edge if (i+j is prime) AND (|i-j| <= multiplier * log2(n))."""
        threshold = int(multiplier * math.log2(self.n))
        def rule(i: int, j: int) -> bool:
            return bool(self.sieve[i + j]) and abs(i - j) <= threshold
        rule.__name__ = f"prime_and_diff_leq_{multiplier}log2n"
        rule.threshold = threshold
        return rule

    def near_prime_sum(self, tolerance: int):
        """Factory: Edge if min distance from (i+j) to a prime <= tolerance."""
        def rule(i: int, j: int) -> bool:
            s = i + j
            for offset in range(tolerance + 1):
                if s + offset <= 2 * self.n and self.sieve[s + offset]:
                    return True
                if s - offset >= 2 and self.sieve[s - offset]:
                    return True
            return False
        rule.__name__ = f"near_prime_tol_{tolerance}"
        return rule

    def prime_and_diff_pow2_relaxed(self, max_pow2_gap: int):
        """Factory: Edge if (i+j is prime) AND (|i-j| is within max_pow2_gap of a power of 2)."""
        def rule(i: int, j: int) -> bool:
            if not self.sieve[i + j]:
                return False
            d = abs(i - j)
            for p in self.pow2s:
                if abs(d - p) <= max_pow2_gap:
                    return True
            return False
        rule.__name__ = f"prime_and_near_pow2_gap_{max_pow2_gap}"
        return rule


# ── Graph Construction ─────────────────────────────────────────────────

def build_graph(n: int, rule: Callable[[int, int], bool], label: str = "") -> nx.Graph:
    """Build graph G_n with nodes [1..n] and edges defined by rule(i,j)."""
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if rule(i, j):
                G.add_edge(i, j)
    G.graph['rule'] = label or getattr(rule, '__name__', 'unknown')
    G.graph['n'] = n
    return G


def build_graph_fast(n: int, rules: EdgeRules, rule_name: str) -> nx.Graph:
    """
    Optimized graph construction using numpy vectorization where possible.
    Handles large n more efficiently than build_graph for known rule types.
    """
    rule_func = getattr(rules, rule_name, None)
    if rule_func is None:
        raise ValueError(f"Unknown rule: {rule_name}")
    return build_graph(n, rule_func, label=rule_name)


def build_erdos_renyi_matched(n: int, target_edges: int, seed: int = 42) -> nx.Graph:
    """Build Erdos-Renyi random graph with same n and approximately same edge count."""
    max_edges = n * (n - 1) // 2
    p = target_edges / max_edges
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    # Relabel to 1-indexed
    mapping = {i: i + 1 for i in range(n)}
    G = nx.relabel_nodes(G, mapping)
    G.graph['rule'] = f'erdos_renyi_p={p:.6f}'
    G.graph['n'] = n
    return G


# ── Metrics ────────────────────────────────────────────────────────────

def graph_metrics(G: nx.Graph, sample_paths: int = 500) -> dict:
    """
    Compute comprehensive metrics for a graph.
    Uses sampling for expensive computations (diameter, avg path length).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    max_edges = n * (n - 1) // 2

    # Components
    components = list(nx.connected_components(G))
    giant = max(components, key=len) if components else set()
    giant_frac = len(giant) / n if n > 0 else 0

    # Degree stats
    degrees = [d for _, d in G.degree()]
    deg_array = np.array(degrees)

    metrics = {
        'n': n,
        'edges': m,
        'density': m / max_edges if max_edges > 0 else 0,
        'num_components': len(components),
        'giant_component_size': len(giant),
        'giant_component_frac': giant_frac,
        'degree_mean': float(deg_array.mean()),
        'degree_std': float(deg_array.std()),
        'degree_min': int(deg_array.min()),
        'degree_max': int(deg_array.max()),
        'degree_median': float(np.median(deg_array)),
    }

    # Only compute path-based metrics on giant component
    if len(giant) > 1:
        G_giant = G.subgraph(giant).copy()

        # Sampled diameter and average path length
        nodes_list = list(giant)
        rng = np.random.RandomState(42)
        sample_nodes = rng.choice(nodes_list, size=min(sample_paths, len(nodes_list)), replace=False)

        max_eccentricity = 0
        total_path_len = 0
        path_count = 0

        for src in sample_nodes:
            lengths = nx.single_source_shortest_path_length(G_giant, src)
            ecc = max(lengths.values())
            if ecc > max_eccentricity:
                max_eccentricity = ecc
            total_path_len += sum(lengths.values())
            path_count += len(lengths) - 1  # exclude self

        metrics['diameter_sampled'] = max_eccentricity
        metrics['avg_path_length_sampled'] = total_path_len / path_count if path_count > 0 else float('inf')

        # Clustering coefficient (sampled for large graphs)
        if n > 2000:
            sample_cc = rng.choice(nodes_list, size=min(1000, len(nodes_list)), replace=False)
            cc_values = [nx.clustering(G_giant, v) for v in sample_cc]
            metrics['clustering_coefficient'] = float(np.mean(cc_values))
        else:
            metrics['clustering_coefficient'] = nx.average_clustering(G_giant)

        # Transitivity (global clustering)
        metrics['transitivity'] = nx.transitivity(G_giant)

    else:
        metrics['diameter_sampled'] = float('inf')
        metrics['avg_path_length_sampled'] = float('inf')
        metrics['clustering_coefficient'] = 0.0
        metrics['transitivity'] = 0.0

    # Is connected
    metrics['is_connected'] = nx.is_connected(G)

    return metrics


def spectral_metrics(G: nx.Graph) -> dict:
    """Compute spectral properties — algebraic connectivity (Fiedler value) and spectral gap."""
    if not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        G = G.subgraph(giant).copy()

    n = G.number_of_nodes()
    if n > 3000:
        # Use sparse computation for large graphs
        from scipy.sparse.linalg import eigsh
        L = nx.laplacian_matrix(G).astype(float)
        # Get the 3 smallest eigenvalues
        eigenvalues = eigsh(L, k=3, which='SM', return_eigenvectors=False)
        eigenvalues = sorted(eigenvalues)
    else:
        L = nx.laplacian_matrix(G).toarray().astype(float)
        eigenvalues = sorted(np.linalg.eigvalsh(L))

    return {
        'algebraic_connectivity': float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0,
        'spectral_gap': float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0,
        'lambda_2': float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0,
        'lambda_3': float(eigenvalues[2]) if len(eigenvalues) > 2 else 0.0,
    }


def robustness_test(G: nx.Graph, removal_fracs: list = None, trials: int = 5) -> dict:
    """
    Test graph robustness under random and targeted node removal.
    Returns giant component fraction after removing each fraction of nodes.
    """
    if removal_fracs is None:
        removal_fracs = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    n = G.number_of_nodes()
    nodes = list(G.nodes())
    rng = np.random.RandomState(42)

    # Random removal
    random_results = {}
    for frac in removal_fracs:
        k = int(n * frac)
        trial_sizes = []
        for t in range(trials):
            remove = rng.choice(nodes, size=k, replace=False)
            H = G.copy()
            H.remove_nodes_from(remove)
            if H.number_of_nodes() > 0:
                giant = max(nx.connected_components(H), key=len)
                trial_sizes.append(len(giant) / (n - k))
            else:
                trial_sizes.append(0.0)
        random_results[frac] = {
            'mean': float(np.mean(trial_sizes)),
            'std': float(np.std(trial_sizes)),
        }

    # Targeted removal (highest degree first)
    degree_sorted = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    targeted_results = {}
    for frac in removal_fracs:
        k = int(n * frac)
        remove = [node for node, deg in degree_sorted[:k]]
        H = G.copy()
        H.remove_nodes_from(remove)
        if H.number_of_nodes() > 0:
            giant = max(nx.connected_components(H), key=len)
            targeted_results[frac] = len(giant) / (n - k)
        else:
            targeted_results[frac] = 0.0

    return {
        'random_removal': random_results,
        'targeted_removal': targeted_results,
    }


def small_world_sigma(G: nx.Graph, nrand: int = 5) -> dict:
    """
    Compute small-world sigma: (C/C_rand) / (L/L_rand).
    sigma > 1 indicates small-world structure.
    Uses sampling for large graphs.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if not nx.is_connected(G):
        giant = max(nx.connected_components(G), key=len)
        G = G.subgraph(giant).copy()
        n = G.number_of_nodes()
        m = G.number_of_edges()

    # Real graph metrics
    C_real = nx.average_clustering(G) if n <= 2000 else float(np.mean(
        [nx.clustering(G, v) for v in np.random.RandomState(42).choice(list(G.nodes()), min(1000, n), replace=False)]
    ))

    # Sampled avg path length
    nodes_list = list(G.nodes())
    rng = np.random.RandomState(42)
    sample = rng.choice(nodes_list, size=min(200, n), replace=False)
    total_len = 0
    count = 0
    for src in sample:
        lengths = nx.single_source_shortest_path_length(G, src)
        total_len += sum(lengths.values())
        count += len(lengths) - 1
    L_real = total_len / count if count > 0 else float('inf')

    # Random graph baselines
    p = 2 * m / (n * (n - 1))
    C_rands = []
    L_rands = []
    for seed in range(nrand):
        R = nx.erdos_renyi_graph(n, p, seed=seed + 100)
        if nx.is_connected(R):
            C_rands.append(nx.average_clustering(R) if n <= 2000 else float(np.mean(
                [nx.clustering(R, v) for v in rng.choice(list(R.nodes()), min(500, n), replace=False)]
            )))
            sample_r = rng.choice(list(R.nodes()), size=min(200, n), replace=False)
            tl = 0
            ct = 0
            for src in sample_r:
                ls = nx.single_source_shortest_path_length(R, src)
                tl += sum(ls.values())
                ct += len(ls) - 1
            L_rands.append(tl / ct if ct > 0 else float('inf'))

    if not C_rands or not L_rands:
        return {'sigma': float('nan'), 'C_real': C_real, 'L_real': L_real,
                'C_rand': float('nan'), 'L_rand': float('nan')}

    C_rand = float(np.mean(C_rands))
    L_rand = float(np.mean(L_rands))

    gamma = C_real / C_rand if C_rand > 0 else float('inf')
    lam = L_real / L_rand if L_rand > 0 else float('inf')
    sigma = gamma / lam if lam > 0 else float('nan')

    return {
        'sigma': sigma,
        'gamma': gamma,
        'lambda': lam,
        'C_real': C_real,
        'C_rand': C_rand,
        'L_real': L_real,
        'L_rand': L_rand,
    }
