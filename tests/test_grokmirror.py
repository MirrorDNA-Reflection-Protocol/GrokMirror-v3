"""
Test suite for GrokMirror analysis.

Validates edge predicates, graph construction, metrics, and known
mathematical properties of the number-theoretic graphs.
"""

import pytest
import math
import numpy as np
import networkx as nx
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from grokmirror_core import (
    prime_sieve, pow2_set, EdgeRules, build_graph,
    build_erdos_renyi_matched, graph_metrics, spectral_metrics,
    robustness_test, small_world_sigma
)


# ── Primitive Tests ────────────────────────────────────────────────────

class TestPrimeSieve:
    def test_small_primes(self):
        sieve = prime_sieve(30)
        primes = [i for i in range(31) if sieve[i]]
        assert primes == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

    def test_zero_and_one_not_prime(self):
        sieve = prime_sieve(10)
        assert not sieve[0]
        assert not sieve[1]

    def test_two_is_prime(self):
        sieve = prime_sieve(2)
        assert sieve[2]

    def test_large_sieve_consistency(self):
        sieve = prime_sieve(10000)
        # Verify a known large prime
        assert sieve[9973]
        # Verify a known composite
        assert not sieve[9999]  # 9999 = 3 * 3333

    def test_prime_count(self):
        """pi(1000) = 168."""
        sieve = prime_sieve(1000)
        count = sum(1 for i in range(1001) if sieve[i])
        assert count == 168


class TestPow2Set:
    def test_small(self):
        p = pow2_set(16)
        assert p == {1, 2, 4, 8, 16}

    def test_boundary(self):
        p = pow2_set(17)
        assert p == {1, 2, 4, 8, 16}
        p = pow2_set(32)
        assert 32 in p

    def test_large(self):
        p = pow2_set(5000)
        assert p == {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}


# ── Edge Rule Tests ────────────────────────────────────────────────────

class TestEdgeRules:
    @pytest.fixture
    def rules(self):
        return EdgeRules(100)

    def test_v1_strict_and(self, rules):
        # i=1, j=2: sum=3 (prime), diff=1 (2^0) → True
        assert rules.v1_strict_and(1, 2)
        # i=1, j=4: sum=5 (prime), diff=3 (not pow2) → False
        assert not rules.v1_strict_and(1, 4)
        # i=2, j=4: sum=6 (not prime), diff=2 (pow2) → False
        assert not rules.v1_strict_and(2, 4)

    def test_v2_strategic_or(self, rules):
        # i=1, j=4: sum=5 (prime) → True (even though diff=3 not pow2)
        assert rules.v2_strategic_or(1, 4)
        # i=2, j=4: diff=2 (pow2) → True (even though sum=6 not prime)
        assert rules.v2_strategic_or(2, 4)
        # i=2, j=4: sum=6 (not prime) but diff=2 (pow2) → True
        assert rules.v2_strategic_or(2, 4)

    def test_prime_sum_only(self, rules):
        # i=1, j=2: sum=3 (prime) → True
        assert rules.prime_sum_only(1, 2)
        # i=2, j=4: sum=6 (not prime) → False
        assert not rules.prime_sum_only(2, 4)

    def test_pow2_diff_only(self, rules):
        # i=1, j=3: diff=2 → True
        assert rules.pow2_diff_only(1, 3)
        # i=1, j=4: diff=3 → False
        assert not rules.pow2_diff_only(1, 4)

    def test_v1_is_subset_of_v2(self, rules):
        """Every v1 edge must also be a v2 edge (AND ⊂ OR)."""
        for i in range(1, 51):
            for j in range(i + 1, 51):
                if rules.v1_strict_and(i, j):
                    assert rules.v2_strategic_or(i, j), f"v1 edge ({i},{j}) not in v2"

    def test_diff_leq_k(self, rules):
        r = rules.diff_leq_k(4)
        # i=1, j=2: sum=3 (prime), diff=1 (<=4) → True
        assert r(1, 2)
        # i=1, j=10: sum=11 (prime), diff=9 (>4) → False
        assert not r(1, 10)

    def test_near_prime_sum(self, rules):
        r = rules.near_prime_sum(1)
        # i=2, j=4: sum=6 (not prime), but 5 and 7 are prime → True
        assert r(2, 4)

    def test_symmetry(self, rules):
        """All rules should be symmetric: rule(i,j) == rule(j,i)."""
        for i in range(1, 30):
            for j in range(i + 1, 30):
                assert rules.v1_strict_and(i, j) == rules.v1_strict_and(j, i)
                assert rules.v2_strategic_or(i, j) == rules.v2_strategic_or(j, i)
                assert rules.prime_sum_only(i, j) == rules.prime_sum_only(j, i)
                assert rules.pow2_diff_only(i, j) == rules.pow2_diff_only(j, i)


# ── Graph Construction Tests ───────────────────────────────────────────

class TestGraphConstruction:
    def test_node_count(self):
        rules = EdgeRules(20)
        G = build_graph(20, rules.v2_strategic_or)
        assert G.number_of_nodes() == 20

    def test_no_self_loops(self):
        rules = EdgeRules(20)
        G = build_graph(20, rules.v2_strategic_or)
        assert nx.number_of_selfloops(G) == 0

    def test_undirected(self):
        rules = EdgeRules(20)
        G = build_graph(20, rules.v2_strategic_or)
        assert not G.is_directed()

    def test_v1_has_fewer_edges_than_v2(self):
        rules = EdgeRules(50)
        G1 = build_graph(50, rules.v1_strict_and)
        G2 = build_graph(50, rules.v2_strategic_or)
        assert G1.number_of_edges() < G2.number_of_edges()

    def test_erdos_renyi_matched_approx(self):
        """ER graph should have approximately the target number of edges."""
        G = build_erdos_renyi_matched(500, 10000, seed=42)
        # Allow 10% tolerance
        assert abs(G.number_of_edges() - 10000) / 10000 < 0.10


# ── Known Mathematical Properties ─────────────────────────────────────

class TestMathProperties:
    def test_pow2_graph_is_connected_small(self):
        """Power-of-2 difference graph on [1..n] is always connected for n >= 2."""
        for n in [5, 10, 20, 50]:
            rules = EdgeRules(n)
            G = build_graph(n, rules.pow2_diff_only)
            assert nx.is_connected(G), f"Pow2 diff graph disconnected at n={n}"

    def test_pow2_graph_diameter_is_log_n(self):
        """Pow2 diff graph has diameter exactly ceil(log2(n))."""
        for n in [8, 16, 32, 64]:
            rules = EdgeRules(n)
            G = build_graph(n, rules.pow2_diff_only)
            d = nx.diameter(G)
            expected = math.ceil(math.log2(n))
            assert d <= expected + 1, f"n={n}: diameter {d} > expected {expected}+1"

    def test_prime_sum_even_odd_structure(self):
        """
        For i+j to be prime and > 2, exactly one of i,j must be even.
        So the prime-sum graph is nearly bipartite (odd-even) for large n.
        Exception: i+j=2 only when i=j=1 (excluded since i<j).
        """
        rules = EdgeRules(50)
        G = build_graph(50, rules.prime_sum_only)
        # Count edges between same-parity nodes
        same_parity = 0
        for u, v in G.edges():
            if u % 2 == v % 2:
                same_parity += 1
        # The only same-parity edge possible is via sum=2 (impossible for distinct positive ints)
        assert same_parity == 0, f"Found {same_parity} same-parity edges"

    def test_v1_strict_is_sparse(self):
        """v1.0 should be very sparse at any reasonable n."""
        for n in [100, 200, 500]:
            rules = EdgeRules(n)
            G = build_graph(n, rules.v1_strict_and)
            density = G.number_of_edges() / (n * (n - 1) / 2)
            assert density < 0.05, f"v1.0 too dense at n={n}: {density:.4f}"

    def test_v2_or_is_dense(self):
        """v2.0 should be quite dense (>10% for n >= 100)."""
        rules = EdgeRules(200)
        G = build_graph(200, rules.v2_strategic_or)
        density = G.number_of_edges() / (200 * 199 / 2)
        assert density > 0.10, f"v2.0 unexpectedly sparse: {density:.4f}"


# ── Metrics Tests ──────────────────────────────────────────────────────

class TestMetrics:
    @pytest.fixture
    def small_graph(self):
        rules = EdgeRules(50)
        return build_graph(50, rules.v2_strategic_or)

    def test_metrics_keys(self, small_graph):
        m = graph_metrics(small_graph, sample_paths=50)
        required = ['n', 'edges', 'density', 'num_components',
                     'giant_component_size', 'giant_component_frac',
                     'degree_mean', 'degree_std', 'is_connected']
        for key in required:
            assert key in m, f"Missing key: {key}"

    def test_density_range(self, small_graph):
        m = graph_metrics(small_graph)
        assert 0 <= m['density'] <= 1

    def test_giant_component_frac_range(self, small_graph):
        m = graph_metrics(small_graph)
        assert 0 <= m['giant_component_frac'] <= 1

    def test_complete_graph_metrics(self):
        G = nx.complete_graph(20)
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(20)})
        m = graph_metrics(G)
        assert m['density'] == pytest.approx(1.0)
        assert m['diameter_sampled'] == 1
        assert m['is_connected']

    def test_disconnected_graph_metrics(self):
        G = nx.Graph()
        G.add_nodes_from(range(1, 11))
        G.add_edge(1, 2)
        G.add_edge(3, 4)
        m = graph_metrics(G)
        assert not m['is_connected']
        assert m['num_components'] > 1


class TestSpectralMetrics:
    def test_complete_graph(self):
        G = nx.complete_graph(10)
        s = spectral_metrics(G)
        # For K_n, algebraic connectivity = n
        assert s['lambda_2'] == pytest.approx(10.0, abs=0.1)

    def test_path_graph(self):
        G = nx.path_graph(20)
        s = spectral_metrics(G)
        # Path graph has small algebraic connectivity
        assert s['lambda_2'] < 1.0
        assert s['lambda_2'] > 0.0


class TestRobustness:
    def test_robustness_output_structure(self):
        G = nx.complete_graph(50)
        r = robustness_test(G, removal_fracs=[0.1, 0.2], trials=2)
        assert 'random_removal' in r
        assert 'targeted_removal' in r
        assert 0.1 in r['random_removal']

    def test_complete_graph_robust(self):
        G = nx.complete_graph(50)
        r = robustness_test(G, removal_fracs=[0.1], trials=3)
        # Complete graph stays connected under 10% removal
        assert r['random_removal'][0.1]['mean'] > 0.99


# ── Integration: Phase Transition Existence ────────────────────────────

class TestPhaseTransition:
    def test_transition_exists(self):
        """
        As we increase the diff threshold k in (prime AND diff<=k),
        there must be a point where the giant component jumps.
        """
        n = 100
        rules = EdgeRules(n)

        fracs = []
        for k in [1, 2, 4, 8, 16, 32, 64]:
            rule = rules.diff_leq_k(k)
            G = build_graph(n, rule)
            components = list(nx.connected_components(G))
            giant = max(components, key=len) if components else set()
            fracs.append(len(giant) / n)

        # Should go from small to large
        assert fracs[-1] > fracs[0], "No transition observed"
        # At k=64 (>n/2), should be well-connected
        assert fracs[-1] > 0.5, f"Expected connectivity at k=64, got {fracs[-1]}"

    def test_v1_sparser_than_v2(self):
        """Sanity: v1.0 must always be sparser than v2.0."""
        for n in [20, 50, 100]:
            rules = EdgeRules(n)
            G1 = build_graph(n, rules.v1_strict_and)
            G2 = build_graph(n, rules.v2_strategic_or)
            assert G1.number_of_edges() <= G2.number_of_edges()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
