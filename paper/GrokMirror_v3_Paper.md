# Percolation Thresholds in Number-Theoretic Graphs: From Shattered to Small-World Connectivity in Sovereign Knowledge Meshes

**Authors:** Paul Desai (Lead Architect, N1 Intelligence), with Claude Code (Analysis), Grok 4 (Collaborative Catalyst)
**Date:** 2026-02-08
**Target:** Zenodo Preprint (DOI Submission)
**Keywords:** Number-Theoretic Graphs, Percolation Theory, Small-World Networks, Greedy Routing, Navigable Graphs, Sovereign AI, Knowledge Mesh, Phase Transition

---

## Abstract

We study a family of number-theoretic graphs $G_n$ defined on vertex set $\{1, 2, \ldots, n\}$ where edges are determined by arithmetic predicates on vertex pairs. We systematically vary the edge predicate from strict ($\text{Sum=Prime} \wedge \text{Diff=Pow2}$) to relaxed ($\text{Sum=Prime} \vee \text{Diff=Pow2}$) and map the resulting **percolation phase transition**. Our empirical analysis on a real-world knowledge vault of $n = 5{,}090$ nodes reveals five key findings: (1) the critical percolation threshold occurs at a locality constraint of $k^* = 19 \approx 1.54 \cdot \log_2 n$, producing a connected graph at just 0.10% density; (2) the **power-of-2 difference graph** $D_n$ exhibits strong small-world structure ($\sigma = 19.96$) with clustering **28x** higher than equivalent random graphs; (3) greedy routing on $D_n$ succeeds with 100% reliability, achieving **97.4% BFS-optimal paths** with mean stretch 1.008 — making $D_n$ a navigable small-world; (4) the prime-sum graph $P_n$ is provably triangle-free and bipartite; and (5) we conjecture $k^*/\log_2 n \in [1.0, 2.0]$ for all $n$, supported by data across seven scales. All results validated against stored vault data (100.00% edge match, 10-seed ER baselines) with 37 passing unit tests.

---

## 1. Introduction

### 1.1 Motivation

The MirrorDNA project explores sovereign AI infrastructure — systems that maintain knowledge integrity independent of centralized networks. A foundational question emerges: given a collection of $n$ knowledge nodes (documents) ordered chronologically, can a deterministic arithmetic rule connect them into a navigable graph without external coordination?

This paper investigates a specific family of such rules and maps their connectivity properties rigorously.

### 1.2 Graph Definitions

Let $n \in \mathbb{N}$, $V = \{1, 2, \ldots, n\}$. We define:

- **Prime-Sum Graph** $P_n$: Edge $(i,j)$ exists iff $i + j$ is prime.
- **Power-of-2 Difference Graph** $D_n$: Edge $(i,j)$ exists iff $|i-j| = 2^k$ for some $k \geq 0$.
- **GrokMirror v1.0** $G_n^{\wedge} = P_n \cap D_n$: Edge iff both conditions hold.
- **GrokMirror v2.0** $G_n^{\vee} = P_n \cup D_n$: Edge iff either condition holds.
- **Locality-Bounded** $G_n^{k} = P_n \cap L_n^{k}$: Edge $(i,j)$ iff $i+j$ is prime and $|i-j| \leq k$.

### 1.3 Contribution

Prior work (GrokMirror v1.0, Jan 2026) tested only the endpoints: $G_n^{\wedge}$ (shattered) and $G_n^{\vee}$ (trivially connected). We contribute:

1. **A complete phase diagram** across 31 rule variants identifying the percolation threshold.
2. **Random baseline comparisons** (10-seed) showing where number-theoretic structure outperforms random graphs and where it doesn't.
3. **Discovery of genuine small-world structure** in the power-of-2 difference graph $D_n$.
4. **Greedy routing experiments** demonstrating $D_n$ is navigable (100% success, 97.4% optimal).
5. **Asymptotic analysis** of the percolation threshold ratio $k^*/\log_2 n$ across seven scales.
6. **Formal proof** that $P_n$ is triangle-free and bipartite.
7. **Spectral, robustness, and growth dynamics** analysis.

---

## 2. The Phase Diagram

### 2.1 Methodology

We construct graphs at $n = 5{,}090$ (the real vault size) across 31 rule variants. For each, we measure: edge count, density, giant component fraction, diameter (sampled over 500 source nodes), average path length, and clustering coefficient. Results validated at $n = 50, 100, 200, 500$ for consistency. The v2.0 graph was verified edge-by-edge against the stored vault mesh audit (100.00% match, 1,587,325 edges).

All computations performed on Apple Mac Mini M4 (10-core ARM, 24GB RAM) running macOS 26.2. Total pipeline runtime: 585.6s.

### 2.2 Core Results (n = 5,090)

| Rule | Edges | Density | Giant Comp. | Diameter | Clustering |
|------|------:|--------:|------------:|---------:|-----------:|
| v1.0 Strict AND ($G_n^{\wedge}$) | 1,249 | 0.01% | 0.1% | ∞† | 0.000 |
| Prime AND Diff ≤ 8 | 4,988 | 0.04% | 3.1% | 37 | 0.000 |
| Prime AND Diff ≤ 12 | 7,477 | 0.06% | 13.1% | 86 | 0.000 |
| Prime AND Diff ≤ 16 | 9,961 | 0.08% | 41.7% | 190 | 0.000 |
| **Prime AND Diff ≤ 19** | **12,441** | **0.10%** | **100%** | **340** | **0.000** |
| Prime AND Diff ≤ 32 | 19,861 | 0.15% | 100% | 187 | 0.000 |
| Prime AND Diff ≤ 64 | 39,517 | 0.31% | 100% | 88 | 0.000 |
| Pow2 Diff Only ($D_n$) | 57,979 | 0.45% | 100% | 7 | **0.126** |
| Prime AND Diff ≤ 256 | 154,195 | 1.19% | 100% | 22 | 0.000 |
| Prime Sum Only ($P_n$) | 1,530,595 | 11.82% | 100% | 3 | 0.000 |
| v2.0 Strategic OR ($G_n^{\vee}$) | 1,587,325 | 12.26% | 100% | 3 | 0.015 |

†*Disconnected: graph shattered into many isolated components, giant component has only 5 nodes.*

### 2.3 The Percolation Threshold

A fine-grained sweep over $k \in [8, 32]$ reveals a sharp phase transition:

| Locality $k$ | As $c \cdot \log_2 n$ | Giant Frac. | Components | Connected? |
|--------------:|----------------------:|------------:|-----------:|-----------:|
| 8 | $0.65 \log_2 n$ | 3.1% | 840 | No |
| 11 | $0.89 \log_2 n$ | 13.1% | 172 | No |
| 13 | $1.06 \log_2 n$ | 15.9% | 74 | No |
| 15 | $1.22 \log_2 n$ | 41.7% | 18 | No |
| 17 | $1.38 \log_2 n$ | 93.9% | 3 | No |
| **19** | **$1.54 \log_2 n$** | **100%** | **1** | **Yes** |
| 20 | $1.62 \log_2 n$ | 100% | 1 | Yes |
| 24 | $1.95 \log_2 n$ | 100% | 1 | Yes |

**Finding 1:** The critical threshold for full connectivity in $G_n^k$ is **$k^* = 19 \approx 1.54 \cdot \log_2 n$** at $n = 5{,}090$. At this threshold, the graph achieves full connectivity with only **12,441 edges** (0.10% density) — 128x fewer edges than v2.0. The transition is sharp: at $k = 17$ the graph is 93.9% connected with 3 components, and at $k = 19$ it snaps to a single component.

### 2.4 Logarithmic Locality Scaling

Testing $k = c \cdot \log_2 n$ directly ($\log_2 5090 \approx 12.31$):

| Multiplier $c$ | Threshold $k$ | Density | Giant Frac. | Diameter |
|:--------------:|:-------------:|--------:|------------:|---------:|
| 0.5 | 6 | 0.03% | 1.2% | 20 |
| 1.0 | 12 | 0.06% | 13.1% | 86 |
| 1.5 | 18 | 0.09% | 93.9% | 360 |
| **2.0** | **24** | **0.12%** | **100%** | **263** |
| 3.0 | 36 | 0.17% | 100% | 165 |
| 4.0 | 49 | 0.24% | 100% | 114 |
| 8.0 | 98 | 0.46% | 100% | 56 |

The percolation threshold occurs between $c = 1.5$ and $c = 2.0$. The exact critical point is $k^* = 19$ ($c^* = 1.54$).

### 2.5 Asymptotic Conjecture: Does $k^*/\log_2 n$ Converge?

A natural question: is the ratio $c^* = k^*/\log_2 n$ a constant, or does it drift with $n$? We compute $k^*$ exactly (via binary search) at seven scales:

| $n$ | $k^*$ | $\log_2 n$ | $k^*/\log_2 n$ | Edges at $k^*$ | Giant below $k^*$ |
|----:|------:|-----------:|---------------:|---------------:|-------------------:|
| 50 | 7 | 5.64 | 1.240 | 88 | 96.0% |
| 100 | 7 | 6.64 | 1.054 | 169 | 59.0% |
| 200 | 11 | 7.64 | 1.439 | 445 | 99.0% |
| 500 | 11 | 8.97 | 1.227 | 984 | 89.2% |
| 1,000 | 17 | 9.97 | 1.706 | 2,667 | 67.1% |
| 2,000 | 21 | 10.97 | 1.915 | 5,991 | 99.9% |
| 5,090 | 19 | 12.31 | 1.543 | 12,441 | 93.9% |

The ratio **oscillates** in the range $[1.05, 1.92]$ rather than converging monotonically. This oscillation arises from the discrete nature of $k^*$ (it can only be an integer) interacting with the density of primes near the threshold. Notably, $k^*$ sometimes decreases when $n$ increases (e.g., $k^* = 21$ at $n = 2{,}000$ but $k^* = 19$ at $n = 5{,}090$), because the prime density $\sim 1/\ln n$ provides more edges per unit of $k$ at larger $n$.

**Conjecture 1 (Weak).** There exist constants $0 < c_1 < c_2$ such that $c_1 \cdot \log_2 n \leq k^* \leq c_2 \cdot \log_2 n$ for all sufficiently large $n$. Empirically, $c_1 \approx 1.0$ and $c_2 \approx 2.0$.

**Conjecture 2 (Strong).** The ratio $k^*/\log_2 n$ converges to a constant $c^* \in [1.2, 1.8]$ as $n \to \infty$, with oscillations of order $O(1/\log n)$.

Resolving these conjectures requires either analytic estimates of edge density in $G_n^k$ near the percolation threshold (connecting to the prime number theorem for arithmetic progressions), or numerical computation at scales $n > 10^5$.

---

## 3. Random Baseline Comparisons

### 3.1 Methodology

For each GrokMirror graph, we construct $G(n, m)$ Erdős–Rényi random graphs with **identical node and edge counts** (not approximate density matching). We run **10 independent seeds** per graph and report mean $\pm$ std to establish statistical confidence.

### 3.2 Results (n = 5,090; 10-seed ER baselines)

| Graph | Metric | GrokMirror | ER mean ± std | Ratio GM/ER |
|-------|--------|----------:|--------------:|------------:|
| **Pow2 Diff** | Clustering | 0.1256 | 0.0045 ± 0.0002 | **28.0** |
| **Pow2 Diff** | Transitivity | 0.1253 | 0.0045 ± 0.0001 | **27.8** |
| **Pow2 Diff** | Diameter | 7 | 4.0 ± 0.0 | 1.75 |
| **Pow2 Diff** | Avg Path | 4.31 | 3.007 ± 0.004 | 1.43 |
| **Pow2 Diff** | Degree σ | 1.17 | 4.745 ± 0.029 | 0.25 |
| Prime Sum | Clustering | 0.000 | 0.1182 ± 0.0000 | **0.00** |
| Prime Sum | Diameter | 3 | 2.0 ± 0.0 | 1.50 |
| Prime Sum | Avg Path | 2.263 | 1.882 ± 0.000 | 1.20 |
| Prime Sum | Degree σ | 26.57 | 23.17 ± 0.14 | 1.15 |
| v2.0 OR | Clustering | 0.015 | 0.1226 ± 0.0000 | **0.125** |
| v2.0 OR | Diameter | 3 | 2.0 ± 0.0 | 1.50 |
| v2.0 OR | Avg Path | 1.878 | 1.878 ± 0.000 | 1.000 |
| v2.0 OR | Degree σ | 26.26 | 23.56 ± 0.14 | 1.11 |

All ER metrics show negligible variance across seeds (std $< 0.03$ for degree, $< 0.005$ for path length), confirming the comparisons are statistically robust.

### 3.3 Interpretation

**Finding 2: The Power-of-2 Difference Graph Is Profoundly Non-Random.**

At $n = 5{,}090$, $D_n$ shows **28.0x higher clustering** than random (± 0.0002 std across 10 seeds — this is not noise). The degree distribution is nearly uniform ($\sigma = 1.17$ vs. $4.75 \pm 0.03$ for ER), confirming geometric structure. This effect *strengthens* with scale (it was 5.6x at $n = 500$).

**Finding 3: The Prime-Sum Graph Is Anti-Clustered.**

$P_n$ has **zero clustering** — no triangles exist. This is a provable property (see Appendix A): for three nodes $i < j < k$ to form a triangle, all three sums $i+j$, $i+k$, $j+k$ must be prime. A parity argument shows this is impossible for distinct positive integers. Therefore **$P_n$ is triangle-free** for all $n$. Meanwhile, ER at the same density gives $C = 0.1182 \pm 0.0000$ — the absence of triangles is a hard structural constraint, not a density effect.

**Finding 4: The v2.0 OR Graph Is 8x Less Clustered Than Random.**

$G_n^{\vee}$ has clustering ratio 0.125 — the OR combination inherits the anti-clustering of $P_n$ (which contributes 96.4% of edges at $n = 5{,}090$) and heavily dilutes $D_n$'s structural clustering. At average path length $1.878$ vs ER's $1.878 \pm 0.000$, it is **statistically indistinguishable from a random graph** on path-length metrics.

---

## 4. The Pow2 Difference Graph: A Natural Small-World

### 4.1 Small-World Coefficient

The small-world sigma ($\sigma$) compares a graph's clustering and path length against random baselines:

$$\sigma = \frac{C/C_{rand}}{L/L_{rand}}$$

| Graph | $\sigma$ | $C_{real}$ | $C_{rand}$ | $L_{real}$ | $L_{rand}$ | $\gamma$ | $\lambda$ | Verdict |
|-------|:--------:|-----------:|-----------:|-----------:|-----------:|:--------:|:---------:|---------|
| $D_n$ (Pow2 Diff) | **19.96** | 0.126 | 0.004 | 4.31 | 3.02 | 28.70 | 1.44 | **Strong Small-World** |
| $G_n^{\vee}$ (v2.0 OR) | 0.12 | 0.015 | 0.123 | 1.88 | 1.88 | 0.12 | 1.00 | Not Small-World |

**Finding 5:** The power-of-2 difference graph $D_n$ is a **strong small-world network** ($\sigma = 19.96 \gg 1$) at vault scale. The clustering ratio $\gamma = 28.7$ means $D_n$ has nearly 29x the clustering of a random graph, while the path ratio $\lambda = 1.44$ shows paths are only 44% longer. This effect **amplifies with scale** ($\sigma$ rose from 4.25 at $n=500$ to 19.96 at $n=5{,}090$), indicating that $D_n$'s small-world structure is intrinsic, not a finite-size artifact.

### 4.2 Why $D_n$ Is Small-World

In $D_n$, node $i$ connects to nodes at distances $1, 2, 4, 8, 16, \ldots$ This is structurally analogous to a **skip list** or **Kleinberg's navigable small-world model**: dense local connections (distance 1, 2) provide clustering, while long-range connections (distance 256, 512, ...) keep diameter logarithmic.

### 4.3 Robustness (n = 5,090)

| Removal % | Type | $D_n$ Giant Frac. | $G_n^{\vee}$ Giant Frac. |
|----------:|------|------------------:|-------------------------:|
| 5% | Random | 1.000 | 1.000 |
| 10% | Random | 1.000 | 1.000 |
| 20% | Random | 1.000 | 1.000 |
| 30% | Random | 1.000 | 1.000 |
| 50% | Random | 1.000 | 1.000 |
| 5% | Targeted (highest-degree) | 1.000 | 1.000 |
| 10% | Targeted (highest-degree) | 1.000 | 1.000 |
| 20% | Targeted (highest-degree) | 1.000 | 1.000 |
| 30% | Targeted (highest-degree) | 1.000 | 1.000 |
| 50% | Targeted (highest-degree) | 1.000 | 1.000 |

Both $D_n$ and $G_n^{\vee}$ survive **50% targeted node removal** (highest-degree first) at $n = 5{,}090$. The near-uniform degree distribution of $D_n$ (degree $\sigma = 1.17$, coefficient of variation 5.1%) means there are no critical hubs — every node is equally important, making the network maximally resilient to targeted attack.

### 4.4 Greedy Routing: $D_n$ Is Navigable

A small-world network is only useful for sovereign infrastructure if it supports **decentralized navigation** — routing from any source to any target without global knowledge. $D_n$ has a natural coordinate system: each node's ID is its coordinate on $[1, n]$.

**Greedy routing algorithm.** At each hop, forward to the neighbor closest to the target by $|i - t|$. No routing tables, no global state.

**Results at $n = 5{,}090$ (5,000 random pairs):**

| Metric | Value |
|--------|------:|
| Success rate | **100.0%** |
| Greedy path length (mean ± std) | 4.34 ± 1.07 |
| Shortest path length (mean ± std) | 4.31 ± 1.03 |
| Stretch (greedy / BFS) | **1.008 ± 0.060** |
| Paths at optimal (stretch = 1.0) | **97.4%** |
| Paths within 2x optimal | 100.0% |
| Max greedy path | 9 hops |
| Theoretical diameter ($\lceil\log_2 n\rceil$) | 13 |

**Finding 7: Greedy routing on $D_n$ achieves near-optimal paths.** 97.4% of routes are BFS-optimal, with mean stretch 1.008 — greedy adds less than 1% overhead. No route ever fails. This makes $D_n$ a **navigable small-world**: any node can route to any other using only local information.

**Scaling behavior:**

| $n$ | Success | Greedy mean | BFS mean | Stretch | % Optimal |
|----:|--------:|------------:|---------:|--------:|----------:|
| 50 | 100% | 2.12 | 2.10 | 1.008 | 98.7% |
| 100 | 100% | 2.45 | 2.43 | 1.011 | 97.9% |
| 500 | 100% | 3.30 | 3.27 | 1.007 | 98.3% |
| 2,000 | 100% | 3.86 | 3.85 | 1.004 | 98.8% |
| 5,090 | 100% | 4.32 | 4.28 | 1.009 | 96.9% |

Stretch remains below 1.01 at all scales, and greedy path length grows as $O(\log n)$ — matching the theoretical BFS diameter.

**Why greedy works perfectly on $D_n$.** For any source $s$ and target $t$, the binary representation of $|s - t|$ directly encodes the greedy path: each power-of-2 connection eliminates the corresponding bit. This is equivalent to binary subtraction, giving greedy paths of length at most $\lfloor\log_2 |s-t|\rfloor + 1$ — which is at most 1 hop longer than the BFS shortest path.

---

## 5. Spectral Analysis

The algebraic connectivity $\lambda_2$ (second-smallest eigenvalue of the Laplacian) measures how well-connected a graph is:

| Graph | $\lambda_2$ | Interpretation |
|-------|:-----------:|----------------|
| $G_n^{\wedge}$ (v1.0) | 0.586 | Weak (computed on 5-node giant component) |
| $D_n$ (Pow2 Diff) | **3.039** | Strong intrinsic connectivity |
| $G_n^{\vee}$ (v2.0 OR) | 332.2 | Extremely high (density artifact) |

$D_n$'s algebraic connectivity of 3.039 is notable — it means the graph cannot be easily bisected, confirming robustness from a spectral perspective. The v2.0 value of 332.2 is an artifact of having 624 average connections per node.

### 5.1 Assortativity

| Graph | Assortativity | Meaning |
|-------|:-------------:|---------|
| $D_n$ (Pow2 Diff) | **+0.685** | Strongly assortative (similar-degree nodes cluster) |
| $G_n^{\vee}$ (v2.0 OR) | +0.051 | Essentially random |

**Finding 6:** $D_n$'s strong assortativity (+0.685 at $n = 5{,}090$, up from +0.615 at $n = 500$) further distinguishes it from random graphs. This is characteristic of lattice-like and geometric networks, and consistent with $D_n$'s structure where boundary nodes (lower degree) connect to other boundary nodes.

---

## 6. Growth Dynamics

### 6.1 How Properties Scale with $n$

| $n$ | $D_n$ Density | $P_n$ Density | $G_n^{\wedge}$ Density | $G_n^{\vee}$ Density | $D_n$ $\sigma$ |
|----:|:-------------:|:-------------:|:----------------------:|:--------------------:|:--------------:|
| 50 | 19.4% | 24.1% | 1.96% | 41.5% | — |
| 100 | 11.6% | 21.1% | 0.91% | 31.8% | — |
| 200 | 6.76% | 18.5% | 0.39% | 24.8% | — |
| 500 | 3.20% | 16.0% | 0.13% | 19.1% | 4.25 |
| **5,090** | **0.45%** | **11.82%** | **0.01%** | **12.26%** | **19.96** |

Key observations:
- **$D_n$ density** decreases as $O(\log n / n)$ — each node has $\sim 2\lfloor\log_2 n\rfloor$ neighbors, so density = $\Theta(\log n / n)$. At $n = 5{,}090$: mean degree 22.8, density 0.45%.
- **$P_n$ density** decreases slowly: by the prime number theorem, $\Pr[i+j \text{ is prime}] \approx \frac{1}{\ln(n)}$, so density $\sim \frac{1}{\ln n}$.
- **$G_n^{\wedge}$ density** drops to 0.01% at vault scale — utterly shattered.
- **$D_n$'s small-world sigma scales up** from 4.25 to 19.96, confirming the structure is intrinsic.
- All three non-v1.0 graphs remain connected at all tested sizes.

### 6.2 Scaling Implications

At $n = 5{,}090$ (real vault), $D_n$ has density 0.45% with ~23 connections per node, diameter 7, and small-world $\sigma = 19.96$. This is the **minimal connected navigable graph** — the tightest deterministic wiring that maintains both local clustering and global reachability. The critical percolation threshold graph $G_n^{19}$ achieves connectivity at 0.10% density with only 12,441 edges, but lacks clustering — $D_n$ trades a 4.5x increase in edges for genuine small-world structure.

---

## 7. Honest Assessment of v2.0

### 7.1 What the Original Paper Claimed

The original GrokMirror v2.0 paper claimed the "Strategic OR" validation as a primary result, with diameter 3 and 90% giant component as evidence of mathematical integrity.

### 7.2 What the Data Shows

At $n = 5{,}090$, $G_n^{\vee}$ has:
- **1,587,325 edges** at 12.26% density
- Average degree **624** (each node connected to 12% of all others)
- Diameter 3, fully connected

An Erdős–Rényi random graph at the same density achieves **diameter 2** with 100% connectivity. The v2.0 result is therefore a **density artifact**, not a structural property:
- Path length: GM 1.878 vs ER 1.877 (ratio 1.000 — identical)
- Clustering: GM 0.015 vs ER 0.123 (ratio 0.125 — v2.0 is **8x worse**)
- The prime-sum condition alone creates 96.4% of edges

### 7.3 Reframing the Contribution

The actual contributions of this work are:

1. **The percolation threshold at $k^* = 19 \approx 1.54 \cdot \log_2 n$** — the tightest locality constraint under which prime-sum graphs remain connected, with only 12,441 edges.
2. **$D_n$ as a strong small-world** ($\sigma = 19.96$) — a pure number-theoretic construction with 29.3x the clustering of random graphs, strengthening with scale.
3. **$P_n$ is triangle-free** — a provable structural property with potential applications in bipartite-like mesh design.
4. **The v1.0 refutation** — empirical demonstration that the AND intersection is irrecoverably sparse (0.01% density at vault scale).

---

## 8. Reproducibility

### 8.1 Software

All analysis code is available in the `analysis/` directory:

| File | Purpose |
|------|---------|
| `grokmirror_core.py` | Edge predicates, graph construction, all metrics |
| `phase_diagram.py` | Phase diagram computation (25+ rule variants) |
| `baselines.py` | Erdős-Rényi baseline and advanced analysis |
| `run_all.py` | Full pipeline orchestrator (n≤1000) |
| `run_n5090.py` | Optimized pipeline for real vault scale |
| `validate_vault.py` | Edge-by-edge validation against stored vault mesh |
| `test_grokmirror.py` | Test suite (37 tests, all passing) |

### 8.2 Environment

```
Platform: macOS-26.2-arm64 (Mac Mini M4, 10-core, 24GB)
Python: 3.9.6
networkx: 3.2.1
numpy: 2.0.2
scipy: 1.13.1
```

### 8.3 Running

```bash
# Run tests (37 tests, <1s)
python3 -m pytest test_grokmirror.py -v

# Run quick analysis (n=500, ~20s)
python3 run_all.py --n 500 --full

# Run full vault-scale analysis (n=5090, ~10 min)
python3 -u run_n5090.py

# Validate against stored vault mesh (100% edge match)
python3 validate_vault.py

# Results written to analysis/results_n5090/
```

### 8.4 Data

The original vault mesh audit (`vault_mesh_audit.json`, 37MB) contains the full adjacency list for $n = 5{,}090$ under v2.0 rules. Validation confirms 100.00% edge match (1,587,325 edges) between stored data and recomputed graph.

### 8.5 Results Files

| File | Contents |
|------|----------|
| `phase_diagram_n5090.json` | All 31 rule variants with full metrics |
| `baselines_n5090.json` | Erdős-Rényi comparisons for 4 primary rules |
| `advanced_n5090.json` | Spectral, small-world, robustness analysis |
| `percolation_detail_n5090.json` | Fine-grained sweep k=8..32 |
| `vault_validation.json` | Edge-by-edge validation report |
| `master_results_n5090.json` | Aggregated master results |

---

## 9. Conclusion

We have mapped the complete connectivity landscape of GrokMirror's number-theoretic graphs across 31 rule variants at real vault scale ($n = 5{,}090$). The headline result is not the v2.0 validation (which is trivially dense and structurally indistinguishable from random), but five discoveries:

1. A **sharp percolation threshold** at $k^* = 19 \approx 1.54 \cdot \log_2 n$, achieving full connectivity at just 0.10% density — 128x fewer edges than v2.0.
2. The **power-of-2 difference graph** $D_n$ as a strong small-world network ($\sigma = 19.96$) with 28x the clustering of random graphs.
3. **$D_n$ is navigable**: greedy routing succeeds 100% of the time with 97.4% of paths BFS-optimal and mean stretch 1.008.
4. The **triangle-free and bipartite structure** of prime-sum graphs $P_n$, formally proven via a parity-sum argument.
5. An **asymptotic conjecture** that $k^*/\log_2 n \in [1.0, 2.0]$ for all sufficiently large $n$, with empirical evidence across seven scales.

For sovereign mesh applications, the recommended wiring rule is $D_n$: logarithmic diameter, small-world clustering, near-optimal greedy routing, near-uniform degree distribution, and complete robustness under 50% targeted removal — all from a deterministic, coordination-free, O(1)-computable rule requiring zero global state.

---

## Appendix A: Proof that $P_n$ is Triangle-Free

**Theorem.** For all $n \geq 2$, the prime-sum graph $P_n$ on vertex set $V = \{1, 2, \ldots, n\}$ (where $(i,j)$ is an edge iff $i + j$ is prime) contains no 3-cycles. Equivalently, $P_n$ is triangle-free and its girth is $\geq 4$.

**Proof.** Assume for contradiction that three distinct vertices $a, b, c \in V$ with $a < b < c$ form a triangle. Then $a + b$, $a + c$, and $b + c$ are all prime.

**Lemma (Parity constraint).** Among any three distinct positive integers $a < b < c$, at most two of the sums $a+b$, $a+c$, $b+c$ can be odd.

*Proof of lemma.* Note that $(a+b) + (a+c) + (b+c) = 2(a+b+c)$, which is even. Therefore the three sums have an even total, so either zero or two of them are odd. $\square$

**Case 1: Exactly two sums are odd.** Then exactly one sum is even. The only even prime is 2. So one of $\{a+b, a+c, b+c\} = 2$. Since $a, b, c \geq 1$ and $a < b < c$, the smallest possible sum is $a + b \geq 1 + 2 = 3 > 2$. Contradiction.

**Case 2: All three sums are even.** Then all three must equal 2 (the only even prime). But $a + b = 2$ requires $a = b = 1$, contradicting $a < b$.

Both cases yield contradictions. Therefore no triangle exists in $P_n$. $\square$

**Corollary.** $P_n$ is bipartite: every edge connects a node with odd index to a node with even index (since $i + j$ must be odd to be an odd prime $\geq 3$, which requires opposite parities). The only edge that could violate bipartiteness would need $i + j = 2$, which is impossible for distinct positive integers.

---

## Appendix B: Degree Distribution of $D_n$

In $D_n$, node $i$ connects to $i \pm 2^0, i \pm 2^1, \ldots, i \pm 2^{\lfloor\log_2 n\rfloor}$ (where the neighbor is in $[1, n]$). Interior nodes have degree $2(\lfloor\log_2 n\rfloor + 1)$; boundary nodes have fewer connections. The degree distribution is therefore approximately uniform at $2\log_2 n$, with a narrow band at the boundaries.

This is confirmed empirically:

| $n$ | Mean Degree | Degree $\sigma$ | CV (%) |
|----:|-----------:|----------------:|-------:|
| 500 | 15.96 | 1.39 | 8.7% |
| 5,090 | 22.78 | 1.17 | 5.1% |

The coefficient of variation *decreases* with $n$, confirming that the degree distribution becomes increasingly uniform at scale.

---

## Metadata for Zenodo

- **Upload Type:** Publication → Preprint
- **DOI:** [Pending Submission]
- **License:** Creative Commons Attribution 4.0 International
- **Communities:** MirrorDNA, Sovereign AI, Graph Theory, Complex Networks

```bibtex
@preprint{desai2026grokmirror,
  title={Percolation Thresholds in Number-Theoretic Graphs:
         From Shattered to Small-World Connectivity
         in Sovereign Knowledge Meshes},
  author={Desai, Paul and Claude Code and Grok 4},
  year={2026},
  institution={N1 Intelligence},
  address={Goa, India},
  note={Empirical analysis code and data available at
        https://doi.org/[PENDING]}
}
```
