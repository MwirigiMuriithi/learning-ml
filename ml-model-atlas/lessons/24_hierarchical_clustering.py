"""
# 24 · Hierarchical (Agglomerative) Clustering
**Family:** Unsupervised · Clustering (tree-based)

## Definition
Start with every point as its own cluster, then repeatedly **merge the two closest clusters** until only one remains. The sequence of merges forms a tree, the **dendrogram**.
Cutting the tree at a chosen height gives any number of clusters, so you do not need to choose k in advance.

## Intuition
Building a family tree from similarity: the two most similar customers pair up first, then similar pairs merge, and so on. The *height* of a merge is how far apart
the two groups were; a very tall merge signals "these really are different groups", so cut just below it.

## The math
    cluster distance ("linkage"):
      single    d(A,B) = min d(a,b)          (can chain; finds elongated clusters)
      complete  d(A,B) = max d(a,b)          (compact clusters)
      average   d(A,B) = mean d(a,b)
      ward      merge the pair giving the smallest increase in total within-cluster variance:  Δ = |A||B|/(|A|+|B|) ||μ_A − μ_B||²
    naive cost O(n³), O(n² log n) with priority queues. Needs the full distance matrix (so ≈ up to tens of thousands of points).

## Practical tips
- Great for small data, taxonomies, gene expression, and when you want to explore several granularities. Use scaling.
- Ward ≈ k-means-like compact clusters; single linkage for chains.

## What the animation shows
36 customers; each frame performs one merge. Left: current clusters (singletons grey, merged clusters coloured, the newest merge circled). Right: the dendrogram growing from the bottom, with the red link as the latest merge.

## Try this
- Switch `linkage="single"` and see long chain-like merges.
- Draw a horizontal cut at distance h and count clusters.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def agglomerate(X, linkage="ward"):
    n = len(X); D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); cl = {i: [i] for i in range(n)}; merges = []; nxt = n
    def dist(a, b):
        A, B = cl[a], cl[b]
        if linkage == "single": return D[np.ix_(A, B)].min()
        if linkage == "complete": return D[np.ix_(A, B)].max()
        if linkage == "average": return D[np.ix_(A, B)].mean()
        return np.sqrt(2 * len(A) * len(B) / (len(A) + len(B))) * np.linalg.norm(X[A].mean(0) - X[B].mean(0))
    while len(cl) > 1:
        ids = list(cl); best = min(((dist(a, b), a, b) for i, a in enumerate(ids) for b in ids[i + 1:]))
        merges.append((best[1], best[2], best[0])); cl[nxt] = cl.pop(best[1]) + cl.pop(best[2]); nxt += 1
    return merges


def main(gif, quick):
    Xall, _ = data.mall_customers(300, seed=2); X = Xall[np.random.default_rng(0).choice(300, 36, replace=False)]; X = (X - X.mean(0)) / X.std(0); n = len(X); M = agglomerate(X, "ward")
    kids = {n + i: (a, b) for i, (a, b, _) in enumerate(M)}; order = []
    def dfs(v): (order.append(v) if v < n else [dfs(c) for c in kids[v]])
    dfs(2 * n - 2); xpos = {v: i for i, v in enumerate(order)}; ypos = {v: 0. for v in range(n)}
    for i, (a, b, h) in enumerate(M): xpos[n + i] = (xpos[a] + xpos[b]) / 2; ypos[n + i] = h
    cut = M[-4][2] * .999; print(f"  largest merge heights: {[round(float(m[2]), 1) for m in M[-5:]]}  -> a cut at {cut:.1f} gives 4 clusters")
    members = {i: [i] for i in range(n)}; fig, (a1, a2) = viz.subplots(1, 2, w=12, h=4.8)

    def labels(m):
        mem = {i: [i] for i in range(n)}
        for i, (a, b, _) in enumerate(M[:m]): mem[n + i] = mem.pop(a) + mem.pop(b)
        return mem

    def update(m):
        a1.clear(); a2.clear(); mem = labels(m); a1.scatter(*X.T, s=40, color="#bbb", edgecolor="white", zorder=3)
        for j, (cid, idx) in enumerate(c for c in mem.items() if len(c[1]) > 1): a1.scatter(*X[idx].T, s=55, color=viz.PALETTE[j % 8], edgecolor="white", zorder=4)
        if m > 0:
            a, b, h = M[m - 1]; allm = mem[n + m - 1]; a1.scatter(*X[allm].T, s=130, facecolor="none", edgecolor="k", linewidth=1.8, zorder=5)
        a1.set(title=f"after {m} merges: {n - m} clusters", xlabel="income (standardized)", ylabel="spending (standardized)")
        for i, (a, b, h) in enumerate(M[:m]):
            c = "#E45756" if i == m - 1 else "k"; a2.plot([xpos[a], xpos[a], xpos[b], xpos[b]], [ypos[a], h, h, ypos[b]], color=c, lw=2 if c != "k" else 1)
        a2.axhline(cut, color="gray", ls=":"); a2.set(xlim=(-1, n), ylim=(0, M[-1][2] * 1.05), title="Dendrogram (height = merge distance)", xlabel="customers (leaf order)"); a2.set_xticks([])
    viz.animate(fig, update, range(0, n, 3 if quick else 1), gif, fps=5)


if __name__ == "__main__":
    run(main, __file__)
