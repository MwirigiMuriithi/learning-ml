"""
# 23 · K-Means Clustering
**Family:** Unsupervised · Clustering (centroid-based)

## Definition
K-means partitions n points into k clusters by repeatedly (1) assigning each point to its nearest **centroid** and (2) moving each centroid to the mean of the points
assigned to it (Lloyd's algorithm). It minimizes the within-cluster sum of squares (inertia).

## Intuition
Place k pins on the map at random. Every customer goes to the nearest pin; then move each pin to the centre of its customers; the borders (Voronoi cells) shift, so repeat.
It always converges, but maybe to a *local* optimum, so initialisation matters (k-means++ spreads the starting pins).

## The math
    assign:  cᵢ = argmin_k ||xᵢ − μ_k||²              update:  μ_k = mean{ xᵢ : cᵢ = k }
    objective (inertia): J = Σᵢ ||xᵢ − μ_{cᵢ}||²   (never increases; guaranteed to converge)
    Choosing k: elbow method (where J stops dropping fast), silhouette score, gap statistic, or business needs.
    k-means++: first centre uniformly at random, each next centre sampled with probability ∝ D(x)² (distance to nearest chosen centre).

## Practical tips
- Scale features; assumes round, similarly sized clusters; sensitive to outliers (use k-medoids / GMM / DBSCAN otherwise).
- Use for customer segmentation, colour quantization, vector quantization, initialising other models.

## What the animation shows
Mall customers (annual income vs spending score) with a deliberately bad random start. Each iteration has two frames: **assign** (points recolour) then **update** (centroids jump to the
cluster means, arrows show the move). Right: inertia plummets then flat-lines. The console prints the elbow curve and compares k-means++.

## Try this
- Set k=3 and k=8 and look at the segments.
- Run 20 random restarts and keep the best inertia.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def assign(X, C): return ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)


def inertia(X, C, l): return ((X - C[l]) ** 2).sum()


def lloyd(X, C, iters=30):
    hist = [(C.copy(), assign(X, C), "init")]
    for _ in range(iters):
        l = assign(X, C); hist.append((C.copy(), l, "assign")); Cn = np.array([X[l == k].mean(0) if (l == k).any() else C[k] for k in range(len(C))]); hist.append((Cn.copy(), l, "update"))
        if np.allclose(Cn, C): break
        C = Cn
    return hist


def kpp(X, k, rng):
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1): d = ((X[:, None] - np.array(C)[None]) ** 2).sum(-1).min(1); C.append(X[rng.choice(len(X), p=d / d.sum())])
    return np.array(C)


def main(gif, quick):
    X, _ = data.mall_customers(300, seed=1); rng = np.random.default_rng(4); k = 5
    el = [min(inertia(X, h[0], h[1]) for h in [lloyd(X, X[rng.choice(len(X), kk, replace=False)])[-1]] for _ in range(3)) for kk in range(1, 9)]
    print("  elbow (inertia for k=1..8):", [int(e) for e in el])
    bad = np.array([X[np.argmin(X[:, 0] + X[:, 1])], X[np.argmin(X[:, 0] + X[:, 1])] + [3, 4], X[np.argmax(X[:, 0] + X[:, 1])], X[np.argmax(X[:, 0] + X[:, 1])] - [3, 4], X[np.argmax(X[:, 0])]])
    H = lloyd(X, bad); Hp = lloyd(X, kpp(X, k, np.random.default_rng(1))); print(f"  bad start final inertia {inertia(X, H[-1][0], H[-1][1]):.0f} vs k-means++ {inertia(X, Hp[-1][0], Hp[-1][1]):.0f}  ({len(H) // 2} vs {len(Hp) // 2} iterations)")
    J = [inertia(X, c, l) for c, l, _ in H]; fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.5)

    def update(i):
        a1.clear(); a2.clear(); C, l, ph = H[i]; viz.scatter(a1, X, l, s=18)
        for kk in range(k): a1.scatter(*C[kk], marker="X", s=190, color=viz.PALETTE[kk], edgecolor="k", linewidth=1.5, zorder=6)
        if ph == "update":
            for kk in range(k): a1.annotate("", C[kk], H[i - 1][0][kk], arrowprops=dict(arrowstyle="->", lw=1.5))
        a1.set(title=f"iteration {(i + 1) // 2}: {ph.upper()} step", xlabel="annual income (k$)", ylabel="spending score", xlim=(0, 105), ylim=(0, 105))
        a2.plot(J[:i + 1], "o-", color=viz.PALETTE[3], ms=3); a2.set(xlim=(0, len(H)), ylim=(0, max(J) * 1.05), title="Inertia (within-cluster sum of squares)", xlabel="half-step")
    viz.animate(fig, update, range(len(H) if not quick else min(len(H), 12)), gif, fps=2)


if __name__ == "__main__":
    run(main, __file__)
