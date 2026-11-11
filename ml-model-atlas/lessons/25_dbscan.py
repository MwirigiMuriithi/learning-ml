"""
# 25 · DBSCAN (Density-Based Clustering)
**Family:** Unsupervised · Clustering (density-based) + outlier detection

## Definition
DBSCAN groups points that are packed closely together and labels isolated points as **noise**. It needs two numbers: a radius **ε** and a minimum neighbourhood size **minPts**.
Clusters can have *any shape*, and the number of clusters is discovered, not chosen.

## Intuition
Think of a crowd at a festival: dense crowds are clusters, lone wanderers are noise. A **core point** has ≥ minPts neighbours within ε. A cluster grows by hopping from core point to core point
(density-reachability); **border points** are within ε of a core point but not dense themselves.

## The math
    N_ε(p) = { q : ||p − q|| ≤ ε }          p is core  ⇔  |N_ε(p)| ≥ minPts
    expand: start at an unvisited core point, give it a new cluster id, and recursively add all points in the neighbourhoods of the core points reached.
    Choosing ε: plot the sorted distance of every point to its k-th nearest neighbour (k = minPts) and pick the "knee" (right panel).
    Complexity O(n log n) with a spatial index. HDBSCAN removes the single-ε limitation (varying densities).

## Practical tips
- Scale features first. Struggles with varying density and high dimensions. Great for geospatial data, anomaly detection and non-convex shapes (k-means fails on moons).

## What the animation shows
Two moons + scattered noise. As the algorithm visits points, the circle shows the ε-neighbourhood being examined. Filled dots = core points, hollow = border points, black × = noise;
colours = clusters, spreading along each moon. Right: the k-distance plot used to pick ε.

## Try this
- Halve ε: clusters shatter. Double it: the moons merge.
- Compare with k-means (k=2) on the same data.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.patches as mp
from mlatlas import data, viz
from mlatlas.cli import run


def dbscan(X, eps, minpts):
    n = len(X); D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1)); nb = [np.where(D[i] <= eps)[0] for i in range(n)]; core = np.array([len(v) >= minpts for v in nb])
    lab = np.full(n, -2); ev = []; cid = 0   # -2 unvisited, -1 noise
    for i in range(n):
        if lab[i] != -2: continue
        if not core[i]: lab[i] = -1; ev.append((i, -1, False)); continue
        lab[i] = cid; ev.append((i, cid, True)); queue = list(nb[i])
        while queue:
            j = queue.pop(0)
            if lab[j] == -1: lab[j] = cid; ev.append((j, cid, core[j]))
            if lab[j] != -2: continue
            lab[j] = cid; ev.append((j, cid, core[j]))
            if core[j]: queue.extend(nb[j])
        cid += 1
    return lab, ev, core, D


def main(gif, quick):
    X, _ = data.moons(240, .07, seed=3); r = np.random.default_rng(0); X = np.r_[X, r.uniform([-1.5, -1], [2.5, 1.5], (25, 2))]; eps, mp_ = .17, 5
    lab, ev, core, D = dbscan(X, eps, mp_); print(f"  found {lab.max() + 1} clusters and {(lab == -1).sum()} noise points")
    kd = np.sort(np.sort(D, 1)[:, mp_ - 1]); idx = viz.pick(len(ev), 14 if quick else 60); fig, (a1, a2) = viz.subplots(1, 2, w=12, h=4.8)

    def update(t):
        a1.clear(); a2.clear(); a1.scatter(*X.T, s=14, color="#ddd"); seen = ev[:t + 1]
        for i, c, isc in seen:
            if c == -1: a1.scatter(*X[i], marker="x", color="k", s=30, zorder=4)
            else: a1.scatter(*X[i], s=34, facecolor=viz.PALETTE[c % 8] if isc else "white", edgecolor=viz.PALETTE[c % 8], linewidth=1.5, zorder=4)
        a1.add_patch(mp.Circle(X[ev[t][0]], eps, fill=False, edgecolor="k", lw=1.5, ls="--")); a1.set_aspect("equal"); a1.set(title=f"visited {t + 1}/{len(ev)} points (ε={eps}, minPts={mp_})", xlim=(-1.6, 2.6), ylim=(-1.1, 1.6))
        a2.plot(kd, color=viz.PALETTE[0]); a2.axhline(eps, color=viz.PALETTE[3], ls="--", label=f"chosen ε = {eps}"); a2.set(title=f"k-distance graph (k={mp_}): the knee suggests ε", xlabel="points sorted by distance", ylabel=f"distance to {mp_}th neighbour"); a2.legend()
    viz.animate(fig, update, idx, gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
