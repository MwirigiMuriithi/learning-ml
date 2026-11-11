"""
# 07 · k-Nearest Neighbors (k-NN)
**Family:** Supervised · Classification / Regression (instance-based, non-parametric)

## Definition
k-NN does no training at all. To classify a new point it finds the k most similar stored examples (nearest by distance)
and takes a majority vote (classification) or an average (regression). The "model" *is* the training data.

## Intuition
"Tell me who your neighbours are and I'll tell you who you are." A small k follows every quirk of the data (low bias,
high variance); a large k blurs everything toward the global majority (high bias, low variance).

## The math
    d(x, xᵢ) = ||x − xᵢ||₂      (or Manhattan, cosine, ...)
    ŷ(x) = argmax_c  Σ_{i ∈ N_k(x)} 1[yᵢ = c]            regression: ŷ = mean of the k neighbours' y
    Weighted variant: each neighbour votes with weight 1/d.

## Practical tips
- **Scale features** (distance is dominated by large-range features). Beware the curse of dimensionality.
- Prediction cost grows with the dataset (use KD-trees / approximate NN such as FAISS or HNSW).
- k chosen by cross-validation; odd k avoids ties in binary problems.

## What the animations show
1. `07_k_nearest_neighbors.gif`: the decision regions for k = 1 … 61 on noisy moons, plus train/test accuracy. k=1 is
   a jagged island map with 100 % training accuracy; huge k is too smooth.
2. `07_k_nearest_neighbors_query.gif`: a query point glides through the data; lines connect it to its 5 neighbours who vote.

## Try this
- Multiply one feature by 100 before computing distances: watch the model ignore the other feature.
- Implement distance-weighted voting and compare accuracy at k = 15.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def votes(order, ytr, k, K=2):
    nn = ytr[order[:, :k]]; cnt = np.stack([(nn == c).sum(1) for c in range(K)], 1)
    return cnt.argmax(1), cnt[:, 1] / k


def main(gif, quick):
    Xtr, ytr = data.moons(200, .35, seed=1); Xte, yte = data.moons(200, .35, seed=2)
    xx, yy, G = viz.grid(Xtr, n=70 if quick else 100)
    sq = lambda A, B: ((A[:, None, :] - B[None]) ** 2).sum(-1)
    og, ot, otr = np.argsort(sq(G, Xtr), 1), np.argsort(sq(Xte, Xtr), 1), np.argsort(sq(Xtr, Xtr), 1)
    ks = [1, 3, 5, 9, 15, 25, 41, 61]; allk = list(range(1, 62, 2))
    acc_te = [(votes(ot, ytr, k)[0] == yte).mean() for k in allk]; acc_tr = [(votes(otr, ytr, k)[0] == ytr).mean() for k in allk]
    print(f"  best k = {allk[int(np.argmax(acc_te))]} (test accuracy {max(acc_te):.2f})")
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4)

    def update(k):
        a1.clear(); a2.clear(); _, P = votes(og, ytr, k); viz.show_map(a1, xx, yy, P, prob=True)
        a1.contour(xx, yy, P.reshape(xx.shape), levels=[.5], colors="k", linewidths=1.5); viz.scatter(a1, Xtr, ytr)
        a1.set(title=f"k = {k}: decision regions", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
        a2.plot(allk, acc_tr, color=viz.PALETTE[2], label="train"); a2.plot(allk, acc_te, color=viz.PALETTE[3], label="test")
        a2.axvline(k, color="k", ls="--"); a2.set(title="Accuracy vs k (small k overfits, large k underfits)", xlabel="k", ylim=(.6, 1.02)); a2.legend()
    viz.animate(fig, update, ks, gif, fps=2)
    # ---- animation 2: one query, five neighbours
    k = 5; T = 18 if quick else 45; path = np.c_[np.linspace(-1.2, 2.2, T), .9 * np.sin(np.linspace(0, 2 * np.pi, T)) * .6 + .25]
    fig, ax = viz.subplots(1, 1, w=6.4, h=4.6)

    def update2(t):
        ax.clear(); q = path[t]; d = ((Xtr - q) ** 2).sum(1); nn = np.argsort(d)[:k]; lab = np.bincount(ytr[nn], minlength=2).argmax()
        viz.scatter(ax, Xtr, ytr, s=14)
        for i in nn: ax.plot([q[0], Xtr[i, 0]], [q[1], Xtr[i, 1]], color="k", lw=1.2, alpha=.7)
        ax.scatter(*q, marker="*", s=260, color=viz.PALETTE[lab], edgecolor="k", linewidth=1.5, zorder=6)
        ax.set(title=f"Query's {k} neighbours vote: {np.bincount(ytr[nn], minlength=2)}  ->  class {lab}", xlim=(-1.9, 2.9), ylim=(-1.4, 1.9))
    viz.animate(fig, update2, range(T), gif.replace(".gif", "_query.gif"), fps=8)


if __name__ == "__main__":
    run(main, __file__)
