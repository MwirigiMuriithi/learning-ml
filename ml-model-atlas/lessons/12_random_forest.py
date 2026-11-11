"""
# 12 · Random Forest (Bagging)
**Family:** Supervised · Classification & Regression (ensemble, variance reduction)

## Definition
A random forest trains many *deep* decision trees, each on a different **bootstrap sample** (rows drawn with replacement) and
each split considering only a **random subset of features**, then averages their votes. This is **bagging** plus feature randomness.

## Intuition
One deep tree has low bias but very high variance. Averaging many *different* high-variance trees cancels their individual
mistakes: the errors are partly independent, so the average is far steadier. Randomness decorrelates the trees, which
is what makes averaging effective (variance of the mean ≈ ρσ² + (1−ρ)σ²/B).

## The math
    for b = 1..B:  draw bootstrap sample D_b (n rows with replacement), grow tree T_b (random m features per split)
    prediction:    classification = majority vote / mean of probabilities ;  regression = mean
    Out-of-bag (OOB) estimate: each row is left out of ≈ 36.8 % of the trees, so score it with only those trees -> a free validation score.
    Feature importance: mean impurity decrease, or permutation importance.

## Practical tips
- Few hyper-parameters, hard to overfit by adding trees, strong default for tabular data, no scaling needed.
- Slower/larger than a single tree; less interpretable; poor at extrapolating beyond the training range.

## What the animation shows
Left: the forest's averaged probability map as trees are added (1 → 60), smoothing from a jagged single-tree map into a gentle boundary.
Middle: the newest single tree on its own (jagged!). Right: test accuracy and OOB accuracy converge.

## Try this
- Set `max_features=None` (all features): trees become more correlated; does the forest improve as quickly?
- Compare with a single depth-12 tree's test accuracy.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.trees import Tree
from mlatlas.cli import run


def main(gif, quick):
    Xtr, ytr = data.moons(300, .35, seed=1); Xte, yte = data.moons(300, .35, seed=2); n = len(ytr)
    M = 12 if quick else 60; rng = np.random.default_rng(0); xx, yy, G = viz.grid(Xtr); trees = []; oob = np.zeros((M, n), bool)
    for m in range(M):
        idx = rng.integers(0, n, n); oob[m, np.setdiff1d(np.arange(n), idx)] = True
        trees.append(Tree(12, task="clf", max_features=1, rng=rng, n_classes=2).fit(Xtr[idx], ytr[idx]))
    Pg = np.array([t.predict_proba(G)[:, 1] for t in trees]); Pte = np.array([t.predict_proba(Xte)[:, 1] for t in trees]); Ptr = np.array([t.predict_proba(Xtr)[:, 1] for t in trees])
    cte = np.cumsum(Pte, 0) / np.arange(1, M + 1)[:, None]; num, den = np.cumsum(Ptr * oob, 0), np.cumsum(oob, 0)
    acc_te = [((cte[m] > .5) == yte).mean() for m in range(M)]
    acc_oob = [(((num[m] / np.maximum(den[m], 1)) > .5) == ytr)[den[m] > 0].mean() for m in range(M)]
    single = np.mean([((Pte[m] > .5) == yte).mean() for m in range(M)]); print(f"  average single-tree test accuracy {single:.3f}  ->  forest of {M}: {acc_te[-1]:.3f}")
    fr = sorted({m for m in [1, 2, 3, 4, 6, 9, 14, 20, 30, 45, 60] if m <= M})
    fig, (a1, a2, a3) = viz.subplots(1, 3, w=14, h=4.3)

    def update(m):
        for a in (a1, a2, a3): a.clear()
        viz.show_map(a1, xx, yy, Pg[:m].mean(0), prob=True); a1.contour(xx, yy, Pg[:m].mean(0).reshape(xx.shape), levels=[.5], colors="k", linewidths=1.8); viz.scatter(a1, Xtr, ytr, s=12)
        a1.set(title=f"Forest of {m} tree(s): test acc {acc_te[m - 1]:.2f}", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
        viz.show_map(a2, xx, yy, (Pg[m - 1] > .5).astype(int)); viz.scatter(a2, Xtr, ytr, s=12); a2.set(title=f"Tree #{m} alone (jagged, overfit)", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
        a3.plot(range(1, m + 1), acc_te[:m], color=viz.PALETTE[3], label="test"); a3.plot(range(1, m + 1), acc_oob[:m], color=viz.PALETTE[2], label="out-of-bag")
        a3.axhline(single, color="gray", ls=":", label="avg single tree"); a3.set(title="Accuracy vs number of trees", xlabel="trees", xlim=(0, M), ylim=(.7, .95)); a3.legend(loc="lower right")
    viz.animate(fig, update, fr, gif, fps=2)


if __name__ == "__main__":
    run(main, __file__)
