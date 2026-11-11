"""
# 10 · Support Vector Machines (SVM) & the Kernel Trick
**Family:** Supervised · Classification (margin-based)

## Definition
An SVM finds the decision boundary with the **largest margin**: the widest street separating the two classes.
Only the points touching the street edges, the **support vectors**, matter; every other point could be deleted without
changing the model. The **kernel trick** lets the same algorithm draw curved boundaries by implicitly mapping points
into a high-dimensional space.

## Intuition
Many lines separate two classes, but the one farthest from both is the most robust to noise. If the classes overlap,
the **soft margin** parameter C says how much you are willing to pay for points inside the street or misclassified
(small C = wide, tolerant street; large C = strict, may overfit).

## The math
    Primal (soft margin):  min ½||w||² + C Σ max(0, 1 − yᵢ(w·xᵢ + b))        (hinge loss)
    Dual:                  max Σ αᵢ − ½ Σᵢⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ,xⱼ)    s.t.  0 ≤ αᵢ ≤ C
    Decision:              f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b           support vectors = points with αᵢ > 0
    Kernels:               linear  K = x·x' ;  RBF  K = exp(−γ||x − x'||²) ;  polynomial  K = (x·x' + 1)ᵈ
    Here we absorb b into the kernel (K+1) and solve the dual by coordinate ascent, one αᵢ at a time:
                           αᵢ ← clip(αᵢ + (1 − yᵢ f(xᵢ)) / Qᵢᵢ , 0, C)

## Practical tips
- Scale features. Tune C and γ together by grid search. Great on small/medium data; O(n²–n³) training cost.
- Large γ = every point is its own island (overfit). Small γ ≈ linear.

## What the animations show
1. `10_support_vector_machine.gif`: dual coordinate ascent epoch by epoch. Left: linear kernel on overlapping blobs; right: RBF kernel
   on moons. Solid line = boundary f=0, dashed = margins f=±1, ringed points = support vectors.
2. `10_support_vector_machine_hyper.gif`: how γ (flexibility) changes the RBF boundary from nearly linear to overfit.

## Try this
- Plot accuracy against C on a log scale for both datasets.
- Use the polynomial kernel (degree 3) on the circles dataset.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def kern(A, B, kind, gamma):
    if kind == "linear": return A @ B.T
    return np.exp(-gamma * ((A[:, None] - B[None]) ** 2).sum(-1))


def fit_dual(X, y, C, kind, gamma, epochs, seed=0):
    rng = np.random.default_rng(seed); K = kern(X, X, kind, gamma) + 1; Q = K * np.outer(y, y); a = np.zeros(len(y)); hist = [a.copy()]
    for _ in range(epochs):
        for i in rng.permutation(len(y)): a[i] = np.clip(a[i] + (1 - Q[i] @ a) / Q[i, i], 0, C)
        hist.append(a.copy())
    return hist


def draw(ax, X, y, a, xx, yy, G, kind, gamma, title):
    ax.clear(); f = (a * y) @ (kern(X, G, kind, gamma) + 1); F = f.reshape(xx.shape)
    viz.show_map(ax, xx, yy, (f > 0).astype(int)); ax.contour(xx, yy, F, levels=[-1, 0, 1], colors=["#777", "k", "#777"], linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
    viz.scatter(ax, X, (y > 0).astype(int), s=20); sv = a > 1e-6
    ax.scatter(*X[sv].T, s=90, facecolor="none", edgecolor="k", linewidth=1.2, zorder=6)
    acc = ((np.sign((a * y) @ (kern(X, X, kind, gamma) + 1))) == y).mean(); ax.set(title=f"{title}  |  {sv.sum()} support vectors, acc {acc:.0%}", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))


def main(gif, quick):
    E = 6 if quick else 18
    Xa, ya = data.blobs(120, k=2, seed=3, std=1.3, spread=2.0); sa = 2 * ya - 1
    Xb, yb = data.moons(160, .22, seed=4); sb = 2 * yb - 1
    ha, hb = fit_dual(Xa, sa, 1., "linear", 0, E), fit_dual(Xb, sb, 5., "rbf", 2., E)
    ga, gb = viz.grid(Xa, n=60), viz.grid(Xb, n=60)
    fig, (a1, a2) = viz.subplots(1, 2, w=12, h=4.8)
    fr = sorted(set([0, 1, 2, 3, 4, 6, 9, 13, E]) & set(range(E + 1)))

    def update(e):
        draw(a1, Xa, sa, ha[e], *ga, "linear", 0, f"Linear kernel, C=1, epoch {e}"); draw(a2, Xb, sb, hb[e], *gb, "rbf", 2., f"RBF kernel γ=2, C=5, epoch {e}")
    viz.animate(fig, update, fr, gif, fps=2)
    # ---- hyper-parameter animation
    gammas = [.05, .2, .5, 1, 2, 5, 12, 30] if not quick else [.2, 2, 30]; fig, (a1, a2) = viz.subplots(1, 2, w=12, h=4.8)
    Xt, yt = data.moons(160, .3, seed=7); st = 2 * yt - 1; gg = viz.grid(Xt, n=60)

    def update2(g):
        a_ = fit_dual(Xt, st, 5., "rbf", g, 25 if not quick else 8)[-1]; draw(a1, Xt, st, a_, *gg, "rbf", g, f"γ = {g}: boundary")
        a2.clear(); a2.set_title("Small γ: smooth/underfit.  Large γ: memorizes noise.", fontsize=9); a2.axis("off")
        a2.text(.5, .5, f"γ = {g}\nreach of each training point ≈ 1/√γ = {1 / np.sqrt(g):.2f}", ha="center", fontsize=14)
    viz.animate(fig, update2, gammas, gif.replace(".gif", "_hyper.gif"), fps=1.5)


if __name__ == "__main__":
    run(main, __file__)
