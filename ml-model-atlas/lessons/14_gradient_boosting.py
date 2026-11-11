"""
# 14 · Gradient Boosting (GBM / XGBoost / LightGBM idea)
**Family:** Supervised · Regression & Classification (ensemble, boosting)

## Definition
Gradient boosting builds a model as a sum of small trees added one at a time, where each new tree is trained to predict the
**negative gradient of the loss** (for squared error: simply the *residuals*, the current mistakes). It is gradient descent
performed in function space.

## Intuition
Start with a crude guess (the mean). Look at what is still wrong (residuals), fit a small tree to explain part of that error,
add a fraction (the learning rate) of it, and repeat. Each tree is a small correction, and many corrections add up to a sharp, accurate model.

## The math
    F₀(x) = mean(y)       (classification: log-odds of the base rate)
    for m = 1..M:
        rᵢ = −∂L(yᵢ, F(xᵢ))/∂F(xᵢ)           squared error: rᵢ = yᵢ − F(xᵢ);   log-loss: rᵢ = yᵢ − σ(F(xᵢ))
        fit a shallow tree h_m to the pseudo-residuals r
        F_m(x) = F_{m−1}(x) + η · h_m(x)       η = learning rate (shrinkage), typically 0.01–0.3
    XGBoost/LightGBM add: second-order (Newton) steps, regularized leaf values, histogram splits, subsampling.

## Practical tips
- Usually the best off-the-shelf model for tabular data. Tune learning rate × number of trees, tree depth (3–8), subsampling; use early stopping.
- Smaller η + more trees = better but slower. Sensitive to noisy labels if too many rounds.

## What the animations show
1. `14_gradient_boosting.gif` (regression): left, prediction F_m hugging the curve; middle, the *residuals* and the small tree fitted to them;
   right, train/test error. The residuals shrink and become noise.
2. `14_gradient_boosting_classification.gif`: the same idea with log-loss on moons: the probability map sculpts itself round by round.

## Try this
- Raise the learning rate to 1.0 and watch test error bounce (overfitting quickly).
- Use depth-1 stumps vs depth-4 trees at the same number of rounds.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.trees import Tree
from mlatlas.cli import run


def main(gif, quick):
    x, y = data.curve(80, seed=3, noise=.3); xte, yte = data.curve(300, seed=11, noise=.3); xt = np.linspace(-1, 1, 300)
    M = 12 if quick else 45; lr = .25; F, Fte, Ft = np.full(len(y), y.mean()), np.full(len(yte), y.mean()), np.full(300, y.mean()); H = []
    tr, te = [np.mean((F - y) ** 2)], [np.mean((Fte - yte) ** 2)]
    for m in range(M):
        r = y - F; t = Tree(2, task="reg").fit(x[:, None], r); H.append(dict(F=Ft.copy(), r=r, h=t.predict(xt[:, None]), pts=x.copy()))
        F += lr * t.predict(x[:, None]); Fte += lr * t.predict(xte[:, None]); Ft += lr * t.predict(xt[:, None]); H[-1]["Fn"] = Ft.copy(); tr.append(np.mean((F - y) ** 2)); te.append(np.mean((Fte - yte) ** 2))
    print(f"  test MSE: start {te[0]:.3f} -> after {M} trees {te[-1]:.3f} (noise floor 0.09)")
    fig, (a1, a2, a3) = viz.subplots(1, 3, w=14, h=4.3)

    def update(m):
        for a in (a1, a2, a3): a.clear()
        h = H[m]; a1.scatter(x, y, s=14, color=viz.PALETTE[0], zorder=3); a1.plot(xt, data.curve_fn(xt), "--", color="gray", label="truth")
        a1.plot(xt, h["F"], color=viz.PALETTE[1], alpha=.5, label=f"F_{m}"); a1.plot(xt, h["Fn"], color=viz.PALETTE[3], lw=2.5, label=f"F_{m + 1}"); a1.set(title=f"Model after {m + 1} tree(s)", ylim=(-2.4, 2.4)); a1.legend(loc="upper left")
        a2.scatter(x, h["r"], s=14, color=viz.PALETTE[4], label="residuals y − F"); a2.plot(xt, h["h"], color="k", lw=2.2, label="new tree fits them"); a2.axhline(0, color="gray", lw=.8)
        a2.set(title=f"Round {m + 1}: fit a small tree to the mistakes", ylim=(-2, 2)); a2.legend(loc="upper left")
        a3.plot(tr[:m + 2], color=viz.PALETTE[2], label="train MSE"); a3.plot(te[:m + 2], color=viz.PALETTE[3], label="test MSE"); a3.axhline(.09, color="gray", ls=":")
        a3.set(title="Error vs rounds", xlabel="boosting round", xlim=(0, M), ylim=(0, max(tr[0], te[0]) * 1.05)); a3.legend()
    viz.animate(fig, update, range(M), gif, fps=4)
    # ---- classification with log-loss
    X, c = data.moons(300, .3, seed=1); Xe, ce = data.moons(300, .3, seed=2); xx, yy, G = viz.grid(X); M2 = 14 if quick else 60
    p0 = c.mean(); F, Fg, Fe = (np.full(len(c), np.log(p0 / (1 - p0))), np.full(len(G), np.log(p0 / (1 - p0))), np.full(len(ce), np.log(p0 / (1 - p0))))
    sg = lambda z: 1 / (1 + np.exp(-z)); ll = lambda z, t: -np.mean(t * np.log(sg(z) + 1e-9) + (1 - t) * np.log(1 - sg(z) + 1e-9)); S = []; L = [(ll(F, c), ll(Fe, ce))]
    for m in range(M2):
        t = Tree(3, task="reg").fit(X, c - sg(F)); F += .3 * t.predict(X) * 3; Fg += .3 * t.predict(G) * 3; Fe += .3 * t.predict(Xe) * 3; S.append(Fg.copy()); L.append((ll(F, c), ll(Fe, ce)))
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4)

    def update2(m):
        a1.clear(); a2.clear(); P = sg(S[m]); viz.show_map(a1, xx, yy, P, prob=True); a1.contour(xx, yy, P.reshape(xx.shape), levels=[.5], colors="k", linewidths=1.8); viz.scatter(a1, X, c)
        a1.set(title=f"Boosted classifier after {m + 1} trees", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
        a2.plot([l[0] for l in L[:m + 2]], color=viz.PALETTE[2], label="train log-loss"); a2.plot([l[1] for l in L[:m + 2]], color=viz.PALETTE[3], label="test log-loss"); a2.set(xlim=(0, M2), ylim=(0, .75), xlabel="boosting round", title="Log-loss"); a2.legend()
    viz.animate(fig, update2, viz.pick(M2, 15 if quick else 40), gif.replace(".gif", "_classification.gif"), fps=4)


if __name__ == "__main__":
    run(main, __file__)
