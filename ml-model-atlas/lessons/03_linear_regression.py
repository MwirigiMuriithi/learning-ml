"""
# 03 · Linear Regression
**Family:** Supervised · Regression

## Definition
Linear regression models a numeric target as a weighted sum of the input features plus a bias:
ŷ = w·x + b. Training finds the weights that minimize the mean squared error between predictions and truth.

## Intuition
Draw the straight line that passes as close as possible to all the points, where "close" means the *sum of squared
vertical distances (residuals)* is as small as it can be. Squaring punishes big misses and makes the math smooth.

## The math
    Loss      L(w,b) = 1/(2n) Σ (w·xᵢ + b − yᵢ)²
    Gradient  ∂L/∂w = 1/n Σ (ŷᵢ − yᵢ)·xᵢ        ∂L/∂b = 1/n Σ (ŷᵢ − yᵢ)
    Update    w ← w − η ∂L/∂w ,  b ← b − η ∂L/∂b
    Closed form (normal equation): w = (XᵀX)⁻¹ Xᵀ y

## Assumptions & pitfalls
- Linear relationship; independent errors with constant variance; no extreme multicollinearity.
- Sensitive to outliers (squared error). Scale features before gradient descent.
- Coefficients are only interpretable when features are not heavily correlated.

## What the animation shows
Left: the line rotating and shifting toward the data, red segments are the residuals being squared. Right: the loss
surface over (w,b) with the optimizer's path sliding down the elongated valley. Afterwards the script fits a
4-feature *house price* model both ways (gradient descent vs normal equation) and prints the coefficients.

## Try this
- Standardize x before descending: the valley becomes round and convergence is far faster.
- Add one huge outlier to the data and watch the line get dragged toward it.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def main(gif, quick):
    X, y = data.sales_vs_ads(); x = X[:, 0]; n = len(y)
    steps = 800 if quick else 2500; lr = .03; w, b = -1.0, 12.0; hist = [(w, b)]
    for _ in range(steps):
        err = w * x + b - y; w, b = w - lr * (err * x).mean(), b - lr * err.mean(); hist.append((w, b))
    wb = np.linalg.lstsq(np.c_[x, np.ones(n)], y, rcond=None)[0]
    print(f"  gradient descent: w={w:.3f} b={b:.3f}   normal equation: w={wb[0]:.3f} b={wb[1]:.3f}   (truth: w=2.2, b=3.0)")
    # --- part B: multiple regression on house prices
    Xh, yh, names = data.house_prices(300); Zh, mu, sd = data.standardize(Xh); ym = yh.mean()
    wh = np.zeros(4); bh = 0.
    for _ in range(2000):
        e = Zh @ wh + bh - (yh - ym); wh -= .1 * Zh.T @ e / len(e); bh -= .1 * e.mean()
    ne = np.linalg.lstsq(np.c_[Zh, np.ones(len(yh))], yh - ym, rcond=None)[0]
    r2 = 1 - ((Zh @ wh + bh - (yh - ym)) ** 2).sum() / ((yh - ym) ** 2).sum()
    print("  house prices (standardized features):   feature      GD     normal-eq")
    for nm, a, c in zip(names, wh, ne[:4]): print(f"                                          {nm:10s} {a:7.2f} {c:7.2f}")
    print(f"  R² = {r2:.3f}")
    Wg, Bg = np.meshgrid(np.linspace(-1, 5, 70), np.linspace(-6, 14, 70))
    L = (((Wg[..., None] * x + Bg[..., None] - y) ** 2).mean(-1)) / 2
    idx = viz.geo(len(hist), 14 if quick else 60); hp = np.array(hist)
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4)

    def update(i):
        a1.clear(); a2.clear(); w_, b_ = hist[i]; xs = np.array([0, 10])
        a1.scatter(x, y, s=16, color=viz.PALETTE[0], zorder=3); a1.plot(xs, w_ * xs + b_, color=viz.PALETTE[3], lw=2.4)
        a1.vlines(x, y, w_ * x + b_, color=viz.PALETTE[3], alpha=.35)
        a1.set(title=f"iteration {i}:  y = {w_:.2f}·x + {b_:.2f}   MSE = {np.mean((w_ * x + b_ - y) ** 2):.1f}", xlabel="ad spend (k$)", ylabel="units sold", ylim=(-5, 35))
        a2.contourf(Wg, Bg, L, levels=np.geomspace(L.min() + .5, L.max(), 18), cmap="Blues_r", alpha=.8)
        a2.plot(hp[:i + 1, 0], hp[:i + 1, 1], color="k", lw=1.2); a2.scatter(w_, b_, s=50, color=viz.PALETTE[3], zorder=5, edgecolor="k")
        a2.scatter(*wb, marker="*", s=140, color="gold", edgecolor="k", zorder=6)
        a2.set(title="Loss surface L(w, b), gradient descent path", xlabel="slope w", ylabel="intercept b"); a2.grid(False)
    viz.animate(fig, update, idx, gif, fps=8)


if __name__ == "__main__":
    run(main, __file__)
