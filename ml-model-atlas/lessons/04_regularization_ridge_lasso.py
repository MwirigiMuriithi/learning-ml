"""
# 04 · Regularization: Ridge, Lasso & Elastic Net
**Family:** Supervised · Regression (penalized linear models)

## Definition
Regularization adds a penalty on the size of the weights to the loss so the model cannot over-use any feature.
**Ridge (L2)** shrinks all weights smoothly toward zero; **Lasso (L1)** can set weights *exactly* to zero (automatic
feature selection); **Elastic Net** mixes both.

## Intuition
With many correlated or noisy features, unpenalized regression happily assigns large opposing weights that fit noise.
The penalty is a budget: weights must "pay" for being large, so only features that truly help earn a big weight. The L1 penalty
has corners on the axes, so the optimum often lands exactly on a corner, which means a zero weight.

## The math
    minimize   1/(2n) ||y − Xw||²  +  λ [ α·||w||₁ + (1−α)/2·||w||₂² ]
    Ridge (α=0): w = (XᵀX/n + λI)⁻¹ Xᵀy/n
    Lasso/Elastic Net via coordinate descent:  wⱼ ← S(ρⱼ, λα) / (1 + λ(1−α)),
    ρⱼ = 1/n Xⱼᵀ(residual + Xⱼwⱼ),  S(r,t) = sign(r)·max(|r|−t, 0)   (soft-thresholding)

## Practical tips
- **Always standardize features** first, otherwise the penalty is unfair across units.
- Choose λ by cross-validation (the U-shaped test-error curve on the right).
- Lasso arbitrarily picks one of several correlated features; Elastic Net keeps groups together.

## What the animation shows
We use the 4 real house-price features plus 6 pure-noise columns. As λ grows: left, ridge paths shrink gradually;
middle, lasso paths hit zero one by one, noise features first; right, test error forms a U shape.

## Try this
- Change `alpha` for the elastic net and see how the path shape morphs between ridge and lasso.
- Make training smaller (50 rows) and see how much more regularization helps.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def soft(r, t): return np.sign(r) * np.maximum(np.abs(r) - t, 0)


def enet(X, y, lam, alpha, iters=150):
    n, d = X.shape; w = np.zeros(d); r = y.copy()
    for _ in range(iters):
        for j in range(d):
            r += X[:, j] * w[j]; w[j] = soft(X[:, j] @ r / n, lam * alpha) / (1 + lam * (1 - alpha)); r -= X[:, j] * w[j]
    return w


def main(gif, quick):
    rng = np.random.default_rng(0)
    X, y, names = data.house_prices(300, seed=1); X = np.c_[X, rng.normal(size=(300, 6))]
    names = names + [f"noise{i}" for i in range(1, 7)]
    Xtr, Xte, ytr, yte = data.train_test_split(X, y, test=.5, seed=2)
    Ztr, mu, sd = data.standardize(Xtr); Zte = (Xte - mu) / sd; ym, ys = ytr.mean(), ytr.std(); ytr_, yte_ = (ytr - ym) / ys, (yte - ym) / ys
    lams = np.logspace(-3, 1, 12 if quick else 40)
    ridge = np.array([enet(Ztr, ytr_, l, 0., 60) for l in lams]); lasso = np.array([enet(Ztr, ytr_, l, 1.) for l in lams[::-1]])[::-1]
    te = lambda W: [np.mean((Zte @ w - yte_) ** 2) for w in W]
    ter, tel = te(ridge), te(lasso)
    print(f"  best ridge λ={lams[np.argmin(ter)]:.3f} (test MSE {min(ter):.3f}),  best lasso λ={lams[np.argmin(tel)]:.3f} (test MSE {min(tel):.3f})")
    fig, (a1, a2, a3) = viz.subplots(1, 3, w=14, h=4.2)
    col = [viz.PALETTE[i] for i in range(4)] + ["#aaaaaa"] * 6

    def update(k):
        for a in (a1, a2, a3): a.clear()
        for ax, W, ttl in [(a1, ridge, "Ridge (L2): shrinks everything"), (a2, lasso, "Lasso (L1): zeroes out features")]:
            for j in range(10): ax.plot(lams, W[:, j], color=col[j], lw=2 if j < 4 else 1, label=names[j] if j < 4 else None)
            ax.axvline(lams[k], color="k", ls="--"); ax.set_xscale("log"); ax.set(title=ttl, xlabel="λ (regularization strength)", ylim=(-.2, 1.0))
        a1.legend(loc="upper right"); nz = int((np.abs(lasso[k]) > 1e-6).sum())
        a2.set_title(f"Lasso (L1): {nz} of 10 features alive at λ={lams[k]:.3f}")
        a3.semilogx(lams, ter, label="ridge", color=viz.PALETTE[0]); a3.semilogx(lams, tel, label="lasso", color=viz.PALETTE[1])
        a3.scatter([lams[k]] * 2, [ter[k], tel[k]], color="k", zorder=5); a3.set(title="Test error (U-shape = bias/variance)", xlabel="λ", ylabel="test MSE (standardized)"); a3.legend()
    viz.animate(fig, update, range(len(lams)), gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
