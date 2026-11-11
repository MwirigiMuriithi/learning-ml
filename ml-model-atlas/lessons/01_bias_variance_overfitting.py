"""
# 01 · Bias, Variance & Overfitting
**Family:** Foundations · Generalization

## Definition
A model's expected test error decomposes into three parts:
**bias²** (error from wrong assumptions: the model is too simple to capture the truth),
**variance** (error from sensitivity to the particular training sample: the model is too flexible) and
**irreducible noise**. *Overfitting* = low training error but high test error (variance too high);
*underfitting* = both errors high (bias too high).

## Intuition
Fit a straight line to a wavy signal: every training set gives nearly the same wrong line (high bias, low variance).
Fit a degree-15 polynomial to 25 noisy points: it threads every point, and a *different* training set gives a wildly
different curve (low bias, high variance). The sweet spot is in between.

## The math
    E[(y − f̂(x))²] = Bias²[f̂(x)] + Var[f̂(x)] + σ²
    Bias[f̂(x)] = E[f̂(x)] − f(x)          Var[f̂(x)] = E[(f̂(x) − E[f̂(x)])²]

## How to fight each problem
- **High variance:** more data, regularization (lesson 04), simpler model, bagging (lesson 12), early stopping, dropout.
- **High bias:** richer features, more flexible model, boosting (lesson 14), less regularization.
- **Always** measure on data the model has not seen (hold-out set or k-fold cross-validation).

## What the animation shows
We draw many different training sets from the same noisy process and fit polynomials of growing degree to each.
Left: thin lines = individual fits (their spread is *variance*), thick line = their average, dashed = truth
(the gap is *bias*). Right: the measured bias², variance and test error as complexity grows (log scale).

## Try this
- Make the training sets bigger (`data.curve(100, ...)`): how does the best degree change?
- Increase `noise`. Why does the noise floor (dotted) rise but the bias/variance curves keep their shape?
"""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from numpy.polynomial import Polynomial
from mlatlas import data, viz
from mlatlas.cli import run


def main(gif, quick):
    warnings.simplefilter("ignore")
    noise = .25
    xt = np.linspace(-1, 1, 200); truth = data.curve_fn(xt)
    degs = list(range(1, 16)); nsets = 8 if quick else 25
    sets = [data.curve(25, seed=s, noise=noise) for s in range(nsets)]
    xte, yte = data.curve(400, seed=999, noise=noise)
    R = {}
    for d in degs:
        polys = [Polynomial.fit(x, y, d) for x, y in sets]
        fits = np.array([p(xt) for p in polys]); mean = fits.mean(0)
        R[d] = dict(fits=fits, mean=mean, bias2=np.mean((mean - truth) ** 2), var=fits.var(0).mean(),
                    tr=np.mean([np.mean((p(x) - y) ** 2) for p, (x, y) in zip(polys, sets)]),
                    te=np.mean([np.mean((p(xte) - yte) ** 2) for p in polys]))
    best = min(degs, key=lambda d: R[d]["te"]); print(f"  best polynomial degree by test error: {best}")
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4)

    def update(d):
        a1.clear(); a2.clear(); r = R[d]
        for f in r["fits"][:12]: a1.plot(xt, f, color=viz.PALETTE[0], alpha=.25, lw=1)
        a1.plot(xt, r["mean"], color="k", lw=2.2, label="average of fits"); a1.plot(xt, truth, "--", color=viz.PALETTE[3], lw=2, label="truth")
        a1.scatter(*sets[0], s=14, color=viz.PALETTE[1], zorder=4, label="one training set")
        a1.set(ylim=(-2.5, 2.5), xlim=(-1, 1), title=f"Polynomial degree {d}: {nsets} different training sets"); a1.legend(loc="upper left")
        ds = [k for k in degs if k <= d]
        for key, lab, c in [("bias2", "bias²", viz.PALETTE[3]), ("var", "variance", viz.PALETTE[0]), ("te", "test error", "k"), ("tr", "train error", viz.PALETTE[2])]:
            a2.semilogy(ds, [R[k][key] + 1e-6 for k in ds], "o-", ms=3, color=c, label=lab)
        a2.axhline(noise ** 2, color="gray", ls=":"); a2.text(1.2, noise ** 2 * 1.15, "noise floor σ²", color="gray", fontsize=8)
        a2.set(xlim=(.5, 15.5), ylim=(1e-3, 1e2), xlabel="model complexity (polynomial degree)", title="Error decomposition"); a2.legend(loc="upper left")
    viz.animate(fig, update, degs, gif, fps=2)


if __name__ == "__main__":
    run(main, __file__)
