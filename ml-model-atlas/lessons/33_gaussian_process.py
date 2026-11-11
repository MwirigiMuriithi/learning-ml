"""
# 33 · Gaussian Processes (GP) Regression
**Family:** Supervised · Bayesian non-parametric regression

## Definition
A Gaussian process is a distribution over *functions*. You choose a **kernel** that says how similar the outputs at two inputs should be; conditioning on the observed points gives, at every new input,
a full Gaussian predictive distribution: a **mean** and a **calibrated uncertainty**. It is regression that knows what it does not know.

## Intuition
Draw random smooth curves (the prior). Every time you observe a data point, throw away all the curves that miss it. What remains is a bundle of plausible functions: tight near data, fanning out
away from it. The kernel's **length-scale** decides how wiggly the plausible functions are.

## The math
    f ~ GP(0, k),    k(x,x') = σ_f² exp(−(x−x')²/2ℓ²)               (RBF kernel)
    data (X, y) with noise σ_n:   K = k(X,X) + σ_n² I
    posterior mean   μ(x*) = k(x*,X) K⁻¹ y
    posterior var    σ²(x*) = k(x*,x*) − k(x*,X) K⁻¹ k(X,x*)
    hyper-parameters (ℓ, σ_f, σ_n) are chosen by maximizing the marginal likelihood  log p(y) = −½ yᵀK⁻¹y − ½ log|K| − (n/2) log 2π.
    Cost O(n³) in the number of points (sparse GPs / inducing points help). Uses: Bayesian optimization, surrogate models, geostatistics (kriging), small-data science.

## What the animation shows
Phase 1: observations arrive one at a time: the mean (red), the ±2σ band and five posterior function samples (thin) tighten around the data. Phase 2: with all data fixed, the length-scale ℓ sweeps
from tiny (overfits, wiggly) to huge (underfits, flat), with the marginal likelihood shown so you can see why it picks a middle value.

## Try this
- Swap the RBF for a periodic kernel and fit the seasonal sales data.
- Add 20 noisy points to the same location and watch σ shrink only slightly (noise ≠ ignorance).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def rbf(a, b, ell, sf=1.): return sf ** 2 * np.exp(-(a[:, None] - b[None]) ** 2 / (2 * ell ** 2))


def posterior(X, y, xs, ell, sn=.15):
    if len(X) == 0: return np.zeros(len(xs)), rbf(xs, xs, ell), 0.
    K = rbf(X, X, ell) + sn ** 2 * np.eye(len(X)); L = np.linalg.cholesky(K); Ks = rbf(xs, X, ell); a = np.linalg.solve(L.T, np.linalg.solve(L, y)); mu = Ks @ a; v = np.linalg.solve(L, Ks.T)
    return mu, rbf(xs, xs, ell) - v.T @ v, -.5 * y @ a - np.log(np.diag(L)).sum() - len(X) / 2 * np.log(2 * np.pi)


def main(gif, quick):
    rng = np.random.default_rng(1); X, y = data.curve(14, seed=5, noise=.12); o = rng.permutation(14); X, y = X[o], y[o]; xs = np.linspace(-1.2, 1.2, 150); z = rng.normal(size=(150, 5)); ells = [.03, .06, .12, .25, .5, 1., 2., 4.] if not quick else [.05, .3, 3.]
    lml = [posterior(X, y, xs, e)[2] for e in np.geomspace(.03, 4, 40)]; best = np.geomspace(.03, 4, 40)[int(np.argmax(lml))]; print(f"  length-scale with the highest marginal likelihood: ℓ ≈ {best:.2f}")
    frames = [("n", k) for k in range(0, 15 if not quick else 8)] + [("l", e) for e in ells]; fig, (a1, a2) = viz.subplots(1, 2, w=12, h=4.6)

    def update(fr):
        a1.clear(); a2.clear(); kind, v = fr; n, ell = (v, .3) if kind == "n" else (14, v); mu, C, ll = posterior(X[:n], y[:n], xs, ell); sd = np.sqrt(np.maximum(np.diag(C), 1e-10)); Lc = np.linalg.cholesky(C + 1e-6 * np.eye(150))
        a1.fill_between(xs, mu - 2 * sd, mu + 2 * sd, color=viz.PALETTE[0], alpha=.2, label="±2σ"); [a1.plot(xs, mu + Lc @ z[:, j], color=viz.PALETTE[0], lw=.8, alpha=.6) for j in range(5)]
        a1.plot(xs, mu, color=viz.PALETTE[3], lw=2.5, label="posterior mean"); a1.plot(xs, data.curve_fn(xs), "--", color="gray", label="truth"); a1.scatter(X[:n], y[:n], color="k", zorder=5, s=25)
        a1.set(ylim=(-3, 3), title=f"{n} observations, length-scale ℓ={ell:.2f}" if kind == "n" else f"all 14 observations, sweeping length-scale ℓ={ell:.2f}"); a1.legend(loc="upper left", fontsize=7)
        ls = np.geomspace(.03, 4, 40); a2.semilogx(ls, lml, color=viz.PALETTE[2]); a2.scatter([ell], [posterior(X, y, xs, ell)[2]], color="k", zorder=5, s=50); a2.set(title="Log marginal likelihood of the data vs length-scale", xlabel="ℓ", ylabel="log p(y)", ylim=(min(lml) - 2, max(lml) + 3))
    viz.animate(fig, update, frames, gif, fps=2)


if __name__ == "__main__":
    run(main, __file__)
