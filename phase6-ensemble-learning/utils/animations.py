"""
animations.py -- the plumbing behind the animated GIFs in the lessons.

Each function builds a small GIF (a few hundred KB) and returns its path.  The
*ideas* are explained in the notebooks; this file is just matplotlib's
FuncAnimation, in case you want to see how the sausage is made.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from course_utils import ASSETS, PALETTE


def _save(fig, update, frames, path: Path, fps: int = 4) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // fps)
    anim.save(str(path), writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------
# 1. Gradient boosting learning a noisy sine wave, one tree at a time
# --------------------------------------------------------------------------
def boosting_animation(path=None, n_rounds: int = 40, learning_rate: float = 0.3, seed: int = 0) -> str:
    from sklearn.tree import DecisionTreeRegressor
    path = Path(path or ASSETS / "boosting_sine.gif")
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 2 * np.pi, 120))
    y = np.sin(x) + 0.3 * np.sin(3 * x) + rng.normal(0, 0.15, x.size)
    grid = np.linspace(0, 2 * np.pi, 400)

    f_train = np.full_like(y, y.mean())
    f_grid = np.full_like(grid, y.mean())
    history = [(f_grid.copy(), (y - f_train).copy(), None)]
    for _ in range(n_rounds):
        resid = y - f_train
        tree = DecisionTreeRegressor(max_depth=2).fit(x[:, None], resid)
        f_train = f_train + learning_rate * tree.predict(x[:, None])
        f_grid = f_grid + learning_rate * tree.predict(grid[:, None])
        history.append((f_grid.copy(), (y - f_train).copy(), tree.predict(grid[:, None]) * learning_rate))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))

    def update(i):
        f, resid, step = history[i]
        a1.clear()
        a2.clear()
        a1.scatter(x, y, s=14, color=PALETTE["gray"], alpha=0.7, label="data")
        a1.plot(grid, f, color=PALETTE["blue"], lw=2.5, label="ensemble prediction")
        a1.set_title(f"After {i} tree{'s' if i != 1 else ''}")
        a1.set_ylim(-1.8, 1.8)
        a1.legend(loc="lower left", fontsize=8)
        a2.scatter(x, resid, s=14, color=PALETTE["red"], alpha=0.8)
        a2.axhline(0, color="black", lw=0.8)
        a2.set_title("Residuals = what the next tree must fix")
        a2.set_ylim(-1.8, 1.8)

    return _save(fig, update, len(history), path, fps=5)


# --------------------------------------------------------------------------
# 2. Bagging: many wobbly trees, one steady average
# --------------------------------------------------------------------------
def bagging_animation(path=None, n_trees: int = 30, seed: int = 1) -> str:
    from sklearn.tree import DecisionTreeRegressor
    path = Path(path or ASSETS / "bagging_average.gif")
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 2 * np.pi, 100))
    y = np.sin(x) + rng.normal(0, 0.35, x.size)
    grid = np.linspace(0, 2 * np.pi, 400)
    preds = []
    for _ in range(n_trees):
        idx = rng.integers(0, x.size, x.size)                      # bootstrap sample
        preds.append(DecisionTreeRegressor(min_samples_leaf=2).fit(x[idx, None], y[idx]).predict(grid[:, None]))
    preds = np.array(preds)

    fig, ax = plt.subplots(figsize=(7, 4))

    def update(i):
        ax.clear()
        ax.scatter(x, y, s=14, color=PALETTE["gray"], alpha=0.6)
        ax.plot(grid, np.sin(grid), color="black", ls="--", lw=1.2, label="truth")
        for p in preds[: i + 1]:
            ax.plot(grid, p, color=PALETTE["orange"], alpha=0.18, lw=1)
        ax.plot(grid, preds[: i + 1].mean(0), color=PALETTE["blue"], lw=3, label=f"average of {i + 1} trees")
        ax.set_ylim(-1.8, 1.8)
        ax.set_title("Bagging: individual trees wobble, the average calms down")
        ax.legend(loc="lower left", fontsize=8)

    return _save(fig, update, n_trees, path, fps=4)


# --------------------------------------------------------------------------
# 3. Bayesian optimisation with a Gaussian-process surrogate
# --------------------------------------------------------------------------
def objective_1d(x):
    """The 'expensive' function we pretend we cannot differentiate (we want its minimum)."""
    return np.sin(3 * x) + 0.5 * x ** 2 - 0.7 * x + 0.1 * np.cos(9 * x)


def expected_improvement(mu, sigma, best, xi: float = 0.01):
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-9)
    z = (best - mu - xi) / sigma
    return (best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def bayesopt_animation(path=None, n_iter: int = 12, seed: int = 3) -> str:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    path = Path(path or ASSETS / "bayesopt.gif")
    rng = np.random.default_rng(seed)
    lo, hi = -2.0, 2.5
    grid = np.linspace(lo, hi, 400)
    X = list(np.linspace(lo + 0.3, hi - 0.3, 4))
    Y = [objective_1d(v) for v in X]
    steps = []
    for _ in range(n_iter):
        # a fixed, sensible kernel keeps the animation stable (real libraries fit these numbers for you)
        gp = GaussianProcessRegressor(ConstantKernel(1.0, "fixed") * Matern(length_scale=0.7, length_scale_bounds="fixed", nu=2.5),
                                      alpha=1e-4, normalize_y=True, optimizer=None)
        gp.fit(np.array(X)[:, None], np.array(Y))
        mu, sd = gp.predict(grid[:, None], return_std=True)
        ei = expected_improvement(mu, sd, min(Y))
        nxt = grid[np.argmax(ei)]
        steps.append((np.array(X), np.array(Y), mu, sd, ei, nxt))
        X.append(nxt)
        Y.append(objective_1d(nxt))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})

    def update(i):
        Xs, Ys, mu, sd, ei, nxt = steps[i]
        a1.clear()
        a2.clear()
        a1.plot(grid, objective_1d(grid), color="black", ls="--", lw=1, label="true function (unknown)")
        a1.plot(grid, mu, color=PALETTE["blue"], lw=2, label="surrogate mean")
        a1.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=PALETTE["blue"], alpha=0.18, label="uncertainty")
        a1.scatter(Xs, Ys, color=PALETTE["red"], zorder=5, s=35, label="evaluated")
        a1.axvline(nxt, color=PALETTE["green"], lw=1.5)
        a1.set_title(f"Bayesian optimisation -- {len(Xs)} evaluations so far  (best so far = {Ys.min():.2f})")
        a1.legend(loc="upper center", fontsize=7, ncol=2)
        a2.fill_between(grid, ei, color=PALETTE["green"], alpha=0.5)
        a2.axvline(nxt, color=PALETTE["green"], lw=1.5)
        a2.set_ylabel("Expected\nimprovement")
        a2.set_xlabel("hyperparameter value")

    return _save(fig, update, len(steps), path, fps=2)


if __name__ == "__main__":
    print(boosting_animation())
    print(bagging_animation())
    print(bayesopt_animation())
