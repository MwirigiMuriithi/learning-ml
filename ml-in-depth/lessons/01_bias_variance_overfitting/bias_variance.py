"""
Lesson 01 - Bias, variance, under- and over-fitting (the idea behind every other lesson)

Problem story: a bakery predicts tomorrow's demand for a seasonal product from one number (a scaled "temperature index" x in
[-1, 1]).  The real relationship is a smooth curve, and every day's sales carry random noise.  We could fit a straight line,
a gentle curve or a wiggly curve.  Which one predicts *future* days best, and why?

Method (NumPy only): we know the true curve, so we can do what a real analyst never can: draw MANY independent training sets,
fit the same model to each, and see how the fits scatter.  For a test point x0:

    E[(y - fhat(x0))^2] =  (E[fhat(x0)] - f(x0))^2   +   Var[fhat(x0)]   +   sigma^2
                              bias^2                    variance          irreducible noise

Animated:
  bias_variance.gif   polynomial degree 0 ... 12: 25 fits (thin lines), their average vs the truth, and the error decomposition
  learning_curves.gif growing the training set: a too-simple model plateaus high, a too-flexible one has a big gap that closes

Run:  python bias_variance.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from numpy.polynomial import legendre as L

SIGMA = 0.3


def f_true(x):
    return np.sin(2.2 * x) + 0.35 * x


def draw(n, r):
    """Fixed design: the bakery always records the same n evenly spread temperature settings; only the day-to-day sales
    noise changes between "parallel universes".  (Random x would let rare gaps blow up high-degree fits and hide the lesson.)"""
    x = np.linspace(-1, 1, n)
    return x, f_true(x) + r.normal(0, SIGMA, n)


def fit_poly(x, y, deg, lam=1e-9):
    """Least-squares polynomial in the Legendre basis (well conditioned on [-1,1]); tiny ridge for safety."""
    A = L.legvander(x, deg)
    return np.linalg.lstsq(A, y, rcond=1e-10)[0]                 # minimum-norm least squares (also defined when deg >= n)


def predict(w, x):
    return L.legvander(x, len(w) - 1) @ w


def decomposition(deg, n=30, reps=300, seed=0, grid=np.linspace(-1, 1, 101)):
    """Monte-Carlo estimate of bias^2, variance and noise, averaged over x ~ U[-1,1] (the grid)."""
    r = rng_(seed)
    preds = np.empty((reps, len(grid)))
    train_err = []
    for i in range(reps):
        x, y = draw(n, r)
        w = fit_poly(x, y, deg)
        preds[i] = predict(w, grid)
        train_err.append(np.mean((predict(w, x) - y) ** 2))
    mean = preds.mean(0)
    bias2 = np.mean((mean - f_true(grid)) ** 2)
    var = np.mean(preds.var(0))
    return bias2, var, np.mean(train_err), preds


def bias_variance_gif(path):
    degs = list(range(0, 13))
    n, reps = 18, 300
    grid = np.linspace(-1, 1, 201)
    res = {d: decomposition(d, n, reps) for d in degs}
    b = np.array([res[d][0] for d in degs]); v = np.array([res[d][1] for d in degs]); tr = np.array([res[d][2] for d in degs])
    total = b + v + SIGMA ** 2
    print("  degree: bias^2 / variance / noise / expected test MSE / mean train MSE")
    for d in degs:
        print(f"     {d:2d}: {b[d]:.4f} / {v[d]:.4f} / {SIGMA**2:.4f} / {total[d]:.4f} / {tr[d]:.4f}")
    best = int(np.argmin(total))
    print(f"  best degree for n={n}: {best}")
    r = rng_(123)
    xs = [draw(n, r) for _ in range(25)]                 # 25 training sets for the left panel
    with Animator(path, fps=1.6, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Bias-variance: refit the same model on 25 different training sets")
        for d in degs:
            for a in (a1, a2, a3):
                a.clear()
            allp = []
            for x, y in xs:
                p = predict(fit_poly(x, y, d), grid)
                allp.append(p)
                a1.plot(grid, p, color=BLUE, lw=0.8, alpha=0.35)
            a1.plot(grid, np.mean(allp, 0), color=ORANGE, lw=2.4, label="average fit")
            a1.plot(grid, f_true(grid), color=INK, lw=1.8, ls="--", label="true curve")
            a1.scatter(*xs[0], s=16, color=INK2, zorder=4, label="one training set")
            a1.set(ylim=(-1.9, 1.9), title=f"Degree {d}", xlabel="temperature index", ylabel="demand"); a1.legend(loc="upper left", ncol=1)
            a2.stackplot(degs, b, v, np.full(len(degs), SIGMA ** 2), colors=[ORANGE, BLUE, MUTED], alpha=0.8, labels=["bias²", "variance", "noise σ²"])
            a2.axvline(d, color=INK, lw=1.5)
            a2.set(xlim=(0, 12), ylim=(0, 0.45), title="Test error = bias² + variance + noise", xlabel="polynomial degree")
            a2.legend(loc="upper right", ncol=1)
            a3.plot(degs, total, "o-", color=INK, label="expected test MSE")
            a3.plot(degs, tr, "o-", color=AQUA, label="training MSE")
            a3.scatter([d], [total[d]], s=90, color=ORANGE, zorder=5)
            a3.set(xlim=(-0.5, 12.5), ylim=(0, 0.45), title="Training error keeps falling", xlabel="polynomial degree"); a3.legend(loc="upper right")
            if d <= 1:
                msg = f"Degree {d}: too rigid (underfit). All fits agree with each other but miss the curve: high bias {b[d]:.3f}, low variance {v[d]:.3f}"
            elif d <= 3:
                msg = f"Degree {d}: bias {b[d]:.3f} has collapsed while variance is only {v[d]:.3f}: close to the sweet spot"
            elif d <= 7:
                msg = f"Degree {d}: bias is already tiny, variance grows ({v[d]:.3f}); training error {tr[d]:.3f} keeps shrinking"
            else:
                msg = f"Degree {d}: flexible (overfit). Each fit chases its own noise: variance {v[d]:.3f}, test error {total[d]:.3f} vs training {tr[d]:.3f}"
            caption(fig, msg)
            anim.grab(fig)
    return res


def learning_gif(path):
    ns = [10, 14, 20, 30, 50, 80, 150, 300, 600]
    degs = [1, 4, 12]
    cols = {1: BLUE, 4: AQUA, 12: ORANGE}
    xt = np.linspace(-1, 1, 400)
    reps = 40
    tr = {d: [] for d in degs}; te = {d: [] for d in degs}
    r = rng_(5)
    for n in ns:
        for d in degs:
            a, b = [], []
            for _ in range(reps):
                x, y = draw(n, r)
                w = fit_poly(x, y, d, lam=1e-6)
                a.append(np.mean((predict(w, x) - y) ** 2))
                b.append(np.mean((predict(w, xt) - f_true(xt)) ** 2) + SIGMA ** 2)     # expected test MSE (noise added analytically)
            tr[d].append(np.mean(a)); te[d].append(np.mean(b))
    print("  learning curves (train MSE / test MSE) by n:")
    for i, n in enumerate(ns):
        print(f"     n={n:3d}: " + "  ".join(f"deg{d}: {tr[d][i]:.3f}/{te[d][i]:.3f}" for d in degs))
    r2 = rng_(77)
    with Animator(path, fps=1.6, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.8), title="More data: variance shrinks, bias stays", gridspec_kw={"width_ratios": [1.3, 1]})
        for i, n in enumerate(ns):
            a1.clear(); a2.clear()
            x, y = draw(n, r2)
            a1.scatter(x, y, s=12, color=INK2, alpha=0.7, zorder=2)
            a1.plot(xt, f_true(xt), color=INK, ls="--", lw=1.6, label="true curve")
            for d in degs:
                a1.plot(xt, predict(fit_poly(x, y, d, lam=1e-6), xt), color=cols[d], lw=2, label=f"degree {d}")
            a1.set(ylim=(-1.9, 1.9), title=f"Fits with n = {n} training points", xlabel="temperature index", ylabel="demand"); a1.legend(loc="upper left", ncol=2)
            for d in degs:
                a2.plot(ns[: i + 1], te[d][: i + 1], "o-", color=cols[d], label=f"degree {d} test")
                a2.plot(ns[: i + 1], tr[d][: i + 1], "o--", color=cols[d], alpha=0.6, label=f"degree {d} train")
            a2.axhline(SIGMA ** 2, color=MUTED, lw=1.2); a2.text(10.5, SIGMA ** 2 + 0.012, "noise floor", color=INK2, fontsize=8)
            a2.set(xscale="log", xlim=(9, 700), ylim=(0, 0.5), title="Learning curves (average of 40 runs)", xlabel="training-set size n", ylabel="MSE")
            if i == 0:
                a2.legend(loc="upper right", ncol=1, fontsize=7)
            if n <= 20:
                msg = f"n = {n}: degree 12 has 13 parameters, so it interpolates the noise (train error ~0, test {te[12][i]:.2f}{', off the chart' if te[12][i] > 0.5 else ''}); the line is steady but wrong"
            elif n < 150:
                msg = f"n = {n}: degree 12's gap between train and test closes as data grow; degree 1 is stuck at test {te[1][i]:.3f}"
            else:
                msg = f"n = {n}: the flexible model has caught up (test {te[12][i]:.3f}); the line still cannot reach the noise floor ({SIGMA**2:.2f}): bias is not cured by data"
            caption(fig, msg)
            anim.grab(fig)


def main():
    print("Lesson 01: bias-variance")
    res = bias_variance_gif(here(__file__, "bias_variance.gif"))
    # verify the decomposition against an *independent* simulation of test MSE with fresh noisy labels
    r = rng_(999)
    for d in (1, 4, 12):
        errs = []
        for _ in range(300):
            x, y = draw(18, r)
            w = fit_poly(x, y, d)
            xt, yt = draw(500, r)
            errs.append(np.mean((predict(w, xt) - yt) ** 2))
        b, v, _, _ = res[d]
        print(f"  check degree {d:2d}: simulated test MSE {np.mean(errs):.4f}  vs  bias^2+var+sigma^2 = {b + v + SIGMA ** 2:.4f}")
    learning_gif(here(__file__, "learning_curves.gif"))


if __name__ == "__main__":
    main()
