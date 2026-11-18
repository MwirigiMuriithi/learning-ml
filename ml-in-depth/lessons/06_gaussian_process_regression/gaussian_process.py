"""
Lesson 06 - Gaussian-process regression (from scratch)

Problem story: a greenhouse sensor was read 11 times over a 10-hour shift (irregular times, one long gap, noisy readings).
The grower wants the temperature curve *and* an honest statement of how sure we are at every moment, including inside the gap.
A Gaussian process (GP) gives both: a distribution over functions, updated by Bayes' rule as readings arrive.

Implemented (NumPy only; scipy.optimize only to maximise the likelihood):
  * kernels: RBF (squared exponential), Matern-3/2, periodic, linear
  * sampling functions from the prior and from the posterior
  * posterior mean / covariance via a Cholesky factorisation
  * log marginal likelihood (data fit + complexity penalty + constant) and its maximisation over (length-scale, signal sd, noise sd)

Animated:
  gp_posterior.gif    prior samples -> add one reading at a time: mean, uncertainty band, sampled functions
  gp_lengthscale.gif  sweep the length-scale: under- vs over-fitting, and the marginal likelihood that picks the best one
  gp_kernels.gif      what each kernel believes before seeing data (prior samples + covariance matrix)

Run:  python gaussian_process.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from scipy.optimize import minimize


# ------------------------------------------------------------------- kernels
def rbf(A, B, ell=1.0, sf=1.0):
    d2 = (A[:, None] - B[None, :]) ** 2
    return sf ** 2 * np.exp(-0.5 * d2 / ell ** 2)


def matern32(A, B, ell=1.0, sf=1.0):
    r = np.abs(A[:, None] - B[None, :]) * np.sqrt(3) / ell
    return sf ** 2 * (1 + r) * np.exp(-r)


def periodic(A, B, ell=1.0, sf=1.0, period=3.0):
    d = np.abs(A[:, None] - B[None, :])
    return sf ** 2 * np.exp(-2 * np.sin(np.pi * d / period) ** 2 / ell ** 2)


def linear(A, B, ell=1.0, sf=1.0):
    return sf ** 2 * (A[:, None] * B[None, :]) / 4.0 + 0 * ell


# ------------------------------------------------------------------- data
def greenhouse(seed=4):
    """Hours into the shift (0-10) and temperature deviation (deg C).  A gap between hours 4.2 and 6.3."""
    r = rng_(seed)
    x = np.array([0.3, 0.9, 1.6, 2.4, 3.1, 3.9, 6.6, 7.3, 8.2, 9.0, 9.7])
    f = lambda t: 1.4 * np.sin(0.9 * t) + 0.5 * np.cos(2.1 * t)
    y = f(x) + r.normal(0, 0.2, len(x))
    return x, y, f


# ------------------------------------------------------------------- GP
class GP:
    def __init__(self, kernel=rbf, ell=1.0, sf=1.0, sn=0.2):
        self.kernel, self.ell, self.sf, self.sn = kernel, ell, sf, sn

    def K(self, A, B):
        return self.kernel(A, B, self.ell, self.sf)

    def fit(self, x, y):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        Kxx = self.K(self.x, self.x) + (self.sn ** 2 + 1e-9) * np.eye(len(self.x))
        self.L = np.linalg.cholesky(Kxx)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))     # (K + sn^2 I)^-1 y
        return self

    def predict(self, xs, full_cov=False):
        """Posterior of f at xs:  mean = K*x alpha,   cov = K** - K*x (K + sn^2 I)^-1 Kx*."""
        Ks = self.K(xs, self.x)
        mean = Ks @ self.alpha
        V = np.linalg.solve(self.L, Ks.T)
        cov = self.K(xs, xs) - V.T @ V
        return (mean, cov) if full_cov else (mean, np.sqrt(np.maximum(np.diag(cov), 1e-12)))

    def lml_terms(self):
        """log p(y|X) = -1/2 y'K^-1 y  -  1/2 log|K|  -  n/2 log 2pi   (data fit, complexity penalty, constant)."""
        n = len(self.x)
        return -0.5 * self.y @ self.alpha, -np.log(np.diag(self.L)).sum(), -0.5 * n * np.log(2 * np.pi)

    def lml(self):
        return sum(self.lml_terms())

    def sample(self, xs, n_samples, r, posterior=True):
        if posterior and hasattr(self, "x"):
            mean, cov = self.predict(xs, full_cov=True)
        else:
            mean, cov = np.zeros(len(xs)), self.K(xs, xs)
        Lc = np.linalg.cholesky(cov + 1e-6 * np.eye(len(xs)))
        return mean[:, None] + Lc @ r.normal(size=(len(xs), n_samples))


def optimise(x, y, kernel=rbf, start=(1.0, 1.0, 0.3)):
    """Maximise the log marginal likelihood over log(ell), log(sf), log(sn)."""
    def neg(p):
        try:
            g = GP(kernel, *np.exp(p)).fit(x, y)
            return -g.lml()
        except np.linalg.LinAlgError:
            return 1e10
    best = None
    for s0 in [start, (0.4, 1.0, 0.1), (3.0, 1.5, 0.5)]:
        res = minimize(neg, np.log(s0), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=800))
        if best is None or res.fun < best.fun:
            best = res
    return np.exp(best.x), -best.fun


# ------------------------------------------------------------------- animations
XS = np.linspace(0, 10, 200)


def posterior_gif(path):
    x, y, f = greenhouse()
    r = rng_(8)
    order = [0, 6, 3, 9, 1, 7, 4, 10, 2, 5, 8]                      # readings arrive in an irregular order
    ell, sf, sn = 0.9, 1.2, 0.2
    with Animator(path, fps=1.5, hold=4.0) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 5), title="Gaussian process: from prior beliefs to a posterior as readings arrive",
                                gridspec_kw={"width_ratios": [1.7, 1]})
        for m in range(0, len(order) + 1):
            a1.clear(); a2.clear()
            g = GP(rbf, ell, sf, sn)
            if m:
                idx = order[:m]; g.fit(x[idx], y[idx])
                mean, sd = g.predict(XS)
                smp = g.sample(XS, 5, rng_(8), True)
            else:
                mean, sd = np.zeros_like(XS), np.full_like(XS, sf)
                smp = g.sample(XS, 5, rng_(8), False)
            a1.fill_between(XS, mean - 2 * sd, mean + 2 * sd, color=BLUE, alpha=0.18, label="±2 sd (95% band)")
            for j in range(5):
                a1.plot(XS, smp[:, j], color=BLUE, lw=0.9, alpha=0.55)
            a1.plot(XS, mean, color=BLUE, lw=2.4, label="posterior mean")
            a1.plot(XS, f(XS), color=INK, lw=1.4, ls="--", label="true curve (unknown)")
            if m:
                a1.scatter(x[order[:m]], y[order[:m]], s=45, color=ORANGE, zorder=5, edgecolor=SURFACE, label="readings")
            a1.set(ylim=(-3.8, 3.8), xlim=(0, 10), title=f"{m} readings", xlabel="hours into the shift", ylabel="temperature deviation (°C)")
            a1.legend(loc="lower left", ncol=2, fontsize=8)
            a2.plot(XS, sd, color=AQUA, lw=2)
            a2.set(xlim=(0, 10), ylim=(0, sf * 1.1), title="Posterior standard deviation", xlabel="hours", ylabel="uncertainty")
            if m:
                a2.scatter(x[order[:m]], np.zeros(m), s=25, color=ORANGE, zorder=5, clip_on=False)
            if m == 0:
                msg = "Prior: every function in the swarm is plausible; uncertainty is the same everywhere (sd = 1.2). Thin lines are random draws"
            elif m < len(order):
                msg = f"{m} reading(s): the band pinches at the data and widens in gaps; the sampled functions are forced through the readings"
            else:
                inside = (XS > 4.5) & (XS < 6.0)
                msg = f"All readings: sd is ~{sd[np.argmin(np.abs(XS - 1.6))]:.2f} near data but {sd[inside].max():.2f} in the gap (hours 4.2-6.6) where the band honestly balloons"
            caption(fig, msg)
            anim.grab(fig)


def lengthscale_gif(path):
    x, y, f = greenhouse()
    sf, sn = 1.2, 0.2
    ells = np.array([0.08, 0.15, 0.3, 0.5, 0.8, 1.2, 1.8, 3.0, 6.0])
    grid = np.geomspace(0.05, 8, 80)
    terms = np.array([GP(rbf, e, sf, sn).fit(x, y).lml_terms() for e in grid])
    tot = terms.sum(1)
    best = grid[tot.argmax()]
    print(f"  length-scale sweep (sf={sf}, sn={sn}): best by marginal likelihood = {best:.2f}")
    with Animator(path, fps=1.4, hold=4.0) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 5), title="Choosing the length-scale ℓ with the marginal likelihood", gridspec_kw={"width_ratios": [1.5, 1]})
        for e in ells:
            a1.clear(); a2.clear()
            g = GP(rbf, e, sf, sn).fit(x, y)
            mean, sd = g.predict(XS)
            a1.fill_between(XS, mean - 2 * sd, mean + 2 * sd, color=BLUE, alpha=0.18)
            a1.plot(XS, mean, color=BLUE, lw=2.4); a1.plot(XS, f(XS), color=INK, ls="--", lw=1.3)
            a1.scatter(x, y, s=40, color=ORANGE, zorder=5, edgecolor=SURFACE)
            fit_, pen, _ = g.lml_terms()
            a1.set(ylim=(-3.8, 3.8), xlim=(0, 10), title=f"ℓ = {e}", xlabel="hours", ylabel="temperature deviation (°C)")
            a2.plot(grid, terms[:, 0], color=AQUA, label="data fit  -½ yᵀK⁻¹y")
            a2.plot(grid, terms[:, 1], color=YELLOW, label="complexity  -½ log|K|")
            a2.plot(grid, tot, color=INK, lw=2.2, label="log marginal likelihood")
            a2.axvline(e, color=ORANGE, lw=2)
            a2.set(xscale="log", ylim=(-60, 10), xlim=(0.05, 8), title="Occam's razor, built in", xlabel="length-scale ℓ")
            a2.legend(loc="lower right", fontsize=8)
            if e < best / 2:
                msg = f"ℓ = {e}: wiggly. Fits every reading (data-fit term good: {fit_:.1f}) but the complexity term is only {pen:+.1f}; the gap is a free-for-all"
            elif e > best * 2:
                msg = f"ℓ = {e}: too smooth. Very simple model (complexity term {pen:+.1f}, a reward) but it cannot follow the readings (data-fit {fit_:.1f})"
            else:
                msg = f"ℓ = {e}: near the best compromise (maximum at ℓ ≈ {best:.2f}): likelihood {g.lml():.1f}"
            caption(fig, msg)
            anim.grab(fig)


def kernels_gif(path):
    xs = np.linspace(0, 10, 120)
    items = [("RBF (smooth, ℓ = 1)", lambda A, B: rbf(A, B, 1.0, 1.0), "infinitely differentiable; wiggles on a scale of ℓ"),
             ("Matern-3/2 (rough, ℓ = 1.5)", lambda A, B: matern32(A, B, 1.5, 1.0), "once differentiable; paths look like rugged terrain"),
             ("Periodic (period 3)", lambda A, B: periodic(A, B, 1.0, 1.0, 3.0), "repeats every 3 units: seasonal behaviour"),
             ("Linear", lambda A, B: linear(A, B), "only straight lines: Bayesian linear regression"),
             ("RBF + Linear", lambda A, B: rbf(A, B, 1.0, 1.0) + linear(A, B), "a trend plus smooth wiggles (kernels add)")]
    with Animator(path, fps=0.9, hold=3.0) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.8), title="Kernels encode beliefs about the function", gridspec_kw={"width_ratios": [1.6, 1]})
        for name, k, note in items:
            a1.clear(); a2.clear()
            K = k(xs, xs)
            Lc = np.linalg.cholesky(K + 1e-6 * np.eye(len(xs)))
            smp = Lc @ rng_(3).normal(size=(len(xs), 5))
            for j in range(5):
                a1.plot(xs, smp[:, j], color=CAT[j], lw=1.6)
            a1.set(title=f"Five functions drawn from the prior: {name}", xlabel="x", ylim=(-4, 4))
            a2.imshow(K, cmap="Blues", extent=(0, 10, 10, 0), aspect="auto"); a2.grid(False)
            a2.set(title="Covariance matrix k(x, x')", xlabel="x'", ylabel="x")
            caption(fig, f"{name}: {note}")
            anim.grab(fig)


def main():
    x, y, f = greenhouse()
    print("Lesson 06: Gaussian processes")
    g = GP(rbf, 0.9, 1.2, 0.2).fit(x, y)
    mean, sd = g.predict(XS)
    inside = (XS > 4.5) & (XS < 6.0)
    near = np.abs(XS[:, None] - x[None]).min(1) < 0.3
    print(f"  fixed hyper-parameters: RMSE vs truth {np.sqrt(np.mean((mean - f(XS)) ** 2)):.3f}; sd near readings {sd[near].mean():.3f}, in the gap {sd[inside].mean():.3f}")
    cover = np.mean(np.abs(f(XS) - mean) <= 2 * sd)
    print(f"  fraction of the true curve inside the 95% band: {cover:.2f}; log marginal likelihood {g.lml():.2f}")
    (ell, sf, sn), lml = optimise(x, y)
    print(f"  optimised: ell={ell:.2f} sf={sf:.2f} sn={sn:.2f} (true noise 0.20); log marginal likelihood {lml:.2f}")
    go = GP(rbf, ell, sf, sn).fit(x, y); mo, so = go.predict(XS)
    print(f"  optimised RMSE vs truth {np.sqrt(np.mean((mo - f(XS)) ** 2)):.3f}")
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
        k = C(1.2 ** 2, "fixed") * RBF(0.9, "fixed") + WhiteKernel(0.2 ** 2, "fixed")
        sk = GaussianProcessRegressor(k, optimizer=None).fit(x[:, None], y)
        m2, s2 = sk.predict(XS[:, None], return_std=True)
        print(f"  [sklearn] max |mean diff| {np.abs(m2 - mean).max():.1e}; max |sd diff| {np.abs(np.sqrt(s2 ** 2 - 0.2 ** 2) - sd).max():.1e} (sklearn's sd includes noise); "
              f"LML sklearn {sk.log_marginal_likelihood_value_:.4f} vs ours {g.lml():.4f}")
    except Exception as ex:
        print("  (sklearn check skipped:", ex, ")")
    posterior_gif(here(__file__, "gp_posterior.gif"))
    lengthscale_gif(here(__file__, "gp_lengthscale.gif"))
    kernels_gif(here(__file__, "gp_kernels.gif"))


if __name__ == "__main__":
    main()
