"""
Lesson 04 - Polynomial Regression (from scratch)

Problem: a sensor records the temperature over one daily cycle (x = time of day
mapped to 0..2*pi, y = temperature anomaly in degC).  The truth is a smooth
S-shaped cycle, but we only have 20 noisy readings.  A straight line cannot bend,
so we expand the single input x into powers [1, x, x^2, ..., x^d] and fit an
ordinary *linear* regression on those new features.

What is animated (polynomial_regression.gif):
  * left   - the fitted curve as the degree grows 1 -> 14 (then 6 re-sampled
             datasets at degree 12, to show how unstable a flexible model is)
  * middle - training error vs test error as the degree grows (underfit ->
             good fit -> overfit)
  * right  - the magnitude of every fitted coefficient (they explode)

Run:  python polynomial_regression.py      -> writes polynomial_regression.gif
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ---------------------------------------------------------------- the model
def poly_features(t, degree):
    """Feature expansion  t -> [1, t, t^2, ..., t^degree]   (Vandermonde matrix)."""
    return np.vander(np.asarray(t, float), degree + 1, increasing=True)


class PolynomialRegression:
    """Linear regression on polynomial features.  Inputs are first mapped to
    [-1, 1] ("standardise"), which keeps t^k in a sane range and the normal
    equations well conditioned."""

    def __init__(self, degree, scale=True):
        self.degree, self.scale = degree, scale
        self.theta = None

    def _t(self, x):
        x = np.asarray(x, float)
        if not self.scale:
            return x
        return (x - self.mid) / self.half       # [lo, hi] -> [-1, 1]

    def fit(self, x, y, method="lstsq"):
        self.mid, self.half = (x.max() + x.min()) / 2, (x.max() - x.min()) / 2
        A = poly_features(self._t(x), self.degree)
        self.cond = np.linalg.cond(A.T @ A)      # cond(A^T A) = cond(A)^2: the price of the normal equation
        if method == "normal":
            # theta = (A^T A)^-1 A^T y   -- squares the condition number, can lose digits
            self.theta = np.linalg.solve(A.T @ A, A.T @ y)
        else:
            # SVD based least squares: same answer, numerically safer
            self.theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def predict(self, x):
        return poly_features(self._t(x), self.degree) @ self.theta


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


# ------------------------------------------------------------- the animation
def main():
    noise = 0.30
    x, y = make_sine(n=20, noise=noise, seed=4)
    xt, yt = make_sine(n=300, noise=noise, seed=99)          # large fresh test set
    xg = np.linspace(0, 2 * np.pi, 400)                      # plotting grid
    truth = np.sin(xg)
    max_deg = 14

    # ---- fit every degree once (for the error curves and the sanity checks)
    models = [PolynomialRegression(d).fit(x, y) for d in range(1, max_deg + 1)]
    tr = [mse(y, m.predict(x)) for m in models]
    te = [mse(yt, m.predict(xt)) for m in models]
    best = int(np.argmin(te)) + 1

    # conditioning with and without standardising the input (degree 12)
    raw = PolynomialRegression(12, scale=False).fit(x, y)
    sc = PolynomialRegression(12, scale=True).fit(x, y)
    print(f"  best degree on test set: {best}  (test MSE {min(te):.3f}, noise floor {noise**2:.3f})")
    print(f"  degree 12: cond(A^T A) raw x = {raw.cond:.1e}   scaled x = {sc.cond:.1e}")
    ne = PolynomialRegression(5).fit(x, y, method="normal")
    ls = PolynomialRegression(5).fit(x, y, method="lstsq")
    print(f"  degree 5: normal-equation vs lstsq max |dtheta| = {np.abs(ne.theta - ls.theta).max():.2e}")
    try:  # optional cross-check against NumPy's own polynomial fit
        ref = np.polynomial.polynomial.polyfit(ls._t(x), y, 5)
        print(f"  agrees with numpy polyfit: {np.allclose(ref, ls.theta, atol=1e-6)}")
    except Exception:
        pass

    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Polynomial regression: more flexibility fits the noise, not the signal")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.85, wspace=0.28)

    # frames: degrees 1..14, then 6 resamples of the data at degree 12
    rs = []
    for k in range(6):
        xr, yr = make_sine(n=20, noise=noise, seed=200 + k)
        rs.append((xr, yr, PolynomialRegression(12).fit(xr, yr)))

    def draw_errors(upto, now):
        a2.clear()
        d = np.arange(1, upto + 1)
        a2.plot(d, tr[:upto], "-o", color=BLUE, markersize=4)
        a2.plot(d, te[:upto], "-o", color=ORANGE, markersize=4)
        a2.axhline(noise ** 2, color=MUTED, linestyle="--", linewidth=1)
        a2.text(max_deg, noise ** 2 * 1.12, "noise floor", ha="right", fontsize=8.5, color=INK2)
        if upto >= 1:
            a2.scatter([now], [te[now - 1]], s=70, color=ORANGE, edgecolor=INK, zorder=5)
        a2.set(title="Error vs. degree: U-shaped test curve", xlabel="polynomial degree", ylabel="MSE (log)", yscale="log")
        a2.set_xlim(0.5, max_deg + 0.5); a2.set_ylim(0.03, 30)
        a2.text(1.2, 0.045, "train", color=BLUE, fontsize=9)
        a2.text(1.2, 8, "test", color=ORANGE, fontsize=9)

    def draw_coefs(m):
        a3.clear()
        mag = np.abs(m.theta) + 1e-12
        a3.bar(np.arange(len(mag)), mag, color=AQUA, width=0.7)
        a3.set(title="Coefficient magnitudes |θₖ|", xlabel="power k (θₖ multiplies tᵏ)", ylabel="|θₖ| (log)", yscale="log")
        a3.set_xlim(-0.8, max_deg + 0.8); a3.set_ylim(1e-3, 1e8)
        a3.text(0.97, 0.93, f"max |θ| = {mag.max():.0e}" if mag.max() >= 100 else f"max |θ| = {mag.max():.1f}",
                transform=a3.transAxes, ha="right", fontsize=9, color=INK2)

    with Animator(here(__file__, "polynomial_regression.gif"), fps=6, hold=3.0) as anim:
        for d in range(1, max_deg + 1):
            m = models[d - 1]
            a1.clear()
            a1.plot(xg, truth, color=MUTED, linewidth=1.5, linestyle="--")
            a1.scatter(x, y, s=24, color=BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=3)
            a1.plot(xg, m.predict(xg), color=ORANGE, zorder=4)
            a1.set(title=f"Degree {d}: the fitted curve", xlabel="time of day (rad)", ylabel="temperature anomaly (°C)")
            a1.set_xlim(0, 2 * np.pi); a1.set_ylim(-2.5, 2.5)
            a1.text(0.03, 0.06, "dashed = true signal", transform=a1.transAxes, fontsize=8.5, color=INK2)
            draw_errors(d, d)
            draw_coefs(m)
            state = "underfit" if d < best - 1 else ("about right" if d <= best + 1 else "overfit")
            caption(fig, f"degree {d:>2} ({state}):  train MSE={tr[d-1]:.3f}   test MSE={te[d-1]:.3f}   cond(AᵀA)={m.cond:.0e}")
            anim.grab(fig); anim.grab(fig)   # each state shown for ~0.33 s

        # second phase: same flexible model, different noisy samples -> wildly different curves
        a1.clear()
        for k, (xr, yr, mr) in enumerate(rs):
            a1.clear()
            a1.plot(xg, truth, color=MUTED, linewidth=1.5, linestyle="--")
            for j in range(k):                                  # older fits fade into a spaghetti
                a1.plot(xg, rs[j][2].predict(xg), color=ORANGE, alpha=0.25, linewidth=1.2)
            a1.scatter(xr, yr, s=24, color=BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=3)
            a1.plot(xg, mr.predict(xg), color=ORANGE, zorder=4)
            a1.set(title=f"Degree 12 on fresh sample #{k+1}: huge variance", xlabel="time of day (rad)", ylabel="temperature anomaly (°C)")
            a1.set_xlim(0, 2 * np.pi); a1.set_ylim(-2.5, 2.5)
            a1.text(0.03, 0.06, "wild edges (Runge effect)", transform=a1.transAxes, fontsize=8.5, color=INK2)
            draw_errors(max_deg, 12)
            draw_coefs(mr)
            caption(fig, f"same degree-12 model, new 20 readings: test MSE={mse(yt, mr.predict(xt)):.2f} (best degree-{best} model: {min(te):.2f})")
            anim.grab(fig); anim.grab(fig)   # each state shown for ~0.33 s


if __name__ == "__main__":
    main()
