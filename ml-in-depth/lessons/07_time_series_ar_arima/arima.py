"""
Lesson 07 - Time series: autocorrelation, AR / ARMA / seasonal-differenced ARIMA models (from scratch)

Problem story: a news site's daily visitors for 350 days: a steady upward trend, a strong weekly rhythm (weekends are quiet) and
day-to-day "momentum" noise (a busy day tends to be followed by another busy day).  The editor wants a 14-day forecast with
honest uncertainty bands, to plan servers and ad sales.

Implemented (NumPy; scipy.optimize only for the MA part):
  * regression with AR errors (trend + day-of-week + AR) and, for comparison, seasonal differencing + AR
  * autocorrelation (ACF) and partial autocorrelation (PACF, Durbin-Levinson)
  * differencing and seasonal differencing (what turns trending, seasonal data into something stationary)
  * AR(p) by least squares, order choice by AIC, ARMA(p, q) by conditional sum of squares
  * multi-step forecasts of the *original* series with prediction intervals from the psi-weights
  * comparison with a seasonal-naive forecast over many forecast origins

Animated:
  ts_ar1.gif         AR(1) with phi from -0.8 to 1.0: the series, its theoretical vs measured ACF (stationarity and the random walk)
  ts_stationary.gif  raw -> seasonal difference -> ACF/PACF: how differencing removes trend and season
  ts_forecast.gif    rolling 14-day forecasts with 80% / 95% bands, refitted at five origins

Run:  python arima.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from scipy.optimize import minimize

SEASON = 7


# ------------------------------------------------------------------- data
def visitors(n=364, seed=6):
    r = rng_(seed)
    t = np.arange(n)
    week = np.array([60, 80, 90, 85, 70, -140, -165], float)            # Mon..Sun effect
    noise = np.zeros(n)
    e = r.normal(0, 22, n)
    for i in range(1, n):
        noise[i] = 0.6 * noise[i - 1] + e[i]
    y = 800 + 1.1 * t + week[t % 7] + noise
    return y


def sim_ar1(phi, n=300, seed=1):
    r = rng_(seed)
    e = r.normal(size=n); x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# ------------------------------------------------------------------- basics
def acf(x, nlags=30):
    x = np.asarray(x, float) - np.mean(x)
    c0 = np.dot(x, x) / len(x)
    return np.array([np.dot(x[: len(x) - k], x[k:]) / len(x) / c0 for k in range(nlags + 1)])


def pacf(x, nlags=30):
    """Partial autocorrelation via the Durbin-Levinson recursion on the sample ACF."""
    r = acf(x, nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    out = [1.0]
    for k in range(1, nlags + 1):
        if k == 1:
            phi[1, 1] = r[1]
        else:
            num = r[k] - phi[k - 1, 1:k] @ r[1:k][::-1]
            den = 1 - phi[k - 1, 1:k] @ r[1:k]
            phi[k, k] = num / den
            phi[k, 1:k] = phi[k - 1, 1:k] - phi[k, k] * phi[k - 1, 1:k][::-1]
        out.append(phi[k, k])
    return np.array(out)


def diff(x, lag=1):
    return x[lag:] - x[:-lag]


# ------------------------------------------------------------------- AR / ARMA fitting
def fit_ar(z, p):
    """z_t = c + phi_1 z_{t-1} + ... + phi_p z_{t-p} + e_t   by ordinary least squares."""
    n = len(z)
    if p == 0:
        c = z.mean(); res = z - c
        return dict(c=c, phi=np.array([]), sigma2=res.var(), n=n, p=0)
    A = np.column_stack([np.ones(n - p)] + [z[p - i - 1: n - i - 1] for i in range(p)])
    b, *_ = np.linalg.lstsq(A, z[p:], rcond=None)
    res = z[p:] - A @ b
    return dict(c=b[0], phi=b[1:], sigma2=res.var(), n=n - p, p=p)


def aic(m):
    return (m["n"] + m["p"]) * np.log(m["sigma2"]) + 2 * (m["p"] + 2)      # same effective sample size for every order


def fit_arma_css(z, p, q):
    """ARMA(p,q) with constant by conditional sum of squares: e_t = z_t - c - sum phi z_{t-i} - sum theta e_{t-j}."""
    def resid(par):
        c, phi, th = par[0], par[1: 1 + p], par[1 + p:]
        e = np.zeros(len(z))
        for t in range(max(p, q), len(z)):
            ar = sum(phi[i] * z[t - i - 1] for i in range(p))
            ma = sum(th[j] * e[t - j - 1] for j in range(q))
            e[t] = z[t] - c - ar - ma
        return e[max(p, q):]
    start = np.r_[np.mean(z) * (1 - 0.0), np.zeros(p + q)]
    res = minimize(lambda par: np.sum(resid(par) ** 2), start, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
    e = resid(res.x)
    return dict(c=res.x[0], phi=res.x[1: 1 + p], theta=res.x[1 + p:], sigma2=e.var(), n=len(e), p=p, q=q)


# ------------------------------------------------------------------- the seasonal-difference model + forecasts
class SeasonalAR:
    """z_t = y_t - y_{t-s} is modelled as AR(p) with a constant; forecasts are integrated back to the original scale.
    Equivalent to ARIMA(p, 0, 0)(0, 1, 0)_s with drift."""

    def __init__(self, p=2, s=SEASON):
        self.p, self.s = p, s

    def fit(self, y):
        self.y = np.asarray(y, float)
        z = diff(self.y, self.s)
        self.m = fit_ar(z, self.p)
        self.z = z
        return self

    def psi(self, h):
        """psi-weights of y_t = sum a_i y_{t-i} + c + e_t, with the AR polynomial phi(B)(1 - B^s) expanded."""
        a = np.zeros(self.p + self.s + 1)
        poly = np.r_[1.0, -self.m["phi"]]                        # phi(B) = 1 - phi1 B - ...
        sea = np.zeros(self.s + 1); sea[0], sea[self.s] = 1.0, -1.0
        full = np.convolve(poly, sea)                            # coefficients of the product polynomial
        a = -full[1:]
        psi = np.zeros(h); psi[0] = 1.0
        for j in range(1, h):
            psi[j] = sum(a[i] * psi[j - i - 1] for i in range(min(j, len(a))) if j - i - 1 >= 0)
        return psi

    def forecast(self, h):
        y, z = list(self.y), list(self.z)
        for _ in range(h):
            zn = self.m["c"] + sum(self.m["phi"][i] * z[-i - 1] for i in range(self.p))
            z.append(zn); y.append(y[-self.s] + zn)
        mean = np.array(y[-h:])
        sd = np.sqrt(self.m["sigma2"] * np.cumsum(self.psi(h) ** 2))
        return mean, sd


class RegAR:
    """Regression with AR errors:  y_t = a + b t + (day-of-week effect) + u_t,   u_t = phi_1 u_{t-1} + ... + e_t.
    Trend and weekly pattern come from least squares; the AR model captures the day-to-day momentum left in the residuals.
    (Right when the weekly pattern is *stable*; seasonal differencing is right when it drifts.)"""

    def __init__(self, p=1, s=SEASON):
        self.p, self.s = p, s

    def design(self, t):
        D = np.zeros((len(t), self.s - 1))
        for k in range(1, self.s):
            D[:, k - 1] = (t % self.s == k)
        return np.column_stack([np.ones(len(t)), t, D])

    def fit(self, y):
        self.y = np.asarray(y, float); self.n = len(y)
        t = np.arange(self.n)
        self.beta, *_ = np.linalg.lstsq(self.design(t), self.y, rcond=None)
        self.resid = self.y - self.design(t) @ self.beta
        self.m = fit_ar(self.resid, self.p)
        return self

    def psi(self, h):
        psi = np.zeros(h); psi[0] = 1.0
        for j in range(1, h):
            psi[j] = sum(self.m["phi"][i] * psi[j - i - 1] for i in range(min(j, self.p)))
        return psi

    def forecast(self, h):
        t = np.arange(self.n, self.n + h)
        u = list(self.resid)
        for _ in range(h):
            u.append(self.m["c"] + sum(self.m["phi"][i] * u[-i - 1] for i in range(self.p)))
        mean = self.design(t) @ self.beta + np.array(u[-h:])
        sd = np.sqrt(self.m["sigma2"] * np.cumsum(self.psi(h) ** 2))
        return mean, sd


def seasonal_naive(y, h, s=SEASON):
    return np.array([y[len(y) - s + (i % s)] for i in range(h)])


# ------------------------------------------------------------------- animations
def stem(ax, vals, color, nlags, band=None):
    ax.vlines(np.arange(len(vals)), 0, vals, color=color, lw=2)
    ax.scatter(np.arange(len(vals)), vals, s=10, color=color, zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    if band is not None:
        ax.axhspan(-band, band, color=BLUE, alpha=0.12, lw=0)
    ax.set(xlim=(-0.5, nlags + 0.5), ylim=(-1.05, 1.05))


def ar1_gif(path):
    phis = [-0.8, 0.0, 0.5, 0.8, 0.95, 1.0]
    nl = 20
    with Animator(path, fps=1.2, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.6), title="AR(1):  x(t) = φ x(t-1) + noise", gridspec_kw={"width_ratios": [1.5, 1]})
        for phi in phis:
            a1.clear(); a2.clear()
            x = sim_ar1(phi, 300)
            a1.plot(x, color=BLUE, lw=1.2)
            a1.set(title=f"φ = {phi}", xlabel="time", ylabel="x", xlim=(0, 300))
            r = acf(x, nl)
            stem(a2, r, AQUA, nl, band=1.96 / np.sqrt(len(x)))
            a2.plot(np.arange(nl + 1), phi ** np.arange(nl + 1), color=ORANGE, lw=1.8, label="theory φᵏ")
            a2.set(title="Autocorrelation function", xlabel="lag k"); a2.legend(loc="upper right")
            if phi < 0:
                msg = "Negative φ: the series flips sign every step; the ACF alternates"
            elif phi == 0:
                msg = "φ = 0: white noise, no memory; every autocorrelation is inside the band"
            elif phi < 0.9:
                msg = f"φ = {phi}: yesterday's value carries over; the ACF decays geometrically like {phi}^k (theory in orange)"
            elif phi < 1:
                msg = f"φ = {phi}: very long memory, slow mean reversion; nearly non-stationary"
            else:
                msg = "φ = 1: a random walk; no mean to return to, variance grows with time (non-stationary). Differencing it gives white noise"
            caption(fig, msg)
            anim.grab(fig)


def stationary_gif(path, y):
    t = np.arange(len(y))
    s1 = y
    s2 = diff(y, SEASON)
    s3 = RegAR(1).fit(y).resid
    stages = [("1. Raw daily visitors", s1, "trend + weekly rhythm: the ACF stays high at every lag (non-stationary)"),
              ("2. Seasonal difference  y(t) − y(t−7)", s2, "trend and weekly pattern are gone, but the ACF dips to about -0.57 at lag 7: the signature of differencing a pattern that was stable"),
              ("3. Residuals of trend + day-of-week regression", s3, "clean AR(1) fingerprint: the ACF decays geometrically and the PACF has one spike at lag 1 (0.6): choose AR(1)")]
    nl = 28
    with Animator(path, fps=0.8, hold=3.0) as anim:
        fig, ((a1, a2), (a3, a4)) = new_fig(2, 2, figsize=(13, 6), title="Differencing turns a trending, seasonal series into a stationary one",
                                              gridspec_kw={"width_ratios": [1.6, 1]})
        for title, s, note in stages:
            for a in (a1, a2, a3, a4):
                a.clear()
            a1.plot(s, color=BLUE, lw=1.1); a1.set(title=title, xlim=(0, len(y)))
            a2.axis("off")
            band = 1.96 / np.sqrt(len(s))
            r, pr = acf(s, nl), pacf(s, nl)
            stem(a3, r, AQUA, nl, band); a3.set(title="ACF (shaded = ±1.96/√n)", xlabel="lag")
            stem(a4, pr, YELLOW, nl, band); a4.set(title="PACF", xlabel="lag")
            a2.text(0.0, 0.5, f"mean {s.mean():.1f}\nsd {s.std():.1f}\nACF lag 1: {r[1]:.2f}\nACF lag 7: {r[7]:.2f}\nPACF lag 1: {pr[1]:.2f}\nPACF lag 2: {pr[2]:.2f}",
                    fontsize=11, va="center", transform=a2.transAxes, color=INK)
            caption(fig, note)
            anim.grab(fig)


def forecast_gif(path, y, p=1, h=14, n_train=336):
    origins = [n_train - 28, n_train - 21, n_train - 14, n_train - 7, n_train]
    origins = [o for o in origins if o + h <= len(y)]
    with Animator(path, fps=0.8, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.8), title=f"Rolling {h}-day forecasts (model refitted at each origin)", gridspec_kw={"width_ratios": [1.8, 1]})
        for o in origins:
            a1.clear(); a2.clear()
            m = RegAR(p).fit(y[:o])
            mean, sd = m.forecast(h)
            lo0 = max(0, o - 70)
            tt = np.arange(lo0, o + h)
            a1.plot(np.arange(lo0, o), y[lo0:o], color=INK2, lw=1.4, label="history")
            a1.plot(np.arange(o, o + h), y[o:o + h], color=INK, lw=1.4, ls="--", label="what actually happened")
            fut = np.arange(o, o + h)
            a1.fill_between(fut, mean - 1.96 * sd, mean + 1.96 * sd, color=BLUE, alpha=0.15, label="95% band")
            a1.fill_between(fut, mean - 1.28 * sd, mean + 1.28 * sd, color=BLUE, alpha=0.25, label="80% band")
            a1.plot(fut, mean, color=BLUE, lw=2.4, label="forecast")
            a1.plot(fut, seasonal_naive(y[:o], h), color=ORANGE, lw=1.4, marker="o", ms=3, label="seasonal naive")
            a1.axvline(o - 0.5, color=MUTED, lw=1)
            a1.set(title=f"Forecast origin: day {o}", xlabel="day", ylabel="visitors"); a1.legend(loc="upper left", ncol=3, fontsize=7.5)
            err = np.abs(mean - y[o:o + h]); err_n = np.abs(seasonal_naive(y[:o], h) - y[o:o + h])
            a2.plot(np.arange(1, h + 1), 1.96 * sd, color=BLUE, lw=2, label="95% half-width")
            a2.plot(np.arange(1, h + 1), err, "o-", color=AQUA, ms=4, label="|error|, model")
            a2.plot(np.arange(1, h + 1), err_n, "o-", color=ORANGE, ms=4, label="|error|, seasonal naive")
            a2.set(title="Uncertainty grows with horizon", xlabel="days ahead", ylabel="visitors"); a2.legend(loc="upper left", fontsize=8)
            inside = np.mean(np.abs(y[o:o + h] - mean) <= 1.96 * sd)
            caption(fig, f"MAE over the next {h} days: model {err.mean():.1f} vs seasonal naive {err_n.mean():.1f}; {inside:.0%} of actual values inside the 95% band")
            anim.grab(fig)


# ------------------------------------------------------------------- main
def main():
    y = visitors()
    print("Lesson 07: time series")
    r = acf(sim_ar1(0.6, 5000), 3)
    print(f"  sample ACF of AR(1) phi=0.6 (n=5000), lags 1-3: {np.round(r[1:], 3)} vs theory {[0.6, 0.36, 0.216]}")
    pr = pacf(sim_ar1(0.6, 5000), 4)
    print(f"  PACF lags 1-4: {np.round(pr[1:], 3)} (AR(1): cuts off after lag 1)")
    z = diff(y, SEASON)
    print(f"  raw ACF(1) {acf(y)[1]:.2f}, ACF(7) {acf(y)[7]:.2f};  after seasonal difference ACF(1) {acf(z)[1]:.2f} ACF(7) {acf(z)[7]:.2f}; after a further first difference ACF(1) {acf(diff(z))[1]:.2f}")
    print("  AR order by AIC on the seasonally differenced series:", {p: round(aic(fit_ar(z, p)), 1) for p in range(0, 6)})
    reg = RegAR(1).fit(y)
    ur = reg.resid
    print(f"  regression residuals: ACF lags 1-3 {np.round(acf(ur, 3)[1:], 2)}, PACF lags 1-4 {np.round(pacf(ur, 4)[1:], 2)}  (AR(1) signature; true phi 0.6)")
    print("  AR order by AIC on the regression residuals:", {p: round(float(aic(fit_ar(ur, p))), 1) for p in range(0, 5)})
    print(f"  RegAR(1): phi={reg.m['phi'][0]:.3f}, sigma={np.sqrt(reg.m['sigma2']):.1f} (true 22), trend slope {reg.beta[1]:.2f} (true 1.10)")
    r_ = rng_(2); e = r_.normal(size=3000); x = e[1:] + 0.6 * e[:-1]
    arma = fit_arma_css(x, 0, 1)
    print(f"  ARMA CSS check: MA(1) with theta=0.6 recovered theta={arma['theta'][0]:.3f}")
    h, errs, errn, errd, cover = 14, [], [], [], []
    for o in range(200, len(y) - h + 1, 7):
        mean, sd = RegAR(1).fit(y[:o]).forecast(h)
        errs.append(np.abs(mean - y[o:o + h]).mean()); errn.append(np.abs(seasonal_naive(y[:o], h) - y[o:o + h]).mean())
        errd.append(np.abs(SeasonalAR(2).fit(y[:o]).forecast(h)[0] - y[o:o + h]).mean())
        cover.append(np.mean(np.abs(y[o:o + h] - mean) <= 1.96 * sd))
    print(f"  rolling-origin ({len(errs)} origins, 14-day horizon): MAE RegAR {np.mean(errs):.1f} | seasonal-difference AR(2) {np.mean(errd):.1f} | seasonal naive {np.mean(errn):.1f}; RegAR 95% band coverage {np.mean(cover):.2f}")
    try:
        from statsmodels.tsa.arima.model import ARIMA
        sm = ARIMA(y, order=(2, 0, 0), seasonal_order=(0, 1, 0, 7), trend="n").fit()
        print("  [statsmodels] available")
    except Exception as ex:
        print("  (statsmodels not installed; checks done against theory: ACF, PACF and MA recovery above)")
    ar1_gif(here(__file__, "ts_ar1.gif"))
    stationary_gif(here(__file__, "ts_stationary.gif"), y)
    forecast_gif(here(__file__, "ts_forecast.gif"), y)


if __name__ == "__main__":
    main()
