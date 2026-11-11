"""
# 35 · Time-Series Forecasting: Smoothing, Holt-Winters & Autoregression
**Family:** Supervised · Forecasting

## Definition
Time-series models predict future values from the past, exploiting **trend** (long-run direction), **seasonality** (repeating cycles) and **autocorrelation** (today depends on yesterday).
We compare three classics on monthly sales: the **seasonal-naive** baseline, **Holt-Winters** (triple exponential smoothing) and an **autoregressive AR(p)** model fitted by least squares.

## Intuition
- Seasonal naive: "next March will look like last March." Hard to beat on simple data.
- Holt-Winters: keep three running estimates, a *level*, a *trend*, and 12 *seasonal offsets*, each updated by exponential smoothing: recent data counts more.
- AR(p): regress today on the last p values; with p ≥ 12 it can learn the yearly cycle and the trend from the data itself.
Always evaluate with **walk-forward (rolling-origin) validation**: never shuffle time series, and never train on the future.

## The math
    Holt-Winters (additive):   ℓₜ = α(yₜ − sₜ₋ₘ) + (1−α)(ℓₜ₋₁ + bₜ₋₁)
                               bₜ = β(ℓₜ − ℓₜ₋₁) + (1−β) bₜ₋₁          sₜ = γ(yₜ − ℓₜ) + (1−γ) sₜ₋ₘ
                               forecast  ŷₜ₊ₕ = ℓₜ + h·bₜ + sₜ₊ₕ₋ₘ
    AR(p):  yₜ = c + φ₁yₜ₋₁ + … + φ_p yₜ₋p + εₜ          fitted by least squares; multi-step forecasts feed predictions back in.
    ARIMA(p,d,q) = AR + differencing d times (removes trend) + moving-average errors q; SARIMA adds seasonal terms.
    Metrics: MAE, RMSE, MAPE, MASE (scaled vs naive).  Modern: Prophet, ETS, gradient boosting on lag features, N-BEATS, transformers.

## What the animation shows
The forecast origin walks forward through the history. Top: the next 12 months predicted by each method against the truth. Bottom: running average MAE of each method across origins so far.

## Try this
- Increase the noise in `data.time_series` and see which method degrades most.
- Change p of the AR model (4, 12, 24).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from mlatlas import data, viz
from mlatlas.cli import run

M = 12


def hw(y, al, be, ga, h=M):
    l, b = y[:M].mean(), (y[M:2 * M].mean() - y[:M].mean()) / M; s = list(y[:M] - l)
    for t in range(M, len(y)):
        l0 = l; l = al * (y[t] - s[t - M]) + (1 - al) * (l + b); b = be * (l - l0) + (1 - be) * b; s.append(ga * (y[t] - l) + (1 - ga) * s[t - M])
    return np.array([l + (k + 1) * b + s[len(y) - M + (k % M)] for k in range(h)])


def ar_fit_forecast(y, p=14, h=M):
    X = np.array([np.r_[1, y[t - p:t][::-1]] for t in range(p, len(y))]); phi = np.linalg.lstsq(X, y[p:], rcond=None)[0]; hist = list(y)
    for _ in range(h): hist.append(phi @ np.r_[1, np.array(hist[-p:])[::-1]])
    return np.array(hist[-h:])


def main(gif, quick):
    y = data.time_series(180, seed=1); grid = [(a, b, g) for a in (.1, .3, .5) for b in (.01, .05, .1) for g in (.1, .3, .5)]
    sse = lambda pr: sum((hw(y[:t], *pr, h=1)[0] - y[t]) ** 2 for t in range(48, 72)); par = min(grid, key=sse); print(f"  Holt-Winters parameters chosen on the first 6 years: α,β,γ = {par}")
    origins = list(range(72, 180 - M + 1, 12 if quick else 4)); names = ["seasonal naive", "Holt-Winters", "AR(14)"]; F = {t: [y[t - M:t], hw(y[:t], *par), ar_fit_forecast(y[:t])] for t in origins}
    mae = {t: [np.abs(f - y[t:t + M]).mean() for f in F[t]] for t in origins}; avg = {n: np.mean([mae[t][i] for t in origins]) for i, n in enumerate(names)}; print("  average 12-month MAE:", {k: round(float(v), 2) for k, v in avg.items()})
    fig = plt.figure(figsize=(11.5, 6), constrained_layout=True); gs = fig.add_gridspec(2, 1, height_ratios=[2, 1]); a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1])

    def update(t):
        a1.clear(); a2.clear(); a1.plot(range(t), y[:t], color="#888", lw=1.3, label="history"); a1.plot(range(t, t + M), y[t:t + M], "k", lw=2.4, label="actual future"); a1.plot(range(t, 180), y[t:], color="#ddd", lw=1, zorder=0)
        for f, n, c in zip(F[t], names, viz.PALETTE): a1.plot(range(t, t + M), f, "o-", ms=3, lw=2, color=c, label=n)
        a1.axvline(t, color="gray", ls=":"); a1.set(xlim=(0, 185), title=f"Forecast origin = month {t}: predicting the next 12 months", xlabel="month", ylabel="sales"); a1.legend(loc="upper left", ncol=5, fontsize=7)
        done = [o for o in origins if o <= t]
        for i, (n, c) in enumerate(zip(names, viz.PALETTE)): a2.plot(done, [np.mean([mae[o][i] for o in done[:j + 1]]) for j in range(len(done))], "o-", ms=3, color=c, label=n)
        a2.set(xlim=(origins[0] - 2, origins[-1] + 2), ylim=(0, 12), title="Running mean absolute error across forecast origins", xlabel="origin month"); a2.legend(loc="upper right", ncol=3, fontsize=7)
    viz.animate(fig, update, origins, gif, fps=3)


if __name__ == "__main__":
    run(main, __file__)
