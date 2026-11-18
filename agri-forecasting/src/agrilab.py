"""
agrilab.py  --  the small helper library that travels with this course.
=======================================================================

Everything in here is *readable on purpose*. Open it, read it, change it.
Sections:

  1. Paths & small utilities (seeds, plotting style, the friendly auto-checker)
  2. Synthetic agricultural market data (the course's main dataset)
  3. Forecast metrics + walk-forward evaluation helpers
  4. Synthetic image datasets (shapes, diseased leaves) for Phase 8

Why synthetic data?
-------------------
Real market data is messy, licensed, and changes underneath you. For *learning*
we want data where we KNOW the truth (trend, seasonality, shocks), so we can
check whether our models discover it. The generator below builds prices from
documented ingredients: trend + seasonality + rainfall effect + fuel cost +
random shocks + autocorrelated noise. When you finish the course, swap in a
real dataset -- every notebook is written so that is a one-line change.
"""
from __future__ import annotations

import os
import pathlib
import random

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# 1. Paths & small utilities
# --------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MARKETS = ["Central", "Highland", "Harbor"]
COMMODITIES = ["maize", "tomato", "onion", "potato"]


def set_seed(seed: int = 42) -> None:
    """Make 'random' reproducible: Python, NumPy and (if installed) PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # torch is optional for Phase 7
        pass


def setup_plots() -> None:
    """One consistent, readable plotting style for every notebook."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (11, 4),
            "figure.dpi": 100,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
        }
    )


def check(label: str, condition: bool, hint: str = "") -> bool:
    """A friendly auto-grader.  Prints a tick or a nudge instead of a scary traceback."""
    if bool(condition):
        print(f"\u2705  {label}")
        return True
    print(f"\u274c  {label}")
    if hint:
        print(f"    \u21b3 hint: {hint}")
    return False


# --------------------------------------------------------------------------
# 2. Synthetic agricultural market data
# --------------------------------------------------------------------------
# Each commodity has its own 'personality'.
#   base         typical price per kg at the start (local currency units)
#   trend        log-growth per year (inflation + demand growth)
#   amp          strength of the yearly price cycle
#   peak_doy     day-of-year when prices tend to peak (lean season before harvest)
#   vol          standard deviation of slow-moving market noise (log scale)
#   rain_sens    how strongly (lagged) rainfall depresses price (more rain -> more supply)
_PROFILES = {
    "maize":  dict(base=42, trend=0.085, amp=0.16, peak_doy=150, vol=0.045, rain_sens=0.060, n_shocks=3),
    "tomato": dict(base=62, trend=0.070, amp=0.26, peak_doy=75,  vol=0.085, rain_sens=0.110, n_shocks=5),
    "onion":  dict(base=70, trend=0.075, amp=0.20, peak_doy=200, vol=0.070, rain_sens=0.070, n_shocks=4),
    "potato": dict(base=48, trend=0.080, amp=0.14, peak_doy=120, vol=0.050, rain_sens=0.080, n_shocks=3),
}
_MARKET_PROFILES = {
    "Central":  dict(mult=1.00, noise=0.025, miss=0.03),
    "Highland": dict(mult=0.88, noise=0.040, miss=0.07),
    "Harbor":   dict(mult=1.12, noise=0.030, miss=0.04),
}


def _ar1(n: int, phi: float, sigma_marginal: float, rng: np.random.Generator) -> np.ndarray:
    """AR(1) noise whose *marginal* standard deviation is sigma_marginal."""
    from scipy.signal import lfilter

    eps = rng.normal(0, sigma_marginal * np.sqrt(1 - phi**2), n)
    return lfilter([1.0], [1.0, -phi], eps)


def make_weather(start="2015-01-01", end="2024-12-31", seed=7) -> pd.DataFrame:
    """Daily rainfall (mm) and temperature (C) with TWO rainy seasons per year."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    doy = dates.dayofyear.to_numpy()
    wet = 0.5 * (1 + np.cos(4 * np.pi * (doy - 110) / 365.25))  # peaks ~ day 110 and ~day 293
    year_factor = {y: float(np.exp(rng.normal(0, 0.25))) for y in np.unique(dates.year)}
    yf = np.array([year_factor[y] for y in dates.year])
    rains = rng.random(len(dates)) < (0.08 + 0.55 * wet)
    amount = rng.gamma(1.2, 9.0 * (0.4 + wet) * yf)
    rain = np.where(rains, amount, 0.0).round(1)
    temp = 22 + 3 * np.sin(2 * np.pi * (doy - 60) / 365.25) - 2.0 * wet + rng.normal(0, 1.2, len(dates))
    return pd.DataFrame({"date": dates, "rainfall_mm": rain, "temp_c": temp.round(1)})


def make_market_prices(seed: int = 2024, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the full daily panel: 4 commodities x 3 markets, 2015-2024."""
    rng = np.random.default_rng(seed)
    if weather is None:
        weather = make_weather()
    dates = pd.DatetimeIndex(weather["date"])
    n = len(dates)
    t_years = np.arange(n) / 365.25
    doy = dates.dayofyear.to_numpy()

    # Fuel index: slow random walk with upward drift (transport cost driver)
    fuel = 100 * np.exp(np.cumsum(rng.normal(0.00005, 0.004, n)) + 0.02 * t_years)
    fuel_z = (np.log(fuel) - np.log(fuel).mean()) / np.log(fuel).std()

    # Rainfall: 60-day accumulated rain, shifted 45 days (crops take time to reach the market)
    rain_acc = weather["rainfall_mm"].rolling(60, min_periods=1).sum()
    rain_z = ((rain_acc - rain_acc.mean()) / rain_acc.std()).shift(45).bfill().to_numpy()

    # A shared 'transport disruption' shock in spring 2020 hits everything.
    d2020 = (dates >= "2020-03-20") & (dates <= "2020-06-30")
    global_shock = np.where(d2020, 0.16 * np.exp(-(np.arange(n) - np.argmax(d2020)) / 90.0), 0.0)

    frames = []
    for com in COMMODITIES:
        p = _PROFILES[com]
        seasonal = p["amp"] * (
            np.cos(2 * np.pi * (doy - p["peak_doy"]) / 365.25)
            + 0.35 * np.cos(4 * np.pi * (doy - p["peak_doy"] - 30) / 365.25)
        )
        # Commodity-specific shocks (pests, floods, export bans...): rise fast, decay slowly.
        shock = np.zeros(n)
        for _ in range(p["n_shocks"]):
            s = int(rng.integers(200, n - 120))
            mag = float(rng.choice([-1, 1]) * rng.uniform(0.18, 0.40))
            k = np.arange(n - s)
            shock[s:] += mag * np.exp(-k / rng.uniform(25, 60))
        national = (
            np.log(p["base"])
            + p["trend"] * t_years
            + seasonal
            - p["rain_sens"] * rain_z
            + 0.05 * fuel_z
            + shock
            + global_shock
            + _ar1(n, 0.985, p["vol"], rng)
        )
        for mkt in MARKETS:
            mp = _MARKET_PROFILES[mkt]
            logp = national + np.log(mp["mult"]) + _ar1(n, 0.90, mp["noise"], rng) + rng.normal(0, 0.012, n)
            price = np.exp(logp)
            volume = 60 * np.exp(-1.2 * (logp - logp.mean())) * np.exp(rng.normal(0, 0.18, n))
            df = pd.DataFrame(
                {
                    "date": dates,
                    "market": mkt,
                    "commodity": com,
                    "price_per_kg": price.round(2),
                    "volume_tonnes": volume.round(1),
                    "fuel_index": fuel.round(1),
                }
            )
            df = df[df["date"].dt.dayofweek != 6]  # markets are closed on Sundays
            df = df[rng.random(len(df)) > mp["miss"]]  # random missing records
            if mkt == "Highland":  # two multi-week outages (road washed out, etc.)
                df = df[~df["date"].between("2017-03-06", "2017-03-24")]
                df = df[~df["date"].between("2021-08-02", "2021-08-20")]
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)

    # A handful of data-entry errors (decimal slips) in Highland -- you will hunt them in Notebook P7-02.
    hl = out.index[(out["market"] == "Highland")].to_numpy()
    for i in rng.choice(hl, 7, replace=False):
        out.loc[i, "price_per_kg"] = round(out.loc[i, "price_per_kg"] * rng.choice([10, 0.1]), 2)
    return out.sort_values(["commodity", "market", "date"]).reset_index(drop=True)


def generate_datasets(outdir: pathlib.Path | str = DATA) -> None:
    """(Re)create every CSV/NPZ file in data/.  Deterministic: same files every time."""
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    weather = make_weather()
    weather.to_csv(outdir / "weather_daily.csv", index=False)
    make_market_prices(weather=weather).to_csv(outdir / "market_prices_daily.csv", index=False)
    X, y, names = make_leaf_dataset(n_per_class=600, size=48, seed=11)
    np.savez_compressed(outdir / "leaf_disease_synthetic.npz", X=X, y=y, class_names=np.array(names))
    print("Datasets written to", outdir)


def load_prices(commodity: str | None = None, market: str | None = None) -> pd.DataFrame:
    """Load the daily price panel (optionally filtered). Dates are parsed for you."""
    df = pd.read_csv(DATA / "market_prices_daily.csv", parse_dates=["date"])
    if commodity:
        df = df[df["commodity"] == commodity]
    if market:
        df = df[df["market"] == market]
    return df.reset_index(drop=True)


def load_weather() -> pd.DataFrame:
    return pd.read_csv(DATA / "weather_daily.csv", parse_dates=["date"])


def price_series(commodity="tomato", market="Central", freq="W", clean=True) -> pd.Series:
    """
    Convenience: a clean, regular price series ready for modelling.
    freq: 'D' daily (forward-filled), 'W' weekly mean, 'ME' monthly mean.
    clean=True removes the decimal-slip outliers (prices >3x or <1/3 of a 31-day rolling median).
    """
    df = load_prices(commodity, market).set_index("date")["price_per_kg"]
    if clean:
        med = df.rolling(31, center=True, min_periods=5).median()
        df = df.where((df < 3 * med) & (df > med / 3))
    s = df.resample("D").mean()
    s = s.interpolate(limit_direction="both")
    if freq != "D":
        s = s.resample(freq).mean()
    s.name = f"{commodity}_{market}_{freq}"
    return s


# --------------------------------------------------------------------------
# 3. Metrics & walk-forward evaluation
# --------------------------------------------------------------------------
def mae(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%). Refuses to divide by (near-)zero actuals."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if np.any(np.abs(y) < eps):
        raise ValueError("MAPE undefined: some actual values are ~0. Use MAE/RMSE/sMAPE/MASE instead.")
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def smape(y, yhat) -> float:
    """Symmetric MAPE (%), bounded between 0 and 200; safe when one value is zero."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y - yhat) / denom) * 100)


def mase(y, yhat, y_train, m: int = 1) -> float:
    """Mean Absolute Scaled Error: error relative to the in-sample seasonal-naive error. <1 beats naive."""
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return float(mae(y, yhat) / scale)


# ---- baseline forecasters: every serious model must beat these ----------------
def naive_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Tomorrow = today.  Repeat the last observed value h times."""
    return np.repeat(float(train.iloc[-1]), h)


def seasonal_naive_forecast(train: pd.Series, h: int, m: int = 52) -> np.ndarray:
    """Next season = last season.  Repeat the last m observations (m=52 for weekly data, 12 for monthly)."""
    last = train.iloc[-m:].to_numpy(float)
    return np.array([last[i % m] for i in range(h)])


def drift_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Naive + the average historical slope (draw a line from first to last observation and extend it)."""
    slope = (float(train.iloc[-1]) - float(train.iloc[0])) / (len(train) - 1)
    return float(train.iloc[-1]) + slope * np.arange(1, h + 1)


def mean_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Always predict the historical average (a *terrible* baseline for trending data)."""
    return np.repeat(float(train.mean()), h)


def walk_forward(y: pd.Series, fit_predict, initial: int, horizon: int = 1, step: int = 1, expanding: bool = True):
    """
    Rolling-origin evaluation -- the honest way to test a forecaster.

    y            : the full time-ordered series
    fit_predict  : function(train_series, horizon) -> array of `horizon` predictions
    initial      : size of the first training window
    horizon      : how many steps ahead each forecast must reach
    step         : how far the origin moves each round
    expanding    : True = grow the training set; False = fixed-size sliding window

    Returns a DataFrame with columns origin_date, target_date, h, actual, forecast.
    """
    rows = []
    for end in range(initial, len(y) - horizon + 1, step):
        start = 0 if expanding else end - initial
        train = y.iloc[start:end]
        preds = np.asarray(fit_predict(train, horizon), float)
        for h in range(1, horizon + 1):
            rows.append(
                dict(
                    origin_date=y.index[end - 1],
                    target_date=y.index[end + h - 1],
                    h=h,
                    actual=float(y.iloc[end + h - 1]),
                    forecast=float(preds[h - 1]),
                )
            )
    return pd.DataFrame(rows)


def score_table(results: dict, y_train=None, m: int = 1) -> pd.DataFrame:
    """results = {model_name: DataFrame from walk_forward}.  Returns a tidy leaderboard."""
    rows = {}
    for name, r in results.items():
        d = dict(MAE=mae(r.actual, r.forecast), RMSE=rmse(r.actual, r.forecast), MAPE=mape(r.actual, r.forecast))
        if y_train is not None:
            d["MASE"] = mase(r.actual, r.forecast, y_train, m)
        rows[name] = d
    return pd.DataFrame(rows).T.sort_values("MAE").round(3)


def make_ts_features(
    y: pd.Series,
    lags=(1, 2, 3, 4, 52),
    roll_windows=(4, 13),
    calendar: bool = True,
    fourier_k: int = 0,
    horizon: int = 1,
    exog: pd.DataFrame | None = None,
    period: float = 365.25,
) -> pd.DataFrame:
    """
    Leak-free supervised table for forecasting.

    Row t describes the TARGET y_t using only information available `horizon` steps before t.
      lag_k      = y_(t-k)                        (k >= horizon)
      rmean_w    = mean of the w values ending at y_(t-horizon)
      rstd_w     = std  of the same window
      month/doy  = calendar features of t itself (the calendar is known in advance -> no leak)
      fourier_*  = sin/cos pairs for yearly seasonality (if fourier_k > 0)
      exog_*     = exogenous columns, shifted by `horizon` so they are known at forecast time
    """
    if min(lags) < horizon:
        raise ValueError(f"every lag must be >= horizon ({horizon}); a lag of {min(lags)} would peek into the future")
    out = pd.DataFrame({"y": y})
    for k in lags:
        out[f"lag_{k}"] = y.shift(k)
    base = y.shift(horizon)
    for w in roll_windows:
        out[f"rmean_{w}"] = base.rolling(w).mean()
        out[f"rstd_{w}"] = base.rolling(w).std()
    if calendar:
        doy = y.index.dayofyear.to_numpy()
        out["month"] = y.index.month
        out["doy_sin"] = np.sin(2 * np.pi * doy / period)
        out["doy_cos"] = np.cos(2 * np.pi * doy / period)
    for k in range(1, fourier_k + 1):
        doy = y.index.dayofyear.to_numpy()
        out[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * doy / period)
        out[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * doy / period)
    if exog is not None:
        for c in exog.columns:
            out[f"exog_{c}"] = exog[c].reindex(y.index).shift(horizon)
    return out


def make_origin_table(y: pd.Series, h: int, exog: pd.DataFrame | None = None, period: float = 365.25) -> pd.DataFrame:
    """
    Supervised table for DIRECT h-step forecasting of a positive price series.

    Row t is a forecast ORIGIN: every feature uses information up to and including t only.
      target    = log(y_{t+h}) - log(y_t)         the log-CHANGE over the next h steps  (NaN for the last h rows)
      mom_k     = log(y_t) - log(y_{t-k})          momentum over the last k steps
      vol_8     = std of the last 8 one-step log-returns
      dev_52    = log(y_t) - mean(log y over the last 52 steps)    (mean-reversion signal)
      tsin/tcos = calendar position of the TARGET date (known in advance)
      <exog>, <exog>_lag4, <exog>_lag8 = observed drivers (e.g. rainfall) at the origin
    Predicting a CHANGE instead of a LEVEL lets tree models handle trending prices (they cannot extrapolate levels).
    """
    ly = np.log(y)
    out = pd.DataFrame(index=y.index)
    out["target"] = ly.shift(-h) - ly
    for k in (1, 2, 3, 4, 8, 13, 26, 52):
        out[f"mom_{k}"] = ly - ly.shift(k)
    out["vol_8"] = ly.diff().rolling(8).std()
    out["dev_52"] = ly - ly.rolling(52).mean()
    step = pd.infer_freq(y.index[:8]) or "W"
    td = (y.index + h * pd.tseries.frequencies.to_offset(step)).dayofyear.to_numpy()
    out["tsin"], out["tcos"] = np.sin(2 * np.pi * td / period), np.cos(2 * np.pi * td / period)
    if exog is not None:
        for c in exog.columns:
            e = exog[c].reindex(y.index)
            out[c], out[f"{c}_lag4"], out[f"{c}_lag8"] = e, e.shift(4), e.shift(8)
    return out


def direct_ml_forecaster(make_model, exog: pd.DataFrame | None = None):
    """
    Turn ANY scikit-learn style regressor into a walk_forward-compatible forecaster (DIRECT strategy:
    one model per horizon).  `make_model` is a zero-argument function returning a fresh, unfitted model.
    Everything is rebuilt from `train` only, so nothing from the future can leak in.
    """

    def fit_predict(train: pd.Series, h: int) -> np.ndarray:
        preds = []
        for k in range(1, h + 1):
            tab = make_origin_table(train, k, exog)
            fit_rows = tab.dropna()
            model = make_model()
            model.fit(fit_rows.drop(columns="target"), fit_rows["target"])
            x_now = tab.drop(columns="target").iloc[[-1]]
            preds.append(float(train.iloc[-1]) * float(np.exp(model.predict(x_now)[0])))
        return np.array(preds)

    return fit_predict


def make_windows(arr: np.ndarray, lookback: int, horizon: int = 1):
    """Turn a 1-D array into supervised (X, y) windows for neural nets. X: (n, lookback), y: (n, horizon)."""
    arr = np.asarray(arr, float)
    X, y = [], []
    for i in range(len(arr) - lookback - horizon + 1):
        X.append(arr[i : i + lookback])
        y.append(arr[i + lookback : i + lookback + horizon])
    return np.array(X), np.array(y)


# --------------------------------------------------------------------------
# 4. Synthetic image datasets (offline, reproducible, tiny)
# --------------------------------------------------------------------------
def _grid(size):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return xx, yy


def _disk(xx, yy, cx, cy, r, soft=1.0):
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return np.clip((r - d) / soft + 0.5, 0, 1)


def make_shapes_dataset(n_per_class=300, size=32, seed=0):
    """4 classes of coloured, rotated shapes on noisy backgrounds: circle, square, triangle, cross."""
    names = ["circle", "square", "triangle", "cross"]
    rng = np.random.default_rng(seed)
    xx, yy = _grid(size)
    X, y = [], []
    for label in range(4):
        for _ in range(n_per_class):
            cx, cy = size / 2 + rng.uniform(-3, 3, 2)
            r = rng.uniform(0.22, 0.34) * size
            th = rng.uniform(0, 2 * np.pi)
            u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
            v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
            if label == 0:
                m = _disk(xx, yy, cx, cy, r)
            elif label == 1:
                m = ((np.abs(u) <= r * 0.9) & (np.abs(v) <= r * 0.9)).astype(np.float32)
            elif label == 2:
                m = ((v >= -r * 0.6) & (v <= r * 1.0 - 1.8 * np.abs(u))).astype(np.float32)
            else:
                w = r * 0.32
                m = (((np.abs(u) <= w) & (np.abs(v) <= r)) | ((np.abs(v) <= w) & (np.abs(u) <= r))).astype(np.float32)
            bg = rng.uniform(0.05, 0.45, 3)
            fg = rng.uniform(0.55, 1.0, 3)
            if rng.random() < 0.5:
                bg, fg = fg, bg  # dark-on-light OR light-on-dark
            img = bg[None, None, :] * (1 - m[..., None]) + fg[None, None, :] * m[..., None]
            img = img + rng.normal(0, 0.05, img.shape)
            X.append(np.clip(img, 0, 1))
            y.append(label)
    X = (np.array(X) * 255).astype(np.uint8)
    y = np.array(y)
    p = rng.permutation(len(y))
    return X[p], y[p], list(names)


def make_leaf_dataset(n_per_class=500, size=48, seed=0):
    """
    Synthetic 'crop disease' images.  Four classes:
      healthy   - clean green leaf (maybe a stray speck or two)
      leaf_spot - several small dark spots with yellow halos
      blight    - one large dark patch creeping in from the leaf edge
      rust      - many tiny orange pustules
    Real data (PlantVillage) is far richer; this set lets you train a CNN on a laptop in seconds.
    """
    names = ["healthy", "leaf_spot", "blight", "rust"]
    rng = np.random.default_rng(seed)
    xx, yy = _grid(size)
    X, y = [], []
    for label in range(4):
        for _ in range(n_per_class):
            bg = rng.choice([0, 1, 2])
            bgc = [np.array([0.45, 0.36, 0.28]), np.array([0.62, 0.62, 0.60]), np.array([0.30, 0.28, 0.26])][bg]
            img = np.ones((size, size, 3), np.float32) * (bgc + rng.normal(0, 0.03, 3))
            cx, cy = size / 2 + rng.uniform(-3, 3, 2)
            a, b = rng.uniform(0.33, 0.44) * size, rng.uniform(0.17, 0.26) * size
            th = rng.uniform(0, np.pi)
            u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
            v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
            rr = np.sqrt((u / a) ** 2 + (v / b) ** 2)
            alpha = np.clip((1 - rr) * 6, 0, 1)
            green = np.array([0.18, 0.52, 0.20]) + rng.normal(0, 0.04, 3)
            leaf = np.ones((size, size, 3), np.float32) * green
            leaf *= (0.85 + 0.25 * (u / a + 1) / 2)[..., None]       # light gradient along the leaf
            leaf[np.abs(v) < 0.8] += 0.10                            # pale midrib
            inside = rr < 0.78

            def paint(mask, color, strength=1.0):
                nonlocal leaf
                m = (mask * (rr < 0.95))[..., None] * strength
                leaf = leaf * (1 - m) + np.array(color, np.float32) * m

            def random_point(rmax=0.7):
                while True:
                    px, py = rng.uniform(0, size, 2)
                    ru = ((px - cx) * np.cos(th) + (py - cy) * np.sin(th)) / a
                    rv = (-(px - cx) * np.sin(th) + (py - cy) * np.cos(th)) / b
                    if np.sqrt(ru**2 + rv**2) < rmax:
                        return px, py

            if label == 0:
                for _ in range(rng.integers(0, 3)):
                    px, py = random_point()
                    paint(_disk(xx, yy, px, py, 0.9), (0.3, 0.35, 0.15), 0.5)
            elif label == 1:
                for _ in range(rng.integers(3, 8)):
                    px, py = random_point()
                    r0 = rng.uniform(1.4, 2.6)
                    paint(_disk(xx, yy, px, py, r0 + 1.5), (0.80, 0.75, 0.25), 0.8)   # halo
                    paint(_disk(xx, yy, px, py, r0), (0.35, 0.22, 0.10), 1.0)          # lesion
            elif label == 2:
                while True:
                    px, py = random_point(0.9)
                    ru = ((px - cx) * np.cos(th) + (py - cy) * np.sin(th)) / a
                    rv = (-(px - cx) * np.sin(th) + (py - cy) * np.cos(th)) / b
                    if np.sqrt(ru**2 + rv**2) > 0.45:
                        break
                big = _disk(xx, yy, px, py, rng.uniform(6, 10), soft=2.5)
                big = big * (0.75 + 0.25 * rng.random((size, size)))
                paint(big, (0.25, 0.18, 0.12), 0.95)
            else:
                for _ in range(rng.integers(22, 45)):
                    px, py = random_point(0.85)
                    paint(_disk(xx, yy, px, py, 0.85), (0.85, 0.42, 0.10), 0.95)
            img = img * (1 - alpha[..., None]) + leaf * alpha[..., None]
            img = img * rng.uniform(0.8, 1.15) + rng.normal(0, 0.025, img.shape)
            X.append(np.clip(img, 0, 1))
            y.append(label)
    X = (np.array(X) * 255).astype(np.uint8)
    y = np.array(y)
    p = rng.permutation(len(y))
    return X[p], y[p], list(names)


def load_leaf_dataset():
    d = np.load(DATA / "leaf_disease_synthetic.npz")
    return d["X"], d["y"], [str(c) for c in d["class_names"]]


if __name__ == "__main__":
    generate_datasets()
