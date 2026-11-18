"""
course_utils.py -- shared helpers for the Time-Series course.

Philosophy: each lesson TEACHES a technique by writing it out in the open
(e.g. you will write the metrics yourself in Lesson 3).  Afterwards we import the
tidy, tested versions from here so later notebooks stay short and readable.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- data loading
def find_data_dir():
    for cand in [Path("data"), Path("../data"), Path("../../data")]:
        if (cand / "agri_prices_weekly.csv").exists():
            return cand
    raise FileNotFoundError("Cannot find the data/ folder. Start Jupyter from inside the course folder.")

DATA = find_data_dir()

def load_weekly(commodity=None):
    """Weekly multi-commodity market file. If commodity is given, returns that commodity
    as a DataFrame indexed by week-ending Sunday (freq='W-SUN')."""
    df = pd.read_csv(DATA / "agri_prices_weekly.csv", parse_dates=["date"])
    if commodity is None:
        return df
    out = df[df.commodity == commodity].drop(columns="commodity").set_index("date").sort_index()
    out.index.freq = "W-SUN"
    return out

def load_airline():
    a = pd.read_csv(DATA / "airline_passengers.csv")
    a["month"] = pd.to_datetime(a["month"])
    s = a.set_index("month")["passengers"].asfreq("MS")
    return s

def load_co2():
    c = pd.read_csv(DATA / "co2_weekly.csv", parse_dates=["date"]).set_index("date")["co2_ppm"]
    return c

# --------------------------------------------------------------------- metrics
def mae(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return np.mean(np.abs(y - yhat))

def rmse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return np.sqrt(np.mean((y - yhat) ** 2))

def mape(y, yhat, eps=1e-8):
    """Mean absolute % error. Returns NaN if any actual is (near) zero -- on purpose!"""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    if np.any(np.abs(y) < eps):
        return np.nan
    return 100 * np.mean(np.abs((y - yhat) / y))

def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = (np.abs(y) + np.abs(yhat))
    ratio = np.where(denom == 0, 0.0, 2 * np.abs(y - yhat) / np.where(denom == 0, 1, denom))
    return 100 * np.mean(ratio)

def mase(y, yhat, y_train, m=1):
    """Mean Absolute Scaled Error: MAE divided by the in-sample MAE of the (seasonal-)naive forecast."""
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    return mae(y, yhat) / scale

def score_table(y, forecasts: dict, y_train=None, m=1):
    """Compare several forecasts at once -> tidy DataFrame sorted by RMSE."""
    rows = {}
    for name, f in forecasts.items():
        r = dict(MAE=mae(y, f), RMSE=rmse(y, f), MAPE=mape(y, f), sMAPE=smape(y, f))
        if y_train is not None:
            r["MASE"] = mase(y, f, y_train, m)
        rows[name] = r
    return pd.DataFrame(rows).T.sort_values("RMSE").round(3)


# ------------------------------------------------------- feature engineering
def calendar_features(index, fourier_k=2):
    """Deterministic calendar features: known in advance for any future date, so they can't leak."""
    ang = 2 * np.pi * index.dayofyear.values / 365.25
    out = pd.DataFrame(index=index)
    out["month"] = index.month
    out["week_of_year"] = index.isocalendar().week.astype(int).values
    for k in range(1, fourier_k + 1):
        out[f"sin{k}"] = np.sin(k * ang)
        out[f"cos{k}"] = np.cos(k * ang)
    return out


def fourier_terms(index, K=3, period=365.25):
    """K pairs of sin/cos harmonics of the yearly cycle -- the standard way to model long seasonal periods."""
    ang = 2 * np.pi * index.dayofyear.values / period
    cols = {}
    for k in range(1, K + 1):
        cols[f"sin{k}"] = np.sin(k * ang)
        cols[f"cos{k}"] = np.cos(k * ang)
    return pd.DataFrame(cols, index=index)

def make_features(df, target="price", lags=(1, 2, 3, 4, 8, 52), windows=(4, 12, 26),
                  exog=(), exog_lags=(1, 4, 8), h=1, fourier_k=2, add_calendar=True):
    """
    Leakage-safe supervised table for forecasting `h` steps ahead.

    Row t holds everything you may know at the forecast origin and the target y[t]
    (so every lag / rolling window is computed on data at least `h` steps old).
    Convention: a feature named price_lag_k is the price k steps BEFORE the target time (k >= h).
    """
    y = df[target]
    X = pd.DataFrame(index=df.index)
    for k in lags:
        assert k >= h, f"lag {k} < horizon {h}: that would peek into the future"
        X[f"{target}_lag_{k}"] = y.shift(k)
    base = y.shift(h)                                   # newest value we are allowed to see
    for w in windows:
        X[f"{target}_rollmean_{w}"] = base.rolling(w).mean()
        X[f"{target}_rollstd_{w}"] = base.rolling(w).std()
    for c in exog:
        for k in exog_lags:
            assert k >= h, f"exog lag {k} < horizon {h}"
            X[f"{c}_lag_{k}"] = df[c].shift(k)
    if add_calendar:
        X = X.join(calendar_features(df.index, fourier_k))
    return X, y


# ------------------------------------------------ ML table: scale-free (return) features
def make_ml_table(df, h=1, price="price", K=3, use_drivers=True):
    """
    Supervised table for h-step-ahead forecasting of PRICE RETURNS (a stationary, scale-free target).

        target  y_t = log(p_t) - log(p_{t-h})            the h-step log return
        features: only data from time <= t-h  ->  no leakage by construction

    Why returns? Tree models cannot predict outside the range of targets they saw in training, so a
    trending price level breaks them; returns stay in a stable range.  Forecast = p_{t-h} * exp(prediction).
    Drivers are lagged >= 8 weeks, so they are valid for every horizon h <= 8.
    """
    lp = np.log(df[price])
    base = lp.shift(h)                                           # newest log-price we may use
    X = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 4, 8, 13, 26, 52):
        X[f"ret_{k}w"] = base - lp.shift(h + k)                  # past k-week returns ending at 'base'
    X["dev_ma12"] = base - lp.rolling(12).mean().shift(h)        # distance from 12-week mean (mean reversion)
    X["vol_13"] = lp.diff().rolling(13).std().shift(h)           # recent volatility
    if use_drivers:
        assert h <= 8, "drivers are lagged 8 weeks: valid for h <= 8 only"
        X["rain6_l8"] = df["rainfall_mm"].rolling(6).mean().shift(8) / 30
        X["fuel_l8"] = np.log(df["fuel_index"] / 100).shift(8)
        X["fest"] = df["festival_flag"]
    X = X.join(fourier_terms(df.index, K))
    y = (lp - base).rename(f"ret_target_h{h}")
    return X, y

def walk_forward_ml(df, model_factory, horizons=(1, 4, 8), initial=470, step=8, price="price", **table_kw):
    """
    Rolling-origin evaluation of a DIRECT multi-horizon ML forecaster (one model per horizon).
    Output has the same schema as walk_forward(): origin, target_date, h, actual, forecast.
    The feature tables are built once -- safe, because row t only contains information from t-h or earlier.
    """
    p = df[price]
    tables = {h: make_ml_table(df, h, price=price, **table_kw) for h in horizons}
    rows = []
    for origin in range(initial, len(df) - max(horizons) + 1, step):
        last_obs = df.index[origin - 1]
        for h in horizons:
            X, y = tables[h]
            Xtr, ytr = X.loc[:last_obs], y.loc[:last_obs]
            ok = Xtr.notna().all(axis=1) & ytr.notna()
            model = model_factory().fit(Xtr[ok], ytr[ok])
            tgt = df.index[origin - 1 + h]
            pred_ret = float(model.predict(X.loc[[tgt]])[0])
            rows.append((last_obs, tgt, h, p.loc[tgt], p.iloc[origin - 1] * np.exp(pred_ret)))
    return pd.DataFrame(rows, columns=["origin", "target_date", "h", "actual", "forecast"])

def leakage_check(feature_fn, series, cut_points=(0.5, 0.7, 0.9), seed=0):
    """
    The 'future-poisoning' test.  Replace everything AFTER a cut point with garbage, recompute the
    features, and verify that every row up to the cut is unchanged.  If a row changes, that feature
    used information from its own future.  Returns True if no leakage detected.
    """
    rng = np.random.default_rng(seed)
    base = feature_fn(series)
    for frac in cut_points:
        cut = int(len(series) * frac)
        poisoned = series.copy()
        poisoned.iloc[cut:] = rng.normal(1e3, 1e2, len(series) - cut)
        alt = feature_fn(poisoned)
        a, b = base.iloc[:cut], alt.iloc[:cut]
        same = ((a == b) | (a.isna() & b.isna())).all().all() if isinstance(a, pd.DataFrame) else ((a == b) | (a.isna() & b.isna())).all()
        if not same:
            bad = [c for c in a.columns if not ((a[c] == b[c]) | (a[c].isna() & b[c].isna())).all()] if isinstance(a, pd.DataFrame) else []
            print(f"🚨 LEAKAGE: rows before the {frac:.0%} cut changed. Culprit column(s): {bad}")
            return False
    print("✅ No leakage detected: past rows never changed when the future was poisoned.")
    return True


# ----------------------------------------------------------- baseline forecasters
# Every forecaster has the signature f(train: pd.Series, h: int) -> np.ndarray of length h
def fc_naive(train, h):
    """Tomorrow = today."""
    return np.repeat(float(train.iloc[-1]), h)

def fc_mean(train, h):
    return np.repeat(float(train.mean()), h)

def fc_ma(train, h, w=8):
    """Flat forecast at the mean of the last w observations."""
    return np.repeat(float(train.iloc[-w:].mean()), h)

def fc_drift(train, h):
    """Extend the straight line between first and last observation."""
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    return float(train.iloc[-1]) + slope * np.arange(1, h + 1)

def fc_snaive(train, h, m=52):
    """Same period last season (m = season length)."""
    last = train.iloc[-m:].values
    return np.resize(last, h)


# ------------------------------------------------- exponential-smoothing forecasters
def fc_ses(train, h):
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    return SimpleExpSmoothing(np.asarray(train, float), initialization_method="estimated").fit().forecast(h)

def fc_holt(train, h, damped=True):
    from statsmodels.tsa.holtwinters import Holt
    return Holt(np.asarray(train, float), damped_trend=damped, initialization_method="estimated").fit().forecast(h)

def fc_hw(train, h, m=52, seasonal="mul", damped=True):
    """Holt-Winters. 'heuristic' initialisation keeps weekly (m=52) fits fast and stable."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    mod = ExponentialSmoothing(np.asarray(train, float), trend="add", damped_trend=damped, seasonal=seasonal,
                               seasonal_periods=m, initialization_method="heuristic")
    return mod.fit().forecast(h)


# ------------------------------------------------------- SARIMAX forecaster factory
def build_exog(df, K=3):
    """Exogenous table for weekly market data: Fourier seasonality + drivers lagged 8 weeks + festival flag.
       Everything here is known for forecast horizons up to h = 8."""
    X = fourier_terms(df.index, K)
    X["rain6_l8"] = df["rainfall_mm"].rolling(6).mean().shift(8) / 30
    X["fuel_l8"] = np.log(df["fuel_index"] / 100).shift(8)
    X["fest"] = df["festival_flag"]
    return X

def sarimax_forecaster(df, cols=(), order=(1, 1, 1), K=3, price="price"):
    """Returns f(train, h) -> price forecasts, fitting SARIMAX on log price with the chosen exog columns."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    X = build_exog(df, K)
    cols = list(cols)
    valid = X.dropna().index
    def f(train, h):
        idx = train.index.intersection(valid)
        tr = np.log(train.loc[idx])
        fut = pd.date_range(train.index[-1], periods=h + 1, freq="W-SUN")[1:]
        mod = SARIMAX(tr, exog=X.loc[idx, cols] if cols else None, order=order, trend="n").fit(disp=False)
        return np.exp(mod.forecast(h, exog=X.loc[fut, cols] if cols else None).values)
    return f

# -------------------------------------------------------------- walk-forward
def walk_forward(series, fit_predict, initial, horizon=1, step=1, expanding=True, window=None):
    """
    Generic rolling-origin evaluation.

    series      : pd.Series indexed by time
    fit_predict : f(train_series, horizon) -> array of `horizon` forecasts
    initial     : number of observations in the first training window
    horizon     : how many steps ahead each forecast looks
    step        : how far the origin moves between folds
    expanding   : True = training window grows; False = fixed-size sliding window of `window`

    Returns a long DataFrame: origin, target_date, h, actual, forecast.
    """
    y = series.dropna()
    rows = []
    for origin in range(initial, len(y) - horizon + 1, step):
        lo = 0 if expanding else origin - (window or initial)
        train = y.iloc[lo:origin]
        fc = np.asarray(fit_predict(train, horizon), float)
        for h in range(1, horizon + 1):
            rows.append((y.index[origin - 1], y.index[origin + h - 1], h, y.iloc[origin + h - 1], fc[h - 1]))
    return pd.DataFrame(rows, columns=["origin", "target_date", "h", "actual", "forecast"])

def scores_by_horizon(wf: pd.DataFrame):
    g = wf.groupby("h")
    return pd.DataFrame({"MAE": g.apply(lambda d: mae(d.actual, d.forecast), include_groups=False),
                         "RMSE": g.apply(lambda d: rmse(d.actual, d.forecast), include_groups=False)})

# ------------------------------------------------------------------- plotting
def set_style():
    plt.rcParams.update({
        "figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": .25,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "legend.frameon": False, "figure.dpi": 100,
    })

def plot_forecasts(train, test, forecasts: dict, title="Forecast comparison", last_train=104):
    """Plot the tail of the training data, the actual test data and any number of forecasts."""
    fig, ax = plt.subplots()
    ax.plot(train.index[-last_train:], train.values[-last_train:], color="#666", label="train")
    ax.plot(test.index, test.values, color="black", lw=2, label="actual")
    for name, f in forecasts.items():
        f = pd.Series(np.asarray(f, float), index=test.index[:len(f)])
        ax.plot(f.index, f.values, lw=1.6, label=name)
    ax.axvline(test.index[0], color="red", ls=":", lw=1)
    ax.set_title(title); ax.legend(ncol=3)
    return ax

# -------------------------------------------------------------- self-checking
def check(label, value, expected, tol=1e-6):
    """Friendly auto-check. check('rows', len(df), 626)"""
    if value is None:
        print(f"⏳ {label}: not answered yet (replace the `None` with your answer)")
        return
    try:
        ok = abs(float(value) - float(expected)) <= tol
    except Exception:
        ok = value == expected
    print(("✅ " if ok else "❌ ") + f"{label}: " + ("correct!" if ok else f"got {value!r} -- not quite, re-read the hints"))

set_style()
