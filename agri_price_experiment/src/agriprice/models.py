"""All forecasting models of the experiment.

Core eight (Appendix C1): naive, seasonal naive, moving average, Ridge, SARIMA,
Decision Tree, Random Forest, HistGradientBoosting.
Diagnostics: ARIMA (non-seasonal), Ridge with crop interactions, MA-4/MA-8 separately.
Advanced (optional, Stage 5): XGBoost, LightGBM.   Quantile HGB for intervals.

ML models are *direct* (one model per horizon) and predict the log change of the
price relative to the latest available price:  y = log(P[t+h] / P_ref[t]).
All learned preprocessing lives inside sklearn Pipelines and is fitted on the
training rows of the current chronological window only.
"""
from __future__ import annotations
import json, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .features import feature_columns
from .splits import train_rows, eval_rows

MODEL_LABEL = {
    "naive": "Naive", "seasonal_naive": "Seasonal naive", "ma": "Moving average",
    "ridge": "Ridge", "sarima": "SARIMA", "arima": "ARIMA (non-seasonal)",
    "decision_tree": "Decision Tree", "random_forest": "Random Forest",
    "hist_gb": "HistGradientBoosting", "ridge_cropint": "Ridge + crop interactions",
    "xgboost": "XGBoost", "lightgbm": "LightGBM",
}
CORE_EIGHT = ["naive", "seasonal_naive", "ma", "ridge", "sarima", "decision_tree", "random_forest", "hist_gb"]
ML_MODELS = ["ridge", "decision_tree", "random_forest", "hist_gb"]


# ----------------------------------------------------------------------------- crop interactions
class CropInteractions(BaseEstimator, TransformerMixin):
    """[X, X*crop_k ...]: lets a linear model have crop-specific slopes (diagnostic only)."""
    def __init__(self, n_num: int = 0, n_market: int = 0, n_crop: int = 4):
        self.n_num, self.n_market, self.n_crop = n_num, n_market, n_crop

    def fit(self, X, y=None): return self

    def transform(self, X):
        X = np.asarray(X)
        num = X[:, : self.n_num]
        crop = X[:, self.n_num + self.n_market: self.n_num + self.n_market + self.n_crop]
        return np.hstack([X] + [num * crop[:, [k]] for k in range(self.n_crop)])


# ----------------------------------------------------------------------------- pipelines
def make_regressor(name: str, params: dict, cfg: dict, seed: int):
    mc = cfg["models"]
    if name in ("ridge", "ridge_cropint"):
        return Ridge(alpha=params.get("alpha", 10.0), random_state=seed)
    if name == "decision_tree":
        return DecisionTreeRegressor(max_depth=params.get("max_depth", 5),
                                     min_samples_leaf=params.get("min_samples_leaf", 20), random_state=seed)
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=params.get("n_estimators", mc["rf_n_estimators"]),
                                     max_depth=params.get("max_depth"), min_samples_leaf=params.get("min_samples_leaf", 5),
                                     max_features=params.get("max_features", 1.0), n_jobs=1, random_state=seed)
    if name.startswith("hist_gb"):
        kw = dict(learning_rate=params.get("learning_rate", 0.05), max_iter=params.get("max_iter", 200),
                  max_depth=params.get("max_depth", 4), l2_regularization=1.0, min_samples_leaf=20,
                  random_state=seed)
        if "quantile" in params:
            kw.update(loss="quantile", quantile=params["quantile"])
        return HistGradientBoostingRegressor(**kw)
    if name == "xgboost":
        from xgboost import XGBRegressor
        a = cfg["advanced"]["xgboost"]
        return XGBRegressor(n_estimators=a["n_estimators"], learning_rate=a["learning_rate"],
                            max_depth=params.get("max_depth", 3), subsample=a["subsample"],
                            colsample_bytree=a["colsample_bytree"], reg_lambda=1.0, n_jobs=1,
                            random_state=seed, verbosity=0)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        a = cfg["advanced"]["lightgbm"]
        return LGBMRegressor(n_estimators=a["n_estimators"], learning_rate=a["learning_rate"],
                             num_leaves=params.get("num_leaves", 7), min_child_samples=a["min_child_samples"],
                             subsample=a["subsample"], subsample_freq=1, colsample_bytree=a["colsample_bytree"],
                             reg_lambda=1.0, n_jobs=1, random_state=seed, verbose=-1)
    raise ValueError(name)


def build_pipeline(name: str, params: dict, fset: str, frame_cols, cfg: dict,
                   include_market_id: bool = True, seed: int = 42) -> Pipeline:
    num, cat = feature_columns(fset, frame_cols)
    if not include_market_id:
        cat = [c for c in cat if c != "market_id"]
    reg = make_regressor(name, params, cfg, seed)
    native_nan = name.startswith("hist_gb") or name in ("xgboost", "lightgbm")
    if native_nan:
        num_t = "passthrough"
    elif name in ("ridge", "ridge_cropint"):
        num_t = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                          ("sc", StandardScaler())])
    else:
        num_t = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True))])
    if name.startswith("hist_gb"):
        cat_t = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        ct = ColumnTransformer([("num", num_t, num), ("cat", cat_t, cat)], remainder="drop",
                               verbose_feature_names_out=False)
        reg.set_params(categorical_features=[False] * len(num) + [True] * len(cat))
        return Pipeline([("prep", ct), ("reg", reg)])
    if name in ("xgboost", "lightgbm"):
        cat_t = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        ct = ColumnTransformer([("num", num_t, num), ("cat", cat_t, cat)], remainder="drop",
                               verbose_feature_names_out=False)
        return Pipeline([("prep", ct), ("reg", reg)])
    cat_t = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ct = ColumnTransformer([("num", num_t, num), ("cat", cat_t, cat)], remainder="drop",
                           verbose_feature_names_out=False)
    steps = [("prep", ct)]
    if name == "ridge_cropint":
        n_market = 5 if "market_id" in cat else 0
        steps.append(("int", CropInteractions(n_num=len(num), n_market=n_market, n_crop=4)))
    steps.append(("reg", reg))
    return Pipeline(steps)


# ----------------------------------------------------------------------------- generic ML fit/predict
def _ml_target(df: pd.DataFrame) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(df["y_obs"].values / df["ref_price"].values)


def training_frame(df: pd.DataFrame, h: int, train_end: int, cfg: dict, handling: bool,
                   exclude_series=None, only_series=None) -> pd.DataFrame:
    m = train_rows(df, h, train_end, cfg["experiment"]["min_origin_week"])
    m &= df["ref_price"].notna() & df["y_obs"].notna() & (df["y_obs"] > 0) & (df["ref_price"] > 0)
    if handling:
        m &= ~df["y_detected_suspect"]
    if exclude_series:
        m &= ~df["series"].isin(exclude_series)
    if only_series:
        m &= df["series"].isin(only_series)
    return df[m]


def evaluation_frame(df: pd.DataFrame, lo: int, hi: int, series=None) -> pd.DataFrame:
    m = eval_rows(df, lo, hi) & df["ref_ok_handled"] & df["ref_price"].notna() & df["y_obs"].notna()
    if series is not None:
        m &= df["series"].isin(series)
    return df[m]


def _finish(ev: pd.DataFrame, pred: np.ndarray, model: str, fset: str, extra: dict | None = None) -> pd.DataFrame:
    keep = ["series", "market_id", "crop_id", "horizon", "origin_week", "target_week", "origin_date",
            "target_date", "ref_price", "y_obs", "y_latent", "y_audit_suspect", "y_event_affected",
            "y_regime", "y_event_ids", "horizon_vol8", "price_stale_weeks" if "price_stale_weeks" in ev else "ref_price"]
    out = ev[list(dict.fromkeys(keep))].copy()
    out["y_pred"] = pred
    out["scored"] = ~out["y_audit_suspect"].astype(bool)
    out["model"], out["feature_set"] = model, fset
    for k, v in (extra or {}).items():
        out[k] = v
    return out


def fit_predict_ml(name: str, params: dict, fset: str, df: pd.DataFrame, h: int, train_end: int,
                   lo: int, hi: int, cfg: dict, handling: bool, exclude_series=None, only_series=None,
                   include_market_id: bool = True, eval_series=None, seed: int | None = None,
                   return_model: bool = False):
    seed = cfg["models"]["random_state"] if seed is None else seed
    tr = training_frame(df, h, train_end, cfg, handling, exclude_series, only_series)
    ev = evaluation_frame(df, lo, hi, eval_series)
    if len(tr) < 50 or len(ev) == 0:
        raise ValueError(f"insufficient rows: train={len(tr)} eval={len(ev)}")
    pipe = build_pipeline(name, params, fset, df.columns, cfg, include_market_id, seed)
    cols = pipe.named_steps["prep"].transformers[0][2] + pipe.named_steps["prep"].transformers[1][2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        pipe.fit(tr[cols], _ml_target(tr))
        t1 = time.perf_counter()
        yhat = pipe.predict(ev[cols])
        t2 = time.perf_counter()
    yhat = np.clip(yhat, -1.2, 1.2)
    out = _finish(ev, ev["ref_price"].values * np.exp(yhat), name, fset)
    out["log_change_pred"] = yhat
    out["n_train"], out["fit_s"], out["pred_s"] = len(tr), t1 - t0, t2 - t1
    out["seed"], out["params"] = seed, json.dumps(params, sort_keys=True)
    if return_model:
        return out, pipe, cols, ev
    return out


# ----------------------------------------------------------------------------- baselines
def baseline_predictions(df: pd.DataFrame, h: int, lo: int, hi: int, cfg: dict,
                         ma_window: int | None = None) -> pd.DataFrame:
    ev = evaluation_frame(df, lo, hi)
    outs = []
    naive = ev["ref_price"].values
    outs.append(_finish(ev, naive, "naive", "-"))
    sn = ev["base_seasonal_naive"].values
    fb = np.isnan(sn) | (sn <= 0)
    o = _finish(ev, np.where(fb, naive, sn), "seasonal_naive", "-")
    o["fallback_to_naive"] = fb
    outs.append(o)
    for W in cfg["models"]["moving_average_windows"]:
        ma = ev[f"base_ma{W}"].values
        fb = np.isnan(ma) | (ma <= 0)
        o = _finish(ev, np.where(fb, naive, ma), f"ma{W}", "-")
        o["fallback_to_naive"] = fb
        outs.append(o)
    return pd.concat(outs, ignore_index=True)


# ----------------------------------------------------------------------------- SARIMA / ARIMA
def sarima_predictions(store, df_by_h: dict, cfg: dict, fit_end: int, lo: int, hi: int,
                       seasonal: bool = True) -> pd.DataFrame:
    """Per-series SARIMA on log price. Fitted once on weeks <= fit_end (order picked by AIC among
    cfg orders), then re-filtered with each new observation to forecast 1..4 weeks ahead.
    Native multi-step (iterated) forecasts - the one exception to the direct-forecast rule."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    sc = cfg["models"]["sarima"]
    P = store.panel
    T, M, C = P.T, len(P.markets), len(P.crops)
    if store.handling:
        series_price = np.where(np.isnan(store.accepted), np.nan, store.accepted)
    else:
        series_price = np.where(P.arrays["wholesale_price_kes_kg"] > 0, P.arrays["wholesale_price_kes_kg"], np.nan)
    horizons = sorted(df_by_h)
    fc_store: dict = {}
    fallbacks = 0
    n_fail = 0
    for m in range(M):
        for c in range(C):
            key = f"{P.markets[m]}|{P.crops[c]}"
            y = np.log(series_price[:, m, c])
            best, best_aic = None, np.inf
            for order in sc["nonseasonal_orders"]:
                so = tuple(sc["seasonal_order"]) + (sc["seasonal_period"],) if seasonal else (0, 0, 0, 0)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod = SARIMAX(y[:fit_end], order=tuple(order), seasonal_order=so, trend="c",
                                      enforce_stationarity=False, enforce_invertibility=False)
                        res = mod.fit(disp=False, maxiter=60)
                    if np.isfinite(res.aic) and res.aic < best_aic:
                        best, best_aic = (res, order, so), res.aic
                except Exception:
                    continue
            if best is None:
                n_fail += 1
                continue
            res, order, so = best
            cur, prev = res, fit_end                       # origin week t uses weeks <= t only
            for t in range(lo, hi + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cur = cur.extend(y[prev:t]); prev = t   # advance Kalman state by new observations
                        fc_store[(key, t)] = np.exp(np.asarray(cur.forecast(max(horizons))))
                except Exception:
                    n_fail += 1
    outs = []
    label = "sarima" if seasonal else "arima"
    for h in horizons:
        ev = evaluation_frame(df_by_h[h], lo, hi)
        pred = np.array([fc_store[(s, t)][h - 1] if (s, t) in fc_store else np.nan
                         for s, t in zip(ev["series"], ev["origin_week"])])
        bad = ~np.isfinite(pred) | (pred <= 0) | (pred > 20 * ev["ref_price"].values)
        pred = np.where(bad, ev["ref_price"].values, pred)
        o = _finish(ev, pred, label, "-")
        o["fallback_to_naive"] = bad
        outs.append(o)
    return pd.concat(outs, ignore_index=True)
