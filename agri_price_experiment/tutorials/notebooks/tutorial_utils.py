"""Small helpers shared by the tutorial notebooks.

IMPORTANT - what this file is and is not
----------------------------------------
* It is the TUTORIALS' OWN simplified implementation. It does NOT import anything from ``src/agriprice``.
  The real experiment lives in ``src/agriprice``; when the two disagree, the pipeline is right and this file is
  the one to fix.
* It only READS the shipped data/results (``data/`` and ``results/``); it never writes into them.
* Everything instructive (the leakage demos, the cleaning rule, the bootstrap, the splits) is written out
  step by step inside the notebooks; this module holds only plumbing and the compact feature builder that
  several notebooks reuse (notebook 08 walks through it).

Tutorial version: 1.0  -  last checked against pipeline version 1.0.0 (agri_price_v1, seed 42)
"""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent] + list(here.parents):
        if (p / "config" / "experiment_config.yaml").exists():
            return p
    raise FileNotFoundError("Could not find the project root (config/experiment_config.yaml). "
                            "Keep the tutorials/ folder inside the project folder.")


ROOT = _find_root()
DATA, RESULTS = ROOT / "data", ROOT / "results"
MARKETS = {"MKT-01": "Nairobi", "MKT-02": "Nakuru", "MKT-03": "Eldoret", "MKT-04": "Kisumu", "MKT-05": "Mombasa"}
CROPS = {"CRP-01": "Maize", "CRP-02": "Tomatoes", "CRP-03": "Irish potatoes", "CRP-04": "Onions"}
TRAIN_END, VAL_END, TEST_END = 104, 130, 156
HORIZONS = (1, 2, 4)
START_DATE = pd.Timestamp("2023-01-02")


def style():
    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 9, "figure.figsize": (9, 3.6)})


# ----------------------------------------------------------------------------- loading
def load_panel(scenario: str = "realistic") -> pd.DataFrame:
    """One of 'clean', 'realistic', 'stressed' (data/synthetic_panel_<scenario>.csv)."""
    df = pd.read_csv(DATA / f"synthetic_panel_{scenario}.csv")
    df["series"] = df["market_id"] + "|" + df["crop_id"]
    return df.sort_values(["series", "week_index"]).reset_index(drop=True)


def wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """weeks x series table of one column (index = week_index, columns = 'MKT-xx|CRP-xx')."""
    return df.pivot(index="week_index", columns="series", values=col).sort_index()


def network_weights() -> dict:
    """Source-market weights of the generator's market network (read from the shipped config)."""
    import yaml
    with open(ROOT / "config" / "experiment_config.yaml") as f:
        return yaml.safe_load(f)["network"]


def mae(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs(y - p)))


# ----------------------------------------------------------------------------- causal cleaning (tutorial copy; notebook 03 explains it)
def causal_outlier_filter(x, thr=0.70, ref_window=5, agree_tol=0.15):
    """Reject implausible prices using only the past. Returns (accepted, rejected_mask)."""
    x = np.asarray(x, float)
    acc = np.full(len(x), np.nan)
    rej = np.zeros(len(x), dtype=bool)
    hist, prev = [], np.nan
    for t, v in enumerate(x):
        if np.isnan(v):
            prev = np.nan
            continue
        ok = True
        if v <= 0:
            ok = False
        elif len(hist) >= 3:
            ref = np.median(hist[-ref_window:])
            if abs(np.log(v / ref)) > thr:
                ok = (not np.isnan(prev)) and prev > 0 and abs(np.log(v / prev)) < agree_tol
        if ok:
            acc[t] = v
            hist.append(v)
        else:
            rej[t] = True
        prev = v
    return acc, rej


def clean_price_wide(raw: pd.DataFrame):
    """Apply the causal filter column by column. Returns (accepted, rejected) as weeks x series frames."""
    acc, rej = {}, {}
    for c in raw.columns:
        a, r = causal_outlier_filter(raw[c].values)
        acc[c], rej[c] = a, r
    return (pd.DataFrame(acc, index=raw.index), pd.DataFrame(rej, index=raw.index))


# ----------------------------------------------------------------------------- features (compact, simplified)
FEATURE_GROUPS = {
    "price": ["lr_1", "lr_2", "lr_4", "lr_8", "lr_13", "dev_mean_4", "dev_mean_8", "dev_mean_13", "vol_8", "seasonal_ref"],
    "calendar": ["tgt_sin1", "tgt_cos1", "tgt_sin2", "tgt_cos2"],
    "supply_demand": ["arr_anom13", "dem_anom13"],
    "weather": ["rain_mean4", "rain_sum13", "drought_l0", "drought_max8"],
    "logistics": ["fuel_chg4", "trans_chg4"],
    "cross_market": ["conn_gap0", "conn_chg1", "regsup_l0"],
}
ABLATION_SETS = {
    "E0 price+calendar": ["price", "calendar"],
    "E1 +supply/demand": ["price", "calendar", "supply_demand"],
    "E2 +weather": ["price", "calendar", "supply_demand", "weather"],
    "E3 +logistics": ["price", "calendar", "supply_demand", "weather", "logistics"],
    "E4 +cross-market": ["price", "calendar", "supply_demand", "weather", "logistics", "cross_market"],
}


def columns_for(groups) -> list[str]:
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def _asof0(x: pd.DataFrame, delay: pd.DataFrame, handling: bool) -> pd.DataFrame:
    """Latest value of an indicator AS KNOWN at the origin: a record flagged delay=1 is not visible yet."""
    v = x.where(delay == 0)
    return v.fillna(x.shift(1)) if handling else v


def build_features(df: pd.DataFrame, h: int, handling: bool = True, min_origin: int = 14) -> pd.DataFrame:
    """Long table: one row per (series, origin week) with simplified features and the target.

    handling=True  -> causal outlier filter + last-observation-carried-forward (cap 8 weeks), indicators carried forward (cap 4)
    handling=False -> raw values; gaps stay NaN, reporting errors stay in
    Target:  y = log(price[t+h] / latest price[t]);  prediction = latest price * exp(y_hat).
    """
    raw = wide(df, "wholesale_price_kes_kg")
    delay = wide(df, "report_delay_weeks")
    if handling:
        acc, rej = clean_price_wide(raw)
        price = acc.ffill(limit=8)
    else:
        price = raw.ffill()
        rej = pd.DataFrame(False, index=raw.index, columns=raw.columns)
    lp = np.log(price)
    ind = {c: wide(df, c) for c in ["market_arrivals_tonnes", "estimated_demand_index", "rainfall_mm", "drought_index",
                                    "fuel_cost_index", "transport_cost_kes_tonne", "regional_supply_index"]}
    if handling:
        ind = {k: v.ffill(limit=4) for k, v in ind.items()}
    F = {}
    for k in (1, 2, 4, 8, 13):
        F[f"lr_{k}"] = lp - lp.shift(k)
    for k in (4, 8, 13):
        F[f"dev_mean_{k}"] = lp - lp.rolling(k, min_periods=k // 2).mean()
    F["vol_8"] = lp.diff().rolling(8, min_periods=5).std()
    F["seasonal_ref"] = lp.shift(52 - h) - lp                      # price one year before the TARGET week
    # calendar facts about the target week (pure date arithmetic -> known in advance)
    weeks = lp.index.values
    tdate = START_DATE + pd.to_timedelta((weeks + h - 1) * 7 + 3, unit="D")
    doy = tdate.dayofyear.values.astype(float)
    for name, f in [("tgt_sin1", np.sin(2 * np.pi * doy / 365.25)), ("tgt_cos1", np.cos(2 * np.pi * doy / 365.25)),
                    ("tgt_sin2", np.sin(4 * np.pi * doy / 365.25)), ("tgt_cos2", np.cos(4 * np.pi * doy / 365.25))]:
        F[name] = pd.DataFrame(np.repeat(f[:, None], lp.shape[1], axis=1), index=lp.index, columns=lp.columns)
    # supply / demand (delay-aware)
    a, d = ind["market_arrivals_tonnes"], ind["estimated_demand_index"]
    F["arr_anom13"] = np.log(_asof0(a, delay, handling)) - np.log(a.rolling(13, min_periods=6).mean().shift(1))
    F["dem_anom13"] = np.log(_asof0(d, delay, handling)) - np.log(d.rolling(13, min_periods=6).mean().shift(1))
    # weather (on time) and logistics
    F["rain_mean4"] = ind["rainfall_mm"].rolling(4, min_periods=2).mean()
    F["rain_sum13"] = ind["rainfall_mm"].rolling(13, min_periods=7).sum()
    F["drought_l0"] = ind["drought_index"]
    F["drought_max8"] = ind["drought_index"].rolling(8, min_periods=4).max()
    fu, tr = ind["fuel_cost_index"], ind["transport_cost_kes_tonne"]
    F["fuel_chg4"] = np.log(fu) - np.log(fu.shift(4))
    F["trans_chg4"] = np.log(_asof0(tr, delay, handling)) - np.log(tr.shift(4))
    # cross-market: weighted price of the source markets (same crop) vs own price
    net = network_weights()
    gap, chg1 = {}, {}
    for s in lp.columns:
        m, c = s.split("|")
        srcs = net[m]
        gap[s] = sum(w * lp[f"{j}|{c}"] for j, w in srcs.items()) - lp[s]
        chg1[s] = sum(w * (lp[f"{j}|{c}"] - lp[f"{j}|{c}"].shift(1)) for j, w in srcs.items())
    F["conn_gap0"], F["conn_chg1"] = pd.DataFrame(gap), pd.DataFrame(chg1)
    F["regsup_l0"] = np.log(_asof0(ind["regional_supply_index"], delay, handling))
    # targets and audit information (audit labels are used for SCORING only, never as inputs)
    y_obs = raw.shift(-h)
    suspect = wide(df, "data_quality_flag").eq("suspect").shift(-h, fill_value=False)
    out = {"ref_price": price, "y_obs": y_obs, "y_suspect": suspect, "y_rejected": rej.shift(-h, fill_value=False),
           "ma4": price.rolling(4, min_periods=2).mean(), "ma8": price.rolling(8, min_periods=4).mean(),
           "price_l52h": price.shift(52 - h)}
    out.update(F)
    long = pd.concat({k: v.stack(future_stack=True) for k, v in out.items()}, axis=1)
    long.index.names = ["origin_week", "series"]
    long = long.reset_index()
    long["market_id"] = long.series.str[:6]; long["crop_id"] = long.series.str[7:]
    long["target_week"] = long.origin_week + h
    long["horizon"] = h
    long = long[(long.origin_week >= min_origin) & (long.target_week <= TEST_END)].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        long["y"] = np.log(long.y_obs / long.ref_price)
    return long.reset_index(drop=True)


# ----------------------------------------------------------------------------- model fitting (tutorial version)
def fit_predict(feats: pd.DataFrame, cols: list[str], kind: str, h: int, train_end: int, lo: int, hi: int,
                handling: bool = True, seed: int = 42, include_ids: bool = True, **params) -> pd.DataFrame:
    """Fit on rows whose TARGET week <= train_end, predict origins lo..hi. Returns evaluation rows with y_pred.

    kind: 'ridge' | 'random_forest' | 'hist_gb'
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    tr = feats[(feats.origin_week <= train_end - h) & feats.ref_price.notna() & feats.y_obs.notna() & (feats.y_obs > 0)]
    if handling:
        tr = tr[~tr.y_rejected]
    ev = feats[(feats.origin_week >= lo) & (feats.origin_week <= hi) & feats.ref_price.notna() & feats.y_obs.notna()]
    cat = ["market_id", "crop_id"] if include_ids else []
    if kind == "ridge":
        num_t = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)), ("sc", StandardScaler())])
        reg = Ridge(alpha=params.get("alpha", 100.0))
    elif kind == "random_forest":
        num_t = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True))])
        reg = RandomForestRegressor(n_estimators=params.get("n_estimators", 150), min_samples_leaf=params.get("min_samples_leaf", 5),
                                    max_features=params.get("max_features", 0.5), n_jobs=1, random_state=seed)
    elif kind == "hist_gb":
        num_t = "passthrough"
        reg = HistGradientBoostingRegressor(learning_rate=params.get("learning_rate", 0.05), max_iter=params.get("max_iter", 150),
                                            max_depth=params.get("max_depth", 3), l2_regularization=1.0, random_state=seed)
    else:
        raise ValueError(kind)
    prep = ColumnTransformer([("num", num_t, cols)] + ([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)] if cat else []))
    pipe = Pipeline([("prep", prep), ("reg", reg)])
    pipe.fit(tr[cols + cat], tr["y"])
    yhat = np.clip(pipe.predict(ev[cols + cat]), -1.2, 1.2)
    out = ev.copy()
    out["y_pred"] = out.ref_price * np.exp(yhat)
    out["scored"] = ~out.y_suspect
    out.attrs["pipeline"] = pipe
    return out


def score(pred: pd.DataFrame, by=None) -> pd.DataFrame | float:
    """MAE on the rows that are scored (injected reporting errors in the TARGET are excluded)."""
    d = pred[pred.scored].assign(abs_err=lambda x: (x.y_obs - x.y_pred).abs())
    if by is None:
        return float(d.abs_err.mean())
    return d.groupby(by).abs_err.mean()


def naive_pred(feats: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    ev = feats[(feats.origin_week >= lo) & (feats.origin_week <= hi) & feats.ref_price.notna() & feats.y_obs.notna()].copy()
    ev["y_pred"] = ev.ref_price
    ev["scored"] = ~ev.y_suspect
    return ev


def eval_hi(h: int, hi: int) -> int:
    """Last origin whose target still exists inside the 156 weeks."""
    return min(hi, TEST_END - h)
