"""Analysis helpers: labels, paired block bootstrap, volatility regimes, group permutation importance."""
from __future__ import annotations
import numpy as np
import pandas as pd

from .models import MODEL_LABEL, fit_predict_ml
from .features import FEATURE_GROUPS, FEATURE_SETS
from .metrics import metric_table

ML_SET = {"ridge", "decision_tree", "random_forest", "hist_gb", "ridge_cropint", "xgboost", "lightgbm"}


def variant(row_model: str, fset: str) -> str:
    """Display label of a model variant, e.g. 'Random Forest [E4]'."""
    if row_model.startswith("ma") and row_model[2:].isdigit():
        return f"Moving average ({row_model[2:]}w)"
    base = MODEL_LABEL.get(row_model, row_model)
    return f"{base} [{fset}]" if row_model in ML_SET else base


def add_variant(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["variant"] = [variant(m, f) for m, f in zip(df["model"], df["feature_set"])]
    return df


def abs_err(df): return (df["y_obs"] - df["y_pred"]).abs()


def paired_block_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
                           block: str = "origin_week") -> dict:
    """Paired comparison of mean absolute error of A vs B on identical rows, resampling origin weeks
    (keeps cross-sectional dependence).  Returns mean diff (A-B), 95% CI and share of resamples with A<B."""
    keys = ["series", "origin_week", "horizon"]
    m = a[keys + ["y_obs", "y_pred", "scored"]].merge(b[keys + ["y_pred"]], on=keys, suffixes=("_a", "_b"))
    m = m[m["scored"]]
    m["d"] = (m["y_obs"] - m["y_pred_a"]).abs() - (m["y_obs"] - m["y_pred_b"]).abs()
    g = m.groupby(block)["d"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    mean = s.sum() / c.sum()
    return dict(mean_diff=float(mean), ci_low=float(np.percentile(boots, 2.5)), ci_high=float(np.percentile(boots, 97.5)),
                prob_a_better=float((boots < 0).mean()), n=int(c.sum()))


def vol_threshold(frames: dict, cutoff_origin: int, q: float = 0.75) -> pd.DataFrame:
    """Per-crop (h-independent) volatility threshold from rows with origin <= cutoff (training information only)."""
    f = frames[min(frames)]
    f = f[f["origin_week"] <= cutoff_origin]
    return f.groupby("crop_id")["vol_8"].quantile(q)


def add_regimes(preds: pd.DataFrame, thr: pd.Series) -> pd.DataFrame:
    p = preds.copy()
    p["high_vol"] = p["horizon_vol8"] > p["crop_id"].map(thr)
    return p


def skill_table(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """MAE, and skill = 1 - MAE/MAE(naive) on the same rows, grouped by `by` (+ variant)."""
    t = metric_table(preds, ["variant"] + by)
    naive = t[t["variant"] == "Naive"].drop(columns="variant")[by + ["MAE"]].rename(columns={"MAE": "MAE_naive"}) if by else None
    if by:
        t = t.merge(naive, on=by, how="left")
    else:
        t["MAE_naive"] = t.loc[t["variant"] == "Naive", "MAE"].iloc[0]
    t["skill_vs_naive"] = 1 - t["MAE"] / t["MAE_naive"]
    return t


# ----------------------------------------------------------------------------- permutation importance by group
def group_permutation_importance(model: str, params: dict, fset: str, frame: pd.DataFrame, h: int, train_end: int,
                                 lo: int, hi: int, cfg: dict, handling: bool, n_rep: int = 8, seed: int = 0) -> pd.DataFrame:
    """Increase in MAE (KES/kg) when ALL columns of a feature group are jointly shuffled across evaluation rows."""
    out, pipe, cols, ev = fit_predict_ml(model, params, fset, frame, h, train_end, lo, hi, cfg, handling, return_model=True)
    ev = ev[~ev["y_audit_suspect"]]
    X = ev[cols].copy(); ref = ev["ref_price"].values; y = ev["y_obs"].values

    def mae_of(Xp):
        yh = np.clip(pipe.predict(Xp), -1.2, 1.2)
        return float(np.mean(np.abs(y - ref * np.exp(yh))))

    base = mae_of(X)
    groups = {g: [c for c in FEATURE_GROUPS[g] + (["price_stale_weeks"] if g == "price" else []) if c in cols]
              for g in FEATURE_SETS[fset]}
    groups["identifiers"] = [c for c in ("market_id", "crop_id") if c in cols]
    rng = np.random.default_rng(seed)
    rows = []
    for g, cs in groups.items():
        if not cs:
            continue
        inc = []
        for _ in range(n_rep):
            Xp = X.copy(); perm = rng.permutation(len(Xp))
            Xp[cs] = X[cs].values[perm]
            inc.append(mae_of(Xp) - base)
        rows.append(dict(group=g, horizon=h, mae_increase=float(np.mean(inc)), sd=float(np.std(inc)), baseline_mae=base))
    return pd.DataFrame(rows)
