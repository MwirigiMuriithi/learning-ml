"""Experiment orchestration: data contexts, tuning, main runs, ablations, generalisation, intervals."""
from __future__ import annotations
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from .config import data_dir, results_dir
from .features import FeatureStore, panel_from_frame
from .models import (ML_MODELS, baseline_predictions, fit_predict_ml, sarima_predictions, build_pipeline,
                     training_frame, evaluation_frame, _ml_target, _finish)
from .splits import partitions, rolling_folds
from .metrics import mae

# (scenario, handling) combinations run in the main experiment
COMBOS = [("clean", True), ("realistic", True), ("realistic", False), ("stressed", True), ("stressed", False)]
PRIMARY = ("realistic", True)


def combo_label(scenario: str, handling: bool) -> str:
    return f"{scenario}|{'handled' if handling else 'raw'}"


class Context:
    """Panel + feature store + per-horizon frames for one (scenario, handling)."""
    def __init__(self, cfg: dict, scenario: str, handling: bool, panel_df: pd.DataFrame | None = None):
        self.cfg, self.scenario, self.handling = cfg, scenario, handling
        if panel_df is None:
            panel_df = pd.read_csv(data_dir() / f"synthetic_panel_{scenario}.csv")
        self.panel_df = panel_df
        self.panel = panel_from_frame(panel_df, cfg)
        self.store = FeatureStore(self.panel, cfg, handling)
        self.frames = {h: self.store.frame(h) for h in cfg["experiment"]["horizons"]}


def _tag(df: pd.DataFrame, ctx: Context, partition: str) -> pd.DataFrame:
    df["scenario"], df["handling"], df["partition"] = ctx.scenario, ("handled" if ctx.handling else "raw"), partition
    return df


# ----------------------------------------------------------------------------- tuning on rolling-origin folds
def tune_model(ctx: Context, model: str, fset: str, h: int, grid: dict, n_estimators_tune: int | None = None):
    """Pooled out-of-sample MAE over the rolling-origin folds for every grid point."""
    cfg = ctx.cfg
    folds = rolling_folds(cfg)
    rows = []
    for params in ParameterGrid(grid):
        p = dict(params)
        if model == "random_forest" and n_estimators_tune:
            p["n_estimators"] = n_estimators_tune
        errs, per_fold = [], []
        for f in folds:
            out = fit_predict_ml(model, p, fset, ctx.frames[h], h, f["train_end"], f["val_start"], f["val_end"],
                                 cfg, ctx.handling)
            o = out[out["scored"]]
            e = np.abs(o["y_obs"] - o["y_pred"]).values
            errs.append(e); per_fold.append(float(e.mean()))
        pooled = float(np.concatenate(errs).mean())
        rows.append(dict(model=model, feature_set=fset, horizon=h, params=json.dumps(dict(params), sort_keys=True),
                         cv_mae=pooled, cv_mae_std_over_folds=float(np.std(per_fold)),
                         fold_maes=json.dumps([round(x, 4) for x in per_fold])))
    return pd.DataFrame(rows)


def grid_for(cfg: dict, model: str) -> dict:
    g = cfg["models"]["grids"]
    if model == "ridge_cropint":
        return g["ridge"]
    if model in g:
        return g[model]
    if model == "xgboost":
        return {"max_depth": cfg["advanced"]["xgboost"]["max_depth"]}
    if model == "lightgbm":
        return {"num_leaves": cfg["advanced"]["lightgbm"]["num_leaves"]}
    raise KeyError(model)


def tune_all(cfg: dict, models: list[str], fsets=("E0", "E4"), log=print) -> tuple[pd.DataFrame, dict]:
    ctx = Context(cfg, *PRIMARY)
    allrows, best = [], {}
    for model in models:
        for fset in fsets:
            for h in cfg["experiment"]["horizons"]:
                t0 = time.time()
                tab = tune_model(ctx, model, fset, h, grid_for(cfg, model),
                                 n_estimators_tune=100 if model == "random_forest" else None)
                allrows.append(tab)
                b = tab.sort_values("cv_mae").iloc[0]
                best.setdefault(model, {}).setdefault(fset, {})[str(h)] = json.loads(b["params"])
                log(f"tuned {model:14s} {fset} h={h}  best={b['params']}  cv_mae={b['cv_mae']:.3f}  ({time.time()-t0:.0f}s)")
    return pd.concat(allrows, ignore_index=True), best


def params_for(best: dict, model: str, fset: str, h: int) -> dict:
    """E0 uses its own tuned parameters; every other feature set uses the E4-tuned parameters."""
    key = "E0" if fset == "E0" else "E4"
    return best.get(model, {}).get(key, {}).get(str(h), {})


# ----------------------------------------------------------------------------- main runs
def run_partition(ctx: Context, best: dict, partition: str, ml_specs: list[tuple[str, str]],
                  with_sarima: bool = True, log=print) -> pd.DataFrame:
    cfg = ctx.cfg
    lo, hi = partitions(cfg)[partition]
    train_end = partitions(cfg)["train"][1] if partition == "validation" else partitions(cfg)["validation"][1]
    outs = []
    for h in cfg["experiment"]["horizons"]:
        df = ctx.frames[h]
        hi_h = hi - h                                   # target must exist inside the dataset
        hi_h = min(hi, cfg["experiment"]["n_weeks"] - h)
        outs.append(baseline_predictions(df, h, lo, hi_h, cfg))
        for model, fset in ml_specs:
            p = params_for(best, model, fset, h)
            outs.append(fit_predict_ml(model, p, fset, df, h, train_end, lo, hi_h, cfg, ctx.handling))
    if with_sarima:
        hi_s = min(hi, cfg["experiment"]["n_weeks"] - 1)
        for seasonal in (True, False):
            t0 = time.time()
            sp = sarima_predictions(ctx.store, ctx.frames, cfg, train_end, lo, hi_s, seasonal=seasonal)
            sp["fit_s"] = (time.time() - t0) / 3
            outs.append(sp)
    res = pd.concat(outs, ignore_index=True)
    # same rows for every candidate: restrict to the origin range valid for the longest horizon is NOT applied;
    # per-horizon comparisons always share rows (all models are evaluated on the same evaluation_frame).
    return _tag(res, ctx, partition)


# ----------------------------------------------------------------------------- rolling-origin CV predictions (ablation)
def run_cv_ablation(ctx: Context, best: dict, models=("ridge", "random_forest", "hist_gb"),
                    fsets=("E0", "E1", "E2", "E3", "E4"), log=print) -> pd.DataFrame:
    cfg = ctx.cfg
    outs = []
    for model in models:
        for fset in fsets:
            for h in cfg["experiment"]["horizons"]:
                p = params_for(best, model, fset, h)
                if model == "random_forest":
                    p = {**p, "n_estimators": 100}
                for f in rolling_folds(cfg):
                    o = fit_predict_ml(model, p, fset, ctx.frames[h], h, f["train_end"], f["val_start"], f["val_end"],
                                       cfg, ctx.handling)
                    o["fold"] = f["fold"]
                    outs.append(o)
            log(f"cv-ablation {model} {fset} done")
    for h in cfg["experiment"]["horizons"]:                 # baselines across the same folds
        for f in rolling_folds(cfg):
            b = baseline_predictions(ctx.frames[h], h, f["val_start"], f["val_end"], cfg)
            b["fold"] = f["fold"]; outs.append(b)
    return _tag(pd.concat(outs, ignore_index=True), ctx, "cv")


# ----------------------------------------------------------------------------- generalisation
def run_generalisation(ctx: Context, best: dict, models=("ridge", "random_forest", "hist_gb"), log=print) -> pd.DataFrame:
    """Test-window diagnostics (history of the held-out series MAY be used as lag features;
    its TARGET rows are never used for training)."""
    cfg = ctx.cfg
    lo, hi = partitions(cfg)["test"]
    train_end = partitions(cfg)["validation"][1]
    pairs = cfg["generalisation"]["held_out_pairs"]
    mk = list(cfg["markets"].keys())
    outs = []
    for h in cfg["experiment"]["horizons"]:
        df = ctx.frames[h]
        hi_h = min(hi, cfg["experiment"]["n_weeks"] - h)
        outs.append(_gen_tag(baseline_predictions(df, h, lo, hi_h, cfg).query("model=='naive'"),
                             "reference", "all", "naive"))
        for model in models:
            p = params_for(best, model, "E4", h)
            # (a) unseen-market: model has NO market_id so it cannot memorise the market
            ref = fit_predict_ml(model, p, "E4", df, h, train_end, lo, hi_h, cfg, ctx.handling, include_market_id=False)
            outs.append(_gen_tag(ref, "seen_market_no_id", "all", model))
            for m in mk:
                held = [s for s in df["series"].unique() if s.startswith(m)]
                o = fit_predict_ml(model, p, "E4", df, h, train_end, lo, hi_h, cfg, ctx.handling,
                                   exclude_series=held, include_market_id=False, eval_series=held)
                outs.append(_gen_tag(o, "unseen_market", m, model))
            # (b) unseen crop-market pairs (market_id and crop_id still available as inputs)
            ref2 = fit_predict_ml(model, p, "E4", df, h, train_end, lo, hi_h, cfg, ctx.handling, eval_series=pairs)
            outs.append(_gen_tag(ref2, "seen_pair", "|".join(pairs), model))
            o2 = fit_predict_ml(model, p, "E4", df, h, train_end, lo, hi_h, cfg, ctx.handling,
                                exclude_series=pairs, eval_series=pairs)
            outs.append(_gen_tag(o2, "unseen_pair", "|".join(pairs), model))
        log(f"generalisation h={h} done")
    return _tag(pd.concat(outs, ignore_index=True), ctx, "test")


def _gen_tag(df, kind, held, model):
    df = df.copy(); df["gen_kind"], df["held_out"], df["model"] = kind, held, model
    return df


# ----------------------------------------------------------------------------- prediction intervals
def run_intervals(ctx: Context, best: dict, coverage: float = 0.8, log=print) -> pd.DataFrame:
    """HGB quantile regression (q10/q90) and split-conformal intervals around HGB point forecasts."""
    cfg = ctx.cfg
    P = partitions(cfg)
    q_lo, q_hi = (1 - coverage) / 2, 1 - (1 - coverage) / 2
    outs = []
    for h in cfg["experiment"]["horizons"]:
        df = ctx.frames[h]
        p = params_for(best, "hist_gb", "E4", h)
        res = {}
        for part, train_end in (("validation", P["train"][1]), ("test", P["validation"][1])):
            lo, hi = P[part]
            hi_h = min(hi, cfg["experiment"]["n_weeks"] - h)
            pt = fit_predict_ml("hist_gb", p, "E4", df, h, train_end, lo, hi_h, cfg, ctx.handling)
            ql = fit_predict_ml("hist_gb", {**p, "quantile": q_lo}, "E4", df, h, train_end, lo, hi_h, cfg, ctx.handling)
            qh = fit_predict_ml("hist_gb", {**p, "quantile": q_hi}, "E4", df, h, train_end, lo, hi_h, cfg, ctx.handling)
            res[part] = (pt, ql, qh)
        # conformal quantile from validation absolute log errors, per crop
        v = res["validation"][0]; v = v[v["scored"]]
        qhat = (np.abs(np.log(v["y_obs"] / v["y_pred"])).groupby(v["crop_id"]).quantile(coverage)).to_dict()
        for part in ("validation", "test"):
            pt, ql, qh = res[part]
            base = pt.copy()
            base["lower_qr"] = np.minimum(ql["y_pred"].values, qh["y_pred"].values)
            base["upper_qr"] = np.maximum(ql["y_pred"].values, qh["y_pred"].values)
            base["lower_qr"] = np.minimum(base["lower_qr"], base["y_pred"])
            base["upper_qr"] = np.maximum(base["upper_qr"], base["y_pred"])
            qc = base["crop_id"].map(qhat).values
            base["lower_cf"] = base["y_pred"] * np.exp(-qc)
            base["upper_cf"] = base["y_pred"] * np.exp(qc)
            base["nominal_coverage"] = coverage
            base["partition"] = part
            outs.append(base)
        log(f"intervals h={h} done")
    out = pd.concat(outs, ignore_index=True)
    out["scenario"], out["handling"] = ctx.scenario, "handled" if ctx.handling else "raw"
    return out
