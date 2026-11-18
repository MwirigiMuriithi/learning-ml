"""
project_utils.py -- the evaluation harness for the capstone project.

THE KEY DESIGN IDEA: a leak-proof harness.
    A model is a plain function   model(train_df, h, future) -> float
      train_df : every column, ONLY for weeks up to the forecast origin (the model never sees later rows)
      h        : the forecast horizon in weeks
      future   : the next h weeks' calendar-known columns (festival_flag) -- deterministic, so allowed
    It must return the price forecast for week origin+h.

Because the harness slices the data before calling your model, a model *cannot* peek at the future, however
buggy its feature code.  (It can still leak through its own scaling/imputing choices, so keep those inside the model.)
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from course_utils import mae, rmse, fourier_terms

HORIZONS = (1, 4, 8, 12)


def extend(train_df, future):
    """train rows + the next h (mostly empty) rows, so features for the TARGET week can be computed.
       Only calendar-known columns are filled for the future rows; everything else stays NaN."""
    ext = train_df.reindex(train_df.index.append(future.index))
    ext.loc[future.index, "festival_flag"] = future["festival_flag"].values
    return ext


def run_walk_forward(d, models, origins, horizons=HORIZONS, verbose=True):
    """
    Rolling-origin evaluation through the leak-proof harness.
    d       : clean weekly DataFrame (index W-SUN, column 'price' and covariates)
    models  : {name: model_fn}
    origins : iterable of integer positions o = number of weeks the model may see (train = d.iloc[:o])
    Returns a tidy DataFrame: model, origin, target_date, h, actual, forecast
    """
    rows = []
    for o in origins:
        train = d.iloc[:o]
        for h in horizons:
            if o + h > len(d):
                continue
            fut = d[["festival_flag"]].iloc[o:o + h]
            tgt, actual = d.index[o + h - 1], d["price"].iloc[o + h - 1]
            for name, fn in models.items():
                rows.append((name, train.index[-1], tgt, h, actual, float(fn(train, h, fut))))
    out = pd.DataFrame(rows, columns=["model", "origin", "target_date", "h", "actual", "forecast"])
    if verbose:
        print(f"{out.origin.nunique()} origins x {len(horizons)} horizons x {out.model.nunique()} models = {len(out)} forecasts")
    return out


def add_blend(wf, members, name, weights=None):
    """Average several models' forecasts (same origins/horizons) into a new 'model'."""
    w = weights or [1 / len(members)] * len(members)
    piv = wf[wf.model.isin(members)].pivot_table(index=["origin", "target_date", "h", "actual"], columns="model", values="forecast")
    blended = sum(wi * piv[m] for wi, m in zip(w, members)).rename("forecast").reset_index()
    blended["model"] = name
    return pd.concat([wf, blended[wf.columns]], ignore_index=True)


def scoreboard(wf, baseline="naive", metric=mae):
    """Metric by model x horizon, plus skill vs the baseline (1 - metric/metric_baseline)."""
    t = wf.groupby(["model", "h"]).apply(lambda g: metric(g.actual, g.forecast), include_groups=False).unstack("h")
    t.columns = [f"h={c}" for c in t.columns]
    t["all"] = wf.groupby("model").apply(lambda g: metric(g.actual, g.forecast), include_groups=False)
    if baseline in t.index:
        skill = 1 - t.div(t.loc[baseline], axis=1)
        skill.columns = [f"skill {c}" for c in skill.columns]
        t = t.join(skill)
    return t.sort_values("all").round(3)


def empirical_interval(dev_wf, model, alpha=0.2):
    """Per-horizon (lo, hi) quantiles of the log forecast ratio log(actual/forecast), learned on the DEV period only."""
    g = dev_wf[dev_wf.model == model].copy()
    g["lr"] = np.log(g.actual / g.forecast)
    return g.groupby("h")["lr"].quantile([alpha / 2, 1 - alpha / 2]).unstack()


def coverage(test_wf, model, q):
    """Share of test actuals that fall inside forecast * exp(quantile band)."""
    g = test_wf[test_wf.model == model].copy()
    lo, hi = g.h.map(q.iloc[:, 0]), g.h.map(q.iloc[:, 1])
    inside = (g.actual >= g.forecast * np.exp(lo)) & (g.actual <= g.forecast * np.exp(hi))
    return float(inside.mean())
