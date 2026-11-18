"""Forecast accuracy metrics (KES/kg unless stated)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def mae(y, p): return float(np.mean(np.abs(y - p)))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def wape(y, p): return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)))
def smape(y, p): return float(np.mean(2 * np.abs(y - p) / (np.abs(y) + np.abs(p))))


def directional_accuracy(y, p, ref):
    """Share of cases where sign(p-ref) == sign(y-ref); undefined (NaN) when the
    model predicts no change at all (e.g. the naive forecast)."""
    dy, dp = np.sign(y - ref), np.sign(p - ref)
    m = (dy != 0) & (dp != 0)
    return float((dy[m] == dp[m]).mean()) if m.sum() else np.nan


def score(g: pd.DataFrame) -> pd.Series:
    y, p, r = g["y_obs"].values, g["y_pred"].values, g["ref_price"].values
    return pd.Series(dict(n=len(g), MAE=mae(y, p), RMSE=rmse(y, p), WAPE=wape(y, p),
                          sMAPE=smape(y, p), DirAcc=directional_accuracy(y, p, r)))


def metric_table(preds: pd.DataFrame, by: list[str], scored_only: bool = True) -> pd.DataFrame:
    d = preds[preds["scored"]] if scored_only else preds
    rows = []
    for key, g in d.groupby(by, sort=False):
        s = score(g)
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **s.to_dict()})
    return pd.DataFrame(rows)
