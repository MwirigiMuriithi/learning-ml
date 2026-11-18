"""Chronological partitions and rolling-origin folds.

A row is (series, forecast-origin week t, horizon h, target week t+h).  The
PARTITION of a row is decided by its origin week.  Training rows are purged so
that their TARGET week is <= the last training week (an h-week embargo), hence
a model fitted for partition P has seen no information dated after the end of
the training window.
"""
from __future__ import annotations
import pandas as pd


def partitions(cfg: dict) -> dict:
    s = cfg["experiment"]["splits"]
    return {
        "train": (1, s["train_end"]),
        "validation": (s["train_end"] + 1, s["val_end"]),
        "test": (s["val_end"] + 1, s["test_end"]),
    }


def rolling_folds(cfg: dict) -> list[dict]:
    r, s = cfg["experiment"]["rolling_origin"], cfg["experiment"]["splits"]
    folds, te = [], r["first_train_end"]
    while te + r["block"] <= s["val_end"]:
        folds.append(dict(fold=len(folds) + 1, train_end=te, val_start=te + 1, val_end=te + r["block"]))
        te += r["step"]
    return folds


def train_rows(df: pd.DataFrame, h: int, train_end: int, min_origin: int) -> pd.Series:
    """Origins whose TARGET week is <= train_end."""
    return (df["origin_week"] >= min_origin) & (df["origin_week"] <= train_end - h)


def eval_rows(df: pd.DataFrame, lo: int, hi: int) -> pd.Series:
    return (df["origin_week"] >= lo) & (df["origin_week"] <= hi)
