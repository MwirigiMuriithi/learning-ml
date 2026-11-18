"""Causal (look-back-only) data-quality handling.

Every function here processes a series strictly in time order and the value at
week t depends only on weeks <= t.  This is what makes cleaning safe to apply
before building forecast-origin features (no hindsight corrections).
"""
from __future__ import annotations
import numpy as np


def causal_outlier_filter(x: np.ndarray, thr: float = 0.70, ref_window: int = 5,
                          agree_tol: float = 0.15):
    """Reject implausible reported prices using only past information.

    A value is rejected if it is <= 0, or if it deviates from the median of the
    last `ref_window` accepted prices by more than `thr` in log terms AND it does
    not agree (within `agree_tol`) with the previous week's report.  The second
    condition keeps genuine persistent level shifts (real shocks) from being
    erased: after one week the new level is accepted.

    Returns (accepted, rejected_mask).  accepted has NaN for missing/rejected.
    """
    T = len(x)
    acc = np.full(T, np.nan)
    rej = np.zeros(T, dtype=bool)
    hist: list[float] = []
    prev_raw = np.nan
    for t in range(T):
        v = x[t]
        if np.isnan(v):
            prev_raw = np.nan
            continue
        ok = True
        if v <= 0:
            ok = False
        elif len(hist) >= 3:
            ref = np.median(hist[-ref_window:])
            if abs(np.log(v / ref)) > thr:
                agrees = (not np.isnan(prev_raw)) and prev_raw > 0 and abs(np.log(v / prev_raw)) < agree_tol
                ok = agrees
        if ok:
            acc[t] = v
            hist.append(v)
        else:
            rej[t] = True
        prev_raw = v
    return acc, rej


def locf(x: np.ndarray, max_age: int | None = None):
    """Last observation carried forward along axis 0.  Returns (filled, age).

    age = weeks since the last real observation (0 when observed now).
    Values older than max_age become NaN again.  Uses past values only.
    """
    T = x.shape[0]
    filled = np.full(x.shape, np.nan)
    age = np.full(x.shape, np.nan)
    last = np.full(x.shape[1:], np.nan)
    last_t = np.full(x.shape[1:], np.nan)
    for t in range(T):
        cur = x[t]
        have = ~np.isnan(cur)
        last = np.where(have, cur, last)
        last_t = np.where(have, t, last_t)
        a = t - last_t
        f = last.copy()
        if max_age is not None:
            f = np.where(a > max_age, np.nan, f)
        filled[t] = f
        age[t] = a
    return filled, age
