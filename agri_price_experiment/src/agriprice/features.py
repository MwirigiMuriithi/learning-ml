"""Forecast-origin feature engineering with an explicit information-availability contract.

Forecast-time data contract
---------------------------
At the end of week t (the forecast origin) the system may use
  * price of week w   (w <= t)          : reported on time;
  * indicator of week w                 : usable only if  w + report_delay[w] <= t
                                          (late records appear one week after period end);
  * calendar facts about the target week t+h (date arithmetic + holiday calendar only);
  * nothing derived from weeks > t, and never `latent_price_kes_kg`, event labels or quality-audit labels.

Every feature is a pure function of the data "as known at t".  `tests/test_no_leakage.py`
proves this by truncating the panel at week T0 and checking that features of all origins <= T0
are bit-identical to the ones computed from the full panel.

Data-quality handling (`handling=True`)
  price      : causal outlier filter -> last-observation-carried-forward (cap) + staleness feature
  indicators : last-observation-carried-forward (cap); lag-0 falls back to lag-1 when the record is late
`handling=False` leaves NaNs and outliers untouched (used to measure the cost of ignoring data quality).
"""
from __future__ import annotations
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd

from .config import market_ids, crop_ids, INDICATOR_FIELDS, DELAYABLE_FIELDS
from .cleaning import causal_outlier_filter, locf
from .generator import holiday_week_flags

# ----------------------------------------------------------------------------- feature groups
FEATURE_GROUPS: dict[str, list[str]] = {
    "calendar": ["tgt_week_of_year", "tgt_month", "tgt_quarter", "tgt_sin1", "tgt_cos1", "tgt_sin2",
                 "tgt_cos2", "tgt_holiday", "org_week_of_year", "org_holiday"],
    "price": ["lp0", "lr_1", "lr_2", "lr_3", "lr_4", "lr_8", "lr_13", "lr_52", "dev_mean_4", "dev_mean_8",
              "dev_mean_13", "dev_mean_26", "vol_4", "vol_8", "seasonal_ref_ratio"],
    "supply_demand": ["arr_l0", "arr_l1", "arr_mean4", "arr_anom13", "dem_l0", "dem_l1", "dem_mean4",
                      "event_flag_l0"],
    "weather": ["rain_l0", "rain_mean4", "rain_mean8", "rain_sum13", "temp_l0", "temp_mean4",
                "drought_l0", "drought_l2", "drought_mean4", "drought_max8"],
    "logistics": ["fuel_l0", "fuel_chg4", "trans_l0", "trans_mean4", "trans_chg4"],
    "cross_market": ["conn_gap0", "conn_chg1", "conn_chg4", "regsup_l0", "regsup_mean4", "regsup_chg1"],
}
HANDLING_ONLY_FEATURES = ["price_stale_weeks"]          # appended to the price group when handling=True
CATEGORICAL = ["market_id", "crop_id"]

FEATURE_SETS: dict[str, list[str]] = {
    "E0": ["calendar", "price"],
    "E1": ["calendar", "price", "supply_demand"],
    "E2": ["calendar", "price", "supply_demand", "weather"],
    "E3": ["calendar", "price", "supply_demand", "weather", "logistics"],
    "E4": ["calendar", "price", "supply_demand", "weather", "logistics", "cross_market"],
}
FEATURE_SET_LABEL = {
    "E0": "E0 price-only", "E1": "E1 +supply/demand", "E2": "E2 +weather",
    "E3": "E3 +logistics", "E4": "E4 full",
}


def feature_columns(fset: str, frame_cols=None, include_ids=True) -> tuple[list[str], list[str]]:
    """(numeric_cols, categorical_cols) for a feature set."""
    num: list[str] = []
    for g in FEATURE_SETS[fset]:
        num += FEATURE_GROUPS[g]
        if g == "price":
            num += HANDLING_ONLY_FEATURES
    if frame_cols is not None:
        num = [c for c in num if c in frame_cols]
    return num, (CATEGORICAL if include_ids else [])


# ----------------------------------------------------------------------------- panel
@dataclass
class Panel:
    """Dense (T, M, C) arrays of the OBSERVED panel plus audit-only arrays."""
    arrays: dict            # observed fields
    delay: np.ndarray       # report_delay_weeks (T,M,C)
    audit: dict             # latent / audit labels (never used as features)
    markets: list
    crops: list
    start_date: pd.Timestamp
    T: int

    def truncate(self, T0: int) -> "Panel":
        return Panel({k: v[:T0].copy() for k, v in self.arrays.items()}, self.delay[:T0].copy(),
                     {k: v[:T0].copy() for k, v in self.audit.items()}, self.markets, self.crops,
                     self.start_date, T0)


def panel_from_frame(df: pd.DataFrame, cfg: dict) -> Panel:
    mk, cr = market_ids(cfg), crop_ids(cfg)
    T = int(df["week_index"].max())
    M, C = len(mk), len(cr)
    ti = df["week_index"].values - 1
    mi = df["market_id"].map({m: i for i, m in enumerate(mk)}).values
    ci = df["crop_id"].map({c: i for i, c in enumerate(cr)}).values

    def arr(col, fill=np.nan, dtype=float):
        a = np.full((T, M, C), fill, dtype=dtype)
        a[ti, mi, ci] = df[col].values
        return a

    fields = ["wholesale_price_kes_kg"] + INDICATOR_FIELDS + ["holiday_or_event_flag"]
    arrays = {f: arr(f) for f in fields}
    delay = arr("report_delay_weeks", 0, int)
    ev = np.empty((T, M, C), dtype=object); ev[:] = ""
    ev[ti, mi, ci] = df["event_ids"].fillna("").values
    audit = {
        "latent": arr("latent_price_kes_kg"),
        "suspect": np.zeros((T, M, C), bool),
        "event_affected": arr("event_affected", False, bool),
        "regime": arr("regime_shift_flag", False, bool),
        "event_ids": ev,
    }
    audit["suspect"][ti, mi, ci] = (df["data_quality_flag"].values == "suspect")
    return Panel(arrays, delay, audit, mk, cr, pd.Timestamp(cfg["experiment"]["start_date"]), T)


# ----------------------------------------------------------------------------- as-known helpers
def asof_lag(x: np.ndarray, delay: np.ndarray, k: int) -> np.ndarray:
    """Value of week t-k as known at origin t (NaN if absent or not yet delivered)."""
    T = x.shape[0]
    out = np.full(x.shape, np.nan)
    if k < T:
        out[k:] = x[: T - k]
        d = np.zeros(x.shape, dtype=int)
        d[k:] = delay[: T - k]
        out = np.where(d > k, np.nan, out)
    return out


def _nanfn(stack, fn, minc):
    cnt = (~np.isnan(stack)).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = fn(stack, axis=0)
    r = np.asarray(r, dtype=float)
    r[cnt < minc] = np.nan
    return r


def wstat(x, delay, W, fn=np.nanmean, minc=None, start=0):
    st = np.stack([asof_lag(x, delay, k) for k in range(start, start + W)], axis=0)
    return _nanfn(st, fn, minc if minc is not None else max(1, W // 2))


def _safe_log(a):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(a)
    out[~np.isfinite(out)] = np.nan
    return out


# ----------------------------------------------------------------------------- core builder
class FeatureStore:
    """Computes origin-level features once; frames per horizon are cheap slices."""

    def __init__(self, panel: Panel, cfg: dict, handling: bool = True):
        self.panel, self.cfg, self.handling = panel, cfg, handling
        self.dq = cfg["data_quality_handling"]
        self._build()

    # ---- cleaning ---------------------------------------------------------
    def _clean_price(self):
        p = self.panel.arrays["wholesale_price_kes_kg"]
        T, M, C = p.shape
        acc = np.full_like(p, np.nan); rej = np.zeros(p.shape, bool)
        for m in range(M):
            for c in range(C):
                acc[:, m, c], rej[:, m, c] = causal_outlier_filter(
                    p[:, m, c], self.dq["outlier_log_threshold"], self.dq["outlier_ref_window"])
        return acc, rej

    def _build(self):
        P, dq = self.panel, self.dq
        T, M, C = P.T, len(P.markets), len(P.crops)
        zero = np.zeros((T, M, C), dtype=int)
        raw = P.arrays["wholesale_price_kes_kg"]
        acc, rej = self._clean_price()
        self.detected_suspect = rej
        self.accepted = acc
        # price series used for features / reference
        pe_h, age_h = locf(acc, dq["price_locf_max_age"])            # handled series (cap)
        self.ref_ok_handled = ~np.isnan(pe_h)
        if self.handling:
            pe, age = pe_h, age_h
        else:
            pe, _ = locf(raw, None)                                  # latest raw value, outliers included
            age = None
        self.ref = pe                                                # latest available price at origin t
        # the series entering lag features:
        self.price_for_lags = pe if self.handling else raw
        self.price_known_for_baselines = pe if self.handling else raw
        lp = _safe_log(self.price_for_lags)
        self.lp = lp
        self.lp_ref = _safe_log(self.ref)
        self.age = age

        # indicators
        ind = {}
        for f in INDICATOR_FIELDS + ["holiday_or_event_flag"]:
            x = P.arrays[f]
            if self.handling and f != "holiday_or_event_flag":
                x, _ = locf(x, dq["indicator_locf_max_age"])
            ind[f] = x
        self.ind = ind
        d_ind = {f: (P.delay if f in DELAYABLE_FIELDS else zero) for f in ind}
        self.d_ind = d_ind

        def lag(f, k):
            v = asof_lag(ind[f], d_ind[f], k)
            if self.handling and k == 0:                              # late record -> use last delivered
                v = np.where(np.isnan(v), asof_lag(ind[f], d_ind[f], 1), v)
            return v

        L = lambda x, k: asof_lag(x, zero, k)                         # price-type lag (no delay)
        F: dict[str, np.ndarray] = {}

        # ---- price features ----
        lp0 = L(lp, 0)
        F["lp0"] = lp0
        for k in (1, 2, 3, 4, 8, 13, 52):
            F[f"lr_{k}"] = lp0 - L(lp, k)
        for W in (4, 8, 13, 26):
            F[f"dev_mean_{W}"] = lp0 - wstat(lp, zero, W, minc=max(2, W // 2))
        ret = np.stack([L(lp, k) - L(lp, k + 1) for k in range(0, 8)], axis=0)    # weekly returns (lag0 = t vs t-1)
        F["vol_4"] = _nanfn(ret[:4], np.nanstd, 3)
        F["vol_8"] = _nanfn(ret[:8], np.nanstd, 5)
        if self.handling:
            F["price_stale_weeks"] = age
        # ---- supply / demand ----
        la = _safe_log(ind["market_arrivals_tonnes"])
        F["arr_l0"] = _safe_log(lag("market_arrivals_tonnes", 0))
        F["arr_l1"] = _safe_log(lag("market_arrivals_tonnes", 1))
        F["arr_mean4"] = _safe_log(wstat(ind["market_arrivals_tonnes"], d_ind["market_arrivals_tonnes"], 4, minc=2))
        F["arr_anom13"] = F["arr_l0"] - _safe_log(
            wstat(ind["market_arrivals_tonnes"], d_ind["market_arrivals_tonnes"], 13, minc=6))
        F["dem_l0"] = _safe_log(lag("estimated_demand_index", 0))
        F["dem_l1"] = _safe_log(lag("estimated_demand_index", 1))
        F["dem_mean4"] = _safe_log(wstat(ind["estimated_demand_index"], d_ind["estimated_demand_index"], 4, minc=2))
        F["event_flag_l0"] = asof_lag(ind["holiday_or_event_flag"], zero, 0)
        # ---- weather ----
        r, tmp, dr = ind["rainfall_mm"], ind["temperature_c"], ind["drought_index"]
        F["rain_l0"] = lag("rainfall_mm", 0)
        F["rain_mean4"] = wstat(r, zero, 4, minc=2)
        F["rain_mean8"] = wstat(r, zero, 8, minc=4)
        F["rain_sum13"] = wstat(r, zero, 13, fn=np.nansum, minc=7)
        F["temp_l0"] = lag("temperature_c", 0)
        F["temp_mean4"] = wstat(tmp, zero, 4, minc=2)
        F["drought_l0"] = lag("drought_index", 0)
        F["drought_l2"] = lag("drought_index", 2)
        F["drought_mean4"] = wstat(dr, zero, 4, minc=2)
        F["drought_max8"] = wstat(dr, zero, 8, fn=np.nanmax, minc=4)
        # ---- logistics ----
        fu, tr = ind["fuel_cost_index"], ind["transport_cost_kes_tonne"]
        F["fuel_l0"] = _safe_log(lag("fuel_cost_index", 0))
        F["fuel_chg4"] = F["fuel_l0"] - _safe_log(lag("fuel_cost_index", 4))
        F["trans_l0"] = _safe_log(lag("transport_cost_kes_tonne", 0))
        F["trans_mean4"] = _safe_log(wstat(tr, d_ind["transport_cost_kes_tonne"], 4, minc=2))
        F["trans_chg4"] = F["trans_l0"] - _safe_log(lag("transport_cost_kes_tonne", 4))
        # ---- cross-market ----
        Wm = np.zeros((M, M))
        for m, srcs in self.cfg["network"].items():
            for j, w in srcs.items():
                Wm[P.markets.index(m), P.markets.index(j)] = w
        def nbr(x):   # weighted neighbour average of x over source markets (same crop)
            return np.einsum("mj,tjc->tmc", Wm, x)
        lp_l0, lp_l1, lp_l4 = L(lp, 0), L(lp, 1), L(lp, 4)
        F["conn_gap0"] = nbr(lp_l0) - lp_l0
        F["conn_chg1"] = nbr(lp_l0 - lp_l1)
        F["conn_chg4"] = nbr(lp_l0 - lp_l4)
        F["regsup_l0"] = _safe_log(lag("regional_supply_index", 0))
        F["regsup_mean4"] = _safe_log(wstat(ind["regional_supply_index"], d_ind["regional_supply_index"], 4, minc=2))
        F["regsup_chg1"] = F["regsup_l0"] - _safe_log(lag("regional_supply_index", 1))
        self.F = F
        self.lag_price_for_seasonal = lambda k: L(lp, k)
        self.lag_price_level = lambda k: L(self.price_known_for_baselines, k)
        # moving-average baselines (arithmetic mean of recently known prices)
        self.ma = {W: wstat(self.price_known_for_baselines, zero, W, minc=max(1, W // 2))
                   for W in self.cfg["models"]["moving_average_windows"]}

    # ---- calendar -------------------------------------------------------------
    def _calendar(self, h: int, T: int):
        start = self.panel.start_date
        wk = np.arange(1, T + 1)
        odate = start + pd.to_timedelta((wk - 1) * 7, unit="D")
        tdate = odate + pd.to_timedelta(7 * h, unit="D")
        # calendar-known holidays over the whole horizon of dates
        all_w = np.arange(1, T + h + 1)
        all_d = start + pd.to_timedelta((all_w - 1) * 7, unit="D")
        hol = holiday_week_flags(all_d, self.cfg)
        tdoy = (tdate + pd.Timedelta(days=3)).dayofyear.values
        return dict(
            odate=odate, tdate=tdate,
            tgt_week_of_year=tdate.isocalendar().week.values.astype(float),
            tgt_month=tdate.month.values.astype(float), tgt_quarter=tdate.quarter.values.astype(float),
            tgt_sin1=np.sin(2 * np.pi * tdoy / 365.25), tgt_cos1=np.cos(2 * np.pi * tdoy / 365.25),
            tgt_sin2=np.sin(4 * np.pi * tdoy / 365.25), tgt_cos2=np.cos(4 * np.pi * tdoy / 365.25),
            tgt_holiday=hol[wk - 1 + h].astype(float), org_holiday=hol[wk - 1].astype(float),
            org_week_of_year=odate.isocalendar().week.values.astype(float))

    # ---- frame per horizon --------------------------------------------------------
    def frame(self, h: int, with_targets: bool = True) -> pd.DataFrame:
        P = self.panel
        T, M, C = P.T, len(P.markets), len(P.crops)
        cal = self._calendar(h, T)
        t_i, m_i, c_i = [a.ravel() for a in np.meshgrid(np.arange(T), np.arange(M), np.arange(C), indexing="ij")]
        data = {
            "market_id": np.array(P.markets)[m_i], "crop_id": np.array(P.crops)[c_i],
            "origin_week": t_i + 1, "target_week": t_i + 1 + h,
            "origin_date": cal["odate"][t_i].strftime("%Y-%m-%d"),
            "target_date": cal["tdate"][t_i].strftime("%Y-%m-%d"),
        }
        for name in FEATURE_GROUPS["calendar"]:
            data[name] = cal[name][t_i]
        for g, cols in FEATURE_GROUPS.items():
            if g == "calendar":
                continue
            for name in cols:
                if name == "seasonal_ref_ratio":
                    continue
                data[name] = self.F[name][t_i, m_i, c_i]
        if self.handling:
            data["price_stale_weeks"] = self.F["price_stale_weeks"][t_i, m_i, c_i]
        # horizon-specific seasonal reference: price one year before the TARGET week (known at origin)
        k = 52 - h
        seas_lp = self.lag_price_for_seasonal(k)
        data["seasonal_ref_ratio"] = (seas_lp - self.F["lp0"])[t_i, m_i, c_i]
        data["ref_price"] = self.ref[t_i, m_i, c_i]
        data["ref_ok_handled"] = self.ref_ok_handled[t_i, m_i, c_i]
        # baselines (as-known information only)
        data["base_seasonal_naive"] = self.lag_price_level(k)[t_i, m_i, c_i]
        for W, arr in self.ma.items():
            data[f"base_ma{W}"] = arr[t_i, m_i, c_i]
        df = pd.DataFrame(data)
        df["series"] = df["market_id"] + "|" + df["crop_id"]
        df["horizon"] = h
        df["horizon_vol8"] = df["vol_8"]
        if with_targets:
            tgt = np.full((T, M, C), np.nan)
            raw = P.arrays["wholesale_price_kes_kg"]
            tgt[: T - h] = raw[h:]
            def shifted(a, fill):
                out = np.full((T, M, C), fill, dtype=a.dtype)
                out[: T - h] = a[h:]
                return out
            df["y_obs"] = tgt[t_i, m_i, c_i]
            df["y_latent"] = shifted(P.audit["latent"], np.nan)[t_i, m_i, c_i]
            df["y_audit_suspect"] = shifted(P.audit["suspect"], False)[t_i, m_i, c_i]
            df["y_event_affected"] = shifted(P.audit["event_affected"], False)[t_i, m_i, c_i]
            df["y_regime"] = shifted(P.audit["regime"], False)[t_i, m_i, c_i]
            ids = np.empty((T, M, C), dtype=object); ids[:] = ""; ids[: T - h] = P.audit["event_ids"][h:]
            df["y_event_ids"] = ids[t_i, m_i, c_i]
            df["y_detected_suspect"] = shifted(self.detected_suspect, False)[t_i, m_i, c_i]
            has_target = df["target_week"] <= T
            df = df[has_target].copy()
        df = df[df["origin_week"] >= self.cfg["experiment"]["min_origin_week"]]
        return df.reset_index(drop=True)
