"""Controlled data-quality layer applied on top of the clean latent world.

The same clean world is degraded differently per scenario:
  clean     - minimal measurement noise, nothing missing
  realistic - missing prices/indicators, noise, a few reporting errors, late data
  stressed  - longer gaps, delayed indicators, market outages, more errors

Real shocks (events) and bad data (quality problems) are kept strictly separate:
events live in `event_ids` / `event_affected`, quality problems in
`data_quality_flag` and the annotation table.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import INDICATOR_FIELDS, DELAYABLE_FIELDS, market_ids, crop_ids

WEATHER_FIELDS = ["rainfall_mm", "temperature_c", "drought_index"]
NATIONAL_FIELDS = ["fuel_cost_index"]
ERROR_MULTIPLIERS = np.array([10.0, 0.1, 2.5, 0.4])
ERROR_PROBS = np.array([0.30, 0.20, 0.25, 0.15])   # remaining 10% -> impossible value (0)


def degrade(world: pd.DataFrame, cfg: dict, scenario: str, seed: int | None = None):
    """Return (observed_panel, annotations) for one scenario."""
    seed = cfg["experiment"]["seed"] if seed is None else seed
    sc = cfg["scenarios"][scenario]
    scen_id = {"clean": 1, "realistic": 2, "stressed": 3}[scenario]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1000 + scen_id]))

    df = world.copy().sort_values(["week_index", "market_id", "crop_id"]).reset_index(drop=True)
    n = len(df)
    N = cfg["experiment"]["n_weeks"]
    mk, cr = market_ids(cfg), crop_ids(cfg)
    ann = []

    def note(mask, issue, field, detail=""):
        sub = df.loc[mask, ["week_index", "week_start", "market_id", "crop_id"]].copy()
        sub["issue"] = issue; sub["field"] = field; sub["detail"] = detail
        ann.append(sub)

    # ---- measurement noise ------------------------------------------------
    price = df["latent_price_kes_kg"].values * np.exp(rng.normal(0, sc["price_noise_sigma"], n))
    df["wholesale_price_kes_kg"] = np.round(price, 2)
    if sc["indicator_noise_sigma"] > 0:
        for f in ["market_arrivals_tonnes", "estimated_demand_index", "transport_cost_kes_tonne"]:
            df[f] = df[f] * np.exp(rng.normal(0, sc["indicator_noise_sigma"], n))
        df["regional_supply_index"] = df["regional_supply_index"] * np.exp(
            rng.normal(0, sc["indicator_noise_sigma"] / 2, n))

    mult_m = df["market_id"].map(lambda m: sc["market_missing_mult"].get(m, 1.0)).values

    # ---- injected recording errors (outliers) ------------------------------
    out_mask = np.zeros(n, dtype=bool)
    k = int(round(sc["outlier_rate"] * n))
    if k > 0:
        idx = rng.choice(n, size=k, replace=False)
        out_mask[idx] = True
        kind = rng.choice(5, size=k, p=np.r_[ERROR_PROBS, 0.10])
        p = df["wholesale_price_kes_kg"].values.copy()
        for i, kd in zip(idx, kind):
            p[i] = 0.0 if kd == 4 else round(p[i] * ERROR_MULTIPLIERS[kd], 2)
        df["wholesale_price_kes_kg"] = p

    # ---- missing prices -----------------------------------------------------
    miss = rng.random(n) < np.clip(sc["price_missing_random_rate"] * mult_m, 0, 1)
    gap_mask = np.zeros(n, dtype=bool)
    if sc["price_missing_gap_rate"] > 0:
        gl = sc["price_gap_len"]; mean_len = np.mean(gl)
        sel = sc["price_gap_series"]
        n_gaps_total = sc["price_missing_gap_rate"] * n / mean_len
        per_series = max(1, int(round(n_gaps_total / max(len(sel), 1))))
        key = df["market_id"] + "|" + df["crop_id"]
        for s_key in sel:
            rows = np.where(key.values == s_key)[0]                  # sorted by week
            for _ in range(per_series):
                L = int(rng.integers(gl[0], gl[1] + 1))
                start = int(rng.integers(0, max(1, N - L)))
                gap_mask[rows[start:start + L]] = True
    outage = np.zeros(n, dtype=bool)
    for o in sc["outages"]:
        outage |= ((df["market_id"] == o["market"]) & df["week_index"].between(o["start"], o["end"])).values
    price_missing = miss | gap_mask | outage
    note(miss & ~gap_mask & ~outage, "missing_price", "wholesale_price_kes_kg", "random")
    note(gap_mask & ~outage, "missing_price", "wholesale_price_kes_kg", "contiguous_gap")
    note(outage, "missing_price", "wholesale_price_kes_kg", "market_reporting_outage")
    note(out_mask & ~price_missing, "outlier_injected", "wholesale_price_kes_kg", "recording error (audit label)")
    out_mask &= ~price_missing
    df.loc[price_missing, "wholesale_price_kes_kg"] = np.nan

    # ---- missing indicators ---------------------------------------------------
    key_wm = df["week_index"].astype(str) + "|" + df["market_id"]
    for f in INDICATOR_FIELDS:
        rate = sc["indicator_missing"].get(f, 0.0)
        if rate <= 0:
            continue
        if f in NATIONAL_FIELDS:                                     # whole week missing for all rows
            weeks = rng.random(N + 1) < rate
            m_ = df["week_index"].map(lambda w: weeks[w]).values
        elif f in WEATHER_FIELDS:                                    # per week-market (same catchment)
            uniq = key_wm.unique()
            draw = dict(zip(uniq, rng.random(len(uniq))))
            u = key_wm.map(draw).values
            m_ = u < np.clip(rate * mult_m, 0, 1)
        else:
            m_ = rng.random(n) < np.clip(rate * mult_m, 0, 1)
        m_ = m_ | outage
        df.loc[m_, f] = np.nan
        note(m_ & ~outage, "missing_indicator", f, "random")
    if sc["indicator_gap_rate"] > 0:
        n_gaps = int(round(sc["indicator_gap_rate"] * n / 3))
        for _ in range(n_gaps):
            mi, ci = rng.choice(mk), rng.choice(cr)
            f = rng.choice([x for x in INDICATOR_FIELDS if x not in NATIONAL_FIELDS])
            L = int(rng.integers(2, 5)); st = int(rng.integers(1, N - L))
            m_ = ((df["market_id"] == mi) & (df["crop_id"] == ci) & df["week_index"].between(st, st + L - 1)).values
            df.loc[m_, f] = np.nan
            note(m_, "missing_indicator", f, "contiguous_gap")
    if outage.any():
        for f in INDICATOR_FIELDS:
            df.loc[outage, f] = np.nan

    # ---- late-arriving indicator records --------------------------------------
    delay = np.zeros(n, dtype=int)
    if sc["delay_prob"] > 0:
        in_mk = df["market_id"].isin(sc["delayed_markets"]).values
        delay = (in_mk & (rng.random(n) < sc["delay_prob"])).astype(int)
    df["report_delay_weeks"] = delay
    note(delay == 1, "delayed_indicators", "|".join(DELAYABLE_FIELDS), "available 1 week after period end")

    # ---- quality flag (precedence: missing > suspect > synthetic_shock > clean) --
    flag = np.where(df["event_affected"].values, "synthetic_shock", "clean").astype(object)
    flag[out_mask] = "suspect"
    flag[price_missing] = "missing"
    df["data_quality_flag"] = flag
    df["scenario"] = scenario

    cols = ["week_start", "week_index", "market_id", "crop_id", "wholesale_price_kes_kg",
            "latent_price_kes_kg", "market_arrivals_tonnes", "estimated_demand_index",
            "rainfall_mm", "temperature_c", "drought_index", "fuel_cost_index",
            "transport_cost_kes_tonne", "regional_supply_index", "holiday_or_event_flag",
            "data_quality_flag", "source_market_id",
            "report_delay_weeks", "event_ids", "event_affected", "regime_shift_flag", "scenario"]
    df = df[cols].sort_values(["market_id", "crop_id", "week_index"]).reset_index(drop=True)
    for f in ["market_arrivals_tonnes", "estimated_demand_index", "rainfall_mm", "temperature_c",
              "drought_index", "fuel_cost_index", "transport_cost_kes_tonne", "latent_price_kes_kg"]:
        df[f] = df[f].round(3)
    df["regional_supply_index"] = df["regional_supply_index"].round(4)
    annotations = pd.concat(ann, ignore_index=True) if ann else pd.DataFrame()
    annotations["scenario"] = scenario
    return df, annotations
