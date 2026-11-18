"""Synthetic-data acceptance criteria (specification section 12).

run_acceptance(cfg) returns one row per criterion (plus a detail table for the event checks).
All checks must pass before any model is trained.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import INDICATOR_FIELDS, market_ids, crop_ids, SCENARIOS
from .generator import generate_clean_world, event_table
from .degrade import degrade
from .features import panel_from_frame, FeatureStore, feature_columns, FEATURE_GROUPS


def _row(check, criterion, passed, detail):
    return dict(check=check, criterion=criterion, passed=bool(passed), detail=detail)


# ----------------------------------------------------------------------------- individual checks
def check_coverage_uniqueness(panels: dict, cfg: dict):
    n_exp = len(market_ids(cfg)) * len(crop_ids(cfg)) * cfg["experiment"]["n_weeks"]
    keys = ["market_id", "crop_id", "week_start"]
    ok_cov = all(len(p) == n_exp for p in panels.values())
    ok_uni = all(not p.duplicated(keys).any() for p in panels.values())
    grid_ok = all(p.groupby(["market_id", "crop_id"]).size().eq(cfg["experiment"]["n_weeks"]).all()
                  for p in panels.values())
    return [
        _row("Coverage", "Exactly 5 markets x 4 crops x 156 weeks before row-level corruption",
             ok_cov and grid_ok, f"{n_exp} rows in each of {len(panels)} scenarios"),
        _row("Uniqueness", "No duplicate market_id + crop_id + week_start keys", ok_uni, "0 duplicate keys" if ok_uni else "duplicates found"),
    ]


def check_price_validity(panels: dict):
    msgs, ok = [], True
    for sc, p in panels.items():
        lat_pos = (p["latent_price_kes_kg"] > 0).all()
        obs = p.dropna(subset=["wholesale_price_kes_kg"])
        med = p.groupby(["market_id", "crop_id"])["latent_price_kes_kg"].transform("median").loc[obs.index]
        bad = (obs["wholesale_price_kes_kg"] <= 0) | (obs["wholesale_price_kes_kg"] > 3.5 * med) | \
              (obs["wholesale_price_kes_kg"] < 0.3 * med)
        unexplained = int((bad & (obs["data_quality_flag"] != "suspect")).sum())
        n_flag = int((obs["data_quality_flag"] == "suspect").sum())
        ok &= bool(lat_pos) and unexplained == 0
        msgs.append(f"{sc}: latent>0={lat_pos}, implausible-but-unflagged={unexplained}, labelled errors={n_flag}")
    return [_row("Price validity", "All prices positive; extremes explained by events or labelled errors", ok, "; ".join(msgs))]


def _harmonic_fit(y: np.ndarray, doy: np.ndarray):
    X = np.column_stack([np.ones_like(doy), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                         np.sin(4 * np.pi * doy / 365.25), np.cos(4 * np.pi * doy / 365.25)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    r2 = 1 - ((y - fit) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    amp = float(np.hypot(beta[1], beta[2]))
    phase = float((np.degrees(np.arctan2(beta[1], beta[2])) % 360) / 360 * 365.25)
    return r2, amp, phase


def check_seasonality(world: pd.DataFrame):
    w = world.copy()
    w["doy"] = pd.to_datetime(w["week_start"]).dt.dayofyear + 3
    w["lp"] = np.log(w["latent_price_kes_kg"])
    w["lp_c"] = w["lp"] - w.groupby(["market_id", "crop_id"])["lp"].transform("mean")
    rows = {}
    for c, g in w.groupby("crop_id"):
        rows[c] = _harmonic_fit(g["lp_c"].values, g["doy"].values.astype(float))
    n_seasonal = sum(r2 > 0.10 for r2, _, _ in rows.values())
    sig = [(round(a, 3), round(p)) for _, a, p in rows.values()]
    distinct = len(set(sig)) == len(sig) and np.ptp([a for _, a, _ in rows.values()]) > 0.02
    detail = "; ".join(f"{c}: R2={r:.2f} amp={a:.3f} peak-doy={p:.0f}" for c, (r, a, p) in rows.items())
    return [_row("Seasonality", "At least some crops show recurring annual patterns, not identical across crops",
                 n_seasonal >= 2 and distinct, detail)]


def check_market_differences(world: pd.DataFrame):
    w = world.sort_values(["market_id", "crop_id", "week_index"]).copy()
    w["lp"] = np.log(w["latent_price_kes_kg"])
    w["ret"] = w.groupby(["market_id", "crop_id"])["lp"].diff()
    spreads, vols = [], []
    for c, g in w.groupby("crop_id"):
        spreads.append(g.groupby("market_id")["lp"].mean().agg(lambda s: s.max() - s.min()))
        v = g.groupby("market_id")["ret"].std()
        vols.append(v.max() / v.min())
    ok = min(spreads) >= 0.05 and min(vols) >= 1.10
    return [_row("Market differences", "Markets have distinct price levels and volatility",
                 ok, f"min across crops of log-level spread = {min(spreads):.3f}; min vol max/min ratio = {min(vols):.2f}")]


def check_persistence(world: pd.DataFrame):
    w = world.sort_values(["market_id", "crop_id", "week_index"]).copy()
    w["lp"] = np.log(w["latent_price_kes_kg"])
    rng = np.random.default_rng(0)
    acfs, null = [], []
    for _, g in w.groupby(["market_id", "crop_id"]):
        x = g["lp"].values; x = x - x.mean()
        acfs.append((x[1:] * x[:-1]).sum() / (x ** 2).sum())
        r = g["lp"].diff().dropna().values
        null.append(abs(np.corrcoef(r[1:], rng.permutation(r)[:-1])[0, 1]))   # shuffled-returns benchmark
    return [_row("Persistence", "Prices exhibit continuity, not independent weekly draws",
                 min(acfs) > 0.5, f"lag-1 autocorrelation of log price: min={min(acfs):.2f}, mean={np.mean(acfs):.2f} "
                                  f"(shuffled benchmark ~{np.mean(null):.2f})")]


def check_missingness(panels: dict, cfg: dict):
    rows, ok = [], True
    for sc in ["realistic"]:
        p = panels[sc]
        pm = p["wholesale_price_kes_kg"].isna().mean()
        out_rate = (p["data_quality_flag"] == "suspect").mean()
        im = {f: p[f].isna().mean() for f in cfg["scenarios"][sc]["indicator_missing"]}
        ok_p = 0.03 <= pm <= 0.05
        ok_o = 0.005 <= out_rate <= 0.011
        ok_i = all(0.045 <= v <= 0.11 for v in im.values())
        ok &= ok_p and ok_o and ok_i
        rows.append(f"realistic: price missing {pm:.2%} (3-5%), errors {out_rate:.2%} (0.5-1%), indicator missing "
                    f"{min(im.values()):.1%}-{max(im.values()):.1%} (5-10%)")
    c = panels["clean"]; s = panels["stressed"]
    ok_c = c["wholesale_price_kes_kg"].isna().sum() == 0 and c[INDICATOR_FIELDS].isna().sum().sum() == 0
    ok_s = s["wholesale_price_kes_kg"].isna().mean() > panels["realistic"]["wholesale_price_kes_kg"].isna().mean()
    rows.append(f"clean has no missing: {ok_c}; stressed price missing {s['wholesale_price_kes_kg'].isna().mean():.2%} > realistic: {ok_s}")
    return [_row("Missingness", "Rates match the selected scenario", ok and ok_c and ok_s, " | ".join(rows))]


# ----------------------------------------------------------------------------- event checks
def check_events(world: pd.DataFrame, cfg: dict):
    mk, cr = market_ids(cfg), crop_ids(cfg)
    w = world.set_index(["market_id", "crop_id", "week_index"]).sort_index()
    res = []

    def series(m, c, f):
        return w.loc[(m, c)][f]

    for ev in cfg["events"]:
        ms = mk if ev["markets"] == "all" else ev["markets"]
        cs = cr if ev["crops"] == "all" else ev["crops"]
        s, e, t = ev["start"], ev["end"], ev["type"]
        ratios = []
        for m in ms:
            for c in cs:
                if t == "drought":
                    r = series(m, c, "rainfall_mm")
                    ratios.append(r.loc[s:e].mean() / r.median())
                elif t == "flood":
                    r = series(m, c, "rainfall_mm"); ratios.append(r.loc[s:e].mean() / r.median())
                elif t in ("harvest_glut", "import_supply_shock"):
                    a = series(m, c, "market_arrivals_tonnes")
                    base = a.loc[max(1, s - 10):s - 1].mean() if s > 11 else a.median()
                    ratios.append(a.loc[s:e].mean() / base)
                elif t == "cross_market_disruption":
                    a = series(m, c, "market_arrivals_tonnes")
                    base = a.loc[max(1, s - 10):s - 1].mean()
                    ratios.append(a.loc[s:e].mean() / base)
                elif t == "demand_surge":
                    d = series(m, c, "estimated_demand_index")
                    ratios.append(d.loc[s:e].mean() / d.loc[max(1, s - 8):s - 1].mean())
                elif t in ("fuel_step", "structural_transport_shift"):
                    f = series(m, c, "fuel_cost_index")
                    pre = f.loc[max(1, s - 8):s - 1].mean()
                    post = f.loc[s:min(s + 3, 156)].mean()
                    ratios.append(post / pre)
        ratio = float(np.mean(ratios))
        crit = {"drought": ("rainfall ratio < 0.6", ratio < 0.6),
                "flood": ("rainfall ratio > 2.0", ratio > 2.0),
                "harvest_glut": ("arrivals ratio > 1.25", ratio > 1.25),
                "import_supply_shock": ("arrivals ratio > 1.25", ratio > 1.25),
                "cross_market_disruption": ("source arrivals ratio < 0.65", ratio < 0.65),
                "demand_surge": ("demand ratio > 1.10", ratio > 1.10),
                "fuel_step": ("fuel index ratio > 1.12", ratio > 1.12),
                "structural_transport_shift": ("fuel index ratio > 1.25", ratio > 1.25)}[t]
        res.append(dict(event_id=ev["id"], event_type=t, expectation=crit[0], measured_ratio=round(ratio, 3), passed=bool(crit[1])))
    detail = pd.DataFrame(res)
    # delayed supply response and cross-market spillover of the drought / disruption events
    extra = []
    ev12 = next(e for e in cfg["events"] if e["id"] == "EV12")
    # difference-in-differences on potatoes (no other test-window event): drought vs control markets,
    # arrivals 6-16 weeks after the dry spell started (lagged effect) relative to a pre-event baseline
    pot = world[world.crop_id == "CRP-03"].pivot(index="week_index", columns="market_id", values="market_arrivals_tonnes")
    treated, control = ev12["markets"], [m for m in pot.columns if m not in ev12["markets"]]
    post = slice(ev12["start"] + 8, ev12["end"] + 7); pre = slice(100, 111)
    did = (pot[treated].loc[post].mean().sum() / pot[treated].loc[pre].mean().sum()) / \
          (pot[control].loc[post].mean().sum() / pot[control].loc[pre].mean().sum())
    lag_ratio = float(did)
    extra.append(f"EV12 drought: potato arrivals in drought markets vs controls (diff-in-diff) = {lag_ratio:.2f}x, "
                 f"measured 8+ weeks after onset (delayed weather-to-supply effect)")
    ev08 = next(e for e in cfg["events"] if e["id"] == "EV08")
    rsi = world[world.market_id == "MKT-01"].groupby("week_index")["regional_supply_index"].mean()
    sp = rsi.loc[ev08["start"] + 1: ev08["end"] + 1].mean() / rsi.loc[100:110].mean()
    extra.append(f"EV08 source disruption: Nairobi regional supply index = {sp:.2f}x baseline (1-week-lag spillover)")
    ok = bool(detail["passed"].all()) and lag_ratio < 0.95 and sp < 0.95
    return [_row("Shock behaviour", "Documented shocks move the relevant variables plausibly", ok,
                 f"{int(detail['passed'].sum())}/{len(detail)} events pass; " + "; ".join(extra))], detail


# ----------------------------------------------------------------------------- leakage / reproducibility
def leakage_truncation_test(panel_df: pd.DataFrame, cfg: dict, handling: bool, cut_points=(60, 100, 130), h=4):
    """Features of origins <= T0 must be identical when all later data is removed."""
    P = panel_from_frame(panel_df, cfg)
    full = FeatureStore(P, cfg, handling).frame(h, with_targets=False)
    cols = feature_columns("E4", full.columns)[0]
    worst = 0.0
    for T0 in cut_points:
        part = FeatureStore(P.truncate(T0), cfg, handling).frame(h, with_targets=False)
        ref = full[full["origin_week"] <= T0].reset_index(drop=True)
        a = ref[cols].to_numpy(float); b = part[cols].to_numpy(float)
        if a.shape != b.shape:
            return False, float("inf")
        same_nan = np.array_equal(np.isnan(a), np.isnan(b))
        diff = np.nanmax(np.abs(np.where(np.isnan(a) | np.isnan(b), 0, a - b))) if a.size else 0
        if not same_nan:
            return False, float("inf")
        worst = max(worst, float(diff))
    return worst == 0.0, worst


def check_no_leakage(panels: dict, cfg: dict):
    # 1) latent never a feature; 2) truncation invariance for handled + raw pipelines on the stressed panel
    P = panel_from_frame(panels["stressed"], cfg)
    fr = FeatureStore(P, cfg, True).frame(2)
    feat_cols = set(sum([feature_columns(f, fr.columns)[0] for f in ["E0", "E1", "E2", "E3", "E4"]], []))
    banned = [c for c in feat_cols if "latent" in c or c.startswith("y_") or "event_ids" in c or "audit" in c]
    ok_a = len(banned) == 0
    ok_h, d_h = leakage_truncation_test(panels["stressed"], cfg, True)
    ok_r, d_r = leakage_truncation_test(panels["stressed"], cfg, False)
    return [_row("No hidden leakage", "All features reconstructable from information available at the forecast origin",
                 ok_a and ok_h and ok_r,
                 f"banned columns in feature sets: {banned or 'none'}; truncation test (cut weeks 60/100/130): "
                 f"handled max|diff|={d_h:.1e}, raw max|diff|={d_r:.1e}")]


def check_reproducibility(world: pd.DataFrame, cfg: dict):
    w2 = generate_clean_world(cfg)
    w3 = generate_clean_world(cfg, seed=cfg["experiment"]["seed"] + 1)
    same = w2.drop(columns=[]).equals(world)
    differs = not np.allclose(w3["latent_price_kes_kg"].values, world["latent_price_kes_kg"].values)
    d1, _ = degrade(world, cfg, "realistic"); d2, _ = degrade(w2, cfg, "realistic")
    same_deg = d1.equals(d2)
    ev_same = event_table(cfg).equals(event_table(cfg))
    return [_row("Reproducibility", "Same configuration and seed generate the same data and event schedule",
                 same and same_deg and differs and ev_same,
                 f"world identical={same}, degraded panel identical={same_deg}, different seed differs={differs}")]


# ----------------------------------------------------------------------------- driver
def run_acceptance(cfg: dict, panels: dict | None = None, world: pd.DataFrame | None = None):
    world = generate_clean_world(cfg) if world is None else world
    if panels is None:
        panels = {sc: degrade(world, cfg, sc)[0] for sc in SCENARIOS}
    rows = []
    rows += check_coverage_uniqueness(panels, cfg)
    rows += check_price_validity(panels)
    rows += check_seasonality(world)
    rows += check_market_differences(world)
    ev_rows, ev_detail = check_events(world, cfg)
    rows += ev_rows
    rows += check_persistence(world)
    rows += check_missingness(panels, cfg)
    rows += check_no_leakage(panels, cfg)
    rows += check_reproducibility(world, cfg)
    return pd.DataFrame(rows), ev_detail
