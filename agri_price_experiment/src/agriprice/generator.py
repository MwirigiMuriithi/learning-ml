"""Synthetic agricultural market world: the CLEAN latent process.

Design (see docs/generator_specification.md)
--------------------------------------------
log P[m,c,t] = mu[c] + alpha[m,c] + S[c,t] + z[m,c,t]

    z[m,c,t] = phi_c * ((1-delta_c) * z[m,c,t-1] + delta_c * sum_j W[m,j] z[j,c,t-1])   # persistence + spillover
             + (1-phi_c) * ( beta_s f(s_eff) + beta_d d_eff + beta_tr rw_m tau
                             + beta_reg log RSI + gamma_dr drought )                      # partial adjustment to drivers
             + sigma_c * noise_m * eps_t                                                    # random variation

This is the design equation of the specification in "partial adjustment" form:
phi*log P[t-1] is replaced by phi*(log P[t-1] - deterministic level), and the
driver coefficients are long-run elasticities (the weekly loading is 1-phi).
Supply, demand, weather, transport and regional supply are generated first (or
alongside, for supply <-> regional supply), so every price movement has a
documented cause.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .config import market_ids, crop_ids


# ----------------------------------------------------------------------------- helpers
def _circ_dist(doy: np.ndarray, centre: float) -> np.ndarray:
    return np.abs((doy - centre + 182.625) % 365.25 - 182.625)


def _ar1(rng, n_steps: int, shape: tuple, rho: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) with marginal std = sigma; first axis is time."""
    out = np.zeros((n_steps,) + shape)
    out[0] = rng.normal(0, sigma, shape)
    innov = sigma * np.sqrt(1 - rho ** 2)
    for t in range(1, n_steps):
        out[t] = rho * out[t - 1] + rng.normal(0, innov, shape)
    return out


def _std_t(rng, shape, df=5):
    return rng.standard_t(df, shape) / np.sqrt(df / (df - 2))


def _lag_window_mean(x: np.ndarray, lag: int, win: int) -> np.ndarray:
    """out[t] = mean(x[t-lag-win+1 .. t-lag]) along axis 0 (zero-padded history)."""
    T = x.shape[0]
    out = np.zeros_like(x, dtype=float)
    for k in range(win):
        s = lag + k
        if s < T:
            out[s:] += x[: T - s]
    return out / win


def _profile(w: np.ndarray, ev: dict) -> np.ndarray:
    """Event intensity profile over weeks w (1-based, burn-in weeks <= 0)."""
    start, end = ev["start"], ev["end"]
    inside = (w >= start) & (w <= end)
    prof = inside.astype(float)
    if ev["type"] == "drought":                       # ramp up over first 2 weeks
        prof = prof * np.clip((w - start + 1) / 3.0, 0.4, 1.0)
    return prof


def holiday_week_flags(week_starts: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    """Calendar-known holiday flag for each week (Christmas window or the week before Easter)."""
    d = cfg["demand"]
    flags = np.zeros(len(week_starts), dtype=bool)
    easter = [pd.Timestamp(x) for x in d["easter_dates"]]
    for i, ws in enumerate(week_starts):
        days = pd.date_range(ws, ws + pd.Timedelta(days=6))
        for day in days:
            xmas = pd.Timestamp(year=day.year, month=12, day=25)
            if abs((day - xmas).days) <= d["christmas_window_days"]:
                flags[i] = True
        for e in easter:
            if (e - pd.Timedelta(days=7)) <= ws + pd.Timedelta(days=6) and ws <= e:
                flags[i] = True
    return flags


# ----------------------------------------------------------------------------- events
def event_table(cfg: dict) -> pd.DataFrame:
    sp = cfg["experiment"]["splits"]
    rows = []
    for ev in cfg["events"]:
        s, e = ev["start"], ev["end"]
        if e <= sp["train_end"]:
            part = "train"
        elif s > sp["val_end"]:
            part = "test"
        elif s > sp["train_end"] and e <= sp["val_end"]:
            part = "validation"
        else:
            part = "spans_boundary"
        mk = market_ids(cfg) if ev["markets"] == "all" else ev["markets"]
        cr = crop_ids(cfg) if ev["crops"] == "all" else ev["crops"]
        rows.append(dict(event_id=ev["id"], event_type=ev["type"], markets="|".join(mk),
                         crops="|".join(cr), start_week=s, end_week=e,
                         effect_tail_weeks=ev.get("tail", 0), intensity=ev["intensity"],
                         half_life_weeks=ev.get("half_life"), partition=part,
                         new_in_test=bool(ev.get("new_in_test", False)),
                         is_structural_change=ev["type"] == "structural_transport_shift",
                         description=ev["description"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def generate_clean_world(cfg: dict, seed: int | None = None, return_hidden: bool = False):
    """Return the long-format CLEAN world (latent price + clean indicators + event labels).

    With return_hidden=True also returns the hidden driver arrays (audit only)."""
    seed = cfg["experiment"]["seed"] if seed is None else seed
    ss = np.random.SeedSequence(seed)
    r_weather, r_supply, r_demand, r_fuel, r_price = [np.random.default_rng(s) for s in ss.spawn(5)]

    ex = cfg["experiment"]
    N, B = ex["n_weeks"], ex["burn_in_weeks"]
    T = N + B
    mk, cr = market_ids(cfg), crop_ids(cfg)
    M, C = len(mk), len(cr)
    mcfg = [cfg["markets"][m] for m in mk]
    ccfg = [cfg["crops"][c] for c in cr]

    wk = np.arange(1 - B, N + 1)                                         # week numbers (<=0 burn-in)
    dates = pd.Timestamp(ex["start_date"]) + pd.to_timedelta((wk - 1) * 7, unit="D")
    doy = (dates + pd.Timedelta(days=3)).dayofyear.values.astype(float)

    # ------------------------------------------------------------------ event effect arrays
    ev_rain = np.ones((T, M))
    ev_temp = np.zeros((T, M))
    ev_arr = np.ones((T, M, C))
    ev_trans = np.ones((T, M))
    ev_dem = np.ones((T, M))
    fuel_step = np.zeros(T)
    affected = np.zeros((T, M, C), dtype=bool)
    regime = np.zeros((T, M, C), dtype=bool)
    surge_core = np.zeros((T, M), dtype=bool)
    ev_ids = np.empty((T, M, C), dtype=object)
    ev_ids[:] = ""

    for ev in cfg["events"]:
        mi = list(range(M)) if ev["markets"] == "all" else [mk.index(m) for m in ev["markets"]]
        ci = list(range(C)) if ev["crops"] == "all" else [cr.index(c) for c in ev["crops"]]
        inten, typ = ev["intensity"], ev["type"]
        prof = _profile(wk, ev)
        for m in mi:
            sens = mcfg[m]["shock_sens"]
            if typ == "drought":
                ev_rain[:, m] *= 1 - 0.85 * inten * prof
                ev_temp[:, m] += 2.0 * inten * prof
            elif typ == "flood":
                ev_rain[:, m] *= 1 + 3.0 * inten * prof
                ev_trans[:, m] *= 1 + 0.5 * inten * prof
                for c in ci:
                    ev_arr[:, m, c] *= 1 - 0.35 * inten * prof
            elif typ == "harvest_glut":
                for c in ci:
                    ev_arr[:, m, c] *= 1 + inten * sens * prof
            elif typ == "import_supply_shock":
                for c in ci:
                    ev_arr[:, m, c] *= 1 + inten * prof
            elif typ == "cross_market_disruption":
                for c in ci:
                    ev_arr[:, m, c] *= 1 - inten * prof
            elif typ == "demand_surge":
                ev_dem[:, m] *= 1 + inten * prof
                surge_core[:, m] |= prof > 0
        if typ == "fuel_step":
            hl = ev.get("half_life", 10)
            after = np.where(wk >= ev["start"], 0.5 ** ((wk - ev["start"]) / hl), 0.0)
            fuel_step += inten * after
        if typ == "structural_transport_shift":
            fuel_step += inten * (wk >= ev["start"])
        # labels
        win = (wk >= ev["start"]) & (wk <= ev["end"] + ev.get("tail", 0))
        for m in mi:
            for c in ci:
                if typ == "structural_transport_shift":
                    regime[win, m, c] = True
                else:
                    affected[win, m, c] = True
                idx = np.where(win)[0]
                for t in idx:
                    ev_ids[t, m, c] = (ev_ids[t, m, c] + "|" + ev["id"]).strip("|")

    # ------------------------------------------------------------------ weather
    wc = cfg["weather"]
    rain_mult = np.array([m["rain_mult"] for m in mcfg])
    clim = (8 + 38 * np.exp(-0.5 * (_circ_dist(doy, 110) / 32) ** 2)
            + 30 * np.exp(-0.5 * (_circ_dist(doy, 315) / 28) ** 2))[:, None] * rain_mult[None, :]
    wsh = wc["rain_shared_weight"]
    xi = r_weather.normal(size=(T, 1)); eta = r_weather.normal(size=(T, M))
    innov = np.sqrt(wsh) * xi + np.sqrt(1 - wsh) * eta
    la = np.zeros((T, M)); la[0] = innov[0]
    for t in range(1, T):
        la[t] = wc["rain_rho"] * la[t - 1] + np.sqrt(1 - wc["rain_rho"] ** 2) * innov[t]
    rs = wc["rain_sigma"]
    rain = clim * np.exp(rs * la - rs ** 2 / 2) * ev_rain
    rain_anom = np.log(rain / clim)

    tbase = np.array([m["temp_base"] for m in mcfg])
    temp = (tbase[None, :] + wc["temp_seasonal_amp"] * np.cos(2 * np.pi * (doy - 60) / 365.25)[:, None]
            + _ar1(r_weather, T, (M,), wc["temp_rho"], wc["temp_noise_sigma"]) + ev_temp)

    def roll_sum(x, k):
        out = np.zeros_like(x)
        for j in range(k):
            out[j:] += x[: T - j]
        return out
    ratio = roll_sum(rain, 8) / roll_sum(clim, 8)
    drought = np.clip((1 - ratio) / 0.4, 0, 4)

    # ------------------------------------------------------------------ demand
    dc = cfg["demand"]
    hol = holiday_week_flags(dates, cfg).astype(float)
    dem_level = np.array([m["demand_level"] for m in mcfg])
    hb = np.array([dc["holiday_boost"][c] for c in cr])
    season_d = 1 + dc["seasonal_amp"] * np.cos(2 * np.pi * (doy - 350) / 365.25)
    dnoise = _ar1(r_demand, T, (M, C), dc["noise_rho"], dc["noise_sigma"])
    demand_mult = (season_d[:, None, None] * (1 + hb[None, None, :] * hol[:, None, None])
                   * ev_dem[:, :, None] * np.exp(dnoise))
    demand_idx = 100 * dem_level[None, :, None] * demand_mult
    d_anom = np.log(demand_mult)

    # ------------------------------------------------------------------ logistics
    lg = cfg["logistics"]
    fu = _ar1(r_fuel, T, (1,), lg["fuel_rho"], lg["fuel_sigma"])[:, 0]
    fuel = 100 * np.exp(fu) * (1 + fuel_step)
    dist = np.array([m["distance_km"] for m in mcfg])
    rate = np.array([lg["rate_kes_per_tonne_km"][c] for c in cr])
    tn = _ar1(r_fuel, T, (M, C), 0.5, lg["transport_noise_sigma"])
    base_transport = dist[:, None] * rate[None, :]
    transport = base_transport[None] * (fuel / 100)[:, None, None] * ev_trans[:, :, None] * np.exp(tn)
    tau = np.log(fuel / 100)[:, None, None] + np.log(ev_trans)[:, :, None] + tn
    route_w = dist / dist.mean()

    # ------------------------------------------------------------------ supply & price (time loop)
    W = np.zeros((M, M))
    for m, srcs in cfg["network"].items():
        for j, w in srcs.items():
            W[mk.index(m), mk.index(j)] = w
    assert np.allclose(W.sum(1), 1.0), "network weights must sum to 1"

    P = {k: np.array([cc[k] for cc in ccfg]) for k in
         ["phi", "sigma", "season_amp", "season_amp2", "harvest_peak_doy", "harvest_amp",
          "arr_sigma", "arr_rho", "storage_lambda", "beta_supply", "scarcity_kappa",
          "beta_demand", "beta_transport", "beta_regional", "gamma_drought", "weather_lag_weeks",
          "a_rain", "a_drought", "a_temp", "delta_spill", "base_price"]}
    base_t = np.array([[cc["base_tonnes"][m] for cc in ccfg] for m in mk])        # (M,C)
    sens = np.array([m["shock_sens"] for m in mcfg])
    noise_mult = np.array([m["noise_mult"] for m in mcfg])
    alpha = np.array([[mcfg[i]["alpha"] + cfg["alpha_mc_adjust"].get(f"{mk[i]}|{cr[j]}", 0.0)
                       for j in range(C)] for i in range(M)])

    # deterministic harvest seasonality in supply and seasonality in log price
    harvest = np.stack([1 + P["harvest_amp"][c] * np.cos(2 * np.pi * (doy - P["harvest_peak_doy"][c]) / 365.25)
                        for c in range(C)], axis=1)                                 # (T,C)
    S = np.stack([-P["season_amp"][c] * np.cos(2 * np.pi * (doy - P["harvest_peak_doy"][c]) / 365.25)
                  + P["season_amp2"][c] * np.cos(4 * np.pi * (doy - P["harvest_peak_doy"][c] - 60) / 365.25)
                  for c in range(C)], axis=1)                                       # (T,C)
    expected = base_t[None] * harvest[:, None, :]                                   # (T,M,C)

    # lagged weather -> supply factor (per crop lag)
    wf = np.zeros((T, M, C))
    temp_exc = np.maximum(temp - wc["tomato_temp_threshold_c"], 0)
    for c in range(C):
        lag = int(P["weather_lag_weeks"][c])
        ra = _lag_window_mean(rain_anom, lag, 4)
        dr = _lag_window_mean(drought, lag, 4)
        te = _lag_window_mean(temp_exc, 2, 2)
        lw = P["a_rain"][c] * np.tanh(ra) - P["a_drought"][c] * sens[None, :] * np.minimum(dr, 2.5) \
            - P["a_temp"][c] * te
        lw = lw - lw[B:].mean(axis=0, keepdims=True)          # centre: no built-in supply bias
        wf[:, :, c] = np.exp(lw)

    kappa_reg = lg["regional_arrival_kappa"]
    arr_noise = np.zeros((T, M, C))
    for c in range(C):
        arr_noise[:, :, c] = (_ar1(r_supply, T, (M,), P["arr_rho"][c], P["arr_sigma"][c])
                              - P["arr_sigma"][c] ** 2 / 2)

    arrivals = np.zeros((T, M, C)); RSI = np.ones((T, M, C)); s_anom = np.zeros((T, M, C))
    s_eff = np.zeros((T, M, C)); z = np.zeros((T, M, C))
    eps = _std_t(r_price, (T, M, C))
    lam = P["storage_lambda"][None, :]
    phi = P["phi"][None, :]; dlt = P["delta_spill"][None, :]
    kap = P["scarcity_kappa"][None, :]
    for t in range(T):
        if t > 0:
            ratio_prev = arrivals[t - 1] / expected[t - 1]
            RSI[t] = W @ ratio_prev
        arrivals[t] = (expected[t] * wf[t] * (1 + kappa_reg * (RSI[t] - 1)) * ev_arr[t]
                       * np.exp(arr_noise[t]))
        s_anom[t] = np.log(arrivals[t] / expected[t])
        prev_eff = s_eff[t - 1] if t > 0 else s_anom[t]
        s_eff[t] = lam * prev_eff + (1 - lam) * s_anom[t]
        f_s = s_eff[t] * (1 + kap * np.minimum(np.maximum(-s_eff[t], 0), 0.6))
        d_eff = d_anom[t] * (1 + 0.8 * np.minimum(np.maximum(-s_eff[t], 0), 0.6))
        target = (P["beta_supply"][None, :] * f_s + P["beta_demand"][None, :] * d_eff
                  + P["beta_transport"][None, :] * route_w[:, None] * tau[t]
                  + P["beta_regional"][None, :] * np.log(RSI[t])
                  + P["gamma_drought"][None, :] * drought[t][:, None])
        zprev = z[t - 1] if t > 0 else np.zeros((M, C))
        persist = phi * ((1 - dlt) * zprev + dlt * (W @ zprev))
        z[t] = (persist + (1 - phi) * target
                + P["sigma"][None, :] * noise_mult[:, None] * eps[t])

    logp = (np.log(P["base_price"])[None, None, :] + alpha[None] + S[:, None, :] + z)
    latent = np.exp(logp)

    # ------------------------------------------------------------------ assemble long format (drop burn-in)
    sl = slice(B, T)
    wk_o = wk[sl]
    ws_o = dates[sl]
    hol_o = hol[sl].astype(bool)
    src_dom = [max(cfg["network"][m], key=cfg["network"][m].get) for m in mk]
    rows = []
    n = len(wk_o)
    t_idx = np.repeat(np.arange(n), M * C)
    m_idx = np.tile(np.repeat(np.arange(M), C), n)
    c_idx = np.tile(np.arange(C), n * M)
    tt = t_idx + B
    df = pd.DataFrame({
        "week_index": wk_o[t_idx],
        "week_start": ws_o[t_idx].strftime("%Y-%m-%d"),
        "market_id": np.array(mk)[m_idx],
        "crop_id": np.array(cr)[c_idx],
        "latent_price_kes_kg": latent[tt, m_idx, c_idx],
        "market_arrivals_tonnes": arrivals[tt, m_idx, c_idx],
        "estimated_demand_index": demand_idx[tt, m_idx, c_idx],
        "rainfall_mm": rain[tt, m_idx],
        "temperature_c": temp[tt, m_idx],
        "drought_index": drought[tt, m_idx],
        "fuel_cost_index": fuel[tt],
        "transport_cost_kes_tonne": transport[tt, m_idx, c_idx],
        "regional_supply_index": RSI[tt, m_idx, c_idx],
        "holiday_or_event_flag": (hol_o[t_idx] | surge_core[tt, m_idx]).astype(int),
        "source_market_id": np.array(src_dom)[m_idx],
        "event_ids": ev_ids[tt, m_idx, c_idx],
        "event_affected": affected[tt, m_idx, c_idx],
        "regime_shift_flag": regime[tt, m_idx, c_idx],
    })
    # hidden drivers kept for the generator audit only (never model inputs)
    if return_hidden:
        return df, dict(z=z[sl], s_anom=s_anom[sl], expected=expected[sl], markets=mk, crops=cr)
    return df
