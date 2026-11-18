#!/usr/bin/env python
"""Generate docs/*.md from the live configuration and code definitions (so they cannot drift)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from agriprice.config import load_config, ROOT, data_dir, results_dir
from agriprice.features import FEATURE_GROUPS, FEATURE_SETS, FEATURE_SET_LABEL

cfg = load_config()
DOCS = ROOT / "docs"; DOCS.mkdir(exist_ok=True)
ex = cfg["experiment"]
mdt = lambda df, ff=".3f": df.to_markdown(index=False, floatfmt=ff)

# ------------------------------------------------------------------ data dictionary
cols = [
 ("week_start", "date (ISO)", "Monday of the reporting week", "no", "model input only via calendar arithmetic"),
 ("week_index", "int 1-156", "Consecutive week number; partitions are defined on it", "no", "known"),
 ("market_id", "MKT-01..05", "Market identifier (Nairobi, Nakuru, Eldoret, Kisumu, Mombasa - illustrative names)", "no", "known"),
 ("crop_id", "CRP-01..04", "Crop identifier (maize, tomatoes, Irish potatoes, onions)", "no", "known"),
 ("wholesale_price_kes_kg", "KES/kg", "OBSERVED weekly wholesale price = latent price x measurement noise, possibly corrupted or missing. **Forecast target.**", "yes", "week w known at the end of week w"),
 ("latent_price_kes_kg", "KES/kg", "True generator price before noise/corruption. **Audit only - never a feature, never a training target.**", "no", "NEVER available to models"),
 ("market_arrivals_tonnes", "tonnes/week", "Quantity arriving at the market (supply)", "yes", "week w known at w+report_delay_weeks"),
 ("estimated_demand_index", "index (100 = market baseline x level)", "Relative demand index incl. seasonal, holiday and surge effects", "yes", "as above"),
 ("rainfall_mm", "mm/week", "Rainfall in the market's supply catchment", "yes", "on time"),
 ("temperature_c", "deg C", "Mean weekly temperature in the catchment", "yes", "on time"),
 ("drought_index", "0-4", "Rainfall-deficit index over the previous 8 weeks (0 = normal)", "yes", "on time"),
 ("fuel_cost_index", "index (100 = baseline)", "National fuel cost index", "yes", "on time (missing weeks hit all markets)"),
 ("transport_cost_kes_tonne", "KES/tonne", "Cost to move the crop to the market (distance x rate x fuel x disruption)", "yes", "week w known at w+report_delay_weeks"),
 ("regional_supply_index", "ratio (1 = expected)", "Weighted supply of connected source markets in the PREVIOUS week relative to their expectation", "yes", "week w known at w+report_delay_weeks"),
 ("holiday_or_event_flag", "0/1", "1 if holiday week (Christmas window, week before Easter) or a documented demand-surge event applies", "no", "known for week w at the end of w; only the pure holiday calendar is used for FUTURE weeks"),
 ("data_quality_flag", "clean / missing / suspect / synthetic_shock (/ corrected in cleaned data)", "Row quality label. Precedence missing > suspect > synthetic_shock > clean. `suspect` is an audit label of an injected recording error, **never a model input**", "no", "audit only"),
 ("source_market_id", "MKT-xx", "Dominant source market (highest network weight)", "no", "static metadata"),
 ("report_delay_weeks", "0 or 1", "1 = indicator record for this week is delivered one week late", "no", "known at delivery; used to build the as-known view"),
 ("event_ids", "string", "Documented events whose effect window covers the row (separate from quality labels)", "no", "audit only"),
 ("event_affected", "bool", "Row lies in an event window (excluding the lasting regime shift)", "no", "audit/evaluation only"),
 ("regime_shift_flag", "bool", "Row lies after the lasting transport-cost regime shift", "no", "audit/evaluation only"),
 ("scenario", "clean / realistic / stressed", "Data-quality scenario of the file", "no", "metadata"),
]
dd = pd.DataFrame(cols, columns=["column", "type / unit", "description", "may be missing", "availability at forecast origin"])
FDESC = {
 "tgt_week_of_year": "ISO week of the TARGET date", "tgt_month": "month of target", "tgt_quarter": "quarter of target",
 "tgt_sin1": "sin(2*pi*doy/365.25) of target", "tgt_cos1": "cos of target (annual)", "tgt_sin2": "second harmonic sin", "tgt_cos2": "second harmonic cos",
 "tgt_holiday": "holiday calendar flag for the target week (calendar only)", "org_week_of_year": "ISO week of the origin", "org_holiday": "holiday flag of the origin week",
 "lp0": "log of latest available (cleaned) price", "lr_1": "log change over last 1 week (origin vs origin-1)", "lr_2": "... 2 weeks", "lr_3": "... 3 weeks", "lr_4": "... 4 weeks",
 "lr_8": "... 8 weeks", "lr_13": "... 13 weeks", "lr_52": "... 52 weeks", "dev_mean_4": "log price minus mean log price of last 4 weeks", "dev_mean_8": "... 8 weeks", "dev_mean_13": "... 13 weeks",
 "dev_mean_26": "... 26 weeks", "vol_4": "std of weekly log changes, last 4 weeks", "vol_8": "std of weekly log changes, last 8 weeks",
 "seasonal_ref_ratio": "log(price one year before the TARGET week) minus log(latest price) - horizon-specific, known at origin", "price_stale_weeks": "weeks since the last accepted price (handling on only)",
 "arr_l0": "log arrivals, latest delivered value", "arr_l1": "log arrivals, one week earlier", "arr_mean4": "log mean arrivals, last 4 delivered weeks", "arr_anom13": "latest arrivals vs 13-week mean (log ratio)",
 "dem_l0": "log demand index, latest", "dem_l1": "log demand index, one week earlier", "dem_mean4": "log mean demand, 4 weeks", "event_flag_l0": "holiday/surge flag of the origin week",
 "rain_l0": "latest rainfall", "rain_mean4": "mean rainfall 4 wks", "rain_mean8": "mean rainfall 8 wks", "rain_sum13": "cumulative rainfall 13 wks", "temp_l0": "latest temperature", "temp_mean4": "mean temperature 4 wks",
 "drought_l0": "latest drought index", "drought_l2": "drought index 2 weeks earlier", "drought_mean4": "mean drought index 4 wks", "drought_max8": "max drought index 8 wks",
 "fuel_l0": "log fuel index, latest", "fuel_chg4": "log change of fuel index over 4 weeks", "trans_l0": "log transport cost, latest delivered", "trans_mean4": "log mean transport cost 4 wks", "trans_chg4": "log change of transport cost over 4 wks",
 "conn_gap0": "weighted log price of source markets minus own log price (same crop)", "conn_chg1": "weighted 1-week log price change of source markets", "conn_chg4": "weighted 4-week log price change of source markets",
 "regsup_l0": "log regional supply index, latest", "regsup_mean4": "log mean regional supply, 4 wks", "regsup_chg1": "1-week log change of regional supply index",
}
grp_of = {c: g for g, cs in FEATURE_GROUPS.items() for c in cs}; grp_of["price_stale_weeks"] = "price"
first_set = {}
for fs, gs in FEATURE_SETS.items():
    for g in gs:
        first_set.setdefault(g, fs)
frows = [(c, grp_of[c], first_set[grp_of[c]], FDESC.get(c, "")) for c in FDESC]
fdd = pd.DataFrame(frows, columns=["feature", "group", "enters at", "definition"])
with open(DOCS / "data_dictionary.md", "w") as f:
    f.write("# Data Dictionary\n\n## Panel columns (`data/synthetic_panel_<scenario>.csv`)\n\n" + mdt(dd) + "\n\n")
    f.write("Rows: 5 markets x 4 crops x 156 weeks = 3,120 per scenario. Key = (`market_id`, `crop_id`, `week_start`).\n\n")
    f.write("## Other data files\n\n- `data/events_schedule.csv` - the documented event calendar (type, markets, crops, weeks, intensity, partition, `new_in_test`).\n"
            "- `data/data_quality_annotations.csv` - every injected data-quality issue (missing price/indicator, outlier, outage, delay) per scenario.\n"
            "- `data/clean_world_audit.csv` - the clean latent world (audit only).\n- `data/manifest.json` - seed, config hash and file checksums.\n\n")
    f.write("## Engineered features (`src/agriprice/features.py`)\n\nFeature sets: " + "; ".join(f"**{k}** = {FEATURE_SET_LABEL[k]}" for k in FEATURE_SETS) +
            ". Every set also contains `market_id` and `crop_id`.\n\n" + mdt(fdd) + "\n")

# ------------------------------------------------------------------ generator specification
crop = pd.DataFrame([{"crop": v["name"], "base KES/kg": v["base_price"], "phi (persistence)": v["phi"], "sigma (weekly)": v["sigma"],
                      "annual amp": v["season_amp"], "harvest peak doy": v["harvest_peak_doy"], "storage lambda": v["storage_lambda"],
                      "beta supply": v["beta_supply"], "beta demand": v["beta_demand"], "beta transport": v["beta_transport"],
                      "beta regional": v["beta_regional"], "gamma drought": v["gamma_drought"], "weather lag (wks)": v["weather_lag_weeks"],
                      "spillover delta": v["delta_spill"]} for v in cfg["crops"].values()])
mkt = pd.DataFrame([{"id": k, "market": v["name"], "demand level": v["demand_level"], "distance km": v["distance_km"], "log premium": v["alpha"],
                     "noise mult": v["noise_mult"], "shock sensitivity": v["shock_sens"], "rain mult": v["rain_mult"]} for k, v in cfg["markets"].items()])
net = pd.DataFrame([{"receiving market": m, **{s: w for s, w in srcs.items()}} for m, srcs in cfg["network"].items()]).fillna(0)
ev = pd.read_csv(data_dir() / "events_schedule.csv")
sc = pd.DataFrame([{"scenario": k, "price noise sd": v["price_noise_sigma"], "indicator noise sd": v["indicator_noise_sigma"],
                    "price missing (random)": v["price_missing_random_rate"], "price missing (gaps)": v["price_missing_gap_rate"],
                    "gap length (wks)": str(v["price_gap_len"]), "outlier rate": v["outlier_rate"], "late-delivery prob": v["delay_prob"],
                    "late markets": ", ".join(v["delayed_markets"]) or "-", "outages": "; ".join(f"{o['market']} w{o['start']}-{o['end']}" for o in v["outages"]) or "-"}
                   for k, v in cfg["scenarios"].items()])
with open(DOCS / "generator_specification.md", "w") as f:
    f.write(f"""# Synthetic Data Generator Specification

Configuration: `config/experiment_config.yaml` (seed **{ex['seed']}**). Code: `src/agriprice/generator.py` (clean world) and `src/agriprice/degrade.py` (data quality).
The data are **fully synthetic and illustrative** - they mimic the *structure* of agricultural wholesale markets, not any real market.

## 1. Structure

{ex['n_weeks']} weeks starting {ex['start_date']}, {len(cfg['markets'])} markets x {len(cfg['crops'])} crops = {len(cfg['markets'])*len(cfg['crops'])} series, {ex['burn_in_weeks']} burn-in weeks discarded.
Weekly log price of crop *c* in market *m*:

```
log P[m,c,t] = log(base_c) + alpha[m,c] + S[c,t] + z[m,c,t]

S[c,t]   = -A1_c cos(2*pi (doy_t - peak_c)/365.25) + A2_c cos(4*pi (doy_t - peak_c - 60)/365.25)        # crop-specific seasonality
z[m,c,t] = phi_c * [ (1-delta_c) z[m,c,t-1] + delta_c * sum_j W[m,j] z[j,c,t-1] ]                      # persistence + cross-market spillover
         + (1-phi_c) * [ beta_s f(s_eff) + beta_d d_eff + beta_tr rw_m tau + beta_reg log RSI + gamma_dr drought ]   # adjustment to drivers
         + sigma_c * noise_m * eps_t                                                                     # random variation (Student-t, df 5)
```

This is the design equation of the specification in *partial-adjustment form*: the lagged-price term `phi*logP[t-1]` is written on deviations from the
deterministic level (`mu + alpha + S`) and the driver coefficients are long-run elasticities (weekly loading `1-phi`). Perishable tomatoes (phi 0.55) react
quickly; storable maize and onions (phi 0.90-0.92) adjust slowly.

* `s_eff` - supply anomaly `log(arrivals/expected)`, smoothed with `storage_lambda` for storable crops. `f(s) = s (1 + kappa min(max(-s,0),0.6))` makes **scarcity hit prices harder than glut relief** (nonlinear).
* `d_eff` - demand anomaly, amplified when supply is short (interaction).
* `tau` - log transport-cost deviation (fuel x disruption), scaled by route length `rw_m`.
* `RSI` - regional supply index: weighted arrivals of source markets in the **previous week** relative to expectation (also feeds back into the market's own arrivals, `kappa={cfg['logistics']['regional_arrival_kappa']}`) - this is the one-week-lag cross-market transmission.
* `drought` - 8-week rainfall-deficit index (small direct anticipatory price effect).
* Arrivals = `expected (harvest seasonality) x weather factor x regional response x event multipliers x AR(1) noise`. The **weather factor depends on rainfall, drought and (tomatoes) heat with a crop-specific delay** (tomatoes 3 weeks, maize 10 weeks), so weather history carries information about *future* supply.
* Weather: bimodal rainfall climatology (long rains Mar-May, short rains Oct-Dec) with spatially correlated lognormal anomalies; temperature seasonal + AR(1).
* Demand: market level x seasonal x holiday boost (Christmas, Easter) x surge events x AR(1) noise.
* Logistics: AR(1) fuel index, step/regime events; `transport = distance x rate x fuel/100 x disruption x noise`.

## 2. Parameters

### Crops
{mdt(crop)}

### Markets
{mdt(mkt)}

### Market network (weights of source markets; row sums = 1)
{mdt(net, '.2f')}

## 3. Documented event schedule

Week numbers are 1-based. Partitions: train 1-{ex['splits']['train_end']}, validation {ex['splits']['train_end']+1}-{ex['splits']['val_end']}, test {ex['splits']['val_end']+1}-{ex['splits']['test_end']}.
`new_in_test = True` marks shocks of a new type or greater strength than seen in training (EV12 longer/stronger drought, EV13 lasting transport-cost regime shift, EV14/EV15 stronger glut/import shock).

{mdt(ev[['event_id','event_type','markets','crops','start_week','end_week','effect_tail_weeks','intensity','partition','new_in_test','description']])}

Event labels (`event_ids`, `event_affected`) are deliberately separate from data-quality labels so that cleaning rules cannot erase genuine market behaviour.

## 4. Data-quality scenarios (applied to the *same* latent world)

{mdt(sc)}

Indicator missingness by field (realistic): {cfg['scenarios']['realistic']['indicator_missing']}; stressed: {cfg['scenarios']['stressed']['indicator_missing']}.
Reporting errors multiply the true value by 10, 0.1, 2.5 or 0.4, or report 0 (an impossible value).
Late-delivered records: supply, demand, transport and regional-supply indicators of selected markets reach the forecaster one week after the period ends.

## 5. Acceptance criteria

See `reports/DATA_AUDIT.md` (coverage, uniqueness, validity, seasonality, market differences, shock behaviour, persistence, missingness, no leakage, reproducibility).
""")

# ------------------------------------------------------------------ provenance
man = json.loads((data_dir() / "manifest.json").read_text()) if (data_dir() / "manifest.json").exists() else {}
with open(DOCS / "provenance_statement.md", "w") as f:
    f.write(f"""# Provenance and Intended-Use Statement

**What this is.** `data/synthetic_panel_*.csv` is a *fully synthetic* weekly wholesale-price panel (5 markets x 4 crops x 156 weeks). It was produced by
`scripts/01_generate_data.py` from `config/experiment_config.yaml` with seed {man.get('seed', ex['seed'])}. No real price, market, weather or trade record
was used, copied, or calibrated against; place names (Nairobi, Nakuru, Eldoret, Kisumu, Mombasa) and KES units are used only to make the structure readable.

**What it is for.** Comparing forecasting methods under *controlled, known* conditions: known causal drivers, documented shocks, adjustable data quality
and a hidden ground truth (`latent_price_kes_kg`) that lets us separate model error from measurement noise.

**What it must not be used for.**
- Any statement about real prices, real markets or real policy; any trading, procurement, credit or aid decision.
- Claims that a method that wins here will win on real data. Real markets contain mechanisms, regime changes and data problems that this generator does not.
- Estimating real-world elasticities: every coefficient is an assumption written in the configuration file.

**Known simplifications.** One national weather/fuel structure with market scaling; stationary price level around a deterministic seasonal pattern (no multi-year inflation or trend);
prices respond to arrivals but arrivals do not respond to past prices; no data revisions (a "corrected" flag appears only in pipeline-cleaned data); errors are gross
(x10/x0.1/x2.5/x0.4/zero) rather than subtle; only one realisation of the world (seed) per run, so conclusions should be confirmed across seeds.

**Reproducibility.** `python scripts/01_generate_data.py` regenerates all files bit-for-bit (SHA-256 in `data/manifest.json`).

File checksums: 
""" + "\n".join(f"- `{k}`: `{v.get('sha256','')[:16]}...`" for k, v in man.get("files", {}).items()) + "\n")

# ------------------------------------------------------------------ forecast-time data contract
with open(DOCS / "forecast_time_data_contract.md", "w") as f:
    f.write("""# Forecast-Time Data Contract

A **forecast origin** is the end of week *t*. The forecast target is the observed wholesale price of week *t+h* (h = 1, 2, 4).

| Item | Available at origin *t*? | Rule in code |
|---|---|---|
| price of week *w* <= *t* | yes | on time; cleaned **causally** (past-only outlier filter, last-observation-carried-forward with age cap, staleness feature) |
| indicator of week *w* <= *t* with `report_delay_weeks = 0` | yes | `asof_lag` |
| indicator of week *w* with `report_delay_weeks = 1` | only from week *w+1* | `asof_lag` masks lag-0 and, in handled mode, falls back to the last delivered value |
| weather, fuel (on-time sources) | yes | no delay |
| calendar facts of the **target** week (week of year, month, holiday calendar) | yes | date arithmetic + configured holiday calendar only |
| demand-surge events, shocks, event labels of future weeks | **no** | not used as features anywhere |
| `latent_price_kes_kg`, `data_quality_flag`, `event_*` columns | **never** | audit/evaluation only; a unit test fails if they appear in a feature set |
| prices / indicators of weeks > *t* | **no** | enforced by tests (truncation invariance) |
| data revisions after *t* | not simulated | no hindsight corrections are possible by construction |

**Chronological partitions** are decided by the origin week. Training rows are *purged*: a row is used for training only if its **target** week is within the training window
(an h-week embargo), so a model fitted for a partition has seen no information dated after that window. Validation: origins 105-130; test: origins 131-156-h. The test period
is used once, after all choices (features, hyper-parameters, model selection) were fixed on training / rolling-origin folds / validation.

**Preprocessing statistics** (imputation medians, scaling, one-hot categories, volatility thresholds) are fitted on the training window of each fit only (inside sklearn Pipelines);
volatility regimes use thresholds computed from origins <= the end of the fitting window.

**Proofs in the test-suite** (`tests/test_no_leakage.py`): (1) removing all data after week T0 leaves every feature of origins <= T0 unchanged (handled and raw pipelines); (2) corrupting every
observation after T0 leaves earlier features unchanged; (3) a late-delivered record is invisible at its own origin and visible one week later; (4) training rows never reach beyond the training window.
""")
print("docs written:", [p.name for p in DOCS.glob("*.md")])
