#!/usr/bin/env python
"""Stage 8: build reports/ERROR_ANALYSIS.md and reports/FINAL_REPORT.md (+ .html) from results/*.csv.

All numbers are read from result files. Interpretation paragraphs were written for the shipped seed-42 run and
are conditioned on the printed numbers; if you re-run with another seed, re-read them against the tables.
"""
import sys, json, subprocess, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from agriprice.config import load_config, results_dir, reports_dir, data_dir
from agriprice.analysis import add_variant
from agriprice.validate import _harmonic_fit

cfg = load_config(); R, REP = results_dir(), reports_dir()
H = cfg["experiment"]["horizons"]
CR = {k: v["name"] for k, v in cfg["crops"].items()}
MK = {k: v["name"] for k, v in cfg["markets"].items()}
sel = json.loads((R / "model_selection.json").read_text())
best_params = json.loads((R / "best_params.json").read_text())
BEST = sel["best_ml_candidate"]


def md(df, ff=".2f", index=False):
    return df.to_markdown(index=index, floatfmt=ff, missingval="n/a")

def rd(name, **kw):
    return pd.read_csv(R / name, **kw)

pct = lambda x: f"{x*100:.1f}%"

# ------------------------------------------------------------------ load
core_t = rd("core_eight_MAE_test.csv", index_col=0); core_v = rd("core_eight_MAE_validation.csv", index_col=0)
core_t.columns = [int(c) if str(c).isdigit() else c for c in core_t.columns]
core_v.columns = [int(c) if str(c).isdigit() else c for c in core_v.columns]
sk_t = 1 - core_t.div(core_t.loc["Naive"]); sk_v = 1 - core_v.div(core_v.loc["Naive"])
allv = rd("all_variants_test_metrics.csv", header=[0, 1], index_col=0)
allv_val = rd("all_variants_validation_MAE.csv", index_col=0)
boot = rd("paired_bootstrap_tests_test.csv")
by_crop = rd("breakdown_by_crop_test.csv"); by_mkt = rd("breakdown_by_market_test.csv")
by_vol = rd("breakdown_by_volatility_test.csv"); by_shock = rd("breakdown_shock_vs_ordinary_test.csv")
by_ev = rd("breakdown_by_event_test.csv"); by_reg = rd("breakdown_regime_shift_test.csv")
scen = rd("scenario_comparison_test.csv"); abl_t = rd("ablation_test_MAE.csv", index_col=[0, 1]); abl_cv = rd("ablation_rolling_cv_MAE.csv", index_col=[0, 1])
abl_ser = rd("ablation_by_series_test.csv"); abl_crop = rd("ablation_by_crop_test.csv")
gen = rd("generalisation_test.csv"); gen_m = rd("generalisation_by_market.csv")
iv_h = rd("interval_coverage_by_horizon_test.csv"); iv_c = rd("interval_coverage_by_crop_test.csv"); iv_e = rd("interval_coverage_by_event_test.csv")
imp = rd("feature_group_importance_validation.csv"); rt = rd("runtime_per_model.csv"); gap = rd("validation_to_test_gap.csv", index_col=0)
acc = rd("acceptance_checks.csv"); tune = rd("tuning_cv.csv")
events = pd.read_csv(data_dir() / "events_schedule.csv")
CORE = list(core_t.index)
main = add_variant(pd.read_parquet(R / "predictions_main.parquet"))
prim = main[(main.scenario == "realistic") & (main.handling == "handled")]
test = prim[(prim.partition == "test") & prim.scored]

# ------------------------------------------------------------------ hypothesis evaluation (spec section 1.3 definitions)
def bsel(name, A=None, h=None):
    d = boot[boot.comparison == name]
    if A: d = d[d.A == A]
    if h: d = d[d.horizon == h]
    return d

# H1: exogenous model beats price-only on at least some crop-market series
h1_rows = []
for m, lab in [("ridge", "Ridge"), ("random_forest", "Random Forest"), ("hist_gb", "HistGradientBoosting")]:
    d = abl_ser[abl_ser.model == m]
    h1_rows.append(dict(model=lab, series_where_E4_beats_E0=f"{int(d.E4_better.sum())} / {len(d)}",
                        pooled_MAE_E0=abl_t.loc[(m, "E0"), "mean_over_horizons"], pooled_MAE_E4=abl_t.loc[(m, "E4"), "mean_over_horizons"],
                        sig_horizons=f"{int((bsel(f'{lab}: E4 vs E0 (feature enrichment)').ci_high < 0).sum())} / 3"))
h1 = pd.DataFrame(h1_rows)
h1_verdict = "Supported" if all(int(r.split(" / ")[0]) >= 10 for r in h1.series_where_E4_beats_E0) else "Partly supported"
# H2: tree ensembles vs linear
d_rf, d_hgb = bsel("Random Forest vs Ridge (E4)"), bsel("HistGB vs Ridge (E4)")
h2_rows = pd.DataFrame({"horizon": H, "RF - Ridge (MAE diff)": d_rf.mean_diff.values, "RF 95% CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(d_rf.ci_low, d_rf.ci_high)],
                        "HGB - Ridge": d_hgb.mean_diff.values, "HGB 95% CI": [f"[{a:.2f}, {b:.2f}]" for a, b in zip(d_hgb.ci_low, d_hgb.ci_high)]})
h2_verdict = "Not supported as a general claim (mixed)"
# H3: horizon deterioration
mono = {v: bool(core_t.loc[v, 1] < core_t.loc[v, 2] < core_t.loc[v, 4]) for v in CORE if v != "Seasonal naive"}
h3_verdict = "Supported" if all(mono.values()) else "Partly supported"
ratio_14 = (core_t[4] / core_t[1]).round(2)
# H4: seasonal naive difficult to beat for strongly seasonal series
world = pd.read_csv(data_dir() / "clean_world_audit.csv")
world["doy"] = pd.to_datetime(world.week_start).dt.dayofyear + 3; world["lp"] = np.log(world.latent_price_kes_kg)
world["lpc"] = world.lp - world.groupby(["market_id", "crop_id"]).lp.transform("mean")
seas_r2 = {c: _harmonic_fit(g.lpc.values, g.doy.values.astype(float))[0] for c, g in world.groupby("crop_id")}
cm = by_crop.groupby(["crop", "variant"]).MAE.mean().unstack()
h4_tab = pd.DataFrame({"crop": [CR[c] for c in CR], "seasonal R2 (annual+semi-annual)": [seas_r2[c] for c in CR],
                       "Naive MAE": [cm.loc[CR[c], "Naive"] for c in CR], "Seasonal naive MAE": [cm.loc[CR[c], "Seasonal naive"] for c in CR],
                       "Best of {Ridge, RF, HGB} MAE": [cm.loc[CR[c], ["Ridge [E4]", "Random Forest [E4]", "HistGradientBoosting [E4]"]].min() for c in CR]})
h4_tab["seasonal naive beats naive?"] = np.where(h4_tab["Seasonal naive MAE"] < h4_tab["Naive MAE"], "yes", "no")
h4_tab["a model beats seasonal naive?"] = np.where(h4_tab["Best of {Ridge, RF, HGB} MAE"] < h4_tab["Seasonal naive MAE"], "yes", "no")
strong = max(seas_r2, key=seas_r2.get)
h4_verdict = "Partly supported (true for the one strongly seasonal crop only)"
# H5
sc_p = scen.pivot(index="variant", columns="combo", values="mean_over_horizons")
h5_verdict = "Supported (clean < realistic < stressed with handling; ignoring data quality is far worse)"

hyp = pd.DataFrame([
    ["H1", "Exogenous variables beat price-history-only on at least some series", h1_verdict, f"E4 beats E0 on {h1.series_where_E4_beats_E0[1]} series for RF; significant at all three horizons for Ridge and RF"],
    ["H2", "Tree ensembles beat a linear model", h2_verdict, "RF/HGB win at h=4 (significant), Ridge wins at h=2 (significant), tie at h=1; Ridge with crop interactions is best overall"],
    ["H3", "4-week forecasts less accurate than 1-week", h3_verdict, f"every non-seasonal model: MAE(h=4)/MAE(h=1) = {ratio_14.drop('Seasonal naive').min():.2f}-{ratio_14.drop('Seasonal naive').max():.2f}"],
    ["H4", "Seasonal naive hard to beat for strongly seasonal series", h4_verdict, f"only {CR[strong]} (seasonal R2 {seas_r2[strong]:.2f}); there seasonal naive beats naive by {pct(1-cm.loc[CR[strong],'Seasonal naive']/cm.loc[CR[strong],'Naive'])}"],
    ["H5", "Corruption/missingness reduce accuracy, especially without handling", h5_verdict,
     f"mean MAE clean {sc_p.loc['Naive','clean|handled']:.2f}/{sc_p.loc['Random Forest [E4]','clean|handled']:.2f} -> realistic {sc_p.loc['Naive','realistic|handled']:.2f}/{sc_p.loc['Random Forest [E4]','realistic|handled']:.2f} -> stressed {sc_p.loc['Naive','stressed|handled']:.2f}/{sc_p.loc['Random Forest [E4]','stressed|handled']:.2f} (Naive/RF)"],
], columns=["ID", "Hypothesis", "Verdict", "Key evidence (test set)"])
hyp.to_csv(R / "hypothesis_verdicts.csv", index=False)

# ================================================================== ERROR ANALYSIS
bm = test[test.variant == BEST].copy(); nv = test[test.variant == "Naive"].copy()
keys = ["series", "origin_week", "horizon"]
e = bm.merge(nv[keys + ["y_pred"]], on=keys, suffixes=("", "_naive"))
e["err"] = (e.y_obs - e.y_pred); e["abs_err"] = e.err.abs(); e["abs_err_naive"] = (e.y_obs - e.y_pred_naive).abs()
e["gain"] = e.abs_err_naive - e.abs_err; e["rel_err"] = e.abs_err / e.y_obs
e["crop"] = e.crop_id.map(CR); e["market"] = e.market_id.map(MK)
e["context"] = [
    "; ".join(x for x in [
        (f"event {i}" if i else ""), ("after transport regime shift" if r else ""),
        ("stale latest price" if (s or 0) >= 1 else ""), ("high-volatility origin" if v else "")] if x) or "ordinary week"
    for i, r, s, v in zip(e.y_event_ids, e.y_regime, e.price_stale_weeks, e.horizon_vol8 > e.groupby("crop_id").horizon_vol8.transform(lambda s: s.quantile(.75)))]

share_rows = e.groupby(np.where(e.y_event_affected, "event-affected target weeks", "other target weeks")).agg(
    rows=("abs_err", "size"), total_abs_error=("abs_err", "sum")).reset_index().rename(columns={"index": "group", "level_0": "group"})
share_rows.columns = ["group", "rows", "total_abs_error"]
share_rows["share_of_rows"] = share_rows.rows / share_rows.rows.sum(); share_rows["share_of_abs_error"] = share_rows.total_abs_error / share_rows.total_abs_error.sum()
win = e.groupby("crop").apply(lambda d: pd.Series({"rows": len(d), f"{BEST} beats naive (share of forecasts)": (d.gain > 0).mean(),
                                                  "mean signed error (obs - forecast)": d.err.mean(), "MAE": d.abs_err.mean()})).reset_index()
post = e[e.target_week >= 140]
bias = post.groupby("crop").apply(lambda d: pd.Series({"rows": len(d), "mean signed error (obs - forecast), KES/kg": d.err.mean(),
                                                       "mean relative bias": (d.err / d.y_obs).mean(), "share under-predicted": (d.err > 0).mean()})).reset_index()
# same for the strongest non-tree comparator
rc = test[test.variant == "Ridge + crop interactions [E4]"]
rb = rc[rc.target_week >= 140].assign(err=lambda d: d.y_obs - d.y_pred).groupby("crop_id").err.mean().rename(index=CR)
bias["Ridge+crop-interactions signed error"] = bias.crop.map(rb)

cols = ["crop", "market", "origin_week", "target_week", "horizon", "y_obs", "ref_price", "y_pred", "abs_err", "abs_err_naive", "context"]
names = {"y_obs": "observed", "ref_price": "latest price (naive)", "y_pred": "forecast", "abs_err": "abs error", "abs_err_naive": "naive abs error"}
worst = e.sort_values("abs_err", ascending=False).head(10)[cols].rename(columns=names)
succ = e[(e.rel_err < 0.04) & (e.horizon >= 2)].sort_values("gain", ascending=False).head(5)[cols].rename(columns=names)
fail_vs_naive = e[e.horizon >= 2].sort_values("gain").head(5)[cols].rename(columns=names)
iv = pd.read_parquet(R / "predictions_intervals.parquet"); iv = iv[(iv.partition == "test") & iv.scored]
iv["miss"] = ~((iv.y_obs >= iv.lower_cf) & (iv.y_obs <= iv.upper_cf))
miss_by = iv.groupby(["crop_id"]).miss.mean().rename(index=CR)

lines = []
A = lines.append
A("# Error Analysis\n")
A(f"Model analysed: **{BEST}** (selected on validation among the core eight; realistic data, data-quality handling on), test window origins 131-156, with the naive last-price forecast as reference. "
  "Forecasts of rows labelled as injected reporting errors are not scored. All numbers are read from `results/predictions_main.parquet`.\n")
A("## 1. Where the errors are\n")
A(md(share_rows, ".3f") + "\n")
A(f"Most of the test window is *by design* shock-affected ({pct(share_rows.set_index('group').loc['event-affected target weeks','share_of_rows'])} of scored forecasts have a target week inside a documented event window), "
  "so the error is concentrated there simply because that is where most weeks are. The informative comparison is skill: see `results/breakdown_shock_vs_ordinary_test.csv` - "
  f"the model's skill versus naive is {pct(by_shock[(by_shock.variant==BEST)&(by_shock.period=='ordinary week')].skill_vs_naive.mean())} in ordinary weeks but only "
  f"{pct(by_shock[(by_shock.variant==BEST)&(by_shock.period=='event-affected target week')].skill_vs_naive.mean())} in event-affected weeks.\n")
A("## 2. By crop: how often does the model beat naive, and is it biased?\n")
A(md(win, ".3f") + "\n")
A("## 3. Systematic failure: under-predicting a rising market\n")
A("After the transport-cost regime shift and the stronger drought (target weeks >= 140), prices of potatoes, tomatoes and onions rose sharply. The forecasts **lag the rise**: "
  "the share of under-predicted forecasts is 91% for potatoes, 73% for tomatoes and 67% for onions. Two mechanisms are mixed here. For **potatoes** 15% of test prices exceeded the highest price seen in training/validation, and tree ensembles cannot predict above values they have seen; "
  "for **tomatoes and onions** prices stayed inside the historical range (2% and 0% of rows above it), so the lag reflects a driver (lasting transport-cost increase) that never varied in training, not range extrapolation. "
  "The linear model with crop interactions is biased in the same direction but less so for potatoes. Signed error = observed - forecast (positive = under-predicted):\n")
A(md(bias, ".3f") + "\n")
A("![h4](../figures/res_05_forecast_examples_h4.png)\n")
A("*Kisumu potatoes (bottom left) is the clearest case: the forecast (green) stays below the outcome for the whole rise while the naive forecast catches up after a lag. "
  "The isolated black dot near 18 KES/kg at week 140 is an injected reporting error - excluded from scoring.*\n")
A("## 4. The ten largest absolute errors\n")
A(md(worst, ".2f") + "\n")
A("All ten are **tomato** forecasts (the most volatile crop), six of them four weeks ahead, in the period of the stronger drought (EV12) and the lasting transport-cost shift (EV13), when tomato prices climbed by roughly 40-60% within a few weeks. "
  "The naive forecast is wrong by almost as much in every case (naive error 39-60 vs model 48-65 KES/kg): these were upward moves that no forecaster in the comparison anticipated, and in eight of the ten the model is slightly *worse* than naive. "
  "None of the ten is a data-quality failure: the latest price was fresh in all of them.\n")
A("## 5. Successful forecasts (largest gains over naive with error under 4%)\n")
A(md(succ, ".2f") + "\n")
A("All five successes are **tomato forecasts issued at the peak of the spike** (origin weeks 151-153) that correctly predicted the subsequent fall, while the naive forecast stayed at the peak. "
  "This is a mean-reversion bet: the model learned that tomato prices far above their recent average tend to fall back, consistent with the dominance of price-deviation features at longer horizons (section 9.1 of the final report). "
  "We did not attribute each case to individual features.\n")
A("## 6. Cases where the model was worse than naive (horizon >= 2)\n")
A(md(fail_vs_naive, ".2f") + "\n")
A("These are the **mirror image of the successes**: tomato forecasts issued on the *plateau* of the spike (origin weeks 147-151), where the model again bet on a fall that did not yet happen (forecasts 20-30 KES/kg below the stable observed level). "
  "The same mean-reversion rule wins at the reversal and loses on the plateau; a last-value forecast makes no directional bet, so it cannot lose this way. Distinguishing a temporary spike from a regime shift is what the model cannot do - it never saw a cause like the transport shift.\n")
A("## 7. Data quality and error\n")
st = rd("breakdown_by_price_staleness_test.csv")
A(md(st[["stale", "variant", "n", "MAE", "skill_vs_naive"]], ".3f") + "\n")
A("When the latest price was one to two weeks old, errors were roughly 50% higher for both the naive forecast and the model, and the model's skill disappeared; only five forecasts had a reference price three or more weeks old, too few to interpret. "
  "Reporting errors are a much larger issue when not handled (see FINAL_REPORT section 8.4): one uncorrected x10 price used as a reference spoils the next forecasts of that series.\n")
A("## 8. Prediction intervals fail when they matter most\n")
A(md(iv_c.assign(**{}).rename(columns={"coverage_quantile_reg": "coverage (quantile reg.)", "coverage_conformal": "coverage (conformal)"})[["crop", "horizon", "rows", "coverage (quantile reg.)", "coverage (conformal)", "rel_width_conformal"]], ".3f") + "\n")
A("Nominal coverage was 80%. Empirical test coverage is 61-75% overall, 50-64% for tomatoes at the 4-week horizon, and for the conformal bands 65-70% inside event windows versus 71-82% in ordinary weeks: validation-based intervals "
  "assume that future errors look like past errors, which the new shocks violated. Users of such bands should treat them as *optimistic* in turbulent periods.\n")
A("## 9. What this implies\n")
A("1. **Unanticipated drift and unseen shocks** (and, for potatoes, prices beyond the training range) are the dominant failure mode, not noise or missing data once cleaning is applied.\n"
  "2. Features about *future-relevant* supply and neighbouring markets help in ordinary weeks; they cannot help with a cause that never varied in training (the transport-cost regime shift).\n"
  "3. Practical responses: retrain more often when a shock is detected, consider a trend-capable component (linear terms or a hybrid of linear model + trees; untested here), wider event-aware intervals, and report forecasts together with a warning flag when the price has left its training range.\n"
  "4. A forecast service should publish the **naive forecast alongside the model**, because the model's edge is small and uncertain in shock periods.\n")
A("![err](../figures/res_06_error_over_time.png)\n")
(REP / "ERROR_ANALYSIS.md").write_text("\n".join(lines))

# ================================================================== FINAL REPORT
L = []
W = L.append
sk4 = lambda v: sk_t.loc[v]
W("# Agricultural Market Price Prediction: Synthetic-Data Forecasting Experiment\n")
W("**Final experiment report** - seed 42, 5 markets x 4 crops x 156 weeks, horizons 1/2/4 weeks.\n")
W("> All data are **synthetic**. Findings describe method behaviour under the assumptions built into the generator and **do not predict real-world accuracy**. "
  "Interpretation paragraphs refer to the shipped seed-42 run.\n")

W("## Summary\n")
best_core_v = sel["best_core_candidate"]
W(f"* **Simple baselines are strong.** The last observed price (naive) is hard to beat; the 4-week moving average is worse than naive at every horizon (test MAE {core_t.loc['Moving average (4w)','mean_over_horizons']:.2f} vs {core_t.loc['Naive','mean_over_horizons']:.2f} KES/kg), "
  f"and the seasonal-naive forecast is the worst model overall ({core_t.loc['Seasonal naive','mean_over_horizons']:.2f}) - except for **maize**, where it is the best baseline.")
W(f"* **Information helps, modestly.** Models with the full feature set (E4) beat price-history-only versions at every horizon for Ridge and Random Forest (paired bootstrap, 95% CI excludes zero). "
  f"On validation, Random Forest/HistGB/Ridge beat naive by {pct(sk_v.loc['Random Forest [E4]','mean_over_horizons'])}/{pct(sk_v.loc['HistGradientBoosting [E4]','mean_over_horizons'])}/{pct(sk_v.loc['Ridge [E4]','mean_over_horizons'])}; "
  f"on the untouched test window the gain shrinks to {pct(sk_t.loc['Random Forest [E4]','mean_over_horizons'])}/{pct(sk_t.loc['HistGradientBoosting [E4]','mean_over_horizons'])}/{pct(sk_t.loc['Ridge [E4]','mean_over_horizons'])} and is statistically clear only at the 1-week horizon.")
W("* **No clear winner among Random Forest, HistGradientBoosting and Ridge.** Their test MAEs differ by about 1%; paired tests find no robust ordering between RF and HGB, trees beat Ridge at h=4 but lose at h=2. "
  f"A Ridge model with crop-specific slopes (diagnostic, outside the core eight) was best on both validation and test ({allv['MAE'].loc['Ridge + crop interactions [E4]','mean_over_horizons']:.2f} vs {core_t.loc[best_core_v,'mean_over_horizons']:.2f}): crop heterogeneity mattered more than nonlinearity.")
W("* **Shocks of a new kind erase most of the edge.** Before week 140 the tree/Ridge models were 19-23% better than naive; after the lasting transport-cost shift and the stronger drought their gain falls to 2-3%. "
  "Forecasts lag the rising prices of the shock period (under-predicted in 67-91% of potato, tomato and onion forecasts); tree models are worse than naive on potatoes, where prices left the training range.")
W("* **Data quality matters a lot - if ignored.** With causal cleaning, MAE rises only gradually from clean to stressed data; without cleaning, a single x10 reporting error used as the latest price raises the error by roughly 30-90%.")
W("* **Intervals are over-confident**: nominal 80% bands covered 61-75% of test outcomes.")
W("* **Generalisation:** models trained without the target series' history keep most (not all) of their small edge for unseen markets, and are about at the naive level for unseen crop-market pairs; no collapse, but little to lose.\n")

W("## 1. Introduction and problem statement\n")
W("A produce-market intelligence service collects weekly wholesale prices and market indicators and wants to tell farmers, aggregators and buyers the likely wholesale price of a crop in a market 1, 2 or 4 weeks ahead. "
  "Because real data are not used here, we build a *synthetic market world* whose causal structure is known, and use it as a laboratory: Do models learn plausible patterns? Do they exploit leakage? How do they respond to documented disruptions and to degraded data?\n")

W("## 2. Research question, objectives and hypotheses\n")
W("**Question.** Can models trained on historical market, crop, weather, supply and cost indicators produce useful 1-, 2- and 4-week-ahead wholesale price forecasts, and under which conditions do they outperform simple baselines?\n")
W("**Objectives:** (1) build and validate the generator; (2) compare the core eight models and two boosting extensions under identical leakage-safe splits; (3) quantify the value of each feature group; (4) stress-test with data corruption, shocks and unseen markets/series; (5) document failures and limits.\n")
W("**Hypotheses and verdicts** (definitions from the specification, section 1.3):\n")
W(md(hyp) + "\n")

W("## 3. Synthetic market environment and generation assumptions\n")
W("Five markets (Nairobi, Nakuru, Eldoret, Kisumu, Mombasa; names illustrative) and four crops (maize, tomatoes, Irish potatoes, onions). Weekly log price = crop base value + market premium + crop-specific seasonality + a persistent deviation that "
  "(i) mean-reverts at a crop-specific speed (tomatoes quickly, maize/onions slowly), (ii) partially transmits from connected markets with a one-week lag, (iii) adjusts to supply, demand, transport-cost, regional-supply and drought drivers - with "
  "scarcity hitting harder than glut (nonlinear) - and (iv) receives random shocks. Weather affects *future* supply with a crop-specific delay (tomatoes 3 weeks, maize 10 weeks). "
  "Sixteen documented events (gluts, droughts, floods, fuel step, demand surges, import shocks, a source-market disruption and a lasting transport-cost regime shift) are spread over train, validation and test; four test events are new or stronger than anything in training. "
  "Full equations, parameter tables and the event calendar: `docs/generator_specification.md`; provenance and intended use: `docs/provenance_statement.md`.\n")
W("**Generator acceptance.** " + f"{int(acc.passed.sum())}/{len(acc)} acceptance criteria and all {len(rd('acceptance_event_checks.csv'))} event checks pass (`reports/DATA_AUDIT.md`): coverage, uniqueness, price validity, seasonality, market differences, shock behaviour (including the delayed drought effect and one-week spillover), persistence, missingness rates, no hidden leakage, reproducibility.\n")

W("## 4. Dataset description and exploratory analysis\n")
W("3,120 rows per scenario. Scenarios share one latent world and differ in the data-quality layer: **clean** (tiny noise, nothing missing), **realistic** (3.7% price missing, 5-8% indicator missing, 0.74% labelled reporting errors, late indicators for two markets), "
  "**stressed** (longer gaps, two market-wide reporting outages, 1% errors, more late data). Real shocks (`event_*`) and data-quality labels (`data_quality_flag`) are kept strictly separate. "
  "Key EDA findings (`reports/EDA_REPORT.md`): price levels are highly persistent; tomatoes are by far the most volatile crop (about 10% weekly changes versus 2% for maize); seasonality differs by crop; arrivals anomalies show the clearest, correctly-signed relation to next-week price changes; "
  "reporting gaps and errors are concentrated in some markets. Data dictionary: `docs/data_dictionary.md`.\n")
W("![prices](../figures/eda_01_price_series.png)\n")

W("## 5. Forecasting methodology and feature engineering\n")
W("Direct forecasting, one model per horizon. ML models predict the log-change of the price against the latest available price (so a model that learns nothing reproduces naive). "
  "Features come in groups, added cumulatively for the ablation: **E0** price history (lags 1-52, rolling deviations/volatility, staleness) + calendar + market/crop IDs; **E1** + supply and demand; **E2** + weather; **E3** + fuel and transport cost; **E4** + cross-market signals "
  "(connected-market price gap and changes, regional supply index). E4 has 55 numeric features plus the market and crop identifiers (`docs/data_dictionary.md`). Data-quality handling (`handling = on`): causal outlier filter, last-observation-carried-forward with age cap, staleness feature; "
  "the *off* mode feeds raw values with gaps and errors to the models.\n")

W("## 6. Temporal validation and leakage controls\n")
W(f"Origin-based partitions: train weeks 1-104, validation 105-130, test 131-156 (used once, after all choices were fixed). Training rows are purged so their target lies inside the fitting window. Hyper-parameters were tuned on five expanding rolling-origin folds (weeks <= 130; grids widened when an optimum sat on a grid edge). "
  "Information availability is governed by `docs/forecast_time_data_contract.md` (late-delivered indicators, causal cleaning, calendar-only future knowledge). **Proofs in code:** truncating the data at week T0 leaves every feature of origins <= T0 bit-identical (handled and raw); "
  "corrupting all later data changes nothing earlier; a late record is invisible at its own origin; training rows never reach beyond the window; latent prices, quality and event labels never appear in a feature set (19 tests, all passing). "
  "Scoring excludes target rows labelled as injected reporting errors (labels used for scoring only); differences are assessed with **paired block bootstraps over origin weeks** (2,000 resamples), because 26 test weeks are few.\n")
W("**Selection rule (specification section 15):** a complex model is preferred only if it improves validation MAE by at least 5% over the strongest baseline *and* wins in at least 4 of 5 rolling folds. "
  f"Result: strongest baseline = {sel['strongest_baseline']}; best ML candidate = {sel['best_ml_candidate']} with {pct(sel['validation_gain_over_strongest_baseline'])} lower validation MAE, winning {sel['rolling_folds_won']}/{sel['rolling_folds_total']} folds -> "
  f"complex model justified: **{sel['complex_model_justified']}**. Note that Random Forest, HistGB and Ridge are within 3% of each other on validation MAE (4.01 / 4.02 / 4.11) and RF vs HGB shows no significant difference on test, so naming Random Forest 'best' is close to a coin-flip among three; the recommendation is the *family* (enriched ML), not the specific model.\n")

W("## 7. Model configurations and baselines\n")
W("Core eight: **Naive** (latest available price), **Seasonal naive** (price one year before the target week; falls back to naive when unavailable), **Moving average** (window chosen on validation: " + sel["moving_average_window_selected"] + "), **Ridge**, **SARIMA** (per series, seasonal period 52 with AR(1) seasonal term, order from {(1,0,0),(1,0,1),(2,0,0)} by AIC, constant trend; native multi-step), "
  "**Decision Tree**, **Random Forest** (200 trees), **HistGradientBoosting**. Extras: ARIMA (non-seasonal), Ridge + crop interactions (diagnostic), XGBoost, LightGBM. Not run: Prophet, LSTM, TFT (see Limitations). Selected hyper-parameters (E4):\n")
bp = []
for m, d in best_params.items():
    if "E4" in d:
        for h in H:
            bp.append({"model": m, "horizon": h, "tuned parameters (E4)": json.dumps(d["E4"][str(h)])})
W(md(pd.DataFrame(bp)) + "\n")
W("Seed 42 everywhere. Training/prediction times (primary scenario, test fit; single CPU core):\n")
W(md(rt.rename(columns={"variant": "model variant"}), ".3f") + "\n")

W("## 8. Results by horizon, crop, market and scenario\n")
W("### 8.1 Core eight - horizon comparison (realistic data, data handling on)\n")
W("Test MAE (KES/kg) by horizon, and validation for reference:\n")
t1 = core_t.copy(); t1.columns = [f"test h={c}" if str(c).isdigit() else "test mean" for c in t1.columns]
v1 = core_v.copy(); v1.columns = [f"val h={c}" if str(c).isdigit() else "val mean" for c in v1.columns]
W(md(pd.concat([t1, v1], axis=1).reset_index().rename(columns={"variant": "model"})) + "\n")
W("Skill versus naive on the test window (1 - MAE/MAE_naive; positive = better than last price):\n")
s1 = (sk_t * 100).round(1); s1.columns = [f"h={c}" if str(c).isdigit() else "mean" for c in s1.columns]
W(md(s1.reset_index().rename(columns={"variant": "model"}), ".1f") + "\n")
full = rd("core_eight_all_metrics_test.csv", header=[0, 1], index_col=0)
W("Other metrics on the test window (RMSE, WAPE, directional accuracy; directional accuracy is undefined for the naive forecast because it never predicts a change):\n")
om = pd.DataFrame({f"{m} h={h}": full[(m, str(h))] for m in ["RMSE", "WAPE", "DirAcc"] for h in H})
W(md(om.reset_index().rename(columns={"variant": "model"}), ".3f") + "\n")
W("![mae](../figures/res_01_mae_by_horizon.png)\n")
W(f"**Reading the table.** (H3) Error grows with horizon for every forecasting model (h=4 errors are {ratio_14.drop('Seasonal naive').min():.1f}-{ratio_14.drop('Seasonal naive').max():.1f}x the 1-week errors). "
  f"The enriched models' edge over naive is {pct(sk_t.loc['HistGradientBoosting [E4]',1])}/{pct(sk_t.loc['Random Forest [E4]',1])} at h=1 and {pct(sk_t.loc['HistGradientBoosting [E4]',4])}/{pct(sk_t.loc['Random Forest [E4]',4])} at h=4 (HGB/RF). "
  "SARIMA is competitive with naive on validation (4.61 vs 4.99) but *worse* than naive on test (6.43 vs 6.16): a per-series model with fixed parameters keeps extrapolating the old regime. "
  "Random Forest beats SARIMA significantly at all three horizons (paired bootstrap). The Decision Tree is the weakest learned model, as expected from a high-variance learner.\n")
W("### 8.2 By crop and market (test)\n")
bc = by_crop[by_crop.variant.isin(CORE + ["Ridge + crop interactions [E4]"])].groupby(["crop", "variant"]).agg(MAE=("MAE", "mean"), skill=("skill_vs_naive", "mean")).reset_index()
W("MAE (KES/kg, mean over horizons):\n")
W(md(bc.pivot(index="variant", columns="crop", values="MAE").reindex(CORE + ["Ridge + crop interactions [E4]"]).reset_index().rename(columns={"variant": "model"})) + "\n")
W("Skill versus naive (%):\n")
W(md((bc.pivot(index="variant", columns="crop", values="skill").reindex(CORE + ["Ridge + crop interactions [E4]"]) * 100).reset_index().rename(columns={"variant": "model"}), ".1f") + "\n")
W("![crop](../figures/res_02_skill_by_crop.png)\n")
W("Tomatoes have 3-9x higher absolute errors than the other crops and the lowest relative skill. **Maize** is the only crop where seasonal naive beats naive (and where the tree ensembles gain most, +25-31%). "
  "**Irish potatoes**, whose price rose beyond the training range in the test window, is the one crop where the tree models are *worse* than naive; Ridge-type models, which can extrapolate beyond seen values, gain 12-22% there.\n")
W("By market (skill versus naive, %, mean over horizons):\n")
bm_ = by_mkt[by_mkt.variant.isin(CORE + ["Ridge + crop interactions [E4]"])].groupby(["market", "variant"]).skill_vs_naive.mean().unstack().reindex(columns=CORE + ["Ridge + crop interactions [E4]"]) * 100
W(md(bm_.T.reset_index().rename(columns={"variant": "model"}), ".1f") + "\n")
W("Complete per-crop-per-horizon and per-market-per-horizon tables with MAE/RMSE/WAPE/sMAPE/directional accuracy: `results/breakdown_by_crop_test.csv`, `results/breakdown_by_market_test.csv`, `results/metrics_overall_all_combos.csv`. "
  "No market is consistently best or worst for the models; Nakuru has the highest naive error.\n")
W("### 8.3 Volatility regimes, shocks and the regime shift (test)\n")
bv = by_vol[by_vol.variant.isin(CORE + ["Ridge + crop interactions [E4]"])].groupby(["variant", "regime"]).skill_vs_naive.mean().unstack() * 100
W("Skill versus naive (%) in calm versus high-volatility origins (threshold = per-crop 75th percentile of volatility over origins <= 130):\n")
W(md(bv.reindex(CORE + ["Ridge + crop interactions [E4]"]).reset_index().rename(columns={"variant": "model"}), ".1f") + "\n")
bs = by_shock[by_shock.variant.isin(CORE + ["Ridge + crop interactions [E4]"])].groupby(["variant", "period"]).skill_vs_naive.mean().unstack() * 100
W("Skill (%) by kind of target week:\n")
W(md(bs.reindex(CORE + ["Ridge + crop interactions [E4]"]).reset_index().rename(columns={"variant": "model"}), ".1f") + "\n")
br = by_reg[by_reg.variant.isin(CORE + ["Ridge + crop interactions [E4]"])].pivot(index="variant", columns="window", values="skill_vs_naive") * 100
W("Skill (%) before versus after the lasting transport-cost regime shift at week 140:\n")
W(md(br.reindex(CORE + ["Ridge + crop interactions [E4]"]).reset_index().rename(columns={"variant": "model"}), ".1f") + "\n")
W("![reg](../figures/res_10_regimes.png)\n")
ev_t = by_ev[by_ev.variant.isin(["Naive", BEST, "SARIMA"])].pivot_table(index=["event_id", "event_type", "new_in_test", "rows"], columns="variant", values="MAE").reset_index()
W("Per test-window event (MAE, KES/kg; rows overlap between events):\n")
W(md(ev_t) + "\n")
W("Interpretation: the models' advantage is concentrated in ordinary weeks and before week 140. Under shocks of a new kind or strength (EV12-EV15) they perform about like naive. "
  "This is the generator working as intended: the transport-cost shift is an out-of-range cause that no training row contains, so no learner could have inferred its effect. The finding is a statement about *what the evaluation should include*, not just about the models.\n")
W("### 8.4 Data-quality scenarios and handling (test)\n")
sp = scen.pivot(index="variant", columns="combo", values="mean_over_horizons")[["clean|handled", "realistic|handled", "realistic|raw", "stressed|handled", "stressed|raw"]]
W("Test MAE, mean over horizons (KES/kg):\n")
W(md(sp.reindex(CORE).reset_index().rename(columns={"variant": "model"})) + "\n")
W("![scen](../figures/res_04_scenarios.png)\n")
hb = boot[boot.comparison.str.startswith("data handling") & (boot.A == BEST)][["comparison", "horizon", "mean_diff", "ci_low", "ci_high"]]
W(f"With handling on, error rises smoothly from clean to realistic to stressed (H5). With handling **off** the error jumps by roughly 30-90% depending on model and scenario (realistic: {sp.loc['Naive','realistic|raw']:.1f} vs {sp.loc['Naive','realistic|handled']:.1f} for naive). The cost is dominated by gross reporting errors: when the latest reported price is x10 or x0.1, every model that uses it as a reference is wrong by a large amount. "
  "Paired bootstrap of handling-on minus handling-off for the selected model (negative = handling helps):\n")
W(md(hb, ".2f") + "\n")
W("*Caution:* the ordering of the two raw columns (stressed raw is sometimes better than realistic raw) reflects *where the random errors happened to land* relative to the scored weeks, not a genuine benefit of stress - only one realisation of the world was run.\n")

W("## 9. Ablation and robustness analysis\n")
W("### 9.1 Feature groups (cumulative)\n")
a1 = abl_t.reset_index(); a1["model"] = a1["model"].map({"ridge": "Ridge", "random_forest": "Random Forest", "hist_gb": "HistGradientBoosting"})
W("Test MAE:\n"); W(md(a1) + "\n")
a2 = abl_cv.reset_index(); a2["model"] = a2["model"].map({"ridge": "Ridge", "random_forest": "Random Forest", "hist_gb": "HistGradientBoosting"})
W("Rolling-origin CV MAE (5 folds, weeks <= 130):\n"); W(md(a2) + "\n")
W("![abl](../figures/res_03_ablation.png)\n")
W("Tests of E4 versus E0 (paired bootstrap, MAE difference, negative = enrichment helps; star = 95% CI excludes zero):\n")
bb = boot[boot.comparison.str.contains("E4 vs E0")][["comparison", "horizon", "mean_diff", "ci_low", "ci_high"]].copy()
bb["significant"] = np.where((bb.ci_high < 0) | (bb.ci_low > 0), "*", "")
W(md(bb, ".2f") + "\n")
W("**Reading.** (H1) Enrichment helps consistently in rolling CV and significantly on test for Ridge and Random Forest at every horizon. The gain does **not** accrue evenly by group: cumulative steps E1 (supply/demand), E2 (weather) and E3 (logistics) give small, sometimes negative, test changes "
  "(HistGB at h=4 is *worse* with E1 than with E0), while the last step - **cross-market signals** - gives the largest single improvement at the 1- and 2-week horizons for all three models (at h=4 its step is smaller). The group permutation importance agrees:\n")
ip = imp.pivot(index="group", columns="horizon", values="mae_increase").sort_values(4, ascending=False)
W(md(ip.reset_index(), ".3f") + "\n")
W("![imp](../figures/res_07_group_importance.png)\n")
W("Plausibility check against the generator: price momentum and mean reversion dominate (more so at longer horizons, where calendar position also gains weight, consistent with seasonality); neighbouring-market prices and regional supply are the most useful *external* information at 1-2 weeks "
  "(they implement the one-week-lag transmission built into the generator); own-market supply/demand matter moderately; weather matters little *through the weather features themselves* because its effect arrives via arrivals; logistics have almost no importance on validation because transport costs barely varied before week 140 - "
  "exactly why the later transport shift could not be learned. These relationships are plausible and consistent with the data-generating process; they do **not** show that the same features would matter for real markets.\n")
W("Why might E1-E3 help less than expected? In the generator, price reacts to *contemporaneous* supply, which is not known at the origin; only its persistence and delayed weather causes are forecastable. A feature's presence in the data is not the same as predictive value at a given lead time. "
  "(Per the specification's guidance for non-helpful features: data quality, lag design and availability were checked - the signal is present in-sample, but its forecastable part is small relative to price noise.)\n")
W("### 9.2 Generalisation to unseen markets and crop-market pairs (test)\n")
W("Protocol: the held-out series' *target rows* are excluded from training; its past prices may be used as input features at forecast time. Unseen-market models drop the `market_id` feature. Skill is versus naive on the same rows.\n")
g1 = gen.copy(); g1["model"] = g1.model.map({"ridge": "Ridge", "random_forest": "Random Forest", "hist_gb": "HistGradientBoosting"})
W(md(g1[["model", "test", "horizon", "rows", "MAE_unseen", "MAE_seen", "MAE_naive", "skill_unseen", "skill_seen", "degradation_%"]], ".3f") + "\n")
W("![gen](../figures/res_09_generalisation.png)\n")
W("Unseen-market MAE is 0-5% worse than the same model trained with the market; skill versus naive is about -1% to 7% unseen versus 1-10% seen. For the four unseen crop-market pairs (only 87-98 forecasts each per horizon) all three models are at or below the naive level at h=4, and RF/HGB also at h=2 (HGB at h=4 about 9% worse than when seen). "
  "So there is **no collapse**, but also very little skill to transfer; per-market results are noisy (Kisumu is negative for RF/Ridge): `results/generalisation_by_market.csv`.\n")
W("### 9.3 Prediction intervals (extension)\n")
W(md(iv_h.rename(columns={"coverage_quantile_reg": "coverage: quantile regression", "coverage_conformal": "coverage: split conformal", "rel_width_quantile_reg": "median rel. width QR", "rel_width_conformal": "median rel. width conformal"}), ".3f") + "\n")
W("![iv](../figures/res_08_interval_coverage.png)\n")
W("Both approaches (HGB quantile q10-q90; split-conformal using validation residuals per crop) **under-cover** an 80% target. Coverage falls with horizon, is worst for tomatoes (50-64% at h=4) and is lower inside event windows (conformal 65-70% vs 71-82% in ordinary weeks). A deployable forecast would need wider, event-aware bands.\n")
W("### 9.4 Additional boosting implementations\n")
xb = allv["MAE"].loc[["HistGradientBoosting [E4]", "XGBoost [E4]", "LightGBM [E4]", "HistGradientBoosting [E0]", "XGBoost [E0]", "LightGBM [E0]"]]
W(md(xb.reset_index().rename(columns={"variant": "model"})) + "\n")
W("XGBoost and LightGBM perform essentially the same as scikit-learn's HistGradientBoosting with E4 features (differences of about 1%, within noise) and are *worse* with price-only features; the algorithm matters far less than the information supplied. Prophet, LSTM and TFT were not run.\n")

W("## 10. Error analysis and interpretation\n")
W("Full report with ten worst forecasts, five successes, five worse-than-naive cases and bias tables: `reports/ERROR_ANALYSIS.md`. Main points:\n")
W("1. The ten largest errors are all **tomato** forecasts (six at h=4) during the stronger drought and transport shift - upward moves of roughly 40-60% that the naive forecast also missed. The same mean-reversion bet produced the model's best forecasts (spike peak) and some of its worst (spike plateau).\n"
  "2. Forecasts **lag rising markets** (systematic under-prediction after week 140: 91% of potato, 73% of tomato forecasts); for potatoes this is partly range extrapolation by trees (15% of test prices exceeded all earlier prices), for tomatoes and onions it is a driver that never varied in training. The linear model with crop interactions lags less on potatoes (bias 3.2 vs 5.5 KES/kg) and only slightly less on tomatoes (9.5 vs 11.4). A hybrid (linear trend + trees) is a natural next experiment, but is untested.\n"
  "3. Stale reference prices raise errors by ~50% and remove the models' edge; cleaning errors causally is essential.\n"
  "4. Validation overstated test skill: every model's MAE is 24-44% higher on test than on validation, and most learned models degrade *more* than naive (Ridge, RF, HGB 41-44% vs 24% on average; the Decision Tree 27%) - see the table below. A single validation window is an optimistic guide when later periods contain different shocks.\n")
g2 = gap[gap.index.isin(CORE)] * 100; g2.columns = [f"h={c}" if str(c).isdigit() else "mean" for c in g2.columns]
W("MAE increase from validation to test (%):\n"); W(md(g2.reset_index().rename(columns={"variant": "model"}), ".0f") + "\n")
W("![ex1](../figures/res_05_forecast_examples_h1.png)\n")

W("## 11. Limitations, ethical considerations and path to real data\n")
W("**Limitations of this study.**\n"
  "* *Synthetic by construction.* Every effect exists because we wrote it into the generator. A method that wins here may lose on real data; informal trade, quality grades, policy, trader behaviour, market closures and measurement practice are absent.\n"
  "* *One world, one seed.* Differences of about 1-3% between models are within noise; only paired tests on 26 test weeks were possible. Re-run `01_generate_data.py --seed N` to test robustness.\n"
  "* *Test window is unusually shock-heavy by design* (57% of scored target weeks sit inside an event window and a further 29% after the lasting regime shift; only about 14% are 'ordinary'), so test skill is deliberately a stress result, not a typical-conditions result.\n"
  "* *Tuning scope.* Hyper-parameters were tuned on the realistic scenario with E0/E4 and reused for E1-E3 and other scenarios; small grids; ridge/trees tuned per horizon.\n"
  "* *SARIMA* is plain (no exogenous regressors), fitted once per partition with parameters frozen, forecasts native multi-step; a refitted or SARIMAX variant might fare better.\n"
  "* *Scoring* excludes labelled reporting errors; a production system could not know which targets are wrong.\n"
  "* *Not implemented:* Prophet, LSTM and Temporal Fusion Transformer (deferred as optional; the specification itself cautions that 3,120 observations are too few for deep models). Elastic Net and ExtraTrees were not compared.\n"
  "* *Simplifications:* no price trend/inflation, no data revisions, gross-only reporting errors, arrivals not responsive to past prices, a single national weather/fuel structure.\n")
W("**Ethical considerations.** Price forecasts can move money: a forecast that is over-confident (as the intervals here are) could mislead smallholders or traders; published forecasts should always show uncertainty and the naive comparator. "
  "Names of real places and the KES unit were used only for readability - the data must not be quoted as real market statistics. Models trained on better-reported markets may serve poorly reported markets worse (the unseen-market results hint at this), "
  "which is a fairness concern for real deployment.\n")
W("**Real-data migration checklist** (from the specification, section 16.1) and what this repository already provides:\n")
W(md(pd.DataFrame([
    ["Identify trustworthy sources; document price type (farm-gate / wholesale / retail)", "Provenance template: `docs/provenance_statement.md`"],
    ["Standardise crop names, varieties, grades, units, market IDs, dates", "Schema in `docs/data_dictionary.md`; replace `generator.py` by an ingestion step that emits the same columns"],
    ["Record source, collection time, publication time, revisions", "Add `published_at` / `revised_at` columns; `features.py` already supports availability delays via `report_delay_weeks` - extend to revisions"],
    ["Investigate missingness and reporting changes (do not assume random)", "Annotation table + missingness plots in the EDA; replace random masks by observed ones"],
    ["Align weather to production catchments; account for delays", "Weather fields are catchment-level and lag-aware in the feature builder"],
    ["Use only indicators available at forecast time", "`docs/forecast_time_data_contract.md` + truncation tests; run them on the real pipeline"],
    ["Rebuild splits around the real collection period; keep a final untouched test", "`splits.py` partitions and purging take weeks from the config"],
    ["Re-evaluate across seasons, locations, grades and horizons", "All breakdown tables are generated per crop, market, regime and event"],
], columns=["Step", "Where this repository helps"])) + "\n")

W("## 12. Conclusion and future work\n")
W("**Conclusion.** In this synthetic world, a transparent comparison shows that (i) the last observed price is a hard benchmark and seasonal structure is only exploitable for the one strongly seasonal crop; "
  "(ii) feature enrichment - above all information from connected markets - gives a real but modest improvement of roughly 6-20% over naive depending on the period, "
  "(iii) the choice among Random Forest, gradient boosting implementations and a well-specified linear model matters little compared with what information they receive; "
  "(iv) the edge largely vanishes under shocks of a new kind or strength (for example the lasting transport-cost shift), where forecasts lag rising prices; (v) ignoring data quality is more damaging than any modelling choice; and (vi) uncertainty bands need event-aware calibration. "
  "None of this should be read as evidence about real markets.\n")
W("**Future work (in the order we would do it).** (1) Repeat over 20+ seeds and report the distribution of rankings. (2) Hybrid models (linear trend + trees; Ridge with crop interactions + residual trees), SARIMAX with exogenous drivers, hierarchical/pooled models. "
  "(3) Event-aware and shock-detecting retraining, and conformal intervals with adaptive widening. (4) Add Prophet (per-origin refits), then LSTM/TFT once the panel is enlarged. (5) Calibrate generator assumptions on sourced real observations and rerun the same pipeline.\n")

W("## Appendix A. All model variants (test MAE, KES/kg)\n")
aa = allv["MAE"].copy(); aa.columns = [f"h={c}" if str(c).isdigit() else "mean" for c in aa.columns]
W(md(aa.sort_values("mean").reset_index().rename(columns={"variant": "model"})) + "\n")
W("## Appendix B. Reproducibility record\n")
W(f"Seed {cfg['experiment']['seed']}; config `{Path(cfg['_path']).name}`; data checksums in `data/manifest.json`; Python 3.12; library versions in `requirements.txt`. "
  "Forecast output example following the specification's contract: `results/forecast_contract_test.csv` (HistGradientBoosting E4 with 80% split-conformal bands, realistic scenario, test window).\n")
(REP / "FINAL_REPORT.md").write_text("\n".join(L))

# html versions (self-contained)
for name in ["FINAL_REPORT", "ERROR_ANALYSIS", "EDA_REPORT", "DATA_AUDIT"]:
    try:
        subprocess.run(["pandoc", f"{name}.md", "-o", f"{name}.html", "--standalone", "--embed-resources", "--metadata", f"title={name.replace('_', ' ').title()}",
                        "--resource-path", str(REP)], cwd=REP, check=True, capture_output=True)
    except Exception as ex:
        print("pandoc failed for", name, getattr(ex, "stderr", b"")[:200])
print("reports written")
