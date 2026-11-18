#!/usr/bin/env python
"""Stages 7-8: evaluation tables, breakdowns, statistics, figures, hypothesis verdicts, error analysis inputs.

Reads results/predictions_*.parquet ; writes results/*.csv, results/findings.json, figures/res_*.png,
results/forecast_contract_test.csv
"""
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agriprice.config import load_config, results_dir, figures_dir, data_dir
from agriprice.experiment import Context, PRIMARY, combo_label
from agriprice.metrics import metric_table
from agriprice.analysis import (add_variant, variant, paired_block_bootstrap, vol_threshold, add_regimes,
                                skill_table, group_permutation_importance)
from agriprice.splits import partitions

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})
cfg = load_config(); R, FIG = results_dir(), figures_dir()
H = cfg["experiment"]["horizons"]
CR = {k: v["name"] for k, v in cfg["crops"].items()}
MK = {k: v["name"] for k, v in cfg["markets"].items()}
best_params = json.loads((R / "best_params.json").read_text())
findings: dict = {}

main = add_variant(pd.read_parquet(R / "predictions_main.parquet"))
main["combo"] = main["scenario"] + "|" + main["handling"]
P_SC, P_H = PRIMARY
PRIM = combo_label(P_SC, P_H)

COLORS = {"Naive": "#888888", "Seasonal naive": "#bbbbbb", "Moving average": "#555555", "Ridge": "#1b6ca8",
          "SARIMA": "#6a4c93", "Decision Tree": "#e9a100", "Random Forest": "#2a9d8f", "HistGradientBoosting": "#c1121f",
          "XGBoost": "#8d5524", "LightGBM": "#e07be0", "ARIMA (non-seasonal)": "#b39ddb", "Ridge + crop interactions": "#7fb3e6"}
def col(v):
    for k, c in COLORS.items():
        if v.startswith(k): return c
    return "#333"

# --------------------------------------------------------------------------- 1. all-metric table
ALL = metric_table(main[main["scored"]], ["combo", "partition", "variant", "horizon"])
ALL.to_csv(R / "metrics_overall_all_combos.csv", index=False)
prim = main[main["combo"] == PRIM]
val, test = prim[prim.partition == "validation"], prim[prim.partition == "test"]

# --------------------------------------------------------------------------- 2. model selection on VALIDATION only
def pooled_mae(df):  # mean over horizons of per-horizon MAE
    return metric_table(df[df.scored], ["variant", "horizon"]).groupby("variant").MAE.mean()
vmae = pooled_mae(val)
ma_choice = min([v for v in vmae.index if v.startswith("Moving average")], key=lambda v: vmae[v])
def ml_choice(base):                      # E0 vs E4 chosen on validation for the headline row
    c = [v for v in vmae.index if v.startswith(base + " [")]
    return min(c, key=lambda v: vmae[v])
core_rows = {"Naive": "Naive", "Seasonal naive": "Seasonal naive", "Moving average": ma_choice,
             "Ridge": "Ridge [E4]", "SARIMA": "SARIMA", "Decision Tree": "Decision Tree [E4]",
             "Random Forest": "Random Forest [E4]", "HistGradientBoosting": "HistGradientBoosting [E4]"}
core_variants = list(core_rows.values())
cand = {k: vmae[v] for k, v in core_rows.items()}
best_core = min(cand, key=cand.get)
best_variant = core_rows[best_core]
ml_core = {k: v for k, v in core_rows.items() if "[" in v}
best_ml_name = min(ml_core, key=lambda k: vmae[ml_core[k]]); best_ml = ml_core[best_ml_name]
sel = dict(validation_mean_MAE={k: round(float(v), 4) for k, v in vmae.items()},
           moving_average_window_selected=ma_choice, best_core_candidate=best_variant, best_ml_candidate=best_ml,
           rule="lowest mean-over-horizons validation MAE (realistic scenario, data handling on); test never used")
(R / "model_selection.json").write_text(json.dumps(sel, indent=2))
findings["best_variant"], findings["best_ml"], findings["ma_choice"] = best_variant, best_ml, ma_choice

# --------------------------------------------------------------------------- 3. headline tables (primary combo)
def table_by_h(df, variants, metric="MAE"):
    t = metric_table(df[df.scored & df.variant.isin(variants)], ["variant", "horizon"])
    piv = t.pivot(index="variant", columns="horizon", values=metric).reindex(variants)
    piv["mean_over_horizons"] = piv.mean(axis=1)
    return piv

head = {}
for part, df in [("validation", val), ("test", test)]:
    for metric in ["MAE", "RMSE", "WAPE", "sMAPE", "DirAcc"]:
        head[(part, metric)] = table_by_h(df, core_variants, metric)
    head[(part, "MAE")].to_csv(R / f"core_eight_MAE_{part}.csv")
core_test = pd.concat({m: head[("test", m)] for m in ["MAE", "RMSE", "WAPE", "sMAPE", "DirAcc"]}, axis=1)
core_test.to_csv(R / "core_eight_all_metrics_test.csv")
naive_test_mae = head[("test", "MAE")].loc["Naive"]
skill_core = 1 - head[("test", "MAE")] / naive_test_mae
skill_core.to_csv(R / "core_eight_skill_vs_naive_test.csv")

all_variants = [v for v in test.variant.unique()]
full_test = pd.concat({m: table_by_h(test, all_variants, m) for m in ["MAE", "RMSE", "WAPE"]}, axis=1)
full_test.to_csv(R / "all_variants_test_metrics.csv")
full_val = table_by_h(val, [v for v in val.variant.unique()], "MAE"); full_val.to_csv(R / "all_variants_validation_MAE.csv")
gap = (head[("test", "MAE")] / head[("validation", "MAE")] - 1)
gap.to_csv(R / "validation_to_test_gap.csv")

# --------------------------------------------------------------------------- 4. breakdowns (test, primary)
tc = test[test.variant.isin(core_variants + [v for v in test.variant.unique() if v.startswith("Ridge + crop")])]
by_crop = skill_table(tc, ["crop_id", "horizon"]); by_crop["crop"] = by_crop.crop_id.map(CR); by_crop.to_csv(R / "breakdown_by_crop_test.csv", index=False)
by_market = skill_table(tc, ["market_id", "horizon"]); by_market["market"] = by_market.market_id.map(MK); by_market.to_csv(R / "breakdown_by_market_test.csv", index=False)

ctx = Context(cfg, *PRIM and (P_SC, P_H))
thr_test = vol_threshold(ctx.frames, partitions(cfg)["validation"][1])
thr_val = vol_threshold(ctx.frames, partitions(cfg)["train"][1])
tcv = add_regimes(tc, thr_test); tcv["regime"] = np.where(tcv.high_vol, "high volatility", "calm")
by_vol = skill_table(tcv, ["regime", "horizon"]); by_vol.to_csv(R / "breakdown_by_volatility_test.csv", index=False)

tce = tc.copy(); tce["period"] = np.where(tce.y_event_affected, "event-affected target week",
                                          np.where(tce.y_regime, "after transport regime shift (no other event)", "ordinary week"))
by_event = skill_table(tce, ["period", "horizon"]); by_event.to_csv(R / "breakdown_shock_vs_ordinary_test.csv", index=False)

rows = []
ev_tab = pd.read_csv(data_dir() / "events_schedule.csv")
for _, ev in ev_tab[ev_tab.partition.isin(["test"])].iterrows():
    if ev.is_structural_change:
        continue
    sub = tc[tc.y_event_ids.str.contains(ev.event_id)]
    if len(sub) == 0: continue
    t = skill_table(sub, [])
    t["event_id"], t["event_type"], t["new_in_test"], t["rows"] = ev.event_id, ev.event_type, ev.new_in_test, len(sub) // len(sub.variant.unique())
    rows.append(t)
by_ev_id = pd.concat(rows, ignore_index=True); by_ev_id.to_csv(R / "breakdown_by_event_test.csv", index=False)

pre = tc[~tc.y_regime & (tc.target_week < 140)]; post = tc[tc.target_week >= 140]
reg_rows = []
for nm, d in [("before week 140", pre), ("from week 140 (regime shift)", post)]:
    t = skill_table(d, []); t["window"] = nm; reg_rows.append(t)
pd.concat(reg_rows).to_csv(R / "breakdown_regime_shift_test.csv", index=False)

# staleness (data-quality at origin) for best ML vs naive
bq = test[test.variant.isin(["Naive", best_ml, best_variant])].copy()
bq["stale"] = pd.cut(bq["price_stale_weeks"].fillna(0), [-1, 0, 2, 100], labels=["fresh (0 wk)", "1-2 wk old", ">=3 wk old"])
stale_tab = skill_table(bq, ["stale"]); stale_tab.to_csv(R / "breakdown_by_price_staleness_test.csv", index=False)

# --------------------------------------------------------------------------- 5. data scenarios & handling (test)
sc_variants = ["Naive", "Seasonal naive", ma_choice, "Ridge [E4]", "SARIMA", "Decision Tree [E4]", "Random Forest [E4]", "HistGradientBoosting [E4]"]
sc_tab = metric_table(main[(main.partition == "test") & main.scored & main.variant.isin(sc_variants)], ["combo", "variant", "horizon"])
sc_piv = sc_tab.pivot_table(index=["variant", "combo"], columns="horizon", values="MAE")
sc_piv["mean_over_horizons"] = sc_piv.mean(axis=1); sc_piv.to_csv(R / "scenario_comparison_test.csv")
sc_val = metric_table(main[(main.partition == "validation") & main.scored & main.variant.isin(sc_variants)], ["combo", "variant", "horizon"])
sc_val.pivot_table(index=["variant", "combo"], columns="horizon", values="MAE").to_csv(R / "scenario_comparison_validation.csv")

# --------------------------------------------------------------------------- 6. ablation (test) & CV ablation
abl = main[(main.combo == PRIM) & (main.partition == "test") & main.model.isin(["ridge", "random_forest", "hist_gb"]) & main.scored]
abl_t = metric_table(abl, ["model", "feature_set", "horizon"])
abl_piv = abl_t.pivot_table(index=["model", "feature_set"], columns="horizon", values="MAE"); abl_piv["mean_over_horizons"] = abl_piv.mean(axis=1)
abl_piv.to_csv(R / "ablation_test_MAE.csv")
if (R / "predictions_cv_ablation.parquet").exists():
    cv = add_variant(pd.read_parquet(R / "predictions_cv_ablation.parquet"))
    cvt = metric_table(cv[cv.scored & cv.model.isin(["ridge", "random_forest", "hist_gb"])], ["model", "feature_set", "horizon"])
    cvp = cvt.pivot_table(index=["model", "feature_set"], columns="horizon", values="MAE"); cvp["mean_over_horizons"] = cvp.mean(axis=1)
    cvp.to_csv(R / "ablation_rolling_cv_MAE.csv")
    cv_naive = metric_table(cv[cv.scored & (cv.model == "naive")], ["horizon"]).set_index("horizon").MAE
    cvf = metric_table(cv[cv.scored & cv.model.isin(["random_forest", "hist_gb", "ridge"]) & cv.feature_set.isin(["E0", "E4"])], ["model", "feature_set", "fold", "horizon"])
    cvf.to_csv(R / "ablation_rolling_cv_by_fold.csv", index=False)
# ---- selection decision rule (spec section 15): complex model only if MEANINGFUL and REPEATABLE gain over strongest baseline
base_names = ["Naive", "Seasonal naive", ma_choice]
strongest_base = min(base_names, key=lambda v: vmae[v])
gain = 1 - vmae[best_ml] / vmae[strongest_base]
key_of = {"Ridge [E4]": "ridge", "Random Forest [E4]": "random_forest", "HistGradientBoosting [E4]": "hist_gb"}
base_key = {"Naive": "naive", "Seasonal naive": "seasonal_naive"}.get(strongest_base, "ma" + ma_choice.split("(")[1].split("w")[0])
folds_won, n_folds = None, None
if (R / "predictions_cv_ablation.parquet").exists() and best_ml in key_of:
    cvm = cv[cv.scored]
    f_c = metric_table(cvm[(cvm.model == key_of[best_ml]) & (cvm.feature_set == "E4")], ["fold", "horizon"]).set_index(["fold", "horizon"]).MAE
    f_b = metric_table(cvm[cvm.model == base_key], ["fold", "horizon"]).set_index(["fold", "horizon"]).MAE
    fm = (f_c.groupby("fold").mean() < f_b.groupby("fold").mean())
    folds_won, n_folds = int(fm.sum()), int(len(fm))
meaningful = bool(gain >= 0.05); repeatable = bool(folds_won is not None and folds_won >= max(1, n_folds - 1))
sel.update(strongest_baseline=strongest_base, validation_gain_over_strongest_baseline=round(float(gain), 4),
           rolling_folds_won=folds_won, rolling_folds_total=n_folds, meaningful_gain_rule=">= 5% lower validation MAE",
           repeatable_rule=">= (folds - 1) rolling-origin folds won", complex_model_justified=bool(meaningful and repeatable),
           recommended_model=best_ml if (meaningful and repeatable) else strongest_base)
(R / "model_selection.json").write_text(json.dumps(sel, indent=2))
findings["selection"] = {k: sel[k] for k in ["strongest_baseline", "validation_gain_over_strongest_baseline", "rolling_folds_won", "rolling_folds_total", "complex_model_justified", "recommended_model"]}
# per-crop E4 vs E0 (test)
ablc = metric_table(abl[abl.feature_set.isin(["E0", "E4"])], ["model", "feature_set", "crop_id", "horizon"])
ablc_p = ablc.pivot_table(index=["model", "crop_id", "horizon"], columns="feature_set", values="MAE").reset_index()
ablc_p["improvement_%"] = (1 - ablc_p["E4"] / ablc_p["E0"]) * 100; ablc_p["crop"] = ablc_p.crop_id.map(CR)
ablc_p.to_csv(R / "ablation_by_crop_test.csv", index=False)
# per-series E4 vs E0 (HGB, RF) test
ser = metric_table(abl[abl.feature_set.isin(["E0", "E4"])], ["model", "feature_set", "series"]).pivot_table(index=["model", "series"], columns="feature_set", values="MAE").reset_index()
ser["E4_better"] = ser["E4"] < ser["E0"]; ser.to_csv(R / "ablation_by_series_test.csv", index=False)

# --------------------------------------------------------------------------- 7. paired bootstrap tests (test, primary)
def sub(v, d=test): return d[d.variant == v]
tests_rows = []
def cmp(name, a_v, b_v, d=test, hs=H, dd=None):
    for h in hs:
        a = (d if dd is None else dd[0]); b = (d if dd is None else dd[1])
        a = a[(a.variant == a_v) & (a.horizon == h)]; b = b[(b.variant == b_v) & (b.horizon == h)]
        r = paired_block_bootstrap(a, b); r.update(comparison=name, A=a_v, B=b_v, horizon=h); tests_rows.append(r)
cmp("best ML vs naive", best_ml, "Naive")
cmp("best core vs naive", best_variant, "Naive")
for m, lab in [("Ridge", "Ridge"), ("Random Forest", "Random Forest"), ("HistGradientBoosting", "HistGradientBoosting"), ("Decision Tree", "Decision Tree")]:
    cmp(f"{lab}: E4 vs E0 (feature enrichment)", f"{m} [E4]", f"{m} [E0]")
cmp("Random Forest vs Ridge (E4)", "Random Forest [E4]", "Ridge [E4]")
cmp("HistGB vs Ridge (E4)", "HistGradientBoosting [E4]", "Ridge [E4]")
cmp("HistGB vs Random Forest (E4)", "HistGradientBoosting [E4]", "Random Forest [E4]")
cmp("best ML vs SARIMA", best_ml, "SARIMA")
cmp("SARIMA vs naive", "SARIMA", "Naive")
cmp("Seasonal naive vs naive", "Seasonal naive", "Naive")
cmp("Ridge+crop-interactions vs Ridge", "Ridge + crop interactions [E4]", "Ridge [E4]")
for sc in ["realistic", "stressed"]:
    a = main[(main.combo == f"{sc}|handled") & (main.partition == "test")]; b = main[(main.combo == f"{sc}|raw") & (main.partition == "test")]
    for v in [best_ml, "Naive", "Moving average (4w)", "SARIMA"]:
        for h in H:
            r = paired_block_bootstrap(a[(a.variant == v) & (a.horizon == h)], b[(b.variant == v) & (b.horizon == h)])
            r.update(comparison=f"data handling ON vs OFF ({sc})", A=v, B=v, horizon=h); tests_rows.append(r)
boot = pd.DataFrame(tests_rows); boot.to_csv(R / "paired_bootstrap_tests_test.csv", index=False)

# --------------------------------------------------------------------------- 8. generalisation
if (R / "predictions_generalisation.parquet").exists():
    g = pd.read_parquet(R / "predictions_generalisation.parquet")
    keys = ["series", "origin_week", "horizon"]
    nv = g[g.gen_kind == "reference"][keys + ["y_pred", "y_obs", "scored"]].rename(columns={"y_pred": "naive_pred"})
    rows = []
    for model in g[g.model != "naive"].model.unique():
        for h in H:
            for seen, unseen, lab in [("seen_market_no_id", "unseen_market", "unseen market"), ("seen_pair", "unseen_pair", "unseen crop-market pair")]:
                u = g[(g.model == model) & (g.gen_kind == unseen) & (g.horizon == h)]
                s = g[(g.model == model) & (g.gen_kind == seen) & (g.horizon == h)]
                m = u[keys + ["y_obs", "y_pred", "scored"]].merge(s[keys + ["y_pred"]], on=keys, suffixes=("_unseen", "_seen")).merge(nv[keys + ["naive_pred"]], on=keys)
                m = m[m.scored]
                e = lambda c: float((m.y_obs - m[c]).abs().mean())
                rows.append(dict(model=model, horizon=h, test=lab, rows=len(m), MAE_unseen=e("y_pred_unseen"), MAE_seen=e("y_pred_seen"), MAE_naive=e("naive_pred")))
    gen_tab = pd.DataFrame(rows)
    gen_tab["skill_unseen"] = 1 - gen_tab.MAE_unseen / gen_tab.MAE_naive; gen_tab["skill_seen"] = 1 - gen_tab.MAE_seen / gen_tab.MAE_naive
    gen_tab["degradation_%"] = (gen_tab.MAE_unseen / gen_tab.MAE_seen - 1) * 100
    gen_tab.to_csv(R / "generalisation_test.csv", index=False)
    per_mkt = []
    for model in g[g.model != "naive"].model.unique():
        for m_ in MK:
            u = g[(g.model == model) & (g.gen_kind == "unseen_market") & (g.held_out == m_) & g.scored]
            n_ = nv[nv.series.str.startswith(m_) & nv.scored]
            mm = u.merge(n_[keys + ["naive_pred"]], on=keys)
            per_mkt.append(dict(model=model, market=MK[m_], MAE_unseen=float((mm.y_obs - mm.y_pred).abs().mean()), MAE_naive=float((mm.y_obs - mm.naive_pred).abs().mean())))
    pm = pd.DataFrame(per_mkt); pm["skill_vs_naive"] = 1 - pm.MAE_unseen / pm.MAE_naive; pm.to_csv(R / "generalisation_by_market.csv", index=False)

# --------------------------------------------------------------------------- 9. intervals
if (R / "predictions_intervals.parquet").exists():
    iv = pd.read_parquet(R / "predictions_intervals.parquet")
    ivt = iv[(iv.partition == "test") & iv.scored].copy()
    for k in ("qr", "cf"):
        ivt[f"cover_{k}"] = (ivt.y_obs >= ivt[f"lower_{k}"]) & (ivt.y_obs <= ivt[f"upper_{k}"])
        ivt[f"width_{k}"] = (ivt[f"upper_{k}"] - ivt[f"lower_{k}"]) / ivt.y_pred
    agg = lambda d: pd.Series({"rows": len(d), "coverage_quantile_reg": d.cover_qr.mean(), "coverage_conformal": d.cover_cf.mean(),
                               "rel_width_quantile_reg": d.width_qr.median(), "rel_width_conformal": d.width_cf.median()})
    iv_h = ivt.groupby("horizon").apply(agg).reset_index(); iv_h.to_csv(R / "interval_coverage_by_horizon_test.csv", index=False)
    iv_c = ivt.assign(crop=ivt.crop_id.map(CR)).groupby(["crop", "horizon"]).apply(agg).reset_index(); iv_c.to_csv(R / "interval_coverage_by_crop_test.csv", index=False)
    iv_e = ivt.assign(event=np.where(ivt.y_event_affected, "event-affected", "ordinary")).groupby(["event", "horizon"]).apply(agg).reset_index()
    iv_e.to_csv(R / "interval_coverage_by_event_test.csv", index=False)

# --------------------------------------------------------------------------- 10. runtime
rt = main[main.fit_s.notna()].drop_duplicates(["combo", "partition", "variant", "horizon"])
rt_tab = rt[(rt.combo == PRIM) & (rt.partition == "test")].groupby("variant").agg(
    training_rows=("n_train", "mean"), fit_seconds_mean=("fit_s", "mean"), predict_seconds_mean=("pred_s", "mean")).round(3)
rt_tab.to_csv(R / "runtime_per_model.csv")

# --------------------------------------------------------------------------- 11. group permutation importance (validation, best ML)
mname = [k for k, v in {"ridge": "Ridge", "decision_tree": "Decision Tree", "random_forest": "Random Forest", "hist_gb": "HistGradientBoosting"}.items() if best_ml.startswith(v)][0]
imp = []
for h in H:
    p = best_params.get(mname, {}).get("E4", {}).get(str(h), {})
    imp.append(group_permutation_importance(mname, p, "E4", ctx.frames[h], h, partitions(cfg)["train"][1], *partitions(cfg)["validation"],
                                            cfg, True))
imp = pd.concat(imp); imp["model"] = best_ml; imp.to_csv(R / "feature_group_importance_validation.csv", index=False)

# =========================================================================== FIGURES
# F1 MAE by horizon (core eight, test)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (part, df) in zip(axes, [("validation", val), ("test", test)]):
    t = head[(part, "MAE")]
    for v in core_variants:
        ax.plot(H, t.loc[v, H].values, marker="o", color=col(v), lw=2 if v == best_variant else 1.3, label=v, ls="--" if v in ("Naive", "Seasonal naive", ma_choice) else "-")
    ax.set_xticks(H); ax.set_xlabel("forecast horizon (weeks)"); ax.set_ylabel("MAE (KES/kg)"); ax.set_title(f"{part} (realistic data, handling on)")
axes[1].legend(fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(FIG / "res_01_mae_by_horizon.png"); plt.close(fig)

# F2 skill heatmap crop x model (test, per horizon)
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
vs = [v for v in core_variants if v != "Naive"]
for ax, h in zip(axes, H):
    t = by_crop[(by_crop.horizon == h) & by_crop.variant.isin(vs)].pivot(index="crop", columns="variant", values="skill_vs_naive")[vs]
    im = ax.imshow(t.values * 100, cmap="RdYlGn", vmin=-40, vmax=40, aspect="auto"); ax.grid(False)
    ax.set_xticks(range(len(vs))); ax.set_xticklabels([v.replace(" [E4]", "").replace("HistGradientBoosting", "HistGB") for v in vs], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(t))); ax.set_yticklabels(t.index); ax.set_title(f"h = {h}")
    for i in range(t.shape[0]):
        for j in range(t.shape[1]): ax.text(j, i, f"{t.values[i, j]*100:.0f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=axes, shrink=.8, label="skill vs naive (%)"); fig.suptitle("Skill versus naive by crop (test; positive = better than last price)")
fig.savefig(FIG / "res_02_skill_by_crop.png", bbox_inches="tight"); plt.close(fig)

# F3 ablation
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
sets = ["E0", "E1", "E2", "E3", "E4"]
for ax, m in zip(axes, ["ridge", "random_forest", "hist_gb"]):
    for h, c in zip(H, ["#9bc1bc", "#f4a261", "#e63946"]):
        y = [abl_piv.loc[(m, s), h] for s in sets]; ax.plot(sets, y, marker="o", color=c, label=f"h={h}")
    ax.set_title({"ridge": "Ridge", "random_forest": "Random Forest", "hist_gb": "HistGradientBoosting"}[m]); ax.set_ylabel("test MAE (KES/kg)")
axes[0].legend(); fig.suptitle("Feature-group ablation (cumulative groups, test)"); fig.tight_layout(); fig.savefig(FIG / "res_03_ablation.png"); plt.close(fig)

# F4 scenarios
fig, ax = plt.subplots(figsize=(10, 4)); combos = ["clean|handled", "realistic|handled", "realistic|raw", "stressed|handled", "stressed|raw"]
vv = ["Naive", ma_choice, "SARIMA", "Ridge [E4]", "Random Forest [E4]", "HistGradientBoosting [E4]"]
w_ = 0.13
for i, v in enumerate(vv):
    y = [sc_piv.loc[(v, c), "mean_over_horizons"] for c in combos]
    ax.bar(np.arange(len(combos)) + i * w_, y, w_, color=col(v), label=v)
ax.set_xticks(np.arange(len(combos)) + 2.5 * w_); ax.set_xticklabels([c.replace("|", "\n") for c in combos]); ax.set_ylabel("test MAE, mean over horizons (KES/kg)")
ax.legend(fontsize=7, ncol=3); ax.set_title("Data-quality scenarios and handling (test)"); fig.tight_layout(); fig.savefig(FIG / "res_04_scenarios.png"); plt.close(fig)

# F5 actual vs predicted (best ML + naive + HGB interval)
ivt_all = iv[(iv.partition == "test")] if (R / "predictions_intervals.parquet").exists() else None
show = [("MKT-01|CRP-02", "Nairobi tomatoes"), ("MKT-03|CRP-01", "Eldoret maize"), ("MKT-04|CRP-03", "Kisumu potatoes"), ("MKT-05|CRP-04", "Mombasa onions")]
for hshow in (1, 4):
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), sharex=True)
    for ax, (s, nm) in zip(axes.ravel(), show):
        d = test[(test.series == s) & (test.horizon == hshow)]
        lat = ctx.panel_df[(ctx.panel_df.market_id == s[:6]) & (ctx.panel_df.crop_id == s[7:])]
        ax.plot(lat.week_index, lat.latent_price_kes_kg, color="#bbb", lw=3, label="true (latent) price", zorder=1)
        o = d[d.variant == "Naive"]; ax.plot(o.target_week, o.y_pred, color="#888", lw=1, ls="--", label="naive")
        b = d[d.variant == best_ml]; ax.plot(b.target_week, b.y_pred, color=col(best_ml), lw=1.6, label=best_ml)
        if ivt_all is not None:
            q = ivt_all[(ivt_all.series == s) & (ivt_all.horizon == hshow)]
            ax.fill_between(q.target_week, q.lower_cf, q.upper_cf, color="#c1121f", alpha=.12, label="80% conformal band (HGB)")
        ob = d[d.variant == "Naive"]; ax.scatter(ob.target_week, ob.y_obs, s=8, color="k", label="observed", zorder=4)
        ax.axvline(140, color="#444", ls=":", lw=1); ax.set_title(nm); ax.set_xlim(131, 156); ax.set_ylabel("KES/kg")
    axes[0, 0].legend(fontsize=7, loc="upper left"); fig.suptitle(f"Forecasts vs outcomes, horizon {hshow} week(s) (test; dotted line = transport regime shift)")
    fig.tight_layout(); fig.savefig(FIG / f"res_05_forecast_examples_h{hshow}.png"); plt.close(fig)

# F6 error over time
fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
for ax, h in zip(axes, (1, 4)):
    for v, c in [("Naive", "#888"), (best_ml, col(best_ml)), ("SARIMA", "#6a4c93")]:
        d = test[(test.variant == v) & (test.horizon == h) & test.scored]
        t = d.assign(e=(d.y_obs - d.y_pred).abs()).groupby("target_week").e.mean(); ax.plot(t.index, t.values, label=v, color=c)
    ev_ = test[(test.horizon == h) & test.y_event_affected].groupby("target_week").size().index
    for w in ev_: ax.axvspan(w - .5, w + .5, color="#ffe8a3", alpha=.35, lw=0)
    ax.axvline(140, color="#444", ls=":"); ax.set_ylabel(f"MAE (KES/kg), h={h}")
axes[0].legend(); axes[1].set_xlabel("target week (shaded: some series inside a documented event window)")
fig.suptitle("Mean absolute error by target week (test)"); fig.tight_layout(); fig.savefig(FIG / "res_06_error_over_time.png"); plt.close(fig)

# F7 feature group importance
fig, ax = plt.subplots(figsize=(8, 3.8)); gi = imp.pivot(index="group", columns="horizon", values="mae_increase").sort_values(H[-1], ascending=True)
gi.plot.barh(ax=ax, color=["#9bc1bc", "#f4a261", "#e63946"]); ax.set_xlabel("increase in validation MAE when group is shuffled (KES/kg)")
ax.set_title(f"Feature-group permutation importance: {best_ml}"); fig.tight_layout(); fig.savefig(FIG / "res_07_group_importance.png"); plt.close(fig)

# F8 intervals + F9 generalisation + F10 vol/event
if ivt_all is not None:
    fig, ax = plt.subplots(figsize=(6.5, 3.6)); x = np.arange(len(H))
    ax.bar(x - .18, iv_h.coverage_quantile_reg, .36, label="quantile regression (q10-q90)", color="#4895ef")
    ax.bar(x + .18, iv_h.coverage_conformal, .36, label="split conformal", color="#2a9d8f"); ax.axhline(.8, color="k", ls="--", label="nominal 80%")
    ax.set_xticks(x); ax.set_xticklabels([f"h={h}" for h in H]); ax.set_ylim(0, 1); ax.set_ylabel("empirical coverage (test)"); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "res_08_interval_coverage.png"); plt.close(fig)
if (R / "generalisation_test.csv").exists():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, tname in zip(axes, ["unseen market", "unseen crop-market pair"]):
        d = gen_tab[gen_tab.test == tname]
        for i, m_ in enumerate(d.model.unique()):
            dd = d[d.model == m_]; ax.bar(np.arange(len(H)) + i * .25 - .25, dd.skill_unseen * 100, .22, label=f"{m_} (unseen)", color=col(variant(m_, "E4")))
            ax.scatter(np.arange(len(H)) + i * .25 - .25, dd.skill_seen * 100, color="k", marker="_", s=120, zorder=5)
        ax.axhline(0, color="k", lw=.7); ax.set_xticks(range(len(H))); ax.set_xticklabels([f"h={h}" for h in H]); ax.set_title(tname); ax.set_ylabel("skill vs naive (%)")
    axes[0].legend(fontsize=7); fig.suptitle("Generalisation: skill vs naive when the series' targets were never seen in training (black ticks = when seen)")
    fig.tight_layout(); fig.savefig(FIG / "res_09_generalisation.png"); plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for ax, (tab, key, ttl) in zip(axes, [(by_vol, "regime", "Volatility regime"), (by_event, "period", "Event vs ordinary weeks")]):
    t = tab[tab.variant.isin(["Naive", best_ml, "SARIMA", "Ridge [E4]"])].groupby([key, "variant"]).MAE.mean().unstack()
    t.plot.bar(ax=ax, color=[col(c) for c in t.columns], rot=15); ax.set_ylabel("MAE (KES/kg), mean over horizons"); ax.set_title(ttl); ax.set_xlabel("")
fig.tight_layout(); fig.savefig(FIG / "res_10_regimes.png"); plt.close(fig)

# =========================================================================== forecast-contract output (Appendix B)
if ivt_all is not None:
    frm = ctx.frames
    rows = []
    for h in H:
        f = frm[h][["series", "origin_week", "lr_4", "price_stale_weeks", "ref_price"]].rename(columns={"ref_price": "latest_price_check"})
        q = ivt_all[ivt_all.horizon == h].drop(columns=["price_stale_weeks"], errors="ignore").merge(f, on=["series", "origin_week"], how="left")
        rows.append(q)
    c = pd.concat(rows)
    ws = pd.to_datetime(c.origin_date); c["issued_at"] = (ws + pd.Timedelta(days=6, hours=18)).dt.strftime("%Y-%m-%dT%H:%M:00Z")
    out = pd.DataFrame({
        "forecast_origin_week": c.origin_date, "target_week": c.target_date, "horizon_weeks": c.horizon,
        "market_id": c.market_id, "crop_id": c.crop_id,
        "predicted_price_kes_kg": c.y_pred.round(2),
        "model_version": [f"hist_gb_h{h}_E4_v1" for h in c.horizon],
        "data_scenario": "realistic", "issued_at": c.issued_at,
        # --- optional extensions beyond Appendix B (uncertainty + context, per section 'useful output') ---
        "lower_80_kes_kg": c.lower_cf.round(2), "upper_80_kes_kg": c.upper_cf.round(2),
        "latest_observed_price_kes_kg": c.ref_price.round(2),
        "recent_4wk_trend_pct": ((np.exp(c.lr_4) - 1) * 100).round(1),
        "data_quality_warning": np.where(c.price_stale_weeks.fillna(0) >= 3, "latest price is >=3 weeks old; forecast less reliable", ""),
        "uncertainty_note": np.where(c.horizon >= 4, "4-week-ahead forecasts carry wide uncertainty; band is empirical 80% from validation errors",
                                    "band is empirical 80% from validation errors"),
    })
    out.to_csv(R / "forecast_contract_test.csv", index=False)
    print("forecast contract rows:", len(out))

# =========================================================================== hypothesis verdicts
def b(name, v=None, h=None):
    d = boot[boot.comparison == name]
    if v is not None: d = d[d.A == v]
    if h is not None: d = d[d.horizon == h]
    return d
H1 = []
for m in ["Ridge", "Random Forest", "HistGradientBoosting"]:
    d = b(f"{m}: E4 vs E0 (feature enrichment)")
    H1.append(dict(model=m, horizons_better=int((d.mean_diff < 0).sum()), horizons_sig=int((d.ci_high < 0).sum()),
                   mean_improvement_pct=float(-(d.mean_diff / table_by_h(test, [f"{m} [E0]"]).iloc[0][H].values).mean() * 100)))
H1 = pd.DataFrame(H1)
h1 = "supported" if (H1.horizons_sig >= 1).any() and (H1.horizons_better.sum() >= 6) else ("partly supported" if H1.horizons_better.sum() >= 4 else "not supported")
d2 = pd.concat([b("Random Forest vs Ridge (E4)"), b("HistGB vs Ridge (E4)")])
h2 = "supported" if (d2.ci_high < 0).sum() >= 4 else ("partly supported" if (d2.mean_diff < 0).sum() >= 4 else "not supported")
d3 = b("best ML vs SARIMA")
h3 = "supported" if (d3.ci_high < 0).sum() >= 2 else ("partly supported" if (d3.mean_diff < 0).sum() >= 2 else "not supported")
sn = skill_table(test[test.variant.isin(["Naive", "Seasonal naive"])], ["crop_id"]); sn = sn[sn.variant == "Seasonal naive"]
sn_crop = {CR[r.crop_id]: float(r.skill_vs_naive) for r in sn.itertuples()}
h4 = "supported" if (max(sn_crop.values()) > -0.05 and min(sn_crop.values()) < -0.5 * abs(max(sn_crop.values())) - 0.05) else "partly supported" if max(sn_crop.values()) > -0.15 else "not supported"
a_ = sc_piv.xs("clean|handled", level=1)["mean_over_horizons"]; b_ = sc_piv.xs("realistic|handled", level=1)["mean_over_horizons"]
c_ = sc_piv.xs("stressed|handled", level=1)["mean_over_horizons"]; br = sc_piv.xs("realistic|raw", level=1)["mean_over_horizons"]; cr_ = sc_piv.xs("stressed|raw", level=1)["mean_over_horizons"]
deg = bool(((b_ > a_).mean() > .6) and ((c_ > b_).mean() > .6))
hb = boot[boot.comparison.str.startswith("data handling")]
helps = float((hb.mean_diff > 0).mean())      # A=handled worse than raw if mean_diff>0 ; we want negative
h5_deg = "supported" if deg else "partly supported"
h5_handling = "supported" if (hb.mean_diff < 0).mean() > .6 else ("partly supported" if (hb.mean_diff < 0).mean() > .3 else "not supported")
findings.update(dict(
    H1=dict(verdict=h1, detail=H1.round(3).to_dict("records")), H2=dict(verdict=h2), H3=dict(verdict=h3),
    H4=dict(verdict=h4, seasonal_naive_skill_by_crop=sn_crop),
    H5=dict(degradation_verdict=h5_deg, handling_verdict=h5_handling,
            mean_MAE_by_scenario={"clean": float(a_.mean()), "realistic": float(b_.mean()), "stressed": float(c_.mean()),
                                  "realistic_raw": float(br.mean()), "stressed_raw": float(cr_.mean())}),
    test_skill_best_ml_by_h={int(h): float(skill_core.loc[best_ml if best_ml in skill_core.index else best_variant, h]) for h in H} if best_ml in skill_core.index else {},
))
(R / "findings.json").write_text(json.dumps(findings, indent=2, default=float))
print(json.dumps(findings, indent=2, default=float)[:3000])
