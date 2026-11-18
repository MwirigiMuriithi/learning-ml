#!/usr/bin/env python
"""Stage 2: synthetic-data acceptance checks + exploratory data analysis (EDA).

Outputs: reports/DATA_AUDIT.md, reports/EDA_REPORT.md, figures/eda_*.png, results/acceptance_checks.csv
EDA statements that could influence modelling decisions use TRAINING weeks only (<= train_end).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf

from agriprice.config import load_config, data_dir, figures_dir, reports_dir, results_dir, SCENARIOS, INDICATOR_FIELDS
from agriprice.validate import run_acceptance
from agriprice.generator import event_table

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})
cfg = load_config()
FIG, REP = figures_dir(), reports_dir()
mk_name = {k: v["name"] for k, v in cfg["markets"].items()}
cr_name = {k: v["name"] for k, v in cfg["crops"].items()}
MK_COL = dict(zip(cfg["markets"], ["#1b6ca8", "#2a9d8f", "#e9a100", "#c1121f", "#6a4c93"]))
CR_COL = dict(zip(cfg["crops"], ["#b08900", "#d62828", "#7f5539", "#6a4c93"]))
sp = cfg["experiment"]["splits"]
TE, VE = sp["train_end"], sp["val_end"]


def shade_splits(ax):
    ax.axvspan(TE + .5, VE + .5, color="#ffe8a3", alpha=.35, lw=0)
    ax.axvspan(VE + .5, sp["test_end"] + .5, color="#ffc9c9", alpha=.35, lw=0)


def md_table(df, floatfmt=".3f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


panels = {s: pd.read_csv(data_dir() / f"synthetic_panel_{s}.csv") for s in SCENARIOS}
world = pd.read_csv(data_dir() / "clean_world_audit.csv")
events = event_table(cfg)
real = panels["realistic"]

# ============================================================== acceptance checks
acc, ev_detail = run_acceptance(cfg, panels, world)
acc.to_csv(results_dir() / "acceptance_checks.csv", index=False)
ev_detail.to_csv(results_dir() / "acceptance_event_checks.csv", index=False)
all_pass = bool(acc["passed"].all() and ev_detail["passed"].all())
print(acc[["check", "passed"]].to_string(index=False)); print("ALL PASS:", all_pass)

# ============================================================== FIGURES
# 1. price series per crop (observed realistic) ----------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
for ax, (c, cn) in zip(axes, cr_name.items()):
    for m in cfg["markets"]:
        g = real[(real.crop_id == c) & (real.market_id == m)]
        ax.plot(g.week_index, g.wholesale_price_kes_kg.where(g.wholesale_price_kes_kg < 3 * g.latent_price_kes_kg.median())  # extreme x10 values clipped for readability,
                color=MK_COL[m], lw=1, label=mk_name[m])
    shade_splits(ax); ax.set_ylabel(f"{cn}\nKES/kg")
axes[0].legend(ncol=5, loc="upper left", fontsize=8)
axes[-1].set_xlabel("week index   (white = train 1-104, yellow = validation 105-130, red = test 131-156)")
fig.suptitle("Weekly wholesale prices by crop and market (realistic scenario; isolated spikes/dips are injected reporting errors)")
fig.tight_layout(); fig.savefig(FIG / "eda_01_price_series.png"); plt.close(fig)

# 2. seasonality ----------------------------------------------------------------------
w = world.copy(); w["month"] = pd.to_datetime(w.week_start).dt.month
w["lp"] = np.log(w.latent_price_kes_kg); w["dev"] = w.lp - w.groupby(["market_id", "crop_id"]).lp.transform("mean")
fig, ax = plt.subplots(figsize=(8, 4))
for c, cn in cr_name.items():
    s = w[w.crop_id == c].groupby("month").dev.mean()
    ax.plot(s.index, s.values * 100, marker="o", color=CR_COL[c], label=cn)
ax.axhline(0, color="k", lw=.6); ax.set_xlabel("month"); ax.set_ylabel("mean log-price deviation (%)")
ax.set_title("Average calendar pattern by crop (clean latent prices)"); ax.legend(); fig.tight_layout()
fig.savefig(FIG / "eda_02_seasonality.png"); plt.close(fig)

# 3. distributions ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(12, 3.6))
for ax, (c, cn) in zip(axes, cr_name.items()):
    data = [real[(real.crop_id == c) & (real.market_id == m)].wholesale_price_kes_kg.dropna().clip(upper=real[real.crop_id == c].latent_price_kes_kg.max() * 1.2) for m in cfg["markets"]]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], cfg["markets"]): patch.set_facecolor(MK_COL[m]); patch.set_alpha(.7)
    ax.set_xticklabels([mk_name[m][:4] for m in cfg["markets"]]); ax.set_title(cn)
axes[0].set_ylabel("KES/kg"); fig.suptitle("Price distributions by market (observed, realistic)"); fig.tight_layout()
fig.savefig(FIG / "eda_03_distributions.png"); plt.close(fig)

# 4. cross-market correlation of weekly returns ---------------------------------------
lw = world.sort_values(["market_id", "crop_id", "week_index"]).copy()
lw["ret"] = np.log(lw.latent_price_kes_kg).groupby([lw.market_id, lw.crop_id]).diff()
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
corr_summary = {}
for ax, (c, cn) in zip(axes, cr_name.items()):
    pv = lw[(lw.crop_id == c) & (lw.week_index <= TE)].pivot(index="week_index", columns="market_id", values="ret")
    cm = pv.corr(); corr_summary[cn] = cm.values[np.triu_indices(5, 1)].mean()
    im = ax.imshow(cm, vmin=-.2, vmax=1, cmap="viridis")
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels([mk_name[m][:3] for m in cm.columns]); ax.set_yticklabels([mk_name[m][:3] for m in cm.index]); ax.set_title(cn); ax.grid(False)
fig.colorbar(im, ax=axes, shrink=.8); fig.suptitle("Cross-market correlation of weekly log price changes (training weeks)")
fig.savefig(FIG / "eda_04_cross_market_corr.png", bbox_inches="tight"); plt.close(fig)

# 5. ACF -------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for c, cn in cr_name.items():
    lv, rt = [], []
    for (m, cc), g in lw[(lw.crop_id == c) & (lw.week_index <= TE)].groupby(["market_id", "crop_id"]):
        x = np.log(g.latent_price_kes_kg.values); lv.append(acf(x, nlags=12)); rt.append(acf(np.diff(x), nlags=12))
    axes[0].plot(np.mean(lv, 0), marker=".", color=CR_COL[c], label=cn); axes[1].plot(np.mean(rt, 0), marker=".", color=CR_COL[c], label=cn)
axes[0].set_title("ACF of log price level (mean over markets)"); axes[1].set_title("ACF of weekly log changes")
for a in axes: a.set_xlabel("lag (weeks)")
axes[0].legend(); fig.tight_layout(); fig.savefig(FIG / "eda_05_acf.png"); plt.close(fig)

# 6. indicators for one series ---------------------------------------------------------
g = real[(real.market_id == "MKT-01") & (real.crop_id == "CRP-02")]
gm = real[(real.market_id == "MKT-03")]
fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
axes[0].plot(g.week_index, g.wholesale_price_kes_kg, color="#d62828"); axes[0].set_ylabel("price KES/kg")
axes[1].plot(g.week_index, g.market_arrivals_tonnes, color="#2a9d8f"); axes[1].set_ylabel("arrivals t")
axes[2].plot(g.week_index, g.estimated_demand_index, color="#1b6ca8"); axes[2].set_ylabel("demand idx")
e = gm[gm.crop_id == "CRP-02"]
axes[3].bar(e.week_index, e.rainfall_mm, color="#4895ef"); axes[3].set_ylabel("rain mm\n(Eldoret)")
axes[4].plot(e.week_index, e.drought_index, color="#bc6c25"); axes[4].set_ylabel("drought idx\n(Eldoret)")
axes[5].plot(g.week_index, g.fuel_cost_index, color="#6a4c93"); axes[5].set_ylabel("fuel idx")
for ax in axes: shade_splits(ax)
fig.suptitle("Nairobi tomatoes: price and drivers (realistic scenario)"); axes[-1].set_xlabel("week index")
fig.tight_layout(); fig.savefig(FIG / "eda_06_indicators.png"); plt.close(fig)

# 7. missingness -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1.6, 1.6, 1.2]})
for ax, sc in zip(axes[:2], ["realistic", "stressed"]):
    p = panels[sc]; p["series"] = p.market_id.str[-2:] + "|" + p.crop_id.str[-2:]
    mat = p.pivot(index="series", columns="week_index", values="wholesale_price_kes_kg").isna().astype(int)
    ax.imshow(mat.values, aspect="auto", cmap="Reds", interpolation="nearest"); ax.grid(False)
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index, fontsize=6); ax.set_title(f"missing prices: {sc}"); ax.set_xlabel("week")
rate = pd.DataFrame({s: panels[s][INDICATOR_FIELDS].isna().mean() for s in SCENARIOS}) * 100
rate.loc["wholesale_price"] = [panels[s].wholesale_price_kes_kg.isna().mean() * 100 for s in SCENARIOS]
rate.plot.barh(ax=axes[2], width=.8, color=["#9bc1bc", "#f4a261", "#e63946"]); axes[2].set_xlabel("% missing"); axes[2].tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig(FIG / "eda_07_missingness.png"); plt.close(fig)

# 8. outliers --------------------------------------------------------------------------
sus = real[real.data_quality_flag == "suspect"]
key = sus.groupby(["market_id", "crop_id"]).size().idxmax()
g = real[(real.market_id == key[0]) & (real.crop_id == key[1])]
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(g.week_index, g.latent_price_kes_kg, color="#555", lw=1, label="latent (audit only)")
ax.plot(g.week_index, g.wholesale_price_kes_kg, color="#1b6ca8", lw=1, label="reported")
s2 = g[g.data_quality_flag == "suspect"]; ax.scatter(s2.week_index, s2.wholesale_price_kes_kg, color="red", zorder=5, label="reporting error")
ax.set_yscale("symlog", linthresh=10); ax.set_title(f"Injected reporting errors: {mk_name[key[0]]} {cr_name[key[1]]}"); ax.legend(); ax.set_xlabel("week index")
fig.tight_layout(); fig.savefig(FIG / "eda_08_outliers.png"); plt.close(fig)

# 9. event timeline --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.2))
cmap = {"harvest_glut": "#2a9d8f", "drought": "#bc6c25", "flood": "#4895ef", "fuel_step": "#6a4c93",
        "demand_surge": "#e9a100", "import_supply_shock": "#8d99ae", "cross_market_disruption": "#c1121f",
        "structural_transport_shift": "#000000"}
for i, r in events.iterrows():
    ax.barh(i, r.end_week - r.start_week + 1 + r.effect_tail_weeks, left=r.start_week, color=cmap[r.event_type], alpha=.5)
    ax.barh(i, r.end_week - r.start_week + 1, left=r.start_week, color=cmap[r.event_type])
    ax.text(r.end_week + r.effect_tail_weeks + 1, i, f"{r.event_id} {r.event_type}" + (" (NEW)" if r.new_in_test else ""), va="center", fontsize=7)
ax.axvline(TE + .5, color="k", ls="--"); ax.axvline(VE + .5, color="k", ls="--")
ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(0, 200); ax.set_xlabel("week index (solid = event, faded = lingering effect window)")
ax.set_title("Documented event schedule (dashed = train | validation | test boundaries)"); fig.tight_layout()
fig.savefig(FIG / "eda_09_event_timeline.png"); plt.close(fig)

# ============================================================== EDA statistics (training weeks only)
tr = real[real.week_index <= TE].copy()
tr["price"] = tr.wholesale_price_kes_kg.where(tr.data_quality_flag != "suspect")
desc = tr.groupby("crop_id").price.describe()[["mean", "std", "min", "50%", "max"]].round(1)
desc.index = [cr_name[i] for i in desc.index]; desc = desc.reset_index().rename(columns={"index": "crop", "50%": "median"})
by_market = tr.groupby("market_id").price.mean().round(1); by_market.index = [mk_name[i] for i in by_market.index]
vol = lw[lw.week_index <= TE].groupby("crop_id").ret.std().round(4)
vol_tab = pd.DataFrame({"crop": [cr_name[i] for i in vol.index], "weekly_log_change_sd": vol.values})
adf_rows = []
for (m, c), g in lw[lw.week_index <= TE].groupby(["market_id", "crop_id"]):
    x = np.log(g.latent_price_kes_kg.values)
    adf_rows.append(dict(crop=cr_name[c], p=adfuller(x, regression="c", autolag="AIC")[1], p_diff=adfuller(np.diff(x), autolag="AIC")[1]))
adf = pd.DataFrame(adf_rows).groupby("crop").agg(share_level_stationary=("p", lambda s: (s < .05).mean()),
                                                   share_diff_stationary=("p_diff", lambda s: (s < .05).mean())).reset_index()
# exploratory driver correlations: weekly log-price change vs lagged driver anomalies (training weeks, handling-free latent data)
lw["arr_anom"] = np.log(lw.market_arrivals_tonnes) - np.log(lw.groupby(["market_id", "crop_id"]).market_arrivals_tonnes.transform(lambda s: s.rolling(13, min_periods=6).mean().shift(1)))
lw["ret_f1"] = lw.groupby(["market_id", "crop_id"]).ret.shift(-1)
lw["drought_l4"] = lw.groupby(["market_id", "crop_id"]).drought_index.shift(4)
lw["dem_anom"] = np.log(lw.estimated_demand_index) - np.log(lw.groupby(["market_id", "crop_id"]).estimated_demand_index.transform(lambda s: s.rolling(13, min_periods=6).mean().shift(1)))
lw["fuel_chg4"] = np.log(lw.fuel_cost_index) - np.log(lw.groupby(["market_id", "crop_id"]).fuel_cost_index.shift(4))
t_ = lw[lw.week_index <= TE]
corr_rows = []
for c, cn in cr_name.items():
    g = t_[t_.crop_id == c]
    corr_rows.append(dict(crop=cn, **{n: g[n].corr(g["ret_f1"]) for n in ["arr_anom", "dem_anom", "drought_l4", "fuel_chg4"]}))
corr_tab = pd.DataFrame(corr_rows).rename(columns={"arr_anom": "arrivals anomaly", "dem_anom": "demand anomaly",
                                                   "drought_l4": "drought idx (lag 4)", "fuel_chg4": "fuel 4-wk change"})
miss_tab = pd.DataFrame({
    "scenario": SCENARIOS,
    "price missing %": [panels[s].wholesale_price_kes_kg.isna().mean() * 100 for s in SCENARIOS],
    "labelled reporting errors %": [(panels[s].data_quality_flag == "suspect").mean() * 100 for s in SCENARIOS],
    "mean indicator missing %": [panels[s][INDICATOR_FIELDS].isna().mean().mean() * 100 for s in SCENARIOS],
    "late-delivered indicator rows %": [panels[s].report_delay_weeks.mean() * 100 for s in SCENARIOS],
    "synthetic-shock rows %": [(panels[s].event_affected).mean() * 100 for s in SCENARIOS]})
mk_miss = real.groupby("market_id").wholesale_price_kes_kg.apply(lambda s: s.isna().mean() * 100).rename(index=mk_name).round(2).reset_index()
mk_miss.columns = ["market", "price missing % (realistic)"]
part_tab = pd.DataFrame([
    ["train", "1-104", "2023-01-02 to 2024-12-23", "fit, tune (rolling-origin)", 5 * 4 * TE],
    ["validation", "105-130", "2024-12-30 to 2025-06-23", "model selection", 5 * 4 * (VE - TE)],
    ["test", "131-156", "2025-06-30 to 2025-12-22", "final one-shot evaluation", 5 * 4 * (sp["test_end"] - VE)]],
    columns=["partition", "weeks (origin)", "dates", "use", "series-weeks"])

# ============================================================== REPORTS
acc_tab = acc.copy(); acc_tab["passed"] = acc_tab["passed"].map({True: "PASS", False: "FAIL"})
with open(REP / "DATA_AUDIT.md", "w") as f:
    f.write("# Synthetic Data Audit (acceptance criteria, specification section 12)\n\n")
    f.write(f"Generator seed **{cfg['experiment']['seed']}**, configuration `config/experiment_config.yaml`. "
            f"Overall result: **{'ALL CRITERIA PASS' if all_pass else 'FAILURES PRESENT'}**.\n\n")
    f.write(md_table(acc_tab[["check", "criterion", "passed", "detail"]]) + "\n\n")
    f.write("## Event-by-event shock verification\n\nEach documented event must move the variable it is meant to move.\n\n")
    f.write(md_table(ev_detail.assign(passed=ev_detail.passed.map({True: "PASS", False: "FAIL"}))) + "\n\n")
    f.write("## Scenario summary\n\n" + md_table(miss_tab, ".2f") + "\n\n")
    f.write("Price missingness by market (realistic scenario; unequal reporting is intentional):\n\n" + md_table(mk_miss, ".2f") + "\n\n")
    f.write("## Data-quality label semantics\n\n- `data_quality_flag = missing`: price not reported.\n- `suspect`: an injected recording error (audit label - **never a model input**; used only to exclude such rows when scoring).\n"
            "- `synthetic_shock`: the week lies inside a documented event window (a genuine market effect, **kept** in training and evaluation).\n"
            "- `clean`: none of the above.  `corrected` appears only in the cleaned data produced by the causal pipeline (see docs).\n")
    f.write("\n![events](../figures/eda_09_event_timeline.png)\n")

with open(REP / "EDA_REPORT.md", "w") as f:
    f.write("# Exploratory Data Analysis (Stage 2)\n\n")
    f.write("All numbers below that could drive modelling choices use **training weeks 1-104 only**. "
            "The dataset is synthetic; nothing here describes real Kenyan markets.\n\n")
    f.write("## Partitions\n\n" + md_table(part_tab) + "\n\n")
    f.write("## Price level and variability (training weeks, realistic scenario, errors excluded)\n\n" + md_table(desc, ".1f") + "\n\n")
    f.write("Mean price by market (KES/kg, pooled over crops): " + ", ".join(f"{k} {v}" for k, v in by_market.items()) + ".\n\n")
    f.write("Weekly log-price change volatility (latent):\n\n" + md_table(vol_tab, ".4f") + "\n\n")
    f.write("![series](../figures/eda_01_price_series.png)\n\n")
    f.write("## Seasonality\n\nCrops peak in different months and with different strength, so a single calendar pattern would not fit all of them.\n\n![season](../figures/eda_02_seasonality.png)\n\n")
    f.write("## Distributions\n\n![dist](../figures/eda_03_distributions.png)\n\n")
    f.write("## Persistence and stationarity\n\nAugmented Dickey-Fuller tests on training weeks (share of the 5 market series per crop with p<0.05):\n\n" + md_table(adf, ".2f") + "\n\n")
    f.write("Price levels are strongly autocorrelated (persistence), which is why last-price (naive) forecasts are a hard benchmark.  "
            "Weekly changes are far less autocorrelated.\n\n![acf](../figures/eda_05_acf.png)\n\n")
    f.write("## Cross-market relationships\n\nMean pairwise correlation of weekly log price changes (training weeks): " +
            ", ".join(f"{k} {v:.2f}" for k, v in corr_summary.items()) + ".\n\n![corr](../figures/eda_04_cross_market_corr.png)\n\n")
    f.write("## Driver relationships (exploratory)\n\nCorrelation between each lagged driver anomaly and the NEXT week's log price change (training weeks, pooled over markets):\n\n" + md_table(corr_tab, ".3f") + "\n\n")
    f.write("Supply anomalies carry the clearest signal, with the expected negative sign (more supply, lower price), strongest for perishable tomatoes.  "
            "Weather and drought act with a delay (see `figures/eda_06_indicators.png`).\n\n![ind](../figures/eda_06_indicators.png)\n\n")
    f.write("## Missingness and reporting errors\n\n" + md_table(miss_tab, ".2f") + "\n\n![miss](../figures/eda_07_missingness.png)\n\n![out](../figures/eda_08_outliers.png)\n\n")
    f.write("## Events and shocks\n\nSee `reports/DATA_AUDIT.md` for verification; schedule in `data/events_schedule.csv`.\n\n![ev](../figures/eda_09_event_timeline.png)\n\n")
    f.write("## Implications for modelling\n\n"
            "1. Use the latest available price as the reference and model the *change* (log ratio) rather than the level.\n"
            "2. Tomatoes are the most volatile crop; scale-free metrics (skill vs naive, WAPE) are needed to compare crops.\n"
            "3. Seasonality differs by crop, so any seasonal baseline must be crop-specific.\n"
            "4. Drivers act with lags (supply immediately, weather with a delay), which supports lag and rolling-window features.\n"
            "5. Reporting gaps and gross errors are concentrated in some markets, so data-quality handling must be causal and market-agnostic.\n")
print("EDA done")
