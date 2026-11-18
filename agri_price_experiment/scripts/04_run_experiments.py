#!/usr/bin/env python
"""Stages 4-6: baselines + core models + advanced models, all scenarios, validation & test partitions;
rolling-origin ablation; generalisation tests; prediction intervals.

Usage:  python scripts/04_run_experiments.py [--part main|ablation|generalisation|intervals|all]
Requires results/best_params.json (from 03_tune.py).
"""
import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from agriprice.config import load_config, results_dir
from agriprice.experiment import (Context, COMBOS, PRIMARY, combo_label, run_partition, run_cv_ablation,
                                  run_generalisation, run_intervals)

ap = argparse.ArgumentParser(); ap.add_argument("--part", default="all"); a = ap.parse_args()
cfg = load_config(); R = results_dir()
best = json.loads((R / "best_params.json").read_text())
log = lambda s: print(s, flush=True)

core_ml = [(m, f) for m in ["ridge", "decision_tree", "random_forest", "hist_gb"] for f in ("E0", "E4")] + [("ridge_cropint", "E4")]
adv_ml = [(m, f) for m in (["xgboost", "lightgbm"] if cfg["advanced"]["enabled"] else []) for f in ("E0", "E4")]
abl_ml = [(m, f) for m in ["ridge", "random_forest", "hist_gb"] for f in ("E1", "E2", "E3")]

if a.part in ("main", "all"):
    outs = []
    for sc, handling in COMBOS:
        t0 = time.time(); ctx = Context(cfg, sc, handling)
        specs = core_ml + adv_ml + (abl_ml if (sc, handling) == PRIMARY else [])
        for part in ("validation", "test"):
            outs.append(run_partition(ctx, best, part, specs, with_sarima=True, log=log))
            log(f"[{combo_label(sc, handling)}] {part} done ({time.time()-t0:.0f}s)")
        pd.concat(outs, ignore_index=True).to_parquet(R / "predictions_main.parquet", index=False)
    log("main done")

ctx = Context(cfg, *PRIMARY)
if a.part in ("ablation", "all"):
    run_cv_ablation(ctx, best, log=log).to_parquet(R / "predictions_cv_ablation.parquet", index=False); log("ablation done")
if a.part in ("generalisation", "all"):
    run_generalisation(ctx, best, log=log).to_parquet(R / "predictions_generalisation.parquet", index=False); log("generalisation done")
if a.part in ("intervals", "all"):
    run_intervals(ctx, best, log=log).to_parquet(R / "predictions_intervals.parquet", index=False); log("intervals done")
