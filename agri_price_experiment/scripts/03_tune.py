#!/usr/bin/env python
"""Stage 3: hyper-parameter tuning on rolling-origin folds (weeks <= val_end only; test never touched).
Tuned on the realistic scenario with data-quality handling; parameters are reused in other scenarios.

  python scripts/03_tune.py                 # tune all models
  python scripts/03_tune.py --only hist_gb  # re-tune selected models and merge into existing results
"""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from agriprice.config import load_config, results_dir
from agriprice.experiment import tune_all

ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*"); a = ap.parse_args()
cfg = load_config(); R = results_dir()
models = ["ridge", "decision_tree", "random_forest", "hist_gb", "ridge_cropint"]
if cfg["advanced"]["enabled"]:
    models += ["xgboost", "lightgbm"]
if a.only:
    models = [m for m in models if m in a.only]
tab, best = tune_all(cfg, models, fsets=("E0", "E4"), log=lambda s: print(s, flush=True))
if a.only and (R / "best_params.json").exists():
    old_best = json.loads((R / "best_params.json").read_text()); old_best.update(best); best = old_best
    old = pd.read_csv(R / "tuning_cv.csv"); tab = pd.concat([old[~old.model.isin(a.only)], tab], ignore_index=True)
tab.to_csv(R / "tuning_cv.csv", index=False)
(R / "best_params.json").write_text(json.dumps(best, indent=2))
print("done")
