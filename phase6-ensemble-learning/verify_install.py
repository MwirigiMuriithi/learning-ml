"""
Run this once:   python verify_install.py
It imports every library, fits a tiny model with each, and tells you exactly what (if anything) is missing.
"""
import importlib, sys, warnings
warnings.filterwarnings("ignore")

REQUIRED = ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "seaborn", "xgboost", "lightgbm", "catboost", "shap", "optuna", "PIL"]
missing = []
print(f"Python {sys.version.split()[0]}\n")
for lib in REQUIRED:
    try:
        m = importlib.import_module(lib)
        print(f"  ok   {lib:<12} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {lib:<12} {type(e).__name__}: {e}")
        missing.append(lib)

if missing:
    print("\nSome libraries are missing. Fix with:\n    pip install -r requirements.txt")
    sys.exit(1)

import numpy as np
from sklearn.datasets import make_regression
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostRegressor
X, y = make_regression(n_samples=300, n_features=8, noise=5, random_state=0)
for name, model in [("XGBoost", xgb.XGBRegressor(n_estimators=20)), ("LightGBM", lgb.LGBMRegressor(n_estimators=20, verbose=-1)),
                    ("CatBoost", CatBoostRegressor(iterations=20, verbose=0))]:
    model.fit(X, y)
    print(f"  fit  {name:<9} R² on train = {model.score(X, y):.3f}" if hasattr(model, "score") else f"  fit  {name}")
import shap
shap.TreeExplainer(lgb.LGBMRegressor(n_estimators=10, verbose=-1).fit(X, y))(X[:5])
print("  ok   SHAP explained a LightGBM model")
print("\nAll good: open JupyterLab (`jupyter lab`) and start with notebooks/00_welcome_and_setup.ipynb")
