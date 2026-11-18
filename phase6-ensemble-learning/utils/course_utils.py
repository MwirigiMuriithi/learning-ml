"""
course_utils.py -- small helpers shared by every notebook in Phase 6.

Nothing in here is "magic": every function is short and readable.  Open this file
whenever you are curious about what a helper does -- reading code is a skill too.

Typical first cell of a notebook:

    import sys; sys.path.append("../utils")
    from course_utils import *
"""
from __future__ import annotations

import time
import warnings
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
SEED = 42

PALETTE = {
    "blue": "#2563eb", "orange": "#f97316", "green": "#16a34a", "red": "#dc2626",
    "purple": "#7c3aed", "teal": "#0d9488", "pink": "#db2777", "gray": "#6b7280",
}


# --------------------------------------------------------------------------
# Look & feel
# --------------------------------------------------------------------------
def set_style() -> None:
    """One consistent, readable plotting style for the whole course."""
    plt.rcParams.update({
        "figure.figsize": (8, 4.5), "figure.dpi": 100, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "axes.prop_cycle": cycler(color=list(PALETTE.values())), "legend.frameon": False,
    })
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*IProgress.*")   # harmless tqdm notice if ipywidgets is missing


def banner(text: str, char: str = "=") -> None:
    line = char * (len(text) + 4)
    print(f"{line}\n  {text}\n{line}")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _ensure_data() -> None:
    if not (DATA / "housing.csv").exists():
        from make_datasets import main  # regenerate if someone deleted the CSVs
        main()


def load_housing() -> pd.DataFrame:
    """Regression dataset: predict `price_k` (thousands of $)."""
    _ensure_data()
    return pd.read_csv(DATA / "housing.csv")


def load_loans() -> pd.DataFrame:
    """Classification dataset: predict `defaulted` (1 = defaulted)."""
    _ensure_data()
    return pd.read_csv(DATA / "loan_default.csv")


def load_crops() -> pd.DataFrame:
    """Regression dataset (capstone): predict `yield_t_ha`."""
    _ensure_data()
    return pd.read_csv(DATA / "crop_yield.csv")


# --------------------------------------------------------------------------
# Self-checking problems
# --------------------------------------------------------------------------
class _Todo:
    """Placeholder for answers you have not written yet."""
    def __repr__(self) -> str:
        return "TODO"

    def __bool__(self) -> bool:
        return False


TODO = _Todo()
_RESULTS: dict[str, bool | None] = {}


def check(name: str, condition, hint: str = "", needs=()) -> None:
    """
    Auto-grader used under every problem.

    condition : a bool, or a zero-argument function returning a bool
    needs     : variables that must be filled in first (anything still TODO -> 'not attempted')
    """
    for v in needs:
        if v is TODO or v is None:
            print(f"⏳ {name}: not attempted yet -- write your code above, then re-run this cell.")
            _RESULTS[name] = None
            return
    try:
        ok = bool(condition() if callable(condition) else condition)
    except Exception as e:  # noqa: BLE001 - we want to show *any* student error kindly
        print(f"❌ {name}: your code raised {type(e).__name__}: {e}")
        if hint:
            print(f"   💡 {hint}")
        _RESULTS[name] = False
        return
    if ok:
        print(f"✅ {name}: correct -- nice work!")
    else:
        print(f"❌ {name}: not quite yet.")
        if hint:
            print(f"   💡 {hint}")
    _RESULTS[name] = ok


def scoreboard() -> None:
    """Print how many problems you have solved so far in this session."""
    done = sum(1 for v in _RESULTS.values() if v)
    todo = sum(1 for v in _RESULTS.values() if v is None)
    print(f"Scoreboard: {done} solved, {len(_RESULTS) - done - todo} to fix, {todo} not attempted")
    for k, v in _RESULTS.items():
        print("  ", "✅" if v else ("⏳" if v is None else "❌"), k)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_report(y_true, y_pred) -> dict:
    """RMSE, MAE, R^2 and MAPE in one go (all on the scale of y)."""
    from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": 100 * float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None)))),
    }


class Timer:
    """with Timer() as t: ...   then t.seconds"""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def plot_decision_boundary(model, X, y, ax=None, title: str = "", h: float = 0.02):
    """Colour the plane by the model's prediction (2-D features only)."""
    ax = ax or plt.gca()
    X = np.asarray(X)
    x_min, x_max = X[:, 0].min() - 0.4, X[:, 0].max() + 0.4
    y_min, y_max = X[:, 1].min() - 0.4, X[:, 1].max() + 0.4
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, alpha=0.25, levels=[-0.5, 0.5, 1.5], colors=[PALETTE["blue"], PALETTE["orange"]])
    ax.scatter(X[:, 0], X[:, 1], c=np.where(np.asarray(y) == 1, PALETTE["orange"], PALETTE["blue"]),
               s=14, edgecolor="white", linewidth=0.4)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return ax


def environment_report() -> pd.DataFrame:
    """Which versions of the course libraries are installed?"""
    import importlib
    warnings.filterwarnings("ignore", message=".*IProgress.*")
    rows = [("python", platform.python_version())]
    for lib in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "xgboost", "lightgbm", "catboost", "shap", "optuna"]:
        try:
            m = importlib.import_module(lib)
            rows.append((lib, getattr(m, "__version__", "?")))
        except ImportError:
            rows.append((lib, "NOT INSTALLED  ->  pip install -r requirements.txt"))
    return pd.DataFrame(rows, columns=["library", "version"])


def show_gif(path):
    """Display a GIF inside the notebook (embedded, so it survives sharing)."""
    from IPython.display import Image, display
    display(Image(data=Path(path).read_bytes(), format="gif"))


# --------------------------------------------------------------------------
# Modelling conveniences (used from Lesson 02 onwards)
# --------------------------------------------------------------------------
HOUSING_CATEGORICAL = ["neighborhood", "property_type", "energy_rating"]


def housing_xy(drop=("listing_id",)):
    """Features X and target y (price_k) for the housing data, ids removed."""
    df = load_housing()
    y = df["price_k"]
    X = df.drop(columns=["price_k", *drop])
    return X, y


def get_housing_split(test_size: float = 0.2, seed: int = SEED):
    """The course-wide train/test split.  Using the *same* split everywhere keeps results comparable."""
    from sklearn.model_selection import train_test_split
    X, y = housing_xy()
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def make_preprocessor(X: pd.DataFrame, cat_cols=None, scale: bool = False, onehot: bool = True):
    """
    Turn a raw DataFrame into model-ready numbers.

      numeric columns     -> median imputation (+ optional standardisation)
      categorical columns -> most-frequent imputation -> one-hot  (or ordinal codes if onehot=False)

    Returns an *unfitted* ColumnTransformer; put it in a Pipeline so that it is fitted on
    training data only (that is how we avoid data leakage -- see Lesson 06).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    cat_cols = [c for c in (cat_cols if cat_cols is not None else HOUSING_CATEGORICAL) if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    num_steps = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    enc = (OneHotEncoder(handle_unknown="ignore") if onehot
           else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    return ColumnTransformer(sparse_threshold=0, transformers=[
        ("num", make_pipeline(*num_steps), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), enc), cat_cols),
    ])


def results_table(rows: list[dict], sort_by: str = "RMSE") -> pd.DataFrame:
    """Collect dicts like {'model': 'RF', 'RMSE': 12.3, ...} into a tidy, sorted table."""
    return pd.DataFrame(rows).sort_values(sort_by).reset_index(drop=True)


# --------------------------------------------------------------------------
# Statistics helper (you wrote this yourself in Lesson 06, Problem 06.3)
# --------------------------------------------------------------------------
def corrected_resampled_ttest(diffs, n_train: int, n_test: int):
    """
    Nadeau & Bengio (2003) corrected resampled t-test for paired cross-validation differences.

    diffs   : per-fold score differences (model A - model B), same folds for both
    returns : (t statistic, two-sided p-value)
    """
    from scipy import stats
    d = np.asarray(diffs, float)
    k = len(d)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * d.var(ddof=1))
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))
