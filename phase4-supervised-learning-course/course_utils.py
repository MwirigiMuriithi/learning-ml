"""
course_utils.py  --  small helper library for the Supervised Learning course.

Import it at the top of every notebook:

    import sys, os
    sys.path.insert(0, os.path.abspath(".."))
    from course_utils import *

What you get
------------
data_path(name)            -> absolute path of a file in the data/ folder
check(label, test)         -> friendly self-check (prints a tick or a cross, never crashes)
regression_scores(...)     -> MAE / RMSE / R2 in one dict
plot_decision_boundary(..) -> colour map of a 2-D classifier
plot_regression_fit(..)    -> scatter + fitted curve for 1-D regression
nice_table(df)             -> tidy display for result tables
"""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------
# Plot style
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (7, 4.2), "figure.dpi": 90, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelsize": 10, "legend.frameon": False,
    "axes.prop_cycle": plt.cycler(color=["#2563eb", "#f97316", "#16a34a", "#dc2626",
                                         "#7c3aed", "#0891b2", "#ca8a04", "#db2777"]),
})
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 3)

RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
def _find_data_dir():
    here = os.path.abspath(os.path.dirname(__file__))
    for cand in (os.path.join(here, "data"), os.path.join(os.getcwd(), "data"),
                 os.path.join(os.getcwd(), "..", "data")):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError("Could not find the data/ folder. Keep the course folder structure intact.")


DATA_DIR = _find_data_dir()


def data_path(name):
    """Absolute path to a dataset in the course data/ folder."""
    return os.path.join(DATA_DIR, name)


# ----------------------------------------------------------------------------
# Self-check helper
# ----------------------------------------------------------------------------
def check(label, test):
    """
    Friendly self-check.  `test` is either a bool or a zero-argument function (lambda).
    It NEVER raises, so 'Run All' always works even when you haven't solved a problem yet.

        check("slope is about 2", lambda: abs(slope - 2) < 0.01)
    """
    try:
        ok = bool(test() if callable(test) else test)
        err = None
    except Exception as exc:  # noqa
        ok, err = False, type(exc).__name__
    if ok:
        print(f"✅  {label}")
    else:
        extra = f"  (your code raised {err})" if err else ""
        print(f"❌  {label}  -- not yet! Re-read the hints and try again.{extra}")


# ----------------------------------------------------------------------------
# Metrics / tables
# ----------------------------------------------------------------------------
def regression_scores(y_true, y_pred, name=None):
    """MAE, RMSE and R^2 in a dict (optionally with a model name)."""
    out = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }
    if name is not None:
        out = {"model": name, **out}
    return out


def nice_table(df, cmap="Blues", subset=None):
    """Display a DataFrame with a light colour gradient (falls back to plain table)."""
    try:
        return df.style.background_gradient(cmap=cmap, subset=subset).format(precision=3)
    except Exception:  # matplotlib/jinja missing
        return df


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def plot_regression_fit(X, y, model=None, ax=None, title=None, grid_points=400, label="model"):
    """Scatter of a 1-D regression problem plus the fitted model's curve."""
    X = np.asarray(X).reshape(-1, 1)
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(X, y, s=18, alpha=0.65, label="data")
    if model is not None:
        xs = np.linspace(X.min(), X.max(), grid_points).reshape(-1, 1)
        ax.plot(xs, model.predict(xs), lw=2.4, color="#dc2626", label=label)
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_decision_boundary(model, X, y, ax=None, title=None, resolution=250, alpha=0.28, s=22,
                           feature_names=None):
    """
    Colour map of a *fitted* classifier trained on exactly 2 features.
    X must be an array (n_samples, 2).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.6, 4.4))
    pad = 0.08 * (X.max(axis=0) - X.min(axis=0))
    x_min, y_min = X.min(axis=0) - pad
    x_max, y_max = X.max(axis=0) + pad
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    classes = np.unique(y)
    palette = ["#2563eb", "#f97316", "#16a34a", "#dc2626", "#7c3aed", "#0891b2"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(palette[: len(classes)])
    ax.contourf(xx, yy, zz, alpha=alpha, cmap=cmap, levels=len(classes), antialiased=True)
    for i, c in enumerate(classes):
        ax.scatter(X[y == c, 0], X[y == c, 1], s=s, color=palette[i], edgecolor="white",
                   linewidth=0.5, label=f"class {c}")
    if feature_names:
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
    if title:
        ax.set_title(title)
    ax.grid(False)
    ax.legend(loc="best", fontsize=8)
    return ax


__all__ = [n for n in dir() if not n.startswith("_")]
