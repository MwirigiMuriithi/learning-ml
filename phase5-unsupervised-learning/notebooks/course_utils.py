"""
course_utils.py  -  small helpers shared by every notebook in Phase 5.

You do NOT need to understand this file to take the course, but you are
welcome to read it. Good programmers read other people's code!

What's inside
-------------
set_style()          pretty, consistent plots
load(name)           load a course dataset as a pandas DataFrame
check(label, fn)     friendly self-check for exercises (never crashes your notebook)
plot_clusters_2d()   scatter plot of clustered points with optional centres
silhouette_plot()    the classic per-sample silhouette "knife" plot
profile_clusters()   summarise what makes each cluster different (used in projects)
"""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# A colour-blind-friendly palette (Okabe-Ito) + a few extras
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#F0E442", "#7F7F7F", "#8C564B", "#17BECF"]
NOISE_COLOR = "#BBBBBB"

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def set_style():
    """Apply the course's plotting style. Call once at the top of a notebook."""
    plt.rcParams.update({
        "figure.figsize": (7, 4.5),
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "legend.frameon": False,
    })


def load(name):
    """Load a course dataset by name (with or without .csv).

    >>> df = load("farmers")

    If the data folder is missing (e.g. you only downloaded the notebooks),
    this regenerates every dataset automatically.
    """
    name = name if name.endswith(".csv") else name + ".csv"
    path = DATA_DIR / name
    if not path.exists():
        script = ROOT / "scripts" / "generate_datasets.py"
        print("Data folder not found - generating datasets (takes ~2 seconds)...")
        subprocess.run([sys.executable, str(script)], check=True)
    return pd.read_csv(path)


def check(label, fn):
    """Self-check helper for exercises.

    Usage:   check("k should be 4", lambda: k == 4)

    - ✅ means your answer passes
    - ❌ means something is off (re-read the hints!)
    - ⏳ means you haven't defined the variable yet (it never crashes)
    """
    try:
        ok = bool(fn())
    except NameError as e:
        print(f"⏳ {label}  (not attempted yet: {e})")
        return
    except Exception as e:  # noqa: BLE001 - we want to be friendly, not strict
        print(f"❌ {label}  ({type(e).__name__}: {e})")
        return
    print(("✅ " if ok else "❌ ") + label + ("" if ok else "   -> open the hints and try again"))


def plot_clusters_2d(X, labels, centers=None, ax=None, title=None, xlabel=None, ylabel=None,
                     s=28, annotate_centers=True):
    """Scatter plot of 2-D data coloured by cluster label. Label -1 (noise) is drawn in grey."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    ax = ax or plt.gca()
    for i, lab in enumerate(sorted(set(labels))):
        mask = labels == lab
        if lab == -1:
            ax.scatter(X[mask, 0], X[mask, 1], s=s * 0.8, c=NOISE_COLOR, marker="x", label="noise", linewidths=1.2)
        else:
            ax.scatter(X[mask, 0], X[mask, 1], s=s, color=PALETTE[i % len(PALETTE)], alpha=0.8,
                       label=f"cluster {lab}", edgecolor="white", linewidth=0.4)
    if centers is not None:
        centers = np.asarray(centers)
        ax.scatter(centers[:, 0], centers[:, 1], s=220, c="black", marker="*", label="centres", zorder=5)
        if annotate_centers:
            for j, (cx, cy) in enumerate(centers):
                ax.annotate(str(j), (cx, cy), color="white", fontsize=7, ha="center", va="center", zorder=6)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def silhouette_plot(X, labels, ax=None, title="Silhouette plot"):
    """The classic silhouette 'knife' plot: one bar per sample, grouped by cluster."""
    from sklearn.metrics import silhouette_samples, silhouette_score
    X = np.asarray(X)
    labels = np.asarray(labels)
    ax = ax or plt.gca()
    vals = silhouette_samples(X, labels)
    avg = silhouette_score(X, labels)
    y_lower = 10
    for i, lab in enumerate(sorted(set(labels))):
        v = np.sort(vals[labels == lab])
        y_upper = y_lower + len(v)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, v, color=PALETTE[i % len(PALETTE)], alpha=0.8)
        ax.text(-0.07, y_lower + 0.5 * len(v), str(lab), fontsize=9)
        y_lower = y_upper + 10
    ax.axvline(avg, color="red", linestyle="--", label=f"average = {avg:.2f}")
    ax.set_xlabel("silhouette coefficient")
    ax.set_ylabel("cluster")
    ax.set_yticks([])
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ax


def profile_clusters(df, labels, cols, round_to=2, include_size=True):
    """Return a tidy table: the *average* of each column inside each cluster.

    This is the single most useful thing you can do after clustering:
    it turns anonymous cluster numbers into stories you can explain to a manager.
    """
    tmp = df[cols].copy()
    tmp["cluster"] = np.asarray(labels)
    prof = tmp.groupby("cluster").mean().round(round_to)
    if include_size:
        prof.insert(0, "size", tmp.groupby("cluster").size())
        prof.insert(1, "share_%", (100 * prof["size"] / prof["size"].sum()).round(1))
    return prof
