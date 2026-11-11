"""
course_utils.py - tiny helper module shared by every notebook in this course.

You do NOT need to read this file to take the course, but you are welcome to:
it is short on purpose.  It gives you three things:

    load("housing.csv")     -> read a CSV from the datasets/ folder
    setup()                 -> consistent plot style + quiet warnings + seeds
    check("label", lambda: <expression>)
                            -> friendly self-check used in the problem sets
"""
import pathlib
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "datasets"

# A small, colour-blind-friendly palette used throughout the course.
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = (
    "#2b6cb0", "#dd6b20", "#2f855a", "#c53030", "#6b46c1", "#718096")
PALETTE = [BLUE, ORANGE, GREEN, RED, PURPLE, GREY]


def load(name: str) -> pd.DataFrame:
    """Load a CSV file from the datasets/ folder, e.g. load('housing.csv')."""
    return pd.read_csv(DATA / name)


def setup(seed: int = 42) -> int:
    """Consistent look & feel, reproducible randomness, quiet warnings."""
    warnings.filterwarnings("ignore")
    random.seed(seed)
    np.random.seed(seed)
    plt.rcParams.update({
        "figure.figsize": (7.5, 4.5),
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "legend.frameon": False,
    })
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.precision", 3)
    pd.set_option("display.width", 120)
    return seed


def check(label: str, test) -> bool:
    """
    Self-check for the problem sets.

        check("slope is about 4", lambda: abs(slope - 4) < 0.5)

    Pass a *lambda* (a tiny function) so that an unsolved problem shows a
    friendly "not yet" instead of crashing with a NameError.
    """
    try:
        ok = bool(test())
    except Exception:
        ok = False
    print(("\u2705 PASSED   " if ok else "\u23f3 NOT YET  ") + label)
    return ok
