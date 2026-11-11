"""
course_utils.py  --  small helpers shared by every notebook in Phase 2.

Nothing magical here: loading functions, a data-profiling table and a plotting style.
Open the file and read it -- it is deliberately short.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ----------------------------------------------------------------------------- loading
def load_raw_farm():
    """The messy farm records exactly as exported (everything the way it arrived)."""
    return pd.read_csv(DATA / "farm_records_raw.csv")


def load_lesson_clean():
    """Farm records after Lessons 1-2 (structurally clean: types, duplicates, spellings,
    impossible values -> NaN). Statistical steps (imputation, scaling...) are still to do.
    Created by notebook 02. If the file is missing, run notebook 02 first."""
    path = DATA / "farm_clean_lesson.csv"
    if not path.exists():
        raise FileNotFoundError("Run notebook 02_data_cleaning_part2.ipynb first - it creates this file.")
    return pd.read_csv(path, parse_dates=["planting_date", "harvest_date"])


def load_prices():
    """Daily commodity prices (long format: one row per date per crop)."""
    return pd.read_csv(DATA / "crop_prices_daily.csv", parse_dates=["date"])


def load_spectra():
    """Soil near-infrared spectra: 40 correlated bands + organic carbon target."""
    return pd.read_csv(DATA / "soil_spectra.csv")


# ----------------------------------------------------------------------------- profiling
def profile(df):
    """One-glance data-quality table: dtype, missing, unique, example value."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(1),
        "n_unique": df.nunique(),
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None for c in df.columns],
    })


def hr(title="", width=70):
    """Print a visual separator -- makes long outputs easier to read."""
    print("\n" + "=" * width)
    if title:
        print(title)
        print("=" * width)


# ----------------------------------------------------------------------------- plotting
def setup_plots():
    """A clean, readable matplotlib style used in all notebooks."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    plt.rcParams.update({
        "figure.figsize": (8, 4.2), "figure.dpi": 100,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.frameon": False,
        "axes.prop_cycle": plt.cycler(color=["#2E7D32", "#EF6C00", "#1565C0", "#C62828", "#6A1B9A", "#00838F"]),
    })


def side_by_side(series_a, series_b, title_a="before", title_b="after", bins=40):
    """Two histograms next to each other - the classic 'before / after' picture."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(pd.Series(series_a).dropna(), bins=bins)
    ax[0].set_title(title_a)
    ax[1].hist(pd.Series(series_b).dropna(), bins=bins, color="#EF6C00")
    ax[1].set_title(title_b)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------- feature recipes (Lesson 04)
CROP_OPTIMUM = {  # agronomy-handbook reference values: crop -> (optimal rainfall mm, optimal temperature C)
    "Maize": (650, 21), "Beans": (450, 20), "Wheat": (500, 17), "Potato": (600, 16), "Sorghum": (400, 24)}
PEST_ORDER = ["Low", "Medium", "High"]
EXPERIENCE_ORDER = ["Beginner", "Intermediate", "Advanced", "Expert"]


def build_farm_features(df):
    """The feature recipes from Lesson 04, packaged so later lessons can reuse them.

    Every recipe is a deterministic row-wise formula - nothing is *learned* from the data,
    so it is safe to apply before splitting into train/test.
    """
    fe = df.copy()
    fe["plant_month"] = fe["planting_date"].dt.month
    fe["plant_year"] = fe["planting_date"].dt.year
    fe["grow_days"] = (fe["harvest_date"] - fe["planting_date"]).dt.days
    fe["npk_total"] = fe["nitrogen_ppm"] + fe["phosphorus_ppm"] + fe["potassium_ppm"]
    fe["n_to_p_ratio"] = fe["nitrogen_ppm"] / (fe["phosphorus_ppm"] + 1)
    fe["moisture_index"] = fe["rainfall_mm"] / fe["avg_temp_c"]
    fe["fert_x_irrig"] = fe["fertilizer_kg_ha"] * fe["irrigation"]
    fe["log_fert"] = np.log1p(fe["fertilizer_kg_ha"])
    fe["log_farm_size"] = np.log1p(fe["farm_size_ha"])
    fe["rain_gap_sq"] = (fe["rainfall_mm"] - fe["crop"].map({k: v[0] for k, v in CROP_OPTIMUM.items()})) ** 2
    fe["temp_gap_sq"] = (fe["avg_temp_c"] - fe["crop"].map({k: v[1] for k, v in CROP_OPTIMUM.items()})) ** 2
    fe["ph_gap_sq"] = (fe["soil_ph"] - 6.3) ** 2
    fe["pest_ord"] = fe["pest_pressure"].map({k: i for i, k in enumerate(PEST_ORDER)})
    fe["exp_ord"] = fe["experience"].map({k: i for i, k in enumerate(EXPERIENCE_ORDER)})
    return fe


# ----------------------------------------------------------------------------- project self-checks
def check_clean(df, verbose=True):
    """Self-check for the capstone: has the raw farm data been cleaned properly? Returns a table of checks."""
    checks = {}
    def safe(name, fn):
        try:
            checks[name] = bool(fn())
        except Exception:
            checks[name] = False
    safe("no exact duplicate rows", lambda: not df.duplicated().any())
    safe("one record per farm_id + planting_date", lambda: not df.duplicated(["farm_id", "planting_date"]).any())
    safe("planting_date / harvest_date are datetimes", lambda: pd.api.types.is_datetime64_any_dtype(df["planting_date"])
         and pd.api.types.is_datetime64_any_dtype(df["harvest_date"]))
    safe("fertilizer_kg_ha is numeric", lambda: pd.api.types.is_numeric_dtype(df["fertilizer_kg_ha"]))
    safe("exactly 5 crop labels", lambda: df["crop"].nunique() == 5)
    safe("exactly 10 county labels", lambda: df["county"].nunique() == 10)
    safe("irrigation is 0/1", lambda: set(df["irrigation"].dropna().unique()) <= {0, 1, 0.0, 1.0})
    safe("soil_ph within 3-10", lambda: df["soil_ph"].dropna().between(3, 10).all())
    safe("rainfall within 50-3000 mm", lambda: df["rainfall_mm"].dropna().between(50, 3000).all())
    safe("avg_temp within -5..40 C", lambda: df["avg_temp_c"].dropna().between(-5, 40).all())
    safe("farm_size_ha > 0", lambda: (df["farm_size_ha"].dropna() > 0).all())
    safe("target has no NaN and is positive", lambda: df["yield_t_ha"].notna().all() and (df["yield_t_ha"] > 0).all())
    safe("harvest after planting (where present)", lambda: (df["harvest_date"].dropna() >= df.loc[df["harvest_date"].notna(), "planting_date"]).all())
    out = pd.Series(checks, name="passed")
    if verbose:
        for k, v in out.items():
            print(("  \u2705 " if v else "  \u274c ") + k)
        print(f"\n{int(out.sum())}/{len(out)} checks passed")
    return out


def check_ml_ready(X, verbose=True):
    """Self-check for the capstone: is this matrix ready to hand to a model?"""
    X = pd.DataFrame(X)
    checks = {
        "all columns numeric": all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes),
        "no missing values": int(X.isna().sum().sum()) == 0,
        "no infinite values": not np.isinf(X.select_dtypes("number")).any().any(),
        "no constant columns": not (X.nunique() <= 1).any(),
        "unique column names": X.columns.is_unique,
    }
    out = pd.Series(checks, name="passed")
    if verbose:
        for k, v in out.items():
            print(("  \u2705 " if v else "  \u274c ") + k)
        print(f"\n{int(out.sum())}/{len(out)} checks passed  |  shape {X.shape}")
    return out
