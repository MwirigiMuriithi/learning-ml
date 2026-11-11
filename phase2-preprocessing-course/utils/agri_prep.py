"""
agri_prep.py  --  reusable scikit-learn transformers for the agricultural data.

You will WRITE these yourself step by step in notebook 07. They live in a module
(not only in a notebook) for one important reason: a saved pipeline (joblib / pickle)
can only be loaded again if Python can import the classes it was built from.
Classes defined inside a notebook live in `__main__` and break when you reload the
model in a different script. Real-world lesson, learned the hard way by many teams.

Every transformer follows the scikit-learn contract:
    fit(X, y=None)   -> LEARN something from the TRAINING data only (store it in attrs ending with `_`)
    transform(X)     -> APPLY what was learned, to ANY data (train, test, tomorrow's new data)
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatures(BaseEstimator, TransformerMixin):
    """Turn the planting date (and optionally the harvest date) into model-friendly numbers.

    Adds: plant_year, plant_month, plant_month_sin, plant_month_cos, plant_dayofyear and
          rain_season (Long / Short / Dry, from the planting month).
    If `harvest_col` is given it also adds grow_days (harvest - planting). Pass harvest_col=None when
    the model must predict at planting time: the actual harvest date is not known yet (see Lesson 6).
    The original date columns are dropped. Nothing is learned from data, so fit() is trivial.
    """
    def __init__(self, plant_col="planting_date", harvest_col="harvest_date"):
        self.plant_col = plant_col
        self.harvest_col = harvest_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        plant = pd.to_datetime(X[self.plant_col])
        X["plant_year"] = plant.dt.year
        X["plant_month"] = plant.dt.month
        X["plant_month_sin"] = np.sin(2 * np.pi * plant.dt.month / 12)
        X["plant_month_cos"] = np.cos(2 * np.pi * plant.dt.month / 12)
        X["plant_dayofyear"] = plant.dt.dayofyear
        X["rain_season"] = np.select([plant.dt.month.isin([3, 4, 5]), plant.dt.month.isin([10, 11, 12])],
                                     ["Long", "Short"], default="Dry")
        drop = [self.plant_col]
        if self.harvest_col is not None:
            harvest = pd.to_datetime(X[self.harvest_col])
            X["grow_days"] = (harvest - plant).dt.days
            drop.append(self.harvest_col)
        return X.drop(columns=drop)


CROP_OPTIMUM = {  # agronomy-handbook reference values: crop -> (optimal seasonal rainfall mm, optimal temperature C)
    "Maize": (650, 21), "Beans": (450, 20), "Wheat": (500, 17), "Potato": (600, 16), "Sorghum": (400, 24)}


class AgriFeatures(BaseEstimator, TransformerMixin):
    """Combine existing columns into agronomically meaningful new ones.

    log_farm_size        : tames the right-skew of farm size
    npk_total            : crude 'total nutrient' indicator
    n_to_p_ratio         : nitrogen / phosphorus balance
    moisture_index       : rainfall per degree Celsius (more rain + cooler = wetter)
    fert_x_irrigation    : fertiliser works better when water is available (interaction)
    rain_gap_sq, temp_gap_sq : squared distance from the crop's ideal rainfall / temperature (hill-shaped effects)
    ph_gap_sq            : squared distance from the ideal soil pH (~6.3)
    Missing inputs simply propagate to NaN; an imputer later in the pipeline fills them.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["log_farm_size"] = np.log1p(X["farm_size_ha"])
        X["npk_total"] = X["nitrogen_ppm"] + X["phosphorus_ppm"] + X["potassium_ppm"]
        X["n_to_p_ratio"] = X["nitrogen_ppm"] / (X["phosphorus_ppm"] + 1)
        X["moisture_index"] = X["rainfall_mm"] / X["avg_temp_c"]
        X["fert_x_irrigation"] = X["fertilizer_kg_ha"] * X["irrigation"]
        X["rain_gap_sq"] = (X["rainfall_mm"] - X["crop"].map({k: v[0] for k, v in CROP_OPTIMUM.items()})) ** 2
        X["temp_gap_sq"] = (X["avg_temp_c"] - X["crop"].map({k: v[1] for k, v in CROP_OPTIMUM.items()})) ** 2
        X["ph_gap_sq"] = (X["soil_ph"] - 6.3) ** 2
        return X


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap extreme values at Tukey fences learned from the TRAINING data.

        lower = Q1 - k * IQR        upper = Q3 + k * IQR
    fit()  learns the fences      -> only ever sees training data
    transform() clips new data to those SAME fences.

    Guard: if a column's IQR is 0 (binary flags, or columns that are 0 for 75%+ of rows) the fences
    would collapse onto a single value and wipe out all variation. Such columns are left uncapped.
    """
    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.cols_ = np.asarray(X.columns, dtype=object)
        q1 = np.nanpercentile(X.to_numpy(float), 25, axis=0)
        q3 = np.nanpercentile(X.to_numpy(float), 75, axis=0)
        iqr = q3 - q1
        degenerate = iqr == 0
        self.lower_ = np.where(degenerate, -np.inf, q1 - self.k * iqr)
        self.upper_ = np.where(degenerate, np.inf, q3 + self.k * iqr)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return self.cols_
