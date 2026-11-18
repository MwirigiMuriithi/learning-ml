"""
make_datasets.py -- builds the three course datasets from scratch.

Why synthetic?  Because with synthetic data *we* know the true recipe that
generated the target, so we can later ask: "did the model discover the recipe?"
That is a luxury real data never gives you.  The data are deliberately messy
(missing values, outliers, skewed targets, interactions) so they behave like
real tabular data.

Run:  python make_datasets.py      (writes CSVs into ../data)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------
# 1. HOUSING  (regression)  -- the main dataset for the Practical Assignment
# --------------------------------------------------------------------------
def make_housing(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # name: (log-price premium, mean distance to centre km, mean school rating, listing weight)
    hoods = {
        "Riverside": (0.35, 4.0, 7.5, 1.0),
        "Hilltop": (0.55, 7.5, 8.2, 0.6),
        "Old Town": (0.20, 1.5, 6.0, 1.0),
        "Greenfield": (0.10, 12.0, 6.8, 0.9),
        "Lakeview": (0.45, 9.0, 7.8, 0.7),
        "Market District": (0.15, 2.5, 5.5, 1.0),
        "Station Heights": (0.25, 5.5, 6.5, 1.0),
        "Parkside": (0.30, 6.0, 7.2, 0.9),
        "Harbor Point": (0.40, 3.0, 7.0, 0.7),
        "Outer Meadows": (-0.15, 18.0, 5.0, 0.8),
        "Industrial East": (-0.25, 10.0, 4.2, 0.6),
        "Sunnydale": (0.00, 14.0, 6.0, 0.9),
    }
    names = list(hoods)
    w = np.array([hoods[k][3] for k in names])
    hood = rng.choice(names, size=n, p=w / w.sum())
    premium = np.array([hoods[h][0] for h in hood])
    dist_mu = np.array([hoods[h][1] for h in hood])
    school_mu = np.array([hoods[h][2] for h in hood])

    distance = np.clip(rng.normal(dist_mu, 0.25 * dist_mu + 0.8), 0.3, 40)

    # property type depends on how far from the centre we are
    logits = np.c_[np.full(n, 0.6) - 0.08 * distance,      # apartment
                   np.full(n, 0.2) + 0.00 * distance,      # townhouse
                   np.full(n, -0.6) + 0.07 * distance]     # detached
    probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    ptype_idx = np.array([rng.choice(3, p=p) for p in probs])
    ptype = np.array(["apartment", "townhouse", "detached"])[ptype_idx]

    size_mu = np.select([ptype == "apartment", ptype == "townhouse"], [np.log(70), np.log(110)], np.log(190))
    size_sd = np.select([ptype == "apartment", ptype == "townhouse"], [0.30, 0.25], 0.30)
    size = np.clip(np.exp(rng.normal(size_mu, size_sd)), 25, 600).round(0)

    bedrooms = np.clip(np.round(size / 38 + rng.normal(0, 0.6, n)), 1, 7).astype(int)
    bathrooms = np.clip(np.round(bedrooms * 0.6 + rng.normal(0, 0.5, n)), 1, None)
    bathrooms = np.minimum(bathrooms, bedrooms).astype(int)

    age_scale = np.where(hood == "Outer Meadows", 8.0, 18.0)
    age = np.clip(rng.exponential(age_scale), 0, 80).round(0)
    renovated = (rng.random(n) < (0.08 + 0.40 * (age > 25))).astype(int)

    transit = np.clip(rng.gamma(2.0, (2 + 0.6 * distance) / 2.0), 1, 90).round(0)
    school = np.clip(rng.normal(school_mu, 1.2), 1, 10).round(1)
    security = np.clip(np.round(rng.normal(school_mu * 0.5 + 1.0, 0.9)), 1, 5).astype(int)

    p_parking = np.select([ptype == "apartment", ptype == "townhouse"], [0.5, 0.85], 0.98)
    has_parking = (rng.random(n) < p_parking).astype(int)
    p_garden = np.select([ptype == "apartment", ptype == "townhouse"], [0.05, 0.6], 0.9)
    has_garden = (rng.random(n) < p_garden).astype(int)
    p_pool = np.where(ptype == "detached", 0.10 + 0.4 * (premium > 0.3), 0.01)
    has_pool = (rng.random(n) < p_pool).astype(int)

    floor = np.where(ptype == "apartment", rng.integers(1, 19, n), 0)
    year = rng.choice(np.arange(2019, 2025), size=n)
    energy = rng.choice(list("ABCDEF"), size=n, p=[0.07, 0.15, 0.26, 0.26, 0.16, 0.10])
    e_eff = pd.Series(energy).map({"A": .06, "B": .04, "C": .02, "D": 0, "E": -.02, "F": -.05}).values

    # ---- the hidden "recipe" for log(price) ----
    is_apt = ptype == "apartment"
    is_det = ptype == "detached"
    logp = (
        np.log(200)
        + premium
        + 0.85 * np.log(size / 80)
        + 0.25 * premium * np.log(size / 80)                     # size matters more in prime areas
        + np.select([ptype == "townhouse", is_det], [0.10, 0.22], 0.0)
        + 0.03 * (bedrooms - 3)
        + 0.05 * (bathrooms - 1)
        - 0.50 * (1 - np.exp(-age / 30)) * (1 - 0.7 * renovated)  # ageing hurts, saturating, renovation helps
        - 0.05 * np.minimum(distance, 10) - 0.01 * np.maximum(distance - 10, 0)
        + np.where(transit <= 8, 0.10, -0.003 * np.minimum(transit - 8, 40))  # threshold effect
        + 0.05 * np.maximum(school - 6, 0)                       # only *good* schools add value
        + 0.03 * (security - 3)
        + has_parking * (0.04 + 0.05 * is_apt)                   # parking matters more for flats
        + 0.05 * has_garden
        + has_pool * np.where(is_det, 0.12, 0.04)
        + np.where(is_apt, 0.012 * np.minimum(floor, 12), 0)
        + e_eff
        + 0.05 * (year - 2019)                                   # market inflation
        + rng.normal(0, 0.10, n)
    )
    price = np.exp(logp)

    # a few mansions / data-entry errors: heavy right tail
    out = rng.random(n) < 0.012
    price[out] *= rng.uniform(1.6, 2.6, out.sum())

    df = pd.DataFrame({
        "listing_id": np.arange(10001, 10001 + n),
        "year_listed": year,
        "neighborhood": hood,
        "property_type": ptype,
        "size_sqm": size,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age_years": age,
        "renovated": renovated,
        "floor": floor,
        "distance_to_center_km": distance.round(2),
        "transit_minutes": transit,
        "school_rating": school,
        "security_rating": security,
        "has_parking": has_parking,
        "has_garden": has_garden,
        "has_pool": has_pool,
        "energy_rating": energy,
        "price_k": price.round(1),   # price in thousands of dollars
    })

    # ---- realistic messiness: missing values (rarer for expensive areas) ----
    low_tier = (premium < 0.2).astype(float)
    df.loc[rng.random(n) < 0.03 + 0.06 * low_tier, "school_rating"] = np.nan
    df.loc[rng.random(n) < 0.04, "age_years"] = np.nan
    df.loc[rng.random(n) < 0.03, "security_rating"] = np.nan
    df.loc[rng.random(n) < 0.08, "energy_rating"] = np.nan
    return df


# --------------------------------------------------------------------------
# 2. LOAN DEFAULT  (binary classification, imbalanced)
# --------------------------------------------------------------------------
def make_loans(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(35, 9, n), 18, 70).round(0)
    income = np.clip(np.exp(rng.normal(np.log(900), 0.55, n)), 150, 15000).round(0)
    employment = np.minimum(rng.exponential(5, n), age - 18).round(1)
    purpose = rng.choice(["business", "education", "emergency", "consumer", "agriculture"], n,
                         p=[0.30, 0.15, 0.20, 0.20, 0.15])
    region = rng.choice(["North", "South", "East", "West", "Central"], n, p=[0.15, 0.2, 0.2, 0.2, 0.25])
    term = rng.choice([3, 6, 12, 18, 24, 36], n, p=[0.15, 0.25, 0.30, 0.12, 0.12, 0.06])
    loan_amount = (income * rng.uniform(0.4, 5.0, n) / 10).round(0) * 10
    collateral = (rng.random(n) < np.where(np.isin(purpose, ["business", "agriculture"]), 0.40, 0.12)).astype(int)
    prev_loans = rng.poisson(2.0, n)
    late = rng.binomial(prev_loans, 0.18)
    util = rng.beta(2, 3, n).round(3)
    sessions = rng.poisson(6, n)
    savings = np.exp(rng.normal(np.log(income * 0.8), 0.9, n))
    savings[rng.random(n) < 0.10] = 0.0
    savings = savings.round(0)

    purpose_risk = pd.Series(purpose).map({"business": 0.2, "education": -0.2, "emergency": 0.5,
                                           "consumer": 0.15, "agriculture": 0.1}).values
    rate = np.clip(9 + 6 * purpose_risk + 1.2 * late + rng.normal(0, 1.5, n), 5, 30).round(1)
    monthly_pay = loan_amount * (1 + rate * term / 1200) / term
    dti = monthly_pay / income

    def logit(b0):
        return (b0
                + 2.2 * np.clip(dti, 0, 1.5)
                + 0.45 * late
                + 1.5 * (util > 0.7)
                - 0.04 * np.minimum(employment, 15)
                - 0.9 * collateral
                + 0.45 * late * (dti > 0.4)                         # interaction
                + purpose_risk
                - 0.7 * np.log1p(savings / income)
                + 0.4 * (age < 25)
                + 0.3 * (sessions < 2))

    lo, hi = -10.0, 5.0
    for _ in range(60):            # bisection: calibrate intercept so ~14% default
        mid = (lo + hi) / 2
        if _sigmoid(logit(mid)).mean() > 0.14:
            hi = mid
        else:
            lo = mid
    p = _sigmoid(logit((lo + hi) / 2))
    default = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "customer_id": np.arange(50001, 50001 + n),
        "age": age, "income_monthly": income, "employment_years": employment,
        "region": region, "loan_purpose": purpose,
        "loan_amount": loan_amount, "term_months": term, "interest_rate_pct": rate,
        "debt_to_income": dti.round(3),
        "num_prev_loans": prev_loans, "num_late_payments": late,
        "credit_utilization": util, "savings_balance": savings,
        "has_collateral": collateral, "app_sessions_per_week": sessions,
        "defaulted": default,
    })
    df.loc[rng.random(n) < 0.12, "savings_balance"] = np.nan
    df.loc[rng.random(n) < 0.05, "employment_years"] = np.nan
    return df


# --------------------------------------------------------------------------
# 3. CROP YIELD  (regression) -- used for the Capstone Project
# --------------------------------------------------------------------------
def make_crops(n: int = 2500, seed: int = 2024) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    crops = {  # base yield t/ha, optimal rain mm, optimal temp C, N saturation (kg/ha)
        "maize": (5.5, 650, 24, 140),
        "wheat": (4.0, 500, 18, 110),
        "beans": (1.8, 450, 21, 40),
        "sorghum": (3.0, 400, 27, 80),
        "potato": (18.0, 600, 17, 160),
    }
    crop = rng.choice(list(crops), n, p=[0.30, 0.22, 0.18, 0.12, 0.18])
    base = np.array([crops[c][0] for c in crop])
    rain_opt = np.array([crops[c][1] for c in crop])
    temp_opt = np.array([crops[c][2] for c in crop])
    n_sat = np.array([crops[c][3] for c in crop])

    region = rng.choice(["Highlands", "Rift Valley", "Coastal", "Lake Basin", "Semi-arid", "Plateau"], n,
                        p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.2])
    alt_mu = pd.Series(region).map({"Highlands": 2100, "Rift Valley": 1700, "Coastal": 50,
                                    "Lake Basin": 1200, "Semi-arid": 900, "Plateau": 1500}).values
    rain_mu = pd.Series(region).map({"Highlands": 900, "Rift Valley": 650, "Coastal": 1000,
                                     "Lake Basin": 1100, "Semi-arid": 350, "Plateau": 700}).values
    altitude = np.clip(rng.normal(alt_mu, 150), 0, 3000).round(0)
    rainfall = np.clip(rng.normal(rain_mu, 0.25 * rain_mu), 80, 1800).round(0)
    temp = np.clip(27 - 0.0055 * altitude + rng.normal(0, 1.5, n), 8, 34).round(1)
    humidity = np.clip(rng.normal(62, 12, n) + 0.01 * (rainfall - 600), 20, 95).round(1)
    sunshine = np.clip(rng.normal(6.5, 1.3, n) - 0.002 * (rainfall - 600), 2, 11).round(1)
    soil = rng.choice(["loam", "clay", "sandy", "silt", "volcanic"], n, p=[0.30, 0.25, 0.2, 0.1, 0.15])
    ph = np.clip(rng.normal(6.2, 0.7, n), 4.0, 8.5).round(2)
    irrigated = (rng.random(n) < np.where(region == "Semi-arid", 0.5, 0.2)).astype(int)
    nitrogen = np.clip(rng.gamma(4, 30, n), 0, 400).round(0)
    phosphorus = np.clip(rng.gamma(3, 12, n), 0, 150).round(0)
    density = np.clip(rng.normal(1.0, 0.15, n), 0.5, 1.6).round(2)
    pest = np.clip(rng.beta(2, 5, n), 0, 1).round(3)
    variety = rng.choice(["standard", "improved", "hybrid"], n, p=[0.35, 0.40, 0.25])
    field = np.clip(np.exp(rng.normal(np.log(1.5), 0.8, n)), 0.1, 30).round(2)
    year = rng.integers(2015, 2025, n)

    eff_rain = rainfall + irrigated * 250                     # irrigation tops up water
    water_stress = np.exp(-((eff_rain - rain_opt) / (0.80 * rain_opt)) ** 2)
    temp_f = 0.35 + 0.65 * np.exp(-((temp - temp_opt) / 7.0) ** 2)
    ph_f = 0.55 + 0.45 * np.exp(-((ph - 6.4) / 1.2) ** 2)
    n_f = 1 - np.exp(-nitrogen / (0.6 * n_sat))                # diminishing returns (Mitscherlich)
    n_f = n_f * (0.35 + 0.65 * water_stress)                   # fertiliser only works with enough water
    p_f = 0.85 + 0.15 * (1 - np.exp(-phosphorus / 25))
    pest_f = 1 - 0.55 * pest ** 1.3
    var_f = pd.Series(variety).map({"standard": 1.0, "improved": 1.12, "hybrid": 1.25}).values
    soil_f = pd.Series(soil).map({"loam": 1.05, "clay": 0.95, "sandy": 0.85, "silt": 1.0, "volcanic": 1.10}).values
    dens_f = 1 - 0.6 * (density - 1.0) ** 2
    sun_f = 0.8 + 0.2 * np.clip((sunshine - 4) / 4, 0, 1)

    y = (base * (0.25 + 0.75 * water_stress) * temp_f * ph_f * (0.3 + 0.7 * n_f) * p_f
         * pest_f * var_f * soil_f * dens_f * sun_f)
    y = y * np.exp(rng.normal(0, 0.07, n))                     # multiplicative noise
    y = np.clip(y * 1.6, 0.05, None)   # rescale to realistic t/ha

    df = pd.DataFrame({
        "field_id": np.arange(1, n + 1), "year": year, "region": region, "crop": crop, "soil_type": soil,
        "seed_variety": variety, "altitude_m": altitude, "rainfall_mm": rainfall, "avg_temp_c": temp,
        "humidity_pct": humidity, "sunshine_hours": sunshine, "soil_ph": ph, "irrigated": irrigated,
        "nitrogen_kg_ha": nitrogen, "phosphorus_kg_ha": phosphorus, "planting_density": density,
        "pest_pressure": pest, "field_size_ha": field, "yield_t_ha": y.round(3),
    })
    df.loc[rng.random(n) < 0.05, "soil_ph"] = np.nan
    df.loc[rng.random(n) < 0.04, "humidity_pct"] = np.nan
    df.loc[rng.random(n) < 0.06, "phosphorus_kg_ha"] = np.nan
    return df


DATA_DICTIONARY = """# Data dictionary

All three datasets are **synthetic** (generated by `utils/make_datasets.py` with fixed seeds),
so every learner gets identical numbers.  They are intentionally messy: missing values,
skew, outliers, non-linear effects and interactions.

## housing.csv  (3,000 rows)  -- regression, target `price_k`
Listings from a fictional city, "Eastbridge".  Prices are in **thousands of dollars**.

| column | meaning |
|---|---|
| listing_id | unique id (not a feature!) |
| year_listed | 2019-2024 (market inflates ~5%/yr) |
| neighborhood | 12 neighbourhoods (categorical) |
| property_type | apartment / townhouse / detached |
| size_sqm | floor area in square metres |
| bedrooms, bathrooms | counts |
| age_years | building age (4% missing) |
| renovated | 1 if renovated |
| floor | floor of apartment (0 for houses) |
| distance_to_center_km | distance to the city centre |
| transit_minutes | walk to nearest station |
| school_rating | 1-10 (about 6% missing) |
| security_rating | 1-5 (3% missing) |
| has_parking, has_garden, has_pool | 0/1 |
| energy_rating | A (best) - F (worst), 8% missing |
| **price_k** | **target** -- asking price, $ thousands |

## loan_default.csv  (5,000 rows)  -- classification, target `defaulted` (~14% positive)
Small short-term loans.

| column | meaning |
|---|---|
| customer_id | unique id (not a feature!) |
| age, income_monthly, employment_years | applicant profile |
| region, loan_purpose | categoricals |
| loan_amount, term_months, interest_rate_pct | loan terms |
| debt_to_income | monthly repayment / monthly income |
| num_prev_loans, num_late_payments | credit history |
| credit_utilization | 0-1 |
| savings_balance | (12% missing) |
| has_collateral | 0/1 |
| app_sessions_per_week | behavioural signal |
| **defaulted** | **target**, 1 = defaulted |

## crop_yield.csv  (2,500 rows)  -- regression, target `yield_t_ha`  (Capstone)
Field-level harvest records across 5 crops and 6 regions.

| column | meaning |
|---|---|
| field_id | unique id (not a feature!) |
| year | 2015-2024 |
| region, crop, soil_type, seed_variety | categoricals |
| altitude_m, rainfall_mm, avg_temp_c, humidity_pct, sunshine_hours | climate |
| soil_ph, nitrogen_kg_ha, phosphorus_kg_ha | soil / fertiliser |
| irrigated | 0/1 |
| planting_density, pest_pressure, field_size_ha | farm practice |
| **yield_t_ha** | **target**, tonnes per hectare |
"""


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    make_housing().to_csv(DATA_DIR / "housing.csv", index=False)
    make_loans().to_csv(DATA_DIR / "loan_default.csv", index=False)
    make_crops().to_csv(DATA_DIR / "crop_yield.csv", index=False)
    (DATA_DIR / "DATA_DICTIONARY.md").write_text(DATA_DICTIONARY)
    print("Wrote datasets to", DATA_DIR)


if __name__ == "__main__":
    main()
