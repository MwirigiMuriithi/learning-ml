"""
generate_datasets.py - re-creates every dataset used in the course.

All synthetic datasets are produced from fixed random seeds, so you always get
exactly the same files.  Run it from this folder:

    python generate_datasets.py
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn import datasets as skd

OUT = pathlib.Path(__file__).resolve().parent


def students(n=200, seed=1):
    rng = np.random.default_rng(seed)
    hours = rng.uniform(0.5, 10, n)
    sleep = np.clip(rng.normal(7, 1, n), 4, 10)
    prev = np.clip(rng.normal(65, 12, n), 30, 100)
    attend = np.clip(rng.normal(85, 10, n), 40, 100)
    score = (35 + 4.0 * hours + 0.25 * (prev - 65) + 0.10 * (attend - 85)
             + 1.0 * (sleep - 7) + rng.normal(0, 4, n))
    return pd.DataFrame({
        "hours_studied": hours.round(1), "sleep_hours": sleep.round(1),
        "prev_score": prev.round(0), "attendance_pct": attend.round(0),
        "exam_score": np.clip(score, 0, 100).round(1)})


def housing(n=900, seed=2):
    rng = np.random.default_rng(seed)
    area = np.clip(rng.normal(120, 40, n), 40, 300)
    bedrooms = np.clip(np.round(area / 45 + rng.normal(0, 0.6, n)), 1, 6)
    bathrooms = np.clip(np.round(bedrooms * 0.6 + rng.normal(0, 0.5, n)), 1, 4)
    age = rng.integers(0, 61, n)
    dist = np.clip(rng.exponential(8, n), 0.5, 40)
    garden = rng.binomial(1, 0.4, n)
    quality = rng.choice([1, 2, 3, 4, 5], n, p=[.08, .22, .34, .24, .12])
    price = (40 + 2.4 * area + 0.004 * area ** 2 + 12 * bathrooms + 25 * garden
             + 30 * quality + 0.15 * quality * area - 1.8 * age - 4.5 * dist)
    price = price * (1 + rng.normal(0, 0.06, n))
    return pd.DataFrame({
        "area_sqm": area.round(1), "bedrooms": bedrooms.astype(int),
        "bathrooms": bathrooms.astype(int), "age_years": age,
        "distance_to_center_km": dist.round(1), "has_garden": garden,
        "neighborhood_quality": quality, "price_k": np.clip(price, 50, None).round(1)})


def customers(n_each=100, seed=3):
    rng = np.random.default_rng(seed)
    # (age mean/sd, income mean/sd, spending mean/sd, visits mean/sd)
    segs = [((48, 7), (85, 10), (22, 7), (3, 1)),        # 0 savers
            ((34, 6), (88, 10), (80, 8), (9, 2)),        # 1 VIPs
            ((23, 3), (30, 7), (75, 9), (11, 2.5)),      # 2 trendy youth
            ((45, 8), (32, 7), (25, 8), (4, 1.5))]       # 3 frugal
    rows = []
    for k, (a, i, s, v) in enumerate(segs):
        rows.append(pd.DataFrame({
            "age": np.clip(rng.normal(*a, n_each), 18, 80).round(0),
            "annual_income_k": np.clip(rng.normal(*i, n_each), 12, 150).round(1),
            "spending_score": np.clip(rng.normal(*s, n_each), 1, 100).round(0),
            "visits_per_month": np.clip(rng.normal(*v, n_each), 1, 20).round(1),
            "segment_truth": k}))
    return pd.concat(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def churn(n=1500, seed=4, target_rate=0.17):
    rng = np.random.default_rng(seed)
    tenure = np.clip(rng.exponential(24, n), 1, 72).round(0)
    charge = np.clip(rng.normal(70, 20, n), 20, 120).round(2)
    calls = rng.poisson(1.5, n)
    contract = rng.choice([1, 12, 24], n, p=[.55, .25, .20])
    late = rng.poisson(0.7, n)
    usage = np.clip(rng.lognormal(3.3, 0.6, n), 1, 400).round(1)
    z = (-0.04 * tenure + 0.02 * (charge - 70) + 0.35 * calls + 0.5 * late
         - 1.2 * (contract == 12) - 2.0 * (contract == 24) - 0.004 * usage)
    lo, hi = -10, 10
    for _ in range(60):  # bisection on the intercept to hit the target rate
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(z + mid)))).mean()
        lo, hi = (mid, hi) if rate < target_rate else (lo, mid)
    p = 1 / (1 + np.exp(-(z + mid)))
    return pd.DataFrame({
        "tenure_months": tenure.astype(int), "monthly_charge": charge,
        "support_calls": calls, "contract_months": contract,
        "late_payments": late, "usage_gb": usage,
        "churned": rng.binomial(1, p)})


def used_cars(n=1200, seed=5):
    rng = np.random.default_rng(seed)
    brands = rng.choice(["Toyota", "Honda", "Ford", "Kia", "BMW"], n,
                        p=[.28, .20, .20, .17, .15])
    base = pd.Series(brands).map({"Toyota": 22000, "Honda": 21000, "Ford": 20000,
                                  "Kia": 18000, "BMW": 38000}).values
    year = rng.integers(2008, 2024, n)
    age = 2024 - year
    mileage = np.clip(age * rng.normal(14000, 4000, n) + rng.normal(3000, 1500, n), 500, None)
    engine = rng.choice([1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], n,
                        p=[.08, .14, .17, .22, .14, .13, .08, .04])
    fuel = rng.choice(["petrol", "diesel", "hybrid"], n, p=[.55, .30, .15])
    trans = rng.choice(["manual", "automatic"], n, p=[.55, .45])
    owners = np.clip(rng.poisson(1.0, n) + 1, 1, 4)
    price = (base * np.exp(-0.10 * age) * np.exp(-0.0000045 * mileage)
             * (1 + 0.12 * (trans == "automatic")) * (1 + 0.00008 * (engine - 1600))
             * np.where(fuel == "hybrid", 1.15, np.where(fuel == "diesel", 1.03, 1.0))
             * 0.96 ** (owners - 1) * np.exp(rng.normal(0, 0.08, n)))
    df = pd.DataFrame({
        "brand": brands, "year": year, "mileage_km": mileage.round(0),
        "engine_cc": engine.astype(float), "fuel_type": fuel,
        "transmission": trans, "owners": owners.astype(float),
        "price_usd": price.round(0)})
    for col, frac in [("mileage_km", .04), ("engine_cc", .03), ("owners", .02)]:
        df.loc[rng.random(n) < frac, col] = np.nan      # realistic missing values
    return df


def noisy_curve(n=80, seed=6):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    return pd.DataFrame({"x": x.round(4),
                         "y": (np.sin(2 * np.pi * x) + rng.normal(0, 0.25, n)).round(4)})


def sklearn_sets():
    iris = skd.load_iris(as_frame=True)
    d = iris.frame.copy()
    d.columns = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "target"]
    d["species"] = d["target"].map(dict(enumerate(iris.target_names)))
    yield "iris.csv", d
    wine = skd.load_wine(as_frame=True).frame
    yield "wine.csv", wine
    bc = skd.load_breast_cancer(as_frame=True).frame
    bc["diagnosis"] = bc["target"].map({0: "malignant", 1: "benign"})
    yield "breast_cancer.csv", bc
    diab = skd.load_diabetes(as_frame=True).frame.rename(columns={"target": "disease_progression"})
    yield "diabetes.csv", diab


if __name__ == "__main__":
    tables = {"students.csv": students(), "housing.csv": housing(),
              "customers.csv": customers(), "churn.csv": churn(),
              "used_cars.csv": used_cars(), "noisy_curve.csv": noisy_curve()}
    tables.update(dict(sklearn_sets()))
    for name, df in tables.items():
        df.to_csv(OUT / name, index=False)
        print(f"wrote {name:<20} {df.shape[0]:>5} rows x {df.shape[1]:>2} cols")
