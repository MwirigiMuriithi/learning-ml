"""
generate_datasets.py
====================
Creates the three datasets used throughout Phase 2.

    python generate_datasets.py

All data is SYNTHETIC (simulated), but it is built to behave like real
East-African smallholder farm data: right-skewed farm sizes, yields that depend
on rainfall / temperature / soil pH / fertilizer, and -- importantly for this
course -- the kinds of *mess* you meet in real life: typos, mixed units,
duplicates, impossible values, missing values and leaky columns.

If you have the real Phase 1 agricultural dataset, you can use it instead in the
project. The skills are identical. The simulated data simply guarantees that every
problem in this course has something interesting to find.

The random seed is fixed, so everybody gets exactly the same files.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).parent
rng = np.random.default_rng(2024)


# ----------------------------------------------------------------------------
# 1.  FARM RECORDS  (the main dataset)
# ----------------------------------------------------------------------------
def make_farm_records(n_farms=600, seasons_per_farm=4):
    counties = {
        #  name          (mean seasonal rain mm, mean temp C)
        "Uasin Gishu": (650, 17.0), "Trans Nzoia": (700, 18.0),
        "Nakuru": (500, 17.5),      "Kakamega": (800, 21.0),
        "Bungoma": (780, 21.0),     "Meru": (600, 19.0),
        "Machakos": (420, 22.0),    "Nyeri": (650, 16.0),
        "Kisii": (900, 19.0),       "Kiambu": (700, 18.0),
    }
    county_names = list(counties)
    crops = ["Maize", "Beans", "Wheat", "Potato", "Sorghum"]
    base_yield = {"Maize": 4.0, "Beans": 1.3, "Wheat": 3.6, "Potato": 14.0, "Sorghum": 2.3}
    opt_rain = {"Maize": 650, "Beans": 450, "Wheat": 500, "Potato": 600, "Sorghum": 400}
    opt_temp = {"Maize": 21, "Beans": 20, "Wheat": 17, "Potato": 16, "Sorghum": 24}
    duration = {"Maize": 160, "Beans": 90, "Wheat": 130, "Potato": 110, "Sorghum": 120}
    base_price = {"Maize": 55, "Beans": 130, "Wheat": 60, "Potato": 35, "Sorghum": 50}
    soils = ["Loam", "Clay", "Sandy", "Silt", "Volcanic"]

    # ---- farm-level (fixed) attributes
    farms = pd.DataFrame({
        "farm_id": [f"F{i:04d}" for i in range(1, n_farms + 1)],
        "county": rng.choice(county_names, n_farms),
        "soil_type": rng.choice(soils, n_farms, p=[.30, .22, .15, .13, .20]),
        "farm_size_ha": np.round(rng.lognormal(0.1, 0.85, n_farms), 2),
        "experience": rng.choice(["Beginner", "Intermediate", "Advanced", "Expert"],
                                 n_farms, p=[.25, .40, .25, .10]),
    })

    # ---- one row per farm per season
    rows = farms.loc[farms.index.repeat(seasons_per_farm)].reset_index(drop=True)
    n = len(rows)
    rows["crop"] = rng.choice(crops, n, p=[.38, .20, .12, .18, .12])
    # each farm plants in 4 DIFFERENT (year, season) slots -> farm_id + planting_date is a true natural key
    combos = np.array([(y, s) for y in range(2019, 2024) for s in (0, 1)])      # s=0 long rains, s=1 short rains
    pick = np.concatenate([rng.choice(len(combos), seasons_per_farm, replace=False) for _ in range(n_farms)])
    year = combos[pick, 0]
    long_rains = combos[pick, 1] == 0
    month = np.where(long_rains, rng.integers(3, 5, n), rng.integers(10, 12, n))
    day = rng.integers(1, 28, n)
    plant = pd.to_datetime(pd.DataFrame({"year": year, "month": month, "day": day}))
    dur = rows["crop"].map(duration).to_numpy() + rng.integers(-10, 11, n)
    rows["planting_date"] = plant
    rows["harvest_date"] = plant + pd.to_timedelta(dur, unit="D")
    rows["season"] = np.where(long_rains, "Long rains", "Short rains")

    mean_rain = rows["county"].map({k: v[0] for k, v in counties.items()}).to_numpy()
    mean_temp = rows["county"].map({k: v[1] for k, v in counties.items()}).to_numpy()
    rows["rainfall_mm"] = np.round(np.clip(rng.normal(mean_rain, 120, n), 150, None), 1)
    rows["avg_temp_c"] = np.round(rng.normal(mean_temp, 1.2, n), 1)

    rows["soil_ph"] = np.round(np.clip(rng.normal(6.0, 0.6, n), 4.3, 8.0), 2)
    rows["nitrogen_ppm"] = np.round(np.clip(rng.normal(30, 10, n), 5, None), 1)
    rows["phosphorus_ppm"] = np.round(rng.lognormal(2.5, 0.5, n), 1)
    rows["potassium_ppm"] = np.round(np.clip(rng.normal(150, 40, n), 30, None), 1)
    rows["organic_matter_pct"] = np.round(np.clip(rng.normal(3.2, 1.0, n), 0.5, 7), 2)
    fert = np.clip(rng.normal(80, 40, n), 0, 250)
    fert[rng.random(n) < 0.15] = 0
    rows["fertilizer_kg_ha"] = np.round(fert, 0)
    rows["irrigation"] = rng.random(n) < 0.25
    rows["pest_pressure"] = rng.choice(["Low", "Medium", "High"], n, p=[.45, .35, .20])

    # ---- the (hidden) yield formula the models will try to learn
    rain_eff = rows["rainfall_mm"] + np.where(rows["irrigation"], 120, 0)
    f_rain = np.exp(-((rain_eff - rows["crop"].map(opt_rain)) / (0.65 * rows["crop"].map(opt_rain))) ** 2)
    f_temp = np.exp(-((rows["avg_temp_c"] - rows["crop"].map(opt_temp)) / 6.0) ** 2)
    f_ph = np.exp(-((rows["soil_ph"] - 6.3) / 1.2) ** 2)
    f_fert = 0.70 + 0.30 * (1 - np.exp(-rows["fertilizer_kg_ha"] / 70))
    f_om = 0.85 + 0.05 * rows["organic_matter_pct"]
    f_pest = rows["pest_pressure"].map({"Low": 1.0, "Medium": 0.88, "High": 0.70})
    f_exp = rows["experience"].map({"Beginner": 0.90, "Intermediate": 1.0, "Advanced": 1.07, "Expert": 1.12})
    noise = rng.lognormal(0, 0.12, n)
    rows["yield_t_ha"] = np.round(
        rows["crop"].map(base_yield) * f_rain * f_temp * f_ph * f_fert * f_om * f_pest * f_exp * noise * 1.35, 2)

    price = rows["crop"].map(base_price) * (1.06 ** (year - 2019)) * rng.lognormal(0, 0.10, n)
    rows["market_price_ksh_kg"] = np.round(price, 1)
    # LEAKY column: only known AFTER harvest AND sale (used in Lesson 6)
    rows["total_revenue_ksh"] = np.round(rows["yield_t_ha"] * rows["farm_size_ha"] * 1000 * rows["market_price_ksh_kg"], 0)

    # sensible column order
    cols = ["farm_id", "county", "soil_type", "farm_size_ha", "experience", "crop", "season",
            "planting_date", "harvest_date", "rainfall_mm", "avg_temp_c", "soil_ph", "nitrogen_ppm",
            "phosphorus_ppm", "potassium_ppm", "organic_matter_pct", "fertilizer_kg_ha", "irrigation",
            "pest_pressure", "market_price_ksh_kg", "total_revenue_ksh", "yield_t_ha"]
    return rows[cols]


def corrupt(df):
    """Inject realistic mess. Returns an object-heavy dataframe like a messy CSV export."""
    df = df.copy()
    n = len(df)
    idx = lambda frac: rng.choice(n, int(frac * n), replace=False)

    # ---------- text-ify some columns (they'll come back from CSV as strings)
    df["planting_date"] = df["planting_date"].dt.strftime("%Y-%m-%d")
    df["harvest_date"] = df["harvest_date"].dt.strftime("%Y-%m-%d")
    # a few dates in another format
    for col in ["planting_date", "harvest_date"]:
        j = idx(0.04)
        df.loc[j, col] = pd.to_datetime(df.loc[j, col]).dt.strftime("%d %b %Y")
    df["fertilizer_kg_ha"] = df["fertilizer_kg_ha"].astype(int).astype(str)
    j = idx(0.10); df.loc[j, "fertilizer_kg_ha"] = df.loc[j, "fertilizer_kg_ha"] + " kg"
    j = idx(0.04); df.loc[j, "fertilizer_kg_ha"] = df.loc[j, "fertilizer_kg_ha"].str.replace(" kg", "") + "kg/ha"
    j = idx(0.05); df.loc[j, "fertilizer_kg_ha"] = rng.choice(["N/A", "unknown", "-"], len(j))
    df["irrigation"] = df["irrigation"].map({True: "Yes", False: "No"})
    j = idx(0.30)
    yes_v, no_v = ["yes", "Y", "TRUE", "1"], ["no", "N", "FALSE", "0"]
    df.loc[j, "irrigation"] = [rng.choice(yes_v) if v == "Yes" else rng.choice(no_v) for v in df.loc[j, "irrigation"]]

    # ---------- inconsistent category spellings
    def mess(series, variants, frac):
        s = series.copy()
        j = idx(frac)
        s.loc[j] = [rng.choice(variants.get(v, [v])) for v in s.loc[j]]
        return s
    df["crop"] = mess(df["crop"], {
        "Maize": ["maize", "MAIZE", " Maize", "Maize ", "Corn", "maize."],
        "Beans": ["beans", "Bean", "BEANS", "Beans "],
        "Wheat": ["wheat", "WHEAT", "Wheat "],
        "Potato": ["potato", "Potatoes", "Irish potato", "POTATO"],
        "Sorghum": ["sorghum", "Sorgum", "SORGHUM"]}, 0.22)
    df["county"] = mess(df["county"], {
        "Uasin Gishu": ["uasin gishu", "Uasin-Gishu", "UasinGishu", "UASIN GISHU"],
        "Trans Nzoia": ["trans nzoia", "Trans-Nzoia", "TransNzoia"],
        "Nakuru": ["nakuru", "NAKURU", "Nakuru "], "Kakamega": ["kakamega", "KAKAMEGA"],
        "Bungoma": ["bungoma", "Bungoma "], "Meru": ["meru", "MERU"],
        "Machakos": ["machakos", "Machakos "], "Nyeri": ["nyeri", "NYERI"],
        "Kisii": ["kisii", "Kisii "], "Kiambu": ["kiambu", "KIAMBU"]}, 0.18)
    df["soil_type"] = mess(df["soil_type"], {
        "Loam": ["loam", "LOAM", "Loamy"], "Clay": ["clay", "CLAY"],
        "Sandy": ["sandy", "Sand"], "Silt": ["silt", "Silty"], "Volcanic": ["volcanic", "VOLCANIC"]}, 0.12)
    df["pest_pressure"] = mess(df["pest_pressure"], {"Low": ["low", "LOW"], "Medium": ["medium", "med", "Med"],
                                                     "High": ["high", "HIGH"]}, 0.10)

    # ---------- invalid values (impossible in the real world)
    df.loc[idx(0.004), "rainfall_mm"] = -rng.integers(10, 400)
    j = idx(0.005); df.loc[j, "soil_ph"] = rng.choice([0, 14.5, -1, 99], len(j))
    j = idx(0.003); df.loc[j, "avg_temp_c"] = rng.choice([99.0, -999.0], len(j))   # sentinel codes
    df.loc[idx(0.003), "yield_t_ha"] = -rng.uniform(0.5, 3)
    df.loc[idx(0.003), "farm_size_ha"] = rng.choice([0, -2.5])
    j = idx(0.006)                                                                  # harvest BEFORE planting
    df.loc[j, "harvest_date"] = "2018-01-15"

    # ---------- outliers (extreme but "possible") and data-entry typos
    j = idx(0.006); df.loc[j, "farm_size_ha"] = np.round(rng.uniform(80, 400, len(j)), 1)
    j = idx(0.003); df.loc[j, "fertilizer_kg_ha"] = (rng.integers(1500, 3000, len(j))).astype(str)
    j = idx(0.003); df.loc[j, "yield_t_ha"] = df.loc[j, "yield_t_ha"] * 10           # misplaced decimal point
    j = idx(0.004); df.loc[j, "nitrogen_ppm"] = rng.uniform(150, 300, len(j))

    # ---------- missing values  (some random, some depend on other columns)
    for col, frac in {"soil_ph": .06, "nitrogen_ppm": .05, "phosphorus_ppm": .05, "potassium_ppm": .05,
                      "rainfall_mm": .04, "avg_temp_c": .03, "organic_matter_pct": .07, "experience": .05,
                      "pest_pressure": .04, "farm_size_ha": .02, "market_price_ksh_kg": .03,
                      "season": .02, "soil_type": .03}.items():
        df.loc[idx(frac), col] = np.nan
    # MAR: small farms are much less likely to report fertilizer use
    small = (df["farm_size_ha"] < 1).fillna(False).to_numpy()
    miss = small & (rng.random(n) < 0.20)
    df.loc[miss, "fertilizer_kg_ha"] = np.nan
    df.loc[idx(0.015), "yield_t_ha"] = np.nan                                         # missing TARGET

    # ---------- duplicates
    exact = df.sample(60, random_state=1)
    near = df.sample(25, random_state=2).copy()                                       # same record, sloppy typing
    near["crop"] = near["crop"].astype(str).str.upper()
    near["county"] = near["county"].astype(str).str.lower()
    alt = rng.random(len(near)) < 0.5                                                # half re-typed in another date format
    near.loc[alt, "planting_date"] = pd.to_datetime(near.loc[alt, "planting_date"], format="mixed").dt.strftime("%d %b %Y")
    out = pd.concat([df, exact, near], ignore_index=True)
    out = out.sample(frac=1, random_state=7).reset_index(drop=True)                   # shuffle
    return out


# ----------------------------------------------------------------------------
# 2.  DAILY CROP PRICES  (time-series feature engineering)
# ----------------------------------------------------------------------------
def make_prices():
    dates = pd.date_range("2019-01-01", "2023-12-31", freq="D")
    t = np.arange(len(dates))
    spec = {   # base price, seasonal amplitude, month of seasonal PEAK (pre-harvest scarcity)
        "Maize": (48, 0.18, 7), "Beans": (115, 0.12, 5), "Potato": (30, 0.22, 9)}
    frames = []
    for crop, (base, amp, peak) in spec.items():
        doy = dates.dayofyear.to_numpy()
        season = 1 + amp * np.cos(2 * np.pi * (doy - peak * 30.4) / 365.25)
        trend = 1 + 0.07 * t / 365.25
        ar = np.zeros(len(t)); e = rng.normal(0, 0.012, len(t))
        for i in range(1, len(t)):
            ar[i] = 0.97 * ar[i - 1] + e[i]
        price = base * trend * season * np.exp(ar)
        vol = np.clip(rng.normal(120, 25, len(t)) / (price / base) ** 0.8, 10, None)
        f = pd.DataFrame({"date": dates, "crop": crop, "price_ksh_kg": np.round(price, 2),
                          "volume_tonnes": np.round(vol, 1)})
        f.loc[rng.choice(len(f), 30, replace=False), "price_ksh_kg"] = np.nan        # market closed / not reported
        spikes = rng.choice(len(f), 4, replace=False)
        f.loc[spikes, "price_ksh_kg"] = f.loc[spikes, "price_ksh_kg"] * 1.8          # data-entry spikes
        frames.append(f)
    out = pd.concat(frames).sort_values(["date", "crop"]).reset_index(drop=True)
    return out


# ----------------------------------------------------------------------------
# 3.  SOIL SPECTRA  (high-dimensional, highly correlated -> PCA / selection)
# ----------------------------------------------------------------------------
def make_spectra(n=700, p=40):
    wl = np.linspace(400, 2400, p)                                   # wavelengths (nm)
    centers = [550, 1100, 1900, 2200]
    W = np.vstack([np.exp(-((wl - c) / 260.0) ** 2) for c in centers])      # 4 smooth "shapes"
    Z = rng.normal(size=(n, 4))
    base = 0.35 + 0.00004 * (wl - 400)                                        # gentle upward baseline
    X = base + 0.10 * Z @ W + rng.normal(0, 0.01, (n, p))
    soc = 2.2 + 0.55 * Z[:, 0] - 0.35 * Z[:, 1] + 0.25 * Z[:, 2] + rng.normal(0, 0.25, n)
    df = pd.DataFrame(X, columns=[f"band_{int(w)}nm" for w in wl]).round(4)
    df.insert(0, "sample_id", [f"S{i:04d}" for i in range(1, n + 1)])
    df["depth_cm"] = rng.choice(["0-20", "20-50"], n)
    df["organic_carbon_pct"] = np.round(np.clip(soc, 0.2, None), 3)
    return df


if __name__ == "__main__":
    clean = make_farm_records()
    raw = corrupt(clean)
    raw.to_csv(OUT / "farm_records_raw.csv", index=False)
    make_prices().to_csv(OUT / "crop_prices_daily.csv", index=False)
    make_spectra().to_csv(OUT / "soil_spectra.csv", index=False)
    print("farm_records_raw.csv :", raw.shape)
    print("crop_prices_daily.csv:", pd.read_csv(OUT / "crop_prices_daily.csv").shape)
    print("soil_spectra.csv     :", pd.read_csv(OUT / "soil_spectra.csv").shape)
