"""
generate_datasets.py
====================
Reproducible generator for every dataset used in the course.

All data are SYNTHETIC. They are inspired by the structure and rhythm of Kenyan
agricultural markets (two rainy seasons, the 2021-2022 drought, steady food
inflation, higher prices in big-city retail markets) but they are NOT official
statistics. Never use them to draw real-world conclusions.

Run:  python generate_datasets.py        (fixed seed -> identical files every time)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2024
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# 1. Reference tables
# ----------------------------------------------------------------------------
MARKETS = [
    # market, county, type, lat, lon, size, premium (log-price)
    ("Wakulima Market",          "Nairobi",     "wholesale", -1.2810, 36.8300, 1.6,  0.02),
    ("City Market",              "Nairobi",     "retail",    -1.2833, 36.8219, 0.8,  0.14),
    ("Kongowea Market",          "Mombasa",     "wholesale", -4.0000, 39.6830, 1.3,  0.08),
    ("Kibuye Market",            "Kisumu",      "wholesale", -0.1022, 34.7500, 1.0,  0.00),
    ("Nakuru Municipal Market",  "Nakuru",      "wholesale", -0.2833, 36.0667, 1.2, -0.03),
    ("Eldoret Municipal Market", "Uasin Gishu", "wholesale",  0.5143, 35.2698, 1.1, -0.08),
    ("Meru Municipal Market",    "Meru",        "retail",     0.0467, 37.6500, 0.7,  0.10),
    ("Thika Market",             "Kiambu",      "retail",    -1.0333, 37.0693, 0.8,  0.11),
    ("Karatina Market",          "Nyeri",       "wholesale", -0.4833, 37.1333, 1.0, -0.05),
    ("Machakos Market",          "Machakos",    "retail",    -1.5177, 37.2634, 0.7,  0.12),
]
COUNTIES = {
    # county: (region, population ~2019 census, rainfall climatology mm/month, temp C)
    "Nairobi":     ("Nairobi",      4_397_073, [60, 45, 95, 190, 150, 30, 15, 20, 25, 60, 150, 90], 18.5),
    "Mombasa":     ("Coast",        1_208_333, [25, 20, 60, 200, 300, 80, 60, 50, 45, 70, 100, 60], 27.0),
    "Kisumu":      ("Nyanza",       1_155_574, [100, 110, 160, 230, 190, 90, 70, 90, 100, 110, 130, 110], 23.0),
    "Nakuru":      ("Rift Valley",  2_162_202, [30, 35, 65, 120, 110, 80, 90, 100, 60, 50, 70, 45], 17.0),
    "Uasin Gishu": ("Rift Valley",  1_163_186, [40, 45, 80, 140, 130, 110, 150, 160, 110, 80, 70, 50], 16.5),
    "Meru":        ("Eastern",      1_545_714, [66, 50, 105, 209, 165, 33, 17, 22, 28, 66, 165, 99], 19.0),
    "Kiambu":      ("Central",      2_417_735, [69, 52, 109, 218, 172, 35, 17, 23, 29, 69, 172, 104], 18.0),
    "Nyeri":       ("Central",        759_164, [72, 54, 114, 228, 180, 36, 18, 24, 30, 72, 180, 108], 16.5),
    "Machakos":    ("Eastern",      1_421_932, [48, 36, 76, 152, 120, 24, 12, 16, 20, 48, 120, 72], 21.0),
}
CROPS = {
    # crop: category, base KES/kg (Jan 2021), seasonal amp, peak month, noise sd, rain beta, base weekly volume kg
    "Maize":    dict(category="Cereal",    base=48,  amp=0.10, peak=6, vol=0.035, rain_beta=-0.01, vbase=40000),
    "Beans":    dict(category="Legume",    base=125, amp=0.08, peak=4, vol=0.030, rain_beta=-0.01, vbase=15000),
    "Potatoes": dict(category="Tuber",     base=38,  amp=0.14, peak=3, vol=0.050, rain_beta=-0.04, vbase=30000),
    "Tomatoes": dict(category="Vegetable", base=65,  amp=0.12, peak=2, vol=0.090, rain_beta=-0.08, vbase=20000),
    "Kale":     dict(category="Vegetable", base=32,  amp=0.08, peak=2, vol=0.060, rain_beta=-0.05, vbase=18000),
    "Cabbage":  dict(category="Vegetable", base=26,  amp=0.10, peak=3, vol=0.060, rain_beta=-0.05, vbase=22000),
    "Onions":   dict(category="Vegetable", base=72,  amp=0.09, peak=1, vol=0.050, rain_beta=-0.03, vbase=14000),
    "Bananas":  dict(category="Fruit",     base=42,  amp=0.04, peak=5, vol=0.030, rain_beta=-0.01, vbase=17000),
}
SHOCKS = {  # drought-driven price pressure (log points) over time
    "Maize":    [("2021-01-01", 0), ("2022-03-01", 0), ("2022-09-01", .30), ("2023-02-01", .32), ("2023-09-01", .05), ("2024-12-31", 0)],
    "Beans":    [("2021-01-01", 0), ("2022-04-01", 0), ("2022-10-01", .12), ("2023-03-01", .12), ("2023-10-01", 0), ("2024-12-31", 0)],
    "Potatoes": [("2021-01-01", 0), ("2022-05-01", 0), ("2022-11-01", .08), ("2023-03-01", .08), ("2023-09-01", 0), ("2024-12-31", 0)],
}

# ----------------------------------------------------------------------------
# 2. Weather (monthly, by county)
# ----------------------------------------------------------------------------
months = pd.period_range("2021-01", "2024-12", freq="M")
weather_rows = []
for county, (_, _, clim, temp) in COUNTIES.items():
    for p in months:
        m = p.month
        factor = 1.0
        if pd.Period("2021-10") <= p <= pd.Period("2022-12"):
            factor = 0.55                      # multi-season drought
        elif pd.Period("2023-03") <= p <= pd.Period("2023-05"):
            factor = 1.45                      # heavy long rains
        elif pd.Period("2023-10") <= p <= pd.Period("2023-12"):
            factor = 1.60                      # El Nino short rains
        rain = clim[m - 1] * factor * rng.gamma(6, 1 / 6)
        t = temp + 1.5 * np.cos(2 * np.pi * (m - 2) / 12) + rng.normal(0, 0.4)
        weather_rows.append(dict(county=county, year=p.year, month=m,
                                 rainfall_mm=round(float(rain), 1), avg_temp_c=round(float(t), 1)))
weather = pd.DataFrame(weather_rows)
weather["period"] = pd.to_datetime(dict(year=weather.year, month=weather.month, day=1)).dt.to_period("M")
weather["rain_z"] = weather.groupby("county")["rainfall_mm"].transform(lambda s: (s - s.mean()) / s.std())

# ----------------------------------------------------------------------------
# 3. Weekly market prices (the main dataset)
# ----------------------------------------------------------------------------
dates = pd.date_range("2021-01-04", "2024-12-30", freq="W-MON")
n_w = len(dates)
ordinals = np.array([d.toordinal() for d in dates])
months_arr = dates.month.to_numpy()
rain_lookup = {(c, p): z for c, p, z in zip(weather.county, weather.period, weather.rain_z)}


def shock_curve(crop):
    pts = SHOCKS.get(crop)
    if pts is None:
        return np.zeros(n_w)
    x = [pd.Timestamp(d).toordinal() for d, _ in pts]
    y = [v for _, v in pts]
    return np.interp(ordinals, x, y)


records = []
crop_names = list(CROPS)
periods = dates.to_period("M")
for (mk, county, mtype, lat, lon, size, prem) in MARKETS:
    # wholesale markets trade everything; retail markets skip two crops
    sold = crop_names if mtype == "wholesale" else [str(c) for c in rng.choice(crop_names, size=6, replace=False)]
    for crop in sold:
        c = CROPS[crop]
        z = np.array([rain_lookup.get((county, p - 1), rain_lookup[(county, p)]) for p in periods])
        season = c["amp"] * np.cos(2 * np.pi * (months_arr - c["peak"]) / 12)
        trend = 0.07 * np.arange(n_w) / 52
        e = np.zeros(n_w)
        sd = c["vol"] * rng.uniform(0.8, 1.2)
        for i in range(1, n_w):
            e[i] = 0.7 * e[i - 1] + rng.normal(0, sd)
        logp = np.log(c["base"]) + trend + season + prem + c["rain_beta"] * z + shock_curve(crop) + e
        price = np.exp(logp)
        logv = np.log(c["vbase"] * size) - 0.6 * (logp - logp.mean()) + rng.normal(0, 0.20, n_w)
        vol = np.exp(logv)
        for i in range(n_w):
            minutes = int(rng.integers(0, 300))
            ts = dates[i] + pd.Timedelta(hours=6, minutes=minutes)
            records.append((dates[i].strftime("%Y-%m-%d"), ts.strftime("%Y-%m-%d %H:%M:%S"),
                            crop, mk, county, round(float(price[i]), 1), "KES/kg", int(round(vol[i], -1))))

clean = pd.DataFrame(records, columns=["date", "recorded_at", "crop", "market", "county",
                                       "price_kes_per_kg", "price_unit", "volume_kg"])
clean = clean.sort_values(["date", "market", "crop"]).reset_index(drop=True)
clean.insert(0, "record_id", [f"R{i:06d}" for i in range(1, len(clean) + 1)])
clean.to_csv(OUT / "market_prices_clean.csv", index=False)

# ----------------------------------------------------------------------------
# 4. The MESSY version (what real data looks like)
# ----------------------------------------------------------------------------
raw = clean.copy()
n = len(raw)
for col in ["county", "price_unit"]:
    raw[col] = raw[col].astype(object)

# (a) missing values -- NOT completely random: Kongowea and tomatoes miss more
p_miss = 0.015 + 0.05 * (raw.market == "Kongowea Market") + 0.02 * (raw.crop == "Tomatoes")
raw.loc[rng.random(n) < p_miss.to_numpy(), "price_kes_per_kg"] = np.nan
raw.loc[rng.random(n) < 0.025, "volume_kg"] = np.nan


# (b) inconsistent county labels
def mess(label):
    kind = int(rng.integers(0, 5))
    return [label.lower(), label.upper(), label + " ", " " + label, label.replace(" ", "-")][kind]


for i in rng.choice(n, size=int(0.05 * n), replace=False):
    raw.at[i, "county"] = mess(raw.at[i, "county"])

# (c) decimal-slip typos (x10 and /10), zeros, impossible volumes
ok = raw.index[raw.price_kes_per_kg.notna()]
up = rng.choice(ok, size=20, replace=False)
raw.loc[up, "price_kes_per_kg"] = raw.loc[up, "price_kes_per_kg"] * 10
down = rng.choice(np.setdiff1d(ok, up), size=12, replace=False)
raw.loc[down, "price_kes_per_kg"] = raw.loc[down, "price_kes_per_kg"] / 10
zeros = rng.choice(np.setdiff1d(ok, np.concatenate([up, down])), size=6, replace=False)
raw.loc[zeros, "price_kes_per_kg"] = 0.0
neg = rng.choice(raw.index[raw.volume_kg.notna()], size=5, replace=False)
raw.loc[neg, "volume_kg"] = -raw.loc[neg, "volume_kg"]

# (d) unit mismatch: some maize reported per 90 kg bag, but column still says per kg
used = np.concatenate([up, down, zeros])
cand = raw.index[(raw.crop == "Maize")
                 & raw.market.isin(["Eldoret Municipal Market", "Nakuru Municipal Market"])
                 & raw.price_kes_per_kg.notna() & ~raw.index.isin(used)]
bag = rng.choice(cand, size=60, replace=False)
raw.loc[bag, "price_kes_per_kg"] = (raw.loc[bag, "price_kes_per_kg"] * 90).round(1)
raw.loc[bag, "price_unit"] = "KES/90kg bag"

# (e) exact duplicates, then shuffle (real exports are rarely sorted)
dups = raw.sample(45, random_state=SEED)
raw = pd.concat([raw, dups], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
raw.to_csv(OUT / "market_prices_raw.csv", index=False)

# ----------------------------------------------------------------------------
# 5. Production by county (Excel, two sheets, title rows above the header)
# ----------------------------------------------------------------------------
AREA = {
    "Nairobi":     dict(Kale=300, Tomatoes=150),
    "Mombasa":     dict(Maize=800, Bananas=400, Tomatoes=200, Kale=150),
    "Kisumu":      dict(Maize=42000, Beans=15000, Tomatoes=3000, Kale=2500, Bananas=2800, Onions=800),
    "Nakuru":      dict(Maize=70000, Beans=25000, Potatoes=22000, Cabbage=5000, Kale=3500, Tomatoes=2500, Onions=1500),
    "Uasin Gishu": dict(Maize=95000, Beans=18000, Potatoes=12000, Cabbage=1500),
    "Meru":        dict(Maize=55000, Beans=32000, Potatoes=15000, Bananas=20000, Tomatoes=3500, Kale=4000, Cabbage=3000, Onions=600),
    "Kiambu":      dict(Maize=30000, Beans=14000, Potatoes=9000, Kale=6000, Cabbage=4000, Tomatoes=3000, Bananas=4500),
    "Nyeri":       dict(Maize=18000, Beans=11000, Potatoes=16000, Cabbage=4500, Kale=5500, Bananas=6000, Tomatoes=900),
    "Machakos":    dict(Maize=52000, Beans=36000, Tomatoes=3500, Onions=1200, Kale=1500, Bananas=2000),
}
YIELD = dict(Maize=2.2, Beans=0.9, Potatoes=12, Tomatoes=18, Kale=14, Cabbage=25, Onions=14, Bananas=14)
prod_rows = []
for county, crops in AREA.items():
    clim_year = sum(COUNTIES[county][2])
    for year in range(2021, 2025):
        rain_year = weather[(weather.county == county) & (weather.year == year)].rainfall_mm.sum()
        mult = float(np.clip(0.70 + 0.30 * rain_year / clim_year, 0.55, 1.15))
        for crop, area in crops.items():
            a = area * (1 + 0.01 * (year - 2021)) * rng.uniform(0.95, 1.05)
            y = YIELD[crop] * mult * rng.normal(1, 0.05)
            prod_rows.append(dict(county=county, crop=crop, year=year, area_ha=round(a),
                                  production_tonnes=round(a * y, 1)))
prod = pd.DataFrame(prod_rows)
prod.loc[rng.random(len(prod)) < 0.03, "area_ha"] = np.nan                                # a few gaps
prod["county"] = prod["county"].astype(object)
prod.loc[(prod.county == "Uasin Gishu") & (prod.year == 2021), "county"] = "Uasin-Gishu"  # merge trap!

counties_df = pd.DataFrame([dict(county=k, region=v[0], population_2019=v[1]) for k, v in COUNTIES.items()])

from openpyxl.styles import Font, PatternFill  # noqa: E402

with pd.ExcelWriter(OUT / "production_by_county.xlsx", engine="openpyxl") as xl:
    prod.to_excel(xl, sheet_name="production", index=False, startrow=2)
    counties_df.to_excel(xl, sheet_name="counties", index=False)
    ws = xl.sheets["production"]
    ws["A1"] = "Crop production by county, 2021-2024 (SYNTHETIC teaching data)"
    ws["A2"] = "Source: generate_datasets.py | area in hectares, production in tonnes"
    for sheet in xl.sheets.values():
        for row in sheet.iter_rows():
            for cell in row:
                cell.font = Font(name="Arial", size=10)
    ws["A1"].font = Font(name="Arial", bold=True, size=13)
    ws["A2"].font = Font(name="Arial", italic=True, size=9)
    for header_row in (ws[3], xl.sheets["counties"][1]):
        for cell in header_row:
            cell.font = Font(name="Arial", bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="2F5D50")
    for sheet in xl.sheets.values():
        for col in "ABCDE":
            sheet.column_dimensions[col].width = 20

# ----------------------------------------------------------------------------
# 6. JSON files: nested markets + flat weather records
# ----------------------------------------------------------------------------
markets_json = []
for (mk, county, mtype, lat, lon, size, prem) in MARKETS:
    sold = sorted(clean.loc[clean.market == mk, "crop"].unique().tolist())
    markets_json.append({
        "market": mk, "county": county, "type": mtype,
        "location": {"lat": lat, "lon": lon},
        "opens": {"days": "Mon-Sat" if mtype == "retail" else "Daily", "hour": 6},
        "crops_traded": sold,
    })
(OUT / "markets.json").write_text(json.dumps(markets_json, indent=2))
weather[["county", "year", "month", "rainfall_mm", "avg_temp_c"]].to_json(
    OUT / "weather_monthly.json", orient="records", indent=1)

# ----------------------------------------------------------------------------
# 7. Dairy farms survey (cross-sectional; used for the EDA playbook lesson)
# ----------------------------------------------------------------------------
N = 1200
d_counties = ["Kiambu", "Nyandarua", "Nakuru", "Uasin Gishu", "Meru", "Bomet", "Nyeri", "Kericho"]
rain_base = dict(zip(d_counties, [1100, 1000, 950, 1150, 1300, 1400, 1250, 1900]))
price_prem = dict(zip(d_counties, [2.0, -1.5, 0.5, 0.0, 0.0, -1.0, 1.0, -0.5]))
cty = rng.choice(d_counties, N, p=[.18, .14, .14, .12, .12, .10, .12, .08])
herd = np.clip(np.round(rng.lognormal(1.35, 0.65, N)), 1, 80).astype(int)
breed = rng.choice(["Friesian", "Ayrshire", "Jersey", "Crossbreed"], N, p=[.35, .25, .10, .30])
p_meal = 1 / (1 + np.exp(-(-0.6 + 0.12 * (herd - 4))))          # bigger herds buy more dairy meal
feed = np.where(rng.random(N) < 0.25 + 0.45 * p_meal, "Silage + dairy meal", "Napier + dairy meal")
feed = np.where(rng.random(N) < 0.45 * (1 - p_meal), "Napier only", feed)
acres = np.round(np.clip(herd * rng.uniform(0.4, 1.4, N) + rng.exponential(1.0, N), 0.5, 120), 1)
rain = np.round(np.array([rain_base[c] for c in cty]) + rng.normal(0, 90, N))
vet = rng.poisson(1.5 + 0.08 * herd + 0.8 * (feed != "Napier only"))
age = np.round(np.clip(rng.normal(49, 11, N), 22, 82))
cooling = rng.random(N) < (0.10 + 0.012 * herd + 0.15 * (cty == "Kiambu"))
breed_eff = pd.Series(breed).map({"Friesian": 11.0, "Ayrshire": 9.5, "Jersey": 8.0, "Crossbreed": 7.0}).to_numpy()
feed_eff = pd.Series(feed).map({"Napier only": 0.0, "Napier + dairy meal": 2.5, "Silage + dairy meal": 4.5}).to_numpy()
mpc = np.clip(breed_eff + feed_eff + 0.004 * (rain - 1200) + 0.15 * vet + rng.normal(0, 1.8, N), 1.5, None)
lact = np.maximum(1, np.round(herd * rng.uniform(0.6, 0.85, N))).astype(int)
total = np.round(lact * mpc, 1)
price = np.round(40 + 3 * cooling + np.array([price_prem[c] for c in cty]) + rng.normal(0, 2, N), 1)
dairy = pd.DataFrame(dict(
    farm_id=[f"F{i:04d}" for i in range(1, N + 1)], county=cty, herd_size=herd, lactating_cows=lact,
    breed=breed, feed_system=feed, farm_acres=acres, annual_rainfall_mm=rain, vet_visits_per_year=vet,
    farmer_age=age, has_cooling=cooling.astype(int), milk_per_cow_l_day=np.round(mpc, 1),
    total_milk_l_day=total, price_kes_per_litre=price))
dairy["monthly_revenue_kes"] = (dairy.total_milk_l_day * dairy.price_kes_per_litre * 30).round(0)
# real-world imperfections
dairy.loc[rng.random(N) < 0.03, "farmer_age"] = np.nan
dairy.loc[rng.random(N) < 0.04, "annual_rainfall_mm"] = np.nan
dairy.loc[rng.random(N) < 0.02, "price_kes_per_litre"] = np.nan
typo = rng.choice(N, 8, replace=False)
dairy.loc[typo, "milk_per_cow_l_day"] = dairy.loc[typo, "milk_per_cow_l_day"] * 10   # decimal slip
bad = rng.choice(N, 5, replace=False)
dairy.loc[bad, "lactating_cows"] = dairy.loc[bad, "herd_size"] + rng.integers(2, 6, 5)  # more milking than owned!
dairy["breed"] = dairy["breed"].astype(object)
low = rng.choice(N, 14, replace=False)
dairy.loc[low, "breed"] = dairy.loc[low, "breed"].str.lower()
dairy.to_csv(OUT / "dairy_farms.csv", index=False)

if __name__ == "__main__":
    print("clean prices :", clean.shape)
    print("raw prices   :", raw.shape)
    print("production   :", prod.shape)
    print("weather      :", weather.shape)
    print("dairy        :", dairy.shape)
