"""
generate_data.py -- builds every dataset used in the course.

Why synthetic data?
-------------------
Real agricultural price series are licensed, patchy, and hard to redistribute.
So the *agricultural* datasets in this course are SIMULATED from a transparent
recipe (trend + harvest seasonality + rainfall shocks + fuel costs + AR noise).
Because we know the recipe, you can check whether your models recover real
structure -- a luxury you never have with real data.

The classic teaching datasets (CO2, sunspots, airline passengers) are REAL.

Run:  python generate_data.py          (takes a few seconds, fully reproducible)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


def _ar1(n, phi, sigma, rng):
    e = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def _impulse(n, t0, size, decay):
    """A shock at time t0 that fades geometrically (e.g. a drought)."""
    x = np.zeros(n)
    for t in range(t0, n):
        x[t] = size * decay ** (t - t0)
    return x


# --------------------------------------------------------------------------
# 1. WEEKLY MULTI-COMMODITY MARKET  (main dataset for lessons 3-8 + project)
# --------------------------------------------------------------------------
def make_weekly(seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-06", "2024-12-29", freq="W-SUN")
    n = len(dates)
    doy = dates.dayofyear.values
    ang = 2 * np.pi * doy / 365.25

    # --- shared drivers (same weather & fuel for every commodity) ----------
    # bimodal rainfall: long rains ~Apr, short rains ~Nov
    rain_clim = 22 + 18 * np.exp(-0.5 * ((doy - 105) / 30) ** 2) \
                   + 14 * np.exp(-0.5 * ((doy - 315) / 28) ** 2)
    drought = np.ones(n)
    for lo, hi in [("2017-03-01", "2017-07-01"), ("2022-03-01", "2022-07-15")]:
        drought[(dates >= lo) & (dates <= hi)] = 0.35         # failed rains
    rain = rng.gamma(shape=2.0, scale=rain_clim * drought / 2.0)
    temp = 24 + 3.5 * np.sin(ang - 1.2) + _ar1(n, 0.5, 0.6, rng)
    fuel = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.012, n)))
    fuel[(dates >= "2022-02-20") & (dates <= "2022-10-01")] *= 1.18   # price spike
    festival = (((dates.month == 12) & (dates.day >= 15)) |
                ((dates.month == 1) & (dates.day <= 7))).astype(int)

    rain_ma = pd.Series(rain).rolling(6, min_periods=1).mean().values
    rain_z = (rain_ma - rain_ma.mean()) / rain_ma.std()
    lag = 8                                    # rain affects price ~8 weeks later
    rain_lag = np.r_[np.full(lag, 0.0), rain_z[:-lag]]
    fuel_l = np.log(np.r_[np.full(2, fuel[0]), fuel[:-2]] / fuel[0])

    specs = {
        # name     base  drift   lvl_sd  seas(A1,ph1,A2,ph2)         rain_b fuel_b  phi  noise  glut
        "maize":  dict(base=45,  drift=0.0012, lvl=0.010, s=(0.16, 0.9, 0.05, 0.3), rb=-0.045, fb=0.20, phi=0.55, nz=0.020),
        "tomato": dict(base=60,  drift=0.0005, lvl=0.012, s=(0.12, 2.6, 0.18, 1.0), rb=-0.030, fb=0.12, phi=0.35, nz=0.075),
        "onion":  dict(base=38,  drift=0.0009, lvl=0.014, s=(0.22, 0.2, 0.06, 2.0), rb=-0.035, fb=0.15, phi=0.60, nz=0.035),
        "rice":   dict(base=95,  drift=0.0010, lvl=0.004, s=(0.04, 1.5, 0.015, 0.5), rb=-0.012, fb=0.25, phi=0.70, nz=0.008),
    }
    rows = []
    for name, p in specs.items():
        level = np.cumsum(rng.normal(p["drift"], p["lvl"], n))
        a1, ph1, a2, ph2 = p["s"]
        seas = a1 * np.cos(ang - ph1) + a2 * np.cos(2 * ang - ph2)
        shock = (_impulse(n, int(np.argmax(dates >= "2017-04-01")), 0.14, 0.93) +
                 _impulse(n, int(np.argmax(dates >= "2022-04-01")), 0.20, 0.94) -
                 _impulse(n, int(np.argmax(dates >= "2020-09-01")), 0.10, 0.90))
        shock = shock * (0.4 if name == "rice" else 1.0)
        noise = _ar1(n, p["phi"], p["nz"], rng)
        logp = (np.log(p["base"]) + level + seas + p["rb"] * rain_lag
                + p["fb"] * fuel_l + 0.04 * festival + shock + noise)
        price = np.exp(logp)
        # arrivals (supply) move opposite to price & seasonality; CONCURRENT => leaky as a predictor!
        arrivals = 120 * np.exp(-0.9 * (logp - logp.mean()) + rng.normal(0, 0.12, n))
        df = pd.DataFrame(dict(date=dates, commodity=name, price=price.round(2),
                               arrivals_tonnes=arrivals.round(1), rainfall_mm=rain.round(1),
                               temp_c=temp.round(1), fuel_index=fuel.round(2),
                               festival_flag=festival))
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(DATA / "agri_prices_weekly.csv", index=False)
    return out


# --------------------------------------------------------------------------
# 2. MESSY DAILY MAIZE FILE  (lesson 2: cleaning & datetime handling)
# --------------------------------------------------------------------------
def make_daily_raw(seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    n = len(dates)
    doy = dates.dayofyear.values
    ang = 2 * np.pi * doy / 365.25
    level = np.cumsum(rng.normal(0.0004, 0.006, n))
    seas = 0.15 * np.cos(ang - 0.9) + 0.04 * np.cos(2 * ang - 0.3)
    dow = dates.dayofweek.values
    dow_eff = np.select([dow == 5, dow == 0, dow == 4], [0.035, -0.02, 0.01], 0.0)  # Sat high
    noise = _ar1(n, 0.5, 0.012, rng)
    price = 52 * np.exp(level + seas + dow_eff + noise)
    rain_clim = 1.2 + 1.3 * np.exp(-0.5 * ((doy - 105) / 30) ** 2) + 1.0 * np.exp(-0.5 * ((doy - 315) / 28) ** 2)
    wet = rng.random(n) < (0.12 + 0.18 * (rain_clim > 1.8))
    rain = np.where(wet, rng.gamma(1.6, 7.0, n), 0.0)       # MANY exact zeros (MAPE trap!)
    df = pd.DataFrame(dict(date=dates.strftime("%Y-%m-%d"), price=price.round(2), rainfall_mm=rain.round(1)))
    df = df[dates.dayofweek != 6].copy()                      # market closed Sundays
    # random missing days (power cuts, data-entry gaps)
    df = df.drop(df.sample(frac=0.03, random_state=1).index)
    # data-entry disasters
    bad = df.sample(6, random_state=3).index
    df.loc[bad[:3], "price"] = (df.loc[bad[:3], "price"] * 10).round(2)    # misplaced decimal
    df.loc[bad[3:5], "price"] = (df.loc[bad[3:5], "price"] / 10).round(2)
    df["price"] = df["price"].astype(object)
    df.loc[bad[5], "price"] = "n/a"
    # duplicate rows
    df = pd.concat([df, df.sample(8, random_state=5)]).sort_index()
    df.loc[df.sample(4, random_state=9).index, "price"] = np.nan
    df.to_csv(DATA / "maize_daily_raw.csv", index=False)
    return df


# --------------------------------------------------------------------------
# 3. CLASSIC REAL DATASETS
# --------------------------------------------------------------------------
def make_classics():
    import statsmodels.api as sm
    co2 = sm.datasets.co2.load_pandas().data.reset_index()
    co2.columns = ["date", "co2_ppm"]
    co2.to_csv(DATA / "co2_weekly.csv", index=False)
    sun = sm.datasets.sunspots.load_pandas().data
    sun["YEAR"] = sun["YEAR"].astype(int)
    sun.columns = ["year", "sunspots"]
    sun.to_csv(DATA / "sunspots_annual.csv", index=False)
    try:  # Box & Jenkins airline passengers (monthly 1949-1960)
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
        air = pd.read_csv(url)
        air.columns = ["month", "passengers"]
        air.to_csv(DATA / "airline_passengers.csv", index=False)
    except Exception as e:
        print("Could not download airline data:", e)


# --------------------------------------------------------------------------
# 4. CAPSTONE PROJECT DATA  (a DIFFERENT market with hidden structure to discover)
# --------------------------------------------------------------------------
def make_project_data(seed=2024):
    """
    'Region B' weekly market: beans and potato.  Differences from the lesson data (students must DISCOVER them):
      * rain affects price with a ~6-week lag (rolling 4-week mean, lags 6-9)
      * potato has a NON-LINEAR heat-stress effect (price rises when temperature 3 weeks ago exceeded 27.5 C)
      * an exchange-rate jump in March 2020 passes through to prices with a ~4-week lag
      * a STRUCTURAL BREAK: a storage subsidy from July 2020 cuts beans' seasonal amplitude by 40 %
      * a drought from mid-Sep 2023 to Jan 2024 hits INSIDE the hold-out year 2024
      * messy data: missing weeks, duplicate rows, decimal-shift typos, a sensor outage in rainfall
    Writes project/data/project_market_train.csv (<= 2023) and project_market_holdout_2024.csv (2024).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-01-06", "2024-12-29", freq="W-SUN")
    n = len(dates)
    doy = dates.dayofyear.values
    ang = 2 * np.pi * doy / 365.25

    rain_clim = 20 + 20 * np.exp(-0.5 * ((doy - 80) / 28) ** 2) + 16 * np.exp(-0.5 * ((doy - 290) / 30) ** 2)
    dry = np.ones(n)
    for lo, hi in [("2016-03-01", "2016-06-30"), ("2019-03-01", "2019-06-30"), ("2023-09-15", "2024-01-31")]:
        dry[(dates >= lo) & (dates <= hi)] = 0.30
    rain = rng.gamma(2.0, rain_clim * dry / 2.0)
    temp = 25 + 3 * np.sin(ang - 1.0) + _ar1(n, 0.5, 0.9, rng)
    fx = 100 * np.exp(np.cumsum(rng.normal(0.0012, 0.008, n)))
    fx[dates >= "2020-03-15"] *= 1.22                                  # currency shock
    festival = (((dates.month == 4) & (dates.day >= 8) & (dates.day <= 21)) |
                ((dates.month == 12) & (dates.day >= 14))).astype(int)

    rain4 = pd.Series(rain).rolling(4, min_periods=1).mean().values
    rz = (rain4 - rain4.mean()) / rain4.std()
    rain_eff = np.r_[np.zeros(6), rz[:-6]]                              # effect arrives 6 weeks later
    heat = np.maximum(0, np.r_[np.full(3, 27.5), temp[:-3]] - 27.5)
    fx_eff = np.log(np.r_[np.full(4, fx[0]), fx[:-4]] / fx[0])

    specs = {
        "beans":  dict(base=120, drift=0.0008, lvl=0.008, A=(0.14, 2.2, 0.05, 0.4), rb=-0.07, fb=0.30, hb=0.0,  phi=0.60, nz=0.022, fest=0.05, brk=True),
        "potato": dict(base=35,  drift=0.0004, lvl=0.012, A=(0.10, 4.0, 0.12, 0.8), rb=-0.04, fb=0.15, hb=0.045, phi=0.40, nz=0.060, fest=0.03, brk=False),
    }
    rows = []
    for name, p in specs.items():
        level = np.cumsum(rng.normal(p["drift"], p["lvl"], n))
        a1, ph1, a2, ph2 = p["A"]
        amp = np.where(dates >= "2020-07-01", 0.6, 1.0) if p["brk"] else 1.0
        seas = amp * (a1 * np.cos(ang - ph1) + a2 * np.cos(2 * ang - ph2))
        noise = _ar1(n, p["phi"], p["nz"], rng)
        logp = (np.log(p["base"]) + level + seas + p["rb"] * rain_eff + p["fb"] * fx_eff
                + p["hb"] * heat + p["fest"] * festival + noise)
        price = np.exp(logp)
        arrivals = 90 * np.exp(-0.8 * (logp - logp.mean()) + rng.normal(0, 0.12, n))
        rows.append(pd.DataFrame(dict(date=dates, commodity=name, price=price.round(2), arrivals_tonnes=arrivals.round(1),
                                      rainfall_mm=rain.round(1), temp_c=temp.round(1), exchange_rate=fx.round(2),
                                      festival_flag=festival)))
    out = pd.concat(rows, ignore_index=True)

    # ---- make it messy (training period only; the hold-out is clean) -------------------------
    rng2 = np.random.default_rng(seed + 1)
    train = out[out.date < "2024-01-01"].copy(); hold = out[out.date >= "2024-01-01"].copy()
    for name in ["beans", "potato"]:
        idx = train.index[(train.commodity == name) & (train.date > "2013-06-01")]
        drop = rng2.choice(idx, 5, replace=False); train = train.drop(drop)                           # missing weeks
        idx = train.index[train.commodity == name]
        typo = rng2.choice(idx, 2, replace=False)
        train.loc[typo[0], "price"] = round(train.loc[typo[0], "price"] * 10, 2)                     # misplaced decimal
        train.loc[typo[1], "price"] = round(train.loc[typo[1], "price"] / 10, 2)
    train = pd.concat([train, train.sample(2, random_state=3)])                                       # duplicate rows
    outage = (train.date >= "2018-05-06") & (train.date <= "2018-05-20")
    train.loc[outage, "rainfall_mm"] = np.nan                                                         # sensor outage
    train = train.sort_values(["commodity", "date"])
    train.to_csv(Path(__file__).parent / "project" / "data" / "project_market_train.csv", index=False)
    hold.to_csv(Path(__file__).parent / "project" / "data" / "project_market_holdout_2024.csv", index=False)
    return train, hold


if __name__ == "__main__":
    w = make_weekly(); print("weekly:", w.shape)
    d = make_daily_raw(); print("daily raw:", d.shape)
    make_classics(); print("classics written")
    t, h = make_project_data(); print("project train:", t.shape, "| holdout:", h.shape)
