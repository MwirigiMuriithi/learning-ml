# 🗃️ Data dictionary (lesson datasets)

All files live in `data/`. Rebuild everything with `python generate_data.py`.

## `agri_prices_weekly.csv`: simulated weekly wholesale market (L0, L3–L8)
2,504 rows = 4 commodities × 626 weeks (week-ending Sunday, 2013-01-06 → 2024-12-29), long format.

| Column | Type | Meaning | Usable as a predictor? |
|---|---|---|---|
| `date` | date | week-ending Sunday | calendar → yes |
| `commodity` | str | maize, tomato, onion, rice | - |
| `price` | float | wholesale price per kg (currency units) | **target**; its lags yes |
| `arrivals_tonnes` | float | quantity delivered that week | ⚠ **No**: recorded concurrently with price (leakage) |
| `rainfall_mm` | float | weekly rainfall in the growing region | only **lagged ≥ horizon** |
| `temp_c` | float | mean temperature | only lagged |
| `fuel_index` | float | transport fuel price index (100 = Jan 2013) | only lagged |
| `festival_flag` | int | 1 in year-end festival weeks | yes (calendar-known) |

**How the prices were built** (see `generate_data.py`): `log price = base + random-walk level with drift + yearly seasonality (1–2 harmonics) + rain effect (6-week mean rainfall, lagged 8 weeks) + fuel pass-through (lag 2) + 4% festival bump + decaying drought shocks (2017, 2022) + AR(1) noise`. Crops differ in noise, seasonal shape and sensitivity (tomato: noisy, two-humped; onion: strongly seasonal; rice: smooth; maize: trending with annual harvest cycle).

## `maize_daily_raw.csv`: deliberately messy daily file (L2)
Daily maize price and rainfall, 2022–2024, with the usual problems: market closed Sundays (structural gaps), ~3 % random missing days, 8 duplicated rows, 5 decimal-shift typos (×10 and ÷10), 5 missing prices, rainfall that is **exactly 0 on 82 % of days** (the MAPE trap).

## Real teaching datasets
| File | Source | Description |
|---|---|---|
| `airline_passengers.csv` | Box, Jenkins & Reinsel (1976) | monthly international airline passengers, 1949–1960; multiplicative seasonality |
| `co2_weekly.csv` | Mauna Loa (via `statsmodels.datasets.co2`) | weekly atmospheric CO₂ (ppm), 1958–2001, some gaps |
| `sunspots_annual.csv` | via `statsmodels.datasets.sunspots` | annual sunspot counts, 1700–2008; the textbook *cyclical* series |

Capstone data are documented in `project/data/DATA_DICTIONARY.md`.
