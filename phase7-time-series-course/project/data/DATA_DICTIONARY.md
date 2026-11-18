# Capstone data dictionary: "Region B"

Files
- `project_market_train.csv`: weekly data 2013-01-06 → 2023-12-31 for **beans** and **potato** (long format). *Messy.*
- `project_market_holdout_2024.csv`: the **2024 hold-out** (52 weeks per commodity). 🔒 Do not open until the final test.

| Column | Meaning | Predictor? |
|---|---|---|
| `date` | week-ending Sunday | calendar |
| `commodity` | beans, potato | - |
| `price` | wholesale price per kg (currency units) | **target** |
| `arrivals_tonnes` | quantity delivered that week | ⚠ concurrent with price → **leakage** |
| `rainfall_mm` | weekly rainfall | lagged only |
| `temp_c` | mean temperature | lagged only |
| `exchange_rate` | local currency per USD index (100 = Jan 2013) | lagged only |
| `festival_flag` | 1 around mid-April and mid-December festivals | calendar-known |

## Known data problems to find (you are expected to discover them)
Duplicate rows, missing weeks, decimal-shift typos in `price`, a gap in `rainfall_mm` (sensor outage). The hold-out file is clean.

## What is *not* documented here: on purpose
How the covariates act on prices (which lags, linear or not), and whether the market's structure is stable, is for **you** to discover from the training data. A solution key is provided in `ANSWER_KEY.md`: read it only after submitting.
