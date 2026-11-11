# Transformation log: AgriReady pipeline

**Use-case:** pre-harvest yield forecast (log yield). **Test:** 20% of farms, never used for fitting.

## Cleaning counts

| step | count |
|---|---|
| raw rows | 2485 |
| duplicate records removed | 85 |
| missing 'season' derived from planting month | 48 |
| harvest dates blanked (before planting) | 14 |
| impossible feature values set to NaN | 35 |
| rows dropped (missing / impossible yield) | 71 |
| clean rows | 2329 |

## Every transformation

| # | Step | Columns | What it does | Why | Learns from data? |
|---|---|---|---|---|---|
| 1 | Type conversion | fertilizer_kg_ha, planting_date, harvest_date | Extract numbers from text ('85 kg' -> 85); parse mixed date formats | Models need numbers/dates, not text | No |
| 2 | Category standardisation | crop, county, soil_type, pest_pressure, irrigation | lower-case, strip punctuation, synonyms (corn->Maize), fuzzy match; Yes/Y/TRUE/1 -> 1 | Dozens of spellings became 5 crops / 10 counties | No (rules) |
| 3 | De-duplication | all rows | Drop repeated (farm_id, planting_date), keep the most complete | Avoid double-counting and train/test duplicates | No |
| 4 | Cross-field logic | season, harvest_date | Derive missing season from planting month; blank harvest < planting | Deterministic repairs, no guessing | No (rules) |
| 5 | Validity rules | soil_ph, rainfall, temperature, farm size, nutrients, fertilizer, price | Out-of-range / sentinel values -> NaN | Impossible values would mislead the model | No (domain rules) |
| 6 | Target cleaning | yield_t_ha | Drop missing, <=0 or above crop ceiling; model log(yield) | Can't learn from impossible answers; log tames skew | No |
| 7 | Leakage exclusions | farm_id, harvest_date, total_revenue_ksh, market_price_ksh_kg, season | Removed before modelling | Not known at prediction time / identifiers / duplicates of other info | No |
| 8 | Grouped split | rows | 80/20 split by farm_id | Test on unseen farms | No |
| 9 | DateFeatures | planting_date | year, month, sin/cos month, day-of-year, rain season | Seasonality | No |
| 10 | AgriFeatures | N, P, K, rainfall, temp, pH, fertiliser, irrigation, farm size, crop | npk_total, N:P ratio, moisture index, fert x irrigation, gap-to-optimum^2, log size | Capture hill-shaped and balance effects | No (row-wise) |
| 11 | Median imputation | numeric features | Fill NaN with the column median | Models can't take NaN; median robust to skew | YES - medians (train only) |
| 12 | Winsorizer (k=10) | numeric features | Clip to Q1-10*IQR / Q3+10*IQR; columns with IQR = 0 (binary flags) are left uncapped | Limit extreme leverage (gentle: aggressive capping hurt; zero-IQR guard prevents erasing 0/1 columns) | YES - fences (train only) |
| 13 | StandardScaler | numeric features | (x - mean) / std | Comparable scales for linear model | YES - mean/std (train only) |
| 14 | Mode imputation + one-hot | crop, county, soil_type, rain_season | Fill with most frequent, then 0/1 columns; unseen -> zeros | Nominal categories have no order | YES - mode & vocabulary (train only) |
| 15 | Mode imputation + ordinal | experience, pest_pressure | Explicit order Beginner<...<Expert, Low<Medium<High | Real order preserved | Mode: YES; order: no |
| 16 | SelectKBest (k=all) | all engineered/encoded columns | Keep top-k by F-test inside the pipeline | Reduce noise; evaluated by GroupKFold | YES - scores (train only) |

## Leakage rule

Every step marked **YES** was fitted on the training set only and merely *applied* to the test set.
