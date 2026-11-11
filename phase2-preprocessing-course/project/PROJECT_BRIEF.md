# 🏆 Capstone Project: AgriReady

**Weeks 3–4 · Phase 2; Data Preprocessing and Feature Engineering**
**Estimated time:** 8–12 hours · **Work alone or in pairs** · **Template:** `project_template.ipynb`

---

## 1. The scenario

The county agriculture office keeps records of thousands of farm seasons: crop, soil tests, weather, inputs applied, pest pressure, and the yield that was harvested. Two analysts have already tried to build models from the raw export and got nonsense: crop names were spelled thirty different ways, some pH values were 99, and one model reported a yield forecast accuracy of 99%, right before it failed on real farms.

You've been hired to fix the foundation. **Your job is not to build the best model. Your job is to turn raw, messy data into trustworthy, machine-learning-ready data, and to prove it.**

**Use-case:** a *pre-harvest yield forecast*: shortly before harvest, forecast each farm's yield (t/ha). Known at that moment: farm attributes, planting date, soil tests, inputs, the season's weather and pest pressure. **Not known:** the actual harvest date, price, revenue.

## 2. The data
`data/farm_records_raw.csv`: the raw, messy agricultural dataset (the simulated stand-in for your Phase 1 data; use your real Phase 1 data if you prefer).
Optional stretch: `data/crop_prices_daily.csv`. See `data/DATA_DICTIONARY.md`.

## 3. Required deliverables

| # | Deliverable | What "good" looks like |
|---|---|---|
| 1 | **A cleaned dataset** (`farm_clean.csv`) | Passes `check_clean` 13/13; created by a reusable function `clean_farm()` with a cleaning log |
| 2 | **A preprocessing pipeline** (`.joblib`) | One scikit-learn `Pipeline`; every learned step fitted on training data only; reloads and predicts on messy new data |
| 3 | **Engineered features** | ≥ 8 new features across date / combination / domain / transform families, built as transformers, each justified |
| 4 | **ML-ready matrices** | `X_train_ready.csv`, `X_test_ready.csv` pass `check_ml_ready` 5/5 |
| 5 | **A documented explanation of each transformation** (`TRANSFORMATION_LOG.md`) | Every step: what, why, which columns, **does it learn from data?** |
| 6 | **Evidence** | Comparison against baselines; shuffled-target test ≈ 0; leakage exclusions justified |

## 4. Milestones (suggested)

| Day | Steps | Output |
|---|---|---|
| 1 | 0–2 | Prediction-moment statement, `issues` table, `clean_farm()` |
| 2 | 3–5 | `EXCLUDE` dict, grouped split, engineered features |
| 3 | 6–7 | Full pipeline, selection decision |
| 4 | 8–9 | Baselines table, shuffled-target test, final hold-out |
| 5 | 10–12 | Exports, transformation log, reflection |

## 5. Rules
* Do **not** touch the raw CSV. Do **not** peek at `reference_solution/` until you've finished (or are truly stuck for > 1 hour on one step).
* **Split before you learn.** The test set is used exactly once.
* Every number in your report must come from code you can re-run.

## 6. Hints by topic
* *Stuck on cleaning?* → Lessons 01–02, especially `standardise()` and the rule table.
* *Don't know what to exclude?* → Lesson 06 timeline table.
* *Features?* → Lesson 04, domain-knowledge section (distance from optimum).
* *Pipeline errors?* → Lesson 07; print `pipe.get_params().keys()`; use `pipe[:-1].transform(X.head())` to see the matrix each step produces.
* *Unsure if you leak?* → the shuffled-target test (Step 8).

## 7. Stretch goals ⭐
1. Time-safe **price context** feature from the daily price series (Step 10).
2. Tune preprocessing (`cap_k`, imputation strategy) with `GridSearchCV` and report which choice mattered.
3. A **second dataset test**: apply your `clean_farm()` and pipeline to a freshly generated dataset (`python data/generate_datasets.py` with another seed): does it still work?
4. Predict at **planting time**: remove the in-season weather columns and see what happens to performance. What does that tell you about the value of weather forecasts?
