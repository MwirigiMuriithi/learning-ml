# 📏 Capstone Rubric (100 points)

| Criterion | Excellent (full) | Adequate (half) | Missing / wrong (0) | Pts |
|---|---|---|---|---|
| **1. Problem framing** (Step 0) | Clear prediction moment; allowed vs excluded information stated; evaluation matches use-case (grouped split) | Statement present but vague | No framing | 8 |
| **2. Data audit / issues log** (Step 1) | Counts for duplicates, types, spellings, impossible values, missingness (incl. MCAR/MAR reasoning) | Some families missing; few numbers | No evidence | 10 |
| **3. Cleaning** (Step 2) | `clean_farm()` passes 13/13; correct *order* of operations; learns nothing from data; cleaning log | Passes most checks; some steps ad hoc | Manual, undocumented, fails checks | 20 |
| **4. Leakage control** (Steps 3–4, 8) | Every excluded column justified; grouped split; all learned steps inside the pipeline; shuffled-target ≈ 0 | One lapse (e.g. scaling before split) | Leaky target/columns in the model | 15 |
| **5. Feature engineering** (Step 5) | ≥ 8 features, all families, as transformers; each justified; no look-ahead | 4–7 features or notebook-only code | < 4 features / unjustified | 12 |
| **6. Pipeline** (Step 6) | ColumnTransformer with correct imputers, scaling, one-hot with `handle_unknown`, **explicit** ordinal order, capping learned on train; reloads and predicts on messy input | Works but missing one required element | Does not run / manual steps | 15 |
| **7. Feature selection** (Step 7) | Canary / noise-floor analysis plus CV-based choice of *k*; decision explained | One method, little evidence | None | 5 |
| **8. Evaluation** (Steps 8–9) | Baselines table; honest final hold-out; per-crop errors reported in t/ha | Only one comparison | Reports training score / test used for tuning | 7 |
| **9. Documentation** (Step 11) | Transformation log complete incl. the **learns-from-data** column; feature dictionary; code readable | Log present but partial | Missing | 8 |
| **Total** | | | | **100** |

### Bonus (up to +5)
Stretch goals completed and honestly evaluated (including "it didn't help").

### Automatic deductions
* −10 if any preprocessing step is fitted on data that includes the test set.
* −10 if `total_revenue_ksh`, `market_price_ksh_kg` or `harvest_date` is used as a predictor for the pre-harvest use-case.
* −5 if the notebook does not run top-to-bottom.

### Grade bands
90–100 outstanding · 75–89 strong · 60–74 meets expectations · < 60 revisit Lessons 06–07 and resubmit.
