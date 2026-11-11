# 🎓 Instructor Notes

For teachers, TAs and self-learners who want to check their work.

## 1. Answer key: the mess hidden in `farm_records_raw.csv`

| Problem family | What was injected | Count in raw file |
|---|---|---|
| Rows | 2,400 true records + 60 exact duplicates + 25 near-duplicates (UPPER-case crop, lower-case county, half re-typed date) | 2,485 |
| Duplicates | exact `duplicated()` → 60; on natural key `farm_id + planting_date` **after parsing dates** → 85 (75 if you compare the date *text*) | 60 / 85 |
| Text instead of numbers | `fertilizer_kg_ha` as `'85'`, `'85 kg'`, `'85kg/ha'`, `'-'`, `'unknown'` (`'N/A'` is auto-read as NaN by `read_csv`) | ≈ 380 non-plain entries |
| Dates | ≈ 4–5% written as `14 Mar 2023` | ≈ 107 planting dates |
| Spellings | crop: 27 variants of 5; county: 37 of 10; irrigation: 10 codes of 2; plus soil type and pest pressure | - |
| Impossible values | soil pH ∈ {0, 14.5, −1, 99} (10); rainfall < 0 (7); temperature sentinels −999/99 (5); farm size ≤ 0 (7); negative yield (~7) | ≈ 36 |
| Cross-field | `harvest_date` = 2018-01-15 (before planting) | 15 (14 after de-dup) |
| Outliers | farm size 80–400 ha (16); fertilizer 1,500–3,000 kg/ha (7); yield × 10 decimal slips (~7); nitrogen 150–300 ppm (~10) | - |
| Missing (MCAR) | 2–7% in many columns (see `profile()`) | - |
| Missing (MAR) | fertiliser missing ≈ 20% on farms < 1 ha vs ≈ 2% on larger farms | - |
| Missing target | ≈ 1.5% | ≈ 38 |
| Leaky column | `total_revenue_ksh = yield × area × price` (computed from the *clean* yield) | all rows |

**Reference cleaning log (`clean_farm`):** 2,485 raw rows → 85 duplicates removed → 14 harvest dates blanked → 35 impossible feature values → NaN → 71 rows dropped (bad target) → **2,329 clean rows**.

## 2. Reference results (so you know when "good" is good)

| Experiment | Result |
|---|---|
| Lesson 3: KNN on maize, unscaled vs scaled (R²) | ≈ −0.01 → ≈ 0.54; Random Forest unchanged |
| Lesson 4: Ridge on log-yield, raw → engineered features | ≈ 0.74 → ≈ 0.93 (Random Forest ≈ 0.92 → 0.93) |
| Lesson 4: 30-day price forecast, naive vs seasonal (MAE) | roughly halved for Maize/Potato |
| Lesson 5: KNN with 0 → 200 noise features | R² ≈ 0.70 → 0.11 |
| Lesson 5: PCA on spectra | 4 components → 95% variance; CV R² ≈ 0.87 (≈ all 40 bands) |
| Lesson 6: select-then-CV on pure noise | ≈ 87% accuracy (wrong) vs ≈ 50% (right) |
| Lesson 6: centered vs trailing window (Maize, MAE) | ≈ 0.50 vs ≈ 0.58 (naive ≈ 0.59) |
| Lesson 7 / project: full pipeline CV R² (log yield) | ≈ 0.93–0.94; crude baseline ≈ 0.07; shuffled-target ≈ 0 |
| Project: hold-out R² in t/ha | ≈ 0.88, MAE ≈ 0.5 t/ha |

Exact decimals vary slightly with library versions.

## 3. Suggested pacing (two weeks, ~12–15 h/week)

| Session | Content | Live-teaching tip |
|---|---|---|
| 1 | Lesson 00 + 01 | Show `np.nan == np.nan`; let students *guess* the MAR pattern from the missingness map before revealing it |
| 2 | Lesson 02 | Ask "is 99 an outlier?": pH vs fertiliser debate (sentinel scan) |
| 3 | Lesson 03 | Whiteboard the 3-farm KNN distance example before running it |
| 4 | Lesson 04 | Have students propose features *before* the idea menu |
| 5 | Lesson 05 | Make a prediction: "where will the canaries rank?" (they're mid-pack, not last!) |
| 6 | Lesson 06 | The noise-leak experiment live: collect guesses of the accuracy first |
| 7 | Lesson 07 | Live-code a custom transformer; deliberately forget `return self` |
| 8–10 | Project | Peer review of each other's `TRANSFORMATION_LOG.md` |

## 4. Common student errors (and which lesson fixes them)
1. `pd.to_numeric(errors="coerce")` without inspecting failures → L1.
2. De-duplicating before standardising text → L1/L2.
3. Using z-score on skewed columns / global outlier rules across crops → L2.
4. Alphabetical `OrdinalEncoder` → L3.
5. Fitting scalers before the split → L3/L6.
6. `diff()`/`rolling()` across stacked groups → L4.
7. Trusting Pearson correlation for hill-shaped effects → L5.
8. Selecting features on all data, then cross-validating → L5/L6.
9. IQR capping applied to binary columns (collapses them to a constant) → L7/Project. *This bug appeared while building the course: `check_ml_ready` caught it.*
10. Classes defined in the notebook break `joblib.load` elsewhere → L7.

## 5. Discussion questions
* When is it *right* to keep an outlier? Who decides?
* The leaky revenue column barely moved the Random Forest score. Why is that *worse* than if it had exploded?
* Group leakage showed no effect in this simulated dataset. What real-world feature of data (unmeasured farm effects) would make it appear?
* Why did aggressive capping (k = 1.5) *hurt* accuracy? What does that say about "standard" outlier rules?
* The stretch goal (price context) did nothing. Is a null result a failure?

## 6. Grading the capstone
Use `project/RUBRIC.md`. Quick automated checks: `check_clean(df)` (13), `check_ml_ready(X)` (5), the shuffled-target test (≈ 0), and "does the notebook run top-to-bottom?".

## 7. Known limitations
* Data are simulated: relationships are cleaner than in reality (e.g. no unmeasured farm effects, so group-CV shows no gap).
* Saved `.joblib` files are tied to the scikit-learn version that produced them; re-run the notebooks rather than loading shipped files across major versions.
* Written and executed with pandas 3 / scikit-learn 1.9; older versions (pandas ≥ 2.0, scikit-learn ≥ 1.3) should work but were not tested.
