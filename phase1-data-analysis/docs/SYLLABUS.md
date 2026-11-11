# Syllabus · Phase 1: Data Analysis & Exploration

## Learning outcomes
By the end of the course you can:

1. **Compute** with NumPy arrays: create, index, slice, filter, reshape, vectorise, broadcast, apply matrix operations, and generate random numbers.
2. **Wrangle** data with pandas: import CSV/Excel/JSON, select/filter, clean, handle missing values and dates, transform, group, aggregate, sort and merge.
3. **Visualise** with Matplotlib and Seaborn: histograms, scatter plots, box plots, heatmaps, distribution plots and correlation visualisations: honestly and clearly.
4. **Explore** an unfamiliar dataset: distributions, relationships, anomalies, trends, correlations, hypotheses.
5. **Judge ML-readiness**: target candidates, features, leakage, splits, baselines.

## Topic coverage matrix (every topic in the Phase 1 brief)

| Required topic | Where it is taught | Practised in |
|---|---|---|
| **NumPy**: arrays and multidimensional arrays | L1 §2–3 | L1 P1–P6 |
| Array indexing and slicing | L1 §4–5 (incl. views vs copies, masks, fancy indexing) | L1 P1–P6 |
| Broadcasting | L2 §3 | L2 P1–P2, reflection |
| Vectorisation | L2 §1 (with timing benchmark) | L2 P5 |
| Matrix operations | L2 §4 (`@`, solve, lstsq, regression teaser) | L2 P4, P7 |
| Numerical operations | L2 §2, §5 (aggregations, NaN, diff, IQR, histogram, corrcoef) | L2 P1, P3 |
| Random number generation | L2 §6 (Generator API, Monte-Carlo, bootstrap) | L2 P6 |
| **Pandas**: Series and DataFrames | L3 §1–2 | L3 P1–P8 |
| Importing CSV, Excel and JSON | L3 §3 (incl. nested JSON, Excel title rows) | L3 P5–P6 |
| Selecting and filtering records | L3 §5 | L3 P1–P4, P8 |
| Grouping and aggregation | L5 §1–2 | L5 P1–P5, P9–P10 |
| Sorting and merging datasets | L3 §5.4 (sorting) · L5 §4 (merge/concat/validate) | L3 P3, P7 · L5 P6–P8 |
| Handling missing values | L4 §5 (diagnose + 4 strategies, scored against truth) | L4 P4, P6 |
| Working with dates and timestamps | L4 §6 (parse, `.dt`, resample, rolling, shift, time zones) | L4 P7–P8 |
| Applying transformations | L4 §7 (`map`, `apply`, `np.select`, `cut`, pipelines) · L3 §5.5 (`assign`) | L4 P9 |
| **Visualisation**: Matplotlib | L6 | L6 P1–P7 |
| Seaborn | L7 | L7 P1–P8 |
| Histograms · scatter plots · box plots | L6 §4–6 · L7 §1–3 | L6 P4–P6 · L7 P1–P2, P8 |
| Heatmaps · distribution plots | L7 §1, §5–6 | L7 P5–P6 |
| Correlation visualisations | L7 §5 (heatmap, pair plot, joint plot, confounding, trends) | L7 P6–P7 |
| **EDA**: distributions, relationships, anomalies, trends, correlations, hypotheses | L8 (8-step playbook) | L8 P1–P10 |
| **Practical assignment**: overview, quality assessment, descriptive stats, visuals, correlation, conclusions | L9 Capstone | Rubric (100 pts) |

## Daily plan

| Day | Notebook | You'll do | Checkpoint (you're done when…) |
|---|---|---|---|
| 1 | 00 + 01 | Setup; arrays, axes, indexing, masks | you can explain *view vs copy* and solve 01-p1…p6 |
| 2 | 02 | vectorise; broadcast; matrices; simulate | you can predict a broadcast shape *before* running it |
| 3 | 03 | read CSV/Excel/JSON; `.loc/.iloc`; filter | you can open an unfamiliar file and describe it in 30 s |
| 4 | 04 | audit, repair, impute, parse dates | you can list the 7 data-quality dimensions and a test for each |
| 5 | 05 | groupby, pivot, merge (safely) | you check shapes and `_merge` after every merge |
| 6–7 | review | redo hardest problems; skim cheat-sheets | two "hard" problems now feel easy |
| 8 | 06 | line/bar/hist/scatter/box; chart quality | your charts have finding-titles, labelled axes, honest scales |
| 9 | 07 | Seaborn; heatmaps; correlation; confounding | you can explain why levels-correlation fools you |
| 10 | 08 | full EDA on dairy farms | you can run the 8 steps without looking |
| 11–13 | 09 | capstone | rubric self-assessment ≥ 80 |
| 14 | review | compare with the reference solution; plan Phase 2 | you can name 3 things the reference found that you didn't (or vice-versa) |

## Assessment
* **Formative:** 64 auto-checked problems (value answers and chart checks); hidden-answer concept questions; reflections.
* **Summative:** the capstone (100 points; see rubric in notebook 09).

| Score | Meaning |
|---|---|
| 90–100 | Ready for Phase 2; could present this analysis to a client |
| 75–89 | Solid; revisit the sections that lost points |
| 60–74 | Re-do Lessons 4, 5 and 8, then re-attempt the weak sections |
| < 60 | Spend another week on Lessons 3–5, then retry |

## For instructors
* Every lesson's solution notebook is executed and verified; answers are stored in `course_answers.json`.
* Regenerate datasets with `python data/generate_datasets.py` (fixed seed → identical files). Changing `SEED` gives each cohort different data, **but** the answer key would then be invalid.
* Suggested live-session format (90 min): 10 min recap quiz · 25 min live-coding a worked example · 35 min pair-solving problems with hints closed · 20 min debrief on the reflection question.
