# 📚 Syllabus: Phase 2: Data Preprocessing & Feature Engineering

## Course description
Raw data is never ready for machine learning. This two-week course teaches the professional workflow for turning raw, messy tabular data into trustworthy numeric inputs (cleaning, transforming, engineering and selecting features) and, above all, doing it **without data leakage**, using scikit-learn pipelines. Everything is practised on an agricultural yield dataset.

## Learning outcomes
By the end you can:
1. **Diagnose** data-quality problems: types, duplicates, missingness mechanisms, inconsistent labels, invalid values, outliers.
2. **Clean** data with documented, repeatable code (`clean_farm()`), separating *structural* from *statistical* cleaning.
3. **Transform** features: scale (min-max, standard, robust), reduce skew (log, power), encode (one-hot, ordinal, frequency, target).
4. **Engineer** features: dates (cyclical), ratios/interactions/domain indices, bins, rolling averages, lags, seasonal indicators.
5. **Select** features: correlation/VIF, F-test, mutual information, RFE/RFECV, permutation importance, Lasso, PCA.
6. **Recognise and prevent leakage** (target, contamination, selection, temporal, group, duplicate).
7. **Build, tune, persist and document** a scikit-learn preprocessing pipeline.

## Week plan
| Week | Day | Lesson | Focus |
|---|---|---|---|
| 3 | Mon | 00, 01 | Big picture, types, duplicates, missing values |
| 3 | Tue–Wed | 02 | Inconsistency, validity rules, outliers |
| 3 | Thu–Fri | 03 | Scaling, log, encoding |
| 4 | Mon | 04 | Feature engineering (dates, windows, lags) |
| 4 | Tue | 05 | Feature selection & PCA |
| 4 | Wed | 06 | Data leakage |
| 4 | Thu | 07 | Pipelines |
| 4 | Fri + weekend | Project | Capstone |

## Assessment
| Component | Weight |
|---|---|
| Practice problems (self-checked against solutions) | formative |
| Lesson quizzes | formative |
| **Capstone project** (see `project/RUBRIC.md`) | **100%** |

## Tools
Python · pandas · NumPy · scikit-learn (preprocessing, feature selection, pipelines, model selection) · matplotlib · Jupyter.

## Expected outcome
You can take a raw dataset and produce structured, machine-learning-ready data, with a pipeline you can defend, reproduce and deploy.

## Recommended further reading
* *scikit-learn user guide:* "Preprocessing data", "Pipelines and composite estimators", "Common pitfalls and recommended practices" (data leakage).
* Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*: section on the *wrong and right way to do cross-validation* (the origin of the select-then-CV experiment in Lesson 6).
* Kuhn & Johnson, *Feature Engineering and Selection*.
* Wickham, "Tidy Data" (Journal of Statistical Software, 2014).
