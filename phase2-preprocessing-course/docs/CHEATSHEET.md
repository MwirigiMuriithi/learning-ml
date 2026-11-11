# 🧾 Cheat sheet: Phase 2 on one page

## The golden rules
1. **Look first** (`info`, `describe`, `profile`, `value_counts`). 2. **Split first, then `fit` on train only.** 3. **Everything learned from data lives inside a Pipeline.** 4. **Document each step: what, why, learns from data?**

## Cleaning
| Task | Code |
|---|---|
| Text → number | `pd.to_numeric(s.str.extract(r"(\d+\.?\d*)", expand=False))` |
| Mixed date formats | `pd.to_datetime(s, format="mixed")` |
| Duplicates | `df.duplicated(subset=key)`; keep most complete: sort by `isna().sum(axis=1)` then `drop_duplicates(keep="first")` |
| Standardise labels | `lower → strip → regex letters-only → synonym dict → difflib.get_close_matches` |
| Validity rules | `df[c].where(df[c].between(lo, hi))` |
| Missing: % per column | `(df.isna().mean()*100).sort_values()` |
| Missing: fill | median (skewed) · mean (symmetric) · mode / "Unknown" (categorical) · group median · `SimpleImputer`, `KNNImputer` · add indicator if informative |
| Outliers | z>3 (non-robust) · IQR 1.5× · MAD>3.5 · per-group · `IsolationForest` |
| Outlier treatment | remove (errors) · cap (`clip`) · log · flag · robust model |

## Transformation
| Need | Tool |
|---|---|
| [0,1] range | `MinMaxScaler` (outlier-sensitive) |
| mean 0 / std 1 | `StandardScaler` |
| outliers present | `RobustScaler` |
| right skew | `np.log1p`, `PowerTransformer("yeo-johnson")` |
| nominal | `OneHotEncoder(handle_unknown="ignore")` |
| ordinal | `OrdinalEncoder(categories=[[...]])` ← explicit order! |
| high cardinality | frequency / `TargetEncoder` (cross-fitted) |
| scale-sensitive models | KNN, SVM, linear/regularised, neural nets, PCA, K-means |
| scale-insensitive | decision trees, random forests, boosting |

## Feature engineering
Dates `.dt.year/month/dayofyear/quarter`; durations `(b-a).dt.days`; cyclical `sin/cos(2π·x/period)`;
ratios (guard denominators), sums, interactions, `(x - optimum)**2`; bins `pd.cut`/`qcut`;
time series **per group**: `g.diff()`, `g.pct_change()`, `g.transform(lambda s: s.rolling(7).mean())`, `g.shift(k)` · trailing windows only · chronological splits with a gap.

## Feature selection
Filter: `VarianceThreshold`, corr/VIF, `f_regression`, `mutual_info_regression`, `chi2`/`f_classif` · Wrapper: `RFE`, `RFECV` · Embedded: `LassoCV`, tree & **permutation** importance · Reduction: `PCA` (scale first).
Add canary noise columns; keep what clears the noise floor *and* helps cross-validation.

## Leakage checklist
- [ ] Only information available at prediction time · [ ] split first (time → chronological; groups → `GroupKFold`) · [ ] de-duplicate before splitting
- [ ] scalers, imputers, encoders, selectors, PCA, caps **inside** the Pipeline · [ ] too-good score? compare with naive baseline, check importances, shuffled-target test

## Pipeline template
```python
pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("cap", Winsorizer(k=10)), ("scale", StandardScaler())]), NUM),
    ("nom", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), NOM),
    ("ord", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(categories=ORDER, handle_unknown="use_encoded_value", unknown_value=-1))]), ORD),
]).set_output(transform="pandas")
pipe = Pipeline([("dates", DateFeatures(harvest_col=None)), ("feats", AgriFeatures()), ("prep", pre),
                 ("select", SelectKBest(f_regression, k="all")), ("model", Ridge())])
cross_val_score(pipe, X, y, cv=GroupKFold(5), groups=farm_id, scoring="r2")
joblib.dump(pipe.fit(X, y), "pipeline.joblib")
```
