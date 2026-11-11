# Glossary (plain English)

**Aggregation**: collapsing many values into one number (sum, mean, max…).
**Array (ndarray)**: NumPy's grid of same-typed numbers; the foundation of data science in Python.
**Autocorrelation**: how strongly a series is correlated with a *lagged copy of itself* (is this week like last week?).
**Axis**: a direction in an array: `axis=0` runs down the rows, `axis=1` across the columns. Reducing along an axis removes it.
**Baseline**: a simple benchmark (e.g. "same as last week") that any model must beat.
**Bin**: one bar-width range of a histogram.
**Boolean mask**: array of True/False used to select elements.
**Bootstrap**: resampling your data with replacement to estimate uncertainty.
**Broadcasting**: NumPy stretching a smaller array across a larger one when shapes are compatible (equal or 1, compared from the right).
**Categorical variable**: takes values from a set of labels (crop, county).
**Coefficient of variation (CV)**: std ÷ mean; relative spread, comparable across different scales.
**Confounder**: a hidden variable that influences both things you are comparing, creating a misleading association.
**Correlation (Pearson r)**: strength of straight-line association, −1…+1. **Spearman**: correlation of ranks; robust to skew/outliers.
**DataFrame / Series**: pandas' table / column, with labelled rows and columns.
**Data leakage**: using information in training that wouldn't be available at prediction time (or is computed from the target).
**Decimal slip**: a typo that shifts a decimal point (48 → 480).
**Detrend**: remove the long-run drift (differences, % changes, dividing by a trend) before analysing a time series.
**dtype**: data type of an array/column (`float64`, `int64`, `str`, `datetime64`).
**ECDF**: empirical cumulative distribution: share of values at or below each x.
**EDA**: exploratory data analysis: understanding data by summarising and visualising it before modelling.
**Fancy indexing**: selecting with a list/array of positions.
**Feature**: an input column for a model. **Target**: the column you want to predict.
**Grain**: what one row represents (weekly price, monthly weather, annual production). Merge tables at the same grain.
**Heatmap**: a grid coloured by value.
**Hypothesis**: a testable explanation for an observation.
**Imputation**: filling in missing values.
**IQR**: interquartile range, Q3 − Q1; the spread of the middle 50%.
**KDE**: kernel density estimate: a smoothed histogram.
**Lag**: a value from an earlier time (`shift(1)` = last period).
**Merge / join**: combining tables on a key. *Inner* keeps matches only; *left* keeps all left rows; *outer* keeps everything.
**MCAR / MAR / MNAR**: missing completely at random / at random given other columns / not at random (depends on the missing value itself).
**NaN**: "not a number": the standard missing-value marker.
**Outlier**: an unusual value. A question, not a verdict: could be an error *or* a discovery.
**Overplotting**: so many points that a scatter becomes a blob (use alpha, sampling, hexbins).
**Pivot table**: two-way summary: one grouping on rows, another on columns.
**Quartile**: the 25th/50th/75th percentiles.
**Resample**: change a time series' frequency (weekly → monthly).
**Seasonality**: a pattern that repeats every year (or week, or day).
**Skewness**: asymmetry of a distribution; positive = long right tail.
**Stratify**: analyse within subgroups to control for a variable.
**Tidy (long) data**: each row an observation, each column a variable.
**Time series**: data ordered in time.
**Transform (pandas)**: group calculation returning a result aligned with the original rows.
**Ufunc**: NumPy "universal function" applied element-wise (`np.sqrt`, `np.log`…).
**Vectorisation**: expressing operations on whole arrays instead of looping over elements.
**View vs copy**: a view shares memory with the original array (editing it changes both); a copy is independent.
**z-score**: (value − mean) ÷ std; how many standard deviations from average.
