# EDA Checklist & Report Template

Print this. Tick it off on **every** new dataset.

## 0 · ASK
- [ ] What is this data for? Who will use the result, for what decision?
- [ ] **What does one row mean?** (the unit of analysis)
- [ ] Is there a data dictionary? If not, write one and *verify* each guess.

## 1 · LOOK
- [ ] shape, columns, dtypes, memory (`df.info()`)
- [ ] `head()`, `tail()`, **`sample()`** (head shows the best-behaved rows)
- [ ] numeric vs categorical vs date vs ID columns; cardinality of each

## 2 · AUDIT (seven dimensions)
- [ ] **Completeness**: missing counts; *is missingness random?* (by group)
- [ ] **Uniqueness**: exact duplicates; duplicate keys
- [ ] **Validity**: impossible values (≤ 0, negative, > physical limits)
- [ ] **Consistency**: spelling/case/whitespace; **units**; mixed formats
- [ ] **Plausibility**: compare each value with its own group's neighbourhood
- [ ] **Continuity**: date range, gaps, regular spacing
- [ ] **Integrity**: keys match across tables; row counts stable after merges
- [ ] Cross-column logic (derived columns as free witnesses)
- [ ] Order of fixes: duplicates → spelling → types → **units** → impossible/outliers → gaps
- [ ] Decision log written; `assert` quality gates pass; imputed rows flagged

## 3 · DESCRIBE (one variable at a time)
- [ ] centre (mean/**median**), spread (std/IQR/**CV**), min/max
- [ ] shape: histogram + skew; log-transform if strongly right-skewed
- [ ] categories: counts/shares; rare levels; class balance

## 4 · RELATE (two variables)
- [ ] numeric × numeric: scatter + correlation (Pearson **and** Spearman)
- [ ] categorical × numeric: box/violin by group; group means with counts
- [ ] categorical × categorical: normalised crosstab heatmap

## 5 · CONTROL (three variables)
- [ ] Stratify by suspected confounders; colour scatter by group
- [ ] Remove trends/seasonality before correlating time series
- [ ] Ask: is a *group* hiding it? is a *trend* faking it? is a *third variable* behind it? how many points?

## 6 · HUNT (anomalies)
- [ ] IQR rule · z-score · domain rules
- [ ] Outlier ≠ error: check internal consistency before removing

## 7 · TREND (if there is time)
- [ ] trend · seasonality · residual shocks · autocorrelation · naive baseline

## 8 · CONCLUDE
- [ ] Findings with **numbers**; hypothesis table (observation → hypothesis → test → ML implication)
- [ ] ML-readiness: target, leakage, redundancy, encoding, scaling, balance, split strategy, baseline
- [ ] Limitations stated honestly

---

## Report skeleton
```
1. Overview               source · rows × columns · period · what one row is · data dictionary
2. Data-quality assessment issue | rows | evidence | decision  (+ decision log)
3. Descriptive statistics  centre · spread · shape of the key variables
4. Visual exploration      each chart: finding as title, units on axes
5. Correlation analysis    heatmap + stratified / detrended / lagged checks
6. Observations & hypotheses   table
7. ML-readiness            target · features · leakage · split · baseline · prep list
8. Limitations & next steps
```

**Every section ends with "So what?": 2–4 plain-English sentences.**
