# Model Card: <Model name>

**Version / date:** …  **Owner:** …

## 1. Intended use
*What decision does the model support? Who uses it? What should it NOT be used for?*

## 2. Data
*Source, time period, size, target definition, exclusions (and why; leakage, identifiers, protected attributes), known quality issues.*

## 3. Model
*Algorithm and key hyper-parameters, preprocessing, how the threshold (if any) was chosen and from which costs.*

## 4. Performance (on a locked test set, touched once)
| Metric | Value | Baseline |
|---|---|---|
| … | … | … |
*Include uncertainty (CV spread / bootstrap) and error by segment (where does it fail?).*

## 5. Fairness & robustness
*Groups audited, metrics (parity, equal opportunity), uncertainty, proxy checks, limits of the audit.*

## 6. Explanations
*Global drivers (importances/coefficients); how an individual decision is explained (reason codes).*

## 7. Limitations & risks
*Data shifts, rare segments, assumptions (e.g. cost ratio), causal caveats.*

## 8. Monitoring plan
*What to track (score drift, error vs actuals, group metrics), how often, and when to retrain.*
