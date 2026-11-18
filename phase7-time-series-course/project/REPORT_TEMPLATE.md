# Model Card: Price Forecasting for Region B (beans & potato)

**Author:** … **Date:** … **Notebook:** `capstone_project_starter.ipynb`

## 1. Purpose and users
What is forecast (price, horizons), for whom, and for which decisions?

## 2. Data
- Source & period; rows per commodity.
- Problems found and how each was handled (table).
- Columns used / **deliberately not used** (and why; e.g. `arrivals_tonnes`).

## 3. What the exploration revealed
Bullet each discovery with its evidence (figure/number): seasonality, rain lead (weeks), exchange-rate effect, non-linearity, structural change, stationarity.

## 4. Method
- Models (one line each) and the features they use.
- Evaluation protocol: horizons, dev period, test period, refitting, metric.
- **How leakage was prevented** (point by point).

## 5. Results
| Commodity | Horizon | Selected model | Dev MAE | Test MAE | Naive test MAE | Skill |
|---|---|---|---|---|---|---|
| | | | | | | |

Comment: where is the model clearly better than naive? Where is it not?

## 6. Uncertainty
Interval method, nominal vs observed coverage by horizon, interpretation.

## 7. Failure analysis
Largest errors (dates, size), likely causes, horizon limits, what the model cannot know.

## 8. Recommendation
For each commodity and horizon: use model X / use naive / do not publish a point forecast. State the monitoring and retraining plan.

## 9. Limitations & next steps
Sample size (number of test origins), simulated/real data caveats, untested ideas, what you'd do with another week.

## 10. Honesty statement
- [ ] Hold-out opened once, after the final models were chosen.
- [ ] Anything changed afterwards is listed here: …
