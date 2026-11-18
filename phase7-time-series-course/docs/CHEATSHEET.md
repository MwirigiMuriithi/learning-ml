# 🧾 Forecasting Cheat Sheet

## The golden rules
1. **Chronological order only**: train < validation < test. Never shuffle.
2. **Every feature must be knowable at the forecast origin** (lag ≥ horizon).
3. **Fit scalers/imputers/selectors on training data only.**
4. **Baselines first** (naive, seasonal naive, drift). Report skill vs naive.
5. **Walk-forward** (many origins), report **by horizon**.
6. **Touch the final test set once.**

## pandas time-series one-liners
```python
df["date"] = pd.to_datetime(df["date"]);  df = df.sort_values("date").set_index("date")
df.loc["2023-06"]                                   # partial-string slicing
df.asfreq("D")                                      # make gaps explicit (NaN)
s.resample("W-SUN").agg({"price": "mean", "rain": "sum"})   # level -> mean, flow -> sum   ("ME","QE","YE" for month/quarter/year END)
s.shift(k)                                          # lag k   (shift(-k) is a LEAD = future! only for targets)
s.shift(1).rolling(4).mean()                        # trailing mean EXCLUDING today
s.rolling(5, center=True)                           # ⚠ peeks into the future: charts only
s.ewm(alpha=.2).mean();  s.expanding().mean()
np.log(s).diff()                                    # weekly log return;  .diff(52) = seasonal difference
s.ffill()   # causal fill;   s.interpolate() / s.bfill()  # use future info -> leakage risk
```

## Diagnostics
```python
from statsmodels.tsa.seasonal import STL;  STL(np.log(y), period=52, robust=True).fit()
from statsmodels.tsa.stattools import adfuller, kpss      # ADF: small p => stationary; KPSS: small p => NON-stationary
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox   # small p => residuals still autocorrelated
```
| Pattern | ACF | PACF |
|---|---|---|
| AR(p) | decays | **cuts off at p** |
| MA(q) | **cuts off at q** | decays |
| Non-stationary | very slow decay | spike at lag 1 |

## Models
```python
# Exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12).fit().forecast(h)
# ARIMA family (use log price; back-transform with np.exp)
from statsmodels.tsa.statespace.sarimax import SARIMAX
SARIMAX(np.log(y), exog=X, order=(1,1,1), seasonal_order=(0,1,1,12), trend="n").fit(disp=False).get_forecast(h)
# long seasonality (weekly, m=52): Fourier terms as exog instead of seasonal_order
# ML on returns:  target = log p_t - log p_{t-h};  forecast = p_{t-h} * exp(pred)
from xgboost import XGBRegressor;  from sklearn.model_selection import TimeSeriesSplit
TimeSeriesSplit(n_splits=5, gap=h-1)
```

## Choosing a model: decision guide
| Situation | Try first |
|---|---|
| Random-walk-like price, short horizon | naive; Ridge on returns |
| Strong seasonality, one series | Holt-Winters / SARIMAX (Fourier) |
| Known drivers (weather, fuel, FX) with lags ≥ horizon | SARIMAX / XGBoost with lagged drivers |
| Non-linear effects, many features | Gradient boosting / XGBoost |
| Many related series, lots of data | global GBM or neural network |
| Need prediction intervals quickly | ETS / SARIMAX (parametric) or empirical errors / conformal |

## Metrics
| Metric | Use | Watch out |
|---|---|---|
| MAE | headline, same units | - |
| RMSE | big misses costly | dominated by outliers |
| MAPE | % reporting | **undefined at 0**, explodes near 0, favours under-forecasts |
| sMAPE | % with symmetric-ish scaling | unstable when both ≈ 0 |
| MASE | across series; <1 beats naive | needs the right seasonal scale `m` |
| Skill | `1 − metric/metric_naive` | always report it |

## Leakage checklist (print me)
- [ ] Shuffled split? (no!)
- [ ] `center=True`, negative `shift`, `bfill`, `interpolate` before splitting?
- [ ] Scaler / imputer / PCA fitted on all data?
- [ ] Feature from the same week as the target (arrivals, today's rain)?
- [ ] Hyper-parameters tuned on the final test set?
- [ ] Walk-forward training window includes only the past?
- [ ] `leakage_check(...)` passes ("future-poisoning test")?
