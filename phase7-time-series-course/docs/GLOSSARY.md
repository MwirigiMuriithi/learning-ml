# 📖 Glossary

**ACF (autocorrelation function)**: correlation of a series with its own lag *k*, plotted for k = 0, 1, 2…
**ADF test**: Augmented Dickey–Fuller; H₀: unit root (non-stationary). Small p ⇒ stationary.
**AIC / BIC**: information criteria: fit penalised for complexity (BIC penalises harder). Lower = better; comparable only for the same data.
**AR(p)**: autoregressive model: value depends linearly on its previous *p* values.
**ARIMA(p,d,q)**: AR(p) + differencing *d* times + MA(q).
**Autocorrelation**: correlation between observations at different times.
**Backtesting**: evaluating a forecasting method on historical data as if forecasting in real time (see walk-forward).
**Baseline**: simple benchmark forecast (naive, seasonal naive, drift, mean) any model must beat.
**Boosting**: training models sequentially, each correcting the previous one's errors (GBM, XGBoost).
**Cyclicality**: rises and falls with no fixed period (e.g. boom–bust, solar cycle).
**Damped trend**: Holt variant whose slope flattens with the horizon.
**Decomposition**: splitting a series into trend, seasonal and remainder components (classical, STL).
**Differencing**: replacing y_t by y_t − y_{t−1} (or y_t − y_{t−m}) to remove trend (or seasonality).
**Direct strategy**: a separate model for each forecast horizon.
**Early stopping**: halting training when validation loss stops improving.
**Exogenous variable**: outside driver of the series (rain, fuel, holidays). Must be known at forecast time.
**Expanding window**: training set grows as the origin moves forward (vs *sliding window*).
**Forecast horizon (h)**: how many steps ahead we predict.
**Forecast origin**: the last time point whose data is used for a forecast.
**Fourier terms**: sin/cos pairs that model a smooth seasonal shape with few parameters.
**GRU / LSTM**: gated recurrent networks that mitigate vanishing gradients.
**Holt-Winters**: exponential smoothing with level, trend and seasonal components.
**KPSS test**: H₀: stationary. Small p ⇒ non-stationary.
**Lag**: a past value of the series (y_{t−k}).
**Leakage**: information from the future (or from the target) sneaking into training/features, inflating scores.
**Ljung–Box test**: tests whether residual autocorrelations are jointly zero (white noise).
**MA(q) / moving average**: (model) value depends on past shocks; (smoother) average of the last *w* observations.
**MAE / RMSE / MAPE / sMAPE / MASE**: error metrics (see Lesson 3).
**Multiplicative vs additive**: whether components multiply (swings grow with level) or add (constant swings).
**Naive forecast**: next value = last value. **Seasonal naive**: next value = value one season ago.
**PACF**: partial autocorrelation: direct effect of lag *k* after removing lags 1…k−1.
**Permutation importance**: drop in performance when one feature is shuffled on validation data.
**Prediction interval**: range expected to contain the future value with stated probability.
**Recursive strategy**: one-step model applied repeatedly, feeding forecasts back as inputs.
**Residual**: actual minus fitted/forecast value; should look like white noise for a good model.
**Rolling-origin evaluation**: walk-forward: repeat train→forecast→score at successive origins.
**SARIMA(p,d,q)(P,D,Q)ₘ**: ARIMA with seasonal AR/MA/differencing at period *m*. **SARIMAX** adds exogenous regressors.
**Seasonality**: repeating pattern with a fixed, known period.
**Skill score**: 1 − error_model / error_baseline (>0 beats the baseline).
**Stationarity**: statistical properties (mean, variance, autocovariance) don't depend on time.
**STL**: Seasonal-Trend decomposition using LOESS; robust, allows changing seasonality.
**Structural break**: permanent change in the data-generating process (policy, technology).
**TimeSeriesSplit**: scikit-learn splitter producing expanding-window, chronological folds.
**Transformer / self-attention**: architecture where every position attends to every other.
**Trend**: long-run direction.
**Unit root / random walk**: non-stationary process where shocks never die out.
**Vanishing gradient**: error signal shrinking exponentially through many recurrent steps.
**Walk-forward validation**: see rolling-origin evaluation.
**White noise**: uncorrelated, mean-zero, constant-variance sequence; the *ideal* residual.
