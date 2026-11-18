# Syllabus: Phase 7: Time-Series Analysis & Forecasting (Weeks 13–15)

**Objective:** develop a strong understanding of sequential data and forecasting techniques, culminating in an honest, leak-free forecasting experiment on agricultural market prices.
**Outcome:** you can develop and evaluate forecasting models using historical time-series data.

> **The one rule that governs the course:** time-series models must respect chronological order. Randomly shuffling observations can leak future information into training and produce misleading evaluation results.

## Learning outcomes (what you can do afterwards)

| Syllabus topic | Where it is taught | You will be able to… |
|---|---|---|
| **1. Fundamentals**: trend, seasonality, cyclicality, noise, stationarity, autocorrelation, partial autocorrelation, lags | L1 | decompose a series, test & fix stationarity, read ACF/PACF |
| **2. Data preparation**: datetime indexing, resampling, frequency conversion, missing timestamps, rolling stats, lag & seasonal features | L2 | turn a messy file into a model-ready, leak-free table |
| **3. Statistical models**: moving average, exponential smoothing, SES, Holt, Holt-Winters, ARIMA, SARIMA, SARIMAX | L4, L5 | fit, diagnose, forecast and interpret them |
| **4. ML forecasting**: linear regression with lags, Random Forest, gradient boosting, XGBoost, feature-based | L6 | frame forecasting as supervised learning without leakage |
| **5. Deep learning**: RNN, LSTM, GRU, Transformer | L7 | build, train and honestly evaluate sequence models |
| **6. Evaluation**: MAE, RMSE, MAPE (zero-care), walk-forward, rolling-origin, horizons, baselines | L3 (+ every lesson) | design a trustworthy evaluation protocol |
| **Libraries**: statsmodels, scikit-learn, XGBoost, pandas, NumPy | all | use them fluently (+ PyTorch in L7) |
| **Practical assignment**: the 8 steps | L8 | run a complete experiment |

## Week-by-week plan

### Week 13: *See the structure, prepare the data*
| Session | Content | Deliverable |
|---|---|---|
| Mon | **L0** Welcome & setup (45 min) · **L1** Fundamentals part 1: components, decomposition | Problems 0.1–0.3, 1.1 |
| Tue | **L1** part 2: stationarity, ACF/PACF | Problems 1.2–1.5 |
| Thu | **L2** Data preparation: datetime, cleaning, resampling, features | Problems 2.1–2.5 |
| Fri | Review & catch-up; re-run `leakage_check` on your own pipelines | - |

### Week 14: *Measure first, then model*
| Session | Content | Deliverable |
|---|---|---|
| Mon | **L3** Evaluation: metrics, baselines, the shuffling experiment, walk-forward | Problems 3.1–3.5 |
| Wed | **L4** Smoothing: MA, SES, Holt, Holt-Winters | Problems 4.1–4.5 |
| Thu–Fri | **L5** ARIMA / SARIMA / SARIMAX | Problems 5.1–5.5 |

### Week 15: *Learn from data, then integrate*
| Session | Content | Deliverable |
|---|---|---|
| Mon | **L6** ML forecasting | Problems 6.1–6.5 |
| Tue | **L7** Deep learning | Problems 7.1–7.5 |
| Wed–Thu | **L8** Guided practical assignment (the 8 steps) | completed notebook + recommendation |
| Fri → | **Capstone project** | see `project/PROJECT_BRIEF.md` |

## Assessment suggestion
| Component | Weight |
|---|---|
| Lesson problems (L0–L7), auto-checks + reasoning comments | 20 % |
| Practical assignment (L8) | 30 % |
| Capstone project (notebook + model card), see `project/RUBRIC.md` | 50 % |

## Prerequisites
Python, NumPy, pandas, matplotlib; basic statistics (mean/variance/correlation, hypothesis tests); scikit-learn workflow (fit/predict, train/test, cross-validation); basic neural-network ideas for L7.

## Recommended reading
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed., free online at otexts.com/fpp3): the best companion text.
- Box, Jenkins, Reinsel & Ljung, *Time Series Analysis: Forecasting and Control*.
- statsmodels time-series documentation; scikit-learn `TimeSeriesSplit` docs.
- Zeng et al. (2023), *"Are Transformers Effective for Time Series Forecasting?"*: a sobering counterpoint for L7.
