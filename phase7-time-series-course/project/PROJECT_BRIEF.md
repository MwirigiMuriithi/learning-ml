# 🏆 Capstone Project: Forecasting Agricultural Market Prices for Region B

**Time:** 8–12 hours over Week 15 and beyond · **Work:** individual (or pairs, if your instructor allows) · **Deliverables:** completed notebook + 1–2 page model card

## 1 · The scenario
The Region B Agricultural Marketing Board publishes weekly wholesale prices for **beans** and **potato**. Traders, cooperatives and the ministry want forecasts **1, 4, 8 and 12 weeks ahead**: and an honest statement of how far to trust them.
The board has given you ten years of history (2013–2023). A *separate sealed file* holds 2024. Your job is to build, validate and defend a forecasting system, then **test it exactly once** on 2024.

## 2 · What makes this project different from the lessons
You have not seen this market. Its data are messy, and *how it works is for you to discover*: how weather and the exchange rate act on prices (with what delays?), whether the relationships are linear, and whether the market's structure stayed constant. A fair number of the "obvious" modelling choices from the lessons will need adapting. Evidence beats intuition.

## 3 · Requirements

### 3.1 Data (Sections 1–2 of the starter)
- Audit the raw file and document **every** problem found.
- Clean it into a regular weekly series per commodity, with **no future information** used to fix the past.

### 3.2 Discovery (Section 3)
Support each finding with a figure or number: trend/seasonality, lead–lag relationships, non-linearity, structural change, stationarity.

### 3.3 Models (Section 5): at least five, per commodity
- ≥ 2 **baselines** (naive + seasonal naive or moving average / exponential smoothing),
- ≥ 1 **ARIMA-family** model (ARIMA/SARIMAX; use Fourier terms for weekly seasonality),
- ≥ 1 **ML model on lag features** (Ridge / RF / GBM / XGBoost), forecasting **returns**,
- ≥ 1 **idea of your own** (blend, sliding window or break indicator, threshold feature, deep model, global model across commodities, …).

### 3.4 Evaluation protocol (non-negotiable)
- Horizons **1, 4, 8, 12** weeks.
- **Development period** (origins 2022-01 → 2023-12, every 4 weeks, expanding window): *all* model selection.
- **Test period:** the 2024 file, evaluated **once**, with the models you selected on the dev period.
- Metrics: MAE and skill vs naive at least; report by horizon *and* commodity.
- Use the provided harness (`project_utils.run_walk_forward`): it hands models only the past.

### 3.5 Uncertainty
Provide prediction intervals for at least one model (analytic, or empirical from dev errors) and **check their coverage** on the hold-out.

### 3.6 Failure analysis & recommendation
Find where the models fail (which weeks, which horizon, which crop) and propose why. Recommend what the Board should do: which model for which horizon, and where **not** to trust a forecast.

## 4 · Leakage checklist (must be satisfied: and mentioned in your report)
- [ ] No shuffled splits; training always precedes validation/test.
- [ ] `arrivals_tonnes` (recorded concurrently with price) is not used as a predictor.
- [ ] Every driver (rain, temperature, exchange rate) is lagged **≥ the horizon** (or you explain why not).
- [ ] No centred windows, backward fill or interpolation using future values in your cleaning/features.
- [ ] Scalers/imputers are fitted only on data available at the origin.
- [ ] Hyper-parameters and model choice come from the dev period only.
- [ ] The hold-out file was opened once, after the models were final. (If you changed anything afterwards, you **say so**.)

## 5 · Suggested timeline
| Block | Work | Time |
|---|---|---|
| 1 | Audit, clean, EDA & discovery | 2–3 h |
| 2 | Baselines + ARIMA-family | 2 h |
| 3 | ML models, blends, your own idea | 2–3 h |
| 4 | Dev comparison, selection, intervals | 1–2 h |
| 5 | One-shot test, failure analysis, model card | 1–2 h |

## 6 · Stretch goals
- A **global** neural network across both commodities (Lesson 7's `dl_utils` is a starting point).
- **Regime-aware** model for the beans structural break.
- **Conformal** or quantile prediction intervals.
- A one-page **decision memo** for non-technical readers (what would you tell the minister?).
- Reproduce the pipeline on **real** price data from an open market-information source.

## 7 · Files
| File | Purpose |
|---|---|
| `capstone_project_starter.ipynb` | your working notebook (scaffolded) |
| `project_utils.py` | evaluation harness (read it!) |
| `data/` | training data, 🔒 hold-out, data dictionary |
| `RUBRIC.md` | how you are graded |
| `REPORT_TEMPLATE.md` | model-card template |
| `ANSWER_KEY.md` | 🔐 hidden structure of the simulation: **after** submission only |
| `../solutions/09_Capstone_Project_Reference_Solution.ipynb` | 🔐 reference solution: after submission only |

> *"The goal is not to predict the future perfectly. The goal is to know, with evidence, how wrong you are likely to be."*
