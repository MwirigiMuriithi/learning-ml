# 📈 Time-Series Analysis & Forecasting
### Phase 7 · Weeks 13–15 · From "what is a time series?" to an honest forecasting system for agricultural market prices

> *"Prediction is very difficult, especially if it's about the future."* by Niels Bohr (probably)
> The craft of forecasting is therefore **not** building the cleverest model: it is **knowing whether to believe it**.

This is a complete, hands-on course: 9 teaching notebooks, a reference solution for every problem set, a guided practical assignment, and a capstone project
with a hidden-structure dataset and a locked hold-out year. Every notebook ships **already executed** (so you can read the results on GitHub/VS Code/Colab before running anything) and was verified end to end.

---

## 🧭 What you will learn

| # | Notebook | Core ideas |
|---|---|---|
| 0 | `00_Welcome_and_Setup` | Forecasting mindset, environment check, your first (humbling) forecast |
| 1 | `01_Time_Series_Fundamentals` | Trend, seasonality, cycles, noise, decomposition (classical & STL), stationarity (ADF/KPSS), differencing, ACF, PACF, lags |
| 2 | `02_Data_Preparation` | Datetime index, messy data, missing timestamps, resampling, rolling stats, lag & seasonal (Fourier) features, **leakage detection** |
| 3 | `03_Forecast_Evaluation` | MAE, RMSE, MAPE (and its traps), sMAPE, MASE, baselines, why shuffling is poison (experiment!), **walk-forward / rolling-origin**, horizons |
| 4 | `04_Statistical_Smoothing_Models` | Moving average, SES, Holt, damped trend, Holt-Winters, ETS prediction intervals |
| 5 | `05_ARIMA_SARIMA_SARIMAX` | AR/MA/I, Box–Jenkins, diagnostics, SARIMA, Fourier terms for weekly seasonality, SARIMAX with lagged drivers |
| 6 | `06_ML_Forecasting` | Supervised framing, return targets, **why trees can't extrapolate**, Linear/Ridge, Random Forest, Gradient Boosting, XGBoost, direct vs recursive, `TimeSeriesSplit` tuning, permutation importance |
| 7 | `07_Deep_Learning_Forecasting` | Windows, RNN (by hand!), vanishing gradients (measured), LSTM, GRU, Transformer/self-attention, early stopping, seeds & ensembles |
| 8 | `08_Practical_Assignment` | The guided 8-step experiment: visualise → decompose → MA baseline → ARIMA/SARIMAX → lag regression → Random Forest → XGBoost → compare |
| 🏆 | `project/` | Capstone: a new market (beans & potato) with a hidden lag structure, a non-linear effect, a structural break, messy data, and a **locked 2024 hold-out** |

## 🗂️ Folder map

```
ts_course/
├── README.md                  ← you are here
├── requirements.txt
├── generate_data.py           ← rebuilds every dataset (readable recipe!)
├── course_utils.py            ← shared helpers: loaders, metrics, walk-forward, features, leakage_check ...
├── dl_utils.py                ← PyTorch helpers for Lesson 7 (RNN/LSTM/GRU/Transformer, training loop)
├── data/                      ← datasets (weekly market, messy daily file, airline, CO2, sunspots)
├── notebooks/                 ← Lessons 0-8 (problems + hints inside)
├── solutions/                 ← reference solutions for every problem + assignment + capstone
├── project/                   ← capstone brief, rubric, starter notebook, data, harness
├── docs/                      ← syllabus, cheat sheet, glossary, data dictionary, teaching notes
└── build/                     ← (optional) the Python sources the notebooks were generated from
```

## 🚀 Quick start

```bash
# 1. create an environment (recommended)
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. start Jupyter FROM THE COURSE FOLDER (important: the notebooks find ../data and ../course_utils.py)
jupyter lab
```
Open `notebooks/00_Welcome_and_Setup.ipynb` and follow the course map.

**Google Colab:** upload the whole folder to Drive, mount it, then run `%cd /content/drive/MyDrive/ts_course/notebooks` in the first cell (and `!pip install xgboost statsmodels` if needed; Colab already has the rest, including PyTorch).

## 🧑‍🏫 How to learn from this course

1. **Predict, then run.** Before each cell, guess what it will print. Prediction before execution builds intuition faster than anything else.
2. **Struggle productively.** Each problem has progressive hints (open one at a time). Solutions are in `solutions/`: read them *after* an honest attempt, and read the *"Why this works"* explanation even if you were right.
3. **Type the code.** Don't just run cells: change parameters, break things, see what happens.
4. **Keep the golden rule:** *never let the model see the future.* Every lesson returns to it.
5. **Compare against baselines**: always.

### 🎚️ Problem difficulty
🟢 Easy (warm-up) · 🟡 Medium (the core skill) · 🔴 Challenge (requires combining ideas). Problems with `check(...)` cells give instant ✅/❌ feedback.

## ⏱️ Suggested pacing (≈ 45–50 hours total)

| Week | Lessons | Study time |
|---|---|---|
| 13 | 0, 1, 2 | ≈ 9 h (including problems) |
| 14 | 3, 4, 5 | ≈ 12 h |
| 15 | 6, 7, 8 + start the capstone | ≈ 14 h + capstone |
| after | Capstone project | ≈ 8–12 h |

## 🧪 About the data

- **Agricultural market data are SIMULATED** from a transparent recipe in `generate_data.py` (trend + harvest seasonality + lagged rainfall + fuel/exchange-rate pass-through + AR noise + droughts). Because *we know the truth*, you can check whether a model recovers it: a luxury real data never offers. Prices are in "currency units per kg"; regions and crops are illustrative.
- **Classic datasets are REAL:** Box & Jenkins airline passengers (monthly 1949–60), Mauna Loa CO₂ (weekly, from `statsmodels`), and annual sunspot numbers (from `statsmodels`).
- After the course, **repeat your best workflow on real market-price data** (national market information systems, FAO GIEWS/FPMA, FEWS NET, USDA AMS and similar open portals publish weekly/monthly series). Check licences, and expect messier data than ours.

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: Cannot find the data/ folder` | Start Jupyter from the course root or open notebooks from `notebooks/` |
| `ValueError: Invalid frequency: M` | You're on an old tutorial's alias: this course uses `"ME"` (pandas ≥ 2.2) |
| `ModuleNotFoundError: torch` | Only Lesson 7 needs it: `pip install torch` (CPU build is fine): or skip Lesson 7 for now |
| A cell takes minutes | Walk-forward cells refit many models; 1–4 minutes is normal in Lessons 5–7 and the capstone. Reduce origins (`step=16`) while experimenting |
| Slightly different numbers than the ones in the text | Library versions (statsmodels optimiser, XGBoost) can shift third decimals. Conclusions should hold; if they don't, that is interesting: investigate |
| Re-generate datasets | `python generate_data.py` (deterministic seeds) |

## 📜 Credits & licence
Course materials © the course author: free for personal and educational use. The airline-passengers series is from Box, Jenkins & Reinsel; CO₂ and sunspot data ship with `statsmodels` (BSD-3). Pedagogical inspirations: Hyndman & Athanasopoulos' *Forecasting: Principles and Practice*, Box–Jenkins methodology, and the teaching style of David Malan (CS50) and Charles Severance (Dr. Chuck): *explain it like they're curious, never like they're slow.*
