# Forecasting & Deep Learning for Agricultural Markets
### A hands-on course: Time-Series Analysis (Phase 7) → Neural Networks (Phase 8) → Capstone

> *"No magic."* Every idea is built from small pieces you can run, break and fix: first by hand (NumPy), then with frameworks (statsmodels, scikit-learn, XGBoost, PyTorch, Keras).

**What you get:** 21 executed lesson notebooks (+ a fully-worked copy of each), a shared helper library, reproducible datasets, 90 auto-graded exercises with progressive hints, 4 projects, a capstone, cheat sheets and an instructor guide.

---

## 1 · Quick start

```bash
# 1. (recommended) create an environment
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. launch
jupyter lab
```
Open **`00_start_here/00_START_HERE.ipynb`** first. Everything runs on a laptop **CPU** (no GPU needed); the slowest notebook takes ~3 minutes.

Tested with Python 3.12, NumPy 2.5, pandas 3.0, scikit-learn 1.9, statsmodels 0.15, XGBoost 3.4, PyTorch 2.15, Keras 3.15. Older versions (pandas ≥ 2.2, torch ≥ 2.2) should work.

## 2 · Folder map

```
00_start_here/        Welcome, toolkit check, data tour, first graded exercise
phase7_timeseries/    Lessons 7.1 – 7.8  (student notebooks: starter cells + hints + checks)
phase8_deep_learning/ Lessons 8.1 – 8.11
capstone/             The capstone notebook + PROJECT_BRIEF.md
solutions/            The same notebooks with every exercise solved (look AFTER you try!)
data/                 Datasets (+ README.md data dictionary)
src/agrilab.py        The helper library: data generators, metrics, walk-forward, feature builders, autograder
build/                Authoring kit that generated these notebooks (see "For authors")
CHEATSHEETS.md        One-page references for time series and deep learning
INSTRUCTOR_GUIDE.md   Pacing, assessment, misconceptions, discussion prompts
```

## 3 · The syllabus

### Phase 7: Time-Series Analysis and Forecasting (weeks 13–15)
| # | Notebook | Core ideas | Time |
|---|---|---|---|
| 7.1 | `P7_01_time_series_fundamentals` | components, decomposition (STL), stationarity (ADF/KPSS), differencing, ACF/PACF, lags | 75 min |
| 7.2 | `P7_02_data_preparation` | datetime index, gaps, outliers, resampling, rolling stats, lag/seasonal features, leak tests | 90 min |
| 7.3 | `P7_03_evaluation_and_baselines` | MAE/RMSE/MAPE/sMAPE/MASE, shuffling leak experiment, baselines, walk-forward, horizons | 90 min |
| 7.4 | `P7_04_smoothing_methods` | moving average, SES, Holt, damped trend, Holt-Winters | 90 min |
| 7.5 | `P7_05_arima_sarima_sarimax` | AR/MA/I, Box-Jenkins, AIC, diagnostics, SARIMA, SARIMAX with Fourier terms and lagged rain | 100 min |
| 7.6 | `P7_06_ml_forecasting` | lag-feature tables, extrapolation trap, Ridge, RF, GBM, XGBoost, direct vs recursive, TimeSeriesSplit | 110 min |
| 7.7 | `P7_07_deep_learning_forecasting` | windows, RNN/LSTM/GRU/Transformer forecasters, seeds, embargo | 100 min |
| 7.8 | `P7_08_practical_assignment` | **the full forecasting experiment** (your 8 steps + report) | 3 h |

### Phase 8: Deep Learning and Neural Networks (weeks 16–17)
| # | Notebook | Core ideas | Time |
|---|---|---|---|
| 8.1 | `P8_01_neurons_and_perceptrons` | neuron, weights/bias, activations, perceptron rule, XOR, layers, parameter counting | 90 min |
| 8.2 | `P8_02_backpropagation_from_scratch` | losses, gradient descent, learning rate, epochs/batches, forward/backprop, gradient checking | 110 min |
| 8.3 | `P8_03_pytorch_fundamentals` | tensors, autograd, `nn.Module`, `DataLoader`, the training loop | 100 min |
| 8.4 | `P8_04_optimization_and_regularization` | SGD→Adam, LR range test, schedulers, dropout, batch-norm, weight decay, early stopping | 110 min |
| 8.5 | `P8_05_project_digit_recognition` | **Project 1**: ANN → CNN, confusion matrix, error analysis | 100 min |
| 8.6 | `P8_06_cnn_image_classification` | convolution by hand, shapes, pooling, feature maps, augmentation | 110 min |
| 8.7 | `P8_07_project_crop_disease` | **Project 3**: leaf classifier, recall/precision, abstention, occlusion maps, transfer learning | 2 h |
| 8.8 | `P8_08_rnn_lstm_gru` | recurrent cell, vanishing gradients, LSTM/GRU gates, forget-bias trick | 110 min |
| 8.9 | `P8_09_transformers` | attention, multi-head, positional encoding, masks, encoder | 110 min |
| 8.10 | `P8_10_keras_tensorflow_rosetta_stone` | PyTorch↔Keras mapping, callbacks, channels-last, functional API | 80 min |
| 8.11 | `P8_11_project_lstm_price_forecasting` | **Project 4**: LSTM forecaster vs the Phase 7 leaderboard | 2.5 h |

### Capstone
`capstone/CAPSTONE_selling_advisor`: forecasts → calibrated intervals → store-or-sell policies → bootstrap → board recommendation (~4 h).

> *Project 2 of your roadmap ("image classification") is Lessons 8.5 and 8.6 together; Project 3 ("basic crop disease classification") is Lesson 8.7.*

## 4 · Coverage map: your roadmap → where it is taught

| Roadmap item | Lesson(s) |
|---|---|
| **Phase 7 · 1 Fundamentals**: time series, trend, seasonality, cyclicality, noise, stationarity, autocorrelation, PACF, lags | 7.1 (lags again in 7.2, 7.6) |
| **7 · 2 Data preparation**: datetime indexing, resampling, frequency conversion, missing timestamps, rolling stats, lag & seasonal features | 7.2 |
| **7 · 3 Statistical models**: MA, exponential smoothing, SES, Holt, Holt-Winters, ARIMA, SARIMA, SARIMAX | 7.4, 7.5 |
| **7 · 4 ML forecasting**: linear regression with lags, Random Forest, Gradient Boosting, XGBoost, feature-based | 7.6 |
| **7 · 5 Deep-learning forecasting**: RNN, LSTM, GRU, Transformer | 7.7 (preview), 8.8, 8.9, 8.11 |
| **7 · 6 Evaluation**: MAE, RMSE, MAPE (zero-care), walk-forward, rolling-origin, horizons, baselines; *never shuffle* | 7.3 (and every later lesson) |
| **7 · Libraries**: statsmodels, scikit-learn, XGBoost, pandas, NumPy | throughout |
| **7 · Practical assignment** (8 steps) | 7.8 |
| **Phase 8 · 1 Fundamentals**: neurons, perceptrons, layers, weights/biases, activations | 8.1 |
| **8 · 2 Training**: forward/backprop, losses, gradient descent, optimisers, learning rates, epochs, batch sizes | 8.2, 8.3, 8.4 |
| **8 · 3 Architectures**: ANN, CNN, RNN, LSTM, GRU, Transformers | 8.1–8.2 (ANN), 8.5–8.6 (CNN), 8.8 (RNN/LSTM/GRU), 8.9 (Transformer) |
| **8 · 4 Optimisation**: dropout, batch-norm, early stopping, regularisation, LR scheduling | 8.4 |
| **8 · Frameworks**: PyTorch primary; TensorFlow & Keras familiarity | 8.3 (PyTorch), 8.10 (Keras/TF) |
| **8 · Projects**: digits, image classification, crop disease, LSTM price forecasting | 8.5, 8.6, 8.7, 8.11 |

## 5 · How the exercises work

Every lesson ends sections with **🧩 Exercises**: a task, a **starter cell** with the variable names you must produce, up to four **💡 hints** (open one at a time), and an auto-**check cell**:

```
✅  lag_1 is last week's price
❌  rmean_4 averages the 4 weeks BEFORE t (not including t)
    ↳ hint: y.shift(1).rolling(4).mean()
```
Run the check before finishing the exercise and you'll see *"⏳ Not ready yet"* rather than a traceback. A solved version of every notebook is in `solutions/` (same file names).

## 6 · Using your own data
Phase 7 notebooks only need a weekly `pd.Series` with a `DatetimeIndex` (the `y` variable near the top of each notebook) and optionally an exogenous DataFrame aligned to it.
Replace `ag.price_series(...)` by your own loader: e.g. `y = pd.read_csv("my.csv", parse_dates=["date"], index_col="date")["price"].resample("W").mean()`: and re-run. Remember to re-check the *leak tests* and re-tune the walk-forward `initial` window to your history length.
Phase 8 image notebooks accept real data by swapping the array-loading cell (see Lesson 8.7 §8 for the PlantVillage recipe).

## 7 · Honest notes
* **Data are synthetic**, built so that the truth is known. Real data are harder: bigger shocks, structural breaks, messier labels. Treat scores as teaching tools, not market benchmarks.
* **Results can vary a little** across library versions and hardware (neural-net training especially). Auto-checks use tolerant thresholds; stories (e.g. "Ridge beats the LSTM here") are robust but exact decimals are not.
* Several lessons *deliberately* show models **losing** to simple baselines. That is the point.

## 8 · Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: agrilab` | start Jupyter anywhere inside the course folder; the first cell searches upward for `src/agrilab.py` |
| `'ME' is not a valid frequency` | upgrade pandas to ≥ 2.2 (or replace `"ME"` by `"M"`) |
| PyTorch install is huge | install the CPU build from <https://pytorch.org/get-started/locally/> |
| Keras import fails | `pip install keras` and `export KERAS_BACKEND=torch` (or install tensorflow) |
| Slow notebooks | lower `n_estimators`, `epochs`, or the number of walk-forward origins (`step=16`) |
| Apple Silicon | PyTorch uses `mps` automatically in Lesson 8.3; all other notebooks use CPU |

## 9 · For authors (regenerating everything)
```bash
python src/agrilab.py          # regenerate data/ (deterministic)
python build/run.py            # rebuild + re-execute every notebook (≈15 min on 1 CPU)
python build/run.py p7_05      # rebuild a single notebook by name fragment
```
Each notebook is defined in `build/nb_*.py` with a tiny DSL (`nbkit.py`): `md`, `code`, `exercise(title, task, hints, starter, solution, check)`. The runner executes the *solution* notebook (so every cell is verified), then derives the student notebook by swapping in starter code.

---
*Teach it once, check it twice.* Enjoy the journey.
