# Agricultural Market Price Prediction - Synthetic-Data Forecasting Experiment

A complete, reproducible implementation of the experiment specified in
*"Agricultural Market Price Prediction: Synthetic-Data Forecasting Experiment"*:
a documented synthetic market world (5 markets x 4 crops x 156 weeks), three data-quality scenarios, leakage-safe
feature engineering, a **core eight-model** comparison plus XGBoost/LightGBM, ablations, stress tests,
generalisation diagnostics, prediction intervals and error analysis.

> **Everything here is synthetic.** Results describe how forecasting *methods behave under the assumptions built
> into the generator*. They say nothing about real Kenyan (or any real) prices. See `docs/provenance_statement.md`.

**Start with `reports/FINAL_REPORT.md`** (also as `FINAL_REPORT.html`, self-contained).

## Quick start

```bash
pip install -r requirements.txt
./run_all.sh                      # full pipeline, ~35-45 min on one CPU core
# or step by step:
export PYTHONPATH=$PWD/src
python scripts/01_generate_data.py          # data (seconds)
python -m pytest tests -q                   # 19 tests: leakage, reproducibility, acceptance (seconds)
python scripts/02_audit_and_eda.py          # acceptance checks + EDA
python scripts/03_tune.py                   # rolling-origin tuning
python scripts/04_run_experiments.py        # all model runs   (--part main|ablation|generalisation|intervals)
python scripts/05_analyse.py                # tables, bootstrap tests, figures, hypothesis verdicts
python scripts/06_make_docs.py              # docs generated from live config
python scripts/07_build_reports.py          # final + error-analysis reports
```
All randomness is seeded (`experiment.seed: 42` in `config/experiment_config.yaml`); the shipped `results/` were produced by exactly this pipeline.
Use `python scripts/01_generate_data.py --seed 7` (then re-run the later stages) to check whether conclusions survive a different world.

## Deliverables (specification section 14) and where to find them

| # | Required deliverable | File(s) |
|---|---|---|
| 1 | Generator specification + configuration | `docs/generator_specification.md`, `config/experiment_config.yaml`, `src/agriprice/generator.py`, `degrade.py` |
| 2 | Main long-format dataset | `data/synthetic_panel_{clean,realistic,stressed}.csv` (+ `.parquet`) |
| 3 | Event/shock schedule + data-quality annotations | `data/events_schedule.csv`, `data/data_quality_annotations.csv` |
| 4 | Data dictionary + provenance statement | `docs/data_dictionary.md`, `docs/provenance_statement.md` |
| 5 | EDA report with time-series and distribution plots | `reports/EDA_REPORT.md`, `figures/eda_*.png`; synthetic-data acceptance audit: `reports/DATA_AUDIT.md` |
| 6 | Experiment scripts with fixed seeds | `scripts/01..07_*.py`, `src/agriprice/`, `run_all.sh`, `tests/` |
| 7 | Model comparison by horizon, crop and market | `reports/FINAL_REPORT.md` section 8, `results/core_eight_*.csv`, `results/breakdown_by_{crop,market}_test.csv`, `results/all_variants_test_metrics.csv` |
| 8 | Ablation and stress-test results | `results/ablation_*.csv`, `results/scenario_comparison_*.csv`, `results/generalisation_*.csv`, `results/breakdown_*`, `results/paired_bootstrap_tests_test.csv` |
| 9 | Error-analysis report (successful and failed forecasts) | `reports/ERROR_ANALYSIS.md` |
| 10 | Final report incl. limitations and path to real data | `reports/FINAL_REPORT.md` (+ `.html`) |
| - | Forecast output contract example (Appendix B) | `results/forecast_contract_test.csv` |

## Design in one paragraph

Each week *t* is a **forecast origin**; the target is the observed wholesale price of week *t+h* (h = 1, 2, 4), one *direct* model per horizon.
Partitions are chronological and decided by origin week: train 1-104, validation 105-130, test 131-156 (used once). Training rows are purged so
their *target* lies inside the training window. A strict information-availability contract (`docs/forecast_time_data_contract.md`) governs every
feature (late-delivered indicators, causal cleaning, calendar-only future knowledge) and is **proved by tests** (truncation invariance).
ML models predict the log-change versus the latest available price. Hyper-parameters are tuned on five expanding rolling-origin folds. Differences are tested
with paired block bootstraps over origin weeks, because 26 test weeks is a small sample.

## Layout

```
config/experiment_config.yaml   single source of truth (world, events, scenarios, grids)
src/agriprice/                  generator, degrade (data quality), cleaning (causal), features, splits, metrics,
                                models, experiment (orchestration), validate (acceptance checks), analysis
scripts/                        numbered pipeline stages
tests/                          pytest suite (no-leakage, purging, cleaning, reproducibility, acceptance)
data/ results/ figures/ reports/ docs/   all generated outputs
```

## Scope and honest deviations from the specification

* **Implemented:** the full core eight (naive, seasonal naive, moving average, Ridge, SARIMA, Decision Tree, Random Forest, HistGradientBoosting),
  plus *ARIMA (non-seasonal)* and *Ridge + crop interactions* as diagnostics, and the advanced **XGBoost** and **LightGBM** (Stage 5).
* **Not implemented (deliberately deferred):** Prophet, LSTM and Temporal Fusion Transformer (Stages 4 and 6). The specification marks them optional
  "only after the core experiment has been evaluated"; with 3,120 base observations the neural models are explicitly not expected to win, and
  Prophet needs per-origin refits that do not fit the single-core budget used here. Nothing in the pipeline prevents adding them. The "all 13 models" comparison is therefore a comparison of 10 families/variants.
* **SARIMA** is a native multi-step (iterated) forecaster fitted once per series per partition (orders picked by AIC), then rolled forward with the Kalman filter - the one exception to the direct-forecast rule. It is plain SARIMA (no exogenous regressors).
* **Partial-adjustment form** of the generator's price equation (equivalent reparameterisation; see `docs/generator_specification.md`).
* **No data revisions** are simulated; the `corrected` quality flag exists only in pipeline-cleaned data.
* Results come from **one world (seed 42)**. Conclusions about small differences should be confirmed across seeds.
