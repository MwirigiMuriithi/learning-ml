# Forecast-Time Data Contract

A **forecast origin** is the end of week *t*. The forecast target is the observed wholesale price of week *t+h* (h = 1, 2, 4).

| Item | Available at origin *t*? | Rule in code |
|---|---|---|
| price of week *w* <= *t* | yes | on time; cleaned **causally** (past-only outlier filter, last-observation-carried-forward with age cap, staleness feature) |
| indicator of week *w* <= *t* with `report_delay_weeks = 0` | yes | `asof_lag` |
| indicator of week *w* with `report_delay_weeks = 1` | only from week *w+1* | `asof_lag` masks lag-0 and, in handled mode, falls back to the last delivered value |
| weather, fuel (on-time sources) | yes | no delay |
| calendar facts of the **target** week (week of year, month, holiday calendar) | yes | date arithmetic + configured holiday calendar only |
| demand-surge events, shocks, event labels of future weeks | **no** | not used as features anywhere |
| `latent_price_kes_kg`, `data_quality_flag`, `event_*` columns | **never** | audit/evaluation only; a unit test fails if they appear in a feature set |
| prices / indicators of weeks > *t* | **no** | enforced by tests (truncation invariance) |
| data revisions after *t* | not simulated | no hindsight corrections are possible by construction |

**Chronological partitions** are decided by the origin week. Training rows are *purged*: a row is used for training only if its **target** week is within the training window
(an h-week embargo), so a model fitted for a partition has seen no information dated after that window. Validation: origins 105-130; test: origins 131-156-h. The test period
is used once, after all choices (features, hyper-parameters, model selection) were fixed on training / rolling-origin folds / validation.

**Preprocessing statistics** (imputation medians, scaling, one-hot categories, volatility thresholds) are fitted on the training window of each fit only (inside sklearn Pipelines);
volatility regimes use thresholds computed from origins <= the end of the fitting window.

**Proofs in the test-suite** (`tests/test_no_leakage.py`): (1) removing all data after week T0 leaves every feature of origins <= T0 unchanged (handled and raw pipelines); (2) corrupting every
observation after T0 leaves earlier features unchanged; (3) a late-delivered record is invisible at its own origin and visible one week later; (4) training rows never reach beyond the training window.
