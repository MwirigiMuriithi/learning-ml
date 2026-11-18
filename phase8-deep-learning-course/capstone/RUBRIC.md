# 📊 Capstone Rubric (100 points)

| Category | Pts | Excellent (full marks) | Developing (partial) | Missing (0) |
|:--|:-:|:--|:--|:--|
| **Disease model** | 20 | Meets accuracy & per-class recall targets; architecture/augmentation justified; error analysis by class **and severity**; confusion matrix interpreted | Meets thresholds but little analysis or justification | Not working / below baseline |
| **Price model** | 20 | Chronological split, no leakage; ≥ 2 baselines (naive + ridge/seasonal); beats 0.8 × naive; model choice justified with evidence; seeds/ensembles considered | Beats naive but weak comparison or unclear splits | Random split / leakage / no baseline |
| **Uncertainty & advice engine** | 15 | Triage tiers behave correctly; intervals calibrated on validation with **coverage measured on test**; hold/sell rule transparent and tested; abstains appropriately | Works but untested, or coverage not reported | Overconfident outputs; no uncertainty |
| **Evaluation rigour** | 15 | Test set used once; uncertainty from small samples acknowledged; at least one controlled ablation/comparison; metrics tied to farmer decisions | Metrics reported without context | Tuned on test data |
| **Model card & ethics** | 15 | All five sections; real numbers; ≥ 3 *specific* limitations; ethics answers address harm, exclusion, consent, over-trust, drift | Generic statements | Missing |
| **Code quality & reproducibility** | 10 | Seeds; clean functions; runs top-to-bottom; comments explain *why*; checks pass | Runs with manual fixes | Does not run |
| **Communication** | 5 | Clear 5-minute talk with one results table, one surprising finding, honest limitations, demo card | Rambling or over-long | None |

**Bonus (up to +10):** a stretch track done well (real data, TFLite export, conformal intervals, robustness audit, small offline model, Transformer comparison) *with honest reporting*, including negative results.

**Automatic deductions:** −10 for using the test set to make modelling choices · −5 for claims not supported by the reported numbers.
