# Capstone Rubric (100 points)

| Criterion | Excellent (full marks) | Adequate | Weak (little/no credit) | Pts |
|---|---|---|---|---|
| **1. Data audit & cleaning** | Finds all problems (duplicates, missing weeks, typos, outage, leakage column); fixes are justified and *causal*; checks that cleaning worked | Most problems found; fixes reasonable but not all justified | Obvious problems missed; uses future-looking fills without noticing | 10 |
| **2. Exploration & discovery** | Quantified evidence for trend/seasonality, lead–lag lags, a non-linear effect, and structural change; implications for modelling stated | Some discoveries with plots; limited interpretation | Generic plots, no insight used later | 15 |
| **3. Evaluation design & leakage control** | Chronological protocol, dev/test separation, harness used correctly, every driver lagged ≥ h, leakage checklist addressed explicitly | Mostly sound with minor lapses acknowledged | Shuffled/random splits, hold-out used for tuning, or unacknowledged leakage | 20 |
| **4. Modelling breadth & quality** | ≥ 5 sensible models incl. baselines, ARIMA-family, ML on returns, and an original idea; clean, reproducible code | Requirements met with limited care | Fewer models, no baselines, targets/features inappropriate (e.g. trees on trending levels) | 15 |
| **5. Comparison & selection** | Scoreboards by horizon & commodity; skill vs naive; per-horizon selection from dev only; awareness of sampling noise | Scoreboards present, selection partly justified | Single overall number, no baseline comparison | 10 |
| **6. Uncertainty** | Intervals with coverage check and honest discussion of miscalibration | Intervals given, coverage not checked | None | 5 |
| **7. Failure analysis & interpretation** | Locates large errors, explains plausible causes (shock, regime, horizon limits), connects to discovered structure | Some error discussion | Only reports the best score | 10 |
| **8. Communication** | Clear model card, correct claims matching the evidence, concrete recommendation per crop & horizon, limitations stated | Understandable, some overclaiming | Confusing or conclusions unsupported by results | 10 |
| **9. Reproducibility & code quality** | Runs top-to-bottom, seeds fixed, readable, commented, one-shot test honest | Minor issues | Doesn't run / unclear | 5 |

### Deductions & bonuses
- **−10 to −30:** using `arrivals_tonnes` (or any same-week information) as a predictor **without** acknowledging that it is not available at forecast time.
- **−10:** changing models after seeing hold-out results *without disclosure*.
- **+1 to +5 (cap 100):** a well-executed stretch goal (global/regime-aware model, conformal intervals, real-data replication, decision memo).

### Grade bands
90–100 outstanding · 80–89 strong · 70–79 solid · 60–69 needs work · < 60 revisit the lessons on evaluation and leakage.

> A modest model with an honest, leak-free evaluation outscores a spectacular result with a hidden leak. Always.
