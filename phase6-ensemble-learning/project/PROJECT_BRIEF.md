# 🌾 Capstone Project Brief: Crop-Yield Forecasting

## Client
A farm-advisory service that visits thousands of smallholder fields each season. Advisers have limited time, seed and fertiliser; they want a **field-level yield forecast** to decide where to go first and what to recommend.

## Your task
Build, validate, tune, ensemble, interpret and **defend** a model that predicts `yield_t_ha` from `data/crop_yield.csv` (2,500 field-season records, 5 crops, 6 regions, 2015–2024). Deliver the completed notebook `10_capstone_project_crop_yield.ipynb` and a ≈ 500-word executive summary for the programme director.

## Constraints & requirements
| | |
|---|---|
| **Deployment** | Predict *next* season, so the **test set is the last two seasons (2023–24)**; all model choices use CV on earlier seasons |
| **Metric** | Yields differ ×10 between crops; justify a **relative** metric (the starter models log-yield, error ≈ % off) |
| **Models** | ≥ 5 families + a baseline, on identical CV folds |
| **Tuning** | ≥ 30 Optuna trials on the most promising family, sensible search space |
| **Ensemble** | out-of-fold blending/stacking ≥ 3 models with a **pre-declared decision rule** |
| **Honesty** | one test-set evaluation; per-crop errors in t/ha **and** %; uncertainty reported |
| **Interpretation** | SHAP/permutation importance; check against agronomy (water is hump-shaped; fertiliser has diminishing returns); no causal claims |

## Milestones
1. **M1 Frame & audit**: roles of columns, missingness, leakage candidates
2. **M2 Metric & validation**: time-forward split + CV design, justification
3. **M3 Benchmark**: fair comparison table + chart
4. **M4 Tune**: Optuna study, gain measured on CV
5. **M5 Ensemble**: OOF blend, weights, decision rule
6. **M6 Final test**: evaluate once; per-crop table
7. **M7 Interpret**: SHAP; verify against domain knowledge
8. **M8 Report**: executive summary

## Stretch tracks (bonus)
* CatBoost with native categoricals (+ the `clone` gotcha)
* Segment robustness (region, seed variety)
* Quantile prediction intervals with LightGBM
* **Classification twin:** the same workflow on `data/loan_default.csv`
* Bootstrap confidence interval for the final test RMSE

## Suggested time plan (≈ 8 h)
Frame & audit 1 h · validation 0.5 h · benchmark 1 h · tuning 1.5 h · ensemble 1 h · final + interpretation 1.5 h · summary 1.5 h.

## Submission checklist
See the bottom of the notebook; grading criteria in `RUBRIC.md`.
