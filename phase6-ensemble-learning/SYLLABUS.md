# 📚 Syllabus: Phase 6: Ensemble Learning & Model Optimization
**Duration:** Weeks 11–12 · **Format:** 10 notebooks + 1 capstone · **Effort:** ≈ 12–14 h/week

## Objective
Understand how multiple models can be combined to improve prediction performance, and how to **optimise, compare and explain** them well enough to make evidence-based model selections.

## Learning outcomes
After Phase 6 you can:

1. **Explain** why ensembles work (bias–variance–noise, correlation, accuracy + diversity) and quantify it.
2. **Build** bagging, Random Forest, AdaBoost, gradient boosting, voting and stacking: from scratch *and* with libraries.
3. **Use** XGBoost, LightGBM and CatBoost with early stopping, native missing/categorical handling, and know what makes each different.
4. **Validate** rigorously: K-fold, stratified, group, time-aware; prevent leakage; use nested CV.
5. **Compare** models with paired folds, the corrected resampled t-test, effect size and the one-standard-error rule.
6. **Tune** with grid, random, successive halving and Bayesian optimisation (GP/EI concepts, TPE/Optuna), with a sensible search space and budget.
7. **Interpret** models with impurity vs permutation importance, PDP/ICE and SHAP; communicate individual predictions; state the limits of interpretation.
8. **Deliver** a complete modelling project with a justified recommendation.

## Schedule

### Week 11: Building ensembles
| Day | Notebook | Core topics | Time |
|---|---|---|---|
| Mon | **00** Welcome & setup · **01** Why ensembles work | environment, datasets, metrics refresher · Galton's ox, Condorcet, correlation floor `ρσ² + (1−ρ)σ²/B`, bias–variance–noise | 1.5 h + 2 h |
| Tue | **02** Bagging & Random Forest | bootstrap (63.2%), bagging from scratch, OOB, `max_features` and tree correlation, n_estimators, extrapolation limit, MDI preview | 3 h |
| Wed | **03** Boosting | AdaBoost by hand + code, gradient boosting as gradient descent in function space, learning-rate × trees, depth = interaction order, early stopping, robust losses | 3 h |
| Thu | **04** XGBoost · LightGBM · CatBoost | 2nd-order objective, leaf weight & gain by hand, γ/λ, leaf-wise growth, GOSS/EFB, ordered boosting & target leakage, benchmark, library rosetta stone | 3 h |
| Fri | **05** Voting & Stacking | hard vs soft voting, diversity heat-maps, OOF stacking, leakage demo, non-negative meta-learner, `StackingRegressor` | 2.5 h |
| Weekend | Review | redo ⭐⭐⭐ problems; read CHEATSHEET.md; explain Lessons 01–05 aloud to a rubber duck | 2 h |

### Week 12: Optimising, comparing, explaining, delivering
| Day | Notebook | Core topics | Time |
|---|---|---|---|
| Mon | **06** Cross-validation & model comparison | split lottery, K-fold/stratified/group/time, leakage on pure noise, learning curves, corrected resampled t-test, nested CV, metrics | 3 h |
| Tue | **07** Hyperparameter tuning | grid vs random (and why), halving, Bayesian optimisation & EI by hand, TPE, Optuna + pruning, tuning playbook | 3 h |
| Wed | **08** Interpretability | MDI's lie, permutation importance, PDP/ICE, Shapley by brute force, TreeSHAP, beeswarm/waterfall, recipe recovery, leak hunting | 3 h |
| Thu–Fri | **09** Practical assignment | five models, repeated CV, test once, segment diagnostics, log-target, significance, one-SE selection, written analysis | 4 h |
| Weekend → | **Capstone** (`project/`) | crop-yield forecasting end to end + executive summary | 6–10 h |

## Topic-coverage checklist (from the official Phase 6 outline)
- [x] Bagging; L01, L02
- [x] Boosting; L01, L03
- [x] Voting classifiers; L05
- [x] Stacking; L05
- [x] Random Forest; L02
- [x] Gradient Boosting Machines; L03
- [x] XGBoost; L04
- [x] LightGBM; L04
- [x] CatBoost; L04
- [x] Hyperparameter tuning; L07
- [x] Grid Search; L07
- [x] Random Search; L07
- [x] Bayesian optimisation concepts; L07
- [x] Cross-validation; L06
- [x] Feature importance; L02, L08
- [x] Model comparison; L06, L09
- [x] Permutation importance; L08
- [x] SHAP; L08
- [x] Understanding prediction explanations; L08
- [x] Practical assignment (Linear, Tree, RF, GB, XGBoost); L09

## Assessment
| component | weight | how |
|---|---|---|
| Lesson problems (auto-checked) | 25% | `scoreboard()` at the end of each notebook (≈ 45 problems) |
| Practical Assignment (L09) | 35% | rubric inside the notebook (100 pts) |
| Capstone project | 40% | `project/RUBRIC.md` (100 pts) |

Suggested mastery bar: **≥ 80% of problems solved without peeking at solutions, ≥ 70/100 on the assignment, ≥ 75/100 on the capstone.**

## Learning principles used throughout
* **Predict, then run**: commit to a guess before seeing output.
* **Build it to understand it**: every major algorithm is implemented in a few lines before the library version appears.
* **Interleaving & retrieval**: quizzes revisit earlier lessons; the capstone revisits everything.
* **Honest evaluation**: the test set is touched once; claims need paired evidence.
* **Explain it**: every project ends with a written explanation for a non-technical reader.
