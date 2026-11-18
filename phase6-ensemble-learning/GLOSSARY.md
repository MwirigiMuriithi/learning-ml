# 📖 Glossary (plain English)

**Acquisition function**: in Bayesian optimisation, the score that decides which hyperparameters to try next (e.g. *expected improvement*); balances exploitation and exploration.
**AdaBoost**: boosting by *re-weighting examples*: mistakes get heavier, and each weak learner gets a "say" α = ½ ln((1−ε)/ε).
**Additivity (SHAP)**: base value + sum of SHAP values = the model's prediction.
**Bagging**: Bootstrap AGGregatING: train models on bootstrap samples and average them. Reduces variance.
**Base learner / weak learner**: one member of an ensemble (typically a small tree).
**Bayesian optimisation**: hyperparameter search that fits a surrogate model of the objective and uses an acquisition function to choose the next trial.
**Bias**: systematic error from a model that is too simple.
**Blending**: stacking using one hold-out set instead of K-fold out-of-fold predictions.
**Bootstrap sample**: a sample of size *n* drawn *with replacement*; contains ≈ 63.2% of the distinct rows.
**Boosting**: build models sequentially, each correcting the ensemble so far. Reduces bias.
**CatBoost**: boosting library with ordered boosting, ordered target statistics and symmetric trees.
**Condorcet jury theorem**: majority vote of independent voters, each better than chance, approaches certainty as voters are added.
**Cross-validation (CV)**: rotating train/validation roles over K folds for a more reliable estimate.
**Data leakage**: information from outside the training fold influencing fitting (scaling/selection on all data, target-derived features…). Inflates scores.
**Decision stump**: a tree with a single split (depth 1).
**Diversity**: how different the members' *errors* are; the second ingredient of a good ensemble (after accuracy).
**Early stopping**: stop adding boosting rounds when validation error stops improving.
**Ensemble**: a combination of several models into one predictor.
**Expected improvement (EI)**: expected amount by which a candidate beats the best result so far: (f*−μ−ξ)Φ(Z)+σφ(Z).
**Extra-Trees**: random forest variant that also picks split thresholds at random.
**GOSS / EFB**: LightGBM speed-ups: gradient-based one-side sampling / exclusive feature bundling.
**Gradient boosting**: boosting where each tree fits the negative gradient (pseudo-residual) of the loss; gradient descent in function space.
**Grid search**: try every combination from a fixed list of hyperparameter values.
**Hard voting / soft voting**: majority of class votes / average of predicted probabilities.
**Histogram-based splitting**: bin features into ≤ 255 buckets and search only bucket boundaries (XGBoost `hist`, LightGBM, HistGB).
**Hyperparameter**: a setting chosen by you before training (learning rate, depth…), as opposed to *parameters* learned from data.
**ICE / PDP**: individual conditional expectation / partial dependence: how predictions respond to one feature.
**Interventional vs path-dependent SHAP**: two ways to define the "missing feature" expectation; they differ with correlated features.
**K-fold / stratified / group / time-series split**: CV schemes for i.i.d. rows / imbalanced classes / clustered rows / time-ordered rows.
**Leaf-wise growth**: grow the single leaf with the largest loss reduction (LightGBM); versus level-wise growth.
**Learning rate (shrinkage)**: multiplier on each tree's contribution; smaller needs more trees but usually generalises better.
**LightGBM**: fast boosting library: histograms, leaf-wise trees, GOSS, EFB.
**MDI (impurity importance)**: built-in forest importance from training-set impurity reduction; biased toward continuous/high-cardinality features.
**Meta-learner**: the second-level model in stacking.
**Nested CV**: CV loop for evaluation around a CV loop for tuning; unbiased estimate of "tune-then-fit".
**Nadeau–Bengio correction**: variance inflation `(1/k + n_test/n_train)` for t-tests on overlapping CV folds.
**Noise (irreducible error)**: randomness in the target that no model can predict.
**OOF (out-of-fold) predictions**: each row's prediction from a model that never saw it; the right input for a stacking meta-learner.
**One-standard-error rule**: among models within 1 SE of the best, choose the simplest.
**OOB error**: validation estimate from the ~36.8% of rows left out of each bootstrap sample.
**Optuna**: Python library for hyperparameter optimisation (TPE sampler, pruners).
**Overfitting**: good on training data, poor on new data.
**Permutation importance**: increase in error when a feature's values are shuffled.
**Pruning (Optuna) / pruning (trees)**: stopping unpromising trials early / cutting tree splits that don't pay for themselves.
**Pseudo-residual**: −∂L/∂F; what the next boosting tree is trained to predict.
**Random forest**: bagged deep trees with a random feature subset at each split.
**Random search**: sample hyperparameters at random from distributions; usually beats grid search.
**Regularisation**: penalties/limits that reduce variance (λ, γ, min leaf size, depth, shrinkage).
**RMSE / MAE / MAPE / RMSLE / R²**: regression metrics (big-error-sensitive / typical error / % error / log error / variance explained).
**Shapley value**: a feature's average marginal contribution over all orderings of features; the unique "fair" attribution.
**SHAP**: SHapley Additive exPlanations; TreeSHAP computes exact values for tree ensembles in polynomial time.
**Stacking**: learn how to combine base-model predictions with a meta-learner trained on OOF predictions.
**Successive halving**: tournament-style search: many candidates on small budgets, survivors get more.
**Surrogate model**: cheap model (GP, TPE densities) of an expensive objective.
**Target encoding / ordered target statistics**: replacing a category by (a prior-smoothed) mean target; the ordered version avoids leaking a row's own label.
**TPE**: Tree-structured Parzen Estimator: models p(x|good) and p(x|bad); Optuna's default sampler.
**Variance**: sensitivity of a model to the particular training sample.
**XGBoost**: boosting library with second-order optimisation and explicit regularisation: w* = −G/(H+λ).
