# 📖 Glossary (plain English)

**Accuracy**: share of predictions that are right. Misleading when classes are imbalanced.
**Accuracy paradox**: a model that always predicts the majority class can score high accuracy while being useless.
**Adjusted R²**: R² penalised for the number of features.
**AUC (ROC-AUC)**: probability a random positive is scored above a random negative; 0.5 = chance, 1 = perfect.
**Bagging**: training many models on bootstrap samples and averaging them (reduces variance).
**Baseline**: the simplest sensible model (e.g. predict the mean); every real model must beat it.
**Bayes' theorem**: P(y|x) = P(x|y)P(y)/P(x): updates a prior belief with evidence.
**Bias**: error from a model that is too simple (underfitting).
**Bias–variance trade-off**: flexible models have low bias/high variance; simple models the opposite; test error is minimised in between.
**Bootstrap**: resampling with replacement to estimate uncertainty (or to build bagged ensembles).
**Calibration**: whether predicted probabilities match observed frequencies.
**Categorical feature**: takes values from a fixed set (neighbourhood, gender); needs encoding.
**Class imbalance**: one class much rarer than the other.
**Class weight**: extra cost for mistakes on a class.
**Classification**: predicting a category.
**Coefficient**: weight of a feature in a linear model.
**Condition number**: how numerically fragile a matrix is; huge for raw polynomial features.
**Confusion matrix**: table of TP, FP, FN, TN.
**Cross-validation (CV)**: rotating train/validation folds to estimate performance and its spread.
**Curse of dimensionality**: in high dimensions distances lose meaning and data becomes sparse.
**Data leakage**: information unavailable at prediction time (or derived from the target) leaking into training.
**Decision boundary**: where the predicted class changes.
**Demographic parity**: equal rate of positive predictions across groups.
**Dummy (one-hot) variable**: 0/1 column per category; drop one to avoid the dummy trap.
**Elastic net**: mix of L1 and L2 penalties.
**Entropy**: average surprise/uncertainty of a node's class distribution (bits).
**Equal opportunity**: equal true-positive rate across groups.
**Estimator / Transformer**: scikit-learn objects: `fit/predict` vs `fit/transform`.
**Extrapolation**: predicting outside the range of the training data; dangerous.
**F1 score**: harmonic mean of precision and recall.
**False positive / negative**: wrong alarm / missed case.
**Feature**: an input variable.
**Feature engineering**: creating better inputs (ratios, transforms, lags).
**Feature importance**: how much a model relies on a feature (impurity-based or permutation).
**Gini impurity**: 1 − Σp²: probability two random items from the node differ in class.
**Gradient descent**: iteratively stepping opposite to the gradient to minimise a loss.
**Heteroscedasticity**: error variance that changes with the level of the target/feature.
**Hinge loss**: max(0, 1 − y·f(x)); the SVM's loss.
**Hyper-parameter**: a setting chosen before training (k, C, depth).
**Information gain**: reduction in entropy from a split.
**Intercept**: prediction when all features are zero.
**Kernel (trick)**: a similarity function K(x,x′) enabling non-linear models without explicit feature maps.
**KNN**: k-nearest neighbours.
**L1 / L2 regularisation**: penalties on |w| (sparsity) / w² (shrinkage).
**Lag feature**: value of a variable at an earlier time step.
**Laplace smoothing**: adding α to counts so no probability is exactly zero.
**Learning curve**: score vs training-set size; diagnoses variance vs bias.
**Learning rate**: step size in gradient descent.
**Linearity gap**: flexible-model score minus linear-model score; measures non-linearity.
**Log loss (cross-entropy)**: penalises confident wrong probabilities.
**Log-odds (logit)**: ln(p/(1−p)); linear in logistic regression.
**MAE / MSE / RMSE / MAPE**: regression error metrics.
**Margin**: width of the street between classes in an SVM (2/‖w‖).
**Maximum likelihood**: choosing parameters that make the observed data most probable.
**Model card**: a short document stating a model's purpose, data, performance, limits and risks.
**Multicollinearity**: strongly correlated features; destabilises coefficients.
**Naive (assumption)**: features independent given the class.
**No Free Lunch**: no algorithm is best on all problems.
**Normal equation**: β = (XᵀX)⁻¹Xᵀy.
**Odds ratio**: e^coefficient in logistic regression; multiplicative change in odds per unit.
**OOB (out-of-bag)**: samples not in a tree's bootstrap; give a free validation score.
**Ordinary least squares (OLS)**: minimising the sum of squared residuals.
**Overfitting**: fitting noise; great train score, poor test score.
**Partial dependence**: average prediction as one feature varies.
**Permutation importance**: drop in score when a feature is shuffled on held-out data.
**Persistence forecast**: "next period = this period"; the baseline for time series.
**Pipeline**: chains preprocessing and a model so everything is fitted inside CV folds.
**Platt scaling**: logistic calibration of SVM scores into probabilities.
**Polynomial features**: x², x³, interaction terms.
**Precision / Recall**: see *F1*.
**Prior**: probability of a class before seeing the evidence.
**Pruning**: simplifying a tree (cost-complexity, `ccp_alpha`).
**R²**: fraction of variance explained.
**Random forest**: bagged, feature-subsampled decision trees.
**Regression**: predicting a number.
**Regularisation**: penalising complexity to reduce variance.
**Residual**: y − ŷ.
**ROC curve**: TPR vs FPR across thresholds.
**Rolling-origin backtest**: repeated train-on-past/predict-next evaluation for time series.
**Scaling / standardisation**: rescaling features to comparable ranges (z-scores).
**Sigmoid**: 1/(1+e^−z); maps scores to probabilities.
**Skill score**: 1 − error_model / error_baseline.
**Softmax**: multi-class generalisation of the sigmoid.
**Stratified split**: preserves class proportions in each fold.
**Support vector**: a training point on/inside the margin that defines the SVM.
**Target**: the variable we predict.
**Threshold**: probability cut-off turning scores into class labels.
**Underfitting**: model too simple; poor on train and test.
**Validation curve**: score vs a hyper-parameter value.
**Variance (model)**: sensitivity of a model to the particular training sample.
**VIF**: variance inflation factor; 1/(1−R²ⱼ), a collinearity diagnostic.
