# 📖 Glossary (A–Z)

Terms are followed by the lesson where they're first taught.

**Accuracy**: fraction of predictions that are correct. Misleading with imbalanced classes. *(7)*
**Alpha (α)**: (a) regularization strength in Ridge/Lasso *(5)*; (b) occasionally the learning rate in gradient descent *(8)*.
**AUC**: Area Under the ROC curve; the probability that a random positive is scored higher than a random negative. 0.5 = chance, 1 = perfect. *(7)*
**Baseline**: a deliberately simple model (e.g. always predict the mean) that any real model must beat. *(6)*
**Batch gradient descent**: each update uses all training examples. *(8)*
**Bias (statistical)**: systematic error from a model that is too simple to capture the pattern; leads to underfitting. *(5)*
**Bias (intercept)**: the constant term `b` in a linear model. *(2, 8)* Not the same as statistical bias!
**Bias–variance tradeoff**: as model complexity rises, bias falls and variance rises; the best model balances them. *(5)*
**Classification**: predicting a category. *(2)*
**Cluster / clustering**: a group of similar examples / the task of finding such groups without labels. *(3)*
**Confusion matrix**: table counting TP, FP, FN, TN (or the multiclass generalisation). *(7)*
**Convergence**: the point at which further training no longer meaningfully reduces the loss. *(4, 8)*
**Convex function**: a "single bowl" cost with one global minimum (e.g. MSE for linear regression). *(8)*
**Cost function**: average loss over the training set; what gradient descent minimises. *(8)*
**Cross-validation (CV)**: rotating which fold is held out for validation and averaging the *k* scores. *(5)*
**Data leakage**: information from outside the training data (test rows, the target itself, future information) sneaking into model building. *(4)*
**Decision boundary**: the border in feature space where a classifier changes its predicted class. *(2)*
**Decision threshold**: probability above which we predict the positive class (default 0.5). *(7)*
**Dimensionality reduction**: representing data with fewer features while keeping as much information as possible (PCA). *(3)*
**Early stopping**: halting training when validation error stops improving. *(4, 8)*
**Elbow method**: choosing *k* where the inertia curve bends. *(3)*
**Epoch**: one full pass over the training data. *(4)*
**F1-score**: harmonic mean of precision and recall. *(7)*
**Feature**: an input variable (column) used for prediction; collectively `X`. *(1)*
**Feature scaling / standardisation**: rewriting each feature as (x − mean) / std. *(3, 8)*
**Fit**: train a model (`model.fit`). *(1)*
**Generalization**: performing well on unseen data. *(5)*
**Gradient**: the vector of slopes of the cost with respect to each parameter. *(8)*
**Gradient descent**: iteratively moving parameters against the gradient to reduce the cost. *(8)*
**Heteroscedasticity**: error variance that changes with the prediction level (a "fan" in the residual plot). *(6)*
**Hyperparameter**: a setting chosen *before* training (k, α, depth, learning rate). *(1)*
**Inertia**: K-Means' total squared distance from points to their cluster centres. *(3)*
**Inference**: using a trained model to predict on new inputs (`predict`). *(1)*
**K-Means**: clustering by alternating assignment to the nearest centre and re-averaging the centres. *(3)*
**KNN (k-nearest neighbours)**: predicts from the *k* most similar training examples. *(2)*
**L1 / Lasso**: penalty on Σ|w|; can drive coefficients to exactly zero. *(5)*
**L2 / Ridge**: penalty on Σw²; shrinks coefficients smoothly. *(5)*
**Label**: the known correct answer for a training example; the target. *(1, 2)*
**Learning rate**: step size in gradient descent. *(8)*
**Lift**: confidence ÷ baseline probability of the consequent; >1 means a real association. *(3)*
**Logistic regression**: a classifier that passes a linear score through the sigmoid to give a probability. *(2, 8)*
**Loss function**: penalty for the error on a single example. *(8)*
**MAE**: mean absolute error. *(6)* **MAPE**: mean absolute percentage error. *(6)*
**Mini-batch**: an update computed from a small random subset (e.g. 32) of the data. *(8)*
**MSE / RMSE**: mean squared error / its square root. *(6)*
**Model**: a function with adjustable parameters that maps features to predictions. *(1)*
**Non-convex**: a cost with several valleys (local minima). *(8)*
**One-hot encoding**: turning a category into 0/1 columns, one per value. *(10)*
**Overfitting**: fitting noise in the training data: low training error, high validation error. *(5)*
**Parameter**: a number learned from data during training (weights, intercept). *(1)*
**PCA**: Principal Component Analysis; projects onto orthogonal directions of maximal variance. *(3)*
**Pipeline**: chains preprocessing steps and a model into one estimator, preventing leakage. *(4, 10)*
**Precision**: TP / (TP + FP). *(7)*
**Predict / `predict_proba`**: hard class or number / per-class probabilities. *(2)*
**R² (coefficient of determination)**: 1 − SS_res/SS_tot; share of variance explained. *(6)*
**Recall (sensitivity, TPR)**: TP / (TP + FN). *(7)*
**Regression**: predicting a number. *(2)*
**Regularization**: penalising model complexity to reduce overfitting. *(5)*
**Residual**: actual − predicted. *(6)*
**ROC curve**: true-positive rate vs false-positive rate across all thresholds. *(7)*
**Sample**: one row/example. *(1)*
**Sigmoid**: σ(z) = 1/(1+e⁻ᶻ); squashes any number into (0, 1). *(8)*
**Silhouette score**: cluster-quality measure from −1 to 1; higher is better. *(3)*
**Specificity (TNR)**: TN / (TN + FP). *(7)*
**Stochastic gradient descent (SGD)**: updates from one example at a time. *(8)*
**Stratified split**: a split that preserves class proportions in each part. *(4)*
**Supervised learning**: learning from labelled examples. *(2)* **Unsupervised**: learning structure without labels. *(3)*
**Target**: the variable we want to predict; `y`. *(1)*
**Test set**: held-out data used once for the final performance estimate. *(4)*
**Training set**: data used to fit parameters. *(4)*
**Underfitting**: model too simple: high error even on training data. *(5)*
**Validation set**: held-out data used to compare models and tune hyperparameters. *(4)*
**Variance (statistical)**: sensitivity of a model to the particular training sample; leads to overfitting. *(5)*
