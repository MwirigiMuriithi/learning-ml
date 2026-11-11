# 🧾 Machine Learning Fundamentals: Cheat Sheet

## 1. Vocabulary in one breath
**Sample** = row · **Feature** (`X`, matrix) = input column · **Target** (`y`, vector) = what we predict · **Training** = `fit` · **Inference** = `predict` · **Parameters** are *learned* (`coef_`, `intercept_`) · **Hyperparameters** are *chosen* (`alpha`, `n_neighbors`, `max_depth`, learning rate).

## 2. Which kind of problem?
| Labels? | Answer is… | Name | Typical models |
|---|---|---|---|
| yes | number | **Regression** | LinearRegression, KNN, trees |
| yes | category | **Classification** | LogisticRegression, KNN, trees |
| no | groups | **Clustering** | K-Means, DBSCAN |
| no | fewer columns | **Dimensionality reduction** | PCA |
| reward signal | actions | Reinforcement | (not in this phase) |

## 3. The scikit-learn pattern
```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X = df.drop(columns="target");  y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # stratify=y for classification

model = make_pipeline(StandardScaler(), LinearRegression())   # preprocessing + model = one object, no leakage
model.fit(X_train, y_train)                                   # TRAIN  (training data only!)
pred = model.predict(X_test)                                  # INFER  (unseen data)
```
**Three-way split:** `X_rest, X_test, y_rest, y_test = split(0.20)` then `X_train, X_val, ... = split(X_rest, test_size=0.25)` → 60/20/20.

## 4. Rules that prevent disasters
1. **Train** to learn · **validation / CV** to choose · **test** to report: **once**.
2. Split **before** scaling, imputing or selecting features; fit those steps on **train only** (`Pipeline`).
3. For every feature ask: *"will I know this when I need the prediction?"*
4. Always report a **baseline** (`DummyRegressor` / `DummyClassifier`).
5. Stratify classification splits. Set `random_state`.
6. Scale for distance- or gradient-based methods (KNN, K-Means, PCA, Ridge/Lasso, gradient descent).

## 5. Generalization
| | Train error | Val error | Cure |
|---|---|---|---|
| **Underfit** (high bias) | high | high (≈ train) | more flexible model, better features, weaker regularization |
| **Overfit** (high variance) | very low | much higher | more data, simpler model, stronger regularization |
| **Good fit** | low | low (≈ train) | - |

**Expected error = Bias² + Variance + Noise σ².** Complexity ↑ ⇒ bias ↓, variance ↑.
**Ridge (L2):** loss + α·Σw² (shrinks) · **Lasso (L1):** loss + α·Σ|w| (shrinks *and* zeroes) · larger α ⇒ simpler.
```python
cross_val_score(model, X_train, y_train, cv=5, scoring="r2")            # → k scores; report mean ± std
GridSearchCV(pipe, {"ridge__alpha": [0.01, 0.1, 1, 10]}, cv=5).fit(X_train, y_train)
```
Time series? `TimeSeriesSplit`. Classification? `StratifiedKFold`.

## 6. Regression metrics   (residual **e = y − ŷ**)
| Metric | Formula | Notes |
|---|---|---|
| **MAE** | mean(\|e\|) | same units as y; robust |
| **MSE** | mean(e²) | squared units; the usual training loss |
| **RMSE** | √MSE | same units; punishes big errors; RMSE ≥ MAE |
| **R²** | 1 − Σe² / Σ(y−ȳ)² | 1 perfect · 0 = predicting the mean · can be negative |
| **MAPE** | mean(\|e/y\|) | scale-free; unstable if y ≈ 0 |

`neg_mean_absolute_error`, `neg_root_mean_squared_error`, `r2`: sklearn *negates* errors in `scoring=`. **Plot residuals** (vs predicted): curves/fans reveal what's missing.

## 7. Classification metrics
```
                    Predicted +      Predicted −
  Actually +        TP               FN  (miss)
  Actually −        FP (false alarm) TN        sklearn: confusion_matrix → [[TN, FP], [FN, TP]]
```
| Metric | Formula | Answers |
|---|---|---|
| Accuracy | (TP+TN)/all | overall correct? (**lies** under imbalance) |
| **Precision** | TP/(TP+FP) | when I say "positive", how often am I right? |
| **Recall** (TPR) | TP/(TP+FN) | of the real positives, how many did I find? |
| Specificity | TN/(TN+FP) | of the real negatives, how many cleared? |
| **F1** | 2PR/(P+R) | harmonic mean (low if either is low) |
| FPR | FP/(FP+TN) | x-axis of ROC |

* **Threshold** (default 0.5) is a dial: ↓ threshold ⇒ recall ↑, precision ↓. Best threshold ≈ cost_FP / (cost_FP + cost_FN) for calibrated probabilities. Choose it on validation/CV data.
* **ROC curve**: TPR vs FPR over all thresholds. **AUC** = P(random positive scored above random negative); 0.5 = chance, 1 = perfect. For rare positives prefer **precision–recall / average precision**.
* Multiclass: per-class metrics + **macro** (equal weight) / **weighted** / micro averages.
* **Choose by cost of errors:** disease/fraud → recall · spam/legal → precision · balanced → accuracy/F1 · ranking → AUC.

## 8. Optimization
| Term | Meaning |
|---|---|
| **Loss** | penalty on one example (squared, absolute, Huber, log loss) |
| **Cost** `J` | average loss over the training set |
| **Gradient** | slope of `J` w.r.t. each parameter |
| **Learning rate α** | step size |

**Gradient descent:** `θ ← θ − α · ∇J(θ)`
Linear regression (ŷ = wx + b, MSE): `∂J/∂w = (2/n) Σ (ŷ−y)·x`  ·  `∂J/∂b = (2/n) Σ (ŷ−y)`
Logistic regression: `p = σ(Xw+b)`, `∂J/∂w = Xᵀ(p−y)/n`, `∂J/∂b = mean(p−y)`, log loss `−mean[y·log p + (1−y)·log(1−p)]`

```python
for _ in range(n_iter):
    err = w*x + b - y                       # ŷ − y
    w, b = w - lr*2*np.mean(err*x), b - lr*2*np.mean(err)   # simultaneous update
```
* **α too small** → crawl · **too large** → cost explodes (diverges) · try 0.001, 0.01, 0.1, 1 and plot the cost.
* **Convergence:** max iterations · |ΔJ| < tol · small gradient · validation stops improving (early stopping).
* **Standardise features** ⇒ round bowl ⇒ far fewer steps. **Batch** (all rows) · **SGD** (1 row, noisy) · **Mini-batch** (16–256, standard).
* Convex (linear/logistic regression): one minimum. Non-convex (neural nets): start point matters.

## 9. Unsupervised
* **K-Means:** init → assign to nearest centre → recompute centres → repeat. Scale first. Choose *k* by **elbow** (inertia) and **silhouette** (−1…1, higher better); **profile** clusters with `groupby().mean()`. Use `n_init=10`.
* **DBSCAN:** density-based; handles odd shapes and noise; needs `eps`, not *k*.
* **PCA:** rotate to axes of max variance (eigenvectors of the covariance matrix); standardise first; keep components for ~90–95% of `explained_variance_ratio_.cumsum()`.
* **Association rules:** support = P(A∧B) · confidence = P(B|A) · **lift** = confidence / P(B) (>1 ⇒ real boost).

## 10. Messy data in a pipeline
```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
prep = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat_cols)])
full = make_pipeline(prep, Ridge(alpha=1.0))
```
Skewed/multiplicative target (prices)? `TransformedTargetRegressor(regressor=full, func=np.log, inverse_func=np.exp)`.

## 11. Debug checklist when results look wrong
* Suspiciously perfect? → leakage (feature derived from target, test data in preprocessing, duplicates across splits).
* Train ≈ test but both bad? → underfitting: richer model/features.
* Train ≪ test? → overfitting: regularize, simplify, more data.
* Accuracy high, recall zero? → class imbalance: look at the confusion matrix, tune threshold.
* Cost rising / `nan`? → learning rate too large; scale features.
* Results change every run? → set `random_state` everywhere.
