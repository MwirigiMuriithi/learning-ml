# 🧾 Phase 6 Cheat Sheet: Ensembles & Model Optimization

## 1 · The big formulas
| idea | formula |
|---|---|
| Variance of an average of B models (variance σ², correlation ρ) | `Var = ρσ² + (1−ρ)σ²/B`  → floor `ρσ²` |
| Error decomposition | `E[(y−f̂)²] = Bias² + Variance + σ²_noise` |
| Bootstrap | P(row not drawn) = (1−1/n)ⁿ → 1/e ≈ 0.368 ⇒ ≈ 63.2% in-bag |
| Majority vote (n independent voters, accuracy p) | `Σ_{k>n/2} C(n,k) pᵏ(1−p)ⁿ⁻ᵏ` |
| AdaBoost | `ε = Σ wᵢ·1[mistake]`, `α = ½ ln((1−ε)/ε)`, `wᵢ ← wᵢ·e^{−α yᵢ h(xᵢ)}` (then normalise) |
| Gradient boosting | `F_m = F_{m−1} + ν·h_m`, `h_m` fits `−∂L/∂F` (squared loss → residual) |
| XGBoost leaf / gain | `w* = −G/(H+λ)`, `Gain = ½[G_L²/(H_L+λ)+G_R²/(H_R+λ)−(G_L+G_R)²/(H_L+H_R+λ)]−γ` (library reports 2× this and compares with γ) |
| Expected improvement (min.) | `EI = (f*−μ−ξ)Φ(Z)+σφ(Z)`, `Z=(f*−μ−ξ)/σ` |
| Corrected resampled t-test | `t = d̄ / √((1/k + n_test/n_train)·s_d²)`, df = k−1 |
| Shapley value | `φ_j = avg over orderings [v(before∪{j}) − v(before)]`; `base + Σφ = f(x)` |

## 2 · Which method attacks what?
| | reduces | trains | overfits with more rounds? |
|---|---|---|---|
| Bagging / Random Forest | variance | parallel | no |
| Boosting (GBM, XGB, LGBM, CatBoost) | bias (+ some variance) | sequential | **yes → early stopping** |
| Voting | variance (needs diverse, comparably good models) | parallel | no |
| Stacking | combines strengths (needs **OOF**) | two-level | meta-learner can |

## 3 · Hyperparameter rosetta stone
| concept | XGBoost | LightGBM | CatBoost | sklearn GBR / HistGB |
|---|---|---|---|---|
| rounds | `n_estimators` | `n_estimators` | `iterations` | `n_estimators` / `max_iter` |
| learning rate | `learning_rate` | `learning_rate` | `learning_rate` | `learning_rate` |
| complexity | `max_depth` | `num_leaves` (+`max_depth`) | `depth` | `max_depth` / `max_leaf_nodes` |
| min leaf size | `min_child_weight` | `min_child_samples` | `min_data_in_leaf` | `min_samples_leaf` |
| row sampling | `subsample` | `subsample`+`subsample_freq` | `subsample` | `subsample` (GBR) |
| col sampling | `colsample_bytree` | `colsample_bytree` | `rsm` | `max_features` |
| L2 | `reg_lambda` | `reg_lambda` | `l2_leaf_reg` | `l2_regularization` (HistGB) |
| min gain | `gamma` | `min_split_gain` | - |: |
| categoricals | `enable_categorical=True` | `category` dtype | `cat_features=[…]` | `categorical_features="from_dtype"` (HistGB) |
| early stopping | `early_stopping_rounds` | `lgb.early_stopping()` | `early_stopping_rounds` | `n_iter_no_change` |

**Typical ranges:** lr 0.01–0.1 · depth 3–8 · num_leaves 8–64 · min leaf 5–50 · subsample/colsample 0.5–1 · λ 1e-3–30 (log).

## 4 · Recipes
```python
# Pipeline (preprocessing INSIDE → no leakage)
pipe = make_pipeline(make_preprocessor(X_train, scale=True), model)

# Repeated, paired CV
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
res = cross_validate(pipe, X, y, cv=rkf, scoring="neg_root_mean_squared_error")

# Out-of-fold predictions for stacking
oof = cross_val_predict(model, X_train, y_train, cv=KFold(5, shuffle=True, random_state=SEED))
meta = LinearRegression(positive=True).fit(oof_matrix, y_train)

# XGBoost with early stopping
m = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.05, early_stopping_rounds=50, enable_categorical=True)
m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

# Optuna
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=50)

# SHAP
sv = shap.TreeExplainer(model)(X_test)
shap.plots.beeswarm(sv); shap.plots.waterfall(sv[0])
```

## 5 · Decision guide
* Need a strong, quick baseline → **Random Forest** (defaults) → then **LightGBM/XGBoost** with early stopping.
* Many categoricals / little tuning time → **CatBoost**. Big data → **LightGBM**.
* Weak/strong models of *different families* available → **average** them, then try **stacking** (OOF!).
* Heavy-tailed positive target → model **log(y)** (and report back-transformed errors).
* Imbalanced classification → stratified CV, **AUC/PR-AUC**, not accuracy.
* Small data + big search → **nested CV**; big data → train/val/test.

## 6 · Pitfall checklist
- [ ] Evaluated on the training set? (OOB / CV / test instead)
- [ ] Preprocessing, selection, encoding **inside** the pipeline?
- [ ] Meta-learner trained on **OOF** predictions?
- [ ] Splitter matches deployment (group / time)?
- [ ] Test set touched **once**? Tuning CV score reported as final? (optimistic!)
- [ ] Differences tested with **paired folds + corrected test**, not eyeballed from one split?
- [ ] Built-in importance trusted? (use permutation / SHAP on held-out data)
- [ ] One feature suspiciously dominant? (leak?)
- [ ] Explanation claims causation? (it doesn't)

## 7 · Metrics quick guide
RMSE (punishes big misses) · MAE (typical miss) · MAPE (% error; bad near 0) · RMSLE (log/relative) · R² (vs mean baseline) · ROC-AUC (ranking) · PR-AUC (rare positives) · log-loss (probability quality).
