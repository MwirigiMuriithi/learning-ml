# 📄 Supervised Learning Cheat-Sheet

## 1. The workflow
Frame (metric + costs) → Explore → Prepare (in a **Pipeline**) → Split (lock the test set) → Train → Evaluate (CV) → Deliver (test once, error analysis, model card).

## 2. Regression metrics
| Metric | Formula | Use |
|---|---|---|
| MAE | mean \|y−ŷ\| | typical error in y's units; robust |
| RMSE | √mean (y−ŷ)² | punishes big misses |
| R² | 1 − SSE/SST | variance explained (can be negative) |
| MAPE / MdAPE | mean / median \|y−ŷ\|/y | % error stakeholders understand |

## 3. Classification metrics
| Metric | Meaning |
|---|---|
| Precision = TP/(TP+FP) | when I say positive, how often right |
| Recall = TP/(TP+FN) | of the real positives, how many I caught |
| F1 = 2PR/(P+R) | harmonic mean |
| ROC-AUC | P(random positive outscores random negative) |
| PR-AUC (avg. precision) | better when positives are rare |
| Log loss | punishes confident mistakes; judges probabilities |
| Threshold for calibrated p | **c_FP / (c_FP + c_FN)** |

## 4. Algorithms at a glance
| Algorithm | Model | Key dials | Scale? | Complexity (train) | Strength | Weakness |
|---|---|---|---|---|---|---|
| Simple LR | ŷ=β₀+β₁x | – | – | O(n) | interpretable | one feature, outliers |
| Multiple LR | ŷ=Xβ; β=(XᵀX)⁻¹Xᵀy | Ridge/Lasso α | helps | O(np²+p³) | interpretable, fast | linear only, collinearity |
| Polynomial | linear on x, x², … | degree, α | **yes** | O(nT²+T³) | smooth curves | extrapolates badly, T=C(p+d,d) |
| Decision tree | recursive splits | max_depth, min_samples_leaf, ccp_alpha | no | O(p n log n·depth) | rules, interactions | unstable, staircase |
| Random forest | avg of decorrelated trees | n_estimators, max_features, leaf | no | O(B·m·n log n) | strong default | less interpretable, no extrapolation |
| SVR / SVM | tube / max margin; kernels | C, γ, ε, kernel | **yes** | O(n²–n³) | small/medium, smooth | slow, tuning, opaque |
| KNN | vote/average of k nearest | k, weights, metric | **yes** | O(1) fit, O(np) predict | simple, local | slow predict, curse of dim. |
| Logistic reg. | σ(w·x+b) | C, penalty | helps | O(np)/iter | calibrated, explainable | linear boundary |
| Naive Bayes | argmax P(y)ΠP(xⱼ\|y) | α, var_smoothing | no | O(np) | tiny data, text, fast | over-confident |

## 5. Choosing (30-second version)
- **Wide (p ≳ n), linear** → Lasso/Ridge, linear SVM, NB.
- **Linear gap ≈ 0** → linear/logistic; **gap ≫ 0** → forest, boosting, SVM-RBF.
- **Rare positives** → PR-AUC, class weights, threshold from costs.
- **Need explanations/regulation** → linear/logistic, shallow tree, reason codes.
- **Tiny data / text** → Naive Bayes, regularised linear.
- **Small n, smooth non-linear** → SVM/SVR; **local structure, few features** → KNN.
- **Time-ordered data** → time-based split, lags, `TimeSeriesSplit`, beat persistence.

## 6. Bias–variance dials
| Algorithm | ↑ complexity | ↓ complexity |
|---|---|---|
| Polynomial | degree ↑ | degree ↓, Ridge |
| Tree | depth ↑, leaf ↓ | depth ↓, leaf ↑, ccp_alpha ↑ |
| Forest | max_features ↑, leaf ↓ (more trees never overfit) | max_features ↓, leaf ↑ |
| KNN | k ↓ | k ↑ |
| SVM/SVR | C ↑, γ ↑ | C ↓, γ ↓ |
| Logistic/Ridge | C ↑ / α ↓ | C ↓ / α ↑ |

## 7. Classic mistakes
1. Preprocessing before splitting (leakage) → use a `Pipeline`.
2. Tuning on the test set. 3. Accuracy on imbalanced data. 4. Unscaled features for KNN/SVM/Ridge.
5. Features computed from the target (or recorded after the outcome). 6. Shuffled CV on time series.
7. Treating a coefficient as a causal effect. 8. Reporting no uncertainty (fairness gaps, small groups).
