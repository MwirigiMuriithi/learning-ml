# 🗓️ Study Plan

Two schedules: the **2-week intensive** (matches Phase 4's Weeks 8–9) and a **4-week relaxed** version.
Time estimates are for *doing* the work (typing, experimenting, attempting problems), not just reading.

## ⚡ Option A: 2-week intensive (≈ 26 h/week)

### Week 8: Regression
| Day | Do | Hours | Checkpoint: you can… |
|---|---|---|---|
| **Mon** | 00 Foundations · 01 Simple LR | 3 | compute MAE/RMSE/R² by hand; derive slope = Cov/Var; explain overfitting vs underfitting |
| **Tue** | 02 Multiple LR | 2.5 | interpret a coefficient "holding others constant"; spot multicollinearity with VIF; build a leakage-free `Pipeline` |
| **Wed** | 03 Polynomial · 04 Decision Tree | 4 | choose a polynomial degree with a validation curve; read a tree; explain why trees can't extrapolate |
| **Thu** | 05 Random Forest | 2.5 | use the variance formula ρσ²+(1−ρ)σ²/B; compute OOB score; use permutation importance |
| **Fri** | 06 SVR · 07 KNN | 4 | rebuild an SVR prediction from support vectors; explain why KNN needs scaling and hates noisy features |
| **Sat** | 08 Regression Showdown | 2.5 | run a *fair* 7-model tournament and justify a winner |
| **Sun** | **Project 1** (House prices): first half (M0–M5) | 4 | pass the leakage and pipeline milestones |

### Week 9: Classification & choosing
| Day | Do | Hours | Checkpoint: you can… |
|---|---|---|---|
| **Mon** | 09 Metrics · 10 Logistic Regression | 4 | explain the accuracy paradox; draw ROC/PR; interpret an odds ratio; derive Xᵀ(p−y)/n |
| **Tue** | 11 Decision Tree Clf · 12 Random Forest Clf | 4.5 | compute Gini/entropy/information gain by hand; explain Condorcet's jury theorem |
| **Wed** | 13 KNN Clf · 14 Naive Bayes | 3.5 | apply Bayes' theorem to a base-rate problem; explain why NB is over-confident |
| **Thu** | 15 SVM | 2.5 | compute a margin; explain C and γ; tune on a log grid |
| **Fri** | 16 Showdown · 17 Selection Guide | 4.5 | pick a threshold from costs; classify a dataset's "fingerprint"; solve the 3 mystery datasets |
| **Sat–Sun** | **Projects 1 (finish), 3, and 2 or 4** | 10+ | deliver models with model cards |

> Core projects: **P1 (regression)** and **P3 (classification)**. Extended: **P2** (fairness & what-ifs) and **P4** (time series). Do all four for a complete portfolio.

## 🌿 Option B: 4-week relaxed (≈ 13 h/week)
| Week | Content |
|---|---|
| 1 | 00, 01, 02, 03 + review quizzes |
| 2 | 04, 05, 06, 07, 08 + Project 1 |
| 3 | 09, 10, 11, 12, 13 |
| 4 | 14, 15, 16, 17 + Projects 2–4 (one per sitting) |

## 🔁 How to study each lesson (the 5-step loop)
1. **Skim** the objectives and roadmap (3 min).
2. **Read + run** top to bottom; *predict outputs before running*.
3. **Break it**: change a parameter in at least two experiments and write one sentence on what changed.
4. **Solve** the problems (hints one at a time): then compare with `solutions/`.
5. **Teach it back**: explain the algorithm out loud in 2 minutes without notes (or write 5 lines). If you can't, re-read the math section.

## 🧪 Self-test after each week
**Week 8**: Without looking: (a) write the normal equation; (b) sketch bias–variance vs model complexity; (c) list when a forest beats a linear model *and* when it doesn't; (d) name the two scaling-sensitive algorithms from this week.
**Week 9**: Without looking: (a) write the log-loss and its gradient; (b) given costs 5:1, state the threshold; (c) list three metrics for imbalanced data; (d) explain the "naive" assumption; (e) explain what a support vector is.

## 🏁 Mastery checklist
- [ ] I can implement OLS, a decision tree, a random forest, KNN, Naive Bayes, logistic regression and a linear SVM from scratch.
- [ ] I never evaluate on data used for training or tuning.
- [ ] I can spot target leakage and time leakage.
- [ ] I choose metrics before algorithms and thresholds from costs.
- [ ] I can justify an algorithm using the dataset fingerprint **and** back it with cross-validation.
- [ ] I can write an honest model card.
