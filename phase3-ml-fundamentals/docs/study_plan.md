# 📅 Three-Week Study Plan

**Pace:** 5 sessions per week, ~1.5–2.5 hours each. Adjust freely: the plan is a guide, not a prison. If you fall behind, **protect Week 6** (generalization & evaluation): it contains the ideas that matter most in practice.

Legend: 📖 lecture reading · 🧩 problems · 🎟️ exit ticket · 🔁 review

---

## 🗓️ Week 5: *What is machine learning, and what can it learn?*

| Session | Do this | Done when you can… |
|---|---|---|
| **Mon** | `00_welcome_and_setup` (20 min) → `01` Parts 1–3: rules vs examples, definition, types of ML | explain, in one sentence, how ML differs from ordinary programming |
| **Tue** | `01` Parts 4–7: features/target, workflow, training vs inference, parameters vs hyperparameters → 🧩 Problems 1.1–1.4 | separate `X` and `y` in code; say which of `coef_` / `alpha` is a hyperparameter |
| **Wed** | `01` Problems 1.5–1.6 + 🎟️ → `02` Parts 1–2 (regression) | read a coefficient out loud ("holding all else equal…") |
| **Thu** | `02` Parts 3–5 (classification, labelled data, prediction) → 🧩 Problems 2.1–2.6 (try 2.6, KNN from scratch) | explain `predict` vs `predict_proba` |
| **Fri** | `03` unsupervised learning (clustering → pattern discovery → PCA) → 🧩 Problems 3.1–3.6 | cluster, choose *k*, profile clusters; explain PCA's "shadow" idea |
| **Weekend (optional)** | 🔁 redo any ⏳ problem; re-run all lectures with changed numbers | |

**Week 5 mini-checkpoint (10 min, no notes):** Name the three families of ML and one example each. What's the difference between a parameter and a hyperparameter? Why scale before K-Means?

---

## 🗓️ Week 6: *Learning honestly: training, generalization and evaluation*

| Session | Do this | Done when you can… |
|---|---|---|
| **Mon** | `04` Parts 1–3: the exam analogy, splitting, stratification, **data leakage** → 🧩 4.1–4.3 | do a 60/20/20 split without arithmetic errors; spot a leaky column |
| **Tue** | `04` Parts 4–5: training loops, model selection → 🧩 4.4–4.6 + 🎟️ | state the rule: *validation to choose, test to report: once* |
| **Wed** | `05` Parts 1–3: overfit/underfit, **bias–variance** | draw the U-shaped test-error curve from memory and label it |
| **Thu** | `05` Parts 4–5: regularization, **cross-validation**, learning curves → 🧩 5.1–5.6 | explain what increasing α does; run `cross_val_score` and report mean ± std |
| **Fri** | `06` regression metrics → 🧩 6.1–6.6 | compute MAE/RMSE/R² by hand; explain why RMSE ≥ MAE; interpret a residual plot |
| **Sat/Sun** | `07` classification metrics (long: split over two sittings) → 🧩 7.1–7.8 | build a confusion matrix; choose precision vs recall for a given problem; explain AUC |

**Week 6 mini-checkpoint (15 min, no notes):** A fraud model has 99.7% accuracy on data with 0.3% fraud (is it good? A model has train RMSE 1.2 and validation RMSE 9.8) diagnose and give two cures. What does AUC = 0.8 mean?

---

## 🗓️ Week 7: *Under the hood, then put it all together*

| Session | Do this | Done when you can… |
|---|---|---|
| **Mon** | `08` Parts 1–3: loss vs cost, **gradient descent**, hand-worked steps | do one gradient step on paper and confirm it in code |
| **Tue** | `08` Parts 4–7: learning rates, convergence, SGD, logistic regression → 🧩 8.1–8.7 | diagnose a diverging learning rate from the cost curve |
| **Wed** | `09` Practical Assignment, Steps 1–5 | have a fitted model + predictions |
| **Thu** | `09` Steps 6–7 (evaluation + written explanation) | write a number-backed, jargon-free explanation |
| **Fri–Sun** | `10` **Capstone**: M1–M3 (Fri), M4–M6 (Sat), M7–M9 + report (Sun) | submit the finished report |

**Final self-assessment:** can you explain, to a friend with no technical background, *how a model learns, how we know it's good, and how it could fail?* If yes: you've got Phase 3. 🎉

---

## ⏱️ If you only have…

* **One week:** Lessons 1 (skim) → 4 → 5 → 6 → 9. Skip 3, 7 (Parts 5–6), 8 (Parts 5–7).
* **Two weeks:** everything except the Capstone; do the Capstone's M1–M4 and M7.
* **A classroom with 90-minute sessions:** one lesson per session; assign the exit ticket as homework; use sessions for the Pause & Predict questions as class polls. See `instructor_notes.md`.

## 🔁 Spaced review (10 minutes each)
* **Day +1:** re-read the 📌 Recap of yesterday's lesson without looking at code.
* **Day +3:** redo one ⭐⭐ problem from scratch in a blank notebook.
* **Day +7:** answer the exit-ticket questions *aloud*.
