# 🧑‍🏫 Instructor Notes

For anyone *teaching* this course (in a classroom, a bootcamp, or as a study-group leader). Self-learners may find the "misconceptions" and "discussion" sections useful too.

## Teaching philosophy

1. **Intuition → picture → math → code → back to intuition.** Every concept is first a story, then a plot, then a formula, then a few lines of NumPy.
2. **Predict before you run.** The "Pause & Predict" boxes are designed for a show of hands or a sticky-note poll. Even wrong guesses prime learning.
3. **Cold-call the failure modes.** Leakage, accuracy paradox, divergence: the course deliberately *shows these failing* so students recognise them in the wild.
4. **Everything is verifiable.** Hand calculations are followed by code that confirms them; self-checks give instant feedback; solutions explain reasoning.
5. **Honest evaluation is a habit, not a lesson.** The "test set is touched once" rule is repeated until it's reflex.

## Suggested session structure (90 minutes)

| Min | Activity |
|---|---|
| 0–10 | Warm-up: 2 exit-ticket questions from the *previous* lesson (cold call) |
| 10–35 | Live-code Parts 1–3 of the lesson; students type along; stop at each 🤔 Pause & Predict |
| 35–50 | Worked examples: students solve before you reveal |
| 50–80 | Problems in pairs (pair programming: driver/navigator, swap every 10 minutes) |
| 80–90 | Debrief: which problem was hardest and why? Preview the next lesson |

## Per-lesson notes

### 01 · Introduction
* **Demo that lands:** the Celsius→Fahrenheit "machine discovers the rule" cell. Ask the class to guess the slope before running.
* **Misconceptions:** "ML = AI = deep learning" (clarify nesting); "the model *is* the algorithm" (the model is the algorithm *plus* learned parameters); confusing hyperparameters with parameters (use the study analogy; the "underscore" trick).
* **Discussion:** Name something at home that is rules-based and something that is learned. What would you need to turn one into the other?

### 02 · Supervised learning
* **Demo:** the three decision-boundary panels: ask students which will generalise best *before* telling them it depends.
* **Misconceptions:** "coefficients = importance" (units!); "logistic *regression* is a regression"; "predict_proba is a true probability" (only if calibrated).
* **Discussion:** What would a labelled dataset for *your* favourite app cost to create? Who labels it?
* **Note:** the lesson deliberately trains on all data; flag it loudly and promise the fix in Lesson 4.

### 03 · Unsupervised
* **Demo:** raw vs scaled K-Means (ARI 0.99 → 0.34 when income is in dollars). Let students change the units themselves.
* **Misconceptions:** "K-Means finds the true number of clusters"; "clusters are real categories" (they're a *useful summary*, validate by profiling); "PCA components are original features".
* **Discussion:** A marketing team asks "how many customer segments are there?" What's your honest answer?

### 04 · Model training
* **Demo:** the churn-rate wobble across random seeds with/without `stratify`.
* **Misconceptions:** second `test_size` arithmetic (0.25, not 0.2); "validation = test"; "scaling before splitting is harmless".
* **Discussion:** Describe a real-world leak you can imagine (hospital, bank, hiring).
* **Assessment idea:** hand students a notebook with a hidden leak and ask them to find it.

### 05 · Generalization
* **The pivotal lesson.** Spend extra time on the bias–variance simulation (200 refits): students *see* variance.
* **Misconceptions:** "more complex = better"; "regularization makes the model worse" (it trades variance for bias); "CV gives the final model" (it gives an *estimate*; you refit afterwards).
* **Discussion:** Mia vs Sam (memoriser vs understander), where in daily life do we overfit? (Revising from past papers only.)
* **Live experiment:** change `random_state=3` in the 20-point split to other seeds and watch the "best degree" jump around: a lesson in why small validation sets are unreliable.

### 06 · Regression metrics
* **Demo:** Anscombe's quartet: same R², four stories.
* **Misconceptions:** "R² = accuracy"; "RMSE and MAE are interchangeable"; "a residual plot is optional".
* **Discussion:** For which real-world problem would you optimise MAE? RMSE? (delivery times vs power-grid load).

### 07 · Classification metrics
* **Demo:** the lazy 85%-accurate model. Ask the class to raise a hand if they'd deploy it.
* **Misconceptions:** mixing up precision/recall (use the smoke-alarm story); thinking 0.5 is "the" threshold; "AUC = accuracy".
* **Role-play:** assign roles (hospital director (wants recall), email provider (wants precision), CFO (wants expected cost)) and negotiate a threshold.

### 08 · Optimization
* **Math-anxiety plan:** every formula is paired with a numeric example and code. Do Worked Example 2 (two steps by hand) on a whiteboard before running anything.
* **Demo:** learning-rate sweep, especially α = 0.9 (oscillation) and α = 1.1 (divergence).
* **Misconceptions:** "gradient descent is only for neural nets"; "the loss must fall monotonically with SGD"; forgetting the simultaneous update.
* **Extension:** momentum / Adam as "gradient descent with memory".

### 09 · Practical assignment
* Can be run as a graded lab (rubric in the notebook). Step 7 (explanation) is the differentiator: share examples of weak vs strong explanations after grading.

### 10 · Capstone
* **Designed insight:** the log-target transformation beats a more flexible model. Don't reveal it in advance; the Milestone 1 plot and the Milestone 4 result (tree beats linear) are the clues.
* **Designed discovery:** the two worst test errors have **missing mileage**. Not every student will notice: reward those who investigate rather than only reading the metric table.
* **Presentation option:** 5-minute "to the dealership owner" talk with one slide (baseline vs model vs where it fails).

## Assessment kit

| Tool | Where | Notes |
|---|---|---|
| Auto self-checks | every problem | formative; students can re-run unlimited times |
| Exit tickets | end of each lesson | 5 questions with answers; use as low-stakes quizzes |
| Weekly mini-checkpoints | `study_plan.md` | 10–15 min, no notes |
| Practical assignment | notebook 09 | 100-point rubric included |
| Capstone | notebook 10 | 100-point rubric included; report is human-graded |

**Anti-copying tip:** the datasets are seeded; to give each student a distinct problem, change the `seed` arguments in `datasets/generate_datasets.py` (or each student's `random_state`): all checks are written to be tolerant of sampling variation *except* a few that pin to specific seeds, which are marked in the problem statement.

## Quick answer-key policy
Distribute `solutions/` only after the deadline of the corresponding lesson. Each solution notebook runs top to bottom and all of its self-checks pass: you can use them as a **reference grader**: if a student's variable values match the solution notebook's, they're right.

## Common student errors (and quick fixes)
| Symptom | Likely cause |
|---|---|
| `ValueError: Expected 2D array` | `X` selected with single brackets; use `df[["col"]]` |
| `ValueError: could not convert string to float` | text column left in `X` (needs encoding; Capstone primer) |
| `NaN` in inputs error | missing values; impute inside a pipeline |
| Great test score, terrible in practice | leakage: hunt for it |
| Self-check says NOT YET but plot looks right | variable name mismatch or earlier cell not run |
| Gradient descent shows `inf`/`nan` | learning rate too large or features not scaled |
