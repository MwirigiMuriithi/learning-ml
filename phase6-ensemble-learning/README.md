# 🎓 Phase 6: Ensemble Learning & Model Optimization
### A two-week, notebook-based course (Weeks 11–12) with worked examples, auto-graded problems, hints, datasets, a practical assignment and a capstone project

> **Core idea of the whole course:** *a committee of average models can beat one brilliant model, and the skill is knowing how to build the committee, tune it, compare it honestly, and explain what it decided.*

**You will be able to:** build and tune ensembles (bagging, boosting, voting, stacking), use XGBoost / LightGBM / CatBoost, run rigorous cross-validation and statistical model comparison, tune with grid / random / Bayesian search, and explain predictions with permutation importance and SHAP: and **make evidence-based model selections**.

---

## 🚀 Quick start (10 minutes)

```bash
# 1. (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate            # Windows:  .venv\Scripts\activate

# 2. install the libraries
pip install -r requirements.txt

# 3. check everything works
python verify_install.py

# 4. start learning
jupyter lab                           # then open notebooks/00_welcome_and_setup.ipynb
```

Run notebooks **top to bottom**, from inside the `notebooks/` folder (that's where `../utils` and `../data` resolve correctly; JupyterLab does this automatically).

> **Mac note:** if LightGBM complains about `libomp`, run `brew install libomp`. **Windows note:** if CatBoost/SHAP fail to install, use Python 3.10–3.12 and upgrade pip first (`python -m pip install -U pip`).

---

## 🗂️ What's inside

```
phase6-ensemble-learning/
├── README.md                ← you are here
├── SYLLABUS.md              ← day-by-day schedule, learning outcomes, topic-coverage checklist
├── CHEATSHEET.md            ← one-page revision sheet (formulas, hyperparameters, decision guide)
├── GLOSSARY.md              ← every key term in plain English
├── requirements.txt · verify_install.py
│
├── notebooks/               ← THE COURSE (student versions: problems are blank)
│   ├── 00_welcome_and_setup.ipynb
│   ├── 01_why_ensembles_work.ipynb
│   ├── 02_bagging_and_random_forest.ipynb
│   ├── 03_boosting_from_adaboost_to_gradient_boosting.ipynb
│   ├── 04_xgboost_lightgbm_catboost.ipynb
│   ├── 05_voting_and_stacking.ipynb
│   ├── 06_cross_validation_and_model_comparison.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_model_interpretability.ipynb
│   └── 09_practical_assignment_regression_showdown.ipynb      ← the graded assignment
│
├── project/                 ← CAPSTONE
│   ├── 10_capstone_project_crop_yield.ipynb
│   ├── PROJECT_BRIEF.md
│   └── RUBRIC.md
│
├── solutions/               ← fully worked, executed solution notebooks (peek only after trying!)
├── data/                    ← housing.csv · loan_default.csv · crop_yield.csv · DATA_DICTIONARY.md
├── utils/                   ← course_utils.py (helpers & auto-grader) · animations.py · make_datasets.py
└── assets/                  ← animated GIFs used in the lessons
```

---

## 🧭 How each lesson is built (the "CS50 / Dr. Chuck" format)

Every notebook follows the same rhythm so you always know where you are:

| block | what it is |
|---|---|
| 🎯 **Learning goals** | exactly what you'll be able to do afterwards |
| 💡 **Intuition first** | an analogy or story *before* any maths |
| 🔮 **Predict-then-run** | you commit to a guess, then run the code: being wrong is where learning happens |
| 🧪 **Worked examples** | I solve it by hand, then the computer confirms |
| 🧱 **From-scratch code** | AdaBoost, bagging, gradient boosting, stacking, Shapley values, Bayesian optimisation: built in a few lines so nothing is magic |
| ✋ **Check your understanding** | quick questions with hidden answers |
| 🧩 **Problems** | ⭐ / ⭐⭐ / ⭐⭐⭐: each has **layered hints** (open them one at a time) and an **instant auto-check** (✅ / ❌ with a nudge) |
| 🏁 **Recap & next-up** | one-screen summary and a teaser |

### How to use the problems
1. Try the problem for **at least 5 minutes**.
2. Still stuck? Open **Hint 1**; then **Hint 2**, and so on. They get progressively more specific.
3. Run the cell: the **self-check** tells you if you're right and what to look at if you're not.
4. Compare with `solutions/<same-notebook>_SOLUTIONS.ipynb` *after* you pass (or after a real struggle). **Type** it, don't paste it.
5. `scoreboard()` at the bottom of each notebook summarises your progress.

---

## 📅 Suggested pace (2 weeks, ~12–14 hours/week)

| | Mon | Tue | Wed | Thu | Fri | Weekend |
|---|---|---|---|---|---|---|
| **Week 11** | L00 + L01 | L02 | L03 | L04 | L05 | review + redo hard problems |
| **Week 12** | L06 | L07 | L08 | Assignment (L09) | Assignment + start capstone | Capstone |

Details, objectives and a topic-coverage checklist are in **SYLLABUS.md**.

---

## 🧪 The datasets (all synthetic, deterministic, intentionally messy)

| file | rows | task | used in |
|---|---|---|---|
| `housing.csv` | 3,000 | regression: price | Lessons 00–09 (the main dataset) |
| `loan_default.csv` | 5,000 | binary classification (14.5% positives) | Lessons 05, 06 + capstone stretch |
| `crop_yield.csv` | 2,500 | regression: yield (t/ha) | **Capstone project** |

Synthetic data lets us ask a question real data never allows: **"did the model discover the true recipe?"**: and in Lesson 08 you'll watch SHAP recover step functions, hinges and interactions we planted on purpose. Full column descriptions: `data/DATA_DICTIONARY.md`. To regenerate: `python utils/make_datasets.py`.

---

## 🧠 Topic coverage (maps 1-to-1 to the Phase 6 syllabus)

| Syllabus topic | Where |
|---|---|
| **Ensemble learning:** bagging · boosting · voting · stacking · Random Forest | L01 (why), L02 (bagging, RF), L03 (boosting), L05 (voting, stacking) |
| **Gradient boosting:** GBM · XGBoost · LightGBM · CatBoost | L03 (GBM from scratch), L04 (all three libraries) |
| **Model optimisation:** hyperparameter tuning · grid · random · Bayesian concepts · cross-validation · feature importance · model comparison | L07 (tuning), L06 (CV & comparison), L02/L08 (importance) |
| **Interpretability:** feature & permutation importance · SHAP · explaining predictions | L08 |
| **Practical assignment:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost on one dataset; evaluate and explain why one wins | L09 |
| **Expected outcome:** optimise models, compare approaches, evidence-based selection | L06 + L07 + L09 + Capstone |

---

## ✅ Self-test: are you done?

You've completed Phase 6 when you can, **without looking anything up**:
- explain, with the formula, why correlation limits what averaging can achieve (L01);
- say what bagging and boosting each reduce (variance / bias) and why boosting needs early stopping (L02–03);
- compute an XGBoost leaf weight and split gain by hand (L04);
- explain why stacking needs out-of-fold predictions (L05);
- design a validation scheme that matches deployment and spot leakage (L06);
- choose a tuning method for a given budget and defend the search space (L07);
- turn a SHAP plot into a sentence a manager understands, and say what it *cannot* tell them (L08);
- finish the assignment and capstone with a written, evidence-backed recommendation.

*Happy modelling, and remember rule number one of this course: **type the code, predict before you run, and break things on purpose.** 🌲🚀*
