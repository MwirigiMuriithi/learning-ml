# 🎓 Supervised Learning Algorithms: The Complete Course
### Phase 4 · Weeks 8–9 · 18 lessons · 134 problems with hints · 4 portfolio projects

> *"Tell me and I forget. Show me and I remember. Let me build it, break it, and fix it, and I understand."*

Welcome! This course takes you from "I can run `model.fit()`" to **"I can choose the right supervised-learning algorithm for a problem, justify the choice with evidence, and deliver it responsibly."** Every algorithm is taught the same way: a story, the intuition, the mathematics, a **from-scratch implementation**, the scikit-learn version, parameter experiments, worked examples, problems with hints, and a quiz.

---

## 🧭 What you will be able to do
After this course you can look at a new dataset and say *why* you'd reach for Lasso, a random forest, an RBF-SVM or logistic regression: then **prove it** with a fair, leakage-free cross-validated experiment, pick a decision threshold from business costs, check the model for fairness, and package it with a model card.

### Phase 4 requirements → where they are covered
| Your Phase 4 objective | Where |
|---|---|
| **Regression:** Simple LR · Multiple LR · Polynomial · Decision Tree · Random Forest · SVR · KNN | Lessons 01 – 07 (one lesson each) + the Regression Showdown (08) |
| **Classification:** Logistic Regression · Decision Tree · Random Forest · KNN · Naive Bayes · SVM | Lessons 10 – 15 (one lesson each) + metrics (09) + Classification Showdown (16) |
| *Underlying concept* | "🧠 The idea" section of every lesson (+ a story hook) |
| *How the algorithm works* | worked-by-hand examples + the **from-scratch** implementation |
| *Mathematical foundation* | "📐 The math" section (derivations, LaTeX) |
| *Important parameters* | parameter experiments + the "full profile" table |
| *Advantages / disadvantages* | the **📋 Cheat-sheet: the full profile** table closing every algorithm lesson |
| *Computational complexity (high level)* | the same table + timing experiments (KNN, SVM, SVR) |
| *Appropriate use cases / limitations* | the same table + Lesson 17 (selection guide) |
| *Implement core algorithms with scikit-learn* | every lesson |
| *Simplified manual implementations* | **all 13 algorithms** are implemented from scratch (closed form, gradient descent, Newton, CART, bagging, sub-gradient SVM/SVR, brute-force KNN, Naive Bayes) and verified against scikit-learn |
| **Projects:** House prices · Student performance · Loan approval · Agricultural prices | `projects/P1` … `P4` |
| **Expected outcome:** *choose an appropriate algorithm from the problem and dataset* | Lessons 08, 16 and **17 (capstone)**, plus all four projects |

---

## 📁 What's in the box
```
supervised_learning_course/
├── README.md                  ← you are here
├── requirements.txt           ← pip install -r requirements.txt
├── environment.yml            ← conda alternative
├── course_utils.py            ← tiny helper library (check(), plotting helpers, data paths)
├── notebooks/                 ← 18 STUDENT lesson notebooks (00 – 17): read, run, solve
├── solutions/                 ← the same lessons, EXECUTED, with worked solutions for every problem
├── projects/                  ← 4 STUDENT project notebooks (P1 – P4) with guided milestones
├── project_solutions/         ← reference solutions for the projects (executed)
├── data/                      ← 9 datasets (CSV) + generate_datasets.py (reproducible, seeded)
├── docs/
│   ├── STUDY_PLAN.md          ← day-by-day schedule (2-week intensive or 4-week relaxed)
│   ├── CHEATSHEET.md          ← one-page algorithm, metrics and decision cheat-sheet
│   ├── GLOSSARY.md            ← ~80 key terms in plain English
│   ├── MODEL_CARD_TEMPLATE.md ← for your project deliverables
│   └── INSTRUCTOR_NOTES.md    ← teaching tips, timing, how to extend & regenerate
└── source/                    ← the generator that wrote every notebook (edit & rebuild if you wish)
```

## 🚀 Quick start (5 minutes)
```bash
# 1. create an environment (Python 3.10+)
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. launch Jupyter from the course folder
jupyter lab            # or: jupyter notebook

# 3. open notebooks/00_Welcome_and_Foundations.ipynb  and press Shift+Enter
```
> ⚠️ **Open each notebook from inside its own folder** (Jupyter does this by default). The first cell of every notebook finds `course_utils.py` and `data/` one level up; keep the folder structure intact.
> ☁️ **Google Colab / other cloud:** upload the *whole* folder (or the zip, then unzip), keep `notebooks/` and `data/` side by side, and `pip install -r requirements.txt` in a cell. Requires **scikit-learn ≥ 1.8** (the L1-penalty code uses the newer API).

## 🎒 How to learn with this course
Each lesson notebook has the same rhythm: treat it like a lecture + lab:

1. **🎬 The hook**: a story that tells you *why* the algorithm exists.
2. **🎯 Objectives & 🗺️ roadmap**: what you'll be able to do by the end.
3. **Concept → math → build it from scratch → scikit-learn**: read the text *before* running the code.
4. **🎛️ Experiments**: change the parameters, watch what breaks.
5. **📋 Profile table**: concept, parameters, pros/cons, complexity, use cases, limitations in one place.
6. **🧪 Worked examples**: solved step by step (do them by hand once!).
7. **🧩 Problems** (⭐ easy → ⭐⭐⭐ hard): each has **collapsible hints** (open one at a time), starter code, and a ✅/❌ **self-check** that never crashes your notebook. Only after a real attempt, compare with `solutions/`.
8. **🧠 Quiz & 🔁 recap**: answers are hidden behind a click.

**The golden rules**
- *Attempt before you peek.* Struggle is where the learning happens.
- *Predict before you run.* Write down what you expect the output to be.
- *Break things on purpose.* Change `k`, `C`, `max_depth`, remove scaling: observe.
- *Never touch the test set until the end*: the single most important habit in applied ML.

## 🗓️ Suggested schedule (details in `docs/STUDY_PLAN.md`)
| Week 8 (Regression | Week 9) Classification & Choosing |
|---|---|
| Day 1: 00 Foundations · 01 Simple LR | Day 1: 09 Metrics · 10 Logistic Regression |
| Day 2: 02 Multiple LR | Day 2: 11 Decision Tree · 12 Random Forest |
| Day 3: 03 Polynomial · 04 Decision Tree | Day 3: 13 KNN · 14 Naive Bayes |
| Day 4: 05 Random Forest | Day 4: 15 SVM |
| Day 5: 06 SVR · 07 KNN | Day 5: 16 Showdown · 17 Selection Guide |
| Weekend: 08 Showdown · Project 1 | Weekend: Projects 2 – 4 |

Total ≈ 35 hours of lessons + ≈ 17 hours of projects. A relaxed 4-week plan is also provided.

## 📚 Lesson catalogue
| # | Notebook | Core ideas | ⏱️ |
|---|---|---|---|
| 00 | Welcome & Foundations | supervised learning, workflow, sklearn API, metrics, bias–variance, CV, pipelines | 75 |
| 01 | Simple Linear Regression | least squares, closed form & gradient descent, residuals, Anscombe, bootstrap | 100 |
| 02 | Multiple Linear Regression | normal equation, inference, VIF, categoricals, log target, Ridge/Lasso | 120 |
| 03 | Polynomial Regression | Vandermonde, degree selection, measured bias–variance, conditioning, Ridge | 100 |
| 04 | Decision Tree Regression | CART from scratch, pruning, instability, extrapolation | 120 |
| 05 | Random Forest Regression | bagging, variance formula, OOB, permutation importance, intervals | 120 |
| 06 | Support Vector Regression | ε-tube, kernels, rebuilding predictions from support vectors | 120 |
| 07 | KNN Regression | distances, scaling, curse of dimensionality, "show me the comps" | 100 |
| 08 | **Regression Showdown** | fair tournament of all 7 regressors, tuning, error analysis, learning curves | 120 |
| 09 | Classification Foundations & Metrics | confusion matrix, accuracy paradox, ROC/PR, log loss, calibration, cost thresholds | 110 |
| 10 | Logistic Regression | sigmoid, odds, max likelihood, Newton vs GD, odds ratios, regularisation, softmax | 130 |
| 11 | Decision Tree Classification | Gini/entropy, information gain by hand, CART & ID3, pruning, decision paths | 130 |
| 12 | Random Forest Classification | Condorcet jury, soft voting, OOB, calibration, class weights | 120 |
| 13 | KNN Classification | voting, k and boundaries, digits, imbalance, thresholds | 110 |
| 14 | Naive Bayes | Bayes' theorem, base-rate trap, Gaussian/Multinomial from scratch, spam filter | 120 |
| 15 | Support Vector Machines | max margin, hinge loss, kernel trick, C/γ, from-scratch SVM | 140 |
| 16 | **Classification Showdown** | tournament, calibration, cost-based threshold & savings | 130 |
| 17 | **Algorithm Selection Guide** | No Free Lunch, dataset fingerprint, linearity gap, 3 mystery datasets | 150 |
| P1 | House Price Prediction | audit, leakage, pipeline, tournament, tuning, error analysis, model card | 240 |
| P2 | Student Performance | regression + at-risk classifier, costs, **fairness audit**, what-if analysis | 240 |
| P3 | Loan Approval | **leakage hunt**, costs, fairness + proxy test, reason codes, `decide()` | 270 |
| P4 | Agricultural Prices | **time-series** validation, lags, baselines, rolling backtest | 270 |

## 🗂️ The datasets (all synthetic, fixed seed, regenerable)
`advertising` · `position_salaries` · `house_prices` (Nairobi-flavoured, KES) · `student_performance` · `loan_approval` (with a leakage trap!) · `agri_prices` (96 months × 6 crops × 5 markets) · `sms_spam_mini` · `mystery_A/B/C`. See `data/README.md`. We also use scikit-learn's built-in breast-cancer, iris and digits data. Synthetic data lets us hide *known* structure (so you can check whether your model finds it) while keeping the messiness of real data: missing values, typos, imbalance, leakage, seasonality.

## ✅ Quality guarantees (how this course was built)
Every notebook was **executed end-to-end**; every problem's reference solution passes its own self-checks; every student notebook runs top-to-bottom without errors before you solve anything; and every number quoted in the narrative was compared against the actual output. Where a result surprised the author (e.g. *Naive Bayes was **not** the worst-calibrated model on loans*; *the degree-2 polynomial led until the SVR was tuned*), the lesson says so: real data doesn't read the textbook.

## 🆘 Troubleshooting
- **`ModuleNotFoundError: course_utils`** → you opened the notebook from the wrong folder; keep `notebooks/` next to `course_utils.py`.
- **`FileNotFoundError` for a CSV** → keep `data/` next to `notebooks/`; or run `python data/generate_datasets.py`.
- **LogisticRegression L1 error / warnings about `penalty`** → upgrade scikit-learn (`pip install -U scikit-learn`, need ≥ 1.8).
- **Slow cells** (SVM, forests in the showdowns) → reduce `n_estimators` / grid sizes, or run on a machine with several cores (`n_jobs=-1`).
- **Plots not showing** → run `%matplotlib inline` in a cell.

## 🙏 A final word from your professor
Machine learning is not about finding the fanciest algorithm. It's about **asking the right question, measuring honestly, and telling the truth about what your model can and cannot do.** If you finish this course able to say *"this problem looks linear and wide, so I'd start with Lasso; here's the experiment that proves it; and here is where it will fail"*: then you've learned the real thing. Now go build something useful. 🚀
