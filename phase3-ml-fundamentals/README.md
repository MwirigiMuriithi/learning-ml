# 🎓 Phase 3: Machine Learning Fundamentals
### A hands-on, notebook-based course · Weeks 5–7

> *Instead of telling a computer **how** to solve a problem, show it **examples** and let it figure out the how.*

This course takes you from "what *is* machine learning?" to a complete, honestly evaluated, clearly explained model: in **three weeks**, using **Python, NumPy, pandas, Matplotlib and scikit-learn**.

Every lesson is a Jupyter notebook written like a lecture: analogies first, then pictures, then math, then code, with worked examples solved *out loud*, problems with **graduated hints** and **automatic self-checks**, an exit ticket and a one-page recap. The course ends with a guided **practical assignment** and a full **capstone project**.

---

## 🚀 Quick start (5 minutes)

```bash
# 1. (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. install the libraries
pip install -r requirements.txt

# 3. launch JupyterLab from the course root folder
jupyter lab
```

Then open **`notebooks/00_welcome_and_setup.ipynb`** and run the first cell. You should see ✅ next to every library.

**Requirements:** Python 3.10+, ~200 MB disk space, no GPU, no internet connection needed after installation (all datasets are included).

---

## 🗺️ The course at a glance

| Week | Notebook | Topics | Time |
|---|---|---|---|
| - | `00_welcome_and_setup` | environment check, how the course works | 20 min |
| **5** | `01_introduction_to_machine_learning` | what ML is · types · workflow · features & target · training vs inference · parameters vs hyperparameters | 90 min |
| | `02_supervised_learning` | regression · classification · labelled data · prediction · KNN from scratch | 100 min |
| | `03_unsupervised_learning` | clustering (K-Means, DBSCAN) · pattern discovery · PCA · market-basket rules | 100 min |
| **6** | `04_model_training` | train/validation/test · stratified splits · data leakage · training loops · model selection | 100 min |
| | `05_model_generalization` | overfitting · underfitting · bias–variance · Ridge/Lasso · cross-validation · learning curves | 120 min |
| | `06_regression_metrics` | MAE · MSE · RMSE · R² · baselines · residual analysis · Anscombe's quartet | 90 min |
| | `07_classification_metrics` | accuracy paradox · confusion matrix · precision · recall · F1 · thresholds & costs · ROC/AUC | 120 min |
| **7** | `08_optimization` | loss & cost · gradient descent from scratch · learning rates · convergence · SGD · logistic regression | 120 min |
| | `09_practical_assignment` | **the 7-step lifecycle** with scikit-learn (load → separate → split → train → predict → evaluate → explain) | 2–3 h |
| | `10_capstone_project` | **used-car price estimator**: missing data, pipelines, model comparison, log-target insight, error analysis, written report | 4–6 h |

Total: ≈ 20–25 hours of focused work. A suggested day-by-day plan is in [`docs/study_plan.md`](docs/study_plan.md).

---

## 📁 What's in the box

```
ml-fundamentals-phase3/
├── README.md                   ← you are here
├── requirements.txt
├── course_utils.py             ← tiny helper: load(), setup(), check()
├── notebooks/                  ← START HERE: the lessons (00 → 10)
├── solutions/                  ← worked solutions + walkthroughs (try first, peek second!)
├── datasets/                   ← 10 CSV files + DATA_DICTIONARY.md + generate_datasets.py
└── docs/
    ├── study_plan.md           ← 3-week, session-by-session schedule
    ├── cheatsheet.md           ← every formula and scikit-learn pattern on two pages
    ├── glossary.md             ← A–Z of every term used
    └── instructor_notes.md     ← teaching notes, misconceptions, discussion questions
```

### The datasets
| File | Rows | Used for |
|---|---|---|
| `students.csv` | 200 | study hours → exam score (simple regression, gradient descent) |
| `housing.csv` | 900 | home prices (regression, metrics, the practical assignment) |
| `customers.csv` | 400 | shopping-mall segments (clustering) |
| `churn.csv` | 1,500 | customer churn, 15% positive (classification metrics, thresholds) |
| `used_cars.csv` | 1,200 | **capstone**: missing values + categories |
| `noisy_curve.csv` | 80 | `sin(2πx)` + noise (over/underfitting, bias–variance) |
| `iris.csv`, `wine.csv`, `breast_cancer.csv`, `diabetes.csv` | classic | classification, PCA, Lasso |

Synthetic datasets are generated from fixed seeds (`datasets/generate_datasets.py`), so everyone gets identical data and identical answers.

---

## 🧭 How to work through a lesson

1. **Read the learning objectives**: they tell you what you'll be able to do afterwards.
2. **Run every cell, and change things.** Notebooks are meant to be scribbled on.
3. **Do the 🤔 Pause & Predict prompts *before* running the code.** Predicting first is how your brain actually learns.
4. **Attempt each 🧩 problem.** Write code in the starter cell, then run the **self-check** cell below it:
   * ✅ `PASSED`: you've got it.
   * ⏳ `NOT YET`: read the statement again, then open **one hint at a time**.
5. **Compare with `solutions/`** only after you've tried: each solution comes with a walkthrough explaining *why*, not just *what*.
6. **Answer the 🎟️ Exit ticket** aloud or on paper before opening the answers.
7. **Screenshot the 📌 Recap**: it's your revision sheet.

### Difficulty guide
⭐ warm-up (minutes) · ⭐⭐ standard (10–20 min) · ⭐⭐⭐ challenge (30+ min; struggling is expected)

---

## ✅ Learning outcomes

After completing Phase 3 you will be able to:

* explain what ML is, its types, and the end-to-end workflow;
* tell features from targets, parameters from hyperparameters, training from inference;
* train regression and classification models and read their decision boundaries/coefficients;
* cluster unlabelled data, choose *k*, profile clusters, and reduce dimensions with PCA;
* split data correctly, **prevent leakage**, and use validation sets and cross-validation;
* **diagnose** overfitting and underfitting and fix them (regularisation, more data, simpler models);
* compute and interpret MAE, MSE, RMSE, R², accuracy, precision, recall, F1, confusion matrices, ROC and AUC, and choose the right one for a business problem;
* implement **gradient descent** from scratch, tune learning rates, recognise convergence;
* **carry a model through the entire lifecycle** and explain its performance to a non-technical audience.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: course_utils` | open Jupyter from the **course root** or keep notebooks inside `notebooks/`; each notebook starts with `sys.path.append("..")` |
| ❌ next to a library in notebook 00 | `pip install -r requirements.txt` in the same environment that runs Jupyter |
| Plots don't appear | run `%matplotlib inline` in a cell, or restart the kernel |
| Results differ slightly from the text | check your scikit-learn version (`pip install -U scikit-learn`); tiny numeric differences are normal, large ones usually mean a different `random_state` |
| A self-check says ⏳ but the answer looks right | variable names must match exactly (e.g. `best_k`, not `bestK`); run all cells *above* the problem first |
| Kernel dies on a big loop | restart (**Kernel → Restart & Run All**) and re-run in order |

To re-create all the datasets: `cd datasets && python generate_datasets.py`.

---

## 📚 Beyond this course

* *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron
* *An Introduction to Statistical Learning* by James, Witten, Hastie, Tibshirani (free PDF)
* scikit-learn **User Guide** and **Examples gallery** (scikit-learn.org)
* Andrew Ng: *Machine Learning Specialization*
* Next phase: decision trees & ensembles, feature engineering, neural networks.

*Happy learning!* 🚀
