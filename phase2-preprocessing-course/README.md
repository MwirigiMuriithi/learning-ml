# 🌱 Phase 2: Data Preprocessing & Feature Engineering
### *From raw, messy data to machine-learning-ready data: a hands-on course*

**Duration:** Weeks 3–4 (≈ 25–35 hours) · **Prerequisite:** Phase 1 (Python, pandas basics, exploratory analysis)
**You will build:** a cleaned dataset, a leak-proof scikit-learn preprocessing pipeline, engineered features and a documented explanation of every transformation: on an agricultural dataset.

> *Models don't fail because they're weak. They fail because we hand them garbage.*

---

## 🚀 Quick start

```bash
# 1. (optional) create an environment
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. install
pip install -r requirements.txt

# 3. start Jupyter from THIS folder and open notebooks/00_welcome_and_setup.ipynb
jupyter lab
```
The datasets are already in `data/`. To regenerate them (identical, seeded): `python data/generate_datasets.py`.
Prefer Google Colab? Upload the whole folder to Drive, open a notebook and run `from google.colab import drive; drive.mount('/content/drive')`, then `%cd` into `notebooks/`.

---

## 🗺️ Course map

| # | Notebook | Topic | Time |
|---|---|---|---|
| 00 | `notebooks/00_welcome_and_setup.ipynb` | The big picture, setup check, pandas/NumPy refresher | 1 h |
| 01 | `notebooks/01_data_cleaning_part1.ipynb` | Data types, duplicates, **missing values** (MCAR/MAR/MNAR, imputation) | 2–3 h |
| 02 | `notebooks/02_data_cleaning_part2.ipynb` | **Inconsistent records, invalid values, outliers**; builds `clean_farm()` | 3 h |
| 03 | `notebooks/03_data_transformation.ipynb` | **Normalisation, standardisation, log transforms, encoding** (one-hot, ordinal) | 3 h |
| 04 | `notebooks/04_feature_engineering.ipynb` | **Dates, ratios, price changes, rolling averages, lags, seasonal indicators** | 3–4 h |
| 05 | `notebooks/05_feature_selection.ipynb` | **Correlation, statistical tests, RFE, importance, PCA** | 3 h |
| 06 | `notebooks/06_data_leakage.ipynb` | **Data leakage**: six kinds, experiments, prevention | 3 h |
| 07 | `notebooks/07_pipelines.ipynb` | **scikit-learn Pipelines**, ColumnTransformer, custom transformers, tuning, saving | 3–4 h |
| 🏆 | `project/project_template.ipynb` | **Capstone:** prepare the agricultural dataset for ML | 8–12 h |

Each lesson has: plain-English concepts with analogies → worked examples on tiny data → the same ideas on real (messy) data → *pause-and-predict* prompts → **5 practice problems with 2–3 progressive hints** → quiz → summary of common mistakes.

### How to study each lesson
1. Run every cell **in order** and read the output. 2. *Predict before you run.* 3. Change numbers, break things on purpose. 4. Do the problems: struggle ≥ 10 min before opening a hint. 5. Check `solutions/solutions_XX.ipynb` **after** your own attempt. 6. Take the quiz.

---

## 📁 What's inside

```
phase2-preprocessing-course/
├── README.md                 ← you are here
├── requirements.txt
├── notebooks/                ← the 8 lessons (outputs included, so you can read them without running)
├── solutions/                ← worked solutions to every practice problem (with "why this works")
├── project/
│   ├── PROJECT_BRIEF.md      ← scenario, deliverables, milestones
│   ├── RUBRIC.md             ← how it is graded (100 pts)
│   ├── project_template.ipynb← guided workspace with TODOs and self-checks
│   └── reference_solution/   ← complete solution + all 9 exported deliverables (open AFTER you finish)
├── data/
│   ├── farm_records_raw.csv  ← main messy dataset (2,485 rows)
│   ├── farm_clean_lesson.csv ← output of Lesson 2 (used by Lessons 3–7)
│   ├── crop_prices_daily.csv ← time series for feature engineering
│   ├── soil_spectra.csv      ← 40 correlated columns for PCA
│   ├── generate_datasets.py  ← regenerates all of the above (seeded)
│   └── DATA_DICTIONARY.md
├── utils/
│   ├── course_utils.py       ← loaders, profile(), plotting, check_clean(), check_ml_ready()
│   └── agri_prep.py          ← reusable transformers: DateFeatures, AgriFeatures, Winsorizer
└── docs/
    ├── SYLLABUS.md           ← outcomes, schedule, assessment
    ├── CHEATSHEET.md         ← one-page reference
    ├── GLOSSARY.md
    └── INSTRUCTOR_NOTES.md   ← answer key to the injected mess, pacing, common errors
```

---

## ❗ Notes & honesty box
* **The data are simulated.** The Phase 1 agricultural dataset was not available when this course was built, so `farm_records_raw.csv` stands in for it: same flavour (Kenyan counties, crops, rainfall, soil tests, yield) with realistic mess deliberately injected. **Use your real Phase 1 dataset for the capstone if you can**: the techniques transfer directly; adapt the column names and validity rules.
* Because the data are simulated, some real-world effects are absent (e.g. unmeasured farm effects), and the course says so where it matters (Lesson 6, group leakage).
* **Environment:** developed and every notebook executed with Python 3.12, pandas 3.0, NumPy 2.5, scikit-learn 1.9. Minimum versions in `requirements.txt` should work but were not all tested. Shipped `.joblib` files are version-specific: re-run the notebooks instead of loading them across major versions.
* **Run order:** notebook `02` creates `data/farm_clean_lesson.csv` (a copy is already included). Solutions for Lesson 2 and later read that file.

## 🧭 Troubleshooting
| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: course_utils` | Start Jupyter so the notebook's folder is `notebooks/` (or `solutions/`, `project/`); the first cell adds `../utils` to the path |
| `FileNotFoundError: farm_clean_lesson.csv` | Run `02_data_cleaning_part2.ipynb` top to bottom |
| `TypeError: ... sparse_output` | Upgrade scikit-learn (`pip install -U scikit-learn`): needs ≥ 1.2 |
| `format='mixed'` error in `to_datetime` | Upgrade pandas to ≥ 2.0 |
| Pickle/joblib load error | Re-create the pipeline by re-running the notebook; make sure `utils/` is on `sys.path` |
