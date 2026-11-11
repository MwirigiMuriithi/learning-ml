# 🌾 Phase 1 · Data Analysis & Exploration

**Duration:** 2 weeks · **Level:** beginner → intermediate · **Stack:** Python, NumPy, pandas, Matplotlib, Seaborn
**Data:** Kenyan agricultural markets (crop prices, volumes, counties, markets, production, weather; *synthetic but realistic*)

> **Goal.** Become comfortable exploring an unfamiliar dataset: understand its structure, judge its quality, find patterns,
> and identify the information that could be useful for machine learning.

---

## 🚀 Quick start (5 minutes)

```bash
# 1. (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate            # Windows:  .venv\Scripts\activate

# 2. install the tools
pip install -r requirements.txt

# 3. launch Jupyter and open notebooks/00_Start_Here.ipynb
jupyter lab
```

Start with **`notebooks/00_Start_Here.ipynb`**: it checks your setup and explains how the course works.
Always launch Jupyter from this folder (or any folder): the notebooks locate `data/` automatically through `course_utils.py`.

---

## 📁 What's inside

```
phase1-data-analysis/
├── README.md                    ← you are here
├── requirements.txt
├── course_utils.py              ← helpers: DATA_DIR, check(), check_plot(), first_look()
├── course_answers.json          ← reference answers used by the auto-checker (no peeking!)
├── notebooks/                   ← THE COURSE (work through in order)
│   ├── 00_Start_Here.ipynb
│   ├── 01_NumPy_Arrays_Indexing_Slicing.ipynb
│   ├── 02_NumPy_Vectorization_Broadcasting_Matrices_Random.ipynb
│   ├── 03_Pandas_Series_DataFrames_Import_Select.ipynb
│   ├── 04_Pandas_Cleaning_Missing_Values_Dates_Transforms.ipynb
│   ├── 05_Pandas_GroupBy_Pivot_Merge.ipynb
│   ├── 06_Matplotlib_Visualization.ipynb
│   ├── 07_Seaborn_Statistical_Visualization.ipynb
│   ├── 08_EDA_Playbook.ipynb
│   └── 09_Capstone_Project.ipynb         ← your project (scaffold + rubric)
├── solutions/                   ← open AFTER you try!
│   ├── 01 … 08 _SOLUTIONS.ipynb          ← worked solutions to every problem
│   └── 09_Project_Reference_Solution.ipynb
├── data/                        ← all datasets + generate_datasets.py + README_DATA.md
└── docs/
    ├── SYLLABUS.md              ← day-by-day plan, topic coverage matrix
    ├── CHEATSHEET.md            ← NumPy / pandas / plotting on one page
    ├── EDA_CHECKLIST.md         ← printable EDA routine + report template
    └── GLOSSARY.md              ← every term in plain English
```

The notebooks in `notebooks/` and `solutions/` are shipped **already executed**, so you can read the expected output
before you run anything. To practise, use *Kernel → Restart Kernel and Clear Outputs* and run them yourself.

---

## 🗓️ The two-week plan

| Day | Notebook | Focus | Hours |
|---|---|---|---|
| **1** | 00 + 01 | Setup · NumPy arrays, indexing, slicing, masks | 3 |
| **2** | 02 | Vectorisation, broadcasting, matrices, random numbers | 3 |
| **3** | 03 | pandas: Series/DataFrame, CSV/Excel/JSON, selecting, filtering | 3 |
| **4** | 04 | Cleaning: quality audit, missing values, dates, transformations | 3.5 |
| **5** | 05 | groupby, pivot, merge: business questions | 3.5 |
| **6–7** | - | *Weekend:* redo 5 problems you found hardest; read the cheat-sheets | 2 |
| **8** | 06 | Matplotlib: charts that tell the truth | 3 |
| **9** | 07 | Seaborn: distributions, heatmaps, correlation | 3 |
| **10** | 08 | The EDA Playbook on a brand-new dataset | 3.5 |
| **11–12** | 09 | **Capstone project** | 6–10 |
| **13** | 09 | Polish, "Restart & Run All", compare with reference solution | 2 |
| **14** | - | Review, self-assessment, plan Phase 2 | 1 |

*Total: roughly 35–45 hours.* Part-time? Stretch the same sequence over 4 weeks.

---

## 🧭 How each lesson works

1. **Objectives** → 2. **Explanations with runnable code** (and 🧠 *Pause and predict* moments) →
3. **🧑‍🏫 Worked examples** (the thinking steps, then code) → 4. **🏋️ Problems** with hidden hints and an **auto-check** →
5. **✅ Check your understanding** → 6. **🧾 Cheat-sheet**.

**The ten-honest-minutes rule:** struggle for ten minutes, then open Hint 1; the final hint is nearly the answer. Only open `solutions/` after you've finished.

**Auto-checks.** In problem cells you set `answer = ...` and the cell calls `check("03-p2", answer)`.
It tells you ✅ or ❌ and *how* you're off (wrong type, length, values): it never shows the reference answer.
Chart problems call `check_plot(ax, ...)`, which verifies title, axis labels and enough lines/bars/points. 

---

## 🎓 The capstone project

`notebooks/09_Capstone_Project.ipynb` gives you a client brief, the data, a 100-point rubric, a scaffold and stretch goals.
You will produce: dataset overview · data-quality assessment (with a decision log) · descriptive statistics · visualisations ·
correlation analysis · observations & hypotheses · ML-readiness assessment.
Compare with `solutions/09_Project_Reference_Solution.ipynb` **after** you finish.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'seaborn'` (etc.) | `pip install -r requirements.txt`, then *Kernel → Restart* |
| `ImportError: Missing optional dependency 'openpyxl'` | `pip install openpyxl` |
| `ModuleNotFoundError: course_utils` | open notebooks from their original folder (`notebooks/` or `solutions/`); the first code cell adds `..` to the path |
| `⚠️ no reference answer found` | keep `course_answers.json` next to `course_utils.py` |
| Resample frequency error `'ME'` | upgrade pandas: `pip install -U "pandas>=2.2"` |
| `ZoneInfoNotFoundError` (Windows) | `pip install tzdata` |
| Charts don't appear | run the cell; in plain scripts add `plt.show()`; in Jupyter they display inline |
| Notebook behaves strangely | *Kernel → Restart Kernel and Run All Cells* (hidden state from out-of-order cells is the usual cause) |
| Using Google Colab | upload the whole folder to Drive, mount it, `%cd` into `notebooks/`, run `!pip install openpyxl` |

---

## ⚠️ About the data

All datasets are **synthetic**, generated by `data/generate_datasets.py` with fixed seeds (so everyone has identical files).
They imitate the *structure and rhythm* of Kenyan agricultural markets: two rainy seasons, the 2021–22 drought, food inflation,
retail premiums in cities, but they are **not official statistics**. Do not quote them as facts about Kenya.

Happy exploring. 🌱
