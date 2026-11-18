# 🌍 Phase 5 · Unsupervised Learning

### *Discover patterns, groups, and relationships in data with no labels.*

**Week 10 · Seven teaching notebooks · Four guided projects · One capstone · ~30 hours of hands-on learning**

> *"No teacher. No answer key. Just data, and curiosity."*

---

## 🎯 What you will be able to do

By the end of this course you can take a table of customers, farmers, markets or crops, or a pile of shop receipts, and:

- **Cluster** it with **K-Means**, **hierarchical clustering** and **DBSCAN**, and know which to choose;
- **Reduce dimensions** with **PCA**, to visualise, compress, denoise, and build composite indices;
- **Mine association rules** with **support, confidence, lift and Apriori**;
- **Judge** whether the patterns are *real* (stability, null tests, internal metrics) when there is no answer key;
- **Profile, name and explain** the groups, and **persuade a decision-maker** to act on them.

## 🚀 Quick start (5 minutes)

```bash
# 1. (optional but recommended) create an isolated environment
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install everything
pip install -r requirements.txt

# 3. check that your machine is ready
python scripts/verify_setup.py

# 4. open the course
jupyter lab notebooks/00_welcome_and_setup.ipynb
```

No installation? Upload the whole folder to **Google Colab** or **JupyterHub**, then run `!pip install mlxtend` in the first cell. (Keep the folder structure so `notebooks/` can find `data/`.)

## 🗺️ The learning path

| # | Notebook | Topic | Time |
|---|---|---|---|
| 0 | `00_welcome_and_setup` | What is unsupervised learning? Setup, datasets, first clustering | 1 h |
| 1 | `01_kmeans_clustering` | K-Means from scratch, k-means++, choosing *k*, scaling, limits | 2 h |
| 2 | `02_hierarchical_clustering` | Agglomerative clustering, linkage, dendrograms, cophenetic correlation | 1.5 h |
| 3 | `03_dbscan_density_clustering` | DBSCAN from scratch, eps/min_samples, k-distance, haversine, HDBSCAN | 1.5 h |
| 4 | `04_pca_dimensionality_reduction` | PCA from scratch, scree, loadings, compression, denoising, indices, t-SNE | 2 h |
| 5 | `05_association_rules_apriori` | Support, confidence, lift, Apriori from scratch, `mlxtend` | 2 h |
| 6 | `06_evaluating_and_profiling_clusters` | Silhouette/DB/CH, stability, Hopkins, profiling, cluster reports | 1.5 h |
| P1 | `07_project1_customer_segmentation` | 🛍️ Customer segmentation | 3 h |
| P2 | `08_project2_farmer_segmentation` | 🌽 Farmer segmentation by production pattern | 3 h |
| P3 | `09_project3_market_clustering` | 🏪 Agricultural market clustering | 3 h |
| P4 | `10_project4_product_grouping` | 🍅 Grouping similar agricultural products | 2.5 h |
| ★ | `11_capstone_agricultural_intelligence_brief` | 🏆 Integrated one-page brief | 5-6 h |

### Syllabus topic → where it's covered

| Syllabus topic | Notebook(s) |
|---|---|
| K-Means | 1 · P1 · P2 · P3 |
| Hierarchical clustering | 2 · P3 · P4 |
| DBSCAN | 3 · P3 · Capstone |
| PCA | 4 · P1-P4 |
| Feature extraction | 4 (§6c, 6d) · P3 |
| Dimensionality-reduction applications | 4 (§6: compression, denoising, indices, visualisation) |
| Support / confidence / lift / Apriori | 5 · Capstone |
| **Practical project 1:** Customer segmentation | P1 |
| **Practical project 2:** Farmer segmentation by production | P2 |
| **Practical project 3:** Agricultural market clustering | P3 |
| **Practical project 4:** Grouping agricultural products | P4 |

## 📁 What's in the box

```
phase5-unsupervised-learning/
├── README.md                  ← you are here
├── COURSE_GUIDE.md            ← syllabus, schedule, how to study, assessment
├── CHEATSHEET.md              ← one-page reference of every algorithm & formula
├── GLOSSARY.md                ← plain-English definitions
├── requirements.txt
├── notebooks/                 ← START HERE: the student notebooks (exercises have stubs)
│   ├── course_utils.py        ← small helper module (plots, self-check, profiling)
│   └── 00 … 11 *.ipynb
├── solutions/                 ← the same notebooks with worked answers (peek only AFTER trying!)
├── data/                      ← 6 datasets + 4 hidden-truth files (+ README data dictionary)
├── scripts/
│   ├── generate_datasets.py   ← regenerates all data (fixed random seed)
│   ├── verify_setup.py        ← checks your installation
│   └── run_all_notebooks.py   ← optional: execute everything end to end
└── instructor/
    └── INSTRUCTOR_GUIDE.md    ← teaching notes, pacing, misconceptions, grading
```

## 🧭 How every lesson is built

1. **A story** that gives the intuition (cooling plants, family trees, the night sky from space, the photographer's problem…).
2. **The idea, then the math, gently.**
3. **Code from scratch** with NumPy, to *see* inside the algorithm, then the professional scikit-learn version.
4. **Worked examples** on agricultural data, with every decision explained.
5. **🧩 Exercises with hints** (revealed one at a time) and **self-checks** (✅ ❌ ⏳).
6. **🎯 A quiz** with hidden answers, and **a cheat-sheet summary**.

## 💡 How to get the most out of it

- **Type the code.** Don't just press Shift+Enter; change numbers and break things.
- **Predict before you run.** Write down what you expect, then compare.
- **Use hints one at a time.** Struggling for 10 minutes teaches more than reading the answer.
- **Never open `data/*_truth.csv` early.** Those are the "answer keys" used only in the 🔓 *Reveal* sections.
- **Do the memos.** Explaining results to a non-technical reader is half of the job.

## 🔒 About the data

All datasets are **synthetic** (computer-generated with a fixed seed), modelled on Kenyan retail and agriculture. They contain *planted* structure so you can learn the methods and later check what you found. No real person, farm or market is described, and numbers should not be used for real policy. See `data/README.md`.

## 🐞 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | `pip install scikit-learn`, then **restart the kernel** |
| `ModuleNotFoundError: No module named 'mlxtend'` | `pip install mlxtend` (needed from Lesson 5) |
| `ModuleNotFoundError: course_utils` | Open notebooks from inside the `notebooks/` folder (the solution notebooks add the path themselves) |
| `FileNotFoundError` for a CSV | `python scripts/generate_datasets.py` (or just call `load("name")`, which regenerates missing data) |
| Plots don't show | Add `%matplotlib inline` at the top of a cell |
| Different numbers from the text | Library versions can shift random results slightly; ranges in the self-checks allow for this. Names/ids of clusters may differ: that's expected |

**Happy exploring! 🚀**
