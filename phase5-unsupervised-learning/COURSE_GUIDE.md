# 📘 Course Guide: Phase 5: Unsupervised Learning

## 1. Objective
> Understand how machine learning can discover patterns **without labelled training data**.

**Expected outcome:** *You can identify patterns and relationships in datasets without predefined target labels, and you can show that they are real, name them, and act on them.*

## 2. Learning outcomes (what you'll be assessed on)

| # | After this course you can… | Bloom level | Where |
|---|---|---|---|
| LO1 | Distinguish supervised from unsupervised problems and choose the right family | Understand | L0, L6 |
| LO2 | Implement and explain K-Means, agglomerative clustering, DBSCAN | Apply / Analyse | L1-L3 |
| LO3 | Choose and defend *k*, linkage, `eps`, `min_samples` | Evaluate | L1-L3, L6 |
| LO4 | Apply PCA for visualisation, compression, denoising, and index building; interpret loadings | Apply / Analyse | L4 |
| LO5 | Compute and interpret support, confidence, lift; run Apriori and filter rules | Apply / Evaluate | L5 |
| LO6 | Evaluate clusterings without labels (internal metrics, stability, null tests) | Evaluate | L6 |
| LO7 | Profile, name and explain clusters; communicate to non-technical audiences | Create | P1-P4, Capstone |
| LO8 | Integrate several analyses into a decision-ready recommendation | Create | Capstone |

## 3. Pacing options

### Option A: "Week 10" intensive (≈ 22 h of work, 5 days)

| Day | Morning | Afternoon |
|---|---|---|
| **Mon** | L0 Welcome (1 h) · L1 K-Means (2 h) | L1 exercises; start L2 |
| **Tue** | L2 Hierarchical (1.5 h) | L3 DBSCAN (1.5 h) |
| **Wed** | L4 PCA (2 h) | L4 exercises · start L5 |
| **Thu** | L5 Association rules (2 h) | L6 Evaluation & profiling (1.5 h) |
| **Fri** | **Project 1** (3 h) | **Project 2** (3 h): or Project 3 |

Projects P3, P4 and the Capstone become the follow-up week.

### Option B: Two weeks (≈ 30 h, comfortable)

| Week | Content |
|---|---|
| **10a** | L0-L6 (11.5 h) + Project 1 |
| **10b** | Projects 2-4, Capstone |

### Option C: Self-paced evenings (2 h/evening, 15 evenings)
One lesson or one project milestone-group per evening. Skip nothing in L1, L5, L6.

## 4. How to study each notebook (the "3-pass" method)

1. **Pass 1: Read.** Run each cell, read the story, look at every plot. Don't open the exercises yet.
2. **Pass 2: Play.** Change parameters. What if *k* = 2? What if `eps` is half? Break it on purpose.
3. **Pass 3: Do.** Do the exercises. Open hints one at a time; use self-checks; attempt the quiz out loud.

*Study rule of thumb:* **20% reading · 40% experimenting · 40% exercises.**

## 5. Assessment plan (suggested)

| Component | Weight | Notes |
|---|---:|---|
| Lesson exercises (self-checks passing) | 15% | Autochecked; ✅ count in notebooks 0-6 |
| Quizzes | 5% | Self-marked; discuss in class |
| Projects 1-4 (25% each of this block) | 40% | Rubrics inside each notebook (100 pts each) |
| **Capstone brief + notebook** | 35% | Rubric inside notebook |
| Peer review | 5% | Checklist at end of the capstone |

**Mastery thresholds:** ≥ 70% = *competent*; ≥ 85% = *proficient*; ≥ 95% with an insight that surprised the grader = *distinguished*.

## 6. Prerequisites (and a 10-minute self-test)

You should be comfortable with: Python basics (lists, functions, loops), NumPy arrays, pandas (`read_csv`, `groupby`, indexing), matplotlib basics, and supervised ML vocabulary from Weeks 6-9 (features, train/test, overfitting).

**Self-test.** Can you (without searching) …
- compute the mean of a DataFrame column grouped by another column?
- write a `for` loop that fits a scikit-learn model for several parameter values?
- explain what standardising a column does?

If not, spend an hour with the NumPy/pandas notes from earlier weeks first.

## 7. Datasets at a glance

| File | Rows | Used for |
|---|---:|---|
| `retail_customers.csv` | 800 | L1, L6, **P1** |
| `farmers.csv` | 700 | L0, L1, L4, L6, **P2**, Capstone |
| `markets.csv` | 103 | L0-L4, **P3**, Capstone |
| `products.csv` | 48 | L2, **P4**, Capstone |
| `grocery_baskets.csv` | 2,500 baskets | L5 |
| `agro_input_baskets.csv` | 3,000 baskets | L5, Capstone |
| `*_truth.csv` | | 🔒 hidden answers: only for 🔓 *Reveal* sections |

Full data dictionary: `data/README.md`.

## 8. Academic honesty and good practice
- Solutions exist so you can **check** your work, not copy it. Try first, compare after.
- If you reuse code from lessons in a project (encouraged!), **comment where it came from** and make sure you understand it.
- Report results *honestly*, including the ones that contradict your expectations (the capstone checks this!).

## 9. Where to go next (Phase 6 and beyond)
- Gaussian mixture models, HDBSCAN, UMAP, spectral clustering
- Anomaly detection (Isolation Forest, one-class SVM)
- Semi-supervised learning and **using your clusters as features** in supervised models
- Time-series clustering and sequential pattern mining
- Recommender systems (a close cousin of association rules)
