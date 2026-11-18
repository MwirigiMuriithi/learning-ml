# 🎓 Instructor Guide: Phase 5: Unsupervised Learning

*For teachers, tutors, and self-study leaders. Students should not need this file.*

## 1. Teaching philosophy (how the notebooks are designed)

| Principle | How it shows up |
|---|---|
| **Intuition before formulas** | Every lesson opens with a story/analogy (cooling plants → K-Means; the night sky from space → DBSCAN; the photographer's problem → PCA; the agro-dealer's shelf → Apriori). |
| **Build it, then use it** | Each algorithm is implemented from scratch in NumPy (K-Means, agglomerative, DBSCAN, PCA, Apriori), checked against scikit-learn, *then* used professionally. |
| **Productive struggle** | Exercises have **hints revealed one at a time**, self-checks (✅ ❌ ⏳) that never crash, and solutions in a separate folder. |
| **Honesty about uncertainty** | The central question of the week: *with no answer key, who grades the exam?* Lesson 6 answers it with four kinds of evidence. Several lessons deliberately show methods *disagreeing* or *failing*. |
| **Data with a secret** | Datasets have planted structure (`*_truth.csv`), revealed *after* students commit, so they feel the difference between "the metric says" and "the truth is". |
| **Communicate or it didn't happen** | Each project ends with a memo for a non-technical stakeholder, graded by rubric. |

## 2. Session plans

Timings assume a mix of live walkthrough (~40%) and student work (~60%).

### Session 0: Welcome (60 min)
- **Hook (5 min):** hold up a handful of mixed coins/seeds; ask students to sort them *without telling them how*. Discuss what "similar" meant to each person → *there is no single right answer*.
- Walk through §1-2 (supervised vs unsupervised; three families). Run the first-clustering cell; **remove the scaler live** to show the failure.
- **Check for understanding:** exercise 2 (S vs U) as a quick poll.
- **Watch for:** students who try to compute "accuracy" for clustering.

### Session 1: K-Means (2 h)
- Live-code `assign_clusters`/`update_centers` while students type along. Use the 4-frame figure to *narrate* the movement.
- **Show the unlucky seed** (`seed=1` or `5`) *on purpose*; it motivates k-means++/`n_init`.
- **Scaling demo** is the lesson's emotional peak: raw 7-feature clustering is *identical* to revenue-only clustering (ARI = 1.00).
- **Misconceptions:** "inertia lowest is best"; "k is found by the algorithm"; "elbow is always visible" (it isn't, for the customer data).
- **Discussion prompt:** *Silhouette picked k=3 for farmers; the extension service wants 4. Who wins and why?*

### Session 2: Hierarchical (90 min)
- Do the **village km example by hand on the board** before any code. Compare single vs complete merge heights.
- Dendrogram reading drill: point at a tall branch → "what does this say?"
- Note the **honest observation**: on the shapes grid, complete linkage unexpectedly succeeds on the rings. Use it to teach *"test on your own data"*.
- **Misconceptions:** y-axis = number of clusters (it's distance); the order of leaves has meaning (only the merge structure does).

### Session 3: DBSCAN (90 min)
- Role-play: students stand in the room, "eps" = arm's length, "min_samples" = 3 → they *become* core/border/noise points. 10 minutes, memorable.
- k-distance plot is a *heuristic*: show the plateau idea.
- The varying-density demo shows a **narrow sweet spot** (eps≈0.5) with failure either side, and HDBSCAN solving it without tuning.
- **Misconceptions:** forgetting radians/`eps = km/6371` (very common); thinking noise is an error; forgetting to scale.

### Session 4: PCA (2 h)
- Start with the banana photograph analogy (physical object if you have one).
- Insist on the 2-D picture (arrows on the ellipse) before any matrices.
- **Sign flips** will happen between students: pre-empt it.
- Digits reconstruction figure is the "wow"; t-SNE section is for *awareness*, not mastery.
- **Misconceptions:** PCs are original features; PC1 "is" the most important variable; low-variance components are useless.

### Session 5: Association rules (2 h)
- Do the 10-receipt table by hand: **DAP⇒Seed (high support, lift 1.11)** vs **Pesticide⇒Sprayer (lower support, lift 3.3)**. This is the whole lesson in one slide.
- Ask students to predict which rules will appear before running Apriori.
- **Misconceptions:** rule = causation; confidence is symmetric; high confidence = interesting; lift < 1 means "bad rule" (it means substitutes).

### Session 6: Evaluation & profiling (90 min)
- Run the **null test live**: same pipeline on uniform noise. The silhouette gap is smaller than the *stability* gap; discuss why.
- Practise the **persona card** with a pair exercise: each pair names two clusters from the heatmap and "pitches" them in 60 seconds.
- Finish with the algorithm-choice table.

## 3. Verified reference results (seeded, scikit-learn 1.9)

Use these to sanity-check student output. Small differences across library versions are normal; cluster *numbers* will differ.

| Where | Quantity | Expected |
|---|---|---|
| L0 | `n_missing` in farmers | 31 (15 + 10 + 6) |
| L1 | Unscaled 7-feature farmers K-Means vs revenue-only: ARI | **1.00** (scaled: ≈ 0.21) |
| L1 | Random-init K-Means on 5-blob data: share of bad local minima | ≈ 47% of 30 single starts (k-means++: 0%) |
| L1 | Farmers k=4 K-Means vs hidden truth: ARI | ≈ **0.82** |
| L1 | Farmers: best silhouette k | 3 (k=4 close; elbow around 3-4) |
| L2 | Cophenetic correlation, products | single .63 · complete .72 · **average .74** · ward .69 |
| L2 | Ward, 7 groups vs true categories: ARI | ≈ 0.81 |
| L3 | Markets, haversine DBSCAN `eps=40 km`, `min_samples=4` | **7 clusters + 7 noise** (the 7 remote markets) |
| L3 | Same at 20 km / 60 km | 8 clusters + 20 noise / 3 clusters + 7 noise |
| L3 | Merged hub pair at 25-40 km | Eldoret + Kitale |
| L4 | Farmers PCA: PC1 / PC1+PC2 / components for 90% | 41% / 64% / **6** |
| L4 | Markets PC1 | **≈ 75%** → "Market Development Index" |
| L4 | Digits: components for 90% variance | 21 (K-Means ARI on 15 PCs ≈ raw 64 pixels, ≈ 0.66) |
| L5 | Grocery, `min_support=0.03` | 90 frequent itemsets; top lift: Wet Wipes ⇔ Diapers ≈ 11.9 |
| L5 | Agro, support ≥ .03, conf ≥ .5, lift ≥ 1.5 | **43 rules**; top: Drinkers & Feeders ⇒ Poultry Feed (lift 9.4) |
| L5 | Agro lowest-lift rule | Pesticide ⇔ DAP, lift **0.60** |
| L6 | Customers: CH peak / DB best / silhouette best | k = 6 / 8 / climbing to 9 |
| L6 | Customers stability (k=2…8) | dips at 4-5 (≈0.87-0.92), recovers ≈0.96 at 6-7 |
| L6 | Hopkins: customers vs noise | ≈ 0.85 vs ≈ 0.50 |
| P1 | Customers k=6: Loyal big spenders | ≈ 15% of customers, ≈ 49% of revenue; top 20% of customers ≈ 57% of spend |
| P2 | Farmers k=4 sizes | ≈ 36 / 29 / 19 / 16 % (subsistence / dairy-mixed / horticulture / commercial) |
| P3 | Markets k=3 sizes | 13 urban hubs / 42 regional / 48 rural |
| P4 | Products Ward vs K-Means (k=7) | ARI ≈ 1.0; silhouette ≈ 0.28 |
| Capstone | Perishable-crop growers' loss vs others | **≈ 14% vs ≈ 18%** (hypothesis rejected); subsistence ≈ 27% |

## 4. Common student errors and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| All clusters look like stripes | Forgot to scale | `StandardScaler` |
| One giant cluster + tiny ones (hierarchical) | Single linkage chaining; or outliers | Ward; log-transform; remove outliers |
| DBSCAN: everything is noise | `eps` too small / unscaled / in degrees instead of radians | k-distance plot; convert units |
| DBSCAN geo gives 1 cluster | Passed km as `eps` to a radians metric | `eps = km/6371` |
| PCA: PC1 explains 99.9% | Unscaled big-number column | Scale; drop revenue |
| `association_rules` is empty | `min_support` too high or wrong boolean table | Lower support; check the one-hot frame |
| Rules in both directions flood the table | Lift is symmetric | De-duplicate by itemset |
| `NaN` error in `fit` | Missing values | Impute (median) |
| Self-check shows ⏳ | Variable not defined (typo) | Match the exact name in the prompt |
| Cluster ids differ from solutions | Arbitrary label order | Expected! Names must match the *student's own* heatmap |

## 5. Assessment notes

- **Auto-checks** guarantee *form* (shapes, ranges, sanity), not *judgement*. Reserve human grading for: choice of *k*, names, the hypothesis test, the memo.
- **Rubrics** are in each project notebook (100 points). The capstone rubric rewards **integration**, **honesty about a failed hypothesis**, and **brevity**.
- **A strong submission** gives *numbers with comparisons*, names clusters by distinctive traits, states what surprised the author, and says what they'd do next.
- **A weak submission** (common pattern): correct code; memo says "we used K-Means and found 4 clusters" with no sizes, no names, no action.
- **Integrity:** solutions are public in the zip; use oral follow-ups ("why 4, not 5?") or modified datasets (`SEED` in `generate_datasets.py`) per cohort. Changing the seed changes numbers, so checks use ranges rather than exact values wherever feasible. (Some exercises, e.g. exact counts of farmers' missing cells, depend on seed 42.)

## 6. Differentiation

| Learner | Support |
|---|---|
| **Needs more scaffolding** | Pair-program L1 §3; allow hints freely; skip challenge (🏆) exercises; do P1 and P4 only |
| **On track** | Full path; challenge exercises optional |
| **Advanced** | All challenges; stretch ideas; replace K-Means with GMM/HDBSCAN in projects; add silhouette-by-cluster analysis; implement k-medoids |

## 7. Extension and integration ideas
- Have students **collect a real mini-dataset** (market prices, class survey) and run the pipeline.
- **Swap datasets:** run Project 2's pipeline on `markets.csv` or Project 3's on `farmers.csv`; ask what breaks.
- **Debate:** *"Is it ethical to segment farmers and treat segments differently?"* (fairness, access, stigma).
- **Bridge to supervised learning:** train a classifier to assign *new* farmers to segments; compare with the decision-tree "field rule".

## 8. Regenerating or modifying data
`python scripts/generate_datasets.py` rebuilds every CSV. Edit the segment recipes (means, shares) at the top of each `make_*` function to create harder or easier problems, e.g. increase `cluster_std`-like spreads to blur segments, or shrink a segment's share to make a "rare group" challenge. After editing, re-run `scripts/run_all_notebooks.py solutions` to see which self-checks need new thresholds.

## 9. Technical notes
- Tested with Python 3.12, NumPy 2.x, pandas 3.x, scikit-learn 1.9, SciPy 1.18, mlxtend 0.25. Requires scikit-learn ≥ 1.3 for `HDBSCAN`.
- The `notebooks/` folder is the student view; `solutions/` notebooks are executed with all checks green (✅).
- Student notebooks ship with **lesson outputs saved** (for browsing) and **self-check outputs cleared**.
- The helper `course_utils.load()` regenerates data automatically if the `data/` folder is missing.
