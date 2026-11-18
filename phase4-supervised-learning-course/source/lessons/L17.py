from nbkit import Lesson


def make():
    nb = Lesson("17_Algorithm_Selection_Guide", 17, "Algorithm Selection Guide; The Capstone",
                "No free lunch: reading a dataset's fingerprint, and choosing the algorithm that fits it",
                150, "🔴 Integrative",
                setup_extra="""import time, warnings
from sklearn.datasets import load_breast_cancer, make_moons, make_classification
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, recall_score
warnings.filterwarnings("ignore")""")
    nb.header(
        objectives=[
            "State the **No Free Lunch** principle and *demonstrate* it: no algorithm wins on every dataset.",
            "Describe a dataset by its **fingerprint**: size, dimensionality, linearity, noise, imbalance, feature types, constraints.",
            "Use a **flowchart and a comparison table** to shortlist algorithms for regression and classification.",
            "Measure how non-linear a problem is with the *linearity gap* (flexible model − linear model).",
            "Write a small rule-based recommender and test it.",
            "Solve three **mystery datasets** by diagnosing their hidden structure, choosing the right metric and algorithm, and *proving* it with cross-validation.",
        ],
        prereqs="All of Lessons 00–16",
        hook=("A master carpenter doesn't own one tool; she owns twenty, and (more importantly) she *knows which one to pick up*. A beginner hammers every problem because the hammer is the tool they know best. "
              "You now own thirteen supervised-learning tools. This final lesson is about **judgement**: looking at a new dataset for thirty seconds and saying *\"this smells linear and wide: try Lasso\"* or "
              "*\"rare positives with curved regions: forest, PR-AUC, and watch the threshold.\"*\n\n"
              "The theorem that justifies this whole course is slightly depressing and wonderfully liberating: **there is no universally best algorithm.** "
              "The best you can do is match the algorithm's assumptions to the data's structure, and *verify with cross-validation*."),
        roadmap=["No Free Lunch: proved by experiment", "The dataset fingerprint", "The linearity gap", "Flowcharts and the master comparison table",
                 "Mystery datasets A, B, C", "Cheat-sheet of the whole course", "Problems & final reflection"])

    # ---------------------------------------------------------- 1 NFL
    nb.section("🍽️", "1. No Free Lunch: proved by experiment",
               "The **No Free Lunch theorem** (Wolpert, 1996) says that, averaged over *all possible* problems, every learning algorithm performs equally. "
               "Practically: an algorithm is good only to the extent that its **assumptions** (inductive bias) match the problem. "
               "Let's see it. Five classification datasets with different structure, six algorithms, identical 5-fold stratified CV:")
    nb.table([
        ["Breast cancer", "569 × 30", "nearly linearly separable, correlated features"],
        ["Moons", "500 × 2", "low-dimensional, curved boundary, noisy"],
        ["Checkerboard (XOR)", "800 × 2", "pure feature *interaction*: no linear signal at all"],
        ["Wide data", "150 × 300", "only 10 of 300 features carry signal (n ≪ p)"],
        ["Loans", "3 000 × 8", "log-odds linear in engineered ratios"],
    ], ["Dataset", "Shape", "Hidden structure"])
    nb.code(r'''def checkerboard(n=800, seed=0):
    r = np.random.default_rng(seed); X = r.uniform(-1, 1, (n, 2)); y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    flip = r.random(n) < 0.05; y[flip] = 1 - y[flip]; return X, y

def loan_xy():
    d = pd.read_csv(data_path("loan_approval.csv"))
    d["dti"] = (d["monthly_debt_kes"] + d["loan_amount_kes"] / d["loan_term_months"] * 1.15) / d["monthly_income_kes"]; d["lti"] = d["loan_amount_kes"] / (d["monthly_income_kes"] * 12)
    return d[["credit_score", "credit_history_years", "dti", "lti", "has_collateral", "age", "dependents", "existing_loans"]].fillna(650).values, d["approved"].values

bc = load_breast_cancer()
datasets = {"cancer": (bc.data, 1 - bc.target), "moons": make_moons(500, noise=0.3, random_state=0), "checkerboard (XOR)": checkerboard(),
            "wide (n=150, p=300)": make_classification(150, 300, n_informative=10, n_redundant=0, class_sep=1.5, random_state=1), "loans": loan_xy()}
models = {"LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
          "Tree": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, random_state=0),
          "Forest": lambda: RandomForestClassifier(200, random_state=0, n_jobs=-1),
          "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(11)),
          "NaiveBayes": lambda: GaussianNB(),
          "SVM-RBF": lambda: make_pipeline(StandardScaler(), SVC())}
skf = StratifiedKFold(5, shuffle=True, random_state=0)
res = pd.DataFrame({dn: {mn: cross_val_score(mk(), X, y, cv=skf, scoring="accuracy").mean() for mn, mk in models.items()} for dn, (X, y) in datasets.items()}).T
display(res.round(3))''')
    nb.code(r'''ranks = res.rank(axis=1, ascending=False).astype(int)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
im = axes[0].imshow(res.values, cmap="YlGnBu", aspect="auto", vmin=0.4, vmax=1); axes[0].set_title("CV accuracy")
im2 = axes[1].imshow(ranks.values, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=6); axes[1].set_title("Rank on each dataset (1 = best)")
for ax, M, fmt in [(axes[0], res.values, "{:.2f}"), (axes[1], ranks.values, "{}")]:
    ax.set_xticks(range(len(res.columns))); ax.set_xticklabels(res.columns, rotation=30); ax.set_yticks(range(len(res.index))); ax.set_yticklabels(res.index); ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=9)
plt.tight_layout(); plt.show()
print("Winner on each dataset:", res.idxmax(axis=1).to_dict())''')
    nb.md("""**Read the rank map: it is the entire lesson in one picture:**

- **Logistic regression** is the champion on cancer and loans (linear structure) and nearly the *worst* on moons and on the XOR checkerboard, where it scores worse than a coin flip: *no* linear boundary can solve XOR.
- **KNN** wins the low-dimensional curved problem (moons) but is the *worst* on wide data (300 noisy dimensions poison distances).
- **Random Forest** shines when structure is non-linear or when many features are irrelevant (XOR, wide) and is solid everywhere.
- **Naive Bayes** is quick but its independence assumption fails on XOR (the label depends on *both* features jointly).
- Three different algorithms win across five datasets: **no algorithm wins everywhere.**

*(Caveat: the 'wide' result for logistic regression uses default regularisation; a tuned L1/L2 penalty would improve it considerably; a reminder that a model's rank also depends on how well you tuned it.)*""")

    # ---------------------------------------------------------- 2 fingerprint
    nb.section("🔎", "2. The dataset fingerprint",
               "Before choosing, **look**. These questions narrow the field in seconds:")
    nb.table([
        ["**Task**", "Number (regression) or category (classification)?", "Metrics, algorithms available"],
        ["**Size n**", "Hundreds? Millions?", "n < 1k: simple models, Naive Bayes, regularised linear, SVM. n > 100k: linear/SGD, forests, boosting; avoid kernel SVM & KNN"],
        ["**Dimensionality p** (and p/n)", "Few features or thousands? p > n?", "p ≳ n: regularised linear models, linear SVM, NB; avoid KNN & deep trees"],
        ["**Linearity**", "Does a straight line/plane explain most of it?", "Linear → linear/logistic; strong non-linearity or interactions → forest, SVM-RBF, boosting"],
        ["**Noise level**", "How much irreducible error?", "Noisy → simpler/regularised models and bagging; avoid fully grown trees, 1-NN"],
        ["**Class balance**", "Rare positives?", "Use PR-AUC/recall, class weights, threshold tuning; accuracy is useless"],
        ["**Feature types**", "Numeric, categorical, text, missing values?", "Trees tolerate raw scales & mixed types; linear/KNN/SVM need scaling + encoding"],
        ["**Scale differences & outliers**", "Wild ranges? Extreme values?", "Scale for distance/gradient models; trees and robust losses (SVR, MAE) cope with outliers"],
        ["**Constraints**", "Need explanations? Latency? Retrain often? Memory?", "Explainability → linear/tree; latency → linear/tree/NB; retrain often → NB/linear; memory → avoid KNN/big forests"],
    ], ["Question", "Ask", "Implication"])
    nb.code(r'''Xl, yl = loan_xy()
print(f"n = {len(yl)}, p = {Xl.shape[1]}, p/n = {Xl.shape[1] / len(yl):.4f}")
print(f"positive share = {yl.mean():.2f}  (balanced enough)  |  missing values: {np.isnan(Xl).sum()}")
C = np.corrcoef(Xl, rowvar=False); np.fill_diagonal(C, 0); print(f"strongest feature-feature |correlation| = {np.abs(C).max():.2f}")
print("feature std range:", Xl.std(axis=0).min().round(2), "to", Xl.std(axis=0).max().round(1), " -> scaling needed for linear/KNN/SVM")''')

    # ---------------------------------------------------------- 3 linearity gap
    nb.section("📏", "3. The linearity gap: how non-linear is my problem?",
               "A quick diagnostic that reuses everything you've learned: compare a **well-regularised linear model** with a **flexible model** (a random forest) under the same cross-validation.\n\n"
               r"$$\text{gap}=\text{score}_{\text{forest}}-\text{score}_{\text{linear}}$$" "\n\n"
               "- **gap ≈ 0 or negative** → the structure is (nearly) linear; prefer the simple, explainable model.\n"
               "- **large positive gap** → non-linearity/interactions matter; invest in forests, SVM-RBF, boosting or feature engineering.\n\n"
               "(R² for regression, ROC-AUC for classification, so the numbers are comparable within a task.)")
    nb.code(r'''def linearity_gap(X, y, task):
    """Return (gap, linear_score, forest_score): forest minus regularised-linear, 5-fold CV."""
    X = np.asarray(X, float)
    if task == "regression":
        lin = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 15))); rf = RandomForestRegressor(100, min_samples_leaf=3, random_state=0, n_jobs=-1)
        cv, scoring = KFold(5, shuffle=True, random_state=0), "r2"
    else:
        lin = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3)); rf = RandomForestClassifier(100, min_samples_leaf=3, random_state=0, n_jobs=-1)
        cv, scoring = StratifiedKFold(5, shuffle=True, random_state=0), "roc_auc"
    s_lin = cross_val_score(lin, X, y, cv=cv, scoring=scoring).mean(); s_rf = cross_val_score(rf, X, y, cv=cv, scoring=scoring).mean()
    return s_rf - s_lin, s_lin, s_rf

rows = []
for name in ["cancer", "moons", "checkerboard (XOR)", "wide (n=150, p=300)", "loans"]:
    X_, y_ = datasets[name]; g, sl, sf = linearity_gap(X_, y_, "classification"); rows.append({"dataset": name, "linear AUC": sl, "forest AUC": sf, "gap": g})
display(pd.DataFrame(rows).set_index("dataset").round(3))''')
    nb.md("The gap flags XOR (huge: linear AUC ≈ 0.5), and tells us cancer and loans are 'linear problems' (gap ≈ 0 or negative). "
          "It costs two cross-validations: cheap insurance before you invest in a complex model.")

    # ---------------------------------------------------------- 4 flowcharts
    nb.section("🧭", "4. Flowcharts and the master comparison table",
               "**Regression**")
    nb.md("""```
START ── Is there a strong, simple (linear/log-linear) relationship?  (linearity gap ≈ 0)
 │
 ├─ YES ── p small, features understood? ──► Multiple Linear Regression (log target if multiplicative)
 │          p large / correlated / p ≳ n ───► Ridge / Lasso / ElasticNet
 │          one smooth curve, one feature ──► Polynomial regression (+ Ridge)
 │
 └─ NO (gap clearly > 0) ── n small–medium, smooth, noisy ──► SVR (RBF): scale X and y, tune C/γ/ε
                          ── n medium–large, mixed types ───► Random Forest (strong default) / gradient boosting
                          ── need rules/explanations ───────► Decision Tree (shallow) + a forest to benchmark
                          ── few features, local structure ─► KNN regression (scaled)
Always: baseline first · log-transform skewed targets · CV · check residuals by segment
```""")
    nb.md("**Classification**")
    nb.md("""```
START ── Rare positives? ──► Fix the METRIC first (PR-AUC, recall at precision X, expected cost) + class weights + threshold
 │
 ├─ Text / counts / tiny data / need instant training ────────────► Naive Bayes (Multinomial) or linear SVM / logistic regression
 ├─ Need probabilities + explanations (credit, medical) ──────────► Logistic Regression (check calibration)
 ├─ Linear gap ≈ 0 ──────────────────────────────────────────────► Logistic Regression (L1/L2 for many features)
 ├─ Non-linear / interactions, mixed types, n ≥ ~1k ─────────────► Random Forest (then boosting)
 ├─ Small–medium n, smooth non-linear, high-dim ─────────────────► SVM (RBF): scale, tune C/γ
 ├─ Few relevant features, local structure (digits, comps) ──────► KNN (scaled, odd k)
 └─ Must show human-readable rules ──────────────────────────────► Decision Tree (pruned): benchmark against a forest
Always: baseline first · stratified CV · several metrics · threshold from costs · check calibration & errors by segment (and fairness)
```""")
    nb.md("### The master comparison table (all thirteen algorithms)")
    nb.table([
        ["Simple Linear Regression", "R", "O(n)", "no", "no", "★★★★★", "sensitive", "linear", "one feature; explaining a trend"],
        ["Multiple Linear Regression", "R", "O(np²+p³)", "helps (GD/penalties)", "no", "★★★★★", "sensitive", "linear", "interpretable baseline; additive effects"],
        ["Polynomial Regression", "R", "O(nT²+T³), T=C(p+d,d)", "yes", "no", "★★★★", "sensitive", "bad", "few features, smooth curves, interactions"],
        ["Decision Tree", "R, C", "O(pn log n·depth)", "no", "partly (NaN ok)", "★★★★★", "robust (X)", "none (flat)", "rules, thresholds, interactions"],
        ["Random Forest", "R, C", "O(B·m·n log n)", "no", "partly", "★★", "robust", "none (flat)", "strong default for tabular data"],
        ["Support Vector Regression / SVM", "R, C", "O(n²–n³)", "**yes**", "no", "★ (kernel)", "SVR robust", "poor (RBF)", "small/medium, smooth, high-dim"],
        ["KNN", "R, C", "O(1) fit; O(np) predict", "**yes**", "no", "★★★★ (comps)", "robust-ish", "none (flat)", "few relevant features, local structure"],
        ["Logistic Regression", "C", "O(np) per iter", "helps", "no", "★★★★★", "sensitive", "linear log-odds", "probabilities + explanations; default"],
        ["Naive Bayes", "C", "O(np)", "no (Gaussian: helps)", "no", "★★★★", "moderate", "n/a", "text, tiny data, streaming"],
    ], ["Algorithm", "Task", "Training cost", "Needs scaling", "Handles missing", "Interpretability", "Outliers", "Extrapolation", "Use for"])
    nb.callout("tip", "The table is a **shortlist generator**, not an oracle. Shortlist two or three candidates, run them through the same cross-validation harness (Lessons 08 and 16), and let the data speak. "
                      "A thoughtful practitioner spends five minutes on the shortlist and an hour on features, metrics and error analysis.")

    # ---------------------------------------------------------- 5 mystery
    nb.section("🕵️", "5. The mystery datasets",
               "Three datasets, **hidden structure**, no documentation. For each: compute the fingerprint, run the linearity gap, **choose a metric**, shortlist algorithms, test them with CV, and name the winner *and why*. "
               "(No peeking at `data/generate_datasets.py`: `make_mystery` gives the game away!)")
    nb.code(r'''A = pd.read_csv(data_path("mystery_A.csv")); B = pd.read_csv(data_path("mystery_B.csv")); C = pd.read_csv(data_path("mystery_C.csv"))
XA, yA = A.drop(columns="y").values, A["y"].values        # regression
XB, yB = B.drop(columns="y").values, B["y"].values        # regression
XC, yC = C.drop(columns="y").values, C["y"].values        # classification (y in {0, 1})
for nm, X_, y_, task in [("A", XA, yA, "regression"), ("B", XB, yB, "regression"), ("C", XC, yC, "classification")]:
    print(f"Mystery {nm}: {task:<14} n = {X_.shape[0]:>4}, p = {X_.shape[1]:>2}, p/n = {X_.shape[1] / X_.shape[0]:.3f}" + (f", positive share = {y_.mean():.3f}" if task == "classification" else ""))''')
    nb.md("**Your mission (problems 3–5):** diagnose A, B and C. Hints: one of them is *wide*, one is *highly non-linear*, and one hides an *accuracy trap*. "
          "The three problems below walk you through each; first, two tool-building problems.")

    # ---------------------------------------------------------- 6 cheat sheet
    nb.section("📚", "6. Cheat-sheet of the whole course", "")
    nb.table([
        ["Simple LR", "ŷ = β₀ + β₁x", "OLS closed form (Cov/Var)", "Slope, intercept", "Outliers, extrapolation"],
        ["Multiple LR", "ŷ = Xβ", "Normal equation / lstsq", "Scaling, VIF, Ridge/Lasso α", "Multicollinearity, non-linearity"],
        ["Polynomial", "linear in β on [1, x, x², …]", "Vandermonde + lstsq", "degree, Ridge α", "Overfit/extrapolate; C(p+d,d) terms"],
        ["Decision tree", "recursive splits; leaf = mean / majority", "greedy impurity reduction", "max_depth, min_samples_leaf, ccp_alpha", "Unstable, staircase, no extrapolation"],
        ["Random forest", "average of decorrelated trees", "bagging + feature subsets", "n_estimators, max_features, leaf size", "Less interpretable; timid probabilities"],
        ["SVR / SVM", "tube / max-margin on support vectors", "convex QP; kernels", "C, γ, ε, kernel", "O(n²–n³); scaling; tuning"],
        ["KNN", "average / vote of k nearest", "none (lazy)", "k, weights, metric", "Slow predict; curse of dimensionality"],
        ["Logistic reg.", "σ(w·x+b); linear log-odds", "max likelihood (Newton/GD)", "C, penalty, threshold", "Linear boundary unless features engineered"],
        ["Naive Bayes", "argmax P(y)ΠP(xⱼ|y)", "counting / means & variances", "α smoothing, var_smoothing", "Over-confident, independence assumption"],
    ], ["Algorithm", "Model", "How it is fitted", "Key dials", "Main weakness"])

    # ---------------------------------------------------------- problems
    nb.problem(
        "Build the fingerprint function",
        "Write `fingerprint(X, y, task)` returning a dict with: `n`, `p`, `p_over_n` (rounded to 3 decimals), `minority_share` (share of the rarer class **for classification**, else `None`, rounded to 3) and "
        "`max_abs_corr` (largest absolute off-diagonal feature correlation, rounded to 2). Compute `fpA`, `fpB`, `fpC` for the mystery datasets.",
        ["`X.shape` gives (n, p). `np.corrcoef(X, rowvar=False)` is the p×p correlation matrix; set its diagonal to 0 with `np.fill_diagonal` before taking `np.abs(...).max()`.",
         "Minority share for 0/1 labels: `min(y.mean(), 1 - y.mean())`.",
         "Return plain Python numbers (use `float(...)`/`int(...)`) so the dict prints nicely."],
        'def fingerprint(X, y, task):\n    pass\n\nfpA = fpB = fpC = None   # your code\n',
        'def fingerprint(X, y, task):\n    X = np.asarray(X, float); n, p = X.shape; Cm = np.corrcoef(X, rowvar=False); np.fill_diagonal(Cm, 0)\n'
        '    share = round(float(min(np.mean(y), 1 - np.mean(y))), 3) if task == "classification" else None\n    return {"n": int(n), "p": int(p), "p_over_n": round(p / n, 3), "minority_share": share, "max_abs_corr": round(float(np.abs(Cm).max()), 2)}\n\n'
        'fpA = fingerprint(XA, yA, "regression"); fpB = fingerprint(XB, yB, "regression"); fpC = fingerprint(XC, yC, "classification")\nprint(fpA); print(fpB); print(fpC)',
        'check("A is wide-ish: p/n = 0.4", lambda: fpA["n"] == 150 and fpA["p"] == 60 and abs(fpA["p_over_n"] - 0.4) < 1e-9)\ncheck("B is big and narrow: n = 3000, p = 8", lambda: fpB["n"] == 3000 and fpB["p"] == 8 and fpB["minority_share"] is None)\ncheck("C has rare positives (< 6%)", lambda: fpC["minority_share"] is not None and fpC["minority_share"] < 0.06)\ncheck("features are essentially uncorrelated in all three (< 0.3)", lambda: max(fpA["max_abs_corr"], fpB["max_abs_corr"], fpC["max_abs_corr"]) < 0.3)',
        explain="Thirty seconds of arithmetic already says: A has few rows relative to its features (regularise!), B has plenty of rows and few features (flexible models are affordable), C is a rare-event problem (metric trap!).")

    nb.problem(
        "A rule-based recommender",
        "Write `recommend(fp, gap, task)` returning a **string**. Rules:\n\n"
        "- regression: if `fp['p_over_n'] > 0.2` and `gap < 0.05` → `'Regularised linear (Lasso/Ridge)'`; elif `gap > 0.15` → `'Boosting / Random Forest / SVR'`; else `'Multiple linear regression (compare with a forest)'`\n"
        "- classification: if `fp['minority_share'] < 0.1`: start the string with `'PR-AUC + class weights/threshold; '` and then append `'Random Forest / boosting'` if `gap > 0.1` else `'Logistic regression (class_weight)'`; "
        "otherwise `'Random Forest / SVM-RBF / boosting'` if `gap > 0.1` else `'Logistic regression'`.\n\n"
        "Compute the gaps with the lesson's `linearity_gap`: `gapA, gapB, gapC` (use the first return value), and store `recA, recB, recC`.",
        ["`gapA = linearity_gap(XA, yA, 'regression')[0]` (A's gap is *negative*: the linear model beats the forest).",
         "Use plain `if / elif / else`; build the classification string by concatenation.",
         "Expected: A → regularised linear; B → boosting/forest/SVR; C → PR-AUC … + Random Forest / boosting."],
        'def recommend(fp, gap, task):\n    pass\n\ngapA = gapB = gapC = None\nrecA = recB = recC = None   # your code\n',
        'def recommend(fp, gap, task):\n    if task == "regression":\n        if fp["p_over_n"] > 0.2 and gap < 0.05: return "Regularised linear (Lasso/Ridge)"\n        if gap > 0.15: return "Boosting / Random Forest / SVR"\n        return "Multiple linear regression (compare with a forest)"\n'
        '    prefix = "PR-AUC + class weights/threshold; " if fp["minority_share"] < 0.1 else ""\n    if gap > 0.1: return prefix + ("Random Forest / boosting" if prefix else "Random Forest / SVM-RBF / boosting")\n    return prefix + ("Logistic regression (class_weight)" if prefix else "Logistic regression")\n\n'
        'gapA = linearity_gap(XA, yA, "regression")[0]; gapB = linearity_gap(XB, yB, "regression")[0]; gapC = linearity_gap(XC, yC, "classification")[0]\n'
        'recA = recommend(fpA, gapA, "regression"); recB = recommend(fpB, gapB, "regression"); recC = recommend(fpC, gapC, "classification")\nprint(round(gapA, 2), recA); print(round(gapB, 2), recB); print(round(gapC, 2), recC)',
        'check("A: the forest LOSES to the linear model (gap < 0)", lambda: gapA < 0)\ncheck("B: strongly non-linear (gap > 0.3)", lambda: gapB > 0.3)\ncheck("C: non-linear classification (gap > 0.1)", lambda: gapC > 0.1)\ncheck("recommendations", lambda: recA.startswith("Regularised") and recB.startswith("Boosting") and recC.startswith("PR-AUC") and recC.endswith("Random Forest / boosting"))',
        explain="Three tiny numbers (p/n, gap, minority share) already steer the shortlist correctly. Real recommenders are of course richer, but the habit of *measuring before choosing* is the point.",
        level="⭐⭐")

    nb.problem(
        "Mystery A: wide and (probably) linear",
        "Evaluate these regressors with `KFold(5, shuffle=True, random_state=0)` and R²: `Linear` (scaled `LinearRegression`), `Ridge` (scaled `RidgeCV(alphas=np.logspace(-2, 3, 20))`), `Lasso` (scaled `LassoCV(cv=5)`), "
        "`Tree` (`DecisionTreeRegressor(min_samples_leaf=5, random_state=0)`), `Forest` (`RandomForestRegressor(200, min_samples_leaf=2, random_state=0, n_jobs=-1)`), `SVR` (scaled `SVR(C=10)`), `KNN` (scaled `KNeighborsRegressor(7)`). "
        "Store the scores in a dict `scoresA`, the winner's name in `winnerA`, and `lasso_vs_linear` = Lasso's R² minus plain Linear's.",
        ["Build a dict of `name -> estimator`, then `cross_val_score(est, XA, yA, cv=kf, scoring='r2').mean()`.",
         "`winnerA = max(scoresA, key=scoresA.get)`.",
         "With 60 features and only 150 rows, plain least squares overfits a little; Lasso zeroes the 55 useless coefficients."],
        'kf = KFold(5, shuffle=True, random_state=0)\nscoresA = {}\nwinnerA = lasso_vs_linear = None   # your code\n',
        'kf = KFold(5, shuffle=True, random_state=0)\ncands = {"Linear": make_pipeline(StandardScaler(), LinearRegression()), "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), "Lasso": make_pipeline(StandardScaler(), LassoCV(cv=5)),\n'
        '         "Tree": DecisionTreeRegressor(min_samples_leaf=5, random_state=0), "Forest": RandomForestRegressor(200, min_samples_leaf=2, random_state=0, n_jobs=-1), "SVR": make_pipeline(StandardScaler(), SVR(C=10)), "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(7))}\n'
        'scoresA = {k: cross_val_score(m, XA, yA, cv=kf, scoring="r2").mean() for k, m in cands.items()}\nwinnerA = max(scoresA, key=scoresA.get); lasso_vs_linear = scoresA["Lasso"] - scoresA["Linear"]\nprint({k: round(v, 3) for k, v in scoresA.items()}, winnerA)',
        'check("seven models evaluated", lambda: len(scoresA) == 7)\ncheck("the winner is a (regularised) linear model", lambda: winnerA in ("Lasso", "Ridge", "Linear"))\ncheck("Lasso (sparse) beats plain least squares", lambda: lasso_vs_linear > 0.01)\ncheck("trees and KNN are far behind (R² < 0.5)", lambda: scoresA["Tree"] < 0.5 and scoresA["KNN"] < 0.5)',
        explain="The signal is a sparse linear combination of 5 of 60 features. Lasso finds exactly that structure; flexible models waste their capacity on 55 noise features with only 150 rows. Flexibility is *not* a virtue when n is small and the truth is simple.",
        level="⭐⭐")

    nb.problem(
        "Mystery B: big, narrow, non-linear",
        "Evaluate on `XB, yB` (same `kf`): `Linear` (scaled), `Tree` (`min_samples_leaf=5`), `Forest` (200 trees, `min_samples_leaf=2`), `SVR` (scaled, `C=10`), `KNN` (scaled, k = 7) and `HGB` (`HistGradientBoostingRegressor(random_state=0)`). "
        "Store `scoresB`, `winnerB`, and `linear_r2` (the plain linear model's R²).",
        ["Same pattern as Mystery A; this time include `HistGradientBoostingRegressor`.",
         "Expect a linear R² near 0.15.",
         "The winner should be a model that can capture both a smooth interaction (sin of a product) and a threshold: boosting or an RBF-SVR."],
        'scoresB = {}\nwinnerB = linear_r2 = None   # your code\n',
        'candsB = {"Linear": make_pipeline(StandardScaler(), LinearRegression()), "Tree": DecisionTreeRegressor(min_samples_leaf=5, random_state=0), "Forest": RandomForestRegressor(200, min_samples_leaf=2, random_state=0, n_jobs=-1),\n'
        '          "SVR": make_pipeline(StandardScaler(), SVR(C=10)), "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(7)), "HGB": HistGradientBoostingRegressor(random_state=0)}\n'
        'scoresB = {k: cross_val_score(m, XB, yB, cv=kf, scoring="r2").mean() for k, m in candsB.items()}\nwinnerB = max(scoresB, key=scoresB.get); linear_r2 = scoresB["Linear"]\nprint({k: round(v, 3) for k, v in scoresB.items()}, winnerB)',
        'check("six models evaluated", lambda: len(scoresB) == 6)\ncheck("linear regression is hopeless (R² < 0.25)", lambda: linear_r2 < 0.25)\ncheck("the winner is boosting or the RBF-SVR", lambda: winnerB in ("HGB", "SVR"))\ncheck("the winner explains > 80% of the variance", lambda: scoresB[winnerB] > 0.80)',
        explain="With 3 000 rows and only 8 features, flexible models are affordable and necessary: the target mixes a sin(x₁x₂) interaction, a square and a threshold-gated term: structure that no linear model can express. Boosting builds it stage by stage; the RBF kernel captures the smooth parts directly.",
        level="⭐⭐")

    nb.problem(
        "Mystery C: the accuracy trap",
        "On `XC, yC` with `StratifiedKFold(5, shuffle=True, random_state=0)`:\n"
        "1. `acc_logreg` and `recall_logreg`: cross-validated **accuracy** and **recall** of a scaled `LogisticRegression(max_iter=3000)`.\n"
        "2. `pr_scores`: dict of cross-validated **PR-AUC** (`scoring='average_precision'`) for `LogReg`, `Tree` (`max_depth=8, min_samples_leaf=5`), `Forest` (200 trees, `min_samples_leaf=2`), `KNN` (scaled, k = 11), `NB` (GaussianNB), `SVM` (scaled `SVC()`); and `best_C` = the winner.\n"
        "3. `trap` = `True` if the logistic regression's accuracy exceeds 0.9 **and** its recall is below 0.1.",
        ["`cross_val_score(model, XC, yC, cv=skf, scoring='accuracy')`, `scoring='recall'`, `scoring='average_precision'`.",
         "Positives are ≈ 4.5 % of the data: always predicting 'negative' is ≈ 95.5 % accurate.",
         "The positives live in *curved* regions (a ring and a product term), so the linear model predicts almost no positives."],
        'skf = StratifiedKFold(5, shuffle=True, random_state=0)\nacc_logreg = recall_logreg = pr_scores = best_C = trap = None   # your code\n',
        'skf = StratifiedKFold(5, shuffle=True, random_state=0); lr_c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))\n'
        'acc_logreg = cross_val_score(lr_c, XC, yC, cv=skf, scoring="accuracy").mean(); recall_logreg = cross_val_score(lr_c, XC, yC, cv=skf, scoring="recall").mean()\n'
        'cands_c = {"LogReg": lr_c, "Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, random_state=0), "Forest": RandomForestClassifier(200, min_samples_leaf=2, random_state=0, n_jobs=-1),\n'
        '           "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(11)), "NB": GaussianNB(), "SVM": make_pipeline(StandardScaler(), SVC())}\n'
        'pr_scores = {k: cross_val_score(m, XC, yC, cv=skf, scoring="average_precision").mean() for k, m in cands_c.items()}\nbest_C = max(pr_scores, key=pr_scores.get); trap = bool(acc_logreg > 0.9 and recall_logreg < 0.1)\nprint(round(acc_logreg, 3), round(recall_logreg, 3), {k: round(v, 3) for k, v in pr_scores.items()}, best_C, trap)',
        'check("logistic regression looks great on accuracy (> 0.9)...", lambda: acc_logreg > 0.9)\ncheck("...but finds almost no positives (recall < 0.1)", lambda: recall_logreg < 0.1)\ncheck("the trap is real", lambda: trap is True)\ncheck("the winner by PR-AUC is the random forest", lambda: best_C == "Forest")\ncheck("the forest has at least 5x logistic regression\'s PR-AUC", lambda: pr_scores["Forest"] > 5 * pr_scores["LogReg"])',
        explain="Accuracy 95 %+ with recall ≈ 0: the model predicts 'negative' for nearly everyone. Judging by PR-AUC exposes it: the forest (which can carve out the ring and the interaction region) finds the rare positives while the linear model cannot. Metric first, algorithm second.",
        level="⭐⭐⭐")

    nb.problem(
        "Mine the No-Free-Lunch table",
        "Using the lesson's `res` (datasets × models accuracy table) and `ranks`: compute `avg_rank` (Series: mean rank of each model across the 5 datasets, sorted ascending), `best_overall` (the lowest average rank), "
        "`n_distinct_winners` (how many different models win at least one dataset) and `logreg_rank_range` = (worst rank − best rank) of `'LogReg'` across datasets.",
        ["`ranks.mean(axis=0).sort_values()`.",
         "`res.idxmax(axis=1).nunique()` counts distinct winners.",
         "`ranks['LogReg'].max() - ranks['LogReg'].min()`."],
        'avg_rank = None\nbest_overall = n_distinct_winners = logreg_rank_range = None   # your code\n',
        'avg_rank = ranks.mean(axis=0).sort_values(); best_overall = avg_rank.index[0]; n_distinct_winners = int(res.idxmax(axis=1).nunique()); logreg_rank_range = int(ranks["LogReg"].max() - ranks["LogReg"].min())\nprint(avg_rank.round(2).to_dict(), best_overall, n_distinct_winners, logreg_rank_range)',
        'check("no single model wins every dataset (>= 3 distinct winners)", lambda: n_distinct_winners >= 3)\ncheck("logistic regression swings by at least 4 ranks", lambda: logreg_rank_range >= 4)\ncheck("the best average rank is held by a flexible, robust model (Forest or SVM-RBF)", lambda: best_overall in ("Forest", "SVM-RBF"))',
        explain="Averaged over these five problems the flexible, robust models rank best, but logistic regression swings from 1st to 5th. A strong general-purpose model (forest/SVM) is a safe default, but the winner on *your* data is whichever model's assumptions match it.",
        level="⭐⭐")

    nb.quiz([
        ("What does the No Free Lunch theorem imply for a practitioner?",
         "No algorithm is best on all problems; you must match the algorithm's assumptions to the problem's structure and verify empirically (cross-validation): there's no substitute for experiments."),
        ("Linearity gap = +0.45 on a regression problem. What do you do next?",
         "Don't rely on a linear model: try a forest, boosting or SVR (and consider engineered interactions). The large gap says non-linear structure carries a lot of the signal."),
        ("You have 80 rows and 400 features. Which of these would you avoid and why: Lasso, KNN, deep random forest?",
         "KNN (distances are meaningless in 400 noisy dimensions) and arguably a deep forest (overfits tiny n); Lasso is designed for p ≫ n."),
        ("A model has 97 % accuracy on a problem with 3 % positives. What is the very first thing you check?",
         "The baseline (always-negative = 97 %) and then recall/precision/PR-AUC: the model may be doing nothing."),
        ("Name three non-accuracy factors that might make you pick a *less accurate* model.",
         "Interpretability/regulation, prediction latency, training/retraining cost, memory footprint, calibration quality, robustness/fairness, maintainability."),
    ])

    nb.md("""---
## 🪞 Final reflection (not graded: but the most valuable part)

Write a short paragraph for each question *in your own words*, as if explaining to a colleague:

1. For **Mystery A**, B and C: what clues in the fingerprint predicted the winner *before* you ran the models?
2. Which algorithm surprised you most in this course, and why?
3. Take a dataset from your own life or work. What is its fingerprint, which two algorithms would you shortlist, and what *metric* would you optimise?
4. What is the most important mistake you now know how to avoid (leakage? accuracy on imbalanced data? unscaled features? tuning on the test set?).

**You are ready for the four projects.** Each one asks you to do exactly what this lesson practised: frame the problem, read the fingerprint, run a fair tournament, interrogate the errors, and tell a convincing story.""")

    nb.footer(["No Free Lunch: every algorithm bakes in assumptions; the best one is the one whose assumptions match your data, and you confirm it with cross-validation.",
               "Read the dataset fingerprint first: task, n, p (and p/n), linearity (linearity gap), noise, class balance, feature types, constraints.",
               "Choose the metric before the algorithm (the accuracy trap!); use baselines, fair CV and error analysis.",
               "Wide + linear → regularised linear; big + non-linear → forest/boosting/SVR; rare positives → PR-AUC, weights, thresholds.",
               "Shortlist 2–3 candidates from the flowchart/table, test them with the same harness, and weigh accuracy against interpretability, speed and cost."],
              "Project 1; House Price Prediction", "../projects/P1_House_Price_Prediction.ipynb")
    return nb
