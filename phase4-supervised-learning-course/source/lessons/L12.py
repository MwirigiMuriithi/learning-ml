from nbkit import Lesson


def make():
    nb = Lesson("12_Random_Forest_Classification", 12, "Random Forest Classification",
                "A jury of decorrelated trees: majority wisdom, probabilities and a free validation score",
                120, "🟡 Intermediate",
                setup_extra="""from sklearn.datasets import make_moons, make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, precision_score, confusion_matrix
from scipy.stats import binom""")
    nb.header(
        objectives=[
            "Explain why a majority vote of **independent, better-than-chance** voters is so powerful (Condorcet's jury theorem) and why correlation limits it.",
            "Distinguish **hard** (majority) voting from **soft** (probability-averaging) voting, and know which scikit-learn uses.",
            "Implement a Random Forest classifier **from scratch** with bootstrap sampling, feature subsets and an OOB score.",
            "Tune `n_estimators`, `max_features`, `min_samples_leaf`, `max_depth` and `class_weight`.",
            "Read probability quality with calibration curves and interpret a forest via permutation importance.",
            "Handle class imbalance with `class_weight` and PR-AUC.",
            "Know when a forest beats logistic regression, and when it doesn't.",
        ],
        prereqs="Lessons 05 and 11",
        hook=("Imagine a jury of 101 people deciding guilty/not guilty, where each juror is right just **60 %** of the time: barely better than a coin flip. "
              "If their errors were independent, the **majority verdict is right about 98 % of the time**. That's the 18th-century *Condorcet jury theorem*, and it is the whole secret of Random Forests: "
              "take many mediocre-but-different voters and let them vote.\n\n"
              "The catch: the voters must be **different**. 101 jurors who all read the same newspaper make one decision 101 times. "
              "Much of the engineering in a Random Forest is about keeping the trees *diverse*."),
        roadmap=["Condorcet and the power (and limit) of voting", "Hard vs soft voting", "From scratch: the forest classifier with OOB",
                 "Seeing the forest smooth the boundary", "Tuning: how many trees, how many features", "Probability quality: calibration",
                 "Loan approvals: forest vs tree vs logistic regression", "Imbalanced data & class weights", "Interpreting with permutation importance",
                 "Cheat-sheet, worked examples, problems, quiz"])

    # ---------------------------------------------------------- 1 condorcet
    nb.section("🗳️", "1. Condorcet and the power (and limit) of voting",
               "If $n$ voters (odd) are each correct with probability $p>0.5$, **independently**, the chance that the majority is correct is a binomial tail:\n\n"
               r"$$P(\text{majority correct})=\sum_{k>n/2}\binom nk p^k(1-p)^{n-k}$$" "\n\n"
               "Let's compute it, and then see what happens when the voters are partly *copies* of each other (a fraction ρ of the time everybody follows the same leader).")
    nb.code(r'''ns = np.arange(1, 202, 2)
fig, ax = plt.subplots(figsize=(8, 4.2))
for p in [0.55, 0.6, 0.7]:
    ax.plot(ns, binom.sf(ns // 2, ns, p), lw=2.2, label=f"independent voters, each p = {p}")
# correlated jury: with probability rho all voters copy ONE shared opinion (correct w.p. p); otherwise they vote independently
p, rho = 0.6, 0.4
ax.plot(ns, rho * p + (1 - rho) * binom.sf(ns // 2, ns, p), "k--", lw=2.2, label=f"p = 0.6 but ρ = {rho} correlated")
ax.axhline(p, color="gray", ls=":"); ax.set_xlabel("number of voters (trees)"); ax.set_ylabel("P(majority is correct)"); ax.legend(fontsize=8); ax.set_title("More voters help; until correlation sets a ceiling"); plt.show()
print(f"101 independent voters with p=0.6 -> majority is right {binom.sf(50, 101, 0.6):.1%} of the time")
print(f"...but with correlation ρ=0.4 the ceiling is only {rho * p + (1 - rho):.1%}")''')
    nb.callout("key", "**Ensemble accuracy rests on two legs:** each voter must beat chance (we use deep trees: low bias), and the voters' errors must be as *independent* as possible. "
                      "Random Forests create diversity with (1) **bootstrap samples**: each tree sees a different ~63 % of the rows: and (2) **random feature subsets** at every split: "
                      "each split sees only `max_features` randomly chosen columns (default √p for classification).")

    # ---------------------------------------------------------- 2 voting
    nb.section("⚖️", "2. Hard vs soft voting",
               "How do we combine the trees' answers?\n\n"
               "- **Hard voting:** each tree votes for a class; the majority wins.\n"
               "- **Soft voting:** average the trees' **predicted probabilities** (each leaf's class proportions) and take the class with the highest average. "
               "This uses more information (a tree that is 90 % sure counts more than one that is 51 % sure). **scikit-learn's forest uses soft voting.**")
    nb.code(r'''tree_probs = np.array([0.90, 0.45, 0.45])                    # P(class 1) from three trees
hard = (tree_probs >= 0.5).mean() > 0.5                       # votes: 1, 0, 0  -> class 0 wins 2-1
soft = tree_probs.mean() >= 0.5                               # average = 0.60 -> class 1
print(f"hard vote -> class {int(hard)}   |   soft vote (mean prob = {tree_probs.mean():.2f}) -> class {int(soft)}")''')
    nb.md("Here the two rules *disagree*: one very confident tree outweighs two lukewarm ones under soft voting. In practice soft voting is usually slightly better and also produces probabilities.")

    # ---------------------------------------------------------- 3 scratch
    nb.section("🔧", "3. Build it from scratch",
               "The same recipe as Lesson 05, but with `DecisionTreeClassifier` base trees, class probabilities, soft *or* hard voting, and an OOB accuracy. "
               "One subtlety: with few classes a bootstrap sample can (rarely) miss a class entirely, so each tree's probability columns must be mapped back onto the full class list.")
    nb.code(r'''class RandomForestClassifierScratch:
    """Random forest classifier: bootstrap rows + random feature subsets + (soft or hard) voting + OOB estimate."""

    def __init__(self, n_estimators=100, max_features="sqrt", min_samples_leaf=1, max_depth=None, voting="soft", random_state=0):
        self.n_estimators, self.max_features, self.min_samples_leaf = n_estimators, max_features, min_samples_leaf
        self.max_depth, self.voting, self.random_state = max_depth, voting, random_state

    def _tree_proba(self, tree, X):                        # map a tree's class columns back onto the full class list
        P = np.zeros((len(X), self.K_)); P[:, tree.classes_] = tree.predict_proba(X); return P

    def fit(self, X, y):
        X = np.asarray(X, float); self.classes_, yi = np.unique(y, return_inverse=True); self.K_ = len(self.classes_); n = len(yi)
        rng = np.random.default_rng(self.random_state); self.trees_ = []
        oob_sum, oob_cnt = np.zeros((n, self.K_)), np.zeros(n)
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, n)                                         # 1) bootstrap sample
            tree = DecisionTreeClassifier(max_features=self.max_features, min_samples_leaf=self.min_samples_leaf,   # 2) random features per split
                                          max_depth=self.max_depth, random_state=int(rng.integers(1 << 30))).fit(X[idx], yi[idx])
            self.trees_.append(tree)
            oob = np.setdiff1d(np.arange(n), idx)                               # rows this tree never saw
            oob_sum[oob] += self._tree_proba(tree, X[oob]); oob_cnt[oob] += 1
        seen = oob_cnt > 0
        self.oob_decision_function_ = np.where(seen[:, None], oob_sum / np.maximum(oob_cnt, 1)[:, None], np.nan)
        self.oob_score_ = np.mean(self.oob_decision_function_[seen].argmax(axis=1) == yi[seen])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        if self.voting == "soft":
            return np.mean([self._tree_proba(t, X) for t in self.trees_], axis=0)         # average probabilities
        votes = np.array([self._tree_proba(t, X).argmax(axis=1) for t in self.trees_])    # each tree's hard vote
        return np.stack([(votes == k).mean(axis=0) for k in range(self.K_)], axis=1)

    def predict(self, X): return self.classes_[self.predict_proba(X).argmax(axis=1)]
    def score(self, X, y): return np.mean(self.predict(X) == np.asarray(y))


Xm, ym = make_moons(600, noise=0.3, random_state=0)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, stratify=ym, random_state=RANDOM_STATE)
single = DecisionTreeClassifier(random_state=0).fit(Xm_tr, ym_tr)
mine = RandomForestClassifierScratch(150, min_samples_leaf=2).fit(Xm_tr, ym_tr)
ref = RandomForestClassifier(150, min_samples_leaf=2, oob_score=True, random_state=0).fit(Xm_tr, ym_tr)
print(f"single tree      : test acc = {single.score(Xm_te, ym_te):.3f}")
print(f"scratch forest   : test acc = {mine.score(Xm_te, ym_te):.3f}   OOB acc = {mine.oob_score_:.3f}")
print(f"sklearn forest   : test acc = {ref.score(Xm_te, ym_te):.3f}   OOB acc = {ref.oob_score_:.3f}")
print(f"hard-vote variant: test acc = {RandomForestClassifierScratch(150, min_samples_leaf=2, voting='hard').fit(Xm_tr, ym_tr).score(Xm_te, ym_te):.3f}")''')
    nb.callout("warn", "On this particular split the hard-vote variant happens to score a little *higher* than soft voting. With only 180 test points, is ±0.6 samples… i.e. a difference of two or three points is **within noise**. "
                       "Don't read a verdict into a single small test set: compare methods with cross-validation before claiming a winner.")
    nb.md("Our 35-line forest performs like scikit-learn's, beats the single tree, and its OOB accuracy (computed without touching the test set) is a faithful estimate of test accuracy.")

    # ---------------------------------------------------------- 4 boundary
    nb.section("🗺️", "4. Seeing the forest smooth the boundary",
               "A single unrestricted tree carves jagged rectangles that chase every noisy point. Averaging 150 of them dissolves the jaggedness:")
    nb.code(r'''fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
plot_decision_boundary(single, Xm_tr, ym_tr, ax=axes[0], title=f"Single tree\ntrain {single.score(Xm_tr, ym_tr):.2f} | test {single.score(Xm_te, ym_te):.2f}")
plot_decision_boundary(ref, Xm_tr, ym_tr, ax=axes[1], title=f"Random forest (150 trees)\ntrain {ref.score(Xm_tr, ym_tr):.2f} | test {ref.score(Xm_te, ym_te):.2f}")
lrm = make_pipeline(StandardScaler(), LogisticRegression()).fit(Xm_tr, ym_tr); plot_decision_boundary(lrm, Xm_tr, ym_tr, ax=axes[2], title=f"Logistic regression (a line)\ntrain {lrm.score(Xm_tr, ym_tr):.2f} | test {lrm.score(Xm_te, ym_te):.2f}")
plt.tight_layout(); plt.show()''')

    # ---------------------------------------------------------- 5 tuning
    nb.section("🎛️", "5. Tuning: how many trees, how many features?",
               "We switch to the **loan-approval** problem (same engineered features as Lessons 10–11; `interest_rate_offered` still excluded).")
    nb.code(r'''loans = pd.read_csv(data_path("loan_approval.csv"))
loans["dti"] = (loans["monthly_debt_kes"] + loans["loan_amount_kes"] / loans["loan_term_months"] * 1.15) / loans["monthly_income_kes"]
loans["lti"] = loans["loan_amount_kes"] / (loans["monthly_income_kes"] * 12)
loans["log_income"] = np.log(loans["monthly_income_kes"]); loans["log_savings"] = np.log1p(loans["savings_balance_kes"])
num = ["credit_score", "credit_history_years", "dti", "lti", "log_income", "log_savings", "has_collateral", "age", "dependents", "existing_loans"]
X = pd.get_dummies(loans[num + ["employment_type", "property_area", "education"]], drop_first=True, dtype=float); X = X.fillna(X.median())
y = loans["approved"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
print("train", Xtr.shape, "| test", Xte.shape, "| approval rate", round(y.mean(), 3))

# reference models
tree_ref = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=0).fit(Xtr, ytr)
lr_ref = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
auc = lambda m: roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
print(f"reference test AUC: tuned tree {auc(tree_ref):.3f} | logistic regression {auc(lr_ref):.3f}")''')
    nb.md("**More trees never hurt.** Using `warm_start`, grow one forest tree-batch by tree-batch and watch the OOB error flatten:")
    nb.code(r'''import warnings
rf_w = RandomForestClassifier(warm_start=True, oob_score=True, max_features="sqrt", min_samples_leaf=3, random_state=0, n_jobs=-1)
sizes = [5, 10, 25, 50, 100, 200, 400]; oob_err, test_err = [], []
for n_trees in sizes:
    rf_w.set_params(n_estimators=n_trees)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); rf_w.fit(Xtr, ytr)
    oob_err.append(1 - rf_w.oob_score_); test_err.append(1 - rf_w.score(Xte, yte))
plt.plot(sizes, oob_err, "o-", label="OOB error"); plt.plot(sizes, test_err, "s-", label="test error"); plt.xscale("log"); plt.xlabel("number of trees"); plt.ylabel("error rate"); plt.legend(); plt.title("Error falls, then plateaus (it does not overfit with more trees)"); plt.show()''')
    nb.md("**`max_features`**: how many columns each split may look at (the decorrelation dial), and **`min_samples_leaf`**: leaf size (the smoothness dial):")
    nb.code(r'''rows = []
for mf in ["sqrt", 0.3, 0.5, 0.8, 1.0]:
    for leaf in [1, 5, 20]:
        rf = RandomForestClassifier(150, max_features=mf, min_samples_leaf=leaf, oob_score=True, random_state=0, n_jobs=-1).fit(Xtr, ytr)
        rows.append({"max_features": str(mf), "min_samples_leaf": leaf, "OOB acc": rf.oob_score_, "test AUC": auc(rf)})
grid = pd.DataFrame(rows); display(grid.pivot(index="max_features", columns="min_samples_leaf", values="test AUC").round(3))''')
    nb.md("Look at the spread: **every setting lands within about 0.015 AUC of the others**: forests are famously forgiving. The mild trend is that *smaller* `max_features` (√p or 0.3) does slightly better than using all features, "
          "because it decorrelates the trees (the lesson of section 1), while leaf size barely matters here. Tune if you need the last fraction of a point, but do not expect miracles from tuning.")

    # ---------------------------------------------------------- 6 calibration
    nb.section("📏", "6. Probability quality: calibration",
               "A forest's probability is the average of its trees' leaf proportions. Are those probabilities *honest*? Compare reliability curves (Lesson 09) for three models:")
    nb.code(r'''rf_best = RandomForestClassifier(300, max_features="sqrt", min_samples_leaf=10, oob_score=True, random_state=0, n_jobs=-1).fit(Xtr, ytr)
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for name, m in [("Logistic regression", lr_ref), ("Tuned tree", tree_ref), ("Random forest", rf_best)]:
    pt, pp = calibration_curve(yte, m.predict_proba(Xte)[:, 1], n_bins=8, strategy="quantile"); ax.plot(pp, pt, "o-", label=name)
ax.plot([0, 1], [0, 1], "k--", label="perfect"); ax.set_xlabel("mean predicted P(approve)"); ax.set_ylabel("observed approval rate"); ax.legend(); ax.set_title("Reliability diagram"); plt.show()''')
    nb.md("Logistic regression hugs the diagonal (it optimises log loss directly). The forest is **under-confident at the extremes**: where it says 14 % the true rate is only ≈ 2 %, where it says 76–88 % the truth is higher: averaging pulls probabilities toward the middle, since a leaf is rarely unanimous across *all* trees. The single tree's probabilities are coarse and erratic (look at its last bin collapsing). "
          "When honest probabilities matter (pricing, risk), calibrate with `CalibratedClassifierCV` (Platt/isotonic) or prefer logistic regression.")

    # ---------------------------------------------------------- 7 loans compare
    nb.section("🏦", "7. Loan approvals: forest vs tree vs logistic regression",
               "")
    nb.code(r'''rows = []
for name, m in [("Logistic regression", lr_ref), ("Tuned decision tree", tree_ref), ("Random forest (leaf=10)", rf_best)]:
    p = m.predict_proba(Xte)[:, 1]; rows.append({"model": name, "accuracy": accuracy_score(yte, p >= .5), "ROC-AUC": roc_auc_score(yte, p), "avg precision": average_precision_score(yte, p)})
display(pd.DataFrame(rows).set_index("model").round(3))
display(pd.DataFrame(confusion_matrix(yte, rf_best.predict(Xte)), index=["actually rejected", "actually approved"], columns=["pred rejected", "pred approved"]))
print(f"OOB accuracy of the forest = {rf_best.oob_score_:.3f}  vs  test accuracy = {rf_best.score(Xte, yte):.3f}")''')
    nb.md("The forest clearly beats the single tree and gets close to logistic regression, which still wins narrowly: the loan decision is generated from (almost) linear log-odds in the engineered ratios: its home turf. "
          "That's a recurring theme: **if you can engineer the right features, a simple model is hard to beat; if you can't, forests find the structure for you.** "
          "Section 8 shows a problem where the forest wins by a mile.")

    # ---------------------------------------------------------- 8 imbalanced
    nb.section("🚨", "8. Imbalanced data and class weights",
               "Back to the 2 % positive 'fraud' data of Lesson 09: now with a forest. `class_weight='balanced'` re-weights classes inversely to their frequency, "
               "so each rare positive counts ~50× more when trees choose splits; `'balanced_subsample'` recomputes the weights within every bootstrap sample.")
    nb.code(r'''Xi, yi = make_classification(n_samples=5000, n_features=10, n_informative=5, weights=[0.98, 0.02], flip_y=0.01, class_sep=1.2, random_state=1)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.3, stratify=yi, random_state=RANDOM_STATE)
rows = []
lr_i = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xi_tr, yi_tr)
rows.append({"model": "Logistic regression", "recall": recall_score(yi_te, lr_i.predict(Xi_te)), "precision": precision_score(yi_te, lr_i.predict(Xi_te), zero_division=0), "PR-AUC": average_precision_score(yi_te, lr_i.predict_proba(Xi_te)[:, 1])})
for cw in [None, "balanced", "balanced_subsample"]:
    rf = RandomForestClassifier(300, min_samples_leaf=2, class_weight=cw, random_state=0, n_jobs=-1).fit(Xi_tr, yi_tr)
    rows.append({"model": f"Random forest, class_weight={cw}", "recall": recall_score(yi_te, rf.predict(Xi_te)), "precision": precision_score(yi_te, rf.predict(Xi_te), zero_division=0), "PR-AUC": average_precision_score(yi_te, rf.predict_proba(Xi_te)[:, 1])})
display(pd.DataFrame(rows).set_index("model").round(3))''')
    nb.md("The forest's PR-AUC dwarfs logistic regression's here: this synthetic problem has clustered, non-linear structure that a linear boundary can't express. "
          "`class_weight='balanced'` lifts recall at a small precision cost; but the *real* lever for imbalance is usually the **threshold** (Lesson 09): forests output good rankings, you pick the operating point.")

    # ---------------------------------------------------------- 9 interpretation
    nb.section("🔎", "9. Interpreting the forest: permutation importance",
               "As in Lesson 05, shuffle one feature on the **held-out** set and measure the loss in AUC. We add a pure-noise column as a sanity probe.")
    nb.code(r'''rng = np.random.default_rng(0)
Xtr_n = Xtr.copy(); Xtr_n["RANDOM_NOISE"] = rng.normal(size=len(Xtr)); Xte_n = Xte.copy(); Xte_n["RANDOM_NOISE"] = rng.normal(size=len(Xte))
rf_n = RandomForestClassifier(300, max_features="sqrt", min_samples_leaf=10, random_state=0, n_jobs=-1).fit(Xtr_n, ytr)
pi = permutation_importance(rf_n, Xte_n, yte, scoring="roc_auc", n_repeats=10, random_state=0, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=Xte_n.columns).sort_values()
colors = ["#dc2626" if i == "RANDOM_NOISE" else "#2563eb" for i in imp.index]
imp.tail(10).plot.barh(color=colors[-10:], figsize=(7, 4)); plt.xlabel("drop in test AUC when the feature is shuffled"); plt.title("Permutation importance (loan forest)"); plt.show()
print("noise column importance:", round(imp["RANDOM_NOISE"], 4), "| top feature:", imp.index[-1])''')

    # ---------------------------------------------------------- 10 profile
    nb.section("📋", "10. Cheat-sheet: the full profile of Random Forest Classification")
    nb.table([
        ["**Underlying concept**", "A committee of many deep, diverse decision trees; the class with the most (probability-weighted) votes wins."],
        ["**How it works**", "For each tree: bootstrap the rows, grow a tree choosing each split from a random subset of `max_features` columns. Predict by averaging the trees' class probabilities. OOB rows give a free validation estimate."],
        ["**Mathematical foundation**", "Condorcet jury theorem; variance of an average of correlated estimators ρσ²+(1−ρ)σ²/B; bootstrap (≈63 % in-bag); Gini/entropy splits; OOB estimation."],
        ["**Important parameters**", "`n_estimators`, `max_features` (√p default), `min_samples_leaf`, `max_depth`, `class_weight` (balanced / balanced_subsample), `bootstrap`, `max_samples`, `oob_score`, `n_jobs`, `random_state`."],
        ["**Advantages**", "Strong out-of-the-box accuracy; robust to noise and irrelevant features; no scaling; handles non-linearity and interactions; OOB score; importances; parallelisable; works for multi-class."],
        ["**Disadvantages**", "Less interpretable than a single tree or logistic regression; probabilities need calibration; large and slower than linear models; can lose to a well-specified linear model; weak on sparse high-dim text."],
        ["**Complexity**", "Training ≈ O(B · √p · n log n) (parallel across trees); prediction O(B · depth); memory O(B · nodes)."],
        ["**Use cases**", "Tabular classification with mixed feature types: fraud, churn, risk, diagnostics, quick strong baselines for almost any structured problem."],
        ["**Limitations**", "Importances can mislead with correlated features; no extrapolation; biased towards frequent classes without weights; many trees = bigger deployment footprint."],
    ], ["Question", "Answer"])

    # ---------------------------------------------------------- 11 worked
    nb.section("🧪", "11. Worked examples")
    nb.md(r"""### Worked Example 1: Majority of 3

Three independent classifiers, each correct with probability $p=0.7$. The majority is right if at least 2 are right:

$$P=\binom32(0.7)^2(0.3)+\binom33(0.7)^3=3(0.49)(0.3)+0.343=0.441+0.343=\mathbf{0.784}$$

A committee of just three mediocre voters beats each one (0.70) by 8 points. With 11 voters it is 0.92.""")
    nb.code(r'''print("P(majority of 3) =", round(binom.sf(1, 3, 0.7), 4), "| of 11 =", round(binom.sf(5, 11, 0.7), 4))''')
    nb.md(r"""### Worked Example 2: Soft voting with three trees

Tree probabilities for (approve, reject): Tree A (0.9, 0.1), Tree B (0.4, 0.6), Tree C (0.45, 0.55). Hard votes: approve, reject, reject → **reject** (2–1). Soft average: approve $=(0.9+0.4+0.45)/3=0.583$ → **approve**. Soft voting sides with the confident tree.""")

    # ---------------------------------------------------------- problems
    nb.problem(
        "Condorcet by the binomial",
        "Using `scipy.stats.binom` (already imported), compute the probability that a **majority** of `n` independent voters is correct, each with accuracy `p`:\n\n"
        "- `p_11` for n = 11, p = 0.6\n- `p_101` for n = 101, p = 0.6\n- `p_101_weak` for n = 101, p = 0.52\n- `n_needed` = the smallest **odd** n for which the majority (p = 0.6) is correct at least 90 % of the time.",
        ["Majority correct means *more than n/2* correct votes: `binom.sf(n // 2, n, p)` (survival function = P(X > n//2)).",
         "For `n_needed` loop over odd n = 1, 3, 5, … until the probability ≥ 0.9.",
         "Notice how much larger a crowd weak voters (p = 0.52) need."],
        'p_11 = p_101 = p_101_weak = n_needed = None   # your code\n',
        'p_11 = binom.sf(11 // 2, 11, 0.6); p_101 = binom.sf(101 // 2, 101, 0.6); p_101_weak = binom.sf(101 // 2, 101, 0.52)\nn_needed = next(n for n in range(1, 2000, 2) if binom.sf(n // 2, n, 0.6) >= 0.9)\nprint(round(p_11, 4), round(p_101, 4), round(p_101_weak, 4), n_needed)',
        'check("11 voters, p=0.6 -> about 0.753", lambda: abs(p_11 - 0.7535) < 1e-3)\ncheck("101 voters, p=0.6 -> about 0.98", lambda: p_101 > 0.97)\ncheck("101 voters, p=0.52 -> only about 0.66", lambda: 0.62 < p_101_weak < 0.70)\ncheck("about 41 voters are enough for 90%", lambda: n_needed == 41)',
        explain="Voters barely better than chance (p = 0.52) lift a crowd of 101 only to about 66 %: you would need roughly a thousand of them to reach 90 %. Individual competence matters as well as independence. At p = 0.6, just 41 independent voters already reach 90 %.")

    nb.problem(
        "Hard vs soft voting",
        "Given `probs`, an array of shape (5 trees, 4 samples) holding each tree's P(class 1) for 4 samples:\n```python\nprobs = np.array([[0.95, 0.55, 0.20, 0.51],\n                  [0.90, 0.55, 0.30, 0.51],\n                  [0.35, 0.60, 0.45, 0.52],\n                  [0.30, 0.52, 0.40, 0.02],\n                  [0.40, 0.51, 0.35, 0.05]])\n```\n"
        "Compute `soft_pred` (class 1 if the mean probability ≥ 0.5) and `hard_pred` (class 1 if **more than half** of the trees have probability ≥ 0.5) for each sample (arrays of 0/1). "
        "Store `disagree` = the indices of the samples where they differ.",
        ["`soft_pred = (probs.mean(axis=0) >= 0.5).astype(int)`.",
         "`votes = (probs >= 0.5)`; `hard_pred = (votes.mean(axis=0) > 0.5).astype(int)`.",
         "`disagree = np.where(soft_pred != hard_pred)[0]`. Sample 0 has two very confident trees; sample 3 has three barely-above-0.5 trees and two near-zero ones."],
        'probs = np.array([[0.95, 0.55, 0.20, 0.51], [0.90, 0.55, 0.30, 0.51], [0.35, 0.60, 0.45, 0.52], [0.30, 0.52, 0.40, 0.02], [0.40, 0.51, 0.35, 0.05]])\nsoft_pred = hard_pred = disagree = None   # your code\n',
        'probs = np.array([[0.95, 0.55, 0.20, 0.51], [0.90, 0.55, 0.30, 0.51], [0.35, 0.60, 0.45, 0.52], [0.30, 0.52, 0.40, 0.02], [0.40, 0.51, 0.35, 0.05]])\nsoft_pred = (probs.mean(axis=0) >= 0.5).astype(int)\nhard_pred = ((probs >= 0.5).mean(axis=0) > 0.5).astype(int)\ndisagree = np.where(soft_pred != hard_pred)[0]\nprint(soft_pred, hard_pred, disagree)',
        'check("soft predictions = [1, 1, 0, 0]", lambda: list(soft_pred) == [1, 1, 0, 0])\ncheck("hard predictions = [0, 1, 0, 1]", lambda: list(hard_pred) == [0, 1, 0, 1])\ncheck("the rules disagree on samples 0 and 3", lambda: list(disagree) == [0, 3])',
        explain="Sample 0: only 2 of 5 trees vote class 1, but those two are very confident (0.95, 0.90), so the *average* (0.58) says class 1: soft voting listens to confidence. Sample 3: 3 of 5 trees barely cross 0.5 while two trees say ≈ 0, so hard voting says class 1 but the mean (0.32) says class 0. Soft voting generally uses more information, which is why scikit-learn's forest averages probabilities.",
        level="⭐⭐")

    nb.problem(
        "Fit a forest and beat the tree",
        "Fit `RandomForestClassifier(300, max_features='sqrt', min_samples_leaf=10, oob_score=True, random_state=0, n_jobs=-1)` on the loan data `Xtr, ytr`. "
        "Store `oob_acc`, `test_acc`, `auc_rf` (test ROC-AUC) and `gain_over_tree = auc_rf - auc(tree_ref)` (the lesson's `auc` helper and `tree_ref`).",
        ["`.oob_score_` is the OOB *accuracy*.",
         "`auc(model)` is already defined in the lesson: `roc_auc_score(yte, model.predict_proba(Xte)[:, 1])`.",
         "The forest should beat the tuned tree by a clear margin."],
        'oob_acc = test_acc = auc_rf = gain_over_tree = None   # your code\n',
        'rf1 = RandomForestClassifier(300, max_features="sqrt", min_samples_leaf=10, oob_score=True, random_state=0, n_jobs=-1).fit(Xtr, ytr)\noob_acc = rf1.oob_score_; test_acc = rf1.score(Xte, yte); auc_rf = auc(rf1); gain_over_tree = auc_rf - auc(tree_ref)\nprint(round(oob_acc, 3), round(test_acc, 3), round(auc_rf, 3), round(gain_over_tree, 3))',
        'check("OOB accuracy tracks test accuracy (gap < 0.04)", lambda: abs(oob_acc - test_acc) < 0.04)\ncheck("forest AUC > 0.91", lambda: auc_rf > 0.91)\ncheck("forest beats the tuned tree by > 0.02 AUC", lambda: gain_over_tree > 0.02)',
        explain="Averaging 300 diverse trees cuts variance: the forest gains a few AUC points over even a well-tuned single tree, and its OOB score agrees with the held-out test accuracy.")

    nb.problem(
        "Tune with randomised search",
        "Use `RandomizedSearchCV` (`n_iter=8`, `cv=StratifiedKFold(3, shuffle=True, random_state=0)`, `scoring='roc_auc'`, `random_state=0`, `n_jobs=-1`) on `RandomForestClassifier(n_estimators=150, random_state=0)` over "
        "`max_features ∈ ['sqrt', 0.3, 0.5]`, `min_samples_leaf ∈ [1, 3, 5, 10, 20]`, `max_depth ∈ [None, 8, 12]`. Store `best_params` and the test AUC `tuned_auc`.",
        ["`RandomizedSearchCV(estimator, param_distributions, n_iter=8, cv=..., scoring='roc_auc', random_state=0, n_jobs=-1).fit(Xtr, ytr)`.",
         "`rs.best_params_` and `roc_auc_score(yte, rs.best_estimator_.predict_proba(Xte)[:, 1])`.",
         "Random search samples 8 of the 45 combinations: usually plenty."],
        'best_params = tuned_auc = None   # your code\n',
        'rs = RandomizedSearchCV(RandomForestClassifier(n_estimators=150, random_state=0), {"max_features": ["sqrt", 0.3, 0.5], "min_samples_leaf": [1, 3, 5, 10, 20], "max_depth": [None, 8, 12]},\n'
        '                        n_iter=8, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc", random_state=0, n_jobs=-1).fit(Xtr, ytr)\nbest_params = rs.best_params_; tuned_auc = roc_auc_score(yte, rs.best_estimator_.predict_proba(Xte)[:, 1])\nprint(best_params, round(tuned_auc, 3))',
        'check("three hyper-parameters searched", lambda: set(best_params) == {"max_features", "min_samples_leaf", "max_depth"})\ncheck("tuned test AUC > 0.91", lambda: tuned_auc > 0.91)',
        explain="Randomised search tests a random subset of the grid and is far cheaper than a full grid. Because forests are forgiving, many different settings score within a hair of each other: the gain from tuning is small but cheap.",
        level="⭐⭐")

    nb.problem(
        "Class weights on imbalanced data",
        "On the fraud data (`Xi_tr, yi_tr, Xi_te, yi_te`) fit two forests `RandomForestClassifier(300, min_samples_leaf=2, random_state=0, n_jobs=-1)`: one with `class_weight=None` and one with `class_weight='balanced'`. "
        "Store each one's **recall** at the default 0.5 threshold (`recall_plain`, `recall_bal`), and the PR-AUC of the plain forest `ap_plain` and of **logistic regression** (`lr_i` from the lesson) `ap_lr`.",
        ["`recall_score(yi_te, model.predict(Xi_te))`.",
         "`average_precision_score(yi_te, model.predict_proba(Xi_te)[:, 1])`.",
         "Expect balanced weights to raise recall, and the forest to beat logistic regression on PR-AUC."],
        'recall_plain = recall_bal = ap_plain = ap_lr = None   # your code\n',
        'mk = lambda cw: RandomForestClassifier(300, min_samples_leaf=2, class_weight=cw, random_state=0, n_jobs=-1).fit(Xi_tr, yi_tr)\nrf_p, rf_b = mk(None), mk("balanced")\nrecall_plain = recall_score(yi_te, rf_p.predict(Xi_te)); recall_bal = recall_score(yi_te, rf_b.predict(Xi_te))\n'
        'ap_plain = average_precision_score(yi_te, rf_p.predict_proba(Xi_te)[:, 1]); ap_lr = average_precision_score(yi_te, lr_i.predict_proba(Xi_te)[:, 1])\nprint(recall_plain, recall_bal, ap_plain, ap_lr)',
        'check("balanced class weights raise recall", lambda: recall_bal > recall_plain)\ncheck("the forest beats logistic regression on PR-AUC here", lambda: ap_plain > ap_lr + 0.1)',
        explain="Weighting rare positives makes trees split to isolate them (more recall). The bigger story: on this non-linear imbalanced problem the forest's ranking quality (PR-AUC) is far better than a linear model's.",
        level="⭐⭐")

    nb.problem(
        "Top drivers by permutation importance",
        "Using the lesson's `rf_n`, `Xte_n`, `yte`: compute permutation importance with `scoring='roc_auc'`, `n_repeats=10`, `random_state=1`. Store the **top three feature names** in `top3` (most important first), "
        "`noise_imp` = the importance of `RANDOM_NOISE`, and `n_useless` = the number of features whose mean importance is **below 0.001**.",
        ["`pi = permutation_importance(rf_n, Xte_n, yte, scoring='roc_auc', n_repeats=10, random_state=1, n_jobs=-1)`.",
         "Series: `pd.Series(pi.importances_mean, index=Xte_n.columns).sort_values(ascending=False)`.",
         "`n_useless = int((imp2 < 0.001).sum())`."],
        'top3 = noise_imp = n_useless = None   # your code\n',
        'pi2 = permutation_importance(rf_n, Xte_n, yte, scoring="roc_auc", n_repeats=10, random_state=1, n_jobs=-1)\nimp2 = pd.Series(pi2.importances_mean, index=Xte_n.columns).sort_values(ascending=False)\ntop3 = imp2.index[:3].tolist(); noise_imp = imp2["RANDOM_NOISE"]; n_useless = int((imp2 < 0.001).sum())\nprint(top3, round(noise_imp, 4), n_useless)',
        'check("credit_score and dti are among the top 3", lambda: "credit_score" in top3 and "dti" in top3)\ncheck("the noise column is (almost) useless", lambda: noise_imp < 0.005)\ncheck("several features are essentially irrelevant", lambda: n_useless >= 3)',
        explain="Credit score and debt-to-income dominate: exactly the drivers logistic regression found in Lesson 10. Two very different models agreeing on the drivers is strong evidence the signal is real.",
        level="⭐⭐")

    nb.quiz([
        ("101 jurors, each right 60 % of the time but all copying the same newspaper. Is the majority 98 % reliable?",
         "No. Condorcet's result needs independent errors. Heavily correlated voters make one decision many times; the ensemble's accuracy is capped near the individual accuracy."),
        ("Why does a forest use random feature subsets *in addition to* bootstrap samples?",
         "Bootstrapping alone yields similar trees when a few features dominate (they'd all split on them first). Random subsets force diversity, lowering tree-to-tree correlation ρ."),
        ("What does `class_weight='balanced'` do to the trees' splits?",
         "It scales each sample's weight inversely to its class frequency, so rare-class mistakes cost much more in the impurity calculation and splits try harder to isolate the rare class."),
        ("The forest's probabilities are never exactly 0 or 1 even for easy cases. Why, and what to do if you need calibrated probabilities?",
         "Probabilities are averages over many trees, so unanimous 0/1 across all trees is rare (and leaves with min_samples_leaf > 1 are mixed). Calibrate with CalibratedClassifierCV or use logistic regression."),
        ("When would you pick logistic regression over a forest?",
         "When the relationship is (nearly) linear in good features, when you need honest probabilities and explainable coefficients, with very small data, or for strict latency/size limits."),
    ])

    nb.footer(["A Random Forest classifier = bootstrap + random feature subsets + soft-voting over many deep trees.",
               "Condorcet: independent better-than-chance voters become excellent as a group; correlation sets the ceiling, so diversity is the design goal.",
               "More trees never hurt; tune `min_samples_leaf`, `max_features`; use `class_weight` + threshold/PR-AUC for imbalance.",
               "OOB accuracy is a free validation score; permutation importance (held-out) tells you what the model relies on.",
               "Forests are a superb default, but probabilities need calibration and a well-specified linear model can still win."],
              "K-Nearest Neighbors Classification", "13_KNN_Classification.ipynb")
    return nb
