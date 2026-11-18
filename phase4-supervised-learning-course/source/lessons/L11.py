from nbkit import Lesson


def make():
    nb = Lesson("11_Decision_Tree_Classification", 11, "Decision Tree Classification",
                "Entropy, Gini and the art of asking the question that clears up the most confusion",
                130, "🟡 Intermediate",
                setup_extra="""from sklearn.datasets import load_iris, load_breast_cancer, make_blobs
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, validation_curve
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix""")
    nb.header(
        objectives=[
            "Explain how a classification tree chooses splits using **Gini impurity** and **entropy / information gain**.",
            "Compute impurity and information gain **by hand** on the classic 'Play Tennis' dataset.",
            "Implement a classification tree **from scratch** (numeric thresholds, Gini or entropy) and match scikit-learn.",
            "Visualise axis-aligned decision regions and explain why trees struggle with diagonal boundaries.",
            "Diagnose overfitting; control it with `max_depth`, `min_samples_leaf` and cost-complexity pruning.",
            "Extract human-readable rules and the decision path of a single prediction.",
            "Compare trees with logistic regression honestly on loan data.",
        ],
        prereqs="Lessons 04, 09, 10",
        hook=("A bank's loan officer doesn't compute log-odds. She works through a checklist: *\"Credit score above 650? Yes. Debt-to-income under 40 %? No. Collateral? Yes → approve.\"* "
              "She learned *which question to ask first* from years of experience: the question that splits good applicants from bad ones most cleanly.\n\n"
              "A **decision tree classifier** learns that checklist from data. The central puzzle: **how do you measure \"cleanly\"?** "
              "The answer (*impurity*) is a beautiful idea borrowed from information theory, and once you grasp it you understand not only trees but also Random Forests and gradient boosting."),
        roadmap=["The idea & impurity measures", "Information gain by hand (Play Tennis)", "ID3 in a dozen lines", "From scratch: numeric CART classifier",
                 "Decision regions & axis-aligned limits", "Overfitting & pruning on loan approvals", "Reading rules & decision paths",
                 "Tree vs logistic regression", "Cheat-sheet, worked examples, problems, quiz"])

    # ---------------------------------------------------------- 1 impurity
    nb.section("🧠", "1. The idea: ask the question that reduces confusion the most",
               "A node is **pure** if all its examples share one class (nothing left to decide) and **impure** if the classes are mixed. "
               "A good question splits an impure node into purer children. We need a number for 'how mixed is this node?'. With class proportions $p_1,\\dots,p_K$:")
    nb.md(r"""| Impurity | Formula | Value for 50/50 | Value for pure |
|---|---|---|---|
| **Gini** | $G=1-\sum_k p_k^2$ | 0.5 | 0 |
| **Entropy** | $H=-\sum_k p_k\log_2 p_k$ | 1 bit | 0 |
| Misclassification error | $1-\max_k p_k$ | 0.5 | 0 |

- **Gini** = the probability that two items drawn at random from the node have *different* classes.
- **Entropy** = the average number of yes/no questions (bits) needed to identify the class of a random item: the *surprise* in the node.

A split is judged by the **weighted average impurity of its children**, and the **gain** is the reduction:

$$\text{Gain}=I(\text{parent})-\sum_{c\in\text{children}}\frac{n_c}{n}\,I(c)$$

With entropy this gain is the classic **information gain**. The tree picks the split with the largest gain, then recurses. A leaf predicts the **majority class** (and $\hat P(k)$ = the proportion of class $k$ in the leaf).""")
    nb.code(r'''p = np.linspace(0.001, 0.999, 400)
gini = 2 * p * (1 - p); ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)); mis = 1 - np.maximum(p, 1 - p)
plt.figure(figsize=(7, 4)); plt.plot(p, gini, lw=2.4, label="Gini"); plt.plot(p, ent, lw=2.4, label="Entropy (bits)"); plt.plot(p, mis, "--", lw=2, label="Misclassification")
plt.xlabel("proportion of class 1 in the node"); plt.ylabel("impurity"); plt.legend(); plt.title("All three are 0 when pure and peak at 50/50"); plt.show()''')
    nb.md("Gini and entropy are almost the same shape (entropy is just a bit more 'curvy'), so they almost always produce very similar trees. Gini is slightly faster (no logarithm) and is scikit-learn's default. "
          "Misclassification error isn't used for *growing* trees because it's too flat: it can't tell that a 90/10 → 70/30 + 100/0 split is progress.")

    # ---------------------------------------------------------- 2 tennis
    nb.section("🎾", "2. Information gain by hand: the Play Tennis dataset",
               "The most famous teaching dataset in machine learning: 14 days, four weather attributes and whether we played tennis.")
    nb.code(r'''tennis = pd.DataFrame([
    ["Sunny", "Hot", "High", "Weak", "No"], ["Sunny", "Hot", "High", "Strong", "No"], ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"], ["Rain", "Cool", "Normal", "Weak", "Yes"], ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"], ["Sunny", "Mild", "High", "Weak", "No"], ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"], ["Sunny", "Mild", "Normal", "Strong", "Yes"], ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"], ["Rain", "Mild", "High", "Strong", "No"]], columns=["Outlook", "Temp", "Humidity", "Wind", "Play"])
display(tennis.T)
print(tennis["Play"].value_counts().to_dict())''')
    nb.md(r"""**Step 1: impurity of the root** (9 Yes, 5 No): $p_{yes}=9/14=0.643,\ p_{no}=0.357$

$$H=-0.643\log_2 0.643-0.357\log_2 0.357=0.410+0.530=\mathbf{0.940\text{ bits}}$$

**Step 2: try splitting on Outlook.** Sunny (2 Yes, 3 No): $H=0.971$; Overcast (4 Yes, 0 No): $H=0$; Rain (3 Yes, 2 No): $H=0.971$.
Weighted: $\frac5{14}(0.971)+\frac4{14}(0)+\frac5{14}(0.971)=0.693$. **Gain(Outlook) = 0.940 − 0.693 = 0.247.**

Repeat for every attribute and compare:""")
    nb.code(r'''def entropy_from_counts(counts):
    counts = np.asarray(counts, float); p = counts[counts > 0] / counts.sum()
    return -np.sum(p * np.log2(p))

def info_gain_table(df, target="Play"):
    base = entropy_from_counts(df[target].value_counts())
    rows = []
    for col in [c for c in df.columns if c != target]:
        weighted = sum(len(g) / len(df) * entropy_from_counts(g[target].value_counts()) for _, g in df.groupby(col))
        rows.append({"attribute": col, "weighted child entropy": weighted, "information gain": base - weighted})
    return base, pd.DataFrame(rows).set_index("attribute").sort_values("information gain", ascending=False)

base_H, gains = info_gain_table(tennis)
print(f"Root entropy = {base_H:.4f} bits"); display(gains.round(4))''')
    nb.md("**Outlook wins** (0.247): it is the best first question. Notice what makes it good: one of its branches (*Overcast*) is **perfectly pure** (always Yes).")

    # ---------------------------------------------------------- 3 ID3
    nb.section("🌳", "3. ID3 in a dozen lines",
               "The recursive recipe (ID3) for categorical attributes: *if the node is pure or we've run out of attributes → leaf; otherwise split on the attribute with the highest gain and recurse on each value.* "
               "Let's grow the full Play-Tennis tree.")
    nb.code(r'''def id3(df, features, target="Play", depth=0):
    classes = df[target].value_counts()
    if len(classes) == 1 or not features:                          # pure node, or no attributes left -> leaf
        print("    " * depth + f"-> {classes.idxmax()}  { {k: int(v) for k, v in classes.items()} }"); return
    best = max(features, key=lambda f: info_gain_table(df[features + [target]])[1].loc[f, "information gain"])
    for value, sub in df.groupby(best):
        print("    " * depth + f"{best} = {value}:")
        id3(sub, [f for f in features if f != best], target, depth + 1)

id3(tennis, ["Outlook", "Temp", "Humidity", "Wind"])''')
    nb.md("Read it as a flowchart: *If Outlook is Overcast → play. If Sunny → depends on Humidity (High → no, Normal → yes). If Rain → depends on Wind (Weak → yes, Strong → no).* "
          "Temperature turned out not to matter at all! The tree performed **automatic feature selection**.")

    # ---------------------------------------------------------- 4 scratch
    nb.section("🔧", "4. Build it from scratch: a CART classifier for numeric features",
               "Real data has numeric features, so we search **thresholds** exactly as in Lesson 04: but now track **class counts**. After sorting a feature, a cumulative sum of one-hot labels "
               "gives the class counts left of *every* possible threshold at once, so each candidate split's impurity costs $O(K)$.")
    nb.code(r'''class DecisionTreeClassifierScratch:
    """CART classifier (Gini or entropy), numeric thresholds, binary splits."""

    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, min_samples_split=2):
        self.criterion, self.max_depth, self.min_samples_leaf, self.min_samples_split = criterion, max_depth, min_samples_leaf, min_samples_split

    def _impurity(self, counts):                                  # counts: (..., K)  -> impurity per row
        n = counts.sum(axis=-1, keepdims=True); p = counts / np.maximum(n, 1)
        if self.criterion == "gini":
            return 1 - (p ** 2).sum(axis=-1)
        return -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(axis=-1)

    def fit(self, X, y):
        X = np.asarray(X, float); self.classes_, yi = np.unique(y, return_inverse=True); self.K_ = len(self.classes_)
        self.n_features_ = X.shape[1]; self.importances_ = np.zeros(self.n_features_)
        self.tree_ = self._grow(X, yi, 0)
        if self.importances_.sum() > 0: self.importances_ /= self.importances_.sum()
        return self

    def _best_split(self, X, yi):
        n, p = X.shape; m = self.min_samples_leaf; onehot = np.eye(self.K_)[yi]; best = (None, None, np.inf)
        for j in range(p):
            order = np.argsort(X[:, j], kind="stable"); xs = X[order, j]
            cum = np.cumsum(onehot[order], axis=0); total = cum[-1]
            i = np.arange(m - 1, n - m)
            if len(i) == 0: continue
            left = cum[i]; right = total - left; nl = (i + 1.0); nr = n - nl
            weighted = (nl * self._impurity(left) + nr * self._impurity(right)) / n          # weighted child impurity
            weighted = np.where(xs[i] < xs[i + 1], weighted, np.inf)
            k = np.argmin(weighted)
            if weighted[k] < best[2]: best = (j, (xs[i[k]] + xs[i[k] + 1]) / 2, weighted[k])
        return best

    def _grow(self, X, yi, depth):
        counts = np.bincount(yi, minlength=self.K_).astype(float); imp = float(self._impurity(counts))
        node = {"counts": counts, "n": len(yi), "impurity": imp}
        if (self.max_depth is not None and depth >= self.max_depth) or len(yi) < self.min_samples_split or imp < 1e-12:
            return node
        j, t, w = self._best_split(X, yi)
        if j is None or imp - w < 1e-12: return node                       # no split improves purity
        self.importances_[j] += len(yi) * (imp - w)
        mask = X[:, j] <= t
        node.update(feature=j, threshold=t, left=self._grow(X[mask], yi[mask], depth + 1), right=self._grow(X[~mask], yi[~mask], depth + 1))
        return node

    def _leaf(self, x):
        node = self.tree_
        while "feature" in node: node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node

    def predict_proba(self, X):
        return np.array([self._leaf(x)["counts"] / self._leaf(x)["n"] for x in np.asarray(X, float)])
    def predict(self, X): return self.classes_[self.predict_proba(X).argmax(axis=1)]
    def score(self, X, y): return np.mean(self.predict(X) == np.asarray(y))

    def show(self, names=None, node=None, indent=""):
        node = self.tree_ if node is None else node
        if "feature" not in node:
            print(f"{indent}-> class {self.classes_[node['counts'].argmax()]}  (counts {node['counts'].astype(int).tolist()})"); return
        nm = names[node["feature"]] if names else f"x{node['feature']}"
        print(f"{indent}if {nm} <= {node['threshold']:.3f}:"); self.show(names, node["left"], indent + "    ")
        print(f"{indent}else:"); self.show(names, node["right"], indent + "    ")


# Sanity check on iris, then verify against sklearn on breast cancer
iris = load_iris()
small = DecisionTreeClassifierScratch(max_depth=2).fit(iris.data[:, 2:], iris.target); small.show(["petal length", "petal width"])''')
    nb.code(r'''bc = load_breast_cancer(); Xb, yb = bc.data, 1 - bc.target                                  # 1 = malignant
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=RANDOM_STATE)
rows = []
for crit in ["gini", "entropy"]:
    mine = DecisionTreeClassifierScratch(criterion=crit, max_depth=4, min_samples_leaf=5).fit(Xb_tr, yb_tr)
    for seed in range(5):                      # sklearn breaks exact ties between equally good splits RANDOMLY (random_state)
        ref = DecisionTreeClassifier(criterion=crit, max_depth=4, min_samples_leaf=5, random_state=seed).fit(Xb_tr, yb_tr)
        rows.append({"criterion": crit, "sklearn seed": seed, "prediction agreement": np.mean(mine.predict(Xb_te) == ref.predict(Xb_te)),
                     "max |proba diff|": np.abs(mine.predict_proba(Xb_te) - ref.predict_proba(Xb_te)).max(),
                     "same tree?": bool(np.abs(mine.predict_proba(Xb_te) - ref.predict_proba(Xb_te)).max() < 1e-12)})
cmp_ = pd.DataFrame(rows); display(cmp_.round(4))
print(f"Our tree is IDENTICAL to scikit-learn's in {cmp_['same tree?'].sum()} of {len(cmp_)} runs; average prediction agreement = {cmp_['prediction agreement'].mean():.1%}")''')
    nb.callout("key", "For most seeds our from-scratch tree is **identical** to scikit-learn's (same predictions, same probabilities). The few differences have one cause: breast-cancer features like *radius*, *perimeter* and *area* are almost perfectly correlated, "
                      "so two candidate splits can tie **exactly**: scikit-learn then picks among them using its random generator, while we always take the first. Different seeds give sklearn itself different (equally good) trees! "
                      "The algorithm is the same: **for every feature and threshold, minimise the children's weighted impurity; recurse until pure or a stopping rule fires.** (This tie-sensitivity is also a hint of the instability of single trees.)")

    # ---------------------------------------------------------- 5 regions
    nb.section("🗺️", "5. Decision regions, and the axis-aligned limitation",
               "A tree can only ask questions like 'is feature *j* ≤ *t*?', so its decision regions are **rectangles**. Look at iris (petal length & width) at increasing depth:")
    nb.code(r'''Xp, yp = iris.data[:, 2:], iris.target
fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, d in zip(axes, [1, 2, 3, None]):
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xp, yp)
    plot_decision_boundary(t, Xp, yp, ax=ax, title=f"max_depth={d}   train acc={t.score(Xp, yp):.2f}", feature_names=["petal length", "petal width"])
plt.tight_layout(); plt.show()''')
    nb.md("Now a **diagonal** boundary; $y=1$ when $x_1+x_2>1$. A logistic regression draws one straight line; a tree needs a staircase of rectangles:")
    nb.code(r'''rng = np.random.default_rng(0); Xd = rng.uniform(0, 1, (400, 2)); yd = (Xd.sum(axis=1) + rng.normal(0, 0.08, 400) > 1).astype(int)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
lrd = LogisticRegression().fit(Xd, yd); plot_decision_boundary(lrd, Xd, yd, ax=axes[0], title=f"Logistic regression: one line (acc {lrd.score(Xd, yd):.2f})")
for ax, d in zip(axes[1:], [3, 8]):
    t = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xd, yd); plot_decision_boundary(t, Xd, yd, ax=ax, title=f"Tree depth {d}: staircase (acc {t.score(Xd, yd):.2f})")
plt.tight_layout(); plt.show()''')
    nb.md("This is the price of axis-aligned splits. Remedies: engineer a feature like $x_1+x_2$, use an ensemble (smoother stairs), or pick a model with oblique boundaries (logistic regression, SVM).")

    # ---------------------------------------------------------- 6 overfit / loans
    nb.section("🏦", "6. Overfitting and pruning on loan approvals",
               "Same loan problem as Lesson 10 (features include the engineered `dti` and `lti` ratios; the `interest_rate_offered` column stays out). Trees need no scaling, so we just one-hot the categories.")
    nb.code(r'''loans = pd.read_csv(data_path("loan_approval.csv"))
loans["emi"] = loans["loan_amount_kes"] / loans["loan_term_months"] * 1.15
loans["dti"] = (loans["monthly_debt_kes"] + loans["emi"]) / loans["monthly_income_kes"]; loans["lti"] = loans["loan_amount_kes"] / (loans["monthly_income_kes"] * 12)
loans["log_income"] = np.log(loans["monthly_income_kes"]); loans["log_savings"] = np.log1p(loans["savings_balance_kes"])
num = ["credit_score", "credit_history_years", "dti", "lti", "log_income", "log_savings", "has_collateral", "age", "dependents", "existing_loans"]
X = pd.get_dummies(loans[num + ["employment_type", "property_area", "education"]], drop_first=True, dtype=float)
X["credit_score"] = X["credit_score"].fillna(X["credit_score"].median()); X["log_savings"] = X["log_savings"].fillna(X["log_savings"].median())
y = loans["approved"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
skf = StratifiedKFold(5, shuffle=True, random_state=0)

depths = [1, 2, 3, 4, 5, 6, 8, 10, 14, None]
tr, va = validation_curve(DecisionTreeClassifier(random_state=0), Xtr, ytr, param_name="max_depth", param_range=[d if d else 40 for d in depths], cv=skf, scoring="roc_auc")
plt.plot(range(len(depths)), tr.mean(1), "o-", label="train AUC"); plt.plot(range(len(depths)), va.mean(1), "s-", label="validation AUC")
plt.xticks(range(len(depths)), [str(d) for d in depths]); plt.xlabel("max_depth"); plt.ylabel("ROC-AUC"); plt.legend(); plt.title("A fully grown tree memorises the training loans"); plt.show()
print("Validation AUC peaks at depth", depths[int(va.mean(1).argmax())], "->", round(va.mean(1).max(), 3), "| unlimited depth:", round(va.mean(1)[-1], 3))''')
    nb.code(r'''full = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
print(f"Unpruned tree: {full.get_n_leaves()} leaves, depth {full.get_depth()}, train acc {full.score(Xtr, ytr):.3f}, TEST acc {full.score(Xte, yte):.3f}, TEST AUC {roc_auc_score(yte, full.predict_proba(Xte)[:, 1]):.3f}")
for leaf in [1, 5, 10, 20, 50]:
    cvs = cross_val_score(DecisionTreeClassifier(min_samples_leaf=leaf, random_state=0), Xtr, ytr, cv=skf, scoring="roc_auc").mean(); print(f"min_samples_leaf={leaf:<3} CV AUC = {cvs:.3f}")''')
    nb.md("**Pruning.** Cost-complexity pruning (Lesson 04) works identically for classifiers: minimise $\\text{impurity}(T)+\\alpha|T|$. Choose $\\alpha$ by cross-validation:")
    nb.code(r'''path = full.cost_complexity_pruning_path(Xtr, ytr); alphas = path.ccp_alphas[:-1]
cand = np.quantile(alphas, np.linspace(0, 0.99, 20))
cv_auc = [cross_val_score(DecisionTreeClassifier(random_state=0, ccp_alpha=a), Xtr, ytr, cv=skf, scoring="roc_auc").mean() for a in cand]
best_alpha = cand[int(np.argmax(cv_auc))]
pruned = DecisionTreeClassifier(random_state=0, ccp_alpha=best_alpha).fit(Xtr, ytr)
print(f"Pruned tree  : {pruned.get_n_leaves()} leaves, depth {pruned.get_depth()}, train acc {pruned.score(Xtr, ytr):.3f}, TEST acc {pruned.score(Xte, yte):.3f}, TEST AUC {roc_auc_score(yte, pruned.predict_proba(Xte)[:, 1]):.3f}")
plt.semilogx(cand + 1e-6, cv_auc, "o-"); plt.axvline(best_alpha + 1e-6, color="r", ls=":"); plt.xlabel("ccp_alpha"); plt.ylabel("CV AUC"); plt.title("Pruning strength vs validation AUC"); plt.show()''')
    nb.callout("key", "Pruning shrinks the tree by a factor of ~20 **and improves test accuracy and AUC substantially**. The big tree's extra branches were fitting noise: "
                      "individual rejected applicants with unlucky combinations, and noise doesn't generalise. "
                      "Never ship an unpruned/unlimited tree.")
    nb.md("**Probabilities from a tree are coarse.** A tree with $L$ leaves can only output $L$ different probabilities (the class proportions in each leaf). Deep trees give extreme 0/1 values; shallow ones, a handful of distinct values:")
    nb.code(r'''for name, m in [("unpruned", full), ("pruned", pruned)]:
    pr_ = m.predict_proba(Xte)[:, 1]; print(f"{name:<9}: {len(np.unique(pr_.round(6)))} distinct probabilities on {len(pr_)} test loans | share of predictions exactly 0 or 1: {np.mean((pr_ == 0) | (pr_ == 1)):.0%}")''')

    # ---------------------------------------------------------- 7 rules
    nb.section("📖", "7. Reading rules and decision paths",
               "The pruned tree is small enough to read: a ready-made, auditable policy:")
    nb.code(r'''small_t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=0).fit(Xtr, ytr)
print(export_text(small_t, feature_names=list(Xtr.columns), show_weights=True))
fig, ax = plt.subplots(figsize=(18, 6.5)); plot_tree(small_t, feature_names=list(Xtr.columns), class_names=["rejected", "approved"], filled=True, rounded=True, fontsize=8, ax=ax, impurity=True, precision=2); plt.show()''')
    nb.md("**Explain one decision:** `decision_path` shows exactly which questions a specific applicant passed through.")
    nb.code(r'''applicant = Xte.iloc[[3]]
node_ids = small_t.decision_path(applicant).indices; t_ = small_t.tree_
print("Applicant features:", applicant.iloc[0][["credit_score", "dti", "lti", "has_collateral"]].round(2).to_dict())
for nid in node_ids[:-1]:
    f, thr = Xtr.columns[t_.feature[nid]], t_.threshold[nid]; val = applicant.iloc[0][f]
    print(f"  asked: {f} <= {thr:.2f}?  value = {val:.2f} -> {'yes' if val <= thr else 'no'}")
print("Decision:", ["REJECT", "APPROVE"][small_t.predict(applicant)[0]], "| P(approve) =", small_t.predict_proba(applicant)[0, 1].round(3))''')

    # ---------------------------------------------------------- 8 tree vs lr
    nb.section("⚔️", "8. Tree versus logistic regression",
               "Same data, honest comparison: tuned tree (pruned) vs logistic regression (from Lesson 10, standardised) on the same split.")
    nb.code(r'''from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
rows = []
for name, m in [("Logistic regression", lr), ("Pruned decision tree", pruned), ("Unpruned decision tree", full)]:
    pr_ = m.predict_proba(Xte)[:, 1]; rows.append({"model": name, "test acc": accuracy_score(yte, pr_ >= 0.5), "test AUC": roc_auc_score(yte, pr_)})
display(pd.DataFrame(rows).set_index("model").round(3))''')
    nb.md("Logistic regression wins here: the true log-odds are (almost) linear in the engineered ratios (its home turf) and a tree must approximate a smooth surface with a few rectangles. "
          "Trees win when the truth is **rule-like, thresholded or interaction-heavy**, or when you cannot engineer the right features. "
          "Their real power appears when many are averaged (next lesson: Random Forest).")

    # ---------------------------------------------------------- 9 profile
    nb.section("📋", "9. Cheat-sheet: the full profile of Decision Tree Classification")
    nb.table([
        ["**Underlying concept**", "A flowchart of yes/no questions on features; each leaf predicts a class (and class proportions as probabilities)."],
        ["**How it works**", "Greedy, recursive binary splitting: choose the (feature, threshold) with the greatest impurity reduction until stopping rules fire; prune."],
        ["**Mathematical foundation**", "Gini impurity 1−Σp², entropy −Σp·log₂p, information gain, weighted child impurity; cost-complexity pruning."],
        ["**Important parameters**", "`criterion` (gini/entropy/log_loss), `max_depth`, `min_samples_leaf`, `min_samples_split`, `max_leaf_nodes`, `ccp_alpha`, `class_weight`, `max_features`."],
        ["**Advantages**", "Readable rules; no scaling; handles numeric+categorical (one-hot) and interactions; automatic feature selection; fast prediction; native multi-class; (sklearn ≥1.3) handles NaN."],
        ["**Disadvantages**", "Overfits easily; unstable (small data changes → different tree); axis-aligned/staircase boundaries; coarse probabilities; biased toward high-cardinality features."],
        ["**Complexity**", "Training ≈ O(p · n log n · depth); prediction O(depth); memory O(#nodes)."],
        ["**Use cases**", "Transparent policies and triage rules, quick exploratory models, feature screening, base learners for forests and boosting."],
        ["**Limitations**", "Rarely the most accurate single model; needs pruning/limits; poor on smooth linear relationships and with highly imbalanced data unless weighted."],
    ], ["Question", "Answer"])

    # ---------------------------------------------------------- 10 worked
    nb.section("🧪", "10. More worked examples")
    nb.md(r"""### Worked Example 1: Gini of a split

A node has 10 loans: 6 approved, 4 rejected. A question sends 5 to the left (**5 approved, 0 rejected**) and 5 to the right (**1 approved, 4 rejected**).

- Parent: $G=1-(0.6^2+0.4^2)=1-0.52=0.48$.
- Left: $G=1-(1^2+0^2)=0$. Right: $G=1-(0.2^2+0.8^2)=1-0.68=0.32$.
- Weighted children: $0.5\cdot0+0.5\cdot0.32=0.16$.
- **Gain** $=0.48-0.16=\mathbf{0.32}$: a very useful split.""")
    nb.code(r'''g = lambda c: 1 - np.sum((np.array(c) / np.sum(c)) ** 2)
print("parent", round(g([6, 4]), 3), "| left", g([5, 0]), "| right", round(g([1, 4]), 3), "| gain", round(g([6, 4]) - 0.5 * g([5, 0]) - 0.5 * g([1, 4]), 3))''')
    nb.md(r"""### Worked Example 2: Why not misclassification error?

Parent: (40 A, 40 B). Split 1 → left (30 A, 10 B), right (10 A, 30 B). Split 2 → left (20 A, 40 B), right (20 A, 0 B). Both have **the same misclassification error** ($20/80=0.25$: it cannot tell them apart!). Gini: split 1 → weighted $0.375$; split 2 → $\frac{60}{80}(0.444)+0=0.333$. Split 2 is better (it carved out a *pure* group), and Gini/entropy see it; misclassification error does not.""")
    nb.code(r'''gi = lambda c: 1 - np.sum((np.array(c) / np.sum(c)) ** 2)
print("split 1 weighted Gini:", round(0.5 * gi([30, 10]) + 0.5 * gi([10, 30]), 4), "| split 2 weighted Gini:", round(60 / 80 * gi([20, 40]) + 20 / 80 * gi([20, 0]), 4))''')

    # ---------------------------------------------------------- problems
    nb.problem(
        "Impurity functions",
        "Write `gini(counts)` and `entropy(counts)` (base 2) for a list of class counts. Compute `g_9_5`, `e_9_5` for `[9, 5]`, `g_even`, `e_even` for `[7, 7]`, and `g_pure`, `e_pure` for `[14, 0]`.",
        ["`p = counts / counts.sum()`; Gini = `1 - (p ** 2).sum()`.",
         "Entropy = `-(p * np.log2(p)).sum()`: drop zero-probability classes first (`p = p[p > 0]`) to avoid log(0).",
         "Expected: Gini([9,5]) ≈ 0.459, entropy ≈ 0.940; even split → 0.5 and 1.0; pure → 0 and 0."],
        'def gini(counts):\n    pass\n\ndef entropy(counts):\n    pass\n\ng_9_5 = e_9_5 = g_even = e_even = g_pure = e_pure = None   # your code\n',
        'def gini(counts):\n    p = np.asarray(counts, float) / np.sum(counts); return 1 - np.sum(p ** 2)\n\ndef entropy(counts):\n    p = np.asarray(counts, float) / np.sum(counts); p = p[p > 0]; return -np.sum(p * np.log2(p))\n\n'
        'g_9_5, e_9_5 = gini([9, 5]), entropy([9, 5]); g_even, e_even = gini([7, 7]), entropy([7, 7]); g_pure, e_pure = gini([14, 0]), entropy([14, 0])\nprint(g_9_5, e_9_5, g_even, e_even, g_pure, e_pure)',
        'check("Gini([9,5]) ≈ 0.4592", lambda: abs(g_9_5 - 0.459184) < 1e-5)\ncheck("entropy([9,5]) ≈ 0.9403", lambda: abs(e_9_5 - 0.940286) < 1e-5)\ncheck("even split: Gini 0.5, entropy 1", lambda: abs(g_even - 0.5) < 1e-9 and abs(e_even - 1) < 1e-9)\ncheck("pure node: both zero", lambda: g_pure == 0 and e_pure == 0)',
        explain="Both measures are 0 for pure nodes and maximal (0.5 / 1 bit) for a 50-50 mix. The 9-vs-5 root of the tennis data sits in between.")

    nb.problem(
        "Information gain of one attribute",
        "Using the `tennis` DataFrame and your `entropy` function: write `info_gain(df, feature, target='Play')` returning the information gain of splitting on `feature`. "
        "Compute `ig_humidity` and `ig_wind`, and set `root_feature` to the attribute with the highest gain among all four attributes.",
        ["Parent entropy: `entropy(df[target].value_counts())`.",
         "For each value group `g` in `df.groupby(feature)`: weight = `len(g) / len(df)` × entropy of `g[target].value_counts()`.",
         "Gain = parent − weighted children. Expected: Humidity 0.1518, Wind 0.0481, root = Outlook."],
        'def info_gain(df, feature, target="Play"):\n    pass\n\nig_humidity = ig_wind = root_feature = None   # your code\n',
        'def info_gain(df, feature, target="Play"):\n    parent = entropy(df[target].value_counts())\n    children = sum(len(g) / len(df) * entropy(g[target].value_counts()) for _, g in df.groupby(feature))\n    return parent - children\n\n'
        'ig_humidity = info_gain(tennis, "Humidity"); ig_wind = info_gain(tennis, "Wind")\nroot_feature = max(["Outlook", "Temp", "Humidity", "Wind"], key=lambda f: info_gain(tennis, f))\nprint(round(ig_humidity, 4), round(ig_wind, 4), root_feature)',
        'check("Humidity gain ≈ 0.1518", lambda: abs(ig_humidity - 0.1518) < 1e-3)\ncheck("Wind gain ≈ 0.0481", lambda: abs(ig_wind - 0.0481) < 1e-3)\ncheck("root question is Outlook", lambda: root_feature == "Outlook")',
        explain="Outlook has the largest gain (0.247) because its Overcast branch is perfectly pure. Wind barely helps (0.048), so the tree asks about it only deep down.",
        level="⭐⭐")

    nb.problem(
        "Find the best Gini threshold",
        "Write `best_gini_split(x, y)` for one numeric feature and binary labels: try every midpoint between consecutive sorted unique x values and return `(threshold, weighted_gini)`. "
        "Test on `xg = [1,2,3,4,5,6,7,8]`, `yg = [0,0,0,1,1,1,1,1]` (→ `res_clean`) and on a noisy version `yg2 = [0,0,1,0,1,1,0,1]` (→ `res_noisy`). "
        "Then confirm that a depth-1 `DecisionTreeClassifier` picks the same threshold for the noisy data: `same_as_sklearn`.",
        ["For threshold t: left = `y[x <= t]`, right = `y[x > t]`; weighted Gini = (nL·G(left) + nR·G(right)) / n.",
         "Gini for a label array: `1 - sum(p²)` over class proportions (use `np.bincount`).",
         "sklearn: `DecisionTreeClassifier(max_depth=1).fit(xg.reshape(-1, 1), yg2).tree_.threshold[0]`."],
        'xg = np.arange(1, 9.); yg = np.array([0, 0, 0, 1, 1, 1, 1, 1]); yg2 = np.array([0, 0, 1, 0, 1, 1, 0, 1])\n\ndef best_gini_split(x, y):\n    pass\n\nres_clean = res_noisy = same_as_sklearn = None   # your code\n',
        'xg = np.arange(1, 9.); yg = np.array([0, 0, 0, 1, 1, 1, 1, 1]); yg2 = np.array([0, 0, 1, 0, 1, 1, 0, 1])\n\ndef best_gini_split(x, y):\n    def G(lbl):\n        return 0.0 if len(lbl) == 0 else 1 - np.sum((np.bincount(lbl, minlength=2) / len(lbl)) ** 2)\n'
        '    vals = np.unique(x); mids = (vals[:-1] + vals[1:]) / 2\n    scores = [(len(y[x <= t]) * G(y[x <= t]) + len(y[x > t]) * G(y[x > t])) / len(y) for t in mids]\n    k = int(np.argmin(scores)); return mids[k], scores[k]\n\n'
        'res_clean = best_gini_split(xg, yg); res_noisy = best_gini_split(xg, yg2)\nstump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(xg.reshape(-1, 1), yg2)\nsame_as_sklearn = bool(abs(stump.tree_.threshold[0] - res_noisy[0]) < 1e-9)\nprint(res_clean, res_noisy, same_as_sklearn)',
        'check("clean data: threshold 3.5 with weighted Gini 0", lambda: abs(res_clean[0] - 3.5) < 1e-9 and abs(res_clean[1]) < 1e-9)\ncheck("noisy data cannot reach zero impurity", lambda: res_noisy[1] > 0.2)\ncheck("matches the scikit-learn stump", lambda: same_as_sklearn)',
        explain="On clean data one cut makes both sides pure (Gini 0). With label noise the best cut still leaves impurity, and that residual impurity is exactly what a deeper tree would be tempted to 'fix' by overfitting.",
        level="⭐⭐")

    nb.problem(
        "Tune a tree on breast-cancer data",
        "Using `Xb_tr, yb_tr, Xb_te, yb_te`: run `GridSearchCV` with `StratifiedKFold(5, shuffle=True, random_state=0)`, scoring `'roc_auc'`, over `max_depth ∈ [2, 3, 4, 5, 6, 8]` and `min_samples_leaf ∈ [1, 5, 10, 20]` for `DecisionTreeClassifier(random_state=0)`. "
        "Store `best_params`, the cross-validated `best_cv_auc` and the test AUC `test_auc` of the refit best tree.",
        ["`GridSearchCV(DecisionTreeClassifier(random_state=0), grid, cv=skf, scoring='roc_auc').fit(Xb_tr, yb_tr)`.",
         "`gs.best_params_`, `gs.best_score_`, and `roc_auc_score(yb_te, gs.best_estimator_.predict_proba(Xb_te)[:, 1])`.",
         "Single trees top out around 0.93–0.95 AUC here (a forest will do better)."],
        'best_params = best_cv_auc = test_auc = None   # your code\n',
        'gs = GridSearchCV(DecisionTreeClassifier(random_state=0), {"max_depth": [2, 3, 4, 5, 6, 8], "min_samples_leaf": [1, 5, 10, 20]}, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").fit(Xb_tr, yb_tr)\n'
        'best_params = gs.best_params_; best_cv_auc = gs.best_score_\ntest_auc = roc_auc_score(yb_te, gs.best_estimator_.predict_proba(Xb_te)[:, 1])\nprint(best_params, round(best_cv_auc, 3), round(test_auc, 3))',
        'check("both parameters tuned", lambda: set(best_params) == {"max_depth", "min_samples_leaf"})\ncheck("CV AUC > 0.92", lambda: best_cv_auc > 0.92)\ncheck("test AUC > 0.9", lambda: test_auc > 0.9)',
        explain="Tuning shrinks the tree to a sensible size. Even so a single tree is limited: averaging many (Random Forest) will push AUC toward 0.99.",
        level="⭐⭐")

    nb.problem(
        "Explain a prediction in words",
        "Write `explain(model, X_row, feature_names)` that returns a **list of strings**, one for each question on the decision path of one sample (use `model.decision_path` and `model.tree_`), formatted like `'credit_score <= 641.50 -> no'` (`yes` if the sample satisfies `<=`). "
        "Run it for the first row of `Xte` with `small_t` (store in `steps`). Set `n_steps = len(steps)` and `matches_depth = (n_steps == number of internal nodes on the path)`: i.e. every non-leaf node on the path.",
        ["`node_ids = model.decision_path(X_row).indices` lists the nodes visited; the last is a leaf.",
         "For each internal node: feature = `feature_names[t.feature[nid]]`, threshold = `t.threshold[nid]`, value = `X_row.iloc[0][feature]`.",
         "Format with `f'{name} <= {thr:.2f} -> {\"yes\" if val <= thr else \"no\"}'`."],
        'def explain(model, X_row, feature_names):\n    pass\n\nsteps = n_steps = matches_depth = None   # your code\n',
        'def explain(model, X_row, feature_names):\n    t = model.tree_; out = []\n    for nid in model.decision_path(X_row).indices:\n        if t.children_left[nid] == -1: continue                       # leaf: no question asked\n'
        '        name, thr = feature_names[t.feature[nid]], t.threshold[nid]; val = X_row.iloc[0][name]\n        out.append(f"{name} <= {thr:.2f} -> {\'yes\' if val <= thr else \'no\'}")\n    return out\n\n'
        'steps = explain(small_t, Xte.iloc[[0]], list(Xtr.columns)); n_steps = len(steps)\nmatches_depth = n_steps == sum(small_t.tree_.children_left[n] != -1 for n in small_t.decision_path(Xte.iloc[[0]]).indices)\nprint(steps)',
        'check("at least one step, at most the tree depth (3)", lambda: 1 <= n_steps <= 3)\ncheck("steps are strings with a yes/no verdict", lambda: all(isinstance(s, str) and s.endswith(("yes", "no")) for s in steps))\ncheck("one step per internal node on the path", lambda: matches_depth)',
        explain="This is the 'reason code' regulators ask for: *which* rules fired for *this* applicant. Trees give it for free; for black-box models it requires extra tooling.",
        level="⭐⭐⭐")

    nb.problem(
        "Prune the loan tree",
        "Using `Xtr, ytr, Xte, yte`, the stratified `skf`, and `full` (the unpruned tree): choose `ccp_alpha` from 20 candidates `np.quantile(alphas, np.linspace(0, 0.99, 20))` (with `alphas = full.cost_complexity_pruning_path(Xtr, ytr).ccp_alphas[:-1]`) by 5-fold CV AUC. "
        "Store `leaves_full`, `leaves_pruned`, `auc_full` and `auc_pruned` (test AUCs).",
        ["Loop over the candidates with `cross_val_score(DecisionTreeClassifier(random_state=0, ccp_alpha=a), Xtr, ytr, cv=skf, scoring='roc_auc')`.",
         "Refit with the best alpha on all training data.",
         "Compare leaf counts via `get_n_leaves()` and test AUCs via `roc_auc_score`."],
        'leaves_full = leaves_pruned = auc_full = auc_pruned = None   # your code\n',
        'alphas_ = full.cost_complexity_pruning_path(Xtr, ytr).ccp_alphas[:-1]; cand_ = np.quantile(alphas_, np.linspace(0, 0.99, 20))\n'
        'scores_ = [cross_val_score(DecisionTreeClassifier(random_state=0, ccp_alpha=a), Xtr, ytr, cv=skf, scoring="roc_auc").mean() for a in cand_]\n'
        'pr2 = DecisionTreeClassifier(random_state=0, ccp_alpha=cand_[int(np.argmax(scores_))]).fit(Xtr, ytr)\nleaves_full = full.get_n_leaves(); leaves_pruned = pr2.get_n_leaves()\n'
        'auc_full = roc_auc_score(yte, full.predict_proba(Xte)[:, 1]); auc_pruned = roc_auc_score(yte, pr2.predict_proba(Xte)[:, 1])\nprint(leaves_full, leaves_pruned, round(auc_full, 3), round(auc_pruned, 3))',
        'check("pruned tree has < 10% of the leaves", lambda: leaves_pruned < 0.1 * leaves_full)\ncheck("pruning improves test AUC by more than 0.05", lambda: auc_pruned > auc_full + 0.05)',
        explain="Memorised noise hurt: the unpruned tree is both far bigger and far worse. Simplicity isn't just convenient, it generalises.",
        level="⭐⭐")

    nb.quiz([
        ("Gini of a node with 3 'yes' and 1 'no'?", "1 − (0.75² + 0.25²) = 1 − 0.625 = 0.375."),
        ("Why does a decision tree classifier often outperform logistic regression on XOR-like data but lose on smooth additive data?",
         "Trees capture thresholds and interactions through successive splits; logistic regression can only draw one linear boundary but represents smooth additive log-odds with very few parameters."),
        ("An unpruned tree has 100 % training accuracy and 77 % test accuracy. What is it doing and what do you do?",
         "Overfitting (memorising noise). Limit depth/leaf size or prune with ccp_alpha, chosen by cross-validation."),
        ("Two splits have the same misclassification error. Why might one still be much better?",
         "Gini/entropy reward purity (a split that produces one *pure* child is closer to finished) whereas misclassification error is too coarse to see it."),
        ("Why are tree-predicted probabilities poor, and how do ensembles help?",
         "Each leaf yields only its class proportion: few distinct, often extreme values. Averaging many trees (forests) gives smoother, better-ranked probabilities."),
    ])

    nb.footer(["Classification trees pick the split that most reduces **impurity** (Gini or entropy); leaves predict the majority class / class proportions.",
               "Information gain = parent entropy − weighted child entropy (Outlook wins the Play-Tennis root).",
               "From scratch: sort + cumulative class counts → every threshold's impurity in one vectorised pass; matches scikit-learn exactly.",
               "Trees draw axis-aligned rectangles; unrestricted trees overfit dramatically: prune (ccp_alpha) or limit leaf size.",
               "Readable and auditable (rules, decision paths) but unstable and rarely best alone: forests fix the variance."],
              "Random Forest Classification", "12_Random_Forest_Classification.ipynb")
    return nb
