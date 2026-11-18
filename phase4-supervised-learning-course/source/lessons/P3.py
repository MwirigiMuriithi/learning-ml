from nbkit import Lesson


def make():
    nb = Lesson("P3_Loan_Approval_Classification", 3, "Loan Approval Classification",
                "Phase 4 project · classification · a leakage trap · costs · fairness · reason codes",
                270, "🔴 Project", kind="Project",
                setup_extra="""import warnings, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, f1_score, log_loss, confusion_matrix
warnings.filterwarnings("ignore")""")
    nb.header(
        objectives=[
            "Detect **target leakage** by interrogating a suspicious column, and quantify how misleading it is.",
            "Build a leakage-free, fair model tournament for a binary decision with business costs.",
            "Choose a decision threshold from **asymmetric costs** using out-of-fold predictions and audit the savings on a locked test set.",
            "Audit a model for **group fairness** (demographic parity, equal opportunity) with bootstrap uncertainty, and test for **proxy** variables.",
            "Produce **reason codes** that explain an individual decision, and package a decision function with a model card.",
        ],
        prereqs="Lessons 09–16 (especially 10, 16)",
        hook=("**From:** Ms. Wambui, Risk Director, Jengo Microfinance  \n**Subject:** Faster, fairer loan pre-screening\n\n"
              "> *\"Our officers review ~3 000 applications a year and decisions vary a lot between officers. We want a pre-screening model that recommends approve / decline, "
              "so officers only review borderline cases. Approving a loan that should have been declined costs us roughly **five times** as much as declining a good applicant. "
              "Regulators will ask us to explain individual declines and to show that we don't discriminate by gender or marital status. Can you help?\"*\n\n"
              "A loan model touches people's lives. This project is where *accuracy* meets **responsibility**: leakage, costs, fairness and explanations are not optional extras."))
    nb.md(r"""## 📦 The data: `data/loan_approval.csv` (3 000 synthetic applications)

| Column | Meaning |
|---|---|
| `loan_id` | identifier |
| `age`, `gender`, `married`, `dependents`, `education` | applicant profile (**`gender` and `married` are protected attributes**) |
| `employment_type`, `monthly_income_kes`, `property_area` | employment/location |
| `loan_amount_kes`, `loan_term_months` | the requested loan |
| `credit_score` (**some missing**), `credit_history_years`, `existing_loans`, `monthly_debt_kes` | credit profile |
| `has_collateral`, `savings_balance_kes` (**some missing**) | security & savings |
| `interest_rate_offered` | ⚠️ **look at this one carefully** |
| **`approved`** | **TARGET**: 1 = approved, 0 = declined (≈ 60 % approved) |

## 🎯 Deliverables
1. This notebook, fully run, with your reasoning in added markdown cells.
2. `loan_model.joblib` (pipeline + threshold) and a working `decide()` function returning a decision with probability and reasons.
3. A model card.

## 🧮 Rubric (100 points)
| Area | Pts | Excellent looks like |
|---|---|---|
| Leakage detection | 15 | Finds the leaky column, explains the mechanism, quantifies the fake gain |
| Pipeline & fair tournament | 20 | No leakage, shared folds, baseline, ≥ 5 models, AUC + F1 + log loss |
| Tuning | 5 | Small, sensible grid |
| Cost-based threshold | 15 | Out-of-fold choice, test-set savings reported |
| Fairness audit | 20 | Parity + equal opportunity with uncertainty; proxy check; reasoned decision on protected attributes |
| Reason codes | 10 | Correct, readable top-3 reasons for a decline |
| Packaging & model card | 15 | Reloadable, honest limitations |

## 🗺️ Milestones
1. Profile · 2. **Leakage hunt** · 3. Features & split · 4. Tournament · 5. Tune · 6. Cost-based threshold · 7. Fairness audit (a, b) · 8. Reason codes · 9. Package & model card""")

    nb.section("📥", "Milestone 0: Shared helper code (given)")
    nb.code(r'''def load_loans():
    d = pd.read_csv(data_path("loan_approval.csv"))
    d["emi"] = d["loan_amount_kes"] / d["loan_term_months"] * 1.15                                   # rough monthly instalment
    d["dti"] = (d["monthly_debt_kes"] + d["emi"]) / d["monthly_income_kes"]                          # debt-to-income
    d["lti"] = d["loan_amount_kes"] / (d["monthly_income_kes"] * 12)                                 # loan-to-annual-income
    d["log_income"] = np.log(d["monthly_income_kes"]); d["log_savings"] = np.log1p(d["savings_balance_kes"])
    return d

NUM = ["credit_score", "credit_history_years", "dti", "lti", "log_income", "log_savings", "has_collateral", "age", "dependents", "existing_loans"]
CAT = ["employment_type", "property_area", "education"]
def make_prep(num=NUM, cat=CAT):
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(num)),
                              ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), list(cat))])
loans = load_loans(); print(loans.shape); display(loans.head(3))''')

    # M1
    nb.section("🔍", "Milestone 1: Profile the data (part of 'Pipeline & tournament')")
    nb.problem(
        "Profile",
        "`n_rows`, `approval_rate` (float), and `missing_cols`: the list of columns with missing values (note: the engineered `log_savings` inherits the gaps of `savings_balance_kes`, so it will appear too).",
        ["`len(loans)`, `loans['approved'].mean()`, `[c for c in loans.columns if loans[c].isna().any()]`.",
         "One of the columns with *lots* of missing values deserves suspicion: how many? Why would a column be missing for a particular subset of applicants?"],
        'n_rows = approval_rate = missing_cols = None   # your code\n',
        'n_rows = len(loans); approval_rate = float(loans["approved"].mean()); missing_cols = [c for c in loans.columns if loans[c].isna().any()]\nprint(n_rows, round(approval_rate, 3), missing_cols); print(loans[missing_cols].isna().sum().to_dict())',
        'check("3000 applications", lambda: n_rows == 3000)\ncheck("about 60% approved", lambda: abs(approval_rate - 0.604) < 0.005)\ncheck("credit_score, savings_balance_kes and interest_rate_offered have gaps (plus the derived log_savings)", lambda: {"credit_score", "savings_balance_kes", "interest_rate_offered"} <= set(missing_cols) <= {"credit_score", "savings_balance_kes", "interest_rate_offered", "log_savings"})',
        explain="`interest_rate_offered` is missing for ~40 % of rows: almost exactly the share of declined loans. Missingness that lines up with the target is a giant red flag.",
        level="⭐")

    # M2
    nb.section("🕵️", "Milestone 2: The leakage hunt (15 pts)",
               "A loan is declined, so no interest rate is ever offered. **The rate is only recorded after the approval decision.** If we use it to predict approval, we're feeding the model the answer. "
               "Prove it, quantify how misleading it is, and decide what to drop.")
    nb.problem(
        "Interrogate the suspicious column",
        "1. `rate_missing_rejected`, `rate_missing_approved`: the share of **declined** / **approved** applications whose `interest_rate_offered` is missing.\n"
        "2. `leak_auc`: the ROC-AUC obtained by using the single indicator `loans['interest_rate_offered'].notna()` as a predictor of `approved` (a one-feature 'model').\n"
        "3. `leaky_cv_auc`: the 5-fold stratified CV (`StratifiedKFold(5, shuffle=True, random_state=0)`) ROC-AUC of `DecisionTreeClassifier(max_depth=2, random_state=0)` trained on just `credit_score` (median-filled) and `rate_filled` (= the rate with missing → 0).\n"
        "4. `honest_cv_auc`: the same tree trained **without** the rate (`credit_score` and `dti` only).\n"
        "5. `drop_cols`: the list of columns that must **not** be used as features: the id, the leaky column and (by policy) the protected attributes.",
        ["`loans.loc[loans['approved'] == 0, 'interest_rate_offered'].isna().mean()` and the same for approved.",
         "`roc_auc_score(loans['approved'], loans['interest_rate_offered'].notna())`.",
         "For the trees: `X_leaky = pd.DataFrame({'credit_score': loans['credit_score'].fillna(loans['credit_score'].median()), 'rate_filled': loans['interest_rate_offered'].fillna(0)})`; use `cross_val_score(..., scoring='roc_auc')`.",
         "`drop_cols = ['loan_id', 'interest_rate_offered', 'gender', 'married']`."],
        'rate_missing_rejected = rate_missing_approved = leak_auc = leaky_cv_auc = honest_cv_auc = drop_cols = None   # your code\n',
        'rate_missing_rejected = float(loans.loc[loans["approved"] == 0, "interest_rate_offered"].isna().mean()); rate_missing_approved = float(loans.loc[loans["approved"] == 1, "interest_rate_offered"].isna().mean())\nleak_auc = roc_auc_score(loans["approved"], loans["interest_rate_offered"].notna())\n'
        'skf = StratifiedKFold(5, shuffle=True, random_state=0); cs = loans["credit_score"].fillna(loans["credit_score"].median())\nX_leaky = pd.DataFrame({"credit_score": cs, "rate_filled": loans["interest_rate_offered"].fillna(0)}); X_honest = pd.DataFrame({"credit_score": cs, "dti": loans["dti"]})\n'
        'tree = DecisionTreeClassifier(max_depth=2, random_state=0)\nleaky_cv_auc = cross_val_score(tree, X_leaky, loans["approved"], cv=skf, scoring="roc_auc").mean(); honest_cv_auc = cross_val_score(tree, X_honest, loans["approved"], cv=skf, scoring="roc_auc").mean()\n'
        'drop_cols = ["loan_id", "interest_rate_offered", "gender", "married"]\nprint(rate_missing_rejected, rate_missing_approved, leak_auc, round(leaky_cv_auc, 3), round(honest_cv_auc, 3))',
        'check("every declined loan has no rate; every approved loan has one", lambda: rate_missing_rejected == 1.0 and rate_missing_approved == 0.0)\ncheck("the missingness indicator alone is a perfect predictor (AUC = 1)", lambda: abs(leak_auc - 1.0) < 1e-9)\ncheck("the leaky tree scores a perfect 1.0 CV AUC", lambda: leaky_cv_auc > 0.999)\ncheck("the honest tree scores far lower (< 0.9)", lambda: honest_cv_auc < 0.9)\ncheck("drop list contains id, leaky column and protected attributes", lambda: set(drop_cols) >= {"loan_id", "interest_rate_offered", "gender", "married"})',
        explain="A depth-2 tree 'predicts' approval perfectly from a column that only exists *after* approval: a fantasy AUC of 1.0 versus a realistic ≈ 0.83. In production the rate wouldn't exist for a new applicant, so the leaky model would collapse. Always ask of every feature: *would I have this value at decision time?*",
        level="⭐⭐")
    nb.md("✍️ *In a markdown cell, explain in your own words (3 sentences) what 'target leakage' is, how you spotted it here, and how you'd check for it on a new dataset.*")

    # M3
    nb.section("🛠️", "Milestone 3; Features and the split")
    nb.code(r'''# 🔄 Checkpoint
loans = load_loans()''')
    nb.problem(
        "Build X, y and the split",
        "1. `X = loans[NUM + CAT]` and `y = loans['approved']` (protected attributes and the leaky/id columns stay **out**); keep `prot = loans[['gender', 'married']]` separately for the fairness audit.\n"
        "2. A **stratified** 75/25 split with `random_state=42` → `Xtr, Xte, ytr, yte`.\n3. `prep = make_prep()` fitted inside pipelines later; verify `prep.fit_transform(Xtr)` has no NaN.",
        ["`train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)`.",
         "`prot.loc[Xte.index]` gives the protected attributes of the test applicants (the split keeps the original index).",
         "Make sure none of `drop_cols` is in `X.columns`."],
        'X = y = prot = None\nXtr = Xte = ytr = yte = None   # your code\n',
        'X = loans[NUM + CAT]; y = loans["approved"]; prot = loans[["gender", "married"]]\nXtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)\nprep = make_prep(); print(Xtr.shape, Xte.shape, round(ytr.mean(), 3), round(yte.mean(), 3))',
        'check("no leaky / id / protected columns in X", lambda: not ({"loan_id", "interest_rate_offered", "gender", "married"} & set(X.columns)))\ncheck("engineered ratios dti and lti are present", lambda: {"dti", "lti"} <= set(X.columns))\ncheck("stratified split: 2250 / 750 with equal approval rates", lambda: len(Xtr) == 2250 and len(Xte) == 750 and abs(ytr.mean() - yte.mean()) < 0.005)\ncheck("preprocessing output has no NaN", lambda: not np.isnan(make_prep().fit_transform(Xtr)).any())',
        explain="`stratify=y` keeps the approval rate identical in train and test, which matters for stable metrics. The protected attributes live in a separate frame so we can audit fairness *without* letting the model use them.",
        level="⭐⭐")

    # M4
    nb.section("🥊", "Milestone 4: The tournament (20 pts together with Milestone 1)",
               "A baseline plus **at least five** classifiers from the toolbox (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, SVM…) in `Pipeline([('prep', prep), ('model', est)])`, "
               "with identical stratified CV (`StratifiedKFold(5, shuffle=True, random_state=0)`) and **three metrics**: ROC-AUC, F1 and log loss.")
    nb.code(r'''# 🔄 Checkpoint
loans = load_loans(); X = loans[NUM + CAT]; y = loans["approved"]; prot = loans[["gender", "married"]]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
prep = make_prep(); skf = StratifiedKFold(5, shuffle=True, random_state=0)''')
    nb.problem(
        "Run the tournament",
        "Build `models` (dict) and `leader`: a DataFrame indexed by model name with columns `cv_auc`, `cv_f1`, `cv_logloss` (positive), sorted by `cv_auc` descending, and `champion` (name of the best by AUC). Include `DummyClassifier(strategy='prior')` as the baseline.",
        ["`cross_validate(model, Xtr, ytr, cv=skf, scoring={'auc': 'roc_auc', 'f1': 'f1', 'nll': 'neg_log_loss'})`.",
         "`-r['test_nll'].mean()` for log loss.",
         "Models: `LogisticRegression(max_iter=2000)`, `DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=0)`, `RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=-1)`, `KNeighborsClassifier(15)`, `GaussianNB()`, `SVC(probability=True, random_state=0)`."],
        'models = {}\nleader = champion = None   # your code\n',
        'mk = lambda est: Pipeline([("prep", prep), ("model", est)])\nmodels = {"Baseline (prior)": mk(DummyClassifier(strategy="prior")), "Logistic Regression": mk(LogisticRegression(max_iter=2000)), "Decision Tree": mk(DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=0)),\n'
        '          "Random Forest": mk(RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=-1)), "KNN (k=15)": mk(KNeighborsClassifier(15)), "Naive Bayes": mk(GaussianNB()), "SVM (RBF)": mk(SVC(probability=True, random_state=0))}\n'
        'rows = {}\nfor n, m in models.items():\n    r = cross_validate(m, Xtr, ytr, cv=skf, scoring={"auc": "roc_auc", "f1": "f1", "nll": "neg_log_loss"})\n    rows[n] = {"cv_auc": r["test_auc"].mean(), "cv_f1": r["test_f1"].mean(), "cv_logloss": -r["test_nll"].mean()}\n'
        'leader = pd.DataFrame(rows).T.sort_values("cv_auc", ascending=False); champion = leader.index[0]; display(leader.round(3)); print("champion:", champion)',
        'check("baseline plus >= 5 real models", lambda: len(models) >= 6)\ncheck("leader has the three metrics, sorted by AUC", lambda: {"cv_auc", "cv_f1", "cv_logloss"} <= set(leader.columns) and leader["cv_auc"].is_monotonic_decreasing)\ncheck("baseline AUC = 0.5", lambda: abs(leader[[("aseline" in i) for i in leader.index]]["cv_auc"].iloc[0] - 0.5) < 1e-9)\ncheck("champion AUC > 0.92", lambda: leader["cv_auc"].iloc[0] > 0.92)\ncheck("log loss is positive", lambda: (leader["cv_logloss"] > 0).all())',
        explain="Logistic regression leads (AUC ≈ 0.93) with the SVM (0.92) and the forest (0.92) close behind; the single tree and KNN trail, and the tree's log loss is no better than the do-nothing baseline's. With engineered ratios the problem is nearly linear in the log-odds, so the simple, explainable model is also the best: good news for a regulated lender.",
        level="⭐⭐⭐")

    # M5
    nb.section("🎛️", "Milestone 5: Tune (5 pts)")
    nb.code(r'''# 🔄 Checkpoint
loans = load_loans(); X = loans[NUM + CAT]; y = loans["approved"]; prot = loans[["gender", "married"]]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
prep = make_prep(); skf = StratifiedKFold(5, shuffle=True, random_state=0)''')
    nb.problem(
        "Grid-search your champion",
        "Tune your champion family with `GridSearchCV` (3-fold stratified, `scoring='roc_auc'`). For logistic regression use `model__C ∈ [0.01, 0.1, 1, 10]` (and optionally the penalty). "
        "Store `tuned_model` (fitted best estimator on `Xtr, ytr`), `grid` (your parameter grid) and `tuned_cv_auc` = its 5-fold CV AUC with the shared `skf`.",
        ["`GridSearchCV(Pipeline([('prep', prep), ('model', LogisticRegression(max_iter=2000))]), {'model__C': [0.01, 0.1, 1, 10]}, cv=3, scoring='roc_auc').fit(Xtr, ytr)`.",
         "`tuned_model = gs.best_estimator_`; `tuned_cv_auc = cross_val_score(tuned_model, Xtr, ytr, cv=skf, scoring='roc_auc').mean()`.",
         "Logistic regression has almost no knobs, so don't expect a big change, and that's fine."],
        'grid = tuned_model = tuned_cv_auc = None   # your code\n',
        'grid = {"model__C": [0.01, 0.1, 1, 10]}\ngs = GridSearchCV(Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))]), grid, cv=StratifiedKFold(3, shuffle=True, random_state=1), scoring="roc_auc").fit(Xtr, ytr)\n'
        'tuned_model = gs.best_estimator_; tuned_cv_auc = cross_val_score(tuned_model, Xtr, ytr, cv=skf, scoring="roc_auc").mean(); print(gs.best_params_, round(tuned_cv_auc, 4))',
        'check("a grid with at least 3 values was searched", lambda: sum(len(v) for v in grid.values()) >= 3)\ncheck("tuned model is fitted", lambda: len(tuned_model.predict_proba(Xte)) == len(Xte))\ncheck("tuned CV AUC > 0.925", lambda: tuned_cv_auc > 0.925)',
        explain="The grid barely matters for logistic regression (all C ≥ 0.1 are equivalent here), but running it documents that you checked.",
        level="⭐")

    # M6
    nb.section("💰", "Milestone 6: Costs → threshold (15 pts)",
               "Ms. Wambui's cost matrix: approving a loan that should have been declined (**FP**) costs **5**; declining a good applicant (**FN**) costs **1**. "
               "Choose τ from **out-of-fold** training predictions (never the test set), then report the **test** cost at the default τ = 0.5 versus your τ, and the resulting confusion matrix.")
    nb.code(r'''# 🔄 Checkpoint: a reference tuned model
loans = load_loans(); X = loans[NUM + CAT]; y = loans["approved"]; prot = loans[["gender", "married"]]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
prep = make_prep(); skf = StratifiedKFold(5, shuffle=True, random_state=0); taus = np.linspace(0.05, 0.95, 91)
ref_model = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, C=1.0))])
def total_cost(y_true, p, tau, c_fp=5, c_fn=1):
    pred = p >= tau; return c_fp * np.sum(pred & (y_true == 0)) + c_fn * np.sum(~pred & (y_true == 1))''')
    nb.problem(
        "Optimal threshold and test savings",
        "Using `ref_model` (or your `tuned_model`):\n1. `oof` = out-of-fold P(approve) on the training set (`cross_val_predict(..., cv=skf, method='predict_proba')[:, 1]`).\n"
        "2. `tau_star` = the threshold in `taus` that minimises `total_cost(ytr.values, oof, tau)`.\n"
        "3. Fit the model on all training data; on the **test set**: `cost_default` (τ = 0.5), `cost_tuned` (τ = `tau_star`), `saving_pct`, and `approve_rate_tuned` (share of test applicants approved at `tau_star`).\n"
        "4. `cm_tuned`: the confusion matrix `[[TN, FP], [FN, TP]]` at `tau_star`.",
        ["`oof = cross_val_predict(model, Xtr, ytr, cv=skf, method='predict_proba')[:, 1]`.",
         "Theory for calibrated probabilities: τ* ≈ 5/(5+1) ≈ 0.83; the empirical optimum is usually a bit lower.",
         "`confusion_matrix(yte, p_test >= tau_star)`."],
        'oof = tau_star = cost_default = cost_tuned = saving_pct = approve_rate_tuned = cm_tuned = None   # your code\n',
        'final = ref_model; oof = cross_val_predict(final, Xtr, ytr, cv=skf, method="predict_proba")[:, 1]\ntau_star = float(taus[int(np.argmin([total_cost(ytr.values, oof, t) for t in taus]))])\nfinal.fit(Xtr, ytr); p_test = final.predict_proba(Xte)[:, 1]\n'
        'cost_default = int(total_cost(yte.values, p_test, 0.5)); cost_tuned = int(total_cost(yte.values, p_test, tau_star)); saving_pct = 100 * (cost_default - cost_tuned) / cost_default\n'
        'approve_rate_tuned = float(np.mean(p_test >= tau_star)); cm_tuned = confusion_matrix(yte, p_test >= tau_star)\nprint(round(tau_star, 2), cost_default, cost_tuned, round(saving_pct, 1), round(approve_rate_tuned, 3)); print(cm_tuned)',
        'check("tau* is strict (between 0.6 and 0.9) because false approvals cost 5x", lambda: 0.6 < tau_star < 0.9)\ncheck("the tuned threshold cuts test cost by more than 15%", lambda: saving_pct > 15)\ncheck("the policy approves fewer applicants than the 60% historical rate", lambda: approve_rate_tuned < 0.6)\ncheck("confusion matrix is 2x2 and sums to the test size", lambda: cm_tuned.shape == (2, 2) and cm_tuned.sum() == len(yte))',
        explain="At τ* ≈ 0.77 the bank approves roughly half the applicants (vs 60 % historically): fewer costly bad approvals (FP ↓) at the price of more good applicants declined (FN ↑). Net cost falls by ≈ 28 %. Ms. Wambui must understand that these savings are conditional on her 5:1 cost assumption.",
        level="⭐⭐⭐")

    # M7a
    nb.section("⚖️", "Milestone 7: Fairness audit (20 pts)",
               "Regulators will ask two questions. **Demographic parity:** are approval *rates* similar across groups? **Equal opportunity:** among applicants who *deserve* approval (true approvals), is the chance of being approved similar (equal true-positive rates)? "
               "Audit **gender** at your threshold with **uncertainty** (bootstrap), then test whether other features act as **proxies** for the protected attribute.")
    nb.code(r'''# 🔄 Checkpoint: reference predictions and threshold
loans = load_loans(); X = loans[NUM + CAT]; y = loans["approved"]; prot = loans[["gender", "married"]]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
prep = make_prep(); skf = StratifiedKFold(5, shuffle=True, random_state=0)
clf = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
p_test = clf.predict_proba(Xte)[:, 1]; tau_star = 0.77; approved_pred = p_test >= tau_star; prot_te = prot.loc[Xte.index]''')
    nb.problem(
        "Group metrics with uncertainty",
        "With `approved_pred`, `yte` and `prot_te['gender']` (values 'Female'/'Male'):\n"
        "1. `approve_rate`: dict gender → share predicted-approved; `dp_gap` = |difference| (demographic parity gap).\n"
        "2. `tpr`: dict gender → share predicted-approved among truly approved (`yte == 1`); `eo_gap` = |difference|.\n"
        "3. `tpr_ci`: bootstrap 95 % CI (2.5th, 97.5th percentile) of TPR(Female) − TPR(Male): 2000 resamples **with replacement within each group's truly-approved test applicants**, `rng = np.random.default_rng(0)`.\n"
        "4. `eo_significant`: True if the CI excludes 0.",
        ["`g = prot_te['gender'].values; yv = yte.values`; TPR(F) = `approved_pred[(g == 'Female') & (yv == 1)].mean()`.",
         "Bootstrap as in Project 2: `rng.choice(flags, size=len(flags), replace=True).mean()` for each group, difference, repeat 2000 times.",
         "Also compare the *historical* approval rates by gender: they are the benchmark the model must not worsen."],
        'approve_rate = tpr = {}\ndp_gap = eo_gap = tpr_ci = eo_significant = None   # your code\n',
        'g = prot_te["gender"].values; yv = yte.values\napprove_rate = {k: float(approved_pred[g == k].mean()) for k in ["Female", "Male"]}; dp_gap = abs(approve_rate["Female"] - approve_rate["Male"])\n'
        'tpr = {k: float(approved_pred[(g == k) & (yv == 1)].mean()) for k in ["Female", "Male"]}; eo_gap = abs(tpr["Female"] - tpr["Male"])\n'
        'rng = np.random.default_rng(0); fF, fM = approved_pred[(g == "Female") & (yv == 1)], approved_pred[(g == "Male") & (yv == 1)]\ndiffs = [rng.choice(fF, len(fF)).mean() - rng.choice(fM, len(fM)).mean() for _ in range(2000)]\n'
        'tpr_ci = tuple(float(v) for v in np.percentile(diffs, [2.5, 97.5])); eo_significant = bool(tpr_ci[0] > 0 or tpr_ci[1] < 0)\nprint(approve_rate, round(dp_gap, 3), tpr, round(eo_gap, 3), [round(v, 3) for v in tpr_ci], eo_significant)',
        'check("approval rates computed for both genders", lambda: set(approve_rate) == {"Female", "Male"})\ncheck("demographic parity gap is small (< 0.06)", lambda: dp_gap < 0.06)\ncheck("TPRs are similar (equal-opportunity gap < 0.1)", lambda: eo_gap < 0.1)\ncheck("the bootstrap CI contains 0: no significant gap", lambda: tpr_ci[0] < 0 < tpr_ci[1] and eo_significant is False)',
        explain="Approval rates differ by ≈ 2 points and TPRs by ≈ 4 points; the bootstrap interval for the TPR gap (about −3 to +12 points) straddles zero. With ~225 truly-approved applicants per gender the audit has decent power, so we can report: no evidence of a gender gap: consistent with the data, in which gender plays no role.",
        level="⭐⭐⭐")

    # M7b
    nb.problem(
        "Proxy test and the decision on protected attributes",
        "1. **Proxy test:** can the protected attribute be *predicted* from the allowed features? `proxy_auc` = CV ROC-AUC (`skf`) of a logistic-regression pipeline (`make_prep()`) predicting `is_female = (loans['gender'] == 'Female')` from `Xtr` (use `is_female.loc[Xtr.index]`). A value near 0.5 means no proxy.\n"
        "2. **Does including the attribute help?** `auc_without` = CV AUC of the approval model on `Xtr` (as in Milestone 4); `auc_with` = CV AUC when `gender` and `married` (dummy-encoded, from `loans`) are added as extra categorical columns. `auc_gain = auc_with − auc_without`.\n"
        "3. `use_protected` = `True`/`False`: your *recommendation* (hint: legal risk without predictive benefit).",
        ["`is_female = (loans['gender'] == 'Female').astype(int)`; `cross_val_score(Pipeline([('prep', make_prep()), ('model', LogisticRegression(max_iter=2000))]), Xtr, is_female.loc[Xtr.index], cv=skf, scoring='roc_auc')`.",
         "For the 'with' model: `Xp = loans.loc[Xtr.index, NUM + CAT + ['gender', 'married']]` and `make_prep(cat=CAT + ['gender', 'married'])`.",
         "If a protected attribute adds no AUC *and* nothing predicts it, dropping it is the safe, defensible policy."],
        'proxy_auc = auc_without = auc_with = auc_gain = use_protected = None   # your code\n',
        'is_female = (loans["gender"] == "Female").astype(int)\nlr_ = lambda p: Pipeline([("prep", p), ("model", LogisticRegression(max_iter=2000))])\nproxy_auc = cross_val_score(lr_(make_prep()), Xtr, is_female.loc[Xtr.index], cv=skf, scoring="roc_auc").mean()\n'
        'auc_without = cross_val_score(lr_(make_prep()), Xtr, ytr, cv=skf, scoring="roc_auc").mean()\nXp = loans.loc[Xtr.index, NUM + CAT + ["gender", "married"]]\nauc_with = cross_val_score(lr_(make_prep(cat=CAT + ["gender", "married"])), Xp, ytr, cv=skf, scoring="roc_auc").mean()\n'
        'auc_gain = auc_with - auc_without; use_protected = False\nprint(round(proxy_auc, 3), round(auc_without, 4), round(auc_with, 4), round(auc_gain, 5), use_protected)',
        'check("no proxy: gender cannot be predicted from the allowed features (AUC < 0.56)", lambda: proxy_auc < 0.56)\ncheck("including protected attributes adds nothing (|gain| < 0.003)", lambda: abs(auc_gain) < 0.003)\ncheck("recommendation: do not use protected attributes", lambda: use_protected is False)',
        explain="Nothing in the allowed features predicts gender (AUC ≈ 0.52) and adding gender/marital status doesn't improve AUC. Using them would add legal and ethical risk for zero benefit: the decision is clear. (In real data proxies such as postcode or occupation often *do* leak protected information; this test is how you find them.)",
        level="⭐⭐⭐")
    nb.md("✍️ *Write a short paragraph for the regulator: what you measured, what you found, and the limits of your audit (only two attributes, one threshold, synthetic data, historical labels that may themselves be biased).*")

    # M8
    nb.section("🗣️", "Milestone 8: Reason codes (10 pts)",
               "Regulators and customers deserve an explanation for each decline. For a **logistic regression** on standardised features, the score is a sum of contributions `coef × transformed_value`; "
               "the most **negative** contributions are the reasons pushing towards decline.")
    nb.code(r'''# 🔄 Checkpoint: fitted reference model and a declined applicant
loans = load_loans(); X = loans[NUM + CAT]; y = loans["approved"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
clf = Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr)
p_test = clf.predict_proba(Xte)[:, 1]; tau_star = 0.77; declined_idx = int(np.where(p_test < tau_star)[0][0])
print("first declined test applicant (position):", declined_idx, "| P(approve) =", round(p_test[declined_idx], 3))''')
    nb.problem(
        "Top-3 reasons for a decline",
        "Write `reason_codes(pipeline, x_row, k=3)` returning the names (strip the `num__`/`cat__` prefix) of the `k` features with the **most negative** contribution `coef × transformed value` for the one-row DataFrame `x_row`. "
        "Apply it to the declined applicant `Xte.iloc[[declined_idx]]` → `reasons`. Set `reasons_ok` = `True` if at least two of the reasons are among `{'dti', 'lti', 'credit_score', 'has_collateral'}`.",
        ["`names = pipeline[0].get_feature_names_out()`; `coef = pipeline[-1].coef_[0]`; `values = pipeline[0].transform(x_row)[0]`.",
         "`contrib = coef * values`; `order = np.argsort(contrib)[:k]` (most negative first).",
         "`names[i].split('__', 1)[1]` removes the prefix."],
        'def reason_codes(pipeline, x_row, k=3):\n    pass\n\nreasons = reasons_ok = None   # your code\n',
        'def reason_codes(pipeline, x_row, k=3):\n    names = pipeline[0].get_feature_names_out(); contrib = pipeline[-1].coef_[0] * pipeline[0].transform(x_row)[0]\n    return [names[i].split("__", 1)[1] for i in np.argsort(contrib)[:k]]\n\n'
        'reasons = reason_codes(clf, Xte.iloc[[declined_idx]]); reasons_ok = len({"dti", "lti", "credit_score", "has_collateral"} & set(reasons)) >= 2\nprint(reasons, reasons_ok)',
        'check("three reasons returned", lambda: isinstance(reasons, list) and len(reasons) == 3)\ncheck("names are clean (no prefixes)", lambda: all("__" not in r for r in reasons))\ncheck("at least two reasons are core credit-risk drivers", lambda: reasons_ok)',
        explain="For this applicant the biggest negatives are a high loan-to-income ratio, no collateral and a below-average credit score: exactly the kind of statement a decline letter needs ('your requested amount is large relative to your income…').",
        level="⭐⭐")

    # M9
    nb.section("📦", "Milestone 9: Package and document (15 pts)",
               "Deliver something the loan officers can call: a saved model **plus** threshold, and a `decide()` function returning the decision, the probability and the reasons.")
    nb.code(r'''# 🔄 Checkpoint: reference model/threshold/reason function
loans = load_loans(); X = loans[NUM + CAT]; y = loans["approved"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
clf = Pipeline([("prep", make_prep()), ("model", LogisticRegression(max_iter=2000))]).fit(Xtr, ytr); tau_star = 0.77
def reason_codes(pipeline, x_row, k=3):
    names = pipeline[0].get_feature_names_out(); contrib = pipeline[-1].coef_[0] * pipeline[0].transform(x_row)[0]
    return [names[i].split("__", 1)[1] for i in np.argsort(contrib)[:k]]''')
    nb.problem(
        "decide() and the model card",
        "1. Save `{'model': clf, 'threshold': tau_star}` with `joblib.dump` to `'loan_model.joblib'`; reload as `bundle`.\n"
        "2. Write `decide(applicant, bundle)` for a one-row DataFrame (columns of `X`) returning a dict `{'decision': 'APPROVE' or 'DECLINE', 'p_approve': float, 'reasons': list}`: reasons only for declines (else `[]`).\n"
        "3. Test it on two applicants: `d_best = decide(...)` for the **most creditworthy** test applicant (highest P(approve)) and `d_declined = decide(...)` for the first declined one (the first test row with `p < tau`, as in Milestone 8).\n"
        "4. Fill `model_card` (keys `intended_use`, `data`, `metrics`, `fairness`, `limitations`) with honest sentences.",
        ["`bundle['model'].predict_proba(applicant)[0, 1]`; compare with `bundle['threshold']`.",
         "Call `reason_codes(bundle['model'], applicant)` only when the decision is DECLINE.",
         "Model card: AUC ≈ 0.93, cost saving, fairness audit results *with limits*, 'pre-screening aid for officers: not an automatic final decision', synthetic data."],
        'bundle = d_best = d_declined = None\nmodel_card = {}\ndef decide(applicant, bundle):\n    pass   # your code\n',
        'joblib.dump({"model": clf, "threshold": tau_star}, "loan_model.joblib"); bundle = joblib.load("loan_model.joblib")\ndef decide(applicant, bundle):\n    p = float(bundle["model"].predict_proba(applicant)[0, 1]); approve = p >= bundle["threshold"]\n    return {"decision": "APPROVE" if approve else "DECLINE", "p_approve": p, "reasons": [] if approve else reason_codes(bundle["model"], applicant)}\n\n'
        'p_all = clf.predict_proba(Xte)[:, 1]; declined_idx = int(np.where(p_all < tau_star)[0][0])\nd_best = decide(Xte.iloc[[int(np.argmax(p_all))]], bundle); d_declined = decide(Xte.iloc[[declined_idx]], bundle)\n'
        'model_card = {"intended_use": "Pre-screening aid that recommends approve/decline so officers can focus on borderline cases; not an automatic final decision.",\n              "data": "3 000 synthetic applications; leaky column interest_rate_offered, the id and protected attributes were excluded; features include engineered debt-to-income and loan-to-income ratios.",\n'
        '              "metrics": "Logistic regression: CV ROC-AUC ≈ 0.93; at the cost-optimal threshold ≈ 0.77 (false approval costs 5x a false decline) the test cost fell by ≈ 28% versus a 0.5 threshold.",\n'
        '              "fairness": "Gender: approval-rate gap ≈ 2 points, equal-opportunity gap ≈ 4 points with a bootstrap CI that includes zero; no feature acts as a gender proxy; protected attributes are not used.",\n'
        '              "limitations": "Synthetic data; historical approvals may encode past bias; audit covered only gender and marital status at one threshold; cost ratio of 5:1 is an assumption to be revisited; the model must be monitored for drift."}\nprint(d_best); print(d_declined)',
        'import os\ncheck("bundle saved and reloaded", lambda: os.path.exists("loan_model.joblib") and set(bundle) == {"model", "threshold"})\ncheck("decide returns the three fields, and the best applicant is approved", lambda: set(d_best) == {"decision", "p_approve", "reasons"} and d_best["decision"] == "APPROVE")\ncheck("a declined applicant gets exactly three reasons", lambda: d_declined["decision"] == "DECLINE" and len(d_declined["reasons"]) == 3)\ncheck("approved applicants get no reasons", lambda: d_best["reasons"] == [])\ncheck("model card has five honest sections", lambda: set(model_card) == {"intended_use", "data", "metrics", "fairness", "limitations"} and all(isinstance(v, str) and len(v) > 40 for v in model_card.values()))',
        explain="The bundle (pipeline + threshold) travels together, so the cost-based decision rule is never separated from the model it was tuned for. `decide()` returns what a loan officer needs: a recommendation, a probability and (for declines) human-readable reasons.",
        level="⭐⭐⭐")

    nb.md("""## 🚀 Stretch goals
- **Calibration:** plot a reliability curve and compare logistic regression, a forest and a calibrated forest (`CalibratedClassifierCV`). Does τ* move toward theory (5/6)?
- **Reject inference:** the historical approvals are *decisions*, not repayment outcomes. Discuss how this biases the data and what extra data (actual defaults) would let you model risk directly.
- **Fairness mitigation:** compute group-specific thresholds that equalise TPR, and discuss whether that is legal/desirable; compare costs.
- **Monitoring plan:** which three quantities would you track monthly to detect drift (score distribution, approval rate by group, realised default rate)?
- **Counterfactual advice:** for a declined applicant, find the smallest change in `loan_amount_kes` that would flip the decision (a simple 'what would it take?' search).

## ✅ Self-assessment
- [ ] Did I catch and *explain* the leaky column?  - [ ] Is every threshold decision based on out-of-fold training data?
- [ ] Did I report fairness with uncertainty, and say what the audit cannot show?  - [ ] Could a declined customer understand my reason codes?""")

    nb.footer(["Leakage (a column recorded after the decision) can give a perfect-looking, useless model: interrogate every suspicious column.",
               "Asymmetric costs (5:1) move the optimal threshold to ≈ 0.77 and cut cost by ≈ 28 %; always choose it on out-of-fold data.",
               "Fairness needs metrics *and* uncertainty *and* a proxy check; protected attributes added nothing, so excluding them is the defensible policy.",
               "Reason codes and a model card make a model usable, and accountable."],
              "Project 4; Agricultural Price Regression", "P4_Agricultural_Price_Regression.ipynb")
    return nb
