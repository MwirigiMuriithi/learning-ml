from nbkit import Lesson


def make():
    nb = Lesson("P2_Student_Performance_Prediction", 2, "Student Performance Prediction",
                "Phase 4 project · regression + classification + costs + a fairness audit + 'what-if' interventions",
                240, "🔴 Project", kind="Project",
                setup_extra="""import warnings, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score, recall_score, mean_absolute_error
warnings.filterwarnings("ignore")""")
    nb.header(
        objectives=[
            "Turn a school's request into **two** prediction tasks: an exam-score regression and an 'at-risk' early-warning classifier.",
            "Compare models fairly; discover that **feature engineering beats a fancier algorithm** when the structure is simple.",
            "Choose an alert threshold from the **costs** of missing an at-risk student versus a wasted tutoring slot.",
            "Run a basic **fairness audit** and judge whether differences between groups are real or just noise (bootstrap).",
            "Use a model for **what-if analysis**: and be honest about the difference between prediction and causation.",
        ],
        prereqs="Lessons 00–16 (especially 02, 09, 10, 16)",
        hook=("**From:** Mrs. Achieng, Deputy Principal, Maji Mazuri Secondary  \n**Subject:** Helping students before it's too late\n\n"
              "> *\"We have records for 1 200 past students: habits, attendance, background and final exam marks. Every year ~15 % of students fail: "
              "and we only find out when it's over. We have 40 tutoring places per term. Could you (a) predict exam scores, (b) flag students *early* who are likely to fail so we can offer tutoring, "
              "(c) tell us whether the flags are fair to boys and girls and to public/private pupils, and (d) tell us what we could actually *do* to help? A missed failing student costs us far more than a tutoring place offered unnecessarily.\"*\n\n"
              "This project is as much about **judgement** (costs, fairness, causation) as it is about modelling."))
    nb.md(r"""## 📦 The data: `data/student_performance.csv` (synthetic)

| Column | Meaning |
|---|---|
| `student_id` | identifier (not a feature) |
| `gender` (F/M), `school_type` (Public/Private) | demographics: handle with care! |
| `study_hours_per_week`, `attendance_pct`, `absences` | habits |
| `previous_exam_score` | last term's score (0–100) |
| `sleep_hours` | average nightly sleep (**some missing**) |
| `parental_education` | Primary / Secondary / College / University (**some missing**) |
| `internet_access`, `tutoring`, `extracurricular` | 0/1 flags |
| `motivation_1_10` | self-reported |
| **`final_score`** | **TARGET 1 (regression)**: final exam mark 0–100 |
| **`passed`** | final_score ≥ 50. We define **`at_risk = 1 − passed`** as **TARGET 2 (classification)** |

## 🎯 Deliverables
1. This notebook, fully run, with your reasoning in added markdown cells.
2. A saved regression pipeline `student_score_model.joblib`.
3. A one-page summary for the Deputy Principal (Milestone 7).

## 🧮 Rubric (100 points)
| Area | Pts | Excellent looks like |
|---|---|---|
| Exploration & data handling | 10 | Missing values handled in a pipeline; sensible encodings |
| Regression tournament | 15 | Baseline, ≥ 4 models, same folds, several metrics |
| Feature engineering | 15 | Evidence-based features (look at the plots!) with measured gain |
| Costs → threshold | 20 | Threshold from out-of-fold predictions, cost saved on test |
| Fairness audit | 15 | Group metrics + uncertainty; no over-claiming |
| What-if analysis | 10 | Sensible simulations + causal caveats |
| Communication | 15 | Honest, clear, actionable summary |

## 🗺️ Milestones
1. Explore · 2. Regression tournament · 3. Feature engineering · 4. At-risk classifier & costs · 5. Fairness audit · 6. What-if interventions · 7. Summary & packaging""")

    nb.section("📥", "Milestone 0: Shared helper code (given)",
               "This cell loads the data and provides `load_students()` and `make_prep()` so every milestone's checkpoint can rebuild its state.")
    nb.code(r'''EDU = {"Primary": 0, "Secondary": 1, "College": 2, "University": 3}
FEATURES = ["gender", "school_type", "study_hours_per_week", "attendance_pct", "previous_exam_score", "sleep_hours", "absences", "parental_education", "internet_access", "tutoring", "extracurricular", "motivation_1_10"]

def load_students():
    s = pd.read_csv(data_path("student_performance.csv"))
    X = s[FEATURES].copy(); X["parental_education"] = X["parental_education"].map(EDU)           # ordered category -> 0..3 (NaN stays NaN)
    return s, X

def make_prep(num_cols, cat_cols=("gender", "school_type")):
    return ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), list(num_cols)),
                              ("cat", OneHotEncoder(drop="first"), list(cat_cols))])

students, X0 = load_students(); print(students.shape); display(students.head())''')

    # ---------------------------------------------------------------- M1
    nb.section("🔍", "Milestone 1: Explore (10 pts)",
               "Look before you model. Which columns have gaps? What fraction fail? Which numeric features correlate most strongly with the final score? Plot at least two relationships: "
               "pay special attention to **study hours vs score** (is it a straight line?) and **sleep vs score** (is more always better?).")
    nb.code(r'''fig, axes = plt.subplots(1, 3, figsize=(16, 3.8))
axes[0].scatter(students["study_hours_per_week"], students["final_score"], s=8, alpha=0.4); axes[0].set_title("study hours vs score")
axes[1].scatter(students["sleep_hours"], students["final_score"], s=8, alpha=0.4, color="#16a34a"); axes[1].set_title("sleep vs score")
axes[2].scatter(students["previous_exam_score"], students["final_score"], s=8, alpha=0.4, color="#f97316"); axes[2].set_title("previous vs final score")
plt.tight_layout(); plt.show()''')
    nb.problem(
        "Audit and correlate",
        "1. `missing_cols`: list of columns of `students` with missing values.\n2. `pass_rate`: the share of students with `passed == 1` (a float).\n"
        "3. `corr`: a Series: the correlation of each numeric predictor (exclude `student_id`, `final_score`, `passed`) with `final_score`, sorted **descending**.\n4. `strongest`: the name of the feature with the strongest *absolute* correlation.",
        ["`students.isna().any()`; `students['passed'].mean()`.",
         "`students.drop(columns=['student_id', 'final_score', 'passed']).select_dtypes('number').corrwith(students['final_score'])`.",
         "`corr.abs().idxmax()`. (Remember: correlation measures *linear* association; the plots may show curvature it understates.)"],
        'missing_cols = pass_rate = corr = strongest = None   # your code\n',
        'missing_cols = [c for c in students.columns if students[c].isna().any()]; pass_rate = float(students["passed"].mean())\ncorr = students.drop(columns=["student_id", "final_score", "passed"]).select_dtypes("number").corrwith(students["final_score"]).sort_values(ascending=False)\nstrongest = corr.abs().idxmax()\nprint(missing_cols, round(pass_rate, 3), strongest); print(corr.round(2).to_dict())',
        'check("two columns with gaps: sleep_hours and parental_education", lambda: set(missing_cols) == {"sleep_hours", "parental_education"})\ncheck("about 85% pass", lambda: abs(pass_rate - 0.849) < 0.005)\ncheck("corr sorted descending, 9 numeric predictors", lambda: len(corr) == 9 and corr.is_monotonic_decreasing)\ncheck("previous exam score is by far the strongest predictor", lambda: strongest == "previous_exam_score" and corr["previous_exam_score"] > 0.7)',
        explain="Last term's score dominates (r ≈ 0.8), while study hours, attendance and absences are modest individually (r ≈ 0.2). The scatter of study hours shows *diminishing returns*: a curve a straight line can't capture, which Milestone 3 exploits.",
        level="⭐")

    # ---------------------------------------------------------------- M2
    nb.section("🥊", "Milestone 2: Regression tournament (15 pts)",
               "Use an 80/20 split (`random_state=42`) and the fixed folds `KFold(5, shuffle=True, random_state=0)`. Compare a **baseline** and at least **four models** (suggestions: Linear, Ridge, Random Forest, SVR, KNN) on **CV R²** and **CV MAE** (mean absolute error in marks; an error a teacher understands).")
    nb.code(r'''# 🔄 Checkpoint
students, X0 = load_students(); y_reg = students["final_score"]
Xtr, Xte, ytr, yte = train_test_split(X0, y_reg, test_size=0.2, random_state=42)
num_cols = [c for c in X0.columns if c not in ("gender", "school_type")]
prep = make_prep(num_cols); kf = KFold(5, shuffle=True, random_state=0)''')
    nb.problem(
        "Compare the regressors",
        "Build `models` (dict name → `Pipeline([('prep', prep), ('model', estimator)])`) with a `DummyRegressor` baseline and ≥ 4 real regressors. "
        "Then build `leader`: a DataFrame indexed by model name with columns `cv_r2` and `cv_mae` (positive marks), sorted by `cv_r2` descending, and `champion` = best model name.",
        ["`cross_val_score(model, Xtr, ytr, cv=kf, scoring='r2')` and `scoring='neg_mean_absolute_error'` (negate it).",
         "Candidates: `LinearRegression()`, `Ridge(10)`, `RandomForestRegressor(200, min_samples_leaf=3, random_state=0, n_jobs=-1)`, `SVR(C=10)`, `KNeighborsRegressor(10)`.",
         "Surprise warning: the simplest model may win. Why would that be (think about how the scores are generated: mostly additive effects)?"],
        'models = {}\nleader = champion = None   # your code\n',
        'mk = lambda est: Pipeline([("prep", prep), ("model", est)])\nmodels = {"Baseline": mk(DummyRegressor()), "Linear": mk(LinearRegression()), "Ridge": mk(Ridge(10)), "Random Forest": mk(RandomForestRegressor(200, min_samples_leaf=3, random_state=0, n_jobs=-1)), "SVR": mk(SVR(C=10)), "KNN": mk(KNeighborsRegressor(10))}\n'
        'leader = pd.DataFrame({n: {"cv_r2": cross_val_score(m, Xtr, ytr, cv=kf, scoring="r2").mean(), "cv_mae": -cross_val_score(m, Xtr, ytr, cv=kf, scoring="neg_mean_absolute_error").mean()} for n, m in models.items()}).T.sort_values("cv_r2", ascending=False)\nchampion = leader.index[0]\ndisplay(leader.round(3))',
        'check("baseline plus at least 4 real models", lambda: len(models) >= 5)\ncheck("leader sorted by R²", lambda: leader["cv_r2"].is_monotonic_decreasing)\ncheck("the baseline R² is ≈ 0", lambda: abs(leader.loc["Baseline", "cv_r2"]) < 0.05)\ncheck("the champion explains > 80% of the variance", lambda: leader["cv_r2"].iloc[0] > 0.80)\ncheck("MAE in marks is single digits", lambda: leader["cv_mae"].iloc[0] < 6)',
        explain="The scores are generated mostly by additive effects, so linear/Ridge regression leads (R² ≈ 0.83, MAE ≈ 3.4 marks); the forest and SVR are close but not better. The lesson of Lesson 08 again: a simple model whose assumptions match the data is hard to beat.",
        level="⭐⭐")

    # ---------------------------------------------------------------- M3
    nb.section("🛠️", "Milestone 3: Feature engineering with evidence (15 pts)",
               "Your exploration plots showed **diminishing returns** from study hours and an **optimal sleep** level (too little *and* too much hurt). A *linear* model can only represent straight lines: "
               "but we can hand it transformed features. Create `sqrt_study = √(study_hours)` and `sleep_dev2 = (sleep_hours − 7.5)²` and **measure** the gain with the same folds.")
    nb.code(r'''# 🔄 Checkpoint
students, X0 = load_students(); y_reg = students["final_score"]
Xtr, Xte, ytr, yte = train_test_split(X0, y_reg, test_size=0.2, random_state=42)
kf = KFold(5, shuffle=True, random_state=0); cat_cols = ("gender", "school_type")''')
    nb.problem(
        "Add the features and measure the gain",
        "1. Write `add_features(X)` returning a **copy** of `X` with the two new columns.\n"
        "2. `r2_before`: CV R² of a `LinearRegression` pipeline on the original features (`make_prep(num_cols)`); `r2_after`: CV R² on `add_features(Xtr)` (remember `num_cols` must now include the new columns). `gain = r2_after - r2_before`.\n"
        "3. `mae_after`: the CV MAE with the new features.",
        ["`X = X.copy(); X['sqrt_study'] = np.sqrt(X['study_hours_per_week']); X['sleep_dev2'] = (X['sleep_hours'] - 7.5) ** 2` (NaN sleep stays NaN → imputed in the pipeline).",
         "`num_cols2 = [c for c in add_features(Xtr).columns if c not in cat_cols]` and `make_prep(num_cols2)`.",
         "The gain should be around +0.02 R²: small in absolute terms, but it means ≈ 12 % of the remaining error vanished with two lines of code."],
        'def add_features(X):\n    pass\n\nr2_before = r2_after = gain = mae_after = None   # your code\n',
        'def add_features(X):\n    X = X.copy(); X["sqrt_study"] = np.sqrt(X["study_hours_per_week"]); X["sleep_dev2"] = (X["sleep_hours"] - 7.5) ** 2; return X\n\n'
        'num0 = [c for c in X0.columns if c not in cat_cols]; num2 = num0 + ["sqrt_study", "sleep_dev2"]\n'
        'lin0 = Pipeline([("prep", make_prep(num0)), ("m", LinearRegression())]); lin2 = Pipeline([("prep", make_prep(num2)), ("m", LinearRegression())])\n'
        'r2_before = cross_val_score(lin0, Xtr, ytr, cv=kf, scoring="r2").mean(); Xtr2 = add_features(Xtr)\nr2_after = cross_val_score(lin2, Xtr2, ytr, cv=kf, scoring="r2").mean(); gain = r2_after - r2_before\nmae_after = -cross_val_score(lin2, Xtr2, ytr, cv=kf, scoring="neg_mean_absolute_error").mean()\nprint(round(r2_before, 4), round(r2_after, 4), round(gain, 4), round(mae_after, 2))',
        'check("add_features returns a copy with both new columns", lambda: {"sqrt_study", "sleep_dev2"} <= set(add_features(Xtr).columns) and "sqrt_study" not in Xtr.columns)\ncheck("the new features improve CV R² by > 0.01", lambda: gain > 0.01)\ncheck("R² after > 0.84", lambda: r2_after > 0.84)\ncheck("MAE below 3.5 marks", lambda: mae_after < 3.5)',
        explain="Two physically motivated transformations made the *linear* model beat the forest and SVR comfortably. Knowing your domain (diminishing returns; U-shaped sleep) is worth more than a bigger algorithm.",
        level="⭐⭐")

    # ---------------------------------------------------------------- M4
    nb.section("🚨", "Milestone 4: The at-risk classifier and the alert threshold (20 pts)",
               "Target: `at_risk = 1 − passed` (≈ 15 % of students). **Costs** (from the Deputy Principal): failing to flag a student who then fails (**FN**) costs **4 units**; offering tutoring to a student who would have passed (**FP**) costs **1 unit**. "
               "Use a stratified 80/20 split (`random_state=42`), stratified 5-fold CV (`random_state=0`), compare **logistic regression** and a **random forest** by CV ROC-AUC, then choose the threshold from **out-of-fold** probabilities of the better model and audit the saving on the test set.")
    nb.code(r'''# 🔄 Checkpoint
students, X0 = load_students(); y_cls = 1 - students["passed"]
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X0, y_cls, test_size=0.2, stratify=y_cls, random_state=42)
num_cols = [c for c in X0.columns if c not in ("gender", "school_type")]; prep = make_prep(num_cols)
skf = StratifiedKFold(5, shuffle=True, random_state=0); taus = np.linspace(0.05, 0.95, 91)
def total_cost(y_true, p, tau, c_fn=4, c_fp=1):
    pred = p >= tau; return c_fn * np.sum(~pred & (y_true == 1)) + c_fp * np.sum(pred & (y_true == 0))
print("at-risk students: train", int(ytr_c.sum()), "| test", int(yte_c.sum()))''')
    nb.problem(
        "Classifier, threshold and savings",
        "1. `cls_auc`: dict `{'LogReg': ..., 'Forest': ...}` of CV ROC-AUC (`Pipeline([('prep', prep), ('model', est)])`; `LogisticRegression(max_iter=2000)` and `RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1)`); `best_cls` = winner's name.\n"
        "2. Using the winner, get out-of-fold probabilities `oof` with `cross_val_predict(..., cv=skf, method='predict_proba')[:, 1]` on the training set; `tau_star` = the threshold in `taus` that minimises `total_cost` on `(ytr_c, oof)`.\n"
        "3. Fit the winner on all training data. On the **test set** store `cost_default` (τ = 0.5), `cost_tuned` (τ = `tau_star`), `recall_tuned` and `flag_rate_tuned` (share of students flagged).",
        ["`cross_val_score(pipe, Xtr_c, ytr_c, cv=skf, scoring='roc_auc').mean()`.",
         "Theory for calibrated probabilities: τ* ≈ c_FP/(c_FP + c_FN) = 1/5 = 0.2.",
         "`flag_rate_tuned = np.mean(p_test >= tau_star)`. Compare with the 40-places-per-term capacity: how many students would be flagged out of 240 in the test cohort?"],
        'cls_auc = {}\nbest_cls = tau_star = cost_default = cost_tuned = recall_tuned = flag_rate_tuned = None   # your code\n',
        'pipes = {"LogReg": Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))]), "Forest": Pipeline([("prep", prep), ("model", RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1))])}\n'
        'cls_auc = {n: cross_val_score(p, Xtr_c, ytr_c, cv=skf, scoring="roc_auc").mean() for n, p in pipes.items()}; best_cls = max(cls_auc, key=cls_auc.get)\n'
        'oof = cross_val_predict(pipes[best_cls], Xtr_c, ytr_c, cv=skf, method="predict_proba")[:, 1]; tau_star = taus[int(np.argmin([total_cost(ytr_c.values, oof, t) for t in taus]))]\n'
        'pipes[best_cls].fit(Xtr_c, ytr_c); p_test = pipes[best_cls].predict_proba(Xte_c)[:, 1]\ncost_default = total_cost(yte_c.values, p_test, 0.5); cost_tuned = total_cost(yte_c.values, p_test, tau_star)\nrecall_tuned = recall_score(yte_c, p_test >= tau_star); flag_rate_tuned = float(np.mean(p_test >= tau_star))\n'
        'print(cls_auc, best_cls, round(tau_star, 2), cost_default, cost_tuned, round(recall_tuned, 3), round(flag_rate_tuned, 3))',
        'check("both models evaluated, AUC > 0.9", lambda: set(cls_auc) == {"LogReg", "Forest"} and min(cls_auc.values()) > 0.9)\ncheck("logistic regression wins on this additive problem", lambda: best_cls == "LogReg")\ncheck("tau* is well below 0.5 (misses are 4x costlier): between 0.1 and 0.4", lambda: 0.1 < tau_star < 0.4)\ncheck("the tuned threshold saves money on the test set", lambda: cost_tuned < cost_default)\ncheck("recall of at-risk students ≥ 0.85", lambda: recall_tuned >= 0.85)\ncheck("flag rate is realistic (15%-35%)", lambda: 0.15 < flag_rate_tuned < 0.35)',
        explain="Because misses cost 4× a wasted tutoring place, the optimal threshold drops to ≈ 0.2: we flag ~22 % of students to catch ≈ 92 % of those who would fail. Present this trade-off, and the capacity limit, to the school; they may prefer a slightly higher threshold if the 40 places are binding.",
        level="⭐⭐⭐")

    # ---------------------------------------------------------------- M5
    nb.section("⚖️", "Milestone 5: Fairness audit (15 pts)",
               "The school asked whether the early-warning system treats **girls and boys** and **public and private pupils** fairly. A basic audit compares, for each group, (a) the **flag rate** and (b) the **recall** among truly at-risk students "
               "(*equal opportunity*: does every at-risk student have the same chance of being flagged?).\n\n"
               "**But beware of tiny groups:** the test set has only ~36 at-risk students, so each gender has ~17–19. A gap between 100 % and 84 % recall could well be chance. "
               "Quantify the uncertainty with a **bootstrap** over the at-risk test students.")
    nb.code(r'''# 🔄 Checkpoint: reference classifier probabilities and threshold
students, X0 = load_students(); y_cls = 1 - students["passed"]
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X0, y_cls, test_size=0.2, stratify=y_cls, random_state=42)
prep = make_prep([c for c in X0.columns if c not in ("gender", "school_type")]); clf = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000))]).fit(Xtr_c, ytr_c)
p_test = clf.predict_proba(Xte_c)[:, 1]; tau_star = 0.22; flagged = p_test >= tau_star''')
    nb.problem(
        "Audit and bootstrap",
        "Using `p_test`, `tau_star`, `flagged`, `yte_c` and the group labels `Xte_c['gender']` / `Xte_c['school_type']`:\n"
        "1. `recall_by_gender` and `flag_rate_by_gender`: dicts `{'F': ..., 'M': ...}`; `recall_gap` = |recall F − recall M|.\n"
        "2. `n_at_risk_by_gender`: dict of how many truly at-risk students each gender has in the test set.\n"
        "3. `boot_ci`: a tuple (2.5th, 97.5th percentile) of the bootstrap distribution of (recall F − recall M): 2000 resamples **with replacement** of the at-risk test students *within each gender*, using `rng = np.random.default_rng(0)`.\n"
        "4. `gap_significant` = `True` if the CI excludes 0.",
        ["Recall for group g: among rows with `yte_c == 1` and `gender == g`, the share with `flagged == True`.",
         "Bootstrap: `flagsF = flagged[at_risk & (gender == 'F')]` (booleans); resample with `rng.choice(flagsF, size=len(flagsF), replace=True).mean()`; repeat for M; store the differences.",
         "`boot_ci = tuple(np.percentile(diffs, [2.5, 97.5]))`. If it spans 0, we cannot claim a real difference."],
        'recall_by_gender = flag_rate_by_gender = n_at_risk_by_gender = {}\nrecall_gap = boot_ci = gap_significant = None   # your code\n',
        'g = Xte_c["gender"].values; yv = yte_c.values\nrecall_by_gender = {k: float(flagged[(g == k) & (yv == 1)].mean()) for k in ["F", "M"]}; flag_rate_by_gender = {k: float(flagged[g == k].mean()) for k in ["F", "M"]}\n'
        'n_at_risk_by_gender = {k: int(((g == k) & (yv == 1)).sum()) for k in ["F", "M"]}; recall_gap = abs(recall_by_gender["F"] - recall_by_gender["M"])\n'
        'rng = np.random.default_rng(0); fF, fM = flagged[(g == "F") & (yv == 1)], flagged[(g == "M") & (yv == 1)]\ndiffs = [rng.choice(fF, len(fF)).mean() - rng.choice(fM, len(fM)).mean() for _ in range(2000)]\n'
        'boot_ci = tuple(np.percentile(diffs, [2.5, 97.5])); gap_significant = bool(boot_ci[0] > 0 or boot_ci[1] < 0)\nprint(recall_by_gender, flag_rate_by_gender, n_at_risk_by_gender, [round(v, 2) for v in boot_ci], gap_significant)',
        'check("recall by gender computed for both groups", lambda: set(recall_by_gender) == {"F", "M"} and all(0 <= v <= 1 for v in recall_by_gender.values()))\ncheck("small groups: 15-25 at-risk students per gender", lambda: all(10 <= v <= 25 for v in n_at_risk_by_gender.values()))\ncheck("the raw gap looks big (> 0.1)", lambda: recall_gap > 0.1)\ncheck("...but the bootstrap CI reaches zero (lower end ≈ 0): the evidence is weak / borderline", lambda: boot_ci[0] <= 0.01 and boot_ci[1] > 0.1 and gap_significant is False)',
        explain="Recall is 100 % for girls and 84 % for boys (alarming at first sight. With only 17–19 at-risk students per group the 95 % bootstrap interval for the gap runs from 0.00 to about 0.32: its lower end *touches* zero, so the evidence is **borderline/weak, not conclusive** (note the artefact: girls' recall is exactly 100 %, so all resampling variation comes from the boys). We cannot claim the model is biased; we also cannot claim it is fair. The responsible report says: 'a gap of up to ~30 points in recall cannot be ruled out with so few cases; no clear significant gap; monitor as more data arrives and consider collecting more labelled outcomes.' (In the synthetic data gender has no effect on scores) real audits don't know the truth!)",
        level="⭐⭐⭐")
    nb.md("✍️ *In a markdown cell: repeat the audit for `school_type` and write 3 sentences on what you would tell the school, including what you would **not** claim.*")

    # ---------------------------------------------------------------- M6
    nb.section("🔮", "Milestone 6: What-if interventions (10 pts)",
               "The school asks: *\"What could we actually do?\"* We can use the regression model for **simulation**: change one lever for each student (keeping everything else fixed) and see the predicted change in score. "
               "Three levers: **tutoring** (for students currently without it), **+5 study hours per week**, and **+10 attendance points** (capped at 100).\n\n"
               "⚠️ **Prediction ≠ causation.** The model learned *associations* in observational data. It's plausible that tutoring causes gains, but students who chose tutoring may differ in unmeasured ways. "
               "Present these as *hypotheses to test with a pilot*, not guaranteed effects.")
    nb.code(r'''# 🔄 Checkpoint: the engineered-feature linear model from Milestone 3
students, X0 = load_students(); y_reg = students["final_score"]
Xtr, Xte, ytr, yte = train_test_split(X0, y_reg, test_size=0.2, random_state=42)
def add_features(X):
    X = X.copy(); X["sqrt_study"] = np.sqrt(X["study_hours_per_week"]); X["sleep_dev2"] = (X["sleep_hours"] - 7.5) ** 2; return X
num2 = [c for c in add_features(Xtr).columns if c not in ("gender", "school_type")]
lin_final = Pipeline([("prep", make_prep(num2)), ("m", LinearRegression())]).fit(add_features(Xtr), ytr)
print("test R² of the model used for what-if analysis:", round(lin_final.score(add_features(Xte), yte), 3))''')
    nb.problem(
        "Simulate three interventions",
        "On the **test students**, using `lin_final` and `add_features` (recompute engineered features after changing a raw column!):\n"
        "- `gain_tutoring`: mean predicted score gain from giving tutoring to the students with `tutoring == 0` (set it to 1);\n"
        "- `gain_study5`: mean gain from +5 `study_hours_per_week` for **all** test students;\n"
        "- `gain_attend10`: mean gain from +10 `attendance_pct` (capped at 100) for all test students;\n"
        "- `best_lever`: the name (`'tutoring'`, `'study_hours'` or `'attendance'`) with the largest gain.",
        ["`base = lin_final.predict(add_features(Xte))`; copy `Xte`, change the column, recompute `lin_final.predict(add_features(X_changed))`, subtract.",
         "For tutoring only average over the rows where `Xte['tutoring'] == 0`.",
         "`np.minimum(100, Xte['attendance_pct'] + 10)` handles the cap."],
        'gain_tutoring = gain_study5 = gain_attend10 = best_lever = None   # your code\n',
        'base = lin_final.predict(add_features(Xte))\nXt = Xte.copy(); Xt["tutoring"] = 1; gain_tutoring = float((lin_final.predict(add_features(Xt)) - base)[(Xte["tutoring"] == 0).values].mean())\n'
        'Xs = Xte.copy(); Xs["study_hours_per_week"] += 5; gain_study5 = float((lin_final.predict(add_features(Xs)) - base).mean())\n'
        'Xa = Xte.copy(); Xa["attendance_pct"] = np.minimum(100, Xa["attendance_pct"] + 10); gain_attend10 = float((lin_final.predict(add_features(Xa)) - base).mean())\n'
        'best_lever = max({"tutoring": gain_tutoring, "study_hours": gain_study5, "attendance": gain_attend10}.items(), key=lambda kv: kv[1])[0]\nprint(round(gain_tutoring, 2), round(gain_study5, 2), round(gain_attend10, 2), best_lever)',
        'check("tutoring gain is about +2.8 marks (2-4)", lambda: 2.0 < gain_tutoring < 4.0)\ncheck("+5 study hours gives a positive gain (1-3)", lambda: 1.0 < gain_study5 < 3.0)\ncheck("+10 attendance points gives a positive gain", lambda: gain_attend10 > 0.5)\ncheck("tutoring is the biggest lever", lambda: best_lever == "tutoring")',
        explain="Per student, tutoring is worth ≈ +2.8 marks, five extra weekly study hours ≈ +1.9 (diminishing returns: √ transform), ten attendance points ≈ +1.3. For a student at 47 the 3 marks matter, which links back to the classifier: *target* the students near the pass mark.",
        level="⭐⭐")

    # ---------------------------------------------------------------- M7
    nb.section("📦", "Milestone 7: Summary and packaging (15 pts)",
               "Write what Mrs. Achieng will actually read. Save the regression pipeline, and compile a structured summary.")
    nb.problem(
        "Package and summarise",
        "1. Save `lin_final` to `'student_score_model.joblib'`, reload it as `loaded`, and predict the score of this student into `demo_score`: female, Public school, 9 study hours/week, 82 % attendance, previous score 55, 7 h sleep, 6 absences, 'Secondary' parental education (→ 1), internet 1, tutoring 0, extracurricular 1, motivation 6. "
        "(Build a one-row DataFrame with the **raw** columns of `X0`, then apply `add_features`.)\n"
        "2. Fill `summary`: a dict with keys `question`, `answer`, `threshold_policy`, `fairness`, `caveats`: each a clear, honest sentence or two for the Deputy Principal.",
        ["`joblib.dump(lin_final, 'student_score_model.joblib'); loaded = joblib.load('student_score_model.joblib')`.",
         "The row must have the same columns/order as `X0`: `pd.DataFrame([{...}])[list(X0.columns)]`.",
         "Keep it honest: the threshold is a cost trade-off, fairness was *tested but under-powered*, what-if gains are associations."],
        'loaded = demo_score = None\nsummary = {}   # your code\n',
        'joblib.dump(lin_final, "student_score_model.joblib"); loaded = joblib.load("student_score_model.joblib")\nrow = pd.DataFrame([{"gender": "F", "school_type": "Public", "study_hours_per_week": 9, "attendance_pct": 82, "previous_exam_score": 55, "sleep_hours": 7, "absences": 6, "parental_education": 1, "internet_access": 1, "tutoring": 0, "extracurricular": 1, "motivation_1_10": 6}])[list(X0.columns)]\n'
        'demo_score = float(loaded.predict(add_features(row))[0])\n'
        'summary = {"question": "Can we predict final marks and flag students likely to fail early enough to help?",\n           "answer": f"Yes: marks are predicted to within about 3.4 marks on average (R² ≈ 0.85) and a logistic-regression early-warning flag catches ~90% of students who later fail.",\n'
        '           "threshold_policy": "Because missing a failing student costs 4x a wasted tutoring place, we flag students with failure probability ≥ 0.22 (about one in five students); raise it if the 40 tutoring places are binding.",\n'
        '           "fairness": "Flag rates and recall were compared by gender and school type; the recall gap by gender (100% vs 84%) was borderline and not conclusive because groups are tiny (≈ 18 at-risk students each), so we cannot rule out a real gap: monitor each term and collect more outcomes.",\n'
        '           "caveats": "Estimated gains from tutoring (+2.8 marks), study (+1.9) and attendance (+1.3) are associations from past data, not proven effects; pilot before scaling. Data are synthetic."}\nprint(round(demo_score, 1), "predicted marks")',
        'import os\ncheck("pipeline saved and reloadable", lambda: os.path.exists("student_score_model.joblib") and loaded is not None)\ncheck("the demo student is predicted around the pass mark (45-65)", lambda: 45 < demo_score < 65)\ncheck("summary has the five sections, each a real sentence", lambda: set(summary) == {"question", "answer", "threshold_policy", "fairness", "caveats"} and all(isinstance(v, str) and len(v) > 40 for v in summary.values()))',
        explain="A good summary answers the client's four questions in plain language, states the cost logic behind the alert threshold, reports the fairness audit *with its limits*, and flags causal caveats. That honesty is what makes a model trustworthy to non-technical decision makers.",
        level="⭐⭐")

    nb.md("""## 🚀 Stretch goals
- **Multi-class outcome:** convert `final_score` into grades (A–E) and train a multi-class classifier; examine the confusion matrix for off-by-one errors.
- **Early-warning timing:** drop `attendance_pct`, `absences` and `previous_exam_score` to simulate a prediction *at the start of term*; how much accuracy do you lose, and is the alert still useful?
- **Calibrate the alert probabilities** with `CalibratedClassifierCV` and check whether the threshold shifts toward theory (0.2).
- **Capacity-constrained policy:** with exactly 40 places in a cohort of 240, rank students by risk and offer tutoring to the top 40; compute the recall you achieve and compare to the cost-optimal policy.
- **Bias mitigation:** retrain without `gender` and `school_type`; do accuracy and group metrics change? Which *proxy* variables might still carry the information?

## ✅ Self-assessment
- [ ] Did the threshold come from out-of-fold training predictions (not the test set)?  - [ ] Did I report uncertainty for group comparisons?
- [ ] Did I separate prediction from causation in the what-if section?  - [ ] Would Mrs. Achieng understand my summary without a statistics degree?""")

    nb.footer(["Simple, well-engineered features (√ study hours, (sleep − 7.5)²) let a linear model beat forests and SVR.",
               "Cost-aware thresholds (FN 4× FP → τ ≈ 0.2) turn a classifier into a policy; remember capacity limits.",
               "Fairness audits need uncertainty: small groups produce alarming-looking but inconclusive gaps.",
               "What-if simulations are hypotheses about associations: pilot before claiming causal effects."],
              "Project 3; Loan Approval Classification", "P3_Loan_Approval_Classification.ipynb")
    return nb
