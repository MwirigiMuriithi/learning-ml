from nbkit import Lesson


def make():
    nb = Lesson("08_Regression_Showdown", 8, "Regression Showdown",
                "All seven regression algorithms, one dataset, one fair fight, and how to pick a winner",
                120, "🟠 Intermediate",
                setup_extra="""import time, warnings
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV, learning_curve
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
warnings.filterwarnings("ignore")""")
    nb.header(
        objectives=[
            "Design a **fair** model comparison: same data, same folds, same preprocessing, same metrics, a baseline, and no peeking at the test set.",
            "Build one reusable preprocessing `Pipeline` + log-target wrapper that every algorithm shares.",
            "Compare all seven regressors on accuracy, stability, training time and prediction time.",
            "Tune each model fairly and understand why the *winner* depends on the data's hidden structure.",
            "Diagnose errors by segment (neighbourhood, price band) and with learning curves.",
            "Use a decision guide to choose an algorithm from the problem's characteristics.",
        ],
        prereqs="Lessons 00–07",
        hook=("You've met seven regression algorithms. Each has a fan club that swears it's the best. The honest answer is the one every good scientist gives: "
              "**\"It depends: let's run the experiment.\"**\n\n"
              "Today we hold a proper tournament on the Nairobi house-price data. A fair tournament needs rules (so no model gets an unfair advantage), "
              "a referee (cross-validation) and a post-match analysis (error diagnostics). "
              "And (spoiler) the winner may not be who you expect, and *why* it wins is the real lesson."),
        roadmap=["Rules of a fair fight", "The contestants", "Round 1: default settings", "Round 2: tuned",
                 "Final: the held-out test", "Post-match analysis (errors by segment)", "Learning curves: would more data help?",
                 "Decision guide", "Problems & quiz"])

    # ---------------------------------------------------------- 1 rules
    nb.section("📜", "1. Rules of a fair fight",
               "1. **Same data, same split.** One locked-away test set; cross-validation with *identical folds* for every model.\n"
               "2. **Same preprocessing** wrapped in a `Pipeline` (so there's no leakage and no model gets cleaner data).\n"
               "3. **Same target treatment.** Prices are multiplicative, so every model predicts $\\log(\\text{price})$ and we convert back with `exp` (done by `TransformedTargetRegressor`): metrics are always reported in **millions of KES**.\n"
               "4. **Several metrics.** R² (relative), RMSE (shillings, punishes big misses), **MdAPE** (median absolute % error: 'typically we're off by x %').\n"
               "5. **A baseline** (predict the mean): a contestant that can't beat it goes home.\n"
               "6. **Tune fairly:** every model gets a modest, comparable tuning budget.\n"
               "7. **Touch the test set once.** Choose using CV; the test score is a final audit, not a selection tool.")
    nb.code(r'''houses = pd.read_csv(data_path("house_prices.csv"))
df = houses[houses["sqft"] <= 6500].reset_index(drop=True)           # remove the 6 data-entry typos
num_cols = ["sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "has_pool", "lot_size_sqft", "distance_to_cbd_km"]
cat_cols = ["neighborhood", "condition"]
X, y = df[num_cols + cat_cols], df["price_million_kes"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("train:", Xtr.shape, " test (locked away):", Xte.shape)

# One shared preprocessing recipe: impute + scale numbers, impute + one-hot categories
preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])

def build(estimator, extra_steps=()):
    """preprocessing -> (optional extra steps) -> model, all predicting log(price) and returning price."""
    pipe = Pipeline([("prep", preprocess)] + list(extra_steps) + [("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)

cv = KFold(n_splits=5, shuffle=True, random_state=0)               # the SAME folds for every contestant
mdape = make_scorer(lambda a, b: np.median(np.abs(a - b) / a), greater_is_better=False)
scoring = {"r2": "r2", "rmse": "neg_root_mean_squared_error", "mdape": mdape}''')

    # ---------------------------------------------------------- 2 contestants
    nb.section("🥊", "2. The contestants",
               "Seven algorithms plus a baseline, each with **reasonable default-ish settings** (no tuning yet):")
    nb.code(r'''sqft_only = Pipeline([("pick", ColumnTransformer([("s", "passthrough", ["sqft"])])), ("lr", LinearRegression())])
contestants = {
    "0 Baseline (mean)":            build(DummyRegressor()),
    "1 Simple LR (sqft only)":      TransformedTargetRegressor(regressor=sqft_only, func=np.log, inverse_func=np.exp),
    "2 Multiple LR":                build(LinearRegression()),
    "3 Polynomial deg-2 + Ridge":   build(Ridge(alpha=10), [("poly", PolynomialFeatures(2, include_bias=False))]),
    "4 Decision Tree":              build(DecisionTreeRegressor(min_samples_leaf=5, random_state=0)),
    "5 Random Forest":              build(RandomForestRegressor(n_estimators=200, max_features=0.33, min_samples_leaf=2, random_state=0, n_jobs=-1)),
    "6 SVR (RBF)":                  build(SVR()),
    "7 KNN (k=7)":                  build(KNeighborsRegressor(n_neighbors=7)),
}
print(len(contestants), "contestants ready")''')

    # ---------------------------------------------------------- 3 round 1
    nb.section("🥇", "3. Round 1: default settings",
               "`cross_validate` gives us the three metrics plus **fit time** and **predict (score) time** for every fold.")
    nb.code(r'''rows = []
for name, model in contestants.items():
    r = cross_validate(model, Xtr, ytr, cv=cv, scoring=scoring)
    rows.append({"model": name, "cv_r2": r["test_r2"].mean(), "r2_std": r["test_r2"].std(), "rmse": -r["test_rmse"].mean(),
                 "mdape_%": -100 * r["test_mdape"].mean(), "fit_s": r["fit_time"].mean(), "predict_s": r["score_time"].mean()})
round1 = pd.DataFrame(rows).set_index("model")
display(round1.sort_values("cv_r2", ascending=False).round(3))''')
    nb.code(r'''fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
order = round1.sort_values("cv_r2").index
axes[0].barh(order, round1.loc[order, "cv_r2"], xerr=round1.loc[order, "r2_std"], color="#2563eb"); axes[0].set_title("CV R² (higher = better)"); axes[0].set_xlim(0, 1)
order2 = round1.sort_values("mdape_%", ascending=False).index
axes[1].barh(order2, round1.loc[order2, "mdape_%"], color="#f97316"); axes[1].set_title("Median abs % error (lower = better)")
axes[2].barh(round1.index, round1["fit_s"], color="#16a34a"); axes[2].set_title("Fit time per fold (s)"); axes[2].set_xscale("log")
for a in axes[1:]: a.set_yticklabels([])
plt.tight_layout(); plt.show()''')
    nb.md("""**Reading round 1**

- The **baseline** (R² ≈ 0) and **simple regression on size alone** are far behind: price depends on location, age, condition: more than one number.
- **Multiple LR** has a large ± in R². Look at the folds: one bad fold is dragging the average. Why? We predict $\\log(\\text{price})$ and convert back with `exp`: one extreme extrapolation on a luxury house becomes a *huge* error in shillings. R² is sensitive to such outliers, while the **MdAPE** (median) stays calm. *Always look at more than one metric.*
- The **polynomial + Ridge** model leads by a visible margin. Remember it is "just" linear regression on squares and pairwise products. Why would it do so well? Keep reading.
- **Random Forest** and **SVR** are close together; the single **Decision Tree** and **KNN** trail.""")
    nb.code(r'''for name in ["2 Multiple LR", "3 Polynomial deg-2 + Ridge", "5 Random Forest"]:
    r = cross_validate(contestants[name], Xtr, ytr, cv=cv, scoring="r2")
    print(f"{name:<28} per-fold R²: {np.round(r['test_score'], 3)}")''')

    nb.callout("key", "**Why does the quadratic model lead round 1?** Think about how house prices are built: *price ≈ neighbourhood multiplier × (quadratic function of size + extras)*. "
                      "Taking logs turns the multiplier into an additive effect, and a degree-2 model with **interactions** (neighbourhood × size, size², pool × size…) can represent this almost exactly. "
                      "A forest or SVR must *learn* that structure from 1 600 examples; the polynomial model *assumes* it, and the assumption is right. "
                      "**When a simple model's assumptions match the data-generating process, it beats fancier models at default settings.** Most real data is messier, so ensembles often win in practice, but you only know by testing. (And wait for round 2: a *well-tuned* flexible model can catch up.)")

    # ---------------------------------------------------------- 4 round 2 tuned
    nb.section("🎛️", "4. Round 2: fair tuning",
               "Each model gets a small grid search (3-fold CV *inside the training set*). Budgets are deliberately comparable: 6–18 combinations each.")
    nb.code(r'''grids = {
    "3 Polynomial deg-2 + Ridge": {"regressor__model__alpha": [0.1, 1, 10, 100]},
    "4 Decision Tree":            {"regressor__model__max_depth": [6, 10, 14, None], "regressor__model__min_samples_leaf": [3, 10, 20]},
    "5 Random Forest":            {"regressor__model__max_features": [0.2, 0.33, 0.6], "regressor__model__min_samples_leaf": [1, 3]},
    "6 SVR (RBF)":                {"regressor__model__C": [1, 10, 100], "regressor__model__gamma": [0.003, 0.01, 0.03], "regressor__model__epsilon": [0.02, 0.05]},
    "7 KNN (k=7)":                {"regressor__model__n_neighbors": [3, 5, 7, 10, 15], "regressor__model__weights": ["uniform", "distance"]},
}
tuned = dict(contestants)                    # untuned ones (baseline, simple LR, multiple LR) stay as they are
tuning_log = []
for name, grid in grids.items():
    t0 = time.perf_counter()
    gs = GridSearchCV(contestants[name], grid, cv=3, scoring="r2", n_jobs=1).fit(Xtr, ytr)
    tuned[name] = gs.best_estimator_
    tuning_log.append({"model": name, "best params": {k.split("__")[-1]: v for k, v in gs.best_params_.items()}, "inner CV R²": round(gs.best_score_, 3), "tuning time (s)": round(time.perf_counter() - t0, 1)})
display(pd.DataFrame(tuning_log).set_index("model"))''')
    nb.code(r'''rows = []
for name, model in tuned.items():
    r = cross_validate(model, Xtr, ytr, cv=cv, scoring=scoring)
    rows.append({"model": name, "cv_r2": r["test_r2"].mean(), "r2_std": r["test_r2"].std(), "rmse": -r["test_rmse"].mean(), "mdape_%": -100 * r["test_mdape"].mean(),
                 "fit_s": r["fit_time"].mean(), "predict_s": r["score_time"].mean()})
round2 = pd.DataFrame(rows).set_index("model")
compare = round1[["cv_r2"]].join(round2[["cv_r2"]], lsuffix="_default", rsuffix="_tuned"); compare["gain"] = compare["cv_r2_tuned"] - compare["cv_r2_default"]
display(compare.sort_values("cv_r2_tuned", ascending=False).round(3))''')
    nb.md("**A plot twist.** Tuning transforms the **SVR**: +0.07 R², from 3rd place to 1st. Its defaults (C = 1, a generic γ) were far from ideal for this data, while the polynomial/Ridge model, forest, tree and KNN gain only 0.005–0.02: "
          "the polynomial was already near its ceiling, and the forest is forgiving (Lesson 05). **Moral:** a model's *default* rank is not its *potential* rank: tune before you conclude. "
          "Notice also how close the finalists end up: a smooth, flexible kernel model (tuned SVR) and a smooth, well-specified polynomial model both capture the same underlying structure.")

    nb.md("### Is the winner *really* better? The one-standard-error rule\nDifferences smaller than the fold-to-fold noise are not worth celebrating. A practical rule: among all models within **one standard error** of the best, prefer the simplest/fastest.")
    nb.code(r'''best = round2["cv_r2"].idxmax()
se = round2.loc[best, "r2_std"] / np.sqrt(5)
tied = round2[round2["cv_r2"] >= round2.loc[best, "cv_r2"] - se].index.tolist()
print(f"Best by CV R²: {best}  (R² = {round2.loc[best, 'cv_r2']:.3f}, one SE ≈ {se:.4f})")
print("Statistically tied (within 1 SE):", tied)''')

    # ---------------------------------------------------------- 5 final test
    nb.section("🏆", "5. The final: the held-out test set",
               "We now audit the tuned models on the 399 houses locked away at the start. *In a real project you would score only your chosen model*: we show everyone for teaching, but the ranking was decided by CV above.")
    nb.code(r'''rows = []
for name, model in tuned.items():
    model.fit(Xtr, ytr); p = model.predict(Xte)
    rows.append({"model": name, "test R²": r2_score(yte, p), "test RMSE (M KES)": np.sqrt(mean_squared_error(yte, p)), "test MdAPE %": 100 * np.median(np.abs(yte - p) / yte)})
final = pd.DataFrame(rows).set_index("model"); display(final.sort_values("test R²", ascending=False).round(3))
champion = final["test R²"].idxmax(); print("Champion on the test set:", champion)''')
    nb.code(r'''champ_pred = tuned[champion].predict(Xte)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(yte, champ_pred, s=10, alpha=0.5); lim = [0, max(yte.max(), champ_pred.max())]; axes[0].plot(lim, lim, "r--"); axes[0].set_xlabel("true price (M KES)"); axes[0].set_ylabel("predicted"); axes[0].set_title(f"{champion}: predicted vs true")
res = (yte - champ_pred) / yte * 100; axes[1].hist(res, bins=40, edgecolor="white"); axes[1].axvline(0, color="r"); axes[1].set_title("Percentage error distribution"); axes[1].set_xlabel("(true − predicted) / true  (%)")
plt.tight_layout(); plt.show()''')

    # ---------------------------------------------------------- 6 post-match
    nb.section("🔬", "6. Post-match analysis, where do the errors live?",
               "A single average hides *who* the model fails. Slice the errors by segment.")
    nb.code(r'''test_df = df.loc[Xte.index].copy()
test_df["pred"] = champ_pred; test_df["ape"] = np.abs(test_df["price_million_kes"] - test_df["pred"]) / test_df["price_million_kes"]
by_hood = test_df.groupby("neighborhood").agg(n=("ape", "size"), mdape_pct=("ape", lambda s: 100 * s.median()), avg_price=("price_million_kes", "mean")).sort_values("mdape_pct")
display(by_hood.round(2))
test_df["band"] = pd.cut(test_df["price_million_kes"], [0, 15, 35, 1000], labels=["cheap (<15M)", "mid (15-35M)", "luxury (>35M)"])
print(test_df.groupby("band", observed=True)["ape"].agg(["size", lambda s: 100 * s.median()]).rename(columns={"<lambda_0>": "MdAPE %"}).round(2))''')
    nb.md("Segments with few examples (e.g. rare luxury properties) typically show larger errors: there's simply less data to learn from. In a real project this analysis tells you "
          "**where not to trust the model** and *what data to collect next*.")

    # ---------------------------------------------------------- 7 learning curves
    nb.section("📈", "7. Learning curves: would more data help?",
               "Train on 20 %, 40 %, … 100 % of the training data and watch the train/validation scores. "
               "If validation is still climbing, more data helps; if train and validation have converged at a mediocre level, you need a **better model or better features**.")
    nb.code(r'''fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, name in zip(axes, ["2 Multiple LR", "3 Polynomial deg-2 + Ridge", "5 Random Forest"]):
    sizes, tr, va = learning_curve(tuned[name], Xtr, ytr, train_sizes=[0.15, 0.3, 0.5, 0.75, 1.0], cv=3, scoring="r2", shuffle=True, random_state=0)
    ax.plot(sizes, tr.mean(1), "o-", label="train"); ax.plot(sizes, va.mean(1), "s-", label="validation"); ax.set_title(name); ax.set_xlabel("training rows"); ax.set_ylim(0.5, 1.02)
axes[0].set_ylabel("R²"); axes[0].legend(); plt.tight_layout(); plt.show()''')
    nb.md("- **Random Forest:** a big train/validation gap that narrows with more data → it is **variance-limited**; more data (or more regularisation) helps.\n"
          "- **Polynomial + Ridge:** tiny gap, plateau → close to its ceiling; more data won't change much, only richer features would.\n"
          "- **Multiple LR:** the validation line is shaky: a few extreme points dominate.")

    # ---------------------------------------------------------- 8 guide
    nb.section("🧭", "8. A decision guide for regression",
               "Everything we learned in Phase 4a, in one table (★ = strength, ✗ = weakness).")
    nb.table([
        ["**Simple LR**", "★★★★★", "✗ one feature", "★★★★★", "Explaining one relationship; baseline"],
        ["**Multiple LR**", "★★★★★", "✗ linear/additive only", "★★★★★", "Interpretable effects, quick baseline, tiny/clean data"],
        ["**Polynomial + Ridge**", "★★★★", "★ smooth curves & interactions", "★★★★", "Few features, smooth structure, when you can reason about interactions"],
        ["**Decision Tree**", "★★★★★ (readable rules)", "★ thresholds & interactions", "★★★★★", "Rules/explainability; building block; not usually the final model"],
        ["**Random Forest**", "★★ (importances, PDPs)", "★★★★ non-linear, interactions", "★★★ (slower predict)", "Strong default for tabular data with mixed types; low tuning"],
        ["**SVR (RBF)**", "★", "★★★★ smooth non-linear", "★★ (O(n²–n³) train)", "Small/medium n, smooth structure, noisy data, outliers"],
        ["**KNN**", "★★★★ ('comps')", "★★★ local patterns", "★★ (slow predict)", "Few relevant features, small n, local structure, quick prototypes"],
    ], ["Algorithm", "Interpretability", "Non-linearity / complexity it can capture", "Speed & scalability", "Reach for it when…"])
    nb.md("""**A practical routine** you can follow on any regression problem:

1. Start with a **baseline** and **multiple linear regression** (on log target if multiplicative).
2. Try a **Random Forest** (robust, little tuning): it tells you how much non-linearity you're missing.
3. If the forest ≫ linear: suspect interactions/non-linearity → engineer features (polynomial, interactions), or try **SVR/gradient boosting**.
4. If linear ≈ forest: keep the simple, interpretable model.
5. Decide with **CV**, report on a **test set once**, and check **errors by segment**.
6. Weigh accuracy against **interpretability, speed, maintenance and data size**: the best model on paper isn't always the best model in production.

*(Stretch: scikit-learn's `HistGradientBoostingRegressor`: a gradient-boosted tree ensemble; usually matches or beats forests on tabular data. It's not on our syllabus yet, but problem 4 lets you taste it.)*""")

    # ---------------------------------------------------------- problems
    nb.problem(
        "Write a reusable leaderboard",
        "Write `leaderboard(models, X, y, cv)` that runs `cross_validate` (scoring R² and RMSE: `{'r2': 'r2', 'rmse': 'neg_root_mean_squared_error'}`) for every model in the dict "
        "and returns a DataFrame indexed by model name with columns `cv_r2`, `r2_std`, `rmse` (positive!), sorted by `cv_r2` descending. "
        "Apply it to `trio = {k: contestants[k] for k in ['2 Multiple LR', '4 Decision Tree', '7 KNN (k=7)']}` and store the result in `lb`.",
        ["Loop over `models.items()`; collect a dict per model into a list; build `pd.DataFrame(rows).set_index('model')`.",
         "`r['test_rmse']` is *negative*: flip the sign.",
         "`.sort_values('cv_r2', ascending=False)`."],
        'def leaderboard(models, X, y, cv):\n    pass\n\ntrio = {k: contestants[k] for k in ["2 Multiple LR", "4 Decision Tree", "7 KNN (k=7)"]}\nlb = None   # your code\n',
        'def leaderboard(models, X, y, cv):\n    rows = []\n    for name, m in models.items():\n        r = cross_validate(m, X, y, cv=cv, scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"})\n'
        '        rows.append({"model": name, "cv_r2": r["test_r2"].mean(), "r2_std": r["test_r2"].std(), "rmse": -r["test_rmse"].mean()})\n'
        '    return pd.DataFrame(rows).set_index("model").sort_values("cv_r2", ascending=False)\n\ntrio = {k: contestants[k] for k in ["2 Multiple LR", "4 Decision Tree", "7 KNN (k=7)"]}\nlb = leaderboard(trio, Xtr, ytr, cv)\ndisplay(lb.round(3))',
        'check("3 rows, 3 columns", lambda: lb.shape == (3, 3))\ncheck("sorted best-first", lambda: lb["cv_r2"].is_monotonic_decreasing)\ncheck("RMSE is positive", lambda: (lb["rmse"] > 0).all())\ncheck("Multiple LR beats KNN in this trio", lambda: lb.loc["2 Multiple LR", "cv_r2"] > lb.loc["7 KNN (k=7)", "cv_r2"])',
        explain="Wrapping an experiment in a function is what turns a one-off notebook into a reusable tool: you will reuse this in every project.")

    nb.problem(
        "Pick a model under a constraint",
        "Business twist: a mobile app must **retrain the model on the phone in under `max_fit_s` seconds**. Write `choose_model(table, max_fit_s)` that, given a table like `round2` "
        "(columns `cv_r2`, `fit_s`), returns the **name of the most accurate model whose `fit_s <= max_fit_s`** (or `None` if none qualify). "
        "Store `pick_fast = choose_model(round2, 0.02)` and `pick_any = choose_model(round2, 1000)`.",
        ["Filter: `eligible = table[table['fit_s'] <= max_fit_s]`.",
         "If `eligible.empty` return `None`; otherwise `eligible['cv_r2'].idxmax()`.",
         "With a huge budget you should get the overall best model."],
        'def choose_model(table, max_fit_s):\n    pass\n\npick_fast = pick_any = None   # your code\n',
        'def choose_model(table, max_fit_s):\n    eligible = table[table["fit_s"] <= max_fit_s]\n    return None if eligible.empty else eligible["cv_r2"].idxmax()\n\npick_fast = choose_model(round2, 0.02); pick_any = choose_model(round2, 1000)\nprint(pick_fast, "|", pick_any)',
        'check("unconstrained pick is the overall CV champion", lambda: pick_any == round2["cv_r2"].idxmax())\n'
        'check("fast pick respects the time limit", lambda: pick_fast is None or round2.loc[pick_fast, "fit_s"] <= 0.02)\n'
        'check("fast pick is the best among eligible models", lambda: pick_fast is None or round2.loc[pick_fast, "cv_r2"] == round2[round2["fit_s"] <= 0.02]["cv_r2"].max())',
        explain="Real model selection is constrained optimisation: maximise accuracy subject to latency, memory, interpretability and cost limits.")

    nb.problem(
        "Where does the champion fail?",
        "Using `test_df` (it has columns `price_million_kes`, `pred`, `ape`) compute the median APE (as a **percentage**) for the three price bands defined in Section 6. "
        "Store them in a Series `band_mdape` (index = band labels) and set `worst_band` to the *name of the band with the highest* MdAPE.",
        ["`test_df.groupby('band', observed=True)['ape'].median() * 100`.",
         "`worst_band = band_mdape.idxmax()` (convert to `str`).",
         "Think about *why* that band is worst: how many training examples live there?"],
        'band_mdape = worst_band = None   # your code\n',
        'band_mdape = test_df.groupby("band", observed=True)["ape"].median() * 100\nworst_band = str(band_mdape.idxmax())\nprint(band_mdape.round(2).to_dict(), "->", worst_band)',
        'check("three bands", lambda: len(band_mdape) == 3)\ncheck("worst_band is the argmax", lambda: worst_band == str(band_mdape.idxmax()))\ncheck("all errors are plausible (< 25%)", lambda: (band_mdape < 25).all())',
        explain="Relative errors are usually largest in the sparsely populated tail (luxury homes): fewer training examples, higher variance in the price formula. Good news for reporting: you can tell clients exactly where the estimate is less reliable.",
        level="⭐⭐")

    nb.problem(
        "Taste of gradient boosting",
        "Add a `HistGradientBoostingRegressor(random_state=0)` to the tournament using `build(...)` (same preprocessing and log target). "
        "Compute its 5-fold CV R² with the lesson's `cv` and store it as `hgb_r2`. Also store `hgb_rank` = its rank (1 = best) among the **default-setting** CV R² values in `round1` plus itself.",
        ["`model = build(HistGradientBoostingRegressor(random_state=0))`; `cross_validate(model, Xtr, ytr, cv=cv, scoring='r2')['test_score'].mean()`.",
         "Combine: `all_r2 = pd.concat([round1['cv_r2'], pd.Series({'hgb': hgb_r2})])`.",
         "`rank = (all_r2 > hgb_r2).sum() + 1`."],
        'hgb_r2 = hgb_rank = None   # your code\n',
        'hgb_model = build(HistGradientBoostingRegressor(random_state=0))\nhgb_r2 = cross_validate(hgb_model, Xtr, ytr, cv=cv, scoring="r2")["test_score"].mean()\n'
        'all_r2 = pd.concat([round1["cv_r2"], pd.Series({"hgb": hgb_r2})]); hgb_rank = int((all_r2 > hgb_r2).sum() + 1)\nprint(round(hgb_r2, 3), "rank", hgb_rank, "of", len(all_r2))',
        'check("gradient boosting is strong (R² > 0.9)", lambda: hgb_r2 > 0.9)\ncheck("it beats the random forest at default settings", lambda: hgb_r2 > round1.loc["5 Random Forest", "cv_r2"])\ncheck("it ranks in the top 3", lambda: hgb_rank <= 3)',
        explain="Boosting builds trees *sequentially*, each correcting the previous ones' errors: a different ensemble strategy from bagging. It often wins on tabular data, but notice it still doesn't beat the model whose assumptions fit the data-generating process (the polynomial).",
        level="⭐⭐")

    nb.problem(
        "Read a learning curve",
        "For the tuned **Random Forest** (`tuned['5 Random Forest']`) and tuned **Polynomial + Ridge** (`tuned['3 Polynomial deg-2 + Ridge']`) compute `learning_curve` with "
        "`train_sizes=[0.2, 1.0]`, `cv=3`, `scoring='r2'`, `shuffle=True, random_state=0`. Store the **train − validation gap at full size** in `gap_rf` and `gap_poly`, "
        "and set `variance_limited = 'rf' if gap_rf > gap_poly else 'poly'`.",
        ["`sizes, tr, va = learning_curve(model, Xtr, ytr, train_sizes=[0.2, 1.0], cv=3, scoring='r2', shuffle=True, random_state=0)`.",
         "At full size use the last column: `tr.mean(1)[-1] - va.mean(1)[-1]`.",
         "A large gap indicates high variance (overfitting capacity)."],
        'gap_rf = gap_poly = variance_limited = None   # your code\n',
        'def gap(model):\n    sizes, tr, va = learning_curve(model, Xtr, ytr, train_sizes=[0.2, 1.0], cv=3, scoring="r2", shuffle=True, random_state=0)\n    return tr.mean(1)[-1] - va.mean(1)[-1]\n\n'
        'gap_rf = gap(tuned["5 Random Forest"]); gap_poly = gap(tuned["3 Polynomial deg-2 + Ridge"])\nvariance_limited = "rf" if gap_rf > gap_poly else "poly"\nprint(round(gap_rf, 3), round(gap_poly, 3), variance_limited)',
        'check("the forest has the larger train/validation gap", lambda: gap_rf > gap_poly)\ncheck("conclusion = rf", lambda: variance_limited == "rf")',
        explain="Deep trees can memorise their bootstrap samples, so even an averaged forest shows a gap; the heavily-regularised polynomial model has almost none. Gap ⇒ 'data-hungry'; no gap and a plateau ⇒ 'needs better features/model'.",
        level="⭐⭐⭐")

    nb.quiz([
        ("Why must every contestant use the same CV folds and preprocessing?",
         "Otherwise differences in score might come from luck of the split or from cleaner data rather than from the algorithm itself."),
        ("Multiple LR shows CV R² = 0.85 ± 0.12 while its MdAPE is a steady 8 %. What explains the contrast?",
         "A few extreme errors (after `exp` back-transformation) wreck R² in one fold; the *median* error is insensitive to those outliers."),
        ("Why can a degree-2 polynomial model beat a Random Forest on this data?",
         "The data-generating process is a neighbourhood multiplier times a quadratic size effect: exactly representable with log target + squares + interactions. A model whose assumptions match reality learns it from far less data."),
        ("Two models differ by 0.002 in CV R², and the fold standard deviation is 0.03. Which do you pick?",
         "They are statistically tied; pick the simpler/faster/more interpretable one."),
        ("Your learning curves show train R² = 0.99 and validation R² = 0.80, with the gap narrowing as data grows. What do you do?",
         "The model is variance-limited: get more data, simplify/regularise the model, or use bagging: it's not a feature problem."),
    ])

    nb.footer(["Fair comparison = same split, same preprocessing, same target treatment, a baseline, several metrics, tune fairly, test once.",
               "No algorithm wins everywhere: the polynomial+Ridge model led at default settings (its assumptions match the data), then a well-tuned SVR edged ahead: tuning can change the ranking.",
               "Look beyond one metric: R² can be fragile; MdAPE tells the business story; time and interpretability matter.",
               "Slice errors by segment and read learning curves to decide *what to do next*.",
               "You can now choose among all seven regressors from the data's size, structure and the business constraints."],
              "Classification Foundations & Metrics", "09_Classification_Foundations_and_Metrics.ipynb")
    return nb
