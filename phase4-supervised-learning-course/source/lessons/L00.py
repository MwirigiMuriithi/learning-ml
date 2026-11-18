from nbkit import Lesson


def make():
    nb = Lesson("00_Welcome_and_Foundations", 0, "Welcome & Foundations of Supervised Learning",
                "The mental model, the workflow, and the toolkit you will use for the rest of the course",
                75, "🟢 Beginner", setup_extra="")
    nb.header(
        objectives=[
            "Explain supervised learning, features, targets, regression and classification in plain English.",
            "Follow the 7-step supervised learning workflow.",
            "Use the scikit-learn `fit` / `predict` / `score` pattern with confidence.",
            "Split data properly and explain *why* we never grade a model on the data it studied.",
            "Compute MAE, MSE, RMSE and R² by hand and with scikit-learn.",
            "Diagnose underfitting vs overfitting using the bias–variance idea.",
            "Use cross-validation and baselines to judge whether a model is actually good.",
        ],
        prereqs="Python basics, NumPy & Pandas (Phases 1–3)",
        hook=("Picture a mama mboga at a Nairobi market. After years of selling tomatoes she can *just tell* what "
              "tomorrow's price will be: it rained heavily last month, fuel went up, it's end of the month. Nobody "
              "ever handed her a formula. She learned from **thousands of examples** where the answer (the price) "
              "was eventually revealed.\n\n**That is supervised learning**: show a computer many examples *with answers*, "
              "and let it discover the pattern, so it can answer correctly for examples it has never seen.\n\n"
              "In the next two weeks you will meet the 13 most important algorithms that do exactly that. "
              "But first, we need a shared language and a safety net. Let's build them."),
        roadmap=["What is supervised learning?", "The 7-step workflow", "The scikit-learn API",
                 "Train / test split (the exam analogy)", "Metrics: how wrong are we?", "Baselines",
                 "Bias–variance: underfitting vs overfitting", "Cross-validation",
                 "Pipelines (preview)", "Map of the 13 algorithms", "Worked examples, problems, quiz"])

    # ------------------------------------------------------------------ 1
    nb.section("🧠", "1. What is supervised learning?",
               "In supervised learning every training example is a pair **(x, y)**:\n\n"
               "- **x**: the *features* (inputs): things we know when we need to predict.\n"
               "- **y**: the *target* (label, output): the thing we want to predict.\n\n"
               "The algorithm learns a function **f** such that **f(x) ≈ y**, and (crucially) it should work on *new* x.")
    nb.table([
        ["Regression", "A **number** (continuous)", "House price, student score, tomorrow's maize price", "Phase 4a (Lessons 01–08)"],
        ["Classification", "A **category** (discrete)", "Loan approved / rejected, spam / not spam, flower species", "Phase 4b (Lessons 09–16)"],
    ], ["Task", "Target type", "Examples", "Where in this course"])
    nb.md("Let's look at a real (small) dataset. **Advertising**: how much a company spent on TV, radio and newspaper ads (in thousands) "
          "and how many thousand units it then sold.")
    nb.code('''ads = pd.read_csv(data_path("advertising.csv"))
print("Shape:", ads.shape, " -> 200 examples, 4 columns")
ads.head()''')
    nb.md("Here the **features** are the three budgets and the **target** is `sales_k_units`. Because the target is a number, "
          "this is a **regression** problem.")
    nb.callout("think", "Suppose a bank wants to predict whether a customer will default on a loan. Is that regression or classification? "
                        "What would the target look like? *(Answer at the bottom of the notebook.)*")

    # ------------------------------------------------------------------ 2
    nb.section("🧭", "2. The 7-step workflow",
               "Every project in this course, and most in your career, follows the same rhythm. Memorise it:")
    nb.md("""```
 1. FRAME      →  What exactly are we predicting? How will we measure success?
 2. EXPLORE    →  Look at the data: distributions, missing values, weird rows.
 3. PREPARE    →  Clean, encode categories, scale numbers (inside a Pipeline!).
 4. SPLIT      →  Lock away a test set. Do NOT touch it until the very end.
 5. TRAIN      →  Fit one or more models on the training set.
 6. EVALUATE   →  Compare using validation / cross-validation. Tune hyper-parameters.
 7. DELIVER    →  One final test-set score, error analysis, and a clear explanation.
```""")
    nb.callout("key", "Most beginners spend 90 % of their energy on step 5 (the fancy algorithm). Professionals spend it on steps 1–4 and 6. "
                      "A mediocre algorithm on well-prepared data beats a brilliant algorithm on messy data almost every time.")

    # ------------------------------------------------------------------ 3
    nb.section("🔧", "3. The scikit-learn API: one pattern to rule them all",
               "Scikit-learn's genius is **consistency**. Every model, from linear regression to SVMs, speaks the same three verbs:\n\n"
               "| Verb | What it does |\n|---|---|\n| `model.fit(X, y)` | **Learn** from the training data |\n"
               "| `model.predict(X_new)` | **Predict** targets for new rows |\n| `model.score(X, y)` | **Grade** itself (R² for regression, accuracy for classification) |\n\n"
               "Conventions: `X` is **2-D** (rows × features): even when you have one feature, and `y` is **1-D**.")
    nb.code('''from sklearn.linear_model import LinearRegression

X = ads[["tv_budget_k"]]          # 2-D: a DataFrame with ONE column (double brackets!)
y = ads["sales_k_units"]          # 1-D: a Series

model = LinearRegression()        # 1. choose the algorithm
model.fit(X, y)                   # 2. learn from examples
new_ad = pd.DataFrame({"tv_budget_k": [100, 200]})
print("Predicted sales for TV budgets 100k and 200k:", model.predict(new_ad).round(2))
print("R² on the data it learned from:", round(model.score(X, y), 3))''')
    nb.callout("trap", "`X = ads['tv_budget_k']` (single brackets) gives a 1-D Series and scikit-learn will complain: "
                       "*\"Expected 2D array, got 1D array instead\"*. Use **double brackets** `ads[['tv_budget_k']]`.")

    # ------------------------------------------------------------------ 4
    nb.section("📝", "4. Train / test split: the exam analogy",
               "Imagine a teacher who gives students the **exact exam questions** to study from, then grades them on those same questions. "
               "Everyone gets 100 %, but did anyone learn? Of course not: they may have just *memorised*.\n\n"
               "Machine learning has the same danger. So we **hide** part of the data (the **test set**), train on the rest "
               "(the **training set**), and grade on the hidden part. The test score estimates how the model will behave on *tomorrow's* real data.")
    nb.code('''from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X = ads[["tv_budget_k", "radio_budget_k", "newspaper_budget_k"]]
y = ads["sales_k_units"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("train:", X_train.shape, " test:", X_test.shape)

# A model that is *allowed* to memorise: an unlimited-depth decision tree
memoriser = DecisionTreeRegressor(random_state=RANDOM_STATE).fit(X_train, y_train)
print(f"R² on TRAIN data: {memoriser.score(X_train, y_train):.3f}   <- looks perfect!")
print(f"R² on TEST  data: {memoriser.score(X_test,  y_test ):.3f}   <- the honest number")''')
    nb.md("See the gap? A perfect training score paired with a much worse test score is the classic fingerprint of **overfitting** "
          "(we will explore it properly in section 7).")
    nb.callout("tip", "`random_state=42` makes the shuffle reproducible, so your results match your classmates' and your own from yesterday. "
                      "Typical splits: 80/20 or 70/30. With *tiny* datasets prefer cross-validation (section 8).")

    # ------------------------------------------------------------------ 5
    nb.section("📏", "5. Metrics for regression: how wrong are we?",
               "For each example, the **error** (residual) is $e_i = y_i - \\hat{y}_i$ (true minus predicted). We need one number that summarises "
               "all errors. The popular choices:")
    nb.md(r"""| Metric | Formula | Units | Personality |
|---|---|---|---|
| **MAE**: Mean Absolute Error | $\frac{1}{n}\sum \lvert y_i-\hat y_i\rvert$ | same as y | "On average I'm off by this much." Robust to outliers. |
| **MSE**: Mean Squared Error | $\frac{1}{n}\sum (y_i-\hat y_i)^2$ | y² | Punishes big mistakes heavily. Smooth → easy to optimise. |
| **RMSE**: Root MSE | $\sqrt{\text{MSE}}$ | same as y | MSE translated back to y's units. The most common report. |
| **R²**: Coefficient of determination | $1-\dfrac{\sum (y_i-\hat y_i)^2}{\sum (y_i-\bar y)^2}$ | none | "Fraction of variance explained." 1 = perfect, 0 = no better than predicting the mean, **can be negative**. |""")
    nb.callout("math", "R² compares your model to the laziest possible model: *always predict the mean $\\bar y$*. "
                       "The numerator is **your** squared error (SSE); the denominator is the lazy model's squared error (SST). "
                       "If you do as badly as the lazy model, SSE/SST = 1 and R² = 0.")
    nb.code('''from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = np.array([200., 250., 300., 410.])
y_pred = np.array([210., 240., 330., 380.])

errors = y_true - y_pred
mae  = np.mean(np.abs(errors))
mse  = np.mean(errors ** 2)
rmse = np.sqrt(mse)
sse  = np.sum(errors ** 2)
sst  = np.sum((y_true - y_true.mean()) ** 2)
r2   = 1 - sse / sst

print(f"errors : {errors}")
print(f"MAE={mae:.2f}  MSE={mse:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
print("--- scikit-learn agrees? ---")
print(f"MAE={mean_absolute_error(y_true, y_pred):.2f}  MSE={mean_squared_error(y_true, y_pred):.2f}  "
      f"RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.2f}  R²={r2_score(y_true, y_pred):.4f}")''')
    nb.callout("tip", "**Which metric should I report?** Use **RMSE or MAE** to tell stakeholders *how many shillings / marks / units* you're off by. "
                      "Use **R²** to compare models on the same data. If a few huge errors would be catastrophic, favour RMSE; "
                      "if you want a typical-error figure that outliers don't distort, favour MAE.")

    # ------------------------------------------------------------------ 6
    nb.section("🪜", "6. Baselines: the bar every model must clear",
               "Before celebrating an R² of 0.7, ask: *compared to what?* A **baseline** is the dumbest sensible model. "
               "For regression, that's \"always predict the training mean\". If your fancy model can't beat it, something is wrong.")
    nb.code('''from sklearn.dummy import DummyRegressor

base = DummyRegressor(strategy="mean").fit(X_train, y_train)
lin  = LinearRegression().fit(X_train, y_train)
for name, m in [("Baseline (mean)", base), ("Linear regression", lin)]:
    pred = m.predict(X_test)
    print(f"{name:<20} RMSE={np.sqrt(mean_squared_error(y_test, pred)):.3f}   R²={r2_score(y_test, pred):.3f}")''')
    nb.md("The baseline's R² is ≈ 0 on the test set (by definition, it explains none of the variance). Linear regression slashes the error: "
          "*now* we can say it learned something real.")

    # ------------------------------------------------------------------ 7
    nb.section("⚖️", "7. Underfitting, overfitting and the bias–variance trade-off",
               "Let's run the most important experiment in machine learning. We create data from a known curve, "
               "$y = \\sin(2\\pi x) + \\text{noise}$, and fit polynomials of increasing flexibility (degree 1 = a straight line, degree 14 = a wiggly monster).")
    nb.code('''from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
true_f = lambda x: np.sin(2 * np.pi * x)
x_tr = rng.uniform(0, 1, 30);   y_tr = true_f(x_tr) + rng.normal(0, 0.3, 30)     # small training set
x_te = rng.uniform(0, 1, 500);  y_te = true_f(x_te) + rng.normal(0, 0.3, 500)    # big fresh test set

fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
xs = np.linspace(0, 1, 300)
for ax, (deg, label) in zip(axes, [(1, "Underfit (degree 1)"), (4, "Good fit (degree 4)"), (14, "Overfit (degree 14)")]):
    m = make_pipeline(PolynomialFeatures(deg), LinearRegression()).fit(x_tr[:, None], y_tr)
    ax.scatter(x_tr, y_tr, s=25, label="train")
    ax.plot(xs, true_f(xs), "g--", lw=1.5, label="truth")
    ax.plot(xs, m.predict(xs[:, None]), "r", lw=2.2, label="model")
    ax.set_ylim(-2, 2)
    tr = np.sqrt(mean_squared_error(y_tr, m.predict(x_tr[:, None])))
    te = np.sqrt(mean_squared_error(y_te, m.predict(x_te[:, None])))
    ax.set_title(f"{label}\\ntrain RMSE={tr:.2f} | test RMSE={te:.2f}")
axes[0].legend(); plt.tight_layout(); plt.show()''')
    nb.md("""- **Degree 1: underfitting (high bias):** too simple to capture the curve. Bad on train *and* test.
- **Degree 4: just right:** captures the trend, ignores the noise.
- **Degree 14: overfitting (high variance):** threads through every training dot (tiny train error) but swings wildly between them: terrible on new data.

Let's sweep *every* degree and draw the famous U-shaped curve.""")
    nb.code('''degrees = range(1, 15)
train_err, test_err = [], []
for d in degrees:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x_tr[:, None], y_tr)
    train_err.append(np.sqrt(mean_squared_error(y_tr, m.predict(x_tr[:, None]))))
    test_err.append(np.sqrt(mean_squared_error(y_te, m.predict(x_te[:, None]))))

plt.plot(degrees, train_err, "o-", label="Train error")
plt.plot(degrees, test_err, "s-", label="Test error")
plt.yscale("log"); plt.xlabel("Model complexity (polynomial degree)"); plt.ylabel("RMSE (log scale)")
plt.axvline(int(np.argmin(test_err)) + 1, color="gray", ls=":", label="best test error")
plt.title("The bias–variance U-curve"); plt.legend(); plt.show()
print("Best degree on the test set:", int(np.argmin(test_err)) + 1)''')
    nb.callout("key", """Training error **always** keeps falling as the model gets more flexible. Test error falls, bottoms out, then **rises**.

- **Bias** = error from wrong assumptions (model too simple) → *underfitting*.
- **Variance** = error from being too sensitive to the particular training sample (model too flexible) → *overfitting*.

Every algorithm in this course has a "complexity dial" (tree depth, *k* in KNN, *C* in SVM...). Your job is to find the sweet spot.""")
    nb.md("**A quick diagnostic table**: memorise it, you will use it constantly:")
    nb.table([
        ["Low", "Low", "✅ Good fit", "Ship it (after checking the test set once)"],
        ["High", "High", "🥶 Underfitting", "More flexible model, better features, less regularisation"],
        ["Low", "High (big gap)", "🔥 Overfitting", "More data, simpler model, regularisation, pruning, bagging"],
    ], ["Train error", "Test error", "Diagnosis", "Typical cure"])

    # ------------------------------------------------------------------ 8
    nb.section("🔁", "8. Cross-validation: a fairer exam",
               "One train/test split can be lucky or unlucky. **K-fold cross-validation** repeats the exam K times: split the data into K "
               "equal folds, train on K−1 folds, test on the remaining one, rotate, and average the K scores. "
               "You get a score **and** an idea of how stable it is (the standard deviation).")
    nb.md("""```
Fold 1: [TEST ][train][train][train][train]
Fold 2: [train][TEST ][train][train][train]
Fold 3: [train][train][TEST ][train][train]     → average the 5 test scores
Fold 4: [train][train][train][TEST ][train]
Fold 5: [train][train][train][train][TEST ]
```""")
    nb.code('''from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for name, mdl in [("Baseline", DummyRegressor()), ("Linear Regression", LinearRegression()),
                  ("Unlimited Decision Tree", DecisionTreeRegressor(random_state=0))]:
    scores = cross_val_score(mdl, X, y, cv=cv, scoring="r2")
    print(f"{name:<24} R² per fold: {np.round(scores, 3)}  ->  mean={scores.mean():.3f} ± {scores.std():.3f}")''')
    nb.callout("tip", "Use `scoring='neg_root_mean_squared_error'` for RMSE. Scikit-learn *maximises* scores, so error metrics come with a minus sign "
                      "(multiply by −1 to read them).")

    # ------------------------------------------------------------------ 9
    nb.section("🚰", "9. Pipelines (a preview you will use everywhere)",
               "Real data needs preprocessing (scaling, encoding, imputing). The classic mistake: compute the scaling statistics on the **whole** "
               "dataset, then split. The test set has now quietly influenced training: this is **data leakage**. "
               "A `Pipeline` bundles preprocessing + model so everything is re-learned *inside* each training fold.")
    nb.code('''from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

pipe = Pipeline([
    ("scale", StandardScaler()),                 # learned on the training part only
    ("knn",   KNeighborsRegressor(n_neighbors=7)),
])
scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2")
print(f"KNN with scaling inside a Pipeline: R² = {scores.mean():.3f} ± {scores.std():.3f}")''')

    # ------------------------------------------------------------------ 10
    nb.section("🗺️", "10. The map of the 13 algorithms you will master",
               "Think of each algorithm as a personality. By the end of the course you'll know which one to hire for which job.")
    nb.table([
        ["01", "Simple Linear Regression", "Regression", "The straight-line thinker: one feature, one slope."],
        ["02", "Multiple Linear Regression", "Regression", "Straight lines in many dimensions; very interpretable."],
        ["03", "Polynomial Regression", "Regression", "A linear model wearing curved glasses."],
        ["04", "Decision Tree (Regression)", "Regression", "20 questions → average of the leaf."],
        ["05", "Random Forest (Regression)", "Regression", "A committee of decorrelated trees."],
        ["06", "Support Vector Regression", "Regression", "Fit within a tube; only the outliers of the tube matter."],
        ["07", "KNN (Regression)", "Regression", "Ask your k nearest neighbours and average."],
        ["10", "Logistic Regression", "Classification", "Squash a linear score into a probability."],
        ["11", "Decision Tree (Classification)", "Classification", "20 questions → majority class of the leaf."],
        ["12", "Random Forest (Classification)", "Classification", "A committee vote."],
        ["13", "KNN (Classification)", "Classification", "You are the company you keep."],
        ["14", "Naive Bayes", "Classification", "Bayes' theorem + a bold independence assumption."],
        ["15", "Support Vector Machine", "Classification", "Find the widest street between classes."],
    ], ["Lesson", "Algorithm", "Task", "One-line personality"])

    # ------------------------------------------------------------------ 11 worked examples
    nb.section("🧪", "11. Worked examples")
    nb.md("""### Worked Example A: Computing metrics by hand

**Question.** A model predicts house prices (in millions): predictions $\\hat y = [12, 18, 25]$ for true prices $y = [10, 20, 30]$.
Compute MAE, RMSE and R².

**Solution.**

1. Errors $y-\\hat y = [-2, 2, 5]$.
2. MAE $= (2+2+5)/3 = 3.0$.
3. MSE $= (4+4+25)/3 = 11$ → RMSE $=\\sqrt{11}\\approx 3.317$.
4. $\\bar y = 20$; SST $= (10-20)^2+(20-20)^2+(30-20)^2 = 200$; SSE $= 4+4+25 = 33$.
5. R² $= 1 - 33/200 = \\mathbf{0.835}$.

Let's verify in code:""")
    nb.code('''y_a = np.array([10, 20, 30]); yh_a = np.array([12, 18, 25])
print("MAE :", mean_absolute_error(y_a, yh_a))
print("RMSE:", round(float(np.sqrt(mean_squared_error(y_a, yh_a))), 3))
print("R²  :", round(r2_score(y_a, yh_a), 3))''')
    nb.md("""### Worked Example B: Diagnosing a model from its scores

| Model | Train R² | Test R² |
|---|---|---|
| A | 0.42 | 0.40 |
| B | 0.99 | 0.55 |
| C | 0.86 | 0.84 |

- **A**: both low, no gap → **underfitting** (high bias). Needs a more flexible model or better features.
- **B**: huge gap → **overfitting** (high variance). Simplify, regularise, or get more data.
- **C**: both high, tiny gap → **good fit**. 

This reasoning is how you'll debug *every* model in the course.""")

    # ------------------------------------------------------------------ problems
    nb.problem(
        "Metrics by hand", 
        "A model predicts `y_pred = [2.5, 5, 4, 8]` for the true values `y_true = [3, 5, 2.5, 7]`.\n\n"
        "Compute **MAE**, **RMSE** and **R²** using only NumPy (no scikit-learn metrics). Store them in `mae`, `rmse`, `r2`.",
        ["Start with `errors = y_true - y_pred`.",
         "MAE is the mean of the *absolute* errors; RMSE is the square root of the mean of the *squared* errors.",
         "R² = 1 − SSE/SST where SSE = Σ(error²) and SST = Σ(y_true − mean(y_true))²."],
        'y_true = np.array([3, 5, 2.5, 7]); y_pred = np.array([2.5, 5, 4, 8])\n'
        'mae = None   # your code\nrmse = None  # your code\nr2 = None    # your code\nprint(mae, rmse, r2)',
        'y_true = np.array([3, 5, 2.5, 7]); y_pred = np.array([2.5, 5, 4, 8])\n'
        'errors = y_true - y_pred\nmae = np.mean(np.abs(errors))\nrmse = np.sqrt(np.mean(errors ** 2))\n'
        'sse = np.sum(errors ** 2); sst = np.sum((y_true - y_true.mean()) ** 2)\nr2 = 1 - sse / sst\nprint(mae, rmse, r2)',
        'check("MAE = 0.75", lambda: abs(mae - 0.75) < 1e-9)\n'
        'check("RMSE ≈ 0.9354", lambda: abs(rmse - 0.935414) < 1e-4)\n'
        'check("R² ≈ 0.7241", lambda: abs(r2 - 0.724138) < 1e-4)',
        explain="SSE = 3.5 and SST = 12.6875, so R² = 1 − 3.5/12.6875 ≈ 0.724. MAE averages |errors| = (0.5+0+1.5+1)/4 = 0.75.")

    nb.problem(
        "Split the advertising data",
        "Split `X` (the three budget columns) and `y` (sales) from the `ads` DataFrame into **80 % train / 20 % test** using "
        "`random_state=42`. Name the pieces `Xtr, Xte, ytr, yte`.",
        ["`train_test_split` returns four objects, in the order X_train, X_test, y_train, y_test.",
         "Use `test_size=0.2` (or `train_size=0.8`)."],
        'Xtr = Xte = ytr = yte = None   # your code\n',
        'Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)',
        'check("train has 160 rows", lambda: len(Xtr) == 160 and len(ytr) == 160)\n'
        'check("test has 40 rows",  lambda: len(Xte) == 40 and len(yte) == 40)\n'
        'check("same split as the lesson", lambda: list(Xtr.index[:3]) == list(X_train.index[:3]))',
        explain="`random_state=42` guarantees the same shuffle as the lesson's `X_train`, which is why the last check passes.")

    nb.problem(
        "Beat the baseline",
        "Using the lesson's `X_train, X_test, y_train, y_test`:\n\n"
        "1. Fit a `DummyRegressor(strategy='mean')` and a `LinearRegression()`.\n"
        "2. Store their **test-set RMSE** in `rmse_base` and `rmse_lin`.\n"
        "3. Set `improvement` = the percentage by which linear regression reduces the baseline RMSE.",
        ["Fit on the *train* split, evaluate on the *test* split.",
         "`np.sqrt(mean_squared_error(y_test, preds))` gives RMSE.",
         "improvement = (rmse_base − rmse_lin) / rmse_base × 100."],
        'rmse_base = rmse_lin = improvement = None   # your code\n',
        'dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)\nlin2 = LinearRegression().fit(X_train, y_train)\n'
        'rmse_base = np.sqrt(mean_squared_error(y_test, dummy.predict(X_test)))\n'
        'rmse_lin = np.sqrt(mean_squared_error(y_test, lin2.predict(X_test)))\n'
        'improvement = (rmse_base - rmse_lin) / rmse_base * 100\nprint(f"Linear regression cuts error by {improvement:.1f}%")',
        'check("linear regression beats the baseline", lambda: rmse_lin < rmse_base)\n'
        'check("improvement is > 50%", lambda: improvement > 50)',
        explain="TV and radio spend are strongly predictive of sales, so the linear model removes most of the unexplained variation.",
        level="⭐")

    nb.problem(
        "Write your own doctor",
        "Write a function `diagnose(train_score, test_score)` that returns one of the strings "
        "`'underfitting'`, `'overfitting'` or `'good fit'` using these rules:\n\n"
        "- if `train_score < 0.6` → `'underfitting'`\n- elif `train_score - test_score > 0.15` → `'overfitting'`\n- else → `'good fit'`",
        ["Check the underfitting rule *first* (a low training score is a problem regardless of the gap).",
         "Use `if / elif / else` and `return` the strings exactly as written."],
        'def diagnose(train_score, test_score):\n    # your code\n    pass\n',
        'def diagnose(train_score, test_score):\n    if train_score < 0.6:\n        return "underfitting"\n'
        '    elif train_score - test_score > 0.15:\n        return "overfitting"\n    return "good fit"',
        'check("model A (0.42 / 0.40) -> underfitting", lambda: diagnose(0.42, 0.40) == "underfitting")\n'
        'check("model B (0.99 / 0.55) -> overfitting",  lambda: diagnose(0.99, 0.55) == "overfitting")\n'
        'check("model C (0.86 / 0.84) -> good fit",     lambda: diagnose(0.86, 0.84) == "good fit")',
        explain="Real diagnosis is a judgement call, but encoding a rule of thumb makes you think about *which* symptoms matter.",
        level="⭐⭐")

    nb.quiz([
        ("Predicting whether a customer will default on a loan: regression or classification?",
         "**Classification**: the target is a category (default / no default)."),
        ("Your model has train R² = 0.97 and test R² = 0.61. What's wrong, and name two cures?",
         "**Overfitting.** Cures: collect more data, simplify the model (smaller depth / fewer polynomial terms), add regularisation, use bagging/ensembles, prune."),
        ("Why must the test set stay untouched until the very end?",
         "If we tune the model by peeking at the test set, it stops being an honest estimate of future performance: it becomes part of training."),
        ("Can R² be negative? What does that mean?",
         "Yes. It means the model is **worse than just predicting the mean**: usually a sign of severe overfitting or a bug."),
        ("What is data leakage, and how do Pipelines help?",
         "Leakage = information from outside the training fold (e.g. test statistics) influencing training. A Pipeline fits the preprocessing steps only on each training fold."),
    ])

    nb.footer(["Supervised learning = learning f(x) ≈ y from labelled examples; **regression** → number, **classification** → category.",
               "Follow the 7-step workflow; the algorithm is only step 5.",
               "`fit` → `predict` → `score` is the universal scikit-learn pattern.",
               "Always evaluate on data the model hasn't seen; always compare against a baseline.",
               "Underfitting = too simple, overfitting = too flexible; cross-validation helps you find the middle."],
              "Simple Linear Regression", "01_Simple_Linear_Regression.ipynb")
    return nb
