from nbkit import Lesson


def make():
    nb = Lesson("02_Multiple_Linear_Regression", 2, "Multiple Linear Regression",
                "Many features, one hyperplane: the workhorse of interpretable prediction",
                120, "🟢🟡 Beginner → Intermediate",
                setup_extra="from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV\nfrom sklearn.metrics import mean_squared_error, r2_score\nfrom sklearn.model_selection import train_test_split, cross_val_score\nfrom scipy import stats")
    nb.header(
        objectives=[
            "Write the model in matrix form and solve it with the **normal equation**.",
            "Implement multiple linear regression from scratch (normal equation *and* vectorised gradient descent).",
            "Interpret a coefficient as 'the effect of x, **holding the other features constant**'.",
            "Compute standard errors, t-statistics and p-values and decide which features matter.",
            "Detect and understand **multicollinearity** with VIF.",
            "Handle categorical and missing data with `ColumnTransformer` + `Pipeline`.",
            "Use a log-transformed target, and tame overfitting with Ridge and Lasso.",
        ],
        prereqs="Lesson 01; basic matrix multiplication",
        hook=("Last lesson, TV spend alone explained about 69 % of sales. But the company *also* buys radio and newspaper ads. "
              "Surely using all three channels gives a better answer?\n\n"
              "And a house price? Nobody prices a house by its size alone: location, age, bathrooms, condition all matter at once. "
              "Real questions have **many** inputs.\n\n"
              "Today the line becomes a **plane** (and, with more features, a *hyperplane*). The idea is the same; the notation is just bigger: "
              "and linear algebra makes it beautiful."),
        roadmap=["From a line to a plane", "The matrix form and the normal equation", "From scratch: normal equation + gradient descent",
                 "Inference: which coefficients are real?", "Multicollinearity & VIF", "Real data: categoricals, missing values, pipelines",
                 "Log target & residuals", "Ridge & Lasso (preview of regularisation)", "Profile, worked examples, problems, quiz"])

    # ------------------------------------------------------------------ 1
    nb.section("🧠", "1. From a line to a plane",
               r"With $p$ features the model is" "\n\n"
               r"$$\hat y=\beta_0+\beta_1x_1+\beta_2x_2+\dots+\beta_px_p$$" "\n\n"
               "**The interpretation changes in a subtle, crucial way.** $\\beta_j$ is the change in $\\hat y$ when $x_j$ rises by one unit "
               "**while every other feature is held constant**. That phrase (*holding the others constant*) is what lets regression separate overlapping effects.")
    nb.code(r'''ads = pd.read_csv(data_path("advertising.csv"))
print(ads.corr().round(2))''')
    nb.md("Notice: **newspaper correlates with sales (≈0.38)** *and* strongly with radio (≈0.56). Does newspaper really sell products, or does it just tag along with radio? "
          "Simple regression can't tell. Multiple regression can. Let's find out, but first, the machinery.")

    # ------------------------------------------------------------------ 2
    nb.section("📐", "2. The matrix form and the normal equation",
               "Stack the $n$ examples into a matrix. Add a column of **ones** for the intercept:")
    nb.md(r"""$$\underbrace{\begin{bmatrix}y_1\\ \vdots\\ y_n\end{bmatrix}}_{\mathbf y}\approx
\underbrace{\begin{bmatrix}1 & x_{11} & \cdots & x_{1p}\\ \vdots & & & \vdots\\ 1 & x_{n1} & \cdots & x_{np}\end{bmatrix}}_{\mathbf X\;(n\times(p+1))}
\underbrace{\begin{bmatrix}\beta_0\\ \beta_1\\ \vdots\\ \beta_p\end{bmatrix}}_{\boldsymbol\beta}$$

The loss is $\text{SSE}(\boldsymbol\beta)=\lVert\mathbf y-\mathbf X\boldsymbol\beta\rVert^2$. Its gradient is $-2\mathbf X^\top(\mathbf y-\mathbf X\boldsymbol\beta)$. Setting it to zero:

$$\mathbf X^\top\mathbf X\,\boldsymbol\beta=\mathbf X^\top\mathbf y\quad\Longrightarrow\quad\boxed{\hat{\boldsymbol\beta}=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y}\qquad\text{(the normal equation)}$$

**Geometric picture.** The columns of $\mathbf X$ span a flat subspace. The fitted values $\hat{\mathbf y}=\mathbf X\hat{\boldsymbol\beta}$ are the **orthogonal projection** of $\mathbf y$ onto that subspace: the closest point in "model-land" to reality. The residual vector is perpendicular to every column of $\mathbf X$ (which is why residuals sum to zero and are uncorrelated with each feature).""")
    nb.callout("warn", r"In code, never write `np.linalg.inv(X.T @ X) @ X.T @ y`. Inverting a matrix is slower and numerically fragile. "
                       "Use `np.linalg.solve(X.T @ X, X.T @ y)` or, better, `np.linalg.lstsq(X, y)` (this is what scikit-learn does under the hood via SVD).")
    nb.md(r"""**When does it break?** $\mathbf X^\top\mathbf X$ must be invertible. If one feature is an exact linear combination of others (e.g. `total = a + b` as a 3rd column), the matrix is **singular** and there are infinitely many equally good $\boldsymbol\beta$: see multicollinearity below.

**Cost:** forming $\mathbf X^\top\mathbf X$ is $O(np^2)$, solving it $O(p^3)$. Fine for hundreds of features; for millions, use gradient descent.""")

    # ------------------------------------------------------------------ 3 hand example
    nb.section("✍️", "3. A hand-worked example",
               r"Three data points, one feature, so $\mathbf X$ has a column of ones and the feature: $\mathbf X=\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix},\ \mathbf y=\begin{bmatrix}1\\2\\2\end{bmatrix}$.")
    nb.md(r"""$$\mathbf X^\top\mathbf X=\begin{bmatrix}3&6\\6&14\end{bmatrix},\quad
\mathbf X^\top\mathbf y=\begin{bmatrix}5\\11\end{bmatrix},\quad
(\mathbf X^\top\mathbf X)^{-1}=\frac{1}{42-36}\begin{bmatrix}14&-6\\-6&3\end{bmatrix}$$

$$\hat{\boldsymbol\beta}=\frac16\begin{bmatrix}14\cdot5-6\cdot11\\-6\cdot5+3\cdot11\end{bmatrix}=\frac16\begin{bmatrix}4\\3\end{bmatrix}=\begin{bmatrix}0.667\\0.5\end{bmatrix}$$

So the line is $\hat y=0.667+0.5x$. Verify:""")
    nb.code(r'''Xh = np.array([[1, 1], [1, 2], [1, 3]], dtype=float); yh = np.array([1, 2, 2], dtype=float)
print("X'X =\n", Xh.T @ Xh, "\nX'y =", Xh.T @ yh)
print("beta (solve) :", np.linalg.solve(Xh.T @ Xh, Xh.T @ yh).round(4))
print("beta (lstsq) :", np.linalg.lstsq(Xh, yh, rcond=None)[0].round(4))''')

    # ------------------------------------------------------------------ 4 scratch
    nb.section("🔧", "4. Build it from scratch",
               "A reusable class with two solvers: the exact **normal equation** and iterative **gradient descent** (vectorised: all features updated at once).\n\n"
               r"Gradient of the MSE: $\nabla L=-\frac{2}{n}\mathbf X^\top(\mathbf y-\mathbf X\boldsymbol\beta)$, update $\boldsymbol\beta\leftarrow\boldsymbol\beta-\eta\nabla L$.")
    nb.code(r'''class LinearRegressionScratch:
    """Multiple linear regression with two solvers: 'normal' (exact) or 'gd' (gradient descent)."""

    def __init__(self, solver="normal", lr=0.05, n_iter=2000):
        self.solver, self.lr, self.n_iter = solver, lr, n_iter

    def fit(self, X, y):
        X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float).ravel()
        Xb = np.c_[np.ones(len(X)), X]                       # add the column of ones
        if self.solver == "normal":
            beta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)       # (X'X) beta = X'y
        else:                                                 # gradient descent (features should be standardised)
            beta = np.zeros(Xb.shape[1]); n = len(y); self.loss_history_ = []
            for _ in range(self.n_iter):
                resid = y - Xb @ beta
                self.loss_history_.append(np.mean(resid ** 2))
                beta += self.lr * (2 / n) * (Xb.T @ resid)     # beta <- beta - lr * gradient
        self.intercept_, self.coef_ = beta[0], beta[1:]
        return self

    def predict(self, X):
        return self.intercept_ + np.asarray(X, dtype=float) @ self.coef_

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        return 1 - np.sum((y - self.predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)


features = ["tv_budget_k", "radio_budget_k", "newspaper_budget_k"]
X_ads, y_ads = ads[features].values, ads["sales_k_units"].values

exact = LinearRegressionScratch("normal").fit(X_ads, y_ads)
sk = LinearRegression().fit(X_ads, y_ads)
print("scratch (normal) :", exact.intercept_.round(4), exact.coef_.round(5), " R²=%.4f" % exact.score(X_ads, y_ads))
print("scikit-learn     :", round(sk.intercept_, 4), sk.coef_.round(5), " R²=%.4f" % sk.score(X_ads, y_ads))

# gradient descent needs standardised features
mu, sd = X_ads.mean(axis=0), X_ads.std(axis=0)
gd = LinearRegressionScratch("gd", lr=0.1, n_iter=500).fit((X_ads - mu) / sd, y_ads)
coef_back = gd.coef_ / sd; int_back = gd.intercept_ - np.sum(gd.coef_ * mu / sd)
print("scratch (GD)     :", int_back.round(4), coef_back.round(5))''')
    nb.md("All three agree. Read the coefficients: **TV ≈ 0.046, radio ≈ 0.174, newspaper ≈ 0.009 (tiny)**. R² jumped from 0.69 (TV only) to about 0.90.\n\n"
          "So the *newspaper* coefficient is tiny **after accounting for radio**. Is it distinguishable from zero? Let's test it statistically.")

    # ------------------------------------------------------------------ 5 inference
    nb.section("🔬", "5. Inference: which coefficients are real?",
               "Because our data is a *sample*, each coefficient has uncertainty. Under the OLS assumptions:\n\n"
               r"$$\widehat{\text{Var}}(\hat{\boldsymbol\beta})=s^2(\mathbf X^\top\mathbf X)^{-1},\qquad s^2=\frac{\text{SSE}}{n-p-1}$$" "\n\n"
               r"The **standard error** of $\hat\beta_j$ is the square root of the $j$-th diagonal entry. The **t-statistic** is $t_j=\hat\beta_j/\text{SE}_j$: how many standard errors the coefficient is away from zero, and the **p-value** is the probability of seeing such a $|t|$ if the true coefficient were 0. "
               "Small p (< 0.05) ⇒ the feature genuinely matters.")
    nb.code(r'''def ols_summary(X, y):
    """Coefficient table (coef, std err, t, p-value, 95% CI) for an OLS fit. X is a DataFrame."""
    names = ["intercept"] + list(X.columns)
    Xb = np.c_[np.ones(len(X)), X.values.astype(float)]
    y = np.asarray(y, dtype=float)
    beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
    resid = y - Xb @ beta
    n, k = Xb.shape
    s2 = resid @ resid / (n - k)                                  # estimated noise variance
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xb.T @ Xb)))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=n - k)
    tcrit = stats.t.ppf(0.975, df=n - k)
    return pd.DataFrame({"coef": beta, "std_err": se, "t": t, "p_value": p,
                         "ci_low": beta - tcrit * se, "ci_high": beta + tcrit * se}, index=names)

print("MULTIPLE regression (all three channels):")
display(ols_summary(ads[features], ads["sales_k_units"]).round(4))
print("SIMPLE regression with newspaper alone:")
display(ols_summary(ads[["newspaper_budget_k"]], ads["sales_k_units"]).round(4))''')
    nb.callout("key", "**Confounding in action.** Newspaper *alone* looks significant (small p-value) because it is correlated with radio, which truly drives sales. "
                      "Add radio and TV to the model and newspaper's effect collapses (coefficient 0.009 instead of 0.092, p ≈ 0.18, and the 95 % interval straddles 0). "
                      "Multiple regression *controls for* the other variables: this is its superpower, and why you should never judge a feature by its solo correlation.")
    nb.md(r"""### Adjusted R²

Plain $R^2$ **never decreases** when you add a feature: even a column of random noise nudges it up. **Adjusted** $R^2$ pays a penalty per feature:
$$R^2_{adj}=1-(1-R^2)\frac{n-1}{n-p-1}$$""")
    nb.code(r'''def adj_r2(r2, n, p): return 1 - (1 - r2) * (n - 1) / (n - p - 1)

rng = np.random.default_rng(0)
base = ads[["tv_budget_k", "radio_budget_k"]].copy()
for k in range(0, 31, 10):
    Xk = base.copy()
    for j in range(k): Xk[f"noise{j}"] = rng.normal(size=len(ads))        # useless random columns
    r2 = LinearRegression().fit(Xk, ads["sales_k_units"]).score(Xk, ads["sales_k_units"])
    print(f"+{k:>2} junk features:  R² = {r2:.4f}   adjusted R² = {adj_r2(r2, len(ads), Xk.shape[1]):.4f}")''')
    nb.md("R² creeps up with pure junk; adjusted R² goes *down*. Better still: judge on **held-out data** (cross-validation), which catches this automatically.")

    # ------------------------------------------------------------------ 6 multicollinearity
    nb.section("🧲", "6. Multicollinearity: when features are too alike",
               "If two features carry almost the same information, the model can't tell *which* one deserves the credit. "
               "Predictions stay fine, but the **individual coefficients become wildly unstable** and uninterpretable.\n\n"
               "**Variance Inflation Factor** diagnoses it: regress feature $j$ on all the *other* features, get that regression's $R_j^2$, and compute\n\n"
               r"$$\text{VIF}_j=\frac{1}{1-R_j^2}$$" "\n\n"
               "Rules of thumb: VIF ≈ 1 → independent; **> 5** worth a look; **> 10** serious.")
    nb.code(r'''# Manual VIF for ONE feature (radio), using the other two as predictors:
others = ads[["tv_budget_k", "newspaper_budget_k"]]
r2_j = LinearRegression().fit(others, ads["radio_budget_k"]).score(others, ads["radio_budget_k"])
print(f"R² of radio ~ tv + newspaper = {r2_j:.3f}  ->  VIF(radio) = {1 / (1 - r2_j):.2f}   (fine, < 5)")''')
    nb.md("Now let's *create* a problem: add a near-copy of TV spend (say, the same budget reported in two systems with tiny differences) and watch the coefficients across 200 bootstrap resamples.")
    nb.code(r'''rng = np.random.default_rng(3)
col = ads.copy(); col["tv_copy"] = col["tv_budget_k"] + rng.normal(0, 3, len(col))
print("corr(tv, tv_copy) =", round(col["tv_budget_k"].corr(col["tv_copy"]), 4))

def boot_coefs(df, cols, n_boot=200):
    out = []
    for _ in range(n_boot):
        s = df.sample(len(df), replace=True, random_state=int(rng.integers(1e9)))
        out.append(LinearRegression().fit(s[cols], s["sales_k_units"]).coef_)
    return pd.DataFrame(out, columns=cols)

b_ok = boot_coefs(col, ["tv_budget_k", "radio_budget_k"])
b_bad = boot_coefs(col, ["tv_budget_k", "tv_copy", "radio_budget_k"])
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].hist(b_ok["tv_budget_k"], bins=25, alpha=.8); axes[0].set_title("TV coefficient (no duplicate)\nstd = %.4f" % b_ok["tv_budget_k"].std())
axes[1].hist(b_bad["tv_budget_k"], bins=25, alpha=.7, label="tv"); axes[1].hist(b_bad["tv_copy"], bins=25, alpha=.7, label="tv_copy")
axes[1].set_title("With a near-duplicate: std = %.4f" % b_bad["tv_budget_k"].std()); axes[1].legend()
plt.tight_layout(); plt.show()
print("tv + tv_copy (the SUM of the two) stays stable: mean = %.4f, std = %.4f" % ((b_bad["tv_budget_k"] + b_bad["tv_copy"]).mean(), (b_bad["tv_budget_k"] + b_bad["tv_copy"]).std()))''')
    nb.callout("tip", "**Cures for multicollinearity:** drop one of the correlated features; combine them (average, ratio, PCA); or use **Ridge** regression (Section 9). "
                      "If you only care about *prediction* (not interpreting coefficients), multicollinearity is far less harmful.")

    # ------------------------------------------------------------------ 7 real data
    nb.section("🏠", "7. Real-world data: categoricals, missing values, pipelines",
               "Now the Nairobi house-price dataset (synthetic, prices in **KES millions**). Real data has holes and typos: let's meet them.")
    nb.code(r'''houses = pd.read_csv(data_path("house_prices.csv"))
print("rows:", len(houses))
print("\nMissing values per column:"); print(houses.isna().sum()[lambda s: s > 0])
bad = houses["sqft"] > 6500
print(f"\n{bad.sum()} listings claim > 6,500 sqft with ordinary prices -> data-entry typos (extra zero). Removing them.")
hc = houses[~bad].reset_index(drop=True)
hc.head()''')
    nb.md("""Two problems for linear regression:

1. **Categorical columns** (`neighborhood`, `condition`): regression only understands numbers. We **one-hot encode**: one 0/1 column per category.
   - *Dummy-variable trap*: if you keep **all** k dummies, they sum to 1 (the same as the intercept column) → perfect multicollinearity. Drop one (`drop='first'`); its effect is absorbed by the intercept, and the other coefficients mean "difference from the dropped category".
2. **Missing values**: impute (median for numbers, most-frequent for categories).

We wrap everything in a `ColumnTransformer` + `Pipeline` so the same steps are applied identically to train, test and future data: **without leakage**.""")
    nb.code(r'''from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

num_cols = ["sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "has_pool", "lot_size_sqft", "distance_to_cbd_km"]
cat_cols = ["neighborhood", "condition"]

preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(drop="first", handle_unknown="ignore")), cat_cols),
])

Xh = hc[num_cols + cat_cols]; yh = hc["price_million_kes"]
Xtr_h, Xte_h, ytr_h, yte_h = train_test_split(Xh, yh, test_size=0.2, random_state=RANDOM_STATE)

lin_house = make_pipeline(preprocess, LinearRegression()).fit(Xtr_h, ytr_h)
pred = lin_house.predict(Xte_h)
print(f"Linear regression on raw price:  TEST R² = {r2_score(yte_h, pred):.3f}   RMSE = {np.sqrt(mean_squared_error(yte_h, pred)):.2f} million KES")''')

    # ------------------------------------------------------------------ 8 log target
    nb.section("📈", "8. Residuals and the log-target trick",
               "House prices are **multiplicative**: a pool adds ~10 % to a house, whether it's a 5 M or a 50 M house: not a flat +2 M. "
               "Linear regression models *additive* effects. The fix: model $\\log(\\text{price})$. Then effects become percentages, and the variance of errors tends to even out.")
    nb.code(r'''log_house = make_pipeline(preprocess, LinearRegression()).fit(Xtr_h, np.log(ytr_h))
pred_log = np.exp(log_house.predict(Xte_h))          # back-transform to millions of KES
print(f"Log-target model:               TEST R² = {r2_score(yte_h, pred_log):.3f}   RMSE = {np.sqrt(mean_squared_error(yte_h, pred_log)):.2f} million KES")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=False)
for ax, (name, fit, res) in zip(axes, [("Raw price", pred, yte_h - pred), ("log(price) model", pred_log, yte_h - pred_log)]):
    ax.scatter(fit, res, s=10, alpha=0.5); ax.axhline(0, color="r"); ax.set_title(f"{name}: residuals vs fitted")
    ax.set_xlabel("fitted price (M KES)"); ax.set_ylabel("residual")
plt.tight_layout(); plt.show()''')
    nb.md("The raw-price residuals **fan out** (expensive houses → bigger errors: heteroscedasticity). The log model improves R² and RMSE. "
          "Now let's *interpret* it with our `ols_summary` (unscaled features, so coefficients have real units):")
    nb.code(r'''interp = hc.dropna().copy()
design = pd.get_dummies(interp[num_cols + cat_cols], columns=cat_cols, drop_first=True, dtype=float)
tbl_log = ols_summary(design, np.log(interp["price_million_kes"]))
display(tbl_log.loc[["intercept", "sqft", "bathrooms", "age_years", "has_pool", "garage_spaces", "condition_Poor", "neighborhood_Karen", "neighborhood_Rongai"]].round(4))
for feat in ["has_pool", "age_years", "neighborhood_Karen"]:
    pct = (np.exp(tbl_log.loc[feat, "coef"]) - 1) * 100
    print(f"{feat:<20}: {pct:+.1f}% on price")''')
    nb.callout("math", "In a **log-target** model, a coefficient $\\beta$ means each unit of the feature multiplies the price by $e^{\\beta}\\approx 1+\\beta$, i.e. about $100\\beta\\,\\%$. "
                       "Dummy coefficients are relative to the dropped baseline category (here the alphabetically-first neighbourhood/condition: *Embakasi* and *Excellent*).")

    # ------------------------------------------------------------------ 9 regularisation
    nb.section("🧯", "9. Ridge and Lasso: a first look at regularisation",
               "With many correlated features, OLS coefficients get huge and unstable. **Regularisation** adds a penalty for large coefficients:\n\n"
               r"$$\text{Ridge:}\ \ \min\ \text{SSE}+\alpha\sum\beta_j^2\qquad\qquad \text{Lasso:}\ \ \min\ \text{SSE}+\alpha\sum|\beta_j|$$" "\n\n"
               r"Ridge has a closed form, $\hat{\boldsymbol\beta}=(\mathbf X^\top\mathbf X+\alpha\mathbf I)^{-1}\mathbf X^\top\mathbf y$: the $\alpha\mathbf I$ also makes the matrix **always invertible**. "
               "Ridge **shrinks** all coefficients smoothly; Lasso can push some **exactly to zero** (automatic feature selection). Always **standardise** first, so the penalty treats every feature fairly.")
    nb.code(r'''Z = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(hc[num_cols]))
ylog = np.log(hc["price_million_kes"].values)
alphas_r = np.logspace(-2, 4, 40); alphas_l = np.logspace(-4, -0.5, 40)
ridge_path = np.array([Ridge(alpha=a).fit(Z, ylog).coef_ for a in alphas_r])
lasso_path = np.array([Lasso(alpha=a, max_iter=10000).fit(Z, ylog).coef_ for a in alphas_l])

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, path, al, ttl in [(axes[0], ridge_path, alphas_r, "Ridge: coefficients shrink smoothly"), (axes[1], lasso_path, alphas_l, "Lasso: coefficients hit exactly zero")]:
    for j, c in enumerate(num_cols): ax.plot(al, path[:, j], label=c)
    ax.set_xscale("log"); ax.set_xlabel("alpha (penalty strength)"); ax.set_ylabel("standardised coefficient"); ax.set_title(ttl)
axes[1].legend(fontsize=7, loc="upper right"); plt.tight_layout(); plt.show()''')
    nb.code(r'''ridge_house = make_pipeline(preprocess, RidgeCV(alphas=np.logspace(-3, 3, 30))).fit(Xtr_h, np.log(ytr_h))
p_ridge = np.exp(ridge_house.predict(Xte_h))
print(f"RidgeCV chose alpha = {ridge_house[-1].alpha_:.3f};  TEST R² = {r2_score(yte_h, p_ridge):.3f}  (plain OLS log model: {r2_score(yte_h, pred_log):.3f})")''')
    nb.md("Here OLS is already well-behaved (only 2000 rows, ~21 features, little collinearity) so Ridge doesn't change much: regularisation shines when features are many, noisy or collinear.")

    # ------------------------------------------------------------------ 10 profile
    nb.section("📋", "10. Cheat-sheet: the full profile of Multiple Linear Regression")
    nb.table([
        ["**Underlying concept**", "A weighted sum of features; the weights are chosen to minimise squared error."],
        ["**How it works**", "Solve the normal equations (X'X)β = X'y (or iterate with gradient descent)."],
        ["**Mathematical foundation**", "Least squares = orthogonal projection of y onto the column space of X; Gauss–Markov; t-tests on coefficients."],
        ["**Important parameters**", "`fit_intercept`, `positive`; scaling (for GD/regularisation); regularisation strength α in Ridge/Lasso/ElasticNet; transformations of target & features."],
        ["**Advantages**", "Fast, interpretable (coefficients = effects), closed form, strong baseline, provides confidence intervals, works well when relationships are roughly linear."],
        ["**Disadvantages**", "Assumes linearity & additivity; sensitive to outliers and multicollinearity; needs encoded categories & imputed gaps; can't discover interactions by itself."],
        ["**Complexity**", "Normal equation: O(n·p² + p³); gradient descent: O(n·p·iterations); prediction O(p) per row."],
        ["**Use cases**", "Pricing, forecasting with a few well-understood drivers, effect estimation ('what is a bedroom worth?'), a baseline for every regression project."],
        ["**Limitations**", "Extrapolation, non-linear/interacting effects, heteroscedastic errors (fix with log/weights), correlation ≠ causation."],
    ], ["Question", "Answer"])

    # ------------------------------------------------------------------ 11 worked
    nb.section("🧪", "11. More worked examples")
    nb.md(r"""### Worked Example 1: Read a coefficient table (illustrative numbers)

| feature | coef | p-value |
|---|---|---|
| intercept | 3.00 | <0.001 |
| TV (per 1k) | 0.045 | <0.001 |
| radio (per 1k) | 0.19 | <0.001 |
| newspaper (per 1k) | −0.001 | 0.89 |

**Q.** Which channel should the company cut, and how much extra sales does 10 k more on radio buy (holding TV and newspaper fixed)?

**A.** Newspaper: its coefficient is statistically indistinguishable from zero (p = 0.89 in this illustrative table), so it adds nothing once TV and radio are in the model. Radio: $10\times0.19=1.9$ thousand units. Per currency unit, radio is also ~4× more effective than TV (0.19 vs 0.045 in this table), so shifting budget from newspaper to radio is the data-backed move (subject to saturation effects the linear model can't see).""")
    nb.md(r"""### Worked Example 2: Percent effects in a log model

**Q.** In a model for $\ln(\text{price})$, the coefficient on `has_pool` is $0.12$ and on `age_years` is $-0.012$. Interpret both.

**A.** Pool: $e^{0.12}-1\approx 12.7\%$ higher price. Each additional year of age: $e^{-0.012}-1\approx-1.2\%$ per year, so a 10-year-older house is about $1-e^{-0.12}\approx 11\%$ cheaper, all else equal.""")
    nb.code(r'''print("pool premium  :", round((np.exp(0.12) - 1) * 100, 1), "%")
print("per year of age:", round((np.exp(-0.012) - 1) * 100, 2), "%    10 years:", round((np.exp(-0.12) - 1) * 100, 1), "%")''')

    # ------------------------------------------------------------------ problems
    nb.problem(
        "Solve the normal equations",
        "Given\n```python\nX = np.array([[1, 2, 1], [1, 3, 2], [1, 5, 2], [1, 7, 4], [1, 8, 3]], dtype=float)   # first column = intercept\ny = np.array([5, 8, 9, 14, 15], dtype=float)\n```\n"
        "Compute `beta` with the **normal equation** using `np.linalg.solve` (not `inv`), then compute the predictions `y_hat` and the residual sum of squares `sse`.",
        [r"Normal equation: $(X^\top X)\beta=X^\top y$. In NumPy: `X.T @ X` and `X.T @ y`.",
         "`beta = np.linalg.solve(A, b)` solves A·beta = b.",
         "`y_hat = X @ beta`; `sse = np.sum((y - y_hat) ** 2)`."],
        'X = np.array([[1, 2, 1], [1, 3, 2], [1, 5, 2], [1, 7, 4], [1, 8, 3]], dtype=float)\ny = np.array([5, 8, 9, 14, 15], dtype=float)\nbeta = y_hat = sse = None   # your code\n',
        'X = np.array([[1, 2, 1], [1, 3, 2], [1, 5, 2], [1, 7, 4], [1, 8, 3]], dtype=float)\ny = np.array([5, 8, 9, 14, 15], dtype=float)\n'
        'beta = np.linalg.solve(X.T @ X, X.T @ y)\ny_hat = X @ beta\nsse = np.sum((y - y_hat) ** 2)\nprint(beta.round(4), round(sse, 4))',
        'ref = np.linalg.lstsq(X, y, rcond=None)[0]\ncheck("beta matches lstsq", lambda: np.allclose(beta, ref))\n'
        'check("residuals are orthogonal to every column of X", lambda: np.allclose(X.T @ (y - y_hat), 0, atol=1e-8))\ncheck("sse is small (< 1.5)", lambda: sse < 1.5)',
        explain="The 'orthogonality' check is the geometric heart of OLS: the residual vector is perpendicular to the column space of X. That's *exactly* what the normal equations say: X'(y − Xβ) = 0.")

    nb.problem(
        "Write a VIF function",
        "Write `vif(X)` that takes a **DataFrame** and returns a Series of VIFs (one per column) using the recipe from Section 6.\n\n"
        "Then compute `vif_ok = vif(col[['tv_budget_k','radio_budget_k','newspaper_budget_k']])` and `vif_bad = vif(col[['tv_budget_k','tv_copy','radio_budget_k']])` "
        "(the `col` DataFrame with the near-duplicate is in the lesson).",
        ["For each column c: regress it on `X.drop(columns=c)` with `LinearRegression` and get its R².",
         "VIF = 1 / (1 − R²).",
         "Return `pd.Series(dict_of_vifs)`."],
        'def vif(X):\n    pass\n\nvif_ok = vif_bad = None   # your code\n',
        'def vif(X):\n    out = {}\n    for c in X.columns:\n        others = X.drop(columns=c)\n        r2 = LinearRegression().fit(others, X[c]).score(others, X[c])\n        out[c] = 1 / (1 - r2)\n    return pd.Series(out)\n\n'
        'vif_ok = vif(col[["tv_budget_k", "radio_budget_k", "newspaper_budget_k"]])\nvif_bad = vif(col[["tv_budget_k", "tv_copy", "radio_budget_k"]])\nprint(vif_ok.round(2).to_dict()); print(vif_bad.round(1).to_dict())',
        'check("healthy features: all VIF < 5", lambda: (vif_ok < 5).all())\ncheck("duplicated TV: VIF > 10 for both copies", lambda: vif_bad["tv_budget_k"] > 10 and vif_bad["tv_copy"] > 10)\ncheck("radio is unaffected (VIF < 5)", lambda: vif_bad["radio_budget_k"] < 5)',
        explain="The two TV columns each explain ~99.9 % of the other, giving VIFs in the hundreds. Radio is untouched.",
        level="⭐⭐")

    nb.problem(
        "Build the house-price pipeline yourself",
        "Using the lesson's `preprocess`, `Xtr_h, Xte_h, ytr_h, yte_h`:\n\n1. Build a pipeline `my_model` = `preprocess` → `LinearRegression`, trained on **log price**.\n"
        "2. Predict the test set, convert back with `np.exp`, and store `my_r2` and `my_rmse` (original KES-million scale).\n3. Store the **median absolute percentage error** (MdAPE) in `mdape`: a business-friendly number: 'typically we're off by x %'.",
        ["`make_pipeline(preprocess, LinearRegression())`; fit with `np.log(ytr_h)`.",
         "Predictions: `np.exp(my_model.predict(Xte_h))`.",
         "MdAPE = `np.median(np.abs(yte_h - preds) / yte_h) * 100`."],
        'my_model = None\nmy_r2 = my_rmse = mdape = None   # your code\n',
        'my_model = make_pipeline(preprocess, LinearRegression()).fit(Xtr_h, np.log(ytr_h))\npreds = np.exp(my_model.predict(Xte_h))\n'
        'my_r2 = r2_score(yte_h, preds); my_rmse = np.sqrt(mean_squared_error(yte_h, preds))\nmdape = np.median(np.abs(yte_h - preds) / yte_h) * 100\nprint(f"R²={my_r2:.3f}  RMSE={my_rmse:.2f}M  MdAPE={mdape:.1f}%")',
        'check("R² > 0.9", lambda: my_r2 > 0.9)\ncheck("RMSE < 5.5 million", lambda: my_rmse < 5.5)\ncheck("typically within 12% of the true price", lambda: mdape < 12)',
        explain="A median error around 7 % from a plain linear model is a respectable baseline. Later lessons will test whether trees, forests and SVR can beat it.",
        level="⭐⭐")

    nb.problem(
        "Choose alpha for Ridge on Advertising + collinearity",
        "Fit a `RidgeCV` with `alphas=np.logspace(-3, 3, 25)` on the **standardised** three-feature-plus-`tv_copy` data "
        "(`col[['tv_budget_k','tv_copy','radio_budget_k','newspaper_budget_k']]`) and compare with plain OLS.\n\n"
        "Store `best_alpha`, and `coef_gap_ols`, `coef_gap_ridge` = the absolute difference between the `tv_budget_k` and `tv_copy` coefficients under each model. "
        "(Ridge should split the credit more evenly.)",
        ["Standardise with `StandardScaler().fit_transform(...)`; use `RidgeCV(alphas=...).fit(Z, y)`; the chosen α is `.alpha_`.",
         "Coefficients are in `.coef_` in the same order as the columns (tv first, tv_copy second).",
         "`coef_gap = abs(coef_[0] - coef_[1])`."],
        'best_alpha = coef_gap_ols = coef_gap_ridge = None   # your code\n',
        'cols4 = ["tv_budget_k", "tv_copy", "radio_budget_k", "newspaper_budget_k"]\nZ4 = StandardScaler().fit_transform(col[cols4]); y4 = col["sales_k_units"]\n'
        'ols = LinearRegression().fit(Z4, y4); rcv = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Z4, y4)\nbest_alpha = rcv.alpha_\n'
        'coef_gap_ols = abs(ols.coef_[0] - ols.coef_[1]); coef_gap_ridge = abs(rcv.coef_[0] - rcv.coef_[1])\nprint(best_alpha, coef_gap_ols, coef_gap_ridge)',
        'check("alpha was chosen from the grid", lambda: 1e-3 <= best_alpha <= 1e3)\ncheck("Ridge splits the credit more evenly than OLS", lambda: coef_gap_ridge < coef_gap_ols)',
        explain="The ℓ2 penalty prefers two medium coefficients over one big and one compensating-negative coefficient (since it penalises squares), so credit is shared.",
        level="⭐⭐⭐")

    nb.problem(
        "Explain a coefficient to a client",
        "Using the lesson's `tbl_log` (OLS on log price, unscaled features), compute:\n\n"
        "- `pool_pct`: % price change from having a pool.\n- `bath_pct`: % price change from one extra bathroom.\n- `sqft_100_pct`: % price change from **100 extra sqft**.\n\n"
        "Formula: `(np.exp(coef * units) - 1) * 100`.",
        ["Look up coefficients with `tbl_log.loc['has_pool', 'coef']`.",
         "For 100 sqft multiply the sqft coefficient by 100 before `np.exp`.",
         "All three should be positive."],
        'pool_pct = bath_pct = sqft_100_pct = None   # your code\n',
        'c = tbl_log["coef"]\npool_pct = (np.exp(c["has_pool"]) - 1) * 100\nbath_pct = (np.exp(c["bathrooms"]) - 1) * 100\nsqft_100_pct = (np.exp(c["sqft"] * 100) - 1) * 100\nprint(pool_pct, bath_pct, sqft_100_pct)',
        'check("pool premium is positive and plausible (3-30%)", lambda: 3 < pool_pct < 30)\ncheck("extra bathroom adds value", lambda: bath_pct > 0)\ncheck("100 sqft adds a few percent (1-15%)", lambda: 1 < sqft_100_pct < 15)',
        explain="Translating coefficients to percentages is the difference between 'the model says 0.0004' and 'each extra 100 sqft adds ~4 %': the second sentence is what a client can act on.",
        level="⭐⭐")

    nb.quiz([
        ("Why add a column of ones to X?", "So the intercept β₀ is learned like any other coefficient: it multiplies a constant 1."),
        ("Newspaper correlates with sales on its own, but its multiple-regression coefficient is ≈ 0. Why?",
         "Newspaper budgets are correlated with radio budgets; radio truly drives sales. Once radio is in the model, newspaper adds no *extra* information: confounding."),
        ("You one-hot encode a 4-category column and keep all 4 dummies alongside the intercept. What goes wrong?",
         "The dummy-variable trap: the 4 columns sum to the intercept column → perfect multicollinearity → XᵀX is singular and coefficients are not unique. Drop one category."),
        ("Adding a useless feature raised R² from 0.8512 to 0.8514. Should you keep it?",
         "No. R² never decreases when features are added. Check adjusted R² or, better, cross-validated/test error."),
        ("When would you choose Lasso over Ridge?", "When you suspect many features are irrelevant and want automatic feature selection (some coefficients exactly 0). Ridge is better when most features carry a bit of signal, especially if they are correlated."),
    ])

    nb.footer(["Multiple linear regression: ŷ = Xβ, solved by the normal equation β = (XᵀX)⁻¹Xᵀy (use `solve`/`lstsq`, never `inv`).",
               "A coefficient = effect of that feature *holding the others constant*; t-stats and p-values say which are real.",
               "Multicollinearity inflates coefficient variance (check VIF); prediction can still be fine.",
               "Encode categories (drop one dummy), impute, scale: inside a Pipeline.",
               "Log-transform multiplicative targets; use Ridge/Lasso when features are many or correlated."],
              "Polynomial Regression", "03_Polynomial_Regression.ipynb")
    return nb
