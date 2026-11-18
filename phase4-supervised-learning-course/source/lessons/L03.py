from nbkit import Lesson


def make():
    nb = Lesson("03_Polynomial_Regression", 3, "Polynomial Regression",
                "A linear model wearing curved glasses, and the art of not over-bending",
                100, "🟡 Beginner → Intermediate",
                setup_extra="from sklearn.linear_model import LinearRegression, Ridge, RidgeCV\nfrom sklearn.preprocessing import PolynomialFeatures, StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.metrics import mean_squared_error, r2_score\nfrom sklearn.model_selection import train_test_split, cross_val_score, validation_curve, KFold")
    nb.header(
        objectives=[
            "Explain why a polynomial model is **still linear regression** (linear in the *parameters*).",
            "Build a polynomial design matrix by hand and fit it with least squares.",
            "Use `PolynomialFeatures` and pipelines in scikit-learn, including interaction terms.",
            "Select the polynomial degree with cross-validation and a validation curve.",
            "Measure **bias and variance** empirically by resampling.",
            "Understand numerical conditioning, feature explosion, and why polynomials extrapolate badly.",
            "Rescue an overfit polynomial with Ridge regularisation.",
        ],
        prereqs="Lessons 01–02",
        hook=("A company's salary ladder: Business Analyst earns 45 000, Manager 80 000, Partner 200 000, CEO **1 000 000**. "
              "Draw a straight line through that and you'll predict that a *Region Manager* earns… more than a Partner. Silly.\n\n"
              "The relationship **curves**. Does that mean we need an entirely new algorithm? No! There's a beautiful trick: "
              "keep the *same* linear-regression machinery, but feed it **extra columns** like $x^2$ and $x^3$. "
              "A straight line in a richer feature space looks curved in the original one."),
        roadmap=["The 'linear in parameters' trick", "Math: Vandermonde design matrix", "Salaries: line vs curve",
                 "From scratch", "Choosing the degree (validation curve)", "Bias & variance, measured",
                 "Traps: conditioning, extrapolation, feature explosion", "Regularised polynomials",
                 "A real dataset: house prices vs size", "Profile, worked examples, problems, quiz"])

    # ---------------------------------------------------------------- 1
    nb.section("🧠", "1. The trick: linear in the parameters",
               r"A degree-$d$ polynomial model is" "\n\n"
               r"$$\hat y=\beta_0+\beta_1x+\beta_2x^2+\dots+\beta_dx^d$$" "\n\n"
               "This is **not** linear in $x$: but it **is** linear in the *coefficients* $\\beta_j$. "
               "Define new features $z_1=x,\\ z_2=x^2,\\ \\dots,\\ z_d=x^d$ and the model is ordinary multiple linear regression on $z$:\n\n"
               r"$$\hat y=\beta_0+\beta_1z_1+\beta_2z_2+\dots+\beta_dz_d$$" "\n\n"
               "Everything from Lesson 02 (the normal equation, R², Ridge) carries over unchanged. Only the **feature engineering** is new.")
    nb.callout("key", "Linear regression's real restriction is *linear in parameters*, not *straight line*. Add transformed features (powers, logs, interactions) "
                      "and the same algorithm fits curves.")

    # ---------------------------------------------------------------- 2 math
    nb.section("📐", "2. The math: the Vandermonde design matrix",
               "For $n$ points and degree $d$, the design matrix has a column for each power:")
    nb.md(r"""$$\mathbf X=\begin{bmatrix}1&x_1&x_1^2&\cdots&x_1^d\\1&x_2&x_2^2&\cdots&x_2^d\\ \vdots&&&&\vdots\\1&x_n&x_n^2&\cdots&x_n^d\end{bmatrix}\qquad\hat{\boldsymbol\beta}=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y$$

(a *Vandermonde matrix*). Two useful facts:

- **Perfect interpolation:** with $d=n-1$ the polynomial passes **exactly** through all $n$ points (training error 0). That is the extreme of overfitting.
- **Multiple features:** for $p$ features and degree $d$ the number of terms (including the constant) is $\binom{p+d}{d}$. For $p=2,d=2$: $1,\,x_1,\,x_2,\,x_1^2,\,x_1x_2,\,x_2^2$ → 6 terms. The cross-term $x_1x_2$ is an **interaction**: "the effect of $x_1$ depends on $x_2$".""")

    # ---------------------------------------------------------------- 3 salaries
    nb.section("💼", "3. Salaries: a line versus a curve",
               "The classic small dataset: 10 job levels and their salaries (KES).")
    nb.code(r'''sal = pd.read_csv(data_path("position_salaries.csv"))
display(sal)
x_s = sal[["level"]].values.astype(float); y_s = sal["salary_kes"].values.astype(float)''')
    nb.code(r'''fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
grid = np.linspace(1, 10, 200).reshape(-1, 1)
for ax, d in zip(axes, [1, 2, 3, 4]):
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x_s, y_s)
    ax.scatter(x_s, y_s, color="black", zorder=3); ax.plot(grid, m.predict(grid), "r", lw=2)
    ax.set_title(f"degree {d}   R² = {m.score(x_s, y_s):.3f}"); ax.set_xlabel("level")
axes[0].set_ylabel("salary (KES)"); plt.tight_layout(); plt.show()''')
    nb.md("Degree 1 misses the shape badly (it even predicts *negative* salaries at the bottom). Degrees 3–4 follow the curve. "
          "Watch what each model predicts for a **Region-Manager-and-a-half** (level 6.5):")
    nb.code(r'''for d in [1, 2, 3, 4]:
    m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x_s, y_s)
    print(f"degree {d}: predicted salary at level 6.5 = {m.predict([[6.5]])[0]:>10,.0f} KES")
print("(Truth lies between Region Manager 150,000 and Partner 200,000)")''')
    nb.md("**An even simpler trick:** salaries roughly *multiply* with each level, so $\\ln(\\text{salary})$ may be nearly a straight line. Transformations can beat extra powers:")
    nb.code(r'''lin_log = LinearRegression().fit(x_s, np.log(y_s))
print("R² of log(salary) ~ level (a straight line!):", round(lin_log.score(x_s, np.log(y_s)), 3))
print("Prediction at 6.5:", f"{np.exp(lin_log.predict([[6.5]])[0]):,.0f} KES")''')

    # ---------------------------------------------------------------- 4 scratch
    nb.section("🔧", "4. Build it from scratch",
               "We need three ingredients: (1) expand $x$ into powers, (2) standardise for numerical stability, (3) solve least squares. "
               "`np.vander(z, d+1, increasing=True)` builds the Vandermonde matrix.")
    nb.code(r'''class PolynomialRegressionScratch:
    """Single-feature polynomial regression, solved by least squares on a Vandermonde matrix."""

    def __init__(self, degree=2, scale=True):
        self.degree, self.scale = degree, scale

    def _design(self, x):
        z = (np.asarray(x, float).ravel() - self.mu_) / self.sd_
        return np.vander(z, self.degree + 1, increasing=True)          # columns: 1, z, z^2, ..., z^d

    def fit(self, x, y):
        x = np.asarray(x, float).ravel()
        self.mu_, self.sd_ = (x.mean(), x.std()) if self.scale else (0.0, 1.0)
        self.beta_ = np.linalg.lstsq(self._design(x), np.asarray(y, float), rcond=None)[0]
        return self

    def predict(self, x):
        return self._design(x) @ self.beta_

    def score(self, x, y):
        y = np.asarray(y, float)
        return 1 - np.sum((y - self.predict(x)) ** 2) / np.sum((y - y.mean()) ** 2)


mine = PolynomialRegressionScratch(degree=4).fit(x_s, y_s)
skl = make_pipeline(PolynomialFeatures(4), LinearRegression()).fit(x_s, y_s)
print(f"Scratch  : R² = {mine.score(x_s, y_s):.5f}   prediction at 6.5 = {mine.predict([6.5])[0]:,.1f}")
print(f"sklearn  : R² = {skl.score(x_s, y_s):.5f}   prediction at 6.5 = {skl.predict([[6.5]])[0]:,.1f}")
print(f"np.polyfit: prediction at 6.5 = {np.polyval(np.polyfit(x_s.ravel(), y_s, 4), 6.5):,.1f}")''')
    nb.md("Same answers (up to rounding), three different routes. Let's peek at the design matrix scikit-learn builds for a tiny input, so there's no mystery:")
    nb.code(r'''demo = PolynomialFeatures(degree=3).fit_transform(np.array([[2.0], [3.0], [5.0]]))
print(pd.DataFrame(demo, columns=["1", "x", "x²", "x³"], index=["x=2", "x=3", "x=5"]))''')

    # ---------------------------------------------------------------- 5 choose degree
    nb.section("🎚️", "5. Choosing the degree",
               "Degree is the **complexity dial**. Too low → underfit; too high → overfit. We pick it with data, never with feelings. "
               "Here is a noisy sample whose true shape is a cubic:")
    nb.code(r'''rng = np.random.default_rng(5)
x_c = rng.uniform(-3, 3, 60)
y_c = 0.5 * x_c ** 3 - 2 * x_c ** 2 + x_c + 3 + rng.normal(0, 3, 60)
plt.scatter(x_c, y_c); plt.title("Noisy data from a hidden cubic"); plt.show()''')
    nb.md("**Validation curve:** for each degree, run 5-fold cross-validation and plot the *training* and *validation* error. "
          "Note we **standardise first, then expand** inside the pipeline.")
    nb.code(r'''pipe = make_pipeline(StandardScaler(), PolynomialFeatures(), LinearRegression())
degs = np.arange(1, 13)
cv = KFold(5, shuffle=True, random_state=RANDOM_STATE)
tr, va = validation_curve(pipe, x_c.reshape(-1, 1), y_c, param_name="polynomialfeatures__degree",
                          param_range=degs, cv=cv, scoring="neg_root_mean_squared_error")
tr, va = -tr, -va
plt.plot(degs, tr.mean(1), "o-", label="train RMSE"); plt.plot(degs, va.mean(1), "s-", label="validation RMSE")
plt.fill_between(degs, va.mean(1) - va.std(1), va.mean(1) + va.std(1), alpha=0.15)
best = degs[va.mean(1).argmin()]; plt.axvline(best, color="gray", ls=":", label=f"best degree = {best}")
plt.xlabel("polynomial degree"); plt.ylabel("RMSE"); plt.legend(); plt.title("Validation curve"); plt.show()
print("The true noise level is 3.0, so RMSE around 2.5–3 means we have captured all the signal (a small sample of 60 can luck out a bit below 3).")''')
    nb.callout("tip", "When two degrees score almost the same, pick the **simpler** one (the 'one-standard-error rule'): simpler models generalise better and are easier to explain.")

    # ---------------------------------------------------------------- 6 bias variance
    nb.section("⚖️", "6. Bias and variance: measured, not just described",
               "We can *measure* the two failure modes by repeating the experiment many times: draw a fresh training set from the same process, refit, and record predictions. "
               "**Bias²** = how far the *average* fitted curve is from the truth. **Variance** = how much individual fits scatter around their own average.")
    nb.code(r'''true_f = lambda x: np.sin(2 * np.pi * x)
rng = np.random.default_rng(11)
xs = np.linspace(0.02, 0.98, 80)
results = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for ax, d in zip(axes, [1, 3, 12]):
    preds = []
    for _ in range(200):
        xt = rng.uniform(0, 1, 25); yt = true_f(xt) + rng.normal(0, 0.3, 25)
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(xt[:, None], yt)
        preds.append(m.predict(xs[:, None]))
    preds = np.array(preds)
    for p in preds[:40]: ax.plot(xs, p, color="#2563eb", alpha=0.15)
    ax.plot(xs, preds.mean(0), "r", lw=2.5, label="average fit"); ax.plot(xs, true_f(xs), "g--", lw=2, label="truth")
    ax.set_ylim(-2.5, 2.5); ax.set_title(f"degree {d}")
    results[d] = (np.mean((preds.mean(0) - true_f(xs)) ** 2), np.mean(preds.var(0)))
axes[0].legend(); plt.tight_layout(); plt.show()

print(pd.DataFrame(results, index=["Bias²", "Variance"]).T.assign(**{"Bias²+Var": lambda d: d.sum(axis=1)}).round(3))
print("Irreducible noise variance = 0.3² = 0.09  (no model can beat this)")''')
    nb.md("- **Degree 1**: the fits all look alike (low variance) but the average is far from the truth (high bias).\n"
          "- **Degree 12**: the average is nearly right (low bias) but individual fits swing wildly (huge variance).\n"
          "- **Degree 3**: the sweet spot. Total error = **Bias² + Variance + irreducible noise**: the *bias–variance decomposition*.")

    # ---------------------------------------------------------------- 7 traps
    nb.section("🪤", "7. Three traps",
               "### Trap 1: extrapolation\nPolynomials are only trustworthy inside the data range; outside, the highest power dominates and the curve shoots off.")
    nb.code(r'''m3 = make_pipeline(PolynomialFeatures(3), LinearRegression()).fit(x_c.reshape(-1, 1), y_c)
m9 = make_pipeline(PolynomialFeatures(9), LinearRegression()).fit(x_c.reshape(-1, 1), y_c)
xx = np.linspace(-4.5, 4.5, 300).reshape(-1, 1)
plt.scatter(x_c, y_c, alpha=0.6); plt.plot(xx, m3.predict(xx), label="degree 3"); plt.plot(xx, m9.predict(xx), label="degree 9")
plt.axvspan(-4.5, -3, color="red", alpha=0.07); plt.axvspan(3, 4.5, color="red", alpha=0.07, label="outside the training range")
plt.ylim(-80, 80); plt.legend(); plt.title("Extrapolation: degree 9 goes wild outside the data"); plt.show()''')
    nb.md("### Trap 2: numerical conditioning\nRaw powers are hugely different in size (e.g. $50^{10}\\approx10^{17}$), so the design matrix becomes nearly singular. "
          "The **condition number** measures how fragile the matrix is (> $10^{8}$ and your coefficients are mostly rounding noise).")
    nb.code(r'''xr = np.linspace(0, 100, 50)
rows = []
for d in [2, 4, 6, 8, 10]:
    raw = np.vander(xr, d + 1, increasing=True)
    z = (xr - xr.mean()) / xr.std(); scaled = np.vander(z, d + 1, increasing=True)
    rows.append((d, np.linalg.cond(raw), np.linalg.cond(scaled)))
print(pd.DataFrame(rows, columns=["degree", "cond (raw x)", "cond (standardised x)"]).to_string(index=False, float_format=lambda v: f"{v:,.3g}"))''')
    nb.callout("tip", "**Always standardise before expanding** (`make_pipeline(StandardScaler(), PolynomialFeatures(d), …)`). It reduces the condition number by many orders of magnitude.")
    nb.md("### Trap 3: feature explosion\nWith many features, the number of polynomial terms $\\binom{p+d}{d}$ grows terrifyingly fast:")
    nb.code(r'''from math import comb
tbl = pd.DataFrame({f"degree {d}": [comb(p + d, d) for p in [1, 2, 5, 10, 20, 50]] for d in [2, 3, 4]},
                   index=[f"p = {p} features" for p in [1, 2, 5, 10, 20, 50]])
display(tbl)
print("With 50 features and degree 3 you'd have", comb(53, 3), "columns - far more than most datasets have rows.")''')
    nb.md("**Remedies:** restrict to specific interactions you believe in (`interaction_only=True`), use only degree 2, use regularisation, or switch to models that discover non-linearity on their own (trees, SVMs with kernels; coming up!).")
    nb.md("Interaction features in action; what scikit-learn builds from two features:")
    nb.code(r'''pf = PolynomialFeatures(degree=2, include_bias=False).fit(pd.DataFrame({"sqft": [1], "bedrooms": [1]}))
print(list(pf.get_feature_names_out()))
print("interaction_only=True ->", list(PolynomialFeatures(2, include_bias=False, interaction_only=True).fit(pd.DataFrame({"sqft": [1], "bedrooms": [1]})).get_feature_names_out()))''')

    # ---------------------------------------------------------------- 8 regularised
    nb.section("🧯", "8. Rescuing a flexible polynomial with Ridge",
               "Instead of *choosing* a low degree, we can use a high degree but **penalise big coefficients** (Ridge, from Lesson 02). "
               "The penalty tames the wiggles. Let's compare on the sine data with a deliberately generous degree of 12.")
    nb.code(r'''rng = np.random.default_rng(0)
x_tr = rng.uniform(0, 1, 30);  y_tr = true_f(x_tr) + rng.normal(0, 0.3, 30)
x_te = rng.uniform(0, 1, 500); y_te = true_f(x_te) + rng.normal(0, 0.3, 500)

models = {
    "OLS degree 12":  make_pipeline(StandardScaler(), PolynomialFeatures(12, include_bias=False), LinearRegression()),
    "Ridge degree 12": make_pipeline(StandardScaler(), PolynomialFeatures(12, include_bias=False), StandardScaler(), RidgeCV(alphas=np.logspace(-6, 2, 40))),
    "OLS degree 3":   make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), LinearRegression()),
}
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.scatter(x_tr, y_tr, color="black", s=20, label="train", zorder=3)
grid = np.linspace(0, 1, 300)
for name, m in models.items():
    m.fit(x_tr[:, None], y_tr)
    rmse = np.sqrt(mean_squared_error(y_te, m.predict(x_te[:, None])))
    ax.plot(grid, m.predict(grid[:, None]), label=f"{name}  (test RMSE {rmse:.2f})", lw=2)
ax.plot(grid, true_f(grid), "g--", label="truth"); ax.set_ylim(-2, 2); ax.legend(fontsize=8); plt.show()''')

    # ---------------------------------------------------------------- 9 house
    nb.section("🏠", "9. A real dataset: house price vs size",
               "Is price vs. size a straight line? Let's let cross-validation decide (ignoring location for now: we want one clean feature).")
    nb.code(r'''houses = pd.read_csv(data_path("house_prices.csv"))
hc = houses[houses["sqft"] <= 6500]                     # remove the data-entry typos found in Lesson 02
Xs, ys = hc[["sqft"]].values, hc["price_million_kes"].values

degs = np.arange(1, 8)
tr, va = validation_curve(make_pipeline(StandardScaler(), PolynomialFeatures(), LinearRegression()), Xs, ys,
                          param_name="polynomialfeatures__degree", param_range=degs, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
plt.figure(figsize=(7, 3.8)); plt.plot(degs, tr.mean(1), "o-", label="train R²"); plt.plot(degs, va.mean(1), "s-", label="validation R²")
plt.xlabel("degree"); plt.ylabel("R²"); plt.legend(); plt.title("House price vs sqft: how curved is it?"); plt.show()
print(pd.Series(va.mean(1), index=degs).round(3).to_dict())

best_d = 2          # we plot degree 2 anyway, to see how little it differs from the line
m_house = make_pipeline(StandardScaler(), PolynomialFeatures(best_d), LinearRegression()).fit(Xs, ys)
g = np.linspace(Xs.min(), Xs.max(), 200).reshape(-1, 1)
plt.scatter(Xs, ys, s=6, alpha=0.3); plt.plot(g, m_house.predict(g), "r", lw=2.5, label="quadratic fit")
plt.plot(g, LinearRegression().fit(Xs, ys).predict(g), "k--", label="straight line"); plt.xlabel("sqft"); plt.ylabel("price (M KES)"); plt.legend(); plt.show()''')
    nb.md("**A surprise worth remembering:** the validation R² is essentially *flat* (≈ 0.50) from degree 1 to 6. Extra curvature buys nothing here, so the straight line is already enough: "
          "a validation curve can tell you **not** to bother. The real story is the scatter: size alone explains only about half of the price variation. "
          "Neighbourhood matters hugely, and capturing *that* is where **trees** (Lesson 04) will shine.")

    # ---------------------------------------------------------------- 10 profile
    nb.section("📋", "10. Cheat-sheet: the full profile of Polynomial Regression")
    nb.table([
        ["**Underlying concept**", "Linear regression on engineered power (and interaction) features → curved fits."],
        ["**How it works**", "Expand x → [1, x, x², …, x^d]; solve least squares for the coefficients."],
        ["**Mathematical foundation**", "Vandermonde design matrix; normal equations; bias–variance decomposition; conditioning."],
        ["**Important parameters**", "`degree` (the complexity dial), `interaction_only`, `include_bias`; scaling; Ridge/Lasso strength α."],
        ["**Advantages**", "Simple, fast, closed-form, captures smooth curves, still interpretable at low degree, reuses all of linear regression's tools."],
        ["**Disadvantages**", "Global (a change in one region moves the whole curve), oscillates (Runge's phenomenon), explodes outside the data, number of terms explodes with features."],
        ["**Complexity**", "1 feature: O(n·d² + d³). p features: terms grow as C(p+d, d) so cost ≈ O(n·T² + T³) with T = number of terms."],
        ["**Use cases**", "Smooth, low-dimensional relationships: growth curves, dose–response, calibration, trend lines, physics-style curves."],
        ["**Limitations**", "Poor extrapolation; high degree ⇒ overfitting & ill-conditioning; unsuitable for many features or sharp changes (steps, thresholds)."],
    ], ["Question", "Answer"])

    # ---------------------------------------------------------------- 11 worked
    nb.section("🧪", "11. Worked examples")
    nb.md(r"""### Worked Example 1: Fit a parabola by hand

**Question.** Fit $\hat y=\beta_0+\beta_1x+\beta_2x^2$ to the points $(0,1),(1,0),(2,1)$.

**Solution.** Three points, three parameters → the parabola passes through all of them. Plug in:
- $x=0$: $\beta_0=1$
- $x=1$: $1+\beta_1+\beta_2=0 \Rightarrow \beta_1+\beta_2=-1$
- $x=2$: $1+2\beta_1+4\beta_2=1 \Rightarrow \beta_1+2\beta_2=0$

Subtract: $\beta_2=1$, so $\beta_1=-2$. The curve is $\hat y=1-2x+x^2=(x-1)^2$. ✔""")
    nb.code(r'''print(np.polyfit([0, 1, 2], [1, 0, 1], 2).round(6), "  <- [β2, β1, β0] (np.polyfit lists the highest power first)")''')
    nb.md(r"""### Worked Example 2: How many terms?

**Question.** You have 8 features and want all terms up to degree 3. How many columns? Is that OK with 500 rows?

**Solution.** $\binom{8+3}{3}=\binom{11}{3}=165$ columns (including the constant). With 500 rows that's 3 rows per parameter: high overfitting risk. Use `interaction_only=True`, degree 2 ($\binom{10}{2}=45$ terms), or regularise.""")
    nb.code(r'''print("C(11,3) =", comb(11, 3), "   degree-2 version C(10,2) =", comb(10, 2))''')

    # ---------------------------------------------------------------- problems
    nb.problem(
        "Build a design matrix by hand",
        "For `x = np.array([1, 2, 3, 4])` build the degree-3 design matrix `A` **without** scikit-learn (columns: 1, x, x², x³). "
        "Then compute the least-squares coefficients `beta` for `y = np.array([2, 5, 10, 17])` with `np.linalg.lstsq`.",
        ["`np.vander(x, 4, increasing=True)` builds the matrix in one line.",
         "Alternative: `np.column_stack([x**0, x**1, x**2, x**3])`.",
         "`np.linalg.lstsq(A, y, rcond=None)[0]` returns the coefficients. (The data is exactly 1 + x², so expect β ≈ [1, 0, 1, 0].)"],
        'x = np.array([1, 2, 3, 4]); y = np.array([2, 5, 10, 17])\nA = beta = None   # your code\n',
        'x = np.array([1, 2, 3, 4]); y = np.array([2, 5, 10, 17])\nA = np.vander(x, 4, increasing=True)\nbeta = np.linalg.lstsq(A, y, rcond=None)[0]\nprint(A); print(beta.round(6))',
        'check("A has shape (4, 4)", lambda: A.shape == (4, 4))\ncheck("last column is x cubed", lambda: np.allclose(A[:, 3], x ** 3))\ncheck("beta ≈ [1, 0, 1, 0]", lambda: np.allclose(beta, [1, 0, 1, 0], atol=1e-8))',
        explain="The data lies exactly on y = 1 + x², so a flexible cubic recovers it perfectly (the x³ and x coefficients are 0).")

    nb.problem(
        "Count the terms",
        "Write `n_terms(p, d)` returning the number of polynomial terms (including the constant) for `p` features and degree `d`. "
        "Verify it against `PolynomialFeatures(d).fit_transform(np.zeros((1, p))).shape[1]` for a few cases and store the answer for **p = 12, d = 3** in `answer`.",
        [r"The formula is $\binom{p+d}{d}$; in Python `math.comb(p + d, d)` (already imported as `comb`).",
         "To verify, build `np.zeros((1, p))` and count the columns that PolynomialFeatures creates."],
        'def n_terms(p, d):\n    pass\n\nanswer = None   # n_terms(12, 3)\n',
        'def n_terms(p, d):\n    return comb(p + d, d)\n\nanswer = n_terms(12, 3)\nprint(answer)',
        'check("matches sklearn for (3, 2)", lambda: n_terms(3, 2) == PolynomialFeatures(2).fit_transform(np.zeros((1, 3))).shape[1])\n'
        'check("matches sklearn for (5, 4)", lambda: n_terms(5, 4) == PolynomialFeatures(4).fit_transform(np.zeros((1, 5))).shape[1])\ncheck("p=12, d=3 -> 455", lambda: answer == 455)',
        explain="12 features at degree 3 already produce 455 columns: more than the rows in many real datasets.")

    nb.problem(
        "Pick the degree with cross-validation",
        "Using the lesson's noisy cubic data `x_c, y_c`:\n\n1. For each degree 1…10, compute the mean **5-fold CV RMSE** (use `cross_val_score` with `scoring='neg_root_mean_squared_error'` "
        "and `cv=KFold(5, shuffle=True, random_state=0)`; build the model as `make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression())`).\n"
        "2. Store the scores in a dict `cv_rmse` (degree → RMSE) and the best degree in `best_degree`.\n3. Apply the **simplicity rule**: store in `simplest_ok` the *smallest* degree whose RMSE is within 5 % of the minimum.",
        ["Loop `for d in range(1, 11)` and fill `cv_rmse[d] = -cross_val_score(...).mean()`.",
         "`best_degree = min(cv_rmse, key=cv_rmse.get)`.",
         "`simplest_ok = min(d for d, v in cv_rmse.items() if v <= 1.05 * min(cv_rmse.values()))`."],
        'cv_rmse = {}\nbest_degree = simplest_ok = None   # your code\n',
        'cv_rmse = {}\nfor d in range(1, 11):\n    pipe_d = make_pipeline(StandardScaler(), PolynomialFeatures(d), LinearRegression())\n'
        '    cv_rmse[d] = -cross_val_score(pipe_d, x_c.reshape(-1, 1), y_c, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_root_mean_squared_error").mean()\n'
        'best_degree = min(cv_rmse, key=cv_rmse.get)\nsimplest_ok = min(d for d, v in cv_rmse.items() if v <= 1.05 * min(cv_rmse.values()))\nprint({k: round(float(v), 2) for k, v in cv_rmse.items()}, best_degree, simplest_ok)',
        'check("tested 10 degrees", lambda: len(cv_rmse) == 10)\ncheck("best raw-CV degree is somewhere in 3-7 (noise makes the exact minimum wobble)", lambda: 3 <= best_degree <= 7)\ncheck("simplest acceptable degree is exactly 3", lambda: simplest_ok == 3)\n'
        'check("degree 1 is clearly worse", lambda: cv_rmse[1] > 1.5 * cv_rmse[3])',
        explain="Degree 1 underfits badly; degrees 3–6 are within a hair of each other, and the *exact* minimum is just noise (here it lands on 5). The simplicity rule cuts through the noise and picks degree 3: exactly the hidden truth.",
        level="⭐⭐")

    nb.problem(
        "Predict the mid-level salary",
        "Fit a **degree-4** polynomial pipeline and a **straight line** to the salary data (`x_s, y_s`). Store the level-6.5 predictions in `pred_poly` and `pred_lin`. "
        "Then fit `log(salary)` on level with a straight line and store the back-transformed level-6.5 prediction in `pred_log`.",
        ["Use `make_pipeline(PolynomialFeatures(4), LinearRegression())` and `.predict([[6.5]])[0]`.",
         "For the log model: fit on `np.log(y_s)`, predict, then `np.exp(...)`.",
         "The truth should lie between 150 000 and 200 000."],
        'pred_poly = pred_lin = pred_log = None   # your code\n',
        'pred_poly = make_pipeline(PolynomialFeatures(4), LinearRegression()).fit(x_s, y_s).predict([[6.5]])[0]\npred_lin = LinearRegression().fit(x_s, y_s).predict([[6.5]])[0]\n'
        'pred_log = np.exp(LinearRegression().fit(x_s, np.log(y_s)).predict([[6.5]])[0])\nprint(pred_poly, pred_lin, pred_log)',
        'check("polynomial prediction is plausible (150k-200k)", lambda: 150_000 < pred_poly < 200_000)\ncheck("straight line badly overshoots (>250k)", lambda: pred_lin > 250_000)\n'
        'check("log-linear prediction is in the right ballpark (140k-220k)", lambda: 140_000 < pred_log < 220_000)',
        explain="The straight line is off by ~100 %. The quartic lands inside the 150–200k bracket; the log-linear model lands close (~208k, slightly high) using only 2 parameters. The *right transformation* often rivals many extra powers.",
        level="⭐⭐")

    nb.problem(
        "Tame the wiggles",
        "Using `x_tr, y_tr, x_te, y_te` from Section 8, build **two** degree-10 models: plain OLS and RidgeCV (`alphas=np.logspace(-6, 2, 40)`). "
        "Both should be pipelines `StandardScaler → PolynomialFeatures(10, include_bias=False) → StandardScaler → (model)`.\n"
        "Store their test RMSEs in `rmse_ols` and `rmse_ridge`.",
        ["Re-use the structure of `models` in Section 8, just change the degree to 10.",
         "`np.sqrt(mean_squared_error(y_te, model.predict(x_te[:, None])))`.",
         "Ridge should achieve a lower test error than plain OLS."],
        'rmse_ols = rmse_ridge = None   # your code\n',
        'ols10 = make_pipeline(StandardScaler(), PolynomialFeatures(10, include_bias=False), StandardScaler(), LinearRegression()).fit(x_tr[:, None], y_tr)\n'
        'rdg10 = make_pipeline(StandardScaler(), PolynomialFeatures(10, include_bias=False), StandardScaler(), RidgeCV(alphas=np.logspace(-6, 2, 40))).fit(x_tr[:, None], y_tr)\n'
        'rmse_ols = np.sqrt(mean_squared_error(y_te, ols10.predict(x_te[:, None])))\nrmse_ridge = np.sqrt(mean_squared_error(y_te, rdg10.predict(x_te[:, None])))\nprint(rmse_ols, rmse_ridge)',
        'check("Ridge generalises better than OLS", lambda: rmse_ridge < rmse_ols)\ncheck("Ridge gets close to the noise floor (< 0.40)", lambda: rmse_ridge < 0.40)',
        explain="Ridge keeps all 10 powers but shrinks the huge, cancelling coefficients that make plain OLS oscillate: variance drops with only a small increase in bias.",
        level="⭐⭐⭐")

    nb.quiz([
        ("Why is polynomial regression still called *linear* regression?",
         "Because the model is linear in its **parameters** β. The non-linearity lives entirely in the engineered features (x², x³, …)."),
        ("You fit degree 15 to 16 points and get training R² = 1.000. Good news?",
         "No: a degree-(n−1) polynomial interpolates every point, including the noise. It's extreme overfitting; check validation error."),
        ("Why standardise before creating polynomial features?",
         "Raw powers span many orders of magnitude, making XᵀX nearly singular (huge condition number) and coefficients numerically unstable."),
        ("What does an interaction term x₁·x₂ capture?", "That the effect of x₁ on y depends on the level of x₂ (e.g. an extra bedroom is worth more in a big house)."),
        ("Give two reasons a high-degree polynomial is dangerous for prediction outside the training range.",
         "The highest-degree term dominates and shoots off to ±∞; the curve oscillates (Runge's phenomenon) and has no data constraining it there."),
    ])

    nb.footer(["Polynomial regression = linear regression on powers of x: linear in β, curved in x.",
               "Degree is the complexity dial: choose it by cross-validation, prefer the simplest degree that works.",
               "Bias² + variance + noise = expected error; low degree → bias, high degree → variance.",
               "Standardise first; beware extrapolation and the C(p+d, d) feature explosion.",
               "Ridge, or a smarter transformation (log), can beat brute-force degree."],
              "Decision Tree Regression", "04_Decision_Tree_Regression.ipynb")
    return nb
