from nbkit import Lesson


def make():
    nb = Lesson("06_Support_Vector_Regression", 6, "Support Vector Regression (SVR)",
                "Fit within a tube of tolerance, and let only the troublemakers define the model",
                120, "🟠 Intermediate → Advanced",
                setup_extra="from sklearn.svm import SVR\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.compose import TransformedTargetRegressor\nfrom sklearn.metrics import mean_squared_error, r2_score\nfrom sklearn.metrics.pairwise import rbf_kernel\nfrom sklearn.model_selection import train_test_split, GridSearchCV, KFold")
    nb.header(
        objectives=[
            "Explain the ε-insensitive loss and the tube picture of SVR.",
            "State the SVR optimisation problem (flatness + slack, trade-off C) and what **support vectors** are.",
            "Describe the **kernel trick** and compute an RBF kernel by hand.",
            "Implement a linear SVR from scratch (sub-gradient descent) and **reconstruct an RBF-SVR's prediction** from its support vectors.",
            "Explain how `C`, `epsilon` and `gamma` change the fit (and why scaling X *and* y matters).",
            "Tune SVR with a cross-validated grid search and know when it is (and isn't) the right tool.",
        ],
        prereqs="Lessons 01–04; comfort with dot products",
        hook=("Imagine a tailor measuring a customer. A 2-millimetre error doesn't matter: nobody can feel it. Only errors beyond, say, 5 mm get remade. "
              "A tailor who obsessed over every millimetre would waste a lifetime; one who ignores the *small* stuff and focuses on the *real* misfits is efficient.\n\n"
              "**Support Vector Regression** has exactly this attitude. Draw a *tube* of half-width ε around the fitted curve: points **inside** the tube cost *nothing*. "
              "Only points on or outside the tube: the **support vectors**: shape the model. "
              "The result is a smooth, robust curve defined by a small set of 'influential' examples."),
        roadmap=["Losses: squared vs absolute vs ε-insensitive", "The SVR optimisation problem", "The kernel trick",
                 "Seeing the tube and the support vectors", "From scratch: linear SVR by sub-gradient descent",
                 "Opening the black box: reconstruct predictions from support vectors", "C, epsilon and gamma in action",
                 "Scaling X *and* y", "Tuning & comparison on house prices", "Cheat-sheet, worked examples, problems, quiz"])

    # ---------------------------------------------------------- 1 loss
    nb.section("📉", "1. Losses: how do we punish a residual?",
               "A **loss function** says how bad a residual $r=y-\\hat y$ is. Regression models differ mainly in the loss they minimise:")
    nb.code(r'''r = np.linspace(-3, 3, 400); eps = 1.0
squared = r ** 2; absolute = np.abs(r); eps_ins = np.maximum(0, np.abs(r) - eps)
huber = np.where(np.abs(r) <= 1, 0.5 * r ** 2, np.abs(r) - 0.5)
for l, lab in [(squared, "squared (OLS)"), (absolute, "absolute (MAE)"), (eps_ins, f"ε-insensitive (ε={eps})"), (huber, "Huber")]:
    plt.plot(r, l, lw=2.3, label=lab)
plt.ylim(0, 5); plt.xlabel("residual  y − ŷ"); plt.ylabel("loss"); plt.legend(); plt.title("Four ways to punish an error"); plt.show()''')
    nb.md("""- **Squared**: tiny errors are ignored, huge errors dominate (outlier-sensitive).
- **Absolute**: every unit of error costs the same.
- **ε-insensitive** (SVR): *zero* inside the tube $|r|\\le\\varepsilon$, then grows **linearly**: tolerant of small errors and gentle on outliers.
- **Huber**: quadratic near zero, linear far away.

Formally: $L_\\varepsilon(r)=\\max\\bigl(0,\\ |r|-\\varepsilon\\bigr)$.""")

    # ---------------------------------------------------------- 2 math
    nb.section("📐", "2. The SVR optimisation problem",
               "We look for a function $f(x)=\\mathbf w\\cdot\\phi(x)+b$ that is **as flat as possible** (small $\\lVert\\mathbf w\\rVert$; a flat function generalises) "
               "while keeping most points within ε of the target. Points that fall outside get **slack** variables $\\xi_i,\\xi_i^*\\ge0$ measuring how far:")
    nb.md(r"""$$\min_{\mathbf w,b,\xi,\xi^*}\ \ \underbrace{\tfrac12\lVert\mathbf w\rVert^2}_{\text{flatness}}\;+\;C\sum_{i=1}^n(\xi_i+\xi_i^*)
\quad\text{s.t.}\quad
\begin{cases}y_i-f(x_i)\le\varepsilon+\xi_i\\ f(x_i)-y_i\le\varepsilon+\xi_i^*\\ \xi_i,\xi_i^*\ge0\end{cases}$$

Equivalently (eliminating the slacks): minimise $\ \tfrac12\lVert\mathbf w\rVert^2+C\sum_i L_\varepsilon\bigl(y_i-f(x_i)\bigr)$.

**The three knobs**

| Parameter | Meaning | Too small | Too large |
|---|---|---|---|
| **C** | Penalty on points outside the tube | Underfit (too flat) | Overfit (chases every point) |
| **ε** | Half-width of the tolerance tube | Many support vectors, slow, noisy | Tube swallows the signal, underfit |
| **γ** (RBF) | Reach of each point's influence | Too smooth / underfit | Wiggly / overfit (each SV is a spike) |

**The dual form and the kernel trick.** Solving with Lagrange multipliers (KKT conditions) gives a model that depends on the data **only through inner products**:

$$\hat f(x)=\sum_{i\in SV}(\alpha_i-\alpha_i^*)\,K(x_i,x)+b$$

Only points **on or outside the tube** have non-zero coefficients $\alpha_i-\alpha_i^*$: the **support vectors**. All points strictly inside the tube have coefficient 0 and could be deleted without changing the model!""")
    nb.md(r"""### The kernel trick

If we replace the inner product $\phi(x)\cdot\phi(x')$ by a *kernel function* $K(x,x')$, we get a model that is linear in a (possibly infinite-dimensional) feature space **without ever computing that space**:

| Kernel | Formula | Behaviour |
|---|---|---|
| Linear | $K=x\cdot x'$ | Straight lines / hyperplanes |
| Polynomial | $K=(\gamma\,x\cdot x'+r)^d$ | Polynomial curves |
| **RBF (Gaussian)** | $K=\exp\bigl(-\gamma\lVert x-x'\rVert^2\bigr)$ | Smooth, flexible, the default. Similarity 1 for identical points, → 0 for distant ones |

An RBF-SVR prediction is a **weighted sum of bumps** centred at the support vectors; $\gamma$ controls the width of each bump (large γ → narrow bumps).""")

    # ---------------------------------------------------------- 3 tube
    nb.section("👀", "3. Seeing the tube and the support vectors",
               "A noisy sine curve; we fit an RBF-SVR and highlight which points became support vectors.")
    nb.code(r'''rng = np.random.default_rng(2)
x = np.sort(rng.uniform(0, 2 * np.pi, 80)); y = np.sin(x) + rng.normal(0, 0.2, 80)
X1 = x.reshape(-1, 1); grid = np.linspace(0, 2 * np.pi, 400).reshape(-1, 1)

svr = SVR(kernel="rbf", C=10, epsilon=0.25, gamma=1.0).fit(X1, y)
fx = svr.predict(grid)
plt.figure(figsize=(8.5, 4.3))
plt.fill_between(grid.ravel(), fx - 0.25, fx + 0.25, color="#fde68a", alpha=0.6, label="ε-tube")
plt.plot(grid, fx, "r", lw=2.5, label="SVR fit")
inside = np.setdiff1d(np.arange(len(x)), svr.support_)
plt.scatter(x[inside], y[inside], s=28, color="#2563eb", label=f"inside tube - ignored ({len(inside)})")
plt.scatter(x[svr.support_], y[svr.support_], s=70, facecolors="none", edgecolors="black", linewidths=1.6, label=f"support vectors ({len(svr.support_)})")
plt.legend(); plt.title("Only points on/outside the tube (circled) define the model"); plt.show()''')
    nb.md("Blue dots strictly inside the yellow tube are **irrelevant**: remove them and refit: you'd get the same curve (up to the solver's numerical tolerance). Let's prove it:")
    nb.code(r'''keep = svr.support_
svr_small = SVR(kernel="rbf", C=10, epsilon=0.25, gamma=1.0).fit(X1[keep], y[keep])
print("Max difference between the two fitted curves after deleting every non-support-vector:",
      np.abs(svr_small.predict(grid) - fx).max().round(6))''')

    # ---------------------------------------------------------- 4 hand kernel
    nb.section("✍️", "4. Two small calculations by hand",
               r"**(a) ε-insensitive loss.** Residuals $[-0.5,\ 0.05,\ 0.3,\ -0.12]$ with $\varepsilon=0.1$ → losses $[0.4,\ 0,\ 0.2,\ 0.02]$ (only the part *beyond* 0.1 counts) → total $0.62$." "\n\n"
               r"**(b) An RBF kernel value.** $x=(1,2)$, $x'=(2,0)$, $\gamma=0.5$: $\lVert x-x'\rVert^2=1+4=5$, so $K=e^{-0.5\cdot5}=e^{-2.5}\approx0.0821$: the points are quite dissimilar.")
    nb.code(r'''res = np.array([-0.5, 0.05, 0.3, -0.12]); print("total ε-insensitive loss:", np.maximum(0, np.abs(res) - 0.1).sum().round(4))
a, b = np.array([[1., 2.]]), np.array([[2., 0.]])
print("RBF kernel by formula :", np.exp(-0.5 * np.sum((a - b) ** 2)).round(6), "   sklearn:", rbf_kernel(a, b, gamma=0.5).round(6)[0, 0])''')

    # ---------------------------------------------------------- 5 scratch linear
    nb.section("🔧", "5. Build it from scratch: linear SVR by sub-gradient descent",
               "The ε-insensitive loss has a *kink* (not differentiable at $|r|=\\varepsilon$), so we use a **sub-gradient**: any valid slope at the kink works. "
               "For the primal objective with a linear model, divide by $Cn$ to get the well-scaled form:\n\n"
               r"$$J(\mathbf w,b)=\frac{1}{2Cn}\lVert\mathbf w\rVert^2+\frac1n\sum_i L_\varepsilon(y_i-\mathbf w\cdot x_i-b)$$" "\n\n"
               r"Sub-gradient: $\nabla_{\mathbf w}J=\frac{\mathbf w}{Cn}-\frac1n\sum_{i\in V}\text{sign}(r_i)\,x_i$ and $\nabla_bJ=-\frac1n\sum_{i\in V}\text{sign}(r_i)$, "
               "where $V$ is the set of points **outside the tube** ($|r_i|>\\varepsilon$): *only violators push on the model*, mirroring the support-vector picture.")
    nb.code(r'''class LinearSVRScratch:
    """Linear epsilon-SVR trained with full-batch sub-gradient descent (inputs should be standardised)."""

    def __init__(self, C=10.0, epsilon=0.1, lr=0.2, n_iter=4000):
        self.C, self.epsilon, self.lr, self.n_iter = C, epsilon, lr, n_iter

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); n, p = X.shape
        w, b = np.zeros(p), 0.0
        self.loss_history_ = []
        for t in range(self.n_iter):
            r = y - (X @ w + b)
            viol = np.abs(r) > self.epsilon                                # points outside the tube
            s = np.sign(r) * viol
            self.loss_history_.append(w @ w / (2 * self.C * n) + np.mean(np.maximum(0, np.abs(r) - self.epsilon)))
            gw = w / (self.C * n) - (X.T @ s) / n
            gb = -np.mean(s)
            step = self.lr / (1 + 0.01 * t)                                # decaying learning rate
            w -= step * gw; b -= step * gb
        self.coef_, self.intercept_ = w, b
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


rng = np.random.default_rng(4)
xl = rng.uniform(0, 10, 70); yl = 2 + 0.7 * xl + rng.normal(0, 0.8, 70); yl[:3] += 6      # three outliers
zx = ((xl - xl.mean()) / xl.std()).reshape(-1, 1); zy = (yl - yl.mean()) / yl.std()

mine = LinearSVRScratch(C=10, epsilon=0.15).fit(zx, zy)
ref_l = SVR(kernel="linear", C=10, epsilon=0.15).fit(zx, zy)
ols = np.polyfit(zx.ravel(), zy, 1)
print(f"scratch  : slope={mine.coef_[0]:.4f} intercept={mine.intercept_:.4f}")
print(f"sklearn  : slope={ref_l.coef_[0][0]:.4f} intercept={ref_l.intercept_[0]:.4f}")
print(f"OLS      : slope={ols[0]:.4f} intercept={ols[1]:.4f}   <- pulled around by the 3 outliers")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
g = np.linspace(zx.min(), zx.max(), 50)
ax[0].scatter(zx, zy, s=18); ax[0].scatter(zx[:3], zy[:3], color="red", s=45, label="outliers")
ax[0].plot(g, mine.coef_[0] * g + mine.intercept_, "k", lw=2.4, label="SVR (scratch)"); ax[0].plot(g, np.polyval(ols, g), "g--", lw=2, label="OLS"); ax[0].legend(); ax[0].set_title("SVR resists the outliers better than OLS")
ax[1].plot(mine.loss_history_); ax[1].set_yscale("log"); ax[1].set_title("Objective J during sub-gradient descent"); ax[1].set_xlabel("iteration")
plt.tight_layout(); plt.show()''')
    nb.md("Our 25-line optimiser lands on (essentially) the same line as scikit-learn's solver, and (thanks to the linear tail of the ε-insensitive loss) it is far less distracted by the three outliers than OLS.")

    # ---------------------------------------------------------- 6 reconstruct
    nb.section("🔬", "6. Opening the black box: rebuild an RBF-SVR's prediction by hand",
               "After `fit`, scikit-learn stores `support_vectors_`, `dual_coef_` (the $\\alpha_i-\\alpha_i^*$) and `intercept_`. "
               "The prediction formula is nothing but $\\hat f(x)=\\sum_i \\text{dual\\_coef}_i\\,K(sv_i,x)+b$. Let's compute it ourselves:")
    nb.code(r'''def rbf(A, B, gamma):
    """K[i, j] = exp(-gamma * ||A_i - B_j||^2)   (all pairs, via broadcasting)"""
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)

new_x = np.array([[0.7], [2.1], [4.0], [5.5]])
K = rbf(svr.support_vectors_, new_x, gamma=1.0)                    # (n_SV x n_new)
manual = svr.dual_coef_ @ K + svr.intercept_                       # weighted sum of bumps + bias
print("manual    :", manual.ravel().round(6))
print("svr.predict:", svr.predict(new_x).round(6))
print("\nnumber of support vectors:", len(svr.support_), "out of", len(x), "| dual coefficients lie in [-C, C]:",
      svr.dual_coef_.min().round(2), "to", svr.dual_coef_.max().round(2))''')
    nb.callout("key", "That is *the entire model*: a list of support vectors, a coefficient for each, and a bias. Prediction cost is therefore **O(n_SV × p)** per row: "
                      "which is why SVR with many support vectors is slower to predict than a linear model. "
                      "Coefficients are bounded by C: points that are *far* outside the tube hit the ceiling ($|\\alpha|=C$); points exactly on the tube edge have smaller values.")

    # ---------------------------------------------------------- 7 params
    nb.section("🎛️", "7. C, epsilon and gamma in action",
               "Three parameter sweeps on the sine data. Watch the curve *and* the support-vector count.")
    nb.code(r'''def panel(ax, **kw):
    p = dict(C=10, epsilon=0.2, gamma=1.0); p.update(kw)
    m = SVR(kernel="rbf", **p).fit(X1, y)
    ax.scatter(x, y, s=12, alpha=0.6); ax.plot(grid, m.predict(grid), "r", lw=2.2); ax.plot(grid, np.sin(grid), "g--", lw=1)
    ax.set_title(", ".join(f"{k}={v}" for k, v in kw.items()) + f"\n{len(m.support_)} support vectors"); ax.set_ylim(-1.8, 1.8)

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, C in zip(axes[0], [0.05, 1, 1000]): panel(ax, C=C)
for ax, g in zip(axes[1], [0.02, 1, 60]):   panel(ax, gamma=g)
for ax, e in zip(axes[2], [0.02, 0.2, 0.9]): panel(ax, epsilon=e)
axes[0, 0].set_ylabel("vary C"); axes[1, 0].set_ylabel("vary gamma"); axes[2, 0].set_ylabel("vary epsilon")
plt.tight_layout(); plt.show()''')
    nb.md("""- **Row 1 (C):** tiny C → flat, underfit curve; huge C → follows noise.
- **Row 2 (γ):** tiny γ → too smooth; huge γ → each support vector makes a narrow spike (overfit, the curve returns to the mean away from data).
- **Row 3 (ε):** small ε → nearly every point is a support vector (slow, model memorises noise); large ε → tube too wide, signal lost.

A useful mental model: **C and γ control complexity (like tree depth); ε controls sparsity/tolerance.** Tune C and γ together: they interact.""")

    # ---------------------------------------------------------- 8 scaling
    nb.section("⚖️", "8. Scaling matters: for X *and* for y",
               "SVR depends on **distances** (RBF) and an absolute tolerance ε. Therefore:\n\n"
               "1. **Scale X.** If `lot_size_sqft` ranges to 30 000 and `bathrooms` to 6, the distance is dominated by lot size alone.\n"
               "2. **Think about the scale of y.** ε is measured in *y's units*, and the penalty C acts on residuals in y's units. A price in millions (4–120) with the default ε = 0.1, C = 1 is badly mismatched. "
               "Standardising y (inside the pipeline, with `TransformedTargetRegressor`) makes the defaults sensible.")
    nb.code(r'''houses = pd.read_csv(data_path("house_prices.csv"))
hc = houses[houses["sqft"] <= 6500].reset_index(drop=True)
num_cols = ["sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "has_pool", "distance_to_cbd_km", "lot_size_sqft"]
Xall = pd.get_dummies(hc[num_cols + ["neighborhood"]], columns=["neighborhood"], dtype=float)
Xall["condition_ord"] = hc["condition"].map({"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3})
yall = hc["price_million_kes"]
Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.2, random_state=RANDOM_STATE)
med = Xtr.median(); Xtr, Xte = Xtr.fillna(med).astype(float), Xte.fillna(med).astype(float)

variants = {
    "1. Raw X, raw y (defaults)":              SVR(),
    "2. Scaled X, raw y (defaults)":           make_pipeline(StandardScaler(), SVR()),
    "3. Scaled X, raw y, C=100":               make_pipeline(StandardScaler(), SVR(C=100)),
    "4. Scaled X AND scaled y (defaults)":     TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR()), transformer=StandardScaler()),
}
for name, m in variants.items():
    m.fit(Xtr, ytr); print(f"{name:<40} test R² = {m.score(Xte, yte):.3f}")''')
    nb.md("Same algorithm, same data, **R² from 0.4 to 0.94** purely through scaling decisions. (Variant 3 also works because a huge C compensates for y's scale, but scaling y is cleaner.)")

    # ---------------------------------------------------------- 9 tuning
    nb.section("🏁", "9. Tuning and comparison on house prices",
               "Grid-search C, γ and ε with 3-fold CV (each fit is fast: n = 1 600).")
    nb.code(r'''pipe = TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR()), transformer=StandardScaler())
grid_s = {"regressor__svr__C": [1, 10, 100], "regressor__svr__gamma": [0.003, 0.01, 0.03, "scale"], "regressor__svr__epsilon": [0.02, 0.1]}
gs = GridSearchCV(pipe, grid_s, cv=3, scoring="r2", n_jobs=-1).fit(Xtr, ytr)
print("best:", gs.best_params_, " CV R² =", round(gs.best_score_, 3))
best_svr = gs.best_estimator_
pred = best_svr.predict(Xte)
print(f"TEST R² = {r2_score(yte, pred):.3f}   RMSE = {np.sqrt(mean_squared_error(yte, pred)):.2f} M KES   "
      f"median abs % error = {100 * np.median(np.abs(yte - pred) / yte):.1f}%")
n_sv = len(best_svr.regressor_[-1].support_)
print(f"Support vectors: {n_sv} of {len(Xtr)} training rows ({n_sv / len(Xtr):.0%})")''')
    nb.md("SVR with an RBF kernel is *smooth* and flexible: here it **matches or beats** the log-linear model and the forest from earlier lessons. Its weakness shows up in **time**:")
    nb.code(r'''import time
rows = []
for n_sub in [200, 400, 800, 1600]:
    Xs, ys = Xtr.iloc[:n_sub], ytr.iloc[:n_sub]
    m = TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR(C=10, gamma=0.01)), transformer=StandardScaler())
    t0 = time.perf_counter(); m.fit(Xs, ys); fit_t = time.perf_counter() - t0
    rows.append({"training rows": n_sub, "fit time (s)": fit_t})
d = pd.DataFrame(rows); d["time ratio vs previous"] = d["fit time (s)"] / d["fit time (s)"].shift(1)
display(d.round(3))''')
    nb.callout("warn", "Kernel SVR training scales between **O(n²) and O(n³)** in the worst case (the table above shows ×1.6–3.6 per doubling on this small, friendly dataset, and the ratio tends to grow with n). Beyond ~50–100 k rows it becomes impractical. "
                       "For large data use `LinearSVR`, SGD-based regressors, or tree ensembles.")

    # ---------------------------------------------------------- 10 profile
    nb.section("📋", "10. Cheat-sheet: the full profile of Support Vector Regression")
    nb.table([
        ["**Underlying concept**", "Fit the flattest function that keeps most points within an ε-tube; points on/outside the tube (support vectors) define it."],
        ["**How it works**", "Solve a convex quadratic programme (dual form). Prediction = Σ (αᵢ − αᵢ*)·K(svᵢ, x) + b."],
        ["**Mathematical foundation**", "ε-insensitive loss; ½‖w‖² + C·Σslack; Lagrangian duality & KKT; Mercer kernels (RBF, polynomial, linear)."],
        ["**Important parameters**", "`kernel`, `C`, `epsilon`, `gamma` (RBF/poly), `degree` (poly), `coef0`, `tol`, `shrinking`. Always scale X (and usually y)."],
        ["**Advantages**", "Smooth non-linear fits; robust to outliers (linear loss); effective in high dimensions; convex ⇒ unique global optimum; kernels allow custom similarity."],
        ["**Disadvantages**", "Needs scaling; sensitive to C/γ/ε; training is O(n²–n³); no native probabilities/intervals; model is hard to interpret; prediction cost grows with #support vectors."],
        ["**Complexity**", "Training ≈ O(n² p) to O(n³); prediction O(n_SV · p); memory O(n²) for the kernel cache. LinearSVR: ≈ O(n · p) per epoch."],
        ["**Use cases**", "Small/medium tabular data with smooth non-linear structure, noisy engineering/sensor data, time-series with engineered lags, when robustness to outliers matters."],
        ["**Limitations**", "Doesn't scale to huge n; hard to explain to stakeholders; poor with categorical data unless encoded; needs careful tuning (grid search); extrapolation is poor with RBF (reverts to the bias)."],
    ], ["Question", "Answer"])

    # ---------------------------------------------------------- 11 worked
    nb.section("🧪", "11. Worked examples")
    nb.md(r"""### Worked Example 1: Which points are support vectors?

A fitted line gives these residuals (true − predicted) for five points: $[0.03,\ -0.20,\ 0.08,\ 0.55,\ -0.09]$. With $\varepsilon=0.1$, which are support vectors and what is the total ε-insensitive loss?

**Solution.** Support vectors lie **on or outside** the tube $|r|\ge\varepsilon$: points 2 ($|−0.20|$) and 4 ($0.55$). Point 5 ($0.09<0.1$) is inside, as are 1 and 3. Losses: $0.20-0.1=0.10$ and $0.55-0.1=0.45$ → total $\mathbf{0.55}$ (points inside contribute nothing).""")
    nb.md(r"""### Worked Example 2: Predict from the dual form

An RBF-SVR (γ = 1) has two support vectors $sv_1=0$, $sv_2=2$, dual coefficients $[1.5,\,-0.5]$ and intercept $b=1$. Predict at $x=1$.

**Solution.** $K(0,1)=e^{-1}=0.3679$ and $K(2,1)=e^{-1}=0.3679$. $\hat f(1)=1.5(0.3679)-0.5(0.3679)+1=0.3679+1=\mathbf{1.368}$.""")
    nb.code(r'''k = np.exp(-1.0); print("prediction:", round(1.5 * k - 0.5 * k + 1, 4))''')

    # ---------------------------------------------------------- problems
    nb.problem(
        "ε-insensitive loss",
        "Write `eps_loss(residuals, eps)` returning the **total** ε-insensitive loss (sum over all residuals). "
        "Compute `loss_a` for residuals `[-0.5, 0.05, 0.3, -0.12]` with ε = 0.1, and `loss_b` for the same residuals with ε = 0.4. "
        "Then set `n_outside_a` to the number of residuals *outside* the ε = 0.1 tube.",
        ["Per residual: `max(0, |r| - eps)`. With NumPy: `np.maximum(0, np.abs(r) - eps)`.",
         "Sum the per-point losses.",
         "Residuals outside the tube: `np.sum(np.abs(r) > eps)`."],
        'def eps_loss(residuals, eps):\n    pass\n\nres = np.array([-0.5, 0.05, 0.3, -0.12])\nloss_a = loss_b = n_outside_a = None   # your code\n',
        'def eps_loss(residuals, eps):\n    return np.sum(np.maximum(0, np.abs(residuals) - eps))\n\nres = np.array([-0.5, 0.05, 0.3, -0.12])\nloss_a = eps_loss(res, 0.1); loss_b = eps_loss(res, 0.4)\nn_outside_a = int(np.sum(np.abs(res) > 0.1))\nprint(loss_a, loss_b, n_outside_a)',
        'check("ε=0.1 -> 0.62", lambda: abs(loss_a - 0.62) < 1e-9)\ncheck("ε=0.4 -> 0.10 (only the -0.5 residual pokes out)", lambda: abs(loss_b - 0.10) < 1e-9)\ncheck("3 residuals outside the ε=0.1 tube", lambda: n_outside_a == 3)',
        explain="A wider tube forgives more: raising ε from 0.1 to 0.4 cuts the loss from 0.62 to 0.10, and only one residual remains outside.")

    nb.problem(
        "Write your own RBF kernel",
        "Write `my_rbf(A, B, gamma)` returning the full kernel matrix between the rows of `A` (m×p) and `B` (k×p). Use broadcasting (no double loop). "
        "Verify against `rbf_kernel` for `A = rng.normal(size=(5, 3))`, `B = rng.normal(size=(4, 3))`, γ = 0.7, storing the max absolute difference in `max_err`. "
        "Also store `diag_ok` = whether `my_rbf(A, A, 0.7)` has all diagonal entries equal to 1.",
        ["Squared distances: `((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)`.",
         "Then `np.exp(-gamma * d2)`.",
         "A point's similarity to itself is exp(0) = 1."],
        'def my_rbf(A, B, gamma):\n    pass\n\nrng = np.random.default_rng(0); A = rng.normal(size=(5, 3)); B = rng.normal(size=(4, 3))\nmax_err = diag_ok = None   # your code\n',
        'def my_rbf(A, B, gamma):\n    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)\n    return np.exp(-gamma * d2)\n\nrng = np.random.default_rng(0); A = rng.normal(size=(5, 3)); B = rng.normal(size=(4, 3))\n'
        'max_err = np.abs(my_rbf(A, B, 0.7) - rbf_kernel(A, B, gamma=0.7)).max()\ndiag_ok = bool(np.allclose(np.diag(my_rbf(A, A, 0.7)), 1))\nprint(max_err, diag_ok)',
        'check("matches sklearn (error < 1e-12)", lambda: max_err < 1e-12)\ncheck("diagonal is all ones", lambda: diag_ok)',
        explain="Broadcasting builds every pairwise difference at once. The kernel matrix is symmetric, positive-semidefinite with ones on its diagonal: a 'similarity matrix'.",
        level="⭐⭐")

    nb.problem(
        "Sparsity vs tolerance",
        "On the sine data (`X1`, `y`) fit `SVR(kernel='rbf', C=10, gamma=1.0, epsilon=e)` for each `e` in `[0.01, 0.05, 0.1, 0.3, 0.6, 1.0]` and store the number of support vectors in a list `n_sv` (same order). "
        "Set `monotone` to `True` if the list never increases as ε grows.",
        ["`len(model.support_)` gives the number of support vectors.",
         "A wider tube holds more points *inside*, so fewer become support vectors.",
         "`monotone = all(a >= b for a, b in zip(n_sv, n_sv[1:]))`."],
        'eps_values = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]\nn_sv = []; monotone = None   # your code\n',
        'eps_values = [0.01, 0.05, 0.1, 0.3, 0.6, 1.0]\nn_sv = [len(SVR(kernel="rbf", C=10, gamma=1.0, epsilon=e).fit(X1, y).support_) for e in eps_values]\n'
        'monotone = all(a >= b for a, b in zip(n_sv, n_sv[1:]))\nprint(n_sv, monotone)',
        'check("6 values recorded", lambda: len(n_sv) == 6)\ncheck("almost all points are SVs when ε is tiny (> 90%)", lambda: n_sv[0] > 0.9 * len(y))\ncheck("count never increases with ε", lambda: monotone)\ncheck("few SVs when ε is huge (< 25%)", lambda: n_sv[-1] < 0.25 * len(y))',
        explain="ε controls *sparsity*: the tube absorbs more points as it widens, so fewer remain as support vectors (cheaper predictions, but a too-wide tube eventually erases the signal).")

    nb.problem(
        "Scale to survive",
        "Using the lesson's `Xtr, ytr, Xte, yte`, build two models: (a) `SVR()` on the raw features, (b) a `TransformedTargetRegressor` wrapping `make_pipeline(StandardScaler(), SVR())` with `transformer=StandardScaler()`. "
        "Store their test R² in `r2_raw` and `r2_scaled`, and `gain = r2_scaled - r2_raw`.",
        ["Fit each on `Xtr, ytr` and use `.score(Xte, yte)`.",
         "The scaled model must scale **both** X (in the pipeline) and y (in `TransformedTargetRegressor`).",
         "The gain should be huge."],
        'r2_raw = r2_scaled = gain = None   # your code\n',
        'm_raw = SVR().fit(Xtr, ytr)\nm_sc = TransformedTargetRegressor(make_pipeline(StandardScaler(), SVR()), transformer=StandardScaler()).fit(Xtr, ytr)\n'
        'r2_raw = m_raw.score(Xte, yte); r2_scaled = m_sc.score(Xte, yte); gain = r2_scaled - r2_raw\nprint(r2_raw, r2_scaled, gain)',
        'check("scaled SVR is strong (R² > 0.9)", lambda: r2_scaled > 0.9)\ncheck("scaling gains more than 0.3 R²", lambda: gain > 0.3)',
        explain="Without scaling, distances are dominated by lot_size_sqft and y's large scale makes the default C/ε inappropriate. Both fixes together turn a poor model into a top performer.")

    nb.problem(
        "Rebuild a prediction from support vectors",
        "Fit `m2 = SVR(kernel='rbf', C=5, gamma=0.5, epsilon=0.1)` on the two standardised advertising features "
        "`Z = StandardScaler().fit_transform(ads[['tv_budget_k', 'radio_budget_k']])` (the `ads` DataFrame is loaded for you) with standardised target `zt`. "
        "Compute `manual` = predictions for the **first 5 rows** using only `m2.support_vectors_`, `m2.dual_coef_`, `m2.intercept_` and the `rbf` function from Section 6 (γ = 0.5). "
        "Store `max_gap` = the max absolute difference with `m2.predict(Z[:5])`.",
        ["`K = rbf(m2.support_vectors_, Z[:5], gamma=0.5)` has shape (n_SV, 5).",
         "`manual = (m2.dual_coef_ @ K + m2.intercept_).ravel()`.",
         "`max_gap = np.abs(manual - m2.predict(Z[:5])).max()` should be ~1e-12."],
        'ads = pd.read_csv(data_path("advertising.csv"))\nZ = StandardScaler().fit_transform(ads[["tv_budget_k", "radio_budget_k"]]); zt = ((ads["sales_k_units"] - ads["sales_k_units"].mean()) / ads["sales_k_units"].std()).values\n'
        'm2 = SVR(kernel="rbf", C=5, gamma=0.5, epsilon=0.1).fit(Z, zt)\nmanual = max_gap = None   # your code\n',
        'ads = pd.read_csv(data_path("advertising.csv"))\nZ = StandardScaler().fit_transform(ads[["tv_budget_k", "radio_budget_k"]]); zt = ((ads["sales_k_units"] - ads["sales_k_units"].mean()) / ads["sales_k_units"].std()).values\n'
        'm2 = SVR(kernel="rbf", C=5, gamma=0.5, epsilon=0.1).fit(Z, zt)\nK2 = rbf(m2.support_vectors_, Z[:5], gamma=0.5)\nmanual = (m2.dual_coef_ @ K2 + m2.intercept_).ravel()\n'
        'max_gap = np.abs(manual - m2.predict(Z[:5])).max()\nprint(max_gap, len(m2.support_), "support vectors")',
        'check("manual prediction equals model.predict", lambda: max_gap < 1e-9)\ncheck("manual has 5 predictions", lambda: len(manual) == 5)',
        explain="The 'black box' is a weighted sum of Gaussian bumps centred on the support vectors. Understanding this gives you the intuition for why γ, C and ε behave the way they do.",
        level="⭐⭐⭐")

    nb.problem(
        "Tune SVR with GridSearchCV",
        "Using `pipe` (the scaled-X-and-y SVR from Section 9) and `Xtr, ytr`, run `GridSearchCV` (cv=3, `scoring='r2'`, `n_jobs=-1`) over "
        "`C ∈ [1, 10]`, `gamma ∈ [0.01, 0.03]` and `epsilon ∈ [0.05]` (remember the `regressor__svr__` prefix). Store `best_params` and the test R² `tuned_r2`.",
        ["Keys: `'regressor__svr__C'`, `'regressor__svr__gamma'`, `'regressor__svr__epsilon'`.",
         "`gs2.best_params_`, `gs2.best_estimator_.score(Xte, yte)`.",
         "Expect R² above 0.93."],
        'best_params = tuned_r2 = None   # your code\n',
        'gs2 = GridSearchCV(pipe, {"regressor__svr__C": [1, 10], "regressor__svr__gamma": [0.01, 0.03], "regressor__svr__epsilon": [0.05]}, cv=3, scoring="r2", n_jobs=-1).fit(Xtr, ytr)\n'
        'best_params = gs2.best_params_; tuned_r2 = gs2.best_estimator_.score(Xte, yte)\nprint(best_params, round(tuned_r2, 3))',
        'check("three hyper-parameters tuned", lambda: len(best_params) == 3)\ncheck("tuned test R² > 0.93", lambda: tuned_r2 > 0.93)',
        explain="With sensible scaling, SVR is *forgiving*: nearly every grid cell scores above 0.93. Tuning refines rather than rescues.",
        level="⭐⭐")

    nb.quiz([
        ("What is a support vector in SVR?", "A training point lying on or outside the ε-tube. Only these have non-zero dual coefficients and influence the fitted function."),
        ("If you double ε, what generally happens to the number of support vectors and to the fit?",
         "Fewer support vectors (more points fall inside the tube) and a smoother/flatter, possibly less accurate fit; if ε is too large the tube swallows the signal."),
        ("Why does the RBF kernel with a very large γ overfit?",
         "Each support vector's bump becomes extremely narrow, so the function is a set of spikes at the training points and reverts to the intercept elsewhere."),
        ("Why must we standardise features before SVR?", "RBF similarity uses Euclidean distance; unscaled features with large ranges dominate the distance and effectively silence the others."),
        ("SVR vs Random Forest on 2 million rows, which is more practical and why?",
         "Random Forest (or a linear/SGD model). Kernel SVR training is O(n²–n³) and becomes impractical at that size."),
    ])

    nb.footer(["SVR minimises ½‖w‖² + C·Σ ε-insensitive loss: flat functions, errors inside the tube are free.",
               "Only support vectors (on/outside the tube) define the model: ŷ = Σ(αᵢ−αᵢ*)K(svᵢ,x)+b.",
               "The kernel trick (RBF) gives smooth non-linear fits without explicit feature expansion.",
               "C and γ set complexity, ε sets tolerance/sparsity; always scale X, and usually y too.",
               "Powerful on small/medium data, but training is O(n²–n³) and the model is hard to interpret."],
              "K-Nearest Neighbors Regression", "07_KNN_Regression.ipynb")
    return nb
