"""
Bespoke curriculum specifications (Seaborn EDA, live diagnostics, sklearn benchmarks, quizzes)
for algorithms in ml-model-atlas.
All data loading strictly uses actual generators available in mlatlas.data or explicit NumPy arrays.
"""

def get_specs():
    specs = {}

    # 00: Gradient Descent & Optimizers
    specs[0] = {
        "eda": (
            "### Step 1: Loss Surface & Landscape Exploration with Matplotlib & Seaborn\nWe examine the elongated valley loss surface $f(x, y) = \\frac{x^2}{20} + y^2$. The surface is 20 times steeper along the $y$-axis than along the $x$-axis, which causes naive gradient descent to oscillate between steep walls rather than moving smoothly down the valley.",
            """# Visualize the optimization landscape
xs = np.linspace(-10, 3, 200)
ys = np.linspace(-4, 4, 200)
X_grid, Y_grid = np.meshgrid(xs, ys)
Z_loss = X_grid**2 / 20.0 + Y_grid**2

plt.figure(figsize=(9, 4.5))
contour = plt.contourf(X_grid, Y_grid, Z_loss, levels=np.geomspace(0.05, 40, 20), cmap="Blues_r", alpha=0.85)
plt.colorbar(contour, label="Loss f(x, y)")
plt.scatter(0, 0, marker="*", color="gold", s=220, edgecolor="black", zorder=5, label="Global Minimum (0, 0)")
plt.title("Optimization Landscape: Elongated Valley (Condition Number = 20)", fontsize=12, fontweight="bold")
plt.xlabel("Parameter x (Flatter Direction)")
plt.ylabel("Parameter y (Steeper Direction)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Live Convergence Diagnostics & Trajectory Plots\nWe run SGD, Momentum, RMSprop, and Adam side by side from the exact same starting point $(-9, 3)$, subject to the same noisy stochastic gradients, and plot their trajectories and loss curves.",
            """# Instantiate optimizers and trace optimization trajectories
opt_instances = {
    "SGD": SGD(lr=0.08),
    "Momentum": Momentum(lr=0.08, beta=0.85),
    "RMSprop": RMSprop(lr=0.08, rho=0.9),
    "Adam": Adam(lr=0.15, beta1=0.85, beta2=0.95),
}
p0 = np.array([-9.0, 3.0])
steps = 60
rng = np.random.default_rng(2)
paths = {k: [p0.copy()] for k in opt_instances}

for _ in range(steps):
    noise = rng.normal(0, 0.4, 2)
    for k, opt in opt_instances.items():
        p = paths[k][-1]
        grad = np.array([p[0] / 10.0, 2.0 * p[1]]) + noise
        paths[k].append(opt.step(p, grad))

paths = {k: np.array(v) for k, v in paths.items()}
cols = {"SGD": "#e45756", "Momentum": "#f58518", "RMSprop": "#54a24b", "Adam": "#4c78a8"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot trajectories on contour
ax1.contour(xs, ys, Z_loss, levels=np.geomspace(0.05, 40, 14), cmap="Greys", alpha=0.6)
for k, P in paths.items():
    ax1.plot(P[:, 0], P[:, 1], color=cols[k], lw=2, label=k, alpha=0.9)
    ax1.scatter(P[-1, 0], P[-1, 1], color=cols[k], s=50, edgecolor="k")
ax1.scatter(0, 0, marker="*", s=180, color="gold", edgecolor="k", zorder=6, label="Minimum")
ax1.set_title("Optimizer Trajectories in Parameter Space", fontsize=11, fontweight="bold")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend(loc="upper left")

# Plot loss reduction curves
for k, P in paths.items():
    losses = [p[0]**2 / 20.0 + p[1]**2 for p in P]
    ax2.semilogy(losses, color=cols[k], lw=2, label=k)
ax2.set_title("Loss Reduction vs. Step (Log Scale)", fontsize=11, fontweight="bold")
ax2.set_xlabel("Step")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Comparison with scikit-learn)\nWe compare against `sklearn.linear_model.SGDRegressor` to verify that stochastic gradient updates and learning rate mechanics match industry standards.",
            """from sklearn.linear_model import SGDRegressor

X_dummy = np.random.randn(100, 2)
y_dummy = X_dummy @ [1.5, -2.0] + 0.1 * np.random.randn(100)

sk_sgd = SGDRegressor(loss="squared_error", learning_rate="constant", eta0=0.01, max_iter=200, random_state=42)
sk_sgd.fit(X_dummy, y_dummy)
print(f"scikit-learn SGD learned weights: {sk_sgd.coef_.round(4)}")"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why does plain SGD oscillate wildly in a narrow valley?</b></summary>
Because the gradient vector points almost perpendicular to the long valley floor (along the direction of steepest curvature). Consequently, steps along the steep walls are large, while progress along the flatter minimum is slow.
</details>

<details>
<summary><b>2. How does Adam correct for both gradient direction and scale?</b></summary>
Adam combines momentum (an exponential moving average of past gradients to maintain direction) with RMSprop (dividing by the square root of past squared gradients to rescale coordinates with uneven curvature).
</details>"""
    }

    # 01: Bias / Variance / Overfitting
    specs[1] = {
        "eda": (
            "### Step 1: Exploratory Data Visualization with Seaborn\nWe examine a noisy non-linear curve dataset generated from a true underlying sinusoidal function $f(x) = \\sin(3x) + 0.5x$ with added Gaussian noise.",
            """x, y = data.curve(n=40, noise=0.25)

plt.figure(figsize=(8, 4.5))
sns.scatterplot(x=x, y=y, color="#3b6fb6", s=65, label="Noisy Observations", edgecolor="w")
x_dense = np.linspace(-1, 1, 200)
plt.plot(x_dense, data.curve_fn(x_dense), color="black", linestyle="--", lw=2, label="True Underlying Function f(x)")
plt.title("Noisy Observations vs. Ground Truth Curve", fontsize=12, fontweight="bold")
plt.xlabel("Input Feature x")
plt.ylabel("Target y")
plt.legend()
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Polynomial Degrees & The Bias-Variance Tradeoff\nWe fit polynomials of degree 1 (high bias, underfitting), degree 3 (balanced), and degree 15 (high variance, overfitting), and plot their fits side by side.",
            """fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
degrees = [1, 3, 15]
titles = ["Degree 1: High Bias (Underfit)", "Degree 3: Balanced Generalization", "Degree 15: High Variance (Overfit)"]
colors = ["#e45756", "#54a24b", "#f58518"]

for ax, deg, title, col in zip(axes, degrees, titles, colors):
    p = np.polyfit(x, y, deg)
    y_pred = np.polyval(p, x_dense)
    ax.scatter(x, y, color="#3b6fb6", s=35, alpha=0.75, label="Data")
    ax.plot(x_dense, data.curve_fn(x_dense), "k--", lw=1.5, label="True f(x)")
    ax.plot(x_dense, y_pred, color=col, lw=2.5, label=f"Poly (deg={deg})")
    ax.set_ylim(-2.0, 2.0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("x")
    ax.legend(loc="upper left", fontsize=8)

plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Comparison with scikit-learn)\nWe compare with `sklearn.preprocessing.PolynomialFeatures` + `LinearRegression`.",
            """from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
pipe.fit(x.reshape(-1, 1), y)
print("scikit-learn Degree-3 Polynomial Coefficients:")
print(pipe.named_steps['linearregression'].coef_.round(4))"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why does high variance lead to catastrophic test errors despite near-zero training error?</b></summary>
A model with high variance (like degree 15 polynomial) has excessive degrees of freedom, allowing it to fit the random stochastic noise of the training sample. When evaluated on new sample points from the same data generator, the erratic oscillations produce massive squared residuals.
</details>

<details>
<summary><b>2. How can we detect whether a model suffers from high bias or high variance?</b></summary>
By inspecting learning curves. If both training error and validation error are high and close together, the model suffers from high bias. If training error is very low but validation error is substantially higher, the model suffers from high variance (overfitting).
</details>"""
    }

    # 02: Classification Metrics, Thresholds & ROC
    specs[2] = {
        "eda": (
            "### Step 1: Exploratory Score Distribution with Seaborn\nWe examine synthetic loan applicant default predictions, analyzing predicted probabilities across positive and negative classes.",
            """X, y, names = data.loan_default(n=300, seed=0)
Z, _, _ = data.standardize(X)
# Generate synthetic confidence scores via logistic function
scores = 1.0 / (1.0 + np.exp(-(Z @ [1.2, -0.8])))

plt.figure(figsize=(9, 4.5))
sns.kdeplot(scores[y == 0], fill=True, color="#4c78a8", label="Repaid (Class 0)", alpha=0.5)
sns.kdeplot(scores[y == 1], fill=True, color="#e45756", label="Defaulted (Class 1)", alpha=0.5)
plt.axvline(0.5, color="black", linestyle="--", lw=2, label="Decision Threshold (0.5)")
plt.title("Predicted Score Distributions for Repaid vs. Defaulted Loans", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Probability / Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: ROC Curve & Confusion Matrix Heatmap\nWe calculate true positive and false positive rates across all thresholds to construct the Receiver Operating Characteristic (ROC) curve.",
            """fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 1. Confusion Matrix Heatmap at threshold = 0.5
y_pred = (scores >= 0.5).astype(int)
cm = np.array([
    [((y == 0) & (y_pred == 0)).sum(), ((y == 0) & (y_pred == 1)).sum()],
    [((y == 1) & (y_pred == 0)).sum(), ((y == 1) & (y_pred == 1)).sum()]
])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1,
            xticklabels=["Pred Repaid (0)", "Pred Default (1)"], yticklabels=["Actual Repaid (0)", "Actual Default (1)"])
ax1.set_title("Confusion Matrix (Threshold = 0.5)", fontsize=11, fontweight="bold")

# 2. ROC Curve sweep
thresholds = np.linspace(0, 1, 100)
tprs, fprs = [], []
P = (y == 1).sum()
N = (y == 0).sum()
for th in thresholds:
    yp = (scores >= th).astype(int)
    tprs.append(((y == 1) & (yp == 1)).sum() / max(P, 1))
    fprs.append(((y == 0) & (yp == 1)).sum() / max(N, 1))

ax2.plot(fprs, tprs, color="#e45756", lw=2.5, label="Model ROC Curve")
ax2.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC=0.5)")
ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
ax2.set_xlabel("False Positive Rate (FPR)")
ax2.set_ylabel("True Positive Rate (TPR / Recall)")
ax2.legend()

plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Comparison with scikit-learn)\nWe compare against `sklearn.metrics.roc_auc_score`.",
            """from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

auc = roc_auc_score(y, scores)
acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)

print(f"ROC-AUC Score: {auc:.4f}")
print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why is accuracy a misleading metric when class imbalance is severe?</b></summary>
If 99% of transactions are legitimate and 1% are fraudulent, a trivial model that always predicts 'legitimate' achieves 99% accuracy while detecting 0 frauds. Precision, Recall, and PR-AUC are far more informative.
</details>

<details>
<summary><b>2. How should you choose an operating threshold in production?</b></summary>
The threshold must balance business costs: if a False Negative is catastrophic (e.g. cancer screening), lower the threshold to prioritize Recall; if a False Positive is costly (e.g. spam blocking legitimate emails), raise the threshold to prioritize Precision.
</details>"""
    }

    # 03: Linear Regression
    specs[3] = {
        "eda": (
            "### Step 1: Exploratory Data Visualization with Seaborn\nWe examine synthetic weekly advertising spend versus units sold. The relationship shows a clear linear trend with Gaussian noise, alongside a multiple regression dataset of house prices.",
            """X_ads, y_ads = data.sales_vs_ads()
x_ads = X_ads[:, 0]

X_house, y_house, house_feat_names = data.house_prices(n=300)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Simple Linear Regression Data (Ad spend vs Sales)
sns.scatterplot(x=x_ads, y=y_ads, color="#3b6fb6", s=60, alpha=0.85, edgecolor="w", ax=ax1)
ax1.set_title("Univariate Dataset: Ad Spend vs. Units Sold", fontsize=11, fontweight="bold")
ax1.set_xlabel("Ad Spend ($1,000s)")
ax1.set_ylabel("Units Sold")

# 2. House Prices correlation heatmap
df_house = pd.DataFrame(X_house, columns=house_feat_names)
df_house["price_k"] = y_house
sns.heatmap(df_house.corr(), annot=True, fmt=".2f", cmap="vlag", ax=ax2)
ax2.set_title("Multivariate Dataset: House Prices Correlation Matrix", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Fitted Model, Residual Diagnostics & Loss Surface\nWe visualize: (1) The fitted univariate regression line with residual error segments, (2) The 2D loss surface $L(w, b)$ with the gradient descent trajectory path, and (3) Multiple regression feature weights.",
            """# 1. Fit Univariate Model with Gradient Descent
w, b = -1.0, 12.0
lr = 0.03
n = len(y_ads)
hist = [(w, b)]
for _ in range(1200):
    err = w * x_ads + b - y_ads
    w -= lr * (err * x_ads).mean()
    b -= lr * err.mean()
    hist.append((w, b))

# Closed-form Normal Equation
wb_normal = np.linalg.lstsq(np.c_[x_ads, np.ones(n)], y_ads, rcond=None)[0]

# 2. Loss surface grid
Wg, Bg = np.meshgrid(np.linspace(-1, 5, 80), np.linspace(-6, 14, 80))
L = (((Wg[..., None] * x_ads + Bg[..., None] - y_ads) ** 2).mean(-1)) / 2.0
hp = np.array(hist)

# Plot 3-panel diagnostics
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))

# Panel 1: Fitted Line & Residuals
ax1.scatter(x_ads, y_ads, color="#3b6fb6", s=40, alpha=0.8, label="Data")
x_line = np.linspace(0, 10, 100)
ax1.plot(x_line, w * x_line + b, color="#e45756", lw=2.5, label=f"Fit: y = {w:.2f}x + {b:.2f}")
ax1.vlines(x_ads, y_ads, w * x_ads + b, color="#e45756", alpha=0.35, label="Residuals")
ax1.set_title("Fitted Regression Line & Residuals", fontsize=11, fontweight="bold")
ax1.set_xlabel("Ad Spend (k$)")
ax1.set_ylabel("Units Sold")
ax1.legend(loc="upper left")

# Panel 2: Loss Surface & Gradient Descent Path
ax2.contourf(Wg, Bg, L, levels=np.geomspace(L.min() + 0.5, L.max(), 20), cmap="Blues_r", alpha=0.85)
ax2.plot(hp[::10, 0], hp[::10, 1], color="k", lw=1.2, label="GD Path")
ax2.scatter(w, b, color="#e45756", s=60, zorder=5, label="Final (w, b)")
ax2.scatter(*wb_normal, marker="*", s=160, color="gold", edgecolor="k", zorder=6, label="Normal Eq.")
ax2.set_title("Loss Surface L(w, b) & Descent Path", fontsize=11, fontweight="bold")
ax2.set_xlabel("Slope (w)")
ax2.set_ylabel("Intercept (b)")
ax2.legend()

# Panel 3: Multiple Regression Feature Weights
Zh, _, _ = data.standardize(X_house)
ym = y_house.mean()
wh = np.linalg.lstsq(np.c_[Zh, np.ones(len(y_house))], y_house - ym, rcond=None)[0][:4]
sns.barplot(x=house_feat_names, y=wh, palette="vlag", ax=ax3)
ax3.axhline(0, color="k", linestyle="--", alpha=0.6)
ax3.set_title("Standardized Coefficients (House Prices)", fontsize=11, fontweight="bold")
ax3.set_ylabel("Effect on Price (k$) per 1-SD")

plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Comparison with scikit-learn)\nWe compare our from-scratch parameters against `sklearn.linear_model.LinearRegression` to verify exact analytical and numerical equivalence.",
            """from sklearn.linear_model import LinearRegression

sk_model = LinearRegression().fit(x_ads.reshape(-1, 1), y_ads)

print("--- Parameter Estimates Comparison ---")
print(f"NumPy Gradient Descent: w = {w:.4f}, b = {b:.4f}")
print(f"scikit-learn OLS:       w = {sk_model.coef_[0]:.4f}, b = {sk_model.intercept_:.4f}")
print(f"Normal Equation:        w = {wb_normal[0]:.4f}, b = {wb_normal[1]:.4f}")
print(f"Mathematical equivalence confirmed within tolerance: {np.isclose(w, sk_model.coef_[0], atol=1e-2)}")"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why does squaring the error punish large outliers more than small ones?</b></summary>
Because squaring is a convex function where the derivative $2(y - \\hat{y})$ grows linearly with error magnitude. An error of 10 units contributes 100 to the loss, whereas an error of 1 unit contributes only 1.
</details>

<details>
<summary><b>2. When is Gradient Descent preferred over the analytical Normal Equation?</b></summary>
When the number of features $d$ is very large (e.g. $d > 10,000$), calculating the inverse $(X^T X)^{-1}$ requires $O(d^3)$ operations and can exceed memory limits. Gradient Descent scales at $O(k \\cdot n \\cdot d)$.
</details>"""
    }

    # 04: Regularization: Ridge, Lasso & Elastic Net
    specs[4] = {
        "eda": (
            "### Step 1: Feature Matrix & Correlation Exploration with Seaborn\nWe examine a house prices dataset extended with 6 pure-noise features to simulate real-world high-dimensional regression with noisy predictors.",
            """rng = np.random.default_rng(0)
X, y, names = data.house_prices(300, seed=1)
X = np.c_[X, rng.normal(size=(300, 6))]
names = names + [f"noise{i}" for i in range(1, 7)]

plt.figure(figsize=(9, 6.5))
corr_matrix = np.corrcoef(X.T)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="vlag", xticklabels=names, yticklabels=names)
plt.title("Feature Correlation Matrix: 4 Informative Features + 6 Pure Noise Features", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Coefficient Regularization Paths (Ridge vs. Lasso)\nWe plot how feature weights shrink as the penalty parameter $\\lambda$ increases. Notice that Lasso drives weights exactly to zero (sparse feature selection), whereas Ridge shrinks them smoothly.",
            """# Prepare standardized data splits
Xtr, Xte, ytr, yte = data.train_test_split(X, y, test=0.5, seed=2)
Ztr, mu, sd = data.standardize(Xtr)
Zte = (Xte - mu) / sd
ym, ys = ytr.mean(), ytr.std()
ytr_, yte_ = (ytr - ym) / ys, (yte - ym) / ys

lams = np.logspace(-3, 1, 30)
ridge = np.array([enet(Ztr, ytr_, l, 0.0, 60) for l in lams])
lasso = np.array([enet(Ztr, ytr_, l, 1.0) for l in lams[::-1]])[::-1]

te = lambda W: [np.mean((Zte @ w - yte_) ** 2) for w in W]
ter, tel = te(ridge), te(lasso)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))
colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"] + ["#aaaaaa"] * 6

# Panel 1: Ridge Paths
for j in range(10):
    ax1.plot(lams, ridge[:, j], color=colors[j], lw=2 if j < 4 else 1, label=names[j] if j < 4 else None)
ax1.set_xscale("log")
ax1.set_title("Ridge (L2): Smooth Shrinkage", fontsize=11, fontweight="bold")
ax1.set_xlabel("Regularization Strength (lambda)")
ax1.set_ylabel("Coefficient")
ax1.legend(loc="upper right", fontsize=8)

# Panel 2: Lasso Paths
for j in range(10):
    ax2.plot(lams, lasso[:, j], color=colors[j], lw=2 if j < 4 else 1, label=names[j] if j < 4 else None)
ax2.set_xscale("log")
ax2.set_title("Lasso (L1): Exact Feature Selection", fontsize=11, fontweight="bold")
ax2.set_xlabel("Regularization Strength (lambda)")

# Panel 3: Test Error U-curve
ax3.semilogx(lams, ter, label="Ridge MSE", color="#4c78a8", lw=2)
ax3.semilogx(lams, tel, label="Lasso MSE", color="#f58518", lw=2)
ax3.set_title("Validation Error (Bias-Variance U-Curve)", fontsize=11, fontweight="bold")
ax3.set_xlabel("Regularization Strength (lambda)")
ax3.set_ylabel("Test MSE (Standardized)")
ax3.legend()

plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Comparison with scikit-learn)\nWe compare our from-scratch regularized models against `sklearn.linear_model.Ridge` and `sklearn.linear_model.Lasso`.",
            """from sklearn.linear_model import Ridge, Lasso

sk_ridge = Ridge(alpha=1.0).fit(Ztr, ytr_)
sk_lasso = Lasso(alpha=0.05).fit(Ztr, ytr_)
print("scikit-learn Ridge coefficients (first 4 informative):", sk_ridge.coef_[:4].round(3))
print("scikit-learn Lasso coefficients (first 4 informative):", sk_lasso.coef_[:4].round(3))
print("scikit-learn Lasso noise coefficients (should be ~0):", sk_lasso.coef_[4:].round(3))"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why does the L1 norm produce exact zeros while L2 does not?</b></summary>
The L1 constraint surface has sharp corners on the coordinate axes (a diamond shape in 2D). When the elliptical OLS loss contours expand, they typically contact the diamond at a vertex where one or more coordinates equal exactly zero.
</details>

<details>
<summary><b>2. Why is feature scaling mandatory before Ridge or Lasso?</b></summary>
Because the penalty $\\sum w_j^2$ or $\\sum |w_j|$ treats all weights equally. A feature measured in millimeters will have a much smaller coefficient than one in kilometers, causing unscaled penalties to unfairly penalize features simply based on their units.
</details>"""
    }

    # 05: Logistic Regression
    specs[5] = {
        "eda": (
            "### Step 1: Exploratory Classification Data Visualization\nWe visualize a synthetic dataset of loan applicants with income and credit debt ratio features, labeled as defaulted (1) or repaid (0).",
            """X, y, names = data.loan_default(n=300, seed=0)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=["#4c78a8", "#e45756"], s=60, alpha=0.9, edgecolor="w")
plt.title("Loan Applicants: Income vs. Debt Ratio", fontsize=12, fontweight="bold")
plt.xlabel("Income ($1,000s/yr)")
plt.ylabel("Debt Ratio (%)")
plt.legend(title="Default (1=Yes)", loc="upper right")
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Decision Boundary & Probability Surface\nWe evaluate the model by computing the decision boundary (where $P(y=1|x) = 0.5$) and plotting the sigmoid probability landscape.",
            """Z, mu, sd = data.standardize(X)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25, 25)))

w = np.zeros(2)
b = 0.0
lr = 0.1
for _ in range(800):
    p = sigmoid(Z @ w + b)
    grad_w = Z.T @ (p - y) / len(y)
    grad_b = np.mean(p - y)
    w -= lr * grad_w
    b -= lr * grad_b

# Create meshgrid for probability contour
x_min, x_max = Z[:, 0].min() - 0.5, Z[:, 0].max() + 0.5
y_min, y_max = Z[:, 1].min() - 0.5, Z[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
grid_points = np.c_[xx.ravel(), yy.ravel()]
probs = sigmoid(grid_points @ w + b).reshape(xx.shape)

plt.figure(figsize=(9, 5.5))
contour = plt.contourf(xx, yy, probs, levels=25, cmap="RdBu_r", alpha=0.75)
plt.colorbar(contour, label="P(Default = 1)")
plt.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2.5, linestyles="--")
sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=y, palette=["#4c78a8", "#e45756"], edgecolor="k", s=55, zorder=5)
plt.title("Logistic Regression: Probability Surface & Decision Boundary (P = 0.5)", fontsize=12, fontweight="bold")
plt.xlabel("Standardized Income")
plt.ylabel("Standardized Debt Ratio")
plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Comparison with scikit-learn)\nWe compare against `sklearn.linear_model.LogisticRegression`.",
            """from sklearn.linear_model import LogisticRegression

sk_clf = LogisticRegression(penalty=None, solver="lbfgs").fit(Z, y)
print("--- Model Parameters Comparison ---")
print(f"NumPy Scratch: w = {w.round(4)}, b = {b:.4f}")
print(f"scikit-learn:  w = {sk_clf.coef_[0].round(4)}, b = {sk_clf.intercept_[0]:.4f}")"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why can't we use Mean Squared Error (MSE) directly with Logistic Regression?</b></summary>
Because passing a linear combination through the non-linear sigmoid function turns the MSE loss surface into a non-convex function with multiple local minima. Binary Cross-Entropy (Log Loss) guarantees a convex surface with a unique global minimum.
</details>

<details>
<summary><b>2. What is the geometric interpretation of the decision boundary?</b></summary>
The decision boundary is the hyperplane defined by $w^T x + b = 0$, where the sigmoid output $\\sigma(0) = 0.5$. On one side, predicted probability is $> 0.5$; on the other, $< 0.5$.
</details>"""
    }

    # 07: k-Nearest Neighbors (k-NN)
    specs[7] = {
        "eda": (
            "### Step 1: Spatial Class Distribution Visualization\nWe visualize interlocking two moons clusters to explore instance-based non-linear boundaries.",
            """X, y = data.moons(n=300, noise=0.25, seed=0)

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=["#4c78a8", "#f58518"], s=55, edgecolor="k")
plt.title("Two Moons Dataset: Non-Linear Decision Geometry", fontsize=12, fontweight="bold")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Live Decision Boundary Surfaces across k\nWe compare decision boundaries for $k=1$ (flexible, sensitive to noise), $k=15$ (balanced), and $k=51$ (smooth, oversmoothed).",
            """fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
ks = [1, 15, 51]
titles = ["k = 1: High Variance (Memorization)", "k = 15: Balanced Boundary", "k = 51: High Bias (Oversmoothed)"]

x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
gx, gy = np.meshgrid(np.linspace(x0_min, x0_max, 100), np.linspace(x1_min, x1_max, 100))
grid_flat = np.c_[gx.ravel(), gy.ravel()]

for ax, k, title in zip(axes, ks, titles):
    dists = np.linalg.norm(grid_flat[:, None, :] - X[None, :, :], axis=2)
    nn_idx = np.argpartition(dists, k, axis=1)[:, :k]
    nn_labels = y[nn_idx]
    preds = (nn_labels.mean(axis=1) >= 0.5).astype(int).reshape(gx.shape)
    
    ax.contourf(gx, gy, preds, levels=[-0.5, 0.5, 1.5], colors=["#4c78a8", "#f58518"], alpha=0.35)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=25, edgecolor="k")
    ax.set_title(title, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Comparison with scikit-learn)\nWe compare against `sklearn.neighbors.KNeighborsClassifier`.",
            """from sklearn.neighbors import KNeighborsClassifier

sk_knn = KNeighborsClassifier(n_neighbors=15).fit(X, y)
print("scikit-learn k-NN (k=15) Accuracy:", sk_knn.score(X, y).round(4))"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why is k-NN described as a 'lazy learner'?</b></summary>
Because it does no explicit computation during training ($O(1)$ fit time); it simply stores the training instances. All computation is deferred until inference time, when distances to all training points must be computed ($O(n \\cdot d)$).
</details>

<details>
<summary><b>2. How does the curse of dimensionality impair k-NN in high dimensions?</b></summary>
In high-dimensional feature spaces, the volume of space grows exponentially with dimension. Data points become sparse, and the distance to the nearest neighbor approaches the distance to the average neighbor, making proximity metrics uninformative.
</details>"""
    }

    # 10: Support Vector Machines (SVM)
    specs[10] = {
        "eda": (
            "### Step 1: Linearly Separable Cluster Exploration\nWe inspect distinct Gaussian clusters to explore maximum-margin hyperplanes.",
            """X, y = data.blobs(n=120, k=2, std=1.1, spread=3.5, seed=0)

plt.figure(figsize=(8, 4.5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=["#4c78a8", "#e45756"], s=60, edgecolor="k")
plt.title("Linearly Separable Classes for SVM Margin Analysis", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Maximum Margin Hyperplane & Support Vectors\nWe plot the decision boundary along with the $+1$ and $-1$ margin hyperplanes and highlight the support vectors that define the margin.",
            """from sklearn.svm import SVC

clf = SVC(kernel="linear", C=1.0).fit(X, y)
w = clf.coef_[0]
b = clf.intercept_[0]

plt.figure(figsize=(9, 5))
x_plot = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
y_decision = (-w[0] * x_plot - b) / w[1]
y_margin_up = (-w[0] * x_plot - b + 1.0) / w[1]
y_margin_down = (-w[0] * x_plot - b - 1.0) / w[1]

plt.plot(x_plot, y_decision, "k-", lw=2, label="Decision Boundary (w*x + b = 0)")
plt.plot(x_plot, y_margin_up, "k--", lw=1.5, label="Upper Margin (+1)")
plt.plot(x_plot, y_margin_down, "k--", lw=1.5, label="Lower Margin (-1)")

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=180, facecolors="none", edgecolors="magenta", lw=2, label="Support Vectors")
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=["#4c78a8", "#e45756"], s=60, edgecolor="k")

plt.title("Support Vector Machine: Maximum Margin & Critical Vectors", fontsize=12, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Comparison with scikit-learn)\nWe evaluate linear vs. RBF kernel accuracy on non-linear data.",
            """clf_rbf = SVC(kernel="rbf", C=1.0, gamma="scale").fit(X, y)
print(f"Linear SVM Accuracy: {clf.score(X, y):.4f}")
print(f"RBF Kernel SVM Accuracy: {clf_rbf.score(X, y):.4f}")"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. What role do non-support vectors play in determining the SVM boundary?</b></summary>
None whatsoever. If all non-support training points are deleted or moved (as long as they remain outside the margin), the optimal separating hyperplane does not change at all. Only points lying on or inside the margin dictate the solution.
</details>

<details>
<summary><b>2. How does the kernel trick avoid computing high-dimensional coordinates?</b></summary>
The kernel function $K(x, z) = \\langle \\phi(x), \\phi(z) \\rangle$ computes the inner product directly in the input space without ever mapping $x$ into the high-dimensional feature space $\\phi(x)$.
</details>"""
    }

    # 11: Decision Trees (CART)
    specs[11] = {
        "eda": (
            "### Step 1: Non-Linear Decision Regions Inspection\nWe inspect a non-linear moons dataset to observe how recursive splitting segments feature space.",
            """X, y = data.moons(n=240, noise=0.2, seed=0)

plt.figure(figsize=(8, 4.5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="tab10", s=50, edgecolor="k")
plt.title("Moons Dataset for Decision Tree Partitioning", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Axis-Aligned Split Partitions\nWe train decision trees with depths 2, 4, and 8 to observe how recursive splitting segments feature space into rectangular regions.",
            """from sklearn.tree import DecisionTreeClassifier

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
depths = [2, 4, 8]

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))

for ax, d in zip(axes, depths):
    tree = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X, y)
    Z = tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="tab10")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", s=25, edgecolor="k")
    ax.set_title(f"Max Depth = {d} (Accuracy: {tree.score(X, y):.2f})", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Feature Importance Analysis)\nWe inspect Gini feature importances computed by `DecisionTreeClassifier`.",
            """tree = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)
print("Gini Feature Importances:", tree.feature_importances_.round(4))"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why are decision trees prone to high variance when unconstrained?</b></summary>
Because a tree with no depth or leaf constraints can continue splitting until every single leaf node is pure (100% training accuracy), creating fragmented rectangular partitions that overfit outliers.
</details>

<details>
<summary><b>2. Why do decision trees struggle with diagonal decision boundaries?</b></summary>
Because every split in a standard CART tree is axis-aligned ($x_j \\le t$). To approximate a diagonal line, the tree must create a staircase pattern requiring numerous splits.
</details>"""
    }

    # 12: Random Forest
    specs[12] = {
        "eda": (
            "### Step 1: Noisy Multi-Class Landscape Inspection\nWe inspect a complex classification dataset where individual decision trees tend to overfit.",
            """X, y = data.moons(n=300, noise=0.3, seed=0)

plt.figure(figsize=(8, 4.5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette=["#4c78a8", "#e45756"], s=50, edgecolor="k")
plt.title("Noisy Moons Dataset: Forest Aggregation", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Single Tree vs. Ensemble Averaging\nWe compare the erratic, jagged boundary of a single deep tree with the smooth, robust decision boundary of a 100-tree Random Forest.",
            """from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

single_tree = DecisionTreeClassifier(random_state=42).fit(X, y)
forest = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))

Z1 = single_tree.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax1.contourf(xx, yy, Z1, alpha=0.35, cmap="coolwarm")
ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=25, edgecolor="k")
ax1.set_title(f"Single Decision Tree: Jagged Boundary (Acc: {single_tree.score(X, y):.2f})", fontsize=11, fontweight="bold")

Z2 = forest.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax2.contourf(xx, yy, Z2, alpha=0.35, cmap="coolwarm")
ax2.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=25, edgecolor="k")
ax2.set_title(f"Random Forest (100 Trees): Smooth Boundary (Acc: {forest.score(X, y):.2f})", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Out-of-Bag Score Validation)\nWe verify Out-Of-Bag (OOB) score approximation without a separate validation set.",
            """rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42).fit(X, y)
print(f"Random Forest OOB Generalization Score: {rf_oob.oob_score_:.4f}")"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why does averaging multiple uncorrelated trees reduce variance without increasing bias?</b></summary>
If each tree has variance $\\sigma^2$ and the trees are uncorrelated, averaging $B$ trees reduces variance to $\\sigma^2 / B$. Because each tree is identically distributed, the expected value (and thus bias) remains unchanged.
</details>

<details>
<summary><b>2. How does feature subsampling (mtry) decorrelate trees?</b></summary>
If one feature is overwhelmingly predictive, every bootstrap tree would split on it first, making the trees highly correlated. By restricting each split to a random subset of $\\sqrt{d}$ features, other features get selected, decorrelating the ensemble.
</details>"""
    }

    # 23: K-Means Clustering
    specs[23] = {
        "eda": (
            "### Step 1: Unlabeled Cluster Distribution Visualization\nWe visualize unlabeled mall customer spending data exhibiting clear spatial clusters.",
            """X, _ = data.mall_customers(n=300, seed=0)

plt.figure(figsize=(8, 4.5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], color="#3b6fb6", s=50, alpha=0.8, edgecolor="w")
plt.title("Mall Customers: Annual Income vs. Spending Score", fontsize=12, fontweight="bold")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Voronoi Partition & Centroid Convergence\nWe fit K-Means ($k=5$) and plot the resulting Voronoi partitions along with the final converged centroid coordinates.",
            """from sklearn.cluster import KMeans

km = KMeans(n_clusters=5, init="k-means++", n_init=10, random_state=42).fit(X)
labels = km.labels_
centers = km.cluster_centers_

plt.figure(figsize=(9, 5))
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150), np.linspace(y_min, y_max, 150))
Z = km.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.25, cmap="Set2")
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette="Set2", s=50, edgecolor="k")
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=220, color="red", edgecolor="black", label="Centroids")
plt.title(f"K-Means Customer Segmentation (k=5, Inertia: {km.inertia_:.1f})", fontsize=12, fontweight="bold")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Elbow Method & Silhouette Score)\nWe calculate inertia across $k=2..8$ to inspect the elbow curve.",
            """from sklearn.metrics import silhouette_score

inertias = [KMeans(k, n_init=5, random_state=42).fit(X).inertia_ for k in range(2, 8)]
sil = silhouette_score(X, labels)
print("Inertias (k=2 to 7):", [round(i, 1) for i in inertias])
print(f"Silhouette Score (k=5): {sil:.4f}")"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why does K-Means struggle with elongated or non-convex clusters?</b></summary>
Because K-Means uses Euclidean distance from a central point, implicitly assuming spherical clusters of equal variance. Points are assigned based on perpendicular bisectors, producing convex Voronoi cells.
</details>

<details>
<summary><b>2. How does k-means++ initialization prevent poor local optima?</b></summary>
Standard random initialization can place multiple centroids in the same cluster. K-means++ chooses the first centroid at random, then selects subsequent centroids with probability proportional to the squared distance from the nearest existing centroid.
</details>"""
    }

    # 27: Principal Component Analysis (PCA)
    specs[27] = {
        "eda": (
            "### Step 1: Correlated Multidimensional Data Inspection\nWe inspect body measurement data with strong covariance across height and weight to observe how maximal variance projection captures the primary information axis.",
            """r = np.random.default_rng(0)
n = 150
height = r.normal(170, 9, n)
weight = 0.9 * (height - 170) + 70 + r.normal(0, 4, n)
X = np.c_[height, weight]

plt.figure(figsize=(7.5, 5))
sns.scatterplot(x=X[:, 0], y=X[:, 1], color="#3b6fb6", s=50, alpha=0.75, edgecolor="w")
plt.title("Correlated Height vs. Weight Data for PCA", fontsize=12, fontweight="bold")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.tight_layout()
plt.show()"""
        ),
        "diag": (
            "### Step 3: Principal Component Vectors & Projection\nWe plot the orthogonal principal component eigenvectors overlaid on the centered data distribution.",
            """from sklearn.decomposition import PCA

X_centered = X - X.mean(axis=0)
pca = PCA(n_components=2).fit(X_centered)

plt.figure(figsize=(8, 5))
plt.scatter(X_centered[:, 0], X_centered[:, 1], color="#3b6fb6", s=40, alpha=0.6, label="Centered Data")

for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    v = comp * 2.5 * np.sqrt(var)
    plt.arrow(0, 0, v[0], v[1], color="#e45756" if i==0 else "#54a24b", width=0.08, head_width=0.4,
              length_includes_head=True, label=f"PC{i+1} ({pca.explained_variance_ratio_[i]*100:.1f}% var)", zorder=5)

plt.title("Principal Components: Orthogonal Axes of Maximal Variance", fontsize=12, fontweight="bold")
plt.xlabel("Centered Height (cm)")
plt.ylabel("Centered Weight (kg)")
plt.legend()
plt.tight_layout()
plt.show()"""
        ),
        "bench": (
            "### Step 4: Industry Benchmark (Explained Variance Verification)\nWe verify eigenvalues and variance explained ratios.",
            """print("Explained Variance Ratios:", pca.explained_variance_ratio_.round(4))
print("Singular Values:", pca.singular_values_.round(4))"""
        ),
        "quiz": """## 🧠 Check Your Understanding

<details>
<summary><b>1. Why must data be centered (and scaled) before performing PCA?</b></summary>
If data is not centered at the origin, the first principal component will point from the origin toward the center of the data cloud rather than along the direction of maximal variance. If unscaled, high-magnitude features artificially dominate the principal axes.
</details>

<details>
<summary><b>2. What is the mathematical relationship between PCA and Singular Value Decomposition (SVD)?</b></summary>
For a centered matrix $X$, the right singular vectors $V$ from $X = U \\Sigma V^T$ are identical to the principal component eigenvectors of the sample covariance matrix $X^T X / (n - 1)$.
</details>"""
    }

    return specs
