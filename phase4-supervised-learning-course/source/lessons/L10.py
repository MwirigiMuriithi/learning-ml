from nbkit import Lesson


def make():
    nb = Lesson("10_Logistic_Regression", 10, "Logistic Regression",
                "From a straight line to a probability: the workhorse of classification",
                130, "🟡 Intermediate",
                setup_extra="""from sklearn.datasets import load_breast_cancer, load_iris, make_moons, make_blobs
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, confusion_matrix, classification_report""")
    nb.header(
        objectives=[
            "Explain why linear regression fails for yes/no targets and how the **sigmoid** fixes it.",
            "Move fluently between **probability, odds and log-odds**, and read coefficients as **odds ratios**.",
            "Derive the log-loss from maximum likelihood and its gradient $\\mathbf X^\\top(\\mathbf p-\\mathbf y)/n$.",
            "Implement logistic regression **from scratch** with gradient descent *and* Newton's method, and match scikit-learn.",
            "Use L2 / L1 regularisation (`C`), and see why separable data breaks the unregularised model.",
            "Draw linear decision boundaries, bend them with polynomial features, and extend to multiple classes (softmax vs one-vs-rest).",
            "Apply it end-to-end to a loan-approval dataset and communicate the result.",
        ],
        prereqs="Lessons 01–02, 09",
        hook=("A teacher wants to predict **pass or fail** from study hours. She fits a straight line through the 0/1 outcomes… and the line predicts "
              "**1.4** for a student who studied 12 hours and **−0.3** for one who studied none. A probability of 140 %? Negative 30 %?\n\n"
              "The fix is an elegant three-step trick: compute a linear score (as in regression), pass it through an S-shaped **squashing function** that maps any number into (0, 1), "
              "and read the output as a **probability**. That is **logistic regression**: despite its name, a *classification* algorithm, and quite possibly the most widely deployed model in the world "
              "(credit scoring, medical risk, click prediction, A/B-test analysis…)."),
        roadmap=["Why linear regression fails for classes", "Sigmoid, odds and log-odds", "Maximum likelihood → log loss → gradient",
                 "A hand-worked example", "From scratch: gradient descent & Newton", "Interpretation: odds ratios", "Regularisation & separable data",
                 "Decision boundaries (and bending them)", "Multi-class: softmax vs one-vs-rest", "Loan approval end-to-end",
                 "Cheat-sheet, worked examples, problems, quiz"])

    # ---------------------------------------------------------- 1 why
    nb.section("🚫", "1. Why not just use linear regression?",
               "Let's simulate pass/fail data where the *true* chance of passing climbs smoothly with study hours, then try both models.")
    nb.code(r'''rng = np.random.default_rng(3)
hours = rng.uniform(0, 10, 150)
p_true = 1 / (1 + np.exp(-1.1 * (hours - 5)))
passed = (rng.random(150) < p_true).astype(int)

lin = LinearRegression().fit(hours[:, None], passed)
logit = LogisticRegression().fit(hours[:, None], passed)
g = np.linspace(-2, 12, 300)[:, None]
plt.figure(figsize=(8, 4.2)); plt.scatter(hours, passed + np.random.default_rng(0).normal(0, 0.02, 150), alpha=0.5, label="students (0 = fail, 1 = pass)")
plt.plot(g, lin.predict(g), "r--", lw=2, label="linear regression"); plt.plot(g, logit.predict_proba(g)[:, 1], "g", lw=2.6, label="logistic regression")
plt.axhline(0, color="gray", lw=0.6); plt.axhline(1, color="gray", lw=0.6); plt.ylim(-0.5, 1.5); plt.xlabel("study hours"); plt.ylabel("P(pass)"); plt.legend(); plt.title("A line escapes [0, 1]; the S-curve never does"); plt.show()
print("Linear regression predicts for 12 hours:", lin.predict([[12]]).round(2), " and for 0 hours:", lin.predict([[0]]).round(2))''')
    nb.md("Linear regression also has a deeper problem: residuals of a 0/1 target can't be normally distributed, and one far-away point can tilt the line. We need a model whose output is **always a valid probability**.")

    # ---------------------------------------------------------- 2 math
    nb.section("📐", "2. Sigmoid, odds and log-odds",
               "**Step 1: a linear score:** $z=\\mathbf w\\cdot\\mathbf x+b$ (any real number). "
               "**Step 2: squash it** with the **sigmoid** (logistic) function:")
    nb.md(r"""$$\sigma(z)=\frac{1}{1+e^{-z}}\qquad\Longrightarrow\qquad P(y=1\mid\mathbf x)=\sigma(\mathbf w\cdot\mathbf x+b)$$

Properties worth memorising: $\sigma(0)=0.5$; $\sigma(z)\to1$ as $z\to+\infty$ and $\to0$ as $z\to-\infty$; **$\sigma(-z)=1-\sigma(z)$**; and the derivative is beautifully simple: $\sigma'(z)=\sigma(z)\,(1-\sigma(z))$.

### Three languages for the same thing
| Language | Formula | Range | Example (p = 0.8) |
|---|---|---|---|
| Probability | $p$ | 0 to 1 | 0.8 |
| **Odds** | $\dfrac{p}{1-p}$ | 0 to ∞ | 4 ("4 to 1 in favour") |
| **Log-odds (logit)** | $\ln\dfrac{p}{1-p}$ | −∞ to +∞ | 1.386 |

The sigmoid is exactly the inverse of the logit, so the model says: **the log-odds are a linear function of the features**:

$$\ln\frac{p}{1-p}=b+w_1x_1+\dots+w_px_p$$

**Interpretation:** raising $x_j$ by one unit adds $w_j$ to the log-odds, i.e. **multiplies the odds by $e^{w_j}$** (the *odds ratio*). Linear regression's "holding everything else constant" logic carries over: on the log-odds scale.""")
    nb.code(r'''z = np.linspace(-7, 7, 300); sig = 1 / (1 + np.exp(-z))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(z, sig, lw=2.5); ax[0].axhline(0.5, color="gray", ls=":"); ax[0].axvline(0, color="gray", ls=":"); ax[0].set_title("Sigmoid σ(z)"); ax[0].set_xlabel("z = w·x + b")
ax[1].plot(z, sig * (1 - sig), lw=2.5, color="#f97316"); ax[1].set_title("Its derivative σ(1−σ): steepest at z = 0"); ax[1].set_xlabel("z")
plt.tight_layout(); plt.show()''')

    nb.md(r"""### Where does the loss come from? Maximum likelihood

If the model says $p_i=P(y_i=1\mid\mathbf x_i)$, the probability it assigns to the **observed** label is $p_i^{y_i}(1-p_i)^{1-y_i}$ (just $p_i$ if $y_i=1$, just $1-p_i$ if $y_i=0$). Assuming independent examples, the **likelihood** of the whole dataset is the product over $i$. Maximise its logarithm, or equivalently *minimise the negative mean log-likelihood*: the **log loss** from Lesson 09:

$$J(\mathbf w,b)=-\frac1n\sum_{i=1}^n\Big[y_i\ln p_i+(1-y_i)\ln(1-p_i)\Big]$$

Two facts make this loss lovable: it is **convex** (one global minimum; no local-minimum worries), and its gradient has the same shape as in linear regression:

$$\boxed{\nabla_{\mathbf w}J=\frac1n\mathbf X^\top(\mathbf p-\mathbf y)\qquad\frac{\partial J}{\partial b}=\frac1n\sum_i(p_i-y_i)}$$

"Prediction minus truth, weighted by the features." (The sigmoid's derivative cancels the log's; algebra at its most generous.) Unlike linear regression there is **no closed-form solution**, so we iterate: gradient descent, or **Newton's method** which uses the curvature (Hessian) $\mathbf H=\frac1n\mathbf X^\top\mathrm{diag}\bigl(p_i(1-p_i)\bigr)\mathbf X$ to take much smarter steps: $\theta\leftarrow\theta-\mathbf H^{-1}\nabla J$ (a.k.a. *iteratively reweighted least squares*).""")

    # ---------------------------------------------------------- 3 hand
    nb.section("✍️", "3. A hand-worked example",
               "Suppose a fitted model is $z=-4+0.8\\times\\text{hours}$. What are the probabilities, odds and the 50 % study time?")
    nb.code(r'''b, w = -4.0, 0.8
tbl = pd.DataFrame({"hours": [2, 5, 8, 10]}); tbl["z (log-odds)"] = b + w * tbl["hours"]
tbl["P(pass)"] = 1 / (1 + np.exp(-tbl["z (log-odds)"])); tbl["odds"] = tbl["P(pass)"] / (1 - tbl["P(pass)"])
display(tbl.round(3))
print("odds ratio per extra hour = e^0.8 =", round(np.exp(0.8), 3), "  (each extra hour multiplies the odds of passing by about 2.2)")
print("50% threshold at hours = -b/w =", -b / w)''')
    nb.md("At 5 hours: $z=0$, probability exactly 0.5, odds 1:1. Every extra hour multiplies the odds by 2.23: odds 1 → 2.2 → 5.0 → 11.0. "
          "The **decision boundary** is where $z=0$, here a single point at $x=-b/w=5$ hours. With two features it's a *line*, with three a *plane*: always a flat boundary.")

    # ---------------------------------------------------------- 4 scratch
    nb.section("🔧", "4. Build it from scratch",
               "We implement the model with two solvers and the L2 penalty written in scikit-learn's convention, "
               "$\\;J=\\text{mean log-loss}+\\dfrac{1}{2Cn}\\lVert\\mathbf w\\rVert^2$ (so **small C = strong regularisation**, and the bias is never penalised).")
    nb.code(r'''def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))               # clip avoids overflow warnings

class LogisticRegressionScratch:
    """Binary logistic regression: solver='gd' (gradient descent) or 'newton' (IRLS). L2 penalty via C (np.inf = none)."""

    def __init__(self, C=1.0, solver="newton", lr=0.5, n_iter=2000, tol=1e-10):
        self.C, self.solver, self.lr, self.n_iter, self.tol = C, solver, lr, n_iter, tol

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float); n, p = X.shape
        Xb = np.c_[np.ones(n), X]                                      # column of ones = intercept
        lam = 0.0 if np.isinf(self.C) else 1.0 / (self.C * n)          # penalty strength on the mean-loss scale
        pen = lam * np.r_[0.0, np.ones(p)]                             # do NOT penalise the intercept
        theta = np.zeros(p + 1); self.loss_history_ = []
        for it in range(self.n_iter):
            prob = sigmoid(Xb @ theta)
            loss = -np.mean(y * np.log(prob + 1e-15) + (1 - y) * np.log(1 - prob + 1e-15)) + 0.5 * np.sum(pen * theta ** 2)
            self.loss_history_.append(loss)
            grad = Xb.T @ (prob - y) / n + pen * theta                  # <- the famous X'(p - y)/n  (+ penalty)
            if self.solver == "gd":
                step = self.lr * grad
            else:                                                       # Newton: use curvature
                H = (Xb * (prob * (1 - prob))[:, None]).T @ Xb / n + np.diag(pen) + 1e-12 * np.eye(p + 1)
                step = np.linalg.solve(H, grad)
            theta -= step
            if np.max(np.abs(step)) < self.tol: break
        self.intercept_, self.coef_ = theta[0], theta[1:]; self.n_iter_ = it + 1
        return self

    def decision_function(self, X): return self.intercept_ + np.asarray(X, float) @ self.coef_
    def predict_proba(self, X):     return sigmoid(self.decision_function(X))
    def predict(self, X, threshold=0.5): return (self.predict_proba(X) >= threshold).astype(int)
    def score(self, X, y): return np.mean(self.predict(X) == np.asarray(y))''')
    nb.code(r'''bc = load_breast_cancer(); Xb_all, yb_all = bc.data, 1 - bc.target                 # 1 = malignant
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb_all, yb_all, test_size=0.3, stratify=yb_all, random_state=RANDOM_STATE)
sc = StandardScaler().fit(Xb_tr); Ztr, Zte = sc.transform(Xb_tr), sc.transform(Xb_te)

newton = LogisticRegressionScratch(C=1.0, solver="newton").fit(Ztr, yb_tr)
gd = LogisticRegressionScratch(C=1.0, solver="gd", lr=0.5, n_iter=5000).fit(Ztr, yb_tr)
ref = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(Ztr, yb_tr)

print(f"Newton : converged in {newton.n_iter_:>4} iterations | max |coef diff| vs sklearn = {np.abs(newton.coef_ - ref.coef_[0]).max():.1e}")
print(f"GD     : ran         {gd.n_iter_:>4} iterations | max |coef diff| vs sklearn = {np.abs(gd.coef_ - ref.coef_[0]).max():.1e}")
print(f"test accuracy; Newton {newton.score(Zte, yb_te):.3f} | GD {gd.score(Zte, yb_te):.3f} | sklearn {ref.score(Zte, yb_te):.3f}")
print(f"test log loss: scratch {log_loss(yb_te, newton.predict_proba(Zte)):.4f} | sklearn {log_loss(yb_te, ref.predict_proba(Zte)[:, 1]):.4f}")''')
    nb.code(r'''fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].plot(newton.loss_history_, "o-", label="Newton"); ax[0].set_title("Newton: converges in a handful of steps"); ax[0].set_xlabel("iteration"); ax[0].set_ylabel("regularised log loss"); ax[0].set_xlim(-0.5, 12)
ax[1].plot(gd.loss_history_[:600], label="gradient descent"); ax[1].axhline(newton.loss_history_[-1], color="r", ls="--", label="optimum (Newton)"); ax[1].set_title("Gradient descent: steady but slower"); ax[1].set_xlabel("iteration"); ax[1].legend()
plt.tight_layout(); plt.show()''')
    nb.callout("key", "Our 35-line class reproduces scikit-learn's coefficients to ~10⁻⁶ (Newton) and the same accuracy. "
                      "Newton needs only ~10 iterations because it exploits curvature; each iteration costs $O(np^2+p^3)$ though, so for huge *p* people use gradient-based solvers (`lbfgs`, `saga`), which is what scikit-learn does by default.")

    # ---------------------------------------------------------- 5 interpretation + applied
    nb.section("🏦", "5. Loan approval end-to-end, and reading the coefficients",
               "Now a realistic problem: **will a loan application be approved?** (3 000 synthetic applications, ~60 % approved.) "
               "We deliberately keep out two columns: `loan_id` (a meaningless identifier) and `interest_rate_offered`: *think about why a column like that might be dangerous*; Project 3 will reveal all.")
    nb.code(r'''loans = pd.read_csv(data_path("loan_approval.csv"))
loans["emi"] = loans["loan_amount_kes"] / loans["loan_term_months"] * 1.15                       # rough monthly instalment
loans["dti"] = (loans["monthly_debt_kes"] + loans["emi"]) / loans["monthly_income_kes"]          # debt-to-income ratio
loans["lti"] = loans["loan_amount_kes"] / (loans["monthly_income_kes"] * 12)                     # loan-to-annual-income
loans["log_income"] = np.log(loans["monthly_income_kes"]); loans["log_savings"] = np.log1p(loans["savings_balance_kes"])

num = ["credit_score", "credit_history_years", "dti", "lti", "log_income", "log_savings", "has_collateral", "age", "dependents", "existing_loans"]
cat = ["employment_type", "property_area", "education"]
X, y = loans[num + cat], loans["approved"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                          ("cat", OneHotEncoder(drop="first"), cat)])
model = make_pipeline(prep, LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
proba = model.predict_proba(Xte)[:, 1]
print(f"accuracy = {accuracy_score(yte, proba >= .5):.3f}   ROC-AUC = {roc_auc_score(yte, proba):.3f}   baseline (always 'approve') = {yte.mean():.3f}")
display(pd.DataFrame(confusion_matrix(yte, proba >= .5), index=["actually rejected", "actually approved"], columns=["pred rejected", "pred approved"]))''')
    nb.md("**Feature engineering pays.** The raw columns (income, loan amount, term, debt) interact multiplicatively; a *linear* log-odds model can't build ratios by itself, but we can hand it the ratios. Compare:")
    nb.code(r'''raw_num = ["credit_score", "credit_history_years", "monthly_income_kes", "loan_amount_kes", "loan_term_months", "monthly_debt_kes", "has_collateral", "age", "dependents", "existing_loans", "savings_balance_kes"]
prep_raw = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), raw_num), ("cat", OneHotEncoder(drop="first"), cat)])
m_raw = make_pipeline(prep_raw, LogisticRegression(max_iter=2000)).fit(loans.loc[Xtr.index, raw_num + cat], ytr)
print(f"AUC with raw columns      : {roc_auc_score(yte, m_raw.predict_proba(loans.loc[Xte.index, raw_num + cat])[:, 1]):.3f}")
print(f"AUC with dti/lti features : {roc_auc_score(yte, proba):.3f}")''')
    nb.md("Now interpret. Because numeric features were standardised, each coefficient is the change in log-odds per **one standard deviation** of that feature; $e^{w}$ is the odds multiplier.")
    nb.code(r'''names = model[0].get_feature_names_out(); coefs = model[-1].coef_[0]
tbl = pd.DataFrame({"coef (log-odds)": coefs, "odds ratio e^coef": np.exp(coefs)}, index=[n.split("__")[1] for n in names]).sort_values("coef (log-odds)")
display(tbl.round(3))
tbl["coef (log-odds)"].plot.barh(color=["#dc2626" if c < 0 else "#16a34a" for c in tbl["coef (log-odds)"]], figsize=(7, 4.5)); plt.axvline(0, color="k", lw=0.8)
plt.title("What pushes a loan toward approval (green) or rejection (red)"); plt.xlabel("log-odds per 1 SD"); plt.show()''')
    nb.md("**Reading it:** a higher debt-to-income ratio is the strongest *negative* driver (odds fall by ~94 % per +1 SD, i.e. ×0.06); a higher credit score is the strongest *positive* one (odds ×4.3 per +1 SD); "
          "collateral and a salaried job help; very little else matters. "
          "A loan officer can *read* this, which is exactly why regulated industries (banking, insurance, healthcare) keep choosing logistic regression.")
    nb.callout("warn", "Coefficients are only comparable across features because we standardised. And, as ever, they describe **association in the data**, not causal effects. "
                       "Also be careful with sensitive attributes (gender, marital status…): we left them out here on purpose; fairness is part of Project 3.")

    # ---------------------------------------------------------- 6 regularisation
    nb.section("🧯", "6. Regularisation and the problem with separable data",
               "### 6a. When classes are perfectly separable, the unregularised model has no answer\n"
               "If one line splits the classes perfectly, the likelihood keeps improving as the weights grow (a steeper sigmoid is *more* confident and *always* right), so the 'optimal' weights are **infinite**. Watch:")
    nb.code(r'''xs = np.array([[1.], [2.], [3.], [4.], [5.], [6.]]); ys = np.array([0, 0, 0, 1, 1, 1])
rows = []
for n_it in [10, 100, 1000, 10000]:
    m = LogisticRegressionScratch(C=np.inf, solver="gd", lr=0.5, n_iter=n_it, tol=0).fit(xs - 3.5, ys)
    rows.append({"GD iterations": n_it, "weight": m.coef_[0], "final loss": m.loss_history_[-1]})
display(pd.DataFrame(rows).round(4))
print("With an L2 penalty (C=1) the answer is finite:", LogisticRegressionScratch(C=1.0, solver="newton").fit(xs - 3.5, ys).coef_.round(3))''')
    nb.md("The weight keeps growing (slowly, logarithmically) while the loss creeps toward 0. This is why scikit-learn applies L2 regularisation **by default** (`C=1.0`): it's a safety net, not a luxury.")
    nb.md("### 6b. The `C` dial: L2 vs L1\n**Small C** → strong penalty → small weights, smoother/simpler model (high bias). **Large C** → weak penalty → flexible (high variance). "
          "**L2** shrinks all weights; **L1** drives many exactly to zero (feature selection).")
    nb.code(r'''Cs = np.logspace(-4, 2, 13); rows = []
for C in Cs:
    l2 = LogisticRegression(C=C, max_iter=5000).fit(Ztr, yb_tr)
    l1 = LogisticRegression(l1_ratio=1, solver="saga", C=C, max_iter=20000).fit(Ztr, yb_tr)       # L1 penalty (scikit-learn >= 1.8)
    rows.append({"C": C, "L2 train acc": l2.score(Ztr, yb_tr), "L2 test acc": l2.score(Zte, yb_te), "L2 ‖w‖": np.linalg.norm(l2.coef_), "L1 non-zero weights": int((l1.coef_ != 0).sum()), "L1 test acc": l1.score(Zte, yb_te)})
res = pd.DataFrame(rows).set_index("C"); display(res.round(3))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].semilogx(res.index, res["L2 train acc"], "o-", label="train"); ax[0].semilogx(res.index, res["L2 test acc"], "s-", label="test"); ax[0].set_xlabel("C"); ax[0].set_title("L2: accuracy vs C"); ax[0].legend()
ax[1].semilogx(res.index, res["L1 non-zero weights"], "o-", color="#7c3aed"); ax[1].set_xlabel("C"); ax[1].set_title("L1: number of features kept"); plt.tight_layout(); plt.show()''')

    # ---------------------------------------------------------- 7 boundaries
    nb.section("🗺️", "7. Decision boundaries (and bending them)",
               "With two features the boundary $w_1x_1+w_2x_2+b=0$ is a straight line. Perfect for blobs; hopeless for interleaved moons: *unless* we feed it polynomial features (just like Lesson 03).")
    nb.code(r'''Xm, ym = make_moons(300, noise=0.25, random_state=1); Xs, ys_ = make_blobs(300, centers=[[-1.5, -1], [1.5, 1.2]], cluster_std=1.1, random_state=2)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
m1 = LogisticRegression().fit(Xs, ys_); plot_decision_boundary(m1, Xs, ys_, ax=axes[0], title=f"Blobs: linear boundary works\n(acc {m1.score(Xs, ys_):.2f})")
m2 = LogisticRegression().fit(Xm, ym); plot_decision_boundary(m2, Xm, ym, ax=axes[1], title=f"Moons: a line can't separate them\n(acc {m2.score(Xm, ym):.2f})")
m3 = make_pipeline(PolynomialFeatures(4), StandardScaler(), LogisticRegression(C=5, max_iter=5000)).fit(Xm, ym); plot_decision_boundary(m3, Xm, ym, ax=axes[2], title=f"Moons + degree-4 polynomial features\n(acc {m3.score(Xm, ym):.2f})")
plt.tight_layout(); plt.show()''')
    nb.md("The model is *still* logistic regression (linear in a richer feature space) which then looks curved in the original space. The same trick as polynomial regression, now for classification.")

    # ---------------------------------------------------------- 8 multiclass
    nb.section("🌸", "8. More than two classes: softmax vs one-vs-rest",
               "Two strategies for $K$ classes:\n\n"
               "- **One-vs-Rest (OvR):** train $K$ binary models ('class k vs everything else'), pick the class with the highest score.\n"
               "- **Multinomial / softmax:** one model with a weight vector per class; probabilities are\n"
               r"$$P(y=k\mid\mathbf x)=\frac{e^{z_k}}{\sum_{j=1}^Ke^{z_j}},\qquad z_k=\mathbf w_k\cdot\mathbf x+b_k$$" "\n"
               "(softmax generalises the sigmoid: for K = 2 it reduces to it.) scikit-learn uses softmax by default with `lbfgs`.")
    nb.code(r'''iris = load_iris(); Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(iris.data, iris.target, test_size=0.3, stratify=iris.target, random_state=RANDOM_STATE)
soft = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xi_tr, yi_tr)
ovr = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(max_iter=1000))).fit(Xi_tr, yi_tr)
print(f"softmax test acc = {soft.score(Xi_te, yi_te):.3f}   |   one-vs-rest test acc = {ovr.score(Xi_te, yi_te):.3f}")

# softmax by hand, from the fitted weights
lr_ = soft[-1]; z = lr_.intercept_ + soft[0].transform(Xi_te[:1]) @ lr_.coef_.T
manual = np.exp(z) / np.exp(z).sum()
print("manual softmax probabilities:", manual.round(4)[0], "\nsklearn predict_proba       :", soft.predict_proba(Xi_te[:1]).round(4)[0], "->", iris.target_names[soft.predict(Xi_te[:1])[0]])''')

    # ---------------------------------------------------------- 9 profile
    nb.section("📋", "9. Cheat-sheet: the full profile of Logistic Regression")
    nb.table([
        ["**Underlying concept**", "A linear score squashed by the sigmoid into a probability; log-odds are linear in the features."],
        ["**How it works**", "Fit weights by minimising the (regularised) log loss with Newton/IRLS or gradient-based solvers; predict P(y=1), compare with a threshold."],
        ["**Mathematical foundation**", "Maximum likelihood; sigmoid/logit; convex log-loss; gradient Xᵀ(p−y)/n; Hessian XᵀWX; L1/L2 priors; softmax for K classes."],
        ["**Important parameters**", "`C` (inverse regularisation), L1/L2 via `l1_ratio`/penalty, `solver` (lbfgs/liblinear/saga/newton-cg), `class_weight`, `max_iter`, the decision threshold (not a model parameter!). Scale features."],
        ["**Advantages**", "Fast, stable, convex; probabilistic outputs; interpretable (odds ratios); works well with limited data; regularisation built in; easy to deploy and monitor."],
        ["**Disadvantages**", "Linear decision boundary; needs feature engineering for interactions/non-linearity; sensitive to outliers & multicollinearity; probabilities can be off with severe imbalance."],
        ["**Complexity**", "Training ≈ O(n·p) per gradient step (×iterations) or O(n·p² + p³) per Newton step; prediction O(p); memory O(p)."],
        ["**Use cases**", "Credit scoring, medical risk, churn, click/conversion prediction, spam baselines, any task where you must explain *why* and need calibrated probabilities."],
        ["**Limitations**", "Cannot model XOR-like or curved boundaries on raw features; perfect separation → needs regularisation; assumes additive effects on the log-odds scale."],
    ], ["Question", "Answer"])

    # ---------------------------------------------------------- 10 worked
    nb.section("🧪", "10. More worked examples")
    nb.md(r"""### Worked Example 1: From log-odds to probability

A spam model has intercept $-3$, and weights $+1.2$ for 'contains the word FREE' and $+0.8$ for 'has ≥ 3 exclamation marks'. An email has both.

$z=-3+1.2+0.8=-1.0$ → $p=\sigma(-1)=\dfrac{1}{1+e^{1}}=0.269$. Odds $=0.269/0.731=0.368=e^{-1}$. Not yet spam at threshold 0.5. If we also add a 'sender unknown' feature (+1.5), $z=0.5$ and $p=0.622$.""")
    nb.code(r'''for z_ in [-1.0, 0.5]: print(f"z={z_:>4} -> p = {1 / (1 + np.exp(-z_)):.3f}, odds = {np.exp(z_):.3f}")''')
    nb.md(r"""### Worked Example 2: One gradient step by hand

One example $x=2$, $y=1$; start with $w=0,\ b=0$; learning rate 0.5. $p=\sigma(0)=0.5$; error $p-y=-0.5$. Gradients: $\partial J/\partial w=x(p-y)=2\times(-0.5)=-1$, $\partial J/\partial b=-0.5$. Update: $w=0-0.5(-1)=0.5$, $b=0-0.5(-0.5)=0.25$. New score at $x=2$: $z=0.5\cdot2+0.25=1.25$, $p=\sigma(1.25)=0.777$: the model became more confident in the right answer.""")
    nb.code(r'''w_, b_ = 0.0, 0.0; x_, y_ = 2.0, 1.0; p_ = 1 / (1 + np.exp(0))
gw, gb = x_ * (p_ - y_), (p_ - y_); w_, b_ = w_ - 0.5 * gw, b_ - 0.5 * gb
print("w, b =", w_, b_, "| new p =", round(1 / (1 + np.exp(-(w_ * x_ + b_))), 3))''')

    # ---------------------------------------------------------- problems
    nb.problem(
        "Sigmoid, odds and log-odds",
        "Write `sigmoid(z)`, `odds(p)` and `logit(p)` (the inverse of the sigmoid). Compute: `p1 = sigmoid(1.386)` (≈ 0.8), `o1 = odds(0.8)`, `l1 = logit(0.8)`, "
        "and verify the round trip with `roundtrip = sigmoid(logit(0.3))` (should return 0.3).",
        ["`sigmoid(z) = 1 / (1 + np.exp(-z))`.",
         "`odds(p) = p / (1 - p)` and `logit(p) = np.log(odds(p))`.",
         "sigmoid∘logit is the identity."],
        'def sigmoid(z):\n    pass\n\ndef odds(p):\n    pass\n\ndef logit(p):\n    pass\n\np1 = o1 = l1 = roundtrip = None   # your code\n',
        'def sigmoid(z):\n    return 1 / (1 + np.exp(-z))\n\ndef odds(p):\n    return p / (1 - p)\n\ndef logit(p):\n    return np.log(odds(p))\n\np1 = sigmoid(1.386); o1 = odds(0.8); l1 = logit(0.8); roundtrip = sigmoid(logit(0.3))\nprint(p1, o1, l1, roundtrip)',
        'check("sigmoid(1.386) ≈ 0.8", lambda: abs(p1 - 0.8) < 1e-3)\ncheck("odds(0.8) = 4", lambda: abs(o1 - 4) < 1e-9)\ncheck("logit(0.8) ≈ 1.386", lambda: abs(l1 - 1.3863) < 1e-3)\ncheck("round trip returns 0.3", lambda: abs(roundtrip - 0.3) < 1e-9)',
        explain="A probability of 0.8 means odds of 4:1 and log-odds of ln 4 ≈ 1.386. The sigmoid and logit are inverse functions, translating between the unbounded linear world and the bounded probability world.")

    nb.problem(
        "Log loss and its gradient, by hand",
        "Data: `X = [[1.0], [2.0], [3.0]]`, `y = [0, 0, 1]`, current parameters `w = 0.5`, `b = -1.0`.\n\n"
        "Compute `p` (the three predicted probabilities), `loss` (mean log loss), and the gradients `grad_w`, `grad_b` using the formulas $\\frac1n X^\\top(p-y)$ and $\\frac1n\\sum(p-y)$. "
        "Then verify `grad_w` numerically with a finite difference `(loss(w+h) − loss(w−h)) / 2h` for `h = 1e-6` and store it in `grad_w_numeric`.",
        ["`z = w * X[:, 0] + b`; `p = 1 / (1 + np.exp(-z))`.",
         "`loss = -np.mean(y*np.log(p) + (1-y)*np.log(1-p))`.",
         "`grad_w = np.mean(X[:, 0] * (p - y))`, `grad_b = np.mean(p - y)`. Write a small helper `L(w, b)` for the numeric derivative."],
        'X = np.array([[1.0], [2.0], [3.0]]); y = np.array([0, 0, 1]); w, b = 0.5, -1.0\np = loss = grad_w = grad_b = grad_w_numeric = None   # your code\n',
        'X = np.array([[1.0], [2.0], [3.0]]); y = np.array([0, 0, 1]); w, b = 0.5, -1.0\ndef L(w_, b_):\n    p_ = 1 / (1 + np.exp(-(w_ * X[:, 0] + b_)))\n    return -np.mean(y * np.log(p_) + (1 - y) * np.log(1 - p_))\n\n'
        'p = 1 / (1 + np.exp(-(w * X[:, 0] + b))); loss = L(w, b)\ngrad_w = np.mean(X[:, 0] * (p - y)); grad_b = np.mean(p - y)\nh = 1e-6; grad_w_numeric = (L(w + h, b) - L(w - h, b)) / (2 * h)\nprint(p.round(4), round(loss, 4), round(grad_w, 4), round(grad_b, 4), round(grad_w_numeric, 4))',
        'check("probabilities ≈ [0.378, 0.5, 0.622]", lambda: np.allclose(p, [0.3775, 0.5, 0.6225], atol=1e-3))\ncheck("loss ≈ 0.5471", lambda: abs(loss - 0.5471) < 1e-3)\ncheck("analytic gradient matches the numeric one", lambda: abs(grad_w - grad_w_numeric) < 1e-6)\ncheck("grad_b ≈ 0.1667", lambda: abs(grad_b - 0.16667) < 1e-3)',
        explain="The finite-difference check is the standard way to verify any gradient you derive by hand: the formula Xᵀ(p−y)/n really is the slope of the log loss.",
        level="⭐⭐")

    nb.problem(
        "Implement predict_proba from a fitted model",
        "Fit `lr_cancer = LogisticRegression(C=1.0, max_iter=10000).fit(Ztr, yb_tr)`. Using **only** `lr_cancer.coef_` and `lr_cancer.intercept_` (no `predict_proba`), compute "
        "`my_proba` for the standardised test set `Zte`, and store `max_gap` = max absolute difference vs `lr_cancer.predict_proba(Zte)[:, 1]`. "
        "Then find `boundary_dist` for the first test sample: its signed distance from the decision boundary, `z / ‖w‖`.",
        ["`z = Zte @ coef_[0] + intercept_[0]`; probability = sigmoid(z).",
         "The signed distance of a point to the hyperplane w·x + b = 0 is `z / np.linalg.norm(w)`.",
         "`max_gap` should be around 1e-15."],
        'lr_cancer = LogisticRegression(C=1.0, max_iter=10000).fit(Ztr, yb_tr)\nmy_proba = max_gap = boundary_dist = None   # your code\n',
        'lr_cancer = LogisticRegression(C=1.0, max_iter=10000).fit(Ztr, yb_tr)\nz_ = Zte @ lr_cancer.coef_[0] + lr_cancer.intercept_[0]\nmy_proba = 1 / (1 + np.exp(-z_))\nmax_gap = np.abs(my_proba - lr_cancer.predict_proba(Zte)[:, 1]).max()\nboundary_dist = z_[0] / np.linalg.norm(lr_cancer.coef_[0])\nprint(max_gap, boundary_dist)',
        'check("manual probabilities equal predict_proba", lambda: max_gap < 1e-10)\ncheck("distance has the same sign as the decision", lambda: np.sign(boundary_dist) == (1 if lr_cancer.predict(Zte[:1])[0] == 1 else -1))',
        explain="A trained logistic regression is just a weight vector and an intercept: ŷ = σ(w·x + b). Deployment can be a one-line formula in any language.")

    nb.problem(
        "Read the loan model",
        "Using `tbl` (coefficients and odds ratios of the loan model): set `strongest_positive` and `strongest_negative` to the *names* of the features with the largest and smallest coefficients, "
        "`or_credit` = the odds ratio of `credit_score`, and `pct_change_dti` = the percentage change in the odds of approval for +1 SD in `dti` (negative number). "
        "Finally `prob_after` = the probability after a customer whose current approval probability is 0.50 improves their credit score by +1 SD (odds multiply by `or_credit`).",
        ["`tbl['coef (log-odds)'].idxmax()` / `.idxmin()`.",
         "% change = `(odds_ratio - 1) * 100`.",
         "New odds = old odds × OR with old odds = 1 (p = 0.5); then p = odds / (1 + odds)."],
        'strongest_positive = strongest_negative = or_credit = pct_change_dti = prob_after = None   # your code\n',
        'strongest_positive = tbl["coef (log-odds)"].idxmax(); strongest_negative = tbl["coef (log-odds)"].idxmin()\nor_credit = tbl.loc["credit_score", "odds ratio e^coef"]\npct_change_dti = (tbl.loc["dti", "odds ratio e^coef"] - 1) * 100\nnew_odds = 1.0 * or_credit; prob_after = new_odds / (1 + new_odds)\nprint(strongest_positive, strongest_negative, round(or_credit, 2), round(pct_change_dti, 1), round(prob_after, 3))',
        'check("credit_score is the strongest positive driver", lambda: strongest_positive == "credit_score")\ncheck("dti is the strongest negative driver", lambda: strongest_negative == "dti")\ncheck("odds ratio of credit score > 3", lambda: or_credit > 3)\ncheck("higher dti cuts the odds by more than 90%", lambda: pct_change_dti < -90)\ncheck("0.5 -> about 0.8 after a +1 SD credit score", lambda: 0.75 < prob_after < 0.9)',
        explain="Odds ×4.3 sounds dramatic, but remember probabilities are bounded: a customer at 50 % moves to about 81 %, not 215 %. Converting back to probabilities is how you communicate results to non-statisticians.",
        level="⭐⭐")

    nb.problem(
        "Pick C by cross-validation",
        "Using `Ztr, yb_tr` (standardised cancer training data), evaluate `LogisticRegression(C=C, max_iter=5000)` for `C ∈ np.logspace(-3, 2, 11)` with **stratified 5-fold CV** (`StratifiedKFold(5, shuffle=True, random_state=0)`, scoring `'roc_auc'`) → dict `cv_auc` (C → mean AUC). "
        "Store `best_C` (the C with the highest AUC; take the **smallest** C in case of ties within 0.0005; prefer more regularisation) and the test AUC `test_auc` of that model.",
        ["`cross_val_score(LogisticRegression(C=C, max_iter=5000), Ztr, yb_tr, cv=skf, scoring='roc_auc').mean()`.",
         "`top = max(cv_auc.values())`; `best_C = min(c for c, v in cv_auc.items() if v >= top - 0.0005)`.",
         "Fit on all of `Ztr` and evaluate with `roc_auc_score(yb_te, model.predict_proba(Zte)[:, 1])`."],
        'cv_auc = {}\nbest_C = test_auc = None   # your code\n',
        'skf = StratifiedKFold(5, shuffle=True, random_state=0)\ncv_auc = {C: cross_val_score(LogisticRegression(C=C, max_iter=5000), Ztr, yb_tr, cv=skf, scoring="roc_auc").mean() for C in np.logspace(-3, 2, 11)}\n'
        'top = max(cv_auc.values()); best_C = min(c for c, v in cv_auc.items() if v >= top - 0.0005)\ntest_auc = roc_auc_score(yb_te, LogisticRegression(C=best_C, max_iter=5000).fit(Ztr, yb_tr).predict_proba(Zte)[:, 1])\nprint(round(best_C, 4), round(top, 4), round(test_auc, 4))',
        'check("11 values of C evaluated", lambda: len(cv_auc) == 11)\ncheck("very strong regularisation (C=0.001) is worse than the best", lambda: cv_auc[min(cv_auc)] < max(cv_auc.values()))\ncheck("test AUC > 0.98", lambda: test_auc > 0.98)',
        explain="On this easy dataset moderate regularisation is best: C too small underfits, C too large starts to chase noise. The 'smallest C within tolerance' rule is a mini one-standard-error rule: prefer the simpler model when scores tie.",
        level="⭐⭐")

    nb.problem(
        "Make a line bend",
        "On the moons data `Xm, ym`, compare the test accuracy of plain `LogisticRegression()` and `make_pipeline(PolynomialFeatures(d), StandardScaler(), LogisticRegression(C=5, max_iter=5000))` for `d = 1…6`. "
        "Use `train_test_split(Xm, ym, test_size=0.3, random_state=0)`. Store accuracies in `acc_by_degree` (dict d → test acc) and `best_d`.",
        ["Split once; fit each pipeline on the training part and score on the test part.",
         "Degree 1 is just a straight line (≈ 0.87).",
         "`best_d = max(acc_by_degree, key=acc_by_degree.get)`: ties go to the first (smallest) degree."],
        'acc_by_degree = {}\nbest_d = None   # your code\n',
        'Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, random_state=0)\n'
        'acc_by_degree = {d: make_pipeline(PolynomialFeatures(d), StandardScaler(), LogisticRegression(C=5, max_iter=5000)).fit(Xm_tr, ym_tr).score(Xm_te, ym_te) for d in range(1, 7)}\n'
        'best_d = max(acc_by_degree, key=acc_by_degree.get); print({k: round(v, 3) for k, v in acc_by_degree.items()}, best_d)',
        'check("six degrees evaluated", lambda: len(acc_by_degree) == 6)\ncheck("a curved boundary beats the straight line", lambda: acc_by_degree[best_d] > acc_by_degree[1] + 0.02)\ncheck("best degree is above 1", lambda: best_d > 1)',
        explain="Feeding squares, cubes and cross-products lets the *linear* log-odds model carve a curved boundary: the same feature-expansion idea as polynomial regression, and the doorway to kernel SVMs.",
        level="⭐⭐")

    nb.quiz([
        ("A logistic regression coefficient is 0.693. What is the odds ratio and what does it mean?",
         "e^0.693 = 2. Each +1 unit of that feature **doubles** the odds of the positive class (holding others constant)."),
        ("Why can't we minimise squared error for logistic regression with the same closed-form trick as linear regression?",
         "The model is non-linear in the weights (through the sigmoid), so the normal equations have no closed-form solution; we optimise the convex log loss iteratively."),
        ("Your classes are perfectly separable and you set `C=np.inf`. What happens to the weights?",
         "They grow without bound: a steeper sigmoid always fits better. Regularisation (finite C) keeps the solution finite."),
        ("Is the decision boundary of logistic regression linear? Can it be made non-linear?",
         "It is linear in the features given (w·x + b = 0). Add polynomial/interaction features (or use a kernel method) to get curved boundaries in the original space."),
        ("Why must we standardise features before comparing coefficients, and before using L1/L2 penalties?",
         "Penalties and coefficient magnitudes depend on feature scale: a feature measured in large units gets a tiny coefficient and is barely penalised, so scales must be comparable."),
    ])

    nb.footer(["Logistic regression: log-odds are linear in the features; the sigmoid turns them into probabilities.",
               "Trained by maximum likelihood = minimising the convex log loss; gradient = Xᵀ(p−y)/n; Newton/IRLS converges in a few steps.",
               "exp(coefficient) is an odds ratio: the key to interpretation (standardise to compare features).",
               "L2/L1 regularisation (C) is essential, especially for separable data; polynomial features bend the boundary.",
               "Softmax extends it to many classes; the threshold, not the model, encodes your business costs."],
              "Decision Tree Classification", "11_Decision_Tree_Classification.ipynb")
    return nb
