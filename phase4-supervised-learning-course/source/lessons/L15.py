from nbkit import Lesson


def make():
    nb = Lesson("15_Support_Vector_Machines", 15, "Support Vector Machines (SVM)",
                "Find the widest street between the classes: then bend space with kernels",
                140, "🟠 Intermediate → Advanced",
                setup_extra="""import time
from sklearn.datasets import load_breast_cancer, load_digits, make_moons, make_circles, make_blobs, make_classification
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, average_precision_score""")
    nb.header(
        objectives=[
            "Explain the **maximum-margin** idea and what a **support vector** is.",
            "Write the hard-margin, soft-margin and hinge-loss formulations and link them to `C`.",
            "Compute margins and distances to the hyperplane by hand ($2/\\lVert\\mathbf w\\rVert$, $|\\mathbf w\\!\\cdot\\!\\mathbf x+b|/\\lVert\\mathbf w\\rVert$).",
            "Implement a linear SVM **from scratch** (hinge-loss sub-gradient descent) and **rebuild an RBF-SVM's decision function from its support vectors**.",
            "Use the **kernel trick** (linear, polynomial, RBF) to separate data no line can.",
            "Tune `C` and `gamma` on a log-scale grid; know why scaling is mandatory.",
            "Handle imbalance (`class_weight`), obtain probabilities (Platt scaling) and choose between `SVC`, `LinearSVC` and `SGDClassifier`.",
        ],
        prereqs="Lessons 06 (SVR), 10 (logistic regression), 09",
        hook=("You must build a road between two villages that don't get along. Many routes separate them: some run uncomfortably close to one village's fence. "
              "A wise engineer picks the route that leaves the **widest possible buffer** on both sides: any mistake in surveying, any new house built slightly off, and the road still separates the two fairly.\n\n"
              "That's the **Support Vector Machine**: of all the lines (hyperplanes) that separate the classes, choose the one with the **maximum margin**. "
              "Only the few houses right at the edge of the buffer: the **support vectors**: matter; the rest of the villagers can move around freely. "
              "Add the *kernel trick* and the straight road can bend into any shape. For two decades (1995–2015) SVMs were the state of the art in classification."),
        roadmap=["The maximum-margin idea", "The math: hard margin, soft margin, hinge loss, dual form", "A hand-worked margin", "Seeing margins & support vectors",
                 "From scratch: linear SVM by sub-gradient descent", "The kernel trick (and rebuilding a prediction by hand)", "C and gamma", "Scaling, tuning with a log-grid",
                 "Probabilities, imbalance, speed & the SVM family", "Cheat-sheet, worked examples, problems, quiz"])

    # ---------------------------------------------------------- 1
    nb.section("🛣️", "1. The maximum-margin idea",
               "Draw a line between two separable classes: there are infinitely many that work. Which is best? The one **farthest from the nearest points of both classes**: it tolerates the most noise. "
               "The distance from the boundary to the closest point is the **margin**. Let's see three valid separating lines, and the max-margin one:")
    nb.code(r'''Xb, yb = make_blobs(60, centers=[[-2, -2], [2, 2]], cluster_std=0.9, random_state=7)
svm = SVC(kernel="linear", C=1e6).fit(Xb, yb)                                  # huge C ~ hard margin
w, b = svm.coef_[0], svm.intercept_[0]
xs = np.linspace(-5, 5, 100)
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(*Xb[yb == 0].T, s=35, label="class 0"); ax.scatter(*Xb[yb == 1].T, s=35, label="class 1")
for slope, icpt in [(-0.6, 0.6), (-1.7, 0.2), (-0.35, -0.2)]:
    ax.plot(xs, slope * xs + icpt, color="gray", lw=1, ls=":")
ax.plot(xs, -(w[0] * xs + b) / w[1], "k", lw=2.5, label="max-margin line")
for off, ls in [(-1, "--"), (1, "--")]: ax.plot(xs, -(w[0] * xs + b - off) / w[1], "k", lw=1, ls=ls)
ax.scatter(*svm.support_vectors_.T, s=160, facecolors="none", edgecolors="red", linewidths=2, label="support vectors")
ax.set_xlim(-5, 5); ax.set_ylim(-5, 5); ax.legend(loc="upper left", fontsize=8); ax.set_title("Many lines separate the classes; one has the widest 'street'"); plt.show()''')
    nb.md("The grey dotted lines all separate the training points perfectly, but they graze points. The solid line sits in the middle of the widest empty street; the dashed lines mark the street's edges, and the **circled points touching the edges are the support vectors**: "
          "delete any other point and the solution won't change.")

    # ---------------------------------------------------------- 2 math
    nb.section("📐", "2. The math: from hard margin to hinge loss",
               "A hyperplane is $\\mathbf w\\cdot\\mathbf x+b=0$. With labels $y_i\\in\\{-1,+1\\}$, point $i$ is correctly classified when $y_i(\\mathbf w\\cdot\\mathbf x_i+b)>0$. "
               "We are free to scale $(\\mathbf w,b)$, so we fix the scale by demanding the closest points satisfy $y_i(\\mathbf w\\cdot\\mathbf x_i+b)=1$. Then the **street's total width is $2/\\lVert\\mathbf w\\rVert$**: "
               "*maximising the margin = minimising $\\lVert\\mathbf w\\rVert$*.")
    nb.md(r"""**Hard-margin SVM** (data perfectly separable):
$$\min_{\mathbf w,b}\ \tfrac12\lVert\mathbf w\rVert^2\quad\text{s.t.}\quad y_i(\mathbf w\cdot\mathbf x_i+b)\ge1\ \ \forall i$$

**Soft-margin SVM** (real data overlaps): allow violations $\xi_i\ge0$ at a price $C$:
$$\min_{\mathbf w,b,\xi}\ \tfrac12\lVert\mathbf w\rVert^2+C\sum_i\xi_i\quad\text{s.t.}\quad y_i(\mathbf w\cdot\mathbf x_i+b)\ge1-\xi_i,\ \ \xi_i\ge0$$

**Hinge-loss view** (eliminating $\xi$: this is what we'll optimise from scratch):
$$\min_{\mathbf w,b}\ \tfrac12\lVert\mathbf w\rVert^2+C\sum_i\underbrace{\max\bigl(0,\ 1-y_i f(\mathbf x_i)\bigr)}_{\text{hinge loss}},\qquad f(\mathbf x)=\mathbf w\cdot\mathbf x+b$$

The hinge loss is **0 for points safely outside the street** ($y f\ge1$) and grows linearly for points inside it or on the wrong side: the same "ignore the comfortable ones, punish the troublemakers" attitude as the ε-tube of SVR (Lesson 06). Compare it with the log-loss of logistic regression, which punishes *every* point a little, even well-classified ones.

| Point's position | $y_if(\mathbf x_i)$ | Hinge loss | Role |
|---|---|---|---|
| Safely outside the street | $\ge1$ | 0 | irrelevant ("non-support") |
| Exactly on the street edge | $=1$ | 0 | **support vector** |
| Inside the street (right side) | $0<\cdot<1$ | $1-yf$ | **support vector** (margin violator) |
| Misclassified | $<0$ | $>1$ | **support vector** |

**Dual form & kernels.** The Lagrangian dual shows the solution is $\ f(\mathbf x)=\sum_{i\in SV}\alpha_iy_i\,K(\mathbf x_i,\mathbf x)+b\ $ with $0\le\alpha_i\le C$ and **only support vectors have $\alpha_i>0$**. "
"Because the data enters only through $K(\mathbf x_i,\mathbf x)=\phi(\mathbf x_i)\cdot\phi(\mathbf x)$, we can swap in a kernel and get a non-linear boundary without ever computing $\phi$ (Section 6).""")
    nb.callout("key", "**The role of C** (same as in SVR, mirror-image of regularisation strength): **small C** → violations are cheap → wide, simple street, many support vectors, more bias; "
                      "**large C** → violations are expensive → narrow street fitted tightly to the training points, fewer support vectors, more variance. scikit-learn's `C` is the *inverse* of the regularisation strength, as in logistic regression.")

    # ---------------------------------------------------------- 3 hand
    nb.section("✍️", "3. A hand-worked margin",
               "**1-D warm-up.** Points $x=1,2$ (class −1) and $x=4,5$ (class +1). The widest street is centred at the midpoint of the *closest* pair (2 and 4): boundary $x=3$.\n\n"
               "Take $w=1,\\ b=-3$: then $f(x)=x-3$. Check the margin constraints: $x=2$: $y f=(-1)(-1)=1$ ✓ (on the edge); $x=4$: $(+1)(1)=1$ ✓; $x=1$: $(-1)(-2)=2\\ge1$; $x=5$: $2\\ge1$. "
               "Margin (street width) $=2/\\lVert w\\rVert=2$: from 2 to 4. The **support vectors are $x=2$ and $x=4$**; $x=1$ and $x=5$ are irrelevant.")
    nb.code(r'''x1d = np.array([[1.], [2.], [4.], [5.]]); y1d = np.array([-1, -1, 1, 1])
m1 = SVC(kernel="linear", C=1e6).fit(x1d, y1d)
print(f"w = {m1.coef_[0][0]:.4f}, b = {m1.intercept_[0]:.4f}  -> boundary at x = {-m1.intercept_[0] / m1.coef_[0][0]:.3f}")
print("support vectors:", m1.support_vectors_.ravel(), "| margin width = 2/|w| =", round(2 / abs(m1.coef_[0][0]), 3))''')
    nb.md(r"""**Two useful formulas.** For a hyperplane $(\mathbf w,b)$ and a point $\mathbf x$:
- **Distance** to the hyperplane: $\;d=\dfrac{|\mathbf w\cdot\mathbf x+b|}{\lVert\mathbf w\rVert}$ (scikit-learn's `decision_function` returns the *scaled* distance $f(\mathbf x)=\mathbf w\cdot\mathbf x+b$).
- **Street width**: $2/\lVert\mathbf w\rVert$.

*Example:* $\mathbf w=(3,4),\ b=-5$. Then $\lVert\mathbf w\rVert=5$, street width $=0.4$. For $\mathbf x=(3,2)$: $f=9+8-5=12$ → distance $12/5=2.4$ on the positive side.""")

    # ---------------------------------------------------------- 4 seeing
    nb.section("👀", "4. Seeing margins and support vectors, and the effect of C",
               "On **overlapping** classes the soft margin earns its keep. We plot the decision function's levels $-1, 0, +1$ (street edges and centre).")
    nb.code(r'''Xo, yo = make_blobs(120, centers=[[-1, -1], [1.3, 1.3]], cluster_std=1.1, random_state=3)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
gx, gy = np.meshgrid(np.linspace(-4.5, 4.5, 250), np.linspace(-4.5, 4.5, 250))
for ax, C in zip(axes, [0.01, 1, 100]):
    m = SVC(kernel="linear", C=C).fit(Xo, yo); Z = m.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    ax.contourf(gx, gy, Z, levels=[-100, 0, 100], alpha=0.12, colors=["#2563eb", "#f97316"]); ax.contour(gx, gy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 2.2, 1])
    ax.scatter(*Xo[yo == 0].T, s=22); ax.scatter(*Xo[yo == 1].T, s=22); ax.scatter(*m.support_vectors_.T, s=110, facecolors="none", edgecolors="red", linewidths=1.5)
    ax.set_title(f"C = {C}: margin width {2 / np.linalg.norm(m.coef_):.2f}, {m.n_support_.sum()} support vectors"); ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5); ax.grid(False)
plt.tight_layout(); plt.show()''')
    nb.md("- **C = 0.01**: very forgiving: a wide street with *many* points inside it (63 support vectors): high bias, stable.\n"
          "- **C = 1**: balanced.\n- **C = 100**: strict: a narrow street hugging the classes. Fewer support vectors, but the boundary now depends heavily on a handful of points (high variance).")

    # ---------------------------------------------------------- 5 scratch
    nb.section("🔧", "5. Build it from scratch: linear SVM by sub-gradient descent",
               "We minimise (dividing the soft-margin objective by $Cn$):\n\n"
               r"$$J(\mathbf w,b)=\frac{1}{2Cn}\lVert\mathbf w\rVert^2+\frac1n\sum_i\max\bigl(0,1-y_if(\mathbf x_i)\bigr)$$" "\n\n"
               r"Sub-gradient: $\nabla_{\mathbf w}J=\dfrac{\mathbf w}{Cn}-\dfrac1n\sum_{i\in V}y_i\mathbf x_i$ and $\nabla_bJ=-\dfrac1n\sum_{i\in V}y_i$, where $V=\{i:\ y_if(\mathbf x_i)<1\}$ are the **margin violators**: "
               "only points *inside* the street push on the boundary. (The support-vector idea, in code.)")
    nb.code(r'''class LinearSVMScratch:
    """Soft-margin linear SVM trained by full-batch sub-gradient descent on the hinge loss (inputs should be standardised)."""

    def __init__(self, C=1.0, lr=0.5, n_iter=5000):
        self.C, self.lr, self.n_iter = C, lr, n_iter

    def fit(self, X, y):
        X = np.asarray(X, float); self.classes_ = np.unique(y); t = np.where(np.asarray(y) == self.classes_[1], 1.0, -1.0)    # labels -> {-1, +1}
        n, p = X.shape; w, b = np.zeros(p), 0.0; self.loss_history_ = []
        for it in range(self.n_iter):
            margin = t * (X @ w + b); viol = margin < 1                                          # violators: inside the street or wrong side
            self.loss_history_.append(w @ w / (2 * self.C * n) + np.mean(np.maximum(0, 1 - margin)))
            gw = w / (self.C * n) - (X[viol].T @ t[viol]) / n; gb = -np.mean(t * viol)
            step = self.lr / (1 + 0.01 * it) ** 0.7; w -= step * gw; b -= step * gb               # decaying step size
        self.coef_, self.intercept_ = w, b; return self

    def decision_function(self, X): return np.asarray(X, float) @ self.coef_ + self.intercept_
    def predict(self, X): return np.where(self.decision_function(X) >= 0, self.classes_[1], self.classes_[0])
    def score(self, X, y): return np.mean(self.predict(X) == np.asarray(y))


bc = load_breast_cancer(); Xc, yc = bc.data, 1 - bc.target                     # 1 = malignant
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.3, stratify=yc, random_state=RANDOM_STATE)
sc = StandardScaler().fit(Xc_tr); Ztr, Zte = sc.transform(Xc_tr), sc.transform(Xc_te)

mine = LinearSVMScratch(C=1.0).fit(Ztr, yc_tr); ref = SVC(kernel="linear", C=1.0).fit(Ztr, yc_tr)
print(f"prediction agreement with sklearn's SVC(linear): {np.mean(mine.predict(Zte) == ref.predict(Zte)):.1%}")
print(f"correlation of weight vectors: {np.corrcoef(mine.coef_, ref.coef_[0])[0, 1]:.4f}")
print(f"test accuracy: scratch {mine.score(Zte, yc_te):.3f} | sklearn {ref.score(Zte, yc_te):.3f}")
fig, ax = plt.subplots(figsize=(6, 3.4)); ax.plot(mine.loss_history_); ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("objective J"); ax.set_title("Sub-gradient descent on the soft-margin objective"); plt.show()''')
    nb.md("Two completely different solvers (our simple sub-gradient loop vs scikit-learn's SMO/libsvm quadratic-programming solver) land on essentially the same hyperplane. "
          "Sub-gradient descent is slow to converge *exactly* (the hinge has a kink), which is why production code uses specialised solvers, but the objective they minimise is the one above.")

    # ---------------------------------------------------------- 6 kernel
    nb.section("🪄", "6. The kernel trick: bending the street",
               "When no line separates the classes (concentric circles, interleaved moons) we map the data to a richer space $\\phi(\\mathbf x)$ where a hyperplane *does* work. "
               "Computing $\\phi$ explicitly may be impossibly expensive (RBF corresponds to **infinitely many** dimensions!), but the SVM only ever needs inner products, "
               "and a **kernel** computes $\\phi(\\mathbf x)\\cdot\\phi(\\mathbf x')$ directly:")
    nb.table([
        ["Linear", "$K=\\mathbf x\\cdot\\mathbf x'$", "straight hyperplane", "-"],
        ["Polynomial", "$K=(\\gamma\\,\\mathbf x\\cdot\\mathbf x'+r)^d$", "polynomial curves up to degree d", "`degree`, `gamma`, `coef0`"],
        ["**RBF / Gaussian**", "$K=\\exp(-\\gamma\\lVert\\mathbf x-\\mathbf x'\\rVert^2)$", "smooth, flexible islands (the default)", "`gamma`"],
        ["Sigmoid", "$K=\\tanh(\\gamma\\,\\mathbf x\\cdot\\mathbf x'+r)$", "rarely used", "`gamma`, `coef0`"],
    ], ["Kernel", "Formula", "Shape", "Parameters"])
    nb.code(r'''Xcirc, ycirc = make_circles(300, noise=0.12, factor=0.4, random_state=0)
Xmoon, ymoon = make_moons(300, noise=0.2, random_state=0)
fig, axes = plt.subplots(2, 3, figsize=(16, 9.4))
for row, (Xd, yd, name) in zip(axes, [(Xcirc, ycirc, "circles"), (Xmoon, ymoon, "moons")]):
    for ax, (kern, kw) in zip(row, [("linear", {}), ("poly", {"degree": 3, "coef0": 1}), ("rbf", {})]):
        m = SVC(kernel=kern, C=1, gamma="scale", **kw).fit(Xd, yd)
        plot_decision_boundary(m, Xd, yd, ax=ax, title=f"{name}: {kern} kernel\naccuracy {m.score(Xd, yd):.2f}, {m.n_support_.sum()} SVs"); ax.scatter(*m.support_vectors_.T, s=60, facecolors="none", edgecolors="k", linewidths=0.6)
plt.tight_layout(); plt.show()''')
    nb.md("A line is hopeless on circles (≈ 50 %), a cubic kernel handles moons nicely, and the RBF kernel adapts to both. "
          "**Intuition for RBF:** each support vector casts a Gaussian 'bump of influence'; the decision function is a weighted sum of bumps (positive-class SVs push up, negative-class SVs push down). "
          "$\\gamma$ sets how narrow the bumps are.")
    nb.md("### Rebuilding an RBF-SVM's decision function by hand\nJust as in Lesson 06: after fitting, `dual_coef_` holds $\\alpha_iy_i$, `support_vectors_` the $\\mathbf x_i$ and `intercept_` the bias. The scikit-learn `decision_function` is exactly $\\sum_i\\alpha_iy_iK(\\mathbf x_i,\\mathbf x)+b$:")
    nb.code(r'''def rbf(A, B, gamma):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2); return np.exp(-gamma * d2)

Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xmoon, ymoon, test_size=0.3, random_state=1)
rbf_svm = SVC(kernel="rbf", C=1.0, gamma=1.5).fit(Xm_tr, ym_tr)
manual = (rbf_svm.dual_coef_ @ rbf(rbf_svm.support_vectors_, Xm_te, 1.5) + rbf_svm.intercept_).ravel()
print("max |manual - decision_function| =", np.abs(manual - rbf_svm.decision_function(Xm_te)).max())
print(f"{len(rbf_svm.support_vectors_)} support vectors out of {len(Xm_tr)} training points | |dual coefficients| ≤ C = 1: max = {np.abs(rbf_svm.dual_coef_).max():.3f}")
print("predicted class = sign of the decision function:", np.array_equal((manual > 0).astype(int), rbf_svm.predict(Xm_te)))''')
    nb.callout("key", "The entire SVM is a list of support vectors with a signed weight each (|weight| ≤ C) and a bias. The weights hitting the ceiling $|\\alpha_iy_i|=C$ are the *margin violators*; those strictly between 0 and C sit exactly on the street edge.")

    # ---------------------------------------------------------- 7 gamma
    nb.section("🎛️", "7. gamma and C together",
               "For the RBF kernel two dials interact: **C** (how much we punish violations) and **gamma** (how local the bumps are). Small γ → broad, smooth boundaries; large γ → tiny islands around individual points (overfitting).")
    nb.code(r'''fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
for ax, g in zip(axes, [0.1, 1, 10, 100]):
    m = SVC(gamma=g, C=1).fit(Xm_tr, ym_tr)
    plot_decision_boundary(m, Xm_tr, ym_tr, ax=ax, title=f"gamma = {g}\ntrain {m.score(Xm_tr, ym_tr):.2f} | test {m.score(Xm_te, ym_te):.2f} | {m.n_support_.sum()} SVs")
plt.tight_layout(); plt.show()''')
    nb.md("`gamma=0.1` is too stiff (underfit), `1–10` is right, and `100` carves islands around single points: training accuracy rises while test accuracy falls. "
          "(`gamma='scale'`, the default, sets $\\gamma=1/(p\\cdot\\mathrm{Var}(X))$: a sensible starting point.)")

    # ---------------------------------------------------------- 8 scaling & tuning
    nb.section("⚖️", "8. Scaling and tuning on a log grid",
               "Like KNN and SVR, SVMs are **distance-based**: unscaled features with big ranges dominate the kernel. Always standardise (inside a Pipeline).")
    nb.code(r'''raw = SVC().fit(Xc_tr, yc_tr); scaled = make_pipeline(StandardScaler(), SVC()).fit(Xc_tr, yc_tr)
print(f"RBF-SVM on raw cancer features   : test accuracy = {raw.score(Xc_te, yc_te):.3f}")
print(f"RBF-SVM on scaled cancer features: test accuracy = {scaled.score(Xc_te, yc_te):.3f}")''')
    nb.md("**Tuning.** C and γ matter on a **multiplicative** scale (0.01 → 0.1 → 1 → 10), so search a *logarithmic* grid, with cross-validation, then inspect the whole landscape rather than just the winner:")
    nb.code(r'''Cs = np.logspace(-2, 3, 6); gammas = np.logspace(-4, 0, 5)
pipe = make_pipeline(StandardScaler(), SVC())
gs = GridSearchCV(pipe, {"svc__C": Cs, "svc__gamma": gammas}, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy").fit(Xc_tr, yc_tr)
heat = gs.cv_results_["mean_test_score"].reshape(len(Cs), len(gammas))
fig, ax = plt.subplots(figsize=(7, 4.6)); im = ax.imshow(heat, cmap="viridis", aspect="auto", origin="lower")
ax.set_xticks(range(len(gammas))); ax.set_xticklabels([f"{g:.0e}" for g in gammas]); ax.set_yticks(range(len(Cs))); ax.set_yticklabels([f"{c:g}" for c in Cs]); ax.set_xlabel("gamma"); ax.set_ylabel("C"); ax.grid(False)
for i in range(len(Cs)):
    for j in range(len(gammas)): ax.text(j, i, f"{heat[i, j]:.3f}", ha="center", va="center", color="white" if heat[i, j] < heat.max() - 0.03 else "black", fontsize=8)
plt.colorbar(im, label="CV accuracy"); ax.set_title("CV accuracy over the (C, gamma) grid"); plt.show()
print("best:", gs.best_params_, " CV acc =", round(gs.best_score_, 3), " TEST acc =", round(gs.best_estimator_.score(Xc_te, yc_te), 3))''')
    nb.md("There is a **ridge** of good settings running diagonally: a larger C wants a smaller γ and vice versa (both control complexity). Corners with tiny C *and* tiny γ underfit; large C *and* large γ overfit. "
          "A broad plateau of near-equal scores means the result is robust to the exact choice.")

    # ---------------------------------------------------------- 9 practical
    nb.section("🧰", "9. Probabilities, imbalance, speed, and the SVM family",
               "### 9a. Probabilities\nAn SVM natively outputs a *signed distance* (`decision_function`), not a probability. `SVC(probability=True)` fits **Platt scaling**: a logistic regression on the scores, using internal 5-fold CV (so training is ~5× slower). "
               "The scores are great for *ranking* (AUC) even without it.")
    nb.code(r'''svc_p = make_pipeline(StandardScaler(), SVC(probability=True, random_state=0)).fit(Xc_tr, yc_tr)
print("first 5 test samples -> decision_function:", svc_p.decision_function(Xc_te[:5]).round(2), "\n                        predict_proba(malignant):", svc_p.predict_proba(Xc_te[:5])[:, 1].round(3))''')
    nb.md("### 9b. Imbalanced data: `class_weight`\n`class_weight='balanced'` multiplies C by a per-class weight, so violating a rare-class point costs far more (wider street around the rare class).")
    nb.code(r'''Xi, yi = make_classification(n_samples=5000, n_features=10, n_informative=5, weights=[0.98, 0.02], flip_y=0.01, class_sep=1.2, random_state=1)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.3, stratify=yi, random_state=RANDOM_STATE)
rows = []
for cw in [None, "balanced"]:
    m = make_pipeline(StandardScaler(), SVC(class_weight=cw)).fit(Xi_tr, yi_tr); p = m.predict(Xi_te)
    rows.append({"class_weight": str(cw), "recall": recall_score(yi_te, p), "precision": precision_score(yi_te, p, zero_division=0), "PR-AUC (decision scores)": average_precision_score(yi_te, m.decision_function(Xi_te))})
display(pd.DataFrame(rows).set_index("class_weight").round(3))''')
    nb.md("**Notice:** `class_weight='balanced'` bought recall (0.34 → 0.63) at a big precision price, and it also *lowered* the PR-AUC of the decision scores (0.66 → 0.48): re-weighting changes what the model learns, it is **not** merely a threshold shift. "
          "On imbalanced problems compare the *ranking* quality (PR-AUC) and then pick the operating point deliberately: don't assume balanced weights are free.")
    nb.md("### 9c. Speed and the family\nKernel `SVC` training costs between $O(n^2)$ and $O(n^3)$ in the worst case: fine up to ~50 000 rows, hopeless beyond. On this easy synthetic problem the empirical growth is gentler (roughly ×2.5–3.3 per doubling of n, i.e. about $n^{1.5}$) but the trend is clear: linear solvers win by orders of magnitude.")
    nb.code(r'''rows = []
for n in [500, 1000, 2000, 4000, 8000]:
    Xs, ys = make_classification(n, 20, n_informative=8, random_state=0)
    t0 = time.perf_counter(); SVC().fit(Xs, ys); a = time.perf_counter() - t0
    t0 = time.perf_counter(); LinearSVC(max_iter=5000).fit(Xs, ys); b_ = time.perf_counter() - t0
    rows.append({"n": n, "SVC (RBF) s": a, "LinearSVC s": b_})
d = pd.DataFrame(rows); d["SVC time ratio"] = d["SVC (RBF) s"] / d["SVC (RBF) s"].shift(1); display(d.round(4))''')
    nb.table([
        ["`SVC`", "kernels (linear/poly/RBF/…), libsvm", "O(n²–n³)", "up to ~50k rows, non-linear problems"],
        ["`LinearSVC`", "linear only, liblinear (squared hinge by default)", "≈ O(n·p)", "large n, many features (text), linear problems"],
        ["`SGDClassifier(loss='hinge')`", "linear SVM trained by stochastic gradient descent", "O(n·p) per epoch, online", "huge or streaming data"],
        ["`NuSVC`", "ν-parametrisation (fraction of SVs/violators)", "like SVC", "when you want to bound the support-vector fraction"],
        ["`OneClassSVM`", "novelty/anomaly detection", "like SVC", "outlier detection"],
    ], ["Class", "What", "Training cost", "Use when"])
    nb.md("**Multi-class:** `SVC` uses **one-vs-one** (a binary SVM for every pair of classes, vote); `LinearSVC` uses one-vs-rest. Quick check on the 10-class digits data:")
    nb.code(r'''dg = load_digits(); Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(dg.data, dg.target, test_size=0.3, stratify=dg.target, random_state=RANDOM_STATE)
svc_d = SVC().fit(Xd_tr, yd_tr); print(f"SVC (RBF, one-vs-one) on digits: test accuracy = {svc_d.score(Xd_te, yd_te):.3f}  (KNN got 0.987)   | support vectors: {svc_d.n_support_.sum()} of {len(Xd_tr)}")
print("decision_function shape (one-vs-one, 45 class pairs):", SVC(decision_function_shape="ovo").fit(Xd_tr, yd_tr).decision_function(Xd_te[:3]).shape)''')

    # ---------------------------------------------------------- 10 profile
    nb.section("📋", "10. Cheat-sheet: the full profile of Support Vector Machines")
    nb.table([
        ["**Underlying concept**", "Find the hyperplane with the widest margin between classes; only points on/inside the margin (support vectors) define it; kernels bend the boundary."],
        ["**How it works**", "Solve a convex quadratic programme: minimise ½‖w‖² + C·Σ hinge loss. Predict with sign(Σ αᵢyᵢK(svᵢ,x) + b)."],
        ["**Mathematical foundation**", "Margin geometry (2/‖w‖), hinge loss, Lagrangian duality & KKT conditions, Mercer kernels, structural risk minimisation."],
        ["**Important parameters**", "`kernel`, `C`, `gamma` (rbf/poly/sigmoid), `degree`, `coef0`, `class_weight`, `probability`, `decision_function_shape`, `tol`, `cache_size`. Always scale features."],
        ["**Advantages**", "Strong accuracy on small/medium, high-dimensional data; effective when p > n; convex ⇒ global optimum; robust (depends only on support vectors); flexible via kernels; memory-efficient model."],
        ["**Disadvantages**", "Needs scaling and careful tuning of C/γ; O(n²–n³) training; no native probabilities; hard to interpret (non-linear kernels); sensitive to noisy overlapping classes; poor with very large n."],
        ["**Complexity**", "Training ≈ O(n²·p) to O(n³) (kernel); LinearSVC ≈ O(n·p); prediction O(n_SV·p) per sample; memory O(n²) kernel cache."],
        ["**Use cases**", "Text/document classification (linear SVM), bioinformatics & small high-dimensional data, image features, handwriting, anomaly detection, any moderately-sized tabular problem needing a strong non-linear baseline."],
        ["**Limitations**", "Doesn't scale to millions of rows (use linear/SGD variants); probabilities via Platt scaling are an add-on; feature importance only for linear kernels; hyper-parameter search can be expensive."],
    ], ["Question", "Answer"])

    # ---------------------------------------------------------- 11 worked
    nb.section("🧪", "11. Worked examples")
    nb.md(r"""### Worked Example 1: Distance, margin and class

A trained linear SVM has $\mathbf w=(2,-1),\ b=-3$. Classify $\mathbf x=(4,1)$ and find its distance from the boundary and whether it is inside the street.

$f=2\cdot4-1\cdot1-3=4>0$ → **positive class**. $\lVert\mathbf w\rVert=\sqrt5\approx2.236$ → distance $4/2.236=\mathbf{1.789}$. The street half-width is $1/\lVert\mathbf w\rVert=0.447$; the point is at 1.789 > 0.447, so it is **safely outside** the street (hinge loss 0, not a support vector).""")
    nb.code(r'''w_, b_, x_ = np.array([2., -1.]), -3., np.array([4., 1.]); f_ = w_ @ x_ + b_
print("f(x) =", f_, "| distance =", round(abs(f_) / np.linalg.norm(w_), 3), "| street half-width =", round(1 / np.linalg.norm(w_), 3), "| hinge loss (true class +1):", max(0, 1 - f_))''')
    nb.md(r"""### Worked Example 2: Hinge loss of three points

Scores $f=(2.3,\ 0.4,\ -0.8)$ for true labels $y=(+1,\ +1,\ +1)$. Hinge $=\max(0,1-yf)$: $(0,\ 0.6,\ 1.8)$. The first point is safely correct (loss 0), the second is correct but **inside the street** (loss 0.6, support vector), the third is **misclassified** (loss 1.8). Mean hinge $=0.8$.""")
    nb.code(r'''f3 = np.array([2.3, 0.4, -0.8]); y3 = np.array([1, 1, 1]); h3 = np.maximum(0, 1 - y3 * f3); print(h3, "mean =", h3.mean())''')

    # ---------------------------------------------------------- problems
    nb.problem(
        "Distances and margins by hand",
        "Write `hyperplane_info(w, b, x)` returning a dict with: `score` (= w·x + b), `distance` (|score| / ‖w‖), `side` (`+1` or `-1`), and `inside_street` (True if |score| < 1). "
        "Use it with `w = [3, 4], b = -5` for `x = [3, 2]` (→ `info1`) and `x = [1, 0.5]` (→ `info2`), and store the street width `2/‖w‖` in `width`.",
        ["`score = np.dot(w, x) + b`; `distance = abs(score) / np.linalg.norm(w)`.",
         "`side = 1 if score >= 0 else -1`; `inside_street = abs(score) < 1`.",
         "For w = (3, 4): ‖w‖ = 5; the second point has score 3 + 2 − 5 = 0 → right on the boundary (inside the street)."],
        'def hyperplane_info(w, b, x):\n    pass\n\nw = np.array([3., 4.]); b = -5.0\ninfo1 = info2 = width = None   # your code\n',
        'def hyperplane_info(w, b, x):\n    score = float(np.dot(w, x) + b)\n    return {"score": score, "distance": abs(score) / np.linalg.norm(w), "side": 1 if score >= 0 else -1, "inside_street": abs(score) < 1}\n\n'
        'w = np.array([3., 4.]); b = -5.0\ninfo1 = hyperplane_info(w, b, np.array([3., 2.])); info2 = hyperplane_info(w, b, np.array([1., 0.5])); width = 2 / np.linalg.norm(w)\nprint(info1, info2, width)',
        'check("(3,2): score 12, distance 2.4, positive side, outside the street", lambda: abs(info1["score"] - 12) < 1e-9 and abs(info1["distance"] - 2.4) < 1e-9 and info1["side"] == 1 and not info1["inside_street"])\n'
        'check("(1,0.5): score 0 -> exactly on the boundary, inside the street", lambda: abs(info2["score"]) < 1e-9 and info2["inside_street"])\ncheck("street width = 0.4", lambda: abs(width - 0.4) < 1e-9)',
        explain="Distance = |f(x)|/‖w‖ and street width = 2/‖w‖. A point with |f(x)| < 1 is inside the street, which makes it a support vector / margin violator.")

    nb.problem(
        "Hinge loss and the soft-margin objective",
        "Write `hinge(y, f)` (mean hinge loss for labels in {−1, +1} and scores `f`) and `svm_objective(w, b, X, y, C)` = `0.5*||w||² + C * sum(hinge per point)`. "
        "With `X = [[1, 2], [2, 0.5], [3, 3], [0.5, 0.2]]`, `y = [1, 1, -1, -1]`, `w = [0.5, 0.5]`, `b = -1`, `C = 2`, store `mean_hinge`, `n_violators` (points with y·f < 1) and `objective`.",
        ["`f = X @ w + b`; per-point hinge = `np.maximum(0, 1 - y * f)`.",
         "Violators: `np.sum(y * f < 1)`.",
         "Objective: `0.5 * w @ w + C * per_point.sum()`."],
        'def hinge(y, f):\n    pass\n\ndef svm_objective(w, b, X, y, C):\n    pass\n\nXh = np.array([[1, 2], [2, 0.5], [3, 3], [0.5, 0.2]]); yh = np.array([1, 1, -1, -1]); wh = np.array([0.5, 0.5]); bh = -1.0\nmean_hinge = n_violators = objective = None   # your code\n',
        'def hinge(y, f):\n    return np.mean(np.maximum(0, 1 - y * f))\n\ndef svm_objective(w, b, X, y, C):\n    return 0.5 * w @ w + C * np.sum(np.maximum(0, 1 - y * (X @ w + b)))\n\nXh = np.array([[1, 2], [2, 0.5], [3, 3], [0.5, 0.2]]); yh = np.array([1, 1, -1, -1]); wh = np.array([0.5, 0.5]); bh = -1.0\n'
        'fh = Xh @ wh + bh; mean_hinge = hinge(yh, fh); n_violators = int(np.sum(yh * fh < 1)); objective = svm_objective(wh, bh, Xh, yh, 2)\nprint(fh, mean_hinge, n_violators, objective)',
        'check("scores f = [0.5, 0.25, 2.0, -0.65]", lambda: np.allclose(Xh @ wh + bh, [0.5, 0.25, 2.0, -0.65]))\ncheck("mean hinge = (0.5 + 0.75 + 3.0 + 0.35) / 4 = 1.15", lambda: abs(mean_hinge - 1.15) < 1e-9)\ncheck("all 4 points violate the margin (y·f < 1)", lambda: n_violators == 4)\ncheck("objective = 0.25 + 2·4.6 = 9.45", lambda: abs(objective - 9.45) < 1e-9)',
        explain="Per-point hinge losses are [0.5, 0.75, 3.0, 0.35]: points 1 and 2 are on the correct side but inside the street (y·f < 1), point 3 (a −1 point scored +2.0) is badly misclassified (loss 3.0), and point 4 is correctly classified but still inside the street (y·f = 0.65). "
                "Sum = 4.6, so mean = 1.15 and the objective is ½‖w‖² + C·Σhinge = 0.25 + 2·4.6 = 9.45. Training would move w and b to shrink exactly these violations.",
        level="⭐⭐")

    nb.problem(
        "C versus margin and support vectors",
        "Using the overlapping blobs `Xo, yo`: for `C` in `[0.01, 0.1, 1, 10, 100]` fit `SVC(kernel='linear', C=C)` and record the **margin width** `2/‖w‖` (list `margins`) and **number of support vectors** (list `n_sv`). "
        "Set `margin_shrinks` = True if margins are non-increasing as C grows, and `sv_shrinks` = True if the support-vector counts are non-increasing.",
        ["`m.coef_[0]` is w; `np.linalg.norm(m.coef_)` is ‖w‖; `m.n_support_.sum()` counts all support vectors.",
         "Larger C = stricter = narrower street = fewer margin violators.",
         "Use `all(a >= b for a, b in zip(lst, lst[1:]))` to test monotonic decrease."],
        'Cs_ = [0.01, 0.1, 1, 10, 100]\nmargins = []; n_sv = []\nmargin_shrinks = sv_shrinks = None   # your code\n',
        'Cs_ = [0.01, 0.1, 1, 10, 100]\nmargins = []; n_sv = []\nfor C in Cs_:\n    m = SVC(kernel="linear", C=C).fit(Xo, yo); margins.append(2 / np.linalg.norm(m.coef_)); n_sv.append(int(m.n_support_.sum()))\n'
        'margin_shrinks = all(a >= b for a, b in zip(margins, margins[1:])); sv_shrinks = all(a >= b for a, b in zip(n_sv, n_sv[1:]))\nprint(np.round(margins, 2), n_sv, margin_shrinks, sv_shrinks)',
        'check("five values recorded", lambda: len(margins) == 5 and len(n_sv) == 5)\ncheck("the street narrows as C grows", lambda: margin_shrinks)\ncheck("support vectors become fewer as C grows", lambda: sv_shrinks)\ncheck("C=0.01 uses most of the data as support vectors (> 40%)", lambda: n_sv[0] > 0.4 * len(yo))',
        explain="With tiny C almost every point may sit inside the wide street (and is therefore a support vector); with large C the street hugs the data and only a few points define it.")

    nb.problem(
        "Tune C and gamma on a log grid",
        "On the cancer data (`Xc_tr, yc_tr, Xc_te, yc_te`) run `GridSearchCV` (`StratifiedKFold(5, shuffle=True, random_state=0)`, `scoring='accuracy'`) for `make_pipeline(StandardScaler(), SVC())` over `C ∈ np.logspace(-1, 2, 4)` and `gamma ∈ np.logspace(-3, -1, 3)`. "
        "Store `best_params` (dict), `best_cv` and `test_acc`.",
        ["Parameter names in a pipeline: `'svc__C'` and `'svc__gamma'`.",
         "`gs.best_params_`, `gs.best_score_`, and `gs.best_estimator_.score(Xc_te, yc_te)`.",
         "You should land near 97–98 % test accuracy."],
        'best_params = best_cv = test_acc = None   # your code\n',
        'gs2 = GridSearchCV(make_pipeline(StandardScaler(), SVC()), {"svc__C": np.logspace(-1, 2, 4), "svc__gamma": np.logspace(-3, -1, 3)}, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="accuracy").fit(Xc_tr, yc_tr)\n'
        'best_params = gs2.best_params_; best_cv = gs2.best_score_; test_acc = gs2.best_estimator_.score(Xc_te, yc_te)\nprint(best_params, round(best_cv, 3), round(test_acc, 3))',
        'check("both hyper-parameters tuned", lambda: set(best_params) == {"svc__C", "svc__gamma"})\ncheck("CV accuracy > 0.96", lambda: best_cv > 0.96)\ncheck("test accuracy > 0.96", lambda: test_acc > 0.96)',
        explain="On this smooth, well-separated problem many (C, γ) pairs perform within a hair of each other: a broad plateau. Scaling mattered far more than the exact grid point.",
        level="⭐⭐")

    nb.problem(
        "Rebuild the decision function from support vectors",
        "Fit `m3 = SVC(kernel='rbf', C=2.0, gamma=0.8).fit(Xm_tr, ym_tr)` on the moons training data. Using only `m3.support_vectors_`, `m3.dual_coef_`, `m3.intercept_` and the lesson's `rbf` function, compute `manual` = decision values for the first 10 rows of `Xm_te`, "
        "`max_gap` vs `m3.decision_function(Xm_te[:10])`, and `n_at_ceiling` = number of support vectors whose |dual coefficient| equals C (within 1e-6): the margin violators.",
        ["`K = rbf(m3.support_vectors_, Xm_te[:10], gamma=0.8)`; `manual = (m3.dual_coef_ @ K + m3.intercept_).ravel()`.",
         "`np.isclose(np.abs(m3.dual_coef_), 2.0, atol=1e-6).sum()`.",
         "Support vectors strictly below the ceiling lie exactly on the street edge."],
        'm3 = SVC(kernel="rbf", C=2.0, gamma=0.8).fit(Xm_tr, ym_tr)\nmanual = max_gap = n_at_ceiling = None   # your code\n',
        'm3 = SVC(kernel="rbf", C=2.0, gamma=0.8).fit(Xm_tr, ym_tr)\nK3 = rbf(m3.support_vectors_, Xm_te[:10], gamma=0.8)\nmanual = (m3.dual_coef_ @ K3 + m3.intercept_).ravel()\nmax_gap = np.abs(manual - m3.decision_function(Xm_te[:10])).max()\nn_at_ceiling = int(np.isclose(np.abs(m3.dual_coef_), 2.0, atol=1e-6).sum())\nprint(max_gap, n_at_ceiling, len(m3.support_vectors_))',
        'check("manual decision values match sklearn", lambda: max_gap < 1e-9)\ncheck("10 values computed", lambda: len(manual) == 10)\ncheck("some (but not all) support vectors sit at the ceiling", lambda: 0 < n_at_ceiling < len(m3.support_vectors_))',
        explain="Support vectors at |α| = C are the margin violators (inside the street or misclassified); those with 0 < |α| < C lie exactly on the street edge. The whole classifier is this weighted sum of Gaussian bumps.",
        level="⭐⭐⭐")

    nb.problem(
        "Balance the rare class",
        "On the fraud data (`Xi_tr, yi_tr, Xi_te, yi_te`) fit `make_pipeline(StandardScaler(), SVC(class_weight=cw))` for `cw=None` and `cw='balanced'`. "
        "Store `recall_plain`, `recall_bal`, `prec_plain`, `prec_bal` (use `zero_division=0`). Set `tradeoff = recall_bal > recall_plain and prec_bal < prec_plain`.",
        ["`recall_score(yi_te, model.predict(Xi_te))`, `precision_score(..., zero_division=0)`.",
         "Balanced weights make errors on the rare class cost more → more positives found, more false alarms.",
         "This is the same trade-off as moving the threshold."],
        'recall_plain = recall_bal = prec_plain = prec_bal = tradeoff = None   # your code\n',
        'mp = make_pipeline(StandardScaler(), SVC()).fit(Xi_tr, yi_tr); mb = make_pipeline(StandardScaler(), SVC(class_weight="balanced")).fit(Xi_tr, yi_tr)\nrecall_plain, recall_bal = recall_score(yi_te, mp.predict(Xi_te)), recall_score(yi_te, mb.predict(Xi_te))\n'
        'prec_plain, prec_bal = precision_score(yi_te, mp.predict(Xi_te), zero_division=0), precision_score(yi_te, mb.predict(Xi_te), zero_division=0)\ntradeoff = bool(recall_bal > recall_plain and prec_bal < prec_plain)\nprint(recall_plain, recall_bal, prec_plain, prec_bal, tradeoff)',
        'check("balanced weights increase recall", lambda: recall_bal > recall_plain)\ncheck("...at a cost in precision", lambda: prec_bal < prec_plain)\ncheck("tradeoff is True", lambda: tradeoff)',
        explain="Weighting shifts the boundary toward the majority class so that more rare points are caught: at the cost of false alarms.",
        level="⭐⭐")

    nb.quiz([
        ("What is a support vector in an SVM classifier?", "A training point on or inside the margin (or misclassified): exactly the points with non-zero dual coefficients. Deleting any other point leaves the solution unchanged."),
        ("You increase C from 1 to 1000. What happens to the margin and to the risk of overfitting?",
         "The margin narrows (violations are punished heavily), the boundary hugs the training points, and overfitting risk rises."),
        ("Why can an RBF-SVM separate concentric circles while a linear SVM can't?",
         "The RBF kernel corresponds to an (infinite-dimensional) feature space in which the circles become linearly separable; the SVM finds the max-margin hyperplane there, which appears as a closed curve in the original space."),
        ("Why is `StandardScaler` essential before an RBF-SVM?", "The kernel depends on Euclidean distances, so features with large numeric ranges would dominate the similarity and effectively mute the others."),
        ("You have 5 million labelled text documents represented by sparse TF-IDF vectors. Which SVM tool do you use and why?",
         "LinearSVC or SGDClassifier(loss='hinge'): kernel SVC is O(n²–n³) and impractical; text is high-dimensional and usually linearly separable, so a linear SVM is both fast and accurate."),
    ])

    nb.footer(["SVM = maximum-margin classifier: minimise ½‖w‖² + C·Σ hinge loss; margin width = 2/‖w‖.",
               "Only support vectors (on/inside the margin) matter: f(x) = Σ αᵢyᵢK(svᵢ, x) + b.",
               "Kernels (linear, polynomial, RBF) bend the boundary without computing the feature map; γ sets locality, C the strictness.",
               "Scale features; tune C and γ on a log grid; use class_weight for imbalance; Platt scaling for probabilities.",
               "Superb on small/medium data, but O(n²–n³): switch to LinearSVC / SGD for large n."],
              "Classification Showdown", "16_Classification_Showdown.ipynb")
    return nb
