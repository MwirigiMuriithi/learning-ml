"""
Lesson 08 - Logistic Regression (from scratch)

Problem: a telecom company wants to predict *customer churn* (will the customer
leave?) from two features: tenure (months as a customer) and support calls in
the last quarter.  Label y = 1 means "churned".

What is implemented (NumPy only):
  * sigmoid, log-loss (binary cross-entropy), gradient  X^T (p - y) / n
  * batch gradient descent with an optional L2 penalty
  * Newton / IRLS (second-order) as a cross-check of the optimum
  * softmax regression (multinomial logistic regression) for k > 2 classes
  * a threshold table (precision / recall at several cut-offs)

What is animated:
  logistic_regression.gif   - probability heat-map + 0.5 boundary moving during
                              training | the sigmoid z -> p with every customer
                              projected on the z-axis | log-loss and accuracy.
  softmax_regression.gif    - the 3-class extension: blended class probabilities
                              and the 3 decision regions forming.

Run:  python logistic_regression.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb


# ---------------------------------------------------------------- the models
def log_loss(y, p, eps=1e-12):
    """Binary cross-entropy  -mean[ y log p + (1-y) log(1-p) ]  (eps avoids log 0)."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegression:
    """P(y=1 | x) = sigmoid(b + w.x).   theta = [b, w1, ..., wd]."""

    def __init__(self, l2=0.0):
        self.l2 = l2          # strength lambda of the ridge penalty (bias is NOT penalised)
        self.theta = None

    @staticmethod
    def _design(X):
        X = np.atleast_2d(X)
        return np.c_[np.ones(len(X)), X]

    def decision_function(self, X):
        """The logit z = theta^T x~ : the log-odds of class 1."""
        return self._design(X) @ self.theta

    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))            # p = 1 / (1 + e^-z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def loss(self, X, y):
        """Penalised objective  J = logloss + (lambda/2n) * |w|^2 (matches the gradient below)."""
        p = self.predict_proba(X)
        pen = self.l2 / (2 * len(y)) * np.sum(self.theta[1:] ** 2)
        return log_loss(y, p) + pen

    def _grad(self, A, y, theta):
        p = sigmoid(A @ theta)
        g = A.T @ (p - y) / len(y)                            # the famous X^T (p - y) / n
        reg = self.l2 / len(y) * theta                        # derivative of (lambda/2n)|w|^2
        reg[0] = 0.0                                          # never shrink the intercept
        return g + reg

    def fit_gd(self, X, y, lr=0.5, n_iters=200, theta0=None):
        """Batch gradient descent. Returns the whole path of theta (n_iters+1, d+1)."""
        A = self._design(X)
        theta = np.zeros(A.shape[1]) if theta0 is None else np.array(theta0, float)
        path = [theta.copy()]
        for _ in range(n_iters):
            theta = theta - lr * self._grad(A, y, theta)      # theta <- theta - eta * grad
            path.append(theta.copy())
        self.theta = theta
        return np.array(path)

    def fit_newton(self, X, y, n_iters=15):
        """Newton / IRLS:  theta <- theta - H^-1 grad,  H = A^T diag(p(1-p)) A / n  (+ ridge).
        Converges in ~6-10 steps; used here to know the true optimum."""
        A = self._design(X)
        n = len(y)
        theta = np.zeros(A.shape[1])
        R = np.eye(A.shape[1]) * self.l2 / n
        R[0, 0] = 0
        for _ in range(n_iters):
            p = sigmoid(A @ theta)
            H = (A.T * (p * (1 - p))) @ A / n + R + 1e-10 * np.eye(A.shape[1])
            theta = theta - np.linalg.solve(H, self._grad(A, y, theta))
        self.theta = theta
        return self


class SoftmaxRegression:
    """Multinomial logistic regression: P(class c | x) = softmax(W x~)_c.
    W has shape (k, d+1).  Gradient of the cross-entropy:  (P - Y)^T X~ / n."""

    def __init__(self, l2=0.0):
        self.l2, self.W = l2, None

    @staticmethod
    def _design(X):
        return np.c_[np.ones(len(X)), np.atleast_2d(X)]

    def predict_proba(self, X, W=None):
        W = self.W if W is None else W
        return softmax(self._design(X) @ W.T, axis=1)         # rows sum to 1

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def fit_gd(self, X, y, k, lr=0.5, n_iters=150):
        A, Y, n = self._design(X), one_hot(y, k), len(y)
        W = np.zeros((k, A.shape[1]))
        path = [W.copy()]
        for _ in range(n_iters):
            P = softmax(A @ W.T, axis=1)
            grad = (P - Y).T @ A / n                           # (k, d+1)
            grad[:, 1:] += self.l2 / n * W[:, 1:]
            W = W - lr * grad
            path.append(W.copy())
        self.W = W
        return np.array(path)


def precision_recall(y, yhat):
    tp = np.sum((y == 1) & (yhat == 1)); fp = np.sum((y == 0) & (yhat == 1)); fn = np.sum((y == 1) & (yhat == 0))
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


# ------------------------------------------------------------- the animation
def churn_animation():
    X3, y, names = make_customer_churn(n=500, seed=4)
    X = X3[:, [0, 2]]                                    # tenure_months, support_calls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test=0.25, seed=1)
    Xs, mu, sd = standardize(Xtr)                        # GD likes round loss bowls
    Xte_s = (Xte - mu) / sd

    # reference optimum (Newton) and a gradient-descent run from a deliberately wrong start
    star = LogisticRegression().fit_newton(Xs, ytr)
    gd = LogisticRegression()
    path = gd.fit_gd(Xs, ytr, lr=0.6, n_iters=120, theta0=[1.5, 2.0, -2.0])   # wrong signs on purpose
    losses = [log_loss(ytr, sigmoid(LogisticRegression._design(Xs) @ t)) for t in path]
    accs = [accuracy(ytr, (LogisticRegression._design(Xs) @ t >= 0).astype(int)) for t in path]
    best = log_loss(ytr, star.predict_proba(Xs))

    # the model object used to draw each frame
    m = LogisticRegression()
    jit = rng_(0).uniform(-0.18, 0.18, len(ytr))        # jitter for the integer 'calls' axis (display only)

    fig = plt.figure(figsize=(13, 4.8))
    fig.suptitle("Logistic regression: a linear score z squashed into a probability", fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    gs = GridSpec(2, 3, figure=fig, left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28, hspace=0.55)
    a1 = fig.add_subplot(gs[:, 0]); a2 = fig.add_subplot(gs[:, 1])
    a3 = fig.add_subplot(gs[0, 2]); a4 = fig.add_subplot(gs[1, 2])
    zz = np.linspace(-6, 6, 200)
    n_it = len(path) - 1

    with Animator(here(__file__, "logistic_regression.gif"), fps=8, hold=2.5) as anim:
        for i in frame_steps(n_it, 36):
            m.theta = path[i]
            # (1) feature space: probability heat-map + 0.5 boundary ---------------------
            a1.clear()
            probability_map(a1, lambda P: m.predict_proba((P - mu) / sd), X, pad=0.3)
            scatter_classes(a1, np.c_[Xtr[:, 0], Xtr[:, 1] + jit], ytr, s=16)
            a1.set(title="Feature space: P(churn) and the p=0.5 line", xlabel="tenure (months)", ylabel="support calls")
            a1.text(0.02, 0.97, "blue = stayed, orange = churned", transform=a1.transAxes, va="top", fontsize=8.5, color=INK2)

            # (2) the sigmoid with customers projected on z ------------------------------
            a2.clear()
            z = m.decision_function(Xs)
            p = sigmoid(z)
            a2.plot(zz, sigmoid(zz), color=INK2, linewidth=1.4)
            a2.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.9)
            a2.axvline(0, color=MUTED, linestyle="--", linewidth=0.9)
            for c in (0, 1):
                a2.scatter(np.clip(z[ytr == c], -6, 6), p[ytr == c], s=12, color=CAT[c], edgecolor=SURFACE, linewidth=0.4, zorder=3)
            a2.set(title="Sigmoid: z = b + w.x  ->  p", xlabel="score z (log-odds)", ylabel="P(churn)")
            a2.set_xlim(-6, 6); a2.set_ylim(-0.03, 1.03)
            a2.text(5.8, 0.06, "predict stay", ha="right", fontsize=8.5, color=INK2)
            a2.text(5.8, 0.93, "predict churn", ha="right", fontsize=8.5, color=INK2)
            a2.text(-5.8, 0.55, "z=0 <=> p=0.5", ha="left", fontsize=8.5, color=INK2)

            # (3) log-loss, (4) accuracy ---------------------------------------------------
            a3.clear()
            a3.plot(range(i + 1), losses[: i + 1], color=BLUE)
            a3.scatter([i], [losses[i]], s=30, color=ORANGE, zorder=5)
            a3.axhline(best, color=MUTED, linestyle="--", linewidth=1)
            a3.set(title="Training log-loss", xlabel="", ylabel="log-loss", xlim=(0, n_it))
            a3.set_xticklabels([])
            a3.text(n_it, best + 0.02, "optimum (Newton)", ha="right", fontsize=8, color=INK2)
            a4.clear()
            a4.plot(range(i + 1), accs[: i + 1], color=BLUE)
            a4.scatter([i], [accs[i]], s=30, color=ORANGE, zorder=5)
            a4.set(title="Training accuracy", xlabel="iteration", ylabel="accuracy", xlim=(0, n_it), ylim=(0.3, 0.9))

            caption(fig, f"iteration {i:>3}:  b={path[i][0]:5.2f}  w_tenure={path[i][1]:5.2f}  w_calls={path[i][2]:5.2f}   "
                         f"log-loss={losses[i]:.3f}   accuracy={accs[i]:.1%}")
            anim.grab(fig)

    # ---- sanity checks / numbers quoted in the lesson -------------------------------------
    print(f"  GD   theta={np.round(gd.theta, 3)}  | Newton theta={np.round(star.theta, 3)}")
    print(f"  test accuracy (GD) = {accuracy(yte, gd.predict(Xte_s)):.3f}, base rate churn = {ytr.mean():.2f}")
    # L2 option
    for lam in (0.0, 10.0, 100.0):
        r = LogisticRegression(l2=lam).fit_newton(Xs, ytr)
        print(f"  L2 lambda={lam:>5}: |w| = {np.linalg.norm(r.theta[1:]):.3f}")
    # threshold table
    pte = gd.predict_proba(Xte_s)
    print("  threshold  precision  recall  accuracy")
    for t in (0.3, 0.5, 0.7):
        pr, rc = precision_recall(yte, (pte >= t).astype(int))
        print(f"   {t:>5.1f}      {pr:6.2f}    {rc:6.2f}   {accuracy(yte, (pte >= t).astype(int)):.3f}")
    # why not MSE: gradient wrt z for a confidently wrong example (y=1, z=-6)
    zc = -6.0; pc = float(sigmoid(zc))
    print(f"  confidently wrong (y=1, z=-6): dLogLoss/dz = {pc - 1:+.4f}   dMSE/dz = {2 * (pc - 1) * pc * (1 - pc):+.5f}")
    try:                                                 # optional cross-check
        from sklearn.linear_model import LogisticRegression as SK
        sk = SK(C=1e6, max_iter=1000).fit(Xs, ytr)
        print(f"  sklearn check: coef={np.round(np.r_[sk.intercept_, sk.coef_[0]], 3)}  (matches Newton: "
              f"{np.allclose(np.r_[sk.intercept_, sk.coef_[0]], star.theta, atol=1e-2)})")
    except ImportError:
        pass


def softmax_animation():
    """3 customer segments (e.g. plan type: basic / family / premium) from (usage, spend)."""
    X, y = make_blobs(n=240, centers=((-1.6, -1.2), (1.8, -0.8), (0.0, 2.0)), std=1.0, seed=2)
    mdl = SoftmaxRegression()
    path = mdl.fit_gd(X, y, k=3, lr=0.4, n_iters=100)
    gx, gy, P = grid_for(X, res=90, pad=0.5)
    base = np.array([to_rgb(CAT[c]) for c in range(3)])
    losses, accs = [], []
    for W in path:
        pr = mdl.predict_proba(X, W)
        losses.append(float(-np.mean(np.log(pr[np.arange(len(y)), y] + 1e-12))))
        accs.append(accuracy(y, pr.argmax(1)))

    fig, (a1, a2) = new_fig(1, 2, figsize=(10, 4.6), title="Softmax regression: one linear score per class, normalised into probabilities",
                            gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.84, wspace=0.25)
    n_it = len(path) - 1
    with Animator(here(__file__, "softmax_regression.gif"), fps=8, hold=2.5, colors=64) as anim:
        for i in frame_steps(n_it, 28):
            pr = mdl.predict_proba(P, path[i]).reshape(gx.shape + (3,))
            # blend class colours by probability, wash towards white where the model is unsure
            img = pr @ base
            conf = pr.max(-1, keepdims=True)
            img = 1 - (1 - img) * (0.25 + 0.6 * (conf - 1 / 3) / (2 / 3))
            a1.clear()
            a1.imshow(np.clip(img, 0, 1), extent=(gx.min(), gx.max(), gy.min(), gy.max()), origin="lower", aspect="auto", alpha=0.9)
            a1.contour(gx, gy, pr.argmax(-1), levels=[0.5, 1.5], colors=[INK], linewidths=1.1)
            scatter_classes(a1, X, y, s=16)
            a1.grid(False)
            a1.set(title="Class probabilities (colour mixing = uncertainty)", xlabel="usage", ylabel="spend",
                   xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
            a2.clear()
            a2.plot(range(i + 1), losses[: i + 1], color=BLUE)
            a2.scatter([i], [losses[i]], s=30, color=ORANGE, zorder=5)
            a2.set(title="Cross-entropy loss", xlabel="iteration", ylabel="loss", xlim=(0, n_it), ylim=(0, np.log(3) + 0.05))
            a2.text(n_it, np.log(3) - 0.08, "log 3 = guessing uniformly", ha="right", fontsize=8.5, color=INK2)
            caption(fig, f"iteration {i:>3}:  cross-entropy={losses[i]:.3f}   accuracy={accs[i]:.1%}")
            anim.grab(fig)
    print(f"  softmax final accuracy = {accs[-1]:.3f}, loss = {losses[-1]:.3f}")


def main():
    churn_animation()
    softmax_animation()


if __name__ == "__main__":
    main()
