"""
Lesson 13 - Support Vector Machines (from scratch)

Problem story: a lab classifies sensor readings as "normal" (-1) or "faulty" (+1).  An SVM draws the boundary with the
widest possible street between the classes; only the readings touching the street (support vectors) matter.

Implemented (NumPy only):
  * LinearSVM_Primal - soft-margin SVM  min  1/2 |w|^2 + C * sum hinge  by sub-gradient descent (no libraries)
  * SVM (dual)       - kernel SVM solved with Platt's simplified SMO; kernels: linear, polynomial, RBF
  * hinge loss, margins, support vectors, decision_function, one-vs-rest multiclass wrapper
  * epsilon-SVR formulation is explained in lesson.md (not implemented)

Animated (four GIFs):
  svm_linear_training.gif  - primal training: line + margin lines settle, support vectors (ringed) emerge, objective falls
  svm_C.gif                - the soft-margin knob C: wide tolerant street -> narrow strict street
  svm_lift.gif             - the kernel idea: circles are not linearly separable, lift (x,y) -> (x,y,x^2+y^2) and a plane works
  svm_rbf_gamma.gif        - RBF kernel width gamma: smooth -> overfit islands, with accuracy and support-vector count

Run:  python svm.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# --------------------------------------------------------------------- kernels
def kernel_matrix(A, B, kind="rbf", gamma=1.0, degree=3, coef0=1.0):
    if kind == "linear":
        return A @ B.T
    if kind == "poly":
        return (gamma * (A @ B.T) + coef0) ** degree
    if kind == "rbf":     # exp(-gamma |a-b|^2), using |a-b|^2 = |a|^2 + |b|^2 - 2 a.b
        d2 = np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d2, 0))
    raise ValueError(kind)


# --------------------------------------------------------------------- primal
class LinearSVM_Primal:
    """minimise  J(w,b) = 1/2 |w|^2 + C * sum_i max(0, 1 - y_i (w.x_i + b))   by sub-gradient descent."""

    def __init__(self, C=1.0):
        self.C = C

    def objective(self, X, y, w, b):
        m = 1 - y * (X @ w + b)
        return 0.5 * w @ w + self.C * np.sum(np.maximum(0, m))

    def fit(self, X, y, n_iters=300, lr0=0.02, record_every=1):
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        self.path = []
        for t in range(n_iters + 1):
            if t % record_every == 0:
                self.path.append((w.copy(), b, self.objective(X, y, w, b)))
            viol = y * (X @ w + b) < 1                    # points inside the street or on the wrong side
            gw = w - self.C * (y[viol, None] * X[viol]).sum(0)    # sub-gradient of the objective
            gb = -self.C * y[viol].sum()
            lr = lr0 / (1 + 0.02 * t) / max(1.0, self.C * 0.1)    # decaying step; smaller if C is large
            w, b = w - lr * gw, b - lr * gb
        self.w, self.b = w, b
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, 1, -1)


# ------------------------------------------------------------------------ dual
class SVM:
    """Kernel SVM, dual form:  max  sum a_i - 1/2 sum_ij a_i a_j y_i y_j K_ij   s.t.  0 <= a_i <= C,  sum a_i y_i = 0.
    Solved by (simplified) Sequential Minimal Optimisation: optimise TWO multipliers at a time analytically.
    Prediction:  f(x) = sum_i a_i y_i K(x_i, x) + b ;  support vectors are the points with a_i > 0."""

    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, degree=3, tol=1e-3, max_passes=6, max_sweeps=300, seed=0):
        self.C, self.kernel, self.gamma, self.degree = C, kernel, gamma, degree
        self.tol, self.max_passes, self.max_sweeps, self.seed = tol, max_passes, max_sweeps, seed

    def _K(self, A, B):
        return kernel_matrix(A, B, self.kernel, self.gamma, self.degree)

    def fit(self, X, y):
        r = rng_(self.seed)
        n = len(y)
        K = self._K(X, X)
        a, b = np.zeros(n), 0.0
        passes = sweeps = 0
        while passes < self.max_passes and sweeps < self.max_sweeps:
            changed = 0
            for i in range(n):
                Ei = (a * y) @ K[:, i] + b - y[i]                        # prediction error on point i
                if (y[i] * Ei < -self.tol and a[i] < self.C) or (y[i] * Ei > self.tol and a[i] > 0):   # KKT violated
                    j = int(r.integers(n - 1)); j += j >= i              # random partner j != i
                    Ej = (a * y) @ K[:, j] + b - y[j]
                    ai, aj = a[i], a[j]
                    if y[i] != y[j]:
                        L, H = max(0, aj - ai), min(self.C, self.C + aj - ai)
                    else:
                        L, H = max(0, ai + aj - self.C), min(self.C, ai + aj)
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]                # curvature along the constraint line (<= 0)
                    if eta >= 0:
                        continue
                    a[j] = np.clip(aj - y[j] * (Ei - Ej) / eta, L, H)    # closed-form 1-D optimum, clipped to the box
                    if abs(a[j] - aj) < 1e-5:
                        continue
                    a[i] = ai + y[i] * y[j] * (aj - a[j])                # keep sum a_i y_i fixed
                    b1 = b - Ei - y[i] * (a[i] - ai) * K[i, i] - y[j] * (a[j] - aj) * K[i, j]
                    b2 = b - Ej - y[i] * (a[i] - ai) * K[i, j] - y[j] * (a[j] - aj) * K[j, j]
                    b = b1 if 0 < a[i] < self.C else b2 if 0 < a[j] < self.C else (b1 + b2) / 2
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
            sweeps += 1
        self.alpha, self.b, self.X, self.y = a, b, X, y
        self.sv = a > 1e-6
        return self

    def decision_function(self, X):
        s = self.sv
        return self._K(X, self.X[s]) @ (self.alpha[s] * self.y[s]) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) > 0, 1, -1)

    def dual_objective(self):
        K = self._K(self.X, self.X)
        ay = self.alpha * self.y
        return self.alpha.sum() - 0.5 * ay @ K @ ay

    @property
    def w(self):                       # only meaningful for the linear kernel
        return (self.alpha * self.y) @ self.X


class OneVsRestSVM:
    """Multiclass by training one binary SVM per class; predict the class with the largest decision value."""

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = [SVM(**self.kw).fit(X, np.where(y == c, 1, -1)) for c in self.classes]
        return self

    def predict(self, X):
        return self.classes[np.argmax([m.decision_function(X) for m in self.models], 0)]


# ------------------------------------------------------------------------ data
def sensor_data(n=120, seed=1, noise=1.15):
    X, y = make_linear_classes(n, noise=noise, seed=seed)
    return X, np.where(y == 1, 1, -1)


def street(ax, w, b, xlim, ylim, color=INK):
    xs = np.array(xlim)
    for off, ls, lw in ((0, "-", 2.0), (1, "--", 1.1), (-1, "--", 1.1)):
        if abs(w[1]) > 1e-9:
            ax.plot(xs, -(w[0] * xs + b - off) / w[1], color=color, linestyle=ls, linewidth=lw, zorder=2)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)


def sv_rings(ax, X, mask):
    ax.scatter(X[mask, 0], X[mask, 1], s=95, facecolor="none", edgecolor=INK, linewidth=1.2, zorder=5)


# ---------------------------------------------------------------------- GIF 1
def linear_training_gif():
    X, y = sensor_data()
    m = LinearSVM_Primal(C=1.0).fit(X, y, n_iters=400)
    ref = SVM(C=1.0, kernel="linear").fit(X, y)
    wref, bref = ref.w, ref.b
    Jref = 0.5 * wref @ wref + ref.C * np.sum(np.maximum(0, 1 - y * (X @ wref + bref)))
    print(f"  primal sub-gradient: J={m.path[-1][2]:.2f}, accuracy {accuracy(y, m.predict(X)):.3f}  |  SMO dual (linear): J={Jref:.2f}, "
          f"{ref.sv.sum()} support vectors, accuracy {accuracy(y, ref.predict(X)):.3f}  (dual objective {ref.dual_objective():.2f} ~ primal optimum)")
    xlim, ylim = (-5, 5), (-4.5, 4.5)
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Linear SVM: push the street as wide as possible while few points sit inside it")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    steps = frame_steps(len(m.path) - 1, 32)
    Js = [p[2] for p in m.path]
    with Animator(here(__file__, "svm_linear_training.gif"), fps=8, hold=3) as anim:
        for t in steps:
            w, b, J = m.path[t]
            marg = y * (X @ w + b)
            a1.clear()
            scatter_classes(a1, X, (y > 0).astype(int), s=22)
            if np.linalg.norm(w) > 1e-6:
                street(a1, w, b, xlim, ylim)
            a1.set_xlim(*xlim); a1.set_ylim(*ylim)
            sv_rings(a1, X, marg <= 1 + 1e-9)
            a1.set(title="Street |w·x+b|<1 (ringed = inside/wrong side)", xlabel="sensor 1", ylabel="sensor 2")
            a1.text(0.02, 0.03, "blue = normal, orange = faulty", transform=a1.transAxes, fontsize=8, color=INK2)
            a2.clear()
            width = [2 / max(np.linalg.norm(p[0]), 1e-9) for p in m.path[: t + 1]]
            a2.plot(range(t + 1), np.minimum(width, 12), color=BLUE)
            a2.set(title="Street width 2/‖w‖", xlabel="iteration", ylabel="width", xlim=(0, len(m.path) - 1), ylim=(0, 12))
            a3.clear()
            a3.plot(range(t + 1), Js[: t + 1], color=BLUE); a3.axhline(Jref, color=GREEN, linestyle="--", linewidth=1.2)
            a3.text(len(m.path) - 2, Jref * 1.02, "dual (SMO) optimum", color=GREEN, ha="right", va="bottom", fontsize=8.5)
            a3.set(title="Objective ½‖w‖² + C·Σ hinge", xlabel="iteration", ylabel="J", xlim=(0, len(m.path) - 1), ylim=(Jref * 0.8, max(Js) * 1.05), yscale="log")
            caption(fig, f"iter {t:>3}: width {('inf' if np.linalg.norm(w) < 1e-9 else format(2 / np.linalg.norm(w), '5.2f'))}  hinge-active points {int(np.sum(marg <= 1))}  J={J:7.2f}  accuracy {np.mean(np.sign(X @ w + b) == y):.2f}")
            anim.grab(fig)


# ---------------------------------------------------------------------- GIF 2
def C_gif():
    X, y = sensor_data(n=140, seed=7, noise=1.5)
    Xt, yt = sensor_data(n=1500, seed=17, noise=1.5)
    Cs = np.geomspace(0.003, 50, 26)
    ms = [SVM(C=c, kernel="linear").fit(X, y) for c in Cs]
    tr = [accuracy(y, m.predict(X)) for m in ms]; te = [accuracy(yt, m.predict(Xt)) for m in ms]
    nsv = [int(m.sv.sum()) for m in ms]
    print(f"  C sweep: C={Cs[0]:.3f}: width {2 / np.linalg.norm(ms[0].w):.2f}, SVs {nsv[0]}, test acc {te[0]:.3f} | C={Cs[-1]:.0f}: width {2 / np.linalg.norm(ms[-1].w):.2f}, SVs {nsv[-1]}, test acc {te[-1]:.3f} | best test acc {max(te):.3f} at C={Cs[int(np.argmax(te))]:.2f}")
    xlim, ylim = (-5.5, 5.5), (-5, 5)
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="The soft-margin knob C: tolerate mistakes (small C) or punish them (large C)")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    with Animator(here(__file__, "svm_C.gif"), fps=6, hold=3) as anim:
        for i, (c, m) in enumerate(zip(Cs, ms)):
            a1.clear()
            scatter_classes(a1, X, (y > 0).astype(int), s=22)
            street(a1, m.w, m.b, xlim, ylim)
            sv_rings(a1, X, m.sv)
            a1.set(title="Street and support vectors (ringed)", xlabel="sensor 1", ylabel="sensor 2")
            a2.clear()
            a2.plot(Cs[: i + 1], nsv[: i + 1], color=BLUE); a2.scatter([c], [nsv[i]], color=ORANGE, zorder=5)
            a2.set(title="Number of support vectors", xlabel="C (log scale)", ylabel="count", xscale="log", xlim=(Cs[0], Cs[-1]), ylim=(0, len(y) + 5))
            a3.clear()
            a3.plot(Cs[: i + 1], tr[: i + 1], color=BLUE); a3.plot(Cs[: i + 1], te[: i + 1], color=ORANGE)
            a3.text(Cs[-1], 0.92, "train", color=BLUE, ha="right", fontsize=8.5); a3.text(Cs[-1], 0.84, "test", color=ORANGE, ha="right", fontsize=8.5)
            a3.set(title="Accuracy", xlabel="C (log scale)", ylabel="accuracy", xscale="log", xlim=(Cs[0], Cs[-1]), ylim=(0.6, 1.0))
            caption(fig, f"C = {c:8.3f}: street width {2 / np.linalg.norm(m.w):5.2f}, {nsv[i]} support vectors, train {tr[i]:.3f}, test {te[i]:.3f}   (small C = wide street, many violations allowed)")
            anim.grab(fig)


# ---------------------------------------------------------------------- GIF 3
def lift_gif():
    X, y01 = make_circles(150, noise=0.07, factor=0.45, seed=3)
    y = np.where(y01 == 1, 1, -1)
    Z = np.c_[X, (X ** 2).sum(1)]                              # explicit feature map phi(x,y) = (x, y, x^2 + y^2)
    lin2d = SVM(C=10, kernel="linear").fit(X, y)
    lin3d = SVM(C=10, kernel="linear").fit(Z, y)
    w3, b3 = lin3d.w, lin3d.b
    print(f"  circles: linear SVM in 2-D accuracy {accuracy(y, lin2d.predict(X)):.3f};  after lifting to (x, y, x^2+y^2): {accuracy(y, lin3d.predict(Z)):.3f}")
    gx, gy = np.meshgrid(np.linspace(-1.4, 1.4, 22), np.linspace(-1.4, 1.4, 22))
    plane = -(w3[0] * gx + w3[1] * gy + b3) / w3[2]
    fig = plt.figure(figsize=(13, 4.8))
    fig.suptitle("The kernel idea: no line separates the circles, but lifting each point to z = x² + y² makes a flat plane do the job",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    a1 = fig.add_subplot(1, 2, 1, projection="3d"); a2 = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.14, top=0.86, wspace=0.12)
    n_lift, n_rot = 14, 16
    sched = [(k / n_lift, -60, False) for k in range(n_lift + 1)] + [(1.0, -60 + 120 * k / n_rot, True) for k in range(1, n_rot + 1)]
    gxx, gyy = np.meshgrid(np.linspace(-1.5, 1.5, 120), np.linspace(-1.5, 1.5, 120))
    zz = w3[2] * (gxx ** 2 + gyy ** 2) + w3[0] * gxx + w3[1] * gyy + b3
    with Animator(here(__file__, "svm_lift.gif"), fps=7, hold=3) as anim:
        for t, az, show_plane in sched:
            a1.clear()
            for c, col in ((-1, BLUE), (1, ORANGE)):
                m = y == c
                a1.scatter(X[m, 0], X[m, 1], t * Z[m, 2], s=14, color=col, depthshade=False)
            if show_plane:
                a1.plot_surface(gx, gy, plane, alpha=0.28, color=GREEN, linewidth=0, shade=False)
            a1.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), zlim=(0, 1.2), title="Feature space (x, y, z = t·(x²+y²))", xlabel="x", ylabel="y", zlabel="z")
            a1.view_init(18, az)
            a2.clear()
            if show_plane:
                a2.contourf(gxx, gyy, zz, levels=[-1e9, 0, 1e9], colors=[BLUE, ORANGE], alpha=0.2)
                a2.contour(gxx, gyy, zz, levels=[0], colors=[INK], linewidths=2)
            scatter_classes(a2, X, (y > 0).astype(int), s=18)
            a2.set(title="Back in the original 2-D plane: the plane becomes a circle" if show_plane else "Original 2-D plane: no straight line separates the classes", xlabel="x", ylabel="y", aspect="equal", xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
            caption(fig, ("lifting: points rise to height t·(x²+y²), t = %.2f" % t) if not show_plane else "the green plane z = c cuts the lifted data cleanly; its intersection with the paraboloid projects to the circle on the right")
            anim.grab(fig)


# ---------------------------------------------------------------------- GIF 4
def rbf_gif():
    X, y01 = make_moons(220, noise=0.28, seed=2); y = np.where(y01 == 1, 1, -1)
    Xt, yt01 = make_moons(1200, noise=0.28, seed=22); yt = np.where(yt01 == 1, 1, -1)
    gammas = np.geomspace(0.03, 150, 26)
    ms = [SVM(C=1.0, kernel="rbf", gamma=g, max_sweeps=150).fit(X, y) for g in gammas]
    tr = [accuracy(y, m.predict(X)) for m in ms]; te = [accuracy(yt, m.predict(Xt)) for m in ms]
    nsv = [int(m.sv.sum()) for m in ms]
    ib = int(np.argmax(te))
    print(f"  RBF gamma sweep (C=1): best test acc {te[ib]:.3f} at gamma={gammas[ib]:.2f};  gamma={gammas[0]:.2f}: train {tr[0]:.2f} test {te[0]:.2f};  gamma={gammas[-1]:.0f}: train {tr[-1]:.2f} test {te[-1]:.2f}, SVs {nsv[-1]}")
    gx, gy, P = grid_for(X, res=130, pad=0.5)
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="RBF kernel width γ: from a nearly straight boundary to islands around every point")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    with Animator(here(__file__, "svm_rbf_gamma.gif"), fps=6, hold=3) as anim:
        for i, (g, m) in enumerate(zip(gammas, ms)):
            f = m.decision_function(P).reshape(gx.shape)
            a1.clear()
            a1.contourf(gx, gy, f, levels=[-1e9, 0, 1e9], colors=[BLUE, ORANGE], alpha=0.2)
            a1.contour(gx, gy, f, levels=[-1, 0, 1], colors=[INK], linewidths=[1, 2, 1], linestyles=["--", "-", "--"])
            scatter_classes(a1, X, (y > 0).astype(int), s=18)
            sv_rings(a1, X, m.sv)
            a1.set(title="Decision regions (ringed = support vectors)", xlabel="x₁", ylabel="x₂")
            a2.clear()
            a2.plot(gammas[: i + 1], nsv[: i + 1], color=BLUE); a2.scatter([g], [nsv[i]], color=ORANGE, zorder=5)
            a2.set(title="Number of support vectors", xlabel="γ (log scale)", ylabel="count", xscale="log", xlim=(gammas[0], gammas[-1]), ylim=(0, len(y) + 5))
            a3.clear()
            a3.plot(gammas[: i + 1], tr[: i + 1], color=BLUE); a3.plot(gammas[: i + 1], te[: i + 1], color=ORANGE)
            a3.text(gammas[-1], 1.0, "train", color=BLUE, ha="right", fontsize=8.5); a3.text(gammas[-1], te[-1] - 0.03, "test", color=ORANGE, ha="right", fontsize=8.5, va="top")
            a3.set(title="Accuracy", xlabel="γ (log scale)", ylabel="accuracy", xscale="log", xlim=(gammas[0], gammas[-1]), ylim=(0.6, 1.02))
            caption(fig, f"γ = {g:8.3f}: train {tr[i]:.3f}, test {te[i]:.3f}, {nsv[i]} support vectors   (large γ = each point only influences its immediate surroundings)")
            anim.grab(fig)


def main():
    linear_training_gif()
    C_gif()
    lift_gif()
    rbf_gif()
    # polynomial kernel and multiclass sanity checks
    Xm, ym = make_blobs(240, centers=((-2, -2), (2, 2), (-2, 3)), std=0.9, seed=3)
    om = OneVsRestSVM(C=1.0, kernel="rbf", gamma=0.5).fit(Xm, ym)
    print(f"  one-vs-rest multiclass SVM (3 blobs): training accuracy {accuracy(ym, om.predict(Xm)):.3f}")
    X, y = make_circles(200, noise=0.1, seed=5); y = np.where(y == 1, 1, -1)
    for kind, kw in (("linear", {}), ("poly", dict(degree=2, gamma=1.0)), ("rbf", dict(gamma=2.0))):
        print(f"  circles, kernel={kind:6s}: training accuracy {accuracy(y, SVM(C=5, kernel=kind, **kw).fit(X, y).predict(X)):.3f}")
    try:
        from sklearn.svm import SVC
        Xr, yr01 = make_moons(200, noise=0.25, seed=9); yr = np.where(yr01 == 1, 1, -1)
        ours = SVM(C=1.0, kernel="rbf", gamma=1.5, tol=1e-4, max_passes=20, max_sweeps=500).fit(Xr, yr)
        sk = SVC(C=1.0, kernel="rbf", gamma=1.5).fit(Xr, yr)
        Xg = np.random.default_rng(1).uniform(-2, 3, (2000, 2))
        print(f"  [sklearn SVC] prediction agreement on 2000 random points: {np.mean(ours.predict(Xg) == sk.predict(Xg)):.3f};  support vectors ours {ours.sv.sum()} vs sklearn {len(sk.support_)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
