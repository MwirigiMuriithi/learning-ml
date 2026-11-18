"""
Lesson 11 - Linear & Quadratic Discriminant Analysis (from scratch)

Problem story: a plant has two machine types.  Each machine is described by two sensor
readings (vibration, temperature).  We want to (a) find the single direction along which the
two types are best separated (Fisher's discriminant) and (b) classify new machines with a
generative model: every class is a Gaussian, Bayes' rule does the rest.

Implemented (NumPy only):
  * LDA  - class means + ONE pooled covariance  -> linear boundary, Fisher direction w = Sw^-1 (mu1-mu0)
  * QDA  - one covariance PER class              -> quadratic boundary
  * RDA-style shrinkage of the covariance (reg parameter)

Animated:
  lda_fisher.gif    - a projection direction rotates; histograms of the projected classes and the
                      Fisher ratio J(w) = (between-class gap)^2 / (within-class spread) trace out the curve
                      whose maximum is exactly the closed-form LDA direction.
  lda_vs_qda.gif    - one class gets a larger and larger covariance: the LDA boundary stays a line and
                      its accuracy falls, the QDA boundary bends and keeps up.

Run:  python lda_qda.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ----------------------------------------------------------------------- models
class QDA:
    """Gaussian class-conditional densities, one covariance per class.
    log-score_k(x) = log pi_k - 1/2 log|S_k| - 1/2 (x-mu_k)^T S_k^-1 (x-mu_k)   (classify by the largest)"""

    def __init__(self, reg=0.0):
        self.reg = reg                      # shrink each covariance toward a scaled identity (RDA flavour)

    def fit(self, X, y):
        self.classes = np.unique(y)
        d = X.shape[1]
        self.pi, self.mu, self.S = [], [], []
        for c in self.classes:
            Xc = X[y == c]
            self.pi.append(len(Xc) / len(X))
            self.mu.append(Xc.mean(0))
            S = np.cov(Xc.T, bias=False)
            self.S.append((1 - self.reg) * S + self.reg * np.trace(S) / d * np.eye(d))
        return self

    def _cov(self, k):
        return self.S[k]

    def scores(self, X):
        out = []
        for k in range(len(self.classes)):
            S = self._cov(k)
            D = X - self.mu[k]
            maha = np.einsum("ij,jk,ik->i", D, np.linalg.inv(S), D)         # (x-mu)^T S^-1 (x-mu)
            out.append(np.log(self.pi[k]) - 0.5 * np.linalg.slogdet(S)[1] - 0.5 * maha)
        return np.array(out).T

    def predict(self, X):
        return self.classes[np.argmax(self.scores(X), 1)]

    def predict_proba(self, X):
        return softmax(self.scores(X), axis=1)


class LDA(QDA):
    """Same model but ALL classes share the pooled within-class covariance Sw.
    Then the quadratic terms cancel and the score is linear:  w_k . x + b_k,  w_k = Sw^-1 mu_k."""

    def fit(self, X, y):
        super().fit(X, y)
        n, K = len(X), len(self.classes)
        Sw = sum((np.sum(y == c) - 1) * self.S[k] for k, c in enumerate(self.classes)) / (n - K)
        d = X.shape[1]
        self.Sw = (1 - self.reg) * Sw + self.reg * np.trace(Sw) / d * np.eye(d)
        return self

    def _cov(self, k):
        return self.Sw

    def fisher_direction(self):
        """Two-class Fisher discriminant: w = Sw^-1 (mu1 - mu0)."""
        w = np.linalg.solve(self.Sw, self.mu[1] - self.mu[0])
        return w / np.linalg.norm(w)

    def transform(self, X, n_components=1):
        """Project on the top eigenvectors of Sw^-1 Sb (for K classes there are up to K-1 of them)."""
        mu = np.mean(np.array(self.mu), 0)
        Sb = sum(self.pi[k] * np.outer(self.mu[k] - mu, self.mu[k] - mu) for k in range(len(self.classes)))
        vals, vecs = np.linalg.eig(np.linalg.solve(self.Sw, Sb))
        order = np.argsort(-vals.real)[:n_components]
        return (X - mu) @ vecs.real[:, order]


def fisher_J(X, y, w):
    """Fisher criterion for a 1-D projection w:  (m1 - m0)^2 / (s0^2 + s1^2)."""
    p = X @ w
    a, b = p[y == 0], p[y == 1]
    return (b.mean() - a.mean()) ** 2 / (a.var() + b.var())


# ------------------------------------------------------------------------- data
def machine_types(n=240, seed=2):
    """Two machine types sharing a correlated covariance (sensors move together)."""
    r = rng_(seed)
    C = np.array([[1.3, 1.0], [1.0, 1.2]])
    L = np.linalg.cholesky(C)
    y = r.integers(0, 2, n)
    mu = np.where(y[:, None] == 1, [2.2, 0.4], [0.0, 0.0])
    return mu + r.normal(size=(n, 2)) @ L.T, y


# --------------------------------------------------------------------- GIF 1
def fisher_animation():
    X, y = machine_types()
    lda = LDA().fit(X, y)
    w_star = lda.fisher_direction()
    th_star = np.arctan2(w_star[1], w_star[0]) % np.pi
    fine = np.linspace(0, np.pi, 361)
    Jfine = np.array([fisher_J(X, y, np.array([np.cos(t), np.sin(t)])) for t in fine])
    th_best_scan = fine[np.argmax(Jfine)]
    print(f"  Fisher direction (closed form) angle = {np.degrees(th_star):.1f} deg;  brute-force scan max at {np.degrees(th_best_scan):.1f} deg")
    print(f"  J at closed-form direction = {fisher_J(X, y, w_star):.3f}  (max of scan {Jfine.max():.3f});  J along raw mean-difference direction = "
          f"{fisher_J(X, y, (lda.mu[1] - lda.mu[0]) / np.linalg.norm(lda.mu[1] - lda.mu[0])):.3f}")

    ang_frames = np.r_[np.linspace(0, np.pi, 34), th_star]            # sweep, then park on the optimum
    center = X.mean(0)
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Fisher's discriminant: rotate a line until the two projected classes overlap least")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.27)
    with Animator(here(__file__, "lda_fisher.gif"), fps=7, hold=3) as anim:
        for k, th in enumerate(ang_frames):
            w = np.array([np.cos(th), np.sin(th)])
            p = (X - center) @ w
            a1.clear()
            scatter_classes(a1, X, y)
            t = np.array([-5.5, 5.5])
            a1.plot(center[0] + t * w[0], center[1] + t * w[1], color=INK, linewidth=1.4, zorder=2)
            proj = center + np.outer(p, w)
            a1.plot(np.c_[X[:, 0], proj[:, 0]].T, np.c_[X[:, 1], proj[:, 1]].T, color=MUTED, linewidth=0.3, alpha=0.5, zorder=1)
            a1.scatter(proj[:, 0], proj[:, 1], s=8, c=[CAT[int(c)] for c in y], zorder=4, edgecolor="none")
            a1.set(title="Data and the projection line", xlabel="vibration", ylabel="temperature", aspect="equal")
            a1.set_xlim(-4, 6.5); a1.set_ylim(-4.5, 5.5)
            a2.clear()
            bins = np.linspace(-6, 6, 31)
            a2.hist(p[y == 0], bins, color=BLUE, alpha=0.65, label="type A")
            a2.hist(p[y == 1], bins, color=ORANGE, alpha=0.65, label="type B")
            a2.set(title="Projected values (1-D)", xlabel="position along the line", ylabel="count", ylim=(0, 45))
            a2.legend(loc="upper right")
            a3.clear()
            a3.plot(np.degrees(fine), Jfine, color=MUTED, linewidth=1)
            done = fine <= th + 1e-9
            if k < len(ang_frames) - 1:
                a3.plot(np.degrees(fine[done]), Jfine[done], color=BLUE)
            a3.scatter([np.degrees(th)], [fisher_J(X, y, w)], s=48, color=ORANGE, zorder=5, edgecolor=INK)
            a3.axvline(np.degrees(th_star), color=GREEN, linestyle="--", linewidth=1.2)
            a3.text(np.degrees(th_star) + 3, Jfine.max() * 0.35, "closed-form LDA\ndirection", color=GREEN, fontsize=8.5)
            a3.set(title="Fisher ratio J(w)", xlabel="direction angle (degrees)", ylabel="(gap)² / (spread)", xlim=(0, 180))
            caption(fig, f"angle {np.degrees(th):5.1f}°:  J = {fisher_J(X, y, w):.3f}   (J = squared distance between class means / sum of within-class variances, along the line)")
            anim.grab(fig)
    return lda


# --------------------------------------------------------------------- GIF 2
def lda_vs_qda_animation():
    r = rng_(5)
    Z0, Z1 = r.normal(size=(150, 2)), r.normal(size=(150, 2))
    T0, T1 = rng_(6).normal(size=(2000, 2)), rng_(7).normal(size=(2000, 2))
    C0 = np.array([[0.6, 0.0], [0.0, 0.6]])
    C1 = np.array([[1.0, 0.7], [0.7, 1.0]])
    mu0, mu1 = np.array([0.0, 0.0]), np.array([1.6, 0.0])
    L0 = np.linalg.cholesky(C0)

    def make(Z, scale, which):
        if which == 0:
            return mu0 + Z @ L0.T
        return mu1 + Z @ np.linalg.cholesky(scale * C1).T

    scales = np.linspace(0.3, 5.0, 28)
    hist = {"lda": [], "qda": []}
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="When class covariances differ, only QDA can bend its boundary")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.27)
    pad = (-3.2, 6.0, -4.5, 4.5)
    gx, gy = np.meshgrid(np.linspace(pad[0], pad[1], 140), np.linspace(pad[2], pad[3], 140))
    P = np.c_[gx.ravel(), gy.ravel()]
    with Animator(here(__file__, "lda_vs_qda.gif"), fps=7, hold=3) as anim:
        for i, s in enumerate(scales):
            X = np.vstack([make(Z0, s, 0), make(Z1, s, 1)]); y = np.r_[np.zeros(150, int), np.ones(150, int)]
            Xt = np.vstack([make(T0, s, 0), make(T1, s, 1)]); yt = np.r_[np.zeros(2000, int), np.ones(2000, int)]
            lda, qda = LDA().fit(X, y), QDA().fit(X, y)
            acc_l, acc_q = accuracy(yt, lda.predict(Xt)), accuracy(yt, qda.predict(Xt))
            hist["lda"].append(acc_l); hist["qda"].append(acc_q)
            for ax, m, name in ((a1, lda, "LDA: shared covariance, linear boundary"), (a2, qda, "QDA: own covariance, curved boundary")):
                ax.clear()
                z = m.predict(P).reshape(gx.shape)
                ax.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], colors=[BLUE, ORANGE], alpha=0.2)
                ax.contour(gx, gy, z, levels=[0.5], colors=[INK], linewidths=1.6)
                scatter_classes(ax, X, y, s=14)
                for k in range(2):                       # fitted 2-sigma covariance ellipses
                    S = m._cov(k); ev, V = np.linalg.eigh(S); t_ = np.linspace(0, 2 * np.pi, 80)
                    E = m.mu[k][:, None] + V @ (2 * np.sqrt(ev)[:, None] * np.vstack([np.cos(t_), np.sin(t_)]))
                    ax.plot(E[0], E[1], color=CAT[k], linewidth=1.2)
                ax.set(title=name, xlabel="vibration", ylabel="temperature"); ax.set_xlim(pad[0], pad[1]); ax.set_ylim(pad[2], pad[3])
            a3.clear()
            a3.plot(scales[: i + 1], hist["lda"], color=BLUE); a3.plot(scales[: i + 1], hist["qda"], color=ORANGE)
            a3.scatter([s, s], [acc_l, acc_q], color=[BLUE, ORANGE], zorder=5, s=30)
            a3.text(scales[-1], 0.995, "QDA", color=ORANGE, ha="right", fontsize=9); a3.text(scales[-1], 0.80, "LDA", color=BLUE, ha="right", fontsize=9)
            a3.set(title="Accuracy on 4000 fresh points", xlabel="class-B covariance scale (A is fixed)", ylabel="accuracy", xlim=(0.3, 5.0), ylim=(0.7, 1.0))
            caption(fig, f"class B covariance x{s:.1f}:  LDA accuracy {acc_l:.3f}   QDA accuracy {acc_q:.3f}   (class A covariance fixed at 0.6 I)")
            anim.grab(fig)
    print(f"  at covariance scale {scales[0]:.1f}: LDA {hist['lda'][0]:.3f}  QDA {hist['qda'][0]:.3f};  at {scales[-1]:.1f}: LDA {hist['lda'][-1]:.3f}  QDA {hist['qda'][-1]:.3f}")


def main():
    lda = fisher_animation()
    lda_vs_qda_animation()
    # multiclass LDA as a dimensionality reducer + optional scikit-learn cross-check
    r = rng_(0)
    Xm = np.vstack([r.normal(m, 1.0, (100, 4)) for m in ([0, 0, 0, 0], [3, 1, 0, 0], [0, 3, 1, 0])]); ym = np.repeat([0, 1, 2], 100)
    Zm = LDA().fit(Xm, ym).transform(Xm, 2)
    print(f"  3-class LDA projection to 2 components: shape {Zm.shape}")
    X, y = machine_types(); Xt, yt = machine_types(2000, seed=9)
    ours = LDA().fit(X, y); qd = QDA().fit(X, y); print(f"  test accuracy LDA {accuracy(yt, ours.predict(Xt)):.3f}  QDA {accuracy(yt, qd.predict(Xt)):.3f}")
    for reg in (0.0, 0.3, 0.9):
        print(f"  RDA-style shrinkage reg={reg}: QDA test acc {accuracy(yt, QDA(reg).fit(X[:30], y[:30]).predict(Xt)):.3f} (30 training points only)")
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as SL, QuadraticDiscriminantAnalysis as SQ
        print(f"  [sklearn] LDA agreement {np.mean(SL().fit(X, y).predict(Xt) == ours.predict(Xt)):.4f}   QDA agreement {np.mean(SQ().fit(X, y).predict(Xt) == qd.predict(Xt)):.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
