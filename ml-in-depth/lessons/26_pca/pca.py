"""
Lesson 26 - Principal Component Analysis (from scratch)

Problem story: a utility records electricity load every 30 minutes (48 numbers per day).  Days differ only in a few ways
(base daily rhythm, hot-afternoon air-conditioning, weekend effect), so the 48-dimensional days live near a low-dimensional
surface.  PCA finds the few directions that carry the variance and gives a compact, interpretable description.

Implemented: PCA via covariance eigendecomposition AND via SVD (checked equal), explained variance, transform / inverse
transform, whitening, reconstruction error, plus the two equivalent views of PCA (max variance == min reconstruction error).

Animations
  pca_rotation.gif    - a line rotates through the 2-D data: variance of the shadow peaks exactly where reconstruction error is minimal
  pca_3d.gif          - a 3-D sensor cloud that is almost flat: principal axes and the PC1-PC2 plane found by PCA
  pca_components.gif  - 48-D daily load curves: scree plot, shape of each component, and reconstruction as components are added

Run:  python pca.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


class PCA:
    def __init__(self, n_components=None, whiten=False):
        self.k, self.whiten = n_components, whiten

    def fit(self, X, method="svd"):
        X = np.asarray(X, float)
        n, d = X.shape
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        if method == "eig":                                   # eigenvectors of the covariance matrix  C = Xc^T Xc / (n-1)
            C = Xc.T @ Xc / (n - 1)
            vals, vecs = np.linalg.eigh(C)
            order = np.argsort(vals)[::-1]
            vals, comps = vals[order], vecs[:, order].T
        else:                                                 # SVD: Xc = U S V^T  ->  components = rows of V^T, variance = S^2/(n-1)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            vals, comps = S ** 2 / (n - 1), Vt
        flip = np.sign(comps[np.arange(len(comps)), np.argmax(np.abs(comps), 1)])      # deterministic sign: largest entry positive
        comps = comps * flip[:, None]
        k = self.k or len(vals)
        self.components_, self.explained_variance_ = comps[:k], vals[:k]
        self.all_variance_ = vals
        self.explained_variance_ratio_ = vals[:k] / vals.sum()
        return self

    def transform(self, X):
        Z = (np.asarray(X, float) - self.mean_) @ self.components_.T
        return Z / np.sqrt(self.explained_variance_) if self.whiten else Z

    def inverse_transform(self, Z):
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        return float(np.mean(np.sum((X - self.inverse_transform(self.transform(X))) ** 2, 1)))


# ------------------------------------------------------------------ data
def sensor_pairs(n=160, seed=1):
    r = rng_(seed)
    C = np.array([[2.6, 1.7], [1.7, 1.5]])
    return r.multivariate_normal([0, 0], C, n)


def sensor_cloud(n=350, seed=2):
    """3 sensors that are driven by only 2 hidden factors (+ a little noise): the cloud is almost a plane."""
    r = rng_(seed)
    Z = r.normal(size=(n, 2)) * [2.4, 1.2]
    W = np.array([[0.8, 0.5, 0.3], [-0.3, 0.7, 0.6]])
    return Z @ W + r.normal(0, 0.12, (n, 3))


def load_curves(n=300, seed=3):
    """Daily electricity load, 48 half-hour points: base rhythm + heat-wave afternoon bump + weekend flattening + noise."""
    r = rng_(seed)
    t = np.linspace(0, 24, 48, endpoint=False)
    base = 1.0 + 0.5 * np.exp(-((t - 8) / 2.0) ** 2) + 0.7 * np.exp(-((t - 19) / 2.5) ** 2) - 0.35 * np.exp(-((t - 3.5) / 3) ** 2)
    heat = np.exp(-((t - 15) / 3.0) ** 2)
    weekend = -0.5 * np.exp(-((t - 8) / 2.2) ** 2) + 0.15 * np.sin(2 * np.pi * t / 24)
    a = r.normal(1.0, 0.1, n); h = r.normal(0, 0.45, n); w = (r.random(n) < 2 / 7).astype(float) * r.normal(1, 0.2, n)
    X = a[:, None] * base + h[:, None] * heat + w[:, None] * weekend + r.normal(0, 0.05, (n, 48))
    return t, X


# ------------------------------------------------------------------ GIF 1
def rotation_gif():
    X = sensor_pairs()
    p = PCA().fit(X)
    C = np.cov(X.T)
    total = np.trace(C)
    th_star = np.arctan2(p.components_[0][1], p.components_[0][0]) % np.pi
    th = np.linspace(0, np.pi, 181)
    U = np.c_[np.cos(th), np.sin(th)]
    var = np.einsum("ij,jk,ik->i", U, C, U)
    err = total - var                                              # mean squared distance to the line (n-1 normalisation)
    print(f"  max projected variance {var.max():.3f} at {np.degrees(th[np.argmax(var)]):.1f} deg;  min reconstruction error {err.min():.3f} at {np.degrees(th[np.argmin(err)]):.1f} deg;  PC1 angle {np.degrees(th_star):.1f} deg;  var+err = {np.ptp(var + err):.1e} (constant {total:.3f})")
    frames = np.r_[np.linspace(0, np.pi, 40, endpoint=False), th_star]
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="PCA's two equivalent views: the line with the largest shadow variance is the line with the smallest reconstruction error")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    mu = X.mean(0)
    with Animator(here(__file__, "pca_rotation.gif"), fps=8, hold=3) as anim:
        for k, a in enumerate(frames):
            u = np.array([np.cos(a), np.sin(a)])
            s = (X - mu) @ u
            proj = mu + np.outer(s, u)
            last = k == len(frames) - 1
            a1.clear()
            a1.plot(np.c_[X[:, 0], proj[:, 0]].T, np.c_[X[:, 1], proj[:, 1]].T, color=ORANGE, linewidth=0.7, alpha=0.7, zorder=1)
            a1.scatter(X[:, 0], X[:, 1], s=14, color=BLUE, zorder=3)
            a1.scatter(proj[:, 0], proj[:, 1], s=9, color=INK, zorder=4)
            a1.plot(mu[0] + np.array([-6, 6]) * u[0], mu[1] + np.array([-6, 6]) * u[1], color=GREEN if last else INK, linewidth=1.8, zorder=2)
            a1.set(title="Data, projection line, reconstruction sticks", xlabel="sensor 1 (centred)", ylabel="sensor 2 (centred)", aspect="equal", xlim=(-5.5, 5.5), ylim=(-4.5, 4.5))
            a2.clear()
            a2.plot(np.degrees(th), var, color=MUTED, linewidth=1)
            m = th <= a + 1e-9
            a2.plot(np.degrees(th[m]), var[m], color=BLUE)
            a2.scatter([np.degrees(a)], [uCu(C, u)], color=ORANGE, zorder=5, edgecolor=INK, s=40)
            a2.axvline(np.degrees(th_star), color=GREEN, linestyle="--", linewidth=1.1)
            a2.set(title="Variance of the projected points", xlabel="line angle (degrees)", ylabel="variance", xlim=(0, 180), ylim=(0, total * 1.02))
            a3.clear()
            a3.plot(np.degrees(th), err, color=MUTED, linewidth=1)
            a3.plot(np.degrees(th[m]), err[m], color=ORANGE)
            a3.scatter([np.degrees(a)], [total - uCu(C, u)], color=BLUE, zorder=5, edgecolor=INK, s=40)
            a3.axvline(np.degrees(th_star), color=GREEN, linestyle="--", linewidth=1.1)
            a3.text(np.degrees(th_star) + 4, total * 0.8, "PC1", color=GREEN, fontsize=9)
            a3.set(title="Mean squared reconstruction error", xlabel="line angle (degrees)", ylabel="error", xlim=(0, 180), ylim=(0, total * 1.02))
            caption(fig, f"angle {np.degrees(a):5.1f}°:  projected variance {uCu(C, u):.2f}  +  reconstruction error {total - uCu(C, u):.2f}  =  {total:.2f} (total variance, constant)")
            anim.grab(fig)


def uCu(C, u):
    return float(u @ C @ u)


# ------------------------------------------------------------------ GIF 2
def cloud_3d_gif():
    X = sensor_cloud()
    p = PCA().fit(X)
    print(f"  3-D cloud: explained variance ratios {np.round(p.explained_variance_ratio_, 3)};  2 components keep {p.explained_variance_ratio_[:2].sum():.1%}")
    mu = X.mean(0); V = p.components_; sd = np.sqrt(p.explained_variance_)
    gu, gv = np.meshgrid(np.linspace(-2.2, 2.2, 8), np.linspace(-2.2, 2.2, 8))
    plane = mu[:, None, None] + 1.0 * (V[0][:, None, None] * gu * sd[0] + V[1][:, None, None] * gv * sd[1])
    Z = p.transform(X)[:, :2]
    fig = plt.figure(figsize=(13, 4.8))
    fig.suptitle("Three sensors, two hidden factors: PCA finds the plane the cloud lives on", fontsize=13, fontweight="bold", x=0.02, ha="left")
    a1 = fig.add_subplot(1, 2, 1, projection="3d"); a2 = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.14, top=0.86, wspace=0.1)
    n = 36
    with Animator(here(__file__, "pca_3d.gif"), fps=8, hold=2) as anim:
        for i in range(n):
            reveal = min(1.0, i / 12)
            a1.clear()
            a1.scatter(X[:, 0], X[:, 1], X[:, 2], s=7, color=BLUE, alpha=0.7, depthshade=False)
            if reveal > 0.3:
                a1.plot_surface(plane[0], plane[1], plane[2], alpha=0.18, color=GREEN, linewidth=0.3, shade=False)
            for j, col in enumerate((ORANGE, MAGENTA, INK)):
                L = 2.2 * sd[j] * V[j] * min(1, reveal * 3 - j * 0.6 + 0.0001 if reveal * 3 - j * 0.6 > 0 else 0)
                a1.plot([mu[0], mu[0] + L[0]], [mu[1], mu[1] + L[1]], [mu[2], mu[2] + L[2]], color=col, linewidth=3)
            a1.set(title="3-D cloud, principal axes, PC1-PC2 plane", xlabel="sensor 1", ylabel="sensor 2", zlabel="sensor 3")
            a1.view_init(22, 25 + 10 * i)
            a2.clear()
            a2.scatter(Z[:, 0], Z[:, 1], s=8, color=BLUE)
            a2.set(title="The same readings in PC1-PC2 coordinates", xlabel=f"PC1 ({p.explained_variance_ratio_[0]:.0%} of variance)", ylabel=f"PC2 ({p.explained_variance_ratio_[1]:.0%})", aspect="equal")
            caption(fig, f"axes: orange PC1, magenta PC2, black PC3. Variance along them: {p.explained_variance_[0]:.2f} / {p.explained_variance_[1]:.2f} / {p.explained_variance_[2]:.3f} - the third axis is just noise, so 3 numbers per reading compress to 2")
            anim.grab(fig)


# ------------------------------------------------------------------ GIF 3
def components_gif():
    t, X = load_curves()
    Xtr, Xte = X[:220], X[220:]
    p = PCA().fit(Xtr)
    cum = np.cumsum(p.all_variance_) / p.all_variance_.sum()
    print(f"  load curves (48-D): variance explained by PC1..PC4 = {np.round(p.all_variance_[:4] / p.all_variance_.sum(), 3)};  3 components keep {cum[2]:.1%}")
    errs = []
    for k in range(0, 9):
        if k == 0:
            errs.append(float(np.mean(np.sum((Xte - Xtr.mean(0)) ** 2, 1))))
        else:
            q = PCA(k).fit(Xtr); errs.append(q.reconstruction_error(Xte))
    print("  held-out reconstruction error (sum of squares per day) for k=0..8:", np.round(errs, 3))
    day = Xte[4]
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="48 numbers per day compress into a few meaningful components")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    with Animator(here(__file__, "pca_components.gif"), fps=1.6, hold=3.5) as anim:
        for k in range(0, 9):
            a1.clear()
            a1.bar(range(1, 11), p.all_variance_[:10] / p.all_variance_.sum(), color=[BLUE if i < k else MUTED for i in range(10)], width=0.62)
            a1.plot(range(1, 11), cum[:10], color=ORANGE, marker="o", markersize=3)
            a1.set(title="Scree plot (bars) and cumulative (line)", xlabel="component", ylabel="fraction of variance", ylim=(0, 1.02), xticks=range(1, 11))
            a2.clear()
            if k > 0:
                a2.plot(t, p.components_[k - 1], color=GREEN, linewidth=2.2)
                a2.axhline(0, color=MUTED, linewidth=0.8)
                a2.set(title=f"Component {k}: loading per half-hour")
            else:
                a2.set(title="(start: only the average day)")
            a2.set(xlabel="hour of day", ylabel="loading", xlim=(0, 24), ylim=(-0.5, 0.65))
            a3.clear()
            rec = Xtr.mean(0) if k == 0 else PCA(k).fit(Xtr).inverse_transform(PCA(k).fit(Xtr).transform(day[None]))[0]
            a3.plot(t, day, color=BLUE, label="actual day")
            a3.plot(t, rec, color=ORANGE, linewidth=2, label=f"reconstruction, k={k}")
            a3.set(title="A held-out day and its reconstruction", xlabel="hour of day", ylabel="load (relative)", xlim=(0, 24)); a3.legend(loc="upper left")
            caption(fig, f"k = {k}: keeps {0 if k == 0 else cum[k - 1]:.1%} of the variance;  held-out reconstruction error {errs[k]:.3f} per day (from 48 numbers down to {k})")
            anim.grab(fig)


def main():
    rotation_gif()
    cloud_3d_gif()
    components_gif()
    # --- eig vs svd, whitening, scikit-learn check ---------------------------------------------------------------------
    t, X = load_curves()
    a, b = PCA().fit(X, "eig"), PCA().fit(X, "svd")
    print(f"  eigendecomposition vs SVD: max |variance diff| = {np.max(np.abs(a.all_variance_ - b.all_variance_)):.2e}, max |component diff| (sign-fixed) = {np.max(np.abs(a.components_[:5] - b.components_[:5])):.2e}")
    w = PCA(5, whiten=True).fit(X); Zw = w.transform(X)
    print(f"  whitening: covariance of the 5 whitened scores = identity? max |C - I| = {np.max(np.abs(np.cov(Zw.T) - np.eye(5))):.2e}")
    try:
        from sklearn.decomposition import PCA as SK
        s = SK(5).fit(X)
        print(f"  [sklearn] explained variance ratio max diff {np.max(np.abs(s.explained_variance_ratio_ - b.all_variance_[:5] / b.all_variance_.sum())):.2e}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
