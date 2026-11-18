"""
Lesson 05 - Regularisation: Ridge, Lasso and Elastic Net (from scratch)

Problem: predict a house's price ($1000s) from 10 features measured on only 30
houses.  Six features carry signal, but three of them (bedrooms, bathrooms, area)
are strongly correlated with each other, and four more are pure noise (street
number, ...).  Plain least squares (OLS) happily gives weight to everything and
its coefficients are unstable.  A *penalty* on the weights fixes that:

    objective(w) = 1/(2n) ||y - Xw||^2  +  lam * [ a*||w||_1 + (1-a)/2*||w||_2^2 ]

    a = 0  -> Ridge (closed form)      a = 1 -> Lasso (coordinate descent)
    0<a<1  -> Elastic Net

What is animated
  regularization_paths.gif    left/middle: coefficient paths of Ridge vs Lasso as lam
                              is swept from huge to tiny; right: 5-fold CV error
                              curves that build up with the sweep.
  regularization_geometry.gif 2-D picture with two correlated features: loss ellipses
                              touch the L2 circle (ridge) or the L1 diamond (lasso)
                              while the budget grows; right: coefficient vs budget.

Run:  python regularization.py     (writes both GIFs, ~30 s)
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403

FEATURES = ["area", "bedrooms", "bathrooms", "age", "dist_center", "school_score",
            "street_no", "lot_shape", "owner_age", "garden_dir"]
N_SIGNAL = 6


# ---------------------------------------------------------------- the data
def make_housing(n=50, seed=0):
    """10 features (already ~unit variance).  bedrooms and bathrooms are correlated
    with area (rho ~ 0.9 / 0.8).  The last 4 features have NO effect on price.
    price ($1000s) = 300 + 60 area + 10 bedrooms + 15 bathrooms - 25 age
                         - 30 distance + 25 school + noise(40)."""
    r = rng_(seed)
    area = r.normal(size=n)
    bed = 0.9 * area + np.sqrt(1 - 0.81) * r.normal(size=n)
    bath = 0.8 * area + 0.6 * r.normal(size=n)
    other = r.normal(size=(n, 7))                    # age, dist, school + 4 irrelevant
    X = np.c_[area, bed, bath, other]
    beta = np.array([60, 10, 15, -25, -30, 25, 0, 0, 0, 0], float)
    y = 300 + X @ beta + r.normal(0, 40, n)
    return X, y, beta


# ------------------------------------------------------------ the models
class _Base:
    """Shared plumbing: standardise X (so one penalty fits all features), centre y
    (so the intercept is not penalised), and map back at predict time."""

    def _prep(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-12
        self.ybar = y.mean()
        return (X - self.mu) / self.sd, y - self.ybar

    def predict(self, X):
        return ((X - self.mu) / self.sd) @ self.coef_ + self.ybar


class Ridge(_Base):
    """minimise 1/(2n)||y-Xw||^2 + lam/2 ||w||^2   ->   closed form
         (X^T X / n + lam I) w = X^T y / n          (always invertible for lam > 0)"""

    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y):
        Xs, yc = self._prep(X, y)
        n, p = Xs.shape
        self.coef_ = np.linalg.solve(Xs.T @ Xs / n + self.lam * np.eye(p), Xs.T @ yc / n)
        return self


def soft_threshold(z, g):
    """S(z, g) = sign(z) * max(|z| - g, 0): shrink toward 0 by g, and clip at exactly 0."""
    return np.sign(z) * np.maximum(np.abs(z) - g, 0.0)


class ElasticNet(_Base):
    """minimise 1/(2n)||y-Xw||^2 + lam*( a*||w||_1 + (1-a)/2*||w||^2 )   by coordinate descent.
    a = l1_ratio:  1 -> Lasso.   Each coordinate has a closed-form 1-D solution:
         w_j <- S( (1/n) X_j^T r_j , lam*a ) / ( (1/n)||X_j||^2 + lam*(1-a) )
    where r_j = y - sum_{k != j} X_k w_k is the partial residual."""

    def __init__(self, lam, l1_ratio=1.0, max_iter=500, tol=1e-7):
        self.lam, self.a, self.max_iter, self.tol = lam, l1_ratio, max_iter, tol
        self.coef_ = None

    def fit(self, X, y, w0=None):
        Xs, yc = self._prep(X, y)
        n, p = Xs.shape
        w = np.zeros(p) if w0 is None else w0.copy()      # warm start = previous lam on a path
        z = (Xs ** 2).sum(0) / n
        r = yc - Xs @ w
        for _ in range(self.max_iter):
            biggest = 0.0
            for j in range(p):
                rho = Xs[:, j] @ r / n + z[j] * w[j]       # (1/n) X_j^T r_j  (add own contribution back)
                new = soft_threshold(rho, self.lam * self.a) / (z[j] + self.lam * (1 - self.a))
                if new != w[j]:
                    r -= Xs[:, j] * (new - w[j])           # cheap residual update
                    biggest = max(biggest, abs(new - w[j]))
                    w[j] = new
            if biggest < self.tol:
                break
        self.coef_ = w
        return self


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


def make_model(kind, lam):
    return {"ridge": lambda: Ridge(lam), "lasso": lambda: ElasticNet(lam, 1.0),
            "enet": lambda: ElasticNet(lam, 0.5)}[kind]()


def path(kind, X, y, lams):
    """Coefficients along a decreasing lam grid (warm-started for the L1 solvers)."""
    out, w = [], None
    for lam in lams:
        m = make_model(kind, lam)
        m.fit(X, y) if (kind == "ridge" or w is None) else m.fit(X, y, w0=w)
        w = m.coef_
        out.append(w.copy())
    return np.array(out)


def cv_curve(kind, X, y, lams, k=5, seed=0):
    """k-fold cross-validated MSE for each lam (how lam is chosen in practice)."""
    idx = rng_(seed).permutation(len(X))
    folds = np.array_split(idx, k)
    err = np.zeros((len(lams), k))
    for f, te in enumerate(folds):
        tr = np.setdiff1d(idx, te)
        w = None
        for i, lam in enumerate(lams):
            m = make_model(kind, lam)
            m.fit(X[tr], y[tr]) if (kind == "ridge" or w is None) else m.fit(X[tr], y[tr], w0=w)
            w = m.coef_
            err[i, f] = mse(y[te], m.predict(X[te]))
    return err.mean(1)


# ------------------------------------------------------------ geometry helpers
def exact_two_feature_problem(rho=0.8, w_true=(1.8, 0.6), n=400, seed=1):
    """Two standardised features with correlation exactly rho and y = X w_true (no noise),
    so the OLS solution is exactly w_true and the loss is a pure quadratic whose level
    sets are ellipses centred there."""
    Z = rng_(seed).normal(size=(n, 2))
    Z = (Z - Z.mean(0))
    Z = Z @ np.linalg.inv(np.linalg.cholesky(Z.T @ Z / n)).T        # exactly identity covariance
    X = Z @ np.linalg.cholesky(np.array([[1, rho], [rho, 1]])).T     # covariance [[1,rho],[rho,1]]
    return X, X @ np.array(w_true)


def lam_for_budget(kind, X, y, t, norm_ord):
    """Bisect lam so that ||w(lam)|| (L2 for ridge, L1 for lasso) equals the budget t."""
    lo, hi = 0.0, 50.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        w = make_model(kind, mid).fit(X, y).coef_
        if np.linalg.norm(w, norm_ord) > t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- animation 1
def spread_labels(ys, gap):
    """Push label y-positions apart so direct labels don't overlap."""
    order = np.argsort(ys)
    pos = np.array(ys, float)[order]
    for i in range(1, len(pos)):
        pos[i] = max(pos[i], pos[i - 1] + gap)
    out = np.empty_like(pos)
    out[order] = pos - (pos.mean() - np.mean(np.array(ys)[order]))
    return out


def animate_paths(X, y, Xt, yt):
    lams = np.logspace(2, -2.5, 60)                     # large lam first (everything shrunk)
    P = {k: path(k, X, y, lams) for k in ("ridge", "lasso", "enet")}
    cv = {k: cv_curve(k, X, y, lams) for k in P}
    test = {k: [mse(yt, make_model(k, l).fit(X, y).predict(Xt)) for l in lams] for k in P}
    ols = Ridge(1e-9).fit(X, y)
    ols_test = mse(yt, ols.predict(Xt))
    best = {k: int(np.argmin(cv[k])) for k in P}
    for k in P:
        m = make_model(k, lams[best[k]]).fit(X, y)
        print(f"  {k:5s}: CV-chosen lam={lams[best[k]]:.3g}  nonzero={int((np.abs(m.coef_) > 1e-9).sum())}/10  "
              f"test MSE={mse(yt, m.predict(Xt)):.0f}  (OLS test MSE {ols_test:.0f}, noise floor 1600)")
    # optional cross-check with scikit-learn
    try:
        from sklearn.linear_model import Ridge as R, Lasso as L, ElasticNet as E
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
        l0 = 0.8
        d = [np.abs(R(alpha=l0 * len(X)).fit(Xs, y).coef_ - Ridge(l0).fit(X, y).coef_).max(),
             np.abs(L(alpha=l0, tol=1e-10, max_iter=10000).fit(Xs, y).coef_ - ElasticNet(l0, 1).fit(X, y).coef_).max(),
             np.abs(E(alpha=l0, l1_ratio=.5, tol=1e-10, max_iter=10000).fit(Xs, y).coef_ - ElasticNet(l0, .5).fit(X, y).coef_).max()]
        print(f"  sklearn cross-check max |dw|: ridge {d[0]:.1e}, lasso {d[1]:.1e}, enet {d[2]:.1e}")
    except ImportError:
        pass

    frames = np.unique(np.round(np.linspace(0, len(lams) - 1, 40)).astype(int))
    xs = np.log10(lams)
    colors = CAT[:N_SIGNAL] + [MUTED] * (10 - N_SIGNAL)
    cmax = max(np.abs(P["ridge"]).max(), np.abs(P["lasso"]).max()) * 1.05

    fig, axes = new_fig(1, 3, figsize=(13, 5), title="Shrinking the weights: ridge squeezes, lasso switches features off")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.17, top=0.86, wspace=0.3)
    a1, a2, a3 = axes
    with Animator(here(__file__, "regularization_paths.gif"), fps=7, hold=3.0) as anim:
        for i in frames:
            for ax, kind, ttl in ((a1, "ridge", "Ridge (L2): all weights shrink smoothly"),
                                  (a2, "lasso", "Lasso (L1): weights hit exactly 0")):
                ax.clear()
                ax.axhline(0, color=INK2, linewidth=0.8)
                for j in range(10):
                    ax.plot(xs[: i + 1], P[kind][: i + 1, j], color=colors[j], linewidth=1.8 if j < N_SIGNAL else 1.0)
                ax.axvline(xs[i], color=INK, linewidth=0.8, linestyle=":")
                ax.set_xlim(xs[0] + 0.1, xs[-1] - 0.9)                      # inverted: strong penalty on the left
                ax.set_ylim(-cmax, cmax)
                ax.set(title=ttl, xlabel="log10(lambda)   (strong penalty  ->  weak)", ylabel="standardised coefficient")
                # direct labels at the current lam, spread vertically
                cur = P[kind][i]
                big = np.argsort(-np.abs(cur))[:7]
                pos = spread_labels(cur[big], cmax * 0.085)
                for pj, j in zip(pos, big):
                    ax.annotate(FEATURES[j], (xs[i], cur[j]), xytext=(xs[i] - 0.06, pj), fontsize=7.5,
                                color=colors[j] if j < N_SIGNAL else MUTED, va="center", ha="left")
                nz = int((np.abs(cur) > 1e-9).sum())
                ax.text(0.03, 0.04, f"non-zero weights: {nz}/10", transform=ax.transAxes, fontsize=9, color=INK2)
            a3.clear()
            for k, c, nm in (("ridge", BLUE, "ridge"), ("lasso", ORANGE, "lasso"), ("enet", AQUA, "elastic net")):
                a3.plot(xs[: i + 1], cv[k][: i + 1], color=c)
                if i >= best[k]:
                    a3.scatter([xs[best[k]]], [cv[k][best[k]]], s=45, color=c, edgecolor=INK, zorder=5)
            a3.axhline(1600, color=MUTED, linestyle="--", linewidth=1)
            a3.text(xs[-1] - 0.85, 1600 * 1.02, "noise floor", fontsize=8, color=INK2, va="bottom", ha="left")
            a3.text(xs[0] - 0.1, cv["lasso"][-1] * 0.0 + 0, "", fontsize=1)
            a3.set_xlim(xs[0] + 0.1, xs[-1] - 0.9)
            a3.set_ylim(1300, max(cv["ridge"].max(), cv["lasso"].max()) * 1.05)
            a3.set(title="5-fold CV error (dots = best lambda)", xlabel="log10(lambda)", ylabel="CV mean squared error")
            a3.text(0.97, 0.08, "blue ridge  orange lasso  green elastic net", transform=a3.transAxes, ha="right", fontsize=8, color=INK2)
            nzr = int((np.abs(P["ridge"][i]) > 1e-9).sum())
            caption(fig, f"lambda={lams[i]:.3g}: ridge keeps {nzr}/10 weights (norm {np.linalg.norm(P['ridge'][i]):.0f}), "
                         f"lasso keeps {int((np.abs(P['lasso'][i]) > 1e-9).sum())}/10 (norm {np.linalg.norm(P['lasso'][i]):.0f});  "
                         f"CV: ridge {cv['ridge'][i]:.0f}, lasso {cv['lasso'][i]:.0f}")
            anim.grab(fig)


# ------------------------------------------------------------- animation 2
def animate_geometry():
    X, y = exact_two_feature_problem()
    w_ols = Ridge(1e-9).fit(X, y).coef_
    C = X.T @ X / len(X)
    print(f"  geometry: OLS w = {np.round(w_ols, 2)}")

    gx, gy = np.meshgrid(np.linspace(-0.7, 3.0, 220), np.linspace(-1.2, 2.2, 220))
    D = np.stack([gx - w_ols[0], gy - w_ols[1]], -1)
    loss = 0.5 * np.einsum("...i,ij,...j->...", D, C, D)           # (1/2n)||y - Xw||^2 up to the constant 0
    fracs = np.linspace(0.03, 1.0, 36)
    nr_, nl_ = np.linalg.norm(w_ols, 2), np.linalg.norm(w_ols, 1)
    sol = {"ridge": [], "lasso": []}
    for f in fracs:
        sol["ridge"].append(make_model("ridge", lam_for_budget("ridge", X, y, f * nr_, 2)).fit(X, y).coef_)
        sol["lasso"].append(make_model("lasso", lam_for_budget("lasso", X, y, f * nl_, 1)).fit(X, y).coef_)
    sol = {k: np.array(v) for k, v in sol.items()}
    print(f"  lasso hits w2 == 0 for budget fraction up to ~{fracs[np.abs(sol['lasso'][:, 1]) < 1e-6].max():.2f}; "
          f"ridge never ({np.abs(sol['ridge'][:, 1]).min():.3f} smallest |w2|)")

    fig, axes = new_fig(1, 3, figsize=(13, 5), title="Why L1 gives sparse solutions: the diamond has corners on the axes")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.17, top=0.86, wspace=0.3)
    a1, a2, a3 = axes
    th = np.linspace(0, 2 * np.pi, 200)
    levels = [0.05, 0.15, 0.35, 0.7, 1.2]
    with Animator(here(__file__, "regularization_geometry.gif"), fps=7, hold=3.0) as anim:
        for i, f in enumerate(fracs):
            for ax, kind, ttl in ((a1, "ridge", "Ridge: ellipse touches the L2 circle"),
                                  (a2, "lasso", "Lasso: ellipse touches the L1 diamond")):
                ax.clear()
                ax.set_aspect("equal")
                ax.contour(gx, gy, loss, levels=levels, colors=[MUTED], linewidths=0.8)
                w = sol[kind][i]
                lw = 0.5 * (w - w_ols) @ C @ (w - w_ols)
                ax.contour(gx, gy, loss, levels=[lw], colors=[ORANGE], linewidths=2.0)   # the contour the solution lies on
                if kind == "ridge":
                    t = f * nr_
                    ax.plot(t * np.cos(th), t * np.sin(th), color=BLUE, linewidth=2)
                else:
                    t = f * nl_
                    ax.plot([t, 0, -t, 0, t], [0, t, 0, -t, 0], color=BLUE, linewidth=2)
                ax.plot(sol[kind][: i + 1, 0], sol[kind][: i + 1, 1], color=INK2, linewidth=1, linestyle=":")
                ax.scatter(*w_ols, marker="*", s=130, color=INK, zorder=6)
                ax.annotate("OLS", w_ols, xytext=(w_ols[0] + 0.08, w_ols[1] + 0.12), fontsize=9)
                ax.scatter(*w, s=60, color=ORANGE, edgecolor=INK, zorder=7)
                ax.axhline(0, color=INK2, linewidth=0.8); ax.axvline(0, color=INK2, linewidth=0.8)
                ax.set_xlim(-0.7, 3.0); ax.set_ylim(-1.2, 2.2)
                ax.set(title=ttl, xlabel="w1 (area)", ylabel="w2 (bedrooms)")
                if kind == "lasso" and abs(w[1]) < 1e-6:
                    ax.annotate("corner: w2 = 0 exactly", w, xytext=(w[0] + 0.15, -0.6), fontsize=9, color=ORANGE,
                                arrowprops=dict(arrowstyle="->", color=ORANGE))
            a3.clear()
            for kind, ls in (("ridge", "--"), ("lasso", "-")):
                a3.plot(fracs[: i + 1], sol[kind][: i + 1, 0], color=BLUE, linestyle=ls)
                a3.plot(fracs[: i + 1], sol[kind][: i + 1, 1], color=ORANGE, linestyle=ls)
            a3.axhline(0, color=INK2, linewidth=0.8)
            a3.set_xlim(0, 1.0); a3.set_ylim(-0.1, 2.0)
            a3.set(title="Coefficients vs budget", xlabel="budget / budget of OLS", ylabel="coefficient")
            a3.text(0.45, 1.5, "w1 (blue)", color=BLUE, fontsize=9)
            a3.text(0.55, 0.62, "w2 (orange)", color=ORANGE, fontsize=9)
            a3.text(0.03, 1.85, "solid = lasso, dashed = ridge", color=INK2, fontsize=8.5)
            caption(fig, f"budget = {f:.2f} x OLS norm:  ridge w=({sol['ridge'][i][0]:.2f}, {sol['ridge'][i][1]:.2f})   "
                         f"lasso w=({sol['lasso'][i][0]:.2f}, {sol['lasso'][i][1]:.2f})   "
                         f"(bigger budget = weaker penalty)")
            anim.grab(fig)


def main():
    X, y, beta = make_housing(n=30, seed=3)
    Xt, yt, _ = make_housing(n=1000, seed=123)
    # note: both calls use the same generative formula; a different seed gives fresh houses
    animate_paths(X, y, Xt, yt)
    animate_geometry()


if __name__ == "__main__":
    main()
