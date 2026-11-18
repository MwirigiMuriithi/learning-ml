"""
Lesson 09 - k-Nearest Neighbours (from scratch)

Problem: a factory monitors a machine with two sensors (vibration, temperature).
Each reading is labelled "healthy" (0) or "fault" (1).  A new reading is classified by
asking its k most similar past readings to vote.  There is NO training phase: the
"model" is the stored data set (a *lazy learner*).

Implemented (NumPy only):
  * KNN  - classification + regression, euclidean / manhattan / chebyshev distance,
           uniform or inverse-distance weights
  * KDTree - a small kd-tree to show how neighbour search avoids scanning all points
  * numeric demos: feature scaling, curse of dimensionality, weighted kNN, kNN regression

Animated:
  knn_query.gif    - a query point travels across the data; its k neighbours light up,
                     vote bars and P(fault) along the path are drawn live.
  knn_k_sweep.gif  - decision regions as k goes 1 -> 40, with train/test accuracy.
  knn_scaling.gif  - the same data with one feature's unit changed (cm -> mm -> ...):
                     without scaling kNN silently ignores the other feature.

Run:  python knn.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------- the model
class KNN:
    """k-nearest-neighbours. task = 'classification' (vote) or 'regression' (average)."""

    def __init__(self, k=5, metric="euclidean", weights="uniform", task="classification"):
        self.k, self.metric, self.weights, self.task = k, metric, weights, task

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y)       # "training" = memorising
        self.n_classes = int(self.y.max()) + 1 if self.task == "classification" else None
        return self

    def distances(self, Q):
        """(m, n) matrix of distances from every query to every stored point."""
        diff = Q[:, None, :] - self.X[None, :, :]
        if self.metric == "euclidean":
            return np.sqrt((diff ** 2).sum(-1))                      # sqrt(sum (a-b)^2)
        if self.metric == "manhattan":
            return np.abs(diff).sum(-1)                              # sum |a-b|
        if self.metric == "chebyshev":
            return np.abs(diff).max(-1)                              # max |a-b|
        raise ValueError(self.metric)

    def neighbours(self, Q):
        """Indices (m,k) and distances (m,k) of the k closest stored points, nearest first."""
        D = self.distances(np.atleast_2d(Q))
        idx = np.argsort(D, axis=1)[:, : self.k]
        return idx, np.take_along_axis(D, idx, axis=1)

    def _w(self, d):
        # uniform: every neighbour counts 1;  distance: closer neighbours count more (1/d)
        return np.ones_like(d) if self.weights == "uniform" else 1.0 / (d + 1e-9)

    def predict_proba(self, Q):
        idx, d = self.neighbours(Q)
        w = self._w(d)
        P = np.zeros((len(idx), self.n_classes))
        for c in range(self.n_classes):
            P[:, c] = (w * (self.y[idx] == c)).sum(1)               # weighted votes per class
        return P / P.sum(1, keepdims=True)

    def predict(self, Q):
        if self.task == "classification":
            return self.predict_proba(Q).argmax(1)
        idx, d = self.neighbours(Q)
        w = self._w(d)
        return (w * self.y[idx]).sum(1) / w.sum(1)                   # (weighted) mean of neighbour targets


# ---------------------------------------------------------------- kd-tree
class KDTree:
    """Tiny kd-tree: split on the median of one coordinate (cycling axes), leaves hold <= leaf points.
    query() returns the k nearest indices AND how many distance evaluations it needed, so we can
    compare with brute force (= n evaluations)."""

    def __init__(self, X, leaf=10):
        self.X, self.leaf = X, leaf
        self.root = self._build(np.arange(len(X)), 0)

    def _build(self, idx, depth):
        if len(idx) <= self.leaf:
            return ("leaf", idx)
        ax = depth % self.X.shape[1]
        order = idx[np.argsort(self.X[idx, ax])]
        mid = len(order) // 2
        return ("node", ax, self.X[order[mid], ax], self._build(order[:mid], depth + 1), self._build(order[mid:], depth + 1))

    def query(self, q, k=1):
        best = []            # list of (dist, index), kept sorted, at most k long
        self.evals = 0

        def visit(node):
            if node[0] == "leaf":
                d = np.sqrt(((self.X[node[1]] - q) ** 2).sum(1))
                self.evals += len(d)
                for di, i in zip(d, node[1]):
                    best.append((di, i))
                best.sort()
                del best[k:]
                return
            _, ax, split, left, right = node
            near, far = (left, right) if q[ax] < split else (right, left)
            visit(near)
            # only cross the splitting plane if the k-th best ball reaches over it
            if len(best) < k or abs(q[ax] - split) < best[-1][0]:
                visit(far)

        visit(self.root)
        return np.array([i for _, i in best]), self.evals


# ---------------------------------------------------------------- data
def machine_data(n=260, seed=5):
    """Two interleaved moons = healthy vs fault readings (vibration, temperature)."""
    X, y = make_moons(n=n, noise=0.28, seed=seed)
    return X, y


def accuracy_for_k(Xtr, ytr, Xte, yte, k, **kw):
    m = KNN(k=k, **kw).fit(Xtr, ytr)
    return accuracy(ytr, m.predict(Xtr)), accuracy(yte, m.predict(Xte))


# ---------------------------------------------------------------- animation 1: a query travels
def query_animation():
    X, y = machine_data()
    k = 7
    m = KNN(k=k).fit(X, y)
    # path of the query: a straight line across the two moons
    T = 42
    t = np.linspace(0, 1, T)
    path = np.c_[-1.4 + 3.6 * t, 1.25 - 1.7 * t + 0.25 * np.sin(2 * np.pi * t)]
    probs = m.predict_proba(path)[:, 1]

    fig = plt.figure(figsize=(13, 4.8))
    fig.suptitle("k-NN: no training, just look up the k closest stored points and let them vote", fontsize=13,
                 fontweight="bold", x=0.02, ha="left")
    gs = GridSpec(1, 3, figure=fig, left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28, width_ratios=[1.35, 0.8, 1.1])
    a1, a2, a3 = (fig.add_subplot(gs[0, j]) for j in range(3))
    with Animator(here(__file__, "knn_query.gif"), fps=8, hold=2.0) as anim:
        for i in range(T):
            q = path[i]
            idx, d = m.neighbours(q)
            idx, d = idx[0], d[0]
            votes = np.bincount(y[idx], minlength=2)
            a1.clear()
            scatter_classes(a1, X, y, s=16)
            a1.scatter(X[idx, 0], X[idx, 1], s=95, facecolor="none", edgecolor=INK, linewidth=1.3, zorder=4)
            for j in idx:
                a1.plot([q[0], X[j, 0]], [q[1], X[j, 1]], color=MUTED, linewidth=0.9, zorder=2)
            a1.add_patch(plt.Circle(q, d.max(), fill=False, color=INK2, linestyle="--", linewidth=1.0, zorder=2))
            a1.plot(path[: i + 1, 0], path[: i + 1, 1], color=MUTED, linewidth=0.8, linestyle=":")
            a1.scatter([q[0]], [q[1]], marker="*", s=230, color=YELLOW, edgecolor=INK, zorder=6)
            a1.set(title=f"Data space: the {k} nearest neighbours (ringed)", xlabel="vibration", ylabel="temperature",
                   xlim=(-1.9, 2.9), ylim=(-1.2, 1.9), aspect="equal")
            a1.text(0.02, 0.97, "blue = healthy, orange = fault\ndashed circle reaches the k-th neighbour", transform=a1.transAxes,
                    va="top", fontsize=8.5, color=INK2)

            a2.clear()
            a2.bar([0, 1], votes, color=[CAT[0], CAT[1]], width=0.6)
            for c in (0, 1):
                a2.text(c, votes[c] + 0.15, str(votes[c]), ha="center", fontsize=10, color=INK)
            a2.set(title="Votes", xticks=[0, 1], xticklabels=["healthy", "fault"], ylim=(0, k + 1), ylabel="neighbours")
            a2.text(0.5, k + 0.55, f"majority: {'fault' if votes[1] > votes[0] else 'healthy'}", ha="center", fontsize=9, color=INK2)

            a3.clear()
            a3.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
            a3.plot(range(i + 1), probs[: i + 1], color=BLUE, drawstyle="steps-post")
            a3.scatter([i], [probs[i]], s=34, color=ORANGE, zorder=5)
            a3.set(title="P(fault) = votes for fault / k", xlabel="step along the query's path", ylabel="P(fault)", xlim=(0, T - 1), ylim=(-0.05, 1.05))
            a3.text(T - 1, 0.53, "decision threshold", ha="right", fontsize=8.5, color=INK2)
            caption(fig, f"step {i + 1:>2}/{T}:  query=({q[0]:+.2f}, {q[1]:+.2f})   nearest distance={d.min():.2f}   "
                         f"votes healthy/fault={votes[0]}/{votes[1]}   P(fault)={probs[i]:.2f}")
            anim.grab(fig)


# ---------------------------------------------------------------- animation 2: sweep over k
def k_sweep_animation():
    X, y = machine_data(n=300, seed=11)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test=0.35, seed=2)
    ks = np.arange(1, 41)
    res = np.array([accuracy_for_k(Xtr, ytr, Xte, yte, k) for k in ks])
    gx, gy, P = grid_for(X, res=100, pad=0.4)

    fig, (a1, a2) = new_fig(1, 2, figsize=(11, 4.8), title="k controls smoothness: k=1 memorises noise, large k blurs the boundary",
                            gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.2, top=0.84, wspace=0.25)
    show = list(range(1, 41, 1))[::1]
    show = [k for k in show if k <= 12 or k % 2 == 0]
    with Animator(here(__file__, "knn_k_sweep.gif"), fps=6, hold=2.5) as anim:
        for k in show:
            mdl = KNN(k=k).fit(Xtr, ytr)
            a1.clear()
            z = mdl.predict(P).reshape(gx.shape)
            a1.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], colors=[CAT[0], CAT[1]], alpha=0.2)
            a1.contour(gx, gy, z, levels=[0.5], colors=[INK], linewidths=1.3)
            scatter_classes(a1, Xtr, ytr, s=15)
            a1.scatter(Xte[:, 0], Xte[:, 1], s=26, marker="s", facecolor="none", edgecolor=INK, linewidth=0.7, zorder=4)
            a1.set(title=f"Decision regions, k = {k}", xlabel="vibration", ylabel="temperature", xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
            a1.text(0.02, 0.97, "dots = train, squares = test", transform=a1.transAxes, va="top", fontsize=8.5, color=INK2)
            a2.clear()
            kk = ks[ks <= k]
            a2.plot(kk, res[: len(kk), 0], color=BLUE)
            a2.plot(kk, res[: len(kk), 1], color=ORANGE)
            a2.scatter([k, k], res[k - 1], s=32, color=[BLUE, ORANGE], zorder=5)
            a2.set(title="Accuracy vs k", xlabel="k (number of neighbours)", ylabel="accuracy", xlim=(0, 41), ylim=(0.7, 1.02))
            label_line(a2, 41, res[-1, 0] + 0.012, "train", color=BLUE, ha="right")
            label_line(a2, 41, res[-1, 1] - 0.025, "test", color=ORANGE, ha="right")
            caption(fig, f"k={k:>2}:  train accuracy={res[k - 1, 0]:.1%}   test accuracy={res[k - 1, 1]:.1%}   "
                         f"(k=1 trivially scores 100% on train: each point is its own neighbour)")
            anim.grab(fig)
    best = int(ks[res[:, 1].argmax()])
    print(f"  k sweep: k=1 train/test = {res[0, 0]:.3f}/{res[0, 1]:.3f}; best test k={best} ({res[best - 1, 1]:.3f}); "
          f"k=40 train/test = {res[-1, 0]:.3f}/{res[-1, 1]:.3f}")


# ---------------------------------------------------------------- animation 3: scaling
def scaling_data(n=320, seed=7):
    """Loan-style data: age (years, ~20-70) and income ($, ~20k-120k). Risk depends on both,
    each roughly equally *after* standardising."""
    r = rng_(seed)
    age = r.uniform(20, 70, n)
    inc = r.uniform(20_000, 120_000, n)
    za, zi = (age - 45) / 14.4, (inc - 70_000) / 28_900
    s = za * 1.0 - zi * 1.0 + 0.35 * r.normal(size=n)
    # a curved boundary so both features matter
    s += 0.35 * za * zi
    y = (s > 0).astype(int)
    return np.c_[age, inc], y


def scaling_animation():
    X, y = scaling_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test=0.35, seed=3)
    Z_tr, mu, sd = standardize(Xtr)
    Z_te = (Xte - mu) / sd
    # multiplier m applied to the standardised income axis: m=1 is fair, m=1000 mimics "income in cents"
    ms = np.geomspace(1, 1000, 18)
    accs = []
    for m_ in ms:
        s_ = np.array([1.0, m_])
        accs.append(accuracy(yte, KNN(k=7).fit(Z_tr * s_, ytr).predict(Z_te * s_)))
    raw = accuracy(yte, KNN(k=7).fit(Xtr, ytr).predict(Xte))
    gx, gy, P = grid_for(X, res=90, pad=0.0)
    gx = np.linspace(15, 75, 90); gy = np.linspace(15_000, 125_000, 90)
    GX, GY = np.meshgrid(gx, gy)
    PG = np.c_[GX.ravel(), GY.ravel()]
    fig, (a1, a2) = new_fig(1, 2, figsize=(11, 4.8), title="Distance-based models obey the units: the biggest-scale feature wins",
                            gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.84, wspace=0.27)
    with Animator(here(__file__, "knn_scaling.gif"), fps=3, hold=3.0) as anim:
        for j, m_ in enumerate(ms):
            s_ = np.array([1.0, m_])
            mdl = KNN(k=7).fit(Z_tr * s_, ytr)
            z = mdl.predict(((PG - mu) / sd) * s_).reshape(GX.shape)
            a1.clear()
            a1.contourf(GX, GY, z, levels=[-0.5, 0.5, 1.5], colors=[CAT[0], CAT[1]], alpha=0.2)
            a1.contour(GX, GY, z, levels=[0.5], colors=[INK], linewidths=1.2)
            scatter_classes(a1, Xtr, ytr, s=14)
            a1.set(title=f"Regions when income is stretched x{m_:,.0f}", xlabel="age (years)", ylabel="income ($)", xlim=(15, 75), ylim=(15_000, 125_000))
            a1.text(0.02, 0.97, "blue = repaid, orange = default", transform=a1.transAxes, va="top", fontsize=8.5, color=INK2)
            a2.clear()
            a2.plot(ms[: j + 1], accs[: j + 1], color=BLUE, marker="o", markersize=3.5)
            a2.scatter([m_], [accs[j]], s=36, color=ORANGE, zorder=5)
            a2.axhline(max(accs[0], 0), color=MUTED, linestyle="--", linewidth=1)
            a2.set(title="Test accuracy as income is stretched", xlabel="stretch factor on the income axis (log)", ylabel="test accuracy",
                   xscale="log", xlim=(0.85, 1200), ylim=(0.45, 1.0))
            a2.text(1000, accs[0] + 0.015, "properly standardised", ha="right", fontsize=8.5, color=INK2)
            caption(fig, f"stretch x{m_:,.0f}:  accuracy={accs[j]:.1%}   "
                         f"(income dominates the distance, age is ignored)" if m_ > 10 else f"stretch x{m_:,.1f}:  accuracy={accs[j]:.1%}   (both features count)")
            anim.grab(fig)
    print(f"  scaling: standardised acc={accs[0]:.3f}; income x1000 acc={accs[-1]:.3f}; raw (age in years, income in $) acc={raw:.3f}")


# ---------------------------------------------------------------- numeric demos
def demos():
    r = rng_(0)
    # (a) weighted vs uniform on the noisy moons
    X, y = machine_data(n=400, seed=21)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test=0.4, seed=4)
    for k in (3, 15, 40):
        u = accuracy(yte, KNN(k=k).fit(Xtr, ytr).predict(Xte))
        w = accuracy(yte, KNN(k=k, weights="distance").fit(Xtr, ytr).predict(Xte))
        print(f"  k={k:>2}: uniform={u:.3f}  distance-weighted={w:.3f}")
    for met in ("euclidean", "manhattan", "chebyshev"):
        print(f"  metric {met:<9}: test acc={accuracy(yte, KNN(k=9, metric=met).fit(Xtr, ytr).predict(Xte)):.3f}")
    # (b) regression kNN: noisy sine
    xs, ys = make_sine(n=80, noise=0.3, seed=1)
    xt = np.linspace(0, 2 * np.pi, 200)
    for k in (1, 5, 20):
        yh = KNN(k=k, task="regression").fit(xs[:, None], ys).predict(xt[:, None])
        print(f"  kNN regression k={k:>2}: MSE vs true sine = {np.mean((yh - np.sin(xt)) ** 2):.3f}")
    # (c) curse of dimensionality: relative contrast of distances, uniform data in [0,1]^d
    print("  curse of dimensionality (n=1000 uniform points; contrast = (d_max - d_min)/d_min from a random query):")
    for d in (1, 2, 5, 10, 50, 200, 1000):
        P = r.uniform(size=(1000, d)); q = r.uniform(size=d)
        dist = np.sqrt(((P - q) ** 2).sum(1))
        print(f"     d={d:>4}: nearest={dist.min():.3f} farthest={dist.max():.3f} contrast={(dist.max() - dist.min()) / dist.min():8.2f}")
    # (d) kd-tree: distance evaluations vs brute force
    for d in (2, 5, 12):
        P = r.uniform(size=(20000, d)); tree = KDTree(P, leaf=16)
        ev, ok = [], True
        for _ in range(60):
            q = r.uniform(size=d)
            idx, e = tree.query(q, k=3)
            brute = np.argsort(((P - q) ** 2).sum(1))[:3]
            ok &= set(idx) == set(brute)
            ev.append(e)
        print(f"  kd-tree d={d:>2}: mean distance evals = {np.mean(ev):8.0f} of 20000 (brute force) ; same answer as brute force: {ok}")
    try:
        from sklearn.neighbors import KNeighborsClassifier
        sk = KNeighborsClassifier(n_neighbors=9).fit(Xtr, ytr)
        same = np.mean(sk.predict(Xte) == KNN(k=9).fit(Xtr, ytr).predict(Xte))
        print(f"  sklearn check: predictions agree on {same:.1%} of test points")
    except ImportError:
        pass


def main():
    query_animation()
    k_sweep_animation()
    scaling_animation()
    demos()


if __name__ == "__main__":
    main()
