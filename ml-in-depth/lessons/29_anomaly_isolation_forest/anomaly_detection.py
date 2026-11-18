"""
Lesson 29 - Anomaly detection: Isolation Forest (from scratch) vs z-score, Mahalanobis, LOF

Problem story
  A factory machine has two operating modes ("idle": cool and quiet, "under load": hot and
  vibrating). Every minute we log (temperature, vibration). 500 readings are healthy, 25 are
  injected faults: some far outside, some in the gap *between* the two modes (a reading that
  is "average" for the whole fleet but impossible for either mode).

What is animated
  GIF 1 (isolation_cuts.gif): random axis-parallel cuts isolating ONE normal reading (left) and
        ONE faulty reading (middle) step by step - the fault needs far fewer cuts. Right: the
        average number of cuts over many random trees converging to very different values.
  GIF 2 (iforest_scores.gif): the anomaly-score heat-map as trees are added, the score histogram
        (normal vs fault) separating, and average precision versus number of trees with the
        z-score / Mahalanobis / LOF baselines drawn as horizontal lines.

Run:  python anomaly_detection.py     -> writes isolation_cuts.gif and iforest_scores.gif
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403

EULER = 0.5772156649


# ---------------------------------------------------------------- Isolation Forest
def c_factor(n):
    """c(n) = average path length of an unsuccessful search in a binary search tree with n points
         = 2 H(n-1) - 2 (n-1)/n,   H(i) ~ ln(i) + Euler.
    It is the *expected* depth needed to isolate a point, used to normalise scores and to
    account for the unbuilt subtree below a depth-limited leaf holding `n` points."""
    n = np.asarray(n, float)
    out = np.zeros_like(n)
    big = n > 2
    out[big] = 2 * (np.log(n[big] - 1) + EULER) - 2 * (n[big] - 1) / n[big]
    out[n == 2] = 1.0
    return out


class IsolationTree:
    """One random tree. Each node picks a random feature q and a random split value p drawn
    uniformly between the min and max of that feature among the points in the node."""

    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.nodes = []          # each node: (feature, split, left, right) or (-1, size, -1, -1) for a leaf

    def fit(self, X, rng):
        self._build(X, 0, rng)
        return self

    def _build(self, X, depth, rng):
        idx = len(self.nodes)
        self.nodes.append(None)
        lo, hi = X.min(0), X.max(0)
        splittable = np.where(hi > lo)[0]
        if depth >= self.max_depth or len(X) <= 1 or len(splittable) == 0:
            self.nodes[idx] = (-1, len(X), -1, -1)                 # external node, remembers its size
            return idx
        q = rng.choice(splittable)                                 # random feature
        p = rng.uniform(lo[q], hi[q])                              # random split value
        mask = X[:, q] < p
        left = self._build(X[mask], depth + 1, rng)
        right = self._build(X[~mask], depth + 1, rng)
        self.nodes[idx] = (q, p, left, right)
        return idx

    def path_length(self, X):
        """h(x) = number of edges from the root to the leaf + c(leaf size)  (vectorised over rows)."""
        out = np.zeros(len(X))
        stack = [(0, np.arange(len(X)), 0)]
        while stack:
            node, rows, depth = stack.pop()
            q, p, l, r = self.nodes[node]
            if q == -1:
                out[rows] = depth + c_factor(np.array([p]))[0]
                continue
            m = X[rows, q] < p
            stack.append((l, rows[m], depth + 1))
            stack.append((r, rows[~m], depth + 1))
        return out


class IsolationForest:
    """Ensemble of random isolation trees, each grown on a small random sub-sample (psi points).
    score(x) = 2 ** ( - E[h(x)] / c(psi) )  in (0, 1]:  ~1 very anomalous, <0.5 normal-looking."""

    def __init__(self, n_trees=100, psi=256, seed=0):
        self.n_trees, self.psi, self.seed = n_trees, psi, seed
        self.trees = []

    def fit(self, X):
        rng = rng_(self.seed)
        psi = min(self.psi, len(X))
        self.psi_ = psi
        depth = int(np.ceil(np.log2(psi)))                         # height limit: normal points are not worth isolating deeper
        self.trees = []
        for _ in range(self.n_trees):
            sub = X[rng.choice(len(X), psi, replace=False)]        # sub-sampling reduces swamping and masking
            self.trees.append(IsolationTree(depth).fit(sub, rng))
        return self

    def mean_path(self, X, n_trees=None):
        trees = self.trees[: n_trees or len(self.trees)]
        return np.mean([t.path_length(X) for t in trees], axis=0)

    def score(self, X, n_trees=None):
        return 2.0 ** (-self.mean_path(X, n_trees) / c_factor(np.array([self.psi_]))[0])


# ---------------------------------------------------------------- baselines
def zscore_score(X):
    """Largest absolute per-feature z-score. Assumes ONE Gaussian blob."""
    return np.abs((X - X.mean(0)) / X.std(0)).max(1)


def mahalanobis_score(X):
    """d(x) = sqrt((x-mu)^T S^-1 (x-mu)): z-score that accounts for correlation. Still ONE ellipse."""
    mu, S = X.mean(0), np.cov(X.T)
    D = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", D, np.linalg.inv(S), D))


def lof_score(X, k=20):
    """Local Outlier Factor: compare the local density of x with the density around its neighbours.
       reach(a,b) = max(kdist(b), d(a,b));  lrd(a) = 1 / mean reach to its k neighbours;
       LOF(a) = mean_b lrd(b) / lrd(a)    (~1 normal, >>1 outlier)."""
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(2))
    nn = np.argsort(D, 1)[:, 1 : k + 1]
    kd = D[np.arange(len(X)), nn[:, -1]]                           # distance to k-th neighbour
    reach = np.maximum(kd[nn], np.take_along_axis(D, nn, 1))
    lrd = 1.0 / (reach.mean(1) + 1e-12)
    return lrd[nn].mean(1) / lrd


# ---------------------------------------------------------------- metrics
def average_precision(y, s):
    """Area under the precision-recall curve (step-wise sum of precision at each true positive)."""
    o = np.argsort(-s)
    hit = y[o] == 1
    prec = np.cumsum(hit) / (np.arange(len(y)) + 1)
    return float(prec[hit].sum() / hit.sum())


def precision_at_k(y, s, k):
    return float(y[np.argsort(-s)[:k]].mean())


# ---------------------------------------------------------------- data
def make_machine_data(seed=4):
    """Healthy readings from two modes + injected faults. Returns X (n,2), y (n,) with 1 = fault."""
    r = rng_(seed)
    idle = r.multivariate_normal([55, 0.9], [[16, 0.5], [0.5, 0.03]], 250)       # cool, quiet
    load = r.multivariate_normal([85, 3.2], [[20, 1.2], [1.2, 0.12]], 250)       # hot, vibrating
    # faults: (a) between the modes (looks "average"), (b) far outside, (c) hot but silent / cold but shaking
    gap = np.c_[r.uniform(64, 76, 9), r.uniform(1.4, 2.6, 9)]
    far = np.c_[r.uniform(100, 112, 6), r.uniform(4.2, 5.2, 6)]
    far = np.vstack([far, np.c_[r.uniform(38, 46, 3), r.uniform(0.0, 0.4, 3)]])
    odd = np.c_[r.uniform(88, 98, 4), r.uniform(0.2, 0.7, 4)]
    odd = np.vstack([odd, np.c_[r.uniform(45, 55, 3), r.uniform(3.6, 4.6, 3)]])
    X = np.vstack([idle, load, gap, far, odd])
    y = np.r_[np.zeros(500, int), np.ones(len(X) - 500, int)]
    return X, y


# ---------------------------------------------------------------- animation helpers
def isolate_sequence(X, i, rng):
    """Random cuts that isolate point i. Returns list of (axis, value, cell_before, cell_after) with
    cell = (x0, x1, y0, y1): the box that still contains i after the cut. Mirrors exactly what an
    isolation tree does along the root-to-leaf path of i."""
    mask = np.ones(len(X), bool)
    cell = [X[:, 0].min() - 2, X[:, 0].max() + 2, X[:, 1].min() - 0.2, X[:, 1].max() + 0.2]
    steps = []
    while mask.sum() > 1:
        P = X[mask]
        lo, hi = P.min(0), P.max(0)
        q = rng.integers(0, 2)
        if hi[q] <= lo[q]:
            q = 1 - q
        p = rng.uniform(lo[q], hi[q])
        before = list(cell)
        keep = (X[:, q] < p) == (X[i, q] < p)
        if X[i, q] < p:
            cell[2 * q + 1] = p
        else:
            cell[2 * q] = p
        mask &= keep
        steps.append((q, p, before, list(cell), int(mask.sum())))
    return steps


def draw_cuts(ax, X, y, i, steps, upto, title, color):
    ax.clear()
    ax.scatter(X[:, 0], X[:, 1], s=9, color=[BLUE if v == 0 else RED for v in y], alpha=0.5, linewidth=0, zorder=2)
    if upto > 0:
        c = steps[upto - 1][3]
        ax.add_patch(plt.Rectangle((c[0], c[2]), c[1] - c[0], c[3] - c[2], color=YELLOW, alpha=0.22, zorder=1))
    for q, p, before, after, left in steps[:upto]:
        if q == 0:
            ax.plot([p, p], [before[2], before[3]], color=INK2, linewidth=1.0, zorder=3)
        else:
            ax.plot([before[0], before[1]], [p, p], color=INK2, linewidth=1.0, zorder=3)
    ax.scatter([X[i, 0]], [X[i, 1]], s=85, marker="*", color=ORANGE, edgecolor=INK, linewidth=0.8, zorder=6)
    ax.set(xlim=(X[:, 0].min() - 3, X[:, 0].max() + 3), ylim=(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3))
    n_left = steps[upto - 1][4] if upto > 0 else len(X)
    done = upto > 0 and n_left == 1
    ax.set_title(f"{title}: {upto} cuts, {n_left} point{'s' if n_left != 1 else ''} left" + (" - isolated" if done else ""), fontsize=8.5)
    ax.set_xlabel("temperature (C)"); ax.set_ylabel("vibration (mm/s)")


def animate_cuts(X, y, forest):
    r = rng_(11)
    # a typical healthy point (in a dense part of a mode) and a typical gap fault
    i_norm = int(np.argmin(np.abs(X[:500, 0] - 84) + 8 * np.abs(X[:500, 1] - 3.2)))
    i_anom = 500 + 2                                              # a gap fault
    # pick random sequences with representative lengths (about the median of 200 draws)
    def typical(i):
        seqs = [isolate_sequence(X, i, rng_(s)) for s in range(60)]
        L = np.array([len(s) for s in seqs])
        return seqs[int(np.argsort(L)[len(L) // 2])]
    sn, sa = typical(i_norm), typical(i_anom)

    # many-tree averages (phase 2): independent random isolation sequences
    T = 150
    Ln = np.array([len(isolate_sequence(X, i_norm, rng_(1000 + s))) for s in range(T)])
    La = np.array([len(isolate_sequence(X, i_anom, rng_(5000 + s))) for s in range(T)])
    print(f"  mean cuts to isolate: normal point {Ln.mean():.1f}, fault {La.mean():.1f}  (c(525)={c_factor(np.array([525]))[0]:.1f})")

    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Isolation: anomalies are 'few and different', so random cuts separate them in a handful of steps")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.82, wspace=0.28)
    nsteps = len(sn)
    tree_counts = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 150]
    with Animator(here(__file__, "isolation_cuts.gif"), fps=3, hold=3.0) as anim:
        # phase 1: one random tree, cut by cut
        sel = list(range(0, nsteps + 1))
        if len(sel) > 22:
            sel = sel[:10] + list(np.unique(np.linspace(10, nsteps, 12).astype(int)))
        for k in sel:
            draw_cuts(a1, X, y, i_norm, sn, min(k, len(sn)), "normal reading", BLUE)
            draw_cuts(a2, X, y, i_anom, sa, min(k, len(sa)), "faulty reading", RED)
            a3.clear()
            a3.set(title="Cuts needed to isolate (this tree)", xlim=(0, nsteps + 1), ylim=(0, 3.2))
            a3.set_yticks([]); a3.set_xlabel("number of cuts")
            for yy, ln, col, nm in [(2.2, len(sn), BLUE, "normal"), (1.0, len(sa), RED, "fault")]:
                a3.barh([yy], [min(k, ln)], height=0.7, color=col, alpha=0.85)
                a3.text(0.2, yy + 0.55, nm, fontsize=9, color=INK2)
                a3.text(min(k, ln) + 0.2, yy, f"{min(k, ln)}", va="center", fontsize=9, color=INK)
            caption(fig, f"one random tree, cut {min(k, nsteps)}: each random cut keeps only the side that contains the star - the fault is alone after {len(sa)} cuts, the normal reading needs {len(sn)}")
            anim.grab(fig)
        # phase 2: average over many random trees
        for t in tree_counts:
            draw_cuts(a1, X, y, i_norm, sn, len(sn), "normal reading (one tree)", BLUE)
            draw_cuts(a2, X, y, i_anom, sa, len(sa), "faulty reading (one tree)", RED)
            a3.clear()
            xs = np.arange(1, t + 1)
            a3.plot(xs, np.cumsum(Ln[:t]) / xs, color=BLUE)
            a3.plot(xs, np.cumsum(La[:t]) / xs, color=RED)
            a3.axhline(c_factor(np.array([525]))[0], color=MUTED, linestyle="--", linewidth=1)
            a3.text(T, c_factor(np.array([525]))[0] + 0.5, "c(n): expected depth of a random point", ha="right", fontsize=8, color=INK2)
            a3.text(t, Ln[:t].mean() - 1.2, f"normal E[h]={Ln[:t].mean():.1f}", ha="right" if t > 40 else "left", color=BLUE, fontsize=9, va="top")
            a3.text(t, La[:t].mean() + 0.8, f"fault E[h]={La[:t].mean():.1f}", ha="right" if t > 40 else "left", color=RED, fontsize=9)
            a3.set(title="Average cuts over many random trees", xlabel="number of trees averaged", xlim=(0, T), ylim=(0, 22), ylabel="E[path length]")
            caption(fig, f"{t:>3} trees averaged: E[h] = {Ln[:t].mean():.1f} cuts for the normal reading vs {La[:t].mean():.1f} for the fault -> shorter path = higher anomaly score")
            anim.grab(fig)


def animate_scores(X, y, forest, baselines):
    gx, gy = np.meshgrid(np.linspace(X[:, 0].min() - 6, X[:, 0].max() + 6, 70), np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 70))
    G = np.c_[gx.ravel(), gy.ravel()]
    # accumulate path lengths tree by tree so each frame is cheap
    Pg = np.cumsum([t.path_length(G) for t in forest.trees], axis=0)
    Px = np.cumsum([t.path_length(X) for t in forest.trees], axis=0)
    cpsi = c_factor(np.array([forest.psi_]))[0]
    sc_g = lambda t: 2.0 ** (-(Pg[t - 1] / t) / cpsi)
    sc_x = lambda t: 2.0 ** (-(Px[t - 1] / t) / cpsi)
    ap_curve = np.array([average_precision(y, sc_x(t)) for t in range(1, len(forest.trees) + 1)])
    counts = [1, 2, 3, 4, 6, 8, 12, 17, 25, 35, 50, 70, 100]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bo", ["#f3f2ee", ORANGE])
    k_faults = int(y.sum())

    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Isolation Forest: averaging many random trees turns noisy cuts into a smooth anomaly landscape")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.82, wspace=0.28)
    with Animator(here(__file__, "iforest_scores.gif"), fps=3, hold=3.0) as anim:
        for t in counts:
            s = sc_x(t)
            a1.clear(); a1.grid(False)
            a1.contourf(gx, gy, sc_g(t).reshape(gx.shape), levels=np.linspace(0.3, 0.9, 13), cmap=cmap, extend="both")
            thr = np.sort(s)[::-1][k_faults - 1]
            flagged = s >= thr
            a1.scatter(X[~flagged, 0], X[~flagged, 1], s=8, color=BLUE, alpha=0.6, linewidth=0)
            a1.scatter(X[flagged, 0], X[flagged, 1], s=26, color=RED, edgecolor=INK, linewidth=0.5, zorder=4)
            a1.set(title=f"Score map, {t} tree{'s' if t > 1 else ''}  (red = top {k_faults} flagged)", xlabel="temperature (C)", ylabel="vibration (mm/s)")

            a2.clear()
            bins = np.linspace(0.3, 0.9, 31)
            a2.hist(s[y == 0], bins=bins, color=BLUE, alpha=0.75, weights=np.ones((y == 0).sum()) / (y == 0).sum(), label="healthy")
            a2.hist(s[y == 1], bins=bins, color=RED, alpha=0.75, weights=np.ones((y == 1).sum()) / (y == 1).sum(), label="fault")
            a2.set(title="Anomaly score distribution", xlabel="score 2^(-E[h]/c)", ylabel="share of group", ylim=(0, 0.45))
            a2.text(0.32, 0.40, "healthy", color=BLUE, fontsize=9); a2.text(0.32, 0.35, "fault", color=RED, fontsize=9)

            a3.clear()
            a3.plot(range(1, t + 1), ap_curve[:t], color=BLUE)
            a3.scatter([t], [ap_curve[t - 1]], color=ORANGE, s=34, zorder=5)
            for (nm, v), col, dy in zip(baselines.items(), [MAGENTA, VIOLET, GREEN], [0.0, 0.0, 0.0]):
                a3.axhline(v, color=col, linewidth=1.2, linestyle="--")
                a3.text(100, v + 0.012, f"{nm} {v:.2f}", ha="right", fontsize=8.5, color=col)
            a3.set(title="Average precision (PR-AUC) vs trees", xlabel="number of trees", ylabel="PR-AUC", xlim=(0, 100), ylim=(0, 1.05))
            a3.text(t + 2, max(ap_curve[t - 1] - 0.08, 0.05), f"iForest {ap_curve[t-1]:.2f}", color=BLUE, fontsize=9)
            caption(fig, f"{t:>3} trees: iForest PR-AUC {ap_curve[t-1]:.2f}, precision@{k_faults} {precision_at_k(y, s, k_faults):.2f}  - one tree is noisy, 100 trees give a stable ranking")
            anim.grab(fig)


def main():
    X, y = make_machine_data()
    forest = IsolationForest(n_trees=100, psi=256, seed=1).fit(X)
    s_if = forest.score(X)
    base = {"z-score": zscore_score(X), "Mahalanobis": mahalanobis_score(X), "LOF": lof_score(X, 20)}
    k = int(y.sum())
    print(f"  {len(X)} readings, {k} faults")
    print(f"  {'method':12s} PR-AUC  precision@{k}")
    results = {"iForest": s_if, **base}
    for nm, s in results.items():
        print(f"  {nm:12s} {average_precision(y, s):.3f}   {precision_at_k(y, s, k):.3f}")
    # sanity: score of a healthy mode centre is < 0.5-ish, a gap fault is much higher
    print(f"  mean score healthy {s_if[y == 0].mean():.3f}  faults {s_if[y == 1].mean():.3f}")

    animate_cuts(X, y, forest)
    animate_scores(X, y, forest, {nm: average_precision(y, s) for nm, s in base.items()})

    # optional cross-check against scikit-learn
    try:
        from sklearn.ensemble import IsolationForest as SkIF
        sk = SkIF(n_estimators=100, max_samples=256, random_state=0).fit(X)
        print(f"  [sklearn cross-check] PR-AUC {average_precision(y, -sk.score_samples(X)):.3f} (ours {average_precision(y, s_if):.3f})")
    except ImportError:
        print("  (scikit-learn not installed; skipping cross-check)")


if __name__ == "__main__":
    main()
