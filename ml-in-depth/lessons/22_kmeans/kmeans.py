"""
Lesson 22 - K-Means clustering (from scratch)

Problem: a shop wants to split 400 customers into marketing segments using
two behaviour features: annual spend ($k) and store visits per month.
Nobody tells the algorithm which customer belongs where (unsupervised).

Implemented here (NumPy only):
  * Lloyd's algorithm (assign -> update), with k-means++ or random initialisation
  * inertia (within-cluster sum of squares), elbow curve, silhouette score
  * mini-batch k-means and k-medoids (small versions, compared in the printout)

Animated:
  kmeans.gif        Lloyd's algorithm step by step: points recoloured (assign),
                    centroids move with trails (update), Voronoi cells shaded,
                    inertia falling after every half-step.
  kmeans_init_k.gif Phase 1: bad random init vs k-means++ reach different local
                    minima. Phase 2: sweep k = 2..8 and watch elbow + silhouette.

Run:  python kmeans.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ------------------------------------------------------------------- data
def make_customers(n=400, seed=7):
    """Five customer segments in (annual spend $k, visits/month)."""
    r = rng_(seed)
    centers = np.array([[0.9, 2.0], [1.8, 8.5], [4.6, 3.0], [5.8, 10.0], [3.4, 6.2]])
    stds = np.array([[0.35, 0.8], [0.45, 1.1], [0.55, 0.9], [0.5, 1.2], [0.4, 0.9]])
    z = r.choice(5, n, p=[0.2, 0.2, 0.2, 0.15, 0.25])
    X = centers[z] + r.normal(size=(n, 2)) * stds[z]
    return X, z


# ------------------------------------------------------------ the algorithm
def sqdist(X, C):
    """Squared Euclidean distances between every point and every centroid: (n, k).
    ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2   (one matrix product instead of a loop)."""
    return (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]


def inertia(X, C, labels):
    """J = sum_i ||x_i - c_{label_i}||^2   (what k-means minimises)."""
    return float(((X - C[labels]) ** 2).sum())


class KMeans:
    def __init__(self, k, init="kmeans++", max_iter=50, seed=0):
        self.k, self.init, self.max_iter, self.seed = k, init, max_iter, seed

    def _init_centroids(self, X, r):
        n = len(X)
        if self.init == "random":                      # k distinct data points
            return X[r.choice(n, self.k, replace=False)].copy()
        # k-means++: spread the seeds out by sampling far points more often
        C = [X[r.integers(n)]]
        for _ in range(1, self.k):
            d2 = np.maximum(sqdist(X, np.array(C)).min(1), 0)  # D(x)^2 = squared distance to nearest chosen centre
            C.append(X[r.choice(n, p=d2 / d2.sum())])  # P(x) proportional to D(x)^2
        return np.array(C)

    def fit(self, X, C0=None):
        r = rng_(self.seed)
        C = self._init_centroids(X, r) if C0 is None else np.array(C0, float)
        labels = sqdist(X, C).argmin(1)
        # history of "half steps" for the animation: (centroids, labels, inertia, phase)
        self.history = [(C.copy(), labels.copy(), inertia(X, C, labels), "init")]
        for it in range(self.max_iter):
            # ASSIGN: every point joins its nearest centroid
            new_labels = sqdist(X, C).argmin(1)
            self.history.append((C.copy(), new_labels.copy(), inertia(X, C, new_labels), "assign"))
            # UPDATE: every centroid moves to the mean of its points (this minimises J for fixed labels)
            newC = C.copy()
            for j in range(self.k):
                m = new_labels == j
                if m.any():
                    newC[j] = X[m].mean(0)
                else:                                   # empty cluster: re-seed at the worst-fitting point
                    newC[j] = X[sqdist(X, C).min(1).argmax()]
            self.history.append((newC.copy(), new_labels.copy(), inertia(X, newC, new_labels), "update"))
            shift = np.abs(newC - C).max()
            C, labels = newC, new_labels
            if shift < 1e-9:                            # centroids stopped moving -> converged
                break
        self.centroids, self.labels_ = C, labels
        self.inertia_ = inertia(X, C, labels)
        self.n_iter_ = it + 1
        return self

    def predict(self, X):
        return sqdist(X, self.centroids).argmin(1)


def best_of(X, k, n_init=5, init="kmeans++", seed=0):
    """Run k-means several times from different seeds, keep the lowest inertia."""
    runs = [KMeans(k, init, seed=seed + s).fit(X) for s in range(n_init)]
    return min(runs, key=lambda m: m.inertia_)


def silhouette(X, labels):
    """Mean silhouette. For point i: a = mean distance to its own cluster, b = mean distance
    to the nearest other cluster, s = (b - a) / max(a, b) in [-1, 1]."""
    D = np.sqrt(np.maximum(sqdist(X, X), 0))
    ks = np.unique(labels)
    s = np.zeros(len(X))
    for i in range(len(X)):
        own = labels == labels[i]
        if own.sum() < 2:
            continue                                    # singleton clusters get s = 0 by convention
        a = D[i, own].sum() / (own.sum() - 1)
        b = min(D[i, labels == c].mean() for c in ks if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


# ------------------------------------------------------------ relatives
def minibatch_kmeans(X, k, batch=40, n_iters=60, seed=0):
    """Mini-batch k-means: each step uses a small random batch; centroids move by a
    per-centroid learning rate 1/count (a running mean). Much cheaper per step on big data."""
    r = rng_(seed)
    C = KMeans(k, seed=seed)._init_centroids(X, r)
    counts = np.zeros(k)
    for _ in range(n_iters):
        B = X[r.choice(len(X), batch, replace=False)]
        lab = sqdist(B, C).argmin(1)
        for x, j in zip(B, lab):
            counts[j] += 1
            C[j] += (x - C[j]) / counts[j]              # running mean update
    return C


def kmedoids(X, k, n_iters=20, seed=0):
    """k-medoids (alternating variant): centres must be real data points and the cost is the
    sum of (not squared) distances, so outliers pull much less than they do in k-means."""
    r = rng_(seed)
    D = np.sqrt(np.maximum(sqdist(X, X), 0))
    med = r.choice(len(X), k, replace=False)
    for _ in range(n_iters):
        lab = D[:, med].argmin(1)
        new = med.copy()
        for j in range(k):
            idx = np.where(lab == j)[0]
            if len(idx):                                # medoid = member with smallest total distance to the others
                new[j] = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
        if (new == med).all():
            break
        med = new
    return X[med], D[:, med].argmin(1)


# ------------------------------------------------------------ drawing helpers
def voronoi(ax, C, X, k, res=110):
    gx, gy, P = grid_for(X, res=res, pad=0.5)
    z = sqdist(P, C).argmin(1).reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=np.arange(-0.5, k + 0.5, 1), colors=CAT[:k], alpha=0.14, antialiased=True)
    ax.set_xlim(gx.min(), gx.max()); ax.set_ylim(gy.min(), gy.max())


def draw_points(ax, X, labels, C, k, trails=None, grey=False):
    for j in range(k):
        m = labels == j
        ax.scatter(X[m, 0], X[m, 1], s=16, color=MUTED if grey else CAT[j], edgecolor=SURFACE, linewidth=0.5, zorder=3)
    if trails is not None:
        for j in range(k):
            tr = np.array([t[j] for t in trails])
            ax.plot(tr[:, 0], tr[:, 1], color=INK, linewidth=1.0, alpha=0.7, zorder=4)
    ax.scatter(C[:, 0], C[:, 1], s=120, marker="X", color=[CAT[j] for j in range(k)], edgecolor=INK, linewidth=1.3, zorder=6)


# ------------------------------------------------------------ animations
def animate_lloyd(X, k, model):
    """GIF 1: every half-step of Lloyd's algorithm."""
    H = model.history
    J = [h[2] for h in H]
    shifts = [0.0]
    for i in range(1, len(H)):
        shifts.append(np.abs(H[i][0] - H[i - 1][0]).max())
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.6),
                                title="K-means alternates two steps: assign points, move centroids")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.84, wspace=0.28)
    names = {"init": "initial centroids (random data points)", "assign": "ASSIGN: each point joins its nearest centroid",
             "update": "UPDATE: each centroid moves to the mean of its points"}
    with Animator(here(__file__, "kmeans.gif"), fps=3, hold=3.0) as anim:
        for i in range(len(H)):
            C, lab, J_i, phase = H[i]
            a1.clear()
            voronoi(a1, C, X, k)
            draw_points(a1, X, lab, C, k, trails=[h[0] for h in H[: i + 1] if h[3] != "assign"], grey=(phase == "init"))
            a1.set(title="Data space (shading = nearest centroid)", xlabel="annual spend (standardised)",
                   ylabel="visits / month (standardised)")
            a2.clear()
            a2.step(range(i + 1), J[: i + 1], where="post", color=BLUE)
            a2.scatter([i], [J[i]], color=ORANGE, s=36, zorder=5)
            a2.set(title="Inertia J after every half-step", xlabel="half-step (odd = assign, even = update)", ylabel="inertia J")
            a2.set_xlim(0, len(H) - 1); a2.set_ylim(0, max(J) * 1.05)
            a3.clear()
            a3.bar(range(1, i + 1), np.maximum(shifts[1: i + 1], 1e-4), color=[BLUE if H[t][3] == "update" else GRID for t in range(1, i + 1)])
            a3.set(title="Largest centroid move", xlabel="half-step", ylabel="distance", yscale="log")
            a3.set_xlim(0, len(H) - 1); a3.set_ylim(1e-4, 5)
            caption(fig, f"step {i:>2}: {names[phase]}   |   inertia J = {J_i:7.1f}")
            anim.grab(fig)


def animate_init_and_k(X, k_true):
    """GIF 2: (phase 1) bad vs good initialisation, (phase 2) sweeping k with elbow + silhouette."""
    k = k_true
    # --- phase 1 : pick the worst random-init run and a k-means++ run
    rand_runs = [KMeans(k, "random", seed=s).fit(X) for s in range(40)]
    bad = max(rand_runs, key=lambda m: m.inertia_)
    good = best_of(X, k, n_init=1, seed=0)
    for s in range(40):                                 # first k-means++ seed that finds the best solution
        g = KMeans(k, "kmeans++", seed=s).fit(X)
        if g.inertia_ <= min(m.inertia_ for m in rand_runs) * 1.001:
            good = g
            break
    print(f"  random init: worst final inertia {bad.inertia_:.1f} ({bad.n_iter_} iters) | "
          f"k-means++ run: {good.inertia_:.1f} ({good.n_iter_} iters) | best random of 40: {min(m.inertia_ for m in rand_runs):.1f}")
    nb, ng = len(bad.history), len(good.history)
    T = max(nb, ng)
    # --- phase 2 : sweep k
    ks = list(range(2, 9))
    fits = {kk: best_of(X, kk, n_init=5) for kk in ks}
    inert = [fits[kk].inertia_ for kk in ks]
    sil = [silhouette(X, fits[kk].labels_) for kk in ks]
    print("  k : inertia / silhouette ->", ", ".join(f"{kk}:{a:.0f}/{b:.2f}" for kk, a, b in zip(ks, inert, sil)))

    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.6))
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.84, wspace=0.28)
    with Animator(here(__file__, "kmeans_init_k.gif"), fps=3, hold=3.0) as anim:
        fig.suptitle("Initialisation decides which local minimum k-means falls into", fontsize=13, fontweight="bold", x=0.02, ha="left")
        for i in range(T):
            ib, ig = min(i, nb - 1), min(i, ng - 1)
            for a, run, ii, name in ((a1, bad, ib, "bad random init"), (a2, good, ig, "k-means++ init")):
                a.clear()
                C, lab, J_i, ph = run.history[ii]
                voronoi(a, C, X, k)
                draw_points(a, X, lab, C, k, trails=[h[0] for h in run.history[: ii + 1] if h[3] != "assign"], grey=(ph == "init"))
                a.set(title=f"{name}: J = {J_i:.0f}", xlabel="annual spend (std.)", ylabel="visits / month (std.)")
            a3.clear()
            a3.step(range(ib + 1), [h[2] for h in bad.history[: ib + 1]], where="post", color=ORANGE)
            a3.step(range(ig + 1), [h[2] for h in good.history[: ig + 1]], where="post", color=BLUE)
            a3.set(title="Inertia per half-step", xlabel="half-step", ylabel="inertia J")
            a3.set_xlim(0, T - 1)
            a3.set_ylim(min(good.inertia_, bad.inertia_) * 0.9, max(bad.history[0][2], good.history[0][2]) * 1.05)
            label_line(a3, T - 1, bad.history[ib][2], "bad init", ORANGE, dx=-0.2, dy=18, ha="right")
            label_line(a3, T - 1, good.history[ig][2], "k-means++", BLUE, dx=-0.2, dy=-18, ha="right")
            caption(fig, f"half-step {i}: same data, same k={k}; the orange run gets stuck at J={bad.inertia_:.0f} while the blue one reaches J={good.inertia_:.0f}")
            anim.grab(fig)
        # phase 2
        fig.suptitle("Choosing k: the elbow in inertia and the peak in silhouette", fontsize=13, fontweight="bold", x=0.02, ha="left")
        for j, kk in enumerate(ks):
            m = fits[kk]
            a1.clear(); a2.clear(); a3.clear()
            voronoi(a1, m.centroids, X, kk)
            draw_points(a1, X, m.labels_, m.centroids, kk)
            a1.set(title=f"best-of-5 clustering with k = {kk}", xlabel="annual spend (std.)", ylabel="visits / month (std.)")
            a2.plot(ks[: j + 1], inert[: j + 1], "-o", color=BLUE, markersize=4)
            a2.scatter([kk], [inert[j]], color=ORANGE, s=40, zorder=5)
            a2.set(title="Elbow: inertia keeps falling with k", xlabel="k", ylabel="inertia J")
            a2.set_xlim(1.6, 8.4); a2.set_ylim(0, max(inert) * 1.05)
            a3.plot(ks[: j + 1], sil[: j + 1], "-o", color=BLUE, markersize=4)
            a3.scatter([kk], [sil[j]], color=ORANGE, s=40, zorder=5)
            a3.set(title="Silhouette: higher = better separated", xlabel="k", ylabel="mean silhouette")
            a3.set_xlim(1.6, 8.4); a3.set_ylim(0, 0.8)
            caption(fig, f"k = {kk}:  inertia = {inert[j]:7.1f}   silhouette = {sil[j]:.3f}   (true number of segments = {k_true})")
            anim.grab(fig)
    return ks, inert, sil


def main():
    Xraw, z = make_customers()
    X, mu, sd = standardize(Xraw)                      # k-means uses Euclidean distance -> put features on one scale
    k = 5

    # animation 1: a deliberately mediocre random init so that several steps are visible
    cands = [KMeans(k, "random", seed=s).fit(X) for s in range(40)]
    demo = sorted(cands, key=lambda m: m.inertia_)[len(cands) // 2]   # median-quality run
    print(f"  Lloyd demo: {demo.n_iter_} iterations, final inertia {demo.inertia_:.1f}")
    animate_lloyd(X, k, demo)

    ks, inert, sil = animate_init_and_k(X, k)
    print(f"  best k by silhouette = {ks[int(np.argmax(sil))]} (true {k})")

    # relatives
    mb = minibatch_kmeans(X, k)
    full = best_of(X, k)
    print(f"  full k-means J={full.inertia_:.1f} | mini-batch J={inertia(X, mb, sqdist(X, mb).argmin(1)):.1f}")
    med, lab = kmedoids(X, k)
    print(f"  k-medoids cost (sum of distances) {np.sqrt(np.maximum(sqdist(X, med), 0)).min(1).sum():.1f}")

    # optional cross-check with scikit-learn
    try:
        from sklearn.cluster import KMeans as SK
        sk = SK(k, n_init=10, random_state=0).fit(X)
        print(f"  sklearn cross-check: inertia {sk.inertia_:.1f} vs ours {full.inertia_:.1f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
