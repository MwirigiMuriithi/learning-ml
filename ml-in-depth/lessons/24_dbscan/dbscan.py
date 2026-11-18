"""
Lesson 24 - DBSCAN (density-based clustering), from scratch

Problem story: a ride-hailing company logs GPS pick-up points.  Demand is concentrated in a few dense hot-spots with
odd shapes (two curved streets and a plaza) plus scattered one-off pick-ups that belong to no hot-spot.  The company does
not know how many hot-spots exist, and wants the stray pick-ups flagged as noise instead of being forced into a cluster.

Implemented (NumPy only):
  * DBSCAN: eps-neighbourhoods, core / border / noise points, cluster expansion by density-reachability
  * a recorded event log so the expansion can be animated point by point
  * the k-distance graph for choosing eps
  * comparison with k-means (which needs K and cannot leave noise out)

Animated:
  dbscan_expand.gif  the algorithm visits points one at a time: eps-circle, core points seed a cluster that spreads
  dbscan_eps.gif     sweep eps: too small = everything is noise, too large = everything merges; k-distance graph; k-means beside it

Run:  python dbscan.py
"""
import sys, pathlib, itertools

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from matplotlib.patches import Circle


# ------------------------------------------------------------------- data
def pickups(seed=5):
    """Pick-up locations (km east, km north): 2 curved streets + 1 plaza + stray noise.  z = hot-spot id, -1 = noise."""
    r = rng_(seed)
    Xm, ym = make_moons(260, noise=0.06, seed=seed)         # two interlocking crescents -> two curved streets
    Xm = Xm * 2.2
    plaza = np.array([5.3, 3.3]) + r.normal(0, 0.28, (70, 2))
    stray = np.c_[r.uniform(-2.2, 7.5, 32), r.uniform(-2.4, 4.8, 32)]
    X = np.vstack([Xm, plaza, stray])
    z = np.r_[ym, np.full(70, 2), np.full(32, -1)]
    return X, z


def pairwise(X):
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ X.T + (X ** 2).sum(1)[None, :]
    return np.sqrt(np.maximum(d2, 0))


# ------------------------------------------------------------ the algorithm
def dbscan(X, eps, min_pts, log=False):
    """Density-Based Spatial Clustering of Applications with Noise.

    core point   : at least `min_pts` points (itself included) within distance eps
    border point : not core, but within eps of a core point
    noise        : neither
    A cluster = all core points connected through chains of eps-neighbours, plus their border points.
    Returns labels (-1 = noise) and, if log=True, a list of events (label snapshot, current point, type) for animation.
    """
    n = len(X)
    D = pairwise(X)
    nbrs = [np.flatnonzero(D[i] <= eps) for i in range(n)]
    is_core = np.array([len(nb) >= min_pts for nb in nbrs])
    labels = np.full(n, -2)                                 # -2 = unvisited, -1 = noise, >=0 cluster id
    events = []
    cid = -1
    for i in range(n):
        if labels[i] != -2:
            continue
        if not is_core[i]:
            labels[i] = -1                                  # provisional noise (may become a border point later)
            if log:
                events.append((labels.copy(), i, "noise"))
            continue
        cid += 1                                            # i is core: start a new cluster and grow it
        labels[i] = cid
        queue = [i]
        if log:
            events.append((labels.copy(), i, "seed"))
        while queue:
            p = queue.pop(0)
            for q in nbrs[p]:
                if labels[q] in (-2, -1):                   # unvisited, or noise that is actually a border point
                    labels[q] = cid
                    if is_core[q]:
                        queue.append(q)                     # core points keep the wave going; border points do not
                    if log:
                        events.append((labels.copy(), q, "core" if is_core[q] else "border"))
    return (labels, is_core, events) if log else (labels, is_core)


def k_distance(X, k):
    """Distance of every point to its k-th nearest neighbour (itself counts as 0th), sorted descending."""
    D = np.sort(pairwise(X), axis=1)
    return np.sort(D[:, k - 1])[::-1]                       # k-th point counting itself => min_pts convention


def kmeans(X, K, seed=0, iters=50):
    r = rng_(seed)
    C = X[r.choice(len(X), K, replace=False)]
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(K)])
    return lab


def purity(z, lab):
    """Share of points in non-noise truth groups whose cluster is mostly one truth group (matches clusters to groups)."""
    m = z >= 0
    tot = 0
    for c in np.unique(lab[m]):
        if c < 0:
            continue
        tot += np.bincount(z[m][lab[m] == c]).max()
    return tot / m.sum()


# ------------------------------------------------------------ plotting helpers
def style_points(ax, X, lab, is_core, eps=None):
    for c in np.unique(lab):
        m = lab == c
        if c == -2:
            ax.scatter(X[m, 0], X[m, 1], s=16, color=GRID, edgecolor=MUTED, linewidth=0.4, zorder=2)
        elif c == -1:
            ax.scatter(X[m, 0], X[m, 1], s=22, color=INK2, marker="x", linewidth=1.0, zorder=3)
        else:
            col = CAT[c % 8]
            ax.scatter(X[m & is_core, 0], X[m & is_core, 1], s=34, color=col, edgecolor=SURFACE, linewidth=0.5, zorder=3)
            ax.scatter(X[m & ~is_core, 0], X[m & ~is_core, 1], s=14, color=col, edgecolor=SURFACE, linewidth=0.5, zorder=3)


# ------------------------------------------------------------ animation 1
def expand_gif(path, eps, min_pts):
    X, z = pickups()
    lab, core, ev = dbscan(X, eps, min_pts, log=True)
    n = len(X)
    nev = len(ev)
    picks = sorted(set(np.unique(np.round(np.geomspace(1, nev, 26)).astype(int) - 1)) | {nev - 1})
    xlim = (X[:, 0].min() - 0.6, X[:, 0].max() + 0.6)
    ylim = (X[:, 1].min() - 0.6, X[:, 1].max() + 0.6)
    with Animator(path, fps=3, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 5.2), title=f"DBSCAN: eps = {eps}, MinPts = {min_pts}", gridspec_kw={"width_ratios": [1.9, 1]})
        for e in picks:
            snap, cur, kind = ev[e]
            a1.clear(); a2.clear()
            style_points(a1, X, snap, core)
            last = e == nev - 1
            if not last:
                a1.add_patch(Circle(X[cur], eps, fill=False, ec=ORANGE, lw=2, zorder=5))
                a1.scatter(*X[cur], s=70, color=ORANGE, zorder=6, edgecolor=SURFACE)
            a1.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="km east", ylabel="km north",
                   title="Pick-ups: colour = cluster, x = noise, grey = not yet visited")
            nclus = int(snap.max()) + 1
            cnt = [np.sum(snap >= 0), np.sum(snap == -1), np.sum(snap == -2)]
            a2.barh(["clustered", "noise (so far)", "unvisited"], cnt, color=[BLUE, INK2, MUTED], height=0.55)
            for i_, v in enumerate(cnt):
                a2.text(v + 4, i_, str(v), va="center", fontsize=9)
            a2.set(xlim=(0, n * 1.08), title=f"Progress: {nclus} cluster(s) found")
            a2.grid(False)
            if last:
                msg = f"Done: {nclus} clusters, {int(np.sum(snap == -1))} noise points (true: 3 hot-spots, 32 strays). Big dots = core, small = border, x = noise"
            elif kind == "seed":
                msg = "A core point (>= MinPts neighbours inside the circle) starts a new cluster"
            elif kind == "core":
                msg = "A neighbour that is itself core joins the cluster and will spread it further"
            elif kind == "border":
                msg = "A border point joins the cluster but cannot spread it (too few neighbours of its own)"
            else:
                msg = "Too few neighbours: labelled noise for now (it can still become a border point later)"
            caption(fig, msg)
            anim.grab(fig)


# ------------------------------------------------------------ animation 2
def eps_gif(path, min_pts):
    X, z = pickups()
    kd = k_distance(X, min_pts)
    eps_list = [0.12, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.8, 1.3, 1.8]
    xlim = (X[:, 0].min() - 0.6, X[:, 0].max() + 0.6)
    ylim = (X[:, 1].min() - 0.6, X[:, 1].max() + 0.6)
    with Animator(path, fps=1.6, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 5), title=f"Choosing eps (MinPts = {min_pts})")
        for eps in eps_list:
            a1.clear(); a2.clear(); a3.clear()
            lab, core = dbscan(X, eps, min_pts)
            K = int(lab.max()) + 1
            style_points(a1, X, lab, core)
            a1.set(xlim=xlim, ylim=ylim, aspect="equal", title=f"DBSCAN eps = {eps}: {K} clusters", xlabel="km east", ylabel="km north")
            a1.text(0.02, 0.03, f"noise {int(np.sum(lab == -1))}", transform=a1.transAxes, fontsize=8.5, color=INK2)
            a2.plot(np.arange(len(kd)), kd, color=BLUE)
            a2.axhline(eps, color=ORANGE, lw=2)
            a2.set(title=f"Sorted {min_pts}-NN distance", xlabel="points, sorted", ylabel="distance to k-th neighbour")
            a2.text(len(kd) * 0.5, eps + 0.03, f"eps = {eps}", color=ORANGE, fontsize=9)
            Kk = max(K, 2)
            km = kmeans(X, Kk)
            style_points(a3, X, km, np.ones(len(X), bool))
            a3.set(xlim=xlim, ylim=ylim, aspect="equal", title=f"k-means with K = {Kk}", xlabel="km east")
            if np.sum(lab == -1) > 0.7 * len(X):
                msg = f"eps = {eps}: too small, almost every point is noise"
            elif K == 3:
                msg = f"eps = {eps}: the knee of the k-distance curve; the three hot-spots are found and strays are flagged as noise"
            elif K > 3:
                msg = f"eps = {eps}: still too small, the streets break into pieces"
            elif K <= 2:
                msg = f"eps = {eps}: too large, separate hot-spots merge and the noise is absorbed"
            else:
                msg = f"eps = {eps}"
            caption(fig, msg)
            anim.grab(fig)


# ------------------------------------------------------------ main
def main():
    X, z = pickups()
    eps, mp = 0.45, 5
    print("Lesson 24: DBSCAN")
    lab, core = dbscan(X, eps, mp)
    K = int(lab.max()) + 1
    print(f"  eps={eps}, MinPts={mp}: {K} clusters, {np.sum(lab == -1)} noise, core {core.sum()}, "
          f"border {np.sum((lab >= 0) & ~core)}; purity vs hot-spots {purity(z, lab):.3f}")
    print(f"  true strays flagged as noise: {np.sum((lab == -1) & (z == -1))}/{np.sum(z == -1)}; hot-spot points wrongly noise: {np.sum((lab == -1) & (z >= 0))}")
    km = kmeans(X, 3)
    print(f"  k-means (K=3, told the right K) purity {purity(z, km):.3f}, flags no noise")
    for e in [0.12, 0.3, 0.45, 0.8, 1.3, 1.8]:
        l, c = dbscan(X, e, mp)
        print(f"   eps {e:4}: clusters {int(l.max()) + 1:2d}, noise {np.sum(l == -1):3d}")
    try:
        from sklearn.cluster import DBSCAN as SK
        sk = SK(eps=eps, min_samples=mp).fit(X)
        same_noise = np.array_equal(sk.labels_ == -1, lab == -1)
        cm = np.zeros(len(X), bool); cm[sk.core_sample_indices_] = True
        # clusters must match on core points (border points may differ by visit order)
        a = {(l1, l2) for l1, l2 in zip(lab[cm], sk.labels_[cm])}
        print(f"  [sklearn] same noise set: {same_noise}; same core flags: {np.array_equal(cm, core)}; "
              f"core-point cluster pairs {sorted(a)} (one-to-one => identical partitions)")
    except Exception as ex:
        print("  (sklearn check skipped:", ex, ")")
    expand_gif(here(__file__, "dbscan_expand.gif"), eps, mp)
    eps_gif(here(__file__, "dbscan_eps.gif"), mp)


if __name__ == "__main__":
    main()
