"""
Lesson 23 - Hierarchical (agglomerative) clustering, from scratch

Problem story: a botanist measured leaf length and width on 45 plants from three species; two of the species are close
relatives.  She wants a *family tree* of the plants (a dendrogram) rather than one fixed number of groups.

Implemented (NumPy only):
  * agglomerative clustering with single / complete / average / Ward linkage, using the Lance-Williams update
    d(k, i+j) = a_i d(k,i) + a_j d(k,j) + b d(i,j) + g |d(k,i) - d(k,j)|
  * the merge table Z = [cluster a, cluster b, height, size]  (same layout as scipy), cutting the tree, leaf ordering
  * cophenetic correlation (how faithfully the tree keeps the original distances)

Animated:
  hier_merge.gif       the 45 plants merge two clusters at a time; the dendrogram grows; merge heights show where to cut
  linkage_compare.gif  4 linkages x 2 data sets, cutting the same tree at k = 8, 7, ... 2 clusters

Run:  python hierarchical.py
"""
import sys, pathlib, itertools

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ------------------------------------------------------------------- data
def plants(seed=3):
    """45 plants: (leaf length cm, leaf width cm).  Species A and B are close relatives, C is distant."""
    r = rng_(seed)
    cen = np.array([[4.0, 2.0], [6.4, 2.6], [9.0, 6.4]])
    std = np.array([[0.45, 0.30], [0.45, 0.30], [0.50, 0.45]])
    z = np.repeat([0, 1, 2], 15)
    X = cen[z] + r.normal(size=(45, 2)) * std[z]
    return X, z


def unequal_blobs(seed=1):
    """Three groups with different sizes and spreads plus a thin bridge of 7 stray points (157 points)."""
    r = rng_(seed)
    sizes = [80, 45, 25]
    cen = np.array([[0.0, 0.0], [4.2, 1.0], [1.6, 4.4]])
    std = [0.9, 0.55, 0.4]
    X = np.vstack([cen[k] + r.normal(size=(s, 2)) * std[k] for k, s in enumerate(sizes)])
    z = np.repeat([0, 1, 2], sizes)
    # a thin "bridge" of 7 stray points between groups 0 and 1 (noise, labelled with the nearer group)
    t = np.linspace(0.3, 0.7, 7)[:, None]
    bridge = (1 - t) * (cen[0] + [1.4, 0.3]) + t * (cen[1] - [1.0, 0.0]) + r.normal(0, 0.05, (7, 2))
    X = np.vstack([X, bridge]); z = np.r_[z, np.where(t[:, 0] < 0.5, 0, 1)]
    return X, z


# ------------------------------------------------------------ the algorithm
def pairwise(X):
    d2 = (X ** 2).sum(1)[:, None] - 2 * X @ X.T + (X ** 2).sum(1)[None, :]
    return np.sqrt(np.maximum(d2, 0))


def linkage(X, method="average"):
    """Agglomerative clustering.  Returns Z with n-1 rows: [id_a, id_b, height, size]; new cluster ids are n, n+1, ...

    Start with n singleton clusters.  Repeat n-1 times: find the closest pair of clusters, merge them, and update the
    distances from the new cluster to all others with the Lance-Williams formula (so we never revisit the raw points).
    """
    n = len(X)
    D = pairwise(X)
    if method == "ward":
        D = D ** 2                                          # Ward's update is exact on *squared* distances
    D = D.copy()
    np.fill_diagonal(D, np.inf)
    ids = list(range(n))                                    # cluster id sitting in each row
    size = np.ones(n)
    active = np.ones(n, bool)
    Z = []
    for step in range(n - 1):
        Dm = np.where(active[:, None] & active[None, :], D, np.inf)
        i, j = np.unravel_index(np.argmin(Dm), Dm.shape)    # closest pair of active clusters
        if i > j:
            i, j = j, i
        h = Dm[i, j]
        ni, nj = size[i], size[j]
        for k in np.flatnonzero(active):
            if k in (i, j):
                continue
            nk = size[k]
            dik, djk = D[i, k], D[j, k]
            if method == "single":
                d = min(dik, djk)
            elif method == "complete":
                d = max(dik, djk)
            elif method == "average":                       # UPGMA: size-weighted mean of the two old distances
                d = (ni * dik + nj * djk) / (ni + nj)
            elif method == "ward":                          # increase in within-cluster sum of squares
                d = ((ni + nk) * dik + (nj + nk) * djk - nk * h) / (ni + nj + nk)
            else:
                raise ValueError(method)
            D[i, k] = D[k, i] = d
        active[j] = False                                   # row i now represents the merged cluster
        Z.append([ids[i], ids[j], np.sqrt(h) if method == "ward" else h, ni + nj])
        ids[i] = n + step
        size[i] = ni + nj
    return np.array(Z)


def cut(Z, k):
    """Labels 0..k-1 obtained by undoing the last k-1 merges."""
    n = len(Z) + 1
    parent = list(range(2 * n - 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for m in range(n - k):                                  # perform only the first n-k merges
        a, b = int(Z[m, 0]), int(Z[m, 1])
        parent[find(a)] = n + m
        parent[find(b)] = n + m
    roots = [find(i) for i in range(n)]
    _, lab = np.unique(roots, return_inverse=True)
    return lab


def members(Z, node):
    """Leaves under a node id."""
    n = len(Z) + 1
    if node < n:
        return [node]
    a, b = int(Z[node - n, 0]), int(Z[node - n, 1])
    return members(Z, a) + members(Z, b)


def leaf_order(Z):
    """Left-to-right order of leaves so that no branches cross."""
    return members(Z, 2 * len(Z))


def cophenetic(Z):
    """Matrix of the height at which every pair of points first joins the same cluster."""
    n = len(Z) + 1
    C = np.zeros((n, n))
    for m in range(n - 1):
        A, B = members(Z, int(Z[m, 0])), members(Z, int(Z[m, 1]))
        for a in A:
            for b in B:
                C[a, b] = C[b, a] = Z[m, 2]
    return C


def match_accuracy(z, lab, K):
    best = 0
    for p in itertools.permutations(range(K)):
        best = max(best, np.mean(np.array(p)[lab] == z))
    return best


# ------------------------------------------------------------ dendrogram drawing
def draw_dendrogram(ax, Z, upto, order, colors=None, hl=None, thresh=None):
    """Draw merges 0..upto (inclusive) as U-shapes.  `colors[m]` colours merge m, `hl` = merge to highlight."""
    n = len(Z) + 1
    xpos = {leaf: i for i, leaf in enumerate(order)}
    ypos = {leaf: 0.0 for leaf in order}
    for m in range(upto + 1):
        a, b = int(Z[m, 0]), int(Z[m, 1])
        xa, xb, ya, yb, h = xpos[a], xpos[b], ypos[a], ypos[b], Z[m, 2]
        c = ORANGE if m == hl else (colors[m] if colors is not None else INK2)
        ax.plot([xa, xa, xb, xb], [ya, h, h, yb], color=c, lw=1.6 if m != hl else 2.6, solid_capstyle="round")
        xpos[n + m] = (xa + xb) / 2
        ypos[n + m] = h
    if thresh is not None:
        ax.axhline(thresh, color=INK, ls="--", lw=1)
    ax.set_xlim(-1, n)
    ax.set_xticks([])
    ax.set_ylim(0, Z[:, 2].max() * 1.08)


# ------------------------------------------------------------ animation 1
def merge_gif(path):
    X, z = plants()
    Z = linkage(X, "average")
    n = len(X)
    order = leaf_order(Z)
    H = Z[:, 2]
    gaps = np.diff(H)
    k_best = 3
    thresh = (H[n - 1 - k_best] + H[n - k_best]) / 2           # between the last 3 and 2 merges... cut giving 3 clusters
    lab3 = cut(Z, k_best)
    cl3 = [CAT[l] for l in range(k_best)]
    # colour a merge by its final 3-cluster when it lies below the cut
    mcol = []
    for m in range(n - 1):
        leaves = members(Z, n + m)
        mcol.append(CAT[lab3[leaves[0]]] if H[m] < thresh else MUTED)

    steps = list(range(0, n - 1))
    shown = sorted(set([0, 1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 34, 38, 40, 41, 42, 43, 44 - 1]) & set(steps))
    if (n - 2) not in shown:
        shown.append(n - 2)
    shown = sorted(set(shown))
    with Animator(path, fps=3, hold=3.0) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 5), title="Agglomerative clustering: merge the two closest clusters, repeat",
                                    gridspec_kw={"width_ratios": [1.0, 1.35, 0.8]})
        for t in shown:
            a1.clear(); a2.clear(); a3.clear()
            final = t == n - 2
            lab = lab3 if final else cut(Z, n - 1 - t)          # clusters after merge t
            # --- left: points, spokes to centroids
            for c in np.unique(lab):
                P = X[lab == c]
                if len(P) > 1:
                    mu = P.mean(0)
                    for p in P:
                        a1.plot([p[0], mu[0]], [p[1], mu[1]], color=MUTED, lw=0.6, alpha=0.7, zorder=1)
            col = np.array([MUTED] * n, dtype=object)
            if final:
                for i in range(n):
                    col[i] = CAT[lab3[i]]
            else:
                A = members(Z, int(Z[t, 0])); B = members(Z, int(Z[t, 1]))
                col[A] = ORANGE; col[B] = AQUA
            a1.scatter(X[:, 0], X[:, 1], c=list(col), s=34, edgecolor=SURFACE, linewidth=0.6, zorder=3)
            a1.set(title="Plants (grey = not in this merge)" if not final else "Cut into 3 clusters", xlabel="leaf length (cm)", ylabel="leaf width (cm)")
            # --- middle: dendrogram
            draw_dendrogram(a2, Z, t, order, colors=mcol if final else None, hl=None if final else t, thresh=thresh if final else None)
            a2.set(title="Dendrogram (average linkage)", ylabel="merge distance")
            # --- right: heights
            a3.plot(np.arange(1, n), H, color=MUTED, lw=1, zorder=1)
            a3.scatter(np.arange(1, n)[: t + 1], H[: t + 1], s=16, color=BLUE, zorder=3)
            a3.scatter([t + 1], [H[t]], s=60, color=ORANGE, zorder=4)
            a3.set(title="Height of each merge", xlabel="merge number", ylabel="distance")
            if final:
                a3.annotate("big jump:\nmerging the last\n3 into 2 is costly", xy=(n - 2, H[n - 3]), xytext=(n * 0.28, H[n - 2] * 0.62), color=INK2, fontsize=8,
                            arrowprops=dict(arrowstyle="->", color=INK2))
            if final:
                msg = (f"Cut at height {thresh:.2f} (dashed): the long vertical gaps mean three natural groups; accuracy against the true species "
                       f"{match_accuracy(z, lab3, 3):.2f}")
            else:
                size_ab = len(A) + len(B)
                msg = f"Merge {t + 1} of {n - 1}: join the two closest clusters (orange + green) at distance {H[t]:.2f} -> a cluster of {size_ab}"
            caption(fig, msg)
            anim.grab(fig)


# ------------------------------------------------------------ animation 2
def compare_gif(path):
    sets = {"unequal blobs": unequal_blobs(), "two moons": make_moons(150, noise=0.05, seed=2)}
    methods = ["single", "complete", "average", "ward"]
    trees = {(s, m): linkage(sets[s][0], m) for s in sets for m in methods}
    Ks = [8, 7, 6, 5, 4, 3, 2]
    with Animator(path, fps=1.5, hold=3.0) as anim:
        fig, axes = new_fig(2, 4, figsize=(13, 6.4), title="Same data, four linkages, cut at K clusters")
        for K in Ks:
            for r, s in enumerate(sets):
                X, z = sets[s]
                for c, m in enumerate(methods):
                    ax = axes[r, c]; ax.clear()
                    lab = cut(trees[(s, m)], K)
                    ax.scatter(X[:, 0], X[:, 1], c=[CAT[l % 8] for l in lab], s=16, edgecolor=SURFACE, linewidth=0.4)
                    acc = match_accuracy(z, lab, K) if K == 2 or K == 3 else None
                    sizes = np.bincount(lab)
                    ax.set_title(f"{m}" + (f"  ({s})" if c == 0 else ""), fontsize=10)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.text(0.02, 0.03, "sizes " + ",".join(map(str, sorted(sizes, reverse=True)[:6])) + ("…" if len(sizes) > 6 else ""),
                            transform=ax.transAxes, fontsize=7.5, color=INK2)
            msg = f"K = {K}: "
            if K == 3:
                msg += "true K for the blobs: single linkage chains through the stray bridge points and misses the groups"
            elif K == 2:
                msg += "true K for the moons: only single linkage follows the curved shapes (the others cut them in half)"
            elif K > 3:
                msg += "more clusters than needed: see what each linkage splits first (single peels off lone points)"
            caption(fig, msg)
            anim.grab(fig)


# ------------------------------------------------------------ main
def main():
    X, z = plants()
    print("Lesson 23: hierarchical clustering")
    try:
        from scipy.cluster.hierarchy import linkage as sl, cophenet, fcluster
        from scipy.spatial.distance import pdist
    except Exception:
        sl = None
    Xb, zb = unequal_blobs()
    Xm, zm = make_moons(150, noise=0.05, seed=2)
    for m in ["single", "complete", "average", "ward"]:
        Z = linkage(X, m)
        C = cophenetic(Z)
        iu = np.triu_indices(len(X), 1)
        cc = np.corrcoef(C[iu], pairwise(X)[iu])[0, 1]
        line = f"  {m:9s} cophenetic corr {cc:.3f}; plants acc(K=3) {match_accuracy(z, cut(Z, 3), 3):.3f}; blobs acc(K=3) {match_accuracy(zb, cut(linkage(Xb, m), 3), 3):.3f}; moons acc(K=2) {match_accuracy(zm, cut(linkage(Xm, m), 2), 2):.3f}"
        if sl is not None:
            Zs = sl(X, method=m)
            err = np.abs(np.sort(Zs[:, 2]) - np.sort(Z[:, 2])).max()
            same = (fcluster(Zs, 3, "maxclust") - 1)
            agree = match_accuracy(same, cut(Z, 3), 3)
            line += f" | scipy: max height diff {err:.1e}, 3-cluster agreement {agree:.3f}, cophenet {cophenet(Zs, pdist(X))[0]:.3f}"
        print(line)
    merge_gif(here(__file__, "hier_merge.gif"))
    compare_gif(here(__file__, "linkage_compare.gif"))


if __name__ == "__main__":
    main()
