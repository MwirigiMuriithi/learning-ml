"""
Lesson 27 - t-SNE (t-distributed Stochastic Neighbour Embedding), from scratch

Problem story: a streaming service describes each of 300 subscribers with 30 numbers (viewing hours per genre, time of day,
device, ...).  They fall into 6 taste segments that are all roughly equally far from each other, and 6 of the 30 columns are
"volume" measures (total hours watched, account age...) that vary a lot but say nothing about taste.  Analysts want a 2-D map
where subscribers with similar taste sit together.  Six equidistant groups cannot be laid out flat by any *linear* projection,
so PCA overlaps them; t-SNE, which preserves *neighbourhoods*, can.

Implemented (NumPy only):
  * input affinities P: a Gaussian around every point whose width is found by binary search to match a target perplexity
  * output affinities Q: a heavy-tailed Student-t (1 degree of freedom) kernel
  * gradient descent on KL(P || Q) with momentum, adaptive gains and "early exaggeration"
  * neighbour-preservation score and PCA for comparison

Animated:
  tsne_optimize.gif   the map unfolds from a point at the origin; the KL objective falls; Q becomes like P
  tsne_perplexity.gif the final map for perplexity 2 ... 100 next to PCA, with the neighbour-preservation score

Run:  python tsne.py        (about 40 s)
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ------------------------------------------------------------------- data
def subscribers(n=300, d=30, k=6, seed=11):
    """k taste segments in d dims.  The last 6 columns are high-variance 'volume' nuisance features."""
    r = rng_(seed)
    taste_dims = d - 6
    centers = r.normal(0, 1, (k, taste_dims))
    centers *= 7.5 / np.linalg.norm(centers, axis=1, keepdims=True)      # equal norms 7.5 -> segments ~10.6 apart
    z = r.integers(0, k, n)
    taste = centers[z] + r.normal(0, 1.0, (n, taste_dims))
    volume = r.normal(0, 3.0, (n, 6))                                                   # irrelevant but dominant variance
    X = np.hstack([taste, volume])
    return X, z


def pairwise_sq(X):
    s = (X ** 2).sum(1)
    return np.maximum(s[:, None] - 2 * X @ X.T + s[None, :], 0)


# ------------------------------------------------------------ the algorithm
def conditional_p(D2, perplexity, tol=1e-5, iters=60):
    """Row i: p_{j|i} = exp(-d2_ij * beta_i) / sum_k exp(-d2_ik * beta_i).  beta_i = 1/(2 sigma_i^2) is found by binary
    search so that the entropy H(P_i) = log(perplexity): 'each point has about `perplexity` effective neighbours'."""
    n = len(D2)
    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        d = np.delete(D2[i], i)
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(iters):
            w = np.exp(-(d - d.min()) * beta)
            sw = w.sum()
            p = w / sw
            H = -(p * np.log(np.maximum(p, 1e-300))).sum()
            diff = H - target
            if abs(diff) < tol:
                break
            if diff > 0:                                   # entropy too high -> Gaussian too wide -> raise beta
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
        P[i, np.arange(n) != i] = p
    return P


def joint_p(X, perplexity):
    P = conditional_p(pairwise_sq(X), perplexity)
    P = (P + P.T) / (2 * len(X))                           # symmetrise so that sum P = 1
    return np.maximum(P, 1e-12)


def q_matrix(Y):
    num = 1.0 / (1.0 + pairwise_sq(Y))                     # Student-t kernel, one degree of freedom
    np.fill_diagonal(num, 0)
    return num, np.maximum(num / num.sum(), 1e-12)


def kl(P, Q):
    m = P > 1e-12
    return float((P[m] * np.log(P[m] / Q[m])).sum())


def tsne(X, perplexity=30, n_iter=750, lr=200.0, exag=12.0, exag_iters=250, seed=0, snap_at=(), P=None):
    """Returns Y (n,2), KL history, and {iteration: Y snapshot}."""
    r = rng_(seed)
    n = len(X)
    if P is None:
        P = joint_p(X, perplexity)
    Y = r.normal(0, 1e-4, (n, 2))
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    hist, snaps = [], {}
    for it in range(n_iter + 1):
        num, Q = q_matrix(Y)
        if it in snap_at:
            snaps[it] = Y.copy()
        hist.append(kl(P, Q))
        if it == n_iter:
            break
        Pe = P * (exag if it < exag_iters else 1.0)        # early exaggeration: clusters form easily when attraction is stronger
        PQ = (Pe - Q) * num
        grad = 4 * (PQ.sum(1)[:, None] * Y - PQ @ Y)       # dKL/dy_i = 4 sum_j (p_ij - q_ij)(y_i - y_j)/(1 + |y_i - y_j|^2)
        mom = 0.5 if it < 250 else 0.8
        same = (grad > 0) == (iY > 0)                      # adaptive gains (delta-bar-delta): speed up while the sign is steady
        gains = np.where(same, gains * 0.8, gains + 0.2)
        gains = np.maximum(gains, 0.01)
        iY = mom * iY - lr * gains * grad
        Y = Y + iY
        Y -= Y.mean(0)
    return Y, np.array(hist), snaps


def pca2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def neighbour_preservation(X, Y, k=10):
    """Mean fraction of each point's k nearest neighbours in the original space that are still among its k nearest in 2-D."""
    DX, DY = pairwise_sq(X), pairwise_sq(Y)
    np.fill_diagonal(DX, np.inf); np.fill_diagonal(DY, np.inf)
    nx = np.argsort(DX, 1)[:, :k]
    ny = np.argsort(DY, 1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nx, ny)]))


def knn_label_acc(Y, z, k=10):
    D = pairwise_sq(Y); np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, 1)[:, :k]
    pred = np.array([np.bincount(z[row]).argmax() for row in nn])
    return float((pred == z).mean())


# ------------------------------------------------------------ plotting
def draw_map(ax, Y, z, title, pad=0.08):
    for c in np.unique(z):
        m = z == c
        ax.scatter(Y[m, 0], Y[m, 1], s=22, color=CAT[c % 8], edgecolor=SURFACE, linewidth=0.4, label=f"segment {c}")
    lo, hi = np.percentile(Y, 1, axis=0), np.percentile(Y, 99, axis=0)      # robust window: a few stragglers may sit off-screen
    span = np.maximum(hi - lo, 1e-9)
    ax.set_xlim(lo[0] - pad * span[0], hi[0] + pad * span[0])
    ax.set_ylim(lo[1] - pad * span[1], hi[1] + pad * span[1])
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def order_by(z):
    return np.argsort(z, kind="stable")


# ------------------------------------------------------------ animation 1
def optimize_gif(path, X, z, perplexity=30):
    n = len(X)
    P = joint_p(X, perplexity)
    n_iter = 750
    show = sorted(set(np.unique(np.round(np.geomspace(1, n_iter + 1, 26)).astype(int) - 1)) | {0, 5, 20, 60, 120, 200, 249, 250, 260, 280, 300, 400, 500, n_iter})
    Y, hist, snaps = tsne(X, perplexity, n_iter=n_iter, snap_at=set(show), P=P)
    o = order_by(z)
    Pimg = np.log10(P[np.ix_(o, o)] / P.max() + 1e-4)
    with Animator(path, fps=3, hold=3.5) as anim:
        fig, (a1, a2, a3, a4) = new_fig(1, 4, figsize=(13, 4.8), title=f"t-SNE: perplexity {perplexity}, from a point at the origin to a map",
                                         gridspec_kw={"width_ratios": [1.55, 1.0, 0.75, 0.75]})
        for it in show:
            for a in (a1, a2, a3, a4):
                a.clear()
            Ys = snaps[it]
            draw_map(a1, Ys, z, f"Map after {it} iterations")
            a2.plot(np.arange(it + 1), hist[: it + 1], color=BLUE)
            a2.axvspan(0, 250, color=YELLOW, alpha=0.15, lw=0)
            a2.set(xlim=(0, n_iter), ylim=(0, hist[1:].max() * 1.05), title="KL(P || Q)", xlabel="iteration")
            a2.text(125, hist[1:].max() * 0.93, "early\nexaggeration", ha="center", fontsize=8, color=INK2)
            _, Q = q_matrix(Ys)
            Qimg = np.log10(Q[np.ix_(o, o)] / Q.max() + 1e-4)
            a3.imshow(Pimg, cmap="Blues", vmin=-4, vmax=0, interpolation="nearest"); a3.set_title("target P"); a3.set_xticks([]); a3.set_yticks([]); a3.grid(False)
            a4.imshow(Qimg, cmap="Blues", vmin=-4, vmax=0, interpolation="nearest"); a4.set_title("current Q"); a4.set_xticks([]); a4.set_yticks([]); a4.grid(False)
            if it == 0:
                msg = "Start: all points almost on top of each other; Q is nearly uniform"
            elif it < 250:
                msg = f"Early exaggeration (P x 12): attraction dominates, so the cloud stays compressed and groups can reorganise freely inside it. KL {hist[it]:.2f}"
            elif it == 250:
                msg = "Exaggeration switched off: repulsion takes over and the compressed cloud starts to expand"
            elif it < n_iter:
                msg = f"Expansion: segments separate and spread so that Q matches P (heat maps sorted by segment: diagonal blocks = neighbours). KL {hist[it]:.2f}"
            else:
                msg = f"Final map: KL {hist[it]:.2f}; neighbour preservation {neighbour_preservation(X, Ys):.2f}"
            caption(fig, msg)
            anim.grab(fig)
    return Y, hist


# ------------------------------------------------------------ animation 2
def perplexity_gif(path, X, z):
    perps = [2, 5, 10, 20, 30, 50, 100]
    pca = pca2(X)
    pres = neighbour_preservation(X, pca)
    runs = {}
    for p in perps:
        Y, h, _ = tsne(X, p, n_iter=500)
        runs[p] = (Y, h[-1], neighbour_preservation(X, Y), knn_label_acc(Y, z))
    print("  perplexity sweep (KL, neighbour preservation, 10-NN segment accuracy in the 2-D map):")
    for p in perps:
        print(f"     perplexity {p:3d}: KL {runs[p][1]:.2f}  preservation {runs[p][2]:.3f}  segment acc {runs[p][3]:.3f}")
    print(f"     PCA            : preservation {pres:.3f}  segment acc {knn_label_acc(pca, z):.3f}")
    with Animator(path, fps=1.4, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Perplexity = how many neighbours each point 'looks at'")
        for i, p in enumerate(perps):
            for a in (a1, a2, a3):
                a.clear()
            draw_map(a1, pca, z, "PCA (2 components)")
            Y, klv, pres_t, acc = runs[p]
            draw_map(a2, Y, z, f"t-SNE, perplexity {p}")
            xs = perps[: i + 1]
            a3.plot(xs, [runs[q][2] for q in xs], "o-", color=BLUE, label="t-SNE")
            a3.axhline(pres, color=ORANGE, ls="--", lw=1.5)
            a3.text(perps[-1], pres + 0.015, "PCA", color=ORANGE, ha="right", fontsize=9)
            a3.set(xscale="log", xlim=(1.6, 130), ylim=(0, 0.6), title="Neighbour preservation (k = 10)", xlabel="perplexity")
            a3.legend(loc="upper left")
            if p <= 5:
                msg = f"Perplexity {p}: very local view; a segment breaks into many small islands"
            elif p <= 30:
                msg = f"Perplexity {p}: segments appear as clean clusters (10-NN segment accuracy {acc:.2f})"
            else:
                msg = f"Perplexity {p}: more global view; clusters stay but fine structure is smoothed (accuracy {acc:.2f})"
            caption(fig, msg + f"; PCA accuracy {knn_label_acc(pca, z):.2f}")
            anim.grab(fig)


# ------------------------------------------------------------ main
def main():
    X, z = subscribers()
    print("Lesson 27: t-SNE")
    P = joint_p(X, 30)
    D2 = pairwise_sq(X)
    Pc = conditional_p(D2, 30)
    H = -(Pc * np.log(np.maximum(Pc, 1e-300))).sum(1)
    print(f"  perplexity calibration: entropy of every row in [{H.min():.4f}, {H.max():.4f}] vs log(30) = {np.log(30):.4f}; sum P = {P.sum():.4f}")
    Y, hist = optimize_gif(here(__file__, "tsne_optimize.gif"), X, z)
    pca = pca2(X)
    print(f"  final KL {hist[-1]:.3f}; neighbour preservation t-SNE {neighbour_preservation(X, Y):.3f} vs PCA {neighbour_preservation(X, pca):.3f}; "
          f"10-NN segment accuracy t-SNE {knn_label_acc(Y, z):.3f} vs PCA {knn_label_acc(pca, z):.3f}")
    perplexity_gif(here(__file__, "tsne_perplexity.gif"), X, z)
    try:
        from sklearn.manifold import TSNE, trustworthiness
        sk = TSNE(2, perplexity=30, init="random", random_state=0, max_iter=750).fit(X)
        print(f"  [sklearn] TSNE KL {sk.kl_divergence_:.3f} (ours {hist[-1]:.3f}; different conventions/optimiser); trustworthiness "
              f"sklearn {trustworthiness(X, sk.embedding_, n_neighbors=10):.3f} vs ours {trustworthiness(X, Y, n_neighbors=10):.3f} vs PCA {trustworthiness(X, pca, n_neighbors=10):.3f}")
    except Exception as ex:
        print("  (sklearn check skipped:", ex, ")")


if __name__ == "__main__":
    main()
