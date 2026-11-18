"""
Lesson 28 - Self-Organizing Map (Kohonen map), from scratch

Problem story
  (1) "Unfolding": a 2-D lattice of neurons is stretched over a blob-shaped cloud of
      points (think: GPS pings around a city city centre). We watch the grid unfold.
  (2) "Customer map": 600 customers described by 4 numbers (recency, frequency,
      spend, tenure) hide 4 segments. A SOM squeezes them onto a 12x12 map; the
      U-matrix (distances between neighbouring neurons) reveals segment borders.

What is animated
  GIF 1 (som_unfolding.gif): grid lines in data space, the Gaussian neighbourhood on the
        lattice around the current winner, and quantisation error + radius + lr curves.
  GIF 2 (som_customer_map.gif): U-matrix emerging, a component plane (spend) and the
        segment map (majority segment per neuron) while training progresses.

Run:  python som.py      -> writes som_unfolding.gif and som_customer_map.gif
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------- the model
class SOM:
    """Kohonen map: a (rows x cols) lattice of neurons, each owning a weight vector in data space."""

    def __init__(self, rows, cols, dim, seed=0, init_data=None):
        self.rows, self.cols, self.dim = rows, cols, dim
        r = rng_(seed)
        # lattice coordinates of every neuron (this is the *map* space, fixed forever)
        gy, gx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        self.grid = np.c_[gy.ravel(), gx.ravel()].astype(float)         # (rows*cols, 2)
        # weight vectors live in *data* space; init: small random blob around the data mean
        if init_data is None:
            self.W = r.normal(0, 0.1, (rows * cols, dim))
        else:
            self.W = init_data.mean(0) + r.normal(0, 0.05 * init_data.std(0), (rows * cols, dim))
        self.rng = r

    def bmu(self, x):
        """Best-matching unit: neuron whose weight vector is closest to x."""
        return int(np.argmin(((self.W - x) ** 2).sum(1)))

    def neighbourhood(self, b, sigma):
        """h(b, j) = exp(-d_lattice(b, j)^2 / (2 sigma^2)) - Gaussian bump on the LATTICE around the winner."""
        d2 = ((self.grid - self.grid[b]) ** 2).sum(1)
        return np.exp(-d2 / (2 * sigma ** 2))

    def fit(self, X, n_iters=3000, lr0=0.5, lr1=0.02, sigma0=None, sigma1=0.6, snapshot_every=None):
        """Online training. Each step: pick a sample, find the BMU, pull the BMU and its lattice
        neighbours toward the sample:  w_j <- w_j + lr(t) * h(b,j; sigma(t)) * (x - w_j).
        lr and sigma decay exponentially:  v(t) = v0 * (v1/v0)^(t/T)."""
        sigma0 = sigma0 or max(self.rows, self.cols) / 2
        hist = {"snap": [], "t": [], "qe": [], "lr": [], "sigma": [], "bmu": [], "x": []}
        for t in range(n_iters):
            frac = t / max(1, n_iters - 1)
            lr = lr0 * (lr1 / lr0) ** frac
            sigma = sigma0 * (sigma1 / sigma0) ** frac
            x = X[self.rng.integers(len(X))]
            b = self.bmu(x)
            h = self.neighbourhood(b, sigma)
            self.W += lr * h[:, None] * (x - self.W)             # the whole algorithm in one line
            if snapshot_every and (t % snapshot_every == 0 or t == n_iters - 1):
                hist["snap"].append(self.W.copy()); hist["t"].append(t)
                hist["qe"].append(self.quantisation_error(X)); hist["lr"].append(lr)
                hist["sigma"].append(sigma); hist["bmu"].append(b); hist["x"].append(x.copy())
        return hist

    # ------------------------------------------------------------ diagnostics
    def quantisation_error(self, X):
        """Mean distance from each sample to its BMU weight (how well the neurons cover the data)."""
        d2 = ((X[:, None, :] - self.W[None]) ** 2).sum(2)
        return float(np.sqrt(d2.min(1)).mean())

    def topographic_error(self, X):
        """Fraction of samples whose 1st and 2nd BMU are NOT lattice neighbours (topology preservation)."""
        d2 = ((X[:, None, :] - self.W[None]) ** 2).sum(2)
        o = np.argsort(d2, 1)[:, :2]
        gd = np.abs(self.grid[o[:, 0]] - self.grid[o[:, 1]]).max(1)   # Chebyshev: 8-neighbourhood
        return float(np.mean(gd > 1))

    def umatrix(self, W=None):
        """Average distance of each neuron's weight to its (up to 8) lattice neighbours."""
        W = self.W if W is None else W
        Wg = W.reshape(self.rows, self.cols, -1)
        U = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                ds = [np.linalg.norm(Wg[i, j] - Wg[a, b])
                      for a in range(max(0, i - 1), min(self.rows, i + 2))
                      for b in range(max(0, j - 1), min(self.cols, j + 2)) if (a, b) != (i, j)]
                U[i, j] = np.mean(ds)
        return U

    def hits(self, X):
        d2 = ((X[:, None, :] - self.W[None]) ** 2).sum(2)
        return d2.argmin(1)


def kmeans_qe(X, k, seed=0, iters=30):
    """Tiny k-means (Lloyd) used only to compare quantisation error with the SOM."""
    r = rng_(seed)
    C = X[r.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
        for j in range(k):
            if np.any(a == j):
                C[j] = X[a == j].mean(0)
    return float(np.sqrt(((X[:, None] - C[None]) ** 2).sum(2).min(1)).mean())


# ---------------------------------------------------------------- data
def make_blob(n=700, seed=0):
    """"City map" GPS pings: a filled, slightly elongated blob (uniform disc, stretched 1.3x)
    with a bit of noise. A square lattice can unfold over it cleanly, which makes the
    unfolding easy to see. (A ring would force the lattice to fold, see lesson pitfalls.)"""
    r = rng_(seed)
    t = r.uniform(0, 2 * np.pi, n)
    rad = 2.0 * np.sqrt(r.uniform(0, 1, n))
    return np.c_[1.3 * rad * np.cos(t), 0.85 * rad * np.sin(t)] + r.normal(0, 0.08, (n, 2))


def make_customers(n=600, seed=1):
    """4 hidden customer segments; features: recency (days since last order, low=good),
    frequency (orders/yr), spend ($/yr), tenure (years)."""
    r = rng_(seed)
    names = ["champions", "newcomers", "at-risk", "bargain hunters"]
    #            recency  freq  spend  tenure
    mu = np.array([[10, 30, 900, 6], [15, 4, 150, 0.5], [120, 12, 500, 5], [40, 25, 180, 3]], float)
    sd = np.array([[6, 5, 150, 1.2], [8, 2, 60, 0.3], [25, 4, 120, 1.2], [15, 5, 50, 1.0]], float)
    y = r.integers(0, 4, n)
    X = mu[y] + r.normal(size=(n, 4)) * sd[y]
    X[:, 0] = X[:, 0].clip(1, None); X[:, 1] = X[:, 1].clip(1, None)
    X[:, 2] = X[:, 2].clip(20, None); X[:, 3] = X[:, 3].clip(0.1, None)
    return X, y, names, ["recency", "frequency", "spend", "tenure"]


# ---------------------------------------------------------------- animations
def draw_grid(ax, W, rows, cols, color=ORANGE):
    Wg = W.reshape(rows, cols, 2)
    for i in range(rows):
        ax.plot(Wg[i, :, 0], Wg[i, :, 1], color=color, linewidth=0.9, alpha=0.9, zorder=2)
    for j in range(cols):
        ax.plot(Wg[:, j, 0], Wg[:, j, 1], color=color, linewidth=0.9, alpha=0.9, zorder=2)
    ax.scatter(W[:, 0], W[:, 1], s=14, color=color, edgecolor=INK, linewidth=0.4, zorder=3)


def animate_unfolding():
    X = make_blob()
    rows = cols = 12
    som = SOM(rows, cols, 2, seed=2, init_data=X)
    T = 4000
    hist = som.fit(X, n_iters=T, lr0=0.6, lr1=0.02, sigma0=6.0, sigma1=0.7, snapshot_every=40)
    snaps = hist["t"]
    sel = frame_steps(len(snaps) - 1, 40)
    print(f"  blob: QE={hist['qe'][-1]:.3f}  topographic error={som.topographic_error(X):.3f}")

    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="A SOM unfolds a 2-D lattice over the data (competitive learning + cooperative neighbours)")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.84, wspace=0.28)
    lim = 3.0
    cmap = LinearSegmentedColormap.from_list("bo", ["#f3f2ee", BLUE])
    with Animator(here(__file__, "som_unfolding.gif"), fps=8, hold=2.5) as anim:
        for k in sel:
            W, t = hist["snap"][k], hist["t"][k]
            # temporarily use the snapshot weights for BMU of the displayed sample
            x = hist["x"][k]
            b = int(np.argmin(((W - x) ** 2).sum(1)))
            sig = hist["sigma"][k]
            a1.clear()
            a1.scatter(X[:, 0], X[:, 1], s=8, color=BLUE, alpha=0.35, linewidth=0, zorder=1)
            draw_grid(a1, W, rows, cols)
            a1.scatter([x[0]], [x[1]], s=90, marker="*", color=YELLOW, edgecolor=INK, zorder=6)
            a1.scatter([W[b, 0]], [W[b, 1]], s=55, color=RED, edgecolor=INK, zorder=6)
            a1.set(title="Data space: the lattice stretched over the data", xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
            a1.text(-lim + 0.1, lim - 0.2, "star = sample, red = its winner", fontsize=8.5, color=INK2, va="top")

            # lattice space: the Gaussian neighbourhood bump around the winner
            gy, gx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
            br, bc = divmod(b, cols)
            h = np.exp(-((gy - br) ** 2 + (gx - bc) ** 2) / (2 * sig ** 2))
            a2.clear()
            a2.imshow(h, cmap=cmap, vmin=0, vmax=1, origin="lower", extent=(-0.5, cols - 0.5, -0.5, rows - 0.5))
            a2.scatter([bc], [br], s=60, color=RED, edgecolor=INK, zorder=5)
            a2.set_xticks(range(0, cols, 2)); a2.set_yticks(range(0, rows, 2)); a2.grid(False)
            a2.set(title=f"Lattice space: neighbourhood h (sigma={sig:.2f})", xlabel="neuron column", ylabel="neuron row")

            a3.clear()
            tt = np.array(snaps[: k + 1])
            a3.plot(tt, hist["qe"][: k + 1], color=BLUE)
            a3.scatter([t], [hist["qe"][k]], s=34, color=ORANGE, zorder=5)
            a3.set(title="Quantisation error (mean distance to winner)", xlabel="training step", ylabel="QE", xlim=(0, T), ylim=(0, max(hist["qe"]) * 1.05))
            a3.text(T * 0.98, max(hist["qe"]) * 0.6, f"lr = {hist['lr'][k]:.3f}\nsigma = {sig:.2f}", ha="right", fontsize=9, color=INK2)
            caption(fig, f"step {t:>4}:  learning rate {hist['lr'][k]:.3f}, radius sigma {sig:.2f}, quantisation error {hist['qe'][k]:.3f}  - big radius early = global ordering, small radius late = fine fitting")
            anim.grab(fig)
    return som, X


def animate_customer_map():
    X, y, names, feats = make_customers()
    Z, mu, sd = standardize(X)
    rows, cols = 12, 12
    som = SOM(rows, cols, 4, seed=3, init_data=Z)
    T = 6000
    hist = som.fit(Z, n_iters=T, lr0=0.5, lr1=0.01, sigma0=6.0, sigma1=0.7, snapshot_every=100)
    print(f"  customers: QE={hist['qe'][-1]:.3f}  topographic error={som.topographic_error(Z):.3f}")
    print(f"  k-means (k=144) QE={kmeans_qe(Z, 144):.3f}  k-means (k=4) QE={kmeans_qe(Z, 4):.3f}")
    sel = frame_steps(len(hist["t"]) - 1, 36)
    seg_cmap = LinearSegmentedColormap.from_list("bo", ["#f3f2ee", ORANGE])
    u_cmap = LinearSegmentedColormap.from_list("u", ["#f3f2ee", BLUE])
    spend_i = feats.index("spend")

    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Customer map: the U-matrix draws borders between segments the SOM discovered on its own")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.2, top=0.84, wspace=0.18)
    final_U = som.umatrix()
    with Animator(here(__file__, "som_customer_map.gif"), fps=8, hold=2.5) as anim:
        for k in sel:
            W, t = hist["snap"][k], hist["t"][k]
            som_tmp = SOM(rows, cols, 4); som_tmp.W = W
            U = som_tmp.umatrix()
            a1.clear(); a1.grid(False)
            a1.imshow(U, cmap=u_cmap, origin="lower", vmin=0, vmax=final_U.max() * 1.1)
            a1.set(title="U-matrix: dark = big gap (border)"); a1.set_xticks([]); a1.set_yticks([])

            a2.clear(); a2.grid(False)
            a2.imshow(W[:, spend_i].reshape(rows, cols) * sd[spend_i] + mu[spend_i], cmap=seg_cmap, origin="lower")
            a2.set(title="Component plane: yearly spend ($)"); a2.set_xticks([]); a2.set_yticks([])

            # segment map: majority segment per neuron (size = number of customers)
            d2 = ((Z[:, None, :] - W[None]) ** 2).sum(2)
            bm = d2.argmin(1)
            a3.clear(); a3.grid(False)
            a3.set(title="Segment map (colour = majority segment)", xlim=(-0.5, cols - 0.5), ylim=(-0.5, rows - 0.5), aspect="equal")
            a3.set_xticks([]); a3.set_yticks([])
            for n in range(rows * cols):
                m = bm == n
                if m.sum() == 0:
                    continue
                maj = np.bincount(y[m], minlength=4).argmax()
                a3.scatter([n % cols], [n // cols], s=18 + 22 * m.sum() ** 0.8, color=CAT[maj], edgecolor=SURFACE, linewidth=0.6, alpha=0.9)
            for c in range(4):
                a3.scatter([], [], color=CAT[c], s=30, label=names[c])
            a3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=7.5, handletextpad=0.2, columnspacing=0.8)
            caption(fig, f"step {t:>4}:  radius sigma {hist['sigma'][k]:.2f}  quantisation error {hist['qe'][k]:.3f}  - labels were never used for training, only to colour the map afterwards")
            anim.grab(fig)
    # purity of the final map
    bm = som.hits(Z)
    pur = 0
    for n in np.unique(bm):
        pur += np.bincount(y[bm == n]).max()
    print(f"  segment purity of the final map: {pur / len(y):.3f}")


def main():
    som, X = animate_unfolding()
    # optional cross-check against k-means on the blob data
    print(f"  blob k-means (k=144) QE={kmeans_qe(X, 144):.3f} (SOM trades a little QE for ordering)")
    animate_customer_map()


if __name__ == "__main__":
    main()
