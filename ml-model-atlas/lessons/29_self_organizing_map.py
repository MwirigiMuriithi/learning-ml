"""
# 29 · Self-Organizing Map (SOM / Kohonen map)
**Family:** Unsupervised · Neural clustering & topology-preserving visualization

## Definition
A SOM is a grid of neurons, each holding a weight vector living in the same space as the data. Training shows data points one by one; the **best-matching unit (BMU)** and its grid
neighbours are pulled toward the point. The result is a 2-D map in which **similar inputs land on nearby cells**: clustering and visualization in one.

## Intuition
Spread a stretchy fishing net over a cloud of points and keep tugging it toward random points. Early on, tugs are big and affect a wide area (the net finds the overall shape); later tugs are small and local (fine tuning).
Here the "points" are random colours: the map organizes itself into a smooth colour atlas, with red next to orange, never next to cyan.

## The math
    BMU:       c = argmin_k ||x − w_k||
    update:    w_k ← w_k + η(t) · h(k, c, t) · (x − w_k)
    neighbourhood  h = exp(−||pos_k − pos_c||² / 2σ(t)²)         η(t) = η₀ e^{−t/τ} ,  σ(t) = σ₀ e^{−t/τ}
    Quantization error = mean ||x − w_BMU||.  U-matrix (distance between neighbouring cells) reveals cluster borders.

## Practical tips
- Good for exploratory visualization of high-dimensional records (customers, sensor profiles, genes). Hex grids, toroidal maps and batch training are common variants.
- Modern alternatives for visualization: UMAP, t-SNE. SOM's advantage is a fixed, reusable grid map.

## What the animation shows
Left: the 14×14 map's weights shown as colours, with the BMU of the current sample ringed and the neighbourhood radius drawn. Right: the roughness (average colour difference between neighbouring cells) collapsing as the map organizes itself.

## Try this
- Feed it 3-D points from `data.mall_customers` plus a third feature; colour cells by the dominant segment.
- Make the grid 4×4 and see it behave like k-means with 16 clusters.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from matplotlib.patches import Circle
from mlatlas import viz
from mlatlas.cli import run


def main(gif, quick):
    rng = np.random.default_rng(0); S = 14; Wt = rng.random((S, S, 3)); X = rng.random((2500, 3)); steps = 800 if quick else 3000; gy, gx = np.mgrid[0:S, 0:S]; qe = []; snaps = {}; tau = steps / 3
    for t in range(steps + 1):
        x = X[t % len(X)]; d = ((Wt - x) ** 2).sum(2); by, bx = np.unravel_index(d.argmin(), d.shape); sig = 6 * np.exp(-t / tau) + .6; eta = .5 * np.exp(-t / (steps / 1.2)) + .02
        h = np.exp(-((gy - by) ** 2 + (gx - bx) ** 2) / (2 * sig ** 2))[..., None]; Wt += eta * h * (x - Wt)
        if t % 10 == 0: qe.append((np.linalg.norm(np.diff(Wt, axis=0), axis=2).mean() + np.linalg.norm(np.diff(Wt, axis=1), axis=2).mean()) / 2)
        if t in set(viz.geo(steps, 14 if quick else 55)): snaps[t] = (Wt.copy(), (bx, by), sig, x.copy(), len(qe))
    print(f"  neighbouring-cell distance (map roughness) {qe[0]:.3f} -> {qe[-1]:.3f}")
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.8)

    def update(t):
        a1.clear(); a2.clear(); W, (bx, by), sig, x, k = snaps[t]; a1.imshow(np.clip(W, 0, 1), origin="lower"); a1.grid(False); a1.add_patch(Circle((bx, by), sig * 1.5, fill=False, edgecolor="white", lw=2)); a1.scatter(bx, by, s=80, color="white", edgecolor="k", zorder=5)
        a1.set(title=f"step {t}: map of colours (white ring = neighbourhood of BMU)"); a1.set_xticks([]); a1.set_yticks([]);
        a2.plot(np.arange(k) * 10, qe[:k], color=viz.PALETTE[3]); a2.set(xlim=(0, steps), ylim=(0, max(qe) * 1.05), title="Map roughness: distance between neighbouring cells", xlabel="step"); a2.scatter([0], [max(qe) * .95], s=600, color=np.clip(x, 0, 1)[None], marker="s", clip_on=False, zorder=5); a2.text(steps * .06, max(qe) * .95, "← current input colour", va="center")
    viz.animate(fig, update, sorted(snaps), gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
