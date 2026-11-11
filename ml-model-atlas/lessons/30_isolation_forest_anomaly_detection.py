"""
# 30 · Isolation Forest (Anomaly Detection)
**Family:** Unsupervised · Anomaly / outlier detection

## Definition
Isolation Forest detects anomalies by how *easy they are to isolate*. It builds many random trees: pick a random feature, pick a random split value between the min and max,
and recurse until each point sits alone. Anomalies are few and different, so random cuts separate them in **very few splits**; normal points deep inside the crowd need many.

## Intuition
Play "20 questions" with random questions. To single out a faulty machine whose temperature is far from everyone else, one or two cuts suffice. To single out one typical machine
in the middle of a dense cloud you must keep cutting. Short average path length ⇒ anomaly.

## The math
    path length h(x) = number of edges from the root to the leaf that isolates x   (+ c(size) correction for unfinished leaves)
    c(n) = 2(ln(n−1) + 0.5772) − 2(n−1)/n         expected path length in an unsuccessful BST search (the normaliser)
    anomaly score  s(x) = 2^{ −E[h(x)] / c(ψ) }    ->  near 1: anomaly,  well below 0.5: normal,  ≈ 0.5: no clear anomaly
    Each tree uses a small random subsample ψ (e.g. 256), which makes it fast (linear time) and robust to "swamping".
    Related: LOF (local density), One-Class SVM, autoencoder reconstruction error (lesson 20), Gaussian mixture density (lesson 26).

## Practical tips
- Choose the score threshold from the expected contamination rate or business cost; no labels needed.
- Axis-parallel cuts can create artefacts (Extended Isolation Forest fixes it). Works well on tabular sensor/finance/log features.

## What the animations show
1. `30_isolation_forest_anomaly_detection.gif`: random cuts isolating an **outlier** (left, ~3 cuts) vs a **normal point** (right, many cuts). The red region is the isolated cell.
2. `30_isolation_forest_anomaly_detection_forest.gif`: adding trees one by one; the anomaly-score map stabilises and the true faults light up.

## Try this
- Add a tight second cluster of 12 anomalies. Does "few and different" still hold? (masking)
- Compare the AUC of Isolation Forest vs a single Gaussian fit.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def cfun(n): return 0. if n <= 1 else (1. if n == 2 else 2 * (np.log(n - 1) + .5772) - 2 * (n - 1) / n)


def build(X, rng, depth=0, maxd=8):
    n = len(X)
    if depth >= maxd or n <= 1 or (X.max(0) == X.min(0)).all(): return {"size": n}
    f = rng.integers(X.shape[1]); lo, hi = X[:, f].min(), X[:, f].max()
    if lo == hi: return {"size": n}
    t = rng.uniform(lo, hi); m = X[:, f] < t
    return {"f": f, "t": t, "l": build(X[m], rng, depth + 1, maxd), "r": build(X[~m], rng, depth + 1, maxd)}


def path(tree, X):
    out = np.zeros(len(X))
    def rec(node, ids, d):
        if "size" in node: out[ids] = d + cfun(node["size"]); return
        m = X[ids, node["f"]] < node["t"]; rec(node["l"], ids[m], d + 1); rec(node["r"], ids[~m], d + 1)
    rec(tree, np.arange(len(X)), 0); return out


def isolate_steps(X, target, rng):
    """Sequence of (cut, bounds) while repeatedly cutting at random and keeping the side with the target point."""
    idx = np.arange(len(X)); b = [X[:, 0].min() - .5, X[:, 0].max() + .5, X[:, 1].min() - .5, X[:, 1].max() + .5]; steps = []
    while len(idx) > 1 and len(steps) < 25:
        f = rng.integers(2); lo, hi = X[idx, f].min(), X[idx, f].max()
        if lo == hi: break
        t = rng.uniform(lo, hi); old = list(b); keep = X[idx, f] < t if X[target, f] < t else X[idx, f] >= t; idx = idx[keep]
        if f == 0: (b.__setitem__(1, t) if X[target, 0] < t else b.__setitem__(0, t))
        else: (b.__setitem__(3, t) if X[target, 1] < t else b.__setitem__(2, t))
        steps.append((f, t, old, list(b)))
    return steps


def main(gif, quick):
    X, y = data.sensor_anomalies(300, 14, seed=1); out_i = int(np.where(y == 1)[0][0]); norm_i = int(np.argmin(np.linalg.norm(X, axis=1)))
    so = [len(isolate_steps(X, out_i, np.random.default_rng(s))) for s in range(200)]; sn = [len(isolate_steps(X, norm_i, np.random.default_rng(s))) for s in range(200)]
    print(f"  average random cuts needed to isolate: outlier {np.mean(so):.1f}  vs  central normal point {np.mean(sn):.1f}")
    A = isolate_steps(X, out_i, np.random.default_rng(3)); B = isolate_steps(X, norm_i, np.random.default_rng(3)); N = max(len(A), len(B)) + 1; fig, (a1, a2) = viz.subplots(1, 2, w=12, h=4.8)
    lim = (X[:, 0].min() - .5, X[:, 0].max() + .5, X[:, 1].min() - .5, X[:, 1].max() + .5)

    def panel(ax, steps, i, k, name):
        ax.clear(); ax.scatter(*X.T, s=10, color="#bbb"); k = min(k, len(steps))
        for f, t, old, new in steps[:k]: (ax.plot([t, t], [old[2], old[3]], "k", lw=1) if f == 0 else ax.plot([old[0], old[1]], [t, t], "k", lw=1))
        if k:
            b = steps[k - 1][3]; ax.add_patch(viz.plt.Rectangle((b[0], b[2]), b[1] - b[0], b[3] - b[2], color=viz.PALETTE[3], alpha=.15))
        ax.scatter(*X[i], s=140, marker="*", color=viz.PALETTE[3], edgecolor="k", zorder=6); ax.set(title=f"{name}: {k} random cuts{' -> isolated!' if k == len(steps) else ''}", xlim=lim[:2], ylim=lim[2:])
    viz.animate(fig, lambda k: (panel(a1, A, out_i, k, "Outlier"), panel(a2, B, norm_i, k, "Normal point")), range(N), gif, fps=2)
    # ---- the forest
    rng = np.random.default_rng(0); xx, yy, G = viz.grid(X, n=60, pad=1); T = 20 if quick else 60; trees = [build(X[rng.choice(len(X), 128, replace=False)], rng) for _ in range(T)]
    PG = np.array([path(t, G) for t in trees]); PX = np.array([path(t, X) for t in trees]); c = cfun(128); fig, (a1, a2) = viz.subplots(1, 2, w=12, h=4.8)
    sc = lambda P: 2 ** (-P.mean(0) / c); sx = sc(PX); thr = np.sort(sx)[-len(np.where(y == 1)[0])]; print(f"  forest of {T}: top-{(y == 1).sum()} scores catch {(y[sx >= thr] == 1).mean():.0%} true faults")

    def update(m):
        a1.clear(); a2.clear(); s = 2 ** (-PG[:m].mean(0) / c); a1.imshow(s.reshape(xx.shape), origin="lower", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap="magma_r", vmin=.35, vmax=.8, aspect="auto"); a1.grid(False)
        a1.scatter(*X[y == 0].T, s=10, color="white", edgecolor="k", linewidth=.3); a1.scatter(*X[y == 1].T, s=40, marker="x", color=viz.PALETTE[3]); a1.set(title=f"Anomaly score map with {m} tree(s) (dark = anomalous)")
        sm = 2 ** (-PX[:m].mean(0) / c); a2.hist(sm[y == 0], bins=25, range=(.3, .9), alpha=.7, color=viz.PALETTE[0], label="normal"); a2.hist(sm[y == 1], bins=25, range=(.3, .9), alpha=.8, color=viz.PALETTE[3], label="true faults"); a2.set(title="Score distribution", xlabel="anomaly score"); a2.legend()
    viz.animate(fig, update, sorted({1, 2, 3, 5, 8, 12, 20, 30, 45, 60} & set(range(1, T + 1))), gif.replace(".gif", "_forest.gif"), fps=2)


if __name__ == "__main__":
    run(main, __file__)
