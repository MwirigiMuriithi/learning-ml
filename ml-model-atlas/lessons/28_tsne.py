"""
# 28 · t-SNE (Non-linear Embedding for Visualization)
**Family:** Unsupervised · Dimensionality reduction (visualization)

## Definition
t-SNE maps high-dimensional points to 2-D so that **neighbours stay neighbours**. It converts distances into neighbour probabilities in both spaces and moves the 2-D points
until the two probability distributions match (minimizing KL divergence).

## Intuition
Imagine points repelling each other in the plane, but each pair also attached by a spring whose strength is how similar they were in the original space.
Similar points get pulled together into clusters, dissimilar ones pushed apart. Heavy-tailed (Student-t) similarities in 2-D give clusters room to separate ("crowding problem").

## The math
    High-D:  p_{j|i} = exp(−||xᵢ−xⱼ||²/2σᵢ²) / Σ_k exp(−||xᵢ−x_k||²/2σᵢ²) ,   P = (p_{j|i} + p_{i|j}) / 2n
             σᵢ chosen per point by binary search so the effective neighbour count = perplexity (≈ 5–50)
    Low-D:   q_ij = (1 + ||yᵢ−yⱼ||²)⁻¹ / Σ_{k≠l} (1 + ||y_k−y_l||²)⁻¹
    Cost:    KL(P||Q) = Σ p_ij log(p_ij/q_ij)          gradient:  4 Σⱼ (p_ij − q_ij)(yᵢ−yⱼ)(1+||yᵢ−yⱼ||²)⁻¹
    Tricks: early exaggeration (×12 for 100 iterations), momentum, adaptive gains.

## Practical tips
- For **visualization only**: distances between clusters and cluster sizes are not meaningful; run with several perplexities and seeds. UMAP is faster and preserves more global structure.
- Do not use t-SNE coordinates as features for a model.

## What the animation shows
240 noisy 8×8 stroke images (64-D, 4 classes). Left: PCA's 2-D linear projection (static) for contrast. Middle: the t-SNE embedding unfolding from a random blob into 4 clusters. Right: the KL divergence falling.

## Try this
- Try perplexity 5 and 100.
- Run t-SNE on `data.mall_customers` plus 5 random noise columns.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def sqd(X): s = (X ** 2).sum(1); return np.maximum(s[:, None] + s[None] - 2 * X @ X.T, 0)


def affinities(X, perp=25):
    D = sqd(X); n = len(X); P = np.zeros((n, n)); target = np.log(perp)
    for i in range(n):
        lo, hi, beta = 0., np.inf, 1.; d = np.delete(D[i], i)
        for _ in range(50):
            p = np.exp(-(d - d.min()) * beta); sp = p.sum(); H = np.log(sp) + beta * ((d - d.min()) * p).sum() / sp; p /= sp
            if abs(H - target) < 1e-4: break
            if H > target: lo = beta; beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else: hi = beta; beta = (beta + lo) / 2
        P[i, np.arange(n) != i] = p
    P = (P + P.T) / (2 * n); return np.maximum(P, 1e-12)


def tsne(X, iters, perp=25, seed=0, every=8):
    n = len(X); P = affinities(X, perp); rng = np.random.default_rng(seed); Y = rng.normal(0, 1e-2, (n, 2)); iY = np.zeros_like(Y); gains = np.ones_like(Y); rec = []
    for it in range(iters):
        ex = 12. if it < 100 else 1.; mom = .5 if it < 100 else .8; num = 1 / (1 + sqd(Y)); np.fill_diagonal(num, 0); Q = np.maximum(num / num.sum(), 1e-12)
        PQ = (ex * P - Q) * num; grad = 4 * (PQ.sum(1)[:, None] * Y - PQ @ Y)
        gains = np.where(np.sign(grad) != np.sign(iY), gains + .2, gains * .8).clip(.01); iY = mom * iY - 200 * gains * grad; Y = Y + iY; Y -= Y.mean(0)
        if it % every == 0 or it == iters - 1: rec.append((Y.copy(), float((P * np.log(P / Q)).sum())))
    return rec


def main(gif, quick):
    Ximg, y = data.shapes(60, seed=0); X = Ximg.reshape(len(y), -1); iters = 160 if quick else 450; rec = tsne(X, iters); Xc = X - X.mean(0); Z = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][:2].T
    print(f"  KL divergence {rec[0][1]:.2f} -> {rec[-1][1]:.2f}")
    fig, (a1, a2, a3) = viz.subplots(1, 3, w=14, h=4.4); kl = [r[1] for r in rec]

    def update(i):
        for a in (a1, a2, a3): a.clear()
        viz.scatter(a1, Z, y, s=14); a1.set(title="PCA (linear) for comparison")
        Y = rec[i][0]; viz.scatter(a2, Y, y, s=14); lim = np.abs(Y).max() * 1.1; a2.set(title=f"t-SNE iteration {min(i * 8, iters)}", xlim=(-lim, lim), ylim=(-lim, lim))
        for c in range(4): a2.scatter([], [], color=viz.PALETTE[c], label=data.SHAPE_NAMES[c], s=20)
        a2.legend(loc="upper right", fontsize=7); a3.plot(kl[:i + 1], color=viz.PALETTE[3]); a3.set(xlim=(0, len(rec)), ylim=(0, max(kl) * 1.05), title="KL divergence (lower = better match)", xlabel="snapshot (×8 iterations)")
    viz.animate(fig, update, viz.pick(len(rec), 12 if quick else 50), gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
