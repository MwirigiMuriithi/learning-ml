"""
# 27 · Principal Component Analysis (PCA)
**Family:** Unsupervised · Dimensionality reduction / feature extraction

## Definition
PCA finds new orthogonal axes (principal components) ordered by how much of the data's **variance** each one captures. Keeping only the first few components compresses
the data with minimal information loss, and makes high-dimensional data visualizable.

## Intuition
Hold a flashlight to a 3-D cloud so its shadow on the wall is as *spread out* as possible: that direction is PC1. Equivalently, PC1 is the line that leaves the smallest
total squared distance from the points to their projections (the residual lines in the animation). The two views are the same by Pythagoras: variance kept + error = total.

## The math
    centre X;  covariance C = XᵀX/(n−1);   eigen-decomposition C v_k = λ_k v_k   (or SVD: X = U S Vᵀ, λ_k = s_k²/(n−1))
    projection  Z = X V_k        reconstruction  X̂ = Z V_kᵀ + mean        explained variance ratio  λ_k / Σλ
    The projection onto unit vector u has variance uᵀCu, maximised by the top eigenvector.
    Relatives: kernel PCA (non-linear), probabilistic PCA/factor analysis, ICA, truncated SVD (LSA), t-SNE/UMAP (visual, non-linear), autoencoders.

## Practical tips
- Standardize variables of different units first. Choose components via the scree plot (cumulative variance ≥ 90–95 %).
- Components are linear mixes: harder to interpret; PCA ignores labels (LDA is the supervised cousin).

## What the animations show
1. `27_pca.gif`: a line rotates through the 2-D cloud. Left: projections and residuals. Right: projected variance (peaks at the PC1 angle) and reconstruction error (lowest there).
2. `27_pca_3d.gif`: a 3-D cloud of body measurements rotates; the best-fit plane (PC1-PC2) and projected points are shown.

## Try this
- Compare PCA of raw vs standardized house-price data.
- Reconstruct the 8×8 shape images from 2, 5, 10 components.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import viz
from mlatlas.cli import run


def pca(X):
    Xc = X - X.mean(0); U, s, Vt = np.linalg.svd(Xc, full_matrices=False); return Vt, s ** 2 / (len(X) - 1)


def main(gif, quick):
    rng = np.random.default_rng(0); h = rng.normal(0, 1, 120); X = np.c_[h * 1.3 + rng.normal(0, .35, 120), h * .9 + rng.normal(0, .55, 120)]; Xc = X - X.mean(0); V, lam = pca(X); a_pc = np.degrees(np.arctan2(V[0, 1], V[0, 0])) % 180
    print(f"  explained variance ratio: {np.round(lam / lam.sum(), 3)}; PC1 angle {a_pc:.1f}°")
    angs = np.linspace(0, 180, 20 if quick else 61)[:-1]; var = lambda a: np.var(Xc @ [np.cos(np.radians(a)), np.sin(np.radians(a))], ddof=1); tot = lam.sum()
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.5); A = np.linspace(0, 180, 181)

    def update(a):
        a1.clear(); a2.clear(); u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]); p = (Xc @ u)[:, None] * u; a1.scatter(*Xc.T, s=14, color=viz.PALETTE[0], zorder=3)
        a1.plot(*(np.array([-4, 4])[:, None] * u).T, "k", lw=2); a1.scatter(*p.T, s=10, color=viz.PALETTE[3], zorder=4)
        for q, r in zip(Xc, p): a1.plot([q[0], r[0]], [q[1], r[1]], color=viz.PALETTE[3], alpha=.25, lw=.8)
        a1.set_aspect("equal"); a1.set(title=f"projection direction {a:.0f}°", xlim=(-4.5, 4.5), ylim=(-3.5, 3.5))
        a2.plot(A, [var(x) for x in A], color=viz.PALETTE[2], label="variance of projection"); a2.plot(A, [tot - var(x) for x in A], color=viz.PALETTE[3], label="reconstruction error (residual variance)")
        a2.axvline(a_pc, color="gray", ls=":"); a2.scatter([a, a], [var(a), tot - var(a)], color="k", zorder=5); a2.set(title="Max variance = min error at the principal axis", xlabel="angle (°)"); a2.legend(loc="center right")
    viz.animate(fig, update, angs, gif, fps=10)
    # ---- 3D
    r = np.random.default_rng(1); n = 200; height = r.normal(170, 9, n); weight = .9 * (height - 170) + 70 + r.normal(0, 4, n); arm = .45 * height + r.normal(0, 2.5, n) - .05 * (weight - 70)
    X3 = np.c_[height, weight, arm]; mu = X3.mean(0); V3, l3 = pca(X3); print(f"  3-D body-measurement data: explained variance {np.round(l3 / l3.sum(), 3)}")
    P = (X3 - mu) @ V3[:2].T @ V3[:2] + mu
    U, W = np.meshgrid(np.linspace(-1, 1, 8) * 3 * np.sqrt(l3[0]), np.linspace(-1, 1, 8) * 3 * np.sqrt(l3[1])); S = mu + U[..., None] * V3[0] + W[..., None] * V3[1]
    fig = viz.plt.figure(figsize=(6.6, 5.4)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*X3.T, s=8, color=viz.PALETTE[0], alpha=.6); ax.scatter(*P.T, s=8, color=viz.PALETTE[3]); ax.plot_surface(S[..., 0], S[..., 1], S[..., 2], alpha=.15, color="gray")
    for k in range(3): ax.plot(*np.c_[mu, mu + 2.5 * np.sqrt(l3[k]) * V3[k]], color="k" if k < 2 else "gray", lw=2)
    ax.set(xlabel="height", ylabel="weight", zlabel="arm length", title="3-D cloud (blue), PC plane, projections (red)")
    viz.animate(fig, lambda a: ax.view_init(18, a), np.linspace(0, 360, 18 if quick else 60, endpoint=False), gif.replace(".gif", "_3d.gif"), fps=10)


if __name__ == "__main__":
    run(main, __file__)
