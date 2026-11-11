"""
# 34 · Matrix Factorization & Recommender Systems
**Family:** Unsupervised/supervised hybrid · Collaborative filtering

## Definition
A recommender predicts ratings people *haven't given yet*. **Matrix factorization** approximates the (mostly empty) user×item rating matrix R by the product of two thin matrices:
user vectors U and item vectors V. The rating of user u for item i is predicted by the dot product u·v_i. Each dimension is a learned "taste" (e.g. action vs romance).

## Intuition
Users and items live in the same small "taste space". If your vector points the same way as a movie's vector, you'll probably like it. We never tell the model what the dimensions mean:
fitting only the *observed* ratings makes similar users and similar items drift together. Here the true taste space is 2-D, so we can literally watch users and movies organise on a map.

## The math
    R̂_ui = μ + U_u · V_i                       (add user/item biases b_u + b_i in practice)
    minimize over U, V:   Σ_{(u,i) observed} (R_ui − R̂_ui)²  +  λ (||U||² + ||V||²)
    SGD:      e = R_ui − U_u·V_i ;  U_u ← U_u + η (e V_i − λ U_u) ;  V_i ← V_i + η (e U_u − λ V_i)
    Alternating least squares (ALS) solves each side in closed form and parallelizes well.
    Evaluate on **held-out ratings** (RMSE). Cold-start problem: no history for new users/items → use content features.
    Beyond: implicit feedback (clicks), factorization machines, two-tower neural networks, transformers for sequences.

## What the animation shows
Panels: (1) the observed ratings (grey = missing), (2) the model's completed matrix as it trains, (3) the 2-D taste space with users (circles) and movies (triangles),
(4) train/test RMSE. The completed matrix fills in plausible values for the grey cells, and test RMSE drops without seeing those entries.

## Try this
- Use 1 or 5 latent factors and compare held-out RMSE (under/overfitting!).
- Increase λ and see how embeddings shrink toward the centre.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from mlatlas import data, viz
from mlatlas.cli import run


def main(gif, quick):
    R, mask, Ut, Vt = data.ratings(40, 30, seed=0, obs=.4); rng = np.random.default_rng(1); test = mask & (rng.random(R.shape) < .2); train = mask & ~test
    k = 2; U = rng.normal(0, .3, (40, k)); V = rng.normal(0, .3, (30, k)); mu = R[train].mean(); lr, lam = .03, .05; E = 60 if quick else 220; ui = np.argwhere(train); rm = lambda m: np.sqrt(np.mean((mu + (U @ V.T)[m] - R[m]) ** 2)); H = []; snaps = {}
    for e in range(E + 1):
        if e % (6 if quick else 4) == 0: H.append((e, rm(train), rm(test))); snaps[e] = (U.copy(), V.copy(), len(H))
        for u, i in ui[rng.permutation(len(ui))]:
            err = R[u, i] - mu - U[u] @ V[i]; Uu = U[u].copy(); U[u] += lr * (err * V[i] - lam * U[u]); V[i] += lr * (err * Uu - lam * V[i])
    print(f"  train RMSE {H[0][1]:.2f} -> {H[-1][1]:.2f};  held-out RMSE {H[0][2]:.2f} -> {H[-1][2]:.2f}   (predicting the mean would give {np.sqrt(np.mean((R[test] - mu) ** 2)):.2f})")
    fig = plt.figure(figsize=(15, 4.2), constrained_layout=True); gs = fig.add_gridspec(1, 4); ax = [fig.add_subplot(gs[0, i]) for i in range(4)]; obs = np.where(train, R, np.nan)

    def update(e):
        for a in ax: a.clear()
        U_, V_, kk = snaps[e]; ax[0].imshow(obs, cmap="viridis", vmin=1, vmax=5, aspect="auto"); ax[0].set(title="Observed ratings (blank = unknown)", xlabel="movies", ylabel="users"); ax[0].grid(False)
        ax[1].imshow(np.clip(mu + U_ @ V_.T, 1, 5), cmap="viridis", vmin=1, vmax=5, aspect="auto"); ax[1].set(title=f"Model's completed matrix (epoch {e})", xlabel="movies"); ax[1].grid(False)
        ax[2].scatter(*U_.T, s=26, color=viz.PALETTE[0], label="users"); ax[2].scatter(*V_.T, s=34, marker="^", color=viz.PALETTE[1], label="movies"); ax[2].axhline(0, color="gray", lw=.5); ax[2].axvline(0, color="gray", lw=.5); ax[2].set(title="Learned 2-D taste space", xlim=(-2, 2), ylim=(-2, 2)); ax[2].legend(loc="upper right")
        h = np.array(H[:kk]); ax[3].plot(h[:, 0], h[:, 1], color=viz.PALETTE[2], label="train"); ax[3].plot(h[:, 0], h[:, 2], color=viz.PALETTE[3], label="held-out"); ax[3].set(xlim=(0, E), ylim=(0, 1.5), title="RMSE", xlabel="epoch"); ax[3].legend()
    viz.animate(fig, update, sorted(snaps), gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
