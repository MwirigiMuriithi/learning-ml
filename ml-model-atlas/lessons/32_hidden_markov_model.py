"""
# 32 · Hidden Markov Models (HMM): Viterbi & Baum-Welch
**Family:** Probabilistic sequence models (supervised or unsupervised)

## Definition
An HMM assumes a system moves through **hidden states** (not observed) following a Markov chain, and each state emits an **observable** symbol/measurement.
From the visible sequence you can: **evaluate** its likelihood (forward algorithm), **decode** the most likely hidden path (Viterbi), and **learn** the parameters (Baum-Welch = EM).

## Intuition
You never see the weather where your friend lives, only what they did each day (walk, shop, clean). Rainy days make cleaning likely, sunny days walking. From a week of activities you can infer
the likely weather pattern. In engineering: hidden machine health (healthy/degraded/failing) from vibration readings; speech phonemes from audio; gene regions from DNA.

## The math
    parameters  π (initial probs), A_ij = P(sₜ₊₁=j | sₜ=i) (transitions), B_i(o) = P(o | state i) (emissions)
    Forward:   αₜ(j) = [Σ_i αₜ₋₁(i) A_ij] · B_j(oₜ)                  P(o) = Σ_j α_T(j)      (O(T·N²))
    Viterbi:   δₜ(j) = max_i [δₜ₋₁(i) A_ij] · B_j(oₜ) ,  ψₜ(j) = argmax_i ... ;  backtrack from argmax δ_T
    Baum-Welch (EM):  E-step: γₜ(i)=P(sₜ=i|o), ξₜ(i,j)=P(sₜ=i,sₜ₊₁=j|o) via forward-backward;
                      M-step: A_ij = Σξ/Σγ ,  B_i(o)=Σ_{t:oₜ=o} γₜ(i)/Σγₜ(i)
    Use log-probabilities or scaling to avoid underflow. Modern relatives: CRFs, Kalman filters (continuous states), RNNs/transformers.

## What the animations show
1. `32_hidden_markov_model_viterbi.gif`: the Viterbi trellis built column by column for the observation sequence walk, shop, clean, clean, walk, shop, shop, clean; best-predecessor arrows; then the backtracked hidden path in red.
2. `32_hidden_markov_model.gif`: Baum-Welch learning a 3-state model from 1500 emitted symbols starting from near-uniform random guesses; learned transition/emission heat-maps (states re-ordered to match the truth) converge to the true ones as the log-likelihood rises.

## Try this
- Add a 3rd observation symbol or a 3rd hidden state to the Viterbi example.
- Use fewer training symbols (100) and watch Baum-Welch overfit/local optima.
"""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from mlatlas import viz
from mlatlas.cli import run


def fwd_bwd(o, pi, A, B):
    T, N = len(o), len(pi); al = np.zeros((T, N)); be = np.ones((T, N)); c = np.zeros(T); al[0] = pi * B[:, o[0]]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, T): al[t] = (al[t - 1] @ A) * B[:, o[t]]; c[t] = al[t].sum(); al[t] /= c[t]
    for t in range(T - 2, -1, -1): be[t] = A @ (B[:, o[t + 1]] * be[t + 1]) / c[t + 1]
    g = al * be; xi = sum(al[t][:, None] * A * (B[:, o[t + 1]] * be[t + 1])[None] / c[t + 1] for t in range(T - 1)); return g, xi, np.log(c).sum()


def viterbi(o, pi, A, B):
    T, N = len(o), len(pi); d = np.zeros((T, N)); psi = np.zeros((T, N), int); d[0] = np.log(pi) + np.log(B[:, o[0]])
    for t in range(1, T): s = d[t - 1][:, None] + np.log(A); psi[t] = s.argmax(0); d[t] = s.max(0) + np.log(B[:, o[t]])
    path = [int(d[-1].argmax())]
    for t in range(T - 1, 0, -1): path.append(int(psi[t][path[-1]]))
    return d, psi, path[::-1]


def main(gif, quick):
    # ---- Viterbi trellis (classic weather example)
    pi = np.array([.6, .4]); A = np.array([[.7, .3], [.4, .6]]); B = np.array([[.1, .4, .5], [.6, .3, .1]]); S = ["Rainy", "Sunny"]; O = ["walk", "shop", "clean"]; obs = [0, 1, 2, 2, 0, 1, 1, 2]
    d, psi, best = viterbi(obs, pi, A, B); print("  Viterbi hidden path:", [S[i] for i in best]); fig, ax = viz.subplots(1, 1, w=11, h=4.4); T = len(obs)

    def update(f):
        ax.clear(); ax.axis("off"); t_max = min(f, T - 1); back = f >= T
        for t in range(t_max + 1):
            for j in range(2):
                inpath = back and best[t] == j and t <= f - T + T - 1 and (t >= T - 1 - (f - T))
                ax.scatter(t, 1 - j, s=1400, color=viz.PALETTE[3] if inpath else "#dde", edgecolor="k", zorder=3); ax.text(t, 1 - j, f"{d[t, j]:.1f}", ha="center", va="center", fontsize=9, zorder=4)
                if t > 0: ax.annotate("", (t - .17, 1 - j), (t - 1 + .17, 1 - psi[t, j]), arrowprops=dict(arrowstyle="->", color="#666", lw=1.3))
        if back:
            k = f - T
            for t in range(T - 1, T - 2 - k, -1):
                if t > 0: ax.annotate("", (t - .17, 1 - best[t]), (t - 1 + .17, 1 - best[t - 1]), arrowprops=dict(arrowstyle="->", color=viz.PALETTE[3], lw=3))
        for t in range(T): ax.text(t, 1.45, O[obs[t]], ha="center", fontsize=10, fontweight="bold")
        for j in range(2): ax.text(-.7, 1 - j, S[j], ha="right", va="center", fontsize=11)
        ax.set(xlim=(-1.6, T - .3), ylim=(-.6, 1.7)); ax.set_title("Viterbi: best log-probability of reaching each hidden state (arrows = best predecessor)" if not back else "Backtracking the most likely hidden path (red)", fontsize=10)
    viz.animate(fig, update, range(2 * T if not quick else 6), gif.replace(".gif", "_viterbi.gif"), fps=2)
    # ---- Baum-Welch learning
    rng = np.random.default_rng(2); N, M, n = 3, 4, 800 if quick else 1500; tA = np.array([[.85, .1, .05], [.1, .8, .1], [.05, .1, .85]]); tB = np.array([[.7, .2, .05, .05], [.05, .7, .2, .05], [.05, .05, .2, .7]]); tp = np.ones(3) / 3
    s = [rng.integers(3)]; ob = []
    for t in range(n): ob.append(rng.choice(M, p=tB[s[-1]])); s.append(rng.choice(3, p=tA[s[-1]]))
    ob = np.array(ob); p, a, b = np.ones(N) / N, rng.dirichlet(np.ones(N) * 8, N), rng.dirichlet(np.ones(M) * 4, N); H = []; ll_true = fwd_bwd(ob, tp, tA, tB)[2]
    for it in range(30 if quick else 60):
        g, xi, ll = fwd_bwd(ob, p, a, b); H.append((a.copy(), b.copy(), ll)); p = g[0]; a = xi / xi.sum(1, keepdims=True); b = np.array([g[ob == k].sum(0) for k in range(M)]).T; b /= b.sum(1, keepdims=True)
    perm = lambda b_: min(itertools.permutations(range(3)), key=lambda q: np.abs(b_[list(q)] - tB).sum()); print(f"  log-likelihood {H[0][2]:.0f} -> {H[-1][2]:.0f} (true model {ll_true:.0f}); max |A error| {np.abs(H[-1][0][list(perm(H[-1][1]))][:, list(perm(H[-1][1]))] - tA).max():.3f}")
    fig = plt.figure(figsize=(12, 4.2), constrained_layout=True); gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.5]); ax = [fig.add_subplot(gs[0, i]) for i in range(5)]

    def update2(i):
        for x in ax: x.clear()
        a_, b_, _ = H[i]; q = list(perm(b_))
        for x, M_, t in ((ax[0], tA, "true transitions A"), (ax[1], a_[q][:, q], f"learned A (iter {i})"), (ax[2], tB, "true emissions B"), (ax[3], b_[q], "learned B")):
            x.imshow(M_, vmin=0, vmax=1, cmap="Blues"); x.grid(False); x.set_title(t, fontsize=9); x.set_xticks(range(M_.shape[1])); x.set_yticks(range(3))
            for r in range(M_.shape[0]):
                for c in range(M_.shape[1]): x.text(c, r, f"{M_[r, c]:.2f}", ha="center", va="center", fontsize=8, color="w" if M_[r, c] > .5 else "k")
        ax[4].plot([h[2] for h in H[:i + 1]], color=viz.PALETTE[2]); ax[4].axhline(ll_true, color="k", ls="--", label="true model"); ax[4].set(xlim=(0, len(H)), ylim=(H[0][2] - 20, ll_true + 40), title="log-likelihood", xlabel="EM iteration"); ax[4].legend()
    viz.animate(fig, update2, range(0, len(H), 2 if quick else 1), gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
