"""
# 19 · Attention & the Transformer
**Family:** Supervised · Deep learning (sequence → sequence, language, vision)

## Definition
**Attention** lets every position in a sequence look at every other position and pull in the information it needs, weighted by relevance.
A **Transformer** stacks attention + feed-forward blocks with residual connections and layer normalization. There is no recurrence, so all
positions are processed in parallel, which is what lets models scale to GPT/BERT size.

## Intuition
Each token asks a question (**query**), advertises what it contains (**key**) and carries a payload (**value**). Similarity between
a query and every key decides how much of each value is mixed in (softmax weights). In this lesson the model must find *the one flagged digit* in a sequence of 8 digits and
report it. We train a tiny 1-layer transformer encoder and **watch its attention learn to lock onto the flagged token**.

## The math
    Q = XW_q ,  K = XW_k ,  V = XW_v
    Attention(Q,K,V) = softmax( Q Kᵀ / √d ) V               (√d keeps the softmax from saturating)
    Multi-head: run h attention "heads" on d/h-dimensional slices, concatenate, project with W_o
    Block:  X ← LayerNorm(X + Attention(X))      X ← LayerNorm(X + FFN(X)),   FFN = ReLU(XW₁+b₁)W₂ + b₂
    Position: attention is order-blind, so add positional information (sinusoidal  sin/cos(pos/10000^(2i/d))  or learned).
    Causal mask (GPT-style) sets future positions' scores to −∞.
    Cost: O(n²) in sequence length, the reason for long-context research (sparse/linear attention, FlashAttention).

## Practical tips
- Warm-up + Adam(W), pre-LayerNorm, dropout, large batches. Pretrain on lots of data, then fine-tune.
- Also dominates vision (ViT), audio, protein folding and more.

## What the animation shows
Rows are 6 fixed test sequences; columns the 8 digit positions (the red frame marks the flagged one). The colour shows how strongly the
classification token attends to each position. Initially attention is spread out; as training progresses it snaps onto the flagged
digit. Right: loss and accuracy. A static figure `19_attention_transformer_math.png` shows positional encoding and the Q·Kᵀ → softmax maths on a toy example.

## Try this
- Remove the flag embedding: can the model still solve the task? (It cannot, and the attention shows why.)
- Add a second head by splitting d into two halves (see `multi_head_attention` in this file).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from mlatlas import viz
from mlatlas.nn import T, Adam, cat, softmax, layer_norm, cross_entropy, param
from mlatlas.cli import run

D, SEQ = 24, 8


def multi_head_attention(X, Wq, Wk, Wv, Wo, heads):
    """Plain-NumPy multi-head self-attention for X of shape (n, d). Returns (output, attention weights per head)."""
    n, d = X.shape; hd = d // heads; Q, K, V = (X @ W for W in (Wq, Wk, Wv)); outs, atts = [], []
    for h in range(heads):
        s = slice(h * hd, (h + 1) * hd); sc = Q[:, s] @ K[:, s].T / np.sqrt(hd); a = np.exp(sc - sc.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
        outs.append(a @ V[:, s]); atts.append(a)
    return np.concatenate(outs, 1) @ Wo, np.array(atts)


def sinusoidal(n, d):
    pos = np.arange(n)[:, None]; i = np.arange(d)[None]; ang = pos / 10000 ** (2 * (i // 2) / d); return np.where(i % 2 == 0, np.sin(ang), np.cos(ang))


class TinyTransformer:
    def __init__(s, rng):
        s.Ed, s.Ef, s.cls = param((10, D), rng, .5), param((2, D), rng, .5), param((1, 1, D), rng, .5)
        s.Wq, s.Wk, s.Wv, s.Wo = (param((D, D), rng) for _ in range(4)); s.W1, s.b1, s.W2, s.b2 = param((D, 2 * D), rng), T(np.zeros(2 * D)), param((2 * D, D), rng), T(np.zeros(D))
        s.g1, s.be1, s.g2, s.be2 = T(np.ones(D)), T(np.zeros(D)), T(np.ones(D)), T(np.zeros(D)); s.Wc, s.bc = param((D, 10), rng), T(np.zeros(10))
    def params(s): return [s.Ed, s.Ef, s.cls, s.Wq, s.Wk, s.Wv, s.Wo, s.W1, s.b1, s.W2, s.b2, s.g1, s.be1, s.g2, s.be2, s.Wc, s.bc]
    def __call__(s, dig, flag):
        B = len(dig); tok = s.Ed[dig] + s.Ef[flag]; X = cat([tok, T(np.zeros((B, 1, D))) + s.cls], axis=1)          # (B, 9, D), last token = [CLS]
        Q, K, V = X @ s.Wq, X @ s.Wk, X @ s.Wv; A = softmax(Q @ K.transpose(0, 2, 1) * (1 / np.sqrt(D)))              # (B, 9, 9) attention weights
        X = layer_norm(X + (A @ V) @ s.Wo, s.g1, s.be1); X = layer_norm(X + (X @ s.W1 + s.b1).relu() @ s.W2 + s.b2, s.g2, s.be2)
        return X[:, SEQ, :] @ s.Wc + s.bc, A.data[:, SEQ, :SEQ]


def batch(rng, B):
    d = rng.integers(0, 10, (B, SEQ)); pos = rng.integers(0, SEQ, B); f = np.zeros((B, SEQ), int); f[np.arange(B), pos] = 1; return d, f, d[np.arange(B), pos], pos


def main(gif, quick):
    # ---- static figure: the maths of attention on a toy example
    r = np.random.default_rng(0); Xt = r.normal(size=(6, 8)); Wq, Wk, Wv, Wo = (r.normal(size=(8, 8)) * .5 for _ in range(4)); _, att = multi_head_attention(Xt, Wq, Wk, Wv, Wo, 2)
    fig, ax = viz.subplots(1, 3, w=13, h=3.8); ax[0].imshow(sinusoidal(20, 32).T, cmap="RdBu_r", aspect="auto"); ax[0].set(title="Sinusoidal positional encoding", xlabel="position", ylabel="dimension"); ax[0].grid(False)
    ax[1].imshow(Xt @ Wq @ (Xt @ Wk).T / np.sqrt(8), cmap="viridis"); ax[1].set(title="scores  Q·Kᵀ/√d (one row per query token)", xlabel="key token", ylabel="query token"); ax[1].grid(False)
    ax[2].imshow(att[0], cmap="magma"); ax[2].set(title="softmax -> attention weights (rows sum to 1), head 0", xlabel="key token"); ax[2].grid(False); fig.savefig(gif.replace(".gif", "_math.png"), dpi=90); plt.close(fig)
    # ---- train
    rng = np.random.default_rng(1); net = TinyTransformer(rng); opt = Adam(net.params(), lr=.002, clip=2.0); steps = 100 if quick else 300; every = 5
    fd, ff, fy, fp = batch(np.random.default_rng(7), 6); ed, ef, ey, _ = batch(np.random.default_rng(8), 300); hist = []; snaps = {}
    for it in range(steps + 1):
        d, f, y, _ = batch(rng, 64); lg, _ = net(d, f); loss = cross_entropy(lg, y)
        if it % every == 0:
            snaps[it] = net(fd, ff)[1]; hist.append((it, float(loss.data), (net(ed, ef)[0].data.argmax(1) == ey).mean()))
        opt.zero(); loss.backward(); opt.step()
    print(f"  loss {hist[0][1]:.2f} -> {hist[-1][1]:.3f}; test accuracy {hist[-1][2]:.2f} (chance 0.10)")
    fig, (a1, a2) = viz.subplots(1, 2, w=11.5, h=4.4)

    def update(it):
        a1.clear(); a2.clear(); a1.imshow(snaps[it], cmap="magma", vmin=0, vmax=1, aspect="auto"); a1.grid(False)
        for i in range(6):
            for j in range(SEQ): a1.text(j, i, str(fd[i, j]), ha="center", va="center", color="white" if snaps[it][i, j] < .6 else "black", fontsize=11, fontweight="bold" if ff[i, j] else "normal")
            a1.add_patch(plt.Rectangle((fp[i] - .5, i - .5), 1, 1, fill=False, edgecolor="#E45756", lw=3))
        a1.set(title=f"Step {it}: attention of [CLS] over the sequence (red frame = flagged)", xlabel="position", ylabel="test sequence"); a1.set_xticks(range(SEQ)); a1.set_yticks(range(6))
        k = it // every + 1; h = hist[:k]; a2.plot([x[0] for x in h], [x[1] for x in h], color=viz.PALETTE[3], label="loss"); a2.plot([x[0] for x in h], [x[2] for x in h], color=viz.PALETTE[2], label="test accuracy")
        a2.set(xlim=(0, steps), ylim=(0, 2.6), xlabel="training step", title="Learning curves"); a2.legend(loc="center right")
    viz.animate(fig, update, sorted(snaps), gif, fps=4)


if __name__ == "__main__":
    run(main, __file__)
