"""
Lesson 21 - Attention and the transformer block (from scratch)

Problem story: a warehouse scanner reads a tray of 6 labelled items.  Each item has an id (one of 10 part numbers) and a measured
weight reading.  A query then asks "what was the reading of part #7?".  The items are shuffled, so the answer cannot be found by
position: the model must *look up by content*.  That is exactly what attention does: a query is compared with every key, the
comparison scores are turned into weights (softmax), and the answer is the weighted mix of the values.

Implemented (NumPy only):
  * scaled dot-product attention (with masks), multi-head attention, sinusoidal positional encoding, layer norm,
    a full pre-norm transformer encoder block (forward pass), and property tests:
        - permutation equivariance of self-attention (no positional encoding)
        - a causal mask gives exactly zero weight to the future
  * a trainable single-head attention *retrieval* model with manual back-propagation (finite-difference checked)

Animated:
  attention_learning.gif   the lookup model learns where to look: attention maps sharpen onto the right item
  attention_mechanics.gif  one query against five keys: dot products -> softmax -> weighted average; then the effect of temperature
  positional_encoding.gif  sinusoidal position codes: more dimensions = finer ruler; the similarity of two positions depends on their distance

Run:  python attention.py        (about 40 s)
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ------------------------------------------------------------------- building blocks
def softmax_rows(S):
    S = S - S.max(-1, keepdims=True)
    e = np.exp(S)
    return e / e.sum(-1, keepdims=True)


def attention(Q, K, V, mask=None):
    """softmax(Q K^T / sqrt(d_k)) V.   mask: boolean (n_q, n_k), True = may attend."""
    d = Q.shape[-1]
    S = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)
    if mask is not None:
        S = np.where(mask, S, -1e9)
    W = softmax_rows(S)
    return W @ V, W


def multi_head(X, P, heads, mask=None):
    """Self-attention with `heads` parallel heads: split the model dimension, attend in each, concatenate, project."""
    n, d = X.shape; dh = d // heads
    Q, K, V = X @ P["Wq"], X @ P["Wk"], X @ P["Wv"]
    outs, Ws = [], []
    for h in range(heads):
        sl = slice(h * dh, (h + 1) * dh)
        o, W = attention(Q[:, sl], K[:, sl], V[:, sl], mask)
        outs.append(o); Ws.append(W)
    return np.concatenate(outs, 1) @ P["Wo"], np.array(Ws)


def sinusoidal_pe(n, d):
    pos = np.arange(n)[:, None]
    i = np.arange(d // 2)[None]
    ang = pos / (10000 ** (2 * i / d))
    pe = np.zeros((n, d)); pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang)
    return pe


def layer_norm(X, eps=1e-5):
    return (X - X.mean(-1, keepdims=True)) / np.sqrt(X.var(-1, keepdims=True) + eps)


def init_block(d, dff, seed=0):
    r = rng_(seed); s = 1 / np.sqrt(d)
    return dict(Wq=r.normal(0, s, (d, d)), Wk=r.normal(0, s, (d, d)), Wv=r.normal(0, s, (d, d)), Wo=r.normal(0, s, (d, d)),
                W1=r.normal(0, s, (d, dff)), b1=np.zeros(dff), W2=r.normal(0, 1 / np.sqrt(dff), (dff, d)), b2=np.zeros(d))


def transformer_block(X, P, heads, mask=None):
    """Pre-norm encoder block:  X + MHA(LN(X));  then + FFN(LN(.)) with a GELU-like ReLU hidden layer."""
    a, W = multi_head(layer_norm(X), P, heads, mask)
    X = X + a
    h = np.maximum(layer_norm(X) @ P["W1"] + P["b1"], 0) @ P["W2"] + P["b2"]
    return X + h, W


# ------------------------------------------------------------------- trainable retrieval model
V_IDS, N_ITEMS, D = 10, 6, 12


def make_batch(B, r):
    ids = np.array([r.choice(V_IDS, N_ITEMS, replace=False) for _ in range(B)])
    vals = r.uniform(-1, 1, (B, N_ITEMS))
    pos = r.integers(0, N_ITEMS, B)
    q = ids[np.arange(B), pos]
    return ids, vals, q, vals[np.arange(B), pos], pos


class Retriever:
    """Single-head attention lookup.   key_j = Wk E[id_j];  value_j = Wv [E[id_j]; v_j];  query = Wq E[q].
    context = sum_j a_j value_j;   y_hat = w . context + b.      Manual backprop through the softmax."""

    def __init__(self, seed=0):
        r = rng_(seed)
        self.p = dict(E=r.normal(0, 0.5, (V_IDS, D)), Wq=r.normal(0, 1 / np.sqrt(D), (D, D)), Wk=r.normal(0, 1 / np.sqrt(D), (D, D)),
                      Wv=r.normal(0, 1 / np.sqrt(D + 1), (D + 1, D)), w=r.normal(0, 1 / np.sqrt(D), D), b=np.zeros(1))

    def forward(self, ids, vals, q):
        p = self.p
        E_it = p["E"][ids]                                          # (B, n, D)
        K = E_it @ p["Wk"]
        U = np.concatenate([E_it, vals[..., None]], 2) @ p["Wv"]    # (B, n, D)
        Qv = p["E"][q] @ p["Wq"]                                    # (B, D)
        S = np.einsum("bd,bnd->bn", Qv, K) / np.sqrt(D)
        A = softmax_rows(S)
        ctx = np.einsum("bn,bnd->bd", A, U)
        y = ctx @ p["w"] + p["b"][0]
        self.c = (ids, vals, q, E_it, K, U, Qv, A, ctx)
        return y, A

    def backward(self, dy):
        p = self.p
        ids, vals, q, E_it, K, U, Qv, A, ctx = self.c
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["w"] = ctx.T @ dy; g["b"] = np.array([dy.sum()])
        dctx = dy[:, None] * p["w"][None]
        dA = np.einsum("bd,bnd->bn", dctx, U)
        dU = A[..., None] * dctx[:, None, :]
        dS = A * (dA - (A * dA).sum(1, keepdims=True))             # softmax Jacobian-vector product
        dS = dS / np.sqrt(D)
        dQv = np.einsum("bn,bnd->bd", dS, K)
        dK = dS[..., None] * Qv[:, None, :]
        g["Wq"] = p["E"][q].T @ dQv
        dEq = dQv @ p["Wq"].T
        Ein = np.concatenate([E_it, vals[..., None]], 2)
        g["Wv"] = np.einsum("bnk,bnd->kd", Ein, dU)
        dEin = dU @ p["Wv"].T
        g["Wk"] = np.einsum("bnk,bnd->kd", E_it, dK)
        dE_it = dK @ p["Wk"].T + dEin[..., :D]
        np.add.at(g["E"], ids.reshape(-1), dE_it.reshape(-1, D))
        np.add.at(g["E"], q, dEq)
        return g


class Adam:
    def __init__(self, params, lr=3e-3):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}; self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = 0.9 * self.m[k] + 0.1 * g; self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            params[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)


def grad_check(seed=0):
    r = rng_(seed); m = Retriever(seed)
    ids, vals, q, y, _ = make_batch(5, r)
    def loss():
        yh, _ = m.forward(ids, vals, q); return 0.5 * np.mean((yh - y) ** 2)
    yh, _ = m.forward(ids, vals, q); g = m.backward((yh - y) / len(y))
    worst = 0
    for k, v in m.p.items():
        for _ in range(6):
            idx = tuple(r.integers(0, s) for s in v.shape)
            old = v[idx]; v[idx] = old + 1e-6; lp = loss(); v[idx] = old - 1e-6; lm = loss(); v[idx] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - g[k][idx]) / (abs(num) + abs(g[k][idx]) + 1e-9))
    return worst


def train_retriever(steps=3000, B=64, seed=0, snaps=()):
    m = Retriever(seed); opt = Adam(m.p, 3e-3); r = rng_(seed + 1)
    tr = rng_(123); T_ids, T_vals, T_q, T_y, T_pos = make_batch(500, tr)
    hist, saved = [], {}
    for s in range(steps + 1):
        if s in snaps or s % 25 == 0 or s == steps:
            yh, A = m.forward(T_ids, T_vals, T_q)
            mse = float(np.mean((yh - T_y) ** 2))
            mass = float(A[np.arange(500), T_pos].mean())                # attention weight on the correct item
            hist.append((s, mse, mass))
            if s in snaps:
                saved[s] = (m.forward(T_ids[:8], T_vals[:8], T_q[:8])[1].copy(), {k: v.copy() for k, v in m.p.items()})
        if s == steps:
            break
        ids, vals, q, y, _ = make_batch(B, r)
        yh, _ = m.forward(ids, vals, q)
        opt.step(m.p, m.backward((yh - y) / B))
    return m, np.array(hist), saved, (T_ids, T_vals, T_q, T_y, T_pos)


# ------------------------------------------------------------------- animations
def learning_gif(path, hist, saved, test):
    T_ids, T_vals, T_q, T_y, T_pos = test
    steps = sorted(saved)
    with Animator(path, fps=1.5, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 5), title="Learning where to look: query = 'what was the reading of part #?'", gridspec_kw={"width_ratios": [1.1, 1.0]})
        for s in steps:
            a1.clear(); a2.clear()
            A = saved[s][0]
            a1.imshow(A, cmap="Blues", vmin=0, vmax=1, aspect="auto"); a1.grid(False)
            for i in range(8):
                a1.add_patch(plt.Rectangle((T_pos[i] - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2.5))
                a1.text(-0.9, i, f"#{T_q[i]}", ha="right", va="center", fontsize=9, color=INK2)
                for j in range(N_ITEMS):
                    a1.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if A[i, j] > 0.5 else INK2)
            a1.set_xticks(range(N_ITEMS), [f"#{k}" for k in T_ids[0]] if False else [f"slot {j + 1}" for j in range(N_ITEMS)], fontsize=8)
            a1.set_yticks([]); a1.set_title("Attention weights of 8 test queries (rows); orange box = the item that holds the answer")
            h = hist[hist[:, 0] <= s]
            a2.semilogy(h[:, 0], h[:, 1], color=BLUE, lw=2, label="test MSE")
            a2.axhline(np.var(T_y), color=MUTED, ls="--", lw=1); a2.text(5, np.var(T_y) * 1.1, f"predicting 0: {np.var(T_y):.2f}", fontsize=8, color=INK2)
            a2b = a2.twinx(); a2b.plot(h[:, 0], h[:, 2], color=ORANGE, lw=2); a2b.set_ylim(0, 1.02); a2b.set_ylabel("attention on the right item", color=ORANGE); a2b.grid(False)
            a2.set(xlim=(0, hist[-1, 0]), ylim=(1e-6, 1), title="Loss and attention accuracy", xlabel="training step", ylabel="MSE (log)")
            cur = h[-1]
            caption(fig, f"step {s}: test MSE {cur[1]:.4f}; on average {cur[2]:.0%} of the attention lands on the correct item (uniform guessing: {1 / N_ITEMS:.0%})")
            anim.grab(fig)


def mechanics_gif(path):
    ang = np.deg2rad([20, 85, 150, 215, 290])
    lens = np.array([1.0, 0.8, 1.1, 0.9, 1.0])
    Kv = np.c_[np.cos(ang), np.sin(ang)] * lens[:, None]            # keys
    Vv = np.array([[2.0, 1.0], [-1.5, 2.0], [-2.0, -0.5], [0.5, -2.0], [1.5, -1.0]])   # values (what each item "says")
    d = 2
    frames = [("angle", a) for a in np.linspace(0, 330, 12)] + [("temp", t) for t in [1.0, 2.0, 4.0, 8.0, 0.5, 0.2]]
    qa0 = np.deg2rad(60)
    with Animator(path, fps=1.5, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="One query attends to five items", gridspec_kw={"width_ratios": [1.1, 0.9, 1.1]})
        for kind, val in frames:
            for a in (a1, a2, a3):
                a.clear()
            if kind == "angle":
                qa, scale = np.deg2rad(val), 3.0
            else:
                qa, scale = qa0, 3.0 * val
            q = np.array([np.cos(qa), np.sin(qa)]) * scale / 3.0 * 3.0 ** 0.5 * (3.0 ** 0.5) / 3.0 ** 0.5   # |q| = scale
            q = np.array([np.cos(qa), np.sin(qa)]) * scale
            s = Kv @ q / np.sqrt(d)
            w = softmax_rows(s[None])[0]
            out = w @ Vv
            for j in range(5):
                a1.annotate("", xy=Kv[j], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=CAT[j], lw=2))
                a1.scatter(*Kv[j], s=60 + 700 * w[j], color=CAT[j], alpha=0.8, zorder=3)
                a1.text(Kv[j][0] * 1.22, Kv[j][1] * 1.22, f"k{j + 1}", color=INK2, fontsize=9, ha="center")
            a1.annotate("", xy=q / max(np.linalg.norm(q), 1e-9) * 1.5, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=3))
            a1.text(*(q / max(np.linalg.norm(q), 1e-9) * 1.7), "query", fontsize=9, ha="center")
            a1.set(xlim=(-1.8, 1.8), ylim=(-1.8, 1.8), aspect="equal", title="Keys and the query (bubble = attention)"); a1.set_xticks([]); a1.set_yticks([])
            a2.bar([f"k{j + 1}" for j in range(5)], w, color=CAT[:5])
            for j in range(5):
                a2.text(j, w[j] + 0.02, f"{w[j]:.2f}\n(q·k/√d = {s[j]:+.1f})", ha="center", fontsize=7.5)
            a2.set(ylim=(0, 1.25), title="softmax weights"); a2.grid(axis="x", visible=False)
            for j in range(5):
                a3.scatter(*Vv[j], s=60 + 700 * w[j], color=CAT[j], alpha=0.8, zorder=3); a3.text(Vv[j][0] + 0.15, Vv[j][1] + 0.15, f"v{j + 1}", fontsize=9, color=INK2)
            a3.scatter(*out, s=140, marker="*", color=INK, zorder=5)
            a3.annotate("output", xy=out, xytext=(out[0] + 0.35, out[1] + 0.45), fontsize=9)
            a3.set(xlim=(-2.8, 2.8), ylim=(-2.8, 2.8), aspect="equal", title="Values and the output = Σ wⱼ vⱼ"); a3.set_xticks([]); a3.set_yticks([])
            ent = -(w * np.log(w + 1e-12)).sum()
            if kind == "angle":
                msg = f"The query points at {int(round(val))}°: the key most aligned with it gets the most weight, and the output moves toward that item's value"
            else:
                msg = f"Query length x{val:g} (temperature {1 / val:.2g}): " + ("nearly uniform, output is the plain average" if val <= 0.5 else "sharper and sharper: approaching a hard lookup" if val >= 4 else "moderate")
                msg += f"; entropy of the weights {ent:.2f} (max {np.log(5):.2f})"
            caption(fig, msg)
            anim.grab(fig)


def pe_gif(path, n=48):
    d = 64
    pe = sinusoidal_pe(n, d)
    with Animator(path, fps=1.2, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Sinusoidal positional encoding: a ruler with many scales", gridspec_kw={"width_ratios": [1.0, 0.8, 1.1]})
        for k in [2, 4, 8, 16, 32, 64]:
            for a in (a1, a2, a3):
                a.clear()
            M = pe.copy(); M[:, k:] = 0
            a1.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto"); a1.grid(False)
            a1.set(title=f"Codes of positions 0-{n - 1} (first {k} dims)", xlabel="dimension", ylabel="position")
            sim = pe[:, :k] @ pe[:, :k].T / (k / 2)
            a2.imshow(sim, cmap="viridis", vmin=-1, vmax=1.05, aspect="equal"); a2.grid(False)
            a2.set(title="Similarity of positions", xlabel="position", ylabel="position")
            lag = np.arange(0, 40)
            a3.plot(lag, [np.mean([sim[i, i + L] for i in range(n - 40)]) for L in lag], color=BLUE, lw=2)
            a3.set(title="Similarity vs distance", xlabel="distance between two positions", ylim=(-1.1, 1.1))
            cols = np.arange(0, k, 2)
            caption(fig, f"{k} dimensions: wavelengths from 2π to {2 * np.pi * 10000 ** ((k - 2) / d):.0f}. " + ("Few frequencies: far-away positions look alike (aliasing)." if k <= 8 else "More frequencies: every position has a unique code, and similarity depends on distance, not absolute position."))
            anim.grab(fig)


# ------------------------------------------------------------------- main
def main():
    print("Lesson 21: attention")
    r = rng_(0); d, n, heads = 16, 7, 4
    P = init_block(d, 32); X = r.normal(size=(n, d))
    out, W = transformer_block(X, P, heads)
    perm = r.permutation(n); out_p, _ = transformer_block(X[perm], P, heads)
    print(f"  transformer block: input {X.shape} -> output {out.shape}, attention weights {W.shape} (heads, queries, keys); rows sum to {W.sum(-1).mean():.3f}")
    print(f"  permutation equivariance (no positional info): max |block(PX) - P block(X)| = {np.abs(out_p - out[perm]).max():.1e}")
    mask = np.tril(np.ones((n, n), bool)); _, Wc = transformer_block(X, P, heads, mask)
    print(f"  causal mask: total attention weight on future positions = {np.triu(Wc[0], 1).sum():.1e}; first query attends 100% to itself: {Wc[0, 0, 0]:.2f}")
    PE = sinusoidal_pe(n, d)
    o1, _ = transformer_block(X[perm] + PE, P, heads); o2, _ = transformer_block(X + PE, P, heads)
    print(f"  with positional encoding added the symmetry is broken (shuffle the tokens, keep the positions): max |difference| {np.abs(o1 - o2[perm]).max():.2f}")
    print(f"  gradient check of the retrieval model (worst relative error): {grad_check():.1e}")
    steps = 3000
    snaps = (0, 25, 50, 100, 200, 400, 800, 1500, 3000)
    m, hist, saved, test = train_retriever(steps, snaps=snaps)
    T_ids, T_vals, T_q, T_y, T_pos = test
    print(f"  baselines on the 500 test queries: predict 0 -> MSE {np.mean(T_y ** 2):.3f}; average of the 6 readings (uniform attention) -> MSE {np.mean((T_vals.mean(1) - T_y) ** 2):.3f}")
    for s, mse, mass in hist:
        if s in snaps:
            print(f"     step {int(s):4d}: test MSE {mse:.4f}, attention on correct item {mass:.3f}")
    learning_gif(here(__file__, "attention_learning.gif"), hist, saved, test)
    mechanics_gif(here(__file__, "attention_mechanics.gif"))
    pe_gif(here(__file__, "positional_encoding.gif"))


if __name__ == "__main__":
    main()
