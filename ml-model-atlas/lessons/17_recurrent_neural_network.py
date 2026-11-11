"""
# 17 · Recurrent Neural Networks (RNN) & Backprop-Through-Time
**Family:** Supervised · Deep learning for sequences

## Definition
An RNN reads a sequence one element at a time and keeps a **hidden state** hₜ, a running summary of everything seen so far. The same weights are reused at every
time step, so it handles sequences of any length (time series, text, sensor streams).

## Intuition
Reading a sentence word by word while keeping a mental note of the story so far. At each step the note is updated from
the new word and the previous note. To train, we "unroll" the network through time (one copy per step, sharing weights) and backpropagate
through all the copies: **backpropagation through time (BPTT)**.

## The math
    hₜ = tanh(xₜ Wₓₕ + hₜ₋₁ Wₕₕ + b)           ŷₜ = hₜ W_y + b_y
    L = 1/T Σₜ (ŷₜ − yₜ)²                       gradient flows backwards through every hₜ
    ∂L/∂Wₕₕ = Σₜ  ... Π (Wₕₕᵀ · diag(1 − h²)) ...      repeated multiplication => gradients vanish (<1) or explode (>1)
    Fixes: gradient clipping (used here), gated cells (LSTM/GRU, lesson 18), attention (lesson 19).
    This file uses the autograd engine in mlatlas/nn.py, so BPTT is automatic; read the 6-line forward pass.

## Practical tips
- Clip gradient norm. Use LSTM/GRU for anything with long-range dependencies. Transformers have mostly replaced RNNs for text.
- For forecasting, teacher forcing (feed the true previous value) is used in training; at inference you feed back predictions.

## What the animation shows
A noisy two-frequency "seasonal demand" signal. Top-left: the RNN's one-step-ahead prediction (red) improving over the true signal (black).
Right: the heat-map of the 16 hidden units over time (the network's memory), bottom-left: loss curve.

## Try this
- Make the sequence length 60 and watch training get harder (vanishing gradients).
- Replace tanh by ReLU and turn clipping off.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from mlatlas import viz
from mlatlas.nn import T, Adam, mse
from mlatlas.cli import run


class RNN:
    def __init__(s, rng, H=16):
        s.H = H; s.Wx = T(rng.normal(0, .5, (1, H))); s.Wh = T(rng.normal(0, .3, (H, H))); s.b = T(np.zeros(H)); s.Wy = T(rng.normal(0, .3, (H, 1))); s.by = T(np.zeros(1))
    def params(s): return [s.Wx, s.Wh, s.b, s.Wy, s.by]
    def run(s, xs):
        h = T(np.zeros((xs.shape[0], s.H))); outs, hs = [], []
        for t in range(xs.shape[1]):
            h = (T(xs[:, t, :]) @ s.Wx + h @ s.Wh + s.b).tanh(); outs.append(h @ s.Wy + s.by); hs.append(h.data[0].copy())
        return outs, np.array(hs).T


def signal(n, rng=None):
    t = np.arange(n); return np.sin(.3 * t) + .5 * np.sin(.11 * t + 1) + (0 if rng is None else rng.normal(0, .05, n))


def main(gif, quick):
    rng = np.random.default_rng(0); long = signal(600, rng); net = RNN(rng); opt = Adam(net.params(), lr=.02, clip=1.0); L = 24; steps = 120 if quick else 400
    test = signal(80)[None, :, None]; losses, snaps = [], {}
    for it in range(steps + 1):
        st = rng.integers(0, 500, 16); xs = np.stack([long[s:s + L + 1] for s in st])[:, :, None]; outs, _ = net.run(xs[:, :-1])
        loss = sum(mse(o, xs[:, t + 1, :]) for t, o in enumerate(outs)) * (1.0 / L); losses.append(float(loss.data))
        if it % (10 if not quick else 20) == 0:
            o, hs = net.run(test[:, :-1]); snaps[it] = (np.array([x.data[0, 0] for x in o]), hs, len(losses))
        opt.zero(); loss.backward(); opt.step()
    print(f"  loss {losses[0]:.3f} -> {losses[-1]:.4f}")
    fig = plt.figure(figsize=(11, 5.4), constrained_layout=True); gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1]); a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0]); a3 = fig.add_subplot(gs[:, 1])

    def update(it):
        for a in (a1, a2, a3): a.clear()
        pr, hs, k = snaps[it]; a1.plot(test[0, 1:, 0], "k", lw=2, label="truth (next value)"); a1.plot(pr, color=viz.PALETTE[3], lw=2, label="RNN prediction"); a1.set(title=f"Training step {it}: one-step-ahead forecast", ylim=(-2, 2)); a1.legend(loc="upper right", ncol=2)
        a2.semilogy(losses[:k], color=viz.PALETTE[0]); a2.set(xlim=(0, steps), ylim=(1e-3, 2), xlabel="step", title="Training loss (MSE)")
        a3.imshow(hs, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1); a3.grid(False); a3.set(title="Hidden state hₜ (16 units × time)", xlabel="time step", ylabel="hidden unit")
    viz.animate(fig, update, sorted(snaps), gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
