"""
# 18 · LSTM & GRU: Gated Memory
**Family:** Supervised · Deep learning for sequences

## Definition
LSTM (Hochreiter & Schmidhuber, 1997) and GRU (Cho et al., 2014) are recurrent cells with **gates**: learned sigmoid switches that
decide what to write into memory, what to keep and what to reveal. They solve the vanishing-gradient problem of plain RNNs
because the memory (cell state) can be carried across many steps almost untouched, giving gradients a "highway".

## Intuition
A plain RNN rewrites its whole memory every step, so an early signal is quickly overwritten. An LSTM has a notebook (cell state cₜ)
with three controls: the **forget gate** (erase old notes?), the **input gate** (write new notes?), the **output gate** (read the notes out now?).
Here the task is a pure memory test: *the first input is a +1 or −1 signal, followed by 29 steps of random noise; at the end, report the sign of the first input.*

## The math (LSTM)
    [i, f, o, g] = [x, h] W + b         i,f,o = σ(·)   g = tanh(·)
    cₜ = f ⊙ cₜ₋₁ + i ⊙ g               (additive update = gradient highway)
    hₜ = o ⊙ tanh(cₜ)
    GRU (fewer parameters, no separate cell):  z = σ([x,h]Wz) , r = σ([x,h]Wr) , h̃ = tanh([x, r⊙h]W) , h = (1−z)⊙h + z⊙h̃
    Forget-gate bias initialised to +1 so the cell remembers by default.

## Practical tips
- LSTM/GRU are the standard for sequential data when transformers are overkill (streams, small data, on-device).
- GRU is faster and often equal in accuracy; LSTM can be better for very long dependencies.

## What the animation shows
Left: test accuracy during training for a vanilla RNN, GRU and LSTM on the memory task (50 % = guessing). Right: the LSTM's **input, forget, output
gates and cell state** over the 30 time steps for one sequence. Watch the cell state lock in the first signal and the forget gate stay open.

## Try this
- Lengthen the noise to 100 steps: which model still learns?
- Set the forget bias to 0 and see how much slower the LSTM trains.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib.pyplot as plt
from mlatlas import viz
from mlatlas.nn import T, Adam, cat, bce_logits, param
from mlatlas.cli import run

TL = 30


class RNNc:
    def __init__(s, rng, H=12): s.H = H; s.W = param((1 + H, H), rng); s.b = T(np.zeros(H)); s.Wy = param((H, 1), rng); s.by = T(np.zeros(1))
    def params(s): return [s.W, s.b, s.Wy, s.by]
    def run(s, x):
        h = T(np.zeros((len(x), s.H)))
        for t in range(x.shape[1]): h = (cat([T(x[:, t]), h]) @ s.W + s.b).tanh()
        return h @ s.Wy + s.by, None


class GRU:
    def __init__(s, rng, H=12):
        s.H = H; s.Wz, s.Wr, s.Wh = (param((1 + H, H), rng) for _ in range(3)); s.bz, s.br, s.bh = (T(np.zeros(H)) for _ in range(3)); s.Wy = param((H, 1), rng); s.by = T(np.zeros(1))
    def params(s): return [s.Wz, s.Wr, s.Wh, s.bz, s.br, s.bh, s.Wy, s.by]
    def run(s, x):
        h = T(np.zeros((len(x), s.H)))
        for t in range(x.shape[1]):
            xt = T(x[:, t]); xh = cat([xt, h]); z = (xh @ s.Wz + s.bz).sigmoid(); r = (xh @ s.Wr + s.br).sigmoid()
            hh = (cat([xt, r * h]) @ s.Wh + s.bh).tanh(); h = (1 - z) * h + z * hh
        return h @ s.Wy + s.by, None


class LSTM:
    def __init__(s, rng, H=12):
        s.H = H; s.W = param((1 + H, 4 * H), rng); b = np.zeros(4 * H); b[H:2 * H] = 1.0; s.b = T(b); s.Wy = param((H, 1), rng); s.by = T(np.zeros(1))
    def params(s): return [s.W, s.b, s.Wy, s.by]
    def run(s, x, trace=False):
        H = s.H; h = T(np.zeros((len(x), H))); c = T(np.zeros((len(x), H))); tr = []
        for t in range(x.shape[1]):
            z = cat([T(x[:, t]), h]) @ s.W + s.b; i = z[:, 0:H].sigmoid(); f = z[:, H:2 * H].sigmoid(); g = z[:, 2 * H:3 * H].tanh(); o = z[:, 3 * H:].sigmoid()
            c = f * c + i * g; h = o * c.tanh()
            if trace: tr.append([i.data[0], f.data[0], o.data[0], c.data[0]])
        return h @ s.Wy + s.by, (np.array(tr).transpose(1, 2, 0) if trace else None)


def batch(rng, B):
    x = rng.normal(0, 1.0, (B, TL, 1)); s = rng.choice([-1., 1.], B); x[:, 0, 0] = s * 2.0; return x, (s > 0).astype(float)[:, None]


def main(gif, quick):
    rng = np.random.default_rng(0); models = {"vanilla RNN": RNNc(rng), "GRU": GRU(rng), "LSTM": LSTM(rng)}
    opts = {k: Adam(m.params(), lr=.01, clip=1.0) for k, m in models.items()}; steps = 40 if quick else 200; every = 5
    xe, ye = batch(np.random.default_rng(99), 200); probe = np.zeros((1, TL, 1)); probe[0, 0, 0] = 2.0; probe[0, 1:, 0] = np.random.default_rng(5).normal(0, 1.0, TL - 1)
    acc = {k: [] for k in models}; snaps = {}
    for it in range(steps + 1):
        x, y = batch(rng, 32)
        for k, m in models.items():
            if it % every == 0: lg, _ = m.run(xe); acc[k].append(((lg.data > 0) == (ye > .5)).mean())
            lg, _ = m.run(x); loss = bce_logits(lg, y); opts[k].zero(); loss.backward(); opts[k].step()
        if it % every == 0: snaps[it] = models["LSTM"].run(probe, trace=True)[1]
    print("  final accuracy:", {k: round(float(v[-1]), 2) for k, v in acc.items()})
    fig = plt.figure(figsize=(12, 5.6), constrained_layout=True); gs = fig.add_gridspec(4, 2, width_ratios=[1.1, 1]); ac = fig.add_subplot(gs[:, 0]); ag = [fig.add_subplot(gs[i, 1]) for i in range(4)]
    names = ["input gate i", "forget gate f", "output gate o", "cell state c"]

    def update(it):
        ac.clear(); k = it // every + 1
        for (nm, v), c in zip(acc.items(), viz.PALETTE): ac.plot(np.arange(len(v[:k])) * every, v[:k], lw=2, color=c, label=nm)
        ac.axhline(.5, color="gray", ls=":"); ac.set(xlim=(0, steps), ylim=(.4, 1.02), xlabel="training step", ylabel="test accuracy", title="Remember the first input across 30 noisy steps"); ac.legend(loc="center right")
        for a, nm, m, (lo, hi) in zip(ag, names, snaps[it], [(0, 1), (0, 1), (0, 1), (-1, 1)]):
            a.clear(); a.imshow(m, aspect="auto", cmap="viridis" if hi == 1 else "RdBu_r", vmin=lo, vmax=hi); a.grid(False); a.set_ylabel(nm, fontsize=8); a.set_yticks([])
            if a is not ag[-1]: a.set_xticks([])
        ag[-1].set_xlabel("time step  (signal arrives at t=0, then noise)"); ag[0].set_title("LSTM internals for one sequence", fontsize=9)
    viz.animate(fig, update, sorted(snaps), gif, fps=4)


if __name__ == "__main__":
    run(main, __file__)
