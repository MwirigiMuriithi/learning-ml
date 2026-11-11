"""
# 00 · Gradient Descent & Optimizers
**Family:** Foundations · Optimization

## Definition
Nearly every model in this course learns by *minimizing a loss function* L(θ) that measures how wrong the model is.
**Gradient descent** repeatedly nudges the parameters θ in the direction that reduces the loss fastest (the negative gradient).
Optimizers are different rules for *how big* and *in which direction* each nudge is.

## Intuition
Picture a hiker in thick fog on a hillside who can only feel the slope under their feet. Plain gradient descent takes a
step straight downhill. In a long narrow valley the slope mostly points at the valley *walls*, so the hiker zig-zags.
Momentum remembers where it has been heading and glides along the valley floor. RMSprop/Adam rescale each direction
so steep directions take small steps and flat directions take big ones.

## The math
    SGD       θ ← θ − η·g
    Momentum  v ← β·v + g          θ ← θ − η·v
    RMSprop   s ← ρ·s + (1−ρ)·g²  θ ← θ − η·g / (√s + ε)
    Adam      m ← β1·m + (1−β1)·g ;  v ← β2·v + (1−β2)·g²
              θ ← θ − η · m̂ / (√v̂ + ε)      (m̂, v̂ are bias-corrected)
    g = ∇L(θ) (+ noise when computed on a mini-batch)

## When to use what
- **SGD (+momentum)**: the workhorse for CNNs; generalizes very well, needs a tuned learning rate schedule.
- **Adam / AdamW**: the default for transformers, RNNs, GANs, anything noisy or badly scaled.
- **Learning rate η** is the most important hyper-parameter in deep learning: too big diverges, too small crawls.

## What the animation shows
Four optimizers start at the same point on the loss surface f(x,y) = x²/20 + y² (a narrow valley, 20× flatter along x),
all receiving the same noisy gradients (like random mini-batches). Right: loss vs step on a log scale.

## Try this
- Change SGD's learning rate to 1.05 and watch it diverge (η·curvature > 2).
- Remove the noise and compare. Which optimizers are most robust to it?
- Replace the surface with the Rosenbrock function f = (1−x)² + 100(y−x²)².
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import viz
from mlatlas.cli import run


class SGD:
    def __init__(s, lr): s.lr = lr
    def step(s, p, g): return p - s.lr * g


class Momentum:
    def __init__(s, lr, beta=.9): s.lr, s.b, s.v = lr, beta, 0
    def step(s, p, g): s.v = s.b * s.v + g; return p - s.lr * s.v


class RMSprop:
    def __init__(s, lr, rho=.9): s.lr, s.rho, s.s = lr, rho, 0
    def step(s, p, g): s.s = s.rho * s.s + (1 - s.rho) * g * g; return p - s.lr * g / (np.sqrt(s.s) + 1e-8)


class Adam:
    def __init__(s, lr, b1=.9, b2=.999): s.lr, s.b1, s.b2, s.m, s.v, s.t = lr, b1, b2, 0, 0, 0
    def step(s, p, g):
        s.t += 1; s.m = s.b1 * s.m + (1 - s.b1) * g; s.v = s.b2 * s.v + (1 - s.b2) * g * g
        return p - s.lr * (s.m / (1 - s.b1 ** s.t)) / (np.sqrt(s.v / (1 - s.b2 ** s.t)) + 1e-8)


def main(gif, quick):
    steps = 25 if quick else 70
    f = lambda p: p[0] ** 2 / 20 + p[1] ** 2
    g = lambda p: np.array([p[0] / 10, 2 * p[1]])
    start = np.array([-8., 3.])
    opts = {"SGD": SGD(.45), "Momentum": Momentum(.05), "RMSprop": RMSprop(.3), "Adam": Adam(.5)}
    rng = np.random.default_rng(1)
    paths = {k: [start.copy()] for k in opts}
    for _ in range(steps):
        noise = rng.normal(0, .3, 2)  # the same "mini-batch noise" for everyone
        for k, o in opts.items():
            p = paths[k][-1]; paths[k].append(o.step(p, g(p) + noise))
    paths = {k: np.array(v) for k, v in paths.items()}
    for k, P in paths.items(): print(f"  {k:9s} final loss {f(P[-1]):.4f}")
    xs, ys = np.meshgrid(np.linspace(-10, 3, 200), np.linspace(-4, 4, 200)); Z = xs ** 2 / 20 + ys ** 2
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4)
    cols = dict(zip(opts, viz.PALETTE))

    def update(t):
        a1.clear(); a2.clear()
        a1.contour(xs, ys, Z, levels=np.geomspace(.05, 40, 14), cmap="Greys", linewidths=.6)
        for k, P in paths.items():
            a1.plot(P[:t + 1, 0], P[:t + 1, 1], color=cols[k], lw=1.6, alpha=.8, label=k)
            a1.scatter(P[t, 0], P[t, 1], color=cols[k], s=45, zorder=5, edgecolor="k")
            a2.semilogy([f(p) + 1e-4 for p in P[:t + 1]], color=cols[k], label=k)
        a1.scatter(0, 0, marker="*", s=130, color="gold", edgecolor="k", zorder=6)
        a1.set(title=f"Loss surface: a narrow valley (step {t})", xlim=(-10, 3), ylim=(-4, 4)); a1.legend(loc="upper left")
        a2.set(title="Loss vs step (log scale)", xlim=(0, steps), ylim=(1e-4, 60), xlabel="step")
    viz.animate(fig, update, range(steps + 1), gif, fps=8)


if __name__ == "__main__":
    run(main, __file__)
