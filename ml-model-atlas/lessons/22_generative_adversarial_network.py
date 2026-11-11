"""
# 22 · Generative Adversarial Networks (GAN)
**Family:** Unsupervised · Generative model (adversarial training)

## Definition
A GAN trains two networks in a game. The **generator G** turns random noise into fake samples; the **discriminator D** tries to tell real samples from fakes.
G is rewarded when D is fooled; D is rewarded when it is not. At equilibrium G's samples match the real distribution and D can only guess (output 0.5).

## Intuition
A counterfeiter and a detective. As the detective gets better at spotting fakes, the counterfeiter must make better ones. We use the simplest possible
setting, 1-D data, so you can see everything: the real data are two bumps (e.g. commuter arrival times: morning and evening rush), and G must learn to map a single
bell-curve of noise onto *two* bumps.

## The math
    D(x) = P(x is real)          G(z), z ~ N(0, I)
    min_G max_D  E_x[log D(x)] + E_z[log(1 − D(G(z)))]
    In practice D minimizes BCE(real→1, fake→0); G uses the "non-saturating" loss  −log D(G(z))  (stronger gradients early on).
    Optimal discriminator for fixed G:  D*(x) = p_data(x) / (p_data(x) + p_G(x))     ->  the green curve below
    Note: the generated bumps come out a little narrower/spikier than the real ones, a mild form of mode collapse that is typical of plain GANs.
    Known troubles: mode collapse (G covers only one bump), oscillating training, vanishing gradients. Remedies: WGAN, spectral norm, careful lr.

## What the animation shows
Left: histogram of real data (grey) vs generated samples (orange) with the discriminator's belief D(x) (green line). Middle: the generator's learned mapping z → G(z); it must become
a staircase to make two bumps from one (the noise has 4 dimensions so it can also spread each bump). Right: the two losses oscillating (healthy for GANs).

## Try this
- Move the bumps further apart (±3) and see if mode collapse appears.
- Train D 5 steps per G step (WGAN-style) and compare stability.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import viz
from mlatlas.nn import T, MLP, Adam, bce_logits
from mlatlas.cli import run


def real(rng, n): return np.where(rng.random((n, 1)) < .5, rng.normal(-1.5, .4, (n, 1)), rng.normal(1.5, .4, (n, 1)))


def main(gif, quick):
    rng = np.random.default_rng(3); G, D = MLP([4, 32, 32, 1], rng), MLP([1, 32, 32, 1], rng, act="relu")
    og, od = Adam(G.params(), lr=2e-3, b1=.5), Adam(D.params(), lr=2e-3, b1=.5); steps = 400 if quick else 3000; B = 128
    zf = rng.normal(size=(1500, 4)); xs = np.linspace(-4, 4, 200)[:, None]; zs = np.zeros((100, 4)); zs[:, 0] = np.linspace(-3, 3, 100); snaps = {}; ld, lg = [], []
    for it in range(steps + 1):
        xr = real(rng, B); z = T(rng.normal(size=(B, 4))); xf = G(z)
        od.zero(); l_d = bce_logits(D(T(xr)), 1.0) + bce_logits(D(T(xf.data)), 0.0); l_d.backward(); od.step()
        og.zero(); od.zero(); l_g = bce_logits(D(xf), 1.0); l_g.backward(); og.step(); ld.append(float(l_d.data)); lg.append(float(l_g.data))
        if it % (20 if quick else 50) == 0: snaps[it] = (G(T(zf)).data[:, 0].copy(), (1 / (1 + np.exp(-D(T(xs)).data[:, 0]))), G(T(zs)).data[:, 0].copy(), len(ld))
    gen = snaps[steps][0]; print(f"  generated mean {gen.mean():.2f} (real 0.0), std {gen.std():.2f} (real ≈1.6); fraction in left bump {np.mean(gen < 0):.2f} (real 0.50)")
    sm = lambda v, k=30: np.convolve(v, np.ones(k) / k, mode="valid")
    fig, (a1, a2, a3) = viz.subplots(1, 3, w=14, h=4.3); rr = real(np.random.default_rng(0), 3000)[:, 0]; tw = a1.twinx()

    def update(it):
        for a in (a1, a2, a3, tw): a.clear()
        g, d, m, k = snaps[it]; a1.hist(rr, bins=60, range=(-4, 4), density=True, color="#ccc", label="real"); a1.hist(g, bins=60, range=(-4, 4), density=True, alpha=.6, color=viz.PALETTE[1], label="generated")
        tw.plot(xs[:, 0], d, color=viz.PALETTE[2], lw=2); tw.set_ylim(0, 1); tw.set_ylabel("D(x)", color=viz.PALETTE[2]); tw.grid(False); tw.spines["right"].set_visible(True); a1.set(title=f"step {it}: distributions and the discriminator", ylim=(0, 1.6)); a1.legend(loc="upper left")
        a2.plot(zs[:, 0], m, color=viz.PALETTE[3], lw=2.5); a2.set(title="Generator mapping  z₁ -> G(z) (other noise dims = 0)", xlabel="noise z₁", ylabel="G(z)", ylim=(-4, 4))
        a3.plot(sm(ld[:k]) if k > 30 else ld[:k], color=viz.PALETTE[0], label="D loss"); a3.plot(sm(lg[:k]) if k > 30 else lg[:k], color=viz.PALETTE[3], label="G loss"); a3.set(xlim=(0, steps), ylim=(0, 3), title="Adversarial losses (smoothed)", xlabel="step"); a3.legend()
    viz.animate(fig, update, sorted(snaps), gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
