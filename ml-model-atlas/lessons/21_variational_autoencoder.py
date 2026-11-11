"""
# 21 · Variational Autoencoder (VAE)
**Family:** Unsupervised · Generative model

## Definition
A VAE is an autoencoder whose code is a *probability distribution*, not a point: the encoder outputs a mean μ and variance σ² and we
sample z ~ N(μ, σ²). An extra loss pulls those distributions toward a standard normal prior N(0, I). The latent space becomes smooth and
dense, so **sampling z from the prior and decoding generates brand-new, realistic data**.

## Intuition
A plain autoencoder can scatter codes anywhere, leaving holes where the decoder produces garbage. The VAE's KL penalty packs all codes into one tidy blob
around the origin, and sampling noise forces nearby codes to decode to similar outputs. Here the data is two interleaved moons; we learn a 2-D latent
space and then generate new moon points from pure noise.

## The math
    encoder: x -> (μ, log σ²)          reparameterization trick:  z = μ + σ ⊙ ε ,  ε ~ N(0, I)   (makes sampling differentiable)
    decoder: z -> x̂
    ELBO loss:  L = E||x − x̂||² / (2s²)  +  KL( N(μ,σ²) || N(0,I) ),   KL = −½ Σ (1 + log σ² − μ² − σ²)
    Trade-off: reconstruction (accuracy) vs KL (regularity). β-VAE scales the KL term to disentangle factors.

## Practical tips
- Output tends to be blurry for images (Gaussian likelihood); VQ-VAE and diffusion models improve on that.
- Watch for "posterior collapse" (KL → 0, decoder ignores z): anneal β from 0 up.

## What the animation shows
Left: the 2-D latent space; every training point's encoded mean, coloured by moon, and the N(0,1) prior circle. Middle: data (grey) vs points *generated from random noise* (red).
Right: reconstruction and KL loss terms. Early generations are a blob; later they concentrate around the two moons. Notice that some samples still land in the gap between them: a VAE trades sharpness for coverage, one reason GANs and diffusion models give crisper samples.

## Try this
- Walk a straight line in latent space between a point of each moon and decode it (latent interpolation).
- Set the KL weight to 0 (plain AE) and sample from the prior: how many generated points are off the moons?
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.nn import T, MLP, Adam
from mlatlas.cli import run


def main(gif, quick):
    rng = np.random.default_rng(0); X, c = data.moons(400, .1, seed=0); X = (X - X.mean(0)) / X.std(0)
    enc, dec = MLP([2, 48, 4], rng), MLP([2, 48, 2], rng); opt = Adam(enc.params() + dec.params(), lr=.005); steps = 150 if quick else 1500; s2 = .07 ** 2; snaps = {}; hist = []
    fixed = rng.normal(size=(300, 2))
    for it in range(steps + 1):
        idx = rng.integers(0, len(X), 128); xb = X[idx]; o = enc(T(xb)); mu, lv = o[:, 0:2], o[:, 2:4]; z = mu + (lv * .5).exp() * T(rng.normal(size=mu.shape))
        rec = ((dec(z) - xb) ** 2).sum(1).mean() * (1 / (2 * s2)); kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean(); loss = rec + kl; hist.append((float(rec.data), float(kl.data)))
        if it % (15 if quick else 30) == 0: snaps[it] = (enc(T(X)).data[:, :2].copy(), dec(T(fixed)).data.copy(), len(hist))
        opt.zero(); loss.backward(); opt.step()
    print(f"  final reconstruction term {hist[-1][0]:.1f}, KL {hist[-1][1]:.2f}")
    fig, (a1, a2, a3) = viz.subplots(1, 3, w=14, h=4.3); th = np.linspace(0, 2 * np.pi, 100)

    def update(it):
        for a in (a1, a2, a3): a.clear()
        mu, gen, k = snaps[it]; viz.scatter(a1, mu, c, s=10); a1.plot(np.cos(th) * 2, np.sin(th) * 2, "k--", lw=1); a1.set(title=f"step {it}: latent space (dashed = 2σ of prior)", xlim=(-4, 4), ylim=(-4, 4))
        a2.scatter(*X.T, s=8, color="#ccc", label="real data"); a2.scatter(*gen.T, s=10, color=viz.PALETTE[3], label="generated from noise"); a2.set(title="Generation: decode random z ~ N(0, I)", xlim=(-2.6, 2.6), ylim=(-2.6, 2.6)); a2.legend(loc="upper left")
        h = np.array(hist[:k]); a3.semilogy(h[:, 0], label="reconstruction", color=viz.PALETTE[0]); a3.semilogy(h[:, 1] + 1e-3, label="KL to prior", color=viz.PALETTE[1]); a3.set(xlim=(0, steps), title="Loss terms", xlabel="step"); a3.legend()
    viz.animate(fig, update, sorted(snaps), gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
