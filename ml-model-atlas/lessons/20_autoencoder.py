"""
# 20 · Autoencoders
**Family:** Unsupervised · Representation learning / dimensionality reduction

## Definition
An autoencoder is a neural network trained to **reconstruct its own input** through a narrow **bottleneck**. The *encoder* compresses x into a
short code z; the *decoder* rebuilds x̂ from z. No labels are needed: the target is the input itself.

## Intuition
If the data really lives on a low-dimensional surface (a curve in 2-D), the bottleneck forces the network to discover coordinates *along* that surface.
Here noisy sensor points lie on a bent arc; a 1-number bottleneck must learn "how far along the arc am I?". It is a **non-linear PCA**: with linear layers it recovers PCA exactly.

## The math
    z = enc(x) = W₂ tanh(W₁x + b₁) + b₂          x̂ = dec(z)
    L = 1/n Σ ||x − x̂||²
    Variants: denoising AE (corrupt x, reconstruct clean x), sparse AE (penalty on z), contractive AE, convolutional AE, VAE (lesson 21).
    Uses: compression, denoising, anomaly detection (large reconstruction error = unusual), feature learning, pre-training.

## What the animation shows
Left: data (grey), each point's reconstruction (coloured by its latent value z) and the decoder's learned curve (black), the manifold it discovered. Middle: histogram of the 1-D codes.
Right: reconstruction loss. The decoder curve starts as a blob near the mean and unfolds along the arc.

## Try this
- Use a 2-D bottleneck on the same data: reconstruction becomes perfect, but is the code still meaningful?
- Add 20 off-curve points and compare their reconstruction error to inliers: that is anomaly detection.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import viz
from mlatlas.nn import T, MLP, Adam, mse
from mlatlas.cli import run


def main(gif, quick):
    rng = np.random.default_rng(0); th = rng.uniform(0, 1.35 * np.pi, 300); X = np.c_[1.6 * np.cos(th), np.sin(th) * 1.2] + rng.normal(0, .07, (300, 2))
    enc, dec = MLP([2, 16, 1], rng), MLP([1, 16, 2], rng); opt = Adam(enc.params() + dec.params(), lr=.02); steps = 150 if quick else 600; snaps = {}; losses = []
    for it in range(steps + 1):
        z = enc(T(X)); loss = mse(dec(z), X); losses.append(float(loss.data))
        if it % (15 if quick else 12) == 0:
            zz = z.data[:, 0]; g = np.linspace(zz.min(), zz.max(), 80)[:, None]; snaps[it] = (dec(T(z.data)).data, zz.copy(), dec(T(g)).data, len(losses))
        opt.zero(); loss.backward(); opt.step()
    print(f"  reconstruction MSE {losses[0]:.3f} -> {losses[-1]:.4f}")
    fig, (a1, a2, a3) = viz.subplots(1, 3, w=14, h=4.3)

    def update(it):
        for a in (a1, a2, a3): a.clear()
        R, z, curve, k = snaps[it]; a1.scatter(*X.T, s=10, color="#bbb"); a1.scatter(*R.T, s=12, c=z, cmap="viridis", zorder=3); a1.plot(*curve.T, "k", lw=2); a1.set(title=f"step {it}: data, reconstructions and decoder curve", xlim=(-2.4, 2.4), ylim=(-1.6, 1.8))
        a2.hist(z, bins=25, color=viz.PALETTE[0]); a2.set(title="1-D latent code z")
        a3.semilogy(losses[:k], color=viz.PALETTE[3]); a3.set(xlim=(0, steps), ylim=(1e-3, 2), title="reconstruction loss", xlabel="step")
    viz.animate(fig, update, sorted(snaps), gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
