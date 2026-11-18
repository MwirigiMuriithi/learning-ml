"""
Lesson 32 - Variational autoencoder (VAE), from scratch

Problem story: a lab measures the spectrum of a chemical sample at 24 wavelengths.  Every spectrum is one smooth peak whose
*position* c and *width* w differ from sample to sample (plus instrument noise).  Two hidden factors generate 24 numbers.
A VAE should discover a 2-D "map of all possible spectra", let us walk through it, and invent brand-new realistic spectra.

Implemented (NumPy + the course's tiny MLP kit in common/nn.py, manual back-propagation):
  * encoder q(z|x) = N(mu(x), diag sigma^2(x)),  decoder p(x|z) = N(decoder(z), s^2 I)
  * the reparameterisation trick  z = mu + sigma * eps
  * the evidence lower bound  ELBO = E[log p(x|z)] - beta * KL(q(z|x) || N(0, I))   (closed-form KL)
  * generation from the prior, latent-space traversal, beta-VAE, finite-difference gradient check

Animated:
  vae_training.gif  the latent space organises itself and the decoded grid of spectra becomes smooth and meaningful
  vae_beta.gif      beta = 0.01 ... 16: sharp-but-gappy map vs smooth-but-blurry map; samples from the prior

Run:  python vae.py        (about 30 s)
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from common.nn import MLP, Adam

NF, ZD, SIGMA_X = 24, 2, 0.1
F = np.linspace(0, 1, NF)


# ------------------------------------------------------------------- data
def spectra(n, seed=0, noise=0.03):
    r = rng_(seed)
    c = r.uniform(0.25, 0.75, n); w = r.uniform(0.05, 0.16, n)
    X = np.exp(-0.5 * ((F[None] - c[:, None]) / w[:, None]) ** 2) + r.normal(0, noise, (n, NF))
    return X, c, w


def peak_stats(X):
    """Estimate centre and width of a spectrum by its (clipped) centroid and second moment."""
    P = np.clip(X, 0, None); P = P / (P.sum(1, keepdims=True) + 1e-9)
    c = P @ F; w = np.sqrt(np.maximum(P @ F ** 2 - c ** 2, 1e-9))
    return c, w


# ------------------------------------------------------------------- model
class VAE:
    def __init__(self, beta=1.0, seed=0, hidden=64):
        self.enc = MLP([NF, hidden, hidden, 2 * ZD], act="tanh", seed=seed)
        self.dec = MLP([ZD, hidden, hidden, NF], act="tanh", seed=seed + 1)
        self.beta = beta
        self.opt = Adam(self.enc.params_and_grads() + self.dec.params_and_grads(), lr=2e-3)

    def encode(self, X):
        o = self.enc.forward(X)
        return o[:, :ZD], np.clip(o[:, ZD:], -8, 4)

    def loss_and_grads(self, X, eps=None, r=None):
        """Returns (recon, kl, total) and fills the parameter gradients.  Loss is averaged over the batch."""
        B = len(X)
        mu, lv = self.encode(X)
        if eps is None:
            eps = r.normal(size=mu.shape)
        z = mu + np.exp(0.5 * lv) * eps                           # reparameterisation trick
        xh = self.dec.forward(z)
        recon = np.sum((xh - X) ** 2, 1) / (2 * SIGMA_X ** 2)      # -log p(x|z) up to a constant
        kl = -0.5 * np.sum(1 + lv - mu ** 2 - np.exp(lv), 1)
        total = (recon + self.beta * kl).mean()
        # backward
        self.enc.zero_grad(); self.dec.zero_grad()
        dxh = (xh - X) / SIGMA_X ** 2 / B
        dz = self.dec.backward(dxh)
        dmu = dz + self.beta * mu / B
        dlv = dz * eps * 0.5 * np.exp(0.5 * lv) + self.beta * 0.5 * (np.exp(lv) - 1) / B
        self.enc.backward(np.concatenate([dmu, dlv], 1))
        return recon.mean(), kl.mean(), total

    def step(self, X, r):
        out = self.loss_and_grads(X, r=r)
        self.opt.step()
        return out

    def decode(self, Z):
        return self.dec.forward(Z)


def grad_check():
    r = rng_(0)
    X, _, _ = spectra(6, 1)
    m = VAE(beta=1.0, seed=3, hidden=8)
    eps = r.normal(size=(6, ZD))
    _, _, _ = m.loss_and_grads(X, eps)
    worst = 0
    for net in (m.enc, m.dec):
        for W, dW in net.params_and_grads():
            for _ in range(4):
                idx = tuple(r.integers(0, s) for s in W.shape)
                g = dW[idx]; old = W[idx]
                W[idx] = old + 1e-6; lp = m.loss_and_grads(X, eps)[2]
                W[idx] = old - 1e-6; lm = m.loss_and_grads(X, eps)[2]
                W[idx] = old
                num = (lp - lm) / 2e-6
                worst = max(worst, abs(num - g) / (abs(num) + abs(g) + 1e-9))
    return worst


def train(beta=1.0, steps=2500, seed=0, X=None, snaps=(), B=100, log_every=50):
    r = rng_(seed + 10)
    m = VAE(beta, seed)
    hist, saved = [], {}
    for s in range(steps + 1):
        if s in snaps:
            saved[s] = ({k: [w.copy() for w in getattr(getattr(m, k), "W")] for k in ("enc", "dec")}, {k: [b.copy() for b in getattr(getattr(m, k), "b")] for k in ("enc", "dec")})
        if s % log_every == 0:
            rec, kl, _ = m.loss_and_grads(X[:1000], r=rng_(1))
            hist.append((s, rec, kl))
        if s == steps:
            break
        m.step(X[r.integers(0, len(X), B)], r)
    return m, np.array(hist), saved


def restore(m, snap):
    Ws, bs = snap
    for k in ("enc", "dec"):
        net = getattr(m, k)
        for i in range(len(net.W)):
            net.W[i][...] = Ws[k][i]; net.b[i][...] = bs[k][i]


# ------------------------------------------------------------------- plotting helpers
def grid_panel(ax, m, rng=2.0, n=5, color=INK):
    zs = np.linspace(-rng, rng, n)
    for i, zy in enumerate(zs[::-1]):
        for zx in zs:
            y = m.decode(np.array([[zx, zy]]))[0]
            ax.plot(zx + (F - 0.5) * 0.7, zy + y * 0.6 - 0.3, color=color, lw=1.5)
    ax.set(xlim=(-rng - 0.5, rng + 0.5), ylim=(-rng - 0.5, rng + 0.5), xlabel="latent z₁", ylabel="latent z₂")


# ------------------------------------------------------------------- animations
def training_gif(path, X, c, w, steps=2500):
    snaps = (0, 25, 50, 100, 200, 400, 800, 1500, steps)
    m, hist, saved = train(1.0, steps, 0, X, snaps)
    print(f"  VAE (beta=1) final: reconstruction term {hist[-1, 1]:.1f}, KL {hist[-1, 2]:.2f} nats")
    with Animator(path, fps=1.5, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.9), title="A VAE learns a map of all possible spectra", gridspec_kw={"width_ratios": [1, 1.1, 0.9]})
        for s in snaps:
            for a in (a1, a2, a3):
                a.clear()
            restore(m, saved[s])
            mu, lv = m.encode(X[:600])
            sc = a1.scatter(mu[:, 0], mu[:, 1], c=c[:600], cmap="coolwarm", s=12, edgecolor="none")
            th = np.linspace(0, 2 * np.pi, 100); a1.plot(np.cos(th), np.sin(th), color=INK2, ls="--", lw=1); a1.plot(2 * np.cos(th), 2 * np.sin(th), color=MUTED, ls=":", lw=1)
            a1.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), aspect="equal", title="Encoded spectra (colour = peak position)", xlabel="z₁", ylabel="z₂")
            grid_panel(a2, m)
            a2.set_title("Decoder output on a 5×5 grid of latent points")
            h = hist[hist[:, 0] <= s]
            a3.plot(h[:, 0], h[:, 1], color=BLUE, lw=2, label="reconstruction term")
            a3.plot(h[:, 0], h[:, 2], color=ORANGE, lw=2, label="KL term")
            a3.set(yscale="log", xlim=(0, steps), ylim=(0.5, 1e4), title="The two parts of the loss", xlabel="step"); a3.legend(loc="upper right")
            r_, k_ = h[-1, 1], h[-1, 2]
            caption(fig, f"step {s}: reconstruction {r_:.0f}, KL {k_:.1f}. " + ("Untrained: all codes sit near the centre and the decoder outputs random wiggles" if s == 0 else
                    "The prior pulls codes into a unit-sized cloud while the decoder learns to reproduce spectra" if s < 800 else "Neighbouring latent points decode to similar spectra: peak position and width change smoothly across the grid"))
            anim.grab(fig)
    return m


def beta_gif(path, X, c, w, steps=2500):
    betas = [0.01, 0.3, 1.0, 4.0, 16.0]
    runs = {}
    r = rng_(77)
    for b in betas:
        m, hist, _ = train(b, steps, 0, X)
        rec, kl = hist[-1, 1], hist[-1, 2]
        Xt, ct, wt = spectra(1000, seed=99)
        mu, _ = m.encode(Xt); xhat = m.decode(mu)
        mse = float(np.mean((xhat - Xt) ** 2))
        zs = r.normal(size=(1000, ZD)); gen = m.decode(zs)
        cg, wg = peak_stats(gen); cd, wd = peak_stats(Xt)
        # 'realistic': generated peak centre and width within the range seen in the data
        ok = np.mean((cg > 0.2) & (cg < 0.8) & (wg > 0.04) & (wg < 0.19))
        runs[b] = (m, mse, kl, ok, mu)
        print(f"     beta={b:5}: recon MSE {mse:.4f}, KL {kl:.2f}, prior samples that look like real peaks {ok:.2f}")
    with Animator(path, fps=1.0, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.9), title="β-VAE: how strongly the prior is enforced", gridspec_kw={"width_ratios": [1, 1.1, 0.9]})
        Xt, ct, wt = spectra(600, seed=99)
        for b in betas:
            for a in (a1, a2, a3):
                a.clear()
            m, mse, kl, ok, mu = runs[b]
            mu = m.encode(Xt)[0]
            a1.scatter(mu[:, 0], mu[:, 1], c=ct, cmap="coolwarm", s=10, edgecolor="none")
            th = np.linspace(0, 2 * np.pi, 100); a1.plot(np.cos(th), np.sin(th), color=INK2, ls="--", lw=1)
            a1.set(xlim=(-4, 4), ylim=(-4, 4), aspect="equal", title=f"Latent codes, β = {b}", xlabel="z₁", ylabel="z₂")
            grid_panel(a2, m); a2.set_title("Decoded latent grid")
            zs = rng_(5).normal(size=(40, ZD)); gen = m.decode(zs)
            for g in gen[:20]:
                a3.plot(F, g, color=BLUE, lw=1, alpha=0.5)
            for x in Xt[:6]:
                a3.plot(F, x, color=ORANGE, lw=1.2, alpha=0.8)
            a3.set(ylim=(-0.4, 1.4), title="20 samples from the prior (blue) vs real (orange)", xlabel="wavelength")
            note = ("sharp but gappy: prior samples often land in empty space" if b < 0.1 else
                    "good compromise: codes fill the prior" if b <= 1.0 else
                    "the prior wins: codes squeezed, reconstructions blur" )
            caption(fig, f"β = {b}: MSE {mse:.4f}, KL {kl:.1f} nats, {ok:.0%} of prior samples look like peaks; {note}")
            anim.grab(fig)
    return runs


def main():
    print("Lesson 32: VAE")
    print(f"  gradient check (worst relative error over random weights, fixed noise): {grad_check():.1e}")
    mu = np.array([[0.7, -0.4]]); lv = np.array([[-0.5, 0.3]])
    r = rng_(0); z = mu + np.exp(0.5 * lv) * r.normal(size=(400000, 2))
    logq = -0.5 * (((z - mu) ** 2) / np.exp(lv) + lv + np.log(2 * np.pi)).sum(1)
    logp = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(1)
    print(f"  closed-form KL {(-0.5 * (1 + lv - mu ** 2 - np.exp(lv)).sum()):.4f} vs Monte-Carlo estimate {np.mean(logq - logp):.4f}")
    X, c, w = spectra(2000, seed=0)
    training_gif(here(__file__, "vae_training.gif"), X, c, w)
    print("  beta sweep:")
    beta_gif(here(__file__, "vae_beta.gif"), X, c, w)
    # PCA baseline for reconstruction with 2 components
    Xt, _, _ = spectra(1000, seed=99)
    mu_, V = X.mean(0), np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:2]
    print(f"  baseline: PCA with 2 components, reconstruction MSE {np.mean(((Xt - mu_) @ V.T @ V + mu_ - Xt) ** 2):.4f}; noise floor (sigma^2) {0.03 ** 2:.4f}")


if __name__ == "__main__":
    main()
