"""
Lesson 31 - Autoencoder (from scratch)

Problem: a fleet of industrial machines reports 3 sensor values (vibration x, temperature-like
height h, vibration z).  The healthy readings do not fill 3-D space: they lie on a rolled-up
2-D sheet (a "swiss roll") whose hidden coordinate t is the machine's wear cycle phase.
An autoencoder (AE) squeezes every reading through a 2-number bottleneck and tries to rebuild it.
To succeed it must discover the sheet's own 2-D coordinates - without ever seeing t.

Animated (two GIFs):
  autoencoder.gif          bottleneck AE: 2-D latent space organising (coloured by hidden t),
                           original vs reconstruction in 3-D, loss curve vs the PCA floor
  autoencoder_variants.gif denoising AE (noise removal), linear AE converging to PCA,
                           anomaly detection by reconstruction error (undercomplete vs overcomplete)

Run:  python autoencoder.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from common.nn import MLP, Adam
from matplotlib.colors import LinearSegmentedColormap

BO = LinearSegmentedColormap.from_list("bo", [BLUE, "#f3f2ee", ORANGE])


# ------------------------------------------------------------------ data
def make_sensor_roll(n=800, seed=0, noise=0.0):
    """Healthy machine states: a swiss roll. t = hidden wear phase, h = second free factor."""
    r = rng_(seed)
    t = r.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    h = r.uniform(0, 10, n)
    X = np.c_[t * np.cos(t), h, t * np.sin(t)] + r.normal(0, noise, (n, 3))
    return X, t, h


def make_anomalies(n, mu, sd, seed=5):
    """Faulty readings: same roll but pushed half a turn outward -> lands in the GAP between sheets."""
    r = rng_(seed)
    t = r.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    h = r.uniform(0, 10, n)
    X = np.c_[(t + np.pi) * np.cos(t), h, (t + np.pi) * np.sin(t)]
    return (X - mu) / sd


# ------------------------------------------------------------------ the model
class Autoencoder:
    """x -> encoder -> code z (bottleneck) -> decoder -> x_hat.

    loss = mean_n || x_hat - x ||^2  [+ l1 * mean_n |z|_1 for the sparse variant]
    `noise` > 0 gives a DENOISING AE: the encoder sees x + eps but the loss compares to the clean x.
    """

    def __init__(self, d_in=3, d_code=2, hidden=(32, 32), act="tanh", code_act=None, seed=0, linear=False):
        h = [] if linear else list(hidden)
        self.enc = MLP([d_in] + h + [d_code], act=act, out_act=code_act, seed=seed)
        self.dec = MLP([d_code] + h[::-1] + [d_in], act=act, seed=seed + 100)
        self.opt = Adam(self.enc.params_and_grads() + self.dec.params_and_grads(), lr=3e-3 if not linear else 1e-2)
        self.rng = np.random.default_rng(seed + 7)

    def encode(self, X): return self.enc.forward(X)
    def decode(self, Z): return self.dec.forward(Z)
    def reconstruct(self, X): return self.decode(self.encode(X))

    def errors(self, X):
        """Per-sample reconstruction error ||x - x_hat||^2  (= anomaly score)."""
        return np.sum((self.reconstruct(X) - X) ** 2, 1)

    def step(self, X, noise=0.0, l1=0.0):
        n = len(X)
        Xin = X + self.rng.normal(0, noise, X.shape) if noise > 0 else X
        z = self.enc.forward(Xin)
        xh = self.dec.forward(z)
        loss = np.mean(np.sum((xh - X) ** 2, 1))
        # dL/dxh = 2 (xh - x) / n   -> backprop through decoder gives dL/dz -> through encoder
        self.enc.zero_grad(); self.dec.zero_grad()
        dz = self.dec.backward(2 * (xh - X) / n)
        if l1 > 0:                      # d/dz of l1 * mean |z|_1 = l1 * sign(z) / n
            dz = dz + l1 * np.sign(z) / n
            loss += l1 * np.mean(np.abs(z).sum(1))
        self.enc.backward(dz)
        self.opt.step()
        return loss


def pca_fit(X, k):
    """Top-k principal subspace via SVD: the best LINEAR rank-k reconstruction."""
    mu = X.mean(0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    V = Vt[:k].T
    mse = float(np.sum(S[k:] ** 2) / len(X))
    return mu, V, mse


def auc(score_pos, score_neg):
    """P(anomaly score > normal score) via ranks (Mann-Whitney)."""
    s = np.r_[score_pos, score_neg]
    ranks = s.argsort().argsort() + 1.0
    n1, n0 = len(score_pos), len(score_neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def subspace_angle(A, B):
    """Largest principal angle (degrees) between the column spaces of A and B."""
    qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(s.min(), -1, 1))))


# ------------------------------------------------------------------ main
def main():
    N_STEPS = 3500
    Xraw, t, h = make_sensor_roll(800, seed=0)
    mu, sd = Xraw.mean(0), Xraw.std(0)
    X = (Xraw - mu) / sd                                    # standardise sensors
    Xte_raw, tte, _ = make_sensor_roll(300, seed=1)
    Xte = (Xte_raw - mu) / sd
    Xan = make_anomalies(60, mu, sd)
    Xtr = (make_sensor_roll(2400, seed=11)[0] - mu) / sd    # larger TRAIN set (X, the first 800 readings, is only for plotting)
    rb = np.random.default_rng(42)
    NOISE = 0.30

    pca_mu, V2, pca_mse = pca_fit(Xtr, 2)
    print(f"  PCA (best linear 2-D) reconstruction MSE = {pca_mse:.3f}")

    ae = Autoencoder(d_code=2, seed=1)                      # undercomplete: 3 -> 2 -> 3
    dae = Autoencoder(d_code=8, hidden=(64, 64), seed=3)    # denoising AE: wider code, trained to map noisy -> clean
    lin = Autoencoder(d_code=2, seed=5, linear=True)        # no hidden layers, no activations
    over = Autoencoder(d_code=16, seed=7)                   # overcomplete: code bigger than input
    Xnoisy_te = Xte + np.random.default_rng(9).normal(0, NOISE, Xte.shape)

    steps = set(frame_steps(N_STEPS, 36).tolist())
    S1, S2 = [], []                                         # snapshots for GIF 1 / GIF 2
    loss_ae, loss_lin = [], []
    for s in range(N_STEPS + 1):
        B = Xtr[rb.integers(0, len(Xtr), 400)]               # mini-batch of 400 readings
        la = ae.step(B); ld = dae.step(B, noise=NOISE); ll = lin.step(B); lo = over.step(B)
        loss_ae.append(float(np.mean(ae.errors(Xtr)))); ll = float(np.mean(lin.errors(Xtr))); loss_lin.append(ll)   # full-data losses
        if s in steps:
            S1.append(dict(s=s, z=ae.encode(X), xh=ae.decode(ae.encode(X))))
            S2.append(dict(s=s, dn=dae.reconstruct(Xnoisy_te), lin=ll,
                           ang=subspace_angle(lin.dec.W[0].T, V2),
                           e_under=(ae.errors(Xte), ae.errors(Xan)),
                           e_over=(over.errors(Xte), over.errors(Xan))))
    print(f"  AE (2-D code) final train MSE = {np.mean(ae.errors(Xtr)):.4f}   linear AE = {loss_lin[-1]:.3f} (PCA {pca_mse:.3f})")
    te_mse = np.mean(ae.errors(Xte))
    print(f"  AE test MSE = {te_mse:.4f}  | denoised-vs-clean MSE = {np.mean((dae.reconstruct(Xnoisy_te) - Xte) ** 2) * 3:.4f}"
          f" vs noisy-vs-clean {np.mean((Xnoisy_te - Xte) ** 2) * 3:.4f}")
    a_u = auc(ae.errors(Xan), ae.errors(Xte)); a_o = auc(over.errors(Xan), over.errors(Xte))
    print(f"  anomaly AUC: undercomplete(2-D) = {a_u:.3f}   overcomplete(16-D) = {a_o:.3f}")

    # optional sparse AE demo: relu code + L1 penalty -> few active units
    sp = Autoencoder(d_code=16, code_act="relu", seed=11); pl = Autoencoder(d_code=16, code_act="relu", seed=11)
    for _ in range(1500):
        B = Xtr[rb.integers(0, len(Xtr), 400)]
        sp.step(B, l1=0.05); pl.step(B)
    print(f"  sparse AE: active code units {np.mean(sp.encode(X) > 1e-6) * 100:.0f}% (L1) vs "
          f"{np.mean(pl.encode(X) > 1e-6) * 100:.0f}% (no L1); recon MSE {np.mean(sp.errors(X)):.3f} vs {np.mean(pl.errors(X)):.3f}")

    # ------------------------------------------------ GIF 1: the bottleneck AE learning
    fig = plt.figure(figsize=(13, 4.8))
    fig.suptitle("An autoencoder discovers the hidden 2-D layout of 3-D sensor data by being forced to rebuild it",
                 fontsize=12.5, fontweight="bold", x=0.02, ha="left")
    a1 = fig.add_subplot(1, 3, 1); a2 = fig.add_subplot(1, 3, 2, projection="3d"); a3 = fig.add_subplot(1, 3, 3)
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.17, top=0.86, wspace=0.22)
    with Animator(here(__file__, "autoencoder.gif"), fps=8, hold=2.5) as anim:
        for sn in S1:
            s, z, xh = sn["s"], sn["z"], sn["xh"]
            a1.clear()
            a1.scatter(z[:, 0], z[:, 1], c=t, cmap=BO, s=12, edgecolor="none")
            a1.set(title="Latent space (2-D code), colour = hidden t", xlabel="code z1", ylabel="code z2")
            a1.text(0.02, 0.03, "blue -> orange = wear phase t\n(never shown to the model)", transform=a1.transAxes, fontsize=8, color=INK2)
            a2.clear()
            a2.scatter(X[:, 0], X[:, 2], X[:, 1], s=5, color=BLUE, alpha=0.45, depthshade=False)
            a2.scatter(xh[:, 0], xh[:, 2], xh[:, 1], s=5, color=ORANGE, alpha=0.8, depthshade=False)
            a2.set(title="Original (blue) vs reconstruction (orange)")
            a2.set_xlim(-2, 2); a2.set_ylim(-2, 2); a2.set_zlim(-2, 2)
            a2.view_init(22, -62)
            for ax_ in (a2.xaxis, a2.yaxis, a2.zaxis):
                ax_.set_ticklabels([])
            a3.clear()
            a3.plot(range(s + 1), loss_ae[: s + 1], color=BLUE)
            a3.axhline(pca_mse, color=ORANGE, linestyle="--", linewidth=1.2)
            a3.text(N_STEPS, pca_mse * 1.25, f"PCA floor (best linear 2-D): {pca_mse:.2f}", ha="right", fontsize=8.5, color=INK2)
            a3.set(title="Reconstruction loss", xlabel="training step", ylabel="MSE per sample", yscale="log")
            a3.set_xlim(0, N_STEPS); a3.set_ylim(2e-3, 4)
            caption(fig, f"step {s:>4}:  reconstruction MSE = {loss_ae[s]:.4f}   (PCA with the same 2 numbers cannot go below {pca_mse:.2f})")
            anim.grab(fig)
    plt.close(fig)

    # ------------------------------------------------ GIF 2: variants
    Xg = Xte * sd + mu   # un-standardised for the denoising view (x, z plane)
    clean_t = np.linspace(1.5 * np.pi, 4.5 * np.pi, 300)
    spiral = (np.c_[clean_t * np.cos(clean_t), clean_t * np.sin(clean_t)] - mu[[0, 2]]) / sd[[0, 2]]
    fig, (b1, b2, b3) = new_fig(1, 3, figsize=(13, 4.8))
    fig.suptitle("Three autoencoder variants: denoising, linear (= PCA), and anomaly detection", fontsize=12.5,
                 fontweight="bold", x=0.02, ha="left")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    r = np.random.default_rng(1)
    with Animator(here(__file__, "autoencoder_variants.gif"), fps=8, hold=2.5) as anim:
        for k, sn in enumerate(S2):
            s = sn["s"]
            b1.clear()
            b1.plot(spiral[:, 0], spiral[:, 1], color=MUTED, linewidth=3, alpha=0.5, zorder=1)
            b1.scatter(Xnoisy_te[:, 0], Xnoisy_te[:, 2], s=9, color=MUTED, alpha=0.7, zorder=2)
            b1.scatter(sn["dn"][:, 0], sn["dn"][:, 2], s=9, color=ORANGE, zorder=3)
            b1.set(title="Denoising AE: noisy in, clean out", xlabel="sensor x", ylabel="sensor z")
            b1.set_xlim(-2.4, 2.4); b1.set_ylim(-2.4, 2.4); b1.set_aspect("equal")
            b1.text(-2.3, 2.15, "grey = noisy input, orange = output,\nthick grey line = true healthy sheet", fontsize=7.5, color=INK2, va="top")

            b2.clear()
            idx = np.arange(k + 1)
            b2.plot([q["s"] for q in S2[: k + 1]], [q["lin"] for q in S2[: k + 1]], color=BLUE)
            b2.axhline(pca_mse, color=ORANGE, linestyle="--", linewidth=1.2)
            b2.text(N_STEPS, pca_mse * 1.02, "PCA optimum (SVD)", ha="right", va="bottom", fontsize=8.5, color=INK2)
            b2.set(title="Linear AE (no activations) lands on PCA", xlabel="training step", ylabel="MSE per sample")
            b2.set_xlim(0, N_STEPS); b2.set_ylim(pca_mse * 0.97, max(q["lin"] for q in S2) * 1.02)
            b2.text(0.97, 0.55, f"angle to PCA subspace: {sn['ang']:.1f} deg", transform=b2.transAxes, ha="right", fontsize=9, color=INK)

            b3.clear()
            for j, (key, lab) in enumerate([("e_under", "2-D code"), ("e_over", "16-D code")]):
                en, ea = sn[key]
                b3.scatter(j - 0.18 + r.uniform(-.1, .1, len(en)) * 0 + np.linspace(-.1, .1, len(en)), en + 1e-6, s=6, color=BLUE, alpha=0.6)
                b3.scatter(j + 0.18 + np.linspace(-.1, .1, len(ea)), ea + 1e-6, s=6, color=RED, alpha=0.8)
                b3.text(j, 1.6e-6, f"AUC {auc(ea, en):.2f}", ha="center", fontsize=9, color=INK)
            b3.set(title="Anomaly score = reconstruction error", ylabel="||x - x_hat||^2", yscale="log")
            b3.set_xticks([0, 1]); b3.set_xticklabels(["undercomplete (2-D code)", "overcomplete (16-D code)"], fontsize=8)
            b3.set_xlim(-0.6, 1.6); b3.set_ylim(1e-6 * 0.8, 20)
            b3.text(1.58, 12, "blue = healthy, red = faulty", ha="right", va="top", fontsize=8, color=INK2)
            caption(fig, f"step {s:>4}:  denoiser MSE vs clean = {np.mean((sn['dn'] - Xte) ** 2) * 3:.3f} (input noise {np.mean((Xnoisy_te - Xte) ** 2) * 3:.3f})"
                         f"   |   linear AE = {sn['lin']:.3f} vs PCA {pca_mse:.3f}")
            anim.grab(fig)
    plt.close(fig)

    # optional cross-check against sklearn PCA
    try:
        from sklearn.decomposition import PCA
        p = PCA(2).fit(Xtr); m = np.mean(np.sum((p.inverse_transform(p.transform(Xtr)) - Xtr) ** 2, 1))
        print(f"  sklearn PCA MSE {m:.4f} vs mine {pca_mse:.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
