"""
Lesson 33 - Generative adversarial networks (GAN and WGAN), from scratch

Problem story: a bank must test its fraud-detection software but cannot share real customer data.  It wants a *generator* that
produces synthetic transaction summaries that look real.  We work with a 2-D toy version of that data: eight tight clusters of
customer behaviour arranged in a ring (eight customer segments).  A good generator must reproduce ALL eight clusters, not only
the easiest one.

Implemented (NumPy + the course's tiny MLP kit in common/nn.py, manual back-propagation through both networks):
  * the original GAN game with the non-saturating generator loss   (D: real=1, fake=0;  G: make D say 1)
  * WGAN with weight clipping (critic scores instead of a probability; a Wasserstein-distance objective)
  * mode coverage and sample-quality metrics

Animated:
  gan_training.gif  generator samples chase the real data while the discriminator's "realness" map reshapes
  gan_vs_wgan.gif   the same task with the Wasserstein critic (both cover all 8 clusters; their quality curves differ)

Run:  python gan.py        (about 1 minute)
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from common.nn import MLP, Adam

NMODES, RADIUS, STD, ZD = 8, 2.0, 0.08, 2
CENTERS = RADIUS * np.c_[np.cos(2 * np.pi * np.arange(NMODES) / NMODES), np.sin(2 * np.pi * np.arange(NMODES) / NMODES)]


def real_batch(n, r):
    k = r.integers(0, NMODES, n)
    return CENTERS[k] + r.normal(0, STD, (n, 2))


def coverage(samples, tol=0.3):
    """(modes with at least 1% of the samples nearby, fraction of samples within `tol` of some mode centre)."""
    d = np.linalg.norm(samples[:, None] - CENTERS[None], axis=2)
    near = d.min(1) < tol
    per = np.array([np.mean(d.argmin(1)[near] == k) * near.mean() for k in range(NMODES)])
    return int((per >= 0.01).sum()), float(near.mean())


class RMSprop:
    def __init__(self, pg, lr=5e-5):
        self.pg, self.lr = pg, lr
        self.v = [np.zeros_like(p) for p, _ in pg]

    def step(self):
        for (p, g), v in zip(self.pg, self.v):
            v[...] = 0.9 * v + 0.1 * g * g
            p -= self.lr * g / (np.sqrt(v) + 1e-8)


def make_nets(seed, wgan):
    G = MLP([ZD, 64, 64, 2], act="relu", seed=seed)
    D = MLP([2, 64, 64, 1], act="leaky_relu", out_act=None if wgan else None, seed=seed + 1)
    return G, D


def train(kind="gan", steps=3000, seed=0, B=128, snaps=(), n_critic=1):
    r = rng_(seed + 5)
    G, D = make_nets(seed, kind == "wgan")
    if kind == "wgan":
        oG, oD = RMSprop(G.params_and_grads(), 5e-5 * 4), RMSprop(D.params_and_grads(), 5e-5 * 4)
        n_critic = 5
    else:
        oG, oD = Adam(G.params_and_grads(), lr=1e-3, b1=0.5), Adam(D.params_and_grads(), lr=1e-3, b1=0.5)
    zfix = rng_(999).normal(size=(1000, ZD))
    hist, saved = [], {}
    gx, gy = np.meshgrid(np.linspace(-3.2, 3.2, 60), np.linspace(-3.2, 3.2, 60))
    grid = np.c_[gx.ravel(), gy.ravel()]
    for s in range(steps + 1):
        if s % 20 == 0 or s in snaps:
            fake = G.forward(zfix); m, q = coverage(fake)
            dr = D.forward(real_batch(500, rng_(1))); df = D.forward(fake[:500])
            if kind == "gan":
                dr, df = sigmoid(dr), sigmoid(df)
            hist.append((s, m, q, float(dr.mean()), float(df.mean())))
            if s in snaps:
                dm = D.forward(grid).reshape(gx.shape)
                saved[s] = (fake.copy(), sigmoid(dm) if kind == "gan" else dm)
        if s == steps:
            break
        for _ in range(n_critic):
            xr = real_batch(B, r); xf = G.forward(r.normal(size=(B, ZD)))
            D.zero_grad()
            if kind == "gan":
                # D loss = -log D(real) - log(1 - D(fake));  logits o:  dL/do = sigmoid(o) - label
                o = D.forward(xr); D.backward((sigmoid(o) - 1) / B)
                o = D.forward(xf); D.backward(sigmoid(o) / B)
                oD.step()
            else:
                # critic loss = mean D(fake) - mean D(real); weights clipped to [-c, c]
                D.forward(xr); D.backward(-np.ones((B, 1)) / B)
                D.forward(xf); D.backward(np.ones((B, 1)) / B)
                oD.step()
                for W in D.W + D.b:
                    np.clip(W, -0.05, 0.05, out=W)
        # generator step
        z = r.normal(size=(B, ZD)); xf = G.forward(z)
        o = D.forward(xf)
        D.zero_grad(); G.zero_grad()
        dout = (sigmoid(o) - 1) / B if kind == "gan" else -np.ones((B, 1)) / B     # non-saturating loss: -log D(G(z))
        dxf = D.backward(dout)
        G.backward(dxf)
        oG.step()
    return G, D, np.array(hist), saved


def draw_run(fig_axes, kind, s, saved, hist, steps):
    a1, a2, a3 = fig_axes
    fake, dmap = saved[s]
    gx = np.linspace(-3.2, 3.2, 60)
    a1.contourf(gx, gx, dmap, levels=12, cmap="RdBu_r" if kind == "gan" else "RdBu_r", alpha=0.35)
    a1.scatter(CENTERS[:, 0], CENTERS[:, 1], s=240, facecolor="none", edgecolor=INK2, lw=1)
    a1.scatter(fake[:, 0], fake[:, 1], s=6, color=ORANGE, alpha=0.7, label="generated")
    a1.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), aspect="equal", title="Samples (orange) vs real clusters (rings)", xlabel="feature 1", ylabel="feature 2")
    h = hist[hist[:, 0] <= s]
    a2.plot(h[:, 0], h[:, 1], color=BLUE, lw=2)
    a2.set(xlim=(0, steps), ylim=(0, 8.4), title="Clusters covered (of 8)", xlabel="step")
    a3.plot(h[:, 0], h[:, 2], color=AQUA, lw=2, label="samples near a real cluster")
    if kind == "gan":
        a3.plot(h[:, 0], h[:, 3], color=INK2, lw=1.4, label="D(real)"); a3.plot(h[:, 0], h[:, 4], color=ORANGE, lw=1.4, label="D(fake)")
    a3.set(xlim=(0, steps), ylim=(0, 1.02), title="Quality and D's verdicts" if kind == "gan" else "Quality (fraction near a real cluster)", xlabel="step"); a3.legend(loc="lower right", fontsize=7.5)
    return h[-1]


def gan_gif(path, steps=3000, seed=0):
    snaps = tuple(sorted(set([0, 50, 100, 200, 300, 500, 800, 1200, 1700, 2300, steps])))
    G, D, hist, saved = train("gan", steps, seed, snaps=snaps)
    with Animator(path, fps=1.5, hold=3.5) as anim:
        fig, axes = new_fig(1, 3, figsize=(13, 4.8), title="GAN: a forger (G) and a detective (D) improve each other", gridspec_kw={"width_ratios": [1.2, 0.9, 1.0]})
        for s in snaps:
            for a in axes:
                a.clear()
            h = draw_run(axes, "gan", s, saved, hist, steps)
            caption(fig, f"step {s}: {int(h[1])}/8 clusters covered, {h[2]:.0%} of samples near a cluster; D(real) {h[3]:.2f}, D(fake) {h[4]:.2f} (0.5 = cannot tell)")
            anim.grab(fig)
    return hist


def compare_gif(path, steps=3000):
    out = {}
    for kind, st in (("gan", steps), ("wgan", steps)):
        for seed in (0, 1, 2):
            G, D, hist, saved = train(kind, st, seed, snaps=(st,))
            out[(kind, seed)] = (hist, saved)
            print(f"     {kind:4s} seed {seed}: clusters covered {int(hist[-1, 1])}/8, samples near a cluster {hist[-1, 2]:.2f}")
            if kind == "wgan" and seed == 0:
                gap = hist[:, 3] - hist[:, 4]
                print("       WGAN critic gap E[D(real)] - E[D(fake)] at steps", {int(hist[i, 0]): round(float(gap[i]), 3) for i in (0, 10, 25, 50, 100, 150)},
                      f"; correlation of the gap with sample quality over training: {np.corrcoef(gap[5:], hist[5:, 2])[0, 1]:+.2f}")
    snaps = tuple(sorted(set([0, 50, 100, 200, 300, 500, 800, 1200, 1700, 2300, steps])))
    Gw, Dw, hw, sw = train("wgan", steps, 0, snaps=snaps)
    Gg, Dg, hg, sg = train("gan", steps, 0, snaps=snaps)
    with Animator(path, fps=1.5, hold=3.5) as anim:
        fig, ((a1, a2), (a3, a4)) = new_fig(2, 2, figsize=(11, 7.4), title="Original GAN vs WGAN on the same 8-cluster data")
        for s in snaps:
            for a in (a1, a2, a3, a4):
                a.clear()
            for ax, kind, sv, hh in ((a1, "gan", sg, hg), (a2, "wgan", sw, hw)):
                fake, dmap = sv[s]
                ax.scatter(CENTERS[:, 0], CENTERS[:, 1], s=240, facecolor="none", edgecolor=INK2, lw=1)
                ax.scatter(fake[:, 0], fake[:, 1], s=6, color=ORANGE if kind == "gan" else BLUE, alpha=0.7)
                row = hh[hh[:, 0] == s][0]
                ax.set(xlim=(-3.2, 3.2), ylim=(-3.2, 3.2), aspect="equal", title=f"{'GAN (BCE loss)' if kind == 'gan' else 'WGAN (clipped critic)'}: {int(row[1])}/8 clusters")
            a3.plot(hg[hg[:, 0] <= s][:, 0], hg[hg[:, 0] <= s][:, 1], color=ORANGE, lw=2, label="GAN")
            a3.plot(hw[hw[:, 0] <= s][:, 0], hw[hw[:, 0] <= s][:, 1], color=BLUE, lw=2, label="WGAN")
            a3.set(xlim=(0, steps), ylim=(0, 8.4), title="Clusters covered (of 8)", xlabel="step"); a3.legend(loc="lower right")
            a4.plot(hg[hg[:, 0] <= s][:, 0], hg[hg[:, 0] <= s][:, 2], color=ORANGE, lw=2)
            a4.plot(hw[hw[:, 0] <= s][:, 0], hw[hw[:, 0] <= s][:, 2], color=BLUE, lw=2)
            a4.set(xlim=(0, steps), ylim=(0, 1.02), title="Samples within 0.3 of a real cluster", xlabel="step")
            caption(fig, f"step {s}")
            anim.grab(fig)
    return out


def main():
    print("Lesson 33: GAN")
    xr = real_batch(20000, rng_(3))
    print(f"  sanity: real data covers {coverage(xr)[0]} of 8 clusters with {coverage(xr)[1]:.2f} of samples within 0.3 of a centre")
    print("  untrained generator:", coverage(make_nets(0, False)[0].forward(rng_(9).normal(size=(1000, ZD)))))
    hist = gan_gif(here(__file__, "gan_training.gif"))
    print(f"  GAN run (seed 0): final clusters {int(hist[-1, 1])}/8, quality {hist[-1, 2]:.2f}; D(real)={hist[-1, 3]:.2f}, D(fake)={hist[-1, 4]:.2f}")
    print("  GAN vs WGAN over three seeds each:")
    compare_gif(here(__file__, "gan_vs_wgan.gif"))


if __name__ == "__main__":
    main()
