"""
Lesson 34 - Denoising diffusion models (DDPM / DDIM), from scratch

Problem story: a lab wants to generate new, realistic "two-regime" machine-state readings (two crescent-shaped families of
operating points, a 2-D stand-in for any complicated data).  A diffusion model learns by watching data being slowly destroyed by
noise, and learns to reverse that destruction one small step at a time.  Sampling = start from pure noise and keep denoising.

Implemented (NumPy + the course's tiny MLP kit in common/nn.py, manual back-propagation):
  * forward process  q(x_t | x_0) = N( sqrt(abar_t) x_0, (1 - abar_t) I )   (closed form, verified against step-by-step noising)
  * noise-prediction network eps_theta(x_t, t)  trained with the simple loss  E || eps - eps_theta(x_t, t) ||^2
  * ancestral DDPM sampling and deterministic DDIM sampling with few steps
  * the link to the score:  eps_theta ~ -sqrt(1 - abar_t) * grad log p_t(x_t);  an analytic Gaussian test of the sampler

Animated:
  diffusion_forward.gif  data dissolving into noise; the signal-to-noise schedule
  diffusion_reverse.gif  noise condensing into the crescents; the learned arrows (score field) and particle paths
  diffusion_samplers.gif the same trained network sampled with 60 (DDPM), 20, 10 and 5 (DDIM) steps

Run:  python diffusion.py        (about 1 minute)
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from common.nn import MLP, Adam

T = 60
BETAS = np.linspace(1e-3, 0.14, T)                 # noise added at each step
ALPHAS = 1 - BETAS
ABAR = np.cumprod(ALPHAS)                           # abar_t = prod alpha_s : fraction of signal variance left at step t (t = 1..T stored at index t-1)


def data(n, seed=0):
    X, _ = make_moons(n, noise=0.06, seed=seed)
    return (X - np.array([0.5, 0.25])) * 1.6        # centre and scale to roughly unit variance


def time_features(t):
    """Sinusoidal embedding of the (normalised) step index, so one network can serve all noise levels."""
    u = (np.asarray(t, float) / T)[:, None]
    f = 2.0 ** np.arange(4)[None] * np.pi
    return np.concatenate([u, np.sin(u * f), np.cos(u * f)], 1)


class Denoiser:
    def __init__(self, seed=0, hidden=128):
        self.net = MLP([2 + 9, hidden, hidden, hidden, 2], act="silu", seed=seed)
        self.opt = Adam(self.net.params_and_grads(), lr=2e-3)

    def eps(self, x, t):
        return self.net.forward(np.concatenate([x, time_features(np.full(len(x), t) if np.isscalar(t) else t)], 1))

    def train_step(self, x0, r):
        B = len(x0)
        t = r.integers(1, T + 1, B)
        noise = r.normal(size=x0.shape)
        ab = ABAR[t - 1][:, None]
        xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * noise
        pred = self.eps(xt, t)
        loss = np.mean((pred - noise) ** 2)
        self.net.zero_grad()
        self.net.backward(2 * (pred - noise) / pred.size)
        self.opt.step()
        return loss


def sample(model, n, r, steps="ddpm", record=False, eps_fn=None):
    """Start from N(0, I) and denoise.  'ddpm' = all T ancestral steps; an integer k = DDIM with k evenly spaced steps."""
    eps_fn = eps_fn or (lambda x, t: model.eps(x, t))
    x = r.normal(size=(n, 2)); traj = [x.copy()]
    if steps == "ddpm":
        for t in range(T, 0, -1):
            e = eps_fn(x, t)
            mean = (x - BETAS[t - 1] / np.sqrt(1 - ABAR[t - 1]) * e) / np.sqrt(ALPHAS[t - 1])
            if t > 1:
                var = BETAS[t - 1] * (1 - ABAR[t - 2]) / (1 - ABAR[t - 1])      # posterior variance
                x = mean + np.sqrt(var) * r.normal(size=x.shape)
            else:
                x = mean
            traj.append(x.copy())
    else:
        ts = np.unique(np.round(np.linspace(T, 1, steps)).astype(int))[::-1]
        for i, t in enumerate(ts):
            e = eps_fn(x, t)
            x0 = (x - np.sqrt(1 - ABAR[t - 1]) * e) / np.sqrt(ABAR[t - 1])      # predicted clean sample
            tn = ts[i + 1] if i + 1 < len(ts) else 0
            abn = ABAR[tn - 1] if tn > 0 else 1.0
            x = np.sqrt(abn) * x0 + np.sqrt(1 - abn) * e                         # deterministic DDIM update (eta = 0)
            traj.append(x.copy())
    return (x, traj) if record else x


def quality(samples, ref):
    """mean distance from a sample to its nearest real point; and share of samples on the upper crescent."""
    d = np.sqrt(((samples[:, None] - ref[None]) ** 2).sum(2)).min(1)
    return float(d.mean()), float(d.max())


def moon_balance(x):
    """The upper moon sits above the line y = -0.4 (x shifted); crude split used only for reporting."""
    return float(np.mean(x[:, 1] > 0.5 * (x[:, 0] - 0.1) * -0.0 + (-0.1)))


def train(steps=8000, seed=0, B=256, snaps=()):
    r = rng_(seed + 4)
    X = data(4000, 1)
    m = Denoiser(seed)
    losses = []
    for s in range(steps + 1):
        if s in snaps:
            pass
        l = m.train_step(X[r.integers(0, len(X), B)], r)
        losses.append(l)
    return m, np.array(losses)


# ------------------------------------------------------------------- animations
def forward_gif(path, X):
    ts = [0, 1, 3, 6, 10, 15, 25, 40, 60]
    r = rng_(0); noise = r.normal(size=X.shape)
    with Animator(path, fps=1.5, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(12, 4.8), title="Forward process: data dissolves into Gaussian noise", gridspec_kw={"width_ratios": [1.2, 1]})
        for t in ts:
            a1.clear(); a2.clear()
            ab = 1.0 if t == 0 else ABAR[t - 1]
            xt = np.sqrt(ab) * X + np.sqrt(1 - ab) * noise
            a1.scatter(xt[:, 0], xt[:, 1], s=6, color=BLUE if t < 60 else MUTED, alpha=0.6)
            a1.set(xlim=(-3.4, 3.4), ylim=(-3.4, 3.4), aspect="equal", title=f"x at step t = {t}", xlabel="feature 1", ylabel="feature 2")
            a2.plot(np.r_[0, np.arange(1, T + 1)], np.r_[1, ABAR], color=INK, lw=2, label="signal fraction ᾱₜ")
            a2.plot(np.r_[0, np.arange(1, T + 1)], np.r_[0, 1 - ABAR], color=ORANGE, lw=2, label="noise fraction 1 − ᾱₜ")
            a2.scatter([t], [ab], s=80, color=INK, zorder=5); a2.scatter([t], [1 - ab], s=80, color=ORANGE, zorder=5)
            a2.set(xlim=(0, T), ylim=(-0.02, 1.02), title="Noise schedule", xlabel="step t"); a2.legend(loc="center right")
            caption(fig, f"t = {t}: x_t = {np.sqrt(ab):.2f}·x₀ + {np.sqrt(1 - ab):.2f}·noise  " + ("(the clean data)" if t == 0 else "(pure noise: the structure is gone)" if t == 60 else ""))
            anim.grab(fig)


def reverse_gif(path, m, ref):
    r = rng_(11)
    n = 1500
    x, traj = sample(m, n, r, "ddpm", record=True)
    q = quality(x, ref)
    print(f"  DDPM (60 steps): mean distance of generated points to the nearest real point {q[0]:.3f} (real-vs-real {quality(data(1500, 7), ref)[0]:.3f}); max {q[1]:.2f}")
    show = [0, 5, 10, 20, 30, 40, 50, 55, 58, 60]
    gx = np.linspace(-3, 3, 13); GX, GY = np.meshgrid(gx, gx); grid = np.c_[GX.ravel(), GY.ravel()]
    paths = np.array(traj)[:, :12]                                  # 12 particle paths
    with Animator(path, fps=1.5, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(12.5, 4.9), title="Reverse process: noise condenses into the data", gridspec_kw={"width_ratios": [1.2, 1]})
        for k in show:
            a1.clear(); a2.clear()
            xt = traj[k]
            t = T - k                                               # current noise level index (T at the start)
            a1.scatter(xt[:, 0], xt[:, 1], s=6, color=BLUE, alpha=0.5)
            if t >= 1:
                e = m.eps(grid, t)
                score = -e / np.sqrt(1 - ABAR[t - 1])
                a1.quiver(grid[:, 0], grid[:, 1], score[:, 0], score[:, 1], color=ORANGE, alpha=0.75, angles="xy", width=0.004)
            a1.set(xlim=(-3.4, 3.4), ylim=(-3.4, 3.4), aspect="equal", title=f"{k} of 60 denoising steps (noise level t = {t})", xlabel="feature 1", ylabel="feature 2")
            a2.scatter(ref[:600, 0], ref[:600, 1], s=4, color=MUTED, alpha=0.5)
            for j in range(12):
                a2.plot(paths[: k + 1, j, 0], paths[: k + 1, j, 1], color=CAT[j % 8], lw=1, alpha=0.8)
                a2.scatter(paths[k, j, 0], paths[k, j, 1], s=22, color=CAT[j % 8], zorder=4)
            a2.set(xlim=(-3.4, 3.4), ylim=(-3.4, 3.4), aspect="equal", title="Paths of 12 individual samples (grey = real data)", xlabel="feature 1")
            caption(fig, f"step {k}: orange arrows = the learned score field (direction toward higher data density at this noise level). "
                         + ("Pure noise." if k == 0 else "Large-scale structure appears first, details last." if k < 50 else "Final polishing."))
            anim.grab(fig)
    return x


def samplers_gif(path, m, ref):
    configs = [("DDPM, 60 steps", "ddpm"), ("DDIM, 20 steps", 20), ("DDIM, 10 steps", 10), ("DDIM, 5 steps", 5), ("DDIM, 2 steps", 2)]
    with Animator(path, fps=1.0, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(11.5, 4.8), title="Same trained network, different samplers", gridspec_kw={"width_ratios": [1.1, 1]})
        res = []
        for name, st in configs:
            x = sample(m, 1500, rng_(21), st)
            q = quality(x, ref); res.append((name, q[0]))
            print(f"     {name:16s}: mean distance to nearest real point {q[0]:.3f}")
        for i, (name, st) in enumerate(configs):
            a1.clear(); a2.clear()
            x = sample(m, 1500, rng_(21), st)
            a1.scatter(ref[:800, 0], ref[:800, 1], s=4, color=MUTED, alpha=0.4)
            a1.scatter(x[:, 0], x[:, 1], s=6, color=ORANGE, alpha=0.6)
            a1.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", title=name, xlabel="feature 1", ylabel="feature 2")
            a2.bar([c[0].replace(", ", "\n") for c in res], [c[1] for c in res], color=[ORANGE if j == i else MUTED for j in range(len(res))])
            a2.set(title="Mean distance of samples to the nearest real point", ylabel="distance (lower is better)"); a2.grid(axis="x", visible=False)
            caption(fig, f"{name}: {'stochastic sampler, every step adds fresh noise' if st == 'ddpm' else 'deterministic DDIM: skips steps, a few steps already give a recognisable shape' if st >= 5 else 'too few steps: blurry bridge between the crescents'}")
            anim.grab(fig)


def main():
    print("Lesson 34: diffusion")
    # forward process closed form vs step by step
    r = rng_(0); x0 = np.full(200000, 1.0)
    x = x0.copy()
    for t in range(1, 21):
        x = np.sqrt(1 - BETAS[t - 1]) * x + np.sqrt(BETAS[t - 1]) * r.normal(size=x.shape)
    print(f"  forward process at t=20: step-by-step mean {x.mean():.4f}, var {x.var():.4f}  vs closed form mean {np.sqrt(ABAR[19]):.4f}, var {1 - ABAR[19]:.4f}; abar_T = {ABAR[-1]:.4f}")
    # analytic test of the sampler: data N(0, s^2) has exact eps*(x,t) = sqrt(1-abar) x / (abar s^2 + 1 - abar)
    s2 = 0.5 ** 2
    exact = lambda xx, t: np.sqrt(1 - ABAR[t - 1]) * xx / (ABAR[t - 1] * s2 + 1 - ABAR[t - 1])
    xs = sample(None, 100000, rng_(1), "ddpm", eps_fn=exact); xd = sample(None, 100000, rng_(2), 10, eps_fn=exact)
    print(f"  sampler test with the exact denoiser for N(0, 0.5^2): DDPM std {xs.std():.3f}, DDIM(10 steps) std {xd.std():.3f} (target 0.500)")
    X = data(1500, 3)
    ref = data(5000, 5)
    m, losses = train(8000)
    print(f"  training loss (noise-prediction MSE): first 100 steps {losses[:100].mean():.3f}, last 500 steps {losses[-500:].mean():.3f}")
    # loss by noise level
    rr = rng_(5); tt = np.arange(1, T + 1)
    per = []
    Xb = data(2000, 8)
    for t in (1, 10, 30, 60):
        nz = rr.normal(size=Xb.shape); xt = np.sqrt(ABAR[t - 1]) * Xb + np.sqrt(1 - ABAR[t - 1]) * nz
        per.append((t, float(np.mean((m.eps(xt, t) - nz) ** 2))))
    print("  held-out loss at noise levels t = 1, 10, 30, 60:", [(t, round(v, 3)) for t, v in per])
    forward_gif(here(__file__, "diffusion_forward.gif"), X)
    reverse_gif(here(__file__, "diffusion_reverse.gif"), m, ref)
    print("  samplers:")
    samplers_gif(here(__file__, "diffusion_samplers.gif"), m, ref)


if __name__ == "__main__":
    main()
