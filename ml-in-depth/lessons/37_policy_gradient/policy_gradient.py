"""
Lesson 37 - Policy gradient (REINFORCE, baselines) from scratch

Story: balance a pole on a motorised cart (the classic CartPole benchmark, common/envs.py).  The agent observes
(cart position, cart velocity, pole angle, pole angular velocity) and pushes left or right.  +1 per step while upright (max 200).

Implemented (NumPy + common/nn.py, manual backprop):
  * a softmax policy network pi(a|s)
  * REINFORCE:   grad J = E[ sum_t grad log pi(a_t|s_t) * (G_t - b(s_t)) ]
  * three variants: raw returns (no baseline), batch-mean baseline, learned state-value baseline (critic)
  * an exact check of the score-function estimator on a 3-armed softmax bandit (and its variance with/without baseline)

Animated:
  pg_cartpole.gif  four snapshots of the policy (untrained ... trained) balancing the pole side by side
  pg_curves.gif    learning curves of the three variants + the policy's push-right probability map as it is learned

Run:  python policy_gradient.py
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from common.nn import MLP, Adam
from common.envs import CartPole, X_LIM, TH_LIM

GAMMA, MAXT, NENV = 0.99, 200, 16


# --------------------------------------------------------------------------- policy
def probs(net, S):
    z = net.forward(S); z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / e.sum(1, keepdims=True)


def collect(net, env, rng):
    """One episode in each of NENV envs (finished ones are masked). Returns arrays of shape (T, n, ...)."""
    S = env.reset_all(); alive = np.ones(NENV, bool)
    Ss, As, Rs, Ms = [], [], [], []
    for t in range(MAXT):
        p = probs(net, S)
        a = (rng.random(NENV) < p[:, 1]).astype(int)
        nxt, r, d, tr, S2 = env.step(a)
        Ss.append(S.copy()); As.append(a); Rs.append(r * alive); Ms.append(alive.copy())
        alive = alive & ~(d | tr)
        S = S2
        if not alive.any():
            break
    return np.array(Ss), np.array(As), np.array(Rs), np.array(Ms)


def returns_to_go(R, M):
    G = np.zeros_like(R); run = np.zeros(R.shape[1])
    for t in range(len(R) - 1, -1, -1):
        run = R[t] + GAMMA * run
        G[t] = run
    return G * M


def train(variant, seed, n_updates=150, lr=1e-3, snap_every=5, quiet=True):
    """variant: 'raw' | 'mean' | 'critic'"""
    rng = np.random.default_rng(seed)
    env = CartPole(NENV, MAXT, seed + 1000)
    pol = MLP([4, 32, 32, 2], act="tanh", seed=seed, init_scale=0.5)
    opt = Adam(pol.params_and_grads(), lr=lr)
    if variant == "critic":
        val = MLP([4, 32, 32, 1], act="tanh", seed=seed + 7)
        vopt = Adam(val.params_and_grads(), lr=lr)
    lens, snaps = [], []
    for u in range(n_updates + 1):
        S, A, R, M = collect(pol, env, rng)
        ep_len = M.sum(0)                                  # steps survived by each env
        lens.append(ep_len.mean())
        if u % snap_every == 0:
            snaps.append((u, pol.clone(), ep_len.mean()))
        if u == n_updates:
            break
        G = returns_to_go(R, M)
        Sf, Af, Gf, Mf = S.reshape(-1, 4), A.reshape(-1), G.reshape(-1), M.reshape(-1)
        idx = np.flatnonzero(Mf)
        Sf, Af, Gf = Sf[idx], Af[idx], Gf[idx]
        if variant == "raw":
            adv = Gf / 50.0                                 # only a constant rescaling, still no baseline
        elif variant == "mean":
            adv = (Gf - Gf.mean()) / (Gf.std() + 1e-8)
        else:
            v = val.forward(Sf)[:, 0]
            adv = Gf - v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            val.zero_grad(); val.backward(((v - Gf) / 50.0 ** 2 * 2 / len(Gf))[:, None]); vopt.step()
        p = probs(pol, Sf)
        d = p.copy(); d[np.arange(len(Af)), Af] -= 1.0       # d(-log pi)/dlogits = p - onehot
        d *= adv[:, None] / len(Af)
        pol.zero_grad(); pol.backward(d); opt.step()
    return np.array(lens), snaps


# ------------------------------------------------------------------ exact estimator check
def bandit_check(seed=0, N=200000):
    """3-armed softmax bandit with rewards 1,2,3 (+noise): compare the exact gradient with the score-function estimate."""
    rng = np.random.default_rng(seed)
    theta = np.array([0.2, -0.1, 0.3]); mu = np.array([1.0, 2.0, 3.0])
    p = np.exp(theta) / np.exp(theta).sum()
    exact = p * (mu - p @ mu)                              # dJ/dtheta_k = pi_k (mu_k - J)
    a = rng.choice(3, size=N, p=p); r = mu[a] + rng.normal(0, 1.0, N)
    score = np.eye(3)[a] - p                               # grad_theta log pi(a)
    out = {}
    for name, b in (("no baseline", 0.0), ("baseline = mean reward", r.mean())):
        g = score * (r - b)[:, None]
        out[name] = (g.mean(0), g.std(0) * 1.0)
    return exact, out


def flat_grad(pol, Sf, Af, adv):
    p = probs(pol, Sf)
    d = p.copy(); d[np.arange(len(Af)), Af] -= 1.0
    d *= adv[:, None] / len(Af)
    pol.zero_grad(); pol.backward(d)
    return np.concatenate([g.ravel() for _, g in pol.params_and_grads()])


def grad_variance(pol, n_batches=60, seed=0):
    """At a FIXED policy, how noisy is the gradient estimate from one batch of 16 episodes, per variant?  (raw scale, no normalisation)"""
    rng = np.random.default_rng(seed); env = CartPole(NENV, MAXT, seed + 5)
    batches = []
    for _ in range(n_batches):
        S, A, R, M = collect(pol, env, rng); G = returns_to_go(R, M)
        idx = np.flatnonzero(M.reshape(-1))
        batches.append((S.reshape(-1, 4)[idx], A.reshape(-1)[idx], G.reshape(-1)[idx]))
    # critic fitted on the first half of the batches, used on the second half (so it is not fitted on the same samples)
    val = MLP([4, 32, 32, 1], act="tanh", seed=3); vopt = Adam(val.params_and_grads(), lr=3e-3)
    Sv = np.concatenate([b[0] for b in batches[:30]]); Gv = np.concatenate([b[2] for b in batches[:30]])
    for _ in range(400):
        v = val.forward(Sv)[:, 0]; val.zero_grad(); val.backward(((v - Gv) * 2 / len(Gv) / 50.0 ** 2)[:, None]); vopt.step()
    out = {}
    for name in ("raw", "mean", "critic"):
        gs = []
        for Sf, Af, Gf in batches[30:]:
            adv = Gf if name == "raw" else (Gf - Gf.mean() if name == "mean" else Gf - val.forward(Sf)[:, 0])
            gs.append(flat_grad(pol, Sf, Af, adv / 50.0))
        gs = np.array(gs)
        out[name] = (np.linalg.norm(gs.mean(0)), np.sqrt(gs.var(0).sum()))
    return out


# ------------------------------------------------------------------ drawing
def draw_cart(ax, s, label, alive=True, step=0):
    x, th = s[0], s[2]
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-0.15, 1.15); ax.set_aspect("equal"); ax.axis("off")
    ax.plot([-X_LIM, X_LIM], [0, 0], color=MUTED, lw=1)
    for xx in (-X_LIM, X_LIM):
        ax.plot([xx, xx], [-0.05, 0.12], color=RED, lw=2)
    col = BLUE if alive else RED
    ax.add_patch(plt.Rectangle((x - 0.2, 0.0), 0.4, 0.15, color=col))
    tip = (x + 0.9 * np.sin(th) * 1.0, 0.15 + 0.9 * np.cos(th))
    ax.plot([x, tip[0]], [0.15, tip[1]], color=ORANGE if alive else RED, lw=4, solid_capstyle="round")
    ax.plot([tip[0]], [tip[1]], "o", color=ORANGE if alive else RED, ms=8)
    ax.set_title(label, fontsize=10)


def greedy_ep(net, seed):
    env = CartPole(1, MAXT, seed); rng = np.random.default_rng(seed)
    S = env.s.copy(); states = [S[0].copy()]
    for _ in range(MAXT):
        p = probs(net, S)
        a = (rng.random(1) < p[:, 1]).astype(int)
        nxt, r, d, tr, S = env.step(a)
        states.append(nxt[0].copy())
        if d[0] or tr[0]:
            break
    return np.array(states)


def cartpole_gif(path, snaps):
    picks = [0, 4, 10, len(snaps) - 1]
    names, eps = [], []
    for i in picks:
        u, net, m = snaps[i]
        eps.append(greedy_ep(net, 3)); names.append(f"update {u}")
    T = 100
    with Animator(path, fps=14, hold=2.0) as anim:
        fig, axes = new_fig(2, 2, figsize=(11, 5.4), title="REINFORCE learns to balance the pole (one sampled episode per snapshot)")
        axes = np.ravel(axes)
        for t in range(0, T + 1, 2):
            for ax, ep, nm in zip(axes, eps, names):
                ax.clear()
                k = min(t, len(ep) - 1); alive = t < len(ep) - 1 or len(ep) - 1 >= MAXT
                draw_cart(ax, ep[k], f"{nm}: " + (f"step {t}" if alive else f"fell at step {len(ep) - 1}"), alive)
            caption(fig, f"step {t}: untrained policy drops the pole within ~{len(eps[0]) - 1} steps; the trained one keeps it up for the whole window")
            anim.grab(fig)
    return [len(e) - 1 for e in eps]


def curves_gif(path, results, snaps):
    names = {"raw": "raw returns (no baseline)", "mean": "batch-mean baseline", "critic": "learned value baseline"}
    cols = {"raw": MAGENTA, "mean": BLUE, "critic": ORANGE}
    n_up = len(next(iter(results.values()))[0])
    gx, gy = np.meshgrid(np.linspace(-TH_LIM, TH_LIM, 40), np.linspace(-2.0, 2.0, 40))
    with Animator(path, fps=2, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.6), title="Three policy-gradient variants on CartPole", gridspec_kw={"width_ratios": [1.3, 1]})
        for (u, net, m) in snaps:
            a1.clear(); a2.clear()
            for k, runs in results.items():
                A = np.array(runs)[:, :u + 1]
                sm = np.array([np.convolve(r, np.ones(5) / 5, mode="same") for r in A]) if u >= 5 else A
                mean = A.mean(0); lo, hi = A.min(0), A.max(0)
                a1.fill_between(np.arange(u + 1), lo, hi, color=cols[k], alpha=0.15, lw=0)
                a1.plot(np.arange(u + 1), mean, color=cols[k], lw=2.2, label=names[k])
            a1.set(xlim=(0, n_up - 1), ylim=(0, 205), xlabel="update (16 episodes each)", ylabel="mean steps balanced",
                   title="Learning curves (mean and range of 3 seeds)"); a1.legend(loc="upper left", fontsize=8)
            Sg = np.stack([np.zeros(gx.size), np.zeros(gx.size), gx.ravel(), gy.ravel()], 1)
            pr = probs(net, Sg)[:, 1].reshape(gx.shape)
            im = a2.pcolormesh(gx, gy, pr, cmap="RdBu_r", vmin=0, vmax=1, shading="auto")
            a2.axvline(0, color=INK, lw=0.6); a2.axhline(0, color=INK, lw=0.6)
            a2.set(xlabel="pole angle (rad)", ylabel="pole angular velocity", title=f"P(push right), update {u}")
            caption(fig, f"update {u}: policy map (right) turns from uniform grey into 'push toward the side the pole is falling'")
            anim.grab(fig)


def main():
    print("Lesson 37: policy gradient")
    exact, out = bandit_check()
    print(f"  bandit: exact dJ/dtheta = {np.round(exact, 4)}")
    for k, (g, sd) in out.items():
        print(f"    score-function estimate, {k:24s}: {np.round(g, 4)}   per-sample sd {np.round(sd, 2)}")
    t0 = time.time()
    results, snaps0 = {}, None
    for v in ("raw", "mean", "critic"):
        runs = []
        for seed in range(3):
            lens, sn = train(v, seed)
            runs.append(lens)
            if v == "critic" and seed == 0:
                snaps0 = sn
        results[v] = runs
        r = np.array(runs)
        print(f"  {v:7s}: mean steps balanced  updates 0-9 {r[:, :10].mean():6.1f} | 40-49 {r[:, 40:50].mean():6.1f} | last 10 {r[:, -10:].mean():6.1f}   (per seed last 10: {np.round(r[:, -10:].mean(1), 1)})   {time.time() - t0:.0f}s")
    u30, net30, _ = snaps0[6]
    gv = grad_variance(net30)
    print(f"  gradient noise of a single 16-episode batch at the policy after {u30} updates (same samples, three baselines):")
    for k, (m, sd) in gv.items():
        print(f"    {k:7s}: |mean gradient| {m:.4f}   total sd of one estimate {sd:.4f}   noise/signal {sd / m:.2f}")
    lens_final = cartpole_gif(here(__file__, "pg_cartpole.gif"), snaps0)
    print(f"  snapshot episodes (sampled policy) survived steps: {lens_final}")
    curves_gif(here(__file__, "pg_curves.gif"), results, snaps0)


if __name__ == "__main__":
    main()
