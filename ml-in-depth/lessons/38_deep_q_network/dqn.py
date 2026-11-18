"""
Lesson 38 - Deep Q-Network (DQN) from scratch

Story: the CartPole of lesson 37 again, but now the agent learns a *value* network Q(s, a) (expected discounted return of
pushing left/right now and acting well afterwards) and acts greedily on it.  A neural network replaces the table of lesson 36.

Implemented (NumPy + common/nn.py, manual backprop):
  * Q-network  [4 -> 64 -> 64 -> 2], epsilon-greedy exploration with a linear decay
  * experience replay buffer (random mini-batches break the correlation between consecutive steps)
  * target network (a slowly updated copy used to build the bootstrap target, which stops the target chasing itself)
  * Huber (clipped-error) loss
  * ablations: no target network, tiny replay buffer ("online"), neither

Animated:
  dqn_training.gif   policy map argmax Q + the states the replay buffer holds, evaluation return, estimated Q-values
  dqn_ablation.gif   the four variants side by side: learning curves and the Q-value estimates

Run:  python dqn.py
"""
import sys, pathlib, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from common.nn import MLP, Adam, grad_check
from common.envs import CartPole, X_LIM, TH_LIM

GAMMA, MAXT, NENV = 0.99, 200, 4


class Replay:
    def __init__(self, cap):
        self.cap, self.n, self.i = cap, 0, 0
        self.S = np.zeros((cap, 4)); self.A = np.zeros(cap, int); self.R = np.zeros(cap)
        self.S2 = np.zeros((cap, 4)); self.D = np.zeros(cap)

    def add(self, s, a, r, s2, d):
        for k in range(len(a)):
            self.S[self.i], self.A[self.i], self.R[self.i], self.S2[self.i], self.D[self.i] = s[k], a[k], r[k], s2[k], d[k]
            self.i = (self.i + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, B, rng):
        j = rng.integers(0, self.n, B)
        return self.S[j], self.A[j], self.R[j], self.S2[j], self.D[j]


def evaluate(q, seed, n=10):
    """Greedy episodes in n environments; returns mean steps balanced."""
    env = CartPole(n, MAXT, seed); S = env.reset_all(); alive = np.ones(n, bool); steps = np.zeros(n)
    for _ in range(MAXT):
        a = q.forward(S).argmax(1)
        _, _, d, tr, S = env.step(a)
        steps += alive; alive &= ~(d | tr)
        if not alive.any():
            break
    return steps.mean()


PROBE = np.random.default_rng(99).uniform([-0.5, -0.5, -0.1, -0.5], [0.5, 0.5, 0.1, 0.5], (200, 4))   # fixed upright-ish states


def train(variant, seed, iters=3000, lr=1e-3, batch=64, eval_every=100, snap_every=300):
    """variant: 'dqn' | 'no_target' | 'tiny_replay' | 'neither'"""
    use_target = variant in ("dqn", "tiny_replay")
    cap = 20000 if variant in ("dqn", "no_target") else 64
    rng = np.random.default_rng(seed)
    env = CartPole(NENV, MAXT, seed + 11)
    q = MLP([4, 64, 64, 2], act="relu", seed=seed, init_scale=0.5)
    tq = q.clone()
    opt = Adam(q.params_and_grads(), lr=lr, clip=10.0)
    buf = Replay(cap)
    S = env.reset_all()
    log = {"it": [], "eval": [], "qmean": [], "snaps": []}
    for it in range(iters + 1):
        if it % eval_every == 0:
            log["it"].append(it); log["eval"].append(evaluate(q, 777 + it)); log["qmean"].append(q.forward(PROBE).max(1).mean())
        if it % snap_every == 0:
            log["snaps"].append((it, q.clone(), buf.S[:buf.n].copy()))
        if it == iters:
            break
        eps = max(0.05, 1.0 - it / 1000.0)
        a = q.forward(S).argmax(1)
        rand = rng.random(NENV) < eps
        a = np.where(rand, rng.integers(0, 2, NENV), a)
        nxt, r, d, tr, S2 = env.step(a)
        buf.add(S, a, r, nxt, d.astype(float))            # terminated -> no bootstrap; truncated (time limit) -> bootstrap
        S = S2
        if buf.n >= 500 or (cap == 64 and buf.n >= 64):
            for _ in range(2):
                s, ac, rw, s2, dn = buf.sample(batch, rng)
                tgt_net = tq if use_target else q
                y = rw + GAMMA * (1 - dn) * tgt_net.forward(s2).max(1)
                qs = q.forward(s)
                td = qs[np.arange(batch), ac] - y
                dout = np.zeros_like(qs); dout[np.arange(batch), ac] = np.clip(td, -1, 1) / batch   # Huber gradient
                q.zero_grad(); q.backward(dout); opt.step()
        if use_target and it % 100 == 0:
            tq.copy_from(q)
    return log


# --------------------------------------------------------------------------- drawing
def policy_map(q, ax, buffer_states=None):
    gx, gy = np.meshgrid(np.linspace(-TH_LIM, TH_LIM, 40), np.linspace(-2.0, 2.0, 40))
    Sg = np.stack([np.zeros(gx.size), np.zeros(gx.size), gx.ravel(), gy.ravel()], 1)
    Q = q.forward(Sg)
    adv = (Q[:, 1] - Q[:, 0]).reshape(gx.shape)
    lim = max(0.5, np.percentile(np.abs(adv), 98))
    ax.pcolormesh(gx, gy, adv, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    if buffer_states is not None and len(buffer_states):
        sub = buffer_states[np.random.default_rng(0).choice(len(buffer_states), min(600, len(buffer_states)), replace=False)]
        ax.scatter(sub[:, 2], sub[:, 3], s=3, color=INK, alpha=0.35)
    ax.set(xlim=(-TH_LIM, TH_LIM), ylim=(-2, 2), xlabel="pole angle (rad)", ylabel="pole angular velocity")


def training_gif(path, log):
    its, ev, qm = log["it"], np.array(log["eval"]), np.array(log["qmean"])
    with Animator(path, fps=1.6, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(14, 4.4), title="DQN on CartPole", gridspec_kw={"width_ratios": [1, 1, 1]})
        for it, q, bs in log["snaps"]:
            for a in (a1, a2, a3):
                a.clear()
            policy_map(q, a1, bs); a1.set_title(f"Q(push right) - Q(push left), iter {it}")
            k = its.index(it)
            a2.plot(its[:k + 1], ev[:k + 1], color=BLUE, lw=2.2, marker="o", ms=3)
            a2.set(xlim=(0, its[-1]), ylim=(0, 210), xlabel="training iteration", ylabel="steps balanced (greedy)", title="Evaluation return")
            a3.plot(its[:k + 1], qm[:k + 1], color=ORANGE, lw=2.2, marker="o", ms=3)
            a3.axhline(86.6, color=MUTED, ls="--", lw=1); a3.text(its[-1] * 0.02, 88, "value of balancing 200 steps (86.6)", fontsize=8, color=MUTED)
            a3.set(xlim=(0, its[-1]), ylim=(0, max(110, qm.max() * 1.05)), xlabel="training iteration", ylabel="mean max Q on probe states", title="What the network believes")
            caption(fig, f"iter {it}: red = push right, blue = push left (map at cart centre); dots = states in the replay buffer; eval {ev[k]:.0f} steps")
            anim.grab(fig)


def ablation_gif(path, results):
    names = {"dqn": "replay + target net (DQN)", "no_target": "replay, no target net", "tiny_replay": "tiny replay, target net", "neither": "no replay, no target net"}
    cols = {"dqn": ORANGE, "no_target": BLUE, "tiny_replay": GREEN, "neither": MAGENTA}
    its = np.array(next(iter(results.values()))[0]["it"]); n = len(its)
    with Animator(path, fps=2.5, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.6), title="Which ingredients does DQN need? (3 seeds each)")
        for k in range(1, n):
            a1.clear(); a2.clear()
            for v, runs in results.items():
                E = np.array([r["eval"] for r in runs])[:, :k + 1]; Q = np.array([r["qmean"] for r in runs])[:, :k + 1]
                a1.fill_between(its[:k + 1], E.min(0), E.max(0), color=cols[v], alpha=0.15, lw=0); a1.plot(its[:k + 1], E.mean(0), color=cols[v], lw=2.2, label=names[v])
                a2.fill_between(its[:k + 1], Q.min(0), Q.max(0), color=cols[v], alpha=0.15, lw=0); a2.plot(its[:k + 1], Q.mean(0), color=cols[v], lw=2.2, label=names[v])
            a2.axhline(86.6, color=MUTED, ls="--", lw=1)
            a1.set(xlim=(0, its[-1]), ylim=(0, 210), xlabel="training iteration", ylabel="steps balanced (greedy)", title="Evaluation return"); a1.legend(loc="upper left", fontsize=8)
            a2.set_yscale("symlog", linthresh=30); a2.set(xlim=(0, its[-1]), ylim=(0, 1e6), xlabel="training iteration", ylabel="mean max Q on probe states (log scale)", title="Estimated value (dashed: 86.6 = best possible)")
            caption(fig, f"iteration {its[k]}: mean of 3 seeds (band = min/max)")
            anim.grab(fig)


def main():
    print("Lesson 38: deep Q-network")
    net = MLP([4, 16, 16, 2], act="relu", seed=1)
    print(f"  gradient check of the Q-network: max relative error {grad_check(net, np.random.default_rng(2).normal(size=(6, 4))):.1e}")
    rr = np.random.default_rng(0); env = CartPole(200, MAXT, 5); S = env.reset_all(); alive = np.ones(200, bool); st = np.zeros(200)
    for _ in range(MAXT):
        _, _, d, tr, S = env.step(rr.integers(0, 2, 200)); st += alive; alive &= ~(d | tr)
    print(f"  random policy: {st.mean():.1f} steps balanced (200 episodes)")
    t0 = time.time(); results = {}
    for v in ("dqn", "no_target", "tiny_replay", "neither"):
        runs = [train(v, s) for s in range(3)]
        results[v] = runs
        E = np.array([r["eval"] for r in runs]); Q = np.array([r["qmean"] for r in runs])
        print(f"  {v:11s}: eval steps  mid(iter 1500) {E[:, 15].mean():6.1f} | last 5 evals {E[:, -5:].mean():6.1f} (per seed {np.round(E[:, -5:].mean(1), 0)}) | best eval {E.max(1).round(0)} | max-Q probe final {Q[:, -1].mean():6.1f} (peak {Q.max():.0f})  {time.time() - t0:.0f}s")
    training_gif(here(__file__, "dqn_training.gif"), results["dqn"][0])
    ablation_gif(here(__file__, "dqn_ablation.gif"), results)


if __name__ == "__main__":
    main()
