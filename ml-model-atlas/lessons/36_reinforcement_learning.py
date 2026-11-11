"""
# 36 · Reinforcement Learning: Q-Learning & Bandits
**Family:** Reinforcement learning (learning from rewards by interaction)

## Definition
An **agent** interacts with an **environment**: it observes a state, takes an action, receives a reward and a new state. The goal is a **policy** maximizing the long-run (discounted) reward.
There is no labelled dataset; the agent must **explore** to discover what pays off and **exploit** what it already knows.

## Intuition
*Cliff walking*: the agent must get from the start (bottom-left) to the goal (bottom-right). Each step costs −1 (so shorter is better); stepping on the cliff costs −100 and teleports back to the start.
Q-learning learns a table Q(state, action) = "how good is this move from here, assuming I act well afterwards". Early on the agent stumbles; later its greedy arrows form a route.
*Bandits* are the one-state special case: several slot machines with unknown payout rates; pull them to maximise winnings (A/B testing, ad selection).

## The math
    Return G = Σ γᵗ rₜ              Q*(s,a) = E[r + γ max_a' Q*(s',a')]        (Bellman optimality)
    Q-learning update:   Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]          (off-policy, model-free)
    ε-greedy exploration: random action with prob ε, else argmax Q.     SARSA uses the action actually taken next (safer path near the cliff).
    Bandit strategies:  ε-greedy ;  UCB1:  argmax μ̂ₐ + √(2 ln t / nₐ) ;  Thompson sampling: sample p̂ₐ ~ Beta(1+wins, 1+losses), pull the argmax.
    Regret = Σ (best arm mean − chosen arm mean).      Deep RL replaces the table with a neural network (DQN) or learns the policy directly (policy gradients, PPO, actor-critic).

## What the animations show
1. `36_reinforcement_learning.gif`: left, the value of each grid cell (max Q) with arrows for the greedy policy and the greedy path (dashed); right, reward per episode. Early policies fall off the cliff, later ones find the shortest safe-ish path.
2. `36_reinforcement_learning_bandits.gif`: three bandit strategies on 5 slot machines: pull counts per arm and cumulative regret. Smart exploration (UCB, Thompson) beats naive ε-greedy.

## Try this
- Implement SARSA and compare the learned path to Q-learning's.
- Change ε to 0.3, or decay it over time.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import viz
from mlatlas.cli import run

R_, C_ = 4, 8; ACT = [(-1, 0), (0, 1), (1, 0), (0, -1)]; ARW = "↑→↓←"; START, GOAL = (3, 0), (3, 7)


def step(s, a):
    r, c = s[0] + ACT[a][0], s[1] + ACT[a][1]; r, c = min(max(r, 0), R_ - 1), min(max(c, 0), C_ - 1)
    if r == 3 and 0 < c < 7: return START, -100, False
    return (r, c), -1, (r, c) == GOAL


def greedy_path(Q):
    s, p = START, [START]
    for _ in range(30):
        s, _, done = step(s, int(Q[s].argmax())); p.append(s)
        if done or s == START and len(p) > 2: break
    return p


def main(gif, quick):
    rng = np.random.default_rng(0); Q = np.zeros((R_, C_, 4)); eps, al, ga = .1, .5, .99; E = 150 if quick else 500; rew = []; snaps = {}; keep = set(viz.geo(E, 14 if quick else 36).tolist()); keep.add(E - 1)
    for ep in range(E):
        s, tot = START, 0
        for _ in range(200):
            a = rng.integers(4) if rng.random() < eps else int(np.argmax(Q[s] + rng.random(4) * 1e-6)); s2, r, done = step(s, a); Q[s][a] += al * (r + ga * Q[s2].max() * (not done) - Q[s][a]); s, tot = s2, tot + r
            if done: break
        rew.append(tot)
        if ep in keep: snaps[ep] = (Q.copy(), greedy_path(Q), len(rew))
    print(f"  average reward: first 20 episodes {np.mean(rew[:20]):.0f}, last 50 {np.mean(rew[-50:]):.1f}; learned greedy path = {len(snaps[E - 1][1]) - 1} steps (the shortest possible is 9, hugging the cliff edge)")
    fig, (a1, a2) = viz.subplots(1, 2, w=12, h=4.4)

    def update(ep):
        a1.clear(); a2.clear(); Qs, path, k = snaps[ep]; V = Qs.max(2); a1.imshow(np.clip(V, -30, 0), cmap="viridis", vmin=-30, vmax=0); a1.grid(False)
        for r in range(R_):
            for c in range(C_):
                if r == 3 and 0 < c < 7: a1.add_patch(viz.plt.Rectangle((c - .5, r - .5), 1, 1, color="#222")); a1.text(c, r, "cliff", color="w", ha="center", va="center", fontsize=8)
                elif (r, c) == GOAL: a1.text(c, r, "GOAL", ha="center", va="center", fontweight="bold", color="w")
                else: a1.text(c, r, ARW[int(Qs[r, c].argmax())], ha="center", va="center", fontsize=16, color="w" if V[r, c] < -12 else "k")
        a1.text(0, 3.38, "S", ha="center", va="center", fontsize=9, color="w"); a1.plot([p[1] for p in path], [p[0] for p in path], "w--", lw=2, alpha=.9)
        a1.set(title=f"After {ep + 1} episodes: value (colour), greedy policy (arrows), greedy path (dashed)"); a1.set_xticks([]); a1.set_yticks([])
        a2.plot(np.convolve(rew[:k], np.ones(10) / 10, mode="valid") if k > 10 else rew[:k], color=viz.PALETTE[3]); a2.set(xlim=(0, E), ylim=(-120, 0), title="Reward per episode (10-episode average)", xlabel="episode")
    viz.animate(fig, update, sorted(snaps), gif, fps=4)
    # ---- bandits
    p_true = np.array([.2, .35, .5, .62, .75]); T = 400 if quick else 1500; runs = 20; best = p_true.max(); res = {}
    def run_alg(name, seed):
        r = np.random.default_rng(seed); n = np.zeros(5); w = np.zeros(5); reg = []; cum = 0
        for t in range(T):
            if name == "ε-greedy": a = r.integers(5) if r.random() < .1 or n.min() == 0 and t < 5 else int(np.argmax(w / np.maximum(n, 1)))
            elif name == "UCB1": a = t if t < 5 else int(np.argmax(w / n + np.sqrt(2 * np.log(t + 1) / n)))
            else: a = int(np.argmax(r.beta(1 + w, 1 + n - w)))
            x = r.random() < p_true[a]; n[a] += 1; w[a] += x; cum += best - p_true[a]; reg.append(cum)
        return np.array(reg), n
    for nm in ("ε-greedy", "UCB1", "Thompson"): out = [run_alg(nm, s) for s in range(runs)]; res[nm] = (np.mean([o[0] for o in out], 0), np.mean([o[1] for o in out], 0))
    print("  mean regret after", T, "pulls:", {k: round(float(v[0][-1]), 1) for k, v in res.items()})
    fig, (a1, a2) = viz.subplots(1, 2, w=12, h=4.4); fr = viz.geo(T, 14 if quick else 40)

    def update2(t):
        a1.clear(); a2.clear()
        for i, ((nm, (reg, _)), c) in enumerate(zip(res.items(), viz.PALETTE)): a1.plot(reg[:t + 1], color=c, lw=2, label=nm)
        a1.set(xlim=(0, T), ylim=(0, max(v[0][-1] for v in res.values()) * 1.05), title="Cumulative regret (lower is better)", xlabel="pull"); a1.legend(loc="upper left")
        # pull counts per arm, estimated by scaling final fractions; recompute prefix counts with fixed seeds for honesty
        for i, (nm, c) in enumerate(zip(res, viz.PALETTE)):
            cnt = np.mean([run_alg2(nm, s, t + 1) for s in range(5)], 0); a2.bar(np.arange(5) + (i - 1) * .27, cnt, .27, color=c, label=nm)
        a2.set(title=f"Pulls per arm after {t + 1} steps (true payout: {', '.join(f'{p:.2f}' for p in p_true)})", xlabel="arm", ylim=(0, T * .9)); a2.legend(loc="upper left", fontsize=7)
    def run_alg2(name, seed, upto):
        r = np.random.default_rng(seed); n = np.zeros(5); w = np.zeros(5)
        for t in range(upto):
            if name == "ε-greedy": a = r.integers(5) if r.random() < .1 or n.min() == 0 and t < 5 else int(np.argmax(w / np.maximum(n, 1)))
            elif name == "UCB1": a = t if t < 5 else int(np.argmax(w / n + np.sqrt(2 * np.log(t + 1) / n)))
            else: a = int(np.argmax(r.beta(1 + w, 1 + n - w)))
            x = r.random() < p_true[a]; n[a] += 1; w[a] += x
        return n
    viz.animate(fig, update2, fr, gif.replace(".gif", "_bandits.gif"), fps=4)


if __name__ == "__main__":
    run(main, __file__)
