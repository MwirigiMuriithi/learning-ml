"""
Lesson 36 - Tabular Q-learning and SARSA on the "cliff" (from scratch)

Problem story: a warehouse robot must drive from the charging dock (bottom left) to the packing station (bottom right).  The
direct row runs along the edge of a loading-bay drop (the "cliff"): stepping into it costs -100 and sends the robot back to
the dock.  Every move costs -1 (battery).  The shortest route hugs the cliff; a safer route goes around, higher up.  The robot's
steering is imperfect while it is still learning (it takes a random move 10% of the time), which is exactly what makes the choice
interesting.

Implemented (NumPy only):
  * the 4 x 12 cliff-walking environment
  * Q-learning (off-policy):  Q(s,a) += alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]
  * SARSA (on-policy):        Q(s,a) += alpha [ r + gamma Q(s',a') - Q(s,a) ]    with a' chosen by the same epsilon-greedy policy
  * value iteration (dynamic programming with the known model) to get the exact optimal Q* and check Q-learning against it

Animated:
  qlearning_learning.gif  Q-values and greedy arrows appear: the value of the goal spreads backwards one step per sweep
  qlearning_vs_sarsa.gif  the optimal-but-risky path (Q-learning) versus the safe path (SARSA), and their online rewards

Run:  python q_learning.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403

H, W = 4, 12
START, GOAL = (3, 0), (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]            # up, right, down, left
ARROWS = ["↑", "→", "↓", "←"]


def step(s, a):
    r, c = s
    dr, dc = MOVES[a]
    nr, nc = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
    if (nr, nc) in CLIFF:
        return START, -100.0, False
    return (nr, nc), -1.0, (nr, nc) == GOAL


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(4))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))          # random tie-break


def run_episode(Q, algo, eps, alpha, gamma, rng, max_steps=300, learn=True):
    s, a = START, None
    a = eps_greedy(Q, s, eps, rng)
    total, path, falls = 0.0, [s], 0
    for _ in range(max_steps):
        s2, r, done = step(s, a)
        total += r
        falls += r < -50
        a2 = eps_greedy(Q, s2, eps, rng)
        if learn:
            if algo == "q":
                target = r + (0 if done else gamma * Q[s2].max())
            else:                                                  # sarsa
                target = r + (0 if done else gamma * Q[s2][a2])
            Q[s][a] += alpha * (target - Q[s][a])
        path.append(s2)
        s, a = s2, a2
        if done:
            break
    return total, path, falls


def new_Q():
    return {(r, c): np.zeros(4) for r in range(H) for c in range(W) if (r, c) not in CLIFF}


def greedy_path(Q, max_steps=60):
    s, path = START, [START]
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s, _, done = step(s, a)
        path.append(s)
        if done:
            break
    return path


def value_iteration(gamma=1.0, iters=500):
    Q = new_Q()
    for _ in range(iters):
        for s in Q:
            if s == GOAL:
                continue
            for a in range(4):
                s2, r, done = step(s, a)
                Q[s][a] = r + (0 if done else gamma * Q[s2].max())
    return Q


def v_grid(Q):
    V = np.full((H, W), np.nan)
    for s, q in Q.items():
        V[s] = q.max()
    return V


def draw_grid(ax, Q, path=None, title="", vmin=-40, vmax=0):
    V = v_grid(Q)
    ax.imshow(np.clip(V, vmin, vmax), cmap="Blues", vmin=vmin, vmax=vmax, aspect="equal")
    ax.grid(False)
    for r in range(H):
        for c in range(W):
            if (r, c) in CLIFF:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=RED, alpha=0.35, lw=0))
            elif (r, c) == GOAL:
                ax.text(c, r, "G", ha="center", va="center", fontsize=13, color=GREEN, fontweight="bold")
            else:
                q = Q[(r, c)]
                if not np.any(q != 0):
                    ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=SURFACE, lw=0))
                if np.any(q != 0):
                    ax.text(c, r, ARROWS[int(np.argmax(q))], ha="center", va="center", fontsize=12, color=INK)
    ax.text(0, 3, "S", ha="center", va="center", fontsize=13, color=ORANGE, fontweight="bold")
    if path is not None:
        p = np.array(path)
        ax.plot(p[:, 1], p[:, 0], color=ORANGE, lw=2.5, alpha=0.9)
    ax.text(5.5, 3, "cliff", ha="center", va="center", fontsize=9, color=RED)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)


# ------------------------------------------------------------------- animations
def learning_gif(path, n_ep=300, seed=2):
    rng = rng_(seed); Q = new_Q()
    shows = {0, 1, 2, 3, 5, 8, 12, 20, 40, 80, 150, n_ep}
    rets, steps_ = [], []
    last_path = [START]
    with Animator(path, fps=1.5, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.4), title="Q-learning on the cliff (ε = 0.1, α = 0.5, γ = 1)", gridspec_kw={"width_ratios": [1.8, 1]})
        for ep in range(n_ep + 1):
            if ep in shows:
                a1.clear(); a2.clear()
                draw_grid(a1, Q, last_path if ep else None, f"After {ep} episodes (dark = high value, arrow = greedy move)")
                if rets:
                    a2.plot(np.arange(1, len(rets) + 1), rets, color=BLUE, lw=0.9, alpha=0.5, label="reward of each episode")
                    k = min(10, len(rets)); sm = np.convolve(rets, np.ones(k) / k, mode="valid")
                    a2.plot(np.arange(k, len(rets) + 1), sm, color=ORANGE, lw=2.4, label="10-episode average")
                a2.set(xlim=(0, n_ep), ylim=(-250, 0), title="Total reward per episode", xlabel="episode"); a2.legend(loc="lower right", fontsize=8) if rets else None
                gp = greedy_path(Q)
                reach = gp[-1] == GOAL
                caption(fig, f"{ep} episodes: " + ("nothing known; the first episodes wander (long, with falls)" if ep == 0 else
                        f"value information spreads backwards from the goal; the greedy path {'reaches the goal in ' + str(len(gp) - 1) + ' steps' if reach else 'does not reach the goal yet'}"))
                anim.grab(fig)
            if ep == n_ep:
                break
            tot, last_path, _ = run_episode(Q, "q", 0.1, 0.5, 1.0, rng)
            rets.append(tot)


def compare_gif(out, n_ep=500, runs=40):
    curves = {"q": np.zeros(n_ep), "s": np.zeros(n_ep)}
    finalQ = {}
    for algo in ("q", "s"):
        for s in range(runs):
            rng = rng_(100 + s); Q = new_Q()
            for ep in range(n_ep):
                tot, _, _ = run_episode(Q, algo, 0.1, 0.5, 1.0, rng)
                curves[algo][ep] += tot / runs
            if s == 0:
                finalQ[algo] = Q
    print(f"  online reward per episode (episodes 100-500, {runs} runs, eps=0.1): Q-learning {curves['q'][100:].mean():.1f}, SARSA {curves['s'][100:].mean():.1f}")
    # greedy paths and how often each agent falls
    for algo in ("q", "s"):
        gp = greedy_path(finalQ[algo]); print(f"  {algo.upper():1s} greedy path: {len(gp) - 1} steps; goes along row {min(p[0] for p in gp)}-{max(p[0] for p in gp)} (cliff edge is row 3)")
    falls = {"q": 0, "s": 0}
    for algo in ("q", "s"):
        rng = rng_(5)
        for _ in range(200):
            falls[algo] += run_episode(finalQ[algo], algo, 0.1, 0.5, 1.0, rng, learn=False)[2]
    print(f"  falls per episode while following each policy with eps = 0.1 (200 episodes): Q-learning {falls['q'] / 200:.2f}, SARSA {falls['s'] / 200:.2f}")
    shows = [10, 50, 100, 200, 350, 500]
    with Animator(out, fps=1.2, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(14, 4.4), title="Same experience, different targets: Q-learning vs SARSA", gridspec_kw={"width_ratios": [1.3, 1.3, 1.0]})
        for ep in shows:
            for a in (a1, a2, a3):
                a.clear()
            # rebuild the Q-tables at this episode for the first run
            Qs = {}
            for algo in ("q", "s"):
                rng = rng_(100); Q = new_Q()
                for _ in range(ep):
                    run_episode(Q, algo, 0.1, 0.5, 1.0, rng)
                Qs[algo] = Q
            draw_grid(a1, Qs["q"], greedy_path(Qs["q"]), f"Q-learning, {ep} episodes")
            draw_grid(a2, Qs["s"], greedy_path(Qs["s"]), f"SARSA, {ep} episodes")
            for algo, col, name in (("q", ORANGE, "Q-learning"), ("s", BLUE, "SARSA")):
                k = 10; sm = np.convolve(curves[algo][:ep], np.ones(k) / k, mode="valid") if ep > k else curves[algo][:ep]
                a3.plot(np.arange(len(sm)) + (k if ep > k else 0), sm, color=col, lw=2.2, label=name)
            a3.set(xlim=(0, n_ep), ylim=(-100, 0), title="Reward while learning (avg of 40 runs)", xlabel="episode"); a3.legend(loc="lower right")
            caption(fig, f"episode {ep}: Q-learning hugs the cliff (best if acting greedily); SARSA detours because it counts its own random slips")
            anim.grab(fig)


def main():
    print("Lesson 36: Q-learning")
    Qs = value_iteration()
    gp = greedy_path(Qs)
    print(f"  value iteration (exact optimum, gamma = 1): start value {Qs[START].max():.0f} (the 13-move route along the cliff edge), optimal path length {len(gp) - 1}")
    # Q-learning with decaying exploration and learning rate should converge to Q*
    rng = rng_(0); Q = new_Q()
    for ep in range(6000):
        eps = max(0.02, 1.0 / (1 + ep / 50)); alpha = max(0.05, 0.5 / (1 + ep / 500))
        run_episode(Q, "q", eps, alpha, 1.0, rng, max_steps=500)
    err = max(abs(Q[s].max() - Qs[s].max()) for s in Q if s != GOAL and Q[s].max() != 0)
    onpath = [s for s in gp if s != GOAL]
    print(f"  Q-learning (decaying eps, alpha; 6000 episodes) vs Q*: max |V - V*| over visited states {err:.2f}; on the optimal path {max(abs(Q[s].max() - Qs[s].max()) for s in onpath):.2f}; greedy policy reaches the goal in {len(greedy_path(Q)) - 1} steps")
    learning_gif(here(__file__, "qlearning_learning.gif"))
    compare_gif(here(__file__, "qlearning_vs_sarsa.gif"))


if __name__ == "__main__":
    main()
