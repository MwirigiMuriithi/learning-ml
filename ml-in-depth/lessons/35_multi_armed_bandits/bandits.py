"""
Lesson 35 - Multi-armed bandits: explore vs exploit (from scratch)

Problem story: a news site has six candidate headlines for the front page.  Each visitor sees ONE headline and either clicks
(reward 1) or not (0).  The true click rates are unknown.  Every impression spent on a bad headline is lost revenue, but
you cannot know which headline is good without showing it.  This is the exploration-exploitation dilemma.

Implemented (NumPy only), five strategies:
  * pure greedy               (always show the headline with the best average so far)
  * epsilon-greedy            (explore a random headline with probability epsilon; also a decaying epsilon)
  * UCB1                      (optimism: average + sqrt(2 ln t / n) confidence bonus)
  * Thompson sampling         (Bayesian: sample a click rate from each headline's Beta posterior, show the best sample)
  * A/B-test-then-commit      (explore uniformly for a while, then lock in)
Metric: cumulative regret = sum over time of (best click rate - click rate of the shown headline), averaged over many runs.

Animated:
  bandit_posteriors.gif  Thompson sampling: the Beta posteriors narrow and the best headline gets almost all impressions
  bandit_regret.gif      regret of all strategies over 5000 visitors (average of 300 runs) and who they show
  bandit_ucb.gif         UCB1's confidence intervals shrinking: optimism in the face of uncertainty

Run:  python bandits.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403

MU = np.array([0.04, 0.06, 0.05, 0.11, 0.09, 0.03])           # true click rates (unknown to the algorithms)
K = len(MU)
BEST = MU.max()


class Agent:
    name = "?"

    def __init__(self, r):
        self.r = r; self.n = np.zeros(K); self.s = np.zeros(K); self.t = 0

    def update(self, a, x):
        self.n[a] += 1; self.s[a] += x; self.t += 1

    def mean(self):
        return np.where(self.n > 0, self.s / np.maximum(self.n, 1), 0.0)


class Greedy(Agent):
    name = "greedy"

    def act(self):
        if self.t < K:
            return self.t                                       # try each once
        return int(np.argmax(self.mean()))


class EpsGreedy(Agent):
    name = "ε-greedy (0.1)"
    eps = 0.1

    def act(self):
        if self.t < K:
            return self.t
        if self.r.random() < self.eps:
            return int(self.r.integers(K))
        return int(np.argmax(self.mean()))


class DecayEps(Agent):
    name = "ε-greedy decaying"

    def act(self):
        if self.t < K:
            return self.t
        eps = min(1.0, 5.0 * K / self.t) * 0.5               # exploration shrinks as 1/t
        if self.r.random() < eps:
            return int(self.r.integers(K))
        return int(np.argmax(self.mean()))


class UCB(Agent):
    name = "UCB1"

    def bounds(self):
        return self.mean() + np.sqrt(2 * np.log(max(self.t, 1)) / np.maximum(self.n, 1))

    def act(self):
        if self.t < K:
            return self.t
        return int(np.argmax(self.bounds()))


class Thompson(Agent):
    name = "Thompson sampling"

    def post(self):
        return 1 + self.s, 1 + self.n - self.s                  # Beta(1 + clicks, 1 + non-clicks) with a uniform prior

    def act(self):
        a, b = self.post()
        return int(np.argmax(self.r.beta(a, b)))


class ABCommit(Agent):
    name = "A/B test, then commit"
    explore = 600

    def act(self):
        if self.t < self.explore:
            return self.t % K
        return int(np.argmax(self.mean()))


def run(agent_cls, T, seed):
    r = rng_(seed)
    ag = agent_cls(rng_(seed + 1000))
    regret = np.zeros(T); pulls = np.zeros((T, K), dtype=np.int32); gap = BEST - MU
    for t in range(T):
        a = ag.act()
        x = float(r.random() < MU[a])
        ag.update(a, x)
        regret[t] = gap[a]; pulls[t, a] = 1
    return np.cumsum(regret), np.cumsum(pulls, 0)


def lai_robbins(T):
    """Asymptotic lower bound on expected regret of any consistent algorithm: sum_a gap_a ln T / KL(mu_a || mu*)."""
    kl = lambda p, q: p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
    return sum((BEST - m) * np.log(T) / kl(m, BEST) for m in MU if m < BEST)


# ------------------------------------------------------------------- animations
def posteriors_gif(path, T=3000, seed=3):
    r = rng_(seed); ag = Thompson(rng_(seed + 1))
    grid = np.linspace(0, 0.3, 400)
    from math import lgamma
    def beta_pdf(x, a, b):
        return np.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + (a - 1) * np.log(x + 1e-12) + (b - 1) * np.log(1 - x + 1e-12))
    shows = {0, 5, 20, 60, 150, 400, 1000, T}
    with Animator(path, fps=1.4, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.8), title="Thompson sampling: beliefs about six headlines' click rates", gridspec_kw={"width_ratios": [1.5, 1]})
        for t in range(T + 1):
            if t in shows:
                a1.clear(); a2.clear()
                a, b = ag.post()
                for k in range(K):
                    a1.plot(grid, beta_pdf(grid, a[k], b[k]) / max(1, beta_pdf(grid, a[k], b[k]).max()) * 1.0 if t == 0 else beta_pdf(grid, a[k], b[k]), color=CAT[k], lw=2, label=f"headline {k + 1} (true {MU[k]:.2f})")
                    a1.axvline(MU[k], color=CAT[k], ls=":", lw=1)
                a1.set(xlim=(0, 0.3), title=f"Posterior of each click rate after {t} visitors" if t else "Prior: every click rate equally plausible", xlabel="click rate", ylabel="density", yticks=[])
                a1.legend(loc="upper right", fontsize=7.5)
                a2.bar([f"{k + 1}" for k in range(K)], ag.n, color=CAT[:K]); a2.set(title="Impressions per headline", xlabel="headline")
                a2.grid(axis="x", visible=False)
                best_share = ag.n[np.argmax(MU)] / max(t, 1)
                caption(fig, f"{t} visitors: headline 4 (true best, 0.11) has {best_share:.0%} of impressions; the posteriors of poor headlines stay wide but are rarely sampled because they rarely win")
                anim.grab(fig)
            if t == T:
                break
            act = ag.act(); ag.update(act, float(r.random() < MU[act]))


def regret_gif(path, T=5000, runs=300):
    classes = [Greedy, EpsGreedy, DecayEps, UCB, Thompson, ABCommit]
    cols = [MUTED, ORANGE, YELLOW, AQUA, BLUE, VIOLET]
    R = {}; shares = {}
    for c in classes:
        reg = np.zeros(T); sh = np.zeros(K)
        for s in range(runs):
            rg, pl = run(c, T, s)
            reg += rg / runs; sh += pl[-1] / T / runs
        R[c.name] = reg; shares[c.name] = sh
    print(f"  average regret after {T} visitors ({runs} runs):")
    for c in classes:
        print(f"     {c.name:24s} {R[c.name][-1]:7.1f}   (share of impressions on the best headline: {shares[c.name][np.argmax(MU)]:.2f})")
    checkpoints = [10, 50, 150, 400, 800, 1500, 2500, 5000]
    with Animator(path, fps=1.2, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.9), title="Regret: the price of not knowing", gridspec_kw={"width_ratios": [1.4, 1]})
        for t in checkpoints:
            a1.clear(); a2.clear()
            for c, col in zip(classes, cols):
                a1.plot(np.arange(t), R[c.name][:t], color=col, lw=2, label=c.name)
            a1.set(xlim=(0, T), ylim=(0, max(v[-1] for v in R.values()) * 1.05), title="Average cumulative regret (lower is better)", xlabel="visitors", ylabel="lost expected clicks"); a1.legend(loc="upper left", fontsize=8)
            names = [c.name for c in classes]
            a2.barh(names, [R[n][t - 1] for n in names], color=cols); a2.invert_yaxis()
            a2.set(title=f"Regret after {t} visitors"); a2.grid(axis="y", visible=False)
            best = min(names, key=lambda n: R[n][t - 1])
            caption(fig, f"after {t} visitors the lowest regret is {best} ({R[best][t - 1]:.1f}); greedy locks onto a wrong headline in some runs and its regret grows linearly")
            anim.grab(fig)
    return R


def ucb_gif(path, T=1500, seed=4):
    r = rng_(seed); ag = UCB(rng_(0))
    shows = [20, 60, 150, 400, 800, T]
    with Animator(path, fps=1.2, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.8), title="UCB1: pick the headline with the highest *optimistic* estimate", gridspec_kw={"width_ratios": [1.3, 1]})
        for t in range(1, T + 1):
            act = ag.act(); ag.update(act, float(r.random() < MU[act]))
            if t in shows:
                a1.clear(); a2.clear()
                m = ag.mean(); ub = ag.bounds()
                for k in range(K):
                    a1.plot([k, k], [m[k], ub[k]], color=CAT[k], lw=6, solid_capstyle="butt", alpha=0.45)
                    a1.scatter([k], [m[k]], s=70, color=CAT[k], zorder=4, edgecolor=SURFACE)
                    a1.scatter([k], [MU[k]], s=90, marker="_", color=INK, zorder=5, linewidth=2)
                a1.set(xticks=range(K), xticklabels=[f"{k + 1}" for k in range(K)], ylim=(0, min(1.0, ub.max() * 1.1)), xlabel="headline", ylabel="click rate", title="Estimate (dot), upper confidence bound (top of bar), truth (black tick)")
                a2.bar([f"{k + 1}" for k in range(K)], ag.n, color=CAT[:K]); a2.set(title="Impressions"); a2.grid(axis="x", visible=False)
                caption(fig, f"{t} visitors: the bonus sqrt(2 ln t / n) shrinks for headlines that were shown often; rarely shown headlines keep a tall bar and get another chance")
                anim.grab(fig)


def main():
    print("Lesson 35: bandits")
    print(f"  true click rates {MU}; best headline = {int(np.argmax(MU)) + 1} ({BEST})")
    # check Thompson posterior: after many pulls the Beta mean matches the true rate
    ag = Thompson(rng_(0)); r = rng_(1)
    for _ in range(5000):
        a = ag.act(); ag.update(a, float(r.random() < MU[a]))
    a, b = ag.post()
    print("  Thompson posterior means after 5000 visitors:", np.round(a / (a + b), 3), " impressions:", ag.n.astype(int))
    posteriors_gif(here(__file__, "bandit_posteriors.gif"))
    regret_gif(here(__file__, "bandit_regret.gif"))
    ucb_gif(here(__file__, "bandit_ucb.gif"))


if __name__ == "__main__":
    main()
