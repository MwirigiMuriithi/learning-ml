"""
Lesson 30 - Hidden Markov models: forward-backward, Viterbi, Baum-Welch (from scratch)

Problem story: a fitness band records heart rate every 10 seconds, but nobody tells it what the wearer is doing.  The activity
(resting / walking / running) is a *hidden* state that changes rarely (people stay in an activity for a while) and produces a
noisy heart rate (the three ranges overlap).  Questions:
  1. filtering   - given the readings so far, how likely is each activity right now?
  2. smoothing   - given ALL readings, how likely was each activity at time t?
  3. decoding    - what is the single most likely activity sequence?   (Viterbi)
  4. learning    - without any labels, learn the transition matrix and the heart-rate ranges   (Baum-Welch = EM)

Implemented (NumPy only): Gaussian-emission HMM with scaled forward / backward recursions, Viterbi in log space, Baum-Welch
(EM), sampling, brute-force checks on short sequences.

Animated:
  hmm_inference.gif  readings arrive; the filtered belief updates; then smoothing and Viterbi clean it up
  hmm_baum_welch.gif EM from a poor starting guess: Gaussians, transition matrix, likelihood and accuracy improve together

Run:  python hmm.py
"""
import sys, pathlib, itertools

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403

NAMES = ["resting", "walking", "running"]


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x[:, None] - mu[None]) / sd[None]) ** 2) / (sd[None] * np.sqrt(2 * np.pi))


class HMM:
    def __init__(self, pi, A, mu, sd):
        self.pi, self.A, self.mu, self.sd = np.array(pi, float), np.array(A, float), np.array(mu, float), np.array(sd, float)

    @property
    def K(self):
        return len(self.pi)

    def sample(self, T, seed=0):
        r = rng_(seed)
        z = np.zeros(T, int); z[0] = r.choice(self.K, p=self.pi)
        for t in range(1, T):
            z[t] = r.choice(self.K, p=self.A[z[t - 1]])
        x = self.mu[z] + self.sd[z] * r.normal(size=T)
        return x, z

    # ---------------------------------------------------------------- inference
    def forward(self, x):
        """alpha_t(k) = P(z_t = k | x_1..t) (normalised each step; c_t = P(x_t | x_1..t-1), log-likelihood = sum log c_t)."""
        B = gauss(x, self.mu, self.sd)
        T = len(x); alpha = np.zeros((T, self.K)); c = np.zeros(T)
        a = self.pi * B[0]; c[0] = a.sum(); alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * B[t]
            c[t] = a.sum(); alpha[t] = a / c[t]
        return alpha, c, B

    def backward(self, B, c):
        T = len(c); beta = np.ones((T, self.K))
        for t in range(T - 2, -1, -1):
            beta[t] = self.A @ (B[t + 1] * beta[t + 1]) / c[t + 1]
        return beta

    def smooth(self, x):
        alpha, c, B = self.forward(x)
        beta = self.backward(B, c)
        gamma = alpha * beta                                        # P(z_t | all x)
        return gamma, alpha, beta, c, B

    def loglik(self, x):
        return float(np.log(self.forward(x)[1]).sum())

    def viterbi(self, x):
        B = np.log(gauss(x, self.mu, self.sd) + 1e-300); T = len(x)
        logA = np.log(self.A + 1e-300)
        d = np.log(self.pi + 1e-300) + B[0]; psi = np.zeros((T, self.K), int)
        for t in range(1, T):
            cand = d[:, None] + logA
            psi[t] = cand.argmax(0); d = cand.max(0) + B[t]
        path = np.zeros(T, int); path[-1] = d.argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, d.max()

    # ---------------------------------------------------------------- learning (Baum-Welch)
    def em_step(self, x):
        gamma, alpha, beta, c, B = self.smooth(x)
        T = len(x)
        xi = np.zeros((self.K, self.K))
        for t in range(T - 1):
            xi += alpha[t][:, None] * self.A * (B[t + 1] * beta[t + 1])[None] / c[t + 1]
        self.pi = gamma[0] / gamma[0].sum()
        self.A = xi / xi.sum(1, keepdims=True)
        Nk = gamma.sum(0)
        self.mu = (gamma * x[:, None]).sum(0) / Nk
        self.sd = np.sqrt((gamma * (x[:, None] - self.mu[None]) ** 2).sum(0) / Nk) + 1e-3
        return float(np.log(c).sum())


def brute_force(h, x):
    """Enumerate all K^T state paths (tiny T only) -> exact likelihood, best path, and marginals."""
    T, K = len(x), h.K
    B = gauss(x, h.mu, h.sd)
    tot, best, bestp = 0.0, -1.0, None
    marg = np.zeros((T, K))
    for path in itertools.product(range(K), repeat=T):
        p = h.pi[path[0]] * B[0, path[0]]
        for t in range(1, T):
            p *= h.A[path[t - 1], path[t]] * B[t, path[t]]
        tot += p
        for t in range(T):
            marg[t, path[t]] += p
        if p > best:
            best, bestp = p, path
    return tot, np.array(bestp), marg / tot


def stationary(A):
    """Long-run share of time in each state: left eigenvector of A for eigenvalue 1."""
    vals, vecs = np.linalg.eig(A.T)
    v = np.real(vecs[:, np.argmax(np.real(vals))])
    return v / v.sum()


def accuracy(z, zhat):
    return float(np.mean(z == zhat))


# ------------------------------------------------------------------- animations
def inference_gif(path, x, z, h):
    T = len(x)
    gamma, alpha, _, _, _ = h.smooth(x)
    vit, _ = h.viterbi(x)
    shown = list(range(5, 61, 5)) + [100, 150, 200, T]
    win = 100
    with Animator(path, fps=1.6, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(2, 1, figsize=(13, 6.2), title="Hidden Markov model: infer the activity from heart rate alone", gridspec_kw={"height_ratios": [1, 1]})
        for k, t in enumerate(shown + ["smooth", "viterbi"]):
            for a in (a1, a2):
                a.clear()
            tt = T if isinstance(t, str) else t
            lo = max(0, tt - win)
            xs = np.arange(lo, tt)
            # truth band
            for s in range(3):
                m = z[lo:tt] == s
                a1.fill_between(xs, 40, 190, where=m, color=CAT[s], alpha=0.16, step="mid", lw=0)
            a1.plot(xs, x[lo:tt], color=INK, lw=1.3)
            a1.scatter(xs, x[lo:tt], s=10, color=INK, zorder=3)
            a1.set(xlim=(lo, lo + win), ylim=(40, 190), ylabel="heart rate (bpm)", title="Readings (background colour = the TRUE hidden activity, which the model never sees)")
            if t == "viterbi":
                for s in range(3):
                    a2.fill_between(xs, 0, 1, where=vit[lo:tt] == s, color=CAT[s], alpha=0.85, step="mid", lw=0, label=NAMES[s])
                a2.set(title=f"Viterbi: the single most probable activity sequence (accuracy {accuracy(z, vit):.2f})")
                msg = "Viterbi decoding picks the best whole path (dynamic programming), so impossible rapid flicker is avoided"
            else:
                P = gamma if t == "smooth" else alpha
                bottom = np.zeros(len(xs))
                for s in range(3):
                    a2.bar(xs, P[lo:tt, s], bottom=bottom, color=CAT[s], width=1.0, label=NAMES[s]); bottom += P[lo:tt, s]
                if t == "smooth":
                    a2.set(title=f"Smoothing: P(activity | ALL readings)   (argmax accuracy {accuracy(z, gamma.argmax(1)):.2f})")
                    msg = "Using future readings too sharpens the beliefs at the boundaries"
                else:
                    a2.set(title=f"Filtering: P(activity now | readings so far)   (argmax accuracy so far {accuracy(z[:tt], alpha[:tt].argmax(1)):.2f})")
                    msg = f"{tt} readings seen: the belief reacts to each new reading but is damped by the 'people rarely switch activity' prior"
            a2.set(xlim=(lo, lo + win), ylim=(0, 1), xlabel="time (10 s per step)"); a2.legend(loc="upper left", ncol=3, fontsize=8)
            caption(fig, msg)
            anim.grab(fig)


def bw_gif(path, x, z, true_h, init, n_iter=40):
    h = HMM(init.pi.copy(), init.A.copy(), init.mu.copy(), init.sd.copy())
    snaps = [(0, h.loglik(x), HMM(h.pi.copy(), h.A.copy(), h.mu.copy(), h.sd.copy()))]
    ll = [h.loglik(x)]
    for it in range(1, n_iter + 1):
        h.em_step(x); ll.append(h.loglik(x))
        snaps.append((it, ll[-1], HMM(h.pi.copy(), h.A.copy(), h.mu.copy(), h.sd.copy())))
    show = [0, 1, 2, 3, 5, 8, 12, 20, 30, 40]
    print("  Baum-Welch log-likelihood by iteration:", {i: round(ll[i], 1) for i in show}, "; monotone:", bool(np.all(np.diff(ll) > -1e-5)))
    grid = np.linspace(40, 190, 300)
    with Animator(path, fps=1.4, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Baum-Welch (EM): learning an HMM without labels", gridspec_kw={"width_ratios": [1.3, 0.9, 1.0]})
        for it in show:
            _, l, m = snaps[it]
            for a in (a1, a2, a3):
                a.clear()
            a1.hist(x, bins=40, density=True, color=MUTED, alpha=0.45)
            order = np.argsort(m.mu)
            dens = np.zeros_like(grid)
            for rnk, s in enumerate(order):
                dd = np.exp(-0.5 * ((grid - m.mu[s]) / m.sd[s]) ** 2) / (m.sd[s] * np.sqrt(2 * np.pi))
                w = stationary(m.A)[s]
                a1.plot(grid, dd * w, color=CAT[rnk], lw=2.2, label=f"state {rnk + 1}: μ={m.mu[s]:.0f}, σ={m.sd[s]:.0f}")
            a1.set(title="Heart-rate histogram and learned state densities", xlabel="bpm", xlim=(40, 190)); a1.legend(loc="upper right", fontsize=7.5)
            A = m.A[np.ix_(order, order)]
            a2.imshow(A, cmap="Blues", vmin=0, vmax=1); a2.grid(False)
            for (i, j), v in np.ndenumerate(A):
                a2.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v > 0.5 else INK, fontsize=10)
            a2.set_xticks(range(3), ["s1", "s2", "s3"]); a2.set_yticks(range(3), ["s1", "s2", "s3"]); a2.set(title="Transition matrix (row = from)", xlabel="to")
            a3.plot(range(it + 1), ll[: it + 1], color=BLUE, lw=2)
            a3.axhline(true_h.loglik(x), color=ORANGE, ls="--", lw=1.5); a3.text(1, true_h.loglik(x) + 3, "true parameters", color=ORANGE, fontsize=8)
            lo_ = min(ll) - 20
            a3.set(xlim=(0, n_iter), ylim=(lo_, true_h.loglik(x) + 25), title="Log-likelihood", xlabel="EM iteration")
            vit, _ = m.viterbi(x)
            # map learned states to true by sorted means
            rank = np.argsort(np.argsort(m.mu))
            acc = accuracy(z, rank[vit])
            caption(fig, f"iteration {it}: log-likelihood {l:.1f}; decoded activity accuracy {acc:.2f} (states matched by their mean heart rate)")
            anim.grab(fig)
    return h


def main():
    print("Lesson 30: hidden Markov models")
    A = [[0.95, 0.04, 0.01], [0.05, 0.90, 0.05], [0.01, 0.07, 0.92]]
    true = HMM([0.5, 0.3, 0.2], A, [62, 86, 108], [8, 11, 12])
    x, z = true.sample(400, seed=4)
    print(f"  data: {len(x)} readings; true activity shares {np.round(np.bincount(z) / len(z), 2)}; number of state switches {int((np.diff(z) != 0).sum())}")
    # --- exactness checks on short sequences
    xs, _ = true.sample(7, seed=1)
    tot, bpath, marg = brute_force(true, xs)
    g, alpha, beta, c, B = true.smooth(xs)
    vp, _ = true.viterbi(xs)
    print(f"  brute force (3^7 paths): likelihood {np.log(tot):.6f} vs forward {true.loglik(xs):.6f}; Viterbi path equal: {np.array_equal(bpath, vp)}; "
          f"smoothed marginals max diff {np.abs(marg - g).max():.1e}")
    gamma, alpha, _, _, _ = true.smooth(x)
    vit, _ = true.viterbi(x)
    print(f"  with the TRUE parameters: filtering accuracy {accuracy(z, alpha.argmax(1)):.3f}, smoothing {accuracy(z, gamma.argmax(1)):.3f}, Viterbi {accuracy(z, vit):.3f}; "
          f"memory-less rule (nearest mean) {accuracy(z, np.abs(x[:, None] - true.mu).argmin(1)):.3f}")
    # --- learning
    init = HMM([1 / 3] * 3, np.full((3, 3), 1 / 3) + 0.02 * rng_(2).random((3, 3)), [75, 90, 105], [20, 20, 20])
    init.A /= init.A.sum(1, keepdims=True)
    fit = bw_gif(here(__file__, "hmm_baum_welch.gif"), x, z, true, init)
    order = np.argsort(fit.mu)
    print(f"  learned means {np.round(fit.mu[order], 1)} (true {true.mu}), sds {np.round(fit.sd[order], 1)} (true {true.sd})")
    print("  learned transition matrix (rows/cols sorted by mean):\n", np.round(fit.A[np.ix_(order, order)], 3), "\n  true matrix:\n", np.array(A))
    rank = np.argsort(np.argsort(fit.mu))
    print(f"  learned-model Viterbi accuracy {accuracy(z, rank[fit.viterbi(x)[0]]):.3f}; log-likelihood learned {fit.loglik(x):.1f} vs true parameters {true.loglik(x):.1f}")
    try:
        from hmmlearn.hmm import GaussianHMM
        print("  [hmmlearn] available")
    except Exception:
        print("  (hmmlearn not installed: verified by exhaustive enumeration instead)")
    inference_gif(here(__file__, "hmm_inference.gif"), x, z, true)


if __name__ == "__main__":
    main()
