"""
Lesson 12 - The Perceptron (from scratch)

Problem story: a bank screens loan requests with two scores (credit score, income).  Past decisions are
labelled approve (+1) / decline (-1).  The perceptron is the oldest learning machine: look at one case, and
ONLY if you get it wrong, nudge the weights toward that case.

Implemented (NumPy only):
  * Perceptron       - the classic mistake-driven rule  w <- w + y x ,  b <- b + y
  * pocket variant   - keep the best weights seen so far (for non-separable data)
  * averaged variant - predict with the average of all weight vectors visited
  * Novikoff bound   - (R / margin)^2 limits the number of mistakes on separable data (checked numerically)

Animated:
  perceptron_learning.gif - update-by-update on separable data: the line jumps after each mistake (offending point
                            ringed), the weight vector walks in weight space, mistakes accumulate below the bound.
  perceptron_xor.gif      - XOR data: the plain perceptron never settles; pocket/averaged variants cope a bit;
                            adding ONE product feature x1*x2 makes the problem separable and the mistakes vanish.

Run:  python perceptron.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


class Perceptron:
    """Labels are +1 / -1.  Decision: sign(w.x + b)."""

    def __init__(self, n_epochs=100, shuffle=False, seed=0):
        self.n_epochs, self.shuffle, self.seed = n_epochs, shuffle, seed

    def fit(self, X, y, record=False):
        r = rng_(self.seed)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        w_sum, b_sum, cnt = np.zeros(d), 0.0, 0           # running sums -> averaged perceptron
        best = (-1.0, w.copy(), b)                       # pocket: (training accuracy, w, b)
        self.history, self.epoch_mistakes = [], []
        for ep in range(self.n_epochs):
            order = r.permutation(n) if self.shuffle else np.arange(n)
            mistakes = 0
            for i in order:
                if y[i] * (X[i] @ w + b) <= 0:           # misclassified (or on the line): UPDATE
                    w = w + y[i] * X[i]                  # move the line toward getting x_i right
                    b = b + y[i]
                    mistakes += 1
                    if record:
                        self.history.append(dict(w=w.copy(), b=b, i=i, epoch=ep))
                w_sum += w; b_sum += b; cnt += 1
            acc = np.mean(np.sign(X @ w + b) == y)
            if acc > best[0]:
                best = (acc, w.copy(), b)
            self.epoch_mistakes.append(mistakes)
            if record is False and mistakes == 0:
                break
        self.w, self.b = w, b
        self.w_pocket, self.b_pocket = best[1], best[2]
        self.w_avg, self.b_avg = w_sum / cnt, b_sum / cnt
        return self

    def predict(self, X, kind="last"):
        w, b = {"last": (self.w, self.b), "pocket": (self.w_pocket, self.b_pocket), "avg": (self.w_avg, self.b_avg)}[kind]
        return np.where(X @ w + b > 0, 1, -1)


def loan_data(n=50, seed=3, gap=0.1):
    """Linearly separable loans.  Points closer than `gap` to the true boundary are removed, so a margin exists.
    The true rule (credit 0.5 + income 1.0 > 0.7) does not pass through the origin, so the bias term matters."""
    r = rng_(seed)
    X = r.uniform(-2, 2, (8 * n, 2))
    wtrue = np.array([0.5, 1.0])
    s = (X @ wtrue - 0.7) / np.linalg.norm(wtrue)
    keep = np.abs(s) > gap
    X, s = X[keep][:n], s[keep][:n]
    return X, np.where(s > 0, 1, -1)


def novikoff(X, y, w, b):
    """Mistake bound (R/gamma)^2 using the separator (w, b) with absorbed bias."""
    A = np.c_[X, np.ones(len(X))]
    wt = np.r_[w, b]
    gamma = np.min(y * (A @ wt)) / np.linalg.norm(wt)
    R = np.max(np.linalg.norm(A, axis=1))
    return (R / gamma) ** 2, R, gamma


def draw_line(ax, w, b, xlim, ylim, **kw):
    if abs(w[1]) > 1e-9:
        xs = np.array(xlim)
        ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)
    elif abs(w[0]) > 1e-9:
        ax.plot([-b / w[0]] * 2, ylim, **kw)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)


def learning_animation():
    X, y = loan_data()
    p = Perceptron(n_epochs=200).fit(X, y, record=True)
    H = p.history
    # when the final epoch makes no mistakes, record=True keeps looping to n_epochs; trim to the last mistake
    bound, R, gamma = novikoff(X, y, np.array([0.5, 1.0]), -0.7)      # margin of the TRUE rule: a valid (not the tightest) separator
    print(f"  separable loans: n={len(X)}  total updates = {len(H)}  Novikoff bound (R/gamma)^2 = {bound:.1f}  (R={R:.2f}, margin={gamma:.3f})")
    assert len(H) <= bound + 1e-9, "perceptron convergence theorem violated?!"
    print(f"  final training accuracy = {np.mean(p.predict(X) == y):.3f}")
    xlim, ylim = (-2.4, 2.4), (-2.4, 2.4)
    Wpath = np.array([[0.0, 0.0]] + [h["w"] for h in H])
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Perceptron: only a mistake changes the weights, and each mistake drags the line toward the missed point")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.27)
    steps = [None] + list(range(len(H)))
    with Animator(here(__file__, "perceptron_learning.gif"), fps=3, hold=3) as anim:
        for s in steps:
            k = 0 if s is None else s + 1
            w, b = (np.zeros(2), 0.0) if s is None else (H[s]["w"], H[s]["b"])
            a1.clear()
            scatter_classes(a1, X, (y > 0).astype(int), s=26)
            if s is not None:
                i = H[s]["i"]
                prev_w, prev_b = (np.zeros(2), 0.0) if s == 0 else (H[s - 1]["w"], H[s - 1]["b"])
                draw_line(a1, prev_w, prev_b, xlim, ylim, color=MUTED, linestyle="--", linewidth=1.2)
                a1.scatter(*X[i], s=170, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=6)
                draw_line(a1, w, b, xlim, ylim, color=INK, linewidth=2)
            else:
                a1.set_xlim(*xlim); a1.set_ylim(*ylim)
            a1.set(title="Data space (ringed = latest mistake)", xlabel="credit score (standardised)", ylabel="income (standardised)")
            a1.text(0.02, 0.97, "orange = approve, blue = decline, dashed = old line", transform=a1.transAxes, fontsize=8, color=INK2, va="top")
            a2.clear()
            a2.plot(Wpath[: k + 1, 0], Wpath[: k + 1, 1], "-o", color=BLUE, markersize=3)
            a2.scatter(*Wpath[k], s=55, color=ORANGE, edgecolor=INK, zorder=5)
            if s is not None:
                a2.annotate("", xy=Wpath[k], xytext=Wpath[k - 1], arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))
            a2.axhline(0, color=GRID); a2.axvline(0, color=GRID)
            a2.set(title="Weight space: w jumps by y·x", xlabel="w₁", ylabel="w₂")
            a2.set_xlim(Wpath[:, 0].min() - 0.6, Wpath[:, 0].max() + 0.6); a2.set_ylim(Wpath[:, 1].min() - 0.6, Wpath[:, 1].max() + 0.6)
            a3.clear()
            a3.step(range(k + 1), [0] + list(range(1, k + 1))[:k], where="post", color=BLUE)
            a3.text(0.04, 0.95, f"Novikoff guarantee: at most (R/γ)² = {bound:.0f} mistakes\n(far above the {len(H)} actually needed: the bound is loose)", transform=a3.transAxes, color=RED, fontsize=8.2, va="top")
            a3.set(title="Cumulative mistakes", xlabel="update number", ylabel="mistakes so far", xlim=(0, len(H) + 1), ylim=(0, len(H) * 1.15))
            acc = np.mean(np.where(X @ w + b > 0, 1, -1) == y)
            caption(fig, f"update {k:>2}: w=({w[0]:+.2f}, {w[1]:+.2f}) b={b:+.1f}   training accuracy {acc:.2f}" + ("" if s is not None else "   (start: w=0, every point is a mistake)"))
            anim.grab(fig)


def make_xor_points(n=120, seed=4):
    r = rng_(seed)
    X = r.uniform(-1.3, 1.3, (n, 2))
    X = X[np.min(np.abs(X), 1) > 0.12][:100]                     # open a gap around the axes
    return X, np.where(X[:, 0] * X[:, 1] > 0, 1, -1)


def xor_animation():
    X, y = make_xor_points()
    Xl = np.c_[X, X[:, 0] * X[:, 1]]                             # ONE extra feature: the product x1*x2
    E = 40
    p = Perceptron(n_epochs=E, shuffle=True, seed=1)
    fits = []
    for e in range(1, E + 1):                                    # refit with 1..E epochs to expose the state after each epoch
        q = Perceptron(n_epochs=e, shuffle=True, seed=1).fit(X, y, record=True)
        fits.append(q)
    lift_mist = Perceptron(n_epochs=E, shuffle=True, seed=1).fit(Xl, y, record=True).epoch_mistakes
    plain_mist = fits[-1].epoch_mistakes
    acc = {k: [np.mean(q.predict(X, k) == y) for q in fits] for k in ("last", "pocket", "avg")}
    print(f"  XOR: plain perceptron training accuracy after epoch {E}: last={acc['last'][-1]:.2f} pocket={acc['pocket'][-1]:.2f} averaged={acc['avg'][-1]:.2f}  (a line cannot beat 0.75)")
    print(f"  XOR with extra feature x1*x2: mistakes per epoch -> {lift_mist[:6]} ... {lift_mist[-3:]}")
    xlim = ylim = (-1.5, 1.5)
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="XOR: no single line works, so the perceptron never converges - until a hidden feature does the bending")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.27)
    with Animator(here(__file__, "perceptron_xor.gif"), fps=6, hold=3) as anim:
        for e in range(E):
            q = fits[e]
            a1.clear()
            scatter_classes(a1, X, (y > 0).astype(int), s=22)
            draw_line(a1, q.w, q.b, xlim, ylim, color=INK, linewidth=1.8)
            draw_line(a1, q.w_pocket, q.b_pocket, xlim, ylim, color=GREEN, linewidth=1.6, linestyle="--")
            draw_line(a1, q.w_avg, q.b_avg, xlim, ylim, color=VIOLET, linewidth=1.6, linestyle=":")
            a1.text(0.02, 0.03, "black = current, green dashed = pocket, violet dotted = averaged", transform=a1.transAxes, fontsize=7.8, color=INK2)
            a1.set(title="Data space (XOR pattern)", xlabel="x₁", ylabel="x₂")
            a2.clear()
            a2.plot(range(1, e + 2), acc["last"][: e + 1], color=INK, label="current")
            a2.plot(range(1, e + 2), acc["pocket"][: e + 1], color=GREEN, label="pocket")
            a2.plot(range(1, e + 2), acc["avg"][: e + 1], color=VIOLET, label="averaged")
            a2.axhline(0.75, color=RED, linestyle="--", linewidth=1)
            a2.text(E, 0.755, "best any line can do on XOR ≈ 0.75", color=RED, fontsize=8, ha="right")
            a2.set(title="Training accuracy per epoch (plain features)", xlabel="epoch", ylabel="accuracy", xlim=(1, E), ylim=(0.3, 1.02)); a2.legend(loc="lower left", ncol=3)
            a3.clear()
            a3.plot(range(1, e + 2), plain_mist[: e + 1], color=INK); a3.plot(range(1, e + 2), lift_mist[: e + 1], color=ORANGE)
            a3.text(E, max(plain_mist) * 0.55, "plain (x₁, x₂)", color=INK, ha="right", fontsize=8.5); a3.text(E, max(plain_mist) * 0.12, "with extra feature x₁·x₂", color=ORANGE, ha="right", fontsize=8.5)
            a3.set(title="Mistakes per epoch", xlabel="epoch", ylabel="mistakes", xlim=(1, E), ylim=(0, max(plain_mist) * 1.1))
            caption(fig, f"epoch {e + 1:>2}: plain perceptron still makes {plain_mist[e]} mistakes; with the x₁·x₂ feature it makes {lift_mist[e]}")
            anim.grab(fig)


def main():
    learning_animation()
    xor_animation()
    X, y = loan_data(); Xt, yt = loan_data(300, seed=11)
    p = Perceptron(n_epochs=100).fit(X, y)
    print(f"  loans: test accuracy {np.mean(p.predict(Xt) == yt):.3f} (train {np.mean(p.predict(X) == y):.3f}); epochs until no mistakes = {len(p.epoch_mistakes)}")
    try:
        from sklearn.linear_model import Perceptron as SP
        s = SP(max_iter=1000, tol=None, shuffle=False).fit(X, y)
        print(f"  [sklearn] Perceptron training accuracy {s.score(X, y):.3f} (ours {np.mean(p.predict(X) == y):.3f})")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
