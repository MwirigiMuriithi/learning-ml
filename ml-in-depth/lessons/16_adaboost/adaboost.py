"""
Lesson 16 - AdaBoost (from scratch, with decision stumps)

Problem story: a camera inspects parts on a conveyor belt.  Two measurements per part; defective parts form a ring around
the good ones, so NO single threshold can separate them.  AdaBoost builds a strong classifier from hundreds of
hopelessly weak rules ("is measurement 1 above 0.3?") by making each new rule concentrate on the parts the previous
rules got wrong.

Implemented: weighted decision stump, AdaBoost.M1 / discrete AdaBoost (labels +-1), exponential-loss tracking,
staged predictions, label-noise experiment.  (Multi-class SAMME is explained in lesson.md.)

Output: adaboost.gif            Run:  python adaboost.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


class Stump:
    """h(x) = s * sign(x_j - thr):  one feature, one threshold, one polarity s in {+1, -1}.  Minimises WEIGHTED error."""

    def fit(self, X, y, w):
        n, d = X.shape
        best = (np.inf, 0, 0.0, 1)
        for j in range(d):
            order = np.argsort(X[:, j])
            xs, ys, ws = X[order, j], y[order], w[order]
            # cut after position i: predict -1 on the left, +1 on the right (polarity +1)
            # error = weight of (+1 labels on the left) + weight of (-1 labels on the right)
            wp = np.cumsum(ws * (ys > 0)); wn = np.cumsum(ws * (ys < 0))
            err_pos = wp[:-1] + (wn[-1] - wn[:-1])
            valid = xs[:-1] != xs[1:]
            for pol, err in ((1, err_pos), (-1, w.sum() - err_pos)):     # polarity -1 flips every prediction
                e = np.where(valid, err, np.inf)
                i = int(np.argmin(e))
                if e[i] < best[0]:
                    best = (float(e[i]), j, float((xs[i] + xs[i + 1]) / 2), pol)
        self.err, self.j, self.thr, self.pol = best
        return self

    def predict(self, X):
        return self.pol * np.where(X[:, self.j] > self.thr, 1, -1)


class AdaBoost:
    """F_T(x) = sum_t alpha_t h_t(x),   prediction = sign(F_T).
        alpha_t = 1/2 ln((1 - err_t) / err_t)          (accurate stumps get a bigger say)
        w_i <- w_i exp(-alpha_t y_i h_t(x_i)) / Z_t    (mistakes get heavier, successes lighter)"""

    def __init__(self, n_rounds=50):
        self.n_rounds = n_rounds

    def fit(self, X, y):
        n = len(y)
        w = np.full(n, 1 / n)
        self.stumps, self.alphas, self.weights_before, self.errs = [], [], [], []
        for t in range(self.n_rounds):
            h = Stump().fit(X, y, w)
            err = np.clip(h.err / w.sum(), 1e-12, 1 - 1e-12)
            if err >= 0.5:                                          # no better than chance: stop
                break
            alpha = 0.5 * np.log((1 - err) / err)
            self.weights_before.append(w.copy())
            self.stumps.append(h); self.alphas.append(alpha); self.errs.append(err)
            w = w * np.exp(-alpha * y * h.predict(X))
            w /= w.sum()
        self.final_weights = w
        return self

    def decision_function(self, X, n_stumps=None):
        m = n_stumps or len(self.stumps)
        return sum(a * h.predict(X) for a, h in zip(self.alphas[:m], self.stumps[:m]))

    def staged_decision(self, X):
        F = np.zeros(len(X))
        for a, h in zip(self.alphas, self.stumps):
            F = F + a * h.predict(X)
            yield F.copy()

    def predict(self, X, n_stumps=None):
        return np.where(self.decision_function(X, n_stumps) >= 0, 1, -1)


def parts_data(n, seed, flip=0.0):
    X, y = make_circles(n, noise=0.13, factor=0.5, seed=seed)
    y = np.where(y == 1, 1, -1)
    if flip:
        y = np.where(rng_(seed + 7).random(n) < flip, -y, y)
    return X, y


def main():
    X, y = parts_data(240, 0); Xt, yt = parts_data(2000, 1)
    T = 40
    ab = AdaBoost(T).fit(X, y)
    stag_tr = [np.mean(np.where(F >= 0, 1, -1) != y) for F in ab.staged_decision(X)]
    stag_te = [np.mean(np.where(F >= 0, 1, -1) != yt) for F in ab.staged_decision(Xt)]
    expl = [np.mean(np.exp(-y * F)) for F in ab.staged_decision(X)]
    print(f"  AdaBoost with {len(ab.stumps)} stumps: train error {stag_tr[-1]:.3f}, test error {stag_te[-1]:.3f};  best single stump error {min(h.err for h in [Stump().fit(X, y, np.full(len(y), 1 / len(y)))]):.3f}")
    # the exponential loss upper-bounds the training error: prod_t Z_t = (1/n) sum exp(-y F); training error <= that
    assert all(e >= s - 1e-12 for e, s in zip(expl, stag_tr)), "exp loss must bound the 0-1 training error"
    Z = np.prod([2 * np.sqrt(e * (1 - e)) for e in ab.errs])
    print(f"  theory: training error <= prod_t 2*sqrt(err_t (1-err_t)) = {Z:.4f} ; exponential loss after T rounds = {expl[-1]:.4f} (equal)")

    gx, gy, P = grid_for(X, res=130, pad=0.4)
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="AdaBoost: each weak stump is trained on re-weighted data, so it targets the previous mistakes")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    rounds = [1, 2, 3, 4, 5, 6, 8, 10, 13, 17, 22, 28, 34, 40]
    sizes = lambda w: 6 + np.minimum(2500 * w, 160)
    with Animator(here(__file__, "adaboost.gif"), fps=2.5, hold=3.5) as anim:
        for m in rounds:
            t = m - 1
            h = ab.stumps[t]; wb = ab.weights_before[t]
            pred = h.predict(X); miss = pred != y
            a1.clear()
            zz = h.predict(P).reshape(gx.shape)
            a1.contourf(gx, gy, zz, levels=[-1.5, 0, 1.5], colors=[BLUE, ORANGE], alpha=0.16)
            for c, col in ((-1, BLUE), (1, ORANGE)):
                k = (y == c)
                a1.scatter(X[k, 0], X[k, 1], s=sizes(wb[k]), color=col, edgecolor=SURFACE, linewidth=0.6, zorder=3)
            a1.scatter(X[miss, 0], X[miss, 1], marker="x", s=24, color=INK, linewidth=0.9, zorder=5)
            a1.set(title=f"Round {m}: stump {'x₁' if h.j == 0 else 'x₂'} {'>' if h.pol > 0 else '<'} {h.thr:.2f}", xlabel="measurement 1", ylabel="measurement 2")
            a1.text(0.02, 0.03, "marker size = sample weight, × = misclassified by this stump", transform=a1.transAxes, fontsize=8, color=INK2)
            a2.clear()
            F = ab.decision_function(P, m).reshape(gx.shape)
            a2.contourf(gx, gy, np.sign(F + 1e-12), levels=[-1.5, 0, 1.5], colors=[BLUE, ORANGE], alpha=0.2)
            wa = ab.weights_before[t + 1] if t + 1 < len(ab.weights_before) else ab.final_weights
            for c, col in ((-1, BLUE), (1, ORANGE)):
                k = (y == c)
                a2.scatter(X[k, 0], X[k, 1], s=sizes(wa[k]), color=col, edgecolor=SURFACE, linewidth=0.6, zorder=3)
            a2.set(title=f"Weighted vote of first {m} stumps", xlabel="measurement 1", ylabel="measurement 2")
            a3.clear()
            a3.plot(range(1, m + 1), stag_tr[:m], color=BLUE); a3.plot(range(1, m + 1), stag_te[:m], color=ORANGE)
            a3.plot(range(1, m + 1), np.minimum(expl[:m], 1.0), color=MUTED, linestyle="--")
            a3.text(T, 0.97, "exp. loss (bound)", color=MUTED, ha="right", fontsize=8.5); a3.text(T, stag_tr[-1] + 0.025, "train error", color=BLUE, ha="right", fontsize=8.5); a3.text(T, stag_te[-1] - 0.04, "test error", color=ORANGE, ha="right", va="top", fontsize=8.5)
            a3.set(title="Error vs rounds", xlabel="rounds", ylabel="error", xlim=(1, T), ylim=(0, 1.0))
            caption(fig, f"round {m:>2}: this stump's weighted error {ab.errs[t]:.3f} → vote strength α = {ab.alphas[t]:.2f};  ensemble train error {stag_tr[t]:.3f}, test error {stag_te[t]:.3f}")
            anim.grab(fig)

    # --- label noise: AdaBoost keeps up-weighting mislabelled points ---------------------------------------------------
    Xn, yn = parts_data(240, 0, flip=0.10); Xnt, ynt = parts_data(2000, 1)             # 10% of TRAINING labels flipped, test clean
    abn = AdaBoost(300).fit(Xn, yn)
    tr = [np.mean(np.where(F >= 0, 1, -1) != yn) for F in abn.staged_decision(Xn)]
    te = [np.mean(np.where(F >= 0, 1, -1) != ynt) for F in abn.staged_decision(Xnt)]
    wmax = abn.final_weights.max() * len(yn)
    flipped = rng_(7).random(240) < 0.10
    print(f"  with 10% label noise: round 10: train err {tr[9]:.3f}, test err {te[9]:.3f} | round 300: train err {tr[-1]:.3f}, test err {te[-1]:.3f}; best test err {min(te):.3f} at round {int(np.argmin(te)) + 1}")
    print(f"  sample weights after 300 rounds: the heaviest point carries {wmax:.0f}x the average weight")
    # --- compare with a deeper base learner count: single stump vs ensemble
    try:
        from sklearn.ensemble import AdaBoostClassifier
        sk = AdaBoostClassifier(n_estimators=T, random_state=0).fit(X, y)
        print(f"  [sklearn] AdaBoost({T}) test error {1 - sk.score(Xt, yt):.3f} vs ours {stag_te[-1]:.3f}")
    except Exception as e:                                           # optional cross-check only
        print(f"  [sklearn check skipped: {type(e).__name__}]")


if __name__ == "__main__":
    main()
