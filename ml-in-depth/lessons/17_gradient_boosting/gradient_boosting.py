"""
Lesson 17 - Gradient Boosting (from scratch, trees from cart.py)

Problem story: a power company forecasts electricity demand from the outdoor temperature (a smooth, curved, noisy
relationship); later it classifies "high-risk" vs "normal" hours from two signals.  Gradient boosting fits a model in
STAGES: each new small tree is trained on the *errors still left* by the ensemble so far.

   F_0(x) = best constant
   F_m(x) = F_{m-1}(x) + nu * h_m(x),     h_m = tree fitted to the negative gradient of the loss at F_{m-1}

   squared loss   : negative gradient = residual  y - F          (leaf value = mean residual)
   logistic loss  : negative gradient = y - p, p = sigmoid(F)    (leaf value = Newton step  sum(y-p) / sum p(1-p))

Outputs
  gb_regression.gif   - stage by stage: prediction curve, residuals + the new tree, train/test error
  gb_lr_depth.gif     - learning rate and tree depth compared side by side over 300 stages
  gb_classification.gif - logistic-loss boosting on a two-moons problem

Run:  python gradient_boosting.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.utils import *  # noqa: F401,F403
from cart import DecisionTree


class GradientBoosting:
    def __init__(self, loss="squared", n_stages=100, lr=0.1, max_depth=2, min_samples_leaf=2, subsample=1.0, seed=0):
        self.loss, self.n_stages, self.lr, self.max_depth = loss, n_stages, lr, max_depth
        self.min_samples_leaf, self.subsample, self.seed = min_samples_leaf, subsample, seed

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        r = np.random.default_rng(self.seed)
        n = len(y)
        if self.loss == "squared":
            self.F0 = y.mean()
        else:                                                    # logistic: start from the log-odds of the base rate
            p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6); self.F0 = np.log(p0 / (1 - p0))
        F = np.full(n, self.F0)
        self.trees, self.pseudo, self.train_F = [], [], []
        for m in range(self.n_stages):
            if self.loss == "squared":
                g = y - F                                        # negative gradient of 1/2 (y-F)^2
            else:
                p = sigmoid(F); g = y - p                        # negative gradient of the log-loss w.r.t. F
            idx = np.arange(n) if self.subsample >= 1 else r.choice(n, int(self.subsample * n), replace=False)   # stochastic boosting
            tree = DecisionTree(task="regression", max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf).fit(X[idx], g[idx])
            if self.loss == "logistic":                          # replace leaf means by one Newton step: sum(g) / sum(p(1-p))
                leaf = tree.apply(X[idx]); ph = sigmoid(F[idx]); hess = ph * (1 - ph)
                for L in np.unique(leaf):
                    mk = leaf == L
                    tree.nodes[L]["value"] = float(np.clip(g[idx][mk].sum() / max(hess[mk].sum(), 1e-9), -6, 6))
                tree._arr = None
            self.trees.append(tree); self.pseudo.append(g.copy())
            F = F + self.lr * tree.predict(X)
        return self

    def decision_function(self, X, n_stages=None):
        F = np.full(len(X), self.F0)
        for t in self.trees[: n_stages if n_stages is not None else len(self.trees)]:
            F = F + self.lr * t.predict(X)
        return F

    def staged(self, X):
        F = np.full(len(X), self.F0)
        yield F.copy()
        for t in self.trees:
            F = F + self.lr * t.predict(X)
            yield F.copy()

    def predict(self, X, n_stages=None):
        F = self.decision_function(X, n_stages)
        return F if self.loss == "squared" else (F > 0).astype(int)

    def predict_proba(self, X, n_stages=None):
        return sigmoid(self.decision_function(X, n_stages))


def true_demand(x):
    return np.sin(1.3 * x) + 0.12 * x * np.cos(0.5 * x) + 0.4 * (x > 6)


def demand_data(n, seed, noise=0.3):
    r = rng_(seed)
    x = np.sort(r.uniform(0, 10, n))
    return x[:, None], true_demand(x) + r.normal(0, noise, n)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


# ------------------------------------------------------------------------------------------- GIF 1
def regression_gif():
    X, y = demand_data(90, 0); Xt, yt = demand_data(500, 1)
    gb = GradientBoosting("squared", n_stages=100, lr=0.3, max_depth=2).fit(X, y)
    tr = [mse(y, F) for F in gb.staged(X)]; te = [mse(yt, F) for F in gb.staged(Xt)]
    xs = np.linspace(0, 10, 400)[:, None]
    stage_F = list(gb.staged(xs))
    ib = int(np.argmin(te))
    print(f"  squared-loss GB (lr 0.3, depth 2): stage 0 test MSE {te[0]:.3f} -> best {te[ib]:.3f} at stage {ib}; stage 100: train {tr[-1]:.3f}, test {te[-1]:.3f}  (noise floor 0.090)")
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Gradient boosting: every small tree fits what the ensemble still gets wrong")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    frames = [0, 1, 2, 3, 4, 6, 8, 12, 18, 28, 45, 70, 100]
    Fm = list(gb.staged(X))
    with Animator(here(__file__, "gb_regression.gif"), fps=2.5, hold=3.5) as anim:
        for m in frames:
            a1.clear()
            a1.plot(xs[:, 0], true_demand(xs[:, 0]), color=MUTED, linestyle="--", linewidth=1.2)
            a1.scatter(X[:, 0], y, s=14, color=BLUE, zorder=3)
            a1.plot(xs[:, 0], stage_F[m], color=ORANGE, linewidth=2.2, zorder=4)
            a1.set(title=f"Ensemble after {m} trees", xlabel="temperature (°C, scaled)", ylabel="electricity demand")
            a1.set_ylim(-2.2, 2.6)
            a2.clear()
            if m == 0:
                a2.scatter(X[:, 0], y - gb.F0, s=14, color=VIOLET); a2.axhline(0, color=MUTED)
                a2.set(title="Residuals of the constant")
            else:
                res = gb.pseudo[m - 1]
                a2.scatter(X[:, 0], res, s=14, color=VIOLET, zorder=3); a2.axhline(0, color=MUTED, linewidth=0.8)
                a2.plot(xs[:, 0], gb.trees[m - 1].predict(xs), color=GREEN, linewidth=2.2, drawstyle="steps-mid", zorder=4)
                a2.set(title=f"Residuals + tree {m} (green)")
            a2.set(xlabel="temperature (°C, scaled)", ylabel="residual", ylim=(-1.9, 1.9))
            a3.clear()
            a3.plot(range(m + 1), tr[: m + 1], color=BLUE); a3.plot(range(m + 1), te[: m + 1], color=ORANGE)
            a3.axhline(0.09, color=MUTED, linestyle="--", linewidth=1); a3.text(100, 0.095, "noise floor", ha="right", va="bottom", color=INK2, fontsize=8.5)
            a3.text(100, tr[-1] * 0.8, "train", color=BLUE, ha="right", va="top", fontsize=8.5); a3.text(100, te[-1] * 1.06, "test", color=ORANGE, ha="right", fontsize=8.5)
            a3.set(title="Mean squared error", xlabel="number of trees", ylabel="MSE", xlim=(0, 100), ylim=(0.0, 1.0))
            caption(fig, f"{m:>3} trees: train MSE {tr[m]:.3f}, test MSE {te[m]:.3f}" + ("   (stage 0 = the constant mean)" if m == 0 else f"   (each tree has 4 leaves and is added with step 0.3)"))
            anim.grab(fig)


# ------------------------------------------------------------------------------------------- GIF 2
def lr_depth_gif():
    X, y = demand_data(90, 0); Xt, yt = demand_data(500, 1)
    configs = [("lr = 1.0, depth 2", 1.0, 2, ORANGE), ("lr = 0.1, depth 2", 0.1, 2, BLUE), ("lr = 0.1, stumps (depth 1)", 0.1, 1, GREEN), ("lr = 0.1, depth 5", 0.1, 5, VIOLET)]
    S = 300
    xs = np.linspace(0, 10, 300)[:, None]
    runs = []
    for name, lr, d, col in configs:
        gb = GradientBoosting("squared", S, lr, d).fit(X, y)
        te = [mse(yt, F) for F in gb.staged(Xt)]; tr = [mse(y, F) for F in gb.staged(X)]
        runs.append((name, col, list(gb.staged(xs)), tr, te))
        print(f"  {name:28s}: best test MSE {min(te):.3f} at stage {int(np.argmin(te)):3d};  at stage {S}: train {tr[-1]:.3f}, test {te[-1]:.3f}")
    fig = plt.figure(figsize=(13, 5.4))
    fig.suptitle("Learning rate and tree depth: how fast and how flexibly the ensemble fits", fontsize=13, fontweight="bold", x=0.02, ha="left")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.8], left=0.05, right=0.985, bottom=0.13, top=0.87, hspace=0.45, wspace=0.25)
    top = [fig.add_subplot(gs[0, i]) for i in range(4)]
    bot = fig.add_subplot(gs[1, :])
    frames = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 160, 230, 300]
    with Animator(here(__file__, "gb_lr_depth.gif"), fps=2.5, hold=3.5) as anim:
        for m in frames:
            for ax, (name, col, curves, tr, te) in zip(top, runs):
                ax.clear()
                ax.plot(xs[:, 0], true_demand(xs[:, 0]), color=MUTED, linestyle="--", linewidth=1)
                ax.scatter(X[:, 0], y, s=8, color=MUTED, zorder=2)
                ax.plot(xs[:, 0], curves[m], color=col, linewidth=2, zorder=4)
                ax.set(title=name, ylim=(-2.2, 2.6), xticks=[], yticks=[])
                ax.text(0.04, 0.05, f"test MSE {te[m]:.3f}", transform=ax.transAxes, fontsize=8.5, color=INK2)
            bot.clear()
            for name, col, curves, tr, te in runs:
                bot.plot(range(m + 1), te[: m + 1], color=col, label=name)
            bot.axhline(0.09, color=MUTED, linestyle="--", linewidth=1)
            bot.set(title="Test MSE vs number of trees", xlabel="number of trees", ylabel="MSE", xlim=(0, S), ylim=(0.0, 0.8)); bot.legend(loc="upper right", ncol=2)
            caption(fig, f"{m:>3} trees. Large steps (orange) and deep trees (violet) fit fast and then overfit; small steps with shallow trees are slower but reach lower test error")
            anim.grab(fig)


# ------------------------------------------------------------------------------------------- GIF 3
def classification_gif():
    X, y = make_moons(250, noise=0.28, seed=2); Xt, yt = make_moons(2000, noise=0.28, seed=3)
    gb = GradientBoosting("logistic", n_stages=150, lr=0.3, max_depth=2).fit(X, y)
    def logloss(yy, p):
        p = np.clip(p, 1e-9, 1 - 1e-9); return float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
    stF, stT = list(gb.staged(X)), list(gb.staged(Xt))
    ltr = [logloss(y, sigmoid(F)) for F in stF]; lte = [logloss(yt, sigmoid(F)) for F in stT]
    atr = [accuracy(y, (F > 0).astype(int)) for F in stF]; ate = [accuracy(yt, (F > 0).astype(int)) for F in stT]
    ib = int(np.argmin(lte))
    print(f"  logistic-loss GB (moons): stage 150 train acc {atr[-1]:.3f}, test acc {ate[-1]:.3f}; best test log-loss {lte[ib]:.3f} at stage {ib}, ends at {lte[-1]:.3f}")
    gx, gy, P = grid_for(X, res=110, pad=0.4)
    gridF = list(gb.staged(P))
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Boosting for classification: trees fit the probability errors y − p, leaves take Newton steps")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    frames = [0, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 150]
    with Animator(here(__file__, "gb_classification.gif"), fps=2.5, hold=3.5) as anim:
        for m in frames:
            a1.clear()
            probability_map(a1, lambda Q: sigmoid(gridF[m]), X, res=110, pad=0.4, alpha=0.4)
            scatter_classes(a1, X, y, s=13)
            a1.set(title=f"P(high-risk) after {m} trees", xlabel="signal 1", ylabel="signal 2")
            a2.clear()
            p = sigmoid(stF[m]); err = y - p
            a2.scatter(X[:, 0], X[:, 1], s=8 + 220 * np.abs(err) ** 1.5, c=[CAT[int(c)] for c in y], alpha=0.8, edgecolor=SURFACE, linewidth=0.5)
            a2.set(title="Remaining error |y − p| (marker size)", xlabel="signal 1", ylabel="signal 2", xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
            a3.clear()
            a3.plot(range(m + 1), ltr[: m + 1], color=BLUE); a3.plot(range(m + 1), lte[: m + 1], color=ORANGE)
            a3.text(150, ltr[-1] * 0.8, "train log-loss", color=BLUE, ha="right", va="top", fontsize=8.5); a3.text(150, lte[-1] * 1.05, "test log-loss", color=ORANGE, ha="right", fontsize=8.5)
            a3.set(title="Log-loss", xlabel="number of trees", ylabel="log-loss", xlim=(0, 150), ylim=(0, 0.75))
            caption(fig, f"{m:>3} trees: train log-loss {ltr[m]:.3f} / acc {atr[m]:.3f};  test log-loss {lte[m]:.3f} / acc {ate[m]:.3f}")
            anim.grab(fig)


def main():
    regression_gif()
    lr_depth_gif()
    classification_gif()
    # --- sub-sampling (stochastic gradient boosting) and cross-check ----------------------------------------------------
    X, y = demand_data(90, 0); Xt, yt = demand_data(500, 1)
    for sub in (1.0, 0.5):
        gb = GradientBoosting("squared", 200, 0.1, 2, subsample=sub, seed=1).fit(X, y)
        print(f"  subsample={sub}: test MSE at 200 trees {mse(yt, gb.predict(Xt)):.3f}")
    try:
        from sklearn.ensemble import GradientBoostingRegressor as SK
        sk = SK(n_estimators=100, learning_rate=0.3, max_depth=2, min_samples_leaf=2, random_state=0).fit(X, y)
        mine = GradientBoosting("squared", 100, 0.3, 2).fit(X, y)
        print(f"  [sklearn] test MSE {mse(yt, sk.predict(Xt)):.3f} vs ours {mse(yt, mine.predict(Xt)):.3f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
