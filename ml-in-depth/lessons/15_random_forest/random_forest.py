"""
Lesson 15 - Random Forest (bagging + random feature subsets), built on our own CART (cart.py)

Problem story: a payments company separates fraudulent from normal transactions using two engineered signals.  A single deep
tree memorises noise and its boundary changes with every sample of data; a forest of many such trees, each trained on a
different bootstrap sample and a random feature subset, votes - and the noise averages out.

Implemented: bootstrap aggregation, random feature subset per split (max_features), out-of-bag (OOB) error, vote averaging,
impurity & permutation feature importance, variance-reduction experiment (single tree vs forest over many resampled datasets).

Outputs: forest_growth.gif, forest_vs_trees.gif       Run:  python random_forest.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.utils import *  # noqa: F401,F403
from cart import DecisionTree


class RandomForest:
    def __init__(self, n_trees=100, max_features="sqrt", max_depth=None, min_samples_leaf=1, bootstrap=True, seed=0):
        self.n_trees, self.max_features, self.max_depth = n_trees, max_features, max_depth
        self.min_samples_leaf, self.bootstrap, self.seed = min_samples_leaf, bootstrap, seed

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, int)
        n = len(y)
        self.K = int(y.max()) + 1
        r = np.random.default_rng(self.seed)
        self.trees, self.inbag = [], np.zeros((self.n_trees, n), bool)
        for b in range(self.n_trees):
            idx = r.integers(0, n, n) if self.bootstrap else np.arange(n)          # sample n rows WITH replacement
            self.inbag[b, idx] = True
            t = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                             seed=int(r.integers(1 << 30)), n_classes=self.K).fit(X[idx], y[idx])
            self.trees.append(t)
        # out-of-bag predictions: each row is predicted only by the trees that never saw it
        self.per_tree_train = np.array([t.predict_proba(X) for t in self.trees])    # (T, n, K)
        self.X_, self.y_ = X, y
        return self

    def tree_proba(self, X):
        return np.array([t.predict_proba(X) for t in self.trees])                   # (T, m, K)

    def predict_proba(self, X, n_trees=None):
        return self.tree_proba(X)[: n_trees or self.n_trees].mean(0)

    def predict(self, X, n_trees=None):
        return np.argmax(self.predict_proba(X, n_trees), 1)

    def oob_curve(self):
        """OOB accuracy using the first m trees, for m = 1..T (only rows that are OOB for >= 1 of those trees count)."""
        mask = ~self.inbag                                                           # (T, n) True if row is OOB for tree
        sums = np.cumsum(self.per_tree_train * mask[:, :, None], 0)
        cnt = np.cumsum(mask, 0)
        acc = []
        for m in range(self.n_trees):
            ok = cnt[m] > 0
            acc.append(np.mean(np.argmax(sums[m][ok], 1) == self.y_[ok]) if ok.any() else np.nan)
        return np.array(acc)

    def oob_accuracy(self):
        return float(self.oob_curve()[-1])

    def feature_importances(self):
        d = self.X_.shape[1]
        return np.mean([t.feature_importances(d) for t in self.trees], 0)

    def permutation_importance(self, X, y, n_repeats=5, seed=0):
        """Drop in accuracy when one feature is shuffled (breaking its link to y) - measured on the data you pass in."""
        r = np.random.default_rng(seed)
        base = accuracy(y, self.predict(X))
        out = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy(); Xp[:, j] = r.permutation(Xp[:, j])
                drops.append(base - accuracy(y, self.predict(Xp)))
            out[j] = np.mean(drops)
        return out


def fraud_data(n, seed):
    X, y = make_moons(n, noise=0.32, seed=seed)
    return X, y


def growth_gif():
    X, y = fraud_data(300, 0); Xt, yt = fraud_data(2000, 1)
    T = 80
    rf = RandomForest(n_trees=T, max_features=1, seed=3).fit(X, y)               # max_features=1 of 2: random feature per split
    gx, gy, P = grid_for(X, res=110, pad=0.5)
    grid_t = rf.tree_proba(P)[:, :, 1]                                           # (T, m) P(fraud) of every tree on the map
    test_t = rf.tree_proba(Xt)
    test_acc_each = np.array([accuracy(yt, np.argmax(p, 1)) for p in test_t])
    cum = np.cumsum(test_t, 0)
    test_curve = np.array([accuracy(yt, np.argmax(cum[m], 1)) for m in range(T)])
    oob = rf.oob_curve()
    print(f"  forest of {T} trees: test acc {test_curve[-1]:.3f}, OOB acc {oob[-1]:.3f};  single deep trees: mean test acc {test_acc_each.mean():.3f} "
          f"(min {test_acc_each.min():.3f}, max {test_acc_each.max():.3f})")
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Random forest: many different, noisy trees vote; the vote is smoother and more accurate than any one tree")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    steps = [1, 2, 3, 4, 6, 8, 11, 15, 20, 27, 35, 45, 60, 80]
    with Animator(here(__file__, "forest_growth.gif"), fps=2.5, hold=3.5) as anim:
        for m in steps:
            k = m - 1                                                            # index of the newest tree
            a1.clear()
            pm = grid_t[:m].mean(0).reshape(gx.shape)
            probability_map(a1, lambda Q: pm.ravel(), X, res=110, pad=0.5, alpha=0.4)   # (re-evaluates a trivial lambda on the same grid)
            scatter_classes(a1, X, y, s=12)
            a1.set(title=f"Forest vote of {m} tree(s): P(fraud)", xlabel="signal 1", ylabel="signal 2")
            a2.clear()
            z = grid_t[k].reshape(gx.shape)
            a2.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], colors=[BLUE, ORANGE], alpha=0.2)
            ib = rf.inbag[k]
            a2.scatter(X[ib, 0], X[ib, 1], s=12, c=[CAT[int(c)] for c in y[ib]], zorder=3)
            a2.scatter(X[~ib, 0], X[~ib, 1], s=26, facecolor="none", edgecolor=[CAT[int(c)] for c in y[~ib]], linewidth=1.1, zorder=3)
            a2.set(title=f"Tree #{m} alone (hollow = out-of-bag)", xlabel="signal 1", ylabel="signal 2", xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
            a3.clear()
            a3.scatter(range(1, m + 1), test_acc_each[:m], s=9, color=MUTED, label="single tree")
            a3.plot(range(1, m + 1), test_curve[:m], color=ORANGE, label="forest (test)")
            a3.plot(range(1, m + 1), oob[:m], color=BLUE, linestyle="--", label="forest (OOB estimate)")
            a3.set(title="Test accuracy vs number of trees", xlabel="number of trees", ylabel="accuracy", xlim=(0, T + 1), ylim=(0.65, 0.95)); a3.legend(loc="lower right")
            caption(fig, f"{m:>2} trees: forest test accuracy {test_curve[k]:.3f}, OOB accuracy {oob[k]:.3f};  this tree alone {test_acc_each[k]:.3f}  (avg single tree {test_acc_each.mean():.3f})")
            anim.grab(fig)
    return rf


def spaghetti_gif():
    X, y = fraud_data(300, 0); Xt, yt = fraud_data(2000, 1)
    T = 24
    rf = RandomForest(n_trees=T, max_features=1, seed=11).fit(X, y)
    gx, gy, P = grid_for(X, res=110, pad=0.5)
    grid_t = rf.tree_proba(P)[:, :, 1]
    test_t = rf.tree_proba(Xt); each = np.array([accuracy(yt, np.argmax(p, 1)) for p in test_t])
    cum = np.cumsum(test_t, 0); ens = np.array([accuracy(yt, np.argmax(cum[m], 1)) for m in range(T)])
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Why averaging works: individual boundaries disagree wildly, their average is stable")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    with Animator(here(__file__, "forest_vs_trees.gif"), fps=3, hold=3.5) as anim:
        for m in range(1, T + 1):
            a1.clear()
            scatter_classes(a1, X, y, s=10)
            for k in range(m):
                a1.contour(gx, gy, grid_t[k].reshape(gx.shape), levels=[0.5], colors=[MAGENTA if k == m - 1 else MUTED], linewidths=2.0 if k == m - 1 else 0.7, alpha=1.0 if k == m - 1 else 0.6)
            a1.set(title=f"{m} single-tree boundaries", xlabel="signal 1", ylabel="signal 2", xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
            a2.clear()
            scatter_classes(a2, X, y, s=10)
            avg = grid_t[:m].mean(0).reshape(gx.shape)
            a2.contourf(gx, gy, avg, levels=[0, 0.5, 1.0001], colors=[BLUE, ORANGE], alpha=0.18)
            a2.contour(gx, gy, avg, levels=[0.25, 0.75], colors=[MUTED], linewidths=0.8, linestyles="--")
            a2.contour(gx, gy, avg, levels=[0.5], colors=[INK], linewidths=2)
            a2.set(title="Majority-vote boundary (dashed 25/75%)", xlabel="signal 1", ylabel="signal 2", xlim=(gx.min(), gx.max()), ylim=(gy.min(), gy.max()))
            a3.clear()
            a3.scatter(range(1, m + 1), each[:m], s=18, color=MUTED, zorder=3)
            a3.plot(range(1, m + 1), ens[:m], color=ORANGE, marker="o", markersize=3)
            a3.set(title="Test accuracy: trees (grey) vs vote", xlabel="number of trees in the vote", ylabel="accuracy", xlim=(0, T + 1), ylim=(0.65, 0.95))
            caption(fig, f"{m:>2} trees: vote accuracy {ens[m - 1]:.3f};  individual trees range {each[:m].min():.3f} - {each[:m].max():.3f}  (mean {each[:m].mean():.3f})")
            anim.grab(fig)


def variance_experiment():
    """Draw 40 independent training sets; compare a single deep tree with a forest on one fixed test set."""
    Xt, yt = fraud_data(2000, 999)
    tree_acc, rf_acc, tree_p, rf_p = [], [], [], []
    for s in range(40):
        X, y = fraud_data(300, 100 + s)
        t = DecisionTree(n_classes=2).fit(X, y); f = RandomForest(40, max_features=1, seed=s).fit(X, y)
        tree_acc.append(accuracy(yt, t.predict(Xt))); rf_acc.append(accuracy(yt, f.predict(Xt)))
        tree_p.append(t.predict_proba(Xt)[:, 1]); rf_p.append(f.predict_proba(Xt)[:, 1])
    print(f"  40 resampled training sets -> single tree: test acc {np.mean(tree_acc):.3f} ± {np.std(tree_acc):.3f};  forest(40): {np.mean(rf_acc):.3f} ± {np.std(rf_acc):.3f}")
    print(f"  prediction variance across training sets (avg over test points): tree {np.mean(np.var(tree_p, 0)):.4f}  forest {np.mean(np.var(rf_p, 0)):.4f}  (x{np.mean(np.var(tree_p, 0)) / np.mean(np.var(rf_p, 0)):.1f} lower)")


def importance_demo():
    X, y, names = make_customer_churn(700, seed=0)
    r = rng_(5)
    X = np.c_[X, r.normal(size=(len(X), 2)), r.integers(0, 1000, len(X))]          # + 2 noise features + a high-cardinality ID-like column
    names = names + ["noise_a", "noise_b", "customer_id"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.3, seed=1)
    rf = RandomForest(150, max_features="sqrt", min_samples_leaf=3, seed=2).fit(Xtr, ytr)
    imp, perm = rf.feature_importances(), rf.permutation_importance(Xte, yte, 8)
    print(f"\n  churn forest: test acc {accuracy(yte, rf.predict(Xte)):.3f}, OOB acc {rf.oob_accuracy():.3f}")
    print("  feature            impurity-importance   permutation-importance (drop in test accuracy)")
    for i in np.argsort(-imp):
        print(f"  {names[i]:15s}     {imp[i]:6.3f}              {perm[i]:+.3f}")


def main():
    growth_gif()
    spaghetti_gif()
    variance_experiment()
    importance_demo()
    try:
        from sklearn.ensemble import RandomForestClassifier as SK
        X, y = fraud_data(300, 0); Xt, yt = fraud_data(2000, 1)
        sk = SK(100, max_features=1, random_state=0).fit(X, y); mine = RandomForest(100, max_features=1, seed=0).fit(X, y)
        print(f"\n  [sklearn] test accuracy: sklearn {sk.score(Xt, yt):.3f} vs ours {accuracy(yt, mine.predict(Xt)):.3f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
