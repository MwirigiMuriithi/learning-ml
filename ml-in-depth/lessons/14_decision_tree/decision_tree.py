"""
Lesson 14 - Decision Trees (CART, from scratch - see cart.py)

Problem story: a lender approves or declines loan applications from two numbers: annual income ($1000s) and
debt-to-income ratio.  The (hidden) policy is a union of rectangles, plus 8% label noise like real human decisions.

What this script does
  * grows a CART tree split by split and animates BOTH the partition of the plane and the tree diagram
  * sweeps max_depth 1..10 and shows underfit -> good fit -> overfit
  * prints: the learned rules as text, Gini vs entropy, feature importances, a regression tree, cost-complexity pruning table,
    and an optional scikit-learn agreement check

Outputs: tree_growth.gif, tree_depth.gif        Run:  python decision_tree.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common.utils import *  # noqa: F401,F403
from cart import DecisionTree

FEATURES = ["income", "debt_ratio"]
CLASSES = ["decline", "approve"]


def loan_data(n=300, seed=0, flip=0.08):
    r = rng_(seed)
    income = r.uniform(20, 120, n)
    dti = r.uniform(0.05, 0.6, n)
    approve = ((income > 55) & (dti < 0.35)) | (income > 95) | ((income > 40) & (dti < 0.14))
    y = approve.astype(int)
    y = np.where(r.random(n) < flip, 1 - y, y)               # noisy human decisions
    return np.c_[income, dti], y


# ------------------------------------------------------------------- tree drawing
def tree_layout(t):
    """x positions by in-order leaf counter; y by depth (static layout from the FULL tree, so frames don't jump)."""
    pos, counter = {}, [0]

    def rec(i):
        nd = t.nodes[i]
        if nd["feature"] < 0:
            pos[i] = (counter[0], -nd["depth"]); counter[0] += 1
        else:
            rec(nd["left"]); rec(nd["right"])
            pos[i] = ((pos[nd["left"]][0] + pos[nd["right"]][0]) / 2, -nd["depth"])
    rec(0)
    return pos, counter[0]


def draw_tree(ax, t, pos, nleaf, k, max_depth):
    """Draw nodes that exist after the first k splits.  A node is 'split' if its order < k."""
    ax.set_axis_off()
    ax.set_xlim(-0.8, nleaf - 0.2); ax.set_ylim(-max_depth - 0.7, 0.7)
    for i, nd in enumerate(t.nodes):
        exists = i == 0 or t.nodes[_parent(t, i)]["order"] < k
        if not exists:
            continue
        x, y = pos[i]
        split = nd["feature"] >= 0 and nd["order"] < k
        if split:
            for c in (nd["left"], nd["right"]):
                if t.nodes[_parent(t, c)]["order"] < k:
                    ax.plot([x, pos[c][0]], [y, pos[c][1]], color=MUTED, linewidth=1, zorder=1)
        if split:
            lab = f"{FEATURES[nd['feature']][:3]} ≤ {nd['threshold']:.2f}" if nd["feature"] == 1 else f"{FEATURES[nd['feature']][:3]} ≤ {nd['threshold']:.0f}"
            ax.text(x, y, lab, ha="center", va="center", fontsize=6.6, color=INK, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=INK, lw=1.3 if nd["order"] == k - 1 else 0.6))
        else:
            p = nd["value"][1]                                      # fraction approved
            col = ORANGE if p >= 0.5 else BLUE
            ax.text(x, y, f"{CLASSES[int(p >= 0.5)][:3]}\n{nd['n_raw']}", ha="center", va="center", fontsize=6.2, color=INK, zorder=3,
                    bbox=dict(boxstyle="round,pad=0.22", fc=col, ec="none", alpha=0.25 + 0.5 * abs(p - 0.5) * 2))


def _parent(t, i):
    if not hasattr(t, "_par"):
        t._par = {}
        for j, nd in enumerate(t.nodes):
            if nd["feature"] >= 0:
                t._par[nd["left"]] = j; t._par[nd["right"]] = j
    return t._par.get(i, 0)


# --------------------------------------------------------------------- GIF 1
def growth_gif():
    X, y = loan_data(300, seed=0)
    Xt, yt = loan_data(2000, seed=1)
    t = DecisionTree(max_depth=4).fit(X, y)
    pos, nleaf = tree_layout(t)
    splits = sorted([nd for nd in t.nodes if nd["feature"] >= 0], key=lambda nd: nd["order"])
    print(f"  full depth-4 tree: {t.n_splits} splits, {t.n_leaves()} leaves, train acc {accuracy(y, t.predict(X)):.3f}, test acc {accuracy(yt, t.predict(Xt)):.3f}")
    gx, gy = np.meshgrid(np.linspace(15, 125, 220), np.linspace(0.0, 0.65, 220))
    P = np.c_[gx.ravel(), gy.ravel()]
    fig = plt.figure(figsize=(13, 5.0))
    fig.suptitle("A tree grows by repeatedly asking the single best yes/no question", fontsize=13, fontweight="bold", x=0.02, ha="left")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.7, 0.75], left=0.05, right=0.985, bottom=0.17, top=0.87, wspace=0.18)
    a1, a2, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    accs = [accuracy(y, t.predict(X, max_splits=k)) for k in range(t.n_splits + 1)]
    tests = [accuracy(yt, t.predict(Xt, max_splits=k)) for k in range(t.n_splits + 1)]
    with Animator(here(__file__, "tree_growth.gif"), fps=2, hold=3.5) as anim:
        for k in range(t.n_splits + 1):
            a1.clear()
            z = t.predict(P, max_splits=k).reshape(gx.shape)
            a1.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], colors=[BLUE, ORANGE], alpha=0.2)
            scatter_classes(a1, X, y, s=12)
            msg = "root: predict the majority class everywhere"
            if k > 0:
                nd = splits[k - 1]
                f, thr, lo, hi = nd["feature"], nd["threshold"], nd["lo"], nd["hi"]
                if f == 0:
                    a1.plot([thr, thr], [lo[1], hi[1]], color=INK, linewidth=2.4, zorder=6)
                else:
                    a1.plot([lo[0], hi[0]], [thr, thr], color=INK, linewidth=2.4, zorder=6)
                l, r = t.nodes[nd["left"]], t.nodes[nd["right"]]
                msg = (f"split {k}: {FEATURES[f]} ≤ {thr:.2f}  (impurity {nd['impurity']:.3f} → "
                       f"{l['impurity']:.3f} | {r['impurity']:.3f},  Gini gain {nd['gain']:.3f})")
            a1.set(title="Partition of the plane (thick = newest cut)", xlabel="income ($1000s)", ylabel="debt-to-income ratio", xlim=(15, 125), ylim=(0.0, 0.65))
            a2.clear()
            draw_tree(a2, t, pos, nleaf, k, 4)
            a2.set_title("The tree (box = question, coloured = prediction, number = applicants)", fontsize=10)
            a3.clear()
            a3.plot(range(k + 1), accs[: k + 1], color=BLUE); a3.plot(range(k + 1), tests[: k + 1], color=ORANGE)
            a3.text(t.n_splits, accs[-1] + 0.01, "train", color=BLUE, ha="right", fontsize=8.5); a3.text(t.n_splits, tests[-1] - 0.02, "test", color=ORANGE, ha="right", va="top", fontsize=8.5)
            a3.set(title="Accuracy", xlabel="number of splits", ylim=(0.5, 1.0), xlim=(0, t.n_splits))
            caption(fig, msg + f"   | train acc {accs[k]:.2f}, test acc {tests[k]:.2f}")
            anim.grab(fig)
    return t, X, y, Xt, yt


# --------------------------------------------------------------------- GIF 2
def depth_gif():
    X, y = loan_data(240, seed=3, flip=0.12)
    Xt, yt = loan_data(3000, seed=4, flip=0.12)
    depths = list(range(1, 11))
    ts = [DecisionTree(max_depth=d).fit(X, y) for d in depths]
    tr = [accuracy(y, t.predict(X)) for t in ts]; te = [accuracy(yt, t.predict(Xt)) for t in ts]
    nl = [t.n_leaves() for t in ts]
    print("  depth sweep (12% label noise): " + "  ".join(f"d{d}: train {a:.2f}/test {b:.2f}/{n} leaves" for d, a, b, n in zip(depths, tr, te, nl)))
    gx, gy = np.meshgrid(np.linspace(15, 125, 200), np.linspace(0.0, 0.65, 200)); P = np.c_[gx.ravel(), gy.ravel()]
    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Depth is the capacity knob: too shallow underfits, too deep memorises noise")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28)
    with Animator(here(__file__, "tree_depth.gif"), fps=2, hold=3.5) as anim:
        for i, d in enumerate(depths):
            a1.clear()
            z = ts[i].predict(P).reshape(gx.shape)
            a1.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], colors=[BLUE, ORANGE], alpha=0.22)
            scatter_classes(a1, X, y, s=12)
            a1.set(title=f"max_depth = {d}", xlabel="income ($1000s)", ylabel="debt-to-income ratio", xlim=(15, 125), ylim=(0.0, 0.65))
            a2.clear()
            a2.plot(depths[: i + 1], nl[: i + 1], "-o", color=BLUE, markersize=4)
            a2.set(title="Number of leaves (model complexity)", xlabel="max_depth", ylabel="leaves", xlim=(1, 10), ylim=(0, max(nl) * 1.08))
            a3.clear()
            a3.plot(depths[: i + 1], tr[: i + 1], "-o", color=BLUE, markersize=4); a3.plot(depths[: i + 1], te[: i + 1], "-o", color=ORANGE, markersize=4)
            a3.text(10, tr[-1] + 0.012, "train", color=BLUE, ha="right", fontsize=8.5); a3.text(10, te[-1] - 0.02, "test", color=ORANGE, ha="right", va="top", fontsize=8.5)
            a3.set(title="Accuracy", xlabel="max_depth", ylabel="accuracy", xlim=(1, 10), ylim=(0.65, 1.02))
            caption(fig, f"depth {d}: {nl[i]} leaves, train {tr[i]:.3f}, test {te[i]:.3f}   (best test accuracy {max(te):.3f} at depth {depths[int(np.argmax(te))]})")
            anim.grab(fig)


def main():
    t, X, y, Xt, yt = growth_gif()
    depth_gif()
    print("\n  Learned rules (depth-3 tree):")
    t3 = DecisionTree(max_depth=3).fit(X, y)
    print("".join("    " + line + "\n" for line in t3.export_text(FEATURES, CLASSES).splitlines()))
    print("  feature importances (impurity decrease):", {k: round(float(v), 3) for k, v in zip(FEATURES, t.feature_importances())})
    for crit in ("gini", "entropy"):
        m = DecisionTree(criterion=crit, max_depth=5).fit(X, y)
        print(f"  criterion={crit:8s}: leaves {m.n_leaves()}, train {accuracy(y, m.predict(X)):.3f}, test {accuracy(yt, m.predict(Xt)):.3f}")
    # cost-complexity pruning: grow a big tree, then collapse the weakest links
    big = DecisionTree().fit(X, y)
    alphas = big.alpha_path()
    print(f"\n  unpruned tree: {big.n_leaves()} leaves, train {accuracy(y, big.predict(X)):.3f}, test {accuracy(yt, big.predict(Xt)):.3f}")
    picks = np.unique(np.round(np.geomspace(1e-3, 0.08, 7), 4))
    for a in picks:
        p = big.pruned(a)
        print(f"    alpha={a:.4f}: {p.n_leaves():3d} leaves, train {accuracy(y, p.predict(X)):.3f}, test {accuracy(yt, p.predict(Xt)):.3f}")
    # regression tree on a noisy sine (piecewise-constant fit)
    r = rng_(0)
    x = np.sort(r.uniform(0, 6, 200))[:, None]; yy = np.sin(x[:, 0]) + r.normal(0, 0.25, 200)
    xt = np.linspace(0, 6, 400)[:, None]
    for d in (1, 3, 6, 12):
        rt = DecisionTree(task="regression", max_depth=d).fit(x, yy)
        print(f"  regression tree depth {d:2d}: train MSE {np.mean((rt.predict(x) - yy) ** 2):.3f}, error vs the TRUE sine {np.mean((rt.predict(xt) - np.sin(xt[:, 0])) ** 2):.3f}")
    try:
        from sklearn.tree import DecisionTreeClassifier as SK
        sk = SK(max_depth=5, random_state=0).fit(X, y); mine = DecisionTree(max_depth=5).fit(X, y)
        print(f"  [sklearn] prediction agreement {np.mean(sk.predict(Xt) == mine.predict(Xt)):.4f}, leaves {sk.get_n_leaves()} vs {mine.n_leaves()}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
