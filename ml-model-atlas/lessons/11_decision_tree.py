"""
# 11 · Decision Trees (CART)
**Family:** Supervised · Classification & Regression (non-linear, rule-based)

## Definition
A decision tree asks a sequence of yes/no questions about single features ("is income ≤ 43?"), splitting the data into
ever purer groups, until a leaf makes a prediction (majority class, or the mean for regression). It is the same as a set of
if/else rules and is the most human-readable ML model.

## Intuition
Twenty-questions for data. At each node, try every feature and every threshold and choose the split that makes the two
children as *pure* as possible. Recurse until a depth limit or until nodes are pure. Deep trees memorise the training set,
so depth/leaf-size is the main complexity dial (and the reason for pruning).

## The math
    Gini impurity   G = 1 − Σ_c p_c²              Entropy   H = − Σ_c p_c log p_c
    Best split      argmin over (feature j, threshold t) of   n_L·G(L) + n_R·G(R)
    Regression      same recipe with variance (SSE) instead of Gini; leaf = mean of y
    Feature importance = total impurity reduction attributed to that feature.

## Strengths & weaknesses
- No scaling needed, handles mixed types/missing values (with tweaks), captures interactions, interpretable.
- Unstable (small data changes → different tree), axis-aligned boundaries only, overfits easily.
  Ensembles (lessons 12–14) fix instability.

## What the animations show
1. `11_decision_tree.gif`: the tree grows level by level on noisy moons; black lines are the chosen split thresholds (thicker =
   earlier), shaded regions are the predictions. Right: train accuracy keeps climbing while test accuracy peaks then falls.
2. `11_decision_tree_regression.gif`: a regression tree approximating a smooth curve with staircase steps.
The console prints the rules of a depth-3 tree on loan data.

## Try this
- Set `min_samples=10` to prune: does the test curve flatten instead of falling?
- Replace Gini by entropy in `trees.py` and compare trees.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.trees import Tree, export_text
from mlatlas.cli import run


def main(gif, quick):
    Xl, yl, nm = data.loan_default(300, seed=1); t = Tree(3, task="clf").fit(Xl, yl)
    print("  Rules learned from loan data (depth 3):"); print("\n".join("    " + l for l in export_text(t, nm, ["repaid", "default"]).split("\n")))
    Xtr, ytr = data.moons(300, .3, seed=1); Xte, yte = data.moons(300, .3, seed=2)
    D = list(range(0, 9 if quick else 11)); xx, yy, G = viz.grid(Xtr)
    trees = [Tree(d, task="clf").fit(Xtr, ytr) for d in D]
    tr = [(tt.predict(Xtr) == ytr).mean() for tt in trees]; te = [(tt.predict(Xte) == yte).mean() for tt in trees]
    print(f"  best depth by test accuracy: {D[int(np.argmax(te))]}")
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4)

    def update(d):
        a1.clear(); a2.clear(); tt = trees[d]; viz.show_map(a1, xx, yy, tt.predict_proba(G)[:, 1], prob=True); viz.scatter(a1, Xtr, ytr)
        for f, th, b, dep in tt.segments:
            (a1.plot([th, th], [b[2], b[3]], "k", lw=max(2.6 - .35 * dep, .6)) if f == 0 else a1.plot([b[0], b[1]], [th, th], "k", lw=max(2.6 - .35 * dep, .6)))
        a1.set(title=f"max depth {d}: {len(tt.segments) + 1} leaves", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
        a2.plot(D[:d + 1], tr[:d + 1], "o-", color=viz.PALETTE[2], label="train"); a2.plot(D[:d + 1], te[:d + 1], "o-", color=viz.PALETTE[3], label="test")
        a2.set(title="Accuracy vs depth (overfitting appears)", xlabel="max depth", xlim=(-.3, D[-1] + .3), ylim=(.6, 1.02)); a2.legend(loc="lower right")
    viz.animate(fig, update, D, gif, fps=2)
    # ---- regression tree
    x, y = data.curve(80, seed=3, noise=.3); xt = np.linspace(-1, 1, 300); fig, ax = viz.subplots(1, 1, w=7, h=4.4)
    Dr = list(range(0, 8 if quick else 9))

    def update2(d):
        ax.clear(); m = Tree(d, task="reg").fit(x[:, None], y); ax.scatter(x, y, s=16, color=viz.PALETTE[0], zorder=3)
        ax.plot(xt, data.curve_fn(xt), "--", color="gray", label="truth"); ax.plot(xt, m.predict(xt[:, None]), color=viz.PALETTE[3], lw=2.5, label="tree")
        ax.set(title=f"Regression tree, depth {d}: {2 ** d} possible leaves (piecewise-constant)", ylim=(-2.2, 2.2)); ax.legend(loc="upper left")
    viz.animate(fig, update2, Dr, gif.replace(".gif", "_regression.gif"), fps=2)


if __name__ == "__main__":
    run(main, __file__)
