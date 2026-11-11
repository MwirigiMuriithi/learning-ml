"""
# 06 · The Perceptron
**Family:** Supervised · Classification (linear), the ancestor of neural networks

## Definition
The perceptron (Rosenblatt, 1958) is a single artificial neuron: ŷ = sign(w·x + b). It learns by looking at one example
at a time and, **only when it makes a mistake**, nudging the boundary toward classifying that example correctly.

## Intuition
If a point is on the wrong side of the line, pull the line toward it (add the point's vector to w if it is a positive
example, subtract if it is negative). Repeat. If the classes can be separated by a line, this is *guaranteed* to stop.

## The math
    ŷ = sign(w·x + b)           labels y ∈ {−1, +1}
    if y (w·x + b) ≤ 0 :        w ← w + η y x ,   b ← b + η y        (otherwise do nothing)
    Perceptron convergence theorem: if a separating margin γ exists and ||x|| ≤ R, at most (R/γ)² mistakes are ever made.

## Limitations (and why they matter)
- If the data is **not** linearly separable (like XOR) it never converges and keeps oscillating.
  That failure (Minsky & Papert, 1969) motivated multi-layer networks (lesson 15).
- No probabilities, and the final line is just *some* separator, not the best one (SVMs, lesson 10, fix that).

## What the animation shows
One frame per weight update. Left: linearly separable customers; the circled point is the mistake being fixed and the
boundary jumps until it separates the classes. Right: XOR data, where no line works and the boundary keeps flailing.

## Try this
- Run with a very small and a very large η: does the number of mistakes change? (Hint: with b=0 start, η is just a scale!)
- Add the feature x₁·x₂ to the XOR data and watch the perceptron suddenly succeed (feature engineering!).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def train(X, y, max_updates, seed=0):
    rng = np.random.default_rng(seed); w, b = rng.normal(0, .5, 2), 0.; H = [(w.copy(), b, -1)]
    for _ in range(200):
        mistakes = 0
        for i in rng.permutation(len(y)):
            if y[i] * (w @ X[i] + b) <= 0:
                w = w + y[i] * X[i]; b = b + y[i]; H.append((w.copy(), b, i)); mistakes += 1
                if len(H) > max_updates: return H
        if mistakes == 0: break
    return H


def main(gif, quick):
    r = np.random.default_rng(2); P = r.uniform(-3, 3, (400, 2)); m = .8 * P[:, 0] + .6 * P[:, 1] - .3
    keep = np.abs(m) > .45; X1 = P[keep][:80]; y1 = (m[keep][:80] > 0).astype(int); s1 = 2 * y1 - 1   # separable with a margin
    X2, y2 = data.xor(120, seed=1); s2 = 2 * y2 - 1
    N = 25 if quick else 60; H1, H2 = train(X1, s1, N), train(X2, s2, N)
    print(f"  separable data: converged after {len(H1) - 1} updates;  XOR: still updating after {len(H2) - 1} (never converges)")
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4)

    def panel(ax, X, y, H, t, title):
        ax.clear(); w, b, i = H[min(t, len(H) - 1)]; viz.scatter(ax, X, y, s=22)
        lim = (X[:, 0].min() - .5, X[:, 0].max() + .5), (X[:, 1].min() - .5, X[:, 1].max() + .5)
        viz.line(ax, w, b, lim[0], lim[1], color="k", lw=2)
        if i >= 0: ax.scatter(*X[i], s=170, facecolor="none", edgecolor=viz.PALETTE[3], linewidth=2.5, zorder=6)
        acc = ((np.sign(X @ w + b)) == (2 * y - 1)).mean(); ax.set_title(f"{title}: update {min(t, len(H) - 1)}, accuracy {acc:.0%}")

    def update(t):
        panel(a1, X1, y1, H1, t, "Separable data"); panel(a2, X2, y2, H2, t, "XOR (not separable)")
    viz.animate(fig, update, range(max(len(H1), len(H2), 5)), gif, fps=5)


if __name__ == "__main__":
    run(main, __file__)
