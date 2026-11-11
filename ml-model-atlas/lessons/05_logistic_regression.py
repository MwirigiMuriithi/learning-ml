"""
# 05 · Logistic Regression
**Family:** Supervised · Classification (linear)

## Definition
Despite the name, logistic regression is a *classifier*. It computes a linear score z = w·x + b and squashes it through
the sigmoid σ(z) = 1/(1+e⁻ᶻ) to get a probability P(y=1 | x). The decision boundary (P = 0.5) is the straight line w·x + b = 0.

## Intuition
Linear regression extended to probabilities. Far on one side of the line the model is confident (P≈0 or 1); near
the line it is unsure (P≈0.5). Training pushes the line to put defaulters on one side and good borrowers on the other,
punishing *confident mistakes* very hard.

## The math
    P(y=1|x) = σ(w·x + b)
    Loss (binary cross-entropy / negative log-likelihood):  L = −1/n Σ [ y log p + (1−y) log(1−p) ]
    Gradient (beautifully simple):  ∂L/∂w = 1/n Σ (pᵢ − yᵢ) xᵢ ,   ∂L/∂b = 1/n Σ (pᵢ − yᵢ)
    Multi-class: softmax instead of sigmoid.  Regularize with an L2/L1 penalty exactly like lesson 04.

## Strengths & weaknesses
- Fast, convex (a single global optimum), well-calibrated probabilities, interpretable (odds ratio = e^w).
- Only linear boundaries unless you add polynomial/interaction features; needs scaled inputs for gradient descent.

## What the animation shows
Loan-default data (income vs debt ratio). Watch the probability map (blue = repay, orange = default) and the black
P=0.5 decision line rotate from a random starting guess to the best separator; right: loss falls, accuracy rises.

## Try this
- Add `debt_ratio**2` as a third feature (needs a 3-D weight vector) and see a curved boundary appear in the original space.
- Change the learning rate: where does training become unstable?
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def main(gif, quick):
    X, y, names = data.loan_default(400, seed=2); Z, mu, sd = data.standardize(X)
    rng = np.random.default_rng(3); w, b = rng.normal(0, 1.5, 2), 0.; lr = .4; steps = 80 if quick else 300; H = []
    for _ in range(steps + 1):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        H.append((w.copy(), b, -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)), ((p > .5) == y).mean()))
        w = w - lr * Z.T @ (p - y) / len(y); b = b - lr * (p - y).mean()
    print(f"  final loss {H[-1][2]:.3f}, accuracy {H[-1][3]:.3f}; weights (standardized) {np.round(H[-1][0], 2)}")
    xx, yy, G = viz.grid(X, pad=5); Gz = (G - mu) / sd
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4); idx = viz.geo(len(H), 12 if quick else 55)
    loss = [h[2] for h in H]

    def update(i):
        a1.clear(); a2.clear(); w_, b_, l, acc = H[i]
        P = 1 / (1 + np.exp(-(Gz @ w_ + b_))); viz.show_map(a1, xx, yy, P, prob=True)
        a1.contour(xx, yy, P.reshape(xx.shape), levels=[.5], colors="k", linewidths=2); viz.scatter(a1, X, y)
        a1.set(title=f"iteration {i}: accuracy {acc:.0%}", xlabel=names[0], ylabel=names[1], xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
        a2.plot(loss, color=viz.PALETTE[3]); a2.scatter(i, loss[i], color="k", zorder=5); a2.set(title="Cross-entropy loss", xlabel="iteration", ylim=(0, max(loss) * 1.05))
    viz.animate(fig, update, idx, gif, fps=8)


if __name__ == "__main__":
    run(main, __file__)
