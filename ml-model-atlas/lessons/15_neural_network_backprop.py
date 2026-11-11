"""
# 15 · Neural Networks (MLP) & Backpropagation
**Family:** Supervised · Deep learning (multi-layer perceptron)

## Definition
A multi-layer perceptron stacks layers of neurons. Each layer computes a linear map followed by a non-linear activation, so the
network can bend space and learn *any* smooth decision boundary (universal approximation). **Backpropagation** is the chain
rule applied efficiently from the loss back to every weight, and gradient descent then updates them all.

## Intuition
The first layers learn to re-describe the inputs (bend and stretch the plane) so that the last layer, a plain linear classifier,
can separate the classes. Without non-linear activations, stacked layers collapse into one linear layer.

## The math (2 hidden layers, tanh, softmax + cross-entropy)
    a1 = tanh(X W1 + b1)      a2 = tanh(a1 W2 + b2)      z = a2 W3 + b3      p = softmax(z)
    L = −1/n Σ log p[i, yᵢ]
    Backward:  dz  = (p − onehot(y)) / n
               dW3 = a2ᵀ dz            da2 = dz W3ᵀ      dz2 = da2 ⊙ (1 − a2²)
               dW2 = a1ᵀ dz2           da1 = dz2 W2ᵀ     dz1 = da1 ⊙ (1 − a1²)
               dW1 = Xᵀ dz1            (biases: column sums)        then  W ← W − η·dW (here with Adam)
    This file writes every line of that by hand and verifies it against numerical gradients.

## Practical tips
- Standardize inputs; initialize with Xavier/He scaling; ReLU/GELU for deep nets; Adam is a strong default.
- Overfitting control: weight decay, dropout, early stopping, more data. Batch/layer norm stabilize deep nets.

## What the animation shows
A 3-arm spiral, which no linear model can classify. Left: the decision regions the network carves out over training;
right: loss and accuracy curves. Early on the boundaries are almost straight lines; then they twist to follow the spirals.

## Try this
- Reduce hidden size to 4 units: what is the best the network can do?
- Swap tanh for ReLU (remember the derivative changes to 1[z>0]).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.nn import AdamNP
from mlatlas.cli import run


def init(rng, sizes):
    P = []
    for a, b in zip(sizes[:-1], sizes[1:]): P += [rng.normal(0, np.sqrt(1 / a), (a, b)), np.zeros(b)]
    return P


def forward(P, X):
    W1, b1, W2, b2, W3, b3 = P; a1 = np.tanh(X @ W1 + b1); a2 = np.tanh(a1 @ W2 + b2); z = a2 @ W3 + b3
    z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True); return a1, a2, p


def loss_grads(P, X, y):
    W1, b1, W2, b2, W3, b3 = P; n = len(y); a1, a2, p = forward(P, X); loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    dz = p.copy(); dz[np.arange(n), y] -= 1; dz /= n
    dW3, db3 = a2.T @ dz, dz.sum(0); dz2 = (dz @ W3.T) * (1 - a2 ** 2); dW2, db2 = a1.T @ dz2, dz2.sum(0)
    dz1 = (dz2 @ W2.T) * (1 - a1 ** 2); dW1, db1 = X.T @ dz1, dz1.sum(0)
    return loss, [dW1, db1, dW2, db2, dW3, db3]


def main(gif, quick):
    X, y = data.spirals(300, k=3, noise=.25, seed=1); rng = np.random.default_rng(0); P = init(rng, [2, 24, 24, 3])
    # gradient check on one weight
    l, g = loss_grads(P, X, y); P[2][3, 5] += 1e-6; lp = loss_grads(P, X, y)[0]; P[2][3, 5] -= 2e-6; lm = loss_grads(P, X, y)[0]; P[2][3, 5] += 1e-6
    print(f"  gradient check  analytic {g[2][3, 5]:.8f}  numeric {(lp - lm) / 2e-6:.8f}")
    E = 500 if quick else 1800; opt = AdamNP(P, lr=.01); xx, yy, G = viz.grid(X, pad=.3); snaps = viz.geo(E, 14 if quick else 50); H = []; frames = {}
    for e in range(E + 1):
        loss, g = loss_grads(P, X, y); acc = (forward(P, X)[2].argmax(1) == y).mean(); H.append((loss, acc))
        if e in snaps: frames[e] = forward(P, G)[2].argmax(1)
        opt.step(g)
    print(f"  final loss {H[-1][0]:.3f}, train accuracy {H[-1][1]:.3f}")
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.6)

    def update(e):
        a1.clear(); a2.clear(); viz.show_map(a1, xx, yy, frames[e], alpha=.4); viz.scatter(a1, X, y, s=18)
        a1.set(title=f"epoch {e}: accuracy {H[e][1]:.0%}", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
        a2.plot([h[0] for h in H[:e + 1]], color=viz.PALETTE[3], label="loss"); a2.plot([h[1] for h in H[:e + 1]], color=viz.PALETTE[2], label="accuracy")
        a2.set(xlim=(0, E), ylim=(0, 1.2), xlabel="epoch", title="Training curves"); a2.legend(loc="center right")
    viz.animate(fig, update, [int(s) for s in snaps], gif, fps=6)


if __name__ == "__main__":
    run(main, __file__)
