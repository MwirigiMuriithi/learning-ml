"""
# 08 · Naive Bayes
**Family:** Supervised · Classification (probabilistic, generative)

## Definition
Naive Bayes applies **Bayes' theorem** to classification and makes one big simplifying ("naive") assumption: given the class,
all features are independent. It models how each class *generates* its features, then inverts that to get P(class | features).

## Intuition
A spam filter learns: spam emails usually have many links and lots of CAPS; ham rarely does. For a new email, multiply
how plausible each observed feature is under "spam" by the spam prior, do the same for "ham", and normalise. The independence
assumption is false (links and CAPS correlate) but the ranking of classes is often still right, so it works surprisingly well.

## The math
    P(c | x) ∝ P(c) · Π_j P(x_j | c)
    Gaussian NB:   P(x_j | c) = N(x_j ; μ_cj , σ²_cj)     (parameters = per-class mean and variance, just counts!)
    Compute in log-space:  log P(c) + Σ_j log P(x_j | c)   to avoid underflow.
    Text (multinomial NB): P(word | c) = (count + α) / (total + α·V)   with Laplace smoothing α.

## Strengths & weaknesses
- Extremely fast, works with tiny datasets and huge feature spaces (text), naturally multi-class, online-updatable.
- Probabilities are poorly calibrated (over-confident); can't learn feature interactions.

## What the animation shows
Spam data (links vs CAPS ratio). As more training emails arrive (4 → 400) watch the fitted per-class Gaussians
(ellipses = 2σ) sharpen, the posterior map stabilise, and (right) the 1-D likelihoods P(links | class).

## Try this
- Print the log-posterior terms for a custom email and see which feature dominated the decision.
- Break the assumption: duplicate a feature 5× and watch the classifier become over-confident.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


class GaussianNB:
    def fit(s, X, y):
        s.cl = np.unique(y); s.pri = np.array([(y == c).mean() for c in s.cl])
        s.mu = np.array([X[y == c].mean(0) for c in s.cl]); s.var = np.array([X[y == c].var(0) + 1e-3 + .01 * X.var(0) for c in s.cl]); return s
    def log_joint(s, X):
        return np.log(s.pri) + np.stack([(-.5 * np.log(2 * np.pi * v) - (X - m) ** 2 / (2 * v)).sum(1) for m, v in zip(s.mu, s.var)], 1)
    def proba(s, X):
        L = s.log_joint(X); L -= L.max(1, keepdims=True); P = np.exp(L); return P / P.sum(1, keepdims=True)


def main(gif, quick):
    X, y = data.spam_emails(400, seed=1)
    full = GaussianNB().fit(X, y); q = np.array([[5., 30.]]); lj = full.log_joint(q)[0]
    print(f"  email with 5 links / 30% caps -> log-joint ham {lj[0]:.2f}, spam {lj[1]:.2f}  => P(spam) = {full.proba(q)[0, 1]:.3f}")
    rng = np.random.default_rng(0); o = rng.permutation(len(y)); first = [i for c in (0, 1) for i in o[y[o] == c][:2]]
    order = np.r_[first, [i for i in o if i not in first]]
    ns = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 150, 250, 400]
    if quick: ns = ns[::3]
    xx, yy, G = viz.grid(X, pad=3); xs = np.linspace(0, 14, 200)
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4)

    def update(n):
        a1.clear(); a2.clear(); idx = order[:n]; m = GaussianNB().fit(X[idx], y[idx]); viz.show_map(a1, xx, yy, m.proba(G)[:, 1], prob=True)
        a1.contour(xx, yy, m.proba(G)[:, 1].reshape(xx.shape), levels=[.5], colors="k", linewidths=1.5); viz.scatter(a1, X[idx], y[idx], s=18)
        for c in (0, 1): viz.ellipse(a1, m.mu[c], np.diag(m.var[c]), viz.PALETTE[c], 2, lw=2)
        a1.set(title=f"trained on {n} emails: P(spam)=orange", xlabel="# links", ylabel="CAPS ratio (%)", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
        for c, nm in ((0, "ham"), (1, "spam")):
            a2.plot(xs, m.pri[c] * np.exp(-(xs - m.mu[c, 0]) ** 2 / (2 * m.var[c, 0])) / np.sqrt(2 * np.pi * m.var[c, 0]), color=viz.PALETTE[c], lw=2.5, label=f"P({nm})·P(links|{nm})")
        a2.hist(X[idx][y[idx] == 0, 0], bins=20, range=(0, 14), density=True, alpha=.25, color=viz.PALETTE[0]); a2.hist(X[idx][y[idx] == 1, 0], bins=20, range=(0, 14), density=True, alpha=.25, color=viz.PALETTE[1])
        a2.set(title="Learned class-conditional likelihood of 'links'", xlabel="# links", ylim=(0, .6)); a2.legend()
    viz.animate(fig, update, ns, gif, fps=2)


if __name__ == "__main__":
    run(main, __file__)
