"""
# 09 · Linear & Quadratic Discriminant Analysis (LDA / QDA)
**Family:** Supervised · Classification (generative, Gaussian class models)

## Definition
LDA and QDA model each class as a multivariate Gaussian N(μ_c, Σ_c) and classify with Bayes' rule.
**LDA** assumes all classes share the *same* covariance matrix, which gives a **linear** boundary.
**QDA** lets every class have its own covariance, which gives a **quadratic (curved)** boundary.

## Intuition
Unlike naive Bayes, these models keep correlations between features (tilted ellipses, not axis-aligned ones). LDA pools all the
data to estimate one shape, which is stable with few samples. QDA is more flexible but needs more data (it estimates d² numbers per class).

## The math
    discriminant  δ_c(x) = −½ log|Σ_c| − ½ (x−μ_c)ᵀ Σ_c⁻¹ (x−μ_c) + log π_c        predict argmax_c δ_c(x)
    LDA: Σ_c = Σ (pooled)  =>  the quadratic terms cancel, leaving the linear  w·x + b
    LDA doubles as supervised dimensionality reduction: project onto the directions maximizing between-class / within-class scatter.

## When to use
- Small data + roughly Gaussian classes: LDA is hard to beat. Classes with clearly different spreads: QDA.
- Regularized discriminant analysis (shrink Σ_c toward Σ) interpolates between the two.

## What the animation shows
Two classes whose covariances start identical and gradually become different (t: 0 → 1). Left: LDA's straight boundary
cannot adapt; right: QDA's boundary curves around the narrower class. Ellipses = fitted 2σ class shapes.

## Try this
- Make the means identical but the covariances different: LDA fails completely, QDA still works.
- Shrink Σ_c toward the pooled Σ with a factor λ and plot accuracy vs λ.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import viz
from mlatlas.cli import run


def fit(X, y, quad):
    cl = [0, 1]; mu = [X[y == c].mean(0) for c in cl]; S = [np.cov(X[y == c].T) for c in cl]; pi = [(y == c).mean() for c in cl]
    pooled = sum((y == c).sum() * S[c] for c in cl) / len(y)
    return mu, (S if quad else [pooled, pooled]), pi


def score(X, mu, S, pi):
    out = []
    for m, s, p in zip(mu, S, pi):
        d = X - m; out.append(-.5 * np.linalg.slogdet(s)[1] - .5 * np.einsum("ij,jk,ik->i", d, np.linalg.inv(s), d) + np.log(p))
    return np.stack(out, 1)


def main(gif, quick):
    rng = np.random.default_rng(5); z = rng.normal(size=(2, 150, 2)); L1 = np.array([[1.1, 0], [.7, .7]]); L2f = np.array([[1.9, .8], [-.9, .35]])
    mu0, mu1 = np.array([-1.3, 0.]), np.array([1.3, .6]); T = 6 if quick else 30
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.6)

    def update(t):
        L2 = (1 - t) * L1 + t * L2f
        X = np.r_[mu0 + z[0] @ L1.T, mu1 + z[1] @ L2.T]; y = np.r_[np.zeros(150), np.ones(150)].astype(int)
        xx, yy, G = viz.grid(np.array([[-6, -5], [6, 5]]), n=80, pad=0)
        for ax, quad, nm in ((a1, False, "LDA (shared covariance, linear boundary)"), (a2, True, "QDA (own covariance, curved boundary)")):
            ax.clear(); mu, S, pi = fit(X, y, quad); sc = score(G, mu, S, pi); viz.show_map(ax, xx, yy, sc.argmax(1))
            ax.contour(xx, yy, (sc[:, 1] - sc[:, 0]).reshape(xx.shape), levels=[0], colors="k", linewidths=2); viz.scatter(ax, X, y, s=14)
            for c in (0, 1): viz.ellipse(ax, mu[c], S[c], "k", 2, lw=1.2, ls="--")
            acc = (score(X, mu, S, pi).argmax(1) == y).mean(); ax.set(title=f"{nm}\nclass spread difference t={t:.2f}, accuracy {acc:.0%}", xlim=(-6, 6), ylim=(-5, 5))
    viz.animate(fig, update, np.linspace(0, 1, T), gif, fps=5)


if __name__ == "__main__":
    run(main, __file__)
