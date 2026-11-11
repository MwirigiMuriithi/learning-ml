"""
# 13 · AdaBoost
**Family:** Supervised · Classification (ensemble, boosting)

## Definition
AdaBoost builds a strong classifier from many **weak learners** (here, one-split "stumps" barely better than guessing)
trained *sequentially*. After each round it **raises the weights of the examples that were misclassified** so the next stump is forced
to focus on the hard cases. The final prediction is a weighted vote.

## Intuition
A team of mediocre specialists. The first stump draws one line. Points it got wrong become heavier (they appear bigger on screen),
so the second stump concentrates on them, and so on. Stumps that were more accurate get a louder voice (larger α).

## The math
    init   wᵢ = 1/n
    for t = 1..T:  fit stump h_t on weighted data;   ε_t = Σ wᵢ·1[h_t(xᵢ) ≠ yᵢ]
                   α_t = ½ ln((1 − ε_t)/ε_t)
                   wᵢ ← wᵢ · exp(−α_t yᵢ h_t(xᵢ)) , then normalise      (y, h ∈ {−1,+1})
    final  H(x) = sign( Σ_t α_t h_t(x) )
    AdaBoost = gradient boosting on the exponential loss; sensitive to label noise and outliers (they get exponentially heavy).

## What the animation shows
A ring-shaped class that no single vertical/horizontal line can separate. Left: point size = current weight, black line = the newest
stump, background = the ensemble's current decision. Right: the ensemble's accuracy climbs while each stump's own weighted error stays close to 0.5.

## Try this
- Add 10 % label noise and watch the weights explode on the flipped points.
- Replace stumps by depth-2 trees and compare how many rounds are needed.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.trees import Tree
from mlatlas.cli import run


def main(gif, quick):
    X, y = data.circles(250, .12, seed=2); s = 2 * y - 1; n = len(y); R = 8 if quick else 30
    w = np.ones(n) / n; xx, yy, G = viz.grid(X); FG = np.zeros(len(G)); F = np.zeros(n); rec = []
    for t in range(R):
        st = Tree(1, task="clf", n_classes=2).fit(X, y, w); h = 2 * st.predict(X) - 1; eps = max(w[h != s].sum(), 1e-9)
        a = .5 * np.log((1 - eps) / eps); FG += a * (2 * st.predict(G) - 1); F += a * h
        rec.append(dict(w=w.copy(), seg=st.segments[0] if st.segments else None, FG=FG.copy(), acc=(np.sign(F) == s).mean(), eps=eps, alpha=a))
        w = w * np.exp(-a * s * h); w /= w.sum()
    print(f"  after {R} stumps: train accuracy {rec[-1]['acc']:.3f}  (each stump alone had ε≈{np.mean([r['eps'] for r in rec]):.2f})")
    fig, (a1, a2) = viz.subplots(1, 2, w=11, h=4.4)

    def update(t):
        a1.clear(); a2.clear(); r = rec[t]; viz.show_map(a1, xx, yy, (r["FG"] > 0).astype(int)); a1.scatter(*X.T, s=8 + 2500 * r["w"], c=np.array(viz.PALETTE)[y], edgecolor="white", linewidth=.5, zorder=3)
        if r["seg"]:
            f, th, b, _ = r["seg"]; (a1.plot([th, th], [b[2], b[3]], "k", lw=3) if f == 0 else a1.plot([b[0], b[1]], [th, th], "k", lw=3))
        a1.set(title=f"round {t + 1}: ensemble of {t + 1} stumps (dot size = weight)", xlim=(xx.min(), xx.max()), ylim=(yy.min(), yy.max()))
        a2.plot(range(1, t + 2), [q["acc"] for q in rec[:t + 1]], "o-", ms=3, color=viz.PALETTE[2], label="ensemble train accuracy")
        a2.plot(range(1, t + 2), [1 - q["eps"] for q in rec[:t + 1]], "s-", ms=3, color=viz.PALETTE[3], label="this stump's weighted accuracy")
        a2.set(xlim=(0, R + 1), ylim=(.4, 1.02), xlabel="boosting round", title=f"α_{t + 1} = {r['alpha']:.2f}  (weak learners add up)"); a2.legend(loc="lower right")
    viz.animate(fig, update, range(R), gif, fps=3)


if __name__ == "__main__":
    run(main, __file__)
