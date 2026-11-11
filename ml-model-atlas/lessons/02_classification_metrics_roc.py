"""
# 02 · Classification Metrics, Thresholds & ROC
**Family:** Foundations · Evaluation

## Definition
A probabilistic classifier outputs a *score* per example. Turning scores into decisions needs a **threshold**, and every
threshold produces a different **confusion matrix** (TP, FP, FN, TN). Metrics summarize it; the **ROC curve** shows all
thresholds at once and **AUC** is the probability that a random positive is scored above a random negative.

## Intuition
Loan officer: lower the approval-risk threshold and you catch more true defaulters (recall ↑) but flag more good
customers (precision ↓). There is no free lunch; you pick the trade-off that matches the *cost* of each mistake.

## The math
    accuracy  = (TP+TN)/N                precision = TP/(TP+FP)
    recall (TPR) = TP/(TP+FN)            specificity = TN/(TN+FP)
    FPR = FP/(FP+TN) = 1 − specificity   F1 = 2·precision·recall/(precision+recall)
    ROC: plot TPR against FPR as the threshold sweeps from 1 to 0.   AUC = area under it (0.5 = coin flip, 1 = perfect)

## Practical tips
- With imbalanced classes accuracy lies; use precision/recall, PR-AUC or cost-weighted metrics.
- Choose the threshold on a validation set, using real business costs.
- Always cross-validate: split into k folds, train on k−1, test on the held-out fold, average.

## What the animation shows
A logistic-regression model (trained inside this file) scores 2000 synthetic loan applicants. As the decision threshold
slides from strict to lenient: the score histograms with the cut line, the moving point on the ROC curve, and the live
confusion matrix with all metrics.

## Try this
- Make the classes more imbalanced (change the default rate) and compare accuracy to F1.
- Add a cost matrix (a missed default costs 10× a false alarm) and find the cost-minimizing threshold.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def main(gif, quick):
    X, y, _ = data.loan_default(2000, seed=4); Z, _, _ = data.standardize(X)
    w, b = np.zeros(2), 0.
    for _ in range(600):
        p = 1 / (1 + np.exp(-(Z @ w + b))); w -= .5 * Z.T @ (p - y) / len(y); b -= .5 * (p - y).mean()
    s = 1 / (1 + np.exp(-(Z @ w + b)))

    def conf(th):
        pr = s >= th
        return (pr & (y == 1)).sum(), (pr & (y == 0)).sum(), ((~pr) & (y == 1)).sum(), ((~pr) & (y == 0)).sum()
    ths = np.linspace(1.001, 0, 300); C = np.array([conf(t) for t in ths])
    fpr, tpr = C[:, 1] / (y == 0).sum(), C[:, 0] / (y == 1).sum()
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2); print(f"  AUC = {auc:.3f}")
    fig, (a1, a2, a3) = viz.subplots(1, 3, w=13.5, h=4.2)
    F = 8 if quick else 45

    def update(th):
        for a in (a1, a2, a3): a.clear()
        tp, fp, fn, tn = conf(th); n = len(y)
        a1.hist(s[y == 0], bins=30, range=(0, 1), alpha=.7, color=viz.PALETTE[0], label="repaid")
        a1.hist(s[y == 1], bins=30, range=(0, 1), alpha=.7, color=viz.PALETTE[1], label="defaulted")
        a1.axvline(th, color="k", lw=2); a1.set(title=f"Model score distribution, threshold = {th:.2f}", xlabel="predicted P(default)"); a1.legend(loc="upper center")
        a2.plot(fpr, tpr, color=viz.PALETTE[0], lw=2); a2.plot([0, 1], [0, 1], "k:", lw=1)
        a2.scatter(fp / (fp + tn), tp / (tp + fn), s=90, color=viz.PALETTE[3], zorder=5, edgecolor="k")
        a2.set(title=f"ROC curve (AUC = {auc:.2f})", xlabel="false positive rate", ylabel="true positive rate")
        a3.imshow([[tn, fp], [fn, tp]], cmap="Blues", vmin=0, vmax=n * .6); a3.grid(False)
        for (i, j), v in zip([(0, 0), (0, 1), (1, 0), (1, 1)], [tn, fp, fn, tp]):
            a3.text(j, i, str(v), ha="center", va="center", fontsize=15, color="k")
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        a3.set_xticks([0, 1], ["pred repaid", "pred default"]); a3.set_yticks([0, 1], ["is repaid", "is default"])
        a3.set_title(f"acc {(tp + tn) / n:.2f}  prec {prec:.2f}  recall {rec:.2f}  F1 {f1:.2f}", fontsize=9)
    viz.animate(fig, update, np.r_[np.linspace(.95, .05, F)], gif, fps=8)


if __name__ == "__main__":
    run(main, __file__)
