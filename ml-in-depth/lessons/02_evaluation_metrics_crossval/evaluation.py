"""
Lesson 02 - Evaluating models: metrics, thresholds and cross-validation (from scratch)

Problem story: a bank screens card transactions for fraud.  Only ~6% are fraudulent, so "accuracy" is a trap
(always answering "genuine" scores 94%).  We train a logistic-regression scorer on two features (amount, unusual-hour score)
and then ask the evaluation questions every project faces:
  1. How good is it?  Which metric, at which decision threshold?  (confusion matrix, precision, recall, F1, ROC, PR, AUC)
  2. Can we trust the number?  (hold-out lottery vs k-fold cross-validation, and the classic leakage trap)

Implemented (NumPy only): confusion matrix, accuracy / precision / recall / F1 / MCC, ROC curve + AUC (trapezoid and rank
formula), precision-recall curve + average precision, log-loss, Brier score, calibration table, k-fold and stratified k-fold,
IRLS logistic regression, and a feature-selection leakage experiment.

Animated:
  threshold_sweep.gif  slide the decision threshold: the confusion matrix, precision/recall and the ROC and PR points move
  cross_validation.gif each fold takes a turn as the validation set; then the single-split lottery vs the k-fold average

Run:  python evaluation.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ------------------------------------------------------------------- data + model
def transactions(n=1000, seed=2, pos_rate=0.06):
    """Two features per transaction: log-amount z-score and unusual-hour score.  y = 1 means fraud."""
    r = rng_(seed)
    y = (r.random(n) < pos_rate).astype(int)
    X = np.where(y[:, None] == 1,
                 r.multivariate_normal([1.5, 1.3], [[0.8, 0.2], [0.2, 0.8]], n),
                 r.multivariate_normal([0.0, 0.0], [[1.0, 0.1], [0.1, 1.0]], n))
    return X, y


def fit_logistic(X, y, lam=1e-2, iters=25):
    """Newton / IRLS:  w <- w - H^-1 g  with H = X'SX + lam I  (S = diag p(1-p)); bias is not penalised."""
    A = np.c_[np.ones(len(X)), X]
    w = np.zeros(A.shape[1])
    R = lam * np.eye(A.shape[1]); R[0, 0] = 0
    for _ in range(iters):
        p = sigmoid(A @ w)
        g = A.T @ (p - y) + R @ w
        H = A.T @ (A * (p * (1 - p))[:, None]) + R + 1e-9 * np.eye(len(w))
        w = w - np.linalg.solve(H, g)
    return w


def score(w, X):
    return sigmoid(np.c_[np.ones(len(X)), X] @ w)


# ------------------------------------------------------------------- metrics
def confusion(y, yhat):
    tp = int(np.sum((y == 1) & (yhat == 1))); fp = int(np.sum((y == 0) & (yhat == 1)))
    fn = int(np.sum((y == 1) & (yhat == 0))); tn = int(np.sum((y == 0) & (yhat == 0)))
    return tp, fp, fn, tn


def prf(tp, fp, fn, tn):
    prec = tp / (tp + fp) if tp + fp else 1.0               # convention: no alarms -> precision 1
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn)
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc, mcc=mcc)


def roc_curve(y, s):
    o = np.argsort(-s, kind="stable")
    ys = y[o]
    tps = np.cumsum(ys); fps = np.cumsum(1 - ys)
    return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]], np.r_[np.inf, s[o]]


def auc_trapezoid(fpr, tpr):
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def auc_rank(y, s):
    """AUC = P(score of a random fraud > score of a random genuine one)  (Mann-Whitney U / ranks, ties count 1/2)."""
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def pr_curve(y, s):
    o = np.argsort(-s, kind="stable")
    ys = y[o]
    tp = np.cumsum(ys); k = np.arange(1, len(y) + 1)
    return tp / ys.sum(), tp / k, s[o]                         # recall, precision, thresholds


def average_precision(y, s):
    rec, prec, _ = pr_curve(y, s)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def calibration_table(y, p, bins=5):
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p <= b)
        out.append((p[m].mean(), y[m].mean(), int(m.sum())))
    return out


# ------------------------------------------------------------------- cross-validation
def kfold_indices(n, k, seed=0, y=None, stratified=False):
    r = rng_(seed)
    folds = [[] for _ in range(k)]
    if stratified and y is not None:
        for c in np.unique(y):
            idx = r.permutation(np.flatnonzero(y == c))
            for i, j in enumerate(idx):
                folds[i % k].append(j)
    else:
        for i, j in enumerate(r.permutation(n)):
            folds[i % k].append(j)
    return [np.array(sorted(f)) for f in folds]


def cv_scores(X, y, k=5, seed=0, stratified=True, lam=1e-2):
    folds = kfold_indices(len(X), k, seed, y, stratified)
    out = []
    for i in range(k):
        va = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
        w = fit_logistic(X[tr], y[tr], lam)
        out.append(auc_rank(y[va], score(w, X[va])))
    return np.array(out)


def leakage_demo(seed=0, n=100, d=2000, n_keep=20):
    """Pure-noise features and random labels: the true AUC is 0.5.  Selecting the 'best' features on ALL the data and
    then cross-validating reports a high AUC (leakage); selecting inside each training fold gives the honest ~0.5."""
    r = rng_(seed)
    X = r.normal(size=(n, d)); y = r.integers(0, 2, n)

    def select(Xt, yt):
        c = np.abs((Xt - Xt.mean(0)).T @ (yt - yt.mean()))
        return np.argsort(-c)[:n_keep]

    def cv(sel_global):
        folds = kfold_indices(n, 5, seed + 1, y, True)
        glob = select(X, y) if sel_global else None
        scs = []
        for i in range(5):
            va = folds[i]; tr = np.concatenate([folds[j] for j in range(5) if j != i])
            cols = glob if sel_global else select(X[tr], y[tr])
            w = fit_logistic(X[tr][:, cols], y[tr], lam=1.0)
            scs.append(auc_rank(y[va], score(w, X[va][:, cols])))
        return np.mean(scs)
    return cv(True), cv(False)


# ------------------------------------------------------------------- animation 1
def threshold_gif(path, y, s):
    fpr, tpr, _ = roc_curve(y, s)
    rec, prec, _ = pr_curve(y, s)
    ths = np.r_[0.9, 0.75, 0.6, 0.5, 0.4, 0.3, 0.22, 0.16, 0.12, 0.09, 0.07, 0.05, 0.03, 0.015]
    prev = y.mean()
    bins = np.linspace(0, 1, 26)
    with Animator(path, fps=1.6, hold=3.5) as anim:
        fig, (a1, a2, a3, a4) = new_fig(1, 4, figsize=(14, 4.9), title="Moving the decision threshold on a fraud scorer (5% fraud)",
                                         gridspec_kw={"width_ratios": [1.3, 1.0, 1.0, 0.8]})
        for t in ths:
            for a in (a1, a2, a3, a4):
                a.clear()
            yhat = (s >= t).astype(int)
            tp, fp, fn, tn = confusion(y, yhat); m = prf(tp, fp, fn, tn)
            a1.hist(s[y == 0], bins=bins, color=BLUE, alpha=0.75, label="genuine")
            a1.hist(s[y == 1], bins=bins, color=ORANGE, alpha=0.85, label="fraud")
            a1.axvline(t, color=INK, lw=2); a1.set_yscale("log"); a1.set(ylim=(0.8, 1000), xlim=(0, 1), title="Scores (log counts)", xlabel="predicted fraud probability")
            a1.legend(loc="upper right")
            a2.plot(fpr, tpr, color=BLUE); a2.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1)
            a2.scatter([fp / (fp + tn)], [tp / (tp + fn)], s=90, color=ORANGE, zorder=5, edgecolor=SURFACE)
            a2.set(xlim=(0, 1), ylim=(0, 1.02), title=f"ROC (AUC {auc_trapezoid(fpr, tpr):.3f})", xlabel="false-positive rate", ylabel="recall (TPR)")
            a3.plot(rec, prec, color=BLUE); a3.axhline(prev, color=MUTED, ls="--", lw=1); a3.text(0.5, prev + 0.02, f"no-skill = prevalence {prev:.2f}", fontsize=8, color=INK2)
            a3.scatter([m["recall"]], [m["precision"]], s=90, color=ORANGE, zorder=5, edgecolor=SURFACE)
            a3.set(xlim=(0, 1), ylim=(0, 1.02), title=f"Precision-recall (AP {average_precision(y, s):.3f})", xlabel="recall", ylabel="precision")
            # confusion matrix as a 2x2 table
            M = np.array([[tn, fp], [fn, tp]])
            a4.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1), aspect="auto"); a4.grid(False)
            for (i, j), v in np.ndenumerate(M):
                a4.text(j, i, str(v), ha="center", va="center", fontsize=13, color="white" if v > M.max() / 2 else INK, fontweight="bold")
            a4.set_xticks([0, 1], ["says genuine", "says fraud"]); a4.set_yticks([0, 1], ["genuine", "fraud"]); a4.set_title("Confusion matrix")
            caption(fig, f"threshold {t:.3f}:  accuracy {m['accuracy']:.3f}   precision {m['precision']:.2f}   recall {m['recall']:.2f}   F1 {m['f1']:.2f}   MCC {m['mcc']:.2f}"
                         f"      (always saying 'genuine' would score accuracy {1 - prev:.3f}, recall 0)")
            anim.grab(fig)


# ------------------------------------------------------------------- animation 2
def cv_gif(path, X, y, k=5):
    folds = kfold_indices(len(X), k, 3, y, True)
    fold_auc = []
    n_splits = 60
    r = rng_(4)
    single = []
    for _ in range(n_splits):                                  # the hold-out lottery: one random 80/20 split each time
        p = r.permutation(len(X)); te = p[: len(X) // 5]; tr = p[len(X) // 5:]
        single.append(auc_rank(y[te], score(fit_logistic(X[tr], y[tr]), X[te])))
    single = np.array(single)
    repeated = np.array([cv_scores(X, y, 5, seed=s).mean() for s in range(60)])
    with Animator(path, fps=1.2, hold=4.0) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="Cross-validation: every observation is tested exactly once",
                                     gridspec_kw={"width_ratios": [1.2, 1.0, 1.1]})
        for i in range(k + 1):
            for a in (a1, a2, a3):
                a.clear()
            if i < k:
                va = folds[i]; tr = np.concatenate([folds[j] for j in range(k) if j != i])
                w = fit_logistic(X[tr], y[tr]); sc = score(w, X[va]); fold_auc.append(auc_rank(y[va], sc))
                a1.scatter(X[tr, 0], X[tr, 1], s=8, color=MUTED, alpha=0.5, label="training")
                a1.scatter(X[va][y[va] == 0, 0], X[va][y[va] == 0, 1], s=22, color=BLUE, edgecolor=SURFACE, label="validation, genuine")
                a1.scatter(X[va][y[va] == 1, 0], X[va][y[va] == 1, 1], s=40, color=ORANGE, edgecolor=SURFACE, label="validation, fraud", zorder=4)
                g = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 2); b = -(w[0] + w[1] * g) / w[2]
                a1.plot(g, b, color=INK, lw=1.5, label="decision line (p = 0.5)")
                a1.set(ylim=(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5), xlim=(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3),
                       title=f"Fold {i + 1}: grey = train, colour = test", xlabel="amount (z)", ylabel="unusual hour")
                a1.legend(loc="lower right", fontsize=7)
                msg = f"Fold {i + 1} of {k}: AUC on its {len(va)} held-out transactions = {fold_auc[-1]:.3f}"
            else:
                a1.hist(single, bins=14, color=ORANGE, alpha=0.8, label="single 80/20 split")
                a1.hist(repeated, bins=14, color=BLUE, alpha=0.85, label="mean of 5-fold CV")
                a1.set(title="60 repetitions", xlabel="AUC estimate", ylabel="count"); a1.legend(loc="upper left")
                msg = (f"The single-split estimate ranges {single.min():.2f}-{single.max():.2f} (sd {single.std():.3f}); the 5-fold average ranges "
                       f"{repeated.min():.2f}-{repeated.max():.2f} (sd {repeated.std():.3f}): same data, less lottery")
            # fold diagram
            for f in range(k):
                for g_ in range(k):
                    col = ORANGE if g_ == f else BLUE
                    a2.add_patch(plt.Rectangle((g_, k - 1 - f), 0.92, 0.8, color=col if (i >= k or f <= i) else GRID, alpha=1 if (f <= i and i < k) or i >= k else 0.5))
                a2.text(-0.15, k - 1 - f + 0.4, f"split {f + 1}", ha="right", va="center", fontsize=8, color=INK2)
            a2.set(xlim=(-1.2, k + 0.2), ylim=(-0.2, k + 0.2), title="5 folds (orange = held out)"); a2.axis("off")
            a2.text(k / 2, -0.15, "blue = used for training", ha="center", color=INK2, fontsize=8)
            a3.bar(np.arange(1, len(fold_auc) + 1), fold_auc, color=AQUA, width=0.6)
            a3.axhline(np.mean(fold_auc), color=INK, lw=1.5, ls="--")
            a3.text(k + 0.45, np.mean(fold_auc), f"mean {np.mean(fold_auc):.3f}\nsd {np.std(fold_auc):.3f}", va="center", fontsize=8.5)
            a3.set(xlim=(0.4, k + 1.6), ylim=(0.5, 1.0), title="AUC per fold", xlabel="fold", xticks=range(1, k + 1))
            caption(fig, msg)
            anim.grab(fig)


# ------------------------------------------------------------------- main
def main():
    X, y = transactions()
    print("Lesson 02: evaluation")
    idx = rng_(1).permutation(len(X)); cut = int(0.7 * len(X))
    tr, te = idx[:cut], idx[cut:]
    w = fit_logistic(X[tr], y[tr]); s = score(w, X[te]); yt = y[te]
    print(f"  data: {len(X)} transactions, {y.sum()} fraud ({y.mean():.1%}); test set {len(te)} with {yt.sum()} fraud")
    m0 = prf(*confusion(yt, np.zeros_like(yt)))
    tp, fp, fn, tn = confusion(yt, (s >= 0.5).astype(int)); m = prf(tp, fp, fn, tn)
    print(f"  always 'genuine': accuracy {m0['accuracy']:.3f}, recall {m0['recall']:.2f}")
    print(f"  model @0.5: TP {tp} FP {fp} FN {fn} TN {tn}; accuracy {m['accuracy']:.3f} precision {m['precision']:.2f} recall {m['recall']:.2f} F1 {m['f1']:.2f} MCC {m['mcc']:.2f}")
    fpr, tpr, _ = roc_curve(yt, s)
    print(f"  AUC trapezoid {auc_trapezoid(fpr, tpr):.4f}  rank formula {auc_rank(yt, s):.4f};  AP {average_precision(yt, s):.3f};  log-loss {log_loss(yt, s):.3f}; Brier {brier(yt, s):.4f} (base-rate model {brier(yt, np.full(len(yt), y[tr].mean())):.4f})")
    print("  calibration (mean predicted, observed rate, n):", [(round(a, 3), round(b, 3), n) for a, b, n in calibration_table(yt, s, 4)])
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, log_loss as sk_ll, brier_score_loss
        print(f"  [sklearn] AUC {roc_auc_score(yt, s):.4f}  AP {average_precision_score(yt, s):.3f}  F1 {f1_score(yt, s >= 0.5):.2f}  log-loss {sk_ll(yt, s):.3f}  Brier {brier_score_loss(yt, s):.4f}")
    except Exception as ex:
        print("  (sklearn check skipped:", ex, ")")
    cvs = cv_scores(X, y, 5)
    print(f"  5-fold stratified CV AUC: {np.round(cvs, 3)} mean {cvs.mean():.3f} sd {cvs.std():.3f}")
    plain = kfold_indices(len(X), 5, 0, y, False)
    print("  positives per fold (plain k-fold):", [int(y[f].sum()) for f in plain], "(stratified:", [int(y[f].sum()) for f in kfold_indices(len(X), 5, 0, y, True)], ")")
    leak, honest = leakage_demo()
    print(f"  leakage demo (pure noise, true AUC 0.5): select features on all data then CV -> {leak:.3f};  select inside each fold -> {honest:.3f}")
    folds = kfold_indices(len(X), 5, 0, y, True)                   # out-of-fold scores: every one of the 1000 transactions scored by a model that never saw it
    oof = np.zeros(len(X))
    for i in range(5):
        va = folds[i]; trn = np.concatenate([folds[j] for j in range(5) if j != i])
        oof[va] = score(fit_logistic(X[trn], y[trn]), X[va])
    fo, to, _ = roc_curve(y, oof)
    print(f"  out-of-fold scores (all 1000): AUC {auc_trapezoid(fo, to):.3f}, AP {average_precision(y, oof):.3f}")
    threshold_gif(here(__file__, "threshold_sweep.gif"), y, oof)
    cv_gif(here(__file__, "cross_validation.gif"), X, y)


if __name__ == "__main__":
    main()
