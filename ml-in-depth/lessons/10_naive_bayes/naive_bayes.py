"""
Lesson 10 - Naive Bayes (from scratch)

Two flavours, one idea:  P(class | x)  is proportional to  P(class) * prod_j P(x_j | class).

  1. Gaussian NB  - continuous features.  Story: a bank scores loan applicants from
     (income, debt-to-income ratio).  Each feature of each class is modelled by a 1-D Gaussian.
  2. Multinomial NB - word counts.  Story: a spam filter on a small synthetic vocabulary
     (each email is a bag of words drawn from a "spam" or a "ham" word distribution).

Implemented (NumPy only): GaussianNB, MultinomialNB (Laplace smoothing alpha, log-space scoring).

Animated:
  naive_bayes.gif        - Gaussian NB being fitted as labelled applicants stream in:
                           posterior map + axis-aligned Gaussian ellipses | marginal densities |
                           held-out accuracy.  "Training" is just counting, so it learns fast.
  naive_bayes_email.gif  - a spam email read word by word: each word adds a log-likelihood-ratio
                           to the running log-odds; the posterior P(spam) moves accordingly.

Run:  python naive_bayes.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse


# ---------------------------------------------------------------- Gaussian NB
class GaussianNB:
    """log P(c | x) = log P(c) + sum_j log N(x_j; mu_cj, var_cj) + const.
    'Training' = per-class counting: priors, means and variances."""

    def __init__(self, var_smoothing=1e-3):
        self.var_smoothing = var_smoothing

    def fit(self, X, y, n_classes=None):
        k = int(y.max()) + 1 if n_classes is None else n_classes
        self.classes = np.arange(k)
        self.prior = np.array([(y == c).mean() for c in self.classes])
        self.mu = np.array([X[y == c].mean(0) for c in self.classes])
        # tiny extra variance keeps a class with 1-2 samples from becoming a spike (variance smoothing)
        self.var = np.array([X[y == c].var(0) for c in self.classes]) + self.var_smoothing * X.var(0).max() + 1e-9
        return self

    def log_joint(self, X):
        X = np.atleast_2d(X)
        out = []
        for c in self.classes:
            ll = -0.5 * (np.log(2 * np.pi * self.var[c]) + (X - self.mu[c]) ** 2 / self.var[c])   # log N(x_j; mu, var)
            out.append(np.log(self.prior[c] + 1e-300) + ll.sum(1))                                 # prior + sum over features
        return np.array(out).T

    def predict_proba(self, X):
        return softmax(self.log_joint(X), axis=1)        # normalise in log space (Bayes' denominator)

    def predict(self, X):
        return self.log_joint(X).argmax(1)


# ---------------------------------------------------------------- Multinomial NB
class MultinomialNB:
    """Bag-of-words NB. theta[c, w] = P(word w | class c) = (count_cw + alpha) / (total_c + alpha*V).
    alpha = 1 is Laplace ('add-one') smoothing: no word ever gets probability exactly 0."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        k, V = int(y.max()) + 1, X.shape[1]
        self.prior = np.array([(y == c).mean() for c in range(k)])
        counts = np.array([X[y == c].sum(0) for c in range(k)])             # (k, V) word counts per class
        with np.errstate(divide="ignore"):
            self.log_theta = np.log((counts + self.alpha) / (counts.sum(1, keepdims=True) + self.alpha * V))
        return self

    def log_joint(self, X):
        # log P(c) + sum_w count_w * log theta_cw  : a dot product, no products of tiny numbers
        with np.errstate(divide="ignore", invalid="ignore"):
            lj = np.log(self.prior)[None, :] + np.nan_to_num(X @ self.log_theta.T, nan=-np.inf)
        return lj

    def predict_proba(self, X):
        return softmax(self.log_joint(X), axis=1)

    def predict(self, X):
        return self.log_joint(X).argmax(1)


# ---------------------------------------------------------------- data
def loan_data(n=500, seed=3):
    """Applicants: x0 = income (standardised), x1 = debt ratio (standardised).
    Within each class the two features are positively correlated (rho=0.55) - which NB ignores."""
    r = rng_(seed)
    y = (r.uniform(size=n) < 0.45).astype(int)
    cov = np.array([[1.0, 0.55], [0.55, 1.0]])
    L = np.linalg.cholesky(cov)
    mu = np.where(y[:, None] == 1, [-0.9, 0.9], [0.9, -0.5])         # default = low income, high debt
    X = mu + r.normal(size=(n, 2)) @ L.T
    return X, y


VOCAB = (["free", "win", "prize", "cash", "offer", "click", "urgent", "winner", "credit", "claim"]          # spammy
         + ["meeting", "report", "lunch", "project", "schedule", "team", "thanks", "review", "deadline", "notes"]  # hammy
         + ["the", "and", "please", "today", "you", "your", "now", "new", "see", "call"])                    # neutral


def make_emails(n=600, seed=0, length=22):
    """Synthetic bag-of-words emails. Word probabilities differ by class; a few words are
    *never* spam in the world but appear as rare ham (so smoothing matters)."""
    r = rng_(seed)
    V = len(VOCAB)
    p_spam = np.r_[np.full(10, 0.075), np.full(10, 0.006), np.full(10, 0.019)]
    p_ham = np.r_[np.full(10, 0.004), np.full(10, 0.07), np.full(10, 0.02)]
    p_spam[VOCAB.index("deadline")] = 0.0            # in this world spammers never write "deadline" ...
    p_spam /= p_spam.sum(); p_ham /= p_ham.sum()
    y = (r.uniform(size=n) < 0.4).astype(int)        # 1 = spam
    X = np.zeros((n, V), int)
    seqs = []
    for i in range(n):
        L = max(6, r.poisson(length))
        words = r.choice(V, size=L, p=p_spam if y[i] else p_ham)
        X[i] = np.bincount(words, minlength=V)
        seqs.append(words)
    return X, y, seqs


# ---------------------------------------------------------------- animation 1: Gaussian NB
def gaussian_animation():
    X, y = loan_data()
    Xte, yte = X[300:], y[300:]
    Xs, ys = X[:300], y[:300]
    # stream the first 300 labelled applicants in order; start once both classes have >= 3 examples
    ns = np.unique(np.round(np.geomspace(8, 300, 30)).astype(int))
    final = GaussianNB().fit(Xs, ys)
    accs = []
    for n in ns:
        accs.append(accuracy(yte, GaussianNB().fit(Xs[:n], ys[:n], n_classes=2).predict(Xte)))
    xg = np.linspace(-4.5, 4.5, 200)

    fig = plt.figure(figsize=(13, 4.8))
    fig.suptitle("Gaussian Naive Bayes: count, average, square - then Bayes' rule does the rest", fontsize=13, fontweight="bold", x=0.02, ha="left")
    gs = GridSpec(1, 3, figure=fig, left=0.05, right=0.985, bottom=0.2, top=0.84, wspace=0.28, width_ratios=[1.3, 1, 0.9])
    a1, a2, a3 = (fig.add_subplot(gs[0, j]) for j in range(3))
    with Animator(here(__file__, "naive_bayes.gif"), fps=6, hold=2.5) as anim:
        for j, n in enumerate(ns):
            m = GaussianNB().fit(Xs[:n], ys[:n], n_classes=2)
            a1.clear()
            probability_map(a1, lambda P: m.predict_proba(P)[:, 1], X, pad=0.2, res=110)
            scatter_classes(a1, Xs[:n], ys[:n], s=14)
            for c in (0, 1):
                for s_, ls in ((1, "-"), (2, ":")):          # 1-sigma and 2-sigma axis-aligned ellipses
                    a1.add_patch(Ellipse(m.mu[c], 2 * s_ * np.sqrt(m.var[c][0]), 2 * s_ * np.sqrt(m.var[c][1]), fill=False,
                                         edgecolor=CAT[c], linestyle=ls, linewidth=1.4, zorder=5))
            a1.set(title="Posterior P(default) and what NB believes", xlabel="income (standardised)", ylabel="debt ratio (standardised)")
            a1.text(0.02, 0.97, "ellipses = 1 and 2 sigma, always axis-aligned\n(features assumed independent)", transform=a1.transAxes, va="top", fontsize=8.5, color=INK2)

            a2.clear()
            for c in (0, 1):
                dens = np.exp(-0.5 * (xg - m.mu[c][0]) ** 2 / m.var[c][0]) / np.sqrt(2 * np.pi * m.var[c][0])
                a2.hist(Xs[:n][ys[:n] == c, 0], bins=np.linspace(-4.5, 4.5, 19), density=True, color=CAT[c], alpha=0.25)
                a2.plot(xg, dens, color=CAT[c])
            a2.set(title="Income | class: the fitted 1-D Gaussians", xlabel="income (standardised)", ylabel="density", xlim=(-4.5, 4.5), ylim=(0, 1.25))
            label_line(a2, 2.0, 1.05, "repaid", color=CAT[0]); label_line(a2, -4.3, 1.05, "default", color=CAT[1])

            a3.clear()
            a3.plot(ns[: j + 1], accs[: j + 1], color=BLUE)
            a3.scatter([n], [accs[j]], s=32, color=ORANGE, zorder=5)
            a3.set(title="Held-out accuracy", xlabel="labelled examples seen", ylabel="accuracy", xscale="log", xlim=(7, 320), ylim=(0.5, 1.0))
            caption(fig, f"{n:>3} examples:  priors default/repaid = {m.prior[1]:.2f}/{m.prior[0]:.2f}   "
                         f"mean income (default / repaid) = {m.mu[1][0]:+.2f} / {m.mu[0][0]:+.2f}   held-out accuracy = {accs[j]:.1%}")
            anim.grab(fig)
    print(f"  GaussianNB final held-out accuracy = {accs[-1]:.3f}  (n=300 train)")
    return final, Xs, ys, Xte, yte


# ---------------------------------------------------------------- animation 2: an email, word by word
def email_animation():
    X, y, seqs = make_emails(n=800, seed=1)
    Xtr, ytr, Xte, yte = X[:600], y[:600], X[600:], y[600:]
    nb = MultinomialNB(alpha=1.0).fit(Xtr, ytr)
    acc = accuracy(yte, nb.predict(Xte))
    llr = nb.log_theta[1] - nb.log_theta[0]                       # per-word evidence for spam: log theta_spam - log theta_ham
    prior_lo = np.log(nb.prior[1] / nb.prior[0])

    # pick a spam email in the test set that contains a 'ham-looking' word (so evidence goes both ways)
    ham_ids = [VOCAB.index(w) for w in ("meeting", "report", "lunch", "project", "schedule", "team", "thanks", "review", "deadline", "notes")]
    cand = [i for i in range(600, 800) if y[i] == 1 and len(seqs[i]) <= 22 and any(w in ham_ids for w in seqs[i]) and nb.predict(X[i:i + 1])[0] == 1]
    words = seqs[cand[0]]
    contrib = llr[words]
    run = prior_lo + np.cumsum(contrib)
    post = sigmoid(run)
    T = len(words)

    fig = plt.figure(figsize=(13, 4.8))
    fig.suptitle("Reading an email word by word: every word adds (or subtracts) evidence in log-odds", fontsize=13, fontweight="bold", x=0.02, ha="left")
    gs = GridSpec(1, 3, figure=fig, left=0.04, right=0.985, bottom=0.2, top=0.84, wspace=0.3, width_ratios=[0.9, 1.4, 0.9])
    a1, a2, a3 = (fig.add_subplot(gs[0, j]) for j in range(3))
    with Animator(here(__file__, "naive_bayes_email.gif"), fps=3, hold=3.0) as anim:
        for t in range(-1, T):
            a1.clear(); a1.axis("off"); a1.grid(False)
            a1.set_title("The email (bag of words)")
            per_line = 3
            for i, w in enumerate(words):
                row, col = divmod(i, per_line)
                if i > t:
                    colr, wt = MUTED, "normal"
                else:
                    colr, wt = (ORANGE if contrib[i] > 0 else BLUE), "bold"
                a1.text(0.02 + col * 0.33, 0.93 - row * 0.085, VOCAB[w], color=colr, fontsize=9.5, fontweight=wt, va="center",
                        bbox=dict(boxstyle="round,pad=0.25", fc="#f3f2ee" if i == t else SURFACE, ec=INK if i == t else "none", lw=1.2))
            a1.set_xlim(0, 1); a1.set_ylim(0, 1)

            a2.clear()
            a2.axhline(0, color=MUTED, linewidth=1)
            if t >= 0:
                xs_ = np.arange(1, t + 2)
                a2.bar(xs_, contrib[: t + 1], color=[ORANGE if c > 0 else BLUE for c in contrib[: t + 1]], width=0.6)
                a2.plot(np.r_[0, xs_], np.r_[prior_lo, run[: t + 1]], color=INK, marker="o", markersize=3.5, linewidth=1.4)
            else:
                a2.scatter([0], [prior_lo], color=INK, s=20)
            lim = max(8, np.abs(run).max() + 1, np.abs(contrib).max() + 1)
            a2.set(title="Evidence per word (bars) and running log-odds (line)", xlabel="words read so far", ylabel="log-odds of spam",
                   xlim=(-0.5, T + 0.5), ylim=(-lim * 0.5, lim * 1.05))
            a2.text(T + 0.4, lim * 0.95, "orange bar = pushes towards spam\nblue bar = pushes towards ham", ha="right", va="top", fontsize=8.5, color=INK2)
            a2.text(0.1, prior_lo + 0.5, "prior", fontsize=8.5, color=INK2)

            a3.clear()
            a3.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
            pp = np.r_[sigmoid(prior_lo), post[: t + 1]]
            a3.plot(range(len(pp)), pp, color=BLUE, marker="o", markersize=3.5)
            a3.set(title="P(spam | words so far)", xlabel="words read", ylabel="posterior", xlim=(-0.5, T + 0.5), ylim=(-0.03, 1.03))
            a3.text(T + 0.3, 0.45, "decision threshold", ha="right", fontsize=8.5, color=INK2, va="top")
            if t < 0:
                caption(fig, f"before reading: prior log-odds = {prior_lo:+.2f}  ->  P(spam) = {sigmoid(prior_lo):.2f}   (just the base rate of spam)")
            else:
                caption(fig, f"word {t + 1:>2} = '{VOCAB[words[t]]}':  evidence = log P(w|spam) - log P(w|ham) = {contrib[t]:+.2f}   "
                             f"running log-odds = {run[t]:+.2f}   P(spam) = {post[t]:.3f}")
            anim.grab(fig)
    print(f"  MultinomialNB spam filter test accuracy = {acc:.3f}  (email of {T} words ends at P(spam)={post[-1]:.3f})")
    return nb, Xtr, ytr, Xte, yte


# ---------------------------------------------------------------- numeric demos
def demos(g, Xs, ys, Xte, yte, nb, Xtr_t, ytr_t, Xte_t, yte_t):
    # (a) Laplace smoothing: 'deadline' never occurs in spam training mails
    j = VOCAB.index("deadline")
    print(f"  'deadline' count in spam training mails = {int(Xtr_t[ytr_t == 1][:, j].sum())}, in ham = {int(Xtr_t[ytr_t == 0][:, j].sum())}")
    nb0 = MultinomialNB(alpha=0.0).fit(Xtr_t, ytr_t)
    e = np.zeros((1, len(VOCAB)), int); e[0, VOCAB.index("free")] = 6; e[0, VOCAB.index("win")] = 5; e[0, j] = 1
    print(f"  mail with 6x free, 5x win, 1x deadline:  alpha=0 -> log P(spam, mail) = {nb0.log_joint(e)[0, 1]}   "
          f"alpha=1 -> {nb.log_joint(e)[0, 1]:.2f}  (P(spam|mail)={nb.predict_proba(e)[0, 1]:.3f})")
    for a in (0.01, 0.1, 1.0, 5.0, 20.0):
        print(f"     alpha={a:>5}: test accuracy = {accuracy(yte_t, MultinomialNB(alpha=a).fit(Xtr_t, ytr_t).predict(Xte_t)):.3f}")
    # (b) log-space trick: probability of a 400-word document
    r = rng_(0); p = r.dirichlet(np.ones(30)); doc = r.choice(30, 400, p=p)
    print(f"  naive product of 400 word probabilities = {np.prod(p[doc]):.3e} (underflows toward 0);  sum of logs = {np.log(p[doc]).sum():.1f}")
    # (c) independence assumption: duplicate a feature -> posteriors become overconfident, accuracy barely moves
    def logloss(P, y): return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))
    for name, A, B in (("2 features", Xs, Xte), ("income duplicated 3x", np.c_[Xs, Xs[:, 0], Xs[:, 0]], np.c_[Xte, Xte[:, 0], Xte[:, 0]])):
        m = GaussianNB().fit(A, ys); P = m.predict_proba(B)
        print(f"  {name:<22}: accuracy={accuracy(yte, P.argmax(1)):.3f}  mean max-prob={P.max(1).mean():.3f}  log-loss={logloss(P, yte):.3f}")
    try:
        from sklearn.naive_bayes import GaussianNB as SG, MultinomialNB as SM
        sg = SG(var_smoothing=1e-3 * 0 + 1e-9).fit(Xs, ys)
        print(f"  sklearn check: GaussianNB agrees on {np.mean(sg.predict(Xte) == GaussianNB(var_smoothing=0).fit(Xs, ys).predict(Xte)):.1%} of test points; "
              f"MultinomialNB agrees on {np.mean(SM(alpha=1.0).fit(Xtr_t, ytr_t).predict(Xte_t) == nb.predict(Xte_t)):.1%}")
    except ImportError:
        pass


def main():
    g, Xs, ys, Xte, yte = gaussian_animation()
    nb, Xtr_t, ytr_t, Xte_t, yte_t = email_animation()
    demos(g, Xs, ys, Xte, yte, nb, Xtr_t, ytr_t, Xte_t, yte_t)


if __name__ == "__main__":
    main()
