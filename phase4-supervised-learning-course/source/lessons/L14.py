from nbkit import Lesson


def make():
    nb = Lesson("14_Naive_Bayes", 14, "Naive Bayes",
                "Bayes' theorem plus one bold, usually-wrong, surprisingly-useful assumption",
                120, "🟡 Intermediate",
                setup_extra="""from sklearn.datasets import load_breast_cancer, load_iris, make_blobs
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, log_loss, confusion_matrix
from scipy.special import logsumexp
from scipy.stats import norm""")
    nb.header(
        objectives=[
            "State **Bayes' theorem** and use it to update beliefs (including the base-rate trap).",
            "Explain the **naive** conditional-independence assumption and why it makes the model tiny and fast.",
            "Compute a Naive Bayes posterior **by hand** (categorical Play-Tennis) with Laplace smoothing.",
            "Implement **Gaussian NB** and **Multinomial NB** from scratch; verify against scikit-learn.",
            "Build a text spam filter with bag-of-words features and inspect which words drive the decision.",
            "Understand why NB ranks well but gives **over-confident probabilities**, and what correlated features do.",
            "Pick the right variant (Gaussian, Multinomial, Bernoulli, Complement) for the data.",
        ],
        prereqs="Lessons 09, 11; conditional probability",
        hook=("A test for a rare disease is advertised as *90 % accurate*. You test positive. Do you have the disease? Most people (including many doctors) say \"90 % likely\". "
              "If the disease affects 1 in 100 people, the correct answer is about **9 %**. The missing ingredient is the **prior**: how common the disease was *before* the test.\n\n"
              "**Bayes' theorem** is the rule for combining prior knowledge with new evidence, and **Naive Bayes** is the machine-learning algorithm built directly on it. "
              "It powered the first great spam filters in the 1990s and still wins on tiny datasets and text. "
              "Its secret is an assumption so naive that it's almost always false, and yet the classifier works remarkably well."),
        roadmap=["Bayes' theorem & the base-rate trap", "The naive assumption", "A hand-worked example (Play Tennis)", "Gaussian NB from scratch",
                 "Multinomial NB from scratch + smoothing", "Spam filter on SMS messages", "Strengths, over-confidence and the double-counting problem",
                 "The NB family", "Cheat-sheet, worked examples, problems, quiz"])

    # ---------------------------------------------------------- 1 bayes
    nb.section("🧮", "1. Bayes' theorem and the base-rate trap",
               "For a hypothesis (class) $y$ and evidence $\\mathbf x$:\n\n"
               r"$$\underbrace{P(y\mid\mathbf x)}_{\text{posterior}}=\frac{\overbrace{P(\mathbf x\mid y)}^{\text{likelihood}}\ \overbrace{P(y)}^{\text{prior}}}{\underbrace{P(\mathbf x)}_{\text{evidence}}}\qquad\text{with}\quad P(\mathbf x)=\sum_k P(\mathbf x\mid y=k)P(y=k)$$" "\n\n"
               "**Worked through the disease example:** prevalence 1 %, *sensitivity* $P(+\\mid\\text{sick})=0.90$, *specificity* $P(-\\mid\\text{healthy})=0.91$ so $P(+\\mid\\text{healthy})=0.09$.")
    nb.code(r'''prior, sens, spec = 0.01, 0.90, 0.91
p_pos = sens * prior + (1 - spec) * (1 - prior)               # total probability of a positive test
posterior = sens * prior / p_pos
print(f"P(positive test) = {p_pos:.4f}   ->   P(sick | positive) = {posterior:.3f}")
print("Per 10,000 people:  sick =", int(10000 * prior), "-> true positives", int(10000 * prior * sens),
      "| healthy =", int(10000 * (1 - prior)), "-> FALSE positives", int(10000 * (1 - prior) * (1 - spec)))''')
    nb.md("Out of every 10 000 people, 90 sick ones test positive, but so do **891 healthy ones**. A positive result is therefore only 90/(90+891) ≈ **9 %** reliable. "
          "The rarity of the disease (prior) dominates. Ignoring base rates is one of the most common reasoning errors in medicine, law and machine learning: "
          "it is exactly the *accuracy paradox* of Lesson 09 in disguise.")

    # ---------------------------------------------------------- 2 naive
    nb.section("🧠", "2. From Bayes' theorem to a classifier",
               "Classify by choosing the class with the highest posterior. The evidence $P(\\mathbf x)$ is the same for every class, so it can be ignored for the decision:\n\n"
               r"$$\hat y=\arg\max_k\ P(y=k)\,P(x_1,\dots,x_p\mid y=k)$$" "\n\n"
               "**The problem:** estimating the joint likelihood $P(x_1,\\dots,x_p\\mid y)$ needs exponentially many parameters (all combinations of feature values). "
               "**The naive assumption:** features are **conditionally independent given the class**:\n\n"
               r"$$P(x_1,\dots,x_p\mid y)=\prod_{j=1}^pP(x_j\mid y)$$" "\n\n"
               "Now we only need $p$ simple one-dimensional distributions *per class*. Training is just counting (or computing means and variances), prediction is a sum of logs:")
    nb.md(r"""$$\boxed{\hat y=\arg\max_k\Big[\ \ln P(y=k)+\sum_{j=1}^p\ln P(x_j\mid y=k)\ \Big]}$$

**Why logs?** Multiplying hundreds of small probabilities underflows to 0.0 in floating point; adding their logs is stable. To turn log-scores back into probabilities use the **log-sum-exp** trick: $P(k\mid\mathbf x)=\exp\big(s_k-\operatorname{logsumexp}(s)\big)$.

**What each variant assumes for $P(x_j\mid y)$:**

| Variant | Feature type | Likelihood model |
|---|---|---|
| `GaussianNB` | continuous | Normal $\mathcal N(\mu_{jk},\sigma^2_{jk})$ per feature per class |
| `MultinomialNB` | counts (word counts, TF-IDF) | Multinomial over the vocabulary |
| `BernoulliNB` | binary (word present / absent) | Bernoulli per feature |
| `CategoricalNB` | categories | Categorical distribution per feature |
| `ComplementNB` | counts, imbalanced text | Uses statistics of the *complement* of each class; often better for skewed text |""")
    nb.callout("key", "Naive Bayes is a **generative** model: it models *how the features are produced by each class* ($P(\\mathbf x\\mid y)$) and inverts that with Bayes. "
                      "Logistic regression is **discriminative**: it models $P(y\\mid\\mathbf x)$ directly. Generative models need less data to converge (they make strong assumptions) but are usually beaten by discriminative models once data is plentiful; Ng & Jordan (2001).")

    # ---------------------------------------------------------- 3 tennis
    nb.section("🎾", "3. A hand-worked example: Play Tennis (categorical NB)",
               "Same 14-day dataset as Lesson 11. **Query:** *Outlook = Sunny, Temp = Cool, Humidity = High, Wind = Strong.* Play or not?")
    nb.code(r'''tennis = pd.DataFrame([
    ["Sunny", "Hot", "High", "Weak", "No"], ["Sunny", "Hot", "High", "Strong", "No"], ["Overcast", "Hot", "High", "Weak", "Yes"],
    ["Rain", "Mild", "High", "Weak", "Yes"], ["Rain", "Cool", "Normal", "Weak", "Yes"], ["Rain", "Cool", "Normal", "Strong", "No"],
    ["Overcast", "Cool", "Normal", "Strong", "Yes"], ["Sunny", "Mild", "High", "Weak", "No"], ["Sunny", "Cool", "Normal", "Weak", "Yes"],
    ["Rain", "Mild", "Normal", "Weak", "Yes"], ["Sunny", "Mild", "Normal", "Strong", "Yes"], ["Overcast", "Mild", "High", "Strong", "Yes"],
    ["Overcast", "Hot", "Normal", "Weak", "Yes"], ["Rain", "Mild", "High", "Strong", "No"]], columns=["Outlook", "Temp", "Humidity", "Wind", "Play"])
query = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
for cls in ["Yes", "No"]:
    sub = tennis[tennis["Play"] == cls]; prior = len(sub) / len(tennis); like = [(sub[f] == v).mean() for f, v in query.items()]
    print(f"{cls:<3}: prior {prior:.3f} x likelihoods {np.round(like, 3)} = {prior * np.prod(like):.5f}")''')
    nb.md(r"""**By hand:**

- **Yes:** $\frac9{14}\times\underbrace{\tfrac29}_{\text{Sunny}}\times\underbrace{\tfrac39}_{\text{Cool}}\times\underbrace{\tfrac39}_{\text{High}}\times\underbrace{\tfrac39}_{\text{Strong}}=0.0053$
- **No:** $\frac5{14}\times\tfrac35\times\tfrac15\times\tfrac45\times\tfrac35=0.0206$

Normalise: $P(\text{No}\mid\cdot)=\dfrac{0.0206}{0.0206+0.0053}=\mathbf{0.795}$ → **don't play.**

**The zero-frequency problem.** If a value never co-occurs with a class in training (e.g. *Outlook = Overcast* never occurs with *No*), the estimate is exactly 0 and **one zero wipes out the whole product**. The fix is **Laplace (additive) smoothing**: add $\alpha$ (usually 1) imaginary observations to every count:
$$\hat P(x_j=v\mid y=k)=\frac{\text{count}(v,k)+\alpha}{\text{count}(k)+\alpha\cdot V_j}$$ ($V_j$ = number of distinct values of feature $j$).""")
    nb.code(r'''class CategoricalNBScratch:
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, df, target):
        self.target, self.features = target, [c for c in df.columns if c != target]; self.df = df
        self.classes_ = sorted(df[target].unique()); self.prior_ = {c: (df[target] == c).mean() for c in self.classes_}
        self.values_ = {f: sorted(df[f].unique()) for f in self.features}; return self
    def _lik(self, f, v, c):
        sub = self.df[self.df[self.target] == c]; return ((sub[f] == v).sum() + self.alpha) / (len(sub) + self.alpha * len(self.values_[f]))
    def predict_proba(self, q):
        s = {c: np.log(self.prior_[c]) + sum(np.log(self._lik(f, q[f], c)) for f in self.features) for c in self.classes_}
        z = logsumexp(list(s.values())); return {c: float(np.exp(v - z)) for c, v in s.items()}

print("alpha=0 (no smoothing)  :", {k: round(v, 3) for k, v in CategoricalNBScratch(alpha=1e-12).fit(tennis, "Play").predict_proba(query).items()})
print("alpha=1 (Laplace)       :", {k: round(v, 3) for k, v in CategoricalNBScratch(alpha=1).fit(tennis, "Play").predict_proba(query).items()})
q2 = {"Outlook": "Overcast", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}
print("\nQuery with Overcast (never seen with 'No'):")
print("alpha≈0 :", {k: round(v, 4) for k, v in CategoricalNBScratch(alpha=1e-12).fit(tennis, "Play").predict_proba(q2).items()}, " <- 'No' is declared IMPOSSIBLE")
print("alpha=1 :", {k: round(v, 4) for k, v in CategoricalNBScratch(alpha=1).fit(tennis, "Play").predict_proba(q2).items()}, " <- uncertainty preserved")''')

    # ---------------------------------------------------------- 4 gaussian
    nb.section("🔧", "4. Gaussian Naive Bayes from scratch",
               "For continuous features we fit, **per class and per feature**, a mean $\\mu_{jk}$ and variance $\\sigma^2_{jk}$: training is just `mean` and `var`. The log-likelihood is the log of a normal density:\n\n"
               r"$$\ln P(x_j\mid y=k)=-\tfrac12\ln(2\pi\sigma_{jk}^2)-\frac{(x_j-\mu_{jk})^2}{2\sigma_{jk}^2}$$" "\n\n"
               "(scikit-learn adds a tiny `var_smoothing` fraction of the largest feature variance to every variance to avoid division by zero; we copy that.)")
    nb.code(r'''class GaussianNBScratch:
    def __init__(self, var_smoothing=1e-9): self.var_smoothing = var_smoothing
    def fit(self, X, y):
        X = np.asarray(X, float); self.classes_, yi = np.unique(y, return_inverse=True)
        eps = self.var_smoothing * X.var(axis=0).max()
        self.mu_ = np.array([X[yi == k].mean(axis=0) for k in range(len(self.classes_))])
        self.var_ = np.array([X[yi == k].var(axis=0) for k in range(len(self.classes_))]) + eps
        self.log_prior_ = np.log(np.bincount(yi) / len(yi)); return self
    def joint_log_likelihood(self, X):
        X = np.asarray(X, float)
        return np.stack([lp - 0.5 * np.sum(np.log(2 * np.pi * v)) - 0.5 * np.sum((X - m) ** 2 / v, axis=1)
                         for lp, m, v in zip(self.log_prior_, self.mu_, self.var_)], axis=1)     # (n_samples, n_classes)
    def predict_proba(self, X):
        J = self.joint_log_likelihood(X); return np.exp(J - logsumexp(J, axis=1, keepdims=True))   # log-sum-exp normalisation
    def predict(self, X): return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]
    def score(self, X, y): return np.mean(self.predict(X) == np.asarray(y))

bc = load_breast_cancer(); Xb, yb = bc.data, 1 - bc.target                          # 1 = malignant
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=RANDOM_STATE)
mine = GaussianNBScratch().fit(Xb_tr, yb_tr); ref = GaussianNB().fit(Xb_tr, yb_tr)
print(f"max |probability difference| scratch vs sklearn: {np.abs(mine.predict_proba(Xb_te) - ref.predict_proba(Xb_te)).max():.1e}")
print(f"test accuracy = {mine.score(Xb_te, yb_te):.3f} | ROC-AUC = {roc_auc_score(yb_te, mine.predict_proba(Xb_te)[:, 1]):.4f}  (logistic regression AUC on the same split: "
      f"{roc_auc_score(yb_te, LogisticRegression(max_iter=5000).fit(Xb_tr, yb_tr).predict_proba(Xb_te)[:, 1]):.4f})")''')
    nb.md("Our class matches scikit-learn to machine precision. Look at the numbers: **an excellent ranker** (AUC ≈ 0.996; on par with logistic regression) from a model whose entire 'training' was computing 120 means and variances. "
          "Visualise what it learned for two features: one Gaussian per class:")
    nb.code(r'''fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, j in zip(axes, [list(bc.feature_names).index("worst radius"), list(bc.feature_names).index("mean texture")]):
    xs = np.linspace(Xb_tr[:, j].min(), Xb_tr[:, j].max(), 300)
    for k, (name, col) in enumerate([("benign", "#2563eb"), ("malignant", "#dc2626")]):
        ax.hist(Xb_tr[yb_tr == k, j], bins=22, density=True, alpha=0.35, color=col); ax.plot(xs, norm.pdf(xs, mine.mu_[k, j], np.sqrt(mine.var_[k, j])), color=col, lw=2.4, label=f"{name} N(μ={mine.mu_[k, j]:.1f})")
    ax.set_title(bc.feature_names[j]); ax.legend(fontsize=8)
plt.suptitle("What Gaussian NB learns: one bell curve per class per feature", y=1.03); plt.tight_layout(); plt.show()''')
    nb.md("Where the two bell curves are far apart (*worst radius*) the feature is highly informative; where they overlap (*mean texture*) it adds little evidence. "
          "Each feature contributes **log-likelihood-ratio evidence**, and the classifier simply adds them up with the prior: a kind of principled voting.")
    nb.md("**Decision boundaries.** With equal variances in both classes the boundary between two Gaussians is a line; with *different* variances it is a **quadratic curve** (like QDA).")
    nb.code(r'''Xs, ys = make_blobs(300, centers=[[0, 0], [2.5, 2.2]], cluster_std=[1.0, 1.9], random_state=4)
g2 = GaussianNB().fit(Xs, ys)
fig, ax = plt.subplots(figsize=(5.8, 4.6)); plot_decision_boundary(g2, Xs, ys, ax=ax, title=f"Gaussian NB with unequal class spreads: curved boundary (acc {g2.score(Xs, ys):.2f})"); plt.show()''')

    # ---------------------------------------------------------- 5 multinomial
    nb.section("🔤", "5. Multinomial Naive Bayes from scratch (for text)",
               "In the **bag-of-words** representation a message becomes a vector of **word counts** over a fixed vocabulary (word order is thrown away). "
               "Multinomial NB models, per class, the probability $\\theta_{wk}$ of each word $w$ appearing, estimated by smoothed relative frequency:\n\n"
               r"$$\hat\theta_{wk}=\frac{N_{wk}+\alpha}{N_k+\alpha V}\qquad\ln P(\text{msg}\mid k)=\sum_w c_w\ln\hat\theta_{wk}$$" "\n\n"
               "where $N_{wk}$ = total count of word $w$ in class-$k$ messages, $N_k=\\sum_wN_{wk}$, $V$ = vocabulary size, $c_w$ = count of $w$ in the message. "
               "Let's load the SMS data (a small synthetic Kenyan-flavoured spam set).")
    nb.code(r'''sms = pd.read_csv(data_path("sms_spam_mini.csv")); print(sms["label"].value_counts().to_dict()); display(sms.sample(5, random_state=2))
y_sms = (sms["label"] == "spam").astype(int)
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(sms["message"], y_sms, test_size=0.3, stratify=y_sms, random_state=RANDOM_STATE)
vec = CountVectorizer(lowercase=True); A = vec.fit_transform(Xs_tr).toarray(); B = vec.transform(Xs_te).toarray()      # word-count matrices
vocab = np.array(vec.get_feature_names_out()); print("training matrix:", A.shape, "(messages x vocabulary words)")''')
    nb.code(r'''class MultinomialNBScratch:
    def __init__(self, alpha=1.0): self.alpha = alpha
    def fit(self, X, y):
        X = np.asarray(X, float); self.classes_, yi = np.unique(y, return_inverse=True)
        counts = np.array([X[yi == k].sum(axis=0) for k in range(len(self.classes_))])               # N_wk
        smoothed = counts + self.alpha
        self.feature_log_prob_ = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))                # ln theta_wk
        self.log_prior_ = np.log(np.bincount(yi) / len(yi)); return self
    def joint_log_likelihood(self, X): return np.asarray(X, float) @ self.feature_log_prob_.T + self.log_prior_
    def predict_proba(self, X):
        J = self.joint_log_likelihood(X); return np.exp(J - logsumexp(J, axis=1, keepdims=True))
    def predict(self, X): return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]
    def score(self, X, y): return np.mean(self.predict(X) == np.asarray(y))

mine = MultinomialNBScratch(alpha=1.0).fit(A, ys_tr); ref = MultinomialNB(alpha=1.0).fit(A, ys_tr)
print(f"max |probability difference| vs sklearn: {np.abs(mine.predict_proba(B) - ref.predict_proba(B)).max():.1e}")
p = mine.predict(B); print(f"test accuracy {accuracy_score(ys_te, p):.3f} | spam precision {precision_score(ys_te, p):.3f} | spam recall {recall_score(ys_te, p):.3f}")
display(pd.DataFrame(confusion_matrix(ys_te, p), index=["actually ham", "actually spam"], columns=["pred ham", "pred spam"]))''')
    nb.md("**What did it learn?** The log-odds of a word, $\\ln\\hat\\theta_{w,\\text{spam}}-\\ln\\hat\\theta_{w,\\text{ham}}$, is its *evidence for spam*: we can simply read the model:")
    nb.code(r'''log_odds = pd.Series(mine.feature_log_prob_[1] - mine.feature_log_prob_[0], index=vocab).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
log_odds.tail(12).plot.barh(ax=axes[0], color="#dc2626"); axes[0].set_title("Strongest SPAM words (log-odds)")
log_odds.head(12).plot.barh(ax=axes[1], color="#2563eb"); axes[1].set_title("Strongest HAM words"); plt.tight_layout(); plt.show()
def spam_prob(text): return mine.predict_proba(vec.transform([text]).toarray())[0, 1]
for t in ["Congratulations you won KES 50,000, claim your prize now", "Are you free tonight at the gym?", "Free lunch tomorrow, call me", "URGENT: send your PIN to verify your account", "I will win the match tomorrow"]:
    print(f"P(spam) = {spam_prob(t):.3f}   <-  {t}")''')
    nb.callout("key", "Notice 'free' and 'win' appear in *both* kinds of message in our data; the model weighs them against the rest of the evidence ('lunch', 'tomorrow', 'gym' vs 'claim', 'prize', 'KES'). "
                      "A transparent, tiny, millisecond-fast model, which is why Naive Bayes filtered the world's email for a decade.")
    nb.md("### Smoothing in action\nWhat happens to a **mostly-ham** message that contains one word seen *only in spam* during training (`voucher`)? With almost no smoothing, that single word multiplies the ham likelihood by ≈ 0:")
    nb.code(r'''test_msg = "Are you free to call me now voucher"       # every word occurs in both classes, except 'voucher' (spam only)
print("training count of 'voucher' in ham messages:", int(A[ys_tr.values == 0][:, list(vocab).index("voucher")].sum()), "| in spam messages:", int(A[ys_tr.values == 1][:, list(vocab).index("voucher")].sum()))
for a in [1e-10, 0.01, 0.1, 1.0, 5.0]:
    m_ = MultinomialNBScratch(alpha=a).fit(A, ys_tr)
    print(f"alpha={a:<7} P(spam | '{test_msg}') = {m_.predict_proba(vec.transform([test_msg]).toarray())[0, 1]:.4f}   | test acc = {m_.score(B, ys_te):.3f}")''')
    nb.md("With $\\alpha\\approx0$ the model is *certain* it's spam because **one word never seen in ham annihilates the ham score**, however friendly the other words are; with $\\alpha=1$ the rest of the evidence ('free', 'call', 'me') gets a vote and the verdict softens to ≈ 0.28. "
          "On clean text the *accuracy* barely changes, but smoothing decides what happens on messages with unfamiliar words: i.e. in the real world.")

    # ---------------------------------------------------------- 6 overconfidence
    nb.section("⚠️", "6. Over-confidence and the double-counting problem",
               "NB's assumption says features are independent given the class. When features are **correlated**, the same evidence is counted several times and the posterior is pushed to the extremes. "
               "See it on the cancer data (many highly-correlated size features):")
    nb.code(r'''gnb = GaussianNB().fit(Xb_tr, yb_tr); pn = gnb.predict_proba(Xb_te)[:, 1]
lrm = LogisticRegression(max_iter=5000).fit(Xb_tr, yb_tr); pl = lrm.predict_proba(Xb_te)[:, 1]
print(f"{'':<22}{'accuracy':>10}{'AUC':>8}{'log loss':>10}{'% probs <0.01 or >0.99':>26}")
for name, pr in [("Gaussian NB", pn), ("Logistic regression", pl)]:
    print(f"{name:<22}{accuracy_score(yb_te, pr >= .5):>10.3f}{roc_auc_score(yb_te, pr):>8.3f}{log_loss(yb_te, pr):>10.3f}{np.mean((pr < .01) | (pr > .99)):>26.0%}")
plt.hist(pn, bins=30, alpha=0.6, label="Gaussian NB"); plt.hist(pl, bins=30, alpha=0.6, label="Logistic regression"); plt.legend(); plt.xlabel("predicted P(malignant)"); plt.title("NB piles its probabilities at 0 and 1"); plt.show()''')
    nb.md("Same ranking quality (AUC), but NB's log loss is far worse: it is **over-confident**: when it is wrong, it is wrong with 99.9 % certainty. Now deliberately **duplicate** one informative feature 10 times (perfectly correlated copies):")
    nb.code(r'''j = list(bc.feature_names).index("worst radius")
rows = []
for n_copies in [0, 1, 3, 10]:
    A_ = np.c_[Xb_tr, np.tile(Xb_tr[:, [j]], n_copies)]; B_ = np.c_[Xb_te, np.tile(Xb_te[:, [j]], n_copies)]
    g_ = GaussianNB().fit(A_, yb_tr); l_ = LogisticRegression(max_iter=10000).fit(A_, yb_tr)
    rows.append({"extra copies of 'worst radius'": n_copies, "NB accuracy": g_.score(B_, yb_te), "NB log loss": log_loss(yb_te, g_.predict_proba(B_)[:, 1]),
                 "LogReg accuracy": l_.score(B_, yb_te), "LogReg log loss": log_loss(yb_te, l_.predict_proba(B_)[:, 1])})
display(pd.DataFrame(rows).set_index("extra copies of 'worst radius'").round(3))''')
    nb.md("Each copy adds the same evidence again, so NB's log loss keeps rising (more extreme, more wrong-when-wrong), while logistic regression (which *learns* how to share weight among correlated features) is unaffected. "
          "**Practical rules:** remove redundant features before NB, or use NB for ranking/classification and *calibrate* its probabilities (`CalibratedClassifierCV`) if you need them.")

    # ---------------------------------------------------------- 7 family
    nb.section("🧬", "7. The Naive Bayes family, which one to use?",
               "")
    nb.table([
        ["`GaussianNB`", "Continuous measurements (sensor, medical, iris)", "Bell-shaped class-conditional features; set `var_smoothing` for stability"],
        ["`MultinomialNB`", "Word counts / TF-IDF; any non-negative counts", "`alpha` smoothing (1.0 default; try 0.1–1)"],
        ["`BernoulliNB`", "Binary word presence, short texts, binary features", "Also models the *absence* of features; `binarize` threshold"],
        ["`ComplementNB`", "Imbalanced text classification", "Often beats MultinomialNB on skewed classes"],
        ["`CategoricalNB`", "Features that are categories (codes 0…K−1)", "`alpha`; every category must be seen or `min_categories` set"],
    ], ["Class", "Use for", "Notes"])
    nb.code(r'''B_bin = (A > 0).astype(int); B_bin_te = (B > 0).astype(int)
for name, m, Xa, Xb_ in [("MultinomialNB (counts)", MultinomialNB(), A, B), ("BernoulliNB (presence/absence)", BernoulliNB(), B_bin, B_bin_te)]:
    m.fit(Xa, ys_tr); print(f"{name:<32} test accuracy = {m.score(Xb_, ys_te):.3f}")
print(f"{'Logistic regression on counts':<32} test accuracy = {LogisticRegression(max_iter=1000).fit(A, ys_tr).score(B, ys_te):.3f}")''')

    # ---------------------------------------------------------- 8 profile
    nb.section("📋", "8. Cheat-sheet: the full profile of Naive Bayes")
    nb.table([
        ["**Underlying concept**", "Pick the class with the highest posterior P(y)·ΠP(xⱼ|y), assuming features are independent given the class."],
        ["**How it works**", "Training: count / compute per-class feature statistics (means & variances, or word frequencies) and class priors. Prediction: sum of log-likelihoods + log-prior; normalise with log-sum-exp."],
        ["**Mathematical foundation**", "Bayes' theorem; conditional independence; maximum-likelihood (or MAP with Laplace smoothing); generative modelling; log-sum-exp."],
        ["**Important parameters**", "`alpha` (smoothing; Multinomial/Bernoulli/Categorical), `var_smoothing` (Gaussian), `priors`, `fit_prior`, `binarize` (Bernoulli), `norm` (Complement)."],
        ["**Advantages**", "Extremely fast to train and predict (one pass; O(n·p)); works with very little data; scales to huge vocabularies; handles many classes; few hyper-parameters; incremental learning with `partial_fit`; very hard to overfit."],
        ["**Disadvantages**", "Independence assumption is rarely true; over-confident, poorly calibrated probabilities; can't learn feature interactions; Gaussian variant assumes bell-shaped features; zero-frequency problem without smoothing."],
        ["**Complexity**", "Training O(n·p); prediction O(p·K) per sample (K classes); memory O(p·K)."],
        ["**Use cases**", "Spam / sentiment / topic classification, document routing, medical screening baselines, real-time and streaming prediction, tiny datasets, a quick benchmark."],
        ["**Limitations**", "Correlated features double-count evidence; weak when feature interactions drive the label (XOR); probabilities need calibration; often outperformed by logistic regression / forests with enough data."],
    ], ["Question", "Answer"])

    # ---------------------------------------------------------- 9 worked
    nb.section("🧪", "9. Worked examples")
    nb.md(r"""### Worked Example 1: Two-word spam filter

Training: 40 % of mail is spam. The word *"free"* appears in 30 % of spam and 2 % of ham; *"meeting"* appears in 1 % of spam and 25 % of ham. An email contains **both** words. P(spam)?

Spam score: $0.40\times0.30\times0.01=0.0012$. Ham score: $0.60\times0.02\times0.25=0.0030$. Normalise: $P(\text{spam})=0.0012/(0.0012+0.0030)=\mathbf{0.286}$ → ham. The strong ham evidence of "meeting" outweighs "free".""")
    nb.code(r'''s, h = 0.40 * 0.30 * 0.01, 0.60 * 0.02 * 0.25; print("P(spam) =", round(s / (s + h), 3))''')
    nb.md(r"""### Worked Example 2: Gaussian NB for one feature

Feature: body temperature. Healthy: $\mu=36.8,\ \sigma=0.4$; Flu: $\mu=38.5,\ \sigma=0.7$; prior Flu = 0.2. A patient has $x=37.8$.

$\mathcal N(37.8;36.8,0.4)=\dfrac{1}{0.4\sqrt{2\pi}}e^{-\frac12(2.5)^2}=0.997\times0.0439=0.0437$; $\mathcal N(37.8;38.5,0.7)=\dfrac1{0.7\sqrt{2\pi}}e^{-\frac12(1)^2}=0.570\times0.607=0.346$.
Scores: healthy $0.8\times0.0437=0.0350$; flu $0.2\times0.346=0.0692$ → $P(\text{flu})=0.0692/0.1042=\mathbf{0.664}$. Even with a low prior, evidence from the temperature tips it to flu.""")
    nb.code(r'''ph = 0.8 * norm.pdf(37.8, 36.8, 0.4); pf = 0.2 * norm.pdf(37.8, 38.5, 0.7); print("P(flu | 37.8) =", round(pf / (ph + pf), 3))''')

    # ---------------------------------------------------------- problems
    nb.problem(
        "Bayes' theorem: the base-rate trap",
        "Write `posterior(prior, sensitivity, specificity)` returning P(disease | positive test). Compute `p_rare` for prior = 0.01 (sensitivity 0.9, specificity 0.91), "
        "`p_common` for prior = 0.20 (same test), and `prior_needed` = the **smallest prior** (to 3 decimals, scanning `np.arange(0.001, 1, 0.001)`) at which a positive result gives at least a 50 % chance of disease.",
        ["P(+) = sens·prior + (1 − spec)·(1 − prior); posterior = sens·prior / P(+).",
         "Expect about 0.092 for prior 1 % and about 0.71 for prior 20 %.",
         "Scan priors and pick the first for which `posterior(...) >= 0.5`."],
        'def posterior(prior, sensitivity, specificity):\n    pass\n\np_rare = p_common = prior_needed = None   # your code\n',
        'def posterior(prior, sensitivity, specificity):\n    p_pos = sensitivity * prior + (1 - specificity) * (1 - prior)\n    return sensitivity * prior / p_pos\n\np_rare = posterior(0.01, 0.9, 0.91); p_common = posterior(0.20, 0.9, 0.91)\n'
        'prior_needed = next(float(p) for p in np.arange(0.001, 1, 0.001) if posterior(p, 0.9, 0.91) >= 0.5)\nprint(round(p_rare, 4), round(p_common, 4), round(prior_needed, 3))',
        'check("rare disease: about 9%", lambda: abs(p_rare - 0.0917) < 1e-3)\ncheck("common disease (20%): about 71%", lambda: abs(p_common - 0.7143) < 1e-3)\ncheck("a 50% posterior needs a prior of about 9%", lambda: 0.08 < prior_needed < 0.10)',
        explain="The very same test is 9 % reliable in a low-prevalence population but 71 % reliable in a high-prevalence one. Posterior = evidence × prior; *who you test* matters as much as *how good the test is*.")

    nb.problem(
        "Naive Bayes by hand on Play Tennis",
        "Using the `tennis` DataFrame, compute (with **no smoothing**) the **normalised** probability of `Play = No` for the query `Outlook=Rain, Temp=Cool, Humidity=High, Wind=Strong`. "
        "Store the unnormalised scores in `score_yes`, `score_no` and the posterior in `p_no`. Then, with Laplace smoothing α = 1 (use `CategoricalNBScratch`), store `p_no_smooth`.",
        ["Score(class) = prior × Π P(feature value | class), with each factor = (# rows of that class with the value) / (# rows of the class).",
         "p_no = score_no / (score_yes + score_no).",
         "`CategoricalNBScratch(alpha=1).fit(tennis, 'Play').predict_proba(q3)['No']`."],
        'q3 = {"Outlook": "Rain", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}\nscore_yes = score_no = p_no = p_no_smooth = None   # your code\n',
        'q3 = {"Outlook": "Rain", "Temp": "Cool", "Humidity": "High", "Wind": "Strong"}\ndef score(cls):\n    sub = tennis[tennis["Play"] == cls]; return len(sub) / len(tennis) * np.prod([(sub[f] == v).mean() for f, v in q3.items()])\n'
        'score_yes, score_no = score("Yes"), score("No"); p_no = score_no / (score_yes + score_no)\np_no_smooth = CategoricalNBScratch(alpha=1).fit(tennis, "Play").predict_proba(q3)["No"]\nprint(round(score_yes, 5), round(score_no, 5), round(p_no, 3), round(p_no_smooth, 3))',
        'check("score(Yes) ≈ 0.0079", lambda: abs(score_yes - 0.00794) < 1e-4)\ncheck("score(No) ≈ 0.0137", lambda: abs(score_no - 0.01371) < 1e-4)\ncheck("posterior of No ≈ 0.633", lambda: abs(p_no - 0.633) < 5e-3)\ncheck("smoothing changes the answer only slightly", lambda: abs(p_no_smooth - p_no) < 0.1)',
        explain="Rain/Cool/High/Strong is predicted 'No' with ~63 % confidence (Yes: 0.0079, No: 0.0137 before normalising). Smoothing nudges the counts slightly (to ~59 %) but cannot change the story when no count was zero: its real job is to rescue the cases where one is.",
        level="⭐⭐")

    nb.problem(
        "Where is the decision boundary?",
        "Two classes with equal spread: class 0 ~ N(0, 1²), class 1 ~ N(3, 1²). Write `gauss_logpdf(x, mu, var)` (the formula in Section 4). "
        "Then find numerically (grid of `np.linspace(-2, 5, 70001)`) the point where the two posteriors are equal: `boundary_equal` for priors (0.5, 0.5) and `boundary_skewed` for priors (0.8, 0.2). "
        "Compare with theory `1.5` and `1.5 + ln(4)/3`.",
        [r"log-posterior (unnormalised) = `np.log(prior) + gauss_logpdf(x, mu, var)`.",
         "The boundary is where `score1 - score0` changes sign: take the grid point with the smallest absolute difference.",
         "When class 0 is more common, the boundary moves *toward class 1*: the rarer class must be more clearly indicated."],
        'def gauss_logpdf(x, mu, var):\n    pass\n\nboundary_equal = boundary_skewed = None   # your code\n',
        'def gauss_logpdf(x, mu, var):\n    return -0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)\n\ngrid = np.linspace(-2, 5, 70001)\n'
        'def boundary(p0, p1):\n    diff = (np.log(p1) + gauss_logpdf(grid, 3, 1)) - (np.log(p0) + gauss_logpdf(grid, 0, 1)); return grid[np.argmin(np.abs(diff))]\n\n'
        'boundary_equal = boundary(0.5, 0.5); boundary_skewed = boundary(0.8, 0.2)\nprint(boundary_equal, boundary_skewed, 1.5 + np.log(4) / 3)',
        'check("log-pdf matches scipy", lambda: abs(gauss_logpdf(0.7, 1, 2.0) - norm.logpdf(0.7, 1, np.sqrt(2.0))) < 1e-12)\ncheck("equal priors: boundary at 1.5", lambda: abs(boundary_equal - 1.5) < 1e-3)\ncheck("skewed priors: boundary at 1.5 + ln(4)/3 ≈ 1.962", lambda: abs(boundary_skewed - (1.5 + np.log(4) / 3)) < 1e-3)',
        explain="The prior tilts the boundary: with 80 % of cases in class 0, a point must be quite far into class-1 territory (≥ 1.96 instead of 1.5) before we call it class 1. This is the same mechanism as lowering/raising a decision threshold in Lesson 09.",
        level="⭐⭐")

    nb.problem(
        "The zero-frequency problem, quantified",
        "Using `A, B, ys_tr, ys_te, vec` (SMS data): for `alpha` in `[1e-10, 0.1, 1.0]` fit `MultinomialNBScratch(alpha)` and store `P(spam | 'Are you free to call me now voucher')` in a dict `p_claim` (keys = alpha). "
        "Set `smoothing_rescues = p_claim[1e-10] > 0.9 and p_claim[1.0] < 0.5`.",
        ["`m = MultinomialNBScratch(alpha=a).fit(A, ys_tr)`; features of the message: `vec.transform([msg]).toarray()`.",
         "`m.predict_proba(...)[0, 1]` is P(spam).",
         "The word 'voucher' never appeared in a ham training message; every other word in the test message appears in both classes."],
        'msg = "Are you free to call me now voucher"\np_claim = {}\nsmoothing_rescues = None   # your code\n',
        'msg = "Are you free to call me now voucher"\np_claim = {a: MultinomialNBScratch(alpha=a).fit(A, ys_tr).predict_proba(vec.transform([msg]).toarray())[0, 1] for a in [1e-10, 0.1, 1.0]}\nsmoothing_rescues = bool(p_claim[1e-10] > 0.9 and p_claim[1.0] < 0.5)\nprint({k: round(v, 4) for k, v in p_claim.items()}, smoothing_rescues)',
        'check("tiny alpha: one unseen word makes the message ~certain spam (> 0.9)", lambda: p_claim[1e-10] > 0.9)\ncheck("alpha=1 restores a sensible verdict (< 0.5)", lambda: p_claim[1.0] < 0.5)\ncheck("smoothing_rescues is True", lambda: smoothing_rescues is True)',
        explain="With α ≈ 0 the ham likelihood of 'voucher' is ≈ 0, wiping out the ham score no matter what else the message says; add-one smoothing gives it a small but non-zero probability so the ham-flavoured words ('free', 'call', 'me') can still vote. (Try a message with a ham-only *and* a spam-only word: both classes get zeroed and the verdict becomes arbitrary; a second reason to smooth.)",
        level="⭐⭐")

    nb.problem(
        "Read the spam words",
        "Using the lesson's fitted `mine` (MultinomialNBScratch α = 1) and `vocab`: compute the spam-vs-ham **log-odds** of every word, and store the 5 words with the **highest** spam log-odds in `top_spam` (list, strongest first) "
        "and the 5 with the **lowest** in `top_ham`. Set `claim_is_spammy = 'claim' in top_spam`.",
        ["`lo = mine.feature_log_prob_[1] - mine.feature_log_prob_[0]`.",
         "`order = np.argsort(lo)`; `vocab[order[-5:][::-1]]` gives the 5 highest, strongest first.",
         "Words that occur only in spam get the largest log-odds."],
        'top_spam = top_ham = claim_is_spammy = None   # your code\n',
        'lo = mine.feature_log_prob_[1] - mine.feature_log_prob_[0]; order = np.argsort(lo)\ntop_spam = list(vocab[order[-5:][::-1]]); top_ham = list(vocab[order[:5]]); claim_is_spammy = "claim" in top_spam\nprint(top_spam, top_ham)',
        'check("5 spam words and 5 ham words", lambda: len(top_spam) == 5 and len(top_ham) == 5)\ncheck("spam words look like spam", lambda: len({"kes", "claim", "prize", "000", "text"} & set(top_spam)) >= 2)\ncheck("the two lists are disjoint", lambda: not set(top_spam) & set(top_ham))',
        explain="The log-odds table *is* the model: each word contributes a fixed amount of evidence for spam or ham, and the verdict is the prior plus the sum of those amounts. That transparency is a big part of NB's enduring popularity.",
        level="⭐⭐")

    nb.problem(
        "Catch double-counting",
        "Duplicate the `worst radius` column **10** times (as in Section 6) for both train and test. Fit `GaussianNB` on the original and on the duplicated data, and compute the test **log loss** of each: `ll_orig`, `ll_dup`. "
        "Do the same for `LogisticRegression(max_iter=10000)`: `ll_lr_orig`, `ll_lr_dup`. Set `nb_hurt_more = (ll_dup - ll_orig) > (ll_lr_dup - ll_lr_orig)`.",
        ["`np.c_[Xb_tr, np.tile(Xb_tr[:, [j]], 10)]` appends 10 copies of column j (get j from `list(bc.feature_names).index('worst radius')`).",
         "Use `log_loss(yb_te, model.predict_proba(X_te)[:, 1])`.",
         "Naive Bayes counts each copy as independent evidence."],
        'll_orig = ll_dup = ll_lr_orig = ll_lr_dup = nb_hurt_more = None   # your code\n',
        'j = list(bc.feature_names).index("worst radius")\nAd = np.c_[Xb_tr, np.tile(Xb_tr[:, [j]], 10)]; Bd = np.c_[Xb_te, np.tile(Xb_te[:, [j]], 10)]\n'
        'll_orig = log_loss(yb_te, GaussianNB().fit(Xb_tr, yb_tr).predict_proba(Xb_te)[:, 1]); ll_dup = log_loss(yb_te, GaussianNB().fit(Ad, yb_tr).predict_proba(Bd)[:, 1])\n'
        'll_lr_orig = log_loss(yb_te, LogisticRegression(max_iter=10000).fit(Xb_tr, yb_tr).predict_proba(Xb_te)[:, 1]); ll_lr_dup = log_loss(yb_te, LogisticRegression(max_iter=10000).fit(Ad, yb_tr).predict_proba(Bd)[:, 1])\n'
        'nb_hurt_more = bool((ll_dup - ll_orig) > (ll_lr_dup - ll_lr_orig)); print(round(ll_orig, 3), round(ll_dup, 3), round(ll_lr_orig, 3), round(ll_lr_dup, 3), nb_hurt_more)',
        'check("duplicates make Naive Bayes worse (log loss increases)", lambda: ll_dup > ll_orig)\ncheck("NB is hurt more than logistic regression", lambda: nb_hurt_more)',
        explain="Naive Bayes treats ten identical copies as ten independent witnesses, multiplying the same evidence ten times and becoming wildly over-confident. Logistic regression splits the weight among the copies, so its probabilities stay sane.",
        level="⭐⭐⭐")

    nb.quiz([
        ("What is the 'naive' assumption?", "That features are conditionally independent given the class, so the joint likelihood factorises into a product of per-feature likelihoods."),
        ("A test has 99 % sensitivity and 99 % specificity, and the disease affects 1 in 10 000. You test positive. Is P(sick) near 99 %?",
         "No, about 1 %. Per 1 000 000 people: 100 sick (99 true positives) versus 999 900 healthy (≈ 9 999 false positives). Posterior ≈ 99 / (99 + 9 999) ≈ 1 %."),
        ("Why do we add logs instead of multiplying probabilities?", "Products of many small numbers underflow to 0; sums of logs are numerically stable (and log-sum-exp renormalises)."),
        ("Why can Naive Bayes be a great *classifier* yet a poor *probability estimator*?",
         "The class with the highest score is often right even when the independence assumption is violated, but the violation inflates the evidence, pushing posteriors to 0/1 (over-confidence)."),
        ("You have 30 labelled emails and need a spam filter today. Which would you try first, Naive Bayes or a random forest?",
         "Naive Bayes: few parameters, strong assumptions, robust with tiny data and instant to train; a forest would overfit 30 examples."),
    ])

    nb.footer(["Naive Bayes = Bayes' theorem + conditional independence: ŷ = argmax [ln P(y) + Σ ln P(xⱼ|y)].",
               "Training is counting (or means/variances); prediction is adding logs: lightning fast and data-efficient.",
               "Laplace smoothing (α) prevents one unseen feature value from zeroing a class; priors tilt decisions toward common classes.",
               "NB ranks well but is over-confident, especially with correlated features (double-counting): calibrate or de-duplicate.",
               "Choose the variant by feature type: Gaussian (continuous), Multinomial/Complement (counts), Bernoulli (binary), Categorical."],
              "Support Vector Machines", "15_Support_Vector_Machines.ipynb")
    return nb
