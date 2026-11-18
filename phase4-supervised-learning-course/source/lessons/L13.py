from nbkit import Lesson


def make():
    nb = Lesson("13_KNN_Classification", 13, "K-Nearest Neighbors Classification",
                "You are the company you keep: classify by a vote of the most similar examples",
                110, "🟢🟡 Beginner → Intermediate",
                setup_extra="""from sklearn.datasets import load_breast_cancer, load_digits, make_moons, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, average_precision_score, roc_auc_score""")
    nb.header(
        objectives=[
            "Describe KNN classification as a **vote** among the k most similar examples, with probabilities = vote shares.",
            "Implement KNN classification **from scratch** and match scikit-learn.",
            "See how **k** moves the decision boundary from jagged (k = 1) to rigid (k = n), and choose it by cross-validation.",
            "Show why scaling matters and compare distance metrics (Euclidean, Manhattan, cosine).",
            "Apply KNN to **handwritten digits** and diagnose its confusions.",
            "Explain why KNN is fragile on **imbalanced** data and how thresholds/weights help.",
            "State KNN's costs: memory, O(n·p) prediction, the curse of dimensionality.",
        ],
        prereqs="Lessons 07, 09",
        hook=("How does a doctor reason about a patient with unusual symptoms? Often by analogy: *\"This reminds me of three previous patients: two had condition A, one had B.\"* "
              "How does your phone recognise your handwriting? By comparing your pen strokes with thousands of labelled examples and asking *\"which digit do the closest ones belong to?\"*\n\n"
              "That is **KNN classification**: no model is trained, no equation is learned. To classify a new point we find its *k* nearest labelled neighbours and let them **vote**. "
              "It is a ten-line algorithm that nonetheless reads handwritten digits with about 98.7 % accuracy, and it teaches, better than any other, how *distance*, *scaling* and *class balance* shape every classifier."),
        roadmap=["The algorithm", "A hand-worked vote", "From scratch", "k and the decision boundary", "Scaling & distance metrics",
                 "Digits: where KNN shines", "Imbalanced data: where KNN struggles", "Curse of dimensionality & speed",
                 "Cheat-sheet, worked examples, problems, quiz"])

    # ---------------------------------------------------------- 1
    nb.section("🧠", "1. The algorithm",
               "To classify a query point $x_q$:\n\n"
               "1. Compute its distance to every training point.\n2. Take the $k$ nearest: the neighbourhood $N_k(x_q)$.\n3. **Vote:**\n\n"
               r"$$\hat P(y=c\mid x_q)=\frac1k\sum_{i\in N_k(x_q)}\mathbb 1[y_i=c]\qquad\hat y=\arg\max_c \hat P(y=c\mid x_q)$$" "\n\n"
               "With `weights='distance'` each neighbour's vote is multiplied by $1/d_i$, so closer neighbours count more.\n\n"
               "**Why it works.** The vote share among nearby points is a direct, *non-parametric* estimate of the class probability around $x_q$. "
               "As the data grows and $k$ grows (but more slowly than $n$) this converges to the best possible classifier (the *Bayes classifier*); "
               "even the crude 1-NN rule has error at most **twice** the Bayes error (Cover & Hart, 1967): a remarkable guarantee for such a simple rule.")
    nb.callout("tip", "Use an **odd k** for two classes so a vote can't tie (with 3+ classes ties are still possible; scikit-learn then picks the smallest class label, "
                      "and distance weighting practically eliminates ties).")

    # ---------------------------------------------------------- 2 hand
    nb.section("✍️", "2. A hand-worked vote",
               "Seven labelled points (two features) and a query at **(3, 3)**. Classify with k = 3 and k = 5.")
    nb.code(r'''pts = np.array([[2, 2.2], [1.3, 3.8], [0.5, 1.2], [4.2, 3.9], [5, 3.7], [3.8, 0.9], [6, 6]], float); lab = np.array(["A", "A", "A", "B", "B", "B", "B"]); q = np.array([3., 3.])
d = np.sqrt(((pts - q) ** 2).sum(axis=1)); order = np.argsort(d)
table = pd.DataFrame({"x1": pts[order, 0], "x2": pts[order, 1], "label": lab[order], "distance": d[order].round(3)}); table.index += 1; display(table)
for k in [3, 5]:
    votes = pd.Series(lab[order[:k]]).value_counts(); print(f"k={k}: votes {votes.to_dict()} -> predict {votes.idxmax()}   P(B) = {votes.get('B', 0) / k:.2f}")
plt.scatter(*pts[lab == "A"].T, s=90, label="class A"); plt.scatter(*pts[lab == "B"].T, s=90, label="class B"); plt.scatter(*q, marker="*", s=300, color="k", label="query")
for k_, ls in [(3, "--"), (5, ":")]: plt.gca().add_patch(plt.Circle(q, d[order[k_ - 1]] + 0.01, fill=False, ls=ls, color="gray"))
plt.axis("equal"); plt.legend(); plt.title("The k = 3 and k = 5 neighbourhoods"); plt.show()''')
    nb.md("Note how the answer depends on **k**: the very nearest points tell one story, a slightly wider neighbourhood another. This sensitivity *is* the bias–variance trade-off, as we'll see.")

    # ---------------------------------------------------------- 3 scratch
    nb.section("🔧", "3. Build it from scratch",
               "Vectorised distances (the identity $\\lVert a-b\\rVert^2=\\lVert a\\rVert^2+\\lVert b\\rVert^2-2a\\!\\cdot\\!b$ makes Euclidean distance a single matrix product), "
               "`argpartition` for the k smallest, then a vote. Optional distance weighting and other Minkowski metrics.")
    nb.code(r'''class KNNClassifierScratch:
    """Brute-force k-NN classifier: uniform or inverse-distance voting; Minkowski distance (p=2 uses a fast matrix identity)."""

    def __init__(self, k=5, weights="uniform", p=2, chunk=200):
        self.k, self.weights, self.p, self.chunk = k, weights, p, chunk

    def fit(self, X, y):
        self.X_ = np.asarray(X, float); self.classes_, self.yi_ = np.unique(y, return_inverse=True); return self

    def _distances(self, Q):
        if self.p == 2:                                                     # ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q.x
            d2 = (Q ** 2).sum(1)[:, None] + (self.X_ ** 2).sum(1)[None, :] - 2 * Q @ self.X_.T
            return np.sqrt(np.maximum(d2, 0))
        return (np.abs(Q[:, None, :] - self.X_[None, :, :]) ** self.p).sum(axis=2) ** (1 / self.p)

    def kneighbors(self, Q):
        Q = np.asarray(Q, float); D, I = [], []
        for s in range(0, len(Q), self.chunk):
            d = self._distances(Q[s:s + self.chunk]); idx = np.argpartition(d, self.k - 1, axis=1)[:, :self.k]
            dk = np.take_along_axis(d, idx, axis=1); o = np.argsort(dk, axis=1)
            D.append(np.take_along_axis(dk, o, axis=1)); I.append(np.take_along_axis(idx, o, axis=1))
        return np.vstack(D), np.vstack(I)

    def predict_proba(self, Q):
        dist, idx = self.kneighbors(Q); labels = self.yi_[idx]                       # (n_query, k) neighbour class indices
        w = np.ones_like(dist) if self.weights == "uniform" else 1 / np.maximum(dist, 1e-12)
        P = np.stack([(w * (labels == c)).sum(axis=1) for c in range(len(self.classes_))], axis=1)
        return P / P.sum(axis=1, keepdims=True)

    def predict(self, Q): return self.classes_[self.predict_proba(Q).argmax(axis=1)]
    def score(self, Q, y): return np.mean(self.predict(Q) == np.asarray(y))


print("hand example, k=3:", KNNClassifierScratch(3).fit(pts, lab).predict([q])[0], " P =", KNNClassifierScratch(3).fit(pts, lab).predict_proba([q]).round(2)[0], "(A, B)")

bc = load_breast_cancer(); Xb, yb = bc.data, 1 - bc.target                    # 1 = malignant
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=RANDOM_STATE)
sc = StandardScaler().fit(Xb_tr); Ztr, Zte = sc.transform(Xb_tr), sc.transform(Xb_te)
for w in ["uniform", "distance"]:
    mine = KNNClassifierScratch(7, w).fit(Ztr, yb_tr); ref = KNeighborsClassifier(7, weights=w).fit(Ztr, yb_tr)
    print(f"weights={w:<9} predictions identical: {np.array_equal(mine.predict(Zte), ref.predict(Zte))} | max |proba diff| = {np.abs(mine.predict_proba(Zte) - ref.predict_proba(Zte)).max():.1e} | test acc = {mine.score(Zte, yb_te):.3f}")''')

    # ---------------------------------------------------------- 4 k
    nb.section("🎚️", "4. k and the decision boundary",
               "The **decision boundary** is where the majority vote flips. Small k: boundaries wrap tightly around individual points (high variance). Large k: smooth, simple boundaries (high bias).")
    nb.code(r'''Xm, ym = make_moons(400, noise=0.35, random_state=3)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(Xm, ym, test_size=0.3, stratify=ym, random_state=RANDOM_STATE)
fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
for ax, k in zip(axes, [1, 5, 25, 150]):
    m = KNeighborsClassifier(k).fit(Xm_tr, ym_tr)
    plot_decision_boundary(m, Xm_tr, ym_tr, ax=ax, title=f"k = {k}\ntrain {m.score(Xm_tr, ym_tr):.2f} | test {m.score(Xm_te, ym_te):.2f}")
plt.tight_layout(); plt.show()''')
    nb.md("- **k = 1:** perfect training accuracy, islands around every stray point: memorisation.\n- **k = 5–25:** the sweet spot; smooth boundary that follows the moons.\n- **k = 150 (≈ ½ the data):** the boundary degenerates toward a straight line / the majority class.")
    nb.code(r'''ks = np.arange(1, 80, 2); skf = StratifiedKFold(5, shuffle=True, random_state=0)
tr, va = validation_curve(KNeighborsClassifier(), Xm_tr, ym_tr, param_name="n_neighbors", param_range=ks, cv=skf, scoring="accuracy")
plt.plot(ks, tr.mean(1), "o-", ms=3, label="train accuracy"); plt.plot(ks, va.mean(1), "s-", ms=3, label="validation accuracy")
plt.axvline(ks[va.mean(1).argmax()], color="gray", ls=":", label=f"best k = {ks[va.mean(1).argmax()]}"); plt.xlabel("k"); plt.ylabel("accuracy"); plt.legend(); plt.title("Validation curve for k (moons data)"); plt.show()''')

    # ---------------------------------------------------------- 5 scaling & metrics
    nb.section("⚖️", "5. Scaling and distance metrics",
               "Breast-cancer measurements live on wildly different scales (area ≈ 500–2500, smoothness ≈ 0.05–0.16). Without scaling, *area* single-handedly decides who is a neighbour.")
    nb.code(r'''print("feature scale range (std):", Xb_tr.std(axis=0).min().round(4), "to", Xb_tr.std(axis=0).max().round(1))
raw = KNeighborsClassifier(5).fit(Xb_tr, yb_tr); scl = KNeighborsClassifier(5).fit(Ztr, yb_tr)
print(f"KNN on RAW features   : test accuracy = {raw.score(Xb_te, yb_te):.3f}")
print(f"KNN on SCALED features: test accuracy = {scl.score(Zte, yb_te):.3f}")
cvk = {k: cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), Xb_tr, yb_tr, cv=skf).mean() for k in range(1, 32, 2)}
plt.plot(list(cvk), list(cvk.values()), "o-"); plt.xlabel("k (odd)"); plt.ylabel("CV accuracy"); plt.title("Choosing k on the (scaled) cancer data"); plt.show()
print("best k by CV:", max(cvk, key=cvk.get), "->", round(max(cvk.values()), 3))''')
    nb.md("**Distance metrics** change what 'similar' means. Common choices: **Euclidean** (L2), **Manhattan** (L1: more robust when a few coordinates differ a lot), "
          "and **cosine** (compares the *direction* of vectors and ignores their length; popular for text and embeddings). We'll compare them on digits next.")

    # ---------------------------------------------------------- 6 digits
    nb.section("✍️", "6. Digits: where KNN shines",
               "scikit-learn ships a classic: 1 797 handwritten digits, each an 8×8 grey-scale image (64 features), 10 classes. Similar-looking digits have similar pixels, so *nearest neighbour* is a natural fit.")
    nb.code(r'''dg = load_digits()
fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for i, ax in enumerate(axes.ravel()): ax.imshow(dg.images[i], cmap="gray_r"); ax.set_title(dg.target[i]); ax.axis("off")
plt.suptitle("Sample digits (8×8 pixels)"); plt.tight_layout(); plt.show()
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(dg.data, dg.target, test_size=0.3, stratify=dg.target, random_state=RANDOM_STATE)
knn_d = KNeighborsClassifier(3).fit(Xd_tr, yd_tr)
print(f"KNN (k=3) test accuracy on digits: {knn_d.score(Xd_te, yd_te):.3f}   (a 10-class problem; random guessing = 0.10)")''')
    nb.code(r'''pred_d = knn_d.predict(Xd_te); cm = confusion_matrix(yd_te, pred_d)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
im = axes[0].imshow(cm, cmap="Blues"); axes[0].set_title("Confusion matrix (rows = truth)"); axes[0].set_xlabel("predicted"); axes[0].set_ylabel("true"); axes[0].set_xticks(range(10)); axes[0].set_yticks(range(10)); axes[0].grid(False)
for i in range(10):
    for j in range(10):
        if cm[i, j]: axes[0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
wrong = np.where(pred_d != yd_te)[0][:8]
axes[1].axis("off"); axes[1].set_title("Some mistakes (true → predicted)")
for n, i in enumerate(wrong):
    ax_ = fig.add_axes([0.52 + (n % 4) * 0.11, 0.52 - (n // 4) * 0.34, 0.1, 0.28]); ax_.imshow(Xd_te[i].reshape(8, 8), cmap="gray_r"); ax_.set_title(f"{yd_te[i]} → {pred_d[i]}", fontsize=9); ax_.axis("off")
plt.show()
print("Metric comparison on digits (k=3):")
for met in ["euclidean", "manhattan", "cosine"]:
    print(f"  {met:<10} test acc = {KNeighborsClassifier(3, metric=met, algorithm='brute').fit(Xd_tr, yd_tr).score(Xd_te, yd_te):.3f}")''')
    nb.callout("key", "Almost 99 % with *no training at all* by KNN is hard to beat when (a) similar examples really do share labels, and (b) there's plenty of labelled data nearby. "
                      "The few mistakes are exactly the ones a human would find ambiguous (a sloppy 8 that looks like a 1). The price: every prediction compares the image with *all* 1 257 stored examples.")

    # ---------------------------------------------------------- 7 imbalance
    nb.section("🚨", "7. Imbalanced data: where KNN struggles",
               "A vote favours whoever is most numerous. With 2 % positives, a rare class rarely wins a neighbourhood vote, and the bigger k, the worse it gets.")
    nb.code(r'''Xi, yi = make_classification(n_samples=5000, n_features=10, n_informative=5, weights=[0.98, 0.02], flip_y=0.01, class_sep=1.2, random_state=1)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.3, stratify=yi, random_state=RANDOM_STATE)
rows = []
for k in [1, 5, 15, 25]:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(Xi_tr, yi_tr); pr = m.predict_proba(Xi_te)[:, 1]
    for tau in [0.5, 0.2]:
        p = (pr >= tau).astype(int); rows.append({"k": k, "threshold": tau, "recall": recall_score(yi_te, p), "precision": precision_score(yi_te, p, zero_division=0), "PR-AUC": average_precision_score(yi_te, pr)})
display(pd.DataFrame(rows).round(3))''')
    nb.md("At the default threshold 0.5 a positive needs a **majority** of its k neighbours to be positive: nearly impossible when 98 % of all points are negative. Recall collapses as k grows. Fixes:\n"
          "1. **Lower the threshold** (to ≈ the base rate region): the table shows recall jumping at τ = 0.2.\n2. Use `weights='distance'` so very close positives count more.\n"
          "3. Resample/undersample the majority class, or choose a smaller k.\n4. Prefer a model that handles weights natively (trees, logistic regression with `class_weight`).")

    # ---------------------------------------------------------- 8 curse & speed
    nb.section("🌌", "8. Curse of dimensionality, speed and probabilities",
               "**Irrelevant features** poison distances (Lesson 07). Let's append noise columns to the standardised cancer data and compare three classifiers:")
    nb.code(r'''rng = np.random.default_rng(0); rows = []
for n_noise in [0, 10, 30, 90]:
    A = np.c_[Ztr, rng.normal(size=(len(Ztr), n_noise))]; B = np.c_[Zte, rng.normal(size=(len(Zte), n_noise))]
    rows.append({"noise features": n_noise, "KNN (k=7)": KNeighborsClassifier(7).fit(A, yb_tr).score(B, yb_te),
                 "Logistic regression": LogisticRegression(max_iter=3000).fit(A, yb_tr).score(B, yb_te),
                 "Random forest": RandomForestClassifier(150, random_state=0, n_jobs=-1).fit(A, yb_tr).score(B, yb_te)})
res = pd.DataFrame(rows).set_index("noise features"); display(res.round(3)); res.plot(marker="o"); plt.ylabel("test accuracy"); plt.title("KNN degrades most once noise dominates (90 junk features)"); plt.show()''')
    nb.md("**Speed.** KNN stores everything (memory O(n·p)) and each prediction costs O(n·p) by brute force: fine for thousands of rows, painful for millions. "
          "kd-/ball-trees help in low dimensions; approximate-nearest-neighbour libraries (FAISS, Annoy) handle huge data.\n\n"
          "**Probabilities.** KNN's probability is a vote share, so it can only take values $0, \\frac1k, \\dots, 1$: coarse for small k. Larger k or distance weighting gives finer values.")
    nb.code(r'''for k in [1, 5, 25]:
    m = KNeighborsClassifier(k).fit(Ztr, yb_tr); pr = m.predict_proba(Zte)[:, 1]
    print(f"k={k:<3} distinct probability values on the test set: {len(np.unique(pr.round(6))):>3}  (max possible: {k + 1})")''')

    # ---------------------------------------------------------- 9 profile
    nb.section("📋", "9. Cheat-sheet: the full profile of KNN Classification")
    nb.table([
        ["**Underlying concept**", "Classify by a (weighted) majority vote among the k most similar training points."],
        ["**How it works**", "Store all training data; for each query compute distances, take the k smallest, vote (vote shares = probabilities)."],
        ["**Mathematical foundation**", "Non-parametric density/probability estimation; Minkowski/cosine distances; Cover–Hart bound (1-NN error ≤ 2× Bayes); consistency as k→∞, k/n→0."],
        ["**Important parameters**", "`n_neighbors` (k), `weights` (uniform/distance), `metric`/`p`, `algorithm` (brute/kd_tree/ball_tree/auto), `leaf_size`. Scale features!"],
        ["**Advantages**", "Trivial to understand and implement; no training; flexible non-linear boundaries; naturally multi-class; strong on local-structure problems (digits, recommendations); explains itself with neighbours."],
        ["**Disadvantages**", "Slow/memory-hungry prediction; needs scaling; hurt by irrelevant & high-dimensional features; sensitive to class imbalance and to k; no learned model or feature importance."],
        ["**Complexity**", "Fit O(1) (or O(n log n) for tree indexes); predict O(n·p) per query brute force (≈O(log n) with trees in low dimensions); memory O(n·p)."],
        ["**Use cases**", "Small/medium datasets with meaningful distances: digits/image patches, recommenders, anomaly detection, geospatial lookups, quick baselines."],
        ["**Limitations**", "Doesn't scale to big data without approximate indexes; poor with many noisy features or mixed categorical data without a custom metric; majority-class bias."],
    ], ["Question", "Answer"])

    # ---------------------------------------------------------- 10 worked
    nb.section("🧪", "10. Worked examples")
    nb.md(r"""### Worked Example 1: Weighted vs unweighted

Query with 5 nearest neighbours: distances $[0.5, 1, 1, 2, 2]$ and classes $[B, A, A, B, B]$.
*Unweighted:* B has 3 votes, A has 2 → **B**. *Inverse-distance weights* $[2, 1, 1, 0.5, 0.5]$: A total $=1+1=2$; B total $=2+0.5+0.5=3$ → still **B** (the closest point is B). Change the closest point to A (classes $[A,A,A,B,B]$): unweighted A=3 vs B=2 → A; weighted A $=2+1+1=4$ vs B $=1$ → A with a much stronger margin (80 % vs 60 %).""")
    nb.code(r'''d_ = np.array([0.5, 1, 1, 2, 2]); w_ = 1 / d_
for c_ in [np.array(["B", "A", "A", "B", "B"]), np.array(["A", "A", "A", "B", "B"])]:
    print(list(c_), "-> weighted shares", {c: round(float(w_[c_ == c].sum() / w_.sum()), 3) for c in ["A", "B"]}, "| unweighted", {c: round(float((c_ == c).mean()), 3) for c in ["A", "B"]})''')
    nb.md(r"""### Worked Example 2: Why the threshold matters for KNN on rare classes

A KNN with k = 10 has 2 positive neighbours out of 10 for a query: $\hat P(+)=0.2$. At the default threshold 0.5 → predict negative. If the positive class is 2 % of the population, a local share of 20 % is **ten times the base rate**: a strong signal that the default threshold ignores. Setting $\tau=0.15$ flags it.""")

    # ---------------------------------------------------------- problems
    nb.problem(
        "Classify by hand (in code)",
        "Write `knn_classify(xq, X, y, k, weighted=False)` (Euclidean distance, no scikit-learn) returning `(label, vote_shares_dict)` where the dict maps each class to its (weighted) vote share. "
        "Use the lesson's `pts`, `lab` and `q = [3, 3]`: store `res3 = knn_classify(q, pts, lab, 3)` and `res5 = knn_classify(q, pts, lab, 5)` and `res5w = knn_classify(q, pts, lab, 5, weighted=True)`.",
        ["Distances: `np.sqrt(((X - xq) ** 2).sum(axis=1))`; neighbours: `np.argsort(d)[:k]`.",
         "Votes: weights `w = np.ones(k)` (or `1 / d[idx]` if weighted); share of class c = `w[y[idx] == c].sum() / w.sum()`.",
         "Return `max(shares, key=shares.get)` as the label. Expected: k=3 → A (2 of 3 are A), k=5 → B."],
        'def knn_classify(xq, X, y, k, weighted=False):\n    pass\n\nres3 = res5 = res5w = None   # your code\n',
        'def knn_classify(xq, X, y, k, weighted=False):\n    d = np.sqrt(((X - xq) ** 2).sum(axis=1)); idx = np.argsort(d)[:k]\n    w = 1 / np.maximum(d[idx], 1e-12) if weighted else np.ones(k)\n    shares = {str(c): float(w[y[idx] == c].sum() / w.sum()) for c in np.unique(y)}\n    return max(shares, key=shares.get), shares\n\n'
        'res3 = knn_classify(q, pts, lab, 3); res5 = knn_classify(q, pts, lab, 5); res5w = knn_classify(q, pts, lab, 5, weighted=True)\nprint(res3, res5, res5w)',
        'check("k=3 -> A (2 of 3 neighbours)", lambda: res3[0] == "A" and abs(res3[1]["A"] - 2 / 3) < 1e-9)\ncheck("k=5 -> B (3 of 5 neighbours)", lambda: res5[0] == "B" and abs(res5[1]["B"] - 0.6) < 1e-9)\ncheck("shares sum to 1", lambda: abs(sum(res5w[1].values()) - 1) < 1e-9)',
        explain="The prediction flips between k = 3 and k = 5 because the 4th and 5th nearest points are class B: the very sensitivity to k that cross-validation is for.")

    nb.problem(
        "Choose k by cross-validation",
        "Using the scaled cancer data `Ztr, yb_tr, Zte, yb_te`: evaluate odd `k` from 1 to 31 with `StratifiedKFold(5, shuffle=True, random_state=0)` accuracy on `Ztr` → dict `cv_acc`. "
        "Store `best_k` (highest CV accuracy; smallest k on ties) and `test_acc` for that k fit on all of `Ztr`.",
        ["`cross_val_score(KNeighborsClassifier(k), Ztr, yb_tr, cv=skf).mean()` for each k.",
         "`best_k = min(k for k, v in cv_acc.items() if v == max(cv_acc.values()))`.",
         "Then `KNeighborsClassifier(best_k).fit(Ztr, yb_tr).score(Zte, yb_te)`."],
        'cv_acc = {}\nbest_k = test_acc = None   # your code\n',
        'cv_acc = {k: cross_val_score(KNeighborsClassifier(k), Ztr, yb_tr, cv=skf).mean() for k in range(1, 32, 2)}\nbest_k = min(k for k, v in cv_acc.items() if v == max(cv_acc.values()))\ntest_acc = KNeighborsClassifier(best_k).fit(Ztr, yb_tr).score(Zte, yb_te)\nprint(best_k, round(cv_acc[best_k], 3), round(test_acc, 3))',
        'check("16 odd values of k tested", lambda: len(cv_acc) == 16)\ncheck("k=1 is not the best", lambda: best_k != 1)\ncheck("test accuracy > 0.94", lambda: test_acc > 0.94)',
        explain="k = 1 is noisy; a handful of neighbours (5–11) gives the best generalisation. (The numerical differences between k = 5, 7, 9 are tiny; within noise.)")

    nb.problem(
        "Price of ignoring scale",
        "Compute `acc_raw` and `acc_scaled`: test accuracy of `KNeighborsClassifier(5)` on the **raw** cancer features (`Xb_tr`/`Xb_te`) and on the **scaled** ones (`Ztr`/`Zte`). "
        "Also find `dominant` = the name (from `bc.feature_names`) of the raw feature with the largest standard deviation in `Xb_tr`: the one that dominates raw distances.",
        ["Fit twice with `k = 5` and use `.score(X_test, y_test)`.",
         "`bc.feature_names[Xb_tr.std(axis=0).argmax()]`.",
         "Scaling should help by a few points; the dominant feature is an 'area' measurement."],
        'acc_raw = acc_scaled = dominant = None   # your code\n',
        'acc_raw = KNeighborsClassifier(5).fit(Xb_tr, yb_tr).score(Xb_te, yb_te); acc_scaled = KNeighborsClassifier(5).fit(Ztr, yb_tr).score(Zte, yb_te)\ndominant = str(bc.feature_names[Xb_tr.std(axis=0).argmax()])\nprint(acc_raw, acc_scaled, dominant)',
        'check("scaling improves accuracy by at least 0.02", lambda: acc_scaled - acc_raw >= 0.02)\ncheck("the dominant raw feature is an area measurement", lambda: "area" in dominant)',
        explain="'worst area' reaches the thousands, so it essentially *is* the distance between two tumours in raw space: the other 29 features are a rounding error.")

    nb.problem(
        "Read the digit confusions",
        "With the lesson's `cm` (confusion matrix of the k = 3 digit classifier): set `accuracy_d` = overall accuracy computed from `cm` (trace / total), "
        "`worst_pair` = the tuple `(true_digit, predicted_digit)` of the **largest off-diagonal** entry (first one if tied), and `n_worst` = its count. "
        "Also `per_class_recall` = a length-10 array of recall for each digit (diagonal / row sum) and `hardest_digit` = the digit with the lowest recall.",
        ["`np.trace(cm) / cm.sum()`.",
         "Copy `cm`, set its diagonal to 0 with `np.fill_diagonal`, then `np.unravel_index(off.argmax(), off.shape)`.",
         "`per_class_recall = np.diag(cm) / cm.sum(axis=1)`; `hardest_digit = int(per_class_recall.argmin())`."],
        'accuracy_d = worst_pair = n_worst = per_class_recall = hardest_digit = None   # your code\n',
        'accuracy_d = np.trace(cm) / cm.sum()\noff = cm.copy(); np.fill_diagonal(off, 0); worst_pair = tuple(int(v) for v in np.unravel_index(off.argmax(), off.shape)); n_worst = int(off.max())\n'
        'per_class_recall = np.diag(cm) / cm.sum(axis=1); hardest_digit = int(per_class_recall.argmin())\nprint(round(accuracy_d, 3), worst_pair, n_worst, per_class_recall.round(2), hardest_digit)',
        'check("overall accuracy ≈ 0.987", lambda: abs(accuracy_d - knn_d.score(Xd_te, yd_te)) < 1e-9)\ncheck("worst pair is a plausible confusion (two different digits)", lambda: worst_pair[0] != worst_pair[1])\ncheck("per-class recall has 10 entries in [0, 1]", lambda: len(per_class_recall) == 10 and ((per_class_recall >= 0) & (per_class_recall <= 1)).all())\ncheck("hardest digit has the minimum recall", lambda: per_class_recall[hardest_digit] == per_class_recall.min())',
        explain="Overall accuracy hides per-class differences: some digits are recognised perfectly, a couple a bit less so. Always read the confusion matrix, not just the headline number.",
        level="⭐⭐")

    nb.problem(
        "Rescue a rare class with the threshold",
        "On the fraud data (`Xi_tr, yi_tr, Xi_te, yi_te`) fit `make_pipeline(StandardScaler(), KNeighborsClassifier(15))`. Using `predict_proba`, store the recall and precision at the default threshold (`rec_50`, `prec_50`) and at `τ = 0.1` (`rec_10`, `prec_10`). "
        "Set `tradeoff_ok = True` if recall went **up** and precision went **down**.",
        ["`pr = model.predict_proba(Xi_te)[:, 1]`; predictions are `(pr >= tau).astype(int)`.",
         "Use `zero_division=0` in `precision_score` in case nothing is predicted positive.",
         "Lowering the threshold increases recall and typically decreases precision."],
        'rec_50 = prec_50 = rec_10 = prec_10 = tradeoff_ok = None   # your code\n',
        'knn_f = make_pipeline(StandardScaler(), KNeighborsClassifier(15)).fit(Xi_tr, yi_tr); pr = knn_f.predict_proba(Xi_te)[:, 1]\n'
        'p50, p10 = (pr >= 0.5).astype(int), (pr >= 0.1).astype(int)\nrec_50, prec_50 = recall_score(yi_te, p50), precision_score(yi_te, p50, zero_division=0)\nrec_10, prec_10 = recall_score(yi_te, p10), precision_score(yi_te, p10, zero_division=0)\n'
        'tradeoff_ok = bool(rec_10 > rec_50 and prec_10 < prec_50)\nprint(rec_50, prec_50, rec_10, prec_10, tradeoff_ok)',
        'check("recall at τ=0.5 is poor (< 0.2)", lambda: rec_50 < 0.2)\ncheck("lowering τ to 0.1 multiplies recall (> 0.4)", lambda: rec_10 > 0.4)\ncheck("recall up, precision down", lambda: tradeoff_ok)',
        explain="With k = 15 almost no positive wins a majority vote. Treating the vote share as a *score* and thresholding at a lower value recovers most positives: at the cost of many false alarms: exactly the precision/recall trade-off of Lesson 09.",
        level="⭐⭐")

    nb.problem(
        "Which distance metric wins on digits?",
        "Fit `KNeighborsClassifier(3, metric=m, algorithm='brute')` on `Xd_tr, yd_tr` for `m` in `['euclidean', 'manhattan', 'cosine', 'chebyshev']` and store the test accuracies in `metric_acc` (dict). "
        "Set `best_metric` (highest accuracy; first in list on ties) and `spread` = max − min accuracy.",
        ["Loop over the metric names and use `.score(Xd_te, yd_te)`.",
         "`best_metric = max(metric_acc, key=metric_acc.get)`.",
         "Chebyshev (largest single-pixel difference) is a poor fit for images."],
        'metric_acc = {}\nbest_metric = spread = None   # your code\n',
        'metric_acc = {m: KNeighborsClassifier(3, metric=m, algorithm="brute").fit(Xd_tr, yd_tr).score(Xd_te, yd_te) for m in ["euclidean", "manhattan", "cosine", "chebyshev"]}\nbest_metric = max(metric_acc, key=metric_acc.get); spread = max(metric_acc.values()) - min(metric_acc.values())\nprint({k: round(v, 3) for k, v in metric_acc.items()}, best_metric, round(spread, 3))',
        'check("four metrics evaluated", lambda: len(metric_acc) == 4)\ncheck("Chebyshev is the worst metric for images", lambda: min(metric_acc, key=metric_acc.get) == "chebyshev")\ncheck("the best metric reaches > 0.98", lambda: metric_acc[best_metric] > 0.98)',
        explain="Euclidean, Manhattan and cosine are all excellent on pixel data; Chebyshev, which looks only at the single most-different pixel, throws away most of the information. 'Similarity' must be chosen to fit the problem.",
        level="⭐⭐")

    nb.quiz([
        ("Why choose an odd k for binary classification?", "To avoid tied votes."),
        ("Why does a larger k make KNN worse on very imbalanced data?",
         "A positive needs a majority of its neighbours to be positive; with few positives, larger neighbourhoods are dominated by the majority class, so almost nothing is predicted positive."),
        ("You doubled the number of training rows. What happens to KNN's training time and prediction time?",
         "Training time stays ≈ 0 (it only stores data); prediction time per query roughly doubles (brute force)."),
        ("Which of these would hurt KNN more: 10 irrelevant extra features, or 10 irrelevant extra *rows*? Why?",
         "Extra features: every distance gets contaminated by noise dimensions. Extra rows only add (harmlessly unhelpful) candidates."),
        ("When is cosine distance preferable to Euclidean?",
         "When the direction of the vector matters more than its magnitude: e.g. text/TF-IDF vectors or embeddings, where document length shouldn't affect similarity."),
    ])

    nb.footer(["KNN classifies by a vote among the k nearest training points; vote shares are (coarse) probabilities.",
               "Small k → jagged, high variance; large k → smooth, high bias; choose k (odd) by cross-validation.",
               "Scale your features and choose a distance that fits your data (Euclidean, Manhattan, cosine).",
               "Excellent on local-structure problems like digits (≈ 98.7 %), weak on imbalanced, noisy or high-dimensional data.",
               "No training, but memory-heavy and O(n·p) per prediction; thresholds are the cure for rare classes."],
              "Naive Bayes", "14_Naive_Bayes.ipynb")
    return nb
