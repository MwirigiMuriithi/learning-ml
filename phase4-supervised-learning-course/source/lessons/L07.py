from nbkit import Lesson


def make():
    nb = Lesson("07_KNN_Regression", 7, "K-Nearest Neighbors Regression",
                "No training, no equations: just 'what did the most similar cases sell for?'",
                100, "🟢🟡 Beginner → Intermediate",
                setup_extra="from sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.ensemble import RandomForestRegressor\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error, r2_score\nfrom sklearn.model_selection import train_test_split, cross_val_score, validation_curve, KFold")
    nb.header(
        objectives=[
            "Describe KNN as a **lazy, instance-based** learner and walk through a prediction by hand.",
            "Compare Euclidean, Manhattan and Minkowski distances and uniform vs distance-weighted voting.",
            "Implement KNN regression **from scratch** with vectorised NumPy and match scikit-learn exactly.",
            "Explain the effect of `k` on bias and variance, and choose it with cross-validation.",
            "Show why **feature scaling** is non-negotiable for KNN.",
            "Demonstrate the **curse of dimensionality** and its effect on KNN.",
            "Explain KNN's cost: O(1) 'training', O(n·p) brute-force prediction, and when kd-trees help.",
        ],
        prereqs="Lessons 00, 04; distance in Euclidean space",
        hook=("When you list a flat for sale, the agent does something very human: she finds **comparables**: \"the three most similar flats sold nearby recently\": "
              "and quotes roughly their average price. She doesn't fit a formula to the entire city.\n\n"
              "That is **K-Nearest Neighbors**: to predict for a new example, find the *k* most similar training examples and average their answers. "
              "It is the simplest machine-learning algorithm there is, and a wonderful way to understand *what 'similar' means* in data, "
              "because the whole model rests on one idea: **distance**."),
        roadmap=["The algorithm in one paragraph", "Distances & weighting", "A hand-worked prediction",
                 "From scratch (and verified)", "Choosing k: bias vs variance", "Scaling is everything",
                 "Show me the comps (explainability)", "The curse of dimensionality", "Speed: brute force vs kd-trees",
                 "Cheat-sheet, worked examples, problems, quiz"])

    # ---------------------------------------------------------- 1
    nb.section("🧠", "1. The algorithm in one paragraph",
               "**Training:** memorise the data. (That's it; KNN is a *lazy learner*; all the work happens at prediction time.)\n\n"
               "**Prediction for a new point $x_q$:**\n\n"
               "1. Compute the distance from $x_q$ to **every** training point.\n"
               "2. Pick the **k** closest (the neighbourhood $N_k(x_q)$).\n"
               "3. Predict the **average** of their targets:\n\n"
               r"$$\hat y(x_q)=\frac1k\sum_{i\in N_k(x_q)}y_i\qquad\text{or, distance-weighted:}\qquad \hat y(x_q)=\frac{\sum_{i\in N_k}w_i\,y_i}{\sum_{i\in N_k}w_i},\ \ w_i=\frac1{d(x_q,x_i)}$$" "\n\n"
               "KNN makes **no assumption** about the shape of the relationship (it is *non-parametric*): the data *is* the model. Watch it bend to a curve:")
    nb.code(r'''rng = np.random.default_rng(1)
x1 = np.sort(rng.uniform(0, 10, 100)); y1 = np.sin(x1) + 0.3 * x1 + rng.normal(0, 0.3, 100)
grid = np.linspace(0, 10, 500).reshape(-1, 1)
fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), sharey=True)
for ax, k in zip(axes, [1, 5, 20, 100]):
    m = KNeighborsRegressor(k).fit(x1.reshape(-1, 1), y1)
    ax.scatter(x1, y1, s=12, alpha=0.6); ax.plot(grid, m.predict(grid), "r", lw=2); ax.set_title(f"k = {k}")
plt.tight_layout(); plt.show()''')
    nb.md("- **k = 1**: jagged: passes through every training point (memorisation, high variance).\n- **k = 5**: smooth but faithful.\n"
          "- **k = 20**: smoother, starting to flatten the peaks (bias creeping in).\n- **k = 100 = n**: every prediction is the global mean: a flat line (maximum bias).\n\n"
          "So **k is the complexity dial**: small k = flexible/high variance; large k = rigid/high bias.")

    # ---------------------------------------------------------- 2 distance
    nb.section("📏", "2. Distances and weighting",
               "KNN's notion of 'similar' is a distance. The **Minkowski** family covers the common choices:")
    nb.md(r"""$$d_p(\mathbf a,\mathbf b)=\Big(\sum_{j=1}^{p_{\text{dim}}}\lvert a_j-b_j\rvert^{\,p}\Big)^{1/p}$$

| p | Name | Intuition |
|---|---|---|
| 1 | **Manhattan** (city-block) | Walk along grid streets; less dominated by one large coordinate gap; good in high dimensions |
| 2 | **Euclidean** (default) | Straight-line distance |
| ∞ | Chebyshev | Largest single coordinate difference |

**Weighting:** `weights='uniform'` gives every neighbour an equal vote; `weights='distance'` gives closer neighbours more say (inverse-distance), which softens the sharp edges of the k-th neighbour boundary.""")

    # ---------------------------------------------------------- 3 hand
    nb.section("✍️", "3. A hand-worked prediction",
               "Five houses (size in 100 sqft → price in millions). Predict the price of a **400 sqft** (x = 4) house with **k = 3**.")
    nb.code(r'''xs = np.array([1, 2, 3, 10, 11.]); ys = np.array([10, 12, 14, 40, 44.]); xq = 4.0
d = np.abs(xs - xq)
order = np.argsort(d)
table = pd.DataFrame({"x": xs, "price": ys, "distance to 4": d}).iloc[order].reset_index(drop=True)
table["rank"] = table.index + 1
display(table)
k = 3
print("3 nearest:", ys[order[:k]], "-> uniform average =", ys[order[:k]].mean())
w = 1 / d[order[:k]]
print("inverse-distance weights:", w.round(3), "-> weighted average =", round(np.sum(w * ys[order[:k]]) / w.sum(), 3))''')
    nb.md("The three nearest neighbours are x = 3, 2, 1 (prices 14, 12, 10) → **uniform prediction 12.0**. With distance weights the nearest (x = 3, distance 1) counts three times as much as x = 1 (distance 3) → **12.73**. "
          "Notice what the model *cannot* do: the two expensive houses at x = 10–11 are far away and have **no** influence.")

    # ---------------------------------------------------------- 4 scratch
    nb.section("🔧", "4. Build it from scratch",
               "The only hard part is computing all pairwise distances **efficiently**. We use broadcasting (no Python loops over points), process queries in chunks to save memory, "
               "and use `np.argpartition` to find the k smallest distances without a full sort.")
    nb.code(r'''class KNNRegressorScratch:
    """Brute-force k-nearest-neighbours regression (uniform or inverse-distance weighting, Minkowski distance)."""

    def __init__(self, k=5, weights="uniform", p=2, chunk=200):
        self.k, self.weights, self.p, self.chunk = k, weights, p, chunk

    def fit(self, X, y):                                   # "training" = memorising
        self.X_, self.y_ = np.asarray(X, float), np.asarray(y, float)
        return self

    def _distances(self, Q):                               # (n_query x n_train) Minkowski distances
        diff = np.abs(Q[:, None, :] - self.X_[None, :, :])
        return (diff ** self.p).sum(axis=2) ** (1 / self.p)

    def kneighbors(self, Q):
        Q = np.asarray(Q, float); all_d, all_i = [], []
        for s in range(0, len(Q), self.chunk):             # chunking keeps memory bounded
            D = self._distances(Q[s:s + self.chunk])
            idx = np.argpartition(D, self.k - 1, axis=1)[:, :self.k]                 # k smallest, unordered
            dk = np.take_along_axis(D, idx, axis=1)
            order = np.argsort(dk, axis=1)                                            # order them nearest-first
            all_d.append(np.take_along_axis(dk, order, axis=1)); all_i.append(np.take_along_axis(idx, order, axis=1))
        return np.vstack(all_d), np.vstack(all_i)

    def predict(self, Q):
        dist, idx = self.kneighbors(Q)
        yk = self.y_[idx]                                  # targets of the neighbours, shape (n_query, k)
        if self.weights == "uniform":
            return yk.mean(axis=1)
        w = 1 / np.maximum(dist, 1e-12)                    # inverse-distance weights (guard against d = 0)
        return (w * yk).sum(axis=1) / w.sum(axis=1)


# sanity check on the hand example
print("scratch prediction for x=4, k=3:", KNNRegressorScratch(3).fit(xs.reshape(-1, 1), ys).predict([[4.0]]),
      "(weighted:", KNNRegressorScratch(3, "distance").fit(xs.reshape(-1, 1), ys).predict([[4.0]]).round(3), ")")''')
    nb.md("Now the real test: our 30 lines versus scikit-learn on the house data. (KNN needs scaled features: we'll see why in Section 6; here we just do it.)")
    nb.code(r'''houses = pd.read_csv(data_path("house_prices.csv"))
hc = houses[houses["sqft"] <= 6500].reset_index(drop=True)
num_cols = ["sqft", "bedrooms", "bathrooms", "age_years", "garage_spaces", "has_pool", "distance_to_cbd_km", "lot_size_sqft"]
Xall = pd.get_dummies(hc[num_cols + ["neighborhood"]], columns=["neighborhood"], dtype=float)
Xall["condition_ord"] = hc["condition"].map({"Poor": 0, "Fair": 1, "Good": 2, "Excellent": 3})
yall = hc["price_million_kes"]
Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.2, random_state=RANDOM_STATE)
med = Xtr.median(); Xtr, Xte = Xtr.fillna(med).astype(float), Xte.fillna(med).astype(float)
sc = StandardScaler().fit(Xtr); Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)       # scaler learned on TRAIN only

for w in ["uniform", "distance"]:
    mine = KNNRegressorScratch(k=7, weights=w).fit(Ztr, ytr.values)
    ref = KNeighborsRegressor(n_neighbors=7, weights=w).fit(Ztr, ytr)
    print(f"weights={w:<9} max |scratch - sklearn| = {np.abs(mine.predict(Zte) - ref.predict(Zte)).max():.2e}   test R² = {ref.score(Zte, yte):.3f}")''')

    # ---------------------------------------------------------- 5 choose k
    nb.section("🎚️", "5. Choosing k: the bias–variance dial",
               "Cross-validate over k on the (scaled) house data. The scaler lives **inside** a pipeline so each fold scales using only its own training part.")
    nb.code(r'''pipe = make_pipeline(StandardScaler(), KNeighborsRegressor())
ks = np.arange(1, 41)
tr, va = validation_curve(pipe, Xtr, ytr, param_name="kneighborsregressor__n_neighbors", param_range=ks,
                          cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
plt.plot(ks, tr.mean(1), "o-", ms=3, label="train R²"); plt.plot(ks, va.mean(1), "s-", ms=3, label="validation R²")
best_k = ks[va.mean(1).argmax()]; plt.axvline(best_k, color="gray", ls=":", label=f"best k = {best_k}")
plt.xlabel("k (number of neighbours)"); plt.ylabel("R²"); plt.legend(); plt.title("Small k: overfit (train ≈ 1). Large k: underfit."); plt.show()
print({int(k): round(float(v), 3) for k, v in zip(ks[[0, 2, 4, 9, 19, 39]], va.mean(1)[[0, 2, 4, 9, 19, 39]])})''')
    nb.callout("tip", "With k = 1 the training R² is exactly 1 (each point's nearest neighbour is itself): a warning sign that training score is meaningless for KNN. "
                      "Always judge KNN with **validation** data. For classification prefer an **odd** k to avoid ties; for regression any k is fine.")

    # ---------------------------------------------------------- 6 scaling
    nb.section("⚖️", "6. Scaling is everything",
               "KNN lives and dies by distance. Compare the raw scales: `lot_size_sqft` ≈ 1 000–30 000 while `bathrooms` ≈ 1–6.")
    nb.code(r'''print(Xtr[["sqft", "lot_size_sqft", "bathrooms", "age_years", "has_pool"]].describe().loc[["min", "max", "std"]].round(1))
unscaled = KNeighborsRegressor(7).fit(Xtr, ytr); scaled = make_pipeline(StandardScaler(), KNeighborsRegressor(7)).fit(Xtr, ytr)
print(f"\nKNN on RAW features   : test R² = {unscaled.score(Xte, yte):.3f}")
print(f"KNN on SCALED features: test R² = {scaled.score(Xte, yte):.3f}")''')
    nb.md("Without scaling, a 5 000-sqft difference in *lot size* swamps a 3-bedroom difference: the 'neighbourhood' is effectively decided by lot size alone. "
          "Standardisation puts every feature on a comparable footing.\n\n"
          "**Scaling is also a modelling decision**: after standardising, every feature has equal say. If you *know* some features matter more, you can scale them up (feature weighting). "
          "Scaling is *not* optional for: KNN, SVM, K-means, PCA, regularised regression, gradient descent. It is *unnecessary* for trees and forests.")

    # ---------------------------------------------------------- 7 comps
    nb.section("🔍", "7. Show me the comps: KNN explains itself",
               "A big advantage of KNN: every prediction comes with **evidence** you can show a client. Let's explain the prediction for the first test house.")
    nb.code(r'''knn5 = make_pipeline(StandardScaler(), KNeighborsRegressor(5)).fit(Xtr, ytr)
q = Xte.iloc[[0]]
dist, idx = knn5[-1].kneighbors(knn5[0].transform(q))
cols = ["neighborhood", "sqft", "bedrooms", "age_years", "has_pool", "condition", "price_million_kes"]
print("QUERY HOUSE:"); display(hc.loc[q.index, cols])
comps = hc.loc[Xtr.index[idx[0]], cols].copy(); comps["distance"] = dist[0].round(2)
print("ITS 5 'COMPARABLES':"); display(comps)
print(f"Prediction = mean of the comps = {comps['price_million_kes'].mean():.2f} M   |   model.predict = {knn5.predict(q)[0]:.2f} M   |   true price = {yte.loc[q.index[0]]:.2f} M")''')

    # ---------------------------------------------------------- 8 curse
    nb.section("🌌", "8. The curse of dimensionality",
               "KNN assumes *near points are similar*. In high dimensions that assumption collapses, for two reasons.\n\n"
               "**(a) Distances concentrate.** Draw random points in a d-dimensional cube; as d grows, the nearest and the farthest point from any query become almost equally far: "
               "'nearest' stops meaning anything.")
    nb.code(r'''rng = np.random.default_rng(0); dims = [1, 2, 5, 10, 20, 50, 100, 300, 1000]; ratio = []
for d in dims:
    r = []
    for _ in range(30):
        pts = rng.uniform(size=(500, d)); qy = rng.uniform(size=(1, d))
        dd = np.sqrt(((pts - qy) ** 2).sum(axis=1)); r.append((dd.max() - dd.min()) / dd.min())
    ratio.append(np.mean(r))
plt.plot(dims, ratio, "o-"); plt.xscale("log"); plt.yscale("log"); plt.xlabel("dimension d"); plt.ylabel("(farthest − nearest) / nearest")
plt.title("As d grows, 'nearest' and 'farthest' converge"); plt.show()''')
    nb.md("**(b) Neighbourhoods stop being local.** To capture just 10 % of uniformly spread data, a hyper-cubic neighbourhood needs side length $0.1^{1/d}$ of the range of *each* axis:")
    nb.code(r'''print(pd.DataFrame({"dimension d": [1, 2, 5, 10, 50, 100], "side length needed to capture 10% of data": [0.1 ** (1 / d) for d in [1, 2, 5, 10, 50, 100]]}).round(3).to_string(index=False))''')
    nb.md("At d = 100 you must span **98 %** of every axis to find 10 % of the data. 'Local' no longer exists. "
          "The practical consequence: **irrelevant features poison KNN**, because they add distance without information. Let's measure it: add pure-noise columns and compare three models:")
    nb.code(r'''rng = np.random.default_rng(0); rows = []
for n_noise in [0, 5, 10, 20, 40]:
    A, B = Xtr.copy(), Xte.copy()
    for j in range(n_noise): A[f"noise{j}"] = rng.normal(size=len(A)); B[f"noise{j}"] = rng.normal(size=len(B))
    knn = make_pipeline(StandardScaler(), KNeighborsRegressor(10)).fit(A, ytr)
    rf = RandomForestRegressor(60, max_features=0.33, min_samples_leaf=2, random_state=0, n_jobs=-1).fit(A, ytr)
    lin = make_pipeline(StandardScaler(), LinearRegression()).fit(A, np.log(ytr))
    rows.append({"noise features": n_noise, "KNN": knn.score(B, yte), "Random forest": rf.score(B, yte), "Linear (log)": r2_score(yte, np.exp(lin.predict(B)))})
res = pd.DataFrame(rows).set_index("noise features"); display(res.round(3)); res.plot(marker="o"); plt.ylabel("test R²"); plt.title("Irrelevant features hurt KNN most"); plt.show()''')
    nb.callout("key", "KNN degrades fastest as junk features pile up; forests are fairly robust (they ignore useless splits) and linear regression barely notices. "
                      "**Remedies:** feature selection, dimensionality reduction (PCA), or switching to a model that learns which features matter.")

    # ---------------------------------------------------------- 9 speed
    nb.section("⏱️", "9. Speed: brute force vs kd-trees",
               "Brute force costs **O(n·p) per query**: every prediction scans the whole training set. For many queries that's painful. "
               "Space-partitioning trees (`kd_tree`, `ball_tree`) organise the data so a query only visits nearby regions: about **O(log n)** per query, *but only in low dimensions* (≲ 15–20). "
               "`algorithm='auto'` chooses for you.")
    nb.code(r'''import time
rng = np.random.default_rng(0); rows = []
for n in [1000, 5000, 20000, 60000]:
    Xs = rng.normal(size=(n, 3)); ys_ = Xs.sum(axis=1); Q = rng.normal(size=(1000, 3))
    row = {"n_train": n}
    for alg in ["brute", "kd_tree"]:
        m = KNeighborsRegressor(5, algorithm=alg).fit(Xs, ys_)
        t0 = time.perf_counter(); m.predict(Q); row[f"{alg} predict 1000 queries (s)"] = time.perf_counter() - t0
    rows.append(row)
display(pd.DataFrame(rows).round(4))''')
    nb.md("Brute-force time grows roughly linearly with n; the kd-tree barely moves. In **high dimensions** (try p = 50) the tree's advantage disappears and it can even be slower than brute force.")

    # ---------------------------------------------------------- 10 profile
    nb.section("📋", "10. Cheat-sheet: the full profile of KNN Regression")
    nb.table([
        ["**Underlying concept**", "Predict the average target of the k most similar training examples."],
        ["**How it works**", "Store data → for a query compute distances to all points → take the k nearest → (weighted) average their targets."],
        ["**Mathematical foundation**", "Minkowski distances; local averaging (a non-parametric estimator); bias–variance controlled by k; curse of dimensionality."],
        ["**Important parameters**", "`n_neighbors` (k), `weights` (uniform/distance), `p` / `metric`, `algorithm` (brute/kd_tree/ball_tree), `leaf_size`. Always scale features."],
        ["**Advantages**", "Dead simple; no training; no assumptions about the function's shape; naturally non-linear; explains itself with 'comparables'; easy to update with new data."],
        ["**Disadvantages**", "Slow, memory-hungry prediction; needs scaling; hurt by irrelevant/high-dimensional features; sensitive to k and the distance metric; no extrapolation; no feature importance."],
        ["**Complexity**", "Fit O(1) (or O(n log n) to build a tree); predict O(n·p) brute force per query, ≈ O(log n) with trees in low dimensions; memory O(n·p)."],
        ["**Use cases**", "Small/medium datasets with few meaningful features; recommender-style 'similar items'; local interpolation (geo-spatial, sensor data); quick baselines."],
        ["**Limitations**", "Doesn't scale to big n or large p; poor with mixed feature types unless a good distance is defined; imbalanced density regions give inconsistent neighbourhood sizes."],
    ], ["Question", "Answer"])

    # ---------------------------------------------------------- 11 worked
    nb.section("🧪", "11. More worked examples")
    nb.md(r"""### Worked Example 1: Manhattan vs Euclidean

Points $a=(1,2,3)$ and $b=(4,0,3)$. Manhattan: $|1-4|+|2-0|+|3-3|=5$. Euclidean: $\sqrt{9+4+0}=\sqrt{13}\approx3.606$. Chebyshev: $\max(3,2,0)=3$. For the same pair, $d_1\ge d_2\ge d_\infty$ always.""")
    nb.code(r'''a, b = np.array([1, 2, 3]), np.array([4, 0, 3])
print("Manhattan:", np.abs(a - b).sum(), "| Euclidean:", round(np.sqrt(((a - b) ** 2).sum()), 3), "| Chebyshev:", np.abs(a - b).max())''')
    nb.md(r"""### Worked Example 2: Why scaling flips the neighbour

Query $q=(sqft{=}1500,\ bedrooms{=}3)$. Candidates: $A=(1500,\ 6)$ and $B=(1590,\ 3)$.

*Raw Euclidean:* $d(q,A)=\sqrt{0+9}=3$, $d(q,B)=\sqrt{90^2+0}=90$ → **A is "nearer"** even though it has double the bedrooms!
*After scaling* (say sqft std = 600, bedrooms std = 1): $d(q,A)=3.0$, $d(q,B)=0.15$ → **B is nearer**, which matches human judgement. Unscaled KNN let the unit of measurement decide similarity.""")

    # ---------------------------------------------------------- problems
    nb.problem(
        "Three distances",
        "Write `minkowski(a, b, p)` for NumPy vectors. Use it to compute `d1`, `d2`, `dinf` for `a = [1, 5, 2, 8]` and `b = [4, 1, 2, 3]` "
        "(for ∞ use `np.abs(a - b).max()`). Then set `ordered = True` if `d1 >= d2 >= dinf`.",
        [r"$d_p=\big(\sum |a_j-b_j|^p\big)^{1/p}$ → `(np.abs(a - b) ** p).sum() ** (1 / p)`.",
         "Expected: d1 = 12, d2 = √50 ≈ 7.071, d∞ = 5.",
         "`ordered = d1 >= d2 >= dinf` (Python chains comparisons)."],
        'def minkowski(a, b, p):\n    pass\n\na = np.array([1, 5, 2, 8.]); b = np.array([4, 1, 2, 3.])\nd1 = d2 = dinf = ordered = None   # your code\n',
        'def minkowski(a, b, p):\n    return (np.abs(a - b) ** p).sum() ** (1 / p)\n\na = np.array([1, 5, 2, 8.]); b = np.array([4, 1, 2, 3.])\nd1 = minkowski(a, b, 1); d2 = minkowski(a, b, 2); dinf = np.abs(a - b).max(); ordered = bool(d1 >= d2 >= dinf)\nprint(d1, d2, dinf, ordered)',
        'check("Manhattan = 12", lambda: abs(d1 - 12) < 1e-9)\ncheck("Euclidean = sqrt(50)", lambda: abs(d2 - np.sqrt(50)) < 1e-9)\ncheck("Chebyshev = 5", lambda: abs(dinf - 5) < 1e-9)\ncheck("d1 >= d2 >= dinf", lambda: ordered)',
        explain="The three metrics measure the same gap differently: a sum, a root-sum-of-squares, and the single worst coordinate.")

    nb.problem(
        "Predict one point by hand (in code)",
        "Write `knn_predict_one(xq, X, y, k, weighted=False)` for 1-D or n-D NumPy data (Euclidean distance): no scikit-learn. "
        "Use it on the lesson's `xs, ys` (reshape `xs` to a column) with query `[4.0]`, k = 3, to fill `pred_uniform` and `pred_weighted` "
        "(expected 12.0 and ≈ 12.727), and for query `[10.5]`, k = 2 to fill `pred_far`.",
        ["Distances: `np.sqrt(((X - xq) ** 2).sum(axis=1))`.",
         "`idx = np.argsort(d)[:k]`; uniform → `y[idx].mean()`.",
         "Weighted: `w = 1 / d[idx]`; `(w * y[idx]).sum() / w.sum()`."],
        'def knn_predict_one(xq, X, y, k, weighted=False):\n    pass\n\nXc = xs.reshape(-1, 1)\npred_uniform = pred_weighted = pred_far = None   # your code\n',
        'def knn_predict_one(xq, X, y, k, weighted=False):\n    d = np.sqrt(((X - xq) ** 2).sum(axis=1)); idx = np.argsort(d)[:k]\n    if not weighted:\n        return y[idx].mean()\n    w = 1 / np.maximum(d[idx], 1e-12)\n    return (w * y[idx]).sum() / w.sum()\n\n'
        'Xc = xs.reshape(-1, 1)\npred_uniform = knn_predict_one(np.array([4.0]), Xc, ys, 3); pred_weighted = knn_predict_one(np.array([4.0]), Xc, ys, 3, weighted=True)\n'
        'pred_far = knn_predict_one(np.array([10.5]), Xc, ys, 2)\nprint(pred_uniform, pred_weighted, pred_far)',
        'check("uniform = 12.0", lambda: abs(pred_uniform - 12.0) < 1e-9)\ncheck("weighted ≈ 12.727", lambda: abs(pred_weighted - 12.7273) < 1e-3)\ncheck("query 10.5 with k=2 -> mean of 40 and 44 = 42", lambda: abs(pred_far - 42) < 1e-9)',
        explain="This function is the entire KNN prediction rule. The scratch class in Section 4 is the same thing, vectorised for many queries at once.",
        level="⭐⭐")

    nb.problem(
        "Choose k by cross-validation",
        "Using `Xtr, ytr, Xte, yte`: for `k` in 1…30, compute the mean 5-fold CV R² (`KFold(5, shuffle=True, random_state=0)`) of `make_pipeline(StandardScaler(), KNeighborsRegressor(k))`. "
        "Store the scores in `cv_scores` (dict k → R²), `best_k`, and the **test R²** of the best model fit on all training data in `test_r2`.",
        ["Loop over k and use `cross_val_score(..., scoring='r2').mean()`.",
         "`best_k = max(cv_scores, key=cv_scores.get)`.",
         "Fit `make_pipeline(StandardScaler(), KNeighborsRegressor(best_k)).fit(Xtr, ytr)` then `.score(Xte, yte)`."],
        'cv_scores = {}\nbest_k = test_r2 = None   # your code\n',
        'cv_scores = {k: cross_val_score(make_pipeline(StandardScaler(), KNeighborsRegressor(k)), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean() for k in range(1, 31)}\n'
        'best_k = max(cv_scores, key=cv_scores.get)\ntest_r2 = make_pipeline(StandardScaler(), KNeighborsRegressor(best_k)).fit(Xtr, ytr).score(Xte, yte)\nprint(best_k, round(cv_scores[best_k], 3), round(test_r2, 3))',
        'check("30 values of k evaluated", lambda: len(cv_scores) == 30)\ncheck("best k is small-to-moderate (2-10)", lambda: 2 <= best_k <= 10)\ncheck("k=1 is worse than the best k", lambda: cv_scores[1] < cv_scores[best_k])\ncheck("test R² > 0.85", lambda: test_r2 > 0.85)',
        explain="The CV curve is U-shaped: k = 1 chases noise, large k blurs distinct neighbourhoods. The optimum (3-ish here) is small because the dataset has strong local structure (neighbourhood × size).",
        level="⭐⭐")

    nb.problem(
        "Quantify the cost of not scaling",
        "Using `KNeighborsRegressor(n_neighbors=7)`: fit one on raw `Xtr` and one inside `make_pipeline(StandardScaler(), …)`. Store test R² as `r2_raw`, `r2_scaled`. "
        "Then find the **single most dominant raw feature**: set `dominant` to the column of `Xtr` with the largest standard deviation.",
        ["`.score(Xte, yte)` for both.",
         "`Xtr.std().idxmax()` returns the name of the highest-variance column.",
         "That feature gets the largest say in raw Euclidean distance."],
        'r2_raw = r2_scaled = dominant = None   # your code\n',
        'r2_raw = KNeighborsRegressor(7).fit(Xtr, ytr).score(Xte, yte)\nr2_scaled = make_pipeline(StandardScaler(), KNeighborsRegressor(7)).fit(Xtr, ytr).score(Xte, yte)\ndominant = Xtr.std().idxmax()\nprint(r2_raw, r2_scaled, dominant)',
        'check("scaling improves R² by more than 0.3", lambda: r2_scaled - r2_raw > 0.3)\ncheck("lot_size_sqft dominates the raw distances", lambda: dominant == "lot_size_sqft")',
        explain="lot_size_sqft has a standard deviation in the thousands, so in raw Euclidean distance the other features are almost invisible.")

    nb.problem(
        "Feel the curse",
        "Add **20** pure-noise columns (`rng = np.random.default_rng(5)`, `rng.normal(size=...)`, named `junk0…junk19`) to both `Xtr` and `Xte` (copies!). "
        "Fit scaled KNN (k = 10) and a `RandomForestRegressor(60, max_features=0.33, min_samples_leaf=2, random_state=0, n_jobs=-1)` on clean and noisy data. "
        "Store the **drops in test R²** (clean − noisy) as `drop_knn` and `drop_rf`.",
        ["Build `A = Xtr.copy(); B = Xte.copy()` then loop `for j in range(20): A[f'junk{j}'] = rng.normal(size=len(A))` (and the same for B).",
         "Clean R²: fit on `Xtr`; noisy R²: fit on `A` and score on `B`.",
         "Expect KNN to lose noticeably more than the forest."],
        'drop_knn = drop_rf = None   # your code\n',
        'rng = np.random.default_rng(5); A, B = Xtr.copy(), Xte.copy()\nfor j in range(20):\n    A[f"junk{j}"] = rng.normal(size=len(A)); B[f"junk{j}"] = rng.normal(size=len(B))\n'
        'mk = lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10)); mr = lambda: RandomForestRegressor(60, max_features=0.33, min_samples_leaf=2, random_state=0, n_jobs=-1)\n'
        'drop_knn = mk().fit(Xtr, ytr).score(Xte, yte) - mk().fit(A, ytr).score(B, yte)\ndrop_rf = mr().fit(Xtr, ytr).score(Xte, yte) - mr().fit(A, ytr).score(B, yte)\nprint(drop_knn, drop_rf)',
        'check("KNN loses at least 0.04 R²", lambda: drop_knn > 0.04)\ncheck("KNN is hurt more than the forest", lambda: drop_knn > drop_rf)',
        explain="Each junk column adds distance without information, diluting the real signal. Forests simply stop splitting on useless features; KNN has no mechanism to ignore them.",
        level="⭐⭐⭐")

    nb.quiz([
        ("Why is KNN called a 'lazy learner'?", "It does no work at training time (it just stores the data); all computation happens at prediction time."),
        ("k = 1 gives training R² = 1. Why is that meaningless?", "Each training point is its own nearest neighbour, so it 'predicts' itself perfectly. Only validation/test performance is informative."),
        ("You add the feature 'house price in US dollars' next to 'bedrooms' without scaling. What happens to the neighbours?",
         "The dollar column dominates the distance (values in millions vs single digits), so neighbours are effectively chosen by price alone and bedrooms are ignored."),
        ("Why does KNN struggle with 500 features?", "Curse of dimensionality: distances concentrate, neighbourhoods stop being local, and irrelevant features add noise to every distance."),
        ("A kd-tree speeds up KNN in 3-D but not in 100-D. Why?", "In high dimensions splits can't prune the search (every region is 'near' the query), so the tree degenerates towards a full scan."),
    ])

    nb.footer(["KNN predicts the (weighted) average of the k nearest training targets: no training, no assumed form.",
               "k is the bias–variance dial: small k = flexible/noisy, large k = smooth/rigid; choose it by cross-validation.",
               "Scale your features: distance is the model, so units decide similarity.",
               "Irrelevant and high-dimensional features hurt KNN badly (curse of dimensionality).",
               "Prediction is O(n·p) by brute force; kd/ball trees help only in low dimensions."],
              "Regression Showdown", "08_Regression_Showdown.ipynb")
    return nb
