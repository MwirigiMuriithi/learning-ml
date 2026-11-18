# 🧾 Phase 5 Cheat Sheet: Unsupervised Learning

## 0 · Golden pipeline for tabular clustering
```
select features → impute → log1p skewed columns → standardise → (PCA if >15 features)
→ cluster for several k/params → evaluate (internal + stability + null test) → profile → name → report
```
**Never** include IDs, labels, or outcomes that merely restate other features.

## 1 · Choosing an algorithm

| Situation | Use |
|---|---|
| Many rows, roughly round groups, want *k* segments | **K-Means** |
| < ~5k rows, want to *see* the structure, unknown *k* | **Hierarchical (Ward)** + dendrogram |
| Odd shapes, outliers matter, geographic data | **DBSCAN / HDBSCAN** |
| Too many/correlated columns; need a picture or an index | **PCA** |
| Baskets: "what goes with what?" | **Apriori / association rules** |

## 2 · K-Means
```python
from sklearn.cluster import KMeans
m = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
m.labels_, m.cluster_centers_, m.inertia_, m.predict(X_new)
```
Loop: **assign** each point to nearest centre ↔ **move** centre to mean. Objective: inertia $J=\sum\lVert x-\mu_{c(x)}\rVert^2$.
Pitfalls: scale! · local minima (`n_init≥10`, k-means++) · outliers · round-clusters assumption.

## 3 · Hierarchical
```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
Z = linkage(X_scaled, method="ward")           # single | complete | average | ward
labels = fcluster(Z, t=k, criterion="maxclust")   # or t=height, criterion="distance"
dendrogram(Z, labels=names, orientation="right")
```
single → chains · complete → compact · average → compromise · **ward → round, equal-sized (default)**.

## 4 · DBSCAN
```python
from sklearn.cluster import DBSCAN
labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)      # -1 = noise
DBSCAN(eps=km/6371, min_samples=4, metric="haversine", algorithm="ball_tree").fit_predict(np.radians(latlon))
```
Core ≥ `min_samples` within `eps` · border near a core · noise otherwise. Choose `eps` from the **k-distance knee**; check the cluster-count plateau. Varying density → **HDBSCAN**.

## 5 · PCA
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.90).fit(X_scaled)
Z = pca.transform(X_scaled); pca.explained_variance_ratio_; pca.components_   # loadings
X_back = pca.inverse_transform(Z)
```
Centre → covariance → eigen-decompose → project. Choose components by cumulative variance (80-95%), scree elbow, Kaiser (eigenvalue>1), or purpose. Scale first. Signs are arbitrary.

## 6 · Association rules
$$\text{support}(A)=\frac{\#\{A\subseteq t\}}{N}\quad \text{conf}(A\Rightarrow B)=\frac{\text{supp}(A\cup B)}{\text{supp}(A)}\quad \text{lift}=\frac{\text{conf}}{\text{supp}(B)}$$
Lift =1 independent · >1 positive · <1 negative · symmetric. Apriori: frequent sets' subsets are frequent → prune.
```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
onehot = pd.DataFrame(TransactionEncoder().fit(baskets).transform(baskets), columns=te.columns_)
rules = association_rules(apriori(onehot, min_support=0.03, use_colnames=True), metric="lift", min_threshold=1.2)
```

## 7 · Evaluation without labels

| Evidence | Tool | Good |
|---|---|---|
| Compact & separated | `silhouette_score` | higher |
| | `davies_bouldin_score` | **lower** |
| | `calinski_harabasz_score` | higher |
| Repeatable | subsample ARI (stability) | → 1 |
| Real structure? | Hopkins; run on uniform noise | ≫ 0.5 vs ≈ 0.5 |
| Agrees with labels (if any) | ARI / NMI | → 1 |
| **Useful** | profile → name → action | the final judge |

## 8 · Profiling & reporting
```python
profile_clusters(df, labels, cols)                           # mean per cluster (course helper)
rel = (profile - df[cols].mean()) / df[cols].std()           # relative profile → sns.heatmap
DecisionTreeClassifier(max_depth=3).fit(df[cols], labels)    # explain clusters in rules
```
Persona card: **name · size · what sets them apart (numbers) · action · caveat**.

## 9 · Ten commandments
1. Look at the data first. 2. Scale. 3. Log-transform skewed money/size columns. 4. Multiple initialisations.
5. *k* is a decision, not a discovery. 6. Test against noise. 7. Test stability. 8. Name your clusters with numbers.
9. Lift, not confidence. 10. Report what *surprised* you, and what you're unsure of.
