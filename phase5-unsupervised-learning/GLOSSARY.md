# 📖 Glossary (plain English)

| Term | Meaning |
|---|---|
| **Unsupervised learning** | Learning patterns from data that has **no target labels** |
| **Feature** | A column describing each row (e.g. `farm_size_acres`) |
| **Cluster** | A group of rows that are similar to each other and different from other groups |
| **Centroid** | The "average point" (centre) of a cluster in K-Means |
| **Inertia (WCSS)** | Sum of squared distances from points to their cluster centre; K-Means minimises it |
| **k-means++** | A smarter way to choose starting centroids: spread them out |
| **Elbow method** | Plot inertia vs *k*; choose where improvements start to flatten |
| **Silhouette score** | −1 to 1; how well each point fits its own cluster vs the next-nearest; higher is better |
| **Davies–Bouldin** | Cluster-overlap score; **lower** is better |
| **Calinski–Harabasz** | Ratio of between- to within-cluster variance; higher is better |
| **Stability** | Do you get the same clusters on different samples? (measured with ARI) |
| **ARI (Adjusted Rand Index)** | Agreement between two groupings, corrected for chance; 1 = identical, ≈0 = random |
| **Hopkins statistic** | Measures if data is "clumpy": ≈0.5 random, → 1 clustered |
| **Dendrogram** | Tree showing the order and distance at which clusters merge |
| **Linkage** | How to measure the distance between two *clusters*: single, complete, average, Ward |
| **Cophenetic correlation** | How faithfully a dendrogram preserves the original distances |
| **Chaining** | Single linkage joining distinct clusters through a bridge of points |
| **DBSCAN** | Density-based clustering with `eps` and `min_samples`; labels outliers as noise |
| **Core / border / noise point** | In DBSCAN: in a crowd / on the edge of one / alone |
| **eps (ε)** | DBSCAN neighbourhood radius |
| **min_samples** | Number of points needed in a neighbourhood to count as dense |
| **k-distance plot** | Sorted distance to each point's k-th neighbour; the "knee" suggests `eps` |
| **Haversine distance** | Great-circle distance between latitude/longitude points on a sphere |
| **HDBSCAN** | DBSCAN across all densities; fewer parameters |
| **PCA** | Rotates data to axes of maximum variance (principal components) |
| **Principal component (PC)** | A new axis: a weighted mix of the original features |
| **Loadings** | The weights that make up a principal component |
| **Explained variance ratio** | Share of total variance a component carries |
| **Scree plot** | Bar chart of explained variance per component |
| **Reconstruction error** | Information lost by projecting down then back up |
| **t-SNE / UMAP** | Non-linear methods for 2-D pictures of high-dimensional data |
| **Curse of dimensionality** | In many dimensions, distances between points become similar and less meaningful |
| **Transaction / basket** | A set of items bought together |
| **Itemset** | Any set of items; a *k*-itemset has *k* items |
| **Support** | Share of baskets containing an itemset |
| **Confidence** | Of baskets with A, the share also containing B: P(B\|A) |
| **Lift** | Confidence ÷ support(B); 1 = independent, >1 = positively associated, <1 = negatively |
| **Leverage / conviction** | Alternative rule-strength measures (difference from independence / how often the rule fails) |
| **Apriori principle** | Subsets of frequent itemsets are frequent; supersets of infrequent ones are infrequent |
| **Profile** | Average feature values per cluster; the basis for naming clusters |
| **Persona** | A named, human description of a cluster ("Lapsed customers") |
| **Imputation** | Filling in missing values (e.g. with the median) |
| **Standardisation** | Rescaling a column to mean 0, std 1 |
| **Log transform** | `log(x)` or `log1p(x)` to tame right-skewed columns |
