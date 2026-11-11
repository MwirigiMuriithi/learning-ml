# 🔤 Glossary

| Term | Plain-English meaning |
|---|---|
| **Canary feature** | A column of pure random noise added on purpose; any selector that ranks it highly is not to be trusted. Its best score is the *noise floor*. |
| **Categorical feature** | A column of labels. *Nominal* = no order (crop); *ordinal* = ordered (Low < Medium < High). |
| **ColumnTransformer** | scikit-learn tool that applies different preprocessing to different columns and glues the results together. |
| **Curse of dimensionality** | With many features, data become sparse and distance-based methods degrade; noise features make it worse. |
| **Cyclical encoding** | Representing a circular quantity (month, hour) with sine and cosine so the end connects to the start. |
| **Data leakage** | Information from outside the training data (test set, future, or the target itself) influencing the model: inflating evaluation scores. |
| **Dummy-variable trap** | Keeping all *k* one-hot columns makes them perfectly redundant (they sum to 1); drop one for linear models. |
| **Feature** | A column used as model input. |
| **Feature engineering** | Creating new features from existing columns. |
| **Feature selection** | Choosing a useful subset of features (filter / wrapper / embedded). |
| **fit / transform** | `fit` *learns* from data (mean, categories, medians…); `transform` *applies* what was learned. `fit` on training data only. |
| **Group leakage** | The same entity (farm, patient) in both train and test sets. |
| **Imputation** | Filling in missing values. |
| **IQR (inter-quartile range)** | Q3 − Q1; basis of Tukey outlier fences. |
| **Lag feature** | The value from *k* steps ago (`shift(k)`). |
| **MAD** | Median absolute deviation; a robust spread measure. |
| **MAR / MCAR / MNAR** | Missing at random (depends on *observed* data) / completely at random / not at random (depends on the missing value itself). |
| **Multicollinearity** | Features strongly correlated with each other. Measured by VIF. |
| **Mutual information** | A measure of any (including non-linear) dependence between two variables. |
| **Natural key** | The set of columns identifying one real-world event (`farm_id + planting_date`). |
| **Outlier** | An unusual value: an error, a rare real event, or a member of another population. |
| **PCA** | Principal component analysis: builds uncorrelated blended features ordered by variance explained. |
| **Permutation importance** | Drop in score when a feature's values are shuffled on held-out data. |
| **Pipeline** | A chain of preprocessing steps and a model fitted and applied as one object. |
| **RFE / RFECV** | Recursive feature elimination (with cross-validated choice of size). |
| **Rolling window** | Statistic (mean, std) over the last *w* observations. *Trailing* = past only; *centered* = includes the future (leaks). |
| **Sentinel value** | A fake number standing for "unknown" (-999, 99). |
| **Skewness** | Asymmetry of a distribution; > 0 = long right tail. |
| **Standardisation / z-score** | (x − mean) / std. |
| **Target** | The quantity to predict (`yield_t_ha`). |
| **Target encoding** | Replacing a category with the mean target of that category (leaks unless cross-fitted). |
| **Winsorising** | Capping extreme values at chosen limits instead of removing them. |
