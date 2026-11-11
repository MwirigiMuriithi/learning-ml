# Cheat-sheet · NumPy · pandas · Matplotlib · Seaborn

## NumPy
```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape, a.ndim, a.size, a.dtype
np.zeros(3); np.ones((2, 2)); np.arange(0, 10, 2); np.linspace(0, 1, 5); np.eye(3)
a[1, 2]; a[1]; a[:, 0]; a[0:2, 1:]; a[::-1]            # index / slice (slices are VIEWS: use .copy())
a[a > 2]; a[(a > 1) & (a < 6)]                          # masks: & | ~ with parentheses
np.where(cond); np.where(cond, x, y); a.argmax(); a.argmin(axis=0)
a.reshape(3, 2); a.reshape(-1); a.T; np.vstack; np.hstack; np.concatenate; np.stack
a.sum(axis=0); a.mean(axis=1); a.std(axis=0, ddof=1); a.cumsum(); np.percentile(a, [25, 75])
a - a.mean(axis=0)                                      # broadcasting: align shapes from the RIGHT (equal or 1)
v[:, np.newaxis]                                        # make a column (n,) -> (n,1)
A @ B; np.linalg.solve(A, b); np.linalg.lstsq(X, y, rcond=None)
np.diff(a); np.nanmean(a); np.isnan(a); np.isclose(x, y); np.clip(a, lo, hi)
rng = np.random.default_rng(42); rng.normal(0, 1, 5); rng.integers(1, 7, 10); rng.choice(x, 3, replace=False)
```

## pandas: load, look, select
```python
import pandas as pd
df = pd.read_csv("f.csv", usecols=[...], parse_dates=["date"], dtype={"c": "category"}, na_values=["N/A"], nrows=1000)
pd.read_excel("f.xlsx", sheet_name="s", skiprows=2);  pd.read_json("f.json");  pd.json_normalize(list_of_dicts)
df.shape; df.info(); df.describe(include="all"); df.head(); df.sample(5); df["c"].value_counts(); df.nunique()
df["c"]; df[["a", "b"]]; df.loc[rows, cols]; df.iloc[0:3, 0:2]; df.set_index("id")
df[df.c > 5]; df[(df.a == "x") & (df.b > 2)]; df[df.c.isin([..])]; df[df.c.between(1, 5)]; df[df.s.str.contains("x")]; df.query("c > @t")
df.sort_values(["a", "b"], ascending=[True, False]); df.nlargest(3, "c"); df.idxmax()
df["new"] = ...; df.assign(new=lambda d: d.a * d.b); df.rename(columns={...}); df.drop(columns="c")
df.loc[mask, "col"] = value                             # NEVER df[mask]["col"] = value
```

## pandas: clean
```python
df.isna().sum(); df.duplicated().sum(); df.drop_duplicates(subset="id")
df["t"].str.strip().str.lower().str.replace("-", " ", regex=False).str.title()
df.dropna(subset=["c"]); df["c"].fillna(v); df.groupby("g")["c"].transform("median"); s.interpolate()
df["flag"] = df["c"].isna()                             # keep the information that it was missing
pd.to_datetime(s, errors="coerce", format="%Y-%m-%d"); s.dt.year / .month / .quarter / .day_name()
df.set_index("date").sort_index(); s.resample("ME").mean(); s.rolling(4).mean(); s.shift(1); s.pct_change(); s.diff()
s.map({...}); np.select([c1, c2], ["a", "b"], default="c"); pd.cut(s, bins, labels); pd.qcut(s, 4)
assert df["id"].is_unique
```

## pandas: group, reshape, merge
```python
df.groupby("g")["x"].mean()
df.groupby(["a", "b"], as_index=False).agg(avg=("x", "mean"), n=("id", "count"))
df.groupby("g")["x"].transform("mean")                  # same length as df
df.groupby("g").filter(lambda d: len(d) > 10)
df.pivot_table(index="a", columns="b", values="x", aggfunc="mean"); pd.crosstab(df.a, df.b); df.melt(id_vars="id")
pd.concat([a, b], ignore_index=True)
a.merge(b, on=["k1", "k2"], how="left", validate="many_to_one", indicator=True)   # check shapes & _merge after EVERY merge
```

## Matplotlib
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4.5))                # always the object-oriented way
ax.plot(x, y, label="…"); ax.bar(x, h); ax.barh(y, w); ax.hist(v, bins=30); ax.scatter(x, y, alpha=.6); ax.boxplot([a, b])
ax.axvline(v); ax.axhline(v); ax.axvspan(x0, x1, alpha=.15); ax.annotate("…", xy=(x, y), xytext=(x2, y2), arrowprops=dict(arrowstyle="->"))
ax.set_title("finding"); ax.set_xlabel("… (unit)"); ax.set_ylabel("… (unit)"); ax.legend(); ax.set_xscale("log")
fig, axes = plt.subplots(2, 2, constrained_layout=True); for ax in axes.flat: …
fig.savefig("chart.png", dpi=150, bbox_inches="tight")
```

## Seaborn
```python
import seaborn as sns; sns.set_theme(style="whitegrid")
sns.histplot(data=df, x="v", hue="g", bins=30, kde=True);   sns.kdeplot / sns.ecdfplot
sns.boxplot / sns.violinplot / sns.stripplot(data=df, x="cat", y="v");   sns.barplot(data=df, x="cat", y="v");   sns.countplot(data=df, x="cat")
sns.scatterplot(data=df, x="a", y="b", hue="g");   sns.regplot(data=df, x="a", y="b");   sns.lineplot(data=df, x="date", y="v", hue="g")
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, center=0)
sns.pairplot(df, vars=[…], hue="g", corner=True);   sns.jointplot(data=df, x=, y=, kind="reg")
sns.relplot(data=df, x=, y=, col="g", kind="line");   sns.displot(data=df, x=, col="g")
```

## Chart choice
| Question | Chart |
|---|---|
| change over time | line |
| compare categories | sorted bar from zero |
| one distribution | histogram / KDE / ECDF / box |
| compare distributions | box / violin |
| two numbers | scatter (+ trend) |
| two categories & a number | heatmap of a pivot |
| many correlations | heatmap (fix vmin/vmax/center) |

## Correlation rules of thumb
* |r| < 0.2 none · 0.2–0.4 weak · 0.4–0.7 moderate · > 0.7 strong.
* Pearson = straight lines; Spearman = ranks (robust to skew/outliers).
* **Detrend** time series (differences / % change) before correlating. **Stratify** to catch confounders.
