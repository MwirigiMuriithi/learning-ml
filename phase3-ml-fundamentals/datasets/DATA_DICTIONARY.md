# 📚 Data Dictionary

All files are plain CSVs with a header row. Synthetic files are created by `generate_datasets.py` from fixed random seeds, so they are identical for every learner. Load any of them in a notebook with `load("filename.csv")`.

---

## `students.csv`: 200 rows · regression
Study habits and exam results (synthetic).

| Column | Type | Meaning |
|---|---|---|
| `hours_studied` | float | hours studied for the exam (0.5–10) |
| `sleep_hours` | float | average nightly sleep before the exam |
| `prev_score` | float | score on the previous exam (0–100) |
| `attendance_pct` | float | % of classes attended |
| **`exam_score`** | float | **target**: exam score (0–100) |

*Built-in truth:* ≈ 4 points per extra study hour, plus smaller effects of previous score, attendance and sleep, and noise (σ = 4).

## `housing.csv`: 900 rows · regression
Past home sales (synthetic). Used for the practical assignment.

| Column | Type | Meaning |
|---|---|---|
| `area_sqm` | float | living area (m²) |
| `bedrooms`, `bathrooms` | int | counts |
| `age_years` | int | building age (0–60) |
| `distance_to_center_km` | float | distance to city centre |
| `has_garden` | 0/1 | garden present |
| `neighborhood_quality` | 1–5 | neighbourhood rating |
| **`price_k`** | float | **target**: sale price in $1,000s |

*Built-in truth:* price rises with area (slightly curved), quality (with an area × quality interaction), garden and bathrooms; falls with age and distance; multiplicative noise of ≈ 6%, which is why residuals "fan out" at high prices.

## `customers.csv`: 400 rows · clustering
Shopping-mall customers (synthetic) with four hidden segments of 100.

| Column | Meaning |
|---|---|
| `age` | years |
| `annual_income_k` | income in $1,000s |
| `spending_score` | 1–100 mall-assigned spending score |
| `visits_per_month` | store visits per month |
| `segment_truth` | **hidden answer key (0–3)**: *do not use as a feature*; only to check clustering afterwards |

Segments: 0 = high-income savers · 1 = VIP shoppers · 2 = trendy low-income youth · 3 = frugal low-income.

## `churn.csv`: 1,500 rows · classification (imbalanced)
Telecom customers (synthetic). **≈ 15% churn.**

| Column | Meaning |
|---|---|
| `tenure_months` | months as a customer |
| `monthly_charge` | monthly bill ($) |
| `support_calls` | calls to support |
| `contract_months` | contract length: 1, 12 or 24 |
| `late_payments` | number of late payments |
| `usage_gb` | monthly data usage |
| **`churned`** | **target**: 1 = left, 0 = stayed |

## `used_cars.csv`: 1,200 rows · regression (capstone)
Used-car listings (synthetic). **Contains missing values and text categories.**

| Column | Type | Missing? | Meaning |
|---|---|---|---|
| `brand` | text | no | Toyota, Honda, Ford, Kia, BMW |
| `year` | int | no | year of manufacture (2008–2023) |
| `mileage_km` | float | **yes (≈ 3.4%)** | kilometres driven |
| `engine_cc` | float | **yes (≈ 2.5%)** | engine size |
| `fuel_type` | text | no | petrol, diesel, hybrid |
| `transmission` | text | no | manual, automatic |
| `owners` | float | **yes (≈ 1.8%)** | previous owners (1–4) |
| **`price_usd`** | float | no | **target**: asking price in US$ |

*Built-in truth:* price = brand base × exp(−0.10 × age) × exp(−4.5e-6 × mileage) × small premiums for automatic / hybrid / bigger engines × 0.96^(owners−1) × log-normal noise (σ = 0.08). The multiplicative structure is why a **log transform** of the target works so well.

## `noisy_curve.csv`: 80 rows · over/underfitting
| Column | Meaning |
|---|---|
| `x` | uniform on [0, 1] |
| `y` | `sin(2πx)` + Gaussian noise (σ = 0.25) |

## Classic datasets (from scikit-learn, saved as CSV)
| File | Rows × cols | Target | Use |
|---|---|---|---|
| `iris.csv` | 150 × 6 | `species` (setosa / versicolor / virginica) and numeric `target` | classification, decision boundaries |
| `wine.csv` | 178 × 14 | `target` (3 grape cultivars) | PCA, multiclass metrics |
| `breast_cancer.csv` | 569 × 32 | `target` (1 = benign, 0 = malignant), `diagnosis` text | binary classification, probabilities |
| `diabetes.csv` | 442 × 11 | `disease_progression` (continuous) | Lasso / feature selection. Features are pre-standardised by scikit-learn. |

*Sources: R. A. Fisher's Iris data (1936); UCI Wine; Wisconsin Diagnostic Breast Cancer; Efron et al., "Least Angle Regression" (2004).*
