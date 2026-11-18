# 🗂️ Datasets

All files are **synthetic** (seeded NumPy generators in `generate_datasets.py`) with realistic mess. Regenerate with `python generate_datasets.py`.

| File | Rows × cols | Used in | Notes |
|---|---|---|---|
| `advertising.csv` | 200 × 4 | L00–L02 | TV/radio/newspaper budgets → sales. Newspaper has **no** true effect but correlates with radio (confounding demo). |
| `position_salaries.csv` | 10 × 3 | L03 | classic non-linear salary ladder. |
| `house_prices.csv` | 2000 × 12 | L02–L08, P1 | 12 Nairobi neighbourhoods, KES millions; 6% missing `lot_size_sqft`, 4% missing `condition`, **6 sqft typos (×10)**. |
| `student_performance.csv` | 1200 × 15 | P2 | targets `final_score` (regression) and `passed`; diminishing returns of study hours; U-shaped sleep effect; ~3% / 2% missing. |
| `loan_approval.csv` | 3000 × 19 | L09–L16, P3 | ~60% approved; missing `credit_score`, `savings_balance_kes`; **`interest_rate_offered` is a leakage trap** (recorded only after approval). |
| `agri_prices.csv` | 2880 × 8 | P4 | 96 months × 6 crops × 5 markets; trend, seasonality, rainfall→supply→price, fuel; 2% missing rainfall. |
| `sms_spam_mini.csv` | 600 × 2 | L14 | templated SMS messages; 8 labels deliberately flipped (label noise). |
| `mystery_A.csv` | 150 × 61 | L17 | 🕵️ hidden structure: don't peek at the generator! |
| `mystery_B.csv` | 3000 × 9 | L17 | 🕵️ |
| `mystery_C.csv` | 4000 × 13 | L17 | 🕵️ |

## Data dictionaries
**house_prices**: `neighborhood`, `sqft`, `bedrooms`, `bathrooms`, `age_years`, `garage_spaces`, `has_pool`, `lot_size_sqft`, `condition` (Poor/Fair/Good/Excellent), `distance_to_cbd_km`, target `price_million_kes`.
**student_performance**: `gender`, `school_type`, `study_hours_per_week`, `attendance_pct`, `previous_exam_score`, `sleep_hours`, `absences`, `parental_education`, `internet_access`, `tutoring`, `extracurricular`, `motivation_1_10`, targets `final_score`, `passed`.
**loan_approval**: profile (`age`, `gender`, `married`, `dependents`, `education`, `employment_type`), finances (`monthly_income_kes`, `loan_amount_kes`, `loan_term_months`, `credit_score`, `credit_history_years`, `existing_loans`, `monthly_debt_kes`, `savings_balance_kes`, `has_collateral`, `property_area`), `interest_rate_offered` ⚠️, target `approved`.
**agri_prices**: `date`, `market`, `crop`, `rainfall_mm`, `avg_temp_c`, `fuel_price_kes_per_litre`, `supply_tons`, target `price_kes_per_kg`.

Scikit-learn's bundled datasets (breast cancer, iris, digits, make_moons/circles/blobs/classification) are used as well: no download needed.
