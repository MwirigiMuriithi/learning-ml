# 📖 Data Dictionary

All three datasets are **simulated** (see `generate_datasets.py`, fixed random seed) to behave like East-African smallholder data. Regenerate them any time with `python generate_datasets.py`.

---
## 1. `farm_records_raw.csv`: the main dataset (2,485 rows × 22 columns)

One row = one **farm × one growing season**. 600 farms, four seasons each (plus injected duplicates).
**Target:** `yield_t_ha`.  *"Known when"* tells you when the value exists: the key to leakage questions (Lesson 6).

| Column | Meaning | Valid / typical values | Known when |
|---|---|---|---|
| `farm_id` | Farm identifier | `F0001`…`F0600` | always (identifier, not a predictor) |
| `county` | Kenyan county | 10 counties (Uasin Gishu, Trans Nzoia, Nakuru, Kakamega, Bungoma, Meru, Machakos, Nyeri, Kisii, Kiambu) | before planting |
| `soil_type` | Dominant soil | Loam, Clay, Sandy, Silt, Volcanic | before planting |
| `farm_size_ha` | Farm area, hectares | > 0; heavy right skew (median ≈ 1 ha) | before planting |
| `experience` | Farmer experience (ordinal) | Beginner < Intermediate < Advanced < Expert | before planting |
| `crop` | Crop grown | Maize, Beans, Wheat, Potato, Sorghum | planting |
| `season` | Rainy season | Long rains (planted Mar–May) / Short rains (Oct–Dec) | planting |
| `planting_date` | Date planted | 2019-03 … 2023-11 | planting |
| `harvest_date` | Date harvested | planting + crop duration (±10 days) | **after harvest** |
| `rainfall_mm` | Total seasonal rainfall | ≈ 150 – 1,300 mm | during season |
| `avg_temp_c` | Mean seasonal temperature (°C) | ≈ 12 – 26 °C | during season |
| `soil_ph` | Soil acidity | physically 0–14; realistic 3–10 | before planting (soil test) |
| `nitrogen_ppm`, `phosphorus_ppm`, `potassium_ppm` | Soil nutrients (parts per million) | ≥ 0 | before planting (soil test) |
| `organic_matter_pct` | Organic matter, % | ≈ 0.5 – 7 | before planting (soil test) |
| `fertilizer_kg_ha` | Fertiliser applied, kg/ha | 0 – ~250 (stored as *text* with units in the raw file) | planting |
| `irrigation` | Irrigated? | yes/no (many spellings in raw) | planting |
| `pest_pressure` | Pest pressure (ordinal) | Low < Medium < High | during season |
| `market_price_ksh_kg` | Sale price, KSh/kg | ≈ 25 – 200 | **at sale (after harvest)** |
| `total_revenue_ksh` | yield × area × price | - | **after sale: computed from the target** |
| **`yield_t_ha`** | **TARGET: yield, tonnes/ha** | maize ≈ 1–8, potato ≈ 5–25, beans ≈ 0.3–2.5 | **after harvest** |

### What is "wrong" with the raw file?
The raw file contains *every* family of problem from Lessons 1–2: duplicates (exact and near), numbers/dates stored as text, inconsistent spellings, missing values (random *and* dependent on other columns), impossible values, sentinel codes, outliers and a leaky column. **Finding them is part of the exercise**: the counts are in `docs/INSTRUCTOR_NOTES.md`.

### `farm_clean_lesson.csv`
Created by notebook `02_data_cleaning_part2.ipynb`. The *structurally* cleaned dataset (2,329 rows): types fixed, spellings standardised, duplicates removed, impossible values → NaN, bad targets dropped. **Missing values are deliberately still present**: imputing them is a statistical step that belongs inside the pipeline.

---
## 2. `crop_prices_daily.csv`: daily market prices (5,478 rows)

| Column | Meaning |
|---|---|
| `date` | Day, 2019-01-01 … 2023-12-31 |
| `crop` | Maize, Beans or Potato (three crops stacked in one **long** table) |
| `price_ksh_kg` | Price in KSh/kg. ~1.6% missing, plus a few spikes (×1.8 data-entry errors) |
| `volume_tonnes` | Volume traded |

Built-in structure: upward trend (~7%/yr), yearly seasonality (peaks: Beans ≈ May/June, Maize ≈ July, Potato ≈ September), autocorrelated noise.

---
## 3. `soil_spectra.csv`: soil near-infrared spectra (700 rows × 43 columns)

| Column | Meaning |
|---|---|
| `sample_id` | Sample identifier |
| `band_400nm` … `band_2400nm` | 40 reflectance bands, strongly correlated neighbours |
| `depth_cm` | `0-20` or `20-50` |
| **`organic_carbon_pct`** | **Target** for the PCA lesson: soil organic carbon, % |

Built so that ≈ 4 hidden "shapes" generate all 40 bands; PCA should rediscover them.
