# 📇 Data cards

## `mnist_subset.npz`: real
Handwritten digits 0–9, 28×28 grayscale (uint8). **20,000 train + 5,000 test**, randomly subsampled (seed 0) from the original 60k/10k.
Keys: `x_train, y_train, x_test, y_test`. Source: LeCun, Cortes & Burges, *The MNIST database of handwritten digits*.
*Limits:* centred, clean digits from US writers; models trained here don't automatically handle messy real scans (see Lesson 07's shift test and handwriting demo).

## `fashion_subset.npz`: real
Zalando article images, 10 classes, 28×28 grayscale. **15,000 train + 5,000 test** subsampled (seed 0) from 60k/10k. Same keys.
Source: Xiao, Rasul & Vollgraf (2017), *Fashion-MNIST* (MIT licence). *Limits:* studio-style product photos, white-on-black silhouettes.

## `crop_leaves.npz`, `crop_leaves_metadata.csv`: 🧪 synthetic
3,000 RGB images, 48×48 (uint8), five **maize leaf** conditions, 600 each: `healthy, gray_leaf_spot, northern_leaf_blight, common_rust, maize_streak`.
Keys: `images (N,48,48,3), labels, lesion_fraction, split ('train'/'val'/'test' = 70/15/15), class_names`.
**How it is made:** `data/build_datasets.py::make_leaf` draws a curved elongated leaf with random orientation, colour and texture on a soil-like background, then paints
class-specific lesions (rectangular grey spots / long tan cigar-shaped blotches / many tiny red-brown pustules / pale yellow streaks along the veins) at one of
three **severity levels** (early infections are faint and sparse), plus dusty "confuser" specks on some healthy leaves, random blur, lighting and sensor noise.
`lesion_fraction` is the share of leaf area covered by lesions (known only because we generated it).
**Limitations (important):** simplified lesion appearance; uniform backgrounds; one leaf, centred; no co-infections, pests, nutrient deficiency, hands, shadows or phone artefacts.
Accuracy here says little about field accuracy. Use it to learn the workflow, then follow Lesson 10 §10 to move to real photos (e.g. PlantVillage maize classes; note its lab-background bias).

## `maize_prices_weekly.csv`: 🧪 synthetic
Weekly maize price (KES per 90 kg bag), rainfall (mm/week) and fuel price (KES/litre) for **Nairobi, Eldoret, Kisumu**, 2011-01-03 → 2025-12-29 (783 weeks × 3 markets).
Columns: `date, market, price_kes_90kg, rainfall_mm, fuel_kes_per_litre`.
**How it is made:** price = trend (~4.5 %/yr inflation) × seasonal harvest cycle × exp(−0.55 × rainfall anomaly lagged ≈ 13 weeks + AR(1) noise) × fuel effect, with market offsets;
rainfall has two rainy seasons (≈April, ≈November) with droughts in 2011, 2017 and 2022; fuel follows a drifting random walk with a 2022 shock.
**Limitations:** relationships are linear-ish *by construction*; real prices react to policy (import duties, export bans), stock levels, regional shocks and data-reporting quirks not modelled here.
Real sources to explore: Kenya's KAMIS, FAO GIEWS / FPMA, WFP VAM market data.

## `example_digits/*.png`: hand-made
Three digits drawn from straight line segments (`digit_1/4/7.png`) for the "your own handwriting" demo in Lesson 07. Replace with photos of your own handwriting.

## Rebuilding
`python data/build_datasets.py` regenerates the two synthetic datasets deterministically (seed 2024). The MNIST/Fashion subsets are fixed files.
