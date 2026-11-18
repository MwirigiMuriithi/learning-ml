"""
build_datasets.py: how the course's synthetic datasets are made
================================================================

Two of our four projects use data we generate ourselves:

  1. crop_leaves.npz: 3,000 small RGB maize-leaf images, 5 classes
  2. maize_prices_weekly.csv: 15 years of weekly maize prices for 3 Kenyan markets

WHY SYNTHETIC?  Real crop-disease photos (e.g. PlantVillage) are multi-gigabyte
downloads, and real market prices need API access.  A generator gives us:
  * a dataset that ships inside the zip and works offline
  * KNOWN ground truth (we can check what a model *should* have learned)
  * the chance to see how data "gets made" (great for understanding bias!)

HONEST WARNING: these are *simulations*, not field data. Models trained on them
teach you the workflow, not real-world agronomy. The capstone shows how to swap
in real data.

Run it again any time:  python build_datasets.py     (deterministic, seed=2024)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
LEAF_CLASSES = ["healthy", "gray_leaf_spot", "northern_leaf_blight", "common_rust", "maize_streak"]
SIZE = 48


# ----------------------------------------------------------------------------
# 1. MAIZE LEAF IMAGES
# ----------------------------------------------------------------------------
def _smooth_noise(rng, size, sigma):
    n = gaussian_filter(rng.standard_normal((size, size)), sigma)
    return (n - n.mean()) / (n.std() + 1e-8)


def make_leaf(rng, label, size=SIZE):
    """Draw ONE leaf image of class `label` (0..4). Returns (uint8 image (size,size,3), lesion_fraction)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.uniform(-3, 3), size / 2 + rng.uniform(-3, 3)
    theta = rng.uniform(0, np.pi)                       # leaf orientation
    c, s = np.cos(theta), np.sin(theta)
    u = (xx - cx) * c + (yy - cy) * s                   # coordinate ALONG the leaf
    v = -(xx - cx) * s + (yy - cy) * c                  # coordinate ACROSS the leaf
    a, b = rng.uniform(19, 23), rng.uniform(6.5, 9.5)   # half-length, half-width
    bend = rng.uniform(-0.012, 0.012)
    v = v - bend * u ** 2                               # slight curve
    width_at_u = b * np.clip(1 - (u / a) ** 2, 0, 1) ** 0.6
    leaf = (np.abs(v) < width_at_u).astype(np.float32)
    leaf = gaussian_filter(leaf, 0.6)

    # background: soil / mulch, brownish-grey with noise
    bg = np.array([rng.uniform(70, 120), rng.uniform(60, 100), rng.uniform(40, 80)])
    img = np.ones((size, size, 3), np.float32) * bg
    img += 12 * _smooth_noise(rng, size, 2)[..., None]

    # leaf colour + texture + midrib
    green = np.array([rng.uniform(35, 75), rng.uniform(115, 165), rng.uniform(30, 65)])
    tex = 1 + 0.07 * _smooth_noise(rng, size, 1.2)
    leaf_rgb = green[None, None, :] * tex[..., None]
    midrib = np.exp(-(v / 0.9) ** 2) * (np.abs(u) < a)
    leaf_rgb = leaf_rgb + midrib[..., None] * np.array([40, 35, 25])
    lesion_mask = np.zeros((size, size), np.float32)

    def paint(mask, color, alpha=0.9):
        nonlocal leaf_rgb, lesion_mask
        m = gaussian_filter(mask.astype(np.float32), 0.5) * (leaf > 0.5) * alpha
        leaf_rgb = leaf_rgb * (1 - m[..., None]) + np.array(color)[None, None, :] * m[..., None]
        lesion_mask = np.maximum(lesion_mask, m)

    def rand_point_in_leaf(margin=0.8):
        uu = rng.uniform(-a * margin, a * margin)
        w = b * max(1 - (uu / a) ** 2, 0.05) ** 0.6
        return uu, rng.uniform(-w * margin, w * margin)

    # Disease SEVERITY: early infections show few, small, faint lesions (hard!); late ones show many, bold lesions.
    sev = rng.choice([0.25, 0.6, 1.0], p=[0.30, 0.35, 0.35])
    fade = 0.5 + 0.5 * sev                                   # early lesions are fainter & smaller

    if label == 1:    # GRAY LEAF SPOT: small rectangular tan-grey lesions along the veins
        for _ in range(max(2, int(rng.integers(5, 13) * sev))):
            u0, v0 = rand_point_in_leaf()
            L, W = rng.uniform(1.5, 3.5) * fade, rng.uniform(0.8, 1.4) * fade
            paint((np.abs(u - u0) < L) & (np.abs(v - v0) < W),
                  [rng.uniform(150, 185), rng.uniform(140, 170), rng.uniform(100, 130)], rng.uniform(0.55, 0.9))
    elif label == 2:  # NORTHERN LEAF BLIGHT: long cigar-shaped tan lesions
        for _ in range(int(rng.integers(1, 4)) if sev > 0.5 else 1):
            u0, v0 = rand_point_in_leaf(0.6)
            L, W = rng.uniform(7, 13) * fade, rng.uniform(1.8, 3.0) * fade
            paint((((u - u0) / L) ** 2 + ((v - v0) / W) ** 2) < 1,
                  [rng.uniform(150, 175), rng.uniform(125, 150), rng.uniform(80, 105)], rng.uniform(0.6, 0.9))
    elif label == 3:  # COMMON RUST: many tiny red-brown pustules
        for _ in range(max(5, int(rng.integers(25, 55) * sev))):
            u0, v0 = rand_point_in_leaf(0.95)
            r = rng.uniform(0.6, 1.1)
            paint(((u - u0) ** 2 + (v - v0) ** 2) < r ** 2,
                  [rng.uniform(140, 175), rng.uniform(55, 80), rng.uniform(25, 40)], rng.uniform(0.6, 0.9))
    elif label == 4:  # MAIZE STREAK VIRUS: pale-yellow streaks running ALONG the leaf
        freq, phase = rng.uniform(0.9, 1.4), rng.uniform(0, 6.28)
        stripes = np.sin(freq * v + phase) > rng.uniform(0.3, 0.6) + (1 - sev) * 0.35
        broken = _smooth_noise(rng, size, 3) > -0.3 + (1 - sev) * 0.5
        paint(stripes & broken & (np.abs(u) < a * 0.95),
              [rng.uniform(200, 225), rng.uniform(205, 225), rng.uniform(70, 110)], 0.85 * fade)
    elif label == 0 and rng.random() < 0.30:   # healthy-but-dusty "confuser": dust specks that mimic early disease
        for _ in range(rng.integers(1, 4)):
            u0, v0 = rand_point_in_leaf()
            paint(((u - u0) ** 2 + (v - v0) ** 2) < rng.uniform(0.5, 1.3) ** 2, [120, 100, 75], rng.uniform(0.5, 0.85))
        lesion_mask[:] = 0                                      # dust is NOT disease: don't count it as lesion area

    out = img * (1 - leaf[..., None]) + leaf_rgb * leaf[..., None]
    if rng.random() < 0.3:                                         # out-of-focus photo
        sg = rng.uniform(0.5, 1.0)
        out = np.stack([gaussian_filter(out[..., c], sg) for c in range(3)], -1)
    out *= rng.uniform(0.7, 1.25)                                  # lighting
    out += rng.normal(0, rng.uniform(4, 10), out.shape)            # sensor noise
    frac = float(lesion_mask.sum() / max(leaf.sum(), 1))
    return np.clip(out, 0, 255).astype(np.uint8), frac


def build_leaves(n_per_class=600, seed=2024):
    rng = np.random.default_rng(seed)
    X, y, frac = [], [], []
    for label in range(len(LEAF_CLASSES)):
        for _ in range(n_per_class):
            img, f = make_leaf(rng, label)
            X.append(img); y.append(label); frac.append(f)
    X, y, frac = np.stack(X), np.array(y, np.int64), np.array(frac, np.float32)
    perm = rng.permutation(len(y))
    X, y, frac = X[perm], y[perm], frac[perm]
    n = len(y); n_tr, n_va = int(.70 * n), int(.15 * n)
    split = np.array(["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va))
    np.savez_compressed(HERE / "crop_leaves.npz", images=X, labels=y, lesion_fraction=frac, split=split,
                        class_names=np.array(LEAF_CLASSES))
    pd.DataFrame({"id": np.arange(n), "label": y, "class_name": [LEAF_CLASSES[i] for i in y],
                  "lesion_fraction": frac.round(4), "split": split}).to_csv(
        HERE / "crop_leaves_metadata.csv", index=False)
    print("leaves:", X.shape, np.bincount(y))


# ----------------------------------------------------------------------------
# 2. WEEKLY MAIZE PRICES  (KES per 90 kg bag), three markets
# ----------------------------------------------------------------------------
def build_prices(seed=2024):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-03", "2025-12-29", freq="W-MON")
    n = len(dates)
    years = np.arange(n) / 52.18
    doy = dates.dayofyear.values
    yr_arr = dates.year.values

    # Rainfall (mm/week): bimodal season -> long rains (~April) & short rains (~Nov).
    seasonal_rain = 18 + 40 * np.exp(-((doy - 105) / 35) ** 2) + 25 * np.exp(-((doy - 315) / 30) ** 2)
    drought = np.ones(n)
    for yr, sev in [(2011, .45), (2017, .5), (2022, .35)]:
        drought[yr_arr == yr] = sev                          # drought years: rain falls to ~40% of normal
    common_rain = rng.gamma(4, 0.25, n)                      # weather shared across markets
    # Fuel price (KES/litre): random walk with upward drift + a 2022 shock
    fuel = 95 + np.cumsum(rng.normal(0.12, 0.9, n))
    fuel = gaussian_filter(fuel, 2)
    fuel += 25 * (yr_arr >= 2022)

    rows = []
    for market, offset, noise_sd in [("Nairobi", 1.00, .020), ("Eldoret", 0.90, .025), ("Kisumu", 0.96, .030)]:
        local = rng.gamma(6, 1 / 6, n)
        rain = seasonal_rain * drought * (0.6 * common_rain + 0.4 * local) * rng.uniform(0.9, 1.1)
        rain_anom = (gaussian_filter(rain, 4) - gaussian_filter(seasonal_rain, 4)) / 25.0
        lagged = np.roll(rain_anom, 13)
        lagged[:13] = 0                                      # rain affects price ~13 weeks later
        trend = 2900 * np.exp(0.045 * years) * offset        # ~4.5 %/yr inflation
        season = 1 + 0.16 * np.cos(2 * np.pi * (doy - 150) / 365.25) + 0.04 * np.cos(4 * np.pi * (doy - 60) / 365.25)
        ar = np.zeros(n)
        for t in range(1, n):
            ar[t] = 0.93 * ar[t - 1] + rng.normal(0, noise_sd)
        price = trend * season * np.exp(-0.55 * lagged + ar) * (1 + 0.0010 * (fuel - 110))
        rows.append(pd.DataFrame({"date": dates, "market": market,
                                  "price_kes_90kg": price.round(0).astype(int),
                                  "rainfall_mm": rain.round(1), "fuel_kes_per_litre": fuel.round(1)}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(HERE / "maize_prices_weekly.csv", index=False)
    print("prices:", df.shape)
    print(df.groupby("market").price_kes_90kg.agg(["min", "mean", "max"]).round(0))


if __name__ == "__main__":
    build_leaves()
    build_prices()
