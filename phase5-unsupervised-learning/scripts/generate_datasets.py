"""
generate_datasets.py
====================
Creates every dataset used in the course. All data is SYNTHETIC (computer-generated)
but designed to look and behave like real data from Kenyan retail and agriculture.

Why synthetic?
  1. Real data is messy and often private. We want you to learn the ideas first.
  2. We *plant* hidden structure in the data, so later (and only later!) we can check
     whether your algorithm found it. In real life you never get that luxury.

Run it yourself:
    python scripts/generate_datasets.py

The random seed is fixed, so you will get exactly the same files every time.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "data"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. RETAIL CUSTOMERS  (Project 1: Customer segmentation)
# ---------------------------------------------------------------------------
def make_retail_customers(n=800, seed=SEED):
    rng = np.random.default_rng(seed)
    # (name, share, age mean/sd, income mean/sd [k KES/yr], recency mean, freq mean, basket mean, discount share, online prob)
    segs = [
        ("Loyal Big Spenders",   0.15, (42, 7),  (1800, 300), 12,  38, 4200, 0.10, 0.45),
        ("Bargain Hunters",      0.22, (34, 9),  (600, 150),  25,  26, 1200, 0.70, 0.35),
        ("Young Trendy",         0.20, (24, 4),  (550, 130),  18,  30, 1800, 0.35, 0.80),
        ("Wealthy Occasional",   0.13, (51, 8),  (2100, 350), 70,   8, 3800, 0.15, 0.30),
        ("Frugal Savers",        0.18, (47, 10), (450, 110),  55,   9,  900, 0.45, 0.15),
        ("Lapsed Customers",     0.12, (38, 11), (850, 250),  210,  3, 1400, 0.40, 0.40),
    ]
    rows, truth = [], []
    cid = 1
    for k, (name, share, age, inc, rec, freq, bsk, disc, onl) in enumerate(segs):
        m = int(round(n * share))
        age_ = np.clip(rng.normal(*age, m), 18, 75).round()
        inc_ = np.clip(rng.normal(*inc, m), 200, 3500).round()
        rec_ = np.clip(rng.gamma(4, rec / 4, m), 1, 365).round()
        freq_ = np.clip(rng.poisson(freq, m) + rng.integers(0, 3, m), 1, 80)
        bsk_ = np.clip(rng.normal(bsk, bsk * 0.2, m), 200, 9000).round()
        disc_ = np.clip(rng.normal(disc, 0.08, m), 0, 1).round(2)
        onl_ = rng.binomial(1, onl, m)
        for i in range(m):
            rows.append(dict(
                customer_id=f"C{cid:04d}",
                age=int(age_[i]),
                gender=rng.choice(["F", "M"], p=[0.55, 0.45]),
                annual_income_kes_k=int(inc_[i]),
                recency_days=int(rec_[i]),
                orders_per_year=int(freq_[i]),
                avg_basket_kes=int(bsk_[i]),
                discount_share=float(disc_[i]),
                prefers_online=int(onl_[i]),
            ))
            truth.append(dict(customer_id=f"C{cid:04d}", true_segment=name))
            cid += 1
    df = pd.DataFrame(rows)
    df["annual_spend_kes"] = (df.orders_per_year * df.avg_basket_kes).astype(int)
    df["spending_score"] = (
        100 * (df.annual_spend_kes.rank(pct=True) * 0.7 + (1 - df.recency_days.rank(pct=True)) * 0.3)
    ).round().astype(int).clip(1, 100)
    # shuffle so rows are not ordered by segment (that would be cheating!)
    order = rng.permutation(len(df))
    df, truth = df.iloc[order].reset_index(drop=True), pd.DataFrame(truth).iloc[order].reset_index(drop=True)
    # a few realistic data problems
    miss = rng.choice(len(df), 12, replace=False)
    df.loc[miss, "annual_income_kes_k"] = np.nan
    return df, truth


# ---------------------------------------------------------------------------
# 2. FARMERS  (Project 2: Farmer segmentation by production patterns)
# ---------------------------------------------------------------------------
def make_farmers(n=700, seed=SEED + 1):
    rng = np.random.default_rng(seed)
    counties = {
        "Nakuru": 0, "Uasin Gishu": 0, "Trans Nzoia": 0, "Narok": 0,
        "Kiambu": 1, "Meru": 1, "Nyeri": 1, "Embu": 1,
        "Machakos": 2, "Kitui": 2, "Makueni": 2,
        "Kakamega": 3, "Kisii": 3, "Bungoma": 3, "Kericho": 3,
    }
    cnames = list(counties)
    # (name, share, acres, yield idx, fert kg/acre, irrigated p, labour days/acre, improved seed p, coop p, dist km, loss %, crops)
    segs = [
        ("Subsistence Smallholder", 0.34, (1.6, .5), 11, 12, 0.03, 38, 0.25, 0.15, 14, 28, ["Maize", "Beans", "Sorghum", "Cassava"]),
        ("Commercial Grain Farmer", 0.16, (42, 15),  26, 70, 0.04, 6,  0.92, 0.55, 8,  9,  ["Maize", "Wheat", "Barley"]),
        ("Intensive Horticulture",  0.20, (3.2, 1.0), 38, 95, 0.85, 62, 0.88, 0.65, 6,  14, ["Tomatoes", "Kale", "Cabbage", "French Beans", "Onions"]),
        ("Dairy-Mixed Farmer",      0.18, (7, 2.5),   19, 45, 0.10, 20, 0.60, 0.80, 5,  7,  ["Napier Grass", "Maize", "Potatoes", "Dairy Fodder"]),
        ("Emerging Market-Oriented",0.12, (5, 1.8),   22, 38, 0.22, 30, 0.70, 0.30, 9,  17, ["Potatoes", "Beans", "Maize", "Bananas", "Avocado"]),
    ]
    rows, truth, fid = [], [], 1
    for name, share, acres, yld, fert, irr, lab, seed_p, coop, dist, loss, crops in segs:
        m = int(round(n * share))
        size = np.clip(rng.lognormal(np.log(acres[0]), 0.25, m), 0.3, 200)
        y = np.clip(rng.normal(yld, yld * 0.18, m), 2, 80)
        f = np.clip(rng.normal(fert, fert * 0.3 + 2, m), 0, 200)
        ir = rng.binomial(1, irr, m)
        lb = np.clip(rng.normal(lab, lab * 0.25, m), 2, 120)
        sd = rng.binomial(1, seed_p, m)
        cp = rng.binomial(1, coop, m)
        ds = np.clip(rng.gamma(4, dist / 4, m), 0.5, 60)
        ls = np.clip(rng.normal(loss, 4, m), 1, 50)
        for i in range(m):
            cty = rng.choice(cnames)
            # horticulture farmers cluster around peri-urban counties, grain in rift valley
            if name == "Commercial Grain Farmer":
                cty = rng.choice(["Nakuru", "Uasin Gishu", "Trans Nzoia", "Narok"])
            elif name == "Intensive Horticulture":
                cty = rng.choice(["Kiambu", "Meru", "Nyeri", "Embu", "Machakos", "Nakuru"])
            elif name == "Dairy-Mixed Farmer":
                cty = rng.choice(["Kiambu", "Nyeri", "Nakuru", "Kericho", "Uasin Gishu"])
            rows.append(dict(
                farmer_id=f"F{fid:04d}", county=cty,
                primary_crop=rng.choice(crops),
                farm_size_acres=round(float(size[i]), 2),
                yield_bags_per_acre=round(float(y[i]), 1),
                fertilizer_kg_per_acre=round(float(f[i]), 1),
                irrigated=int(ir[i]),
                labour_days_per_acre=round(float(lb[i]), 1),
                improved_seed=int(sd[i]),
                coop_member=int(cp[i]),
                distance_to_market_km=round(float(ds[i]), 1),
                post_harvest_loss_pct=round(float(ls[i]), 1),
                experience_years=int(np.clip(rng.normal(14, 8), 1, 45)),
                extension_visits_per_year=int(np.clip(rng.poisson(3 if cp[i] else 1), 0, 12)),
            ))
            truth.append(dict(farmer_id=f"F{fid:04d}", true_segment=name))
            fid += 1
    df = pd.DataFrame(rows)
    price = rng.normal(3200, 300, len(df))  # KES per bag (generic)
    df["annual_revenue_kes"] = (
        df.farm_size_acres * df.yield_bags_per_acre * price * (1 - df.post_harvest_loss_pct / 100)
    ).round(-2).astype(int)
    order = rng.permutation(len(df))
    df, truth = df.iloc[order].reset_index(drop=True), pd.DataFrame(truth).iloc[order].reset_index(drop=True)
    # realistic messiness
    for col, k in [("fertilizer_kg_per_acre", 15), ("post_harvest_loss_pct", 10), ("distance_to_market_km", 6)]:
        df.loc[rng.choice(len(df), k, replace=False), col] = np.nan
    # a few extreme outliers (data entry mistakes / giant estates)
    df.loc[rng.choice(len(df), 3, replace=False), "farm_size_acres"] *= 8
    return df, truth


# ---------------------------------------------------------------------------
# 3. MARKETS  (Project 3: Agricultural market clustering; also DBSCAN geo demo)
# ---------------------------------------------------------------------------
def make_markets(seed=SEED + 2):
    rng = np.random.default_rng(seed)
    hubs = {  # name: (lat, lon, n_markets, spread_deg)
        "Nairobi":  (-1.286, 36.817, 22, 0.10),
        "Mombasa":  (-4.043, 39.668, 12, 0.08),
        "Kisumu":   (-0.091, 34.768, 11, 0.09),
        "Nakuru":   (-0.303, 36.080, 13, 0.09),
        "Eldoret":  (0.514, 35.270, 11, 0.10),
        "Meru":     (0.047, 37.650, 10, 0.08),
        "Nyeri":    (-0.420, 36.951, 9, 0.07),
        "Kitale":   (1.019, 35.002, 8, 0.08),
    }
    # (type, lat/lon agnostic) traders, volume, volatility, dist to city, road, cold storage p, perishable share, commodities, mobile pay, market days
    types = {
        "Urban Wholesale Hub":        dict(tr=(900, 200),  vol=(420, 90), vola=(0.14, 0.03), dist=(4, 2),   road=4.6, cold=0.75, per=(0.70, 0.08), com=(48, 6), mob=(0.88, 0.05), days=7),
        "Regional Aggregation Market":dict(tr=(350, 90),   vol=(180, 45), vola=(0.20, 0.04), dist=(45, 15), road=3.5, cold=0.30, per=(0.45, 0.10), com=(32, 5), mob=(0.65, 0.08), days=5),
        "Rural Periodic Market":      dict(tr=(90, 30),    vol=(25, 9),   vola=(0.34, 0.06), dist=(85, 25), road=2.1, cold=0.03, per=(0.30, 0.10), com=(15, 4), mob=(0.38, 0.10), days=2),
        "Border/Export Gateway":      dict(tr=(500, 130),  vol=(300, 70), vola=(0.27, 0.05), dist=(30, 12), road=4.0, cold=0.55, per=(0.35, 0.09), com=(26, 5), mob=(0.72, 0.08), days=6),
    }
    rows = []
    mid = 1
    for city, (lat0, lon0, n, spread) in hubs.items():
        for j in range(n):
            # first market in each hub city = urban hub, near the centre; others spread out
            if j == 0:
                t = "Urban Wholesale Hub"
            elif j < 3 and city in ("Nairobi", "Mombasa"):
                t = rng.choice(["Urban Wholesale Hub", "Regional Aggregation Market"])
            elif city in ("Kitale",) and j < 3:
                t = "Border/Export Gateway"
            else:
                t = rng.choice(list(types), p=[0.04, 0.38, 0.50, 0.08])
            p = types[t]
            rows.append(dict(
                market_id=f"M{mid:03d}",
                market_name=f"{city} {t.split()[0]} {j+1}",
                nearest_hub=city,
                latitude=round(lat0 + rng.normal(0, spread * (0.4 if t.startswith("Urban") else 1.6)), 4),
                longitude=round(lon0 + rng.normal(0, spread * (0.4 if t.startswith("Urban") else 1.6)), 4),
                avg_daily_traders=int(max(15, rng.normal(*p["tr"]))),
                avg_daily_volume_tonnes=round(float(max(3, rng.normal(*p["vol"]))), 1),
                price_volatility=round(float(max(0.05, rng.normal(*p["vola"]))), 3),
                distance_to_city_km=round(float(max(0.5, rng.normal(*p["dist"]))), 1),
                road_quality_1_5=round(float(np.clip(rng.normal(p["road"], 0.5), 1, 5)), 1),
                has_cold_storage=int(rng.random() < p["cold"]),
                perishable_share=round(float(np.clip(rng.normal(*p["per"]), 0.05, 0.95)), 2),
                n_commodities=int(max(5, rng.normal(*p["com"]))),
                mobile_payment_share=round(float(np.clip(rng.normal(*p["mob"]), 0.05, 0.99)), 2),
                market_days_per_week=int(np.clip(round(p["days"] + rng.normal(0, 0.6)), 1, 7)),
            ))
            rows[-1]["_true"] = t
            mid += 1
    # remote isolated markets (will become DBSCAN "noise")
    remote = [(3.1, 35.6), (2.3, 37.9), (-2.6, 40.2), (3.5, 39.5), (1.7, 40.0), (-2.9, 37.8), (4.1, 36.8)]
    p = types["Rural Periodic Market"]
    for (la, lo) in remote:
        rows.append(dict(
            market_id=f"M{mid:03d}", market_name=f"Remote Rural {mid}", nearest_hub="(none)",
            latitude=la, longitude=lo,
            avg_daily_traders=int(max(15, rng.normal(60, 15))),
            avg_daily_volume_tonnes=round(float(max(3, rng.normal(14, 5))), 1),
            price_volatility=round(float(rng.normal(0.45, 0.06)), 3),
            distance_to_city_km=round(float(rng.normal(190, 40)), 1),
            road_quality_1_5=round(float(np.clip(rng.normal(1.6, 0.4), 1, 5)), 1),
            has_cold_storage=0, perishable_share=round(float(np.clip(rng.normal(0.25, 0.08), 0.05, 0.9)), 2),
            n_commodities=int(max(5, rng.normal(9, 2))),
            mobile_payment_share=round(float(np.clip(rng.normal(0.22, 0.08), 0.05, 0.9)), 2),
            market_days_per_week=int(np.clip(round(rng.normal(1.5, 0.5)), 1, 7)),
            _true="Rural Periodic Market"))
        mid += 1
    df = pd.DataFrame(rows)
    truth = df[["market_id", "_true"]].rename(columns={"_true": "true_type"})
    df = df.drop(columns="_true")
    return df, truth


# ---------------------------------------------------------------------------
# 4. PRODUCTS  (Project 4: Grouping similar agricultural products)
# ---------------------------------------------------------------------------
def make_products(seed=SEED + 3):
    rng = np.random.default_rng(seed)
    # product, category
    catalog = [
        ("Maize", "Cereal"), ("Wheat", "Cereal"), ("Rice", "Cereal"), ("Sorghum", "Cereal"), ("Millet", "Cereal"), ("Barley", "Cereal"),
        ("Beans", "Legume"), ("Green Grams", "Legume"), ("Cowpeas", "Legume"), ("Pigeon Peas", "Legume"), ("Groundnuts", "Legume"), ("Soybeans", "Legume"),
        ("Tomatoes", "Vegetable"), ("Kale", "Vegetable"), ("Cabbage", "Vegetable"), ("Onions", "Vegetable"), ("Spinach", "Vegetable"),
        ("Capsicum", "Vegetable"), ("Carrots", "Vegetable"), ("French Beans", "Vegetable"), ("Indigenous Vegetables", "Vegetable"),
        ("Bananas", "Fruit"), ("Mangoes", "Fruit"), ("Avocado", "Fruit"), ("Oranges", "Fruit"), ("Pineapples", "Fruit"),
        ("Watermelon", "Fruit"), ("Passion Fruit", "Fruit"), ("Papaya", "Fruit"),
        ("Potatoes", "Root & Tuber"), ("Cassava", "Root & Tuber"), ("Sweet Potatoes", "Root & Tuber"), ("Arrowroot", "Root & Tuber"), ("Yams", "Root & Tuber"),
        ("Tea", "Cash Crop"), ("Coffee", "Cash Crop"), ("Cotton", "Cash Crop"), ("Sugarcane", "Cash Crop"), ("Pyrethrum", "Cash Crop"), ("Macadamia", "Cash Crop"),
        ("Milk", "Livestock Product"), ("Eggs", "Livestock Product"), ("Beef", "Livestock Product"), ("Goat Meat", "Livestock Product"),
        ("Chicken", "Livestock Product"), ("Honey", "Livestock Product"), ("Tilapia", "Livestock Product"), ("Hides & Skins", "Livestock Product"),
    ]
    # category profile: price KES/kg, price CV, seasonality amp, shelf-life d, elasticity, wholesale share, export share, margin %, volume (t/month, log10)
    prof = {
        "Cereal":            (55,  0.12, 0.18, 360, 0.4, 0.55, 0.03, 12, 4.4),
        "Legume":            (140, 0.17, 0.22, 300, 0.6, 0.45, 0.05, 18, 3.6),
        "Vegetable":         (60,  0.42, 0.35, 6,   0.9, 0.70, 0.08, 28, 4.0),
        "Fruit":             (85,  0.38, 0.45, 10,  1.1, 0.60, 0.15, 30, 3.8),
        "Root & Tuber":      (45,  0.25, 0.20, 40,  0.5, 0.50, 0.02, 20, 3.9),
        "Cash Crop":         (180, 0.22, 0.28, 180, 0.3, 0.15, 0.80, 15, 4.3),
        "Livestock Product": (260, 0.14, 0.12, 12,  0.8, 0.30, 0.10, 22, 4.1),
    }
    rows = []
    for name, cat in catalog:
        p, cv, seas, shelf, el, wh, ex, mg, vol = prof[cat]
        rows.append(dict(
            product=name,
            avg_price_kes_per_kg=round(float(p * rng.lognormal(0, 0.45)), 1),
            price_cv=round(float(np.clip(rng.normal(cv, cv * 0.25), 0.04, 0.9)), 3),
            seasonality_amplitude=round(float(np.clip(rng.normal(seas, seas * 0.3), 0.03, 0.9)), 3),
            shelf_life_days=int(max(2, rng.lognormal(np.log(shelf), 0.35))),
            demand_elasticity=round(float(np.clip(rng.normal(el, 0.2), 0.1, 2.0)), 2),
            wholesale_share=round(float(np.clip(rng.normal(wh, 0.1), 0.05, 0.95)), 2),
            export_share=round(float(np.clip(rng.normal(ex, ex * 0.35 + 0.01), 0.0, 0.95)), 2),
            avg_margin_pct=round(float(np.clip(rng.normal(mg, 5), 3, 55)), 1),
            monthly_volume_tonnes=int(10 ** rng.normal(vol, 0.35)),
            markets_traded_in=int(np.clip(rng.normal(60 if cat in ("Cereal", "Vegetable", "Root & Tuber") else 38, 12), 5, 110)),
            _cat=cat,
        ))
    df = pd.DataFrame(rows)
    truth = df[["product", "_cat"]].rename(columns={"_cat": "category"})
    df = df.drop(columns="_cat")
    return df, truth


# ---------------------------------------------------------------------------
# 5. BASKETS  (Association rule learning)
# ---------------------------------------------------------------------------
def _make_baskets(missions, noise_items, n, seed, prefix):
    """missions: list of (probability, {item: prob_in_basket})"""
    rng = np.random.default_rng(seed)
    probs = np.array([m[0] for m in missions]); probs = probs / probs.sum()
    rows = []
    for t in range(1, n + 1):
        mi = rng.choice(len(missions), p=probs)
        basket = set()
        for item, p in missions[mi][1].items():
            if rng.random() < p:
                basket.add(item)
        # random extra items (impulse buys)
        for item, p in noise_items.items():
            if rng.random() < p:
                basket.add(item)
        if not basket:
            basket.add(rng.choice(list(noise_items)))
        for item in sorted(basket):
            rows.append(dict(transaction_id=f"{prefix}{t:05d}", item=item))
    return pd.DataFrame(rows)


def make_grocery_baskets():
    missions = [
        (0.22, {"Bread": .85, "Milk": .75, "Butter": .45, "Eggs": .40, "Tea Leaves": .30}),               # breakfast run
        (0.20, {"Maize Flour": .90, "Cooking Oil": .70, "Sugar": .50, "Salt": .30, "Tomatoes": .40, "Onions": .55}),  # staples (ugali)
        (0.15, {"Rice": .85, "Cooking Oil": .55, "Onions": .65, "Tomatoes": .60, "Spices": .45}),          # pilau night
        (0.12, {"Soda": .70, "Crisps": .60, "Chocolate": .40, "Bread": .15}),                                # snacks
        (0.10, {"Detergent": .75, "Bath Soap": .60, "Toothpaste": .45, "Toilet Paper": .50}),               # household
        (0.12, {"Sukuma Wiki": .80, "Tomatoes": .55, "Onions": .60, "Maize Flour": .50}),                   # fresh veg
        (0.09, {"Baby Formula": .65, "Diapers": .75, "Wet Wipes": .50, "Milk": .30}),                       # baby
    ]
    noise = {"Bread": .03, "Milk": .04, "Sugar": .04, "Eggs": .03, "Soda": .03, "Tea Leaves": .03, "Rice": .02,
             "Bath Soap": .02, "Salt": .02, "Cooking Oil": .03, "Spices": .02, "Tomatoes": .03, "Onions": .03}
    return _make_baskets(missions, noise, 2500, SEED + 4, "G")


def make_agro_baskets():
    missions = [
        (0.20, {"Hybrid Maize Seed": .90, "DAP Fertilizer": .85, "CAN Fertilizer": .55, "Herbicide": .20}),   # maize planting season
        (0.14, {"Pesticide": .85, "Knapsack Sprayer": .45, "Protective Gloves": .55, "Fungicide": .55}),      # crop protection
        (0.14, {"Dairy Meal": .90, "Mineral Lick": .60, "Veterinary Drugs": .45, "Acaricide": .45}),            # dairy
        (0.12, {"Vegetable Seeds": .85, "Drip Kit": .40, "Foliar Feed": .55, "Fungicide": .40, "Pesticide": .35}),  # horticulture
        (0.10, {"Poultry Feed": .90, "Chick Starter": .50, "Veterinary Drugs": .45, "Drinkers & Feeders": .40}),
        (0.10, {"Potato Seed": .90, "DAP Fertilizer": .65, "Fungicide": .65, "Pesticide": .30}),
        (0.08, {"Tools (Hoe/Panga)": .75, "Wheelbarrow": .25, "Protective Gloves": .30, "Water Pump": .10}),
        (0.12, {"CAN Fertilizer": .75, "Herbicide": .55, "Hybrid Maize Seed": .15}),                             # top-dressing
    ]
    noise = {"Protective Gloves": .03, "Pesticide": .03, "Veterinary Drugs": .03, "DAP Fertilizer": .03,
             "Tools (Hoe/Panga)": .03, "Dairy Meal": .02, "Vegetable Seeds": .02, "Herbicide": .02}
    return _make_baskets(missions, noise, 3000, SEED + 5, "A")


if __name__ == "__main__":
    c, ct = make_retail_customers(); c.to_csv(OUT / "retail_customers.csv", index=False); ct.to_csv(OUT / "retail_customers_truth.csv", index=False)
    f, ft = make_farmers();         f.to_csv(OUT / "farmers.csv", index=False);          ft.to_csv(OUT / "farmers_truth.csv", index=False)
    m, mt = make_markets();         m.to_csv(OUT / "markets.csv", index=False);          mt.to_csv(OUT / "markets_truth.csv", index=False)
    p, pt = make_products();        p.to_csv(OUT / "products.csv", index=False);         pt.to_csv(OUT / "products_truth.csv", index=False)
    make_grocery_baskets().to_csv(OUT / "grocery_baskets.csv", index=False)
    make_agro_baskets().to_csv(OUT / "agro_input_baskets.csv", index=False)
    for p_ in sorted(OUT.glob("*.csv")):
        print(f"{p_.name:32s} {sum(1 for _ in open(p_)) - 1:>6,d} rows")
