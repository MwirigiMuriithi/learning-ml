"""
Dataset generator for the Supervised Learning course.

All datasets are SYNTHETIC (generated with a fixed random seed) but designed to
behave like real-world data: they have missing values, noise, non-linear effects,
a few data-entry errors, class imbalance and even a "leakage trap" for you to find.

Run:  python generate_datasets.py
The CSV files are written next to this script.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. advertising.csv  (Lessons 01-02)  tv / radio / newspaper -> sales
# ---------------------------------------------------------------------------
def make_advertising(n=200):
    tv = rng.uniform(5, 300, n)
    radio = np.clip(rng.normal(25, 14, n), 0, 50)
    # newspaper is correlated with radio but has NO real effect on sales
    newspaper = np.clip(0.9 * radio + rng.normal(20, 18, n), 0.5, 115)
    sales = 3.0 + 0.045 * tv + 0.19 * radio + 0.0 * newspaper + rng.normal(0, 1.6, n)
    df = pd.DataFrame({"tv_budget_k": tv.round(1), "radio_budget_k": radio.round(1),
                       "newspaper_budget_k": newspaper.round(1), "sales_k_units": sales.round(2)})
    df.to_csv(os.path.join(HERE, "advertising.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 2. position_salaries.csv (Lesson 03)  a tiny, wildly non-linear dataset
# ---------------------------------------------------------------------------
def make_position_salaries():
    df = pd.DataFrame({
        "position": ["Business Analyst", "Junior Consultant", "Senior Consultant", "Manager",
                     "Country Manager", "Region Manager", "Partner", "Senior Partner", "C-level", "CEO"],
        "level": list(range(1, 11)),
        "salary_kes": [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000],
    })
    df.to_csv(os.path.join(HERE, "position_salaries.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 3. house_prices.csv (Lessons 02-08, Project 1)   Nairobi-flavoured, KES millions
# ---------------------------------------------------------------------------
def make_house_prices(n=2000):
    hoods = {  # name: (price multiplier, mean distance to CBD in km, pool prob, lot multiplier)
        "Westlands": (1.50, 5, 0.12, 1.0), "Kilimani": (1.35, 4, 0.10, 1.0),
        "Lavington": (1.60, 7, 0.20, 1.6), "Kileleshwa": (1.40, 5, 0.12, 1.2),
        "Karen": (1.80, 15, 0.30, 3.0), "South B": (1.00, 6, 0.03, 0.9),
        "Embakasi": (0.70, 14, 0.01, 0.9), "Kasarani": (0.75, 12, 0.02, 1.1),
        "Ruaka": (0.90, 16, 0.04, 1.2), "Syokimau": (0.80, 20, 0.02, 1.4),
        "Rongai": (0.65, 25, 0.01, 1.6), "Kahawa": (0.70, 16, 0.01, 1.3),
    }
    names = list(hoods)
    probs = np.array([.10, .10, .06, .08, .05, .10, .10, .09, .08, .07, .08, .09])
    probs = probs / probs.sum()
    nb = rng.choice(names, n, p=probs)
    mult = np.array([hoods[h][0] for h in nb])
    dist = np.clip(rng.normal([hoods[h][1] for h in nb], 1.5), 0.5, None)
    sqft = np.clip(rng.lognormal(np.log(1450), 0.42, n), 400, 6500)
    bedrooms = np.clip(np.round(sqft / 520 + rng.normal(0, 0.6, n)), 1, 7).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, n), 1, None)
    age = np.clip(rng.exponential(12, n), 0, 60).round(0)
    garage = rng.choice([0, 1, 2], n, p=[.35, .45, .20])
    pool = (rng.random(n) < np.array([hoods[h][2] for h in nb])).astype(int)
    lot = sqft * rng.uniform(1.2, 3.5, n) * np.array([hoods[h][3] for h in nb])
    cond = rng.choice(["Poor", "Fair", "Good", "Excellent"], n, p=[.07, .28, .45, .20])
    cond_bonus = pd.Series(cond).map({"Poor": -2.5, "Fair": -0.8, "Good": 0.8, "Excellent": 2.5}).values

    base = (4 + 0.0075 * sqft + 1.2e-6 * sqft ** 2 + 1.2 * bathrooms + 0.6 * garage + 3.5 * pool
            - 0.12 * age - 0.06 * dist + cond_bonus + 0.0006 * lot)
    price = mult * base * rng.lognormal(0, 0.08, n)
    price = np.clip(price, 3, None).round(2)

    df = pd.DataFrame({
        "listing_id": np.arange(1, n + 1), "neighborhood": nb, "sqft": sqft.round(0).astype(int),
        "bedrooms": bedrooms, "bathrooms": bathrooms, "age_years": age.astype(int),
        "garage_spaces": garage, "has_pool": pool, "lot_size_sqft": lot.round(0),
        "condition": cond, "distance_to_cbd_km": dist.round(1), "price_million_kes": price,
    })
    # --- realism: missing values + a few data-entry typos (sqft x10) -------------
    df.loc[rng.random(n) < 0.06, "lot_size_sqft"] = np.nan
    df.loc[rng.random(n) < 0.04, "condition"] = np.nan
    typo_rows = rng.choice(n, 6, replace=False)
    df.loc[typo_rows, "sqft"] = df.loc[typo_rows, "sqft"] * 10
    df.to_csv(os.path.join(HERE, "house_prices.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 4. student_performance.csv (Project 2)
# ---------------------------------------------------------------------------
def make_students(n=1200):
    study = np.clip(rng.normal(12, 5, n), 0, 35)
    attend = np.clip(rng.normal(85, 10, n), 40, 100)
    prev = np.clip(rng.normal(62, 15, n), 10, 100)
    sleep = np.clip(rng.normal(7, 1.2, n), 4, 10)
    absences = rng.poisson((100 - attend) * 0.4)
    edu = rng.choice(["Primary", "Secondary", "College", "University"], n, p=[.10, .35, .30, .25])
    edu_bonus = pd.Series(edu).map({"Primary": 0, "Secondary": 1, "College": 2.5, "University": 4}).values
    internet = (rng.random(n) < 0.78).astype(int)
    tutoring = (rng.random(n) < 0.25).astype(int)
    extra = (rng.random(n) < 0.5).astype(int)
    gender = rng.choice(["F", "M"], n)
    school = rng.choice(["Public", "Private"], n, p=[.7, .3])
    motivation = np.clip(np.round(rng.normal(6, 2, n)), 1, 10).astype(int)

    score = (-2 + 0.55 * prev + 3.2 * np.sqrt(study) + 0.12 * attend - 0.35 * absences + 3 * tutoring
             + edu_bonus + 2.5 * internet - 0.8 * (sleep - 7.5) ** 2 + 1.0 * motivation
             + 1.5 * (school == "Private") + rng.normal(0, 4, n))
    score = np.clip(score, 0, 100).round(1)
    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1), "gender": gender, "school_type": school,
        "study_hours_per_week": study.round(1), "attendance_pct": attend.round(1),
        "previous_exam_score": prev.round(1), "sleep_hours": sleep.round(1), "absences": absences,
        "parental_education": edu, "internet_access": internet, "tutoring": tutoring,
        "extracurricular": extra, "motivation_1_10": motivation, "final_score": score,
    })
    df["passed"] = (df["final_score"] >= 50).astype(int)
    df.loc[rng.random(n) < 0.03, "sleep_hours"] = np.nan
    df.loc[rng.random(n) < 0.02, "parental_education"] = np.nan
    df.to_csv(os.path.join(HERE, "student_performance.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 5. loan_approval.csv (Lessons 09, 16, Project 3)  includes a LEAKAGE TRAP column
# ---------------------------------------------------------------------------
def make_loans(n=3000):
    age = rng.integers(21, 66, n)
    gender = rng.choice(["Female", "Male"], n)
    married = rng.choice(["Yes", "No"], n, p=[.6, .4])
    dependents = np.clip(rng.poisson(1.2, n), 0, 5)
    education = rng.choice(["Graduate", "Not Graduate"], n, p=[.65, .35])
    emp = rng.choice(["Salaried", "Self-Employed", "Business Owner"], n, p=[.55, .25, .20])
    income = rng.lognormal(np.log(80000), 0.55, n).round(-2)
    term = rng.choice([12, 24, 36, 60, 120, 180, 240], n, p=[.10, .15, .20, .25, .10, .10, .10])
    loan = (income * 12 * rng.uniform(0.15, 2.5, n)).round(-3)
    credit = np.clip(rng.normal(650, 80, n), 300, 850).round(0)
    hist = np.clip(rng.gamma(2.5, 3.0, n), 0, 30).round(1)
    n_loans = rng.poisson(1.0, n)
    debt = (n_loans * income * rng.uniform(0.03, 0.12, n)).round(-2)
    area = rng.choice(["Urban", "Semi-urban", "Rural"], n, p=[.45, .35, .20])
    collateral = (rng.random(n) < 0.45).astype(int)
    savings = rng.lognormal(np.log(150000), 1.0, n).round(-2)

    emi = loan / term * 1.15
    dti = (debt + emi) / income
    lti = loan / (income * 12)
    z = (0.2 + 0.014 * (credit - 650) - 4.0 * (dti - 0.30) - 0.8 * (lti - 0.8) + 0.9 * collateral
         + 0.04 * hist + 0.5 * (emp == "Salaried") - 0.25 * (area == "Rural")
         + 0.3 * (education == "Graduate") + 0.000002 * np.minimum(savings, 1e6) + rng.logistic(0, 0.7, n))
    approved = (z > 0).astype(int)

    df = pd.DataFrame({
        "loan_id": [f"LN{100000 + i}" for i in range(n)], "age": age, "gender": gender, "married": married,
        "dependents": dependents, "education": education, "employment_type": emp,
        "monthly_income_kes": income, "loan_amount_kes": loan, "loan_term_months": term,
        "credit_score": credit, "credit_history_years": hist, "existing_loans": n_loans,
        "monthly_debt_kes": debt, "property_area": area, "has_collateral": collateral,
        "savings_balance_kes": savings,
    })
    # LEAKAGE TRAP: an interest rate is only assigned AFTER a loan is approved.
    rate = np.clip(18 - (credit - 650) * 0.012 + rng.normal(0, 1.0, n), 9, 26).round(2)
    df["interest_rate_offered"] = np.where(approved == 1, rate, np.nan)
    df["approved"] = approved
    df.loc[rng.random(n) < 0.03, "credit_score"] = np.nan
    df.loc[rng.random(n) < 0.05, "savings_balance_kes"] = np.nan
    df.to_csv(os.path.join(HERE, "loan_approval.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 6. agri_prices.csv (Project 4)   monthly wholesale prices, 2018-2025
# ---------------------------------------------------------------------------
def make_agri(n_months=96):
    crops = {"Maize": (45, 0.0), "Tomatoes": (60, 1.6), "Potatoes": (40, 3.1),
             "Onions": (70, 4.4), "Cabbage": (30, 5.5), "Beans": (120, 2.2)}  # base KES/kg, season phase
    scales = {"Maize": 900, "Tomatoes": 400, "Potatoes": 600, "Onions": 250, "Cabbage": 350, "Beans": 150}
    markets = {"Nairobi": 1.00, "Mombasa": 1.12, "Kisumu": 0.94, "Nakuru": 0.90, "Eldoret": 0.86}
    dates = pd.date_range("2018-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)
    month = dates.month.values
    fuel = 105 * (1 + 0.006 * t) + rng.normal(0, 3, n_months).cumsum() * 0.5
    fuel = np.maximum(fuel, 90)
    rows = []
    for mk, mk_mult in markets.items():
        # bimodal rainfall: long rains Mar-May, short rains Oct-Dec
        seasonal_rain = (60 + 110 * np.exp(-((month - 4) ** 2) / 3.0) + 80 * np.exp(-((month - 11) ** 2) / 2.5))
        rain = np.clip(seasonal_rain * rng.lognormal(0, 0.35, n_months), 0, None)
        temp = 22 + 3.0 * np.sin((month - 2) / 12 * 2 * np.pi) + rng.normal(0, 0.8, n_months) + (6 if mk == "Mombasa" else 0)
        rain_lag = np.r_[np.full(2, rain.mean()), rain[:-2]]  # rain two months ago drives supply
        for crop, (base, phase) in crops.items():
            scale = scales[crop]
            supply = scale * np.clip(0.55 + rain_lag / rain_lag.mean() * 0.45, 0.3, 2.2) * rng.lognormal(0, 0.10, n_months)
            e = np.zeros(n_months)
            for i in range(1, n_months):
                e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.05)
            seasonal = 1 + 0.12 * np.sin(month / 12 * 2 * np.pi + phase)
            price = (base * mk_mult * (1 + 0.004 * t) * (fuel / 105) ** 0.25 * seasonal
                     * (supply / scale) ** -0.45 * np.exp(e))
            for i in range(n_months):
                rows.append((dates[i], mk, crop, round(rain[i], 1), round(temp[i], 1), round(fuel[i], 1),
                             round(supply[i], 1), round(price[i], 2)))
    df = pd.DataFrame(rows, columns=["date", "market", "crop", "rainfall_mm", "avg_temp_c",
                                     "fuel_price_kes_per_litre", "supply_tons", "price_kes_per_kg"])
    df = df.sort_values(["date", "market", "crop"]).reset_index(drop=True)
    df.loc[rng.random(len(df)) < 0.02, "rainfall_mm"] = np.nan
    df.to_csv(os.path.join(HERE, "agri_prices.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 7. sms_spam_mini.csv (Lesson 14)  tiny text dataset for Naive Bayes
# ---------------------------------------------------------------------------
def make_sms(n_ham=420, n_spam=180):
    names = ["Wanjiku", "Otieno", "Amina", "Kamau", "Njeri", "Brian", "Faith", "Mwangi", "Achieng", "Hassan"]
    places = ["the office", "campus", "the market", "church", "home", "the stage", "town", "the gym"]
    times = ["tonight", "tomorrow", "at 5pm", "this weekend", "on Friday", "after class", "at lunch"]
    things = ["the report", "the meeting notes", "your charger", "the book", "the keys", "my laptop", "the photos"]
    ham_t = [
        "Hey {n}, are you free {t}? Let's meet at {p}",
        "Can you send me {th} before {t}?",
        "I will be late to {p}, please start without me",
        "Did the team win the match? Tell me everything {t}",
        "Don't forget to carry {th} when you come to {p}",
        "Thanks {n}! I got your message, see you {t}",
        "Mum says dinner is ready, come home {t}",
        "Are you still coming to {p} {t}? Call me when you leave",
        "Lunch {t}? I am free to talk now if you want",
        "Happy birthday {n}! Hope you win big in life, enjoy your day",
        "Please call me when you reach {p}, I have a free minute",
        "I left {th} at {p}, can you pick it up {t}?",
        "The lecture was moved to {t}, tell {n} too",
        "Sorry I missed your call. Let me call you back {t}",
        "Good morning {n}, how was your trip to {p}?",
    ]
    spam_t = [
        "WINNER! You have won KES {amt}. Claim your prize now by calling {ph}",
        "FREE entry in our weekly draw! Text WIN to {sc} to claim your prize",
        "Congratulations! You are selected for a cash prize of KES {amt}. Reply YES to claim",
        "URGENT! Your account will be suspended. Send your PIN to {ph} to verify now",
        "Get a loan of up to KES {amt} instantly, no documents needed. Call {ph} now",
        "Free airtime offer! Click the link and claim {amt} bonus today only",
        "You have been chosen to receive a FREE phone. Text CLAIM to {sc}",
        "Limited offer: double your money in 24 hours. Send KES {amt} to {ph}",
        "Dear customer, you won a gift voucher worth KES {amt}. Call {ph} to claim now",
        "Cheap deals!! Buy 1 get 1 free, offer ends today, text BUY to {sc}",
    ]
    rows = []
    for _ in range(n_ham):
        t = ham_t[rng.integers(len(ham_t))]
        rows.append((t.format(n=rng.choice(names), p=rng.choice(places), t=rng.choice(times),
                              th=rng.choice(things)), "ham"))
    for _ in range(n_spam):
        t = spam_t[rng.integers(len(spam_t))]
        rows.append((t.format(amt=f"{rng.choice([5, 10, 20, 50, 100, 500])},000",
                              ph=f"07{rng.integers(10000000, 99999999)}",
                              sc=str(rng.integers(20000, 99999))), "spam"))
    df = pd.DataFrame(rows, columns=["message", "label"]).sample(frac=1, random_state=1).reset_index(drop=True)
    # real data is messy: flip a few labels
    flip = rng.choice(len(df), 8, replace=False)
    df.loc[flip, "label"] = df.loc[flip, "label"].map({"ham": "spam", "spam": "ham"})
    df.to_csv(os.path.join(HERE, "sms_spam_mini.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# 8. mystery_A/B/C.csv (Lesson 17)  three datasets with HIDDEN structure.
#    No peeking at this function before you have solved the mystery challenge! :)
# ---------------------------------------------------------------------------
def make_mystery():
    r = np.random.default_rng(10)                                   # A: small n, many features, few matter, linear
    XA = r.normal(size=(150, 60)); bA = np.zeros(60); bA[:5] = [3, -2, 1.5, 2.5, -1]
    A = pd.DataFrame(XA, columns=[f"x{i + 1}" for i in range(60)]); A["y"] = (XA @ bA + r.normal(0, 1.5, 150)).round(3)
    A.round(4).to_csv(os.path.join(HERE, "mystery_A.csv"), index=False)

    r = np.random.default_rng(11)                                   # B: big n, few features, strongly non-linear with interactions
    XB = r.uniform(-2, 2, (3000, 8))
    yB = 3 * np.sin(XB[:, 0] * XB[:, 1]) + XB[:, 2] ** 2 + 2 * (XB[:, 3] > 0) * XB[:, 4] + r.normal(0, 0.3, 3000)
    B = pd.DataFrame(XB, columns=[f"x{i + 1}" for i in range(8)]); B["y"] = yB.round(3)
    B.round(4).to_csv(os.path.join(HERE, "mystery_B.csv"), index=False)

    r = np.random.default_rng(12)                                   # C: rare positives (~4.5%) defined by non-linear regions
    XC = r.normal(size=(4000, 12))
    pos = ((XC[:, 0] ** 2 + XC[:, 1] ** 2) > 7.0) | ((XC[:, 2] * XC[:, 3]) > 3.2)
    yC = pos.astype(int); flip = r.random(4000) < 0.01; yC = np.where(flip, 1 - yC, yC)
    C = pd.DataFrame(XC, columns=[f"x{i + 1}" for i in range(12)]); C["y"] = yC
    C.round(4).to_csv(os.path.join(HERE, "mystery_C.csv"), index=False)
    return C


if __name__ == "__main__":
    for fn in [make_advertising, make_position_salaries, make_house_prices, make_students,
               make_loans, make_agri, make_sms, make_mystery]:
        d = fn()
        print(f"{fn.__name__:<26} -> {d.shape}")
