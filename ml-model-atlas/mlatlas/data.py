"""Synthetic but practical datasets. Every generator is seeded, so every run is reproducible."""
import numpy as np


def _r(seed):
    return np.random.default_rng(seed)


def standardize(X, mu=None, sd=None):
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) + 1e-9 if sd is None else sd
    return (X - mu) / sd, mu, sd


def train_test_split(X, y, test=.3, seed=0):
    p = _r(seed).permutation(len(y)); k = int(len(y) * (1 - test))
    return X[p[:k]], X[p[k:]], y[p[:k]], y[p[k:]]


# ---------- regression ----------
def sales_vs_ads(n=60, seed=0):
    """Weekly ad spend (k$) -> units sold. True line: 3 + 2.2*x."""
    r = _r(seed); x = r.uniform(0, 10, n)
    return x[:, None], 3.0 + 2.2 * x + r.normal(0, 2.0, n)


def house_prices(n=300, seed=0):
    """Size, rooms, age, distance to centre -> price (k$)."""
    r = _r(seed)
    size = r.uniform(40, 250, n); rooms = np.clip(np.round(size / 45 + r.normal(0, .7, n)), 1, 8)
    age = r.uniform(0, 50, n); dist = r.uniform(1, 30, n)
    price = 20 + 1.1 * size + 8 * rooms - 0.8 * age - 2.5 * dist + r.normal(0, 15, n)
    return np.c_[size, rooms, age, dist], price, ["size_m2", "rooms", "age_yrs", "dist_km"]


def curve_fn(x):
    return np.sin(3 * x) + .5 * x


def curve(n=40, seed=0, noise=.25):
    """Noisy samples of a smooth non-linear signal (for bias/variance, trees, boosting, GPs)."""
    r = _r(seed); x = r.uniform(-1, 1, n)
    return x, curve_fn(x) + r.normal(0, noise, n)


# ---------- classification ----------
def loan_default(n=300, seed=0):
    """income (k$/yr), debt ratio (%) -> default (1) or repaid (0)."""
    r = _r(seed)
    income = np.clip(r.normal(55, 18, n), 12, 130); debt = np.clip(r.beta(2.2, 4.5, n) * 100, 2, 95)
    z = 0.085 * (debt - 32) - 0.07 * (income - 55)
    y = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return np.c_[income, debt], y, ["income_k", "debt_ratio_pct"]


def spam_emails(n=400, seed=0):
    """num_links, caps_ratio_pct -> spam (1) / ham (0)."""
    r = _r(seed); y = (r.random(n) < .35).astype(int)
    links = np.where(y == 1, r.normal(6, 2.5, n), r.normal(1.5, 1.2, n))
    caps = np.where(y == 1, r.normal(35, 15, n), r.normal(8, 5, n))
    return np.c_[np.clip(links, 0, None), np.clip(caps, 0, 100)], y


def blobs(n=300, k=3, seed=0, std=1.0, spread=4.0):
    r = _r(seed); ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + .5
    C = spread * np.c_[np.cos(ang), np.sin(ang)]
    y = r.integers(0, k, n)
    return C[y] + r.normal(0, std, (n, 2)), y


def moons(n=300, noise=.2, seed=0):
    r = _r(seed); m = n // 2
    t1, t2 = r.uniform(0, np.pi, m), r.uniform(0, np.pi, m)
    X = np.r_[np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), .5 - np.sin(t2)]] + r.normal(0, noise, (2 * m, 2))
    y = np.r_[np.zeros(m), np.ones(m)].astype(int); p = r.permutation(2 * m)
    return X[p], y[p]


def circles(n=300, noise=.1, seed=0, factor=.5):
    r = _r(seed); m = n // 2
    t1, t2 = r.uniform(0, 2 * np.pi, m), r.uniform(0, 2 * np.pi, m)
    X = np.r_[np.c_[np.cos(t1), np.sin(t1)], factor * np.c_[np.cos(t2), np.sin(t2)]] + r.normal(0, noise, (2 * m, 2))
    y = np.r_[np.zeros(m), np.ones(m)].astype(int); p = r.permutation(2 * m)
    return X[p], y[p]


def spirals(n=300, k=3, noise=.2, seed=0):
    r = _r(seed); m = n // k; X = []; y = []
    for j in range(k):
        rad = np.linspace(.05, 1, m); t = np.linspace(j * 4, (j + 1) * 4, m) + r.normal(0, noise, m)
        X.append(np.c_[rad * np.sin(t), rad * np.cos(t)]); y += [j] * m
    X = np.vstack(X); y = np.array(y); p = r.permutation(len(y))
    return X[p], y[p]


def xor(n=200, seed=0, std=.35):
    r = _r(seed); b = r.integers(0, 2, (n, 2))
    return (b * 2 - 1) * 1.5 + r.normal(0, std, (n, 2)), b[:, 0] ^ b[:, 1]


# ---------- unsupervised ----------
def mall_customers(n=300, seed=0):
    """annual income (k$), spending score (1-100); 5 hidden customer segments."""
    r = _r(seed); C = np.array([[25, 20], [25, 80], [55, 50], [85, 20], [85, 80]], float)
    y = r.integers(0, 5, n)
    return np.clip(C[y] + r.normal(0, [6, 7], (n, 2)), 1, 100), y


def sensor_anomalies(n=300, n_out=12, seed=0):
    """Machine temperature vs vibration: a healthy cloud plus a few faults."""
    r = _r(seed); cov = [[1, .6], [.6, 1]]
    X = r.multivariate_normal([0, 0], cov, n)
    out = r.uniform(-5, 5, (n_out, 2)); out = out[np.linalg.norm(out, axis=1) > 3]
    y = np.r_[np.zeros(n), np.ones(len(out))].astype(int)
    return np.r_[X, out], y


ITEMS = ["bread", "butter", "milk", "eggs", "jam", "coffee", "sugar", "tea", "chips", "soda", "beer", "diapers"]


def baskets(n=600, seed=0):
    """Supermarket baskets with planted rules (bread->butter->jam, coffee->sugar, diapers->beer, chips->soda)."""
    r = _r(seed); out = []
    base = {"bread": .45, "milk": .45, "eggs": .3, "coffee": .25, "tea": .2, "chips": .2, "soda": .1,
            "beer": .06, "diapers": .12, "sugar": .06}
    for _ in range(n):
        s = {k for k, p in base.items() if r.random() < p}
        if "bread" in s and r.random() < .7: s.add("butter")
        if "butter" in s and r.random() < .5: s.add("jam")
        if "coffee" in s and r.random() < .6: s.add("sugar")
        if "diapers" in s and r.random() < .6: s.add("beer")
        if "chips" in s and r.random() < .6: s.add("soda")
        if s: out.append(frozenset(s))
    return out


def ratings(n_users=40, n_items=30, seed=0, obs=.35):
    """Movie ratings 1-5 generated from 2 hidden taste dimensions; only `obs` fraction observed."""
    r = _r(seed); ut, it = r.uniform(0, 2 * np.pi, n_users), r.uniform(0, 2 * np.pi, n_items)
    U = np.c_[np.cos(ut), np.sin(ut)] * r.uniform(.6, 1.2, (n_users, 1))
    V = np.c_[np.cos(it), np.sin(it)] * r.uniform(.6, 1.2, (n_items, 1))
    R = np.clip(3 + 1.6 * U @ V.T + r.normal(0, .3, (n_users, n_items)), 1, 5)
    return R, r.random(R.shape) < obs, U, V


SHAPE_NAMES = ["horizontal", "vertical", "diag \\", "diag /"]


def shapes(n_per=60, seed=0):
    """Tiny 8x8 noisy images of oriented strokes (a mini-MNIST for CNNs / t-SNE)."""
    r = _r(seed); imgs = []; labs = []; i = np.arange(1, 7)
    for c in range(4):
        for _ in range(n_per):
            im = np.zeros((8, 8)); o = r.integers(-1, 2)
            if c == 0: im[r.integers(1, 7), 1:7] = 1
            elif c == 1: im[1:7, r.integers(1, 7)] = 1
            elif c == 2: im[i, np.clip(i + o, 0, 7)] = 1
            else: im[i, np.clip(7 - i + o, 0, 7)] = 1
            imgs.append(im + r.normal(0, .25, im.shape)); labs.append(c)
    X = np.array(imgs); y = np.array(labs); p = r.permutation(len(y))
    return X[p], y[p]


def time_series(n=180, seed=0):
    """Monthly sales: trend + yearly seasonality + noise."""
    r = _r(seed); t = np.arange(n)
    return 100 + .6 * t + 15 * np.sin(2 * np.pi * t / 12) + 6 * np.sin(4 * np.pi * t / 12 + 1) + r.normal(0, 3.5, n)
