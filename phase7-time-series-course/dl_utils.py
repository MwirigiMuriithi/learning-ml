"""
dl_utils.py -- PyTorch helpers for Lesson 7 (and the capstone project).

Single source of truth for the deep-learning forecasters, so that notebooks stay short.
Read this file! Everything is deliberately small and commented.

Design choices
--------------
* Input  : a WINDOW of the last L weeks.  Shape (batch, L, n_features)   [batch_first everywhere]
* Output : cumulative log-returns for horizons 1..H   (direct multi-horizon forecast)
* Features per week are scale-free (returns, deviations, drivers, calendar), standardised with
  statistics from the TRAINING windows only.
* Early stopping uses the most recent slice of the training windows (chronological, never random).
"""
import copy, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from course_utils import fourier_terms

torch.set_num_threads(1)

FEATURES = ["ret", "dev_ma12", "rain6", "fuel", "fest", "sin1", "cos1", "sin2", "cos2"]


# ------------------------------------------------------------------ data -> windows
def make_step_features(df, price="price"):
    """One row of features per week (all known at that week -> usable inside an input window)."""
    lp = np.log(df[price])
    f = pd.DataFrame(index=df.index)
    f["ret"] = lp.diff()                                         # weekly log return
    f["dev_ma12"] = lp - lp.rolling(12).mean()                   # distance from the 12-week mean
    f["rain6"] = df["rainfall_mm"].rolling(6).mean() / 30        # recent rainfall
    f["fuel"] = np.log(df["fuel_index"] / 100)                   # fuel cost level (log)
    f["fest"] = df["festival_flag"].astype(float)
    ft = fourier_terms(df.index, K=2)
    for c in ("sin1", "cos1", "sin2", "cos2"):
        f[c] = ft[c]
    return f[FEATURES], lp


def make_windows(step_feats, lp, L=52, H=8):
    """
    Build supervised windows.
        X[i] = features of weeks  [t-L+1 ... t]           (the L weeks up to and including the origin t)
        Y[i] = [lp[t+1]-lp[t], ..., lp[t+H]-lp[t]]        (cumulative log-returns for h = 1..H)
    Returns X (N, L, F), Y (N, H), origins (N,) = integer position of t in the DataFrame.
    """
    F = step_feats.values.astype(np.float32); lpv = lp.values
    ok_start = int(np.argmax(~np.isnan(F).any(axis=1)))          # first row with all features defined
    Xs, Ys, org = [], [], []
    for t in range(ok_start + L - 1, len(lpv) - H):
        Xs.append(F[t - L + 1: t + 1])
        Ys.append(lpv[t + 1: t + H + 1] - lpv[t])
        org.append(t)
    return np.stack(Xs), np.stack(Ys).astype(np.float32), np.array(org)


class Standardizer:
    """Fit mean/std on TRAIN data only (never on val/test!)."""
    def fit(self, X, Y):
        self.mx = X.reshape(-1, X.shape[-1]).mean(0); self.sx = X.reshape(-1, X.shape[-1]).std(0) + 1e-8
        self.my = Y.mean(0); self.sy = Y.std(0) + 1e-8
        return self
    def tx(self, X): return ((X - self.mx) / self.sx).astype(np.float32)
    def ty(self, Y): return ((Y - self.my) / self.sy).astype(np.float32)
    def iy(self, Y): return Y * self.sy + self.my


# ------------------------------------------------------------------------ models
class SeqForecaster(nn.Module):
    """RNN / LSTM / GRU encoder + linear head that predicts H cumulative returns."""
    def __init__(self, kind, n_features, H=8, hidden=32, dropout=0.1):
        super().__init__()
        self.kind = kind
        cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
        self.rnn = cell(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, H)
    def forward(self, x):
        out, _ = self.rnn(x)                    # out: (batch, L, hidden) -- hidden state at every week
        return self.head(self.drop(out[:, -1])) # use the LAST hidden state: it has "read" the whole window


class AttnBlock(nn.Module):
    """One Transformer encoder block: self-attention + feed-forward, with residuals and LayerNorm."""
    def __init__(self, d, heads=2, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.last_attn = None
    def forward(self, x):
        h = self.n1(x)
        a, w = self.attn(h, h, h, need_weights=True, average_attn_weights=True)
        self.last_attn = w.detach()             # (batch, L, L): how much each week attends to every other week
        x = x + self.drop(a)
        return x + self.drop(self.ff(self.n2(x)))


class TransformerForecaster(nn.Module):
    def __init__(self, n_features, H=8, L=52, d=32, heads=2, layers=2, dropout=0.1):
        super().__init__()
        self.kind = "transformer"
        self.inp = nn.Linear(n_features, d)
        self.pos = nn.Parameter(torch.zeros(1, L, d))          # learned positional embedding: "which week of the window am I?"
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([AttnBlock(d, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, H)
    def forward(self, x):
        h = self.inp(x) + self.pos[:, : x.shape[1]]
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h)[:, -1])


def build_model(kind, n_features, H=8, L=52, hidden=32, dropout=0.1):
    if kind == "transformer":
        return TransformerForecaster(n_features, H, L, d=hidden, dropout=dropout)
    return SeqForecaster(kind, n_features, H, hidden, dropout)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------- training
def train_model(kind, Xtr, Ytr, hidden=32, dropout=0.1, epochs=60, lr=3e-3, weight_decay=1e-3,
                batch_size=32, patience=10, val_frac=0.15, gap=8, seed=0, verbose=False):
    """
    Fit one forecaster.  Returns (model, standardizer, history).
    * windows are standardised with TRAIN statistics
    * the last `val_frac` of the windows (after a `gap` of windows to avoid target overlap) are the
      validation set used for EARLY STOPPING -- chronological, never random
    * shuffling windows *inside* the training set for SGD is fine: each window is self-contained
    """
    torch.manual_seed(seed); np.random.seed(seed)
    n = len(Xtr); n_val = max(int(n * val_frac), 10)
    tr_idx = np.arange(0, n - n_val - gap); va_idx = np.arange(n - n_val, n)
    sc = Standardizer().fit(Xtr[tr_idx], Ytr[tr_idx])
    xt, yt = torch.tensor(sc.tx(Xtr[tr_idx])), torch.tensor(sc.ty(Ytr[tr_idx]))
    xv, yv = torch.tensor(sc.tx(Xtr[va_idx])), torch.tensor(sc.ty(Ytr[va_idx]))
    model = build_model(kind, Xtr.shape[-1], Ytr.shape[-1], Xtr.shape[1], hidden, dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    best, best_state, wait, hist = math.inf, None, 0, {"train": [], "val": []}
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(xt)); tot = 0.0
        for i in range(0, len(xt), batch_size):
            b = perm[i: i + batch_size]
            opt.zero_grad(); loss = loss_fn(model(xt[b]), yt[b]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)          # guards against exploding gradients
            opt.step(); tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad(): vloss = loss_fn(model(xv), yv).item()
        hist["train"].append(tot / len(xt)); hist["val"].append(vloss)
        if vloss < best - 1e-5: best, best_state, wait = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience: break
        if verbose and ep % 10 == 0: print(f"epoch {ep:3d}  train {hist['train'][-1]:.4f}  val {vloss:.4f}")
    model.load_state_dict(best_state); model.eval()
    return model, sc, hist


def predict(model, sc, X):
    """Returns predicted cumulative log-returns, shape (N, H), in original units."""
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(sc.tx(X))).numpy()
    return sc.iy(out)


# ------------------------------------------------------------- walk-forward (DL)
def walk_forward_dl(df, kind, horizons=(1, 4, 8), L=52, H=8, initial=470, step=16, price="price", seeds=(0,), **train_kw):
    """
    Rolling-origin evaluation with refitting at every origin (expanding window).  Same output schema as
    walk_forward(): origin, target_date, h, actual, forecast.   If several seeds are given, the
    forecasts are averaged (a tiny ensemble -- it reduces the run-to-run variance of neural nets).
    """
    sf, lp = make_step_features(df, price)
    X, Y, org = make_windows(sf, lp, L, H)
    p = df[price].values; rows = []
    for origin in range(initial, len(df) - max(horizons) + 1, step):
        t = origin - 1                                         # last observed week
        tr = org + H <= t                                      # windows whose full target lies inside the known data
        te = org == t
        if te.sum() == 0: continue
        preds = [predict(*train_model(kind, X[tr], Y[tr], seed=s, **train_kw)[:2], X[te])[0] for s in seeds]
        pr = np.mean(preds, axis=0)
        for h in horizons:
            rows.append((df.index[t], df.index[t + h], h, p[t + h], p[t] * np.exp(pr[h - 1])))
    return pd.DataFrame(rows, columns=["origin", "target_date", "h", "actual", "forecast"])


# ------------------------------------------------------------ single-split evaluation
def evaluate_split(df, kind, t0=469, h=8, L=52, H=8, seeds=(0,), price="price", **train_kw):
    """
    Train ONCE on the windows whose targets are fully known at week t0, then forecast the price h weeks
    ahead from every later origin (no refitting).  Handy for quick experiments (seeds, window length,
    hyper-parameters).  Returns a dict with per-seed MAE, ensemble MAE, naive MAE and the raw arrays.
    """
    sf, lp = make_step_features(df, price)
    X, Y, org = make_windows(sf, lp, L, H)
    tr, te = org + H <= t0, org > t0
    p = df[price].values
    base, actual = p[org[te]], p[org[te] + h]
    fcs, hists = [], []
    for s in seeds:
        model, sc, hist = train_model(kind, X[tr], Y[tr], seed=s, **train_kw)
        fcs.append(base * np.exp(predict(model, sc, X[te])[:, h - 1])); hists.append(hist)
    mae = lambda a, b: float(np.mean(np.abs(a - b)))
    return dict(per_seed_mae=[mae(actual, f) for f in fcs], ens_mae=mae(actual, np.mean(fcs, axis=0)),
                naive_mae=mae(actual, base), best_val=[min(h_["val"]) for h_ in hists],
                actual=actual, base=base, forecasts=fcs, hists=hists, n_test=int(te.sum()))
