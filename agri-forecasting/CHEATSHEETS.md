# Cheat sheets

## A · Time series (Phase 7)

### Components & tests
| Item | Rule of thumb |
|---|---|
| Additive vs multiplicative | swings grow with level → **take logs** (multiplicative → additive) |
| Stationary? | ADF p < .05 **and** KPSS p > .05; fix with differencing (`diff()`, seasonal `diff(s)`), logs |
| Seasonal strength | `1 - Var(resid)/Var(seasonal+resid)` from STL |
| Return | `np.log(y).diff()` ≈ % change |

### ACF / PACF fingerprints
| Pattern | Model |
|---|---|
| ACF decays slowly | non-stationary → difference |
| PACF cuts off at *p*, ACF tails off | **AR(p)** |
| ACF cuts off at *q*, PACF tails off | **MA(q)** |
| spikes at lag *s, 2s…* | seasonality → SARIMA(P,D,Q)ₛ or Fourier terms |

### Model chooser
`trend? seasonality?` → none: SES · trend: Holt (damped for long horizons) · both: Holt-Winters (mult. if swings scale) · autocorrelation structure / exogenous drivers: ARIMA/SARIMAX · many features, nonlinearity: Ridge → RF/XGBoost (predict **changes**, not levels).

### Metrics
| Metric | Use | Caution |
|---|---|---|
| MAE | explain to farmers | – |
| RMSE | punish big misses | outlier-dominated |
| MAPE | % error | **undefined near 0 actuals** |
| sMAPE | bounded % | odd near 0 |
| MASE | < 1 beats naive | needs training scale (`m`) |

### Evaluation protocol
Chronological split · **walk-forward** (expanding or sliding) · report **error by horizon** · baselines: naive, seasonal naive, drift · refit schedule stated · embargo ≥ horizon between blocks.

### Leakage checklist
- [ ] features use only data available at the origin (**shift first, then roll**)
- [ ] no centered windows / `bfill` / interpolation across the split
- [ ] scalers, encoders, selected hyper-parameters fitted on **training only**
- [ ] exogenous variables known at forecast time (lag them or forecast them)
- [ ] corrupt the target → rebuild features → nothing at row *t* changes

---
## B · Deep learning (Phase 8)

### The training loop (memorise)
```python
for epoch in range(n_epochs):
    model.train()
    for xb, yb in loader:
        opt.zero_grad()            # 1
        loss = loss_fn(model(xb), yb)   # 2, 3
        loss.backward()            # 4
        opt.step()                 # 5
    model.eval()
    with torch.no_grad(): val_loss = ...
```

### Loss ↔ output layer
| Task | Last layer | Loss (PyTorch) |
|---|---|---|
| regression | none | `MSELoss` / `L1Loss` / `HuberLoss` |
| binary | 1 logit | `BCEWithLogitsLoss` (targets float 0/1) |
| multi-class | *n* logits | `CrossEntropyLoss` (targets **class ids**, `long`): never apply softmax first |

### Shapes
| Data | PyTorch | Keras |
|---|---|---|
| tabular | `(N, F)` | `(N, F)` |
| sequences | `(N, T, F)` with `batch_first=True` | `(N, T, F)` |
| images | `(N, C, H, W)` | `(N, H, W, C)` |
Conv output: `⌊(W-K+2P)/S⌋+1` · conv params: `C_out(C_in·K²+1)` · dense: `n_in·n_out+n_out`.

### Defaults that usually work
AdamW lr 1e-3 (weight decay 1e-4…1e-2) · batch 32–128 · He init + ReLU · BatchNorm in CNNs · dropout 0.1–0.5 · cosine or one-cycle schedule · early stopping (patience 8–15) · clip gradients (norm 1.0) for RNNs · LSTM forget-bias 1–3.

### Debugging ladder
1. Print shapes and dtypes. 2. Overfit **one tiny batch** to ~0 loss. 3. Check loss at init ≈ `ln(n_classes)`. 4. Plot train vs validation curves. 5. Gradient-check new layers. 6. Compare with a linear baseline. 7. Try several seeds.

| Curves | Diagnosis → fix |
|---|---|
| both high | underfit → bigger model, more epochs, better features |
| train low, val high | overfit → early stop, augmentation, dropout, weight decay, more data |
| loss NaN | LR too high / bad scaling / log(0) → lower LR, clip, stable softmax |
| loss flat | LR too low, dead ReLUs, wrong labels/shape → LR range test, check data |

### Choosing an architecture
tabular → linear/GBM first · images → CNN (+augmentation, transfer learning) · short sequences → GRU/LSTM · long sequences / many series / lots of data → Transformer · always benchmark against the simple model.
