# 🧾 Deep-Learning Cheat Sheet

## The five-step training loop (PyTorch)
```python
model.train()
for xb, yb in train_loader:
    pred = model(xb)                 # 1. forward
    loss = loss_fn(pred, yb)         # 2. loss
    optimizer.zero_grad()            # 3. clear old gradients
    loss.backward()                  # 4. backprop
    optimizer.step()                 # 5. update weights
model.eval()
with torch.no_grad(): ...            # evaluate / predict
```
Packaged version: `dl.fit(model, train_dl, val_dl, loss_fn, optimizer, epochs, scheduler=..., early_stopper=..., grad_clip=...)`.

## Shapes
| Data | Shape |
|:--|:--|
| Tabular batch | `(N, features)` |
| Images (PyTorch) | `(N, C, H, W)`: **channels first**; NumPy/PIL/Keras use `(N, H, W, C)` |
| Sequences | `(N, T, features)` with `batch_first=True` |
| Dense layer | `(N, n_in) @ (n_in, n_out) → (N, n_out)` · params `n_in·n_out + n_out` |
| Conv layer | out size `⌊(n + 2p − k)/s⌋ + 1` · params `(k²·C_in + 1)·C_out` |
| LSTM / GRU / RNN params | `4× / 3× / 1× (h·i + h² + 2h)` |
| Transformer block params | `≈ 4d² + 2·d·d_ff + 9d + d_ff` |

## Which output layer + loss?
| Task | Output layer | Loss (PyTorch) | Labels |
|:--|:--|:--|:--|
| Regression | Linear (1 neuron) | `MSELoss` / `L1Loss` / `HuberLoss` | float, same shape |
| Binary | 1 logit | `BCEWithLogitsLoss` | float {0,1} |
| Multi-class | k logits | `CrossEntropyLoss` (**raw logits**, no softmax!) | int64 class index |
| Multi-label | k logits | `BCEWithLogitsLoss` | float k-hot |

Sanity check: **initial loss ≈ ln k** (10 classes → 2.30).

## Activations
ReLU = default hidden · GELU = Transformers · sigmoid = binary output / gates · tanh = RNN state · softmax = multi-class probabilities.
Init: ReLU → He/Kaiming · tanh/sigmoid → Xavier/Glorot.

## Optimiser defaults
`Adam(lr=1e-3)` / `AdamW(lr=1e-3, weight_decay=0.01–0.1)` · `SGD(lr=0.1, momentum=0.9)` for CNNs (tune carefully).
Sweep LR on a **log scale**. Batch size 32–256. Clip gradients (`clip_grad_norm_(…, 1.0)`) for RNNs.

## Regularisation menu (in order of bang-for-buck)
More data / **augmentation** → **early stopping** → **dropout** (0.1–0.5) → **weight decay** → smaller model → BatchNorm/LayerNorm (mostly for optimisation).
Augmentation must **preserve the label** and is applied to **training data only**.

## Debugging checklist
| Symptom | Likely cause |
|:--|:--|
| Shape error | Print `x.shape` at every layer; `Flatten` size; channels first vs last |
| `expected Long / Float` | Labels must be `int64`; features `float32` |
| Loss `nan` | LR too high; bad inputs; `log(0)`; clip gradients |
| Loss stuck ≈ ln k | LR too low; dead ReLUs; labels/inputs misaligned; **can't overfit one batch ⇒ bug** |
| Loss never near 0 on easy data | Softmax applied before `CrossEntropyLoss` |
| Train ↑ val ↓ | Overfitting → regularise / more data |
| Different predictions each run | Forgot `model.eval()` (dropout / BatchNorm) |
| Gradients explode over steps | Forgot `zero_grad()` |
| Great test score, bad in production | Leakage or distribution shift (look at splits & data sources) |

## Evaluation
Train / validation / **test (once)**. Baselines first (majority class, naive forecast, logistic/ridge). Report per-class **recall/precision/F1**,
confusion matrix, **top-k**, **calibration / confidence**, and for forecasting **MAE vs naive**, **interval coverage**. Split time series **chronologically**.

## PyTorch ↔ Keras
| PyTorch | Keras |
|:--|:--|
| `nn.Linear(i, o)` | `Dense(o)` |
| `nn.Conv2d(c_in, c_out, k, padding=1)` | `Conv2D(c_out, k, padding="same")` |
| `nn.CrossEntropyLoss()` | `SparseCategoricalCrossentropy(from_logits=True)` |
| hand-written loop / `dl.fit` | `model.compile(...)` + `model.fit(...)` |
| `torch.save(model.state_dict(), f)` | `model.save("m.keras")` |
| `loss.backward()` | `tf.GradientTape()` |
| `DataLoader` | `tf.data.Dataset` |

## Course helper API (`utils/dlcourse.py`)
`set_seed` · `get_device` · `check` · `load_mnist/fashion/leaves/prices` · `fit` · `evaluate` · `predict_proba` · `count_params` · `EarlyStopper` ·
`show_images` · `plot_history` · `plot_confusion` · `plot_decision_boundary`
