"""
dlcourse.py: the small helper library used by every notebook
=============================================================

Philosophy: the *lessons* teach the ideas with code you can read line by line.
This file only holds the boring plumbing (loading data, drawing pictures) so the
notebooks stay focused.  Everything here is short and readable: open it and look!

Contents
--------
  Reproducibility : set_seed, get_device
  Self-checking   : check            (the ✅ / ❌ you see after each problem)
  Data loaders    : load_mnist, load_fashion, load_leaves, load_prices
  Training        : fit, evaluate, predict_proba, count_params, EarlyStopper
  Plots           : show_images, plot_history, plot_confusion, plot_decision_boundary
"""
from __future__ import annotations

import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STRICT = os.environ.get("DL_STRICT") == "1"   # used only when the course itself is tested

FASHION_CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
LEAF_CLASSES = ["healthy", "gray_leaf_spot", "northern_leaf_blight", "common_rust", "maize_streak"]


# =============================================================================
# Reproducibility
# =============================================================================
def set_seed(seed: int = 42) -> None:
    """Make random numbers repeatable (Python, NumPy and PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:          # PyTorch not installed yet (fine for the NumPy-only lessons)
        pass


def get_device():
    """Return the best available device: CUDA GPU, Apple-silicon GPU (mps) or CPU."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_plots() -> None:
    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": .25,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.figsize": (7, 4)})


setup_plots()


# =============================================================================
# Self-checking
# =============================================================================
def check(name: str, condition: bool, hint: str = "") -> None:
    """Print a friendly ✅ / ❌ for one test.  (Raises only in the course's own test-suite.)"""
    if bool(condition):
        print(f"✅ {name}")
    else:
        print(f"❌ {name}" + (f"   → hint: {hint}" if hint else ""))
        if STRICT:
            raise AssertionError(f"check failed: {name}")


# =============================================================================
# Data loaders
# =============================================================================
def _norm(x, flatten):
    x = x.astype("float32") / 255.0
    return x.reshape(len(x), -1) if flatten else x


def load_mnist(n_train=None, n_test=None, flatten=False):
    """Real MNIST handwritten digits (20k train / 5k test subset).
    Returns x_train, y_train, x_test, y_test.  Images are float32 in [0,1], shape (N,28,28)
    (or (N,784) if flatten=True)."""
    d = np.load(DATA / "mnist_subset.npz")
    xtr, ytr, xte, yte = d["x_train"][:n_train], d["y_train"][:n_train], d["x_test"][:n_test], d["y_test"][:n_test]
    return _norm(xtr, flatten), ytr.astype("int64"), _norm(xte, flatten), yte.astype("int64")


def load_fashion(n_train=None, n_test=None, flatten=False):
    """Real Fashion-MNIST clothing images (15k train / 5k test subset). Same format as load_mnist."""
    d = np.load(DATA / "fashion_subset.npz")
    xtr, ytr, xte, yte = d["x_train"][:n_train], d["y_train"][:n_train], d["x_test"][:n_test], d["y_test"][:n_test]
    return _norm(xtr, flatten), ytr.astype("int64"), _norm(xte, flatten), yte.astype("int64")


def load_leaves():
    """Synthetic maize-leaf dataset (see data/build_datasets.py).
    Returns dict with keys: x_train, y_train, x_val, y_val, x_test, y_test, lesion_train
    Images are float32 in [0,1], shape (N, 48, 48, 3): channels LAST (convert for PyTorch!)."""
    d = np.load(DATA / "crop_leaves.npz")
    X, y, frac, split = d["images"].astype("float32") / 255.0, d["labels"], d["lesion_fraction"], d["split"]
    out = {}
    for s in ("train", "val", "test"):
        m = split == s
        out[f"x_{s}"], out[f"y_{s}"], out[f"lesion_{s}"] = X[m], y[m].astype("int64"), frac[m]
    return out


def load_prices(market: str | None = "Nairobi"):
    """Weekly maize prices (KES / 90 kg bag).  Pass market=None to get all three markets."""
    import pandas as pd
    df = pd.read_csv(DATA / "maize_prices_weekly.csv", parse_dates=["date"])
    if market is not None:
        df = df[df.market == market].drop(columns="market").reset_index(drop=True)
    return df


# =============================================================================
# Training helpers (PyTorch)
# =============================================================================
def count_params(model, trainable_only: bool = True) -> int:
    """Number of learnable numbers in a model."""
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


class EarlyStopper:
    """Stop training when the validation loss hasn't improved for `patience` epochs
    and remember the best weights.  (You'll build your own version in Lesson 14.)"""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.counter, self.best_state, self.best_epoch = float("inf"), 0, None, 0

    def step(self, val_loss, model, epoch) -> bool:
        """Returns True when it is time to stop."""
        if val_loss < self.best - self.min_delta:
            self.best, self.counter, self.best_epoch = val_loss, 0, epoch
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        return self.counter >= self.patience


def evaluate(model, loader, loss_fn, device=None, task="classification"):
    """Average loss (and accuracy for classification) over a DataLoader, with gradients off."""
    import torch
    device = device or next(model.parameters()).device
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * len(xb)
            if task == "classification":
                correct += (out.argmax(1) == yb).sum().item()
            n += len(xb)
    return total_loss / n, (correct / n if task == "classification" else None)


def predict_proba(model, x, device=None, batch_size=512):
    """Softmax probabilities for a tensor of inputs x (no gradients)."""
    import torch
    device = device or next(model.parameters()).device
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            outs.append(torch.softmax(model(x[i:i + batch_size].to(device)), dim=1).cpu())
    return torch.cat(outs)


def fit(model, train_loader, val_loader, loss_fn, optimizer, epochs=10, scheduler=None,
        device=None, early_stopper: EarlyStopper | None = None, task="classification",
        grad_clip: float | None = None, verbose=True):
    """The full training loop, packaged.  It does exactly what you wrote by hand in Lesson 5:

        for each epoch:
            for each mini-batch:  forward → loss → zero_grad → backward → step
            evaluate on validation data, record history, (maybe) step the scheduler / stop early

    Returns a history dict with train_loss, val_loss, train_acc, val_acc, lr (one entry per epoch).
    """
    import torch
    device = device or get_device()
    model.to(device)
    hist = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc", "lr")}
    per_batch_sched = scheduler is not None and scheduler.__class__.__name__ in ("OneCycleLR", "CyclicLR")
    on_plateau = scheduler is not None and scheduler.__class__.__name__ == "ReduceLROnPlateau"
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if per_batch_sched:
                scheduler.step()
            run_loss += loss.item() * len(xb)
            n += len(xb)
            if task == "classification":
                correct += (out.argmax(1) == yb).sum().item()

        tr_loss = run_loss / n
        tr_acc = correct / n if task == "classification" else None
        va_loss, va_acc = evaluate(model, val_loader, loss_fn, device, task)
        hist["lr"].append(optimizer.param_groups[0]["lr"])
        if scheduler is not None and not per_batch_sched:
            scheduler.step(va_loss) if on_plateau else scheduler.step()
        for k, v in zip(("train_loss", "val_loss", "train_acc", "val_acc"), (tr_loss, va_loss, tr_acc, va_acc)):
            hist[k].append(v)

        if verbose:
            msg = f"epoch {epoch:3d}/{epochs} | train loss {tr_loss:.4f} | val loss {va_loss:.4f}"
            if task == "classification":
                msg += f" | train acc {tr_acc:.3f} | val acc {va_acc:.3f}"
            print(msg)
        if early_stopper is not None and early_stopper.step(va_loss, model, epoch):
            if verbose:
                print(f"⏹  early stopping at epoch {epoch} (best epoch: {early_stopper.best_epoch})")
            model.load_state_dict(early_stopper.best_state)
            break
    hist["seconds"] = time.time() - t0
    if verbose:
        print(f"done in {hist['seconds']:.1f}s")
    return hist


# =============================================================================
# Plots
# =============================================================================
def show_images(x, y=None, class_names=None, preds=None, n=16, ncols=8, cmap="gray", title=None, figscale=1.5):
    """Grid of images.  x: (N,H,W) or (N,H,W,3).  Titles show label (and prediction, red if wrong)."""
    x = np.asarray(x)
    n = min(n, len(x))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * figscale, nrows * figscale * 1.15))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(x[i], cmap=cmap if x[i].ndim == 2 else None, vmin=0, vmax=1)
        if y is not None:
            name = lambda k: class_names[int(k)] if class_names is not None else str(int(k))
            t, color = name(y[i]), "black"
            if preds is not None:
                t, color = f"{name(y[i])}→{name(preds[i])}", ("green" if preds[i] == y[i] else "red")
            ax.set_title(t, fontsize=8, color=color)
    if title:
        fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_history(hist, title="Training history", figsize=(11, 3.8)):
    """Loss (and accuracy if present) curves for train vs validation."""
    has_acc = hist.get("train_acc") and hist["train_acc"][0] is not None
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=figsize)
    axes = np.atleast_1d(axes)
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    axes[0].plot(ep, hist["train_loss"], "o-", label="train", ms=3)
    axes[0].plot(ep, hist["val_loss"], "s-", label="validation", ms=3)
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss"); axes[0].legend()
    if has_acc:
        axes[1].plot(ep, hist["train_acc"], "o-", label="train", ms=3)
        axes[1].plot(ep, hist["val_acc"], "s-", label="validation", ms=3)
        axes[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy"); axes[1].legend()
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, class_names=None, normalize=True, figsize=(6, 5), cmap="Blues"):
    """Confusion matrix heat-map.  Rows = truth, columns = prediction."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    shown = cm / cm.sum(1, keepdims=True) if normalize else cm
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(shown, cmap=cmap)
    k = len(cm)
    names = class_names if class_names is not None else list(range(k))
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion matrix" + (" (row-normalised)" if normalize else ""))
    ax.grid(False)
    for i in range(k):
        for j in range(k):
            val = f"{shown[i, j]:.2f}" if normalize else f"{cm[i, j]}"
            ax.text(j, i, val, ha="center", va="center", fontsize=7,
                    color="white" if shown[i, j] > shown.max() * .6 else "black")
    plt.colorbar(im, fraction=.046)
    plt.tight_layout()
    plt.show()
    return cm


def plot_decision_boundary(predict_fn, X, y, ax=None, h=0.02, title=None, pad=0.5):
    """Colour the plane by predicted class.  predict_fn(grid_points[N,2]) -> class labels or probabilities."""
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = np.asarray(predict_fn(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=.3, cmap="coolwarm", levels=np.linspace(Z.min(), Z.max() + 1e-9, 12) if Z.max() > Z.min() else 2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
    ax.set(title=title or "", xlabel="$x_1$", ylabel="$x_2$")
    ax.grid(False)
    return ax
