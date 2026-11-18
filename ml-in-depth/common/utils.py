"""
Shared helpers for the "ML in Depth" course.

Everything here is plain NumPy + Matplotlib + Pillow. The models themselves are
always implemented from scratch inside each lesson - this file only provides
(1) a consistent look, (2) a tiny GIF writer, (3) synthetic datasets, and
(4) a few plotting helpers.

Typical lesson script header:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from common.utils import *
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Look & feel (validated categorical order, light surface)
# --------------------------------------------------------------------------
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]  # use in this order
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e5e4e0", "#fcfcfb"

__all__ = [n for n in dir() if n.isupper()]  # extended at bottom


def style():
    """Apply the course-wide matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "font.family": "DejaVu Sans",
        }
    )


style()


def new_fig(nrows=1, ncols=1, figsize=(9, 5), title=None, **kw):
    """Create a figure. Returns (fig, axes). A suptitle is added when given."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, x=0.02, ha="left")
    return fig, axes


def caption(fig, text, y=0.01):
    """One-line explanatory caption at the bottom of a frame (what is happening *now*)."""
    for t in getattr(fig, "_caption_artists", []):
        t.remove()
    art = fig.text(0.02, y, text, fontsize=9.5, color=INK2, ha="left", va="bottom")
    fig._caption_artists = [art]
    return art


# --------------------------------------------------------------------------
# GIF writer
# --------------------------------------------------------------------------
class Animator:
    """Collect matplotlib frames and write a compact GIF.

        with Animator(here(__file__, "demo.gif"), fps=8) as anim:
            fig, ax = new_fig()
            for t in range(40):
                ax.clear(); ...draw...
                anim.grab(fig)          # snapshot the current figure

    Frames are quantised to a small palette, so files stay small (~1 MB).
    `hold` = seconds to pause on the final frame.
    """

    def __init__(self, path, fps=8, colors=96, hold=2.0, dpi=72):
        self.path = Path(path)
        self.fps, self.colors, self.hold, self.dpi = fps, colors, hold, dpi
        self.frames: list[Image.Image] = []

    def grab(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=self.dpi)
        buf.seek(0)
        im = Image.open(buf).convert("RGB")
        self.frames.append(
            im.quantize(colors=self.colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        )

    def save(self):
        if not self.frames:
            raise RuntimeError("Animator has no frames")
        dur = int(1000 / self.fps)
        durations = [dur] * len(self.frames)
        durations[-1] = int(self.hold * 1000)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(
            self.path,
            save_all=True,
            append_images=self.frames[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=1,
        )
        kb = self.path.stat().st_size / 1024
        print(f"  wrote {self.path.name}: {len(self.frames)} frames, {kb:,.0f} KB")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.save()
        plt.close("all")
        return False


def here(file, name):
    """Path next to the calling script:  here(__file__, 'out.gif')."""
    return Path(file).resolve().parent / name


def frame_steps(n_iters, n_frames, log=True):
    """Which iterations to draw. Log spacing shows the fast early progress of
    iterative algorithms (gradient descent, EM, ...) without 500 identical frames."""
    n_frames = min(n_frames, n_iters + 1)
    if log:
        s = np.unique(np.round(np.geomspace(1, n_iters + 1, n_frames)).astype(int) - 1)
    else:
        s = np.unique(np.round(np.linspace(0, n_iters, n_frames)).astype(int))
    return s


# --------------------------------------------------------------------------
# Synthetic datasets (all NumPy, all seeded)
# --------------------------------------------------------------------------
def rng_(seed):
    return np.random.default_rng(seed)


def make_house_prices(n=120, seed=0):
    """Regression: floor area (m^2) -> price ($1000s). Linear trend + noise.
    Returns x (n,), y (n,)."""
    r = rng_(seed)
    x = r.uniform(40, 220, n)
    y = 35 + 1.45 * x + r.normal(0, 22, n)
    return x, y


def make_sine(n=60, noise=0.25, seed=0, lo=0.0, hi=2 * np.pi):
    """Non-linear 1-D regression: y = sin(x) + noise (e.g. daily temperature cycle)."""
    r = rng_(seed)
    x = np.sort(r.uniform(lo, hi, n))
    y = np.sin(x) + r.normal(0, noise, n)
    return x, y


def make_blobs(n=300, centers=((-2, -2), (2, 2), (-2, 3)), std=0.8, seed=0):
    """Gaussian clusters, e.g. customers by (spend, visits). Returns X (n,2), y (n,)."""
    r = rng_(seed)
    centers = np.asarray(centers, float)
    k = len(centers)
    y = r.integers(0, k, n)
    X = centers[y] + r.normal(0, std, (n, centers.shape[1]))
    return X, y


def make_moons(n=300, noise=0.15, seed=0):
    r = rng_(seed)
    n1 = n // 2
    t1 = r.uniform(0, np.pi, n1)
    t2 = r.uniform(0, np.pi, n - n1)
    a = np.c_[np.cos(t1), np.sin(t1)]
    b = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([a, b]) + r.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n1, int), np.ones(n - n1, int)]
    p = r.permutation(n)
    return X[p], y[p]


def make_circles(n=300, noise=0.08, factor=0.45, seed=0):
    r = rng_(seed)
    n1 = n // 2
    t1 = r.uniform(0, 2 * np.pi, n1)
    t2 = r.uniform(0, 2 * np.pi, n - n1)
    outer = np.c_[np.cos(t1), np.sin(t1)]
    inner = factor * np.c_[np.cos(t2), np.sin(t2)]
    X = np.vstack([outer, inner]) + r.normal(0, noise, (n, 2))
    y = np.r_[np.zeros(n1, int), np.ones(n - n1, int)]
    p = r.permutation(n)
    return X[p], y[p]


def make_spirals(n=300, noise=0.12, turns=1.5, seed=0):
    r = rng_(seed)
    n1 = n // 2
    t = np.sqrt(r.uniform(0.02, 1, n1)) * turns * 2 * np.pi
    a = np.c_[t * np.cos(t), t * np.sin(t)] / (turns * 2 * np.pi)
    b = -a
    X = np.vstack([a, b]) + r.normal(0, noise, (2 * n1, 2))
    y = np.r_[np.zeros(n1, int), np.ones(n1, int)]
    p = r.permutation(2 * n1)
    return X[p], y[p]


def make_xor(n=300, noise=0.25, seed=0):
    r = rng_(seed)
    X = r.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X = X + r.normal(0, noise * 0.3, X.shape)
    return X, y


def make_linear_classes(n=200, margin=0.0, noise=1.0, seed=0):
    """Two roughly linearly separable Gaussian classes (e.g. spam vs ham scores)."""
    r = rng_(seed)
    y = r.integers(0, 2, n)
    mu = np.where(y[:, None] == 1, [1.6 + margin, 1.2 + margin], [-1.6 - margin, -1.2 - margin])
    X = mu + r.normal(0, noise, (n, 2))
    return X, y


def make_customer_churn(n=600, seed=0):
    """Practical tabular classification. Features: tenure (months), monthly_spend ($),
    support_calls (count). Label churn=1 more likely for short tenure, high spend,
    many support calls. Returns X (n,3), y (n,), feature_names."""
    r = rng_(seed)
    tenure = r.gamma(3.0, 8.0, n).clip(1, 72)
    spend = r.normal(60, 20, n).clip(15, 140)
    calls = r.poisson(2.0, n).astype(float)
    z = -0.9 - 0.06 * tenure + 0.025 * (spend - 60) + 0.55 * calls + 1.2
    p = 1 / (1 + np.exp(-z))
    y = (r.uniform(size=n) < p).astype(int)
    return np.c_[tenure, spend, calls], y, ["tenure_months", "monthly_spend", "support_calls"]


def make_timeseries(n=200, seed=0):
    """Trend + weekly season + noise (e.g. daily sales)."""
    r = rng_(seed)
    t = np.arange(n)
    y = 50 + 0.15 * t + 8 * np.sin(2 * np.pi * t / 7) + r.normal(0, 2.5, n)
    return t, y


# --------------------------------------------------------------------------
# Small ML utilities (shared by lessons; deliberately tiny)
# --------------------------------------------------------------------------
def train_test_split(X, y, test=0.25, seed=0):
    r = rng_(seed)
    idx = r.permutation(len(X))
    k = int(len(X) * (1 - test))
    a, b = idx[:k], idx[k:]
    return X[a], X[b], y[a], y[b]


def standardize(X, mean=None, std=None):
    mean = X.mean(0) if mean is None else mean
    std = X.std(0) + 1e-12 if std is None else std
    return (X - mean) / std, mean, std


def accuracy(y, yhat):
    return float(np.mean(np.asarray(y) == np.asarray(yhat)))


def one_hot(y, k=None):
    k = int(np.max(y)) + 1 if k is None else k
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1
    return out


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def grid_for(X, res=160, pad=0.6):
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x0, x1, res), np.linspace(y0, y1, res))
    return gx, gy, np.c_[gx.ravel(), gy.ravel()]


def decision_regions(ax, predict, X, res=160, pad=0.6, alpha=0.22, n_classes=None):
    """Paint the plane by predicted class. `predict(P)` maps (m,2) -> int labels (m,).
    Returns the grid so callers can reuse it for contours."""
    gx, gy, P = grid_for(X, res, pad)
    z = np.asarray(predict(P)).reshape(gx.shape)
    k = n_classes or int(z.max()) + 1
    from matplotlib.colors import ListedColormap

    ax.contourf(gx, gy, z, levels=np.arange(-0.5, max(k, 2) + 0.5, 1), colors=CAT[: max(k, 2)], alpha=alpha, antialiased=True)
    ax.set_xlim(gx.min(), gx.max())
    ax.set_ylim(gy.min(), gy.max())
    return gx, gy


def probability_map(ax, proba, X, res=160, pad=0.6, alpha=0.35, levels=11):
    """Heat-map of P(class 1) with the 0.5 contour drawn as the decision boundary."""
    gx, gy, P = grid_for(X, res, pad)
    z = np.asarray(proba(P)).reshape(gx.shape)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("bo", [BLUE, "#f3f2ee", ORANGE])
    ax.contourf(gx, gy, z, levels=np.linspace(0, 1, levels), cmap=cmap, alpha=alpha, antialiased=True)
    ax.contour(gx, gy, z, levels=[0.5], colors=[INK], linewidths=1.6)
    ax.set_xlim(gx.min(), gx.max())
    ax.set_ylim(gy.min(), gy.max())
    return gx, gy


def scatter_classes(ax, X, y, s=22, labels=None, edge=SURFACE):
    for c in np.unique(y):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=s, color=CAT[int(c) % len(CAT)], edgecolor=edge, linewidth=0.8,
                   label=None if labels is None else labels[int(c)], zorder=3)


def label_line(ax, x, y, text, color=INK2, dx=0.0, dy=0.0, ha="left", **kw):
    """Direct label next to a line end instead of a legend box."""
    ax.annotate(text, (x, y), xytext=(x + dx, y + dy), color=color, fontsize=8.5, ha=ha, va="center", **kw)


__all__ = [n for n in dir() if not n.startswith("_") and n not in {"annotations", "io", "Path", "matplotlib", "Image"}]
