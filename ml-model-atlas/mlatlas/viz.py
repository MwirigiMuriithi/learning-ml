"""Shared plotting + animation helpers."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Ellipse

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2", "#EECA3B", "#9D755D"]
CMAP = ListedColormap(PALETTE)
PROB = LinearSegmentedColormap.from_list("prob", [PALETTE[0], "#f4f4f4", PALETTE[1]])  # 0 -> blue, 1 -> orange

plt.rcParams.update({
    "figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": .25, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "font.size": 9, "figure.facecolor": "white",
})


def subplots(nr=1, nc=1, w=10, h=4.2, **kw):
    return plt.subplots(nr, nc, figsize=(w, h), constrained_layout=True, **kw)


def animate(fig, update, frames, path, fps=10, dpi=64):
    """Render update(frame) for every frame into an animated GIF."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    anim = FuncAnimation(fig, update, frames=list(frames), blit=False, cache_frame_data=False)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print("  saved", os.path.basename(path))


def pick(total, n):
    """n evenly spaced integers in [0, total-1]."""
    return np.unique(np.linspace(0, total - 1, min(n, total)).round().astype(int))


def geo(total, n):
    """n integers in [0, total-1], dense at the start (learning is fastest early)."""
    return np.unique(np.r_[0, np.geomspace(1, max(total - 1, 1), n).round().astype(int)])


def grid(X, n=90, pad=.5):
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


def show_map(ax, xx, yy, Z, prob=False, alpha=.32):
    """Paint a classifier's decision regions (Z = class ids) or probability map (prob=True)."""
    ext = (xx.min(), xx.max(), yy.min(), yy.max())
    Zr = np.asarray(Z).reshape(xx.shape)
    if prob:
        ax.imshow(Zr, origin="lower", extent=ext, cmap=PROB, vmin=0, vmax=1, alpha=alpha + .25,
                  aspect="auto", interpolation="bilinear")
    else:
        ax.imshow(Zr, origin="lower", extent=ext, cmap=CMAP, vmin=-.5, vmax=len(PALETTE) - .5,
                  alpha=alpha, aspect="auto", interpolation="nearest")
    ax.grid(False)


def scatter(ax, X, y, s=16, **kw):
    ax.scatter(X[:, 0], X[:, 1], c=np.array(PALETTE)[np.asarray(y).astype(int) % len(PALETTE)],
               s=s, edgecolor="white", linewidth=.5, zorder=3, **kw)


def line(ax, w, b, xlim, ylim, **kw):
    """Draw the line w0*x + w1*y + b = 0 inside the given limits."""
    w = np.asarray(w, float)
    if abs(w[1]) > 1e-9:
        xs = np.array(xlim, float)
        ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)
    else:
        ax.axvline(-b / (w[0] + 1e-12), **kw)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def ellipse(ax, mean, cov, color, nstd=2, **kw):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * nstd * np.sqrt(np.maximum(vals[::-1], 1e-12))
    ax.add_patch(Ellipse(mean, w, h, angle=ang, fill=False, edgecolor=color, **kw))
