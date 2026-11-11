"""
# 16 · Convolutional Neural Networks (CNN)
**Family:** Supervised · Deep learning for images / grids

## Definition
A CNN processes images with **convolution** layers: small learnable filters (here 3×3) slide across the image,
and each filter produces a *feature map* showing where its pattern occurs. **Pooling** shrinks the maps and adds tolerance
to small shifts. Dense layers at the end classify. Weight sharing makes CNNs far more efficient than fully-connected nets on images.

## Intuition
A filter is a tiny pattern detector (an edge, a corner, a stroke direction). Sliding it over the picture asks "is this pattern here?" at
every position. Stacking layers builds from edges → textures → parts → objects. We train on 8×8 images of strokes
(horizontal, vertical, two diagonals) and **watch the filters turn into oriented-edge detectors by themselves**.

## The math
    conv:   Z[f,i,j] = Σ_{k,l} W[f,k,l] · X[i+k, j+l] + b_f          (cross-correlation; "valid" padding)
    ReLU:   A = max(Z, 0)             max-pool 2×2:  M[f,i,j] = max of the 2×2 block of A
    dense:  logits = flatten(M) W₂ + b₂ ;   softmax + cross-entropy
    Backprop through the layers by hand:  dense → un-flatten → route the gradient to the arg-max of each pooling window
    → multiply by 1[Z>0] → dW[f] = Σ over positions of  dZ[f,i,j] · (image patch at i,j)
    Output size = (N − K + 2P)/S + 1.

## What the animation shows
Top row: the four 3×3 filters evolving from random noise. Middle row: the input image and the four feature maps after ReLU (bright = filter fired).
Bottom: loss/accuracy curves. The sample cycles through the four classes so you can see which filter reacts to which stroke direction.

## Try this
- Increase filters to 8, or add a second conv layer.
- Shift the test strokes by two pixels: translation tolerance comes from conv + pooling.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from mlatlas import data, viz
from mlatlas.nn import AdamNP
from mlatlas.cli import run
import matplotlib.pyplot as plt


class CNN:
    def __init__(s, rng, F=4):
        s.F = F; s.W = rng.normal(0, .3, (F, 3, 3)); s.b = np.zeros(F); s.W2 = rng.normal(0, .2, (F * 9, 4)); s.b2 = np.zeros(4)

    def params(s): return [s.W, s.b, s.W2, s.b2]

    def forward(s, X):
        N = len(X); P = swv(X, (3, 3), axis=(1, 2))                                        # (N,6,6,3,3) all image patches
        Z = np.einsum("nijkl,fkl->nfij", P, s.W) + s.b[None, :, None, None]; A = np.maximum(Z, 0)
        Ar = A.reshape(N, s.F, 3, 2, 3, 2); M = Ar.max(axis=(3, 5)); flat = M.reshape(N, -1)
        return flat @ s.W2 + s.b2, (P, Z, A, Ar, M, flat)

    def backward(s, dl, c):
        P, Z, A, Ar, M, flat = c; dM = (dl @ s.W2.T).reshape(M.shape)
        mask = Ar == M[:, :, :, None, :, None]; dAr = mask * dM[:, :, :, None, :, None] / mask.sum(axis=(3, 5), keepdims=True)
        dZ = dAr.reshape(A.shape) * (Z > 0)
        return [np.einsum("nfij,nijkl->fkl", dZ, P), dZ.sum(axis=(0, 2, 3)), flat.T @ dl, dl.sum(0)]


def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def main(gif, quick):
    X, y = data.shapes(80, seed=0); Xtr, ytr, Xte, yte = X[:240], y[:240], X[240:], y[240:]
    rng = np.random.default_rng(1); net = CNN(rng); opt = AdamNP(net.params(), lr=.01); E = 8 if quick else 30
    H = []; snaps = []
    def snapshot():
        lg, _ = net.forward(Xte); H.append(((np.log(softmax(lg)[np.arange(len(yte)), yte] + 1e-9)).mean() * -1, (lg.argmax(1) == yte).mean()))
        snaps.append((net.W.copy(), net.b.copy()))
    snapshot()
    for e in range(E):
        p = rng.permutation(len(ytr))
        for i in range(0, len(p), 32):
            idx = p[i:i + 32]; lg, c = net.forward(Xtr[idx]); d = softmax(lg); d[np.arange(len(idx)), ytr[idx]] -= 1; opt.step(net.backward(d / len(idx), c))
        snapshot()
    print(f"  test accuracy after {E} epochs: {H[-1][1]:.3f}")
    fig = plt.figure(figsize=(10, 6.4), constrained_layout=True); gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.2])
    af = [fig.add_subplot(gs[0, j + 1]) for j in range(4)]; ai = fig.add_subplot(gs[1, 0]); am = [fig.add_subplot(gs[1, j + 1]) for j in range(4)]; ac = fig.add_subplot(gs[2, :])
    ex = [np.where(yte == c)[0][0] for c in range(4)]; probe = CNN(rng)

    def update(e):
        for a in af + am + [ai, ac]: a.clear()
        probe.W, probe.b = snaps[e]; probe.W2, probe.b2 = net.W2, net.b2; k = ex[e % 4]; img = Xte[k:k + 1]; _, c = probe.forward(img); A = c[2][0]
        for j in range(4): af[j].imshow(snaps[e][0][j], cmap="RdBu_r", vmin=-1, vmax=1); af[j].set_title(f"filter {j}", fontsize=8); af[j].axis("off")
        ai.imshow(img[0], cmap="gray"); ai.set_title(f"input: {data.SHAPE_NAMES[yte[k]]}", fontsize=8); ai.axis("off")
        for j in range(4): am[j].imshow(A[j], cmap="magma", vmin=0, vmax=max(A.max(), 1)); am[j].set_title(f"feature map {j}", fontsize=8); am[j].axis("off")
        ac.plot([h[0] for h in H[:e + 1]], color=viz.PALETTE[3], label="test loss"); ac.plot([h[1] for h in H[:e + 1]], color=viz.PALETTE[2], label="test accuracy")
        ac.set(xlim=(0, E), ylim=(0, 1.5), xlabel="epoch", title=f"Epoch {e}: accuracy {H[e][1]:.0%}"); ac.legend(loc="upper right")
    viz.animate(fig, update, range(E + 1), gif, fps=3)


if __name__ == "__main__":
    run(main, __file__)
