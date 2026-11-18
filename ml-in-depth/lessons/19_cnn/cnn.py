"""
Lesson 19 - Convolutional Neural Network (from scratch, vectorised NumPy)

Problem: "sensor defect inspection".  A tiny 8x8 thermal sensor tile is imaged; we must
classify the defect type from the noisy picture:
    0 horizontal scratch   1 vertical scratch   2 diagonal crack "/"
    3 diagonal crack "\"   4 hot-pixel cluster (2x2 blob)
Every defect appears at a random position (shift), with random length and heavy Gaussian noise.
A CNN handles the shifts for free because it looks for the same local pattern everywhere.

Network:  image 1x8x8 -> Conv(6 filters 3x3, stride 1) -> ReLU -> MaxPool 2x2 -> Flatten -> Dense(5) -> softmax
Everything is implemented by hand: im2col-style convolution forward AND backward, max-pool
backward (route the gradient to the arg-max), softmax cross-entropy, Adam, and a numerical
gradient check for the convolution backward pass.

Animated (cnn.gif):  the 6 learned 3x3 filters going from random noise to edge / line detectors,
the feature maps they produce for one fixed input, and the loss / accuracy curves.

Run:  python cnn.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

DIVERGING = LinearSegmentedColormap.from_list("bo", [BLUE, "#f3f2ee", ORANGE])
CLASSES = ["horiz. scratch", "vert. scratch", "diag /", "diag \\", "hot spot"]


# ------------------------------------------------------------------ data
def make_defect_images(n=600, size=8, noise=0.35, seed=0):
    """Synthetic sensor tiles. Returns X (n,1,size,size), y (n,)."""
    r = np.random.default_rng(seed)
    X = np.zeros((n, 1, size, size))
    y = r.integers(0, 5, n)
    for i in range(n):
        img = X[i, 0]
        L = int(r.integers(4, 7))                      # defect length 4..6 pixels
        o = int(r.integers(0, size - L + 1))           # random start along the line
        p = int(r.integers(1, size - 1))               # random row/column (the *shift*)
        c = y[i]
        if c == 0:   img[p, o:o + L] = 1
        elif c == 1: img[o:o + L, p] = 1
        elif c == 2:                                   # "/" : row decreases as column grows
            for t in range(L): img[o + L - 1 - t, o + t] = 1
        elif c == 3:
            for t in range(L): img[o + t, o + t] = 1
        else:
            a, b = r.integers(0, size - 1, 2); img[a:a + 2, b:b + 2] = 1
    X += r.normal(0, noise, X.shape)                   # sensor noise
    return X, y


# ------------------------------------------------------------------ layers
class Conv2D:
    """Cross-correlation layer (what deep-learning libraries call 'convolution').

    out[n,f,i,j] = b[f] + sum_{c,u,v} W[f,c,u,v] * x[n,c, s*i+u, s*j+v]
    Vectorised by 'im2col': every k x k patch becomes one row of a matrix, so the whole layer
    is one matrix product  cols @ W_mat.T  (same weights reused at every position = weight sharing).
    """

    def __init__(self, c_in, c_out, k=3, stride=1, pad=0, seed=0):
        r = np.random.default_rng(seed)
        self.k, self.s, self.p = k, stride, pad
        self.W = r.normal(0, np.sqrt(2.0 / (c_in * k * k)), (c_out, c_in, k, k))   # He init (ReLU follows)
        self.b = np.zeros(c_out)

    def forward(self, x):
        self.x_shape = x.shape
        xp = np.pad(x, ((0, 0), (0, 0), (self.p,) * 2, (self.p,) * 2))
        win = sliding_window_view(xp, (self.k, self.k), axis=(2, 3))[:, :, ::self.s, ::self.s]   # (N,C,oh,ow,k,k)
        N, C, oh, ow = win.shape[:4]
        self.cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N, oh, ow, C * self.k * self.k)      # (N,oh,ow,C*k*k)
        out = self.cols @ self.W.reshape(len(self.W), -1).T + self.b                             # (N,oh,ow,F)
        return out.transpose(0, 3, 1, 2)

    def backward(self, dout):
        """dout: (N,F,oh,ow). Returns dx; stores dW, db."""
        N, F, oh, ow = dout.shape
        d = dout.transpose(0, 2, 3, 1)                                         # (N,oh,ow,F)
        self.dW = np.tensordot(d, self.cols, axes=([0, 1, 2], [0, 1, 2])).reshape(self.W.shape)   # sum over positions: shared weights
        self.db = d.sum((0, 1, 2))
        C = self.W.shape[1]
        dcols = (d @ self.W.reshape(F, -1)).reshape(N, oh, ow, C, self.k, self.k)
        _, _, H, Wd = self.x_shape
        dxp = np.zeros((N, C, H + 2 * self.p, Wd + 2 * self.p))
        for u in range(self.k):                                                # scatter-add each patch pixel back
            for v in range(self.k):
                dxp[:, :, u:u + self.s * oh:self.s, v:v + self.s * ow:self.s] += dcols[:, :, :, :, u, v].transpose(0, 3, 1, 2)
        return dxp[:, :, self.p:self.p + H, self.p:self.p + Wd]


class MaxPool2:
    """2x2 max pooling, stride 2. Backward: the gradient goes only to the winning pixel of each window."""

    def forward(self, x):
        N, C, H, W = x.shape
        self.x = x
        self.out = x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))
        return self.out

    def backward(self, dout):
        N, C, H, W = self.x.shape
        xr = self.x.reshape(N, C, H // 2, 2, W // 2, 2)
        mask = (xr == self.out[:, :, :, None, :, None])
        mask = mask / mask.sum(axis=(3, 5), keepdims=True)                     # split evenly on (rare) ties
        return (mask * dout[:, :, :, None, :, None]).reshape(N, C, H, W)


class CNN:
    """conv -> ReLU -> maxpool -> flatten -> dense -> softmax."""

    def __init__(self, n_filters=6, n_classes=5, size=8, seed=0):
        self.conv = Conv2D(1, n_filters, 3, seed=seed)
        self.pool = MaxPool2()
        feat = n_filters * ((size - 2) // 2) ** 2
        r = np.random.default_rng(seed + 1)
        self.Wd = r.normal(0, np.sqrt(1.0 / feat), (feat, n_classes))
        self.bd = np.zeros(n_classes)

    def params(self):
        return [self.conv.W, self.conv.b, self.Wd, self.bd]

    def forward(self, x):
        self.z1 = self.conv.forward(x)                 # (N,F,6,6) feature maps before ReLU
        self.a1 = np.maximum(self.z1, 0)               # ReLU
        self.p1 = self.pool.forward(self.a1)           # (N,F,3,3)
        self.flat = self.p1.reshape(len(x), -1)
        return softmax(self.flat @ self.Wd + self.bd)

    def loss(self, P, y):
        return -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))

    def backward(self, P, y):
        n = len(y)
        dz = P.copy(); dz[np.arange(n), y] -= 1; dz /= n                       # softmax + CE
        dWd, dbd = self.flat.T @ dz, dz.sum(0)
        dflat = dz @ self.Wd.T
        dp1 = dflat.reshape(self.p1.shape)
        da1 = self.pool.backward(dp1)
        dz1 = da1 * (self.z1 > 0)                                              # ReLU'
        self.conv.backward(dz1)
        return [self.conv.dW, self.conv.db, dWd, dbd]

    def predict(self, x):
        return self.forward(x).argmax(1)

    def n_params(self):
        return sum(p.size for p in self.params())


class Adam:
    def __init__(self, params, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in params]; self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            p -= self.lr * (self.m[i] / (1 - self.b1 ** self.t)) / (np.sqrt(self.v[i] / (1 - self.b2 ** self.t)) + self.eps)


# ------------------------------------------------------------------ gradient check
def conv_gradient_check(eps=1e-6):
    """Check Conv2D.backward (with stride 2 and padding 1 and 2 input channels) and the whole CNN
    against centred finite differences.  Relative error ~1e-8 => backward is right."""
    r = np.random.default_rng(0)
    conv = Conv2D(2, 3, k=3, stride=2, pad=1, seed=1)
    x = r.normal(size=(2, 2, 6, 6)); target = r.normal(size=conv.forward(x).shape)
    f = lambda: 0.5 * np.sum((conv.forward(x) - target) ** 2)                 # scalar loss = 0.5*||out - t||^2
    dx = conv.backward(conv.forward(x) - target)
    worst = 0.0
    for arr, g in ((conv.W, conv.dW), (conv.b, conv.db), (x, dx)):
        for idx in np.ndindex(*arr.shape):
            old = arr[idx]; arr[idx] = old + eps; jp = f(); arr[idx] = old - eps; jm = f(); arr[idx] = old
            num = (jp - jm) / (2 * eps); worst = max(worst, abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx])))
    worst_conv = worst
    net = CNN(seed=3); X, y = make_defect_images(4, seed=5)
    P = net.forward(X); grads = net.backward(P, y); worst = 0.0
    for p, g in zip(net.params(), grads):
        for idx in np.ndindex(*p.shape):
            old = p[idx]; p[idx] = old + eps; jp = net.loss(net.forward(X), y)
            p[idx] = old - eps; jm = net.loss(net.forward(X), y); p[idx] = old
            num = (jp - jm) / (2 * eps); worst = max(worst, abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx])))
    return worst_conv, worst


# ------------------------------------------------------------------ training
def train(net, Xtr, ytr, Xva, yva, iters=600, batch=32, lr=0.01, seed=0, snaps_at=()):
    r = np.random.default_rng(seed)
    opt = Adam(net.params(), lr=lr)
    h = dict(it=[], loss=[], vloss=[], acc=[], vacc=[], snap={})
    order, pos = r.permutation(len(ytr)), 0
    for it in range(iters + 1):
        if it % 4 == 0 or it == iters:
            P, Pv = net.forward(Xtr), net.forward(Xva)
            h["it"].append(it); h["loss"].append(net.loss(P, ytr)); h["vloss"].append(net.loss(Pv, yva))
            h["acc"].append(accuracy(ytr, P.argmax(1))); h["vacc"].append(accuracy(yva, Pv.argmax(1)))
        if it in snaps_at:
            h["snap"][it] = [p.copy() for p in net.params()]
        if it == iters:
            break
        if pos + batch > len(order):
            order, pos = r.permutation(len(ytr)), 0
        idx = order[pos:pos + batch]; pos += batch
        P = net.forward(Xtr[idx])
        opt.step(net.backward(P, ytr[idx]))
    return h


def shift_test(net, X, y, shift=2):
    """Translation robustness: roll every test image by `shift` pixels. Accuracy should stay high."""
    return accuracy(y, net.predict(np.roll(X, shift, axis=3)))


# ------------------------------------------------------------------ main
def main():
    wc, wn = conv_gradient_check()
    print(f"  gradient check: Conv2D (stride 2, pad 1, 2 channels) worst rel. error = {wc:.1e};  whole CNN = {wn:.1e}")

    X, y = make_defect_images(900, noise=0.35, seed=0)
    Xtr, ytr, Xva, yva = X[:600], y[:600], X[600:], y[600:]
    net = CNN(seed=1)
    iters = 600
    sel = frame_steps(iters, 38)
    h = train(net, Xtr, ytr, Xva, yva, iters=iters, snaps_at=set(sel.tolist()))
    print(f"  CNN ({net.n_params()} params): train acc {h['acc'][-1]:.3f}  val acc {h['vacc'][-1]:.3f}  val loss {h['vloss'][-1]:.3f}")
    print(f"  accuracy after shifting all val images by 2 px (np.roll): {shift_test(net, Xva, yva):.3f}")
    dense_params = 64 * 54 + 54 + 54 * 5 + 5
    print(f"  parameters: CNN {net.n_params()}  vs  dense MLP 64-54-5: {dense_params}")
    sklearn_check(Xtr, ytr, Xva, yva)

    # fixed probe image for the feature-map panels: a clean-ish diagonal "/" crack
    probe_i = int(np.where(yva == 2)[0][0])
    probe = Xva[probe_i:probe_i + 1]
    show = MLP_free_copy(net)

    F = net.conv.W.shape[0]
    fig = plt.figure(figsize=(13, 5.4))
    fig.suptitle("A CNN learns its own edge and line detectors from noise", fontsize=13, fontweight="bold", x=0.02, ha="left")
    gs = GridSpec(2, 12, figure=fig, left=0.03, right=0.985, bottom=0.17, top=0.80, wspace=0.18, hspace=0.45)
    ax_in = fig.add_subplot(gs[0, 0]); ax_pr = fig.add_subplot(gs[1, 0])
    ax_f = [fig.add_subplot(gs[0, 1 + i]) for i in range(F)]
    ax_m = [fig.add_subplot(gs[1, 1 + i]) for i in range(F)]
    ax_l = fig.add_subplot(gs[0, 8:]); ax_a = fig.add_subplot(gs[1, 8:])
    its = np.array(h["it"])
    with Animator(here(__file__, "cnn.gif"), fps=8, hold=2.5) as anim:
        for it in sel:
            for p, q in zip(show.params(), h["snap"][it]):
                p[...] = q
            P = show.forward(probe)[0]
            vm = np.abs(show.conv.W).max()
            ax_in.clear(); ax_in.imshow(probe[0, 0], cmap="Greys", vmin=-0.5, vmax=1.2)
            ax_in.set_title("input\n(true: diag /)", fontsize=8.5); ax_in.set_xticks([]); ax_in.grid(False); ax_in.set_yticks([])
            ax_pr.clear(); ax_pr.barh(range(5), P, color=[ORANGE if k == P.argmax() else BLUE for k in range(5)], height=0.7)
            ax_pr.set_yticks(range(5)); ax_pr.set_yticklabels(CLASSES, fontsize=6.5); ax_pr.set_xlim(0, 1); ax_pr.invert_yaxis()
            ax_pr.set_xticks([0, 1]); ax_pr.set_title("P(class)", fontsize=8.5); ax_pr.grid(False)
            for f in range(F):
                ax_f[f].clear(); ax_f[f].imshow(show.conv.W[f, 0], cmap=DIVERGING, vmin=-vm, vmax=vm)
                ax_f[f].set_title(f"filter {f + 1}", fontsize=8.5); ax_f[f].set_xticks([]); ax_f[f].set_yticks([]); ax_f[f].grid(False)
                ax_m[f].clear(); ax_m[f].imshow(show.a1[0, f], cmap="Blues", vmin=0, vmax=max(1.0, show.a1.max()))
                ax_m[f].set_title("feature map", fontsize=8, color=INK2); ax_m[f].set_xticks([]); ax_m[f].set_yticks([]); ax_m[f].grid(False)
            k = np.searchsorted(its, it, side="right")
            ax_l.clear(); ax_l.plot(its[:k], h["loss"][:k], color=BLUE); ax_l.plot(its[:k], h["vloss"][:k], color=ORANGE, linewidth=1.3)
            ax_l.set(xlim=(0, iters), ylim=(0, 1.8), title="loss", xlabel=""); ax_l.set_xticklabels([])
            ax_l.text(iters * 0.6, 1.5, "train", color=BLUE, fontsize=8.5); ax_l.text(iters * 0.6, 1.3, "validation", color=ORANGE, fontsize=8.5)
            ax_a.clear(); ax_a.plot(its[:k], h["acc"][:k], color=BLUE); ax_a.plot(its[:k], h["vacc"][:k], color=ORANGE, linewidth=1.3)
            ax_a.set(xlim=(0, iters), ylim=(0, 1.02), title="accuracy", xlabel="training iteration (mini-batch of 32)")
            caption(fig, f"iteration {it:>3}:  train acc={h['acc'][k - 1]:.1%}  val acc={h['vacc'][k - 1]:.1%}  loss={h['loss'][k - 1]:.3f}"
                         f"   |  filters: blue = negative, orange = positive weight (colour scale per frame); feature map = ReLU(filter * input)")
            anim.grab(fig)


def MLP_free_copy(net):
    """A second CNN object used only for drawing, so snapshots never disturb the trained net."""
    c = CNN(n_filters=net.conv.W.shape[0])
    return c


def sklearn_check(Xtr, ytr, Xva, yva):
    """Optional cross-check: scikit-learn's MLP on flattened pixels (no convolution) as a baseline."""
    try:
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier((54,), max_iter=600, random_state=0).fit(Xtr.reshape(len(Xtr), -1), ytr)
        print(f"  [sklearn] dense MLPClassifier on raw pixels: val acc = {clf.score(Xva.reshape(len(Xva), -1), yva):.3f}")
    except ImportError:
        print("  [sklearn not installed - skipping cross-check]")


if __name__ == "__main__":
    main()
