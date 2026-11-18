"""
Lesson 18 - Multi-Layer Perceptron and Backpropagation (from scratch)

Problem: "fraud vs normal" card transactions.  Each transaction is summarised by
two engineered scores (x1, x2).  Normal and fraudulent transactions wind around
each other like two interleaved spirals, so NO straight line can separate them -
the network has to bend space.

The network:  2 -> 14 -> 10 -> 2 -> 2(softmax)   (tanh hidden units).
The last hidden layer has only 2 units on purpose: we can *plot* it, and watch
the classes get untangled into two blobs that a single straight line separates.

Everything is manual:  forward pass, chain-rule backward pass, softmax+cross-entropy,
Xavier / He initialisation, mini-batch SGD with momentum, Adam, weight decay,
dropout, early stopping and a numerical gradient check.

Animated (two GIFs):
  mlp_backprop.gif          decision boundary + network diagram (edge width/colour = weight,
                            node colour = activation for one highlighted transaction) + loss/accuracy
  mlp_feature_space.gif     input space vs. the 2-unit last hidden layer (feature space)

Run:  python mlp_backprop.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

DIVERGING = LinearSegmentedColormap.from_list("bo", [BLUE, "#f3f2ee", ORANGE])


# ------------------------------------------------------------------ the network
class MLP:
    """Fully-connected network.  Row-vector convention: a batch is (n, features).

    Layer l:   z_l = a_{l-1} @ W_l + b_l        (pre-activation)
               a_l = f(z_l)                     (activation; last layer: softmax)
    """

    def __init__(self, sizes, act="tanh", seed=0, weight_decay=0.0, dropout=0.0):
        self.sizes, self.act, self.wd, self.p_drop = list(sizes), act, weight_decay, dropout
        self.rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for m, n in zip(sizes[:-1], sizes[1:]):
            # Why careful init?  We want Var(z) ~ Var(a_prev) at every layer so that signals
            # (forward) and gradients (backward) neither shrink to 0 nor blow up.
            #   tanh/sigmoid -> Xavier/Glorot: Var(W) = 1/fan_in
            #   ReLU         -> He:            Var(W) = 2/fan_in   (ReLU zeroes half the units)
            var = 2.0 / m if act == "relu" else 1.0 / m
            self.W.append(self.rng.normal(0, np.sqrt(var), (m, n)))
            self.b.append(np.zeros(n))

    # ---- activation f and its derivative f'(z), written in terms of z and a=f(z)
    def _f(self, z):
        return np.maximum(z, 0) if self.act == "relu" else np.tanh(z)

    def _df(self, z, a):
        return (z > 0).astype(float) if self.act == "relu" else 1 - a * a      # tanh' = 1 - tanh^2

    # ---- forward ----------------------------------------------------------
    def forward(self, X, train=False):
        """Returns softmax probabilities; caches what backward needs."""
        a = np.asarray(X, float)
        self.cache = [(None, a, None)]              # (z, a, dropout-mask) per layer; layer 0 = input
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            if l == L - 1:
                a = softmax(z)                       # p_k = exp(z_k) / sum_j exp(z_j)
                self.cache.append((z, a, None))
            else:
                a = self._f(z)
                mask = None
                if train and self.p_drop > 0 and l < L - 2:     # keep the tiny 2-unit layer intact
                    # inverted dropout: drop units with prob p, rescale survivors by 1/(1-p)
                    mask = (self.rng.uniform(size=a.shape) > self.p_drop) / (1 - self.p_drop)
                    a = a * mask
                self.cache.append((z, a, mask))
        return a

    def loss(self, P, y):
        """Mean cross-entropy  -(1/n) sum_i log p_i[y_i]  + (wd/2) sum ||W||^2."""
        ce = -np.mean(np.log(P[np.arange(len(y)), y] + 1e-12))
        return ce + 0.5 * self.wd * sum((W * W).sum() for W in self.W)

    # ---- backward: the chain rule, one layer at a time ----------------------
    def backward(self, y):
        """Returns lists dW, db.  Derivation (see lesson.md section 3):

        output layer:  dz_L = (p - onehot(y)) / n               softmax + CE collapse to this
        every layer:   dW_l = a_{l-1}^T @ dz_l   (+ wd * W_l)
                       db_l = sum_rows(dz_l)
                       da_{l-1} = dz_l @ W_l^T                   send the error backwards
                       dz_{l-1} = da_{l-1} * mask * f'(z_{l-1})  pass through the activation (and dropout)
        """
        L = len(self.W)
        n = len(y)
        P = self.cache[-1][1]
        dz = P.copy()
        dz[np.arange(n), y] -= 1.0
        dz /= n
        dW, db = [None] * L, [None] * L
        self.grad_norms = [0.0] * L
        for l in range(L - 1, -1, -1):
            a_prev = self.cache[l][1]
            dW[l] = a_prev.T @ dz + self.wd * self.W[l]
            db[l] = dz.sum(0)
            self.grad_norms[l] = float(np.linalg.norm(dW[l]))
            if l > 0:
                da = dz @ self.W[l].T
                z_prev, _, mask = self.cache[l]
                if mask is not None:
                    da = da * mask                      # dropped units pass no gradient
                dz = da * self._df(z_prev, self._f(z_prev))
        return dW, db

    def predict(self, X):
        return self.forward(X).argmax(1)

    def hidden(self, X, layer=-2):
        """Activations of a hidden layer (default: last hidden layer = the 2-unit feature space)."""
        self.forward(X)
        return self.cache[layer][1]

    # ---- parameter snapshot helpers
    def get_params(self):
        return [w.copy() for w in self.W], [b.copy() for b in self.b]

    def set_params(self, params):
        self.W, self.b = [w.copy() for w in params[0]], [b.copy() for b in params[1]]


# ------------------------------------------------------------------ optimisers
class SGDMomentum:
    """v <- mu*v - lr*g ;  theta <- theta + v   (a heavy ball rolling downhill)."""

    def __init__(self, net, lr=0.1, mu=0.9):
        self.net, self.lr, self.mu = net, lr, mu
        self.vW = [np.zeros_like(w) for w in net.W]
        self.vb = [np.zeros_like(b) for b in net.b]

    def step(self, dW, db):
        for l in range(len(self.net.W)):
            self.vW[l] = self.mu * self.vW[l] - self.lr * dW[l]
            self.vb[l] = self.mu * self.vb[l] - self.lr * db[l]
            self.net.W[l] += self.vW[l]
            self.net.b[l] += self.vb[l]


class Adam:
    """m <- b1*m + (1-b1)*g ; v <- b2*v + (1-b2)*g^2 ; theta <- theta - lr * m_hat / (sqrt(v_hat)+eps)."""

    def __init__(self, net, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps, self.t = net, lr, b1, b2, eps, 0
        self.params = lambda: net.W + net.b
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]

    def step(self, dW, db):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params(), dW + db)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)          # bias correction (m, v start at 0)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)      # in-place update of W/b arrays


def train(net, opt, Xtr, ytr, epochs=200, batch=64, Xva=None, yva=None, snap_every=None, seed=0, patience=None):
    """Mini-batch training loop.  Returns history dict (+ parameter snapshots for animation).
    If `patience` is given, early stopping restores the weights with best validation loss."""
    r = np.random.default_rng(seed)
    n = len(ytr)
    hist = dict(loss=[], acc=[], vloss=[], vacc=[], snaps=[], snap_epochs=[], gnorm=[])
    best, best_params, bad = np.inf, None, 0
    for ep in range(epochs + 1):
        if snap_every and (ep % snap_every == 0 or ep == epochs):
            hist["snaps"].append(net.get_params()); hist["snap_epochs"].append(ep)
        P = net.forward(Xtr)
        hist["loss"].append(net.loss(P, ytr)); hist["acc"].append(accuracy(ytr, P.argmax(1)))
        if Xva is not None:
            Pv = net.forward(Xva)
            hist["vloss"].append(net.loss(Pv, yva)); hist["vacc"].append(accuracy(yva, Pv.argmax(1)))
            if patience:
                if hist["vloss"][-1] < best - 1e-4:
                    best, best_params, bad = hist["vloss"][-1], net.get_params(), 0
                else:
                    bad += 1
                    if bad >= patience:
                        net.set_params(best_params); hist["stopped"] = ep; break
        if ep == epochs:
            break
        perm = r.permutation(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            net.forward(Xtr[idx], train=True)          # dropout only active here
            dW, db = net.backward(ytr[idx])
            opt.step(dW, db)
        hist["gnorm"].append(list(net.grad_norms))
    return hist


# ------------------------------------------------------------------ gradient check
def gradient_check(seed=0, act="tanh", eps=1e-5):
    """Compare backprop with the centred finite difference  (J(w+e)-J(w-e)) / 2e  for every weight.
    A relative error < 1e-6 means the chain rule was implemented correctly.
    (ReLU has a kink at 0: if a pre-activation sits within eps of 0 the finite difference straddles it
    and the check can fail spuriously - e.g. seed 0 does so for ReLU. Try other seeds.)"""
    r = np.random.default_rng(seed)
    net = MLP([3, 5, 4, 3], act=act, seed=seed, weight_decay=0.01)
    X, y = r.normal(size=(7, 3)), r.integers(0, 3, 7)
    net.forward(X)
    dW, db = net.backward(y)
    worst = 0.0
    for params, grads in ((net.W, dW), (net.b, db)):
        for p, g in zip(params, grads):
            for idx in np.ndindex(*p.shape):
                old = p[idx]
                p[idx] = old + eps; jp = net.loss(net.forward(X), y)
                p[idx] = old - eps; jm = net.loss(net.forward(X), y)
                p[idx] = old
                num = (jp - jm) / (2 * eps)
                worst = max(worst, abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx])))
    return worst


def vanishing_gradient_demo(depth=12, width=16, seed=0):
    """Gradient norm of each layer in a deep sigmoid-free tanh net: Xavier vs a too-small init."""
    X, y = make_spirals(200, seed=1)
    out = {}
    for name, scale in (("Xavier init", 1.0), ("tiny init (x0.3)", 0.3)):
        net = MLP([2] + [width] * depth + [2], act="tanh", seed=seed)
        net.W = [w * (scale if i > 0 else 1) for i, w in enumerate(net.W)]
        net.forward(X); net.backward(y)
        out[name] = net.grad_norms
    return out


# ------------------------------------------------------------------ drawing helpers
def layer_positions(sizes):
    """(x, y) for every node of a layered diagram in [0,1]x[0,1]."""
    pos = []
    for li, n in enumerate(sizes):
        x = li / (len(sizes) - 1)
        ys = np.linspace(0.04, 0.96, n) if n > 1 else [0.5]
        pos.append([(x, y) for y in ys])
    return pos


def draw_network(ax, W, activations, pos, wmax):
    """Edges: width ~ |w|, blue = positive, orange = negative.  Nodes: colour = activation."""
    ax.set_axis_off()
    for l, Wl in enumerate(W):
        segs, cols, lws = [], [], []
        for i, p0 in enumerate(pos[l]):
            for j, p1 in enumerate(pos[l + 1]):
                w = Wl[i, j]
                segs.append([p0, p1]); cols.append(BLUE if w > 0 else ORANGE)
                lws.append(0.2 + 3.2 * min(abs(w) / wmax, 1))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, alpha=0.55, zorder=1))
    for l, pl in enumerate(pos):
        a = activations[l]
        for k, (x, y) in enumerate(pl):
            ax.scatter([x], [y], s=95, c=[a[k]], cmap=DIVERGING, vmin=-1, vmax=1, edgecolor=INK, linewidth=0.8, zorder=3)
    for l, name in enumerate(["input", "hidden 1", "hidden 2", "2-D feature", "P(class)"]):
        ax.text(l / (len(pos) - 1), 1.03, name, ha="center", fontsize=7.5, color=INK2)
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.04, 1.1)


# ------------------------------------------------------------------ main
def main():
    # --- 1. gradient check: prove backprop is right before using it
    for act in ("tanh", "relu"):
        print(f"  gradient check ({act}): worst relative error = {gradient_check(seed=1, act=act):.2e}")

    # --- 2. data: interleaved "fraud" (orange, class 1) vs "normal" (blue, class 0) transactions
    X, y = make_spirals(n=420, noise=0.07, turns=1.25, seed=4)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test=0.25, seed=1)
    names = ["normal", "fraud"]

    # --- 3. train the main network, snapshotting the weights for the animation
    net = MLP([2, 14, 10, 2, 2], act="tanh", seed=2)
    opt = Adam(net, lr=0.004)
    epochs = 300
    h = train(net, opt, Xtr, ytr, epochs=epochs, batch=48, Xva=Xva, yva=yva, snap_every=2, seed=3)
    print(f"  main net: train acc {h['acc'][-1]:.3f}  val acc {h['vacc'][-1]:.3f}  final loss {h['loss'][-1]:.3f}")

    # --- 4. cheap experiments printed for lesson.md (regularisation + optimisers + vanishing gradients)
    regularisation_experiment()
    for k, v in vanishing_gradient_demo().items():
        print(f"  grad norm per layer, {k}: first layer {v[0]:.1e}  ...  last layer {v[-1]:.1e}")

    # --- 5. GIF 1: boundary + network + curves
    pos = layer_positions(net.sizes)
    wmax = max(np.abs(w).max() for w in h["snaps"][-1][0]) * 0.8
    probe_i = int(np.where(ytr == 1)[0][5])           # the highlighted transaction
    probe = Xtr[probe_i:probe_i + 1]
    test = MLP(net.sizes, act="tanh")
    snaps = h["snap_epochs"]
    sel = frame_steps(len(snaps) - 1, 40)
    gx, gy, G = grid_for(Xtr, res=90, pad=0.25)

    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title="An MLP bends space: backprop nudges every weight to cut the loss")
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.17, top=0.83, wspace=0.2)
    with Animator(here(__file__, "mlp_backprop.gif"), fps=8, hold=2.5) as anim:
        for k in sel:
            ep = snaps[k]
            test.set_params(h["snaps"][k])
            Pg = test.forward(G)[:, 1].reshape(gx.shape)
            a1.clear()
            a1.contourf(gx, gy, Pg, levels=np.linspace(0, 1, 11), cmap=DIVERGING, alpha=0.4)
            a1.contour(gx, gy, Pg, levels=[0.5], colors=[INK], linewidths=1.5)
            scatter_classes(a1, Xtr, ytr, s=16)
            a1.scatter(probe[:, 0], probe[:, 1], marker="*", s=260, color=YELLOW, edgecolor=INK, zorder=6)
            a1.set(title="Decision boundary (star = highlighted sample)", xlabel="score x1", ylabel="score x2")
            a1.set_xlim(gx.min(), gx.max()); a1.set_ylim(gy.min(), gy.max()); a1.grid(False)

            test.forward(probe)
            acts = [test.cache[0][1][0] / max(1, np.abs(probe).max())] + [test.cache[i][1][0] for i in (1, 2, 3)]
            pr = test.cache[4][1][0]
            acts.append(pr * 2 - 1)                     # map probability 0..1 onto the colour range -1..1
            a2.clear()
            draw_network(a2, h["snaps"][k][0], acts, pos, wmax)
            a2.set_title(f"Network: P(fraud) for the star = {pr[1]:.2f}", pad=14)

            a3.clear()
            e = np.arange(ep + 1)
            a3.plot(e, h["loss"][: ep + 1], color=BLUE); a3.plot(e, h["vloss"][: ep + 1], color=ORANGE, linewidth=1.3)
            a3.set(title="Cross-entropy loss and accuracy", xlabel="epoch", ylabel="loss")
            a3.set_xlim(0, epochs); a3.set_ylim(0, 0.8)
            a3.text(epochs * 0.55, 0.72, "train loss", color=BLUE, fontsize=8.5)
            a3.text(epochs * 0.55, 0.65, "validation loss", color=ORANGE, fontsize=8.5)
            a3.text(epochs * 0.98, 0.45, f"train acc {h['acc'][ep]:.0%}\nval acc {h['vacc'][ep]:.0%}", ha="right", fontsize=9, color=INK)
            caption(fig, f"epoch {ep:>3}:  loss={h['loss'][ep]:.3f}   train acc={h['acc'][ep]:.1%}   val acc={h['vacc'][ep]:.1%}   "
                         f"|  edge width = |weight|, blue = +, orange = -; node colour = activation")
            anim.grab(fig)

    # --- 6. GIF 2: untangling the classes in the 2-unit hidden layer
    fig, (b1, b2, b3) = new_fig(1, 3, figsize=(13, 4.8), title="The last hidden layer (2 units) untangles the classes until a straight line splits them")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.17, top=0.83, wspace=0.25)
    with Animator(here(__file__, "mlp_feature_space.gif"), fps=8, hold=2.5) as anim:
        for k in sel:
            ep = snaps[k]
            test.set_params(h["snaps"][k])
            b1.clear()
            scatter_classes(b1, Xtr, ytr, s=16)
            b1.set(title="Input space (what we see)", xlabel="score x1", ylabel="score x2"); b1.grid(False)

            Hh = test.hidden(Xtr)
            Wo, bo = test.W[-1], test.b[-1]
            b2.clear()
            # the output layer is linear in h:  logit1 - logit0 = d.h + c ;  boundary is where that = 0
            d = Wo[:, 1] - Wo[:, 0]; c = bo[1] - bo[0]
            hx = np.linspace(-1.1, 1.1, 60)
            gh = np.array(np.meshgrid(hx, hx)).reshape(2, -1).T
            pz = sigmoid(gh @ d + c).reshape(60, 60)
            b2.contourf(hx, hx, pz, levels=np.linspace(0, 1, 11), cmap=DIVERGING, alpha=0.35)
            b2.contour(hx, hx, pz, levels=[0.5], colors=[INK], linewidths=1.6)
            scatter_classes(b2, Hh, ytr, s=16)
            b2.set(title="Feature space = hidden layer 3 (2 units, tanh)", xlabel="hidden unit 1  (black line = output layer's straight boundary)", ylabel="hidden unit 2")
            b2.set_xlim(-1.1, 1.1); b2.set_ylim(-1.1, 1.1); b2.grid(False)

            b3.clear()
            # separation score: linear logit margin, plotted as histograms of (d.h + c) per class
            s = Hh @ d + c
            bins = np.linspace(-12, 12, 41)
            for cls in (0, 1):
                b3.hist(np.clip(s[ytr == cls], -12, 12), bins=bins, color=CAT[cls], alpha=0.65, edgecolor=SURFACE)
            b3.axvline(0, color=INK, linewidth=1.2)
            b3.set(title="Output-layer logit margin per class", xlabel="logit(fraud) - logit(normal)", ylabel="# transactions")
            b3.set_xlim(-12, 12); b3.set_ylim(0, 60)
            caption(fig, f"epoch {ep:>3}:  train acc={h['acc'][ep]:.1%}.  Early on both classes are smeared over the same region; "
                         f"later the hidden layer pulls them to opposite corners.")
            anim.grab(fig)

    sklearn_check(Xtr, ytr, Xva, yva)


def _set(net, params):
    net.set_params(params)
    return net


def regularisation_experiment():
    """Small noisy training set + big network = overfitting.  Compare the remedies on validation data."""
    X, y = make_spirals(n=500, noise=0.18, turns=1.25, seed=7)
    Xtr, ytr, Xva, yva = X[:70], y[:70], X[100:], y[100:]
    cfgs = [("no regularisation", dict()), ("weight decay 3e-3", dict(weight_decay=3e-3)),
            ("dropout 0.3", dict(dropout=0.3)), ("early stopping", dict(patience=25))]
    print("  overfitting experiment (70 training points, net 2-64-64-2):")
    for name, kw in cfgs:
        pat = kw.pop("patience", None)
        net = MLP([2, 64, 64, 2], act="tanh", seed=0, **kw)
        h = train(net, Adam(net, lr=0.01), Xtr, ytr, epochs=400, batch=70, Xva=Xva, yva=yva, patience=pat)
        print(f"    {name:<20s} train acc {accuracy(ytr, net.predict(Xtr)):.2f}  val acc {accuracy(yva, net.predict(Xva)):.2f}"
              f"  val loss {net.loss(net.forward(Xva), yva):.3f}" + (f"  (stopped at epoch {h['stopped']})" if "stopped" in h else ""))


def sklearn_check(Xtr, ytr, Xva, yva):
    """Optional cross-check: scikit-learn's MLPClassifier should reach a similar accuracy."""
    try:
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier((14, 10, 2), activation="tanh", max_iter=2000, random_state=0).fit(Xtr, ytr)
        print(f"  [sklearn cross-check] MLPClassifier val acc = {clf.score(Xva, yva):.3f}")
    except ImportError:
        print("  [sklearn not installed - skipping cross-check]")


if __name__ == "__main__":
    main()
