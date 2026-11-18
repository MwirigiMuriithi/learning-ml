"""
Minimal NumPy neural-network kit with MANUAL backpropagation.

Used by lessons that need "a small network as a component" (autoencoder, VAE,
GAN, diffusion, DQN, policy gradient). The lesson on MLPs (18) implements its
own backprop from scratch for teaching; everyone else may import this.

    from common.nn import MLP, Adam, SGD

    net = MLP([2, 32, 32, 1], act="tanh", out_act=None, seed=0)
    opt = Adam(net.params_and_grads(), lr=1e-3)
    for step in range(1000):
        pred = net.forward(X)                 # (n, 1)
        dpred = 2 * (pred - y) / len(X)       # dLoss/dpred  (you supply this)
        net.zero_grad()
        dX = net.backward(dpred)              # returns dLoss/dInput (needed for GANs, etc.)
        opt.step()

All arrays are float64. Batches are rows: shape (n, features).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- activations
def _identity(x):  return x, lambda g: g
def _relu(x):
    m = x > 0
    return x * m, lambda g: g * m
def _leaky(x, a=0.2):
    m = np.where(x > 0, 1.0, a)
    return x * m, lambda g: g * m
def _tanh(x):
    t = np.tanh(x)
    return t, lambda g: g * (1 - t * t)
def _sigmoid(x):
    s = 1 / (1 + np.exp(-np.clip(x, -60, 60)))
    return s, lambda g: g * s * (1 - s)
def _softplus(x):
    out = np.logaddexp(0, x)
    s = 1 / (1 + np.exp(-np.clip(x, -60, 60)))
    return out, lambda g: g * s
def _silu(x):
    s = 1 / (1 + np.exp(-np.clip(x, -60, 60)))
    return x * s, lambda g: g * (s + x * s * (1 - s))

ACTS = {None: _identity, "none": _identity, "relu": _relu, "leaky_relu": _leaky, "tanh": _tanh,
        "sigmoid": _sigmoid, "softplus": _softplus, "silu": _silu}


class MLP:
    """Fully-connected network. sizes=[in, h1, ..., out]. Hidden layers use `act`,
    the last layer uses `out_act` (None = linear)."""

    def __init__(self, sizes, act="relu", out_act=None, seed=0, init_scale=1.0):
        r = np.random.default_rng(seed)
        self.sizes, self.act_name, self.out_act_name = list(sizes), act, out_act
        self.W, self.b, self.dW, self.db = [], [], [], []
        for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            a = out_act if last else act
            gain = np.sqrt(2.0 / m) if a in ("relu", "leaky_relu", "silu") else np.sqrt(1.0 / m)
            self.W.append(r.normal(0, gain * init_scale, (m, n)))
            self.b.append(np.zeros(n))
            self.dW.append(np.zeros((m, n)))
            self.db.append(np.zeros(n))
        self._cache = []

    # -- forward / backward ---------------------------------------------
    def forward(self, x):
        self._cache = []
        h = np.asarray(x, float)
        L = len(self.W)
        for i in range(L):
            z = h @ self.W[i] + self.b[i]
            f = ACTS[self.out_act_name if i == L - 1 else self.act_name]
            out, back = f(z)
            self._cache.append((h, back))
            h = out
        return h

    def backward(self, dout):
        """Accumulates dW/db (call zero_grad() first) and returns dLoss/dInput."""
        g = np.asarray(dout, float)
        for i in reversed(range(len(self.W))):
            h_in, back = self._cache[i]
            gz = back(g)
            self.dW[i] += h_in.T @ gz
            self.db[i] += gz.sum(0)
            g = gz @ self.W[i].T
        return g

    def zero_grad(self):
        for a, b in zip(self.dW, self.db):
            a[...] = 0
            b[...] = 0

    def params_and_grads(self):
        out = []
        for i in range(len(self.W)):
            out += [(self.W[i], self.dW[i]), (self.b[i], self.db[i])]
        return out

    # -- utilities -----------------------------------------------------------
    def clone(self):
        m = MLP(self.sizes, self.act_name, self.out_act_name)
        m.copy_from(self)
        return m

    def copy_from(self, other):
        for i in range(len(self.W)):
            self.W[i][...] = other.W[i]
            self.b[i][...] = other.b[i]

    def soft_update(self, other, tau=0.01):
        for i in range(len(self.W)):
            self.W[i][...] = (1 - tau) * self.W[i] + tau * other.W[i]
            self.b[i][...] = (1 - tau) * self.b[i] + tau * other.b[i]

    def __call__(self, x):
        return self.forward(x)


class SGD:
    def __init__(self, pg, lr=0.01, momentum=0.0):
        self.pg, self.lr, self.mom = pg, lr, momentum
        self.v = [np.zeros_like(p) for p, _ in pg]

    def step(self):
        for (p, g), v in zip(self.pg, self.v):
            v[...] = self.mom * v - self.lr * g
            p += v


class Adam:
    def __init__(self, pg, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, clip=None):
        self.pg, self.lr, self.b1, self.b2, self.eps, self.clip = pg, lr, b1, b2, eps, clip
        self.m = [np.zeros_like(p) for p, _ in pg]
        self.v = [np.zeros_like(p) for p, _ in pg]
        self.t = 0

    def step(self):
        self.t += 1
        if self.clip:  # global-norm gradient clipping
            n = np.sqrt(sum((g * g).sum() for _, g in self.pg))
            s = min(1.0, self.clip / (n + 1e-12))
        else:
            s = 1.0
        for (p, g), m, v in zip(self.pg, self.m, self.v):
            g = g * s
            m[...] = self.b1 * m + (1 - self.b1) * g
            v[...] = self.b2 * v + (1 - self.b2) * g * g
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def grad_check(net: MLP, x, seed=0, eps=1e-5):
    """Max relative error between analytic and numerical gradients of sum(out**2)/2."""
    out = net.forward(x)
    net.zero_grad()
    net.backward(out)
    worst = 0.0
    r = np.random.default_rng(seed)
    for (p, g) in net.params_and_grads():
        for _ in range(5):
            idx = tuple(r.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps; lp = 0.5 * np.sum(net.forward(x) ** 2)
            p[idx] = old - eps; lm = 0.5 * np.sum(net.forward(x) ** 2)
            p[idx] = old
            num = (lp - lm) / (2 * eps)
            worst = max(worst, abs(num - g[idx]) / (abs(num) + abs(g[idx]) + 1e-12))
    return worst


if __name__ == "__main__":
    for act in ["relu", "tanh", "sigmoid", "leaky_relu", "softplus", "silu"]:
        net = MLP([3, 8, 8, 2], act=act, out_act="tanh", seed=1)
        print(f"{act:11s} grad-check max rel err = {grad_check(net, np.random.default_rng(2).normal(size=(5, 3))):.2e}")
    # tiny training sanity check: fit y = sin(x)
    X = np.linspace(-3, 3, 200)[:, None]; Y = np.sin(X)
    net = MLP([1, 32, 32, 1], act="tanh", seed=0); opt = Adam(net.params_and_grads(), lr=5e-3)
    for _ in range(1500):
        p = net.forward(X); net.zero_grad(); net.backward(2 * (p - Y) / len(X)); opt.step()
    print("final MSE on sin(x):", float(np.mean((net.forward(X) - Y) ** 2)))
