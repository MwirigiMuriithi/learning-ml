"""A tiny reverse-mode autograd engine on NumPy arrays (about 150 lines) plus layers and optimizers.

It powers the RNN / LSTM / Transformer / Autoencoder / VAE / GAN lessons so those models can be written the way
they are written in papers, while backpropagation (incl. backprop-through-time) is done for us. Read it once:
every op stores a closure that pushes gradient to its inputs; backward() runs them in reverse topological order.
"""
import numpy as np


def _unb(g, shape):
    """Sum a gradient back down to the shape an input had before broadcasting."""
    while g.ndim > len(shape): g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1: g = g.sum(axis=i, keepdims=True)
    return g


def _w(x): return x if isinstance(x, T) else T(x)


class T:
    __array_ufunc__ = None  # make `ndarray + T` defer to T

    def __init__(self, data, parents=()):
        self.data = np.array(data, dtype=float); self.grad = np.zeros_like(self.data)
        self._p, self._bw = parents, None

    shape = property(lambda s: s.data.shape)

    def __add__(a, b):
        b = _w(b); o = T(a.data + b.data, (a, b))
        def bw(): a.grad += _unb(o.grad, a.data.shape); b.grad += _unb(o.grad, b.data.shape)
        o._bw = bw; return o
    __radd__ = __add__

    def __mul__(a, b):
        b = _w(b); o = T(a.data * b.data, (a, b))
        def bw(): a.grad += _unb(o.grad * b.data, a.data.shape); b.grad += _unb(o.grad * a.data, b.data.shape)
        o._bw = bw; return o
    __rmul__ = __mul__

    def __neg__(a): return a * -1.0
    def __sub__(a, b): return a + _w(b) * -1.0
    def __rsub__(a, b): return _w(b) + a * -1.0
    def __truediv__(a, b): return a * (_w(b) ** -1.0)
    def __rtruediv__(a, b): return _w(b) * (a ** -1.0)

    def __pow__(a, k):
        o = T(a.data ** k, (a,))
        def bw(): a.grad += k * a.data ** (k - 1) * o.grad
        o._bw = bw; return o

    def __matmul__(a, b):
        b = _w(b); o = T(a.data @ b.data, (a, b))
        def bw():
            a.grad += _unb(o.grad @ np.swapaxes(b.data, -1, -2), a.data.shape)
            b.grad += _unb(np.swapaxes(a.data, -1, -2) @ o.grad, b.data.shape)
        o._bw = bw; return o

    def _un(a, f, df):
        o = T(f(a.data), (a,))
        def bw(): a.grad += df(a.data, o.data) * o.grad
        o._bw = bw; return o

    def exp(a): return a._un(np.exp, lambda x, y: y)
    def log(a): return a._un(np.log, lambda x, y: 1 / x)
    def tanh(a): return a._un(np.tanh, lambda x, y: 1 - y ** 2)
    def sigmoid(a): return a._un(lambda x: .5 * (1 + np.tanh(x / 2)), lambda x, y: y * (1 - y))
    def relu(a): return a._un(lambda x: np.maximum(x, 0), lambda x, y: (x > 0).astype(float))
    def leaky_relu(a, s=.1): return a._un(lambda x: np.where(x > 0, x, s * x), lambda x, y: np.where(x > 0, 1., s))
    def softplus(a): return a._un(lambda x: np.logaddexp(0, x), lambda x, y: .5 * (1 + np.tanh(x / 2)))

    def sum(a, axis=None, keepdims=False):
        o = T(a.data.sum(axis=axis, keepdims=keepdims), (a,))
        def bw():
            g = o.grad
            if axis is not None and not keepdims: g = np.expand_dims(g, axis)
            a.grad += np.broadcast_to(g, a.data.shape)
        o._bw = bw; return o

    def mean(a, axis=None, keepdims=False):
        n = a.data.size if axis is None else np.prod([a.data.shape[i] for i in np.atleast_1d(axis)])
        return a.sum(axis, keepdims) * (1.0 / n)

    def reshape(a, *shape):
        o = T(a.data.reshape(*shape), (a,))
        def bw(): a.grad += o.grad.reshape(a.data.shape)
        o._bw = bw; return o

    def transpose(a, *axes):
        o = T(np.transpose(a.data, axes), (a,))
        def bw(): a.grad += np.transpose(o.grad, np.argsort(axes))
        o._bw = bw; return o

    def __getitem__(a, idx):
        o = T(a.data[idx], (a,))
        def bw(): np.add.at(a.grad, idx, o.grad)
        o._bw = bw; return o

    def backward(self):
        topo, seen, stack = [], set(), [(self, False)]
        while stack:
            v, done = stack.pop()
            if done: topo.append(v); continue
            if id(v) in seen: continue
            seen.add(id(v)); stack.append((v, True)); stack.extend((p, False) for p in v._p)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            if v._bw: v._bw()


def cat(ts, axis=-1):
    o = T(np.concatenate([t.data for t in ts], axis), tuple(ts))
    def bw():
        cuts = np.cumsum([t.data.shape[axis] for t in ts])[:-1]
        for t, g in zip(ts, np.split(o.grad, cuts, axis)): t.grad += g
    o._bw = bw; return o


def softmax(x, axis=-1):
    e = (x - x.data.max(axis=axis, keepdims=True)).exp()
    return e / e.sum(axis, keepdims=True)


def log_softmax(x, axis=-1):
    s = x - x.data.max(axis=axis, keepdims=True)
    return s - s.exp().sum(axis, keepdims=True).log()


def cross_entropy(logits, y):
    lp = log_softmax(logits); return -(lp[np.arange(len(y)), y]).mean()


def bce_logits(z, y):
    return (z.softplus() - z * y).mean()


def mse(p, y): return ((p - y) ** 2).mean()


def layer_norm(x, g, b, eps=1e-5):
    xc = x - x.mean(-1, keepdims=True)
    return xc * ((xc * xc).mean(-1, keepdims=True) + eps) ** -0.5 * g + b


def param(shape, rng, scale=None):
    scale = np.sqrt(2.0 / sum(shape[-2:])) if scale is None else scale
    return T(rng.normal(0, scale, shape))


class Linear:
    def __init__(s, i, o, rng): s.W, s.b = param((i, o), rng), T(np.zeros(o))
    def __call__(s, x): return x @ s.W + s.b
    def params(s): return [s.W, s.b]


class MLP:
    def __init__(s, sizes, rng, act="tanh"):
        s.layers = [Linear(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]; s.act = act
    def __call__(s, x):
        for l in s.layers[:-1]: x = getattr(l(x), s.act)()
        return s.layers[-1](x)
    def params(s): return [p for l in s.layers for p in l.params()]


class Adam:
    """Adam for autograd tensors. clip= global gradient-norm clipping (essential for RNNs)."""
    def __init__(s, params, lr=1e-2, b1=.9, b2=.999, eps=1e-8, clip=None):
        s.p, s.lr, s.b1, s.b2, s.eps, s.clip, s.t = list(params), lr, b1, b2, eps, clip, 0
        s.m = [np.zeros_like(p.data) for p in s.p]; s.v = [np.zeros_like(p.data) for p in s.p]
    def zero(s):
        for p in s.p: p.grad = np.zeros_like(p.data)
    def step(s):
        if s.clip:
            n = np.sqrt(sum((p.grad ** 2).sum() for p in s.p))
            if n > s.clip:
                for p in s.p: p.grad = p.grad * (s.clip / n)
        s.t += 1
        for p, m, v in zip(s.p, s.m, s.v):
            m *= s.b1; m += (1 - s.b1) * p.grad; v *= s.b2; v += (1 - s.b2) * p.grad ** 2
            p.data -= s.lr * (m / (1 - s.b1 ** s.t)) / (np.sqrt(v / (1 - s.b2 ** s.t)) + s.eps)


class AdamNP:
    """Adam for plain NumPy arrays (used by the hand-written backprop lessons). Updates arrays in place."""
    def __init__(s, params, lr=1e-2, b1=.9, b2=.999, eps=1e-8):
        s.p, s.lr, s.b1, s.b2, s.eps, s.t = params, lr, b1, b2, eps, 0
        s.m = [np.zeros_like(p) for p in params]; s.v = [np.zeros_like(p) for p in params]
    def step(s, grads):
        s.t += 1
        for p, g, m, v in zip(s.p, grads, s.m, s.v):
            m *= s.b1; m += (1 - s.b1) * g; v *= s.b2; v += (1 - s.b2) * g * g
            p -= s.lr * (m / (1 - s.b1 ** s.t)) / (np.sqrt(v / (1 - s.b2 ** s.t)) + s.eps)
