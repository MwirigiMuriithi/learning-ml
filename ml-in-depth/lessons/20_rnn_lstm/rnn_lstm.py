"""
Lesson 20 - Recurrent networks: vanilla RNN and LSTM, with back-propagation through time (from scratch)

Problem story: a factory sensor emits a trigger code (+1 or -1) at the very first tick of a production run, then streams noisy
readings for T ticks.  At the end of the run the controller must say which code started the run.  The code must therefore be
*remembered* across T noisy steps.  A vanilla RNN's memory fades as T grows (vanishing gradients); an LSTM keeps it with gates.

Implemented (NumPy only, manual BPTT, verified by finite-difference gradient checks):
  * RNN cell        h_t = tanh(x_t Wx + h_{t-1} Wh + b)
  * LSTM cell       input / forget / output gates, cell state c_t   (forget-gate bias initialised to +1)
  * Adam with gradient clipping, binary cross-entropy on the final hidden state
  * gradient flow measurement  ||dL/dh_t||  versus lag

Animated:
  memory_race.gif   RNN vs LSTM learning the delayed recall for T = 10, 30, 50 (loss/accuracy as training proceeds)
  lstm_gates.gif    a trained LSTM reading a sequence tick by tick: the gates open at tick 0, close, and the cell holds the code
  gradient_flow.gif how much learning signal from the final error reaches each earlier tick (vanishing gradients)

Run:  python rnn_lstm.py        (about 1-2 minutes)
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ------------------------------------------------------------------- data
def make_runs(B, T, r):
    """x[:, :, 0] = noise every tick;  x[:, 0, 1] = trigger code (+-1);  label = (code > 0)."""
    code = r.choice([-1.0, 1.0], B)
    X = np.zeros((B, T, 2))
    X[:, :, 0] = r.normal(0, 1.0, (B, T))
    X[:, 0, 1] = code
    return X, (code > 0).astype(float)


# ------------------------------------------------------------------- models
class RNN:
    kind = "RNN"

    def __init__(self, nx, nh, seed=0):
        r = rng_(seed)
        self.nh = nh
        self.p = dict(Wx=r.normal(0, 1 / np.sqrt(nx), (nx, nh)), Wh=r.normal(0, 1 / np.sqrt(nh), (nh, nh)), b=np.zeros(nh),
                      Wo=r.normal(0, 0.5, (nh, 1)), bo=np.zeros(1))

    def forward(self, X):
        B, T, _ = X.shape
        H = np.zeros((B, T + 1, self.nh))
        for t in range(T):
            H[:, t + 1] = np.tanh(X[:, t] @ self.p["Wx"] + H[:, t] @ self.p["Wh"] + self.p["b"])
        logit = (H[:, T] @ self.p["Wo"] + self.p["bo"])[:, 0]
        self.cache = (X, H)
        return logit

    def backward(self, dlogit):
        X, H = self.cache
        B, T, _ = X.shape
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        g["Wo"] = H[:, T].T @ dlogit[:, None]; g["bo"] = np.array([dlogit.sum()])
        dh = dlogit[:, None] @ self.p["Wo"].T
        flow = np.zeros(T + 1); flow[T] = np.linalg.norm(dh) / np.sqrt(B)
        for t in range(T - 1, -1, -1):
            dz = dh * (1 - H[:, t + 1] ** 2)
            g["Wx"] += X[:, t].T @ dz; g["Wh"] += H[:, t].T @ dz; g["b"] += dz.sum(0)
            dh = dz @ self.p["Wh"].T
            flow[t] = np.linalg.norm(dh) / np.sqrt(B)
        self.flow = flow
        return g


class LSTM:
    kind = "LSTM"

    def __init__(self, nx, nh, seed=0):
        r = rng_(seed)
        self.nh = nh
        b = np.zeros(4 * nh); b[nh:2 * nh] = 1.0                  # forget-gate bias = 1: "remember by default"
        self.p = dict(W=r.normal(0, 1 / np.sqrt(nx + nh), (nx + nh, 4 * nh)), b=b, Wo=r.normal(0, 0.5, (nh, 1)), bo=np.zeros(1))

    def forward(self, X):
        B, T, _ = X.shape; nh = self.nh
        H = np.zeros((B, T + 1, nh)); C = np.zeros((B, T + 1, nh))
        G = np.zeros((B, T, 4 * nh)); Z = np.zeros((B, T, X.shape[2] + nh))
        for t in range(T):
            z = np.concatenate([X[:, t], H[:, t]], 1); a = z @ self.p["W"] + self.p["b"]
            i, f, o = sigmoid(a[:, :nh]), sigmoid(a[:, nh:2 * nh]), sigmoid(a[:, 2 * nh:3 * nh]); gg = np.tanh(a[:, 3 * nh:])
            C[:, t + 1] = f * C[:, t] + i * gg
            H[:, t + 1] = o * np.tanh(C[:, t + 1])
            G[:, t] = np.concatenate([i, f, o, gg], 1); Z[:, t] = z
        logit = (H[:, T] @ self.p["Wo"] + self.p["bo"])[:, 0]
        self.cache = (X, H, C, G, Z)
        return logit

    def backward(self, dlogit):
        X, H, C, G, Z = self.cache
        B, T, _ = X.shape; nh = self.nh
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        g["Wo"] = H[:, T].T @ dlogit[:, None]; g["bo"] = np.array([dlogit.sum()])
        dh = dlogit[:, None] @ self.p["Wo"].T; dc = np.zeros((B, nh))
        flow = np.zeros(T + 1); flow[T] = np.linalg.norm(dh) / np.sqrt(B)
        for t in range(T - 1, -1, -1):
            i, f, o, gg = (G[:, t, k * nh:(k + 1) * nh] for k in range(4))
            tc = np.tanh(C[:, t + 1])
            dc = dc + dh * o * (1 - tc ** 2)
            da = np.concatenate([dc * gg * i * (1 - i), dc * C[:, t] * f * (1 - f), dh * tc * o * (1 - o), dc * i * (1 - gg ** 2)], 1)
            g["W"] += Z[:, t].T @ da; g["b"] += da.sum(0)
            dz = da @ self.p["W"].T
            dh = dz[:, X.shape[2]:]; dc = dc * f
            flow[t] = np.linalg.norm(dh) / np.sqrt(B)
        self.flow = flow
        return g


def bce(logit, y):
    p = sigmoid(logit)
    return float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))), p


class Adam:
    def __init__(self, params, lr=0.01, clip=1.0):
        self.lr, self.clip, self.t = lr, clip, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}; self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        norm = np.sqrt(sum((g ** 2).sum() for g in grads.values()))
        sc = min(1.0, self.clip / (norm + 1e-12))                # global gradient-norm clipping
        for k in params:
            g = grads[k] * sc
            self.m[k] = 0.9 * self.m[k] + 0.1 * g; self.v[k] = 0.999 * self.v[k] + 0.001 * g * g
            params[k] -= self.lr * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)
        return norm


def train(model, T, steps=500, B=64, lr=0.01, seed=0, log_every=10):
    r = rng_(seed + 100)
    opt = Adam(model.p, lr)
    Xv, yv = make_runs(400, T, rng_(999))
    hist = []
    for s in range(steps + 1):
        if s % log_every == 0:
            l, p = bce(model.forward(Xv), yv)
            hist.append((s, l, float(np.mean((p > 0.5) == (yv > 0.5)))))
        if s == steps:
            break
        X, y = make_runs(B, T, r)
        logit = model.forward(X)
        _, p = bce(logit, y)
        opt.step(model.p, model.backward((p - y) / B))
    return np.array(hist)


def grad_check(cls, seed=0):
    r = rng_(seed); m = cls(2, 4, seed); X, y = make_runs(5, 6, r)
    X[:, :, 0] += 0.3
    logit = m.forward(X); _, p = bce(logit, y); g = m.backward((p - y) / len(y))
    worst = 0
    for k, v in m.p.items():
        for idx in [tuple(r.integers(0, s) for s in v.shape) for _ in range(5)]:
            old = v[idx]; v[idx] = old + 1e-5; lp, _ = bce(m.forward(X), y); v[idx] = old - 1e-5; lm, _ = bce(m.forward(X), y); v[idx] = old
            num = (lp - lm) / 2e-5
            worst = max(worst, abs(num - g[k][idx]) / (abs(num) + abs(g[k][idx]) + 1e-9))
    return worst


# ------------------------------------------------------------------- animations
def race_gif(path, Ts=(10, 30, 50), steps=600):
    runs = {}
    for T in Ts:
        for cls in (RNN, LSTM):
            h = train(cls(2, 16, seed=1), T, steps=steps, seed=1)
            runs[(T, cls.kind)] = h
            print(f"     T={T:2d} {cls.kind:4s}: final loss {h[-1, 1]:.3f}, accuracy {h[-1, 2]:.3f}; steps to 90% accuracy: "
                  f"{(h[h[:, 2] >= 0.9][0, 0] if np.any(h[:, 2] >= 0.9) else float('nan')):.0f}")
    n = len(runs[(Ts[0], 'RNN')])
    with Animator(path, fps=3, hold=3.5) as anim:
        fig, axes = new_fig(1, len(Ts), figsize=(13, 4.6), title="Learning to remember a code across T noisy ticks")
        for k in sorted(set(list(range(0, n, 3)) + [n - 1])):
            for ax, T in zip(axes, Ts):
                ax.clear()
                for kind, col in (("RNN", BLUE), ("LSTM", ORANGE)):
                    h = runs[(T, kind)]
                    ax.plot(h[: k + 1, 0], h[: k + 1, 2], color=col, lw=2, label=f"{kind} accuracy")
                ax.axhline(0.5, color=MUTED, ls="--", lw=1)
                ax.set(xlim=(0, steps), ylim=(0.4, 1.02), title=f"T = {T} ticks of delay", xlabel="training step", ylabel="accuracy (held-out)")
                if T == Ts[0]:
                    ax.legend(loc="lower right")
            accs = {T: (runs[(T, 'RNN')][k, 2], runs[(T, 'LSTM')][k, 2]) for T in Ts}
            caption(fig, f"step {int(runs[(Ts[0], 'RNN')][k, 0])}: " + "   ".join(f"T={T}: RNN {a:.2f} / LSTM {b:.2f}" for T, (a, b) in accs.items()) + "      (0.5 = guessing)")
            anim.grab(fig)
    return runs


def gates_gif(path, T=24, steps=500):
    m = LSTM(2, 16, seed=1)
    train(m, T, steps=steps, seed=1)
    r = rng_(5)
    X, y = make_runs(2, T, r)
    X[0, 0, 1], X[1, 0, 1] = 1.0, -1.0
    logit = m.forward(X)
    _, H, C, G, _ = m.cache
    nh = m.nh
    gi, gf, go = G[:, :, :nh], G[:, :, nh:2 * nh], G[:, :, 2 * nh:3 * nh]
    # which cell unit differs most between the two codes at the end?
    unit = int(np.argmax(np.abs(C[0, T] - C[1, T])))
    print(f"  trained LSTM (T={T}): logits for code +1/-1: {logit[0]:+.2f}/{logit[1]:+.2f}; cell unit {unit} at the end: {C[0, T, unit]:+.2f} vs {C[1, T, unit]:+.2f}")
    print(f"  mean input gate at tick 0 {gi[:, 0].mean():.2f}, afterwards {gi[:, 1:].mean():.2f}; mean forget gate {gf[:, 1:].mean():.2f}")
    with Animator(path, fps=3, hold=3.5) as anim:
        fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.8), title=f"Inside a trained LSTM (T = {T}): gates and memory")
        for t in range(T + 1):
            for a in (a1, a2, a3):
                a.clear()
            a1.plot(np.arange(T), X[0, :, 0], color=MUTED, lw=1, label="noise channel")
            a1.scatter([0], [X[0, 0, 1]], s=90, color=ORANGE, zorder=5, label="code at tick 0 (+1)")
            a1.scatter([0], [X[1, 0, 1]], s=90, color=BLUE, zorder=5, label="same, code -1")
            a1.axvline(t - 0.5, color=INK, lw=1.5)
            a1.set(title="Input stream", xlabel="tick", xlim=(-0.5, T), ylim=(-3.2, 3.2)); a1.legend(loc="lower right", fontsize=7.5)
            for gate, col, name in ((gi, AQUA, "input gate (write)"), (gf, YELLOW, "forget gate (keep)"), (go, VIOLET, "output gate (show)")):
                a2.plot(np.arange(1, t + 1), gate[0, :t].mean(1), color=col, lw=2, label=name)
            a2.set(xlim=(0, T), ylim=(0, 1.02), title="Average gate openness", xlabel="tick"); a2.legend(loc="center right", fontsize=7.5)
            a3.plot(np.arange(0, t + 1), C[0, : t + 1, unit], color=ORANGE, lw=2.4, label="cell, code +1")
            a3.plot(np.arange(0, t + 1), C[1, : t + 1, unit], color=BLUE, lw=2.4, label="cell, code -1")
            a3.set(xlim=(0, T), ylim=(min(C[:, :, unit].min(), -0.5) * 1.1, max(C[:, :, unit].max(), 0.5) * 1.1), title=f"Cell state of unit {unit} (the memory)", xlabel="tick"); a3.legend(loc="center right", fontsize=7.5)
            if t == 0:
                msg = "Before reading anything: empty memory. The code (orange = +1, blue = -1) arrives at tick 0, then only noise follows"
            elif t < T:
                msg = (f"Tick {t}: the forget gate (~{gf[0, t - 1].mean():.2f}) keeps most of the cell each tick, so the code is not overwritten; "
                       f"cell value is {C[0, t, unit]:+.1f} for code +1 and {C[1, t, unit]:+.1f} for code -1")
            else:
                msg = f"End: the cell still separates +1 ({C[0, T, unit]:+.2f}) from -1 ({C[1, T, unit]:+.2f}); the output layer reads the answer from it"
            caption(fig, msg)
            anim.grab(fig)


def flow_gif(path, T=50):
    r = rng_(7)
    X, y = make_runs(64, T, r)
    out = {}
    for cls in (RNN, LSTM):
        m = cls(2, 16, seed=1)
        logit = m.forward(X); _, p = bce(logit, y); m.backward((p - y) / len(y))
        out[cls.kind] = m.flow / m.flow[T]
    print("  gradient reaching tick t relative to the last tick (untrained nets, T=50):")
    for lag in (1, 5, 10, 20, 30, 50):
        print(f"     lag {lag:2d}: RNN {out['RNN'][T - lag]:.2e}  LSTM {out['LSTM'][T - lag]:.2e}")
    lags = np.arange(T + 1)
    with Animator(path, fps=3, hold=3.5) as anim:
        fig, (a1, a2) = new_fig(1, 2, figsize=(13, 4.6), title="How much of the final error's learning signal reaches tick t?")
        for k in list(range(1, T + 1, 3)) + [T]:
            a1.clear(); a2.clear()
            for kind, col in (("RNN", BLUE), ("LSTM", ORANGE)):
                a1.semilogy(lags[: k + 1], out[kind][T - lags[: k + 1]], color=col, lw=2.4, label=kind)
            a1.set(xlim=(0, T), ylim=(1e-8, 3), title="||dL/dh|| at lag ticks before the end (log scale)", xlabel="lag (ticks back)", ylabel="relative gradient"); a1.legend(loc="lower left")
            a2.bar(["RNN", "LSTM"], [out["RNN"][T - k], out["LSTM"][T - k]], color=[BLUE, ORANGE])
            a2.set(yscale="log", ylim=(1e-8, 3), title=f"Gradient at lag {k}"); a2.grid(axis="x", visible=False)
            caption(fig, f"lag {k}: RNN gradient {out['RNN'][T - k]:.1e}, LSTM {out['LSTM'][T - k]:.1e} (ratio {out['LSTM'][T - k] / out['RNN'][T - k]:.0f}x). A signal that tiny cannot teach the early weights anything")
            anim.grab(fig)


def main():
    print("Lesson 20: RNN / LSTM")
    print(f"  gradient check (worst relative error over random entries): RNN {grad_check(RNN):.1e}, LSTM {grad_check(LSTM):.1e}")
    flow_gif(here(__file__, "gradient_flow.gif"))
    gates_gif(here(__file__, "lstm_gates.gif"))
    print("  memory race:")
    race_gif(here(__file__, "memory_race.gif"))


if __name__ == "__main__":
    main()
