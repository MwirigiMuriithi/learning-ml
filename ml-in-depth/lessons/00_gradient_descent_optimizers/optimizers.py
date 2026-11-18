"""
Lesson 00 - Gradient Descent & Optimisers (from scratch)

Problem story: every model in this course is "a loss function + a way to walk
downhill on it".  Here we study the walking itself.  We race six optimisers
(GD, Momentum, Nesterov, AdaGrad, RMSprop, Adam) on three 2-D landscapes
  * an ill-conditioned bowl      (a long narrow valley, like un-scaled features)
  * the Rosenbrock banana        (a curved valley, the classic stress test)
  * a saddle with two minima     (a flat saddle + a shallow and a deep basin)
and then study the knobs that matter in practice on a real little problem,
linear regression on house prices: learning rate (too small / good /
oscillating / diverging) and batch size (full batch vs mini-batch vs 1 sample).

Animated (two GIFs):
  optimizer_race.gif   - trails on three contour maps + loss-vs-step curve
  lr_and_batch.gif     - learning-rate effects and gradient noise (batch size)

Run:  python optimizers.py
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ------------------------------------------------------------ test surfaces
# Each surface = (f, grad, hessian-or-None, start point, plotting box)
def bowl(p):            # f = 0.5 (x^2 + 20 y^2): curvature 1 along x, 20 along y -> condition number 20
    return 0.5 * (p[0] ** 2 + 20 * p[1] ** 2)
def bowl_g(p):
    return np.array([p[0], 20 * p[1]])
def bowl_h(p):
    return np.diag([1.0, 20.0])

def rosen(p):           # f = (1-x)^2 + 20 (y - x^2)^2 : global minimum 0 at (1, 1) inside a curved valley
    return (1 - p[0]) ** 2 + 20 * (p[1] - p[0] ** 2) ** 2
def rosen_g(p):
    x, y = p
    return np.array([-2 * (1 - x) - 80 * x * (y - x ** 2), 40 * (y - x ** 2)])

def saddle(p):          # f = x^2 - y^2 + y^4/4 + 0.3 y : saddle near origin, minima near (0,-1.5) (deep) and (0,+1.3) (shallow)
    x, y = p
    return x ** 2 - y ** 2 + 0.25 * y ** 4 + 0.3 * y
def saddle_g(p):
    x, y = p
    return np.array([2 * x, -2 * y + y ** 3 + 0.3])

SURFACES = {
    "Ill-conditioned bowl": dict(f=bowl, g=bowl_g, x0=(-3.0, 1.0), box=(-3.6, 1.0, -1.4, 1.4)),
    "Rosenbrock valley": dict(f=rosen, g=rosen_g, x0=(-1.5, 1.5), box=(-2.0, 2.0, -0.6, 2.6)),
    "Saddle + two basins": dict(f=saddle, g=saddle_g, x0=(1.5, 2.4), box=(-2.0, 2.0, -2.2, 2.8)),
}


# --------------------------------------------------------------- optimisers
class Optimizer:
    """step(x, grad_fn) -> new x.  `grad_fn(x)` is passed (not a gradient value) so
    that Nesterov can evaluate the gradient at a look-ahead point."""
    name = "base"
    def __init__(self, lr): self.lr = lr
    def step(self, x, grad_fn): raise NotImplementedError

class GD(Optimizer):
    name = "GD"
    def step(self, x, grad_fn):
        return x - self.lr * grad_fn(x)                      # x <- x - lr * g

class Momentum(Optimizer):
    name = "Momentum"
    def __init__(self, lr, beta=0.9):
        super().__init__(lr); self.beta = beta; self.v = 0.0
    def step(self, x, grad_fn):
        self.v = self.beta * self.v + grad_fn(x)             # v <- beta v + g   (velocity = running sum of gradients)
        return x - self.lr * self.v                          # x <- x - lr v

class Nesterov(Optimizer):
    name = "Nesterov"
    def __init__(self, lr, beta=0.9):
        super().__init__(lr); self.beta = beta; self.v = 0.0
    def step(self, x, grad_fn):
        g = grad_fn(x - self.lr * self.beta * self.v)        # gradient at the look-ahead point
        self.v = self.beta * self.v + g
        return x - self.lr * self.v

class AdaGrad(Optimizer):
    name = "AdaGrad"
    def __init__(self, lr, eps=1e-8):
        super().__init__(lr); self.eps = eps; self.G = 0.0
    def step(self, x, grad_fn):
        g = grad_fn(x)
        self.G = self.G + g ** 2                             # G <- G + g^2  (only ever grows -> steps shrink)
        return x - self.lr * g / (np.sqrt(self.G) + self.eps)

class RMSprop(Optimizer):
    name = "RMSprop"
    def __init__(self, lr, rho=0.9, eps=1e-8):
        super().__init__(lr); self.rho, self.eps = rho, eps; self.s = 0.0
    def step(self, x, grad_fn):
        g = grad_fn(x)
        self.s = self.rho * self.s + (1 - self.rho) * g ** 2  # moving average of g^2 (forgets the past)
        return x - self.lr * g / (np.sqrt(self.s) + self.eps)

class Adam(Optimizer):
    name = "Adam"
    def __init__(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(lr); self.b1, self.b2, self.eps = b1, b2, eps; self.m = 0.0; self.s = 0.0; self.t = 0
    def step(self, x, grad_fn):
        g = grad_fn(x); self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g        # first moment  (momentum)
        self.s = self.b2 * self.s + (1 - self.b2) * g ** 2   # second moment (RMSprop)
        mh = self.m / (1 - self.b1 ** self.t)                # bias correction: m, s start at 0
        sh = self.s / (1 - self.b2 ** self.t)
        return x - self.lr * mh / (np.sqrt(sh) + self.eps)

# Per-surface learning rates (each tuned a little so the race is fair, not rigged)
LRS = {
    "Ill-conditioned bowl": dict(GD=0.09, Momentum=0.012, Nesterov=0.012, AdaGrad=0.8, RMSprop=0.08, Adam=0.25),
    "Rosenbrock valley":    dict(GD=0.006, Momentum=0.002, Nesterov=0.002, AdaGrad=1.0, RMSprop=0.02, Adam=0.1),
    "Saddle + two basins":  dict(GD=0.05, Momentum=0.03, Nesterov=0.03, AdaGrad=0.3, RMSprop=0.03, Adam=0.08),
}
CLASSES = [GD, Momentum, Nesterov, AdaGrad, RMSprop, Adam]


def run(opt, grad, x0, n_steps, clip=None):
    """Return the path (n_steps+1, d). Optional gradient clipping by global norm."""
    x = np.array(x0, float); path = [x.copy()]
    gf = grad if clip is None else (lambda z: clip_norm(grad(z), clip))
    for _ in range(n_steps):
        x = opt.step(x, gf); path.append(x.copy())
    return np.array(path)


def clip_norm(g, c):
    """Gradient clipping: if ||g|| > c rescale to length c (keeps direction)."""
    n = np.linalg.norm(g)
    return g if n <= c else g * (c / n)


# ----------------------------------------------------- schedules (for notes)
def step_decay(lr0, t, every=50, drop=0.5): return lr0 * drop ** (t // every)
def cosine(lr0, t, T): return 0.5 * lr0 * (1 + np.cos(np.pi * t / T))
def warmup(lr0, t, w=10): return lr0 * min(1.0, (t + 1) / w)


# -------------------------------------------------- linear-regression loss
class LinReg:
    """Mean-squared-error loss of y ~ b + w x, with full-batch or mini-batch gradients."""
    def __init__(self, x, y):
        self.A = np.c_[np.ones(len(x)), x]; self.y = y
    def loss(self, th, idx=None):
        A, y = (self.A, self.y) if idx is None else (self.A[idx], self.y[idx])
        return float(np.mean((A @ th - y) ** 2))
    def grad(self, th, idx=None):
        A, y = (self.A, self.y) if idx is None else (self.A[idx], self.y[idx])
        return 2 / len(y) * A.T @ (A @ th - y)          # (2/n) A^T (A th - y)


def sgd_path(prob, th0, lr, n_steps, batch, seed):
    """Mini-batch SGD: each step uses a random subset of `batch` rows (None = all)."""
    r = rng_(seed); th = np.array(th0, float); path = [th.copy()]; n = len(prob.y)
    for _ in range(n_steps):
        idx = None if batch is None else r.choice(n, batch, replace=False)
        th = th - lr * prob.grad(th, idx); path.append(th.copy())
    return np.array(path)


# ------------------------------------------------------------- animations
def race_gif():
    n_steps = 300
    results = {}
    for sname, S in SURFACES.items():
        for C in CLASSES:
            opt = C(LRS[sname][C.name])
            results[(sname, C.name)] = run(opt, S["g"], S["x0"], n_steps)
    colors = dict(zip([c.name for c in CLASSES], CAT[:6]))

    fig, axes = new_fig(1, 4, figsize=(13, 4.8), title="Optimiser race: same start, six ways of walking downhill")
    fig.subplots_adjust(left=0.045, right=0.915, bottom=0.2, top=0.82, wspace=0.25)
    grids = {}
    for sname, S in SURFACES.items():
        x0_, x1_, y0_, y1_ = S["box"]
        gx, gy = np.meshgrid(np.linspace(x0_, x1_, 140), np.linspace(y0_, y1_, 140))
        z = np.array([[S["f"]((a, b)) for a, b in zip(ra, rb)] for ra, rb in zip(gx, gy)])
        grids[sname] = (gx, gy, z)
    rosen_loss = {c.name: np.array([rosen(p) for p in results[("Rosenbrock valley", c.name)]]) for c in CLASSES}

    with Animator(here(__file__, "optimizer_race.gif"), fps=8, hold=2.5) as anim:
        for i in frame_steps(n_steps, 40):
            for ax, (sname, S) in zip(axes[:3], SURFACES.items()):
                ax.clear()
                gx, gy, z = grids[sname]
                lv = np.quantile(z, np.linspace(0, 1, 22) ** 1.6)
                ax.contourf(gx, gy, z, levels=np.unique(lv), cmap="Blues_r", alpha=0.75)
                ax.grid(False)
                for C in CLASSES:
                    p = results[(sname, C.name)][: i + 1]
                    ax.plot(p[:, 0], p[:, 1], color=colors[C.name], linewidth=1.3)
                    ax.scatter(*p[-1], s=30, color=colors[C.name], edgecolor=INK, linewidth=0.6, zorder=5)
                ax.scatter(*S["x0"], marker="s", s=30, color=INK, zorder=6)
                ax.set(title=sname, xlim=S["box"][:2], ylim=S["box"][2:])
                ax.tick_params(labelsize=7)
            a4 = axes[3]; a4.clear()
            for C in CLASSES:
                a4.plot(range(i + 1), rosen_loss[C.name][: i + 1], color=colors[C.name])
            a4.set(title="Rosenbrock loss vs step", xlabel="step", yscale="log", xlim=(0, n_steps), ylim=(1e-5, 200))
            # direct labels at the right edge of the loss panel, ordered by final loss (no legend box)
            fin = sorted(CLASSES, key=lambda c: rosen_loss[c.name][-1])
            for k, C in enumerate(fin):
                a4.text(1.02, 0.93 - 0.075 * k, C.name, transform=a4.transAxes, color=colors[C.name], fontsize=8.5, va="center", fontweight="bold")
            cur = "  ".join(f"{C.name}={rosen_loss[C.name][i]:.1e}" for C in CLASSES)
            caption(fig, f"step {i:>3}  Rosenbrock loss:  {cur}")
            anim.grab(fig)
    return results


def lr_batch_gif():
    x, y = make_house_prices(n=80, seed=3)
    prob = LinReg(x / 100.0, y / 100.0)                 # area in 100 m^2, price in $100k: keeps numbers O(1)
    th_star, *_ = np.linalg.lstsq(prob.A, prob.y, rcond=None)
    best = prob.loss(th_star)
    th0 = np.array([0.0, -0.5])
    H = 2 / len(y) * prob.A.T @ prob.A
    lam = np.linalg.eigvalsh(H)
    print(f"  linreg Hessian eigenvalues {lam.round(3)} -> GD is stable only for lr < 2/lambda_max = {2 / lam.max():.3f}")

    # --- learning-rate cases on the same problem
    lr_cases = [("too small", 0.01, CAT[0]), ("good", 0.15, CAT[2]), ("oscillating", 0.34, CAT[3]), ("diverging", 0.38, CAT[7])]
    n1 = 60
    paths = {nm: sgd_path(prob, th0, lr, n1, None, 0) for nm, lr, _ in lr_cases}
    # --- batch-size cases (same lr)
    n2 = 200
    bcases = [("full batch (80)", None, CAT[0]), ("mini-batch 8", 8, CAT[1]), ("single sample", 1, CAT[7])]
    bpaths = {nm: sgd_path(prob, th0, 0.1, n2, b, 1) for nm, b, _ in bcases}

    bs = np.linspace(-0.6, 1.6, 120); ws = np.linspace(-0.6, 1.6, 120)
    W, B = np.meshgrid(ws, bs)
    L = np.array([[prob.loss(np.array([b, w])) for w in ws] for b in bs])

    fig, axes = new_fig(1, 4, figsize=(13, 4.8), title="Two knobs that decide whether training works: step size and gradient noise")
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.2, top=0.82, wspace=0.3)
    a1, a2, a3, a4 = axes
    nsteps_frames = sorted(set(frame_steps(n1, 24, log=False)))
    total = len(nsteps_frames) + 16
    with Animator(here(__file__, "lr_and_batch.gif"), fps=8, hold=2.5) as anim:
        for f in range(total):
            i = nsteps_frames[min(f, len(nsteps_frames) - 1)]
            j = int(round(n2 * min(1, (f + 1) / total) ** 1.5))
            # (1) LR trails on loss surface
            a1.clear(); a1.grid(False)
            a1.contourf(W, B, np.log(L), levels=16, cmap="Blues_r", alpha=0.8)
            for nm, lr, col in lr_cases:
                p = paths[nm][: i + 1]
                p = np.clip(p, -5, 5)
                a1.plot(p[:, 1], p[:, 0], "-o", color=col, markersize=2.5, linewidth=1.1)
            a1.scatter(th_star[1], th_star[0], marker="*", s=150, color=GREEN, edgecolor=INK, zorder=6)
            a1.set(title="Learning rate: paths in (w, b)", xlabel="slope w", ylabel="intercept b", xlim=(-0.6, 1.6), ylim=(-0.6, 1.6))
            # (2) LR loss curves
            a2.clear()
            for nm, lr, col in lr_cases:
                l = [min(prob.loss(t), 1e3) for t in paths[nm][: i + 1]]
                a2.plot(range(i + 1), l, color=col)
            a2.set(title="Loss vs step", xlabel="step", yscale="log", xlim=(0, n1), ylim=(best * 0.8, 1e3))
            for k, (nm, lr, col) in enumerate(lr_cases):
                a2.text(0.03, 0.35 - 0.09 * k, f"{nm}  (lr={lr})", transform=a2.transAxes, color=col, fontsize=8.5, fontweight="bold")
            # (3) batch paths
            a3.clear(); a3.grid(False)
            a3.contourf(W, B, np.log(L), levels=16, cmap="Blues_r", alpha=0.8)
            for nm, b, col in bcases:
                p = bpaths[nm][: j + 1]
                a3.plot(p[:, 1], p[:, 0], color=col, linewidth=1.0, alpha=0.9)
            a3.scatter(th_star[1], th_star[0], marker="*", s=150, color=GREEN, edgecolor=INK, zorder=6)
            a3.set(title="Batch size: paths in (w, b)", xlabel="slope w", ylabel="intercept b", xlim=(-0.6, 1.6), ylim=(-0.6, 1.6))
            # (4) batch loss curves (always full-data loss)
            a4.clear()
            for nm, b, col in bcases:
                l = [prob.loss(t) for t in bpaths[nm][: j + 1]]
                a4.plot(range(j + 1), l, color=col, linewidth=1.2)
            a4.set(title="Full-data loss vs step (lr=0.1)", xlabel="step", yscale="log", xlim=(0, n2), ylim=(best * 0.8, 1.5))
            for k, (nm, b, col) in enumerate(bcases):
                a4.text(0.97, 0.93 - 0.09 * k, nm, transform=a4.transAxes, color=col, fontsize=8.5, fontweight="bold", ha="right")
            lastlosses = "  ".join(f"{nm.split(' (')[0]}={prob.loss(paths[nm][i]):.3g}" if np.isfinite(prob.loss(paths[nm][i])) else f"{nm}=inf" for nm, _, _ in lr_cases)
            caption(fig, f"lr step {i:>2}: {lastlosses}   |   SGD step {j:>3}: noise from small batches never fully settles")
            anim.grab(fig)
    for nm, b, _ in bcases:
        print(f"  batch {nm:>16}: final full-data loss {prob.loss(bpaths[nm][-1]):.4f} (optimum {best:.4f})")
    for nm, lr, _ in lr_cases:
        l = prob.loss(paths[nm][-1])
        print(f"  lr {lr:<5} {nm:>11}: loss after {n1} steps = {l:.4g}")


def main():
    results = race_gif()
    print("Rosenbrock final distance to (1,1):",
          {c.name: round(float(np.linalg.norm(results[('Rosenbrock valley', c.name)][-1] - 1)), 3) for c in CLASSES})
    print("Bowl final loss:", {c.name: float(f"{bowl(results[('Ill-conditioned bowl', c.name)][-1]):.2e}") for c in CLASSES})
    print("Saddle final y:", {c.name: round(float(results[('Saddle + two basins', c.name)][-1][1]), 2) for c in CLASSES})
    lr_batch_gif()

    # ---- the "conceptual" methods, shown numerically
    p0 = np.array(SURFACES["Ill-conditioned bowl"]["x0"])
    newton = p0 - np.linalg.solve(bowl_h(p0), bowl_g(p0))      # x <- x - H^-1 g : exact in ONE step for a quadratic
    print(f"  Newton step on the bowl from {p0}: -> {newton} (minimum is the origin)")
    cl = clip_norm(np.array([300.0, -400.0]), 5.0)
    print(f"  clipping g=(300,-400) to norm 5 -> {cl}")
    print("  LR schedules at t=0,50,100:", [round(step_decay(0.1, t), 4) for t in (0, 50, 100)], [round(cosine(0.1, t, 100), 4) for t in (0, 50, 100)])
    try:                                                         # optional cross-check with scipy's L-BFGS
        from scipy.optimize import minimize
        r = minimize(rosen, SURFACES["Rosenbrock valley"]["x0"], jac=rosen_g, method="L-BFGS-B")
        print(f"  scipy L-BFGS on Rosenbrock: x={r.x.round(4)} after {r.nit} iterations (vs 300 steps for our Adam)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
