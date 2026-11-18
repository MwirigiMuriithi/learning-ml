"""
Lesson 03 - Linear Regression (from scratch)

Problem: predict house price ($1000s) from floor area (m^2).
Two ways to fit y = w*x + b:
  1. Normal equation  (closed form, one shot)
  2. Gradient descent (iterative - this is what we animate, because it is the
     template for almost every other model in this course)

Run:  python linear_regression.py      -> writes linear_regression.gif
"""
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from common.utils import *  # noqa: F401,F403


# ---------------------------------------------------------------- the model
class LinearRegression:
    """y_hat = X @ theta, theta = [bias, w1, w2, ...]."""

    def __init__(self):
        self.theta = None

    @staticmethod
    def _design(X):
        X = np.atleast_2d(X)
        return np.c_[np.ones(len(X)), X]

    def fit_normal_equation(self, X, y):
        """theta = (X^T X)^-1 X^T y   (solved with lstsq for numerical stability)."""
        A = self._design(X)
        self.theta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return self

    def fit_gradient_descent(self, X, y, lr=0.1, n_iters=100, theta0=None):
        """Repeat: theta <- theta - lr * dMSE/dtheta.  Returns the full path."""
        A = self._design(X)
        n = len(y)
        theta = np.zeros(A.shape[1]) if theta0 is None else np.array(theta0, float)
        path = [theta.copy()]
        for _ in range(n_iters):
            residual = A @ theta - y                 # (n,)  prediction error
            grad = (2 / n) * A.T @ residual          # d MSE / d theta
            theta = theta - lr * grad
            path.append(theta.copy())
        self.theta = theta
        return np.array(path)

    def predict(self, X):
        return self._design(X) @ self.theta


def mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


def r2(y, yhat):
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


# ------------------------------------------------------------- the animation
def main():
    x, y = make_house_prices(n=80, seed=3)
    xs, mu, sd = standardize(x[:, None])       # standardise so GD is well-conditioned
    xs = xs[:, 0]

    # closed-form answer (the "target" GD is chasing)
    ne = LinearRegression().fit_normal_equation(xs[:, None], y)
    b_star, w_star = ne.theta

    # gradient descent from a deliberately bad start
    gd = LinearRegression()
    path = gd.fit_gradient_descent(xs[:, None], y, lr=0.06, n_iters=60, theta0=[0.0, -40.0])
    # path columns: [b, w]
    loss = [mse(y, p[0] + p[1] * xs) for p in path]

    # loss surface over (w, b)
    ws = np.linspace(-60, 110, 120)
    bs = np.linspace(-20, 300, 120)
    W, B = np.meshgrid(ws, bs)
    L = np.array([[mse(y, b + w * xs) for w in ws] for b in bs])

    fig, (a1, a2, a3) = new_fig(1, 3, figsize=(13, 4.6), title="Linear regression learns by rolling downhill on the loss surface")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.2, top=0.84, wspace=0.28)
    xl = np.linspace(x.min() - 5, x.max() + 5, 50)

    with Animator(here(__file__, "linear_regression.gif"), fps=8, hold=2.5) as anim:
        for i in frame_steps(len(path) - 1, 34):
            b, w = path[i]
            yhat = b + w * xs

            # (1) data space ------------------------------------------------
            a1.clear()
            a1.vlines(x, y, yhat, color=ORANGE, alpha=0.35, linewidth=1.2, zorder=1)
            a1.scatter(x, y, s=18, color=BLUE, edgecolor=SURFACE, linewidth=0.6, zorder=3)
            xl_s = (xl - mu[0]) / sd[0]
            a1.plot(xl, b + w * xl_s, color=ORANGE, zorder=4)
            a1.set(title="Data space: the line and its errors", xlabel="floor area (m²)", ylabel="price ($1000s)")
            a1.set_xlim(xl[0], xl[-1]); a1.set_ylim(y.min() - 30, y.max() + 30)
            a1.text(0.03, 0.95, "orange sticks = residuals\n(loss = mean of their squares)", transform=a1.transAxes,
                    va="top", fontsize=8.5, color=INK2)

            # (2) parameter space -------------------------------------------
            a2.clear()
            a2.contourf(W, B, np.log(L), levels=18, cmap="Blues_r", alpha=0.85)
            a2.plot(path[: i + 1, 1], path[: i + 1, 0], "-o", color=ORANGE, markersize=3, linewidth=1.5)
            a2.scatter([w], [b], s=60, color=ORANGE, edgecolor=INK, zorder=5)
            a2.scatter([w_star], [b_star], marker="*", s=190, color=GREEN, edgecolor=INK, zorder=6)
            a2.set(title="Parameter space: loss surface (log)", xlabel="slope w", ylabel="intercept b")
            a2.text(w_star - 4, b_star + 14, "closed-form optimum", fontsize=8.5, color=INK, ha="right")

            # (3) loss curve --------------------------------------------------
            a3.clear()
            a3.plot(range(i + 1), loss[: i + 1], color=BLUE)
            a3.scatter([i], [loss[i]], color=ORANGE, zorder=5, s=36)
            a3.axhline(mse(y, ne.predict(xs[:, None])), color=MUTED, linestyle="--", linewidth=1)
            a3.set(title="Loss vs. iteration", xlabel="iteration", ylabel="MSE", yscale="log")
            a3.set_xlim(0, len(path) - 1)
            a3.text(len(path) - 2, mse(y, ne.predict(xs[:, None])) * 1.12, "best possible", ha="right", fontsize=8.5, color=INK2)

            caption(fig, f"iteration {i:>2}:  slope={w:6.1f}   intercept={b:6.1f}   MSE={loss[i]:8.1f}   R²={r2(y, yhat):.3f}")
            anim.grab(fig)

    # sanity: both solvers agree
    print(f"  normal equation: b={b_star:.2f} w={w_star:.2f} | gradient descent: b={gd.theta[0]:.2f} w={gd.theta[1]:.2f}")


if __name__ == "__main__":
    main()
