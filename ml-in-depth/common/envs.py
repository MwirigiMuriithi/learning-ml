"""Small vectorised environments for the reinforcement-learning lessons (NumPy only).

CartPole follows the classic Barto-Sutton-Anderson dynamics (same constants as the well-known Gym version):
state = (cart position x, cart velocity, pole angle theta, pole angular velocity); actions 0 = push left, 1 = push right;
reward +1 per step while the pole stays within 12 degrees and the cart within 2.4 m; episode capped at max_steps.
"""
import numpy as np

GRAV, MC, MP, LEN, FORCE, TAU = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
TOTAL = MC + MP
PML = MP * LEN
X_LIM, TH_LIM = 2.4, 12 * 2 * np.pi / 360


class CartPole:
    """n independent copies stepped together; finished copies auto-reset (their last state/reward are returned first)."""

    def __init__(self, n=1, max_steps=200, seed=0):
        self.n, self.max_steps = n, max_steps
        self.rng = np.random.default_rng(seed)
        self.s = self.rng.uniform(-0.05, 0.05, (n, 4))
        self.t = np.zeros(n, int)

    def reset_all(self):
        self.s = self.rng.uniform(-0.05, 0.05, (self.n, 4)); self.t[:] = 0
        return self.s.copy()

    def step(self, a):
        """a: int array (n,). Returns next_state (before reset), reward, done (terminated), truncated, and the state to act from next."""
        x, xd, th, thd = self.s.T
        force = np.where(np.asarray(a) == 1, FORCE, -FORCE)
        ct, st = np.cos(th), np.sin(th)
        temp = (force + PML * thd ** 2 * st) / TOTAL
        thacc = (GRAV * st - ct * temp) / (LEN * (4.0 / 3.0 - MP * ct ** 2 / TOTAL))
        xacc = temp - PML * thacc * ct / TOTAL
        x, xd, th, thd = x + TAU * xd, xd + TAU * xacc, th + TAU * thd, thd + TAU * thacc
        nxt = np.stack([x, xd, th, thd], 1)
        self.t += 1
        done = (np.abs(x) > X_LIM) | (np.abs(th) > TH_LIM)
        trunc = (self.t >= self.max_steps) & ~done
        reward = np.ones(self.n)
        self.s = nxt.copy()
        fin = done | trunc
        if fin.any():
            self.s[fin] = self.rng.uniform(-0.05, 0.05, (fin.sum(), 4)); self.t[fin] = 0
        return nxt, reward, done, trunc, self.s.copy()


def rollout(policy, seed=0, max_steps=200):
    """Run one episode with `policy(state)->action` and return the list of states."""
    env = CartPole(1, max_steps, seed)
    s = env.s.copy(); states = [s[0].copy()]
    for _ in range(max_steps):
        nxt, r, d, tr, s = env.step(np.array([policy(states[-1])]))
        states.append(nxt[0].copy())
        if d[0] or tr[0]:
            break
    return np.array(states)
