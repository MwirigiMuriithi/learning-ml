"""CART decision tree (classification: weighted Gini, regression: weighted SSE). Used by trees/forests/boosting lessons."""
import numpy as np


class Tree:
    def __init__(self, max_depth=3, min_samples=2, task="clf", max_features=None, rng=None, n_classes=None):
        self.max_depth, self.min_samples, self.task = max_depth, min_samples, task
        self.max_features, self.rng, self.K = max_features, rng or np.random.default_rng(0), n_classes

    def fit(self, X, y, w=None):
        X = np.asarray(X, float); n, d = X.shape
        self.X, self.d = X, d
        self.w = np.ones(n) if w is None else np.asarray(w, float)
        if self.task == "clf":
            y = np.asarray(y, int); self.K = self.K or int(y.max()) + 1; self.Y = np.eye(self.K)[y]
        self.y = np.asarray(y)
        self.segments, self.importance = [], np.zeros(d)
        if d >= 2: b = (X[:, 0].min() - .5, X[:, 0].max() + .5, X[:, 1].min() - .5, X[:, 1].max() + .5)
        else: b = (X[:, 0].min() - .1, X[:, 0].max() + .1, 0, 1)
        self.root = self._grow(np.arange(n), 0, b)
        return self

    def _value(self, idx):
        w = self.w[idx]
        if self.task == "clf":
            p = (self.Y[idx] * w[:, None]).sum(0); return p / max(p.sum(), 1e-12)
        return np.array([(w * self.y[idx]).sum() / max(w.sum(), 1e-12)])

    def _total_impurity(self, idx):
        w = self.w[idx]
        if self.task == "clf":
            p = (self.Y[idx] * w[:, None]).sum(0); tot = p.sum()
            return tot * (1 - ((p / tot) ** 2).sum()) if tot > 0 else 0.
        y = self.y[idx]; m = (w * y).sum() / max(w.sum(), 1e-12)
        return (w * (y - m) ** 2).sum()

    def _split(self, idx, feats):
        best = (None, None, np.inf); w = self.w[idx]
        for f in feats:
            x = self.X[idx, f]; o = np.argsort(x, kind="stable"); xs = x[o]
            if self.task == "clf":
                cw = np.cumsum(self.Y[idx][o] * w[o, None], 0); left = cw[:-1]; right = cw[-1] - left
                wl, wr = left.sum(1), right.sum(1)
                gl = 1 - ((left / np.maximum(wl, 1e-12)[:, None]) ** 2).sum(1)
                gr = 1 - ((right / np.maximum(wr, 1e-12)[:, None]) ** 2).sum(1)
                score = wl * gl + wr * gr
            else:
                y = self.y[idx][o]; ww = w[o]
                cw, cy, cyy = np.cumsum(ww), np.cumsum(ww * y), np.cumsum(ww * y * y)
                wl, wr = cw[:-1], cw[-1] - cw[:-1]; sl, sr = cy[:-1], cy[-1] - cy[:-1]; ql, qr = cyy[:-1], cyy[-1] - cyy[:-1]
                score = (ql - sl ** 2 / np.maximum(wl, 1e-12)) + (qr - sr ** 2 / np.maximum(wr, 1e-12))
            score = np.where(xs[1:] > xs[:-1], score, np.inf)
            if len(score) == 0: continue
            i = int(np.argmin(score))
            if score[i] < best[2]: best = (f, (xs[i] + xs[i + 1]) / 2, score[i])
        return best

    def _grow(self, idx, depth, bounds):
        node = {"leaf": True, "value": self._value(idx), "n": len(idx)}
        if depth >= self.max_depth or len(idx) < self.min_samples: return node
        parent = self._total_impurity(idx)
        if parent <= 1e-12: return node
        feats = np.arange(self.d)
        if self.max_features: feats = self.rng.choice(self.d, min(self.max_features, self.d), replace=False)
        f, t, score = self._split(idx, feats)
        if f is None or score >= parent - 1e-12: return node
        x = self.X[idx, f]; L, R = idx[x <= t], idx[x > t]
        if len(L) == 0 or len(R) == 0: return node
        self.importance[f] += parent - score; self.segments.append((f, t, bounds, depth))
        bl, br = list(bounds), list(bounds)
        if f == 0: bl[1], br[0] = t, t
        elif f == 1: bl[3], br[2] = t, t
        return {"leaf": False, "f": f, "t": t, "n": len(idx), "value": node["value"],
                "l": self._grow(L, depth + 1, tuple(bl)), "r": self._grow(R, depth + 1, tuple(br))}

    def _pred(self, X):
        X = np.asarray(X, float); out = np.zeros((len(X), self.K if self.task == "clf" else 1))
        def rec(node, ids):
            if len(ids) == 0: return
            if node["leaf"]: out[ids] = node["value"]; return
            m = X[ids, node["f"]] <= node["t"]; rec(node["l"], ids[m]); rec(node["r"], ids[~m])
        rec(self.root, np.arange(len(X))); return out

    def predict_proba(self, X): return self._pred(X)

    def predict(self, X):
        p = self._pred(X); return p.argmax(1) if self.task == "clf" else p[:, 0]


def export_text(tree, names, class_names=None):
    lines = []
    def rec(n, ind):
        pad = "  " * ind
        if n["leaf"]:
            v = n["value"]
            if tree.task == "clf":
                c = int(v.argmax()); lines.append(f"{pad}-> predict {class_names[c] if class_names else c}  (p={v.max():.2f}, n={n['n']})")
            else: lines.append(f"{pad}-> predict {v[0]:.2f}  (n={n['n']})")
            return
        lines.append(f"{pad}if {names[n['f']]} <= {n['t']:.2f}:"); rec(n["l"], ind + 1)
        lines.append(f"{pad}else:"); rec(n["r"], ind + 1)
    rec(tree.root, 0)
    return "\n".join(lines)
