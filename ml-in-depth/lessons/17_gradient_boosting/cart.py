"""
cart.py - a compact CART (Classification And Regression Tree) written from scratch with NumPy.

Used by lesson 14 (decision tree), and copied next to lessons 15 (random forest) and 17 (gradient boosting).

Features
  * classification (gini / entropy) and regression (variance reduction = squared error)
  * sample weights (needed by AdaBoost-style and weighted problems)
  * exhaustive best-split search per feature in O(m log m) using cumulative sums of the sorted samples
  * max_depth, min_samples_split, min_samples_leaf, max_features (random subset per split -> random forests)
  * breadth-first growth, every split gets a creation index `order`, so a PARTIAL tree (first k splits) can be evaluated
    and drawn -> used for the step-by-step animations
  * per-node bounding boxes (lo, hi) for plotting 2-D partitions
  * cost-complexity (weakest-link) pruning, impurity-based feature importance, text export
"""
from __future__ import annotations

from collections import deque
import copy

import numpy as np


class DecisionTree:
    def __init__(self, task="classification", criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, seed=0, n_classes=None):
        self.task, self.criterion = task, criterion
        self.max_depth, self.min_samples_split, self.min_samples_leaf = max_depth, min_samples_split, min_samples_leaf
        self.max_features, self.seed, self.n_classes = max_features, seed, n_classes

    # ------------------------------------------------------------------ impurity
    def _impurity_from_counts(self, C):
        """C: (..., K) weighted class counts -> impurity (gini or entropy)."""
        tot = C.sum(-1, keepdims=True)
        p = C / np.maximum(tot, 1e-300)
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, -1)
        return -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, 1e-300)), 0.0), -1)

    def _node_stats(self, idx):
        w = self.w[idx]
        if self.task == "classification":
            counts = self.Y[idx].T @ w
            return counts / counts.sum(), float(self._impurity_from_counts(counts)), float(counts.sum())
        m = np.sum(w * self.y[idx]) / w.sum()
        return m, float(np.sum(w * (self.y[idx] - m) ** 2) / w.sum()), float(w.sum())

    def _new_node(self, idx, depth, lo, hi):
        value, imp, n = self._node_stats(idx)
        self.nodes.append(dict(value=value, impurity=imp, n=n, n_raw=len(idx), depth=depth, feature=-1, threshold=0.0,
                               left=-1, right=-1, lo=lo, hi=hi, order=10 ** 9, gain=0.0, idx=idx))
        return len(self.nodes) - 1

    # ------------------------------------------------------------------ split search
    def _best_split(self, idx, node):
        d = self.X.shape[1]
        k = d
        if self.max_features is not None:
            k = int(np.sqrt(d)) if self.max_features == "sqrt" else int(self.max_features)
            k = max(1, min(k, d))
        feats = self.rng.choice(d, k, replace=False) if k < d else np.arange(d)
        msl = self.min_samples_leaf
        best = (0.0, None, None)                       # (gain, feature, threshold)
        w_all = self.w[idx]
        W = w_all.sum()
        for f in feats:
            x = self.X[idx, f]
            order = np.argsort(x, kind="stable")
            xs, ws = x[order], w_all[order]
            m = len(xs)
            valid = xs[:-1] != xs[1:]
            pos = np.arange(1, m)                      # number of samples going left for each candidate cut
            valid &= (pos >= msl) & (m - pos >= msl)
            if not valid.any():
                continue
            if self.task == "classification":
                Cl = np.cumsum(self.Y[idx][order] * ws[:, None], 0)[:-1]
                Cr = Cl[-1:] * 0 + (self.Y[idx].T @ w_all) - Cl
                nl, nr = Cl.sum(1), Cr.sum(1)
                child = (nl * self._impurity_from_counts(Cl) + nr * self._impurity_from_counts(Cr)) / W
            else:
                y = self.y[idx][order]
                cw, cwy, cwyy = np.cumsum(ws), np.cumsum(ws * y), np.cumsum(ws * y * y)
                tw, twy, twyy = cw[-1], cwy[-1], cwyy[-1]
                sse_l = cwyy[:-1] - cwy[:-1] ** 2 / cw[:-1]
                sse_r = (twyy - cwyy[:-1]) - (twy - cwy[:-1]) ** 2 / np.maximum(tw - cw[:-1], 1e-300)
                child = (sse_l + sse_r) / W
            gain = node["impurity"] - child
            gain = np.where(valid, gain, -np.inf)
            j = int(np.argmax(gain))
            if gain[j] > best[0] + 1e-12:
                best = (float(gain[j]), int(f), float((xs[j] + xs[j + 1]) / 2))
        return best if best[1] is not None else None

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, sample_weight=None):
        self.X = np.asarray(X, float)
        n, d = self.X.shape
        self.w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        if self.task == "classification":
            y = np.asarray(y, int)
            self.K = self.n_classes or int(y.max()) + 1
            self.Y = np.eye(self.K)[y]
        self.y = np.asarray(y)
        self.rng = np.random.default_rng(self.seed)
        self.nodes = []
        pad = 0.05 * (self.X.max(0) - self.X.min(0))
        root = self._new_node(np.arange(n), 0, self.X.min(0) - pad, self.X.max(0) + pad)
        queue, order = deque([root]), 0
        while queue:                                    # breadth-first: shallow splits are created first
            nid = queue.popleft()
            nd = self.nodes[nid]
            idx = nd["idx"]
            if (self.max_depth is not None and nd["depth"] >= self.max_depth) or len(idx) < self.min_samples_split or nd["impurity"] <= 1e-12:
                continue
            s = self._best_split(idx, nd)
            if s is None:
                continue
            gain, f, thr = s
            nd.update(feature=f, threshold=thr, gain=gain, order=order)
            order += 1
            go = self.X[idx, f] <= thr
            lo_l, hi_l = nd["lo"].copy(), nd["hi"].copy(); hi_l[f] = thr
            lo_r, hi_r = nd["lo"].copy(), nd["hi"].copy(); lo_r[f] = thr
            nd["left"] = self._new_node(idx[go], nd["depth"] + 1, lo_l, hi_l)
            nd["right"] = self._new_node(idx[~go], nd["depth"] + 1, lo_r, hi_r)
            queue.extend([nd["left"], nd["right"]])
        self.n_splits = order
        for nd in self.nodes:
            nd.pop("idx", None)
        self._arr = None
        return self

    # ------------------------------------------------------------------ predict
    def _arrays(self):
        if getattr(self, "_arr", None) is None:
            g = lambda k, dt: np.array([nd[k] for nd in self.nodes], dt)
            self._arr = dict(f=g("feature", int), t=g("threshold", float), l=g("left", int), r=g("right", int), o=g("order", np.int64))
            self._vals = np.array([nd["value"] for nd in self.nodes])
        return self._arr

    def apply(self, X, max_splits=None):
        """Index of the leaf each sample falls into. `max_splits=k` uses only the first k splits created."""
        X = np.asarray(X, float)
        a = self._arrays()
        lim = 10 ** 9 if max_splits is None else max_splits
        cur = np.zeros(len(X), int)
        rows = np.arange(len(X))
        while True:
            f = a["f"][cur]
            internal = (f >= 0) & (a["o"][cur] < lim)
            if not internal.any():
                return cur
            go_left = X[rows, np.maximum(f, 0)] <= a["t"][cur]
            cur = np.where(internal, np.where(go_left, a["l"][cur], a["r"][cur]), cur)

    def predict_proba(self, X, max_splits=None):
        self._arrays()
        return self._vals[self.apply(X, max_splits)]

    def predict(self, X, max_splits=None):
        self._arrays()
        v = self._vals[self.apply(X, max_splits)]
        return np.argmax(v, 1) if self.task == "classification" else v

    # ------------------------------------------------------------------ introspection
    def _reachable(self):
        """Node ids reachable from the root (pruned-away descendants stay in the list but are unreachable)."""
        out, stack = [], [0]
        while stack:
            i = stack.pop()
            out.append(i)
            if self.nodes[i]["feature"] >= 0:
                stack += [self.nodes[i]["left"], self.nodes[i]["right"]]
        return out

    def n_leaves(self):
        return sum(self.nodes[i]["feature"] < 0 for i in self._reachable())

    def depth(self):
        return max(self.nodes[i]["depth"] for i in self._reachable())

    def feature_importances(self, d=None):
        d = d or self.X.shape[1]
        imp = np.zeros(d)
        for i in self._reachable():
            nd = self.nodes[i]
            if nd["feature"] >= 0:
                l, r = self.nodes[nd["left"]], self.nodes[nd["right"]]
                imp[nd["feature"]] += nd["n"] * nd["impurity"] - l["n"] * l["impurity"] - r["n"] * r["impurity"]
        return imp / imp.sum() if imp.sum() > 0 else imp

    def export_text(self, names=None, class_names=None, nid=0, indent=""):
        nd = self.nodes[nid]
        if nd["feature"] < 0:
            v = nd["value"]
            lab = (class_names[int(np.argmax(v))] if class_names else int(np.argmax(v))) if self.task == "classification" else f"{v:.2f}"
            return f"{indent}-> {lab}  (n={nd['n_raw']}, impurity={nd['impurity']:.3f})\n"
        nm = names[nd["feature"]] if names else f"x{nd['feature']}"
        s = f"{indent}if {nm} <= {nd['threshold']:.2f}:\n" + self.export_text(names, class_names, nd["left"], indent + "    ")
        s += f"{indent}else:  # {nm} > {nd['threshold']:.2f}\n" + self.export_text(names, class_names, nd["right"], indent + "    ")
        return s

    # ------------------------------------------------------------------ cost-complexity pruning
    def _risk(self, nd):
        return nd["n"] * ((1 - nd["value"].max()) if self.task == "classification" else nd["impurity"])

    def _subtree_stats(self, nid):
        nd = self.nodes[nid]
        if nd["feature"] < 0:
            return self._risk(nd), 1
        rl, ll = self._subtree_stats(nd["left"])
        rr, lr = self._subtree_stats(nd["right"])
        return rl + rr, ll + lr

    def pruned(self, alpha):
        """Weakest-link pruning: repeatedly collapse the internal node with the smallest
        g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)  while g(t) <= alpha.   (R = total misclassification / squared error)"""
        t = copy.deepcopy(self)
        total = self.nodes[0]["n"]
        while True:
            best, bn = np.inf, None
            for nid in t._reachable():
                nd = t.nodes[nid]
                if nd["feature"] >= 0:
                    rT, nl = t._subtree_stats(nid)
                    g = (t._risk(nd) - rT) / (nl - 1) / total
                    if g < best - 1e-15:
                        best, bn = g, nid
            if bn is None or best > alpha:
                break
            nd = t.nodes[bn]
            nd.update(feature=-1, left=-1, right=-1, order=10 ** 9)
        t._arr = None
        t.n_splits = sum(t.nodes[i]["feature"] >= 0 for i in t._reachable())
        return t

    def alpha_path(self):
        """Effective alphas at which the tree shrinks (sorted), found by pruning one weakest link at a time."""
        t, total, alphas = copy.deepcopy(self), self.nodes[0]["n"], [0.0]
        while t.n_leaves() > 1:
            best, bn = np.inf, None
            for nid in t._reachable():
                nd = t.nodes[nid]
                if nd["feature"] >= 0:
                    rT, nl = t._subtree_stats(nid)
                    g = (t._risk(nd) - rT) / (nl - 1) / total
                    if g < best - 1e-15:
                        best, bn = g, nid
            alphas.append(max(best, alphas[-1]))
            t.nodes[bn].update(feature=-1, left=-1, right=-1, order=10 ** 9)
        return np.array(alphas)


if __name__ == "__main__":
    # quick self-test on XOR-like data and a noisy sine
    r = np.random.default_rng(0)
    X = r.uniform(-1, 1, (400, 2)); y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    t = DecisionTree(max_depth=4).fit(X, y)
    print("xor accuracy", np.mean(t.predict(X) == y), "leaves", t.n_leaves())
    x = np.sort(r.uniform(0, 6, 200))[:, None]; yy = np.sin(x[:, 0]) + r.normal(0, 0.2, 200)
    rt = DecisionTree(task="regression", max_depth=5).fit(x, yy)
    print("regression mse", np.mean((rt.predict(x) - yy) ** 2))
    print("alpha path length", len(t.alpha_path()), "pruned leaves at alpha=0.01:", t.pruned(0.01).n_leaves())
