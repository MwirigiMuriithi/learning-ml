"""
# 31 · Association Rules & the Apriori Algorithm
**Family:** Unsupervised · Pattern mining (market-basket analysis)

## Definition
Association-rule mining finds items that frequently occur together in transactions and states them as rules "if {bread} then {butter}".
**Apriori** finds all *frequent itemsets* efficiently by growing them level by level and pruning with one key fact: **a set can only be frequent if all its subsets are frequent**.

## Intuition
Count single items first; throw away rare ones. Only combine survivors into pairs; count and prune; combine surviving pairs into triples, and so on.
Whole branches of the search space are never counted. Then turn frequent sets into rules and rank them by how much they beat chance (lift).

## The math
    support(A)        = P(A) = (# baskets containing A) / N
    confidence(A→B)   = support(A∪B) / support(A) = P(B | A)
    lift(A→B)         = confidence / support(B)     (>1: positively associated, 1: independent, <1: substitutes)
    Apriori principle: support(S) ≤ support(T) for any T ⊂ S   ⇒  prune a candidate if any subset is infrequent.
    Faster alternatives: FP-Growth, Eclat. Applications: recommendations, store layout, cross-selling, log/alert correlation, web-usage mining.

## Practical tips
- Choose min-support to the order of "enough baskets to matter"; sort rules by lift, not only confidence (popular items inflate confidence).
- Association is not causation. Beware of many spurious rules at tiny support.

## What the animation shows
600 synthetic supermarket baskets with planted patterns (bread→butter→jam, coffee→sugar, diapers→beer, chips→soda). Each stage grows the bars of candidate itemsets;
those below the dashed min-support line turn grey and are pruned. The last frames plot every discovered rule: support vs confidence, coloured by lift.

## Try this
- Lower min support to 2 % and watch the number of candidates explode.
- Add the rule "tea → lemon" to `data.baskets` and confirm Apriori finds it.
"""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from mlatlas import data, viz
from mlatlas.cli import run


def apriori(B, minsup):
    n = len(B); sup = lambda s: sum(s <= b for b in B) / n; levels = []; cand = [frozenset([i]) for i in data.ITEMS]; k = 1
    while cand:
        sc = {c: sup(c) for c in cand}; levels.append(sc); freq = [c for c, s in sc.items() if s >= minsup]
        cand = list({a | b for a in freq for b in freq if len(a | b) == k + 1 and all(frozenset(s) in set(freq) for s in itertools.combinations(a | b, k))}); k += 1
    return levels


def main(gif, quick):
    B = data.baskets(600, seed=1); minsup = .05; L = apriori(B, minsup); allsup = {c: s for lv in L for c, s in lv.items()}; rules = []
    for lv in L[1:]:
        for s, sp in lv.items():
            if sp < minsup: continue
            for r in range(1, len(s)):
                for a in itertools.combinations(s, r):
                    a = frozenset(a); b = s - a; conf = sp / allsup[a]; rules.append((a, b, sp, conf, conf / allsup[b]))
    rules.sort(key=lambda r: -r[4]); print(f"  min support {minsup:.0%}: " + ", ".join(f"{len(lv)} candidates -> {sum(s >= minsup for s in lv.values())} frequent (size {i + 1})" for i, lv in enumerate(L)))
    print("  top rules by lift:"); [print(f"    {sorted(a)} -> {sorted(b)}  support {s:.2f} conf {c:.2f} lift {l:.1f}") for a, b, s, c, l in rules[:6]]
    show = [sorted(lv.items(), key=lambda kv: -kv[1])[:14] for lv in L[:3]]; fig, ax = viz.subplots(1, 1, w=9, h=5.2); G = 5; frames = [(i, g) for i in range(3) for g in range(1, G + 1)] + [("pr", 0)] * 3

    def update(fr):
        ax.clear(); i, g = fr
        if i == "pr":
            a = np.array([[r[2], r[3], r[4]] for r in rules]); sc = ax.scatter(a[:, 0], a[:, 1], c=a[:, 2], cmap="viridis", s=60, edgecolor="k")
            for r in rules[:5]: ax.annotate(f"{'+'.join(sorted(r[0]))}→{'+'.join(sorted(r[1]))}", (r[2], r[3]), fontsize=7, xytext=(4, 4), textcoords="offset points")
            ax.set(title="Discovered rules (colour = lift)", xlabel="support", ylabel="confidence"); return
        items = show[i]; v = np.array([s for _, s in items]) * min(1, g / G * 1.0) if g < G else np.array([s for _, s in items]); y = np.arange(len(items))[::-1]
        ok = [s >= minsup for _, s in items]; ax.barh(y, v, color=[viz.PALETTE[2] if o or g < G else "#ccc" for o in ok]); ax.axvline(minsup, color="k", ls="--"); ax.set_yticks(y, ["+".join(sorted(c)) for c, _ in items], fontsize=8)
        ax.set(title=f"Level {i + 1}: counting {'items' if i == 0 else str(i + 1) + '-itemsets'} ({len(L[i])} candidates; grey = pruned below min support {minsup:.0%})", xlabel="support", xlim=(0, .6))
    viz.animate(fig, update, frames if not quick else frames[::2], gif, fps=3)


if __name__ == "__main__":
    run(main, __file__)
