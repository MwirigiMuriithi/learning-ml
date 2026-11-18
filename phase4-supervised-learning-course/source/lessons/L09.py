from nbkit import Lesson


def make():
    nb = Lesson("09_Classification_Foundations_and_Metrics", 9, "Classification Foundations & Metrics",
                "Why 99 % accuracy can be worthless, and how to measure a classifier honestly",
                110, "🟡 Beginner → Intermediate",
                setup_extra="""from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score, log_loss, classification_report, balanced_accuracy_score)""")
    nb.header(
        objectives=[
            "Frame a classification problem: classes, hard labels vs probabilities, decision thresholds.",
            "Build and read a **confusion matrix** (TP, FP, FN, TN).",
            "Compute accuracy, precision, recall, specificity, F1 and balanced accuracy: by hand and with scikit-learn.",
            "Explain the **accuracy paradox** on imbalanced data.",
            "Trade precision against recall by moving the threshold; read ROC and precision-recall curves.",
            "Compute ROC-AUC from scratch and interpret it probabilistically.",
            "Choose a threshold from **business costs**, and use stratified splits / class weights.",
        ],
        prereqs="Lessons 00–08",
        hook=("A hospital proudly announces its new cancer-screening test: **99 % accurate!** Impressive: until you learn that only 1 in 100 patients actually has cancer, "
              "and the test is simply a rubber stamp that says *\"healthy\"* to everyone. It is 99 % accurate and **catches zero cancers**.\n\n"
              "That is the *accuracy paradox*, and it has fooled engineers, managers and entire companies. "
              "Before we learn any classification algorithm, we must learn how to **measure** one, because the right metric depends on *what mistakes cost*. "
              "A missed cancer, a blocked genuine payment and a spam email in your inbox are three very different kinds of wrong."),
        roadmap=["Classification in one page", "The confusion matrix", "The metric zoo", "The accuracy paradox",
                 "Probabilities & thresholds", "ROC, AUC and precision–recall curves", "Log loss & calibration",
                 "Costs → thresholds", "Stratification, class weights, multi-class metrics", "Metric cheat-sheet, worked examples, problems, quiz"])

    # ---------------------------------------------------------- 1
    nb.section("🧠", "1. Classification in one page",
               "In **classification** the target is a *category*. With two categories it's **binary** classification: we call one class **positive (1)**: the thing we're hunting for "
               "(fraud, disease, default, spam), and the other **negative (0)**.\n\n"
               "Most classifiers don't output a label directly. They output a **score or probability** $\\hat p=P(y=1\\mid x)$, and a **decision threshold** $\\tau$ turns it into a label:\n\n"
               r"$$\hat y=\begin{cases}1&\hat p\ge\tau\\0&\hat p<\tau\end{cases}\qquad(\text{default }\tau=0.5)$$" "\n\n"
               "The threshold is a *business decision*, not a mathematical constant. Keep that in mind: it will be the punchline of this lesson.\n\n"
               "With more than two categories (**multi-class**, e.g. iris species) the model picks the most probable class, and metrics are averaged over classes (Section 9).")

    # ---------------------------------------------------------- 2 CM
    nb.section("🔲", "2. The confusion matrix",
               "Every prediction lands in one of four boxes, defined by (truth, prediction):")
    nb.md(r"""|  | **Predicted positive** | **Predicted negative** |
|---|---|---|
| **Actually positive** | **TP**: True Positive (hit) | **FN**: False Negative (miss, "Type II error") |
| **Actually negative** | **FP**: False Positive (false alarm, "Type I error") | **TN**: True Negative (correct rejection) |

*Mnemonic:* the second letter is the **prediction** (P/N); the first letter says whether the prediction was **True** (right) or **False** (wrong).""")
    nb.md("**A hand-worked example.** A fraud detector reviews 10 000 transactions, of which 100 are really fraud. It flags 150 transactions; 80 of the flagged ones are genuinely fraudulent.\n\n"
          "- TP = 80 (frauds caught) - FP = 150 − 80 = **70** (innocent customers flagged) - FN = 100 − 80 = **20** (frauds missed) - TN = 10 000 − 80 − 70 − 20 = **9 830**.")
    nb.code(r'''TP, FP, FN, TN = 80, 70, 20, 9830
cm = pd.DataFrame([[TP, FN], [FP, TN]], index=["actually fraud", "actually legit"], columns=["flagged fraud", "flagged legit"]); display(cm)''')

    # ---------------------------------------------------------- 3 metric zoo
    nb.section("🦁", "3. The metric zoo",
               "From four numbers we derive everything. Each answers a different question:")
    nb.md(r"""| Metric | Formula | The question it answers |
|---|---|---|
| **Accuracy** | $\dfrac{TP+TN}{TP+TN+FP+FN}$ | "What fraction of *all* predictions were right?" |
| **Precision** (positive predictive value) | $\dfrac{TP}{TP+FP}$ | "When it says *positive*, how often is it right?" (cost of false alarms) |
| **Recall** (sensitivity, TPR) | $\dfrac{TP}{TP+FN}$ | "Of all *real* positives, how many did we catch?" (cost of misses) |
| **Specificity** (TNR) | $\dfrac{TN}{TN+FP}$ | "Of all real negatives, how many did we correctly clear?" |
| **False-positive rate** | $\dfrac{FP}{FP+TN}=1-\text{specificity}$ | "How many innocents do we wrongly flag?" |
| **F1 score** | $\dfrac{2PR}{P+R}$ (harmonic mean) | "One number balancing precision and recall": low if *either* is low |
| **Fβ** | $\dfrac{(1+\beta^2)PR}{\beta^2P+R}$ | β>1 favours recall, β<1 favours precision |
| **Balanced accuracy** | $\tfrac12(\text{recall}+\text{specificity})$ | Accuracy that isn't fooled by class imbalance |""")
    nb.code(r'''acc = (TP + TN) / (TP + TN + FP + FN); prec = TP / (TP + FP); rec = TP / (TP + FN); spec = TN / (TN + FP)
f1 = 2 * prec * rec / (prec + rec)
print(f"accuracy    = {acc:.4f}   <- 99.1%: sounds fantastic!")
print(f"precision   = {prec:.4f}   <- only 53% of the flags are real fraud")
print(f"recall      = {rec:.4f}   <- we caught 80% of the frauds")
print(f"specificity = {spec:.4f}\nF1          = {f1:.4f}\nbalanced acc= {(rec + spec) / 2:.4f}")''')
    nb.callout("key", "The same model is simultaneously '99.1 % accurate' and '53 % precise'. Neither number is wrong; they answer different questions. "
                      "**Always ask: which mistake is more expensive?** If a missed fraud costs thousands but a false alarm costs a phone call → chase *recall*. "
                      "If a false alarm means blocking a genuine customer's wedding payment → protect *precision*.")

    # ---------------------------------------------------------- 4 paradox
    nb.section("🎭", "4. The accuracy paradox",
               "Let's make a dataset where only 2 % of cases are positive and compare an honest model with a lazy 'always say negative' baseline.")
    nb.code(r'''Xi, yi = make_classification(n_samples=5000, n_features=10, n_informative=5, weights=[0.98, 0.02], flip_y=0.01, class_sep=1.2, random_state=1)
Xi_tr, Xi_te, yi_tr, yi_te = train_test_split(Xi, yi, test_size=0.3, stratify=yi, random_state=RANDOM_STATE)
print(f"positives in the data: {yi.mean():.1%}")

lazy = DummyClassifier(strategy="most_frequent").fit(Xi_tr, yi_tr)
real = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xi_tr, yi_tr)
rows = {}
for name, m in [("Lazy: always 'negative'", lazy), ("Logistic regression", real)]:
    p = m.predict(Xi_te)
    rows[name] = {"accuracy": accuracy_score(yi_te, p), "precision": precision_score(yi_te, p, zero_division=0), "recall": recall_score(yi_te, p),
                  "F1": f1_score(yi_te, p), "balanced acc": balanced_accuracy_score(yi_te, p)}
display(pd.DataFrame(rows).T.round(3))''')
    nb.md("The lazy model's accuracy is ≈ 98 %: *higher than or close to the real model's*, yet recall = 0. It never finds a single positive. "
          "On imbalanced data, **accuracy rewards ignorance**. Prefer precision/recall/F1, balanced accuracy, and curve-based metrics.")

    # ---------------------------------------------------------- 5 scratch
    nb.section("🔧", "5. Metrics from scratch",
               "Everything comes from four counts. Let's build the confusion matrix and metrics ourselves, then check against scikit-learn.")
    nb.code(r'''def confusion_counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    TP = int(np.sum((y_true == 1) & (y_pred == 1))); FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0))); TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    return TP, FP, FN, TN

def metrics_from_counts(TP, FP, FN, TN):
    safe = lambda a, b: a / b if b else 0.0                       # avoid dividing by zero
    prec, rec = safe(TP, TP + FP), safe(TP, TP + FN)
    return {"accuracy": safe(TP + TN, TP + FP + FN + TN), "precision": prec, "recall": rec, "specificity": safe(TN, TN + FP),
            "F1": safe(2 * prec * rec, prec + rec)}

y_demo = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]); p_demo = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
c = confusion_counts(y_demo, p_demo); print("TP, FP, FN, TN =", c)
mine = metrics_from_counts(*c); print({k: round(v, 3) for k, v in mine.items()})
print("sklearn confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_demo, p_demo))
print("sklearn precision/recall/F1:", precision_score(y_demo, p_demo), recall_score(y_demo, p_demo), round(f1_score(y_demo, p_demo), 3))''')
    nb.callout("trap", "scikit-learn's `confusion_matrix` is laid out **[[TN, FP], [FN, TP]]** (rows = truth, columns = prediction, classes sorted ascending): "
                       "the *opposite* of the textbook layout above where TP is top-left. Always label your axes!")

    # ---------------------------------------------------------- 6 thresholds
    nb.section("🎚️", "6. Probabilities and thresholds",
               "We'll use the **Wisconsin breast-cancer** dataset (569 tumours, 30 measurements). We define **malignant = positive (1)**: the class we must not miss. "
               "(In scikit-learn's file the original target is reversed: 0 = malignant, so we flip it.)")
    nb.code(r'''bc = load_breast_cancer()
Xb = pd.DataFrame(bc.data, columns=bc.feature_names); yb = 1 - bc.target          # 1 = malignant
print("malignant share:", yb.mean().round(3), " | samples:", len(yb), " | features:", Xb.shape[1])
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.3, stratify=yb, random_state=RANDOM_STATE)
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xb_tr, yb_tr)
prob = clf.predict_proba(Xb_te)[:, 1]                                             # P(malignant)
print("test accuracy at τ = 0.5:", round(accuracy_score(yb_te, prob >= 0.5), 3))
display(pd.DataFrame(confusion_matrix(yb_te, prob >= 0.5), index=["actually benign", "actually malignant"], columns=["pred benign", "pred malignant"]))''')
    nb.md("Now sweep the threshold from 0 to 1 and watch precision and recall move in opposite directions:")
    nb.code(r'''taus = np.linspace(0.01, 0.99, 99)
P = [precision_score(yb_te, prob >= t, zero_division=1) for t in taus]; R = [recall_score(yb_te, prob >= t) for t in taus]; F = [f1_score(yb_te, prob >= t) for t in taus]
plt.figure(figsize=(8, 4.2)); plt.plot(taus, P, label="precision"); plt.plot(taus, R, label="recall"); plt.plot(taus, F, "--", label="F1")
plt.axvline(0.5, color="gray", ls=":"); plt.xlabel("decision threshold τ"); plt.ylabel("score"); plt.legend(); plt.title("Raise τ → fewer alarms (precision ↑) but more misses (recall ↓)"); plt.show()
for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
    TP_, FP_, FN_, TN_ = confusion_counts(yb_te, prob >= t); print(f"τ={t}: caught {TP_} malignant, missed {FN_}, false alarms {FP_}")''')
    nb.md("For *cancer screening* we'd happily accept a few false alarms (extra biopsies) to miss as few tumours as possible → a **low** threshold. "
          "For an automatic 'block this card' system we might choose a **high** one. Same model, different threshold, different policy.")

    # ---------------------------------------------------------- 7 ROC
    nb.section("📈", "7. ROC curve, AUC and the precision–recall curve",
               "Instead of picking one threshold, sweep them all. The **ROC curve** plots the **true-positive rate (recall)** against the **false-positive rate** for every threshold. "
               "A random classifier follows the diagonal; a perfect one hugs the top-left corner.\n\n"
               "**AUC** (area under the ROC curve) summarises it in one number: 0.5 = coin flip, 1.0 = perfect. Its meaning is wonderfully concrete:\n\n"
               "> AUC = the probability that a **randomly chosen positive** gets a *higher score* than a **randomly chosen negative**.\n\n"
               "Let's build the curve ourselves: sort by score, then walk down the list, counting.")
    nb.code(r'''def roc_from_scratch(y_true, scores):
    y_true = np.asarray(y_true); order = np.argsort(-scores)                  # best-scored first
    ys = y_true[order]
    tps = np.cumsum(ys); fps = np.cumsum(1 - ys)                               # positives / negatives accepted so far
    tpr = np.r_[0, tps / tps[-1]]; fpr = np.r_[0, fps / fps[-1]]
    auc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2)              # trapezoid rule
    return fpr, tpr, auc

fpr, tpr, auc_mine = roc_from_scratch(yb_te, prob)
fpr_s, tpr_s, _ = roc_curve(yb_te, prob)
print(f"AUC (scratch) = {auc_mine:.5f}    AUC (sklearn) = {roc_auc_score(yb_te, prob):.5f}")

# the probabilistic meaning: compare every positive with every negative
pos, neg = prob[yb_te == 1], prob[yb_te == 0]
auc_pairs = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
print(f"P(random positive scores higher than random negative) = {auc_pairs:.5f}   <- the same number!")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].plot(fpr, tpr, lw=2.4, label=f"model (AUC = {auc_mine:.3f})"); axes[0].plot([0, 1], [0, 1], "k--", label="random guessing")
for t in [0.1, 0.5, 0.9]:
    i = np.argmin(np.abs(fpr_s - np.mean(prob[yb_te == 0] >= t))); axes[0].scatter(np.mean(prob[yb_te == 0] >= t), np.mean(prob[yb_te == 1] >= t), s=60, zorder=3); axes[0].annotate(f"τ={t}", (np.mean(prob[yb_te == 0] >= t) + 0.02, np.mean(prob[yb_te == 1] >= t) - 0.05))
axes[0].set_xlabel("false-positive rate"); axes[0].set_ylabel("true-positive rate (recall)"); axes[0].legend(); axes[0].set_title("ROC curve")
pr, rc, _ = precision_recall_curve(yb_te, prob); axes[1].plot(rc, pr, lw=2.4, label=f"AP = {average_precision_score(yb_te, prob):.3f}")
axes[1].axhline(yb_te.mean(), color="k", ls="--", label="baseline = prevalence"); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].legend(); axes[1].set_title("Precision–recall curve")
plt.tight_layout(); plt.show()''')
    nb.callout("tip", "**ROC-AUC vs PR-AUC.** ROC-AUC is threshold-free and robust, but on *highly imbalanced* data it can look rosy because the huge number of negatives keeps the false-positive *rate* small. "
                      "The **precision–recall curve** (summarised by *average precision*) focuses on the positives and is more revealing when positives are rare.")
    nb.code(r'''print("Imbalanced (2% positives) problem:")
sc_i = real.predict_proba(Xi_te)[:, 1]
print(f"  ROC-AUC           = {roc_auc_score(yi_te, sc_i):.3f}   (looks respectable...)")
print(f"  Average precision = {average_precision_score(yi_te, sc_i):.3f}   (...but the PR view shows how hard finding rare positives really is; baseline = prevalence = {yi_te.mean():.3f})")''')

    # ---------------------------------------------------------- 8 log loss
    nb.section("🧮", "8. Log loss and calibration: judging the probabilities themselves",
               "AUC only cares about the *ranking*. But if we promise a customer 'there is a 90 % chance this is fraud', those probabilities had better be honest. "
               "**Log loss** (cross-entropy) scores the probabilities:\n\n"
               r"$$\text{LogLoss}=-\frac1n\sum_i\Big[y_i\ln\hat p_i+(1-y_i)\ln(1-\hat p_i)\Big]$$" "\n\n"
               "Being **confidently wrong** is punished brutally: predicting $\\hat p=0.01$ for a true positive costs $-\\ln 0.01=4.6$, versus $-\\ln0.5=0.69$ for a shrug. Lower is better.")
    nb.code(r'''def log_loss_scratch(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps); y = np.asarray(y)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
print("scratch  :", round(log_loss_scratch(yb_te, prob), 5), "  sklearn:", round(log_loss(yb_te, prob), 5))
for p_ in [0.5, 0.9, 0.99, 0.01]: print(f"true positive predicted with p={p_:<5} -> loss {-np.log(p_):.2f}")''')
    nb.md("**Calibration** asks: *of the cases where the model says 70 %, do about 70 % turn out positive?* A **reliability diagram** bins the predictions and compares:")
    nb.code(r'''from sklearn.calibration import calibration_curve
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for name, m, Xe, ye in [("Logistic (cancer)", clf, Xb_te, yb_te)]:
    pt, pp = calibration_curve(ye, m.predict_proba(Xe)[:, 1], n_bins=6, strategy="quantile"); ax.plot(pp, pt, "o-", label=name)
ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated"); ax.set_xlabel("mean predicted probability"); ax.set_ylabel("observed fraction positive"); ax.legend(); ax.set_title("Reliability diagram"); plt.show()''')

    # ---------------------------------------------------------- 9 costs / strat
    nb.section("💰", "9. Costs → thresholds, and other practicalities",
               "### 9a. From costs to a threshold\n"
               "Suppose a **missed cancer** (FN) costs $c_{FN}=10$ units (delayed treatment) while a **false alarm** (FP) costs $c_{FP}=1$ (an extra test). "
               "For **well-calibrated** probabilities, predicting positive is the cheaper choice whenever expected cost of acting is lower than of ignoring:\n\n"
               r"$$(1-\hat p)\,c_{FP}<\hat p\,c_{FN}\iff \hat p>\frac{c_{FP}}{c_{FP}+c_{FN}}=\frac{1}{11}\approx0.09$$" "\n\n"
               "The optimal threshold is **0.09, not 0.5**! Let's verify empirically by computing total cost over a threshold sweep:")
    nb.code(r'''c_fn, c_fp = 10, 1
costs = []
for t in taus:
    TP_, FP_, FN_, TN_ = confusion_counts(yb_te, prob >= t); costs.append(c_fn * FN_ + c_fp * FP_)
best_t = taus[int(np.argmin(costs))]
plt.plot(taus, costs); plt.axvline(1 / 11, color="green", ls="--", label=f"theory: 1/11 = {1/11:.2f}"); plt.axvline(best_t, color="red", ls=":", label=f"empirical best = {best_t:.2f}")
plt.axvline(0.5, color="gray", ls=":", label="default 0.5"); plt.xlabel("threshold τ"); plt.ylabel("total cost on test set"); plt.legend(); plt.title("Cost-sensitive threshold"); plt.show()
print(f"cost at τ=0.5: {costs[49]}   |   cost at best τ ({best_t:.2f}): {min(costs)}")''')
    nb.md("The empirical optimum (≈ 0.2) is not exactly the theoretical 0.09: with only 171 test cases the cost curve is flat and noisy over a wide range, and the model's probabilities are only *roughly* calibrated. "
          "But both are far below the default 0.5, and moving there cuts the total cost by about two thirds. The theory tells you the *direction and magnitude*; validation data fine-tunes it.")
    nb.md("### 9b. Stratified splits\nWith rare positives, a random split can leave the test set with almost none. **Stratified** splitting (`stratify=y`, `StratifiedKFold`) keeps the class proportions the same in every fold. Use it by default for classification (scikit-learn's `cross_val_score` already stratifies for classifiers when `cv` is an int).")
    nb.code(r'''skf = StratifiedKFold(5, shuffle=True, random_state=0)
print("positive share per fold (stratified):", [round(float(yi[te].mean()), 4) for _, te in skf.split(Xi, yi)])
rand_shares = [round(float(yi[np.random.default_rng(s).choice(len(yi), 1000, replace=False)].mean()), 4) for s in range(5)]
print("positive share in 5 random 1000-row samples:", rand_shares)''')
    nb.md("### 9c. Class weights\nAnother way to fight imbalance is to tell the algorithm that mistakes on the rare class cost more: `class_weight='balanced'` re-weights the loss inversely to class frequency. "
          "It trades precision for recall (roughly equivalent to lowering the threshold). Look at the numbers below: recall doubles, but precision collapses and F1 actually gets *worse*: class weights are a lever, not a free lunch. Whether the trade is worthwhile depends entirely on the costs from section 9a.")
    nb.code(r'''for cw in [None, "balanced"]:
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=cw)).fit(Xi_tr, yi_tr); p = m.predict(Xi_te)
    print(f"class_weight={str(cw):<9} precision={precision_score(yi_te, p, zero_division=0):.3f}  recall={recall_score(yi_te, p):.3f}  F1={f1_score(yi_te, p):.3f}")''')
    nb.md("### 9d. Multi-class metrics\nFor more than two classes, compute per-class precision/recall, then average: **macro** (every class counts equally), **weighted** (by class size), **micro** (pool all decisions: equals accuracy for single-label problems).")
    nb.code(r'''iris = load_iris(); Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(iris.data, iris.target, test_size=0.3, stratify=iris.target, random_state=RANDOM_STATE)
mc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xr_tr, yr_tr); pr_ = mc.predict(Xr_te)
print(classification_report(yr_te, pr_, target_names=iris.target_names, digits=3))
print("confusion matrix (rows = truth):\n", confusion_matrix(yr_te, pr_))''')

    # ---------------------------------------------------------- 10 cheat sheet
    nb.section("📋", "10. Metric cheat-sheet: which one when?")
    nb.table([
        ["Balanced classes, equal costs", "Accuracy, F1, ROC-AUC", "Everyone agrees; accuracy is fine here"],
        ["Rare positives (fraud, disease)", "Recall, precision, **PR-AUC**, F1/Fβ, balanced accuracy", "Accuracy is misleading"],
        ["Missing a positive is very costly (screening, safety)", "**Recall**/sensitivity, Fβ with β>1, low threshold", "Accept false alarms"],
        ["False alarms are very costly (auto-blocking, legal action)", "**Precision**, Fβ with β<1, high threshold", "Accept some misses"],
        ["Need a threshold-free ranking quality", "ROC-AUC (or PR-AUC if rare)", "Good for comparing models"],
        ["Probabilities drive decisions (pricing, risk)", "Log loss, Brier score, calibration curve", "Ranking isn't enough"],
        ["Known unit costs", "Expected cost; threshold = c_FP/(c_FP + c_FN) if calibrated", "The most business-relevant metric"],
        ["Many classes", "Macro-F1 (rare classes matter), weighted-F1, confusion matrix", "Inspect per-class results"],
    ], ["Situation", "Prefer", "Note"])
    nb.callout("warn", "**Data leakage warning (a preview).** Metrics are only honest if the model never saw information it wouldn't have *at prediction time*. "
                       "Example: predicting 'will this loan be approved?' using a column that only exists *after* approval (such as 'interest rate offered') gives a miraculous AUC ≈ 1.0, and a useless model. "
                       "If a result looks too good to be true, hunt for leakage. You'll meet this in Project 3.")

    # ---------------------------------------------------------- 11 worked
    nb.section("🧪", "11. Worked examples")
    nb.md(r"""### Worked Example 1: Choose the metric

| Scenario | Costly mistake | Metric to optimise |
|---|---|---|
| Airport weapon scanner | Missing a weapon (FN) | **Recall** (accept many false alarms) |
| Email spam filter | Flagging an important email (FP) | **Precision** (let some spam through) |
| Ranking leads for a sales team | Order of leads | **ROC-AUC / PR-AUC** |
| Loan approvals with known default losses | Money lost on FP and on FN | **Expected cost** at a tuned threshold |""")
    nb.md(r"""### Worked Example 2: F1 by hand

A model finds 30 of 40 real positives (TP = 30, FN = 10) and raises 20 false alarms (FP = 20). Precision $=30/50=0.60$, recall $=30/40=0.75$, $F_1=\dfrac{2(0.6)(0.75)}{0.6+0.75}=\dfrac{0.9}{1.35}=\mathbf{0.667}$. Note F1 (0.667) sits closer to the *lower* of the two numbers than the arithmetic mean (0.675) would: it punishes imbalance between them.""")
    nb.code(r'''P_, R_ = 0.6, 0.75; print("F1 =", round(2 * P_ * R_ / (P_ + R_), 4), "| arithmetic mean =", (P_ + R_) / 2)''')

    # ---------------------------------------------------------- problems
    nb.problem(
        "Confusion matrix and metrics by hand",
        "Given\n```python\ny_true = np.array([1,0,1,1,0,0,1,0,0,1,0,1])\ny_pred = np.array([1,0,0,1,0,1,1,0,0,0,0,1])\n```\n"
        "Using the lesson's `confusion_counts`, set `TP, FP, FN, TN` and compute `precision`, `recall`, `f1` and `accuracy` (plain Python/NumPy arithmetic).",
        ["`TP, FP, FN, TN = confusion_counts(y_true, y_pred)`.",
         "precision = TP / (TP + FP); recall = TP / (TP + FN).",
         "F1 = 2·precision·recall / (precision + recall)."],
        'y_true = np.array([1,0,1,1,0,0,1,0,0,1,0,1]); y_pred = np.array([1,0,0,1,0,1,1,0,0,0,0,1])\nTP = FP = FN = TN = precision = recall = f1 = accuracy = None   # your code\n',
        'y_true = np.array([1,0,1,1,0,0,1,0,0,1,0,1]); y_pred = np.array([1,0,0,1,0,1,1,0,0,0,0,1])\nTP, FP, FN, TN = confusion_counts(y_true, y_pred)\nprecision = TP / (TP + FP); recall = TP / (TP + FN)\nf1 = 2 * precision * recall / (precision + recall); accuracy = (TP + TN) / len(y_true)\nprint((TP, FP, FN, TN), round(precision, 3), round(recall, 3), round(f1, 3), round(accuracy, 3))',
        'check("TP=4, FP=1, FN=2, TN=5", lambda: (TP, FP, FN, TN) == (4, 1, 2, 5))\ncheck("precision = 0.8", lambda: abs(precision - 0.8) < 1e-9)\ncheck("recall = 2/3", lambda: abs(recall - 2 / 3) < 1e-9)\ncheck("F1 matches sklearn", lambda: abs(f1 - f1_score(y_true, y_pred)) < 1e-9)\ncheck("accuracy = 0.75", lambda: abs(accuracy - 0.75) < 1e-9)',
        explain="4 hits, 2 misses, 1 false alarm, 5 correct rejections. Precision 4/5, recall 4/6, accuracy 9/12.")

    nb.problem(
        "Expose the lazy model",
        "On the imbalanced data (`Xi_tr, yi_tr, Xi_te, yi_te`), build a `DummyClassifier(strategy='most_frequent')` and the lesson's `real` model. "
        "Store each one's **accuracy** and **balanced accuracy** on the test set in `acc_lazy, bacc_lazy, acc_real, bacc_real`. "
        "Set `accuracy_fooled = acc_lazy > acc_real - 0.02` (is the lazy model's accuracy within 0.02 of (or better than) the real one?).",
        ["`accuracy_score(y, model.predict(X))` and `balanced_accuracy_score`.",
         "Balanced accuracy of the lazy model is exactly 0.5.",
         "The real model's accuracy is only slightly higher than the lazy one."],
        'acc_lazy = bacc_lazy = acc_real = bacc_real = accuracy_fooled = None   # your code\n',
        'acc_lazy = accuracy_score(yi_te, lazy.predict(Xi_te)); bacc_lazy = balanced_accuracy_score(yi_te, lazy.predict(Xi_te))\nacc_real = accuracy_score(yi_te, real.predict(Xi_te)); bacc_real = balanced_accuracy_score(yi_te, real.predict(Xi_te))\naccuracy_fooled = bool(acc_lazy > acc_real - 0.02)\nprint(acc_lazy, bacc_lazy, acc_real, bacc_real, accuracy_fooled)',
        'check("lazy accuracy > 0.97", lambda: acc_lazy > 0.97)\ncheck("lazy balanced accuracy = 0.5", lambda: abs(bacc_lazy - 0.5) < 1e-9)\ncheck("real model has clearly higher balanced accuracy", lambda: bacc_real > bacc_lazy + 0.05)\ncheck("accuracy alone cannot separate them", lambda: accuracy_fooled)',
        explain="Accuracy differences are tiny (≈ 98 % vs 98 %), while balanced accuracy separates them clearly (0.5 vs ≈ 0.6+). Choose metrics that notice the rare class.")

    nb.problem(
        "A threshold for 98 % recall",
        "Using `prob` and `yb_te` (cancer test set, positive = malignant): scan `taus = np.linspace(0.01, 0.99, 99)` and find the **largest** threshold `tau_star` whose recall is **≥ 0.98**. "
        "Store `recall_at` and `precision_at` for that threshold, and `fn_at` (number of missed malignant tumours).",
        ["Loop over `taus`, compute `recall_score(yb_te, prob >= t)`, keep those ≥ 0.98, take the maximum threshold.",
         "Then compute precision and the false-negative count at `tau_star`.",
         "Higher thresholds trade recall for precision, so the *largest* qualifying τ gives the best precision."],
        'tau_star = recall_at = precision_at = fn_at = None   # your code\n',
        'ok = [t for t in taus if recall_score(yb_te, prob >= t) >= 0.98]\ntau_star = max(ok)\nrecall_at = recall_score(yb_te, prob >= tau_star); precision_at = precision_score(yb_te, prob >= tau_star)\nfn_at = confusion_counts(yb_te, prob >= tau_star)[2]\nprint(tau_star, recall_at, precision_at, fn_at)',
        'check("recall at tau_star is at least 0.98", lambda: recall_at >= 0.98)\ncheck("no larger threshold also reaches 0.98 recall", lambda: all(recall_score(yb_te, prob >= t) < 0.98 for t in taus if t > tau_star))\ncheck("precision is still high (> 0.85)", lambda: precision_at > 0.85)',
        explain="This is how screening programmes are specified in practice: 'catch ≥ 98 % of cases' fixes the recall target, and the model/threshold must then deliver the best precision it can.",
        level="⭐⭐")

    nb.problem(
        "AUC from pairs",
        "Write `auc_pairs(y, scores)` that computes ROC-AUC as the probability that a random positive outscores a random negative (count ties as ½) using broadcasting: no `roc_auc_score`. "
        "Store `auc1 = auc_pairs(yb_te, prob)` and `auc2` for the **imbalanced** problem `auc_pairs(yi_te, sc_i)`.",
        ["`pos = scores[y == 1]`, `neg = scores[y == 0]`.",
         "`(pos[:, None] > neg[None, :]).mean()` compares all pairs; add `0.5 * (pos[:, None] == neg[None, :]).mean()` for ties.",
         "Compare with `roc_auc_score`."],
        'def auc_pairs(y, scores):\n    pass\n\nauc1 = auc2 = None   # your code\n',
        'def auc_pairs(y, scores):\n    y = np.asarray(y); pos, neg = scores[y == 1], scores[y == 0]\n    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()\n\nauc1 = auc_pairs(yb_te, prob); auc2 = auc_pairs(yi_te, sc_i)\nprint(auc1, auc2)',
        'check("matches sklearn (cancer)", lambda: abs(auc1 - roc_auc_score(yb_te, prob)) < 1e-9)\ncheck("matches sklearn (imbalanced)", lambda: abs(auc2 - roc_auc_score(yi_te, sc_i)) < 1e-9)',
        explain="AUC really is just a ranking statistic: the fraction of positive/negative pairs the model orders correctly. That's why it ignores thresholds and calibration.",
        level="⭐⭐")

    nb.problem(
        "Cost-sensitive decision",
        "A missed malignant tumour costs 10, a false alarm costs 1. Using `prob`, `yb_te`:\n"
        "- `cost_default` = total cost at τ = 0.5\n- `tau_opt` = the threshold in `taus` that **minimises** total cost (first one in case of ties)\n- `cost_opt` = its cost\n- `saving_pct` = percentage saving versus the default.",
        ["Total cost = `10 * FN + 1 * FP` from `confusion_counts(yb_te, prob >= t)`.",
         "Compute costs for every τ in `taus`; `np.argmin` gives the best index.",
         "Theory says the optimum is near 1/11 ≈ 0.09, though noisy with only 171 test cases."],
        'cost_default = tau_opt = cost_opt = saving_pct = None   # your code\n',
        'def total_cost(t):\n    TP_, FP_, FN_, TN_ = confusion_counts(yb_te, prob >= t); return 10 * FN_ + 1 * FP_\n\ncost_default = total_cost(0.5)\ncosts_all = [total_cost(t) for t in taus]\ntau_opt = taus[int(np.argmin(costs_all))]; cost_opt = min(costs_all)\nsaving_pct = (cost_default - cost_opt) / cost_default * 100\nprint(cost_default, tau_opt, cost_opt, round(saving_pct, 1))',
        'check("tuned threshold is cheaper than 0.5", lambda: cost_opt < cost_default)\ncheck("optimal τ is well below 0.5 (< 0.3)", lambda: tau_opt < 0.3)\ncheck("saving is positive", lambda: saving_pct > 0)',
        explain="When misses are 10× worse than false alarms, the best operating point moves to a low threshold: we deliberately raise more alarms to avoid costly misses. Default τ = 0.5 silently assumes both mistakes cost the same.",
        level="⭐⭐")

    nb.quiz([
        ("A model has 95 % accuracy on data where 95 % of cases are negative. Is it good?",
         "Not necessarily: 'always negative' scores the same. Check recall/precision, balanced accuracy, or PR-AUC."),
        ("You lower the decision threshold from 0.5 to 0.2. What happens to recall and precision (typically)?",
         "Recall goes up (more positives caught), precision goes down (more false alarms)."),
        ("Interpret ROC-AUC = 0.85 in a sentence.",
         "A randomly chosen positive gets a higher score than a randomly chosen negative 85 % of the time."),
        ("Why is the PR curve preferred to ROC when positives are very rare?",
         "ROC's false-positive *rate* has a huge number of negatives in the denominator, so even many false alarms look small; precision exposes them directly."),
        ("False alarms cost 1 and misses cost 4. What is the optimal threshold for calibrated probabilities?",
         "c_FP / (c_FP + c_FN) = 1/5 = 0.2."),
    ])

    nb.footer(["A classifier outputs scores; a **threshold** (a business decision) turns them into labels.",
               "The confusion matrix (TP, FP, FN, TN) produces every metric: accuracy, precision, recall, specificity, F1…",
               "Accuracy is dangerous on imbalanced data: use recall/precision/F1, balanced accuracy, PR-AUC.",
               "ROC-AUC = probability a random positive outranks a random negative; log loss and calibration judge the probabilities.",
               "Optimal thresholds come from costs; use stratified splits and class weights for imbalance."],
              "Logistic Regression", "10_Logistic_Regression.ipynb")
    return nb
