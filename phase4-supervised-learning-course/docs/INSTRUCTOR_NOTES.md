# 🧑‍🏫 Instructor Notes

## Philosophy
Each lesson follows a *lecture arc* (hook → concept → math → build → use → experiment → consolidate → practise). The tone is deliberately conversational; the rigour is in the **code and checks**, not the prose. Two principles run throughout:

1. **Build it, then trust it.** Every algorithm is implemented from scratch and compared with scikit-learn (usually to ~1e-6 or better; tree classifiers differ only by tie-breaking, which the lesson explains).
2. **Measure honestly.** Baselines, leakage, stratified/time-aware CV, uncertainty and costs appear in every showdown and project.

## Suggested use
- **Self-study:** follow `docs/STUDY_PLAN.md`.
- **Flipped classroom:** assign the notebook as pre-reading; spend class time on the 🤔 "pause and think" boxes, the quizzes, and ⭐⭐⭐ problems.
- **Live coding:** the *from scratch* sections (≈ 30–60 lines each) are ideal to code live; students then compare with scikit-learn.
- **Assessment:** problems carry automatic self-checks (`check(...)` prints ✅/❌ and never crashes); projects have rubrics (100 points).

## Timing guide
Lessons 75–150 min each (≈ 35 h total); projects 240–270 min (≈ 17 h). The slowest cells are in Lessons 08, 12, 16, 17 (grid searches); each solutions notebook executes in < 2 min on a laptop.

## Good discussion questions
- 01: Why do squared errors give a closed form but absolute errors don't?  
- 03: Why is polynomial regression "linear"? What breaks at degree 14?  
- 05: Why can adding trees never overfit, yet a deep tree does?  
- 08: Why did the polynomial model lead at default settings, and what changed after tuning the SVR?  
- 09/16: Who should choose the decision threshold: the data scientist or the business? On what evidence?  
- 14: When is a "wrong" assumption (independence) still useful?  
- P2/P3: A fairness gap appears in a tiny group: what do you report?

## Known "teachable surprises" (kept on purpose)
- Single decision tree < log-linear model on house prices (L04); forest ≈ close (L05).
- Degree-2 polynomial + Ridge leads at defaults; tuned SVR overtakes (L08).
- On loans, the *tree* (not Naive Bayes) has the worst log loss (L16).
- Hard voting beating soft voting on one small split is noise (L12).
- Bootstrap CI for a fairness gap touching zero exactly (P2): report "borderline".
- Same-month supply barely helps in P4; the leakage lesson is *which features are available at forecast time*.

## Regenerating or customising
`source/` contains the generator. Each lesson is a Python module that builds both the student and solution notebooks from one source (`nbkit.Lesson`: `md`, `code`, `problem(...)`).
```bash
pip install nbformat nbconvert nbclient ipykernel
cd source
python make_all.py L05           # rebuild one lesson (executes the solutions notebook and runs all self-checks)
python make_all.py --verify-student   # rebuild all; also check that every student notebook runs top-to-bottom
python make_all.py L05 --no-exec      # structural build only (does NOT refresh solution outputs)
```
The build **fails** if any self-check in a solutions notebook prints ❌, which is how the course's numbers stay honest when you edit data or text.

## Datasets
`data/generate_datasets.py` regenerates every CSV deterministically (seed 42; mystery sets use seeds 10–12). Change a seed or a coefficient to create new cohorts: then rebuild and read the new outputs before trusting the prose!

## Licence & reuse
Use freely for teaching and learning. All data are synthetic; no real persons, loans or prices are represented.
