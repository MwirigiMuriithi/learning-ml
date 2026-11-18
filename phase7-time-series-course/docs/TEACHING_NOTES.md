# 🧑‍🏫 Teaching notes (for instructors and self-learners who like a map)

## Lesson design pattern
Every lesson: **objectives → story/analogy → worked example → graded problems with progressive hints → recap, common mistakes, preview.**
Each notebook is also a *script for a 90–120 minute session*: ~35 % live coding, ~25 % discussion, ~40 % student problems.

## Five "aha" moments to engineer (and where they live)
| Moment | Lesson | How |
|---|---|---|
| "Shuffling makes models look 5× better, and it's fake" | L3 §4 | same model, two splits, side-by-side MAE |
| "The naive forecast is hard to beat" | L0, L3, L4 | α ≈ 1 in SES; h = 1 ML models lose to naive |
| "Seasonality is predictable information" | L4–L5 | Holt-Winters / Fourier terms halve the error |
| "A forest can't follow a trend" | L6 §2 | prediction plateaus at the training maximum |
| "Same code, different seed, different answer" | L7 §8 | LSTM across 5 seeds; ensembles stabilise |

## Suggested discussion questions
1. A model has MAPE 3 % on a random split and 12 % on a chronological split. Which number do you tell the minister? Why?
2. Why is `rolling(5, center=True)` fine in a chart and wrong in a feature?
3. You have rainfall data for the *current* week. Can you use it to forecast *next* week's price? What about next-month's?
4. When would you prefer a damped Holt over ARIMA? When ARIMA over XGBoost?
5. Your XGBoost beats naive at h = 8 but loses at h = 1. Is that a bug?

## Pacing tips
- L2 and L5 are the heaviest. If time is short, assign L2 §5–7 and L5 §8 (SARIMAX tournament) as self-study.
- L7 can be run as a demonstration; the gradient-flow plot and the seed-variance table carry the message.
- Encourage students to **change one thing at a time** in the tournaments (window, K Fourier terms, α).

## Grading problem sets
Auto-checks (`check(...)`) verify numeric answers; the *explanations* (comments, short markdown answers) carry the real marks. Compare against `solutions/*_SOLUTIONS.ipynb`. Accept alternative valid methods: the solutions say "there is often more than one valid way".

## Known limitations (be upfront with students)
- Agricultural data are simulated; real data are messier, with structural change, reporting errors and policy shocks.
- Walk-forward evaluations use 10–20 origins to keep runtimes short; differences of a few percent are within noise. A good habit to instil: *report uncertainty about the comparison itself*.
- Library versions can move third decimals; conclusions should be stable.
