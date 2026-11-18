# 🏆 Capstone Brief: FieldAdvisor

**Context.** Smallholder maize farmers must decide *what is wrong with my crop?* and *when should I sell?*: often with a basic phone and little connectivity.
**Mission.** Build the intelligence behind a prototype decision-support app, evaluate it honestly, and document its limits.

## What you build
1. **Leaf-disease classifier** (CNN): five classes: healthy, gray leaf spot, northern leaf blight, common rust, maize streak virus.
2. **Maize-price forecaster** (Ridge and/or LSTM/GRU): 1–4 weeks ahead for Nairobi, with an 80 % prediction interval.
3. **Advice engine**: turns model outputs into a farmer-friendly card with:
   - diagnosis **only when confident** (otherwise "retake photo" or "refer to extension officer"),
   - price outlook with range,
   - a transparent **HOLD / HOLD (low confidence) / SELL NOW** recommendation using a storage-cost threshold.
4. **System evaluation** on the held-out test data.
5. **Model card + ethics review**, and a **5-minute presentation**.

## Constraints & rules
- Data: the bundled synthetic leaf images and price series (Section 5 of the notebook lists real-data stretch tracks).
- **Chronological** price splits (train ≤ 2020, val 2021–22, test 2023–25). No test data in training, tuning, scaling or interval calibration.
- The **test set is used once** for the final report. Pre-register your success criteria first (see the notebook).
- Everything must be **reproducible**: seeds set, runs top-to-bottom, dependencies in `requirements.txt`.
- CPU-only; total training time under ~10 minutes.

## Milestones
| # | Deliverable | Self-check in notebook |
|:-:|:--|:--|
| 1 | Leaf data pipeline (channels-first, train-only normalisation) | ✅ |
| 2 | Disease CNN ≥ 85 % test accuracy, every-class recall ≥ 65 % (aim 70 %+) | ✅ |
| 3 | Price forecaster beating 0.8 × naive MAE at 4 weeks | ✅ |
| 4 | Advice engine passing behavioural tests (abstains when unsure) | ✅ |
| 5 | System report: accuracy, recalls, MAE vs naive, interval coverage | ✅ |
| 6 | Model card (5 sections, real numbers, ≥ 3 limitations) | ✅ |
| 7 | End-to-end demo of three advice cards | ✅ |

## Deliverables
`16_capstone_fieldadvisor.ipynb` (completed) · `MODEL_CARD.md` (export your `model_card` string) · written ethics answers · slides (5 minutes).

## Suggested timeline (2 days)
**Day 1:** M1–M3 (get a baseline pipeline working *first*, then improve) · **Day 2 morning:** M4–M5 · **Day 2 afternoon:** M6–M7, ethics answers, slides, rehearsal.

## Tips from your professor
- Get an end-to-end *ugly* version working before polishing any model.
- Compare against baselines at every step; write down what you tried (even what failed).
- When a number looks great, ask what could be leaking. When it looks bad, look at the images.
- The best capstones are *honest*: they show uncertainty, failure cases and limitations clearly.
