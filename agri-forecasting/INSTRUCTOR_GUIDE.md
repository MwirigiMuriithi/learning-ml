# Instructor guide

## Pacing (matches the 5-week roadmap)
| Week | Content | Class / lab split |
|---|---|---|
| 13 | 7.1, 7.2, 7.3 | 1 h lecture on components & leakage → labs; **leakage demo (7.3 §3) live** |
| 14 | 7.4, 7.5, 7.6 | smoothing → ARIMA → ML; stress "error by horizon" |
| 15 | 7.7, **7.8 assignment** | assignment due end of week; peer review of reports |
| 16 | 8.1, 8.2, 8.3, 8.4 | live-code the NumPy MLP (8.2): students type along |
| 17 | 8.5–8.11 + **capstone** | projects in pairs; capstone as take-home (4 h) |

## Learning outcomes → assessment
| Outcome | Evidence |
|---|---|
| builds leak-free forecasting pipelines | 7.2 leak test; 7.8 checks |
| evaluates honestly | 7.3 exercises; 7.8 leaderboard; capstone Step 2 |
| explains and implements backprop | 8.2 exercises (dense backward, gradient check, training loop) |
| trains and debugs networks | 8.3–8.4 exercises; 8.5–8.7 projects |
| communicates uncertainty and decisions | capstone Steps 3–6 |

Suggested weights: 7.8 assignment 25 % · Phase 8 projects (8.5, 8.7, 8.11) 30 % · capstone 35 % · participation / lab notebooks 10 %.

## Common misconceptions (and the lesson that dismantles each)
| Misconception | Where addressed |
|---|---|
| "More data points from shuffling = better validation" | 7.3 §3 (shuffle experiment) |
| "ADF p < .05 means no seasonality" | 7.1 pitfall |
| "A flexible model always beats a simple one" | 7.6, 7.7, 8.4, 8.11: linear baseline wins |
| "MAPE is always fine" | 7.3 §2 |
| "LSTMs remember long histories automatically" | 8.8: forget-bias init matters; default init fails |
| "Attention maps fully explain the model" | 8.9 caution |
| "Accuracy is enough" | 8.5 confusion matrix; 8.7 recall/precision/abstention |
| "Validation score of one run is reliable" | 7.7, 8.8 seeds |

## Discussion prompts
1. Your forecast says +12 % in 6 weeks with an 80 % interval of ±20 %. Do you store? What else do you need to know?
2. Why did the 80 % intervals cover only ~65 % at first? Which assumption (exchangeability) is violated by prices?
3. When would you *not* use a neural network for this cooperative?
4. How would you detect, in production, that the model has stopped working?
5. Who is harmed if the disease classifier is over-confident?

## Lab hygiene
Require "Restart & Run All" before submission. Use `ag.set_seed`. Ask for the leak test output. Let students compare leaderboards: rank *stability* across seeds/origins is a richer discussion than rank.

## Extending
Swap in real data (see README §6). Add crops or markets in `src/agrilab.py` (`_PROFILES`, `_MARKET_PROFILES`) and regenerate. New lessons: copy any `build/nb_*.py`, use `Notebook(...).header/md/code/exercise`, then `python build/run.py <name>`.
