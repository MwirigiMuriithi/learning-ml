# 🎤 5-Minute Presentation Guide

**Audience:** a cooperative board and an extension officer: smart, busy, not ML experts. **Goal:** convince them you built something *useful and honest*.

| Time | Slide | Content |
|:-:|:--|:--|
| 0:00 | 1. The farmer's problem | Wanjiru's two questions (what's wrong with my crop? when should I sell?). One image. |
| 0:30 | 2. What FieldAdvisor does | The pipeline diagram: photo → CNN; history → forecaster; both → advice card. |
| 1:15 | 3. Disease model | Architecture in one line; accuracy **and per-class recall**; one confusion-matrix insight; the severity finding (mild infections are hard). |
| 2:15 | 4. Price model | Baselines vs your model on one chart/table (MAE vs naive); the interval with its **measured coverage**. |
| 3:00 | 5. The advice card (live or screenshot) | Show a confident case, an uncertain case that says "retake photo", and a HOLD/SELL case. |
| 3:45 | 6. Honest limitations | Synthetic data; small test sets; what would break in the field; who is excluded. |
| 4:30 | 7. Next steps | Real data, offline TFLite model, local-language UI, pilot with the cooperative, monitoring plan. |
| 5:00 | Q&A | |

**Rules of thumb:** one idea per slide · numbers with baselines · show a failure case · say "we don't know yet" when true · rehearse out loud to 4:30.
**Likely questions:** *Why should I trust it? What if it's wrong? Does it work on my phone? What about my variety/region? Who sees my photos?*
