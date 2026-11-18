# 🔐 ANSWER KEY: open only after you have submitted your project

The capstone data were simulated (`generate_data.py → make_project_data`) with the following hidden structure:

1. **Rainfall → price with a ~6-week lag**, via the 4-week rolling mean of rainfall (so lags 6–9 matter). Coefficient (on the z-scored driver) −0.07 for beans, −0.04 for potato. Consequence: rain is only legitimately usable for horizons ≤ 6; for h = 8, 12 only older rain (lag ≥ h) is available and carries less signal.
2. **Exchange rate → price with a 4-week lag** (pass-through 0.30 beans, 0.15 potato), with a one-off **+22 % currency jump on 2020-03-15**.
3. **Potato heat stress (non-linear):** +4.5 % price per °C that temperature *3 weeks earlier* exceeded **27.5 °C** (zero below). A linear model on raw temperature misses this; trees and a hinge feature `max(0, temp − 27.5)` capture it.
4. **Structural break (beans):** from **2020-07-01** a storage subsidy cuts the seasonal amplitude by **40 %**. A model trained on the whole history averages two regimes; a sliding window or break indicator helps.
5. **Drought inside the hold-out:** rainfall is cut to 30 % of normal from **2023-09-15 to 2024-01-31**; with the 6-week lag, its price impact lands mostly in the first half of 2024. Earlier droughts: 2016 and 2019 (Mar–Jun).
6. **Seasonality:** beans (one annual hump + small second harmonic; potato) two humps per year (second harmonic dominant).
7. **Festival bumps:** +5 % beans, +3 % potato.
8. **Mess:** 5 missing weeks per commodity; 2 duplicate rows; 2 decimal-shift typos per commodity (×10 and ÷10); 3-week rainfall outage in May 2018 (hold-out is clean).
9. **Trap column:** `arrivals_tonnes` is generated *from* the price (concurrent): using it gives spectacular and meaningless results.

A worked reference solution is in `../solutions/09_Capstone_Project_Reference_Solution.ipynb`.
