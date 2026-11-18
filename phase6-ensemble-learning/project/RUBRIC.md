# 📏 Capstone Rubric (100 points + 5 bonus)

Score each criterion 0–4 (0 = missing, 1 = weak, 2 = adequate, 3 = strong, 4 = excellent) and multiply by the weight.

| criterion | weight | 4 = excellent looks like… | 2 = adequate looks like… | 0–1 looks like… |
|---|---|---|---|---|
| **Framing & metric** | ×2.5 (10) | Metric explicitly justified for a target spanning crops with ×10 different scales; baseline defined and quantified in plain language ("typical error ×3") | A sensible metric, thin justification | Raw RMSE on t/ha with no discussion |
| **Data audit & leakage control** | ×2.5 (10) | Ids removed; `year` decision argued; missingness quantified and handled inside the pipeline; no target-derived features | Mostly right, some choices unexplained | Ids/target in features; imputation before splitting |
| **Validation design** | ×3.75 (15) | Time-forward test; CV inside training seasons; identical folds for all models; test untouched until M6 | CV used, test mostly protected | Single random split; peeked at test |
| **Model benchmark** | ×3.75 (15) | ≥ 5 families + baseline, mean ± spread, a chart, discussion of *why* families differ | Table of scores, little interpretation | Models compared on different splits/metrics |
| **Tuning** | ×2.5 (10) | ≥ 30 trials; log-scale for rates; sensible ranges; CV gain measured; notes that tuned CV is optimistic | Tuning done, space not justified | Manual guessing / no baseline comparison |
| **Ensembling** | ×2.5 (10) | OOF predictions; non-negative meta-learner; **pre-declared** decision rule applied | Blend built correctly; rule post-hoc | In-sample stacking (leakage) |
| **Final evaluation** | ×2.5 (10) | One test evaluation; per-crop errors in t/ha and %; comparison with baseline; consistency with CV discussed | Test RMSE reported, no per-crop view | Model chosen by test score |
| **Interpretation** | ×2.5 (10) | SHAP/permutation results checked against agronomy; notes correlated-feature caveats; no causal claims | Importance chart described | Built-in importance only; causal claims |
| **Executive summary** | ×2.5 (10) | ≈ 500 words, quantified, plain-language, states uncertainty, risks, recommendation and monitoring | Complete but technical / vague on risks | Missing sections or unsupported claims |
| **Bonus** | +5 | Any stretch task done properly with evidence | | |

**Grade bands:** 90+ distinction · 75–89 pass with merit · 60–74 pass · < 60 revisit Lessons 06–08 and resubmit.

### Common deductions
* −5 to −10: test set used for any decision before M6
* −5: preprocessing fitted outside the CV loop
* −5: claiming causation from SHAP
* −3: reporting only a single number without spread
