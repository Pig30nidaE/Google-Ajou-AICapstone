# YMJ Exp.11 Nested-CV Cause Experiment

- Mode: full
- Repeated outer CV: 10-fold × 10
- Inner CV for B: 5-fold
- Subjects: 162 (CN=111, MCI=51)

## Mean repeated-OOF metrics

| Condition | ROC-AUC | Accuracy | Sensitivity | Specificity | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| B_strict_tuned | 0.5424 | 0.5864 | 0.3275 | 0.7054 | 0.3309 | 0.5164 |
| C_strict_fixed | 0.5208 | 0.5710 | 0.2980 | 0.6964 | 0.3023 | 0.4972 |
| F_no_rfe_fixed | 0.5353 | 0.5877 | 0.3451 | 0.6991 | 0.3444 | 0.5221 |
| G_global_rfe_fixed | 0.8254 | 0.7926 | 0.5843 | 0.8883 | 0.6397 | 0.7363 |

## Paired subject-bootstrap AUC differences

| Contrast | Mean difference | 95% percentile CI |
|---|---:|---:|
| G_minus_C_global_selection_effect | +0.3051 | [+0.2495, +0.3614] |
| B_minus_C_nested_tuning_effect | +0.0220 | [-0.0065, +0.0497] |
| F_minus_C_no_rfe_effect | +0.0141 | [-0.0247, +0.0564] |

Interpretation signs:

- G−C > 0: using all labels for feature selection creates optimistic performance.
- B−C < 0: inner-CV tuning is less stable than the preregistered fixed setting.
- F−C > 0: fold-local RFE harms performance relative to regularized all-feature LR.

Subject-level OOF predictions and fold assignments are stored only in `private/` and are not included in the public ZIP.
