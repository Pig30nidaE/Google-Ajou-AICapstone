# Fixed-domain 3-class XGBoost result

Single fixed model, five-fold subject-disjoint OOF; no nested CV or tuning.

| Metric | Estimate | Conditional 95% CI |
|---|---:|---:|
| accuracy | 0.6149 | [0.5460, 0.6839] |
| macro_f1 | 0.5481 | [0.4591, 0.6338] |
| balanced_accuracy | 0.5937 | [0.4842, 0.6989] |
| macro_ovr_auc | 0.7246 | [0.6368, 0.8057] |
| recall_CN | 0.6486 | [0.5586, 0.7297] |
| recall_MCI | 0.5490 | [0.4118, 0.6667] |
| recall_Dementia | 0.5833 | [0.3333, 0.8333] |

Confusion matrix: rows=true, columns=predicted, order CN/MCI/Dementia.

```
[[72 29 10]
 [20 28  3]
 [ 2  3  7]]
```

Only 12 Dementia subjects; bootstrap does not include retraining uncertainty.
The cohort has been used in earlier research; these are not external validation results.
All-zero circadian rows are preserved. No changes were made after observing scores.
