# Protocol comparison — R (reproduction) / G (subject-independent) / N (nested CV)

Written before any run. Every difference has a stated reason; a cell that is identical across the
three columns is shared code (`src/cheon/`).

| Component | Reproduction (R) | Subject-independent (G) | Nested CV (N) | Reason for difference |
|---|---|---|---|---|
| Cohort | AI-Hub Training + Validation pooled: 174 subjects, 12,183 daily records (A02, A03) | same | same | — |
| Label | CN = 0 vs MCI + Dem = 1 (7,737 / 4,446 records) | same | same | — |
| Raw features | 24 activity + 27 sleep scalars, `sleep_bedtime_*` as clock hours (A06) | same | same | — |
| Derived features | sleep_time (A07), 12 MET statistics (A08), 4 activity-class counts, 4 hypnogram counts (A09) → 72 (Figure 2) | same | same | — |
| Missing values | none exist in the 72 features (A13); no imputer | same | same | — |
| Scaling | none (A13) | same | same | — |
| Base-model comparison (Table 1) | 7 algorithms, library defaults, all 5 repeats | same models, subject folds, all 5 repeats | not run | N is about selection isolation; the base-model table is a fidelity check for R and a leakage diagnostic for G |
| Feature ranking (SHAP) | default LightGBM fit + explained on **all records** (A15) | fit + explained on the **training fold** (A17) | fit + explained on the **inner-training** part for selection, and on the **outer-training** set for the final outer model (A19) | Leakage prevention: the ranking is a learned step and may not see evaluation records |
| Feature-count selection | forward path k = 1..72 on the 5 outer folds, k* = argmax (paper: 40) | **no search**; k fixed at the paper's 40 (diagnostic curve saved, not used) | forward path on the inner folds, k* = argmax of inner-mean AUC, chosen per outer fold | G cannot search k without an inner loop; N adds exactly that loop |
| Hyperparameters | Table 2 fixed (A18); also default | Table 2 fixed (same); also default | selected per outer fold on the inner folds from the 16-point paper-anchored grid (A20); the fixed-Table-2 variant (N-A) and the default variant are reported for the same outer folds | N isolates hyperparameter selection from the outer test subjects |
| Outer split | `StratifiedKFold(5, shuffle=True)` on **records**; subjects appear on both sides (overlap count logged as a diagnostic) | `StratifiedGroupKFold(5, shuffle=True)`, group = subject; overlap asserted = 0 | same as G; overlap asserted = 0 | The single change that defines G |
| Inner split | N/A | N/A | `StratifiedGroupKFold(5)` on the outer-training subjects; inner overlap asserted = 0; outer-test subjects asserted absent from every inner set | Defines N |
| Model | LightGBM (scikit-learn API), `random_state = seed`, 8 threads | same | same | — |
| Threshold | model default (0.5 on probability) (A11) | same | same (not tuned) | — |
| Metrics | Accuracy, ROC-AUC, Precision (macro), Recall (macro), F1-macro + sensitivity, specificity, balanced accuracy, PR-AUC | same | same | — |
| Aggregation | per-repeat fold mean → across-repeat mean ± SD, 95 % CI (A21) | same | same | — |
| Seeds / repeats | 42–46 (5) | 42–46 (5) | 42–46 (5), one process per seed, merged (A26) | — |
| Reported stages | `baseline_lgbm_default_all72`, `fs_default_selected_k`, `fs_default_k40`, `final_tuned_selected_k`, `final_tuned_k40`, `final_tuned_all72`, `table1_*` | `baseline_lgbm_default_all72`, `fs_default_k40`, `final_tuned_k40` (**primary**), `final_tuned_all72`, `table1_*` | `nested_selected_k_selected_params` (**primary**), `nested_selected_k_paper_params` (N-A), `nested_selected_k_default_params` | Stage names are shared so R↔G pairs line up exactly (`baseline_lgbm_default_all72`, `fs_default_k40`, `final_tuned_k40`, `final_tuned_all72`) |

## Primary comparison lines

- **R vs paper**: `baseline_lgbm_default_all72` ↔ Table 1 (0.9010); `fs_default_selected_k` / `fs_default_k40` ↔ 0.9037; `final_tuned_selected_k` / `final_tuned_k40` ↔ 0.9492.
- **R → G, splitter-only pairs** (no learned step at all): `baseline_lgbm_default_all72` and `final_tuned_all72`.
- **R → G, recipe pairs** (learned ranking moves inside the training fold): `fs_default_k40`, `final_tuned_k40` (G primary).
- **G → N**: `final_tuned_k40` (G) ↔ `nested_selected_k_paper_params` (N-A, same hyperparameters, k chosen inside) ↔ `nested_selected_k_selected_params` (N primary, k and hyperparameters chosen inside).

## Check before running

Every row above has either identical cells or a reason in the last column. No unexplained difference remains.
