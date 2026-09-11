# Comparison with the existing Cheon-related implementations in this repository

Produced **after** the three experiments (R, G, N) were implemented, tested and launched, by six
read-only audit agents that were given `PAPER_PROTOCOL.md` and `ASSUMPTIONS.md` as the reference and
were forbidden to open `Data/`, modify files, or run code. Nothing in any other folder was changed.

Five implementations turned out to be Cheon reproductions; one candidate is not:

| Tag | Path | What it is |
|---|---|---|
| **T-R** | `Taehyun/Paper_Reproduction/Cheon_LGBM.py` | 3-step script: default LightGBM → all-data SHAP top-40 → Table-2 params; 5×5 record CV; stdout only |
| **T-N** | `Taehyun/Paper_Reproduction_Nested/Cheon_LGBM.py` | 4 scenarios (record/subject × fixed/Optuna), 5×5 repeats; stdout + PNG only |
| **S-1** | `SangHyo/Reproduction/XAI_Paper_Reproduction/` | 6-script pipeline: 7 baselines → OOF-SHAP forward selection → Table-2 final + SHAP/DRS → report → audit |
| **S-2** | `SangHyo/Reproduction/XAI_Paper_Reproduction2/` | Single "paper-exact" Colab notebook, same stages, with a self-audit and a forced top-40 |
| **S-B** | `SangHyo/Binary/Binary_PaperLGBM_NoMMSE/` | Deliberately *not* a reproduction: the paper's features evaluated at **subject** level + a leakage diagnostic |
| — | `Taehyun/xai/`, `Taehyun/previous/xai/` | Not a Cheon reproduction: a generic post-hoc SHAP wrapper (`ShapAnalyzer`) for the team's 3-class models; reads no data, trains no model, has no CV. Byte-identical copies |

## Master comparison

| Item | New implementation (this directory) | T-R | T-N | S-1 | S-2 | S-B |
|---|---|---|---|---|---|---|
| **Cohort** | Train + Validation pooled, **asserted** 174 subjects | pooled, no count asserted | pooled, README claims 176 subjects | pooled, 174 verified | pooled, asserted 174 | **Training only** (141) for CV; 33 as hold-out |
| **Record count** | **12,183 asserted** (row-wise pairing) | unknown (inner wake-date join + listwise deletion) | ≈12.1k (inner wake-date join) | 12,183 via **left** join (33 rows have no sleep) | 12,183 via **left** join (same) | 9,673 Training days → **141 subject rows** |
| **Record construction** | row-wise pairing of the two tables (only construction reproducing 12,183) | wake-date inner join, longest-sleep dedup | wake-date inner join, longest-sleep dedup | wake-date left join, longest-sleep dedup | wake-date left join, longest-sleep dedup | wake-date inner join + interval filters |
| **Missing values** | none exist in the 72 features (verified); no imputer | complete-case deletion over ~107 columns, whole dataset | **median imputation fit on the whole pooled dataset** before CV | median imputer inside each fold's pipeline | median imputer inside each fold's pipeline | median imputation fit on the training fold |
| **Features** | **exactly the 72 of Figure 2** | ≈107; adds 20 statistics of the **5-min HR and RMSSD logs the paper removed**; drops both bedtime clocks and sleep_time; no kurtosis/skew/autocorr | same family, ≈ 57 derived + all numeric raw; same HR/RMSSD additions and same omissions | 86; 68 of the paper's 72 (no kurtosis, skewness, autocorrelation, quantile_50) + 18 extra | 86, same as S-1 | 69; 67 verbatim + 2 clock features; no sleep_time, sleep_midpoint_time, sleep_period_id; + 69 subject-level `std` |
| **Feature selection** | SHAP forward path k = 1…72; scope per arm (all records / train fold / inner folds) | **no path**: k = 40 hard-coded, ranking from a model fit on **all** records | ranking from fold models, path with a 50-tree model, **record-level CV even inside subject scenarios** | global OOF-SHAP ranking, path k ≤ 80 scored on the **same folds**, k* = 28 but final forced to 40 | same, ranking made with the **tuned** model, k* = 39 overridden to 40 | **none** (no SHAP, no selection) |
| **Split** | R record-level; **G/N subject-level with overlap asserted 0**; N has a subject-aware inner loop | record-level, 5 seeds | record: S1/S2; subject: S3/S4 — but the **inner Optuna CV is record-level** in S4 | record-level, one seed (a `--grouped` flag exists, never run) | record-level, one seed (`groups` passed but ignored) | subject-level by construction (one row per subject) |
| **Model** | LightGBM + the 6 other Table-1 models (defaults) | LightGBM only | LightGBM only | all 7, with scaler/imputer pipelines | all 7, same | LightGBM only |
| **Hyperparameters** | defaults and Table 2 as constants (R, G); paper-anchored 16-point grid selected inside inner folds (N) | Table 2 constants | Table 2 (S1/S3); **Optuna 10 trials, unseeded sampler**, tuned at 150 trees then refit at 500 (S2/S4) | Table 2 constants (optional 72-point grid, not run) | Table 2 constants | **custom** (400 trees, 15 leaves, lr 0.03, `class_weight='balanced'`); Table 2 only in the leaky diagnostic |
| **Statistics** | per-repeat fold mean → mean ± SD, 95 % t-CI; pooled-fold and pooled-OOF as sensitivity | 25 pooled fold scores, mean ± 1.96·SD/√25 (invalid CI) | 25 pooled fold scores, t-CI with n = 25 (invalid) | pooled OOF, single seed, no CI | pooled OOF, single seed, no CI | one AUC on repeat-averaged subject OOF, no CI |
| **Metrics** | Accuracy, ROC-AUC, macro P/R, F1-macro + auxiliaries | P/R/F1 **binary-averaged** (not comparable to the paper's F1-macro) | P/R/F1 binary-averaged | macro + positive-class | macro + positive-class | AUC only (+ accuracy on the hold-out) |
| **Artifacts** | JSON/CSV/log/manifest/environment per run, hashed subject ids | **none** (stdout only) | **none** (stdout + PNG) | full outputs + audit JSON | full outputs + audit JSON | FINAL_REPORT.json + hashed OOF |
| **Privacy** | subject ids hashed at load, never written | ids not written (no artifacts) | ids not written (no artifacts) | **raw e-mail written to output CSVs** | **raw e-mail written to output CSVs** | hashed |

## Reported numbers, and how they compare with this run

Only S-1, S-2 and S-B have saved results. T-R and T-N report no numbers anywhere in the repository.

| Stage | Paper | This run (R, 5 seeds) | S-1 | S-2 |
|---|---|---|---|---|
| LightGBM, all features, defaults | 0.9010 | **0.9009 ± 0.0021** | 0.9004 | 0.8995 |
| Forward selection at k = 40 | 0.9037 | **0.9031 ± 0.0010** | 0.9000 | 0.9026 |
| Forward selection at own best k | 0.9037 (k = 40) | **0.9050** (k* = 33–47) | 0.9038 (k* = 28) | 0.9053 (k* = 39) |
| Table-2 params, final | 0.9492 | **0.9500 ± 0.0006** | 0.9478 | 0.9479 |
| Dementia risk score, CN / impaired mean | 7.59 / 15.71 | not computed | 5.23 / 20.80 (in-sample SHAP) | 7.42 / 15.62 (OOF SHAP) |

All three land within ≈0.002 of the paper at every stage even though they use different feature sets
(72 vs 86 vs 86), different record constructions and different aggregation rules. That insensitivity is
itself evidence about what drives the number: with record-level folds the model can identify the subject
from any sufficiently rich daily fingerprint, so the exact feature list barely matters.

Subject-level results, for the same reason, agree across implementations:

| Honest estimate | This run | S-B |
|---|---|---|
| day-level, subject-grouped folds | — | 0.4811 (`day_groupkfold_roc_auc_honest`) |
| subject-level prediction | 0.5062 ± 0.0168 (G, day-level model, subject folds) | 0.5214 (subject-aggregated features, subject folds) / 0.4582 (subject mean of grouped day scores) |
| leaky day-level reference | 0.9500 (R) | 0.9526 (`day_random_kfold_roc_auc_LEAKY`) |

## The manuscript's 0.9033 and 0.4562

Neither number occurs anywhere in this repository (a repo-wide search over `.md`, `.py`, `.json`, `.txt`
files found only unrelated coincidences: a Cliff's delta and a YDF prediction value). The manuscript is
kept outside this development repository, so its numbers could not be traced to a stored artifact. The
nearest saved values are S-1's 0.9004 / S-2's 0.8995 (Table-1 stage) and S-B's 0.4582 / 0.4811 (honest
subject-level estimates). Our own run neither targeted nor reproduced 0.9033 / 0.4562; see
`FINAL_REPORT.md` §7b for what this implies for the manuscript.

## Findings about the existing implementations

Ordered by how much they affect a published claim. Each was reported by a reader agent with file:line
evidence and re-checked by an independent verifier (verdicts in the next section).

1. **Feature-selection leakage in every reproduction (T-R, T-N, S-1, S-2).** The SHAP ranking that
   defines the "top 40" is computed once from all 12,183 records (T-R, S-1, S-2 pooled-OOF) and the
   forward path is then scored on the very same folds, with the same seed. Our R arm reproduces this
   deliberately (A15) and quantifies it; G and N remove it.
2. **T-N's subject-level scenarios are not leakage-free despite their name.** The inner Optuna CV is
   record-level `StratifiedKFold(3)`, and the SHAP/forward-selection CV inside them is record-level too,
   so the same subject appears on both sides of the inner split; median imputation is additionally fit on
   the whole pooled dataset before any split.
3. **T-R and T-N use the 5-minute heart-rate and RMSSD logs** (20 statistics), which the paper explicitly
   removed as un-imputable (p.164 §3.1), while dropping the two bedtime clock features and `sleep_time`
   that the paper adds. Their "top 40" therefore cannot be the paper's, and three of Figure 6's top-20
   features cannot appear in it.
4. **S-1 and S-2 lack four of the paper's 12 MET statistics** (kurtosis, skewness, autocorrelation,
   quantile_50) and add 18 columns of their own; two of Figure 6's top-20 features (kurtosis, rank 10;
   autocorrelation, rank 20) cannot appear. Both also define `sleep_time` from raw timestamps in seconds,
   which is an exact duplicate of `sleep_duration` (OOF mean |SHAP| = 0.0), so the paper's added
   sleep-time variable is effectively absent.
5. **Invalid confidence intervals (T-R, T-N).** 25 correlated fold scores from 5 repeats of the same data
   are treated as independent (z = 1.96 or t with n = 25) and the SD uses ddof = 0. Our A21 rule exists
   precisely to avoid this.
6. **Non-comparable metrics (T-R, T-N).** Precision/Recall/F1 use sklearn's binary averaging on class 1
   but are printed against the paper's F1-**macro** column.
7. **Raw subject e-mail addresses are written to output CSVs by S-1 and S-2**
   (`outputs/data/daily_binary_lifelog.csv`, `outputs/final/dementia_risk_scores.csv`). These files are
   `.gitignore`d, so they are local only, but this is a privacy issue worth fixing in those folders. We
   did not modify them.
8. **S-B's README promises ≈0.71 subject-level AUC; its own saved run gives 0.5214.** The README was
   never reconciled with the result. Our G arm (0.506) agrees with the run, not the README.
9. **S-2's own audit records `passed=false`** (its best k is 39, not 40) yet the surrounding documents
   describe the run as paper-exact; the audit checks feature *count*, not feature *identity*.
10. **Neither Taehyun script writes any artifact**, and `Taehyun/Paper_Reproduction/Cheon_LGBM.py` cannot
    run in this checkout at all (its `Taehyun/data/` copies of the activity/sleep CSVs are absent), so no
    number can currently be attributed to either.
11. **S-1 contains a latent `NameError`** (`Pipeline` referenced at module scope in
    `positive_probability`, imported only inside other functions) and a stale default raw-data path.

None of the existing implementations is wrong to have been written: S-B is explicitly a leakage
diagnostic, and S-1/S-2 reproduce the paper's headline numbers closely. The issue is that none of them
can support a *subject-independent* or *nested* claim, which is what R/G/N here were built to provide.

## Adversarial verification of the findings above

Each reader's claims were re-checked by an independent agent that re-opened the cited lines and was
instructed to refute wherever possible and to default to "unclear" rather than "confirmed".

| | count |
|---|---|
| claims checked | 122 |
| confirmed | 119 |
| unclear | 3 |
| refuted | 0 |

The three unclear verdicts are all cases where the *code* was confirmed but the *consequence* was not
determinable without running anything: the shap output layout for a binary tree model in the `xai`
package (version-dependent, unpinned); whether S-B's below-chance day-level diagnostic is caused by the
Table-2 configuration (the verifier notes S-B's own regularised model also scored ≈0.52, so
under-regularisation is not established); and whether the fourth constant column dropped by S-2 is
`sleep_is_longest` (checking it would require opening `Data/`, which the agents were forbidden to do).

The verifiers also surfaced items the readers had missed. The ones that matter here:

- **Independent corroboration of our assumption A11.** S-1's own output records Decision-Tree
  ROC-AUC = 0.6707 = recall_macro, the same identity we used in the paper's Table 1 (0.6806 = 0.6806) to
  infer that Cheon et al. macro-average recall. Two independent implementations show the signature.
- **T-N's Optuna search never leaves its random start.** `n_trials=10` with the default TPE sampler
  (`n_startup_trials=10`) means all ten trials are random samples; the "nested tuning" is a 10-point
  random search, and the sampler is unseeded.
- **S-1's shipped audit JSON records absolute paths from a different project tree** (a machine-learning
  assignment folder), so the provenance of its published numbers cannot be established from this
  repository.
- **S-1's Table-1 agreement is not like-for-like**: 0.9004 vs the paper's 0.9010 is computed on its own
  86-feature universe, not the paper's 72.
- **S-2's missing-data footprint is exactly 33 of 12,183 rows** (0.271 %), each with all 39 sleep columns
  empty — the price of its left join.
- The identifier exposure in S-1/S-2 is confined to gitignored local outputs; nothing is tracked by git.
- **S-B hashes subject ids without a salt** (plain SHA-256 of the e-mail), which is reversible by
  dictionary attack for a known address list; ours is salted.
- **S-B's hold-out accuracy (0.6364) is below its own all-CN baseline (0.7879)**, which its report states
  but never compares.
- `Taehyun/Binary_LGBM_RF.py:155` does use the `xai` SHAP wrapper inside a Cheon-shaped pipeline
  (SHAP computed on training rows, then forward selection under record-level `StratifiedKFold`), so the
  wrapper is adjacent to this work even though it is not itself a reproduction.

Raw reader and verifier outputs: `audit_raw/` and `audit_raw2/`; the workflow script is
`audit_workflow.js`.
