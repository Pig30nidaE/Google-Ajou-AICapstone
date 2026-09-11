# Cheon 2025 Validation Report

Cheon H., Park H., Lee B., Hong S., Joung J. (2025), *A Study on Dementia Risk Assessment Using Lifelog
Data with Explainable Artificial Intelligence*, JKIIE 51(2):161–170, re-implemented from the PDF only and
run under three validation designs on one shared pipeline (R → G → N). Executed 2026-09-09
(08:27–11:50 UTC), git commit `fecf4807`, Python 3.12.8, LightGBM 4.7.0, shap 0.52.0, scikit-learn 1.9.0.
Primary statistic everywhere: per-repeat mean over 5 folds, then mean ± SD [95 % t-CI] over 5 repeats
(seeds 42–46). Full tables: `RESULT_COMPARISON.md` / `results/summary_tables.md`.

## 1. Paper protocol

Extracted in `PAPER_PROTOCOL.md` with page references (journal pages 161–170). In short: AI-Hub
"치매 고위험군 웨어러블 라이프로그", 174 subjects (≥ 55 y; CN incl. aAD / MCI / Dem), 35–122 days each, one row
per day, 12,183 daily records; label CN = 0 (7,737 records) vs MCI + Dem = 1 (4,446), ratio 1.74 : 1;
72 lifelog features (Figure 2: 24 activity scalars, 27 sleep scalars, sleep_time, 12 statistics of the
1-min MET log, 4 activity-class counts, 4 hypnogram counts); no MMSE/CIST input; 5-fold cross-validation
with fold-mean aggregation; seven candidate algorithms, LightGBM chosen by ROC-AUC (Table 1: 0.9010);
forward selection ordered by SHAP importance, best at 40 features (0.9037); grid search →
min_data_in_leaf 41, num_leaves 330, n_estimators 1000, learning_rate 0.08 (Table 2) → 0.9492.
Not reported (recorded as NOT REPORTED and decided in `ASSUMPTIONS.md` A01–A26): which AI-Hub partitions,
activity–sleep pairing, seeds, stratification/shuffle, base-model hyperparameters, SHAP scope/explainer,
missing-value handling, scaling, search ranges for three of the four tuned parameters, threshold,
precision/recall averaging, software versions.

The paper matches the local data exactly once Training and Validation are pooled and the activity and sleep
tables are paired row by row: 174 subjects (CN 111 / MCI 51 / Dem 12), 12,183 records, 7,737 vs 4,446.

## 2. Reproduction fidelity

Paper:
Table 1 LightGBM ROC-AUC 0.9010 (Accuracy 0.8262, Precision 0.8276, Recall 0.7904, F1-macro 0.8025);
forward selection 40 features → 0.9037; tuned final → 0.9492.

Reproduction (R, record-level `StratifiedKFold(5, shuffle)`, 5 seeds):

| Stage | Paper | R primary (mean ± SD [95 % CI]) | R seed 42 | \|diff\| |
|---|---|---|---|---|
| LightGBM, all 72, defaults (Table 1) | 0.9010 | 0.9009 ± 0.0021 [0.8983, 0.9036] | 0.9001 | 0.0001 |
| forward selection, best k (paper 40) | 0.9037 | 0.9050 ± 0.0006 at k* ∈ {47, 35, 44, 41, 33} | 0.9043 (k* = 47) | 0.0013 |
| forward selection at the paper's k = 40 | 0.9037 | 0.9031 ± 0.0010 [0.9019, 0.9044] | 0.9022 | 0.0006 |
| Table-2 params at k = 40 (final model) | 0.9492 | 0.9500 ± 0.0006 [0.9493, 0.9507] | 0.9502 | 0.0008 |
| Table-2 params at our k* | 0.9492 | 0.9497 ± 0.0006 [0.9490, 0.9504] | 0.9504 | 0.0005 |

All five Table-1 metrics of the LightGBM row are reproduced within 0.002 (Accuracy 0.8277, Precision
0.8297, Recall 0.7918, F1-macro 0.8041). All seven base models are within 0.003 ROC-AUC of Table 1 except
logistic regression (0.6187 vs 0.6067). The pooled forward-selection curve peaks at k = 41 (0.9042) with
a plateau of 0.9009–0.9042 for k ≥ 30, i.e. Figure 4. The seed-42 SHAP top-20 shares 14 features with
Figure 6, the top-40 contains all 20, and sleep_breath_average is the top feature in every seed as in the
paper. No seed, preprocessing, feature or parameter was adjusted after seeing results.

Conclusion: under the paper's own validation design the paper is reproduced essentially exactly.

## 3. Subject-independent evaluation

Reproduction (R, final paper configuration: top-40 by SHAP, Table-2 hyperparameters):
0.9500 ± 0.0006 [0.9493, 0.9507]

Subject-independent (G, identical configuration, `StratifiedGroupKFold(5, shuffle)` by subject, SHAP
ranking fit on the training fold, k = 40 fixed, Table-2 hyperparameters):
0.5062 ± 0.0168 [0.4854, 0.5271]

Delta (G − R): −0.4438

Splitter-only pair (no learned selection at all; default LightGBM on all 72 features):
R 0.9009 [0.8983, 0.9036] → G 0.5088 [0.4836, 0.5340]; delta −0.3921.
Second splitter-only pair (Table-2 params on all 72): R 0.9490 → G 0.5058; delta −0.4432.

Every G stage's CI contains 0.5 (t-test of the repeat means vs 0.5: p = 0.45 final, 0.39 baseline); per-fold
AUC 0.38–0.60; G's Accuracy 0.57 is below the all-CN rate (7,737 / 12,183 = 0.635). The seven base models
under G all fall to 0.50–0.56. Within G, selection and tuning change nothing (paired vs baseline:
+0.004, −0.003, −0.003; CIs include 0).

## 4. Nested CV evaluation

Subject-independent fixed-paper-settings (G, k = 40, Table-2 params):
0.5062 ± 0.0168 [0.4854, 0.5271]

Nested (N, primary: k and hyperparameters selected on inner `StratifiedGroupKFold(5)` folds of the
outer-training subjects, 16-point paper-anchored grid A20):
0.4851 ± 0.0268 [0.4518, 0.5184]

Nested, fixed Table-2 params with inner-selected k (N-A):
0.4907 ± 0.0232 [0.4618, 0.5195]

G and N share identical outer folds, so the paired difference over the 25 outer folds is
N − G = −0.021 [−0.052, +0.010] (primary; N wins 9/25) and −0.016 [−0.043, +0.012] (N-A).
Inner-CV AUC at the selected configuration averages 0.536 vs 0.485 outer (optimism +0.051); k* ranges
1–70 (median 5); the 1000-tree Table-2 schedule is selected in 3 of 25 folds. The selection procedure
of the paper, applied honestly, selects noise.

Ladder: R 0.950 → G 0.506 → N 0.485 (final configuration); R 0.901 → G 0.509 (no selection).

## 5. What changed between conditions

| | R | G | N |
|---|---|---|---|
| Question | Does the paper reproduce? | Same recipe with evaluation subjects excluded from training | … and excluded from feature-count and hyperparameter selection too |
| Split | `StratifiedKFold(5, shuffle)` on daily records; every fold shares all 174 subjects between train and test | `StratifiedGroupKFold(5, shuffle)` by subject; overlap = 0 | outer as G; inner `StratifiedGroupKFold(5)` on the outer-training subjects; overlap = 0 at both levels |
| SHAP ranking | all 12,183 records (paper-faithful, A15) | training fold only | inner-training (selection) / outer-training (final model) |
| Feature count k | argmax of the record-level CV curve (paper: 40) | fixed 40 (paper's outcome) | argmax of the inner-fold curve, per outer fold |
| Hyperparameters | Table 2 fixed (+ defaults) | Table 2 fixed (+ defaults) | chosen on inner folds from the 16-point paper-anchored grid (A20); Table-2-fixed and default variants reported on the same outer folds |
| Everything else | identical: cohort, label, 72 features, no imputation/scaling, LightGBM, threshold, metrics, seeds 42–46, 5 folds, aggregation | identical | identical |

The only R→G change that is *not* the splitter is the scope of the learned SHAP ranking (all records →
training fold), which leakage prevention requires; it is stated, not hidden (`RESULT_COMPARISON.md`).
The stage pairs `baseline_lgbm_default_all72` and `final_tuned_all72` involve no learned selection and are
therefore pure splitter-only comparisons.

## 6. Leakage audit

Automated tests (`tests/`, run with `.venv/bin/python -m pytest tests -q`):

| Test | What it checks | Result |
|---|---|---|
| `test_subject_overlap.py` | G/N outer train∩test subjects = ∅ on synthetic folds, on a synthetic end-to-end G and N run, and on the real `split_summary.csv` / `fold_metrics.csv` of the full runs; N inner train∩validation = ∅; outer-test subjects absent from every inner set; the disjointness assertion raises on overlap; R's record split *does* share subjects (documented leak) | pass |
| `test_preprocessing_scope.py` | no imputer/scaler class or `fillna` anywhere in `src/cheon`; feature construction is row-wise (features of a 200-row subset equal the full-table rows); the 72 features contain 0 NaN/inf on the real data, so imputation is vacuous | pass |
| `test_feature_selection_scope.py` | a spy around `rank_features` proves G ranks on exactly the training-fold rows, N's inner rankings use only outer-training rows and never an outer-test subject, R ranks on all rows by design; N's selected k and hyperparameters equal the argmax of the saved inner-fold tables | pass |
| `test_forbidden_features.py` | the 72 names are all `activity_*`/`sleep_*`; MMSE/diagnosis/identifier names are rejected; the loader never opens a `CognitiveFunction`/`mmse` file (spy on `pandas.read_csv`); no raw identifier or diagnosis column survives loading | pass |
| `test_features.py` | series parsing with trailing separator; KST clock conversion | pass |

Runtime assertions inside the pipeline: `assert_subject_disjoint` on every G/N fold and every N inner
fold; an explicit check that no outer-test subject appears in any inner set; `leakage_assertions.json` and
the `subject_overlap` column of every fold row record the outcome (0 in G/N; 174 in R).

MMSE was checked against the PDF: the paper never uses MMSE or CIST scores as inputs (CIST is only the
recommended follow-up screening), so MMSE files are forbidden inputs here and are never opened.

## 7. Reproduction differences from the paper

Things the new implementation could not take from the paper and had to decide (full list with reasons in
`ASSUMPTIONS.md`): pooling of the AI-Hub Training/Validation partitions (implied by the counts, A02);
row-wise activity–sleep pairing (the only construction giving 12,183 records, A03); clock conversion and
the sleep_time definition (A06–A07); the exact MET-statistic definitions (A08); stratified + shuffled folds
and the seeds (A10); macro averaging of precision/recall and the 0.5 threshold (A11); library defaults for
the seven base models and no scaling (A12–A13); SHAP explainer and its dataset-wide scope in R (A14–A15);
default LightGBM along the forward path and the argmax stopping rule (A16); the grid search was not re-run
(A18); software versions (A23). Repeats (5 seeds) were added on top of the paper's single 5-fold run (A24).

Observed during implementation (not assumptions): `median` and `quantile_50` of the MET series are the same
column; `sleep_time` computed from full timestamps would equal `sleep_duration/3600` exactly; with
min_data_in_leaf = 41 on ≈9.7k training rows the tuned trees reach only ≈180–200 leaves, so the paper's
num_leaves refinement 300 → 320 → 330 (p.165) cannot have changed the model.

## 7b. Comparison with the existing Cheon implementations in this repository

Audited after the fact by read-only agents (`EXISTING_CODE_COMPARISON.md`). Five implementations
reproduce Cheon: `Taehyun/Paper_Reproduction` (T-R), `Taehyun/Paper_Reproduction_Nested` (T-N),
`SangHyo/Reproduction/XAI_Paper_Reproduction` (S-1), `.../XAI_Paper_Reproduction2` (S-2) and, as an
explicit leakage diagnostic rather than a reproduction, `SangHyo/Binary/Binary_PaperLGBM_NoMMSE` (S-B).
`Taehyun/xai` is a generic SHAP wrapper, not a reproduction.

| Item | This implementation | Existing implementations |
|---|---|---|
| Cohort | pooled, 174 asserted | pooled (T-R, T-N, S-1, S-2; T-N's README claims 176); S-B uses Training only |
| Record count | 12,183 asserted, row-wise pairing | 12,183 via a **left** join with 33 sleep-less rows (S-1, S-2); unverified wake-date inner join (T-R, T-N); 9,673 days → 141 subject rows (S-B) |
| Missing values | none exist; no imputer | whole-dataset median imputation before CV (T-N); complete-case deletion (T-R); fold-internal median imputer (S-1, S-2, S-B) |
| Features | exactly the 72 of Figure 2 | ≈107 incl. the 5-min HR/RMSSD logs the paper removed, without the bedtime clocks or sleep_time (T-R, T-N); 86 with 4 of the paper's MET statistics missing and 18 extra (S-1, S-2); 69 + subject-level SDs (S-B) |
| Feature selection | forward path, scope per arm, k = 1…72 | k = 40 hard-coded from an all-data ranking (T-R); path scored on the same folds as the ranking (S-1, S-2), best k overridden to 40; record-level even inside subject scenarios (T-N); none (S-B) |
| Split | R record; G/N subject, overlap asserted 0; N subject-aware inner loop | record-level (T-R, S-1, S-2); T-N has subject scenarios but a **record-level inner CV**; S-B is subject-level |
| Model | 7 models; LightGBM primary | LightGBM only (T-R, T-N, S-B); all 7 (S-1, S-2) |
| Hyperparameters | defaults + Table 2; inner-selected grid in N | Table 2 constants (T-R, T-N-S1/S3, S-1, S-2); unseeded 10-trial Optuna tuned at 150 trees, refit at 500 (T-N-S2/S4); custom values (S-B) |
| Statistics | repeat-mean → mean ± SD, 95 % t-CI | 25 pooled fold scores with an invalid CI (T-R, T-N); pooled OOF, one seed, no CI (S-1, S-2); one AUC, no CI (S-B) |

Where results exist, they agree with ours: Table-1 stage 0.9004 (S-1) / 0.8995 (S-2) vs our 0.9009;
final stage 0.9478 / 0.9479 vs our 0.9500; leaky day-level reference 0.9526 (S-B) vs our R 0.9500; honest
subject-level 0.5214 / 0.4811 / 0.4582 (S-B) vs our G 0.506. The agreement across three quite different
feature sets supports the interpretation that the record-level number is a subject-identification
artefact rather than a property of a particular feature list.

Every claim below was re-checked by an independent verifier agent instructed to refute it: 119 of 122
claims confirmed, 3 unclear (consequences not determinable without running code), 0 refuted. One
verifier finding corroborates our own assumption A11: an existing implementation's Decision-Tree row
shows ROC-AUC = recall_macro (0.6707 = 0.6707), the same identity we used in the paper's Table 1 to infer
macro averaging.

Substantive findings about the existing code (all with file:line evidence in
`EXISTING_CODE_COMPARISON.md`): feature-selection leakage in all four reproductions; T-N's
"zero-leakage" scenarios are not leakage-free (record-level inner CV plus whole-dataset imputation);
T-R/T-N use the 5-minute heart-rate log the paper removed; S-1/S-2 are missing four of the paper's MET
statistics, and their `sleep_time` duplicates `sleep_duration` exactly; invalid confidence intervals and
binary-instead-of-macro metrics in T-R/T-N; S-1 and S-2 write raw subject e-mail addresses into local
output CSVs (a privacy issue in those folders, which were not modified); S-B's README claims ≈0.71 where
its own artifact records 0.5214. Neither manuscript number (0.9033, 0.4562) exists anywhere in this
repository, so they could not be attributed to a stored artifact.

## 8. Limitations

- R is faithful to what the paper *reports*; where the paper is silent, A01–A25 fill the gaps, so R is one
  admissible reading of the protocol, not the authors' code.
- 174 subjects (63 positive) is small: subject-level fold AUCs vary widely between folds and seeds; the
  repeat-level 95 % CIs are the honest uncertainty and are wide.
- N's hyperparameter candidate set is operational (16 paper-anchored configurations, A20), not the paper's
  (unreported) search space; the Table-2-fixed variant N-A is reported alongside for that reason.
- The AI-Hub Validation partition is pooled into the cohort (as the paper did), so no untouched external
  hold-out exists; all three arms are internal cross-validation estimates.
- No demographics exist in the release, so age/sex confounding cannot be examined.
- The team's folder guide (`SangHyo/AGENTS.md`) was read before implementation for repository conventions;
  it contains summary numbers of earlier team experiments (including a leaky-vs-honest LightGBM table) but
  no Cheon implementation code. No existing implementation was opened before the three experiments were
  complete and launched; their comparison (`EXISTING_CODE_COMPARISON.md`) was produced afterwards by
  read-only audit agents.
- Knowing that team folder existed, the expectation that subject-level performance would be near chance
  was not blind. It did not enter the code: R, G and N share one pipeline, the assumptions were fixed
  before the runs, and no seed, feature or parameter was changed after seeing a result.
- Absence of evidence is not evidence of absence: G and N show that *this* feature set, at day level,
  with 174 subjects, does not separate the classes. A larger cohort or different features could.

## 9. Files and commands

```
SangHyo/Reproduction/cheon_2025_validation/
  PAPER_PROTOCOL.md  ASSUMPTIONS.md  PROTOCOL_COMPARISON.md  RESULT_COMPARISON.md
  EXISTING_CODE_COMPARISON.md  FINAL_REPORT.md  README.md
  configs/{reproduction,subject_independent,nested_cv}.yaml
  src/cheon/{data,features,splits,model,selection,evaluation,manifest,pipeline}.py
  run.py  merge_runs.py  launch_all.sh  report_tables.py  tests/
  results/reproduction/         metrics.json fold_metrics.csv oof_predictions.csv split_summary.csv run_manifest.json
                                environment.txt stdout.log timing.json feature_selection.csv shap_importance.csv selected_features.json
  results/subject_independent/  + selected_features_by_fold.csv feature_selection_diagnostic.json
  results/nested_cv/            + outer_fold_metrics.csv inner_selection.csv selected_features_by_fold.csv
                                  selected_hyperparameters_by_fold.csv leakage_assertions.json  (merged from results/nested_cv_seed{42..46}/)
  results/summary_tables.md     (report_tables.py output)
  results/*_smoke/              wiring-check runs (1 seed, 1 fold, max_k=6) — never reported as performance
  results/launcher/             per-process stdout of the full runs + ALL_DONE timestamp
  audit_workflow.js, audit_raw/ the read-only audit of the existing implementations (§7b) and its raw findings
```

```bash
cd SangHyo/Reproduction/cheon_2025_validation
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python lightgbm shap scikit-learn pandas numpy scipy pyyaml pytest psutil
.venv/bin/python -m pytest tests -q
.venv/bin/python run.py --config configs/reproduction.yaml --smoke
./launch_all.sh            # R, G and five single-seed N processes, then merge_runs.py
.venv/bin/python report_tables.py > results/summary_tables.md
```

Note: the repository `.gitignore` excludes `results/`, `*.csv`, `*.txt` and `*.log`, so the result artifacts
exist locally only (repository convention: results are not shared through git). Nothing was committed or
pushed.
