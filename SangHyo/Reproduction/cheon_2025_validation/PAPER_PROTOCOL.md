# Cheon et al. 2025 Protocol

Source of truth: Cheon H., Park H., Lee B., Hong S., Joung J. (2025). *A Study on Dementia Risk
Assessment Using Lifelog Data with Explainable Artificial Intelligence* (설명가능 인공지능을 활용한
라이프로그 기반 치매 위험도 산정 방법에 관한 연구). Journal of the Korean Institute of Industrial
Engineers 51(2), 161–170. DOI 10.7232/JKIIE.2025.51.2.161.

PDF: `papers/Domain/설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도 산정 방법에 관한 연구.pdf`
(10 PDF pages; PDF page p = journal page 160 + p). Page numbers below are **journal pages**.

This document was written from the PDF only, before any existing reproduction code in this
repository was opened. Everything not backed by a page reference is marked NOT REPORTED and is
handled in `ASSUMPTIONS.md`.

---

## Cohort

| Item | Paper statement | Paper page / Section / Table-Figure |
|---|---|---|
| Dataset | AI-Hub “치매 고위험군 웨어러블 라이프로그” (Wearable lifelog of high-risk dementia group), https://www.aihub.or.kr/ | p.162 §1 (last para.); p.163 §3.1 |
| Device | Ring-type daily sleep/activity collector; sleep data (sleep time, sleep efficiency, sleep depth, …) and activity data (activity time, exercise time, recovery time, …) | p.163 §3.1 |
| Subjects | 174 persons, age ≥ 55, pathological diagnosis by a specialist: CN (normal cognition, including asymptomatic AD), MCI, Dementia; AD high-risk group prioritised | p.163 §3.1; p.168 §5 |
| Inclusion / exclusion criteria | Figure 1 (informed consent; no severe hearing/vision/language disability; no alcohol/substance disorder, stroke, CNS disease; excludes illiteracy, epilepsy/depressive/bipolar/schizophrenia/substance disorders, brain-related history, long-term psychiatric medication, disabilities hindering participation, severe head trauma with LOC, cancer within 3 years post-treatment, metal implants, non-consent) | p.163 Figure 1 |
| De-identification | Personal information removed | p.164 §3.1 |
| Days per subject | 35–122 days per person (≥ 1 month each) | p.164 §3.1 |
| Which AI-Hub partition (Training / Validation) | NOT REPORTED explicitly. Implied: the AI-Hub release has 141 Training + 33 Validation subjects = 174, and the paper's record total (7,737 + 4,446 = 12,183) equals the Training + Validation daily-row total (9,705 + 2,478 = 12,183). Both AI-Hub partitions were therefore pooled. | inferred from p.163 §3.1 (174) and p.164 §3.1 (record counts); verified against local data (see `ASSUMPTIONS.md` A02) |
| CN / MCI / Dem subject counts | NOT REPORTED at subject level (only 174 total) | — |

## Classification task

| Item | Paper statement | Page / Section |
|---|---|---|
| Classes | Two classes from the pathological diagnosis: normal-cognition group (CN) vs cognitive-impairment group (MCI, Dem) | p.164 §3.1 |
| Encoding | CN → 0, impaired (MCI + Dem) → 1 | p.164 §3.1 |
| Record counts | CN 7,737 records; impaired 4,446 records; ratio 1.74 : 1 | p.164 §3.1; p.169 §5 |
| Positive class for ROC | ROC curve for the cognitive-impairment class | p.165 §3.2(2) |
| MMSE / CIST used as input? | No. CIST is mentioned only as the recommended follow-up screening test for high-risk persons; MMSE is never mentioned. The independent variables are exclusively the lifelog features of Figure 2. | p.161 abstract; p.162 §1; p.164 Figure 2 |

## Raw files

The paper does not name individual files. It describes the input as the ring-collected daily sleep
data and daily activity data of the AI-Hub dataset (p.163 §3.1). The feature names in Figure 2
(p.164) are the raw AI-Hub column names (`activity_*`, `sleep_*`) plus derived names, so the daily
activity table and the daily sleep table are the raw inputs. The 5-minute heart-rate log was removed
(p.164 §3.1). Cognitive-test (MMSE) files are not used.

## Analysis unit

One row = one day of one subject (“하루 단위로 기록된 라이프로그 데이터”, p.164 §3.1). The
conclusion restates that the model was built on day-level, not person-level, data because of the
small number of subjects (p.169 §5). Total 12,183 daily records (p.164 §3.1).

## Preprocessing

All from p.164 §3.1 unless noted.

1. Removed “meaningless” data and converted the rest into a modelable form.
2. Removed columns whose values are all identical (example given: lifelog measurement start time
   and measurement end time).
3. Removed data for which missing-value handling was impossible (example given: 5-minute
   heart-beat log data).
4. Timestamp-format columns (sleep start time, sleep end time) converted to a real number
   corresponding to the 24-hour clock (“24시에 대응하는 실숫값”).
5. Added a sleep-time variable = sleep end time − sleep start time.
6. Annotated activity-log time series: extracted a variable for the count of each measurement value,
   and/or added descriptive statistics (standard deviation, variance, mean, quartiles).
7. Missing-value imputation method: NOT REPORTED (only the removal in item 3).
8. Outlier handling: NOT REPORTED.
9. Scaling / normalisation: NOT REPORTED.
10. Categorical encoding: only the label encoding (CN = 0, impaired = 1).
11. Training/Validation partition merge: NOT REPORTED (implied, see Cohort).
12. Activity–sleep date matching rule: NOT REPORTED.
13. Duplicate sleep-record handling: NOT REPORTED (`sleep_period_id` is kept as a feature, Figure 2).

## Feature engineering

Figure 2 (p.164) lists **72 independent variables**. Transcribed from the figure image:

Raw activity scalars (24): activity_average_met, activity_cal_active, activity_cal_total,
activity_daily_movement, activity_high, activity_inactive, activity_inactivity_alerts, activity_low,
activity_medium, activity_met_min_high, activity_met_min_inactive, activity_met_min_low,
activity_met_min_medium, activity_non_wear, activity_rest, activity_score,
activity_score_meet_daily_targets, activity_score_move_every_hour, activity_score_recovery_time,
activity_score_stay_active, activity_score_training_frequency, activity_score_training_volume,
activity_steps, activity_total.

Raw sleep scalars (27): sleep_awake, sleep_bedtime_end, sleep_bedtime_start, sleep_breath_average,
sleep_deep, sleep_duration, sleep_efficiency, sleep_hr_average, sleep_hr_lowest, sleep_light,
sleep_midpoint_at_delta, sleep_midpoint_time, sleep_onset_latency, sleep_period_id, sleep_rem,
sleep_restless, sleep_rmssd, sleep_score, sleep_score_alignment, sleep_score_deep,
sleep_score_disturbances, sleep_score_efficiency, sleep_score_latency, sleep_score_rem,
sleep_score_total, sleep_temperature_delta, sleep_total.

Derived (21):
- sleep_time — “Actual sleep duration” (the added end − start variable, §3.1).
- activity_met_1min_{std, variance, kurtosis, skewness, mean, median, min, max, autocorrelation,
  quantile_25, quantile_50, quantile_75} — statistics of the 1-minute MET log per day (12).
- activity_class_5min_count_{1,2,3,4} — number of rest / inactive / low-intensity /
  medium-intensity periods in the 5-minute activity-class log per day (4).
- sleep_hypnogram_5min_count_{1,2,3,4} — number of deep / light / REM / awake periods in the
  sleep-state log (4).

Total = 24 + 27 + 21 = 72. The forward-selection curve in Figure 4 (p.167) runs to ≈72 on the x-axis,
consistent with 72 candidate features.

Not among the 72 (therefore dropped by the paper): measurement start/end time (removed as
all-identical, §3.1), 5-minute heart-rate log (removed, §3.1), 5-minute RMSSD log, `sleep_is_longest`,
`sleep_temperature_deviation`, activity-class counts for classes 0 (non-wear) and 5 (high),
subject identifier. Figure 2 only; the paper gives no reason for the last three.

Definitions NOT REPORTED: autocorrelation lag; quantile interpolation; kurtosis convention
(Fisher/Pearson); whether `median` and `quantile_50` differ; units of sleep_time and of the
converted clock values.

## Feature selection

| Item | Paper statement | Page / Section |
|---|---|---|
| Method | Forward selection (전진 선택법) on the selected model | p.165 §3.2(2) |
| Ranking | SHAP Importance evaluated for all features; features added one at a time starting from the most important | p.165 §3.2(2) |
| Stopping rule | The feature set of the best-performing model among the constructed ones is chosen (global best over the path, Figure 4 shows the full curve to ≈72) | p.165 §3.2(2); p.167 Figure 4 |
| Result | Top 40 by SHAP importance → ROC-AUC 0.9037 (best) | p.167 §4.1; Figure 4 |
| SHAP explainer type, data used for SHAP, aggregation (e.g. mean \|SHAP\|), model hyperparameters during the FS path | NOT REPORTED | — |

## Model

| Item | Paper statement | Page / Section |
|---|---|---|
| Candidate models (7) | Logistic Regression, Decision Tree, K-Nearest Neighbor, Support Vector Machine, Multi-Layer Perceptron, Random Forest, LightGBM | p.165 §3.2(2) |
| Selection criterion | Highest ROC-AUC (positive class = impaired) | p.165 §3.2(2) |
| Selected model | LightGBM (Accuracy 82.62 %, ROC-AUC 0.9010, F1-macro 80.25 %) | p.167 §4.1; p.166 Table 1 |
| Hyperparameters of the 7 base models | NOT REPORTED (presumably defaults; nothing stated) | — |

## Hyperparameters

| Item | Paper statement | Page / Section |
|---|---|---|
| Tuning method | Grid search; a coarse range is searched, the best value by cross-validated ROC-AUC is found, then a narrower range around it is searched again; repeated | p.165 §3.2(3) |
| Reported search trajectory | `num_leaves`: 100–1000 step 100 → 300; then 220–380 step 20 → 320; then 310–330 step 10 → final | p.165 §3.2(3) |
| Final values (Table 2) | min_data_in_leaf = 41; num_leaves = 330; n_estimators = 1000; learning_rate = 0.08 | p.167 Table 2 |
| Search ranges for min_data_in_leaf, n_estimators, learning_rate | NOT REPORTED | — |
| Tuning CV | “교차 검증을 통해” (through cross-validation); fold count for tuning not separately stated (5-fold is the only CV described) | p.165 §3.2(3) |
| Other LightGBM parameters | NOT REPORTED | — |

## Validation design

| Item | Paper statement | Page / Section |
|---|---|---|
| Scheme | K-fold cross-validation, K = 5; dataset split into 5 folds, each fold used once as validation, the rest as training; models trained on the training set and evaluated on the validation set | p.164–165 §3.2(1); p.165 Figure 3 |
| Aggregation | Performance = (1/5) Σ Performance_i (mean of the 5 fold scores) | p.165 Figure 3 |
| Unit of splitting | Records (days). The paper does not mention subject-level grouping; it explicitly states the model is day-level (p.169 §5). | p.164 §3.1; p.169 §5 |
| Stratification | NOT REPORTED |
| Shuffling / random seed | NOT REPORTED |
| Repeats | None mentioned (single 5-fold run) | p.165 Figure 3 |
| Separate hold-out test set | None | — |

## Metrics

Table 1 (p.166) columns: Accuracy, ROC-AUC, Precision, Recall, F1-macro. ROC-AUC is the primary
metric (p.165 §3.2(2)). Averaging of Precision/Recall is NOT REPORTED; F1 is explicitly macro.
(Observation: for the Decision-Tree row, ROC-AUC = Recall = 0.6806, which is what macro-averaged
recall gives for a hard-prediction classifier; see `ASSUMPTIONS.md` A11.) Decision threshold: NOT
REPORTED. Sensitivity / specificity: not reported in the paper.

## Reported results

### Table 1 — 7 base models, 5-fold CV, all features (p.166)

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-macro |
|---|---|---|---|---|---|
| LightGBM | 0.8262 | 0.9010 | 0.8276 | 0.7904 | 0.8025 |
| Random forest | 0.8055 | 0.8835 | 0.8325 | 0.7491 | 0.7659 |
| Decision tree | 0.7041 | 0.6806 | 0.6808 | 0.6806 | 0.6807 |
| K-Nearest Neighbor | 0.6572 | 0.6595 | 0.6229 | 0.6111 | 0.6136 |
| Multi-Layer Perceptron | 0.5953 | 0.6348 | 0.6188 | 0.5634 | 0.5142 |
| Support vector machine | 0.6393 | 0.6249 | 0.6609 | 0.5083 | 0.4114 |
| Logistic regression | 0.6457 | 0.6067 | 0.6113 | 0.5331 | 0.4830 |

### Feature selection (p.167 §4.1, Figure 4)

LightGBM + forward selection by SHAP importance: best ROC-AUC **0.9037** with the **top 40**
features. Figure 4: curve rises steeply to ~0.90 by ~25 features, plateaus ~0.900–0.904 thereafter;
dashed vertical line at 40, dashed horizontal at ≈0.9037.

### Hyperparameter tuning (p.167 §4.1, Table 2, Figure 5)

Tuned LightGBM (Table 2 values) on the selected features: ROC-AUC **0.9492**. Figure 5 shows the
ROC curve of the final model.

### SHAP summary (p.167 Figure 6; p.167–168 §4.2)

Top-20 features in Figure 6, in order: sleep_breath_average, sleep_hr_average, sleep_hr_lowest,
sleep_bedtime_end, sleep_rmssd, sleep_score_rem, sleep_restless, sleep_midpoint_at_delta,
activity_class_5min_count_3, activity_met_1min_kurtosis, activity_cal_total,
activity_score_meet_daily_targets, activity_score, sleep_score_deep, activity_score_training_volume,
activity_met_1min_max, activity_class_5min_count_4, activity_class_5min_count_2,
activity_daily_movement, activity_met_1min_autocorrelation. Text: sleep_breath_average is most
important (high value → positive SHAP); activity_class_5min_count_3 is the most important activity
feature.

### Dementia risk score (p.166 Eq. 2; p.168 §4.2–4.3, Figure 7)

Score = sum of positive SHAP values per daily record. CN: min 1.06, max 24.99, mean 7.59; impaired:
min 1.79, max 31.28, mean 15.71. One-sample one-sided t-test of impaired scores against the CN mean:
p → 0 (p < 0.05). (Not part of the three experiments here; recorded for completeness.)

## Items explicitly stated by the paper

- Dataset name and source; 174 subjects; ≥ 55 y; CN(+aAD)/MCI/Dem; 35–122 days per subject (p.163–164).
- Day-level analysis unit (p.164, p.169).
- Binary label CN = 0 vs MCI + Dem = 1; 7,737 vs 4,446 records; ratio 1.74 : 1 (p.164).
- Removal of all-identical columns and of the 5-minute heart-rate log; clock → real number; sleep
  time = end − start; count and descriptive-statistic features from time-series logs (p.164).
- The 72-feature list (Figure 2, p.164).
- 5-fold cross-validation with fold-mean aggregation (p.164–165, Figure 3).
- Seven candidate algorithms; selection by ROC-AUC of the impaired class; Table 1 values (p.165–166).
- Forward selection ordered by SHAP importance, one feature at a time, best set kept; 40 features →
  0.9037 (p.165, p.167, Figure 4).
- Iterative grid search; the num_leaves trajectory; final Table 2 values; tuned ROC-AUC 0.9492
  (p.165, p.167).
- Metrics: Accuracy, ROC-AUC, Precision, Recall, F1-macro (Table 1).

## Items not reported by the paper

- Which AI-Hub partitions were used (only implied by counts).
- Subject-level CN/MCI/Dem counts.
- Activity–sleep record matching rule; duplicate handling.
- Missing-value imputation method (only that the 5-min HR log was removed because it could not be
  handled); outlier handling; scaling.
- Exact conversion of clock time to a real number; unit of sleep_time; whether sleep_time is computed
  from full timestamps or from the converted clock values.
- Definitions of autocorrelation (lag), quantiles, kurtosis, skewness.
- Why classes 0 and 5 of the activity-class log and the 5-min RMSSD log are absent from Figure 2.
- Whether the 5 folds are stratified; whether the data were shuffled; random seed; number of repeats.
- Hyperparameters of the seven base models (Table 1 stage).
- SHAP explainer, data used to compute SHAP importance, importance aggregation, and the model
  configuration used during the forward-selection path.
- Grid-search ranges for min_data_in_leaf, n_estimators, learning_rate; other LightGBM parameters;
  the CV used inside the grid search.
- Averaging mode of Precision/Recall; decision threshold.
- Software / library versions.

## Operational assumptions required for reproduction

Every decision made to fill a gap above is recorded with an ID (A01, A02, …) in `ASSUMPTIONS.md`.
None of them is presented anywhere in this directory as the paper's method.
