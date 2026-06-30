# Paper Implementation Notes

Source PDF:

`docs/설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도 산정 방법에 관한 연구.pdf`

## Extracted Method

1. Use AI-Hub dementia high-risk wearable lifelog data from 174 subjects.
2. Build daily-row lifelog data from wearable activity and sleep records.
3. Encode target as binary:
   - `CN -> 0`
   - `MCI`, `Dem` -> `1`
4. Paper class counts:
   - CN daily rows: `7,737`
   - impaired daily rows: `4,446`
   - total daily rows: `12,183`
5. Remove unusable columns:
   - constant timestamp-like columns
   - 5-minute heart-rate logs with unresolvable missingness
6. Convert timestamp columns to 24-hour numeric values.
7. Add sleep duration from bedtime end minus bedtime start.
8. Convert annotated sequence logs into count/statistical features.
9. Compare seven models with 5-fold cross validation:
   - Logistic Regression
   - Decision Tree
   - K-Nearest Neighbor
   - Support Vector Machine
   - Multi-Layer Perceptron
   - Random Forest
   - LightGBM
10. Select best model by ROC-AUC. Paper selected LightGBM.
11. Compute SHAP importance, add features by descending importance, and evaluate forward selection.
12. Paper reports best forward selection at top 40 features with ROC-AUC `0.9037`.
13. Tune LightGBM by grid search. Paper final parameters:
   - `min_data_in_leaf=41`
   - `num_leaves=330`
   - `n_estimators=1000`
   - `learning_rate=0.08`
14. Paper final tuned ROC-AUC: `0.9492`.
15. Compute Dementia Risk Score per daily row as the sum of positive SHAP values for the impaired class.
16. Validate DRS by one-sided one-sample t-test:
   - H0: impaired mean DRS <= CN mean DRS
   - H1: impaired mean DRS > CN mean DRS

## Implementation Choices

- The provided raw activity files contain exactly `12,183` daily rows after combining train and validation splits, matching the paper count.
- The provided raw sleep files contain 12 fewer rows. To match the paper's count while preserving sleep features, preprocessing uses activity rows as the base table and left-joins sleep by `patient_id`, `sample_date`, and `split`.
- Missing values from the left join are imputed inside each CV fold by `SimpleImputer(strategy="median")`.
- The paper describes daily-row 5-fold CV and later states that daily rows were used because only 174 subjects were available. Therefore row-level `StratifiedKFold` is the default. Subject-group CV is available with `--grouped` only as a diagnostic experiment.
- The paper does not disclose all untuned baseline hyperparameters, exact fold seed, or all feature engineering details. These remain explicit code defaults rather than hidden assumptions.

## Local Preprocessing Verification

The preprocessing script was run locally with:

```bash
/Users/pig30nidae/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  training/XAI_Paper_Reproduction/scripts/01_preprocess_daily_binary.py \
  --output-dir /private/tmp/xai_repro_check
```

Observed output summary:

```json
{
  "rows": 12183,
  "subjects": 174,
  "class_counts": {"0": 7737, "1": 4446},
  "split_counts": {"train": 9705, "val": 2478},
  "feature_count": 86,
  "merge_policy": "left_activity"
}
```

The implementation includes the paper-discussed features:

- `sleep_breath_average`
- `sleep_hr_average`
- `sleep_hr_lowest`
- `activity_class_5min_count_3`

## Output Audit

`scripts/06_audit_reproduction_outputs.py` verifies a completed run against the main paper-reproduction requirements:

- required data, baseline, feature-selection, final-model, DRS, and report outputs exist
- preprocessing count matches the paper: `12,183` rows, `174` subjects, CN `7,737`, impaired `4,446`
- all seven paper baseline models are present in a full run
- the paper top-40 feature list is available
- final LightGBM parameters match the paper in strict mode
- DRS is computed for all rows in strict mode
- impaired DRS mean is greater than CN mean with one-sided t-test `p < 0.05`

Use `--allow-smoke` only for reduced local checks.
