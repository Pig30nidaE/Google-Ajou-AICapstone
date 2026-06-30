# RF/LGBM Discrete Dataset Guide

- 생성일: 2026-05-31 10:05:33
- 설명: 논문 RF/LGBM 비연속형 변수 처리 설명에 맞춘 재현형 전처리 CSV

## X에서 제외할 컬럼

`patient_id`, `sample_date`, `split`, `binary_class`는 모델 입력 X에서 제외합니다. `binary_class`만 y로 사용합니다.

## 결측치와 split

- 논문 미기재: 결측치 처리 방식, median imputation, 결측률 50% 이상 피처 제거.
- 구현상 선택: train 기준 median을 fit하고 val에는 transform만 적용.
- 구현상 선택: 결측률 피처 제거 적용 여부 `True`, threshold `0.50`.
- `split` 컬럼은 원본 `1.Training`/`2.Validation` 출처를 표시하는 참고 컬럼입니다.
- 논문 RF/LGBM 재현 모드에서는 `split=train`만 사용해 70/30 row-level stratified split과 Random Search를 수행합니다.
- 임상 일반화 성능 확인 모드에서는 전체 CSV를 사용하되 `patient_id` 기준 GroupKFold/GroupShuffleSplit을 사용합니다.

## 전처리 원칙

- RF/LGBM용 이산형 변수만 남깁니다.
- 제외: `5min`, `1min` 연속형 시계열, timestamp/datetime/date/time/start/end, list/array, id/email/sample/label/class/diag 계열 메타 컬럼.
- Sleep `sample_date`는 수면 종료일 기준입니다.
- Activity는 일 단위 집계이며, `score`/`average`/`efficiency`/`met` 계열은 mean, 나머지 시간/횟수/총량 계열은 sum입니다.

- 원본 `1.Training`은 `train`, `2.Validation`은 `val`로 유지합니다.

## Activity

- 파일: `rf_lgbm_activity_discrete.csv`
- rows: 12183
- patients: 174
- features: 23
- missing before/after: 0 -> 0
- dropped features: 없음

### Split Distribution

| split | rows | patients |
| --- | --- | --- |
| train | 9705 | 141 |
| val | 2478 | 33 |

### Label Distribution

| split | 0 | 1 |
| --- | --- | --- |
| train | 5781 | 3924 |
| val | 1956 | 522 |

### Features

- `activity_average_met`
- `activity_cal_active`
- `activity_cal_total`
- `activity_daily_movement`
- `activity_high`
- `activity_inactive`
- `activity_inactivity_alerts`
- `activity_low`
- `activity_medium`
- `activity_met_min_high`
- `activity_met_min_inactive`
- `activity_met_min_low`
- `activity_met_min_medium`
- `activity_non_wear`
- `activity_rest`
- `activity_score`
- `activity_score_meet_daily_targets`
- `activity_score_move_every_hour`
- `activity_score_stay_active`
- `activity_score_training_frequency`
- `activity_score_training_volume`
- `activity_steps`
- `activity_total`

## Sleep

- 파일: `rf_lgbm_sleep_discrete.csv`
- rows: 12171
- patients: 174
- features: 24
- missing before/after: 0 -> 0
- dropped features: 없음

### Split Distribution

| split | rows | patients |
| --- | --- | --- |
| train | 9694 | 141 |
| val | 2477 | 33 |

### Label Distribution

| split | 0 | 1 |
| --- | --- | --- |
| train | 5776 | 3918 |
| val | 1955 | 522 |

### Features

- `sleep_awake`
- `sleep_breath_average`
- `sleep_deep`
- `sleep_duration`
- `sleep_efficiency`
- `sleep_hr_average`
- `sleep_hr_lowest`
- `sleep_is_longest`
- `sleep_light`
- `sleep_onset_latency`
- `sleep_rem`
- `sleep_restless`
- `sleep_rmssd`
- `sleep_score`
- `sleep_score_alignment`
- `sleep_score_deep`
- `sleep_score_disturbances`
- `sleep_score_efficiency`
- `sleep_score_latency`
- `sleep_score_rem`
- `sleep_score_total`
- `sleep_temperature_delta`
- `sleep_temperature_deviation`
- `sleep_total`
