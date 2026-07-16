# Three-Class Performance Lab - Local EDA Report

이 보고서는 Training 자료만으로 계산한 식별자 비노출 집계입니다. Validation 라벨은 열지 않았고, Validation 원천자료는 헤더와 subject-overlap 확인용 식별자 열만 읽었습니다.

## 핵심 결과

- Training subject: CN 85명, MCI 47명, DEM 9명.
- 세 Training 라벨 사본의 subject 집합과 정규화 라벨은 모두 일치: `True`.
- Activity 9,705행/141명, Sleep 9,705행/141명.
- 날짜 기준 activity-sleep matched 9,673건, activity-only 32건, sleep-only 21건.
- Protocol/coverage negative control 중 |Cliff's delta| >= 0.33인 feature: 14개. 이 변수들은 성능 후보가 아니라 누수 경고로 해석해야 합니다.

## 데이터 계약 및 격리

- Target mapping: CN=0, MCI=1, DEM/Dementia=2.
- Activity 날짜는 `activity_day_start`, sleep 날짜는 수면 종료일(`sleep_bedtime_end`)을 사용했습니다.
- 이 설명용 EDA 집계에서는 같은 subject-date의 sleep을 `sleep_is_longest`와 duration 우선순위로 1건화했습니다. 최종 discovery pipeline은 장비 flag를 feature/선택에 쓰지 않고, prediction index 이전 episode만 남긴 뒤 duration, bedtime start/end의 고정 시간 규칙으로 main sleep을 다시 선택합니다. 원본 중복 수는 별도 audit에만 둡니다.
- 모든 효과크기는 일별 행이 아니라 subject median/IQR로 계산했습니다.
- Validation 라벨은 읽지 않았으며 모델/feature 선택에 사용할 수 없습니다.

## Modality audit

| modality | rows | subjects | date range | missing date rows | duplicate subject-date rows | raw missing cell fraction |
|---|---:|---:|---|---:|---:|---:|
| activity | 9,705 | 141 | 2020-10-17 - 2021-02-17 | 0 | 0 | 0.0000 |
| sleep | 9,705 | 141 | 2020-10-17 - 2021-02-17 | 0 | 11 | 0.0000 |

## Class별 coverage

| class | activity valid days median [IQR] | sleep valid days median [IQR] | matched-day ratio median [IQR] |
|---|---:|---:|---:|
| CN | 65.000 [17.000] | 64.000 [18.000] | 1.000 [0.000] |
| MCI | 67.000 [14.000] | 67.000 [14.000] | 1.000 [0.000] |
| DEM | 61.000 [23.000] | 61.000 [23.000] | 1.000 [0.000] |

## 가장 큰 training-only class 효과

### MCI vs CN

| feature | class median | CN median | Cliff's delta | robust SMD |
|---|---:|---:|---:|---:|
| `sleep__sleep_midpoint_at_delta__subject_median` | 7102.000 | 8270.000 | -0.266 | -0.344 |
| `sleep__sleep_restless__subject_day_iqr` | 8.000 | 7.000 | 0.254 | 0.450 |
| `activity__activity_class_5min__longest_inactive_run__subject_day_iqr` | 6.500 | 7.000 | -0.229 | -0.225 |
| `activity__activity_class_5min__longest_inactive_run__subject_median` | 13.000 | 15.000 | -0.228 | -0.654 |
| `sleep__sleep_hypnogram_5min__deep_bout_count__subject_median` | 5.000 | 6.000 | -0.227 | -0.674 |
| `sleep__sleep_score_alignment__subject_day_iqr` | 0.000 | 7.000 | -0.225 | -0.578 |
| `sleep__sleep_hypnogram_5min__entropy__subject_day_iqr` | 0.130 | 0.115 | 0.215 | 0.536 |
| `sleep__sleep_score_disturbances__subject_day_iqr` | 10.500 | 9.500 | 0.212 | 0.431 |
| `activity__activity_class_5min__longest_rest_run__subject_day_iqr` | 15.500 | 13.750 | 0.203 | 0.402 |
| `activity__activity_score_move_every_hour__subject_day_iqr` | 0.000 | 5.000 | -0.202 | -1.349 |
| `sleep__sleep_score_alignment__subject_median` | 100.000 | 100.000 | 0.197 | 0.000 |
| `sleep__derived_bedtime_end_hour__subject_median` | 5.898 | 6.525 | -0.197 | -0.520 |

### DEM vs CN

| feature | class median | CN median | Cliff's delta | robust SMD |
|---|---:|---:|---:|---:|
| `activity__activity_class_5min__active_ratio_first_half__subject_day_iqr` | 0.083 | 0.132 | -0.718 | -1.396 |
| `activity__activity_class_5min__class_1_ratio__subject_median` | 0.418 | 0.326 | 0.591 | 1.736 |
| `sleep__sleep_light__subject_median` | 19500.000 | 14400.000 | 0.584 | 1.295 |
| `activity__activity_rest__subject_median` | 666.000 | 505.000 | 0.582 | 2.328 |
| `sleep__sleep_score_deep__subject_day_iqr` | 35.000 | 5.500 | 0.552 | 1.980 |
| `activity__activity_total__subject_day_iqr` | 73.000 | 113.500 | -0.549 | -1.362 |
| `activity__activity_average_met__subject_day_iqr` | 0.094 | 0.156 | -0.511 | -1.058 |
| `sleep__sleep_restless__subject_median` | 46.000 | 33.000 | 0.511 | 1.323 |
| `sleep__sleep_midpoint_time__subject_median` | 18180.000 | 14055.000 | 0.505 | 1.586 |
| `activity__activity_low__subject_day_iqr` | 66.500 | 96.750 | -0.497 | -0.996 |
| `sleep__sleep_hypnogram_5min__transition_rate__subject_median` | 0.262 | 0.303 | -0.493 | -1.207 |
| `sleep__sleep_hypnogram_5min__bout_count__subject_day_iqr` | 10.500 | 9.000 | 0.489 | 0.799 |

## Coverage/protocol negative controls

아래 항목은 collection length, 결측, valid sequence length, duplicate-day, first-date 같은 관리·프로토콜 변수입니다. 큰 효과가 있더라도 질병 신호로 간주하거나 바로 모델에 투입하면 안 됩니다.

| feature | strongest class | Cliff's delta vs CN | n |
|---|---|---:|---:|
| `sleep__sleep_hr_5min__valid_count__subject_median` | DEM | 0.459 | 9 |
| `sleep__sleep_rmssd_5min__valid_count__subject_median` | DEM | 0.459 | 9 |
| `sleep__sleep_hypnogram_5min__valid_count__subject_median` | DEM | 0.456 | 9 |
| `protocol_sleep_date_coverage_ratio` | DEM | -0.433 | 9 |
| `protocol_activity_date_coverage_ratio` | DEM | -0.418 | 9 |
| `protocol_sleep_first_date_offset_days` | MCI | -0.414 | 47 |
| `protocol_activity_first_date_offset_days` | MCI | -0.414 | 47 |
| `sleep__sleep_hr_5min__nonzero_count__subject_median` | DEM | 0.403 | 9 |
| `sleep__sleep_rmssd_5min__nonzero_count__subject_median` | DEM | 0.403 | 9 |
| `sleep__sleep_rmssd_5min__valid_count__subject_day_iqr` | DEM | 0.362 | 9 |
| `sleep__sleep_hr_5min__valid_count__subject_day_iqr` | DEM | 0.362 | 9 |
| `sleep__sleep_rmssd_5min__nonzero_count__subject_day_iqr` | DEM | 0.359 | 9 |

## MMSE exclusion

Training/Validation MMSE 값은 feature EDA에서 제외했습니다. MMSE 원천에는 `DIAG_NM`, 임상의/검사 메타데이터 및 진단과 매우 가까운 인지검사 점수가 함께 있어, prediction index와 진단 생성 절차가 확정되기 전 사용하면 target leakage 또는 임상적 순환성이 생길 수 있습니다. Validation MMSE에서는 schema와 overlap 확인을 위한 식별자 열 외 값을 읽지 않았습니다.

## 모델링 시사점

1. 완전관측 조건으로 subject를 버리지 않고 141명 모두를 유지하되, primary 입력은 최근 observed event의 생체 요약(`event_summary_v1`)과 coverage 비노출 sequence(`event_sequence28_v1`)로 제한합니다.
2. EDA 효과크기는 해석·감사용이며 이번 run의 고정 4개 primary 후보, feature 집합, hyperparameter를 추가하거나 바꾸는 supervised 선택 근거로 사용하지 않습니다.
3. Observed count, valid length/ratio, calendar gap, missing fraction, padding mask는 primary 입력에서 제외하고 `coverage_only_v1` negative control 또는 고정 `mask_tcn_35d_legacy_v1` comparator에서만 사용합니다.
4. Parsed intraday HR/RMSSD, sleep-stage/activity-state 비율·전이·엔트로피는 허용된 생체 후보로 유지하되, Validation benchmark를 이용한 feature 제거·threshold·ensemble 선택은 금지합니다.

## 한계

- DEM Training subject가 9명뿐이므로 큰 효과도 불확실합니다. 효과크기는 탐색적 연관성이지 통계적·임상적 인과가 아닙니다.
- 데이터 수집 프로토콜과 diagnosis timing 정보가 제한되어 chronology 가정은 최종 파이프라인에서 다시 검증해야 합니다.
- `class_feature_summary.csv`는 class별 aggregate만 포함하며 subject-level 행은 저장하지 않습니다.
