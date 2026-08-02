# Google YDF CNBoost — Training-only EDA

이 보고서는 Training 사람 단위 집계만 사용했습니다. Validation은 열지 않았고 MMSE 파일은 경로조차 찾지 않았습니다.

## 먼저 볼 결론

- Training은 CN 85명, MCI 47명, DEM 9명입니다.
- 모델 후보로 들어가는 웨어러블 요약은 2640개입니다.
- CN 대 비CN에서 |Cliff's delta| 0.33 이상인 상위 후보는 0개입니다. 단일 변수 하나로 해결되기보다 여러 작은 패턴을 묶어야 한다는 뜻입니다.
- 이전 실험에서 강했던 관측일 수, 첫 날짜, mask, calendar gap은 수집 방식의 차이일 수 있어 주 모델에서 차단했습니다.
- 아래 순위는 이해를 위한 EDA입니다. 실제 학습 변수 선택은 각 CV 학습 fold 안에서 다시 계산됩니다.

## CN 대 비CN 상위 패턴

| feature | cliffs_delta_group_minus_cn | direction_free_univariate_auc | bootstrap_delta_ci_low | bootstrap_delta_ci_high | bootstrap_sign_consistency |
|---|---|---|---|---|---|
| `sleep__event7__sleep__raw__sleep_light__p90` | 0.313 | 0.657 | 0.123 | 0.478 | 1.000 |
| `sleep__event28__sleep__stage__entropy__iqr` | 0.300 | 0.650 | 0.120 | 0.475 | 1.000 |
| `sleep__event28__sleep__stage__entropy__mad` | 0.282 | 0.641 | 0.094 | 0.465 | 1.000 |
| `activity__event14__activity__circadian__peak_cos__p10` | 0.282 | 0.641 | 0.090 | 0.456 | 0.996 |
| `sleep__event7__sleep__hr__lowest_clock_cos__theil_sen_rank_slope` | 0.277 | 0.639 | 0.075 | 0.467 | 0.998 |
| `activity__event7__activity__circadian__peak_cos__p10` | 0.268 | 0.634 | 0.090 | 0.453 | 0.992 |
| `sleep__event14__sleep__clock__bedtime_sin__p10` | -0.266 | 0.633 | -0.428 | -0.086 | 0.992 |
| `sleep__event14__sleep__stage__deep_ratio__p90` | -0.265 | 0.632 | -0.447 | -0.066 | 1.000 |
| `activity__event14__activity__circadian__peak_cos__median` | 0.264 | 0.632 | 0.074 | 0.456 | 0.996 |
| `sleep__event14__sleep__raw__sleep_awake__p10` | 0.264 | 0.632 | 0.064 | 0.458 | 0.996 |
| `sleep__event14__sleep__raw__sleep_awake__trimmed_mean_10` | 0.260 | 0.630 | 0.086 | 0.443 | 0.998 |
| `sleep__event14__sleep__stage__deep_ratio__trimmed_mean_10` | -0.257 | 0.628 | -0.437 | -0.064 | 0.994 |
| `activity__event7__activity__met__trimmed_mean_10__theil_sen_rank_slope` | 0.255 | 0.628 | 0.070 | 0.448 | 0.994 |
| `activity__event7__activity__raw__activity_daily_movement__mad` | -0.255 | 0.628 | -0.440 | -0.072 | 0.998 |
| `sleep__event14__sleep__duration__deep_ratio__trimmed_mean_10` | -0.255 | 0.627 | -0.447 | -0.063 | 0.990 |

## MCI 대 CN

| feature | cliffs_delta_group_minus_cn | direction_free_univariate_auc |
|---|---|---|
| `sleep__event14__sleep__clock__bedtime_sin__p10` | -0.283 | 0.642 |
| `sleep__event7__sleep__clock__bedtime_sin__p10` | -0.282 | 0.641 |
| `activity__event14__activity__circadian__peak_cos__median` | 0.280 | 0.640 |
| `sleep__event14__sleep__raw__sleep_awake__p10` | 0.276 | 0.638 |
| `sleep__event7__sleep__raw__sleep_score_alignment__median` | 0.276 | 0.638 |
| `sleep__event7__sleep__raw__sleep_midpoint_at_delta__p10` | -0.270 | 0.635 |
| `sleep__event28__sleep__clock__bedtime_sin__trimmed_mean_10` | -0.269 | 0.635 |
| `sleep__event14__sleep__raw__sleep_score_alignment__trimmed_mean_10` | 0.268 | 0.634 |
| `sleep__event28__sleep__raw__sleep_score_alignment__trimmed_mean_10` | 0.268 | 0.634 |
| `sleep__event14__sleep__clock__bedtime_sin__trimmed_mean_10` | -0.266 | 0.633 |
| `sleep__event28__sleep__clock__bedtime_sin__median` | -0.265 | 0.632 |
| `sleep__event28__sleep__hr__lowest_clock_sin__p10` | -0.264 | 0.632 |

## DEM 대 CN

| feature | cliffs_delta_group_minus_cn | direction_free_univariate_auc |
|---|---|---|
| `activity__event7__activity__class__medium_ratio_within_wear__iqr` | -0.723 | 0.861 |
| `activity__event7__activity__class__medium_longest_run_ratio_within_wear__mad` | -0.689 | 0.844 |
| `activity__event7__activity__class__medium_ratio_within_wear__mad` | -0.688 | 0.844 |
| `activity__event7__activity__class__wear_state_entropy__late_half_minus_early_half` | 0.663 | 0.831 |
| `activity__event7__activity__raw__activity_medium__iqr` | -0.641 | 0.820 |
| `activity__event7__activity__class__wear_state_entropy__theil_sen_rank_slope` | 0.629 | 0.814 |
| `sleep__event14__sleep__stage__rem_ratio__theil_sen_rank_slope` | 0.626 | 0.813 |
| `activity__event7__activity__class__medium_longest_run_ratio_within_wear__iqr` | -0.626 | 0.813 |
| `activity__event7__activity__class__inactive_longest_run_ratio_within_wear__iqr` | 0.613 | 0.807 |
| `sleep__event14__sleep__stage__rem_ratio__late_half_minus_early_half` | 0.605 | 0.803 |
| `sleep__event7__sleep__raw__sleep_restless__median` | 0.604 | 0.802 |
| `sleep__event28__sleep__raw__sleep_light__p90` | 0.603 | 0.801 |

## 패턴 묶음

| family | feature_count | median_abs_delta | max_abs_delta | max_direction_free_auc |
|---|---|---|---|---|
| sleep:sleep_stage | 192 | 0.146 | 0.313 | 0.657 |
| sleep:variability | 468 | 0.069 | 0.300 | 0.650 |
| activity:level | 552 | 0.047 | 0.282 | 0.641 |
| sleep:change | 324 | 0.062 | 0.277 | 0.639 |
| sleep:circadian | 84 | 0.137 | 0.266 | 0.633 |
| sleep:level | 192 | 0.151 | 0.264 | 0.632 |
| activity:change | 276 | 0.065 | 0.255 | 0.628 |
| activity:variability | 372 | 0.057 | 0.255 | 0.628 |
| sleep:cardiac | 180 | 0.059 | 0.232 | 0.616 |

## 학습 설계에 반영한 점

1. 직접 3-class YDF, CN 대 비CN을 먼저 보는 계층형 YDF, 확률형 YDF Random Forest를 고정 후보로 둡니다.
2. 수면/활동 한쪽에만 변수가 몰리지 않도록 fold 안에서 두 modality의 최소 후보 수를 보장합니다.
3. 중앙값뿐 아니라 IQR/MAD, 상태 전이, 엔트로피, 최근 변화, 수면 시각의 원형 표현을 유지합니다.
4. feature 선택·모델 조정·blend 결정은 모두 nested CV의 학습 부분 안에서만 합니다.
5. Validation은 label-free 예측을 먼저 저장한 뒤 역사적 benchmark로 한 번 평가합니다.

## 주의할 점

- DEM은 9명뿐이므로 한두 사람만 달라져도 점수가 크게 바뀝니다.
- MCI와 CN의 웨어러블 차이는 대체로 작습니다. 높은 accuracy만 보고 전부 CN에 가깝게 예측하는 모델을 선택하면 안 됩니다.
- 효과크기는 연관성이지 원인이나 임상적 진단 기준이 아닙니다.
- 이 EDA 결과 자체로 feature를 고정하면 전체 Training 라벨을 미리 본 셈이 되므로 학습 코드는 EDA 순위를 직접 가져다 쓰지 않습니다.
