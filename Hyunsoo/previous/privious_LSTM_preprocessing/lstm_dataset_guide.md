# LSTM Dataset Guide

- 생성일: 2026-05-31 11:58:05
- 설명: 논문 LSTM 전처리 설명에 맞춘 재현형 전처리 데이터

## 산출 파일

- `lstm_dataset.pkl`: 3차원/4차원 모델 입력을 담은 메인 pickle
- `lstm_window_index.csv`: pickle window 순서 확인용 CSV
- `lstm_discrete_daily.csv`: sliding window 전 discrete daily feature 확인용 CSV

## Pickle Keys / Shapes

- `X_continuous_seq`: 5분 단위 continuous 4D 입력, (6940, 7, 288, 5)
- `X_continuous_flat_seq`: day-level LSTM용 continuous flatten 입력, (6940, 7, 1440)
- `X_discrete_seq`: 일 단위 discrete 3D 입력, (6940, 7, 47)
- `X_integrated_seq`: continuous flatten + discrete 통합 입력, (6940, 7, 1487)
- `y`: window label, (6940,)
- `patient_id`, `window_start_date`, `window_end_date`, `split`: window별 메타 배열
- `continuous_scaler_params`, `discrete_scaler_params`: train 기준 MinMax scaler 통계
- continuous feature 수: 5
- continuous flat feature 수: 1440
- discrete feature 수: 47
- integrated feature 수: 1487

## Window / Split

- 전체 window 수: 6940
- 전체 patient 수: 170
- 원본 label 기준 patient 수: 174
- strict 7일 window 기준 최종 patient 수: 170
- strict 7일 window 기준 제외 patient 수: 4
- 제외 patient_id: nia+088@rowan.kr, nia+112@rowan.kr, nia+219@rowan.kr, nia+229@rowan.kr
- sequence days: 7
- daily sequence length: 288
- padding value: -1.0
- window 기준: `patient_id + window_start_date + window_end_date`
- `lstm_window_index.csv`의 `window_index`는 pickle 배열의 첫 번째 축 index와 일치합니다.
- `split` 컬럼은 원본 `1.Training`/`2.Validation` 출처 표시용 참고 컬럼입니다.
- 모델팀 검증 방식: 전체 pickle을 사용하여 `patient_id` 기준 5-fold GroupKFold를 직접 수행합니다.
- 일반 KFold 및 row 단위 random split은 사용하지 않습니다.

### Split별 Window 수

| split | windows |
| --- | --- |
| train | 5500 |
| val | 1440 |

### Split별 Patient 수

| split | patients |
| --- | --- |
| train | 138 |
| val | 32 |

### Split별 Label 분포

| split | 0 | 1 |
| --- | --- | --- |
| train | 3359 | 2141 |
| val | 1118 | 322 |

## MinMax / 결측 / Padding

- 논문 명시: LSTM 학습 전 MinMax 정규화와 일주일 시퀀스 작업을 수행합니다.
- 구현상 선택: continuous와 discrete 모두 train 기준으로 scaler를 fit하고 val에는 transform만 적용합니다.
- 구현상 선택: continuous padding/missing placeholder `-1.0`은 scaler fit에서 제외하고 정규화 후에도 `-1.0`으로 유지합니다.
- 구현상 선택: discrete 결측치는 train median으로 대체한 뒤 train 기준 MinMax를 적용합니다.
- 구현상 선택: val 값이 train min/max 범위를 벗어나면 0~1 범위로 clipping합니다.
- 논문 미기재, 구현상 선택: 결측률 50% 이상 feature 제거, activity 일 단위 집계, 연속 7일 window만 사용, timestep 결측과 padding을 `-1.0`으로 처리, 1min to 5min 변환.
- activity와 sleep이 모두 존재하는 `patient_id + sample_date`만 유지합니다. 한쪽 modality가 없는 날을 `-1.0`으로 채우는 방식은 사용하지 않습니다.
- 1min to 5min 변환: `activity_met_1min`은 5개씩 묶어 평균을 사용했습니다. 1분 단위 범주형 시계열이 추가될 경우 5개 단위 최빈값을 사용하도록 함수가 준비되어 있습니다.

## Integrated Input

- 통합 방식: day-level LSTM 입력
- timestep = 7일
- feature = 하루 288개 5분 단위 continuous flatten + 일 단위 discrete feature
- 5min-level LSTM(timestep=2016)도 가능하지만, 시퀀스 길이 증가와 일 단위 discrete 결합 문제 때문에 본 구현에서는 day-level 통합 방식을 사용합니다.
- 이 통합 방식은 논문 미기재, 구현상 선택입니다.

## Feature Names

### Continuous

- `activity_class_5min`
- `activity_met_5min`
- `sleep_hr_5min`
- `sleep_hypnogram_5min`
- `sleep_rmssd_5min`

### Discrete

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

## 전처리 통계

- 날짜 누락으로 제외된 continuous window 수: 4166
- 날짜 누락으로 제외된 discrete window 수: 4166
- continuous에만 있는 window 수: 0
- discrete에만 있는 window 수: 0
- activity만 있어 제외한 daily sample 수: 33
- sleep만 있어 제외한 daily sample 수: 21
- dropped discrete features: []

## 모델팀 사용 예시

```python
import pickle

with open('lstm_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X_cont_4d = data['X_continuous_seq']
X_cont_flat = data['X_continuous_flat_seq']
X_disc = data['X_discrete_seq']
X_int = data['X_integrated_seq']
y = data['y']
groups = data['patient_id']
```