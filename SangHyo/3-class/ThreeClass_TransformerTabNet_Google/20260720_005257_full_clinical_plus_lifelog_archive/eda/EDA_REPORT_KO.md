# CN / MCI / DEM Training EDA 보고서

이 보고서는 **학습 데이터만** 살펴본 결과입니다. 모델 학습이나 Validation 라벨 확인은 하지 않았습니다.

## 먼저 알아둘 점

- 사람 수는 CN 85명, MCI 47명, DEM 9명입니다.
- Gait/Sleep/CognitiveFunction의 라벨 사본 3개는 사람과 정답이 모두 같습니다.
- DEM은 9명뿐이라 한두 명의 결과가 점수를 크게 바꿉니다. 그래서 단일 accuracy보다 Macro F1과 class별 결과를 함께 봐야 합니다.
- Activity와 Sleep은 각각 9,705행이며, 두 자료가 같은 사람·날짜에 맞는 경우는 9,673건입니다.
- 가공 후 사람당 입력 후보는 754개입니다. 실제 학습에서는 각 fold의 Training 부분 안에서만 줄입니다.

## 눈에 띄는 패턴

### 1. 인지검사 점수는 강하지만, 정답 열은 반드시 제외해야 합니다

MMSE TOTAL 중앙값은 CN 28.0, MCI 26.0, DEM 22.0입니다. 
다만 MMSE 파일의 `DIAG_NM`은 이번 정답과 완전히 같은 값입니다. 이 열을 넣으면 모델이 질병 패턴을 배우는 것이 아니라 정답을 복사하므로 코드가 강제로 차단합니다. 질문별 점수와 TOTAL만 기본 모드에서 사용합니다.

### 2. DEM에서는 활동량 감소와 수면 변화가 비교적 크게 보입니다

| feature | class_median | cn_median | cliffs_delta |
| --- | --- | --- | --- |
| sleep → sleep_score_latency → q10 | 62.000 | 67.000 | -0.625 |
| activity → seq_activity_state_state_1_ratio → q10 | 0.356 | 0.278 | 0.605 |
| activity → seq_activity_state_state_1_ratio → median | 0.431 | 0.344 | 0.603 |
| sleep → sleep_light → q90 | 23292.000 | 18234.000 | 0.603 |
| sleep → sleep_light → mean | 19578.621 | 14164.435 | 0.595 |
| sleep → sleep_light → median | 19500.000 | 14400.000 | 0.584 |
| sleep → sleep_score_deep → std | 23.639 | 15.202 | 0.584 |
| activity → activity_rest → median | 666.000 | 505.000 | 0.582 |
| activity → activity_low → std | 52.668 | 81.730 | -0.576 |
| activity → seq_activity_state_state_1_ratio → mean | 0.437 | 0.345 | 0.574 |

위 표는 Training 안의 연관성입니다. DEM 사람이 매우 적으므로 질병의 원인이나 확정된 임상 패턴으로 해석하면 안 됩니다.

### 3. MCI와 CN은 차이가 더 작고 겹침이 큽니다

| feature | class_median | cn_median | cliffs_delta |
| --- | --- | --- | --- |
| sleep → clock_bedtime_start_sin → median | -0.560 | -0.428 | -0.270 |
| sleep → clock_bedtime_start_sin → mean | -0.518 | -0.357 | -0.267 |
| sleep → sleep_midpoint_at_delta → median | 7102.000 | 8270.000 | -0.266 |
| sleep → sleep_score_alignment → mean | 94.937 | 93.261 | 0.265 |
| sleep → sleep_restless → iqr | 8.000 | 7.000 | 0.254 |
| sleep → sleep_score_alignment → q10 | 83.800 | 76.400 | 0.248 |
| sleep → sleep_score_alignment → std | 10.576 | 13.612 | -0.242 |
| sleep → sleep_midpoint_at_delta → q10 | 3277.200 | 4914.600 | -0.241 |
| sleep → clock_bedtime_start_sin → q10 | -0.788 | -0.724 | -0.239 |
| sleep → sleep_score_alignment → iqr | 0.000 | 7.000 | -0.225 |

MCI 구분이 세 class 중 가장 어려울 가능성이 큽니다. 이 때문에 class별 가중치, Macro F1 중심 선택, 여러 모델의 확률 평균을 사용합니다.

### 4. 수집 일수 자체도 class마다 조금 다릅니다

| class | activity days median [Q1-Q3] | sleep days median [Q1-Q3] |
| --- | ---: | ---: |
| CN | 65 [57-74] | 64 [56-74] |
| MCI | 67 [63-77] | 67 [63-77] |
| DEM | 61 [51-74] | 61 [51-74] |

수집 일수나 빈 날짜 수를 주 모델에 넣으면 병이 아니라 수집 방식의 차이를 외울 수 있습니다. 그래서 날짜·수집량·ID는 입력에서 제외했습니다.

## MMSE 문항에서 보이는 차이

| comparison | feature | class_median | cn_median | cliffs_delta |
| --- | --- | --- | --- | --- |
| DEM vs CN | mmse → orientation_sum | 17.000 | 20.000 | -0.927 |
| DEM vs CN | mmse → total | 22.000 | 28.000 | -0.910 |
| DEM vs CN | mmse → memory_language_sum | 34.000 | 38.000 | -0.800 |
| DEM vs CN | mmse → q03 | 1.000 | 2.000 | -0.754 |
| DEM vs CN | mmse → q05 | 1.000 | 2.000 | -0.556 |
| DEM vs CN | mmse → q16_3 | 1.000 | 2.000 | -0.497 |
| DEM vs CN | mmse → q12_2 | 1.000 | 2.000 | -0.495 |
| DEM vs CN | mmse → q13_2 | 1.000 | 2.000 | -0.467 |
| MCI vs CN | mmse → total | 26.000 | 28.000 | -0.389 |
| MCI vs CN | mmse → memory_language_sum | 37.000 | 38.000 | -0.342 |
| MCI vs CN | mmse → q13_3 | 1.000 | 2.000 | -0.339 |
| MCI vs CN | mmse → q13_2 | 1.000 | 2.000 | -0.332 |
| MCI vs CN | mmse → orientation_sum | 19.000 | 20.000 | -0.303 |
| MCI vs CN | mmse → q12_5 | 2.000 | 2.000 | -0.280 |
| MCI vs CN | mmse → q03 | 2.000 | 2.000 | -0.168 |
| MCI vs CN | mmse → q09 | 2.000 | 2.000 | -0.137 |

## 전처리 결정

1. 한 사람당 한 행으로 합쳐 사람 단위로만 나눕니다.
2. Activity와 Sleep은 중앙값, 변동폭, 최근-초기 변화, 완만한 추세로 요약합니다.
3. 수면이 하루에 여러 개면 장비 ID 대신 가장 긴 수면과 시간 순서로 하나를 고릅니다.
4. 마지막 Activity 시점보다 미래인 Sleep 행은 제외합니다.
5. 결측값 채우기·크기 맞추기·특징 선택은 매 fold의 Training 부분에서만 배웁니다.
6. `DIAG_NM`, `DIAG_SEQ`, ID, 의사명, 검사순번, 절대 날짜는 어떤 모델에도 넣지 않습니다.

## 성능 목표를 읽는 법

Accuracy 0.8 이상을 목표로 탐색하지만, 141명·DEM 9명 자료에서 이를 미리 보장할 수는 없습니다. 
내부 nested-CV와 한 번만 확인하는 Validation 결과를 따로 저장하며, Accuracy·Macro F1·ROC-AUC와 class별 F1을 모두 보고 판단합니다.

## 산출물

- 사용 가능한 특징: 754개
- 결측률 40% 초과 특징: 0개
- `top_effects.csv`: class별 큰 연관성 목록
- `feature_quality.csv`: 결측·고유값 점검
- `class_counts.png`, `top_wearable_effects.png`: 빠르게 보는 그림

이 분석은 관련성을 보여줄 뿐, 임상 진단이나 인과관계를 뜻하지 않습니다.
