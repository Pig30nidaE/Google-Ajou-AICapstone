# assumptions.md — 논문 미보고 항목과 본 재현이 채택한 가정

작성일: 2026-08-02

원칙:

1. 논문에 명시된 값은 **그대로** 사용한다.
2. 미보고 항목은 **보수적 기본값**을 쓰고 여기에 기록한다.
3. 미보고 항목을 "원 논문 구현"이라고 부르지 않는다.
   해당 설정 이름에는 반드시 `assumption_variant`를 포함한다.
4. 모든 가정은 config로 변경 가능해야 한다. 코드에 상수로 박지 않는다.

---

## A. 데이터·전처리 가정

### A-1. 날짜 파생 `assumption_variant_date_derivation`

**논문**: 표 1에 `date` 컬럼이 있다고 기재. **실제 데이터에는 없음.**

**가정**:
```python
activity_date = to_datetime(activity_day_start).tz_convert("Asia/Seoul").date()
sleep_date    = to_datetime(sleep_bedtime_end).tz_convert("Asia/Seoul").date()
```

**근거**: `activity_day_start`는 매일 04:00 고정 경계이고, `sleep_bedtime_end`(기상 시각)가
그 활동일과 같은 날에 떨어진다. 두 파생 날짜의 일치율은 실측 **99.65%**다.

**대안**: `sleep_bedtime_start` 기준(config `sleep_date_source: bedtime_start`).
전날 저녁 취침이 전날로 배정되어 활동일과 어긋난다. 민감도 분석용.

---

### A-2. activity ↔ sleep 결합 `assumption_variant_join_mode`

**논문**: 결합키 미보고. 코드 스니펫은 각각 집계 후 사용하므로 일별 결합을 우회한다.

**가정 (기본 `positional`)**: 피험자 내 원본 행 순서 i ↔ i로 결합한다.

**근거**: 두 파일의 피험자별 행 수가 **모든 피험자에서 동일**하고, 행 순서의 EMAIL 배열이
**완전히 동일**하다. 배포 시점부터 1:1로 정렬되어 있다고 보는 것이 자연스럽다.
12,183행이 손실 없이 보존된다.

**대안 `date`**: `(subject_id, date)` inner join. train 9,684행(21행 손실), val 2,478행.
sleep의 `bedtime_end` 날짜 중복(train 11, val 1) 때문에 손실이 발생한다.

> RF/XGB의 피험자 평균 집계에서는 두 모드의 결과가 **거의 같다**(집계 후 EMAIL 병합).
> 차이는 시계열 표현에서만 실질적이다.

---

### A-3. 특징 집합 `paper_code_verbatim`

**논문**: "총 58개의 변수가 학습에 사용됐다."

**가정**: 논문에 제시된 RF/XGBoost 코드(`groupby.mean(numeric_only=True)` + drop 목록)를
**문자 그대로** 적용했을 때 남는 **49개**를 기본 특징집합으로 한다. 딥러닝 입력 코드와
최종 shape는 보고되지 않았으므로, LSTM/Bi-LSTM/1D-CNN에도 같은 49개를 쓰는 것은 별도의
`assumption_variant_shared_49_features`다.

**근거**: 논문 서술(58)과 제시 코드(49)가 모순될 때, 트리 모델에는 실행 가능한 코드 쪽이
더 강한 근거다.
서술을 맞추려면 존재하지 않는 컬럼 2개를 만들어내고 문자열 6개를 임의로 수치화해야 하는데,
그 방법이 논문에 없으므로 재현 불가능하다.

**대안**: `feature_set: paper_declared_58` — 논문이 나열한 56개 중 실재하는 54개만 쓰고
문자열 6개를 파생 수치로 변환. **명시적 assumption_variant이며 기본값 아님.**

---

### A-3-1. `paper_table16_lifelog` — 논문이 말한 "최종 분석 변수군"의 재구성

**논문 §3.2 마지막 문장**:

> "변수 선택에 있어서는 다중공선성을 제거하고 통계적 유의성과 임상적 중요성을 모두
> 고려하여 최종 분석 변수군을 구성하였다."

이 문장은 전처리 절(§3.2)에 있으므로 **두 경로(트리·딥러닝) 모두에 적용**된다.
그런데 **그 변수군이 무엇인지 논문은 ML 모델에 대해 밝히지 않는다.**

기존 `paper_code_verbatim`(49개)은 이 문장을 **전혀 구현하지 않는다.** 즉 논문이
"변수 선택을 했다"고 쓴 단계를 통째로 건너뛴 것이며, 이것이 재현 실패의 유력한
원인이다. 141명에 49개 특징은 과적합이 심하다.

**가정**: 논문이 인쇄한 유일한 구체적 라이프로그 변수 목록인 **표 16(학술지 표 13)의
12개**를 그 "최종 분석 변수군"으로 읽는다.

```
sleep_breath_average, sleep_hr_average, sleep_hr_lowest, sleep_efficiency,
sleep_midpoint_time, sleep_restless, sleep_score_disturbances,
activity_cal_active, activity_cal_total, activity_daily_movement,
activity_inactive, activity_met_min_medium
```

**이 가정의 약점(정직하게 기록)**: 표 16은 §3.4 로지스틱 회귀의 결과표이고, 논문이
이 12개를 §3.3의 ML 모델에 썼다고 **명시하지 않았다.** 따라서 이것은 재구성 가설이며
논문 진술이 아니다. config 이름과 결과 파일에 그렇게 표시된다.

**이 가정을 지지하는 정황**:

- 표 16은 "완전 선형 종속성을 제거한 뒤" 남은 목록이라고 서술된다 —
  §3.2의 "다중공선성을 제거"와 같은 절차다.
- 표 16에는 `sleep_efficiency`(p=0.756)처럼 **유의하지 않은 변수도 포함**돼 있다.
  즉 유의성 필터링 결과가 아니라 **최종 모형에 남은 변수 목록**이다.
  (그래서 p-value로 재유도하면 논문 목록과 달라진다.)
- 초록이 꼽은 유의 예측변수 `수면 중 호흡률`, `수면 중 뒤척임`이 이 목록에 있다.

**사용법**: `configs/paper_literal_table16.yaml`. `paper_reproduction.yaml`(49개)과
같은 seed로 돌려 비교하면 변수 선택이 격차를 얼마나 설명하는지 분리해서 볼 수 있다.

---

### A-4. 결측치 처리

**논문**: 미보고.

**가정**: activity/sleep 실측 결측 **0건**이므로 실험 A에서는 결측 처리를 하지 않는다.
실험 B·C에서는 fold-local `SimpleImputer(strategy="median")`를 파이프라인에 넣되,
결측이 없으면 항등 연산이 된다. 시퀀스 패딩 구간은 결측이 아니라 **마스크**로 처리한다.

---

### A-5. 정규화 `assumption_variant_scaler_scope`

**논문**: "시간 순서를 유지한 채 **정규화** 후 3차원 텐서" — 방법·적합 범위 모두 미보고.

**가정**:
- 방법: `StandardScaler` (평균 0, 분산 1). 트리 모델(RF/XGB)에는 미적용(불변).
- 적합 범위: **실험 A에서만 두 변형을 모두 제공**한다.

| config 값 | 의미 | 용도 |
| --- | --- | --- |
| `train_only` (기본) | train 141명에서만 fit | 정직한 재현 |
| `all_data` | 174명 전체에서 fit | 논문이 이랬을 경우의 낙관 편향 크기 측정 |

**실험 B·C에서는 `train_only`만 허용**한다. `all_data`를 넣으면 감사기가 예외를 던진다.

---

### A-6. 무분산·행정 변수

**논문**: 언급 없음. `sleep_is_longest`(상수 1), `sleep_period_id`(세션 인덱스)를 포함한다.

**가정**:
- 실험 A: 논문 코드 그대로 **포함**. 무분산 경고만 기록.
- 실험 B·C: `drop_zero_variance: true`, `drop_administrative: true` 기본.
  단 무분산 판정은 **fold의 학습부분에서만** 수행한다(전체 데이터로 판정하면 누수).

---

## B. 분할 가정

### B-1. 실험 A 분할 `official_partition` (기본)

**논문**: "동일한 전처리 데이터셋을 80:20 비율로 학습/검증용으로 분할"

**가정**: 이 "80:20"은 AI-Hub 공식 Training(141) / Validation(33) 분할이다 → 81:19.

**근거**: `reproduction_spec.md` §2. 보고된 Accuracy가 모두 `k/33`, Recall이 모두 `k/7`로
정확히 분해되고, 실제 Validation이 정확히 33명·양성 7명이다. AUC도 분모 182 = 7×26으로
떨어진다. 174명을 `test_size=0.2`로 무작위 분할하면 test가 35명이 되어 재현되지 않는다.

**대안 변형** (모두 config로 선택 가능):

| 변형 | 설명 | 목적 |
| --- | --- | --- |
| `assumption_variant_random_subject_holdout` | 174명을 층화 무작위 80:20 (test 35명) | 문구를 문자 그대로 읽었을 때 |
| `assumption_variant_random_row_holdout` | 12,183 **행**을 무작위 80:20 | **누수 크기 정량화 전용.** 성능 주장 금지 |

`assumption_variant_random_row_holdout`은 실행 시 **경고를 출력하고**, 결과 JSON에
`leakage_expected: true`, `interpret_as: "leakage_diagnostic_not_performance"`를 기록한다.

---

### B-2. 실험 B·C 분할

**가정**: `StratifiedGroupKFold(n_splits=5)`, `groups = subject_id`, 층화는 이진 라벨.
반복은 `repeats`(기본 5), seed는 `seed + repeat_index`.
실험 C의 inner는 `StratifiedGroupKFold(n_splits=3)`.

**근거**: 프롬프트 §4 권장안 + 저장소 `SangHyo/AGENTS.md` §2의 최소 계약.

---

## C. 모델 가정

### C-1. Random Forest — **전부 미보고**

논문에는 RF 하이퍼파라미터가 **하나도** 없다.

**가정**: 대부분 scikit-learn 기본값을 사용하되, 본문이 각 트리가 information gain으로
분할한다고 설명한 유일한 단서에 맞춰 `criterion: entropy`를 사용한다. 이것도 실제 RF
설정표가 아니라 **본문 설명을 실행 가능하게 옮긴 가정**이다.

```yaml
n_estimators: 100
criterion: "entropy"    # information-gain 서술에서 추론한 가정
max_depth: null
min_samples_split: 2
min_samples_leaf: 1
max_features: "sqrt"
class_weight: null      # 논문이 불균형 보정을 하지 않았음
random_state: <config.seed>
```

`class_weight: null`의 근거: 논문 결론부가 "향후 class weight 조정, focal loss 등을
**적용할 수 있을 것**"이라고 쓴다 → 본 연구에서는 미적용.

---

### C-2. XGBoost — 부분 보고

**논문 명시 (그대로 사용)**:
```yaml
n_estimators: 100
learning_rate: 0.1
subsample: 1.0
colsample_bytree: 1.0
reg_alpha: 0
reg_lambda: 1
eval_metric: "logloss"
```
`use_label_encoder=False`는 xgboost 2.x에서 제거된 인자다. 전달하지 않고,
호환 로그만 남긴다.

**미보고 → 가정** (XGBoost 기본값):
```yaml
max_depth: 6            # 논문은 "최대 깊이"를 탐색했다고만 쓰고 결과값을 누락
objective: "binary:logistic"
random_state: <config.seed>
```

> 논문 본문은 트리 수·최대 깊이·learning rate·subsample을 Random Search로 탐색했다고
> 서술하면서, 결과 문자열에는 `max_depth`가 없다. 탐색 결과 목록이 불완전하다.

**본 재현은 Random Search를 다시 돌리지 않는다.** 논문이 최종 채택값을 제시했으므로
그 값을 고정한다(프롬프트 §1-8, §4-A-6).

---

### C-3. LSTM / Bi-LSTM / 1D-CNN — **전부 미보고**

논문에는 세 신경망의 구조가 **단 하나도** 기재되어 있지 않다. 개념 설명만 있다.

**놓쳤다가 뒤늦게 찾은 단서 — 논문은 Keras를 썼다**

§3.3.2가 쓴 층 이름이 `Conv1D`, `MaxPooling1D`, `AveragePooling1D`다. 이것은
**Keras/TensorFlow 표기**이며 PyTorch는 `Conv1d`, `MaxPool1d`로 쓴다. 따라서 미보고
항목은 **임의의 보수적 값이 아니라 Keras 기본값**으로 두는 것이 덜 발명적이다.

| 항목 | Keras 기본값 | 본 재현 반영 |
| --- | --- | --- |
| `Conv1D(padding=...)` | `'valid'` | `conv_padding: valid` (paper-literal arm) |
| `model.fit(batch_size=...)` | `32` | `batch_size: 32` (기존 16에서 변경) |
| `Adam(learning_rate=...)` | `0.001` | 동일 |
| `pad_sequences(padding=...)` | `'pre'` | `padding: pre` (일치) |
| `LSTM(return_sequences=False)` | 마지막 시점 출력 | `readout: last_hidden` |
| `Dense(1, activation='sigmoid')` | — | 동일 |

**CNN 구조도 논문 서술을 넘어서 있었다**: §3.3.2는 1D-CNN을 정확히 3단계로 적는다 —
"1) Conv1D 계층 → 2) Pooling 계층 → 3) Fully Connected 계층(평탄화)". 즉 conv 블록은
**하나**다. 기존 기본값 `filters: (64, 64)`는 2블록으로 논문 서술을 넘어선 것이었다.
paper-literal arm은 `filters: [64]`를 쓴다.

**가정** (`assumption_variant_minimal_architecture`). 작은 표본(141명)에 맞춘 최소 구조:

| 항목 | LSTM | Bi-LSTM | 1D-CNN |
| --- | --- | --- | --- |
| 층 | LSTM 1층 | Bi-LSTM 1층 | Conv1d ×2 |
| hidden / filters | 64 | 64 (양방향 → 128) | 64, 64 |
| kernel_size | — | — | 3 |
| readout / pooling | 마지막 hidden state | 마지막 forward/backward state 연결 | MaxPool1d(2) ×2 → Flatten |
| dropout | 0.2 | 0.2 | 0.2 |
| head | Linear(64→1) | Linear(128→1) | Flatten된 전체 feature map → Linear(→1) |
| 출력 | sigmoid | sigmoid | sigmoid |

공통 기본값(모두 가정):
```yaml
loss: "bce"                 # 논문 미보고
optimizer: "adam"           # 논문 미보고
learning_rate: 0.001        # 논문 미보고
batch_size: 16              # 논문 미보고. 141명 기준 보수적
max_epochs: 100             # 논문 미보고
early_stopping: true        # 실험 B·C만; 실험 A는 false
early_stopping_patience: 10 # 논문 미보고
early_stopping_metric: auc  # 실험 B·C만 (아래 C-3-1 참조)
validation_fraction: 0.2    # 실험 B·C만. 층화 분할
pos_weight: null            # 논문이 불균형 보정 미적용
```

**실험 A**는 논문에 내부 holdout/early stopping 보고가 없으므로
`early_stopping: false`, `validation_fraction: 0.0`으로 두고 공식 Training **141명 전부**를
고정 100 epoch 학습한다. 100 epoch 자체는 여전히 가정이다. 실험 B·C의 early stopping은
재현 대상이 아니라 검증설계 확장을 위한 계산상 가정이다.

### C-3-1. 기존 실험 A early stopping 실패와 수정

**논문**: early stopping 자체를 언급하지 않는다. 전부 본 재현의 가정이다.

**기존 코드의 실패**: `20260803_014223_utc`는 논문에 없는 20%(28명) 모니터 분할을
Training 141명에서 떼어 optimizer에는 113명만 넣었다. CNN은 12 epoch 뒤 epoch 1을
복원했고, 33명 전부를 음성으로 분류했다. 이는 논문 방법을 그대로 재구성했다고 보기
어렵다.

**수정**:

1. **실험 A**: 내부 분할과 early stopping을 제거하고 141명 전부를 고정 epoch 학습한다.
2. **실험 B·C**: early stopping을 유지하되 모니터 분할을 층화하고 ROC-AUC를 감시한다.
3. 작은 모니터 분할에서 AUC가 동률이면 loss로 tie-break한다.

실험 A의 100 epoch와 실험 B·C의 모니터 정책은 모두 논문 미보고 가정이다. 향후 설정을
바꿀 때 33명 Validation 성능으로 최적 epoch를 고르면 안 된다.

**남은 안전장치**: optimizer step이 없거나, 최종 보존된 파라미터가 초기값에서 변하지
않았거나, 학습 loss/파라미터 변화량이 비유한이면 `degenerate_training: true`가 기록되고
`FINAL_REPORT.json`과 실행 로그에 경고가 뜬다. epoch 0은 첫 **학습 후** epoch이므로 그
인덱스만으로 미학습 판정을 하지 않는다.

**중요**: 미보고 설정 때문에 정확한 수치 재현 가능 여부는 판정할 수 없다. 다만 같은
33명에서 1D-CNN이 전부 음성을 예측하고 AUC/F1/Recall과 모델 순위가 크게 다르면,
논문의 **핵심 결론은 재구성되지 않은 것**으로 별도 판정한다.

---

### C-4. 시퀀스 길이·패딩 `assumption_variant_sequence_length`

**논문**: "시계열 데이터의 패딩 구조"라는 언급만. 길이·방향·마스킹 규칙 전부 미보고.

**가정**:
```yaml
sequence_length: 122        # 실험 A: 전체 코호트 최대 관측일수, 재구성 전용 가정
padding: "pre"              # 앞쪽 0-padding, 최근 관측이 시퀀스 끝
truncation: "last"          # 길면 최근 T일 사용
mask_padding: true          # 관측/달력-gap과 외부 padding을 boolean mask로 구분
min_observations: 1         # 최소 관측일수 미달 피험자 제외 (실측상 최소 35일이라 무해)
```

실험 A에서 122는 논문이 실제로 썼다고 확인된 값이 아니다. 다만 논문이 전체 데이터를 먼저
3차원 텐서로 만든 뒤 분할한 것으로 읽힐 여지가 있고, 기존 train-derived T=120이 Validation
양성 1명의 첫 2일을 잘랐기 때문에 **full-cohort-shape 재구성 가정**으로 명시한다. 일반화
성능을 주장하는 실험 B·C에서는 `sequence_length: max`를 각 fold의 training에서만 정하고,
test가 shape를 결정하지 못하게 한다.

LSTM/Bi-LSTM은 `pack_padded_sequence`를 위해 외부 padding만 뒤로 옮기되 내부 달력 gap의
위치를 보존한다. Flatten을 쓰는 1D-CNN은 위치 민감하므로 설정된 pre/post padding을 그대로
사용한다. 관측되지 않은 외부 padding과 내부 달력 gap은 표준화 후 정확히 0으로 복원한다.

**대안**: `sequence_length: 35`(전 피험자 실관측 보장 최소값), `56`, `"median"`.
프롬프트 §4-C에 따라 실험 C의 inner CV에서만 탐색 가능하며,
**민감도 분석으로 표시**한다.

**비연속 날짜 처리** `assumption_variant_gap_handling`:

실측상 연속 간격 비율은 91.0%, train 141명 중 **131명**이 결측일을 갖는다(최대 23일).

| 모드 | 설명 |
| --- | --- |
| `compress` (기본) | 결측일을 **무시**하고 관측 행만 날짜순으로 이어붙인다 |
| `calendar` | 달력 격자에 채우고 결측일은 마스크 처리 (시퀀스가 최대 124일로 길어짐) |

`compress`가 기본인 이유: 논문이 결측일 보간을 언급하지 않았고, 관측 행만 텐서로 쌓는
것이 "정규화 후 3차원 텐서"라는 서술에 가장 가깝다.

---

## D. 평가 가정

### D-1. 결정 임계값

**논문**: 미보고. Accuracy/F1/Recall을 보고하므로 임계값은 존재했다.

**가정**:
- 실험 A: **0.5**
- 실험 B: **0.5** 고정 (사전 고정)
- 실험 C: **inner CV에서만** 선택. outer test는 임계값 결정에 절대 미사용

---

### D-2. 피험자 단위 통합

**논문**: 실험 A는 이미 1인 1예측이므로 통합이 불필요하다.

**가정**: 일별 예측이 여러 개 나오는 표현(`daily_record`)에서는 피험자별 **예측확률 평균**을
주 평가값으로 한다. 중앙값·다수결은 민감도 분석으로만 제공한다(프롬프트 §8).

---

### D-3. 논문 AUC와의 비교 가능성

논문은 ROC-AUC만 보고한다. 본 재현은 PR-AUC·Brier 등을 추가 산출하지만,
**논문 대비 비교표에는 논문이 보고한 4개 지표(Accuracy, AUC, F1, Recall)만** 넣는다.
나머지는 별도 열로 분리한다.

---

## E. 실행·환경 가정

| 항목 | 가정 | 근거 |
| --- | --- | --- |
| random seed | 42 (config 중앙관리) | 논문 미보고 |
| 반복 횟수 | 실험 B `repeats: 5`, 실험 C `repeats: 3` | 정확도와 계산예산 절충 |
| device | CUDA 있으면 GPU, 없으면 CPU 자동 | 프롬프트 §7 |
| 딥러닝 프레임워크 | PyTorch | 저장소에 이미 torch 계열(`pytorch-tabnet`) 사용 이력 |
| 결과 저장 위치 | Colab: `/content/drive/MyDrive/reproduction_lim_2025_result/<UTC_RUN_ID>/` | `SangHyo/AGENTS.md` §6 |
| 피험자 ID | EMAIL의 SHA-256 앞 12자 | `SangHyo/AGENTS.md` §6 (원본 이메일 저장 금지) |

---

## F. 명시적으로 **하지 않은** 것

프롬프트 §1의 금지사항과 §4의 "과도한 탐색 금지"에 따라 아래는 구현하지 않았다.

- Optuna 및 대규모 하이퍼파라미터 탐색
- 논문에 없는 추가 특징공학(rolling window, 주파수 변환, `CONVERT` 5분 계열 등)
- SMOTE 등 오버샘플링 (논문 미적용)
- 논문에 없는 모델(Transformer, TabNet, LightGBM 등) 추가
- 33명 Validation을 새 독립 test로 재사용하는 주장
  — `SangHyo/AGENTS.md` §2-5에 따라 이 33명은 **historical benchmark**이며,
    실험 A의 재현 대상일 뿐 새로운 일반화 근거가 아니다.

---

## G. 가정 변경 시 확인할 것

`configs/*.yaml`에서 위 가정을 바꿀 때는 다음을 반드시 함께 갱신한다.

1. 이 문서의 해당 항목
2. 결과 JSON의 `assumptions` 블록 (코드가 config에서 자동 기록)
3. `unresolved_questions.md`에서 해당 질문이 해소되었는지
