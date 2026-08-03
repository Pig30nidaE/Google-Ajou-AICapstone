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

**가정**: 논문 코드(`groupby.mean(numeric_only=True)` + drop 목록)를 **문자 그대로**
실행한 결과인 **49개**를 기본 특징집합으로 한다.

**근거**: 논문 서술(58)과 논문 코드(49)가 모순될 때, **코드가 실제로 실행된 것**이다.
서술을 맞추려면 존재하지 않는 컬럼 2개를 만들어내고 문자열 6개를 임의로 수치화해야 하는데,
그 방법이 논문에 없으므로 재현 불가능하다.

**대안**: `feature_set: paper_declared_58` — 논문이 나열한 56개 중 실재하는 54개만 쓰고
문자열 6개를 파생 수치로 변환. **명시적 assumption_variant이며 기본값 아님.**

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

**가정**: scikit-learn 기본값을 보수적으로 사용.

```yaml
n_estimators: 100
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

**가정** (`assumption_variant_minimal_architecture`). 작은 표본(141명)에 맞춘 최소 구조:

| 항목 | LSTM | Bi-LSTM | 1D-CNN |
| --- | --- | --- | --- |
| 층 | LSTM 1층 | Bi-LSTM 1층 | Conv1d ×2 |
| hidden / filters | 64 | 64 (양방향 → 128) | 64, 64 |
| kernel_size | — | — | 3 |
| pooling | — | — | MaxPool1d(2) → GlobalAvgPool |
| dropout | 0.2 | 0.2 | 0.2 |
| head | Linear(→1) | Linear(→1) | Linear(→1) |
| 출력 | sigmoid | sigmoid | sigmoid |

공통:
```yaml
loss: "bce"                 # 논문 미보고
optimizer: "adam"           # 논문 미보고
learning_rate: 0.001        # 논문 미보고
batch_size: 16              # 논문 미보고. 141명 기준 보수적
max_epochs: 100             # 논문 미보고
early_stopping_patience: 10 # 논문 미보고
early_stopping_metric: auc  # 논문 미보고 (아래 C-3-1 참조)
validation_fraction: 0.2    # 논문 미보고. 층화 분할
pos_weight: null            # 논문이 불균형 보정 미적용
```

### C-3-1. early stopping 모니터 지표를 AUC로 정한 이유

**논문**: early stopping 자체를 언급하지 않는다. 전부 본 재현의 가정이다.

**처음 선택(loss)의 실패**: 141명 중 20%인 28명을 모니터 분할로 떼고 BCE loss를
감시했더니, loss가 epoch 0에서 최소였고 이후 계속 올라 early stopping이
**초기화 직후 가중치를 복원**했다. train loss는 0.713 → 0.481로 내려가고 있었는데도
보고된 것은 사실상 학습되지 않은 모델이었다. 게다가 모니터 분할이 **층화되지 않아**
양성 비율이 seed마다 요동쳤다.

**수정**:

1. 모니터 분할을 **층화**한다. 실제 데이터에서 양성 수가 11로 고정된다.
2. 모니터 지표를 **ROC-AUC**로 바꾼다. 주 보고 지표가 AUC이고, 28명 BCE loss는
   신경망이 확신을 갖기 시작하면 순위가 좋아지는 중에도 올라간다.
3. **AUC 동점 시 loss로 tie-break**한다. 작은 모니터 분할에서 AUC는 1.0에 포화될 수
   있고, 그러면 다시 epoch 0이 복원된다.

**선택 근거의 독립성**: 이 결정은 **학습 동역학**(best epoch, 모니터 분할 내부 점수)만
보고 내렸다. 33명 test AUC를 보고 고르지 않았다. test로 골랐다면 그것이야말로 이
프로젝트가 막으려는 누수다.

**남은 안전장치**: 그럼에도 epoch 0이 복원되면 `degenerate_training: true`가 기록되고
`FINAL_REPORT.json`의 `degenerate_training_models`와 실행 로그에 경고가 뜬다.
해당 모델 수치는 성능으로 인용하지 않는다.

**중요**: 이 구조로 논문의 1D-CNN AUC 0.810이 재현되지 않아도, 그것은 재현 실패가 아니라
**논문이 구조를 보고하지 않았기 때문**이다. 결과 보고 시 반드시 함께 기재한다.

---

### C-4. 시퀀스 길이·패딩 `assumption_variant_sequence_length`

**논문**: "시계열 데이터의 패딩 구조"라는 언급만. 길이·방향·마스킹 규칙 전부 미보고.

**가정**:
```yaml
sequence_length: "max"      # 데이터 최대 관측일수(174명 기준 122)
padding: "pre"              # 앞쪽 0-padding, 최근 관측이 시퀀스 끝
truncation: "last"          # 길면 최근 T일 사용
mask_padding: true          # LSTM은 pack_padded_sequence, CNN은 마스크 곱
min_observations: 1         # 최소 관측일수 미달 피험자 제외 (실측상 최소 35일이라 무해)
```

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
| 반복 횟수 | 실험 B·C 모두 `repeats: 5` 기본 | 프롬프트 §4-B 권장 |
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
