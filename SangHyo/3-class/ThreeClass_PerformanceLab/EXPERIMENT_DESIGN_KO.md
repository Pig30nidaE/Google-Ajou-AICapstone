# ThreeClass PerformanceLab 사전등록 실험 설계

- 문서 버전: `performance_lab_preregistered_v1`
- 사전등록일: 2026-07-16
- 주 과제: wearable lifelog 기반 `CN(0) / MCI(1) / DEM(2)` 동시점 분류
- 개발 cohort: `Data/1.Training`의 141명만 사용
- 주 평가 단위: subject level
- 주 평가 지표: 3-class Macro F1

이 문서는 다음 실행에서 결과를 보기 전에 고정할 데이터 표현, 후보 모델,
교차검증, 선택 및 중단 규칙을 정의한다. 실행 결과를 본 뒤 같은 run 안에서
feature, 모델, seed, class weight, threshold, ensemble weight를 추가하거나 바꾸지
않는다. 변경이 필요하면 별도 버전과 별도 fresh split을 사용하는 새 실험으로
등록한다.

## 1. 출발점과 검증할 가설

이전 `ThreeClass_NextStage` 실행에서 calendar-day 값, 관측 mask,
`days-since-observed`를 입력한 `mask_tcn_35d` 후보는 이미 확인한 outer 결과에서
Nested Macro F1 `0.404 +/- 0.101`로 관찰되었다. 이 값은 adaptive pipeline을
평가한 outer prediction을 사후 비교하면서 발견한 결과다. 따라서 독립적으로
확립된 성능이나 새 실험의 보장된 목표값이 아니다. 이 후보를 그 결과에 근거해
같은 outer fold의 우승 모델로 승격하면 outer fold를 모델 선택에 재사용하게 된다.

이번 실험에서 `mask_tcn_35d`는 과거의 49개 daily value, value/mask/delta 변환,
TCN 구조와 fold 내부 epoch-selection/refit을 고정한 **legacy comparator**로만
사용한다. 다만 새 prediction-index 계약에 맞춰 exact end-timestamp future guard와
결정론적 main-sleep tie rule을 추가한다. 따라서 이는 chronology-corrected fresh-seed
비교이지 과거 0.404의 완전한 수치 재현이 아니며, primary pipeline 선택에는
참여하지 않는다.

이번 실험의 주 가설은 다음과 같다.

> 수집 일수, calendar gap, padding 길이를 모델에 직접 노출하지 않고, 최근 관측을
> 절대 날짜가 아닌 observed-event 순서로 표현하면 acquisition coverage shortcut을
> 줄이면서 실제 activity/sleep 패턴의 out-of-subject 분류력을 더 안정적으로 평가할
> 수 있다.

이는 아직 측정되지 않은 가설이다. 이 설계 문서는 성능 개선을 주장하지 않으며,
아래 stop/go 조건을 통과한 fresh nested-CV 결과가 있을 때만 개선 가능성을
논의한다.

단, 이번 표현과 feature family의 기획에는 Training 141명 전체의 class별 aggregate
EDA가 사용되었다. Learned imputation/model selection은 outer fold 밖으로 새지 않지만,
사람의 feature-family 결정까지 포함해 outer subject가 완전히 미노출된 것은 아니다.
따라서 nested-CV는 이 사전 고정 설계에 조건부인 leakage-audited 내부 추정치로
보고하며, 독립 확증으로 표현하지 않는다.

## 2. 예측 계약

### 2.1 Target과 class mapping

| Canonical ID | 정규화할 원시 label |
| --- | --- |
| `0` | `CN` |
| `1` | `MCI` |
| `2` | `Dem`, `DEM`, `Dementia` |

- 목표는 미래의 치매 전환 시점이나 발병 위험을 예측하는 예후 모델이 아니다.
- 각 subject의 마지막 유효 activity row의 `activity_day_end` timestamp를 index로
  삼아, 그 시점에 부여된 인지상태를 분류하는 **동시점 상태 분류**다. Timestamp는
  원문 `+09:00` offset을 보존해 비교한다.
- 날짜는 event를 정렬하고 index를 정하는 데만 사용한다. 연도, 월, 일, 요일,
  절대 timestamp, 수집 시작일/종료일은 feature로 제공하지 않는다.
- 수면의 시계시간은 생체 리듬 값이므로 필요한 경우 `sin/cos` clock phase로만
  표현할 수 있다. calendar date와 timezone 식별값은 feature가 아니다.

### 2.2 Cohort와 데이터 역할

- Discovery, EDA, feature 검토, 모든 model selection에는 Training 141명
  `(CN 85, MCI 47, DEM 9)`만 사용한다.
- 한 subject는 모든 split에서 정확히 한쪽에만 존재한다.
- 불완전한 7일/35일 관측 때문에 subject를 제거하지 않는다. 최종 train subject
  수는 반드시 141명이어야 한다.
- `Data/2.Validation`은 역사적으로 여러 번 사용된 benchmark다. 독립 holdout으로
  부르지 않으며 모델 선택의 근거로 사용하지 않는다.

## 3. Validation의 물리적 격리

실행은 역할이 다른 두 notebook으로 분리한다.

```text
01_train_only_discovery_colab.ipynb
  - Data/1.Training만 경로로 받음
  - Data/2.Validation 경로, source, label을 참조하거나 fingerprint하지 않음
  - EDA, nested CV, negative control, 최종 Training refit과 freeze 수행

02_frozen_benchmark_colab.ipynb
  - GO 판정과 frozen artifact가 존재할 때만 실행 가능
  - frozen hash 검증 후 validation source를 label 없이 deterministic transform
  - label-free prediction과 input manifest를 먼저 저장
  - 명시적 acknowledgment 뒤 validation label을 한 번만 로드해 평가
```

Discovery notebook에는 validation 관련 경로 상수나 자동 탐색 코드 자체를 두지
않는다. `02_frozen_benchmark_colab.ipynb`는 feature, model, weight, seed,
preprocessor, class rule을 변경할 수 없으며 동일 frozen run의 완료 marker가 있으면
재평가를 거부한다. NO-GO run에서는 validation label을 열지 않는다.

## 4. 허용 데이터와 금지 feature

### 4.1 허용

- Activity/Gait source의 센서 및 일별 요약값
- Sleep source의 센서 및 일별 요약값
- 원시 1분/5분 sequence에서 결정론적으로 계산한 생체·행동 요약
- 수면 단계 비율·entropy·transition, HR/HRV/호흡, HR drop, activity intensity,
  MET, steps, rest, score 계열
- clock-of-day에서 계산한 순환형 `sin/cos` feature

### 4.2 금지

- MMSE `TOTAL`, `Q*`, `Q12_TOTAL` 및 기타 cognitive-test 값
- `DIAG_NM`, `DIAG_SEQ`와 source 안의 중복 진단 필드
- `EMAIL`, `SAMPLE_EMAIL`, 원본 subject ID, hash 전 ID
- `MMSE_NUM`, `DOCTOR_NM`, sample order, 파일 row order
- `sleep_period_id`와 장비/record 식별자
- 절대 날짜, 연도, 월, 요일, 수집 시작/종료 날짜
- validation에서만 발견된 column이나 validation 분포로 선택한 rule

원본 subject ID는 modality join과 split 구성 중 메모리에서만 사용한다. 모든 저장
artifact에는 secret-key 기반 subject hash만 남긴다. 비밀 key는 출력하거나
artifact에 저장하지 않는다.

## 5. Primary representation: observed-event index

### 5.1 기본 원칙

Primary representation은 calendar에 빈 날을 채운 길이, 관측일 수, 최대 gap,
padding mask를 모델에 주지 않는다. 각 modality 안에서 유효한 관측 event를 시간
순서로 정렬하고 최근 event를 기준으로 상대 순서를 부여한다.

- Activity: 최근 최대 28개 유효 activity event
- Sleep: 최근 최대 28개 main-sleep event
- Activity event는 `activity_day_end <= prediction_index`, sleep event는
  `sleep_bedtime_end <= prediction_index`인 행만 허용한다. 선택 tensor/summary에
  index 이후 event가 하나라도 있으면 즉시 중단한다.
- 같은 현지 wake-date에 수면 episode가 여러 개이면 가장 긴 main sleep 하나를
  대표 event로 사용한다. duration 동률은 bedtime start, bedtime end의 시간순으로
  결정하며 `sleep_period_id`와 원본 row order는 사용하지 않는다.
- Index 이후라 제외된 sleep/activity event 수와 duplicate episode 수는 aggregate
  audit에만 저장하고 primary feature에는 넣지 않는다.
- event 위치는 절대 날짜가 아닌 정규화 observed-event rank `[0, 1]`
- 28개보다 적은 subject도 제거하지 않음

여기서 “coverage-invariant”는 관측 수가 생체값의 추정 오차에 미치는 영향까지
수학적으로 완전히 제거한다는 뜻이 아니다. count, gap, mask를 primary model에
명시적으로 제공하지 않고 동일 차원의 생체 표현을 사용한다는 운영적 계약이다.
남을 수 있는 간접 coverage 신호는 별도 negative control로 감사한다.

### 5.2 `event_summary_v1`

각 daily 생체 feature에 대해 최근 최대 28개 observed event에서 다음 고정 통계를
계산한다.

- median
- 10% trimmed mean
- p10, p90
- IQR, MAD
- normalized event-rank에 대한 robust slope
- 마지막 절반 median - 첫 절반 median

관측 event 수, valid ratio, span day, missing-day count는 넣지 않는다. 통계를
계산할 원시값이 부족하면 결측으로 유지하고 fold-train imputer가 처리한다.
feature 정의는 label을 보기 전에 고정하며 supervised ANOVA/SHAP ranking을 하지
않는다.

### 5.3 `event_sequence28_v1`

Activity와 sleep 각각의 observed event sequence를 정규화 rank 위의 28개 위치로
결정론적으로 보간한다.

- 모델 입력에 calendar gap, padding mask, observed count, days-since-observed를
  추가하지 않는다.
- feature별 원시 결측은 fold-train median/IQR 계약으로 처리한다.
- activity와 sleep은 독립 rank 축으로 전처리한 뒤 두 encoder의 pooled embedding을
  late fusion한다.
- 보간 방법, feature order, 28-step 길이는 run 중 변경하지 않는다.

## 6. Coverage-only negative control

`coverage_only_v1`은 생체값을 전혀 사용하지 않고 수집 프로토콜만으로 label이
예측되는지를 검사한다.

허용되는 negative-control feature:

- activity observed day 수
- sleep observed night 수
- paired day 수
- modality별 관측 span 길이
- 최대/평균 calendar gap
- raw field missing ratio
- main-sleep episode 중복 수

모델은 median imputation + scaling + multinomial logistic regression 하나로
고정한다. 이 control은 primary 후보 선택에 참여할 수 없고 ensemble에도 들어갈 수
없다. 분류력이 primary와 비슷하면 생체 패턴이 아니라 acquisition protocol을
학습했을 위험으로 판정한다.

## 7. 고정 candidate set

Primary selection에 참여할 후보는 다음 네 개뿐이다.

| Candidate | 입력 | 고정 목적 |
| --- | --- | --- |
| `event_elastic_v1` | `event_summary_v1` | 저분산 선형 기준선 |
| `event_extra_trees_v1` | `event_summary_v1` | 작은 표본용 비선형 tree 기준선 |
| `event_tcn28_v1` | `event_sequence28_v1` | coverage-invariant temporal 후보 |
| `event_elastic_tcn_equal_v1` | 위 두 확률의 `0.5 / 0.5` | 사전 고정 보완성 후보 |

고정 comparator/control:

| 이름 | 역할 | Selection 참여 |
| --- | --- | --- |
| `mask_tcn_35d_legacy_v1` | 과거 post-hoc TCN의 chronology-corrected fresh-seed 비교 | 불가 |
| `coverage_only_v1` | acquisition shortcut negative control | 불가 |
| `class_prior_v1` | training-fold class prior 확률 기준선 | 불가 |

이번 run에는 TabPFN, MiniRocket, LDA, hierarchical/pairwise model, SMOTE,
synthetic augmentation, 추가 boosting 후보를 넣지 않는다. 결과를 본 뒤 같은 run에
후보를 추가하지 않는다.

### 7.1 고정 모델 계약

`event_elastic_v1`

- fold-local median imputation
- fold-local RobustScaler
- multinomial logistic regression, elastic-net
- `C=0.1`, `l1_ratio=0.5`, `class_weight="balanced"`
- raw softmax probability와 argmax 사용

`event_extra_trees_v1`

- fold-local median imputation
- `n_estimators=1000`, `max_depth=5`, `min_samples_leaf=4`
- `max_features=0.35`, `class_weight="balanced_subsample"`
- probability는 tree 평균, decision은 argmax

`event_tcn28_v1`

- modality별 1D encoder와 late fusion
- hidden 24, kernel size 3, dilation `1, 2, 4, 8`
- residual block당 convolution 2개, GroupNorm, GELU, dropout 0.35
- AdamW `lr=8e-4`, weight decay `2e-3`, gradient clipping 1.0
- sqrt-balanced fold-train class weight, label smoothing 0.05
- epoch 120 고정, early stopping과 내부 holdout 사용 안 함
- stochastic refit seed 두 개의 probability 평균

`event_elastic_tcn_equal_v1`

- `event_elastic_v1`과 `event_tcn28_v1` probability를 정확히 0.5씩 평균
- weight 탐색 없음

`mask_tcn_35d_legacy_v1`

- 과거 allowlist의 daily value 49개(activity 18 + sleep 31)를 그대로 사용하고,
  35 calendar-day value + observed mask + normalized delta의 147 channel 입력
- non-wear, daily sleep count, sequence length/validity 파생값과 mask/delta가 포함되므로
  acquisition shortcut을 의도적으로 보존한 감사용 비교기
- exact `activity_day_end` index 이후 수면을 먼저 제외하고, duration/start/end/digest
  순서로 main sleep을 고르는 chronology-corrected 규칙
- hidden 24, dilation `1, 2, 4, 8`, dropout 0.35의 기존 구조를 유지
- 각 fold-train의 25% stratified epoch-selection subset에서 최대 300 epoch,
  patience 30으로 Macro F1 epoch를 선택하고, seed에 100003을 더해 fold-train 전체를
  선택 epoch만큼 처음부터 refit
- 성능을 보고 value allowlist, mask/delta, loss, hidden, epoch rule, threshold를
  바꾸지 않음
- primary selection과 final bundle에 포함할 수 없음

공통 stochastic model seed는 `[17011, 27011]`로 고정하고 probability를 평균한다.
모든 class weight는 해당 fold의 train subject count만으로 계산한다.

## 8. Fold-local preprocessing 계약

다음 순서를 모든 inner fold, outer fold, full refit에서 동일하게 적용한다.

1. subject 목록을 먼저 split한다.
2. 원시 sequence parsing, 날짜 정렬, main-sleep 선택처럼 학습되지 않는 결정론적
   변환만 수행한다.
3. all-missing/constant feature 제거 규칙을 fold-train에서 학습한다.
4. median, clipping bound가 필요한 경우 0.5/99.5 percentile, scaler 통계를
   fold-train에서만 계산한다.
5. 변환된 feature 이름과 순서를 저장하고 inner-valid/outer-valid에 그대로 적용한다.
6. class weight, model fitting과 stochastic seed ensemble도 fold-train만 사용한다.
7. finite value, feature dimension/order, label consistency, subject overlap 0을 assert한다.

Supervised feature selection, SHAP ranking, validation-aware pruning은 이번 run에서
수행하지 않는다. Oversampling도 하지 않는다.

## 9. 사전등록 CV

### 9.1 Split

- Outer: fresh seed 5개 x stratified 3-fold
- Outer seeds: `[137, 1009, 2027, 4099, 8191]`
- 각 outer-train 내부: seed 2개 x stratified 3-fold inner OOF
- Inner seed 생성 규칙:
  - `inner_seed_1 = outer_seed + 50021`
  - `inner_seed_2 = outer_seed + 90001`
- split 대상은 141개 unique subject row다.
- 모든 outer-valid fold에 세 class가 있어야 하며 DEM 3명이 들어가는지 저장·검증한다.
- 모든 inner-valid fold에도 세 class가 있어야 하며 DEM 2명이 들어가는지 저장·검증한다.

이 seed들은 이전 주요 실행의 `42`, `2024`, `3407`, `811`, `7`과 겹치지 않는
fresh evaluation seed다.

### 9.2 Inner selection

각 outer-train 안에서 네 primary candidate의 repeated inner-OOF probability를 만든다.
selection 순서는 다음과 같다.

1. mean subject-level Macro F1이 가장 높은 후보
2. 최고점과 차이가 `0.01` 이내이면 더 단순한 후보 선택:
   `event_elastic_v1` -> `event_extra_trees_v1` -> `event_tcn28_v1`
   -> `event_elastic_tcn_equal_v1`
3. 복잡도가 같으면 balanced accuracy가 높은 후보
4. 여전히 같으면 multiclass log loss가 낮은 후보
5. 완전 동률이면 이름의 사전등록 표 순서

### 9.3 Outer 평가와 주 보고값

각 outer fold에서는 inner에서 선택한 rule을 outer-train 전체에 동일 계약으로 refit한
뒤 outer-valid probability를 한 번 생성한다.

주 보고값은 outer seed별로 141명 전체를 한 번씩 덮는 complete OOF prediction의
Macro F1 다섯 개에 대한 평균과 표본 표준편차다. 함께 저장·보고할 값:

- 5개 repeat별 Macro F1과 confusion matrix
- 15개 outer fold별 subject/class count와 Macro F1
- balanced accuracy, accuracy, multiclass log loss
- class별 precision/recall/F1/support
- class별 OVR AUROC/AUPRC where defined
- class-prior, coverage-only, legacy calendar TCN과의 paired delta
- candidate selection 빈도와 inner-outer gap

두 outer repeat의 probability를 다시 평균한 지표는 더 많은 split-model을 섞는
diagnostic일 뿐 주 성능으로 보고하지 않는다.

## 10. 금지된 사후 결정

- class별 scale, class별 threshold, MCI/DEM bias grid를 사용하지 않는다.
- global temperature, isotonic, Platt calibration을 추가하지 않는다.
- 모든 candidate의 decision은 원 probability의 `argmax`다.
- adaptive ensemble weight, stacking, fold별 후보 추가를 사용하지 않는다.
- benchmark 결과로 model, feature, seed, epoch를 바꾸지 않는다.
- `mask_tcn_35d_legacy_v1`이 높게 나와도 이번 run의 primary winner로 승격하지 않는다.
  새 독립 실험의 사전등록 후보로만 제안할 수 있다.

## 11. Final selection과 freeze

### 11.1 Primary candidate 결정

Nested 결과에서 다음 순서로 한 candidate를 결정한다.

1. 15개 outer fold의 inner selection에서 선택된 빈도가 가장 높은 primary candidate
2. 빈도가 같으면 deployment-matched outer OOF Macro F1 평균이 높은 candidate
3. 평균 차이가 0.01 이내이면 더 단순한 candidate
4. 그래도 같으면 log loss가 낮은 candidate

Legacy comparator와 negative control은 선택 대상이 아니다.

### 11.2 Full Training freeze

- 선택 candidate를 Training 141명 전체에 고정 seed `[17011, 27011]`로 refit한다.
- full-training feature/preprocess 통계와 feature order를 저장한다.
- selection OOF는 final rule 선택에도 사용되었으므로 낙관적 diagnostic으로 명시한다.
- 주 일반화 추정치는 계속 repeated nested-CV다.
- freeze JSON에 code hash, config hash, input hash, split seed, class mapping, prediction
  index, feature manifest, preprocessing 통계, model seed를 기록한다.
- freeze 이후 artifact hash가 하나라도 달라지면 benchmark notebook은 중단한다.

## 12. Stop/Go 규칙

현재까지의 정직한 historical reference는 nested Macro F1 약 `0.358`이다. 이는 새
fresh split과 동일한 paired estimate가 아니므로 단독 채택 근거로 사용하지 않고
contextual threshold로만 사용한다.

### 12.1 즉시 STOP

다음 중 하나라도 발생하면 run은 NO-GO로 동결하고 validation label을 열지 않는다.

- Training subject가 141명이 아니거나 class count가 `85/47/9`와 다름
- modality label 불일치, subject overlap, forbidden feature, PII artifact 발견
- fold-local fit 위반 또는 transformed NaN/Inf 발생
- outer/inner fold 중 한 class가 빠짐
- coverage-only mean Macro F1이 selected primary보다 `0.03` 미만 낮음
- coverage-only와 selected primary의 repeat별 성능 방향이 불안정해 shortcut을
  배제할 수 없음
- selected primary가 MCI 또는 DEM을 5 repeat 중 2회 이상 전혀 맞히지 못함
- candidate selection OOF와 nested primary estimate의 차이가 `0.03`을 초과
- 실행 중 사전등록 config를 변경함

### 12.2 GO를 논의할 최소 조건

아래를 모두 만족할 때만 frozen benchmark 실행을 허용한다.

- repeated nested-CV Macro F1 평균이 `0.378` 이상
  - historical 0.358 대비 약 +0.02를 요구하는 사전등록 gate이며 개선을 미리
    주장하는 값이 아님
- final candidate가 `event_elastic_v1`보다 복잡한 경우에만, Elastic 대비
  repeat-level paired Macro F1 delta가 5회 중 최소 4회 양수
- 같은 조건에서만 15 outer fold 중 최소 10개에서 paired delta가 양수
- 같은 조건에서만 MCI와 DEM의 repeat-mean F1이 각각 `event_elastic_v1`보다
  0.02 이상 악화되지 않음
- selected primary와 coverage-only Macro F1 차이가 최소 0.03
- selected primary log loss가 같은 fold의 class-prior log loss보다 0.05를 넘게
  악화되지 않음
- repeat Macro F1 표준편차가 0.10 미만이고 개선 방향이 한두 subject 적중만으로
  설명되지 않음
- selection OOF - nested gap 절댓값이 0.03 이하

여기서 용어와 계산은 다음처럼 고정한다.

- `nested primary`: 각 outer fold에서 inner CV가 고른 후보를 그 outer-valid에
  적용해 만든 deployment-matched selection pipeline이다. 5개 repeat별 완전 OOF
  Macro F1의 평균이 주 일반화 추정치다.
- `final candidate`: 15개 outer fold에서 가장 자주 선택된 단일 selectable
  candidate다. 빈도 동률이면 candidate별 outer-repeat Macro F1 평균, 0.01 이내면
  단순성 순서, 이후 log loss 순서를 적용한다.
- `selection OOF`: 위 final candidate의 candidate-specific outer OOF를 repeat별로
  평가한 평균이다. Final candidate 결정에도 사용됐으므로 낙관적 diagnostic이다.
- GO에는 nested primary와 final candidate의 Macro F1 평균이 모두 0.378 이상이어야
  한다. `selection OOF - nested primary` 절댓값도 0.03 이하여야 한다.
- Coverage 방향 안정성은 final candidate Macro F1이 coverage-only보다 큰 repeat가
  5회 중 최소 4회이고, 두 평균의 차이가 최소 0.03인 것으로 정의한다.
- Final candidate의 MCI/DEM repeat-mean F1은 elastic보다 각각 0.02 넘게 낮을 수
  없다. 이 incremental 비교는 final candidate가 elastic보다 복잡할 때만 적용한다.
  MCI 또는 DEM recall이 0인 repeat는 final candidate 종류와 무관하게 class별 최대
  1회만 허용한다.
- Final candidate log loss는 같은 outer prediction의 class-prior log loss보다
  0.05 넘게 나쁠 수 없다.
- 한두 subject 적중 민감도는 final-candidate-minus-elastic Macro F1 delta로
  계산한다. 각 제거 집합마다 5개 outer-repeat에서 계산한 delta의 평균을 그
  subject 또는 subject-pair의 delta 하나로 정의한다. Final candidate가
  elastic보다 복잡할 때, 141개의
  leave-one-subject-out delta 중 90% 이상, 그리고 가능한 모든 subject pair(최대
  10,000개; 초과 시 keyed-hash 정렬의 앞 10,000쌍) 제거 delta 중 90% 이상이
  양수여야 한다. Final candidate가 `event_elastic_v1` 자체이면 이 네 가지
  incremental-complexity gate(repeat/fold win, MCI/DEM delta, leave-one/two)는
  `not applicable/pass`로 기록한다. 이 예외는 단순 모델 선호 원칙을 지키기 위한
  것이며, 절대 Macro F1, coverage 차이, 분산, zero-recall, log-loss, nested gap은
  그대로 모두 통과해야 한다. 원본 ID나 제거 대상은 저장하지 않고 비율만 저장한다.
- Run identity는 config, core, notebook, requirements, 설계 문서, Training input,
  smoke/full mode, global seed, 실제 package/runtime 계약과 비밀 hash key의 비가역
  verifier를 포함한다. 기존 checkpoint의 identity가 완전히 같지 않으면 resume을
  거부한다.

조건을 통과해도 “성능 개선 확정”이 아니라 Training cohort 안에서의 재현 가능한
개선 후보로 표현한다. 최종 확증에는 수집 프로토콜이 다른 독립 외부 cohort가
필요하다.

## 13. Colab 실행 산출물

### 13.1 Discovery notebook 필수 artifact

```text
run_config.json
nested_cv_config.json
environment.json
code_snapshot/
  01_train_only_discovery_colab.ipynb
  performance_lab_core.py
data_audit.json
training_input_manifest.json
feature_manifest_event_summary.json
feature_manifest_event_sequence.json
coverage_audit.json
fold_assignments_hashed.csv
inner_candidate_metrics.csv
inner_fold_split_audit.csv
outer_fold_metrics.csv
outer_repeat_metrics.csv
candidate_outer_predictions_hashed.parquet
coverage_negative_control_metrics.json
legacy_mask_tcn_metrics.json
nested_cv_report.json
selection_report.json
stop_go_decision.json
FINAL_TRAINING_REPORT.json
privacy_audit_pre_freeze.json
privacy_audit.json
TRAINING_COMPLETE.json
```

아래 네 파일은 Full run이 GO gate를 통과했을 때만 생성하는 GO-only artifact다.
NO-GO 및 smoke run에서 이 파일들이 없는 것은 실패가 아니라 의도된 계약이다.

```text
frozen_config_before_validation.json
final_model_bundle.joblib
selected_preprocessor.joblib
FINAL_TRAINING_REFIT.json
```

`environment.json`에는 Python, NumPy, pandas, scikit-learn, PyTorch/CUDA, GPU,
mixed precision, strict deterministic-algorithm 상태,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, 실행시간과 package version을 기록한다.
Checkpoint는 outer fold와
model seed 단위로 저장해 Colab 재시작 후 같은 run hash로 재개할 수 있어야 한다.

### 13.2 GO 후 benchmark artifact

```text
benchmark_source_audit.json
benchmark_input_manifest_before_label.json
benchmark_predictions_hashed_before_label.csv
historical_benchmark_metrics.json
historical_benchmark_confusion_matrix.png
historical_benchmark_per_class_metrics.csv
historical_benchmark_drift_diagnostics.csv
BENCHMARK_EVALUATION_COMPLETE.json
```

Benchmark report는 반드시 `historically reused official benchmark`라고 표기하고,
nested-CV 결과와 별도 표로 제시한다. benchmark의 높은 단일 class AUROC나 recall을
작은 support를 무시한 일반화 증거로 해석하지 않는다.

## 14. 결과 보고 원칙

- 주 결과는 repeated nested-CV subject-level Macro F1 평균과 SD다.
- Accuracy는 보조 지표이며 class-prior collapse 여부를 함께 적는다.
- MCI와 DEM support, confusion matrix, per-class F1/recall을 모든 표에 포함한다.
- `mask_tcn_35d`의 과거 0.404는 post-hoc hypothesis였음을 반복해서 명시한다.
- coverage-only 결과와 primary-minus-coverage delta를 숨기지 않는다.
- feature 중요도는 성능 동결 후 OOF/held-out association으로만 계산한다.
- SHAP/importance를 임상적 인과관계로 표현하지 않는다.
- GO 조건을 통과하지 못하면 “개선 실패” 또는 “불안정”을 그대로 보고하며,
  benchmark를 열어 더 좋은 숫자를 찾지 않는다.
