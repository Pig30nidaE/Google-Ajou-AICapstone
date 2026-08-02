# CN vs MCI+Dem ROC-AUC 최적화: 두 신규 모델 설계 분석

작성일: 2026-07-30

## 1. 목적과 고정 조건

이 문서는 `SangHyo/` 아래의 기존 이진·삼진 분류, Gemma 특징 생성,
Google 모델, 논문 재현 실험과 저장된 결과 artifact를 검토한 뒤 다음 두 신규
실험의 방향을 고정하기 위한 문서다.

1. `Binary_Gemma_CognitiveFeature_AUC`
   - Gemma API가 CN과 MCI의 미세한 인지 차이를 표현하는 연속형 특징을 생성한다.
   - 생성 특징을 원래 MMSE item-level 특징 및 웨어러블 특징과 함께 사용한다.
2. `Binary_Google_YDF_AUC`
   - Google의 Yggdrasil Decision Forests(YDF) sparse-oblique GBT를 중심으로
     ROC-AUC를 최대화한다.

공통 과제는 다음과 같다.

- 음성 class: `CN`
- 양성 class: `MCI + Dem`
- Training 141명: CN 85명, MCI 47명, Dem 9명
- 주 평가 지표: ROC-AUC만 사용
- accuracy, balanced accuracy, threshold, calibration 성능은 모델 선택에 사용하지
  않는다.
- nested 여부와 무관하게 non-nested feature/model/weight 선택을 허용한다.
- 그에 따른 선택 낙관편향은 허용하되, 직접적인 데이터 누수는 엄격히 금지한다.

## 2. MMSE 포함 여부에 대한 결론

두 신규 모델의 primary configuration에는 **MMSE를 포함한다**.

웨어러블-only 정식 OOF는 대부분 0.45~0.57이었고, 가장 유망했던 개별 sequence
Transformer도 ROC-AUC 0.6254였다. 반면 MMSE item-level 특징을 사용하는 단순
규제 LR+SVM은 정식 subject OOF 0.765756을 달성했다. ROC-AUC만 최우선으로
한다면 MMSE를 제외할 근거가 없다.

단, 원천 임상 요약 컬럼 `MMSE_NUM`을 그대로 사용하지 않는다. 30개 raw MMSE
문항으로부터 다음 39개 특징을 결정적으로 다시 계산한다.

- raw item 30개
- derived `TOTAL` 1개
- orientation, attention, recall 등 domain 합계 6개
- `num_failed` 1개
- `recall_deficit` 1개

웨어러블-only 또는 no-MMSE 결과는 ablation과 오류 점검 용도로만 저장하며
primary champion 선택에는 사용하지 않는다.

## 3. 기존 실험 결과 요약

점수의 평가 조건이 서로 다르므로 정식 subject OOF, non-nested 점수,
Google-only 단일 모델 근거를 구분해야 한다.

| 구분 | 실험/모델 | ROC-AUC | 해석 |
|---|---|---:|---|
| 정식 최고 | `Binary_MMSE_MaxAUC` | **0.765756** | repeated subject OOF, bootstrap 95% CI `[0.6840, 0.8457]` |
| 정식 Google 탐색 | `Binary_Google_MaxAUC_Tuned` | 0.717227 | nested OOF |
| 같은 실행의 non-nested | `Binary_Google_MaxAUC_Tuned` | **0.801681** | 직접 feature 누수는 없으나 탐색 낙관편향 `+0.08445` |
| 정식 안정성 탐색 | `Binary_Google_OrdinalStable` | 0.756933 | nested OOF, 7개 arm 중 승자 선택 bias 포함 |
| 전체 관련 실험의 non-nested 최고 | `Binary_Google_OrdinalStable` | **0.810294** | 사용자 조건상 허용되는 최고 관측값, optimism 약 `+0.0534` |
| Google-only 단일 family 근거 | YDF sparse-oblique GBT | **0.788866** | `MaxAUC_Tuned`의 non-nested/inner 평가에서 최고 Google 모델 |
| 최신 개인별 Gemma 특징 | `GeminiFeaturePipeline` LR | **0.718067** | MMSE+base+Gemma pooled mean-OOF |
| 최신 전역 Gemma program | `GemmaFeatureProgramPipeline` full | **0.729097** | repeat-mean AUC, MMSE-only 0.735142보다 낮음 |

`OrdinalStable`의 0.810294는 이번 사용자의 non-nested 허용 조건에서는 최적화
목표로 사용할 수 있다. 다만 정식 일반화 성능이나 Google-only 성능으로
표현해서는 안 된다. 이 점수는 LR, SVM, YDF 등을 포함한 arm 선택 결과다.

### 3.1 주요 정식 실험 계보

| 계열 | 실험 | OOF ROC-AUC | 핵심 결론 |
|---|---|---:|---|
| wearable-only | `Binary_Wearable_GoogleModels` | 0.5370 | 1,077개 후보와 대규모 탐색이 일반화되지 않음 |
| wearable-only | `Binary_Wearable_SequenceFusion_Google` ensemble | 0.5664 | 개별 Transformer만 0.6254 |
| wearable-only | `Binary_Wearable_TabNet_Google` | 0.4462 | 고차원 TabNet 실패 |
| wearable-only | `Binary_Wearable_BalancedFusion_Google` | 0.4756 | 복잡한 gated fusion 이득 없음 |
| wearable-only | `Binary_PaperLGBM_NoMMSE` | 0.5214 | 사람 단위 평가에서는 무작위 수준 |
| MMSE/fusion | `Binary_Clinical_MMSE_Fusion` | 0.6973 | MMSE의 명확한 상승 확인 |
| MMSE/fusion | `Binary_MMSE_DomainFusion` | 0.7095 | 단순 domain fusion |
| MMSE/fusion | `Binary_EDA_Selective` | 0.7174 | 적은 특징이 복잡한 모델 이상 |
| Google | `Binary_Google_YDF_Ensemble` | 0.6727 | 일반 YDF GBT+RF만으로는 부족 |
| mixed ensemble | `Binary_MetaEnsemble_Google` | 0.7082 | 복잡한 meta 결합 이득 없음 |
| Google | `Binary_Google_FinalBest` | 0.7116 | 14개 MMSE 중심 특징, threshold 변경은 AUC 불변 |
| MMSE/paper | `Binary_PaperLGBM_MMSE` | 0.6924 | MMSE 추가로 상승하나 목표 미달 |
| MMSE | `Binary_MMSE_MaxAUC` | **0.765756** | 현재 정식 최고 |

### 3.2 직접 누수가 만든 잘못된 고득점

`XAI_Paper_Reproduction2` 및 paper-style day-random 평가에서는 같은 사람의
서로 다른 날짜가 train/test fold 양쪽에 들어갔다. 같은 특징 공학을 사람 단위로
다시 평가하면 성능이 다음과 같이 붕괴했다.

| 입력 | day-random ROC-AUC | subject OOF ROC-AUC |
|---|---:|---:|
| wearable-only | 0.9526 | 0.5214 |
| wearable+MMSE | 0.99998 | 0.6924 |

따라서 0.95~1.00 점수는 신규 모델의 기준점이나 재사용 근거가 아니다.

### 3.3 다른 target의 고득점

`Binary_Google_DemScreen`, `Binary_Google_DemRankAUC_select1`,
`Codex_Dementia_ROCAUC`는 Dem vs CN+MCI 과제다. `DemRankAUC`의 약 0.92
점수는 현재의 CN vs MCI+Dem 과제와 직접 비교하거나 feature 선택 근거로 사용할
수 없다.

## 4. 최신 Gemma 실험 분석

### 4.1 개인별 Gemma 특징: `GeminiFeaturePipeline`

완료된 full run은 141명 전부에 대해 `gemma-4-31b-it` 응답을 생성하고 strict
schema 검증을 통과했다.

| 모델 | pooled mean-OOF ROC-AUC | repeat-mean ROC-AUC |
|---|---:|---:|
| MMSE+base LR | 0.694118 | 0.685000 |
| MMSE+base+Gemma LR | **0.718067** | **0.706176** |
| 차이 | +0.023950 | +0.021176 |

paired bootstrap 차이의 95% CI는 `[-0.01567, 0.06331]`로 0을 포함했다.
GBDT에서는 MMSE+base 0.693697에서 Gemma 추가 0.685504로 하락했다.
wearable-only에서도 Gemma 추가는 유의미한 개선을 만들지 못했다.

원인은 프롬프트가 진단, 위험도, 임상 측정, 집단 비교를 모두 금지하여 다음과
같은 일반 생활패턴 압축 특징만 생성했기 때문이다.

- routine regularity
- sleep timing variability와 continuity
- activity stability와 exertion
- diurnal contrast
- trend와 volatility
- weekday/weekend difference
- cross-domain coherence
- atypical-day burden
- reliability

이 특징은 진단 중립적이고 유용할 수 있으나 CN과 MCI의 미세한 경계를 직접
표현하지 않는다.

### 4.2 전역 Gemma 특징 프로그램: `GemmaFeatureProgramPipeline`

Gemma가 환자별 결과가 아니라 하나의 전역 wearable feature program을 생성하고,
이를 모든 사람에게 결정적으로 적용했다.

| 구성 | repeat-mean ROC-AUC | subject-mean ROC-AUC |
|---|---:|---:|
| full: MMSE+wearable+program | 0.729097 | 0.743908 |
| MMSE-only | **0.735142** | **0.749160** |
| program-only | 0.571492 | 0.572899 |
| wearable-only | 0.538136 | 0.549580 |
| wearable+program | 0.540215 | 0.542017 |

full-minus-MMSE 차이는 약 -0.00525였고, modal fusion weight는 MMSE 1,
wearable 0, program 0이었다. 기존 program prompt가 class와 MMSE를 모르며,
허용 연산도 `signed_mean`, `signed_product`, `absolute_gap`으로 제한되어 새
신호를 만들지 못했다.

### 4.3 재사용할 요소와 교체할 요소

재사용할 요소:

- API retry와 rate-limit 처리
- prompt/schema/model/payload 기반 deterministic cache key
- API key 비출력
- deidentified subject payload
- strict JSON schema validation
- 중단 후 재개
- subject-row alignment 검증
- repeated subject OOF와 paired AUC 비교

교체할 요소:

- diagnosis-neutral prompt
- MMSE 사용 금지 규칙
- 지나치게 제한적인 전역 feature DSL
- raw 특징을 Gemma 압축 특징으로 대체하는 구성

## 5. 신규 모델 1: `Binary_Gemma_CognitiveFeature_AUC`

### 5.1 핵심 가설

현재 과제의 어려운 경계는 CN vs Dem이 아니라 CN vs MCI다. 최신 champion
분석에서도 CN vs MCI AUC는 약 0.7258, CN vs Dem AUC는 약 0.9386이었다.
따라서 Gemma는 심한 전반적 손상보다 다음과 같은 **고기능 구간의 미세한
인지 불일치**를 특징으로 만들어야 한다.

- 보존된 orientation 대비 저하된 delayed recall
- 높은 derived TOTAL 안에 숨은 소수 item 실패
- attention과 recall 사이의 불균형
- 시간 orientation의 미세한 오류
- 특정 기억 문항의 반복 실패
- 단일 domain 오류와 다중 domain 오류의 구분
- 인지 특징과 수면/활동 안정성 사이의 불일치

### 5.2 입력

개인별 Gemma payload에는 다음만 포함한다.

- raw MMSE item 30개
- raw item으로부터 결정적으로 계산한 39개 MMSE 특징

현재 구현의 primary v1은 wearable summary를 Gemma API에 보내지 않는다.
기존 두 Gemma 실험에서 wearable/program 특징의 추가 이득이 없거나 음수였기
때문에, CN–MCI 경계에 직접 연결되는 MMSE 문항 패턴에 API 역할을 집중한다.

관측일 수, 수집량, missingness count는 API/cache 완전성 감사에는 사용할 수 있지만
분류 feature에는 넣지 않는다. 이 값들은 질병 신호가 아니라 수집 프로토콜의
대리변수가 될 수 있기 때문이다.

다음은 payload, prompt 변수, cache metadata에서 모두 제외한다.

- `DIAG_NM`
- `DIAG_SEQ`
- `DOCTOR_NM`
- subject ID와 파일명
- 원천 `MMSE_NUM`
- `MMSE_KIND`
- 해당 개인의 binary label
- 절대 날짜

### 5.3 task-aware Gemma prompt

프롬프트에는 과제 자체를 숨기지 않는다. Gemma에게 CN과 MCI+Dem을 구분하는
feature engineer 역할을 명시한다. 이는 해당 개인의 정답 label을 제공하는
직접 누수가 아니다.

프롬프트가 생성할 연속형 특징 후보는 다음과 같다.

1. `memory_specific_deficit`
2. `orientation_memory_gap`
3. `attention_recall_discordance`
4. `ceiling_adjusted_subtle_error`
5. `temporal_orientation_weakness`
6. `multi_domain_error_burden`
7. `preserved_function_with_focal_failure`
8. `mci_boundary_evidence`
9. `global_severity_evidence`
10. `evidence_reliability`

각 값은 `[0, 1]` 범위의 finite number로 제한하고, 근거가 부족하면 추측 대신
낮은 reliability를 반환한다. 최종 진단 문자열은 반환하지 않지만,
`mci_boundary_evidence`와 같은 연속 순위 특징은 허용한다.

API 호출은 temperature 0 또는 API가 지원하는 가장 결정적인 설정으로 고정한다.
같은 model, prompt, schema, payload에는 항상 같은 cache를 사용한다.

### 5.4 downstream 모델

Gemma 특징은 raw MMSE 39개를 대체하지 않고 추가한다. 후보 입력 block은 다음과
같다.

- MMSE 39개
- Gemma cognitive 특징 10개

이 첫 모델은 Gemma 특징의 효과를 Google 모델 효과와 분리하기 위해 후보 learner를
규제 Logistic Regression과 RBF SVM으로 제한하고, 반복 subject OOF에서
ROC-AUC가 가장 높은 Gemma 포함 단일 모델 또는 rank blend를 non-nested 방식으로
선택한다. MMSE-only 모델은 ablation으로 함께 계산하되 Gemma 모델의 champion
후보에는 포함하지 않는다. selection score 외의 accuracy나 threshold 지표는
사용하지 않는다.

### 5.5 성공 기준

- 최소 기준: 최신 개인별 Gemma 0.718067 초과
- 실질 개선 기준: 정식 MMSE anchor 0.765756 초과
- 도전 기준: 허용된 non-nested 최고 0.810294 초과

Gemma-minus-no-Gemma paired subject bootstrap CI도 저장하되, 모델 선택은
ROC-AUC 절대값으로 수행한다.

## 6. 신규 모델 2: `Binary_Google_YDF_AUC`

### 6.1 핵심 가설

기존 Google 모델 중 가장 강한 단일 근거는 일반 axis-aligned YDF가 아니라
sparse-oblique YDF GBT의 0.788866이다. 상관된 MMSE item과 소수 wearable
summary의 선형 조합을 oblique split이 직접 학습하는 것이 현재 표본 크기에 가장
적합하다.

### 6.2 입력 feature bank

- MMSE 39개
- 기존 wearable summary에서 관측일 수 2개와 non-wear coverage 1개를
  제거하고 활동 구성비 3개로 대체한 안전한 측정 summary 112개
- 총 151개 후보

반복적으로 강했던 특징은 다음과 같다.

- derived `TOTAL`
- `num_failed`
- recall과 `recall_deficit`
- `Q13_2`, `Q13_3`, `Q12_5`, `Q03`
- temporal orientation
- attention
- place orientation
- deep-sleep score std/CV
- restless-sleep std
- awake std/mean
- light-sleep ratio mean
- move-every-hour mean
- inactivity-alert mean

### 6.3 non-nested 선택

사용자가 non-nested 선택을 허용했으므로 다음 후보 절차를 같은 Training repeated
OOF에서 비교하고 그중 가장 높은 ROC-AUC를 고른다.

1. 각 OOF fold의 **training subject만** 사용해 direction-free univariate
   ROC-AUC로 feature ranking
2. 그 training fold에서 상위 70개 유지
3. 그 training fold에서 절대 상관 0.99 이상 중복 제거
4. YDF 설정과 seed/rank ensemble weight를 repeated subject OOF AUC로 선택

각 held-out subject의 label은 그 subject를 점수화하는 모델의 feature ranking,
전처리 또는 fit에 들어가지 않는다. 다만 같은 repeated OOF 결과로 feature-count,
YDF 설정과 ensemble을 고르고 동시에 최고 점수를 보고하므로 model-selection
낙관편향은 남는다. 이번 조건에서는 이를 허용된 non-nested 선택으로 명시하되,
held-out label이나 동일 사람의 다른 날짜가 학습 쪽에 들어가는 직접 누수와
구분한다.

### 6.4 우선 재현할 YDF 설정

기존 Google-only 최고 **하이퍼파라미터 설정**을 첫 고정 candidate로 사용한다.
다만 기존 151개 bank에 있던 두 관측일 수 특징과 non-wear coverage는 분류
입력에서 제거했으므로, 이 설정은 안전한 151개 bank로 이식한 것이며 과거
점수의 수치적 exact reproduction을 뜻하지 않는다.

```text
learner = YDF GradientBoostedTrees
split_axis = SPARSE_OBLIQUE
top_k = 70
correlation_threshold = 0.99
num_trees = 600
max_depth = 5
min_examples = 20
shrinkage = 0.08
subsample = 0.6
num_candidate_attributes_ratio = 0.3
l2_regularization = 0
sparse_oblique_normalization = STANDARD_DEVIATION
sparse_oblique_num_projections_exponent = 1.5
sparse_oblique_projection_density_factor = 3.0
```

동일 family의 여러 seed 확률을 percentile rank로 변환한 뒤 평균하는 Google-only
ensemble을 추가 후보로 둔다. 일반 GBT/RF를 추가할 때는 OOF AUC를 실제로 높이는
경우에만 포함한다. 기존 0.801681 mixed ensemble은 LR/SVM weight가 약 83%이므로
Google-only champion의 근거로 그대로 사용하지 않는다.

### 6.5 성공 기준

- 재현 기준: Google-only 관측 근거 0.788866 이상
- 1차 목표: mixed non-nested 0.801681 이상
- 도전 기준: 전체 non-nested 최고 0.810294 이상

모든 점수는 Google-only인지 mixed인지 명시한다.

## 7. 직접 데이터 누수 방지 규칙

이번 실험에서 엄격히 막을 대상은 다음과 같다.

1. **subject leakage**
   - 모든 split은 subject 단위다.
   - 한 사람의 모든 날짜와 window는 하나의 fold에만 존재한다.
2. **target/admin column leakage**
   - `DIAG_NM`, `DIAG_SEQ`, `DOCTOR_NM`, ID, 파일명은 feature가 될 수 없다.
3. **clinical summary leakage**
   - `MMSE_NUM`, `MMSE_KIND`를 직접 사용하지 않는다.
   - raw item에서 필요한 합계와 domain 특징을 다시 계산한다.
4. **Gemma payload leakage**
   - 개인별 label, 진단명, ID를 전송하거나 cache에 포함하지 않는다.
   - prompt는 과제와 일반 판별 원리를 알 수 있지만 해당 사람의 정답은 알 수 없다.
5. **row alignment leakage/오류**
   - API 결과는 익명화된 내부 row key로만 다시 결합한다.
   - 누락, 중복, 순서 변경을 실행 전에 검증한다.
6. **Validation 재사용**
   - historical Validation 33명은 feature/model 선택에 사용하지 않는다.
   - 최종 참고 평가를 수행하더라도 Training OOF와 별도로 표시한다.

허용하지만 보고서에 반드시 표시할 대상은 다음과 같다.

- 전체 Training을 이용한 feature ranking
- non-nested hyperparameter 선택
- 여러 arm 중 최고 AUC 선택
- OOF를 이용한 ensemble weight 선택

이는 선택 낙관편향이지만, 금지 대상인 직접 feature/subject leakage와는 구분한다.

## 8. 피해야 할 기존 설계

- day/window random split
- 1,000개 이상 무차별 feature를 소표본 TabNet에 투입
- 진단 중립적 Gemma 요약만으로 raw MMSE를 대체
- 복잡한 fusion을 AUC 검증 없이 고정
- threshold 또는 accuracy 최적화
- historical Validation 33명으로 반복 선택
- Dem-vs-rest 실험의 고득점을 현재 과제에 전용
- test batch 전체에서 rank normalization을 다시 계산하는 transductive 처리
- all-zero MMSE를 임의의 정상 점수로 해석하는 missingness 처리

## 9. 단일 엔트리포인트와 결과 계약

신규 모델 폴더는 다음 두 개다. 외부 실행 진입점은 각각 `run.py` 하나이며,
내부 모듈·README·계약 테스트를 함께 둔다.

```text
SangHyo/Binary_Gemma_CognitiveFeature_AUC/
├── run.py
├── README_KO.md
└── tests/

SangHyo/Binary_Google_YDF_AUC/
├── run.py
├── README_KO.md
└── tests/
```

각 실험의 외부 실행 진입점은 `run.py` 하나다. 내부 module을 둘 수는 있지만 별도
launcher는 만들지 않는다.

각 Training `run.py`는 최소한 다음 종류의 artifact를 생성한다.

- 실행 config
- source/data/leakage audit
- feature manifest
- repetition/fold별 OOF prediction
- subject-mean OOF prediction
- candidate별 ROC-AUC
- 선택된 feature/model/weight
- `FINAL_REPORT.json`과 `TRAINING_COMPLETE.json`

Gemma 모델은 추가로 다음을 저장한다.

- label과 ID가 제거된 payload hash
- model/prompt/schema version
- response validation 결과
- cache hit/miss와 retry 통계
- Gemma feature table

API key, 원본 개인 식별자, 진단 label이 포함된 payload는 artifact에 저장하지 않는다.

## 10. 최종 보고 원칙

최종 결과에서는 아래 세 숫자를 혼합하지 않는다.

1. 정식 subject OOF anchor: 0.765756
2. 허용된 non-nested 전체 최고: 0.810294
3. Google-only 단일 모델 근거: 0.788866

신규 모델이 non-nested 선택을 사용했다면 headline에도 이를 표시한다. 사용자
목표대로 모델 선택은 ROC-AUC 하나만으로 수행하되, 점수가 직접 누수 없이
생성되었는지를 각 실행의 `LEAKAGE_AUDIT.json`으로 재현 가능하게 증명한다.
