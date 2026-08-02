# ROC-AUC Champion 기술 보고서

최종 갱신: 2026-07-28

## 1. 문서의 지위

이 문서는 `Binary_Google_ROCAUC_Champion`의 설계 근거, 누수 계약, 평가
estimand와 산출물 규약을 기술합니다. **실험 결과 보고서가 아닙니다.**
구현은 완료됐지만 아직 이 새 코드를 `default`/`max` profile로 정식 학습하지
않았으므로 새 모델의 ROC-AUC, 기존 모델 대비 개선 또는 소요 시간의 실측값은
없습니다.

목표는 기존 프로젝트의 모든 관측을 이용해 유망한 후보를 구성하되, 후보 탐색
자체를 nested CV의 학습 절차 안에 넣어 subject-level ROC-AUC를 정직하게
평가하는 것입니다. “최고 모델”은 이름이나 inner score가 아니라, 완료된 outer
OOF 산출물이 증명해야 합니다.

## 2. 과제와 데이터

### 2-1. 예측 과제

- 음성 class: `CN=0`
- 양성 class: `MCI 또는 Dem=1`
- primary metric: subject-level ROC-AUC
- secondary: PR-AUC, Balanced Accuracy, MCI+Dem Recall, CN Specificity,
  Accuracy, confusion matrix
- Training: 141명, CN 85 / MCI 47 / Dem 9
- historical Validation: 33명, CN 26 / MCI 4 / Dem 3

Validation의 all-CN Accuracy는 26/33=0.7879입니다. Accuracy만 최대화하면
양성을 거의 찾지 못해도 높은 값이 나오므로 primary로 사용하지 않습니다.

`Binary_Google_DemScreen`의 CN+MCI vs Dem(12명) 과제는 label과 난이도가
다릅니다. wearable full-cohort 반복 평균 AUC 0.7184는 참고할 수 있지만 이
보고서의 CN vs MCI+Dem baseline이나 개선 근거로 사용하지 않습니다.

### 2-2. 두 modality 계약

#### `wearable`

- Activity와 Sleep만 읽습니다.
- `3.CognitiveFunction`의 디렉터리, SourceData와 LabelingData를 열거나
  탐색하지 않습니다.
- cognitive path가 access audit에 한 번이라도 나타나면 fail-closed입니다.
- 일별 행이나 sequence crop이 아니라 **사람**이 split 단위입니다.

#### `mmse`

- Activity, Sleep과 명시적 MMSE allowlist만 읽을 수 있습니다.
- MMSE allowlist는 `SAMPLE_EMAIL`, `TOTAL`과 **30개 점수 문항**뿐입니다.
- 식별자는 join 후 제거하고 모델 특징이 될 수 없습니다.
- `DIAG_NM`, `DIAG_SEQ`, `DOCTOR_NM`, `MMSE_NUM`, `MMSE_KIND`, `EMAIL`은
  source-level `usecols`에서 제외합니다. 읽은 뒤 삭제하는 방식이 아닙니다.
- Validation MMSE의 all-zero placeholder 검사는 라벨을 보지 않고 수행하며,
  해당 시험은 결측으로 처리합니다.

두 트랙은 같은 라벨 정의와 split seed를 사용해 paired comparison이 가능하지만,
특징 접근 경로와 모델은 분리됩니다.

## 3. 기존 실험의 증거

### 3-1. MMSE 계열

| 실험 | nested/subject OOF AUC | 핵심 관측 |
| --- | ---: | --- |
| `Binary_MMSE_MaxAUC` | **0.765756**, bootstrap 95% CI [0.684001, 0.845656] | 39개 MMSE 특징, 규제 LR+RBF SVM |
| `Binary_Google_OrdinalStable` winner | 0.7569 | MMSE-only 0.7494 대비 +0.0076 |
| `Binary_Google_MaxAUC_Tuned` | 0.717227 | non-nested 0.801681, optimism +0.08445 |
| SOTA DualTrack MMSE oblique GBT | 0.695588 | 95% CI [0.608124, 0.781920] |
| SOTA DualTrack MMSE stacking | 0.676681 | 복잡한 fusion의 안정적 이득 없음 |

`Binary_MMSE_MaxAUC`의 historical Validation AUC는 0.634615였습니다. OOF
0.7658과 함께 이 차이를 보면 작은 Validation을 이용해 모델을 고르면 안 된다는
점이 분명합니다.

OrdinalStable의 all-feature winner와 같은 run의 MMSE-only 차이는 0.0076뿐이고,
별도 paired bootstrap 차이 구간도 0을 포함했습니다. MaxAUC_Tuned는 약
10.6시간의 큰 탐색에도 AUC가 낮아졌습니다. 따라서 새 설계는 MMSE MaxAUC를
버리지 않고 anchor로 보존하며, 작은 고정 후보군만 비교합니다.

### 3-2. 웨어러블 계열

| 실험/arm | subject OOF AUC | 핵심 관측 |
| --- | ---: | --- |
| SequenceFusion 개별 Transformer | **0.625420** | wearable-only의 가장 유망한 제한적 신호 |
| SequenceFusion 고정 앙상블 | 0.566387 | 약한 branch가 anchor를 희석 |
| Wearable GoogleModels | 0.5370 | 1,077개 후보 탐색과 불안정 선택 |
| SOTA DualTrack wearable oblique GBT | 0.523319 | 95% CI [0.423312, 0.626998] |
| SOTA DualTrack wearable soft vote | 0.512395 | 95% CI [0.412343, 0.614712] |
| SOTA DualTrack wearable stacking | 0.494958 | chance 부근 |

SequenceFusion의 historical Validation AUC는 앙상블 0.4121이었습니다. SOTA
DualTrack의 feature-selection mean Jaccard도 wearable 0.125, MMSE fusion
0.132이며 모든 fold에서 항상 선택된 특징이 없었습니다. 이는 큰 supervised
feature search보다 규제, 반복 평가와 fallback이 우선이라는 근거입니다.

### 3-3. 누수의 관측 크기

같은 특징 공학에서 split만 바꾼 재현 결과는 다음과 같습니다.

| 입력 | 하루 무작위 K-fold | subject OOF |
| --- | ---: | ---: |
| wearable-only | **0.9526** | 0.5214 |
| wearable+MMSE | **0.99998** | 0.6924 |

같은 사람의 다른 날짜가 train/test에 동시에 들어가면 모델은 질병 신호가 아니라
사람 고유 패턴을 재식별합니다. 따라서 일별 표본 수가 많아도 유효 표본 수는
141명이며, 모든 crop/window는 subject group을 따라야 합니다.

## 4. 설계 원리와 문헌 근거

Varma와 Simon은 모델 선택과 성능 추정에 같은 CV를 사용하면 편향이 생기며,
nested CV가 독립 test에 가까운 추정을 제공함을 보였습니다
([원 논문](https://doi.org/10.1186/1471-2105-7-91)).
Vabalas 등은 제한된 표본의 simulation에서 pooled feature selection이
hyperparameter tuning보다 더 큰 편향을 만들 수 있음을 보였습니다
([PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224365)).
Cawley와 Talbot은 유한 표본에서 model-selection criterion 자체도 과적합될 수
있다고 지적했습니다
([JMLR](https://www.jmlr.org/papers/v11/cawley10a.html)).

이에 따라 본 설계는 다음 원칙을 사용합니다.

1. 모델 family·특징·blend 선택을 outer-training 내부에서만 수행
2. 광범위 random search보다 사전 고정한 작은 후보 library 사용
3. 복잡한 후보가 약한 개선만 보이면 강한 anchor로 fallback
4. outer OOF는 최종 선택된 한 모델이 아니라 **선택 절차 전체**를 평가
5. Validation은 모든 선택이 끝난 뒤 label-free freeze 후 한 번만 확인

상관된 고차원 특징에는 그룹화 효과가 있는 Elastic-Net을 우선합니다
([Zou와 Hastie](https://doi.org/10.1111/j.1467-9868.2005.00503.x)).
앙상블은 cross-validated prediction으로 결합 규칙을 학습해야 한다는
Super Learner 원리를 따르되
([van der Laan et al.](https://doi.org/10.2202/1544-6115.1309)),
141명에서는 자유로운 meta-model 대신 참조 ECDF와 동일가중 평균으로 자유도를
제한합니다.

## 5. 특징 구성

### 5-1. wearable 공통 입력

- 모든 사람에게 같은 최근 28개 관측을 사용합니다.
- 절대 날짜, 원본 행 순번, 관측 일수, sequence length, coverage, padding,
  mask와 missingness count를 모델 특징으로 내보내지 않습니다.
- 감사한 113-channel source schema의 feature count와 SHA-256을 고정해,
  새 `date`/`timestamp`/`day`/`elapsed`/`start`/`end` 계열 열이나 schema
  변경을 fail-closed로 거부합니다.
- Activity와 Sleep의 허용된 생리 신호만 사용합니다.
- 고정 28일 창에서 median, IQR, MAD, p10, p90, normalized Theil-Sen slope,
  최근 7일과 이전 21일 median 차이를 label-blind하게 계산합니다.
- 이 subject-level raw summary 계산은 다른 피험자의 분포나 label을 보지 않습니다.
- 이후 population median, clipping, scaling, correlation/feature selection은
  반드시 CV training fold 안에서만 fit합니다.

Sequence Transformer branch도 한 사람의 모든 sequence view를 같은 outer/inner
fold에 둡니다. Transformer 구조의 일반적 근거는
[Vaswani et al.](https://papers.nips.cc/paper/7181-attention-is-all-you-need)이며,
이 데이터에서의 채택 근거는 기존 개별 OOF AUC 0.6254입니다. 그 수치는 새
split에서 재현되어야 할 가설이지 사전 보장값이 아닙니다.

### 5-2. MMSE 입력

- `TOTAL`과 30개 허용 문항
- 시간·장소 지남력, 등록, 주의, 회상, 언어의 domain score/fraction
- reconstructed total, failed items, recall deficit
- 사전 고정한 임상 cutoff indicator

모든 변환은 label-blind하고 점수 규약만 사용합니다. supervised 선택은 inner
fold 안에서만 수행합니다. `TOTAL`과 문항에서 파생된 상관 특징을 함께 넣을 수
있으므로, 규제 모델이 기본입니다. MaxAUC anchor view는 이전 champion과
동등한 `TOTAL` 1개 + domain score 6개 + 문항 30개 + failed-items 1개 +
recall-deficit 1개, 총 **39개 특징**으로 코드가 단언합니다.

## 6. 후보 branch

정확한 후보 library는 `evaluation.py`의 `branch_library()`에 고정돼 있고,
실제로 선택·학습된 branch와 설정은 `fold_manifests.json`,
`deployment/deployment.json`, `FINAL_REPORT.json`에 기록됩니다.

### 6-1. `mmse` 트랙

- **`mmse_maxauc_anchor`**: 39개 MMSE anchor 특징에 median imputation과
  standard scaling을 적용한 규제 Logistic Regression + RBF SVM. 현재
  training partition 안 3-fold OOF Balanced Accuracy 0.55 gate로 두 모델의
  비음수 가중치를 정합니다.
- **`fusion_elastic_top25`**: MMSE+wearable 전체에서 fold-local
  direction-free AUC/correlation screen으로 최대 25개를 고른 Elastic-Net
  Logistic Regression
- **`fusion_rbf_top25`**: 같은 fusion view와 top-25 screen을 쓰는 RBF SVM
- **`fusion_catboost_top25`**: 같은 fusion view와 top-25 screen을 쓰는
  깊이 3의 강하게 규제한 CatBoost
- **`fusion_tabpfn_top64`**: `--tabpfn`으로 포함할 때만 평가하는
  공식 TabPFN v2.6, 최대 64개 특징

표 전처리는 현재 fold에서 median → 1/99% winsorization → median/IQR
scaling(SD fallback, 상수 제거) 순서입니다. Elastic-Net은
`C=0.1, l1_ratio=0.25`, RBF SVM은 `C=1, gamma="scale"`이며 두 모델 모두
`class_weight="balanced"`입니다. CatBoost는 300 trees, depth 3,
learning rate 0.03, L2 5와 balanced class weight로 고정합니다. MMSE track의
fusion 후보가 gate를 통과하지
못하면 MMSE anchor를 유지합니다. TabPFN branch는 이 전처리를 쓰지 않고
all-NaN/상수 열만 현재 fold에서 제거하며, 원값과 NaN을 native 모델에
전달합니다. top-64 선택 계산용 shadow에만 현재 fold median을 사용합니다.

### 6-2. `wearable` 트랙

- **`sequence_transformer_anchor`**: 28일 × 8개 고정 view를 쓰는
  SequenceFusion Transformer. 현재 training partition의 20% subject
  early-holdout에서 epoch를 선택한 뒤 전체 partition에 그 epoch 수로 refit
- **`wearable_core_ridge`**: 사전 고정 wearable core view 전체를 쓰는
  Ridge Logistic Regression (`C=0.1`)
- **`wearable_elastic_top25`**: wearable summary 전체에서 fold-local
  top-25 screen을 거친 Elastic-Net Logistic Regression
- **`wearable_rbf_top25`**: 같은 wearable view와 top-25 screen을 쓰는 RBF SVM
- **`wearable_catboost_top25`**: 같은 wearable view와 top-25 screen을 쓰는
  깊이 3의 강하게 규제한 CatBoost
- **`wearable_tabpfn_top64`**: `--tabpfn`으로 포함할 때만 평가하는
  공식 TabPFN v2.6, 최대 64개 특징

이 트랙의 어떤 branch도 CognitiveFunction 파일을 읽을 수 없습니다. TabPFN은
작은 표형 데이터에 맞춰 synthetic dataset으로 사전학습된 tabular foundation
model이라는 근거가 있습니다
([Hollmann et al., Nature 2025](https://www.nature.com/articles/s41586-024-08328-6)).
그러나 이 프로젝트에서 더 높다는 보장은 없습니다. 구현은 package의 움직이는
default를 사용하지 않고
`TabPFNClassifier.create_default_for_version(ModelVersion.V2_6)`을 호출합니다.
Prior Labs 공식 저장소는 v2.6 기본 모델이 purely synthetic data로 학습됐다고
명시합니다
([공식 TabPFN 저장소](https://github.com/PriorLabs/TabPFN)).
선택·학습된 TabPFN branch의 model manifest에는 package version,
`model_version="v2.6"`, checkpoint path와 SHA-256, moving default 미사용을
기록합니다. 인증이나 checkpoint 로딩이 실패하면 다른 버전이나 실제 데이터
fine-tuned checkpoint로 자동 전환하지 않고 run이 실패합니다.

## 7. repeated nested CV 알고리즘

### 7-1. profile

| profile | outer split | inner split | 용도 |
| --- | --- | --- | --- |
| `default` | 5-fold × 5 repeats | 4-fold × 2 repeats | 정식 기본 |
| `max` | 5-fold × 10 repeats | 4-fold × 2 repeats | 더 많은 반복의 정식 확인 |

축소 학습 mode는 없습니다. CLI의 `--mode`는 `full`만 허용하고,
`--profile default|max`가 outer repeat 수와 bootstrap 횟수(default 5,000회;
max 10,000회)를 정합니다. seed와 outer-test subject hash는 manifest에
기록합니다. stratification은 각 fold에 두 class가 존재하도록 하기 위한
장치이며, 반복 fold를 독립 표본으로 해석하지 않습니다. inner CV는
Transformer를 포함한 각 branch에 outer-training 내부의 **동일한** 8개 fold
(4 folds × 2 repeats)를 사용합니다. split seed는 해당 outer fold에서 한 번
고정하고, branch 이름은 모델 내부 RNG에만 반영합니다. 따라서 fold 승수 비교는
동일한 held-out 사람끼리 짝지어집니다. aggregate 비교는 각 사람에게서 나온
두 번의 held-out score 평균을 사용합니다.

TabPFN 포함 여부는 별도 `--tabpfn auto|on|off`로 정합니다. `auto`는
`TABPFN_TOKEN` 환경변수가 있을 때만 포함하며, `on`은 v2.6 branch를
명시적으로 포함하고 `off`는 제외합니다.

### 7-2. 한 outer fold의 절차

```text
outer-train subjects
  ├─ repeated inner subject splits 고정
  ├─ 각 branch:
  │    ├─ inner-train에서만 전처리·특징/epoch 선택
  │    └─ inner-held-out 연속 score 생성
  ├─ branch별 inner OOF reference ECDF fit
  ├─ anchor에서 시작해 candidate 하나를 더한 equal blend를 순차 비교
  ├─ aggregate OOF ΔAUC + 8-fold 승수 gate로 최대 2개 candidate 추가
  ├─ 선택된 branch를 outer-train 전체에 refit
  ├─ inner OOF reference ECDF로 outer-test score 변환
  └─ outer-test subject score를 정확히 한 번 저장
```

outer-test label은 branch, feature, epoch, ECDF, threshold 또는 gate 선택에
사용되지 않습니다. 표 모델 hyperparameter는 위 후보 library에 고정돼 있어
fold 안팎에서 탐색하지 않습니다.

## 8. 참조 ECDF와 보수적 blend

서로 다른 모델은 probability, margin, logit처럼 scale이 다른 score를 냅니다.
ROC-AUC는 순위 지표이므로 모델별 inner OOF 분포의 empirical CDF를 공통 척도로
사용합니다.

branch \(j\)의 outer-training 내부 OOF score를 정렬한 길이 \(n\)의 reference라
하고, 새 점수 \(t\)보다 작은 reference 수를 \(L_j(t)\), 같거나 작은 수를
\(R_j(t)\)라 하면 실제 구현의 smoothed mid-rank ECDF는

\[
\hat F_j(t)=\frac{L_j(t)+R_j(t)+1}{2(n+1)}
\]

입니다. tie에는 mid-rank를 사용하고, 상수 score는 0.5로 보냅니다. outer-test
점수 \(s^*_{j}\)는 test fold의 다른 사람과 함께 rank하지 않고
\(\hat F_j(s^*_{j})\)로 변환합니다. 따라서 outer-test batch 구성이나 label이
score 정규화에 영향을 주지 않습니다.

통과 후보 \(J\)의 score는

\[
s_{\mathrm{blend}}=\frac{1}{|J|}\sum_{j\in J}\hat F_j(s_j)
\]

의 동일가중 평균입니다. 작은 inner OOF에서 연속 weight를 최적화하면 또 다른
hyperparameter search가 되므로 학습 가중치를 두지 않습니다.

두 inner repeat에서 한 사람은 정확히 두 번 held-out score를 얻습니다. aggregate
inner OOF에서는 이 두 score를 사람별로 평균한 뒤 한 번만 AUC를 계산합니다.
현재 선택 집합의 정책을 \(p\), 여기에 candidate \(c\)를 하나 추가한 정책을
\(p+c\)라 하면 aggregate 차이는

\[
\Delta_{\mathrm{all},c\mid p}
=AUC_{\mathrm{inner\ OOF}}(p+c)
-AUC_{\mathrm{inner\ OOF}}(p)
\]

입니다. 동시에 8개 inner fold \(k\)에서
\(\Delta_{k,c\mid p}=AUC_k(p+c)-AUC_k(p)\)를 계산합니다. candidate를 현재
동일가중 blend에 추가하려면 다음을 모두 만족해야 합니다.

- \(\Delta_{\mathrm{all},c\mid p}\ge 0.005\)
- \(\sum_{k=1}^{8} I(\Delta_{k,c\mid p}>0)>4\), 즉 8개 중 최소 5개 fold 승리

첫 단계에서 \(p\)는 anchor입니다. fold AUC 동률은 승리로 세지 않습니다.
gate를 통과한 candidate가 여러 개면 aggregate AUC gain, strict fold
win fraction, branch 이름의 사전 고정 순서로 하나를 추가합니다. anchor를
포함한 선택 branch 수가 3개가 될 때까지 또는 더는 통과 후보가 없을 때까지
반복합니다. 따라서 첫 단계에서 모두 실패하면 anchor fallback이고, 한 후보를
추가한 뒤 다음 후보가 실패하면 현재 2-branch 정책을 유지합니다. 성공해도
outer-test에서 실제로 개선된다고 보장하지 않으며, outer OOF가 이를 평가합니다.

## 9. 불균형과 threshold

85:56은 ROC-AUC 학습을 위해 합성 표본이 필요한 수준의 극단적 불균형이
아닙니다. 본 설계는 SMOTE를 사용하지 않습니다. class weight가 있는 후보와
없는 후보를 비교하더라도 선택은 inner fold 안에서만 합니다.

ROC-AUC에는 hard label이 아니라 continuous decision score를 사용합니다.
threshold를 바꾸어도 ROC-AUC는 바뀌지 않으므로 threshold tuning은 champion
선택과 분리합니다. scikit-learn도 같은 data에서 classifier 학습과 threshold
선택을 함께 하지 말라고 안내합니다
([공식 문서](https://scikit-learn.org/stable/modules/classification_threshold.html)).

## 10. Primary estimand와 불확실성

각 트랙의 `nested_oof_report.json`은 다음을 별도 구조로 보고합니다.

1. `primary_selected_policy.repeat_level_roc_auc.mean`: repeat마다 모든 outer
   fold OOF를 모아 계산한 AUC의 평균
2. `primary_selected_policy.repeat_level_roc_auc.sd`: 위 repeat AUC의 모집단 SD
   (`ddof=0`)
3. `primary_selected_policy.subject_mean_repeated_oof.evaluation.primary.roc_auc`:
   사람별 repeated OOF score 평균의 AUC
4. 같은 `evaluation.primary.bootstrap`: 3번 추정량의 stratified subject
   bootstrap percentile CI

4번 CI를 1번 point estimate의 CI처럼 붙이지 않습니다. K-fold 결과는 train
set이 겹쳐 단순 fold variance가 불확실성을 과소평가할 수 있습니다
([Bengio와 Grandvalet](https://www.jmlr.org/papers/v5/grandvalet04a.html)).

Champion과 anchor는 같은 subject와 split의 score를 사용하므로 paired
stratified subject bootstrap으로
`AUC(champion)-AUC(anchor)`를 보고합니다. 95% interval이 0을 포함하면
“개선 확정”으로 선언하지 않습니다.

보조 지표에는 다음이 포함됩니다.

- PR-AUC / average precision
- Training inner OOF에서만 고른 threshold의 cross-fitted Accuracy와
  Balanced Accuracy
- frozen historical Validation에서 같은 threshold로 계산한 MCI+Dem Recall,
  CN Specificity, Precision, F1와 confusion matrix
- all-CN Accuracy/Balanced Accuracy baseline
- historical Validation의 CN vs MCI, CN vs Dem 보조 평가
- fold별 inner branch AUC, descriptive outer AUC와 branch/fallback 선택 빈도
- fold manifest의 selected feature 목록에서 재계산 가능한 선택 안정성

## 11. 최종 refit과 Validation two-track freeze

Training OOF 평가가 끝난 뒤 트랙별 선택 규칙을 Training 141명에 적용합니다.
최종 Validation 순서는 다음 state machine을 따릅니다.

```text
TRAIN_LABELS_OPEN
  → MMSE_FINAL_REFIT
  → WEARABLE_FINAL_REFIT
  → MMSE_LABEL_FREE_VALIDATION_SCORES_WRITTEN
  → WEARABLE_LABEL_FREE_VALIDATION_SCORES_WRITTEN
  → BOTH_FILES_SHA256_FROZEN_AND_VERIFIED
  → VALIDATION_LABELS_OPENED_ONCE
  → HISTORICAL_VALIDATION_EVALUATED
```

다음 조건 중 하나면 labels-open 단계 전에 중단합니다.

- train/validation subject ID overlap
- 각 트랙의 중복 subject ID 또는 feature schema 불일치
- wearable access audit에 CognitiveFunction path가 존재
- MMSE source access가 고정 allowlist와 다르거나 feature schema에 금지 열 존재
- 예측 파일 hash 재계산 불일치
- 두 트랙 중 하나의 freeze 누락
- 저장 모델 재로딩 score가 허용오차를 초과

Validation label은 Gait와 Sleep 사본을 한 번 읽어 일치 여부를 확인합니다.
CognitiveFunction label 사본은 어느 트랙도 읽지 않습니다.

Validation metric은
`status="historical_validation_reused_not_independent_test"`와 함께 저장합니다.
여기서 높은 값이 나와도 branch나 seed를 바꾸지 않으며, 새 외부 test 성능으로
표현하지 않습니다.

## 12. 산출물과 감사 가능성

정식 run에는 최소한 다음 증거가 필요합니다.

- `LAUNCHER_STATUS.status == "complete"`
- `TRAINING_COMPLETE.json`
- `run_manifest.json`의 실행 환경, dependency, profile과 seed
- `fold_manifests.json`의 outer-test subject hash, branch 설정과 선택 로그
- 선택된 TabPFN branch가 있으면 v2.6 checkpoint 경로와 SHA-256
- track별 data access audit
- track별 repeated OOF prediction과 repeat/fold provenance
- inner 선택, ECDF reference, blend gate와 fallback 로그
- anchor와 champion paired comparison
- label-free Validation 예측 두 파일과 SHA-256 freeze manifest
- historical Validation report
- 최종 model serialization round-trip 결과

현재 구현의 주요 구조:

```text
<result>/<UTC_RUN_ID>/
├── LAUNCHER_STATUS.json
├── run_manifest.json
├── data_access_audit.json
└── training/
    ├── FINAL_REPORT.json
    ├── TRAINING_COMPLETE.json
    ├── deployment_round_trip.json
    ├── mmse/
    │   ├── oof_predictions_hashed.csv
    │   ├── nested_oof_report.json
    │   ├── fold_manifests.json
    │   ├── nested_cv_progress.json
    │   └── deployment/
    │       ├── deployment.json
    │       └── models/
    ├── wearable/
    │   └── 같은 OOF·fold·deployment 구조
    ├── validation_predictions_label_free_hashed_mmse.csv
    ├── validation_predictions_label_free_hashed_wearable.csv
    ├── VALIDATION_PREDICTIONS_FROZEN.json
    └── historical_validation_report.json
```

중단된 run이 일부 파일을 남겨도 `LAUNCHER_STATUS.status`가 `complete`가 아니거나
`training/FINAL_REPORT.json`이 없으면 성능 보고 대상으로 인정하지 않습니다.

## 13. 실행 시간과 자원

- 정식 실행 위치: 사용자 Google Colab Pro+
- 권장 runtime: A100 + High-RAM
- 전체 두 트랙 목표 wall time: 6시간 이내
- launcher hard limit: 6시간(`SIGALRM`, 지원되는 Linux/Colab 환경)
- 기본 계약: CUDA 필수. `--allow-cpu`로 해제할 수 있으나 매우 느릴 수 있음
- `default`와 `max`의 실제 wall time: **모두 미실측**

특히 `max`가 6시간 안에 끝난다는 보장은 없습니다. 시간이나 인증 문제를 피하려면
run 시작 전에 `--profile`과 `--tabpfn`을 결정해야 합니다. 실행 도중 또는
historical Validation을 본 뒤 branch를 제거하거나 seed를 바꾸지 않습니다.
시간 제한에 걸린 run은 실패 상태로 종료하며 완료된 repeat 일부만 골라 최고
성능을 주장하지 않습니다.

이 구현 작업에서는 `default`/`max` 정식 학습을 실행하지 않았습니다. 사용자가
코드를 실행해 전달한 완료 artifacts만 새 성능·시간의 근거로 분석합니다.

## 14. MMSE incorporation bias와 기타 한계

MMSE 사용은 이 실험에서 허용됐지만, 임상 CN/MCI/Dem 판정이 MMSE를 포함한
인지검사에 일부 의존했다면 predictor가 outcome 정의 과정에 들어간 것입니다.
PROBAST는 이런 incorporation bias가 predictor-outcome 관계와 성능을 낙관적으로
만들 수 있다고 설명합니다
([PROBAST 설명](https://pmc.ncbi.nlm.nih.gov/articles/PMC9291738/),
[PROBAST+AI](https://www.bmj.com/content/388/bmj-2024-082505)).

따라서:

- MMSE track은 “MMSE를 포함한 인지 스크리닝 상한”으로 해석
- wearable track만 “웨어러블-only”라고 표현
- 두 트랙 성능 차이를 웨어러블의 임상적 부가가치로 오해하지 않음
- 진단과 독립된 endpoint 또는 새 외부 cohort에서 추가 검증

그 밖의 한계는 다음과 같습니다.

- 같은 141명을 이용한 다수의 과거 실험에서 생긴 project-level selection bias
- 33명 historical Validation의 반복 사용과 양성 7명의 큰 표본 변동
- Dem 9명이 CN vs MCI+Dem AUC를 끌어올릴 수 있는 과제 구성
- wearable 기간과 기기 상태의 분포 이동 가능성
- optional pretrained model의 checkpoint provenance와 재현성
- nested CV도 미래 기관·기기·인구집단의 external validity를 보장하지 않음

## 15. 완료 판정

이 폴더를 “ROC-AUC 기준 최고 모델”로 부를 수 있는 최소 조건은 다음과 같습니다.

1. `--mode full`에서 `default` 또는 `max` profile의 모든 repeat를 완료
2. 두 트랙 모두 누수·접근·freeze·round-trip 계약 통과
3. hashed subject OOF score로 수치를 재계산 가능
4. champion과 anchor의 paired 차이 및 CI 보고
5. historical Validation을 모델 선택에 사용하지 않음
6. 새 성능을 기존 결과가 아니라 해당 run의 완료 artifact에서 인용

새 champion의 OOF AUC가 anchor보다 낮으면 실험은 실패한 것이 아니라,
fallback과 정직한 평가가 제대로 작동했다는 음성 결과입니다. 외부 cohort 없이
내부 소수점 차이만으로 일반화 SOTA를 주장하지 않습니다.

## 16. 참고 문헌·공식 문서

- Varma, S. & Simon, R. (2006).
  [Bias in error estimation when using cross-validation for model selection](https://doi.org/10.1186/1471-2105-7-91).
- Cawley, G. C. & Talbot, N. L. C. (2010).
  [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html).
- Vabalas, A. et al. (2019).
  [Machine learning algorithm validation with a limited sample size](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224365).
- Zou, H. & Hastie, T. (2005).
  [Regularization and variable selection via the elastic net](https://doi.org/10.1111/j.1467-9868.2005.00503.x).
- van der Laan, M. J., Polley, E. C. & Hubbard, A. E. (2007).
  [Super Learner](https://doi.org/10.2202/1544-6115.1309).
- Vaswani, A. et al. (2017).
  [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need).
- Tomita, T. M. et al. (2020).
  [Sparse Projection Oblique Randomer Forests](https://www.jmlr.org/papers/v21/18-664.html).
- Hollmann, N. et al. (2025).
  [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6).
- Bengio, Y. & Grandvalet, Y. (2004).
  [No Unbiased Estimator of the Variance of K-Fold Cross-Validation](https://www.jmlr.org/papers/v5/grandvalet04a.html).
- scikit-learn.
  [Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage),
  [decision-threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html).
- imbalanced-learn.
  [Resampling leakage pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html).
