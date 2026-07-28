# Codex_Dementia_ROCAUC

AI Hub 「치매 고위험군 웨어러블 라이프로그」에서 다음 사람 단위 이진분류를
평가하는 새 코드입니다.

- 음성 0: CN + MCI
- 양성 1: Dem
- Primary: subject-level ROC-AUC
- Secondary: PR-AUC, Dem recall, F1, balanced accuracy, MCC, specificity

이 폴더를 작성하면서 **학습은 실행하지 않았습니다**. 따라서 새 ROC-AUC,
기존 대비 개선 폭, fold 결과를 만들거나 추정하지 않습니다. 수행한 검증은
문법 검사, import 가능한 경로, 합성 데이터 split/leakage assertion뿐이며
estimator `.fit()` 호출 수는 0입니다.

## 1. 기존 실험에서 확인한 기준과 한계

같은 라벨 정의를 정식으로 다룬 기존 결과의 source of truth는 다음 파일입니다.

`SangHyo/Binary_Google_DemScreen/Binary_Google_DemScreen_result/20260728_051820_utc/training/FINAL_REPORT.json`

| 기존 arm | 평가 | ROC-AUC |
|---|---|---:|
| wearable-only full | 174명 pooled, 20-repeat nested OOF AUC의 평균 | 0.7184 ± 0.0363 |
| wearable-only full | subject별 20-repeat OOF score 평균의 AUC | 0.7803 |
| wearable+MMSE full | 174명 pooled, repeat OOF AUC의 평균 | 0.8284 ± 0.0450 |
| wearable-only filtered | label-blind QC 민감도 arm, 172명/DEM 11 | 0.8304 ± 0.0395 |
| wearable+MMSE filtered | QC+MMSE 민감도 arm | 0.9225 ± 0.0279 |

주의사항:

- 174명은 원 Training과 historical Validation을 합친 코호트이고 독립 test가
  없습니다.
- 기존 보고서의 `0.7184` 옆 CI는 실제로 다른 estimand인 `0.7803`의
  subject-bootstrap CI입니다. 새 코드는 두 estimand를 별도 필드로 저장합니다.
- 기존 `top_univariate`는 전체 174명 라벨을 본 descriptive AUC이며 OOF가
  아닙니다.
- Hyunsoo의 wearable `0.9087`은 성능을 본 뒤 Dem 한 명을 제외하고 같은
  코호트에서 특징을 고정한 post-selection OOF입니다. 강한 후속 가설이지만
  깨끗한 기준 성능으로 사용하지 않습니다.
- 기존 DemScreen의 SMOTE 스위치는 실제 learner 설정으로 전달되지 않았고,
  YDF 부재 시 다른 모델로 조용히 바뀔 수 있었습니다. 새 코드는 optional
  모델 의존성이 없으면 명시적으로 실패하며 silent fallback을 금지합니다.

## 2. 새 평가 설계

Primary는 official Training 141명(CN 85, MCI 47, Dem 9)에서 수행합니다.

```text
outer: StratifiedGroupKFold 3-fold × 고정 repeat
  outer-training 안에서만:
    inner 3-fold × repeat
    fold-local 전처리/특징 선택/리샘플링
    전체 후보의 가벼운 screening
    상위 후보 Optuna AUC tuning
    OOF probability/rank-ECDF blending
    operating threshold 선택
  outer-validation:
    사람별 예측 1회
```

한 피험자는 한 행/한 시퀀스이고 `EMAIL`이 group입니다. 모든 outer/inner
fold에서 train/validation subject 교집합 0, 양쪽 class 존재, Validation Dem
최소 1명, repeat마다 모든 피험자 OOF 1회를 assertion으로 확인합니다.
inner score는 튜닝·blend·threshold 선택에 다시 쓰이므로 성능 주장용 수치가
아닙니다. 완전한 적응 절차의 성능은 한 번도 선택에 쓰이지 않은 outer OOF로만
보고합니다.

historical Validation 33명(CN 26, MCI 4, Dem 3)은 다음 순서로만 사용합니다.

1. Training-only nested OOF와 최종 refit 완료
2. Validation 라벨을 열지 않고 세 track 예측 저장
3. 예측 CSV, 모델, subject 순서의 SHA-256 freeze
4. 모든 track freeze 성공 확인
5. 그 뒤에만 Gait/Sleep label 사본을 열어 descriptive 평가

이 Validation은 과거 프로젝트에서 반복 사용됐으므로 freeze를 하더라도 독립
외부 test가 아닙니다.

## 3. 세 track

| track | 입력 | 지위 |
|---|---|---|
| `wearable` | Activity/Sleep 생물학적 신호 | primary |
| `wearable_protocol` | wearable + 관측일/결측 coverage proxy | 사전 선언 sensitivity |
| `wearable_mmse` | wearable + allowlist MMSE 점수 | diagnostic incorporation bias가 있는 reference |

`wearable`/`wearable_protocol` loader는 `3.CognitiveFunction`을 열면 실패합니다.
`wearable_mmse`도 `usecols` allowlist로 `TOTAL`과 30문항만 읽습니다.
`DIAG_NM`, `DIAG_SEQ`, `DOCTOR_NM`, `MMSE_NUM`, `MMSE_KIND`, ID는 모델
특징에 들어갈 수 없습니다.

## 4. 특징

원본 Activity/Sleep의 일별 scalar와 고주파 payload에서 이미 label-free로 만든
하루 벡터를 사용합니다. 각 피험자 안에서만 다음을 계산합니다.

- mean, median, std, min/max/range, p10/p25/p75/p90, IQR, CV
- skewness, kurtosis, first-last delta, normalized slope
- 최근 7일과 전체의 차이/비율
- lag-1 autocorrelation, entropy, spectral peak ratio
- rolling 7-day variability
- 별도 sensitivity view의 결측률, 유효일, 최장 연속결측, sequence length
- 사전 지정된 활동·수면·MMSE interaction

절대 날짜와 enrollment order는 특징으로 내보내지 않습니다. weekday/weekend
특징은 가능하지만 이번 기본 schema에서는 모집 시기 proxy 위험을 피하려고
제외했습니다.

피험자 간 통계가 필요한 다음 처리는 모두 estimator pipeline 안에서만 fit됩니다.

- median imputation + missing indicator
- winsorization
- univariate top-k selection
- robust scaling
- RandomOverSampler / SMOTE / ADASYN
- model, TabNet pretraining, TSMixer channel selection

## 5. 모델과 선택

공유 outer split에서 다음 고정 configuration의 honest model별 OOF를 저장합니다.

- prespecified `activity_low` variability 1-feature logistic anchor
- elastic-net logistic regression, RBF SVM
- ExtraTrees, RandomForest, HistGradientBoosting
- BalancedRandomForest, EasyEnsemble
- LightGBM, XGBoost, CatBoost
- MLP
- supervised TabNet
- fold-local unsupervised pretraining + supervised TabNet
- 실제 일별 sequence용 compact TSMixer

TabNet은 class weighting, virtual batch, decision steps, feature/attention
dimension, relaxation factor, sparsity와 learning rate를 inner OOF에서
튜닝합니다.

`standard`는 넓은 screening 후 상위 2개 후보를 outer-training의 inner OOF
ROC-AUC로 12 trial 튜닝합니다. `max`는 10 outer repeats, 후보당 40 trial이고
TabNet 두 variant와 TSMixer 튜닝을 강제합니다. Optuna pruning과 후보별 timeout이
설정돼 있습니다.

앙상블은 사전 고정 anchor에서 시작하되, 다른 단일 후보가 inner OOF AUC를
최소 0.0025 높이고 inner repeat의 엄격한 과반에서 이기면 그 후보를 안정적
base로 승격할 수 있습니다. 이후 probability 평균과 rank-ECDF 평균을 비교하고
가중치는 inner OOF에서만 고릅니다. 최종 blend 역시 같은 최소 향상 폭과
repeat 과반 규칙을 통과하지 못하면 안정적 base로 돌아갑니다. outer 대상끼리
rank를 계산하지 않습니다. 선택된 각 모델은 inner repeat 수와 같은 수의
사전 고정 seed로 outer-training 전체에 재학습하고 확률을 평균해, 반복 OOF와
배포 predictor의 분산 축소 구조를 맞춥니다. 자유로운 stacking은 Training
Dem이 20명 미만이라 사전 규칙으로 비활성화합니다.

TimesFM은 forecasting 모델이고 이 저장소에 검증된 frozen representation 계약이
없으므로 실행하지 않습니다. 생략 이유가 결과 JSON에 자동 기록됩니다.

## 6. 설치

저장소 루트에서:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r Codex_Dementia_ROCAUC/requirements_core.txt
python -m pip install -r Codex_Dementia_ROCAUC/requirements_models.txt
```

Colab에서 두 파일을 한 번에 설치하려면:

```bash
python -m pip install -r Codex_Dementia_ROCAUC/requirements_colab.txt
```

CUDA가 있으면 TabNet/TSMixer가 자동 사용하고 표형 모델은 CPU를 사용합니다.

## 7. 학습 없는 검증 명령

코드/합성 split/leakage 계약만 확인:

```bash
python Codex_Dementia_ROCAUC/run.py validate-code
```

실제 raw schema, ID, class count, 파일 hash만 확인:

```bash
python Codex_Dementia_ROCAUC/run.py audit-data \
  --data-root Data \
  --output-dir Codex_Dementia_ROCAUC_audit
```

학습 없이 outer split registry만 저장:

```bash
python Codex_Dementia_ROCAUC/run.py make-splits \
  --data-root Data \
  --output-dir Codex_Dementia_ROCAUC_splits
```

## 8. 사용자가 실행할 학습 명령

`standard`:

```bash
python Codex_Dementia_ROCAUC/run.py train \
  --profile standard \
  --data-root Data \
  --output-dir /content/drive/MyDrive/Codex_Dementia_ROCAUC_standard \
  --device auto \
  --n-jobs 4 \
  --execute-training I_UNDERSTAND_THIS_RUNS_TRAINING
```

전체 TabNet/TSMixer 튜닝을 포함한 `max`:

```bash
python Codex_Dementia_ROCAUC/run.py train \
  --profile max \
  --data-root Data \
  --output-dir /content/drive/MyDrive/Codex_Dementia_ROCAUC_max \
  --device auto \
  --n-jobs 4 \
  --execute-training I_UNDERSTAND_THIS_RUNS_TRAINING
```

출력 폴더가 비어 있지 않으면 학습을 거부합니다. acknowledgement가 정확하지
않아도 학습을 시작하지 않습니다.

## 9. 주요 결과 파일

```text
<output>/
├── CONFIG.json
├── ENVIRONMENT.json
├── ALL_TRACKS_FROZEN.json
├── FINAL_REPORT.json
├── RUN_STATUS.json
└── <track>/
    ├── primary_oof/
    │   ├── split_registry_outer.json
    │   ├── inner_splits/*.json
    │   ├── oof_predictions_long.csv
    │   ├── oof_predictions_subject_mean.csv
    │   ├── fixed_model_oof_long.csv
    │   ├── model_comparison.csv
    │   ├── training_fitted_importance.csv
    │   ├── curves/*_roc_curve.{csv,png}
    │   ├── curves/*_pr_curve.{csv,png}
    │   ├── PRIMARY_REPORT.json
    │   └── PRIMARY_COMPLETE.json
    ├── deployment/
    │   ├── deployment.pkl
    │   ├── deployment_manifest.json
    │   └── deployment_inner_splits.json
    ├── historical_validation_predictions_label_free.csv
    ├── HISTORICAL_PREDICTIONS_FROZEN.json
    ├── historical_validation_labeled_predictions.csv
    └── HISTORICAL_VALIDATION_REPORT.json
```

`PRIMARY_REPORT.json`은 다음을 구분합니다.

- repeat별 complete OOF AUC의 평균/표준편차: split noise
- subject별 repeated cross-fitted score 평균의 AUC + stratified subject
  bootstrap CI: sampling uncertainty
- 고정 모델별 outer OOF 비교
- 선택된 ensemble 절차의 outer OOF

새 실행이 완료되기 전에는 최고 모델이나 개선 폭이 존재하지 않습니다. 또한 새
primary(Training-only 141)와 기존 pooled-174 결과는 평가 코호트가 달라 단순
차이를 개선 폭으로 보고하지 않습니다.
