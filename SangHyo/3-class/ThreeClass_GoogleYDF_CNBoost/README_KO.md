# CN / MCI / DEM — Google YDF CNBoost

이 폴더는 **MMSE 없이 웨어러블 활동·수면 기록만으로** CN, MCI, DEM을 구분하는 다음 단계 실험입니다. 목표는 사람 단위 ROC-AUC와 accuracy를 높이면서, 특히 CN과 비CN을 더 안정적으로 나누는 것입니다.

코드만 준비했으며 모델 학습은 실행하지 않았습니다. `eda.py`는 Training-only EDA이고, `train.py`가 실제 학습 파일입니다.

## 무엇이 달라졌나요?

### 1. MMSE를 단순 제외하는 수준보다 더 강하게 막았습니다

이전 `wearable_only` 모드는 최종 입력에 MMSE를 합치지는 않았지만 MMSE 파일을 찾아 읽었습니다. 새 loader의 입력 구조에는 MMSE 경로 필드 자체가 없습니다.

- 읽는 원천: activity, sleep
- 정답 확인: 세 label 사본의 `DIAG_NM` 일치 여부
- 읽지 않는 원천: `SourceData/3.CognitiveFunction/*mmse.csv`
- 금지 변수: MMSE, 진단명, ID, 절대 날짜, 관측일 수, 수집 간격, mask, raw/valid length, non-wear 비율

실행 결과의 `training_feature_audit.json`과 `feature_manifest.json`에도 `mmse_source_opened=false`, `mmse_values_used=false`가 기록됩니다.

### 2. Google 계열 모델을 중심으로 단순한 후보만 비교합니다

주 모델은 Google의 [Yggdrasil Decision Forests(YDF)](https://ydf.readthedocs.io/en/latest/)입니다. 보조 신경망인 [TabNet도 Google Research가 발표한 모델](https://research.google/pubs/tabnet-attentive-interpretable-tabular-learning/)입니다.

고정 후보는 다섯 가지입니다.

1. `ydf_multiclass`: CN/MCI/DEM을 한 번에 학습
2. `ydf_hierarchical`: 먼저 CN 대 비CN, 그다음 MCI 대 DEM을 학습
3. `ydf_random_forest`: 확률을 투표하는 YDF Random Forest로 GBT와 다른 오류를 보완
4. `ydf_ovr`: 클래스마다 “해당 클래스인가?”를 따로 학습한 뒤 확률을 합침
5. `tabnet`: 다른 방식의 판단을 추가하기 위한 Google Research 보조 모델

과거 실험처럼 수천 개의 ensemble weight와 class scale을 검색하지 않습니다. 미리 정한 10개 단일/혼합 규칙만 inner OOF에서 비교하고, 최고점과 0.01 이내라면 더 단순한 규칙을 고릅니다. 역사적 Validation을 보고 가중치나 threshold를 바꾸는 코드는 없습니다.

YDF는 주로 CPU를 사용합니다. A100은 TabNet에 사용되며, YDF에는 Colab High-RAM과 CPU thread가 더 중요합니다.

### 3. 최근 7·14·28개의 실제 관측을 따로 봅니다

달력상 7일, 14일, 28일을 채우면 결측 mask가 수집 방식의 차이를 외울 수 있습니다. 이 실험은 날짜 grid 대신 각 사람의 **최근 7/14/28번 실제 관측 event**를 사용합니다.

각 구간에서 다음을 만듭니다.

- 중앙값과 10% 절사평균
- 낮은 쪽/높은 쪽 분위수
- IQR과 MAD 같은 일간 변동
- 이상치에 덜 민감한 Theil–Sen 추세
- 최근 절반과 이전 절반의 차이
- 활동 상태 1~5의 비율·연속 구간·전이·엔트로피
- wear 구간에 맞춘 MET와 활동 리듬
- 수면 단계·분절·심박·RMSSD·취침/기상 시각

활동 state 5(high)를 누락하던 이전 집계와 MET의 정상적인 0까지 지우던 문제는 재사용한 `PerformanceLab` feature 계약에서 이미 교정돼 있습니다.

### 4. “CN의 정상 범위”에서 얼마나 벗어나는지도 fold 안에서 계산합니다

일부 변수는 MCI와 DEM이 CN에서 서로 반대 방향으로 움직입니다. 단순 평균을 내면 이 차이가 상쇄될 수 있습니다.

새 전처리는 각 학습 fold의 CN만으로 중앙값과 IQR을 계산하고, 각 값이 그 정상 범위에서 얼마나 멀리 떨어졌는지 절댓값 변수를 추가합니다. 이 기준은 outer/inner validation 사람을 보지 않고 매 fold에서 새로 계산합니다. MCI 대 DEM 단계에는 CN이 없으므로 이 변환을 사용하지 않습니다.

## 이전 결과에서 무엇을 참고했나요?

비교 가능한 MMSE-free 결과는 아직 높지 않았습니다.

| 실험 | Accuracy | Macro F1 | Macro ROC-AUC | CN recall / F1 |
|---|---:|---:|---:|---:|
| 기존 lifelog-only 기준선 | 0.532 | 0.358 | 0.530 | 0.753 / 0.681 |
| NextStage adaptive | 0.489 | 0.350 | 0.542 | 0.629 / 0.616 |
| NextStage mask TCN 사후 진단 | 0.546 | 0.404 | 0.566 | 0.741 / 0.675 |
| Transformer/TabNet/YDF + MMSE | 0.610 | 0.491 | 0.706 | 0.824 / 0.745 |

마지막 행은 MMSE가 포함돼 이번 실험과 직접 비교할 수 없습니다. 다만 그 실험에서 YDF가 평균 accuracy와 CN recall이 가장 높고, 6개 fold 중 5개 blend에 선택됐다는 점을 모델 가설로만 참고했습니다.

`mask_tcn_35d`는 MMSE 없이 상대적으로 좋았지만, 관측 mask와 마지막 관측 후 경과일을 직접 사용했습니다. 실제 EDA에서는 수집 시작일 같은 protocol 변수가 생체 변수보다 CN을 더 잘 나눴습니다. 따라서 그 수치는 생체 신호 개선 근거로 사용하지 않고 새 주 모델에서는 해당 shortcut을 차단했습니다.

Training-only EDA에서 반복된 생체 패턴은 다음과 같습니다.

- 수면 midpoint와 취침/기상 시각
- restless, awake, light/deep 수면의 수준과 일간 변동
- 수면 단계 entropy·transition·bout
- lowest HR 시점, HR drop, RMSSD
- 활동 rest/low/high 비율과 상태 전이
- 최근 활동·수면 변화량

새 7/14/28-event EDA에서 CN 대 비CN의 가장 큰 단일 신호는 최근 7회 light sleep 상단값(방향 보정 AUC 0.657), 최근 28회 수면단계 entropy IQR(0.650), 최근 활동 리듬의 peak 시각(약 0.64)이었습니다. MCI 대 CN 최고 단일 AUC도 약 0.642라 여전히 어렵지만, DEM 대 CN에서는 최근 7회 medium activity 변동이 약 0.861까지 나왔습니다. 이 수치는 같은 Training을 본 단변량 설명값이지 CV 성능이 아닙니다.

즉 MCI와 CN의 단일 변수 차이는 작고, DEM과 CN의 차이는 상대적으로 컸습니다. 그래서 큰 모델 하나보다 CN gate, 직접 3-class, probability Random Forest, one-vs-rest를 나란히 고정 비교합니다.

## 평가 방법

Training 141명은 CN 85명, MCI 47명, DEM 9명입니다. DEM이 매우 적어 3-fold보다 많은 fold를 쓰지 않습니다.

- outer CV: 사전에 정한 새로운 seed 5개 × stratified 3-fold
- inner CV: 각 outer-training 안에서 3-fold
- 전처리, 변수 선택, class weight, hyperparameter, blend 선택: 모두 inner/outer 학습 fold 안에서만 수행
- 주 지표: repeat별 subject macro ROC-AUC, subject accuracy, CN-vs-rest AUC
- 함께 확인: macro F1, balanced accuracy, CN precision/recall/specificity, MCI/DEM recall, log loss
- 비교 기준: class-prior와 all-CN baseline

모델 선택 점수는 다음처럼 고정했습니다.

```text
0.45 × macro OVR ROC-AUC
+ 0.30 × CN-vs-rest ROC-AUC
+ 0.15 × subject accuracy
+ 0.10 × macro F1
```

비CN recall이 0.20보다 낮으면 감점을 주어, 모두 CN으로 예측해 accuracy만 높이는 해를 막습니다.

주 결과는 각 repeat의 전체 OOF 점수 평균과 표준편차입니다. 여러 repeat의 확률을 먼저 평균한 단일 OOF 점수는 보조 진단으로만 표시합니다.

## 실행 방법

가장 간단한 실행 파일은 [`../base_sanghyo.ipynb`](../base_sanghyo.ipynb)입니다. 기존 `SangHyo/base.ipynb`는 수정하지 않았고, 새 notebook만 이 폴더를 가리키도록 만들었습니다. Colab에서 위에서 아래로 실행하면 최신 GitHub 저장소를 clone하고 결과와 zip archive를 Google Drive의 `MyDrive/SangHyo_CNBoost_Results`에 저장합니다.

Colab A100/High-RAM에서 저장소를 clone하고 Google Drive를 mount한 뒤 실행합니다.

```bash
python -m pip install -r SangHyo/ThreeClass_GoogleYDF_CNBoost/requirements_colab.txt

python SangHyo/ThreeClass_GoogleYDF_CNBoost/eda.py \
  --training-root /content/drive/MyDrive/GoogleAI_contest/Data/1.Training \
  --output-dir /content/drive/MyDrive/SangHyo_CNBoost_Results/eda

python SangHyo/ThreeClass_GoogleYDF_CNBoost/train.py \
  --training-root /content/drive/MyDrive/GoogleAI_contest/Data/1.Training \
  --validation-root /content/drive/MyDrive/GoogleAI_contest/Data/2.Validation \
  --output-dir /content/drive/MyDrive/SangHyo_CNBoost_Results/training
```

짧은 동작 점검은 `--fast`를 추가합니다. 이 모드는 seed 1개, 후보별 trial 1개, 작은 tree/epoch를 사용하므로 성능 보고용이 아닙니다.

TabNet을 빼고 YDF 네 후보만 확인하려면 `--skip-tabnet`을 추가합니다. 배포용 full-training checkpoint가 불필요한 진단 실행에는 `--no-full-checkpoint`를 사용할 수 있습니다.

## 출력과 체크포인트

주요 결과:

- `nested_cv_report.json`: repeat별/평균·표준편차 OOF 결과
- `outer_fold_metrics.csv`: fold별 결과
- `nested_oof_repeat_XX_hashed.csv`: repeat별 OOF 확률
- `FINAL_REPORT.json`: 핵심 요약
- `validation_report.json`: 이미 여러 번 사용된 33명 Validation의 역사적 결과

모델 체크포인트:

- `models/repeat_.../candidate/`: 모든 outer-fold selector와 YDF/TabNet 모델
- 각 candidate의 `checkpoint_identity.json`: seed, split hash, class 순서, code hash
- 각 candidate의 `CHECKPOINT_COMPLETE.json`: 저장 완료 표식
- `models/full_training_refit/`: Training 전체로 다시 맞춘 배포용 후보
- `FULL_CHECKPOINT_COMPLETE.json`: full checkpoint 완료 표식

Validation label은 `validation_predictions_label_free_hashed.csv`와 `VALIDATION_PREDICTIONS_FROZEN.json`이 먼저 저장된 다음에만 열립니다. 이 Validation은 이미 과거 실험에서 반복 사용됐으므로 fresh test가 아니라 역사적 참고 결과로 해석해야 합니다.

## EDA 출력

`eda.py`는 모델을 학습하지 않습니다.

- `EDA_REPORT_KO.md`: 쉬운 요약
- `feature_effects.csv`: MCI/DEM/비CN 대 CN 효과
- `cn_top_features_bootstrap.csv`: CN 상위 패턴과 bootstrap 범위
- `feature_family_summary.csv`: 수면·활동 패턴 묶음 요약
- `class_counts.png`, `cn_top_effects.png`: 그림
- `eda_audit.json`: MMSE/Validation 비접근 및 개인정보 비저장 확인

EDA 순위는 설명용이며 `train.py`가 직접 읽지 않습니다. 이를 통해 전체 Training 라벨로 미리 변수를 고른 뒤 같은 Training nested CV를 평가하는 낙관 편향을 줄였습니다.

## 현실적인 기대치

목표가 높은 것은 좋지만 141명, 특히 DEM 9명만으로 accuracy 0.8 이상을 보장할 수는 없습니다. 이번 단계의 성공 기준은 단일 validation accuracy가 아니라 다음을 함께 만족하는지입니다.

1. fresh 5-repeat nested macro ROC-AUC와 accuracy가 이전 MMSE-free 기준보다 일관되게 높음
2. CN-vs-rest AUC가 올라가면서 비CN recall이 무너지지 않음
3. fold/seed에 따라 결과 방향이 과도하게 뒤집히지 않음
4. all-CN 및 class-prior baseline보다 macro F1·balanced accuracy가 분명히 높음
5. MMSE나 coverage shortcut 없이 얻은 결과임

이 조건을 통과하기 전에는 0.8이라는 숫자만으로 임상적 성능을 주장하지 않는 것이 안전합니다.
