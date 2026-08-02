# Wearable SequenceFusion: CN vs MCI+DEM

이 폴더는 Activity와 Sleep만으로 `CN=0` 대 `MCI+DEM=1`을 분류하는 새
실험입니다. 목표 Accuracy는 0.90 이상이지만, 이는 **실험 목표이지 보장 성능이
아닙니다**. MMSE 값·문항·파일과 CognitiveFunction 폴더는 전혀 읽지 않습니다.
라벨은 Gait와 Sleep 폴더에 있는 두 사본이 정확히 같은지만 확인합니다.

## 이전 결과에서 확인한 문제

직전 `Binary_Wearable_GoogleModels`의 Training nested OOF Accuracy는 약 0.532,
historical Validation Accuracy는 24/33=0.727이었습니다. Validation은 CN이 26명이라
모두 CN으로 예측해도 26/33=0.788입니다. 따라서 Accuracy 하나만으로 성능 향상을
판단할 수 없고 impaired recall, balanced accuracy, ROC-AUC와 confusion matrix를
같이 봐야 합니다.

직전 전처리는 2,640개 후보를 만들었지만 10개 outer fold 모두에서 동일한 24개를
골랐습니다. 이 24개는 사실상 알파벳순으로 앞선 `event28 median`이어서 event7/14,
IQR, MAD, 추세와 최근 변화가 사라졌습니다. Transformer의 평균 AUC가 약 0.606으로
가장 높았지만 log loss가 매우 커 calibration도 불안정했습니다.

`SangHyo/previous/Experiment2.py`의 Conv1D+BiLSTM은 구조적 참고 대상입니다. 다만
저장소에는 0.84 결과를 검증할 metrics/예측 산출물이 없습니다. 당시 Validation은
32명(CN 26명)이므로 0.84375가 subject accuracy였다면 27/32이고, all-CN 기준
26/32보다 한 명을 더 맞힌 값일 수 있습니다. 또한 전체 Training에서 RF top-20을
먼저 고른 뒤 CV를 수행해 feature-selection 누수가 있었습니다.

## 새 전처리

새 파이프라인은 모든 사람에게 정확히 **8개 × 28 observed-event view**를 부여합니다.
각 view는 같은 길이이며 padding, mask, 관측 일수, 결측률, calendar gap, 절대 날짜,
ID와 원본 행 순서를 입력하지 않습니다. 따라서 기록이 긴 사람이 더 많은 window로
학습을 지배하던 **행 가중치 불균형**을 제거합니다. 다만 8개 crop은 각 사람의 전체
관측 구간에 고르게 놓이므로, 기록 길이는 직접 feature가 아니어도 crop 중복도와
포괄 기간에 간접 영향을 줄 수 있습니다. 이 점은 결과 해석 시 별도 한계로 남깁니다.

일별 입력에는 안전한 Activity/Sleep scalar와 다음 생리적 요약을 포함합니다.

- activity class/MET의 분위수, 변동성, 강도 비율, entropy와 전이율
- sleep HR/RMSSD의 robust 통계와 변화, sleep-stage 비율·entropy·전이율
- bedtime/wake clock의 sin/cos 표현과 수면단계 duration ratio

각 outer fold의 Training 사람만 사용해 signed-log1p 허용 목록, median 대치,
1/99 percentile winsorization, median/IQR scaling을 fit합니다. YDF에는 각
28-event view의 median, IQR, MAD, p10, p90, normalized Theil-Sen slope,
최근 7일-이전 21일 차이를 만들고, 사람별 8개 view를 같은 가중치로 평균합니다.
이 summary bank 자체는 label 없이 생성됩니다. 이후 각 outer-training fold 안에서만
32회 stratified bootstrap 안정성 선택을 수행해 최대 160개를 남깁니다. Activity/Sleep
각 40개와 각 통계 종류 8개를 우선 선택한 뒤 높은 상관의 중복을 제거하려 시도하며,
실제 선택 수는 fold별 manifest에 기록합니다. 따라서
특정 통계나 알파벳 앞부분만 남았던 이전 선택기의 문제를 반복하지 않습니다.

## 모델

- `conv_bilstm`: 기존 실험을 참고하되 fold-safe 전처리와 subject-balanced view를 쓰는
  regularized Conv1D + BiLSTM
- `sequence_transformer`: Google Research의 원 Transformer 구조를 28-event wearable
  sequence에 적용한 작은 temporal Transformer
- `ydf_gbt`, `ydf_rf`: Google Yggdrasil Decision Forests의 GBT와 Random Forest

네 확률은 사전에 고정한 비율 `YDF GBT 0.20 + YDF RF 0.10 + ConvBiLSTM 0.40 +
Sequence Transformer 0.30`으로 결합합니다. Validation을
보고 모델이나 가중치를 고르지 않습니다. 신경망 epoch만 outer-training 안의 별도
subject holdout으로 고른 뒤 outer-training 전체를 다시 학습합니다. 평가는 5-fold ×
2-repeat subject CV이며 같은 사람의 view가 서로 다른 fold에 들어가지 않습니다.
outer fold에서 선택된 epoch의 median을 동결한 뒤, Validation 예측용 모델은 Training
141명 전체로 두 seed를 각각 refit합니다. 따라서 80% fold 모델만 평균해 최종 예측하는
것이 아니라 모든 Training 사람을 사용한 확률 평균을 동결합니다.

OOF threshold는 0.35~0.75의 거친 사전 고정 grid에서 Training OOF만으로 선택합니다.
impaired recall 0.40과 CN specificity 0.60을 동시에 만족하는 점만 비교하며, 가능한
점이 없으면 두 class recall 중 더 낮은 값을 최대화하는 보수적 fallback을 사용합니다.
OOF `threshold=0.5`가 조정되지 않은 1차 지표이고, 선택 threshold의 OOF 지표는 같은
OOF label로 threshold를 선택했다는 주의 문구와 함께 2차 지표로 저장합니다.

## Validation 해석

Validation 확률은 정답을 열기 전에 아래 파일로 먼저 저장하고 SHA-256을 동결합니다.

```text
validation_predictions_label_free_hashed.csv
VALIDATION_PREDICTIONS_FROZEN.json
```

그 뒤에만 Gait/Sleep label 사본을 열어 평가합니다. 그러나 이 33명 Validation은 이미
이전 여러 실험에서 재사용된 **historical benchmark**입니다. 여기서 0.90 이상이 나와도
새로운 독립 test 또는 fresh test에서 재현하기 전에는 일반화 성능 0.90으로 주장하면
안 됩니다. 33명에서 목표 통과는 최소 30/33=0.909이며 한 명이 약 3.03%p입니다.

## 실행

루트 [base.ipynb](../../base.ipynb)의 셀 2에서 `RUN_FILE`만 바꿉니다.

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Wearable_SequenceFusion_Google/run.py"
```

`run.py` 하나가 dependency 확인, Training-only EDA, 전처리, repeated CV,
Validation label-free freeze와 historical 평가까지 실행합니다. 기본은 `full`입니다.
빠른 배선 확인은 실행 전에 다음만 지정합니다.

```python
import os
os.environ["BINARY_SEQUENCE_RUN_MODE"] = "smoke"
```

smoke 결과는 성능으로 보고하지 않습니다. 10-fold 전체 full 실행 시간은 로컬에서
실측하지 않았고, 각 YDF native fit도 최대 5분으로 제한했습니다. full에는 요청하신 Colab **A100을
권장하며 기본적으로 CUDA가 없으면 중단**합니다. ConvBiLSTM과 Transformer가 A100을
사용하고 Google YDF는 같은 Colab 인스턴스의 CPU 코어를 사용합니다. 다른 GPU나
별도 TPU는 필요하지 않습니다. YDF 때문에 CPU도 일부 사용되지만 A100 런타임에
포함된 CPU면 충분합니다. soft limit는 5시간 45분, hard limit는 6시간입니다.

결과는 매 실행마다 Google Drive의
`/content/drive/MyDrive/Binary_Wearable_SequenceFusion_Google_result/<UTC run id>/`
에 저장됩니다. UTC 실행 ID를 분리하므로 이전 결과를 덮어쓰지 않습니다. Drive가
마운트되지 않았으면 로컬 런타임에 잘못 저장하지 않고 즉시 중단합니다. 핵심 파일은
`training/FINAL_REPORT.json`, `training/oof_report.json`,
`training/historical_validation_report.json`, hashed prediction CSV와
`training/models/final_full_training_refits/` 아래의 최종 체크포인트입니다.
