# CN / MCI / DEM 분류 실험 안내서

이 폴더는 한 사람의 활동, 수면, 인지검사 정보를 모아 `CN / MCI / DEM`을
구분하는 새 실험입니다. 기존 `SangHyo` 실험은 수정하지 않았고,
`base_sanghyo.ipynb`만 이 폴더를 실행하도록 바꿨습니다.

코드는 준비했지만 **모델 학습은 실행하지 않았습니다.** 학습은 Colab A100에서
사용자가 직접 실행하도록 구성했습니다.

## 한눈에 보기

- 목표: Accuracy 0.80 이상
- 함께 확인할 값: Macro F1 0.70 이상, ROC-AUC 0.85 이상
- 학습 단위: 하루가 아니라 사람 1명
- Training: 141명 (`CN 85 / MCI 47 / DEM 9`)
- 모델: FT-Transformer, TabNet, Google YDF
- 최종 답: 세 모델의 확률을 안쪽 검증 결과에 따라 섞은 ensemble
- 결과 저장: Google Drive 폴더와 ZIP 압축본

위 숫자는 **목표 기준**입니다. 특히 DEM이 9명뿐이므로 0.8을 미리 보장할 수는
없습니다. 코드도 목표에 못 미치는 결과를 숨기지 않고 그대로 기록합니다.

## EDA에서 확인한 내용

Training 원본만 사용해 EDA를 실제로 실행했습니다. Validation 라벨과 예측 모델은
사용하지 않았습니다. 자세한 결과는
[`eda_outputs/EDA_REPORT_KO.md`](eda_outputs/EDA_REPORT_KO.md)에 있습니다.

눈에 띈 점은 다음과 같습니다.

1. Activity와 Sleep은 각각 9,705행이고 모든 141명이 들어 있습니다.
2. 한 사람당 수집 일수 중앙값은 대략 61~67일입니다. class마다 수집량이 조금
   달라, 수집 일수 자체를 모델에 넣으면 병이 아니라 수집 방식을 외울 수 있습니다.
3. DEM은 CN보다 쉬는 시간이 길고, 낮은 활동 상태가 많으며, 수면의 light 단계와
   몇몇 수면 점수의 차이가 비교적 크게 보였습니다.
4. MCI와 CN의 차이는 DEM과 CN보다 작았습니다. MCI가 가장 어려운 구분일 가능성이
   큽니다.
5. MMSE TOTAL 중앙값은 `CN 28 / MCI 26 / DEM 22`였습니다. 다만 범위가 겹치므로
   점수 하나만으로 완벽히 나뉘지는 않습니다.
6. MMSE 파일의 `DIAG_NM`은 정답과 100% 같습니다. 이를 입력하면 성능이 아니라
   정답 복사가 되므로 강제로 차단했습니다.

EDA가 만든 입력 후보는 754개입니다. 모두 쓰지 않고, 매 검증 fold의 Training
부분에서만 쓸 특징을 다시 고릅니다. 이렇게 해야 검증 사람의 정보가 전처리에
미리 들어가지 않습니다.

## 어떤 정보를 쓰는가

기본값인 `clinical_plus_lifelog`은 다음을 함께 씁니다.

- Activity: 활동량, 걸음 수, 칼로리, MET, 활동 상태의 비율과 변화
- Sleep: 수면 단계, 수면 점수, 심박, HRV, 호흡, 취침 시각의 주기 표현
- MMSE: 문항별 점수와 TOTAL

아래 정보는 어떤 모드에서도 쓰지 않습니다.

- `DIAG_NM`, `DIAG_SEQ`: 정답 또는 사실상 정답인 열
- 이메일/사람 ID, 의사명, 검사 순번, 기기 기록 ID
- 연·월·일 같은 절대 날짜
- 수집 일수, 빈 날짜 수, 수집 시작일처럼 수집 방식을 드러내는 값
- 마지막 Activity 시점보다 미래인 Sleep 기록

인지검사 없이 웨어러블만 비교하고 싶으면 노트북에서 다음과 같이 바꿉니다.

```python
FEATURE_MODE = "wearable_only"
```

두 모드는 질문이 다릅니다. 기본 모드는 “현재 사용할 수 있는 인지검사와 생활기록을
합치면 얼마나 잘 구분하는가”, 비교 모드는 “생활기록만으로 얼마나 구분하는가”를
봅니다. 보고서에서 둘을 같은 성능처럼 섞어 말하면 안 됩니다.

## 세 모델을 함께 쓰는 이유

### FT-Transformer

754개 후보 중 fold 안에서 고른 특징들을 작은 token처럼 다루고, 서로 어떤 조합이
중요한지 찾습니다. A100을 가장 직접적으로 활용하는 모델입니다. Transformer는
Google 연구진이 처음 제안한 구조입니다.

### TabNet

여러 단계에 걸쳐 “이번 판단에서 어떤 특징을 볼지” 고르는 표 데이터 모델입니다.
활동·수면·MMSE 중 상황에 따라 다른 정보에 집중할 수 있습니다. TabNet 역시
Google 연구에서 발표했습니다.

### Google YDF

Yggdrasil Decision Forests의 Gradient Boosted Trees를 사용합니다. 데이터가 141명처럼
작을 때 신경망만 사용하면 불안정할 수 있어, 표 데이터에 강한 나무 모델을 함께
둡니다. YDF는 CPU를 사용하지만 A100 Colab의 고용량 RAM/CPU에서 함께 실행됩니다.

세 모델은 같은 실수를 하지 않을 가능성이 있습니다. 안쪽 검증에서만 ensemble
비율과 CN/MCI/DEM 확률 보정을 고른 뒤, 바깥쪽 사람에게 적용합니다.

공식 참고 자료:

- [Google Research: Attention Is All You Need](https://research.google/pubs/attention-is-all-you-need/)
- [Google Research: TabNet](https://research.google/pubs/tabnet-attentive-interpretable-tabular-learning/)
- [Google YDF 공식 저장소](https://github.com/google/yggdrasil-decision-forests)
- [YDF Gradient Boosted Trees 문서](https://ydf.readthedocs.io/en/stable/py_api/GradientBoostedTreesLearner/)

## 검증 방식

사람 한 명이 Training과 검증에 동시에 들어가지 않도록, 처음부터 한 사람당 한 행을
만듭니다.

```text
Training 141명
  └─ 바깥 3-fold × 2회: 최종 내부 성능 측정
       └─ 각 바깥 Training 안의 3-fold
            ├─ 모델 설정 탐색
            ├─ 세 모델을 섞는 비율 선택
            └─ class별 확률 보정 선택
```

Validation은 위 선택에 참여하지 않습니다. 모든 바깥 모델의 label-free 예측 CSV를
먼저 저장한 뒤에만 Validation 라벨을 열어 한 번 평가합니다.

기록하는 값은 다음과 같습니다.

- Accuracy: 전체 중 맞힌 비율
- Macro F1: CN, MCI, DEM을 똑같이 중요하게 보고 계산한 점수
- ROC-AUC: 각 class를 다른 class보다 앞에 놓는 능력
- Balanced accuracy: 세 class의 맞힌 비율을 같은 비중으로 평균
- class별 precision, recall, F1과 confusion matrix
- log loss: 맞고 틀림뿐 아니라 확률이 지나치게 자신만만한지도 확인

## 실행 방법

1. Google Drive에 이 저장소 전체를 둡니다.
2. Colab에서 [`../base_sanghyo.ipynb`](../base_sanghyo.ipynb)를 엽니다.
3. 런타임을 A100 GPU와 High-RAM으로 설정합니다.
4. 경로 셀의 자동 탐색이 실패하면 `PROJECT_ROOT_OVERRIDE`만 수정합니다.
5. 먼저 `RUN_MODE = "smoke"`로 실행 흐름을 확인합니다.
6. 새 런타임을 시작해 `RUN_MODE = "full"`로 위에서 아래까지 실행합니다.

기본 Drive 결과 위치는 다음과 같습니다.

```text
/content/drive/MyDrive/SangHyo_CN_MCI_DEM_Results/
```

다른 곳에 저장하려면 노트북에서 `DRIVE_RESULTS_ROOT_OVERRIDE`를 바꿉니다. 학습 결과,
EDA, 모델, 설정, 예측표가 모두 이 폴더에 직접 기록되고 마지막 셀에서 ZIP도 만듭니다.

## Full 실행 설정

- 바깥 검증: 3-fold × 2회
- 안쪽 검증: 각 바깥 fold 안에서 3-fold
- Transformer 탐색: 바깥 fold마다 24회
- TabNet 탐색: 바깥 fold마다 24회
- YDF 탐색: 바깥 fold마다 40회
- 기본 seed: `20260719`

시간을 줄이기 위해 탐색 횟수를 줄일 수는 있지만, 그러면 같은 실험으로 비교하기
어렵습니다. Full 결과를 정식 결과로 쓸 때는 기본값을 유지하는 편이 좋습니다.

## 주요 파일

```text
ThreeClass_TransformerTabNet_Google/
├── README_KO.md                     # 지금 보고 있는 쉬운 안내서
├── eda.py                           # Training-only EDA
├── feature_engineering.py           # 활동·수면·MMSE 가공
├── preprocessing.py                 # fold 안에서만 전처리/특징 선택
├── models.py                        # Transformer, TabNet, Google YDF
├── train.py                         # nested-CV, ensemble, Validation 평가
├── requirements_colab.txt
├── eda_outputs/                     # 실제 실행한 집계 EDA 결과
└── tests/test_static_contracts.py
```

## 학습 뒤 꼭 볼 파일

```text
training/FINAL_REPORT.json
training/nested_cv_report.json
training/outer_fold_metrics.csv
training/nested_oof_predictions_hashed.csv
training/validation_predictions_label_free_hashed.csv
training/validation_report.json
training/selected_features_by_fold.csv
training/TRAINING_COMPLETE.json
eda/EDA_REPORT_KO.md
```

`FINAL_REPORT.json`의 `target_check_nested_oof`가 목표 통과 여부를 보여줍니다.
Validation 목표는 `validation_report.json`에서 따로 확인합니다. OOF가 높고 Validation이
낮으면 과적합 또는 Training/Validation 차이일 수 있습니다.

## 성능을 해석할 때 주의할 점

- DEM 9명에서는 한 명이 약 11%입니다. fold별 DEM F1의 흔들림을 꼭 봐야 합니다.
- MMSE를 포함한 점수는 웨어러블만의 성능이 아닙니다.
- MMSE는 현재 진단과 가까운 검사입니다. 실제 서비스에서 예측 시점에 같은 검사가
  없으면 `wearable_only` 결과가 더 맞는 기준입니다.
- 높은 내부 점수도 새로운 병원·기기·수집 방식에서 그대로 유지된다는 뜻은 아닙니다.
- 이 코드는 연구용 분류 실험이며 의료 진단 도구가 아닙니다.

## 이번에 일부러 하지 않은 일

- `DIAG_NM`을 넣어 100%에 가까운 가짜 성능 만들기
- 같은 사람의 여러 날짜를 서로 다른 fold에 나누기
- Validation 결과를 본 뒤 특징, threshold, ensemble 비율 다시 조정하기
- 전체 데이터로 결측값과 scale을 먼저 계산하기
- 0.8에 못 미친 결과를 지우거나 좋은 seed만 골라 보고하기

이 제한이 있어야 실제로 새 사람을 만났을 때의 성능에 조금 더 가까운 값을 얻을 수
있습니다.

