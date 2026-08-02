# 웨어러블 기반 CN vs MCI+DEM 이진 분류

이 폴더는 사람별 Activity와 Sleep 기록만 사용해 다음 두 집단을 분류합니다.

- `CN` → class 0
- `IMPAIRED` (`MCI + DEM`, 저장 형식에 따라 `MCI_DEM`으로 표시 가능) → class 1

Training 기준으로는 `CN 85명` 대 `MCI 47명 + DEM 9명 = 56명`입니다. 하루 한
건이 아니라 **사람 한 명이 학습 표의 한 행**입니다. 목표 Accuracy는 `0.90` 이상이지만,
이는 실험 목표이지 코드가 보장하는 결과가 아닙니다. 데이터 수가 작고 두 집단의
웨어러블 분포가 겹칠 수 있으므로 목표 미달 결과도 그대로 저장하고 보고해야 합니다.

## MMSE는 완전히 제외합니다

이 실험의 입력은 Activity와 Sleep뿐입니다. Training과 Validation의 MMSE 파일을
읽지 않으며, MMSE TOTAL·문항별 점수·MMSE 파일 안의 진단 열을 특징으로 사용하지
않습니다. 파일 경로나 열 이름에 `MMSE`가 들어간 값도 특징 허용 목록에 들어갈 수
없도록 fail-closed 검사를 둡니다. 따라서 이 결과를 “웨어러블 + 인지검사” 성능으로
해석하면 안 됩니다.

정답은 `LabelingData/1.Gait`와 `LabelingData/2.Sleep`의 두 사본만 읽어 서로
일치하는지 확인합니다. `SourceData/3.CognitiveFunction`뿐 아니라
`LabelingData/3.CognitiveFunction` 경로도 탐색하거나 열지 않습니다.

다음 값도 모델 입력에서 제외합니다.

- `DIAG_NM`, `DIAG_SEQ`와 그 밖의 정답 또는 정답을 직접 파생한 열
- 이메일, 원본 사람 ID, 기기/기록 ID
- 절대 날짜와 원본 행 순서
- Validation 정답에서 계산한 통계, 특징, threshold 또는 ensemble 가중치

사람 ID는 Activity·Sleep·label을 연결하고 fold를 나누는 데만 사용합니다. 저장
산출물에는 원본 ID 대신 해시 ID를 사용합니다.

## EDA와 전처리

EDA는 Training 데이터만으로 다음을 확인하고 결과를 `outputs/<run_id>/eda/` 아래에
저장합니다.

1. 파일/열 계약, 중복, 결측률과 비정상 값
2. class 수와 사람별 기록 일수 분포
3. Activity·Sleep 변수의 분포, 극단값과 사람 단위 feature 품질
4. 사람 단위 집계 특징의 class별 요약과 효과 크기
5. 수집량 또는 결측 패턴이 정답의 대리변수가 될 위험

일별 원본은 사람 단위의 robust 통계, 분위수, 변동성, 시간대/수면 단계 비율과 같은
특징으로 집계합니다. 한 사람의 일별 행을 서로 다른 fold로 나누지 않습니다. 결측값
대치, scale, 분산/상관 기반 제거 및 특징 선택은 매 fold의 Training 부분에서만
학습한 뒤 해당 fold의 검증 사람에게 적용합니다. 전체 표에서 먼저 전처리 통계를
계산하면 검증 정보가 새므로 금지합니다.

사람별 특징 생성기는 저장소의 기존 감사 구현을 재사용하되, 두 Python 구현 파일의
SHA-256을 새 파이프라인에 고정합니다. 구현이 바뀌면 조용히 다른 실험을 실행하지
않고 검토 후 hash를 명시적으로 갱신하라는 오류로 중단합니다.

## 모델과 출처를 정확히 구분하기

세 모델군의 확률을 Training의 안쪽 검증 결과만으로 비교하거나 ensemble합니다.

- **Yggdrasil Decision Forests (YDF)**: Google이 공개한 decision-forest
  라이브러리의 Gradient Boosted Trees/Random Forest 계열입니다.
- **TabNet**: Google Cloud AI 연구진이 제안한 attention 기반 표 데이터 모델입니다.
- **수치 토큰 Transformer**: Google 연구진이 제안한 원래 Transformer 구조를
  수치형 표 데이터에 작게 적용한 이 실험 전용 모델입니다. **FT-Transformer 자체는 Google 모델이 아니며
  Yandex Research 연구진이 제안했습니다.** 이 코드는
  FT-Transformer라고 부르거나 그 출처를 Google로 표현하지 않습니다.

1차 자료:

- [Google Research — TabNet](https://research.google/pubs/tabnet-attentive-interpretable-tabular-learning/)
- [Google Research — Attention Is All You Need](https://research.google/pubs/attention-is-all-you-need/)
- [Google — Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
- [Yandex Research — FT-Transformer 공식 구현/논문](https://github.com/yandex-research/rtdl-revisiting-models)

즉 “Google 계열 모델 활용”은 YDF와 TabNet을 직접 사용하고 Google-origin
Transformer를 포함한다는 뜻입니다. FT-Transformer의 출처까지 Google로 돌리는
뜻은 아닙니다.

## 사람 단위 repeated nested CV

Training 141명에서 사람 단위 repeated nested cross-validation을 수행합니다.

```text
Training subjects
└─ repeated outer stratified folds
   ├─ outer validation subjects: 최종 OOF 성능 측정에만 사용
   └─ outer training subjects
      └─ inner stratified folds
         ├─ 사전 고정된 후보 모델의 OOF 확률 비교
         ├─ ensemble 가중치 선택
         └─ binary threshold 선택
```

같은 사람의 Activity/Sleep 기록이 학습과 검증에 동시에 나타날 수 없습니다. 일별
행을 무작위 K-fold에 넣는 방식은 표본 수를 부풀리고 개인 습관을 외우게 하므로
사용하지 않습니다. 보고서에는 outer OOF Accuracy와 함께 ROC-AUC, F1, balanced
accuracy, confusion matrix 및 반복/fold별 변동을 기록합니다.

모델 hyperparameter는 대규모 Optuna/search로 찾지 않고 코드에 사전 등록한 보수적
설정을 사용합니다. fold-local 전처리는 각 Training fold에만 fit하지만, Validation
결과를 보고 모델 설정이나 특징 규칙을 다시 고르지 않습니다.

## Validation 예측 동결

Validation 33명은 모델/특징/threshold/ensemble 선택에 쓰지 않습니다. 모든 선택은
Training nested CV에서 끝냅니다. 이후 10개 outer-fold 모델의 cross-fitted
ensemble로 Validation의 **라벨 없는 예측**을 먼저 만들고 다음 산출물과 SHA-256을
저장해 동결합니다.

```text
validation_predictions_label_free_hashed.csv
VALIDATION_PREDICTIONS_FROZEN.json
```

동결 뒤에만 Validation label을 열어 한 번 평가합니다. 결과를 본 다음 seed, 특징,
threshold 또는 ensemble 비율을 고쳐 같은 Validation에 다시 맞추면 그 점수는 최종
검증 점수가 아닙니다.

Validation이 정확히 33명이라면 Accuracy `0.90` 이상에는 최소 `30/33` 정답이
필요합니다. `29/33 = 0.879`이고 `30/33 = 0.909`이므로 “대략 90%”와 “목표 통과”를
혼동하지 않습니다.

## 실행

저장소 루트의 [`base.ipynb`](../../base.ipynb) 셀 2에서 `USER_FOLDER`는 그대로 두고
`RUN_FILE`만 아래처럼 바꿉니다.

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Wearable_GoogleModels/run.py"
```

`run.py`가 EDA, 전처리, nested CV, 최종 학습, Validation 예측 동결과 보고서 생성을
한 번에 조정하므로 다른 Python 파일을 따로 실행할 필요가 없습니다. 기본 모드는
`full`입니다. hyperparameter search 대신 사전 고정한 보수적 모델 설정과 신경망
epoch 상한을 사용합니다. Training에는 **5시간 45분 soft budget**을 전달하고,
launcher는 **6시간 hard limit**을 둡니다. Linux/macOS/Colab에서는 hard guard가
6시간을 넘긴 실행을 중단하며, hard guard를 제공하지 않는 플랫폼에서는 경고를
출력합니다. soft budget에 먼저 도달하면 부분 checkpoint와 진행 파일을 남기고
실패로 종료하므로 이를 완료 점수로 보고하지 않습니다.

먼저 설치와 데이터 계약만 빠르게 확인하려면 실행 전에 환경변수를 설정합니다.

```bash
BINARY_RUN_MODE=smoke python SangHyo/Binary_Wearable_GoogleModels/run.py
```

Colab에서는 `run.py`가 읽히기 전에 다음을 한 번 실행한 뒤 같은 `RUN_FILE`로 실행할
수 있습니다.

```python
import os
os.environ["BINARY_RUN_MODE"] = "smoke"
```

`smoke` 결과는 배선 확인용이며 정식 성능으로 보고하지 않습니다. 정식 실험은 새
런타임에서 환경변수를 지우거나 `full`로 설정해 처음부터 실행합니다.

## 주요 파일과 산출물

```text
Binary_Wearable_GoogleModels/
├── run.py                    # base.ipynb가 실행하는 단일 진입점
├── data.py                   # Activity/Sleep 로딩, label 병합, 사람 단위 집계
├── eda.py                    # Training-only EDA
├── preprocessing.py          # fold-local 전처리와 특징 선택
├── models.py                 # YDF, TabNet, Transformer 모델
├── train.py                  # repeated nested CV, ensemble, freeze/evaluation
├── requirements_colab.txt
└── tests/test_static_contracts.py
```

각 실행은 `outputs/<run_id>/` 아래에 설정과 seed, EDA, fold별/OOF 지표, 선택 특징,
모델 또는 checkpoint, label-free Validation 확률, freeze manifest와 최종 보고서를
남깁니다. launcher는 완료 또는 실패 상태도 별도 JSON으로 기록합니다. 대용량 출력,
cache와 checkpoint는 Git에 커밋하지 않습니다.

## 성능과 누수 해석 주의

- Accuracy 0.90은 목표이며 사전 보장이 아닙니다. 가장 좋은 seed만 골라 목표 달성으로
  보고하지 않습니다.
- nested OOF가 모델 선택의 근거이고, Validation은 동결 뒤 한 번 보는 최종 확인입니다.
- 사람 수가 작으므로 단일 split 점수보다 반복 outer fold와 fold별 변동을 봅니다.
- 한 사람의 여러 날이 서로 다른 fold에 들어가면 그 실험은 무효입니다.
- 전체 데이터로 imputer/scaler/feature selector를 먼저 fit하면 누수입니다.
- 수집 일수, 결측률, 기기/사이트 차이가 진단의 대리변수가 아닌지 EDA에서 확인합니다.
- 33명 Validation에서 한 명은 Accuracy 약 3.03%p에 해당합니다.
- 이 코드는 연구용 분류 실험이며 의료 진단 도구가 아닙니다.
