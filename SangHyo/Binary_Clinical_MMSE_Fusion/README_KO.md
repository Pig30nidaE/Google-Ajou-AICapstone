# Binary_Clinical_MMSE_Fusion (웨어러블 + MMSE)

## 한 줄 요약

Activity·Sleep 웨어러블 요약에 **MMSE(인지검사) 점수를 더해** `CN`(정상) vs
`MCI+DEM`(인지저하)을 구분합니다. 위의 `Binary_Wearable_ConvBiLSTM_NoMMSE`가
"웨어러블만"이라면, 이 폴더는 **"웨어러블 + 인지검사"** 버전으로, MMSE가
성능에 얼마나 도움이 되는지 직접 비교하기 위한 짝입니다.

## MMSE를 어떻게 쓰나 (그리고 무엇을 조심하나)

- 이 실험은 목적상 `SourceData/3.CognitiveFunction`의 MMSE 파일을 **일부러
  엽니다.** 그리고 **32개의 실제 인지검사 점수**(문항 Q01~Q19, 구역 합계
  `Q12_TOTAL`, 총점 `TOTAL`)만 특징으로 씁니다.
- **절대 특징으로 쓰지 않는 것**: `DIAG_NM`(진단명 = 바로 정답이라 쓰면
  컨닝), `DIAG_SEQ`(진단 순서 코드), `DOCTOR_NM`·`MMSE_NUM`·`MMSE_KIND`(검사
  자체가 아닌 행정용 값). 코드에 이 컬럼들이 특징에 섞이면 즉시 중단하는
  안전장치(fail-closed)를 넣었습니다.
- 정답 라벨은 MMSE 파일이 아니라 **Gait/Sleep의 라벨 사본**에서만 읽습니다
  (웨어러블 실험과 동일). 즉 "정답은 라벨 파일에서, MMSE는 점수만" 원칙입니다.

## 왜 트리·회귀 앙상블인가 (아주 쉽게)

MMSE 점수는 사람마다 한 줄로 딱 떨어지는 **깨끗한 표 데이터**입니다. 이런
데이터는 큰 신경망보다 **잘 정규화된 트리·회귀 모델**이 더 안정적이고 빠릅니다
(141명처럼 적은 표본에서 특히 그렇습니다). 그래서 GPU 없이 CPU만으로 몇 분 안에
끝나는 견고한 앙상블을 썼습니다.

- 입력 = 웨어러블 요약(각 채널의 중앙값·IQR·추세, 168개) + MMSE 점수(32개).
- 모델 = `gbt`(HistGradientBoosting) + `logreg`(L2 로지스틱 회귀) +
  `rf`(랜덤포레스트), 모두 클래스 가중치로 불균형 보정.
- 각 fold의 학습 데이터 안에서만 결측 대치·정규화·특징 선택(최대 40개)을
  하고, 세 모델을 **내부 교차검증 성능**에 비례해 가중 평균합니다.

## 누수를 막는 방법

- 사람 단위 **5-겹 × 5-반복 중첩 교차검증**(웨어러블 실험보다 반복을 늘려
  변동을 더 촘촘히 봄).
- 모든 전처리·특징 선택·가중치·임계값은 그 fold의 학습 데이터에서만 학습.
- 검증 33명은 예측을 먼저 저장(+SHA-256 동결)한 **뒤에야** 정답을 열어 한 번만
  채점. 학습·검증에 겹치는 사람이 있으면 즉시 중단.

## 목표와 정직한 기대치

- 목표: 정확도 **0.90 이상**, 균형정확도 **0.80 이상**.
- 정직한 예상: MMSE를 넣으면 웨어러블만일 때보다 **확실히 좋아집니다.** 다만
  0.90/0.80까지 항상 도달한다고 보장할 수는 없습니다. 이유는 임상적으로
  **MCI(경도인지장애) 환자의 MMSE 총점이 정상(CN)과 많이 겹치기** 때문입니다
  (이 데이터: CN 총점 평균 27.6 vs MCI 25.6, 치매 19.2). 즉 치매는 잘
  갈라지지만 "CN vs MCI"가 어려워서, MMSE를 써도 이 경계가 성능의 한계를
  만듭니다.
- 그래서 정확도 하나가 아니라 **균형정확도·인지저하 재현율·CN 특이도·ROC-AUC·
  혼동행렬·부트스트랩 신뢰구간**을 함께 보세요. 검증 33명은 CN이 26명이라
  **전부 CN으로 찍어도 정확도 0.79**가 나옵니다.

## 실행 방법 (Colab, base.ipynb)

루트 `base.ipynb` 셀 2에서 아래만 바꿉니다.

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Clinical_MMSE_Fusion/run.py"
```

기본 모드는 `full`입니다. 빠른 배선 확인만 하려면:

```bash
python run.py --mode smoke --output-dir /content/drive/MyDrive/tmp_mmse_smoke
```

### 필요한 실행 환경

- **GPU가 필요 없습니다.** CPU만으로 충분하고, 고용량 RAM도 필요 없습니다.
- 예상 시간: 전체 실행이 **몇 분**이면 끝납니다(6시간 상한 대비 매우 여유).

결과는 `/content/drive/MyDrive/Binary_Clinical_MMSE_Fusion_result/<UTC_실행ID>/`
아래에 실행마다 새 폴더로 저장됩니다.

## 주요 산출물

```text
eda/eda_summary.json                         # 데이터 요약, MMSE/웨어러블 단일 특징 AUC
training/nested_cv_report.json               # 중첩 OOF 성능(핵심)
training/oof_predictions_hashed.csv          # 사람별 OOF 예측(해시 ID)
training/fold_metrics.csv                    # fold별 성능
training/VALIDATION_PREDICTIONS_FROZEN.json  # 검증 예측 동결(SHA-256)
training/validation_predictions_label_free_hashed.csv
training/validation_report.json              # 검증 33명 채점(1회)
training/FINAL_REPORT.json                   # 최종 요약
training/TRAINING_COMPLETE.json
```

## 두 폴더를 함께 보기

`Binary_Wearable_ConvBiLSTM_NoMMSE`(웨어러블만)와 이 폴더(웨어러블+MMSE)의
`nested_cv_report.json`을 나란히 비교하면 **"인지검사가 성능을 얼마나
올리는가"** 를 정직하게 확인할 수 있습니다. 두 실험은 같은 141/33 분할과 같은
누수 방지 규칙을 쓰므로 비교가 공정합니다.

## 해석상 주의

MMSE는 임상에서 진단을 내릴 때 참고하는 검사이므로, "MMSE로 진단을 예측"하는
것은 어느 정도 **자기참조적**입니다. 이 실험은 MMSE가 실제로 얼마나 정보를
더하는지를 웨어러블만 버전과 비교해 보여주기 위한 것이며, 결과를 독립적인
진단 성능으로 과대 해석하면 안 됩니다. 검증 33명도 재사용된 과거 벤치마크이며,
이 코드는 연구용이지 의료 진단 도구가 아닙니다.
