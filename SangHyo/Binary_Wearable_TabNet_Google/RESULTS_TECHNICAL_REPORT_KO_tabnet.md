# Binary Wearable TabNet + YDF 결과 기술 보고서

작성일: 2026-07-23  
분류 문제: `CN` 대 `MCI + DEM`  
입력 데이터: 웨어러블 `Activity`와 `Sleep`만 사용

## 1. 먼저 보는 결론

이번 실행은 **smoke test가 아니라 full 학습**이다. Google Colab의
`NVIDIA A100-SXM4-80GB`에서 약 1시간 36분 동안 실행되었고, 학습과
체크포인트 저장도 정상적으로 끝났다.

다만 성능 목표는 달성하지 못했다.

- 가장 중요하게 봐야 하는 Nested OOF Accuracy는 **50.4%**였다.
- 두 클래스를 공평하게 평가하는 Balanced Accuracy는 **43.0%**였다.
- 실제 MCI+DEM 56명 중 찾아낸 사람은 **4명(7.1%)**뿐이었다.
- ROC-AUC는 **44.6%**로, 새로운 사람을 안정적으로 구분한다고 보기 어렵다.
- 별도 Validation의 Accuracy는 **75.8%**였지만, 그 데이터에서는 모든
  사람을 CN으로만 예측해도 **78.8%**가 나온다. 따라서 75.8%를 좋은
  결과라고 해석하면 안 된다.

즉, 이번 실험은 “코드가 짧게만 실행된 실패”가 아니다. 충분한 full
학습과 저장은 완료되었지만, 모델이 CN과 MCI+DEM의 차이를 안정적으로
학습하지 못한 경우다.

근거 파일은 [최종 보고서](outputs/training/FINAL_REPORT.json),
[Nested CV 상세 결과](outputs/training/nested_cv_report.json),
[fold별 결과](outputs/training/outer_fold_metrics.csv)에 있다.

## 2. MMSE 배제 여부

이번 결과 해석에서는 MMSE를 포함한 과거 실험과 논문의 성능을 핵심
비교 대상으로 사용하지 않았다.

현재 실행 기록에는 다음 내용이 명시되어 있다.

- 열린 입력 종류: `Activity`, `Sleep`
- `cognitive_test_used`: `false`
- 인지검사 원본 파일을 열었는가: `false`
- 인지검사 특징을 만들었는가: `false`

따라서 이 보고서의 성능은 MMSE 점수 없이 얻은 웨어러블-only 결과다.
MMSE가 포함된 과거 수치는 입력 조건이 다르므로 직접 비교에서 제외했다.

## 3. 평가 지표를 아주 쉽게 설명하면

| 지표 | 쉬운 의미 |
| --- | --- |
| Accuracy | 전체 사람 중 정답을 맞힌 비율 |
| MCI+DEM Recall | 실제 MCI+DEM인 사람을 놓치지 않고 찾아낸 비율 |
| CN Specificity | 실제 CN인 사람을 CN으로 올바르게 분류한 비율 |
| Balanced Accuracy | MCI+DEM Recall과 CN Specificity의 평균 |
| ROC-AUC | 임계값 하나에 고정하지 않고, MCI+DEM을 CN보다 높은 위험도로 정렬하는 능력 |

Balanced Accuracy는 두 클래스의 정답률을 같은 비중으로 평균낸다. 그래서
CN이 더 많은 데이터에서 Accuracy만 높고 MCI+DEM을 거의 못 찾는 문제를
드러내는 데 유용하다. 정의는
[scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)에서도
확인할 수 있다.

이번 데이터에서 OOF Balanced Accuracy 43.0%는 다음 계산이다.

```text
(CN을 맞힌 비율 78.8% + MCI+DEM을 맞힌 비율 7.1%) / 2
= 약 43.0%
```

## 4. 실행이 full이었는지 확인

| 확인 항목 | 결과 |
| --- | --- |
| 실행 모드 | `full` |
| smoke 여부 | `false` |
| 전체 시간 | 5,794초, 약 1시간 36분 34초 |
| GPU | NVIDIA A100-SXM4-80GB |
| 외부 교차검증 | 5 folds × 2 repeats = 10개 outer fold |
| 내부 교차검증 | 각 outer fold 안에서 3 folds |
| 튜닝 | outer fold마다 TabNet 4회 + YDF 4회 |
| TabNet 반복 학습 | 최종 fold 모델마다 seed 3개 |

따라서 학습이 예상보다 빨리 끝난 이유를 smoke 설정에서 찾을 수는 없다.
실행 상태는 [Launcher 상태](outputs/LAUNCHER_STATUS.json), 설정은
[실행 설정](outputs/training/run_config.json), 장비 정보는
[환경 정보](outputs/training/environment.json)에서 확인할 수 있다.

## 5. 데이터와 EDA 결과

학습 데이터는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| 전체 사람 수 | 141명 |
| CN | 85명 |
| MCI+DEM | 56명 |
| Activity와 Sleep이 정렬된 일별 행 | 9,673일 |
| 한 사람의 관측 길이 | 최소 35일, 중앙값 66일, 최대 120일 |
| 일별 웨어러블 특징 | 119개 |
| 최종 후보 특징 | 1,077개 |
| 모델에 사용한 관측 구간 | 각 사람의 마지막 28개 정렬 관측 |

데이터 품질 자체에는 큰 이상이 보이지 않았다.

- 119개 일별 특징 중 값이 항상 같은 특징은 0개였다.
- 결측이 20% 이상인 일별 특징은 0개였다.
- 최종 후보 특징의 전체 결측 비율은 약 0.007%로 매우 낮았다.
- ID, 진단명, 절대 날짜, 수집 일수, coverage, non-wear 정보는 모델
  특징에서 제외되었다.
- Validation 데이터와 정답은 학습 EDA와 특징 선택에 사용되지 않았다.

EDA에서 단일 특징 하나만 본 최대 direction-free AUC는 약 65.7%였다.
상위권에는 깊은 수면 점수의 변동, 수면 단계 entropy, 깨어 있는 시간,
취침 시각과 같은 수면 특징이 많았다. 다만 이 값은 학습 데이터 안에서
한 특징씩 둘러본 탐색 결과일 뿐, 새로운 사람에 대한 최종 성능이 아니다.

자세한 내용은 [학습 EDA 보고서](outputs/eda/EDA_REPORT_KO.md),
[EDA 요약 JSON](outputs/eda/eda_summary.json),
[일별 데이터 EDA](outputs/eda/daily_sequence/EDA_REPORT_KO.md)에서 확인할
수 있다.

## 6. 어떤 모델을 사용했는가

두 Google 계열 모델을 결합했다.

1. **TabNet**
   - 표 형태 데이터에서 중요한 특징을 단계적으로 선택하도록 설계된
     신경망이다.
   - 각 fold에서 seed 3개를 학습해 평균했다.
   - 출처: [Google Research의 TabNet 소개](https://research.google/pubs/tabnet-attentive-interpretable-tabular-learning/)

2. **Yggdrasil Decision Forests(YDF)**
   - Google의 결정 트리 계열 라이브러리다.
   - 이번 실험에서는 Gradient Boosted Trees를 사용했다.
   - 출처: [YDF 공식 GBT 문서](https://ydf.readthedocs.io/en/latest/py_api/GradientBoostedTreesLearner/)

모델 선택, 특징 선택, 보정, 결합 비율은 각 outer fold의 학습 부분
안에서만 결정했다. 한 사람을 평가할 때 그 사람의 정답이 모델 선택에
들어가지 않도록 구성한 점은 적절하다.

## 7. 핵심 성능

### 7.1 가장 중요한 Nested OOF 결과

OOF는 각 사람을 **그 사람을 학습에 사용하지 않은 모델**로 예측한
결과다. 현재처럼 표본이 작을 때는 한 번의 Validation 점수보다 이
결과를 우선해서 보는 것이 안전하다.

| 평가 | Accuracy | Balanced Accuracy | MCI+DEM Recall | CN Specificity | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| OOF, 기준값 0.50 | **50.4%** | **43.0%** | **7.1%** | 78.8% | **44.6%** |
| OOF, 선택 기준값 0.39 | 50.4% | 50.6% | 51.8% | 49.4% | 44.6% |

기준값을 0.50에서 0.39로 낮추면 MCI+DEM을 더 많이 찾지만, 그만큼 CN을
MCI+DEM으로 잘못 판단한다. Balanced Accuracy가 50.6%에 그쳐 실질적인
분리 능력이 좋아졌다고 보기는 어렵다. ROC-AUC는 순위 성능이므로
기준값을 바꿔도 그대로다.

0.50 기준 bootstrap 95% 구간도 넓고 낮았다.

- Accuracy: 44.7% ~ 56.0%
- Balanced Accuracy: 37.4% ~ 48.6%
- ROC-AUC: 34.7% ~ 53.9%

현재 표본에서는 우연한 변동을 넘어서는 안정적인 성능 근거가 부족하다.

### 7.2 Historical Validation 결과

| 평가 | Accuracy | Balanced Accuracy | MCI+DEM Recall | CN Specificity | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cross-fold bagging, 기준값 0.50 | **75.8%** | 58.5% | 28.6% | 88.5% | 56.6% |
| Cross-fold bagging, 기준값 0.39 | 39.4% | 51.1% | 71.4% | 30.8% | 56.6% |
| 전체 학습 후 full refit, 기준값 0.50 | 72.7% | 67.0% | 57.1% | 76.9% | 58.8% |
| 모든 사람을 CN으로 예측 | **78.8%** | 50.0% | 0.0% | 100.0% | 해당 없음 |

Cross-fold bagging이 현재 코드의 주 평가 모델이다. Full refit은 141명
전체로 다시 학습한 배포용 보조 모델이므로 독립 OOF 성능이 없고, 더
좋은 평가 근거로 취급하면 안 된다.

Validation 상세 내용은
[Validation 보고서](outputs/training/validation_report.json)와
[최종 보고서](outputs/training/FINAL_REPORT.json)에 있다.

## 8. 혼동행렬을 사람 수로 읽기

### 8.1 Nested OOF, 기준값 0.50

```text
실제 CN 85명
  ├─ CN으로 맞힘: 67명
  └─ MCI+DEM으로 잘못 판단: 18명

실제 MCI+DEM 56명
  ├─ MCI+DEM으로 맞힘: 4명
  └─ CN으로 놓침: 52명
```

전체 141명 중 71명을 맞혔지만, 임상적으로 더 관심 있는 MCI+DEM을
56명 중 4명밖에 찾지 못했다.

### 8.2 Nested OOF, 선택 기준값 0.39

```text
실제 CN 85명
  ├─ CN으로 맞힘: 42명
  └─ MCI+DEM으로 잘못 판단: 43명

실제 MCI+DEM 56명
  ├─ MCI+DEM으로 맞힘: 29명
  └─ CN으로 놓침: 27명
```

MCI+DEM 탐지는 늘었지만 CN 오경보도 43명으로 크게 늘었다. 단순히
기준값만 낮추는 것으로는 두 클래스를 함께 잘 맞힐 수 없었다.

### 8.3 Historical Validation, 기준값 0.50

```text
실제 CN 26명
  ├─ CN으로 맞힘: 23명
  └─ MCI+DEM으로 잘못 판단: 3명

실제 MCI+DEM 7명
  ├─ MCI+DEM으로 맞힘: 2명
  └─ CN으로 놓침: 5명
```

25명을 맞혀 Accuracy는 75.8%지만, 실제 MCI+DEM은 7명 중 2명만
찾았다.

## 9. Validation Accuracy 75.8%가 왜 오해를 부르는가

Validation은 33명 중 CN이 26명이고 MCI+DEM이 7명이다. CN 비율이
매우 높다.

```text
아무 특징도 보지 않고 33명 전원을 CN이라고 예측
→ CN 26명을 맞힘
→ Accuracy = 26 / 33 = 78.8%
```

현재 모델의 75.8%는 이 단순 기준보다 낮다. 따라서 “75.8%니까 거의
목표에 도달했다”가 아니라, “CN이 많은 데이터라 Accuracy가 높아 보인
것”에 가깝다.

또한 MCI+DEM이 7명뿐이어서 단 한 사람의 결과가 Recall을 약 14.3%p씩
바꾼다. 이 Validation은 이전 실험에서도 여러 번 확인한 historical
benchmark이므로, 완전히 새로운 독립 test로 볼 수도 없다.

이 때문에 다음 실험에서도 다음 순서로 판단해야 한다.

1. subject-level Nested OOF Balanced Accuracy와 ROC-AUC
2. MCI+DEM Recall과 CN Specificity가 동시에 충분한지
3. 95% 신뢰구간과 fold 간 변동
4. Historical Validation은 마지막 참고값

## 10. 성능이 낮은 주요 원인

### 10.1 사람 수에 비해 후보 특징이 너무 많다

전체 사람은 141명인데 후보 특징은 1,077개다. 한 outer fold에서 실제
학습에 쓰는 사람은 112명 또는 113명뿐이다. 특징 선택을 하더라도 작은
표본의 우연한 차이를 중요한 신호로 고를 가능성이 크다.

### 10.2 fold마다 선택 특징이 크게 달랐다

특징 목록의 fold 간 평균 Jaccard 유사도는 TabNet 약 0.220, YDF 약
0.213이었다. 쉽게 말하면 서로 다른 fold가 고른 특징의 겹침이 낮았다.
10개 fold 모두에서 선택된 특징도 TabNet은 2개, YDF는 1개뿐이었다.

근거는
[특징 선택 안정성](outputs/training/feature_selection_stability.json)에
있다.

### 10.3 내부 튜닝 결과가 바깥 평가로 이어지지 않았다

- TabNet 평균 ROC-AUC: 내부 튜닝 약 55.2% → outer 평가 약 43.1%
- YDF 평균 ROC-AUC: 내부 튜닝 약 49.1% → outer 평가 약 46.5%

특히 TabNet은 내부 데이터에서는 더 좋아 보였지만, 보지 않은 outer
사람에게서는 크게 떨어졌다. 이는 작은 데이터에서 튜닝이 내부 fold의
우연한 패턴에 맞춰졌다는 신호다.

### 10.4 fold별 결과가 불안정했다

10개 outer fold 중 9개에서 ROC-AUC가 50%보다 낮았고, 5개 fold에서는
MCI+DEM을 한 명도 찾지 못했다. 한 fold만 상대적으로 좋았기 때문에
특정 split에 따라 결과가 크게 달라졌다.

### 10.5 예측 위험도의 방향도 안정적이지 않았다

OOF에서 모델이 출력한 MCI+DEM 확률의 평균은 다음과 같았다.

- 실제 CN: 약 0.418
- 실제 MCI+DEM: 약 0.382

오히려 실제 CN의 평균 위험도가 더 높았다. ROC-AUC 44.6%와 같은
방향의 문제다. “확률을 반대로 뒤집으면 해결된다”는 뜻은 아니다.
작은 표본과 fold 변동 때문에 예측 순서 자체가 안정적이지 않다는
뜻으로 해석해야 한다.

### 10.6 확률 보정도 자주 실패했다

TabNet과 YDF를 합쳐 20개의 fold별 calibrator 중 13개가 음의 기울기를
보여 identity fallback을 사용했다. 내부 OOF에서도 모델 확률과 정답의
관계가 안정적으로 같은 방향을 유지하지 못했다는 뜻이다.

### 10.7 주원인은 단순 결측 문제로 보이지 않는다

결측률은 매우 낮고 값이 항상 같은 일별 특징도 없었다. 따라서 이번
실패를 “데이터 파일에 값이 많이 비어 있어서”라고 설명하기는 어렵다.
핵심 문제는 작은 사람 수, 지나치게 많은 후보 특징, 약한 일반화 신호,
그리고 split에 따른 불안정성에 더 가깝다.

## 11. MMSE-free 이전 실험에서 얻을 수 있는 제한적인 힌트

웨어러블 Activity/Sleep만 사용한 이전 SequenceFusion 실험에서는
개별 sequence Transformer가 OOF Balanced Accuracy 약 58.8%,
ROC-AUC 약 62.5%로 이번 TabNet 중심 결과보다 나았다. 다만 그 실험의
Historical Validation도 불안정했으므로, 이 수치를 그대로 재현 성능으로
간주할 수는 없다.

이 비교의 의미는 “Transformer가 정답”이라는 것이 아니라, 28일을
1,077개의 요약 특징으로 크게 펼치는 방식보다 **날짜 순서가 있는
시계열 표현을 다시 시험할 가치가 있다**는 정도다.

근거:
[MMSE-free SequenceFusion 최종 보고서](../Binary_Wearable_SequenceFusion_Google/outputs/20260722_070723_utc/training/FINAL_REPORT.json)

MMSE가 포함되었거나 포함 여부가 명확하지 않은 과거 성능은 이 비교에서
제외했다.

## 12. 다음 실험의 개선 방향

### 12.1 목표 지표를 Balanced Accuracy 우선으로 바꾼다

Accuracy 90%는 계속 목표로 두되, 모델과 임계값을 고를 때는 Balanced
Accuracy를 먼저 최적화해야 한다. MCI+DEM Recall 또는 CN Specificity
중 하나만 지나치게 낮은 후보는 높은 Accuracy가 나와도 탈락시켜야 한다.

권장 목표 확인 순서는 다음과 같다.

1. Nested OOF Balanced Accuracy 75% 이상
2. MCI+DEM Recall과 CN Specificity가 모두 충분한지
3. Nested OOF Accuracy 90% 이상
4. ROC-AUC와 95% 신뢰구간

### 12.2 특징 공간을 작고 안정적으로 만든다

- 모든 1,077개 후보를 넓게 탐색하기보다 수면 단계, 수면 시각,
  심박, 활동량, 비활동, 일주기처럼 해석 가능한 웨어러블 채널을 먼저
  제한한다.
- 최솟값이 35일이므로 모든 사람에게 동일한 마지막 35개 정렬 관측을
  사용하는 안을 비교한다.
- 중앙값, IQR, 추세, 최근 변화처럼 소수의 robust summary만 사용한다.
- 특징 선택은 반드시 각 outer-training fold 안에서만 수행한다.
- fold마다 반복해서 선택되는 특징을 기록해 안정성을 확인한다.

### 12.3 표 요약 모델과 시계열 모델을 함께 비교한다

권장 후보는 다음과 같다.

- class weight를 적용한 정규화 Logistic Regression
- Google YDF의 보수적인 GBT
- 작은 구조의 Google Research TabNet
- 같은 길이의 일별 행을 사람 단위로 묶어 평가하는 Google YDF
  multi-instance 모델
- 이전 웨어러블-only 실험에서 상대적으로 나았던 작은 Transformer

여러 모델을 무조건 평균하면 약한 모델이 좋은 모델을 끌어내릴 수 있다.
따라서 각 outer-training의 inner OOF에서 최소 품질을 통과한 모델만
결합하고, 아무 모델도 통과하지 못하면 가장 단순한 모델로 fallback하는
방식이 안전하다.

### 12.4 현재 확률 보정 문제를 직접 막는다

- class 1이 항상 `MCI+DEM`인지 자동 검증한다.
- inner OOF에서 음의 calibration slope가 나오면 identity로 억지 사용하지
  않고 해당 모델을 결합 후보에서 제외한다.
- 모델 선택, 보정, 결합 비율, 임계값에 **같은 inner OOF 예측**을 사용해
  seed와 예측 출처가 어긋나지 않게 한다.
- fold별 선택 임계값을 다른 refit 모델에 그대로 복사하기보다, 임계값을
  뺀 margin을 결합해 fold 간 확률 크기 차이를 줄인다.

### 12.5 검증을 더 반복하되 정보 누출은 막는다

- subject 단위 outer 5-fold를 여러 번 반복한다.
- 모든 전처리, 특징 선택, 스케일링, 모델 선택, 보정은 outer-training
  안에서만 학습한다.
- Historical Validation은 모델 선택에 사용하지 않고 마지막에 한 번
  참고한다.
- Accuracy, Balanced Accuracy, 두 클래스 Recall, ROC-AUC, 혼동행렬,
  bootstrap 95% 구간을 함께 저장한다.

### 12.6 장비 사용 시 참고

A100은 TabNet과 신경망 학습에 적합하다. 다만 YDF, Logistic Regression,
EDA는 CPU 중심이므로 A100 사용 중에도 CPU 코어와 RAM이 중요하다. 전체
실행 중 GPU 사용률이 계속 높지 않아도 반드시 오류는 아니다.

### 12.7 이 분석을 반영한 후속 코드

후속 코드는
[Binary_Wearable_BalancedFusion_Google](../Binary_Wearable_BalancedFusion_Google/)
폴더에 만들었다. `base.ipynb`에서는 다음 실행 파일만 지정하면 된다.

```python
RUN_FILE = "Binary_Wearable_BalancedFusion_Google/run.py"
```

이 코드는 아직 학습하지 않았으므로 개선 성능을 미리 주장할 수는 없다.
실행 후에는 먼저 반복 Nested OOF의 Balanced Accuracy와 신뢰구간을 현재
결과와 비교해야 한다. 또한 같은 141명의 이전 분석을 참고해 후속 구조를
정했으므로, 코드 내부 누수와 별개로 반복 실험에 따른 선택 편향은 남는다.
최종 성능 주장은 새로운 외부 대상자 데이터로 다시 확인해야 한다.

## 13. 체크포인트 상태

체크포인트는 정상적으로 저장되었다.

| 항목 | 결과 |
| --- | ---: |
| outer fold bundle | 10개 |
| 각 bundle의 TabNet 모델 | seed별 3개 |
| 각 bundle의 YDF 모델 | 1개 |
| full refit bundle | 1개 |
| 전체 TabNet zip | 33개 |
| 전체 YDF 모델 | 11개 |
| 저장 후 다시 불러오기 확인 | 통과 |
| checkpoint tree 검증 | 통과 |

각 bundle에는 모델 파일뿐 아니라 전처리 상태, 선택 특징, 보정기,
결합 설정, manifest와 hash가 함께 저장된다. 저장된 모델을 새 CPU
프로세스에서 다시 불러와 같은 예측을 내는 round-trip 확인도 통과했다.

자세한 목록은
[체크포인트 인덱스](outputs/training/checkpoint_index.json)에 있다.

## 14. 한계

- 학습 대상이 141명으로, 복잡한 딥러닝과 큰 특징 공간을 안정적으로
  평가하기에는 작다.
- Historical Validation은 33명뿐이고 MCI+DEM은 7명뿐이다.
- Historical Validation은 이전 실험에서도 확인되었으므로 완전히 새로운
  독립 test가 아니다.
- 웨어러블 신호만으로 얻은 분류 결과이며, 의료 진단을 대신할 수 없다.
- Accuracy 90%와 Balanced Accuracy 75%는 다음 실험의 목표이지, 코드
  변경만으로 보장되는 값이 아니다.
- 엄격한 OOF에서도 성능이 계속 낮다면 모델 복잡도를 더 늘리기보다
  대상자 수 확대, 라벨 품질 점검, 독립 외부 데이터 확보가 우선이다.

웨어러블 기반 인지저하 연구 전반에서도 작은 표본과 외부 검증 부족은
반복해서 지적되는 한계다. 관련 검토는
[wearable 기반 인지저하 systematic review](https://pubmed.ncbi.nlm.nih.gov/41730193/)를
참고할 수 있다.

## 15. 최종 판단

이번 실험에서 확인된 사실은 명확하다.

1. full 학습은 정상 완료되었고 smoke가 아니다.
2. MMSE 없이 Activity와 Sleep만 사용했다.
3. 체크포인트는 빠짐없이 저장되고 재로딩 검증까지 통과했다.
4. 그러나 가장 신뢰해야 할 Nested OOF 성능은 Accuracy 50.4%,
   Balanced Accuracy 43.0%, MCI+DEM Recall 7.1%로 낮았다.
5. Validation Accuracy 75.8%는 all-CN 기준 78.8%보다 낮아 좋은
   분류력의 근거가 아니다.
6. 다음 실험은 더 큰 모델보다 작은 특징 공간, 시계열 표현, balanced
   objective, 품질 기반 모델 결합, 반복 subject-level Nested CV에
   초점을 맞추는 것이 합리적이다.

현재 결과는 실패 원인을 확인하는 데는 유용하지만, 목표 성능을 달성한
최종 모델로 사용해서는 안 된다.
