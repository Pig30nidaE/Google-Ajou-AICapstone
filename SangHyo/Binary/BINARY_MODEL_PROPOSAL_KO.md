# CN vs MCI+Dem 이진분류 최종 모델 제안

최종 갱신: **2026-07-30**

## 1. 선정 기준과 요약

이 문서는 `SangHyo/`에 저장된 실험 결과 중 **nested cross-validation으로
평가를 완료한 모델만** 비교해, MMSE를 사용하는 경우와 사용하지 않는 경우에
가장 높은 ROC-AUC를 기록한 모델을 하나씩 제안한다.

여기서 nested 평가는 사람을 다음과 같이 두 단계로 나누는 방법을 뜻한다.

1. 바깥쪽 분할의 평가 대상자는 최종 성능을 측정할 때만 사용한다.
2. 모델 선택, 가중치 결정, 변수 선택과 학습 설정 결정은 바깥쪽 학습 데이터
   안에서 다시 나눈 안쪽 분할만 사용한다.

따라서 평가 대상자의 정답을 보고 모델이나 변수를 고르는 직접적인 데이터 누수를
막을 수 있다. 모든 분할은 사람 단위로 수행해, 같은 사람의 기록이 학습용과
평가용에 동시에 들어가지 않게 했다.

학습 데이터는 총 141명으로, CN 85명과 MCI·Dem 56명(MCI 47명, Dem 9명)이다.
평가 지표는 `CN=0`, `MCI 또는 Dem=1`에 대한 ROC-AUC 하나를 우선한다.

| 구분 | 제안 모델 | Nested OOF ROC-AUC | 95% 신뢰구간 |
| --- | --- | ---: | --- |
| **MMSE 포함** | MMSE 39개 변수 기반 규제 로지스틱 회귀 + RBF SVM | **0.765756** | [0.684001, 0.845656] |
| **MMSE 미포함** | 28일·8개 입력 Sequence Transformer | **0.629307** | [0.534346, 0.724477] |

이전 보고서에서 제안했던 Google YDF 모델의 0.7834와 Gemma 인지 특징 모델의
0.7817은 같은 OOF 결과를 보며 후보와 결합 비율을 고른 non-nested 개발
점수다. 이번 문서에서는 성능이 조금 낮더라도 선택 과정과 바깥 평가가 분리된
nested 결과를 우선하므로 두 모델을 최종 후보에서 제외했다.

---

## 2. MMSE 포함 모델

### 제안 모델

**`Binary_MMSE_MaxAUC`의 MMSE 전용 품질 게이트 앙상블**

다음 두 모델의 예측을 결합한다.

1. 규제가 적용된 Logistic Regression
2. 비선형 관계를 보완하는 RBF SVM

각 바깥쪽 학습 데이터 안에서 두 모델을 다시 평가하고, 일정 수준 이상의 성능을
낸 모델만 결합한다. 전체 학습 데이터로 만든 최종 배포 모델의 비중은 Logistic
Regression **0.47737**, RBF SVM **0.52263**이다.

### 전처리 방식

- 한 사람의 MMSE 기록을 한 행으로 정리한다.
- 다음과 같이 총 39개 변수를 사용한다.
  - MMSE 총점 1개
  - 시간 지남력, 장소 지남력, 기억등록, 주의집중, 지연회상, 언어 점수 6개
  - MMSE 세부 문항 30개
  - 실패 문항 수와 지연회상 저하 정도 2개
- 진단명, 진단 코드, 의사명, 검사 번호와 같은 진단·행정·식별 정보는 입력에서
  제외한다.
- 비어 있는 값은 현재 학습 묶음의 중앙값으로 채운다.
- 각 변수는 현재 학습 묶음의 평균과 표준편차를 사용해 단위를 맞춘다.
- 중앙값과 표준화 기준은 매번 학습용 사람에게서만 계산하고 평가 대상자에게
  그대로 적용한다.
- 웨어러블 활동·수면 정보는 사용하지 않는다. 기존 실험에서 이를 추가했을 때
  MMSE 신호가 안정적으로 개선되지 않았기 때문이다.

### 데이터 분할 방식

- 사람 단위 `StratifiedKFold`를 사용해 **바깥쪽 5분할을 5번 반복**한다.
- 각 반복에서 모든 사람은 한 번씩 바깥쪽 평가 대상이 된다.
- 각 바깥쪽 학습 데이터는 다시 **안쪽 3분할**로 나눈다.
- 안쪽 분할은 Logistic Regression과 RBF SVM의 품질을 평가하고 결합 비율을
  정하는 데만 사용한다.
- 바깥쪽 평가 대상자의 정답은 전처리, 모델 선택, 가중치 결정에 사용하지 않는다.
- 각 사람에게서 얻은 5개의 바깥쪽 평가 예측을 평균한 뒤 전체 ROC-AUC를
  계산한다.
- 95% 신뢰구간은 사람 단위 bootstrap을 1,000번 수행해 계산했다.

Historical Validation 33명은 모델 선택에 사용하지 않았다. 이 데이터는 과거
실험에서 여러 번 확인한 자료이므로 새로운 독립 테스트셋이 아니라 참고용
벤치마크로만 취급한다.

### 모델 선택 / 하이퍼파라미터 튜닝

큰 탐색을 수행하기보다, 작은 표본에서 비교적 안정적이었던 두 모델과 설정을
고정해 사용했다.

| 구성 모델 | 설정 |
| --- | --- |
| Logistic Regression | `C=0.1`, `class_weight="balanced"`, `max_iter=5000` |
| RBF SVM | `C=1.0`, `gamma="scale"`, `class_weight="balanced"`, 확률 출력 사용 |

각 바깥쪽 학습 데이터의 안쪽 3분할에서 Balanced Accuracy를 계산한다.
0.55 이상인 모델만 결합 대상으로 삼고, 각 모델의 점수가 0.5보다 얼마나
높은지에 비례해 가중치를 준다. 둘 다 기준을 통과하지 못하면 안쪽 평가가 더
좋은 모델 하나만 사용한다.

이 선택 규칙은 ROC-AUC 자체를 대규모로 반복 최적화하지 않고, 성능이 낮은
모델이 결합 결과를 떨어뜨리는 것을 막기 위한 안전장치다. 최종 배포 모델의
안쪽 평균 Balanced Accuracy는 Logistic Regression **0.6572**, RBF SVM
**0.6709**였다.

### 최종 결론 및 코멘트

- 사람별 반복 예측을 평균한 nested OOF ROC-AUC: **0.765756**
- 사람 단위 bootstrap 95% 신뢰구간: **[0.684001, 0.845656]**
- 직접적인 데이터 누수 점검: 통과
- Historical Validation ROC-AUC: **0.634615** (여러 번 사용된 33명의
  참고값이므로 최종 성능으로 해석하지 않음)

현재 저장된 완료 실험 중 nested 평가를 적용한 MMSE 포함 모델로는 가장 높은
ROC-AUC다. 같은 기준의 다른 주요 결과인
`Binary_Google_ROCAUC_Champion` MMSE 앵커 0.762290과
`Binary_Google_OrdinalStable` 0.756933보다 근소하게 높다.

복잡한 웨어러블 결합이나 대규모 모델 탐색보다 MMSE 39개와 두 개의 규제 모델을
사용한 간단한 구성이 더 안정적이었다. 다만 신뢰구간이 넓고, MMSE가 임상 진단
과정에도 사용됐을 수 있으므로 이 점수만으로 실제 임상 성능을 확정할 수는 없다.

근거 파일:

- [`Binary_MMSE_MaxAUC/FINAL_REPORT.json`](Binary_MMSE_MaxAUC/Binary_MMSE_MaxAUC_result/20260727_042357_utc/training/FINAL_REPORT.json)
- [`Binary_MMSE_MaxAUC/deployment.json`](Binary_MMSE_MaxAUC/Binary_MMSE_MaxAUC_result/20260727_042357_utc/deployment/deployment.json)
- [`Binary_MMSE_MaxAUC/README_KO.md`](Binary_MMSE_MaxAUC/README_KO.md)

---

## 3. MMSE 미포함 모델

### 제안 모델

**`Binary_Google_ROCAUC_Champion`의 28일·8개 입력 Sequence Transformer**

최근 28일의 활동·수면 기록을 여덟 가지 고정된 형태로 구성해 각각 예측한 뒤,
결과를 평균한다. MMSE와 CognitiveFunction 파일은 전혀 사용하지 않는다.

### 전처리 방식

- Activity와 Sleep에서 얻은 생리 신호 113개만 사용한다.
- 모든 사람에게 같은 기준을 적용해 최근 28개 관측으로 여덟 개의 입력 묶음을
  만든다.
- 날짜, 원본 행 번호, 관측 일수, 결측 개수, 데이터 길이, 기기 미착용 정보처럼
  실제 건강 상태보다 수집 환경을 나타낼 수 있는 값은 입력에서 제외한다.
- 값이 한쪽으로 크게 치우치는 일부 변수에는 미리 정한 `signed log1p` 변환을
  적용한다.
- 비어 있는 값은 중앙값으로 채우고, 지나치게 큰 값의 영향을 줄인 뒤 변수의
  단위를 맞춘다.
- 중앙값, 극단값 처리 기준과 단위 조정 기준은 현재 학습용 사람에게서만 구한다.
- 한 사람에게서 만든 여덟 개 입력은 항상 모두 학습용이거나 모두 평가용이다.
- 여덟 번의 예측 확률을 평균해 그 사람의 최종 점수를 만든다.

### 데이터 분할 방식

- 바깥쪽 분할은 사람 단위 **5분할 × 5회 반복**이다.
- 각 바깥쪽 학습 데이터 안에서 다시 **4분할 × 2회 반복** 평가를 수행한다.
- 모델을 몇 epoch 학습할지는 현재 학습용 사람 중 20%를 임시 확인용으로 떼어
  정한다.
- 학습 횟수를 결정한 뒤 현재 학습용 전체를 그 횟수만큼 다시 학습한다.
- 보조 모델을 추가할지는 안쪽 평가 결과만으로 결정하며, 바깥쪽 평가 대상자의
  정답은 사용하지 않는다.
- 각 사람에게서 얻은 5개의 바깥쪽 평가 점수를 평균해 최종 ROC-AUC를 계산한다.
- 95% 신뢰구간은 사람 단위로 5,000번 다시 추출해 계산했다.

### 모델 선택 / 하이퍼파라미터 튜닝

Sequence Transformer의 기본 구조는 바깥 평가 전에 고정했다. 안쪽 평가에서는
Ridge, Elastic-Net, RBF SVM, CatBoost를 보조 모델로 더할 수 있는지 확인했다.
보조 모델은 다음 두 조건을 모두 만족할 때만 추가했다.

1. 안쪽 ROC-AUC가 기존 구성보다 최소 0.005 높을 것
2. 여덟 개 안쪽 분할 중 절반을 초과해 기존 구성보다 좋은 결과를 낼 것

보조 모델을 선택하는 전체 nested 정책의 ROC-AUC는 0.617227이었다. 반면 평가
전에 구조를 고정한 Sequence Transformer 단독은 0.629307이었다. 보조 모델
추가가 안정적인 개선으로 이어지지 않았으므로, 최종 제안은 Transformer
단독이다.

| 항목 | 설정 |
| --- | --- |
| 사용 기간 / 입력 묶음 | 최근 28개 관측 / 8개 |
| 입력 변수 | 113개 |
| Transformer 폭 | 64 |
| Encoder | 2개 층, 4개 attention head |
| Feed-forward 폭 | 128 |
| Dropout | 본체 0.25, 예측부 0.30 |
| Optimizer | AdamW |
| Learning rate / weight decay | `3e-4` / `2e-4` |
| Batch size | 32 |
| 최대 epoch / patience | 120 / 16 |
| 전체 재학습에서 선택된 epoch | 13 |

두 집단의 인원 차이를 완화하기 위해 집단별 가중치를 사용하고, 모델이 지나치게
확신하지 않도록 약한 label smoothing을 적용했다.

### 최종 결론 및 코멘트

- 사람별 반복 예측을 평균한 nested OOF ROC-AUC: **0.629307**
- 분할 반복별 ROC-AUC 평균: **0.595525**
- 분할 반복별 표준편차: **0.020109**
- 사람 단위 bootstrap 95% 신뢰구간: **[0.534346, 0.724477]**
- 직접적인 데이터 누수 점검: 통과
- MMSE 데이터 사용: 없음
- Historical Validation ROC-AUC: **0.497253** (여러 번 사용된 33명의
  참고값이므로 최종 성능으로 해석하지 않음)

현재 저장된 nested 결과 중 MMSE를 사용하지 않은 모델로는 가장 높은
ROC-AUC다. 이전 Sequence Transformer 결과인 0.6254와 비슷한 값이 새로운
분할에서도 나왔다는 점은 긍정적이다.

그러나 신뢰구간에 0.5가 포함되고 Historical Validation에서는 무작위 수준의
결과가 나왔다. 따라서 현재 단계에서는 임상 선별에 바로 사용할 모델이라기보다,
웨어러블 정보에 제한적인 구분 신호가 있을 가능성을 보여주는 연구용 모델로
보는 것이 안전하다.

근거 파일:

- [`Binary_Google_ROCAUC_Champion/FINAL_REPORT.json`](Binary_Google_ROCAUC_Champion/Binary_Google_ROCAUC_Champion_result/20260728_101249_utc/training/FINAL_REPORT.json)
- [`Binary_Google_ROCAUC_Champion/wearable nested OOF`](Binary_Google_ROCAUC_Champion/Binary_Google_ROCAUC_Champion_result/20260728_101249_utc/training/wearable/nested_oof_report.json)
- [`Binary_Google_ROCAUC_Champion/wearable deployment`](Binary_Google_ROCAUC_Champion/Binary_Google_ROCAUC_Champion_result/20260728_101249_utc/training/wearable/deployment/deployment.json)

---

## 4. 최종 제안

1. **MMSE를 사용할 수 있다면** `Binary_MMSE_MaxAUC`의 Logistic
   Regression+RBF SVM 결합 모델을 권장한다. Nested OOF ROC-AUC는
   **0.765756**이다.
2. **MMSE를 사용할 수 없다면** 최근 28일 활동·수면 기록을 사용하는 Sequence
   Transformer 단독 모델을 권장한다. Nested OOF ROC-AUC는 **0.629307**이다.
3. 이전 보고서의 YDF 0.7834와 Gemma 0.7817은 non-nested 개발 점수이므로,
   이번 nested 기준의 최종 모델과 같은 수준의 평가값으로 비교하지 않는다.
4. 두 결과 모두 141명 안에서 여러 번 나눈 평가다. 실제 적용 전에는 모델과
   설정을 그대로 고정하고, 지금까지 사용하지 않은 새로운 참여자 데이터에서
   다시 확인해야 한다.
