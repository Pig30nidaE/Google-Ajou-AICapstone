# Validation accuracy 0.84848 앙상블 기술 설명서

작성일: 2026-07-21

실험 ID: **20260720_005257_full_clinical_plus_lifelog**

분류 대상: **CN / MCI / DEM**

## 1. 이 문서의 목적

이 문서는 Validation accuracy **0.84848**을 기록한 실험이 무엇을 입력으로
사용했고, 어떻게 학습했으며, 어떤 경우에 잘하거나 못했는지를 쉬운 말로 설명합니다.

먼저 가장 중요한 사실은 다음과 같습니다.

> 이 결과는 하나의 모델이 만든 점수가 아닙니다.
>
> Transformer, TabNet, Google YDF를 6개 데이터 분할에서 학습한
> **총 18개 모델의 앙상블 결과**입니다.

또한 이 모델은 활동·수면 기록만 사용하는 모델이 아닙니다. **MMSE 문항별 점수와
TOTAL을 함께 사용한 임상 정보 보조 모델**입니다.

## 2. 한눈에 보는 결론

| 항목 | 내용 |
| --- | --- |
| Training | 141명: CN 85 / MCI 47 / DEM 9 |
| Validation | 33명: CN 26 / MCI 4 / DEM 3 |
| 입력 | Activity 312개 + Sleep 408개 + MMSE 34개 = 754개 후보 |
| 기본 모델 | 자체 구현 Transformer + Google TabNet + Google YDF |
| 최종 시스템 | 2회 반복 × 3-fold × 모델 3개 = 18개 체크포인트 |
| Validation accuracy | 28/33 = **0.84848** |
| Validation macro F1 | **0.63636** |
| Validation macro ROC-AUC | **0.72790** |
| Nested OOF accuracy | **0.60993** |
| 가장 큰 약점 | Validation MCI 4명을 모두 CN으로 오분류 |

정확한 해석은 다음과 같습니다.

> 이 작은 Validation에서는 CN과 매우 뚜렷한 DEM은 잘 구분했지만,
> MCI와 CN은 구분하지 못했습니다. CN이 전체의 78.8%이기 때문에
> 전체 accuracy가 높게 보이는 효과도 큽니다.

## 3. CN, MCI, DEM은 무엇인가

- **CN**: 인지 기능이 정상 범위인 집단
- **MCI**: 경도인지장애 집단
- **DEM**: 치매 집단

모델은 한 사람마다 세 개의 확률을 출력합니다.

~~~text
예시
CN 확률  0.72
MCI 확률 0.24
DEM 확률 0.04

가장 큰 값이 CN이므로 최종 예측은 CN
~~~

내부 class 순서는 항상 **CN=0, MCI=1, DEM=2**입니다.

## 4. 전체 처리 흐름

~~~text
Activity 원본 ─┐
Sleep 원본    ├─→ 사람 한 명당 754개 입력 후보
MMSE 원본     ┘
                         ↓
             fold 전용 전처리와 특징 선택
                         ↓
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
   Transformer         TabNet         Google YDF
        └────────────────┼────────────────┘
                         ↓
             fold별 확률 가중 평균
                         ↓
             MCI·DEM 확률 배율 보정
                         ↓
              위 과정을 6개 fold에서 반복
                         ↓
               6개 fold 결과를 동일 평균
                         ↓
                  CN / MCI / DEM 예측
~~~

각 fold마다 데이터, 전처리기, 선택된 특징, 모델 설정과 혼합 비율이 다릅니다.
따라서 특정 Transformer 파일 하나만 가져오거나 가장 좋아 보이는 fold 하나만
가져오면 accuracy 0.84848 결과를 재현할 수 없습니다.

## 5. 모델에 들어간 정보

### 5.1 Activity

Training Activity는 141명의 9,705개 일별 기록입니다. 원본 숫자와 짧은 시간 간격의
활동 상태·MET 수열을 다음과 같이 요약했습니다.

- 활동 점수, 걸음 수, 칼로리, 활동 시간
- 낮음·중간·높음·휴식 상태
- 활동 상태가 얼마나 자주 바뀌는지
- 한 상태가 얼마나 오래 이어지는지
- 활동 패턴의 다양성
- MET의 평균, 변동, 낮은 구간과 높은 구간

하루 단위 값을 다시 사람 단위로 묶을 때는 다음 여덟 가지 요약값을 만들었습니다.

- 중앙값
- 평균
- 표준편차
- 하위 10% 값
- 상위 10% 값
- 중간 50% 범위
- 시간에 따른 완만한 증가·감소
- 최근 절반과 초기 절반의 차이

최종 Activity 후보는 **312개**입니다.

### 5.2 Sleep

Training Sleep도 141명의 9,705개 원본 기록입니다.

- 수면 시간과 수면 점수
- light, deep, REM, awake 관련 정보
- 수면 단계가 바뀌는 빈도와 다양성
- 심박과 RMSSD
- 취침·기상 시각
- 수면 중 불안정성과 최근 변화

하루에 여러 수면 기록이 있으면 가장 긴 수면을 중심으로 한 개만 선택했습니다.
마지막 Activity 시점보다 미래에 기록된 Sleep은 사용하지 않도록 시간 검사를
적용했습니다. 취침 시각은 23시와 0시가 멀리 떨어진 숫자로 보이지 않도록
sin/cos 형태의 원형 값으로 바꿨습니다.

최종 Sleep 후보는 **408개**입니다.

### 5.3 MMSE

이 실험은 MMSE를 포함합니다. 사용한 34개 후보는 다음과 같습니다.

- Q01부터 Q19까지의 세부 문항 점수
- 일부 문항의 하위 점수
- TOTAL
- 시간·장소 지남력 관련 합계
- 기억·언어 관련 합계

Training에서 MMSE TOTAL 중앙값은 다음과 같았습니다.

| class | MMSE TOTAL 중앙값 |
| --- | ---: |
| CN | 28 |
| MCI | 26 |
| DEM | 22 |

MMSE는 세 집단과 강하게 연결된 정보입니다. 특히 Validation의 DEM 3명은 MMSE가
매우 낮은 쪽에 모였고, MCI 4명은 CN과 크게 겹쳤습니다. 이 패턴은 모델이 DEM은
잘 맞히고 MCI는 전부 CN으로 판단한 결과와 일치합니다.

### 5.4 사용하지 않은 정보

다음 값은 성능을 부당하게 높이거나 사람을 식별할 수 있어 특징에서 제외했습니다.

- **DIAG_NM**: 진단 정답과 같은 값
- DIAG_SEQ, DOCTOR_NM, MMSE_NUM, MMSE_KIND
- 이메일과 subject ID
- 수면 기기 기록 ID
- 연·월·일 같은 절대 날짜
- 수집 일수, 빈 날짜 수, 수집 간격

특히 MMSE 원본의 DIAG_NM은 실제 정답과 100% 같습니다. 이 값을 모델에 넣으면
학습이 아니라 정답 복사가 되므로 절대로 사용하지 않았습니다.

## 6. 전처리는 어떻게 했는가

전처리는 전체 141명에서 한 번 계산하지 않았습니다. **각 학습 fold 안에서만**
다시 계산했습니다. 이렇게 해야 검증 대상자의 정보가 중앙값, 크기 조정 또는
특징 선택에 미리 섞이지 않습니다.

처리 순서는 다음과 같습니다.

1. 무한대를 결측값으로 바꿉니다.
2. 결측 비율이 40%를 넘는 특징을 제거합니다.
3. 값이 모두 같아 구분에 도움이 되지 않는 특징을 제거합니다.
4. 빈 값은 해당 학습 fold의 중앙값으로 채웁니다.
5. 너무 극단적인 값은 학습 fold의 1%~99% 범위 안으로 제한합니다.
6. 중앙값을 빼고 IQR로 나누어 서로 다른 숫자 크기를 맞춥니다.
7. 생활기록 특징은 학습 fold의 ANOVA 점수로 순위를 정합니다.
8. 상관계수가 0.985 이상인 거의 같은 특징은 하나만 남깁니다.

MMSE 특징은 임상 정보 모드의 핵심이므로, 해당 fold에서 실제로 값이 변하는 문항은
우선 보존했습니다. 생활기록을 합친 최종 특징 수는 모델과 fold에 따라
**64개에서 192개**입니다.

이 전처리 자체도 모델마다 다르므로 각 체크포인트의
**preprocessor.joblib** 파일이 반드시 필요합니다.

## 7. 세 가지 기본 모델

### 7.1 숫자형 Transformer

이 저장소의 Transformer는 PyTorch로 직접 만든 간소화된 FT-Transformer 계열입니다.
Google이 배포한 완성 모델을 가져온 것은 아닙니다.

작동 방식은 다음과 같습니다.

1. 숫자 특징 하나를 작은 벡터인 “토큰”으로 바꿉니다.
2. 전체 정보를 모으는 CLS 토큰을 앞에 붙입니다.
3. Attention이 MMSE·수면·활동 특징 사이의 관계를 찾습니다.
4. CLS 결과를 CN, MCI, DEM 세 점수로 바꿉니다.
5. softmax를 적용해 세 확률의 합이 1이 되게 합니다.

예를 들어 “MMSE가 낮고, 휴식 시간이 길며, 수면 단계 변화가 크다”와 같은 여러
조건의 조합을 함께 볼 수 있습니다.

학습 안정화를 위해 다음을 사용했습니다.

- 적은 class를 더 자주 보여주는 가중 표본 추출
- AdamW
- cosine 학습률 감소
- label smoothing
- 입력에 작은 잡음 추가
- gradient clipping
- 학습 후반부 모델 가중치 평균

실제로 선택된 Transformer는 fold에 따라 특징 64~96개, 토큰 크기 32~48,
attention head 4~8개, encoder layer 2~4개, 180~420 epoch를 사용했습니다.

표 데이터용 Transformer의 대표 참고 논문은
[Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)입니다.
현재 코드는 논문의 공식 모델을 그대로 복사한 것이 아니라 핵심 아이디어를 단순화한
자체 구현입니다.

### 7.2 TabNet

TabNet은 표 데이터를 위해 만든 신경망입니다. 모든 특징을 매번 똑같이 보지 않고,
여러 판단 단계를 거치면서 “이번 단계에서는 어떤 특징을 볼 것인가”를 선택합니다.

이 실험에서는 내부 표현 크기, 판단 단계 수, 특징 선택 방식, 학습률, 규제,
epoch 등을 Optuna로 찾았습니다. 실제 선택값은 fold마다 달랐으며 64~128개의
특징, 4~7개의 판단 단계, 200~500 epoch를 사용했습니다.

TabNet은 Google 연구진이 발표한 모델입니다.
[Google Research TabNet 설명](https://research.google/pubs/tabnet-attentive-interpretable-tabular-learning/)

### 7.3 Google YDF Gradient Boosted Trees

YDF는 여러 결정 나무를 사용하는 Google의 라이브러리입니다. 이 실험에서는
Gradient Boosted Trees를 사용했습니다.

쉽게 말하면 첫 번째 나무가 틀린 부분을 다음 나무가 보완하고, 그다음 나무가 남은
오류를 다시 보완합니다. 작은 표 데이터에서는 신경망과 다른 방식으로 판단하기
때문에 앙상블의 다양성을 높일 수 있습니다.

실제 선택된 YDF 모델은 특징 64~192개, 나무 300~1,000개, 깊이 2~6을
사용했습니다. [YDF 공식 문서](https://ydf.readthedocs.io/en/latest/)

## 8. class 불균형은 어떻게 다뤘는가

Training은 CN 85명, MCI 47명, DEM 9명으로 크게 불균형합니다. 아무 조치 없이
학습하면 모델이 CN만 자주 답하는 편이 유리해질 수 있습니다.

각 모델은 class 수의 반대 방향으로 가중치를 주되, 가중치 강도 자체도 Optuna가
찾도록 했습니다.

- Transformer: 적은 class의 사람을 더 자주 뽑음
- TabNet: 손실 계산에서 적은 class의 오류를 더 크게 반영
- YDF: class weight로 적은 class의 오류를 더 크게 반영

최종 혼합 단계에서는 MCI와 DEM 확률에 추가 배율을 적용했습니다. 다만 실제
Validation 결과를 보면 이 조치만으로 MCI 문제를 해결하지는 못했습니다.

## 9. 학습과 검증 구조

### 9.1 왜 nested cross-validation을 썼는가

데이터가 141명으로 작기 때문에 모델 설정을 고른 사람과 성능을 확인할 사람을
같게 두면 점수가 과장될 수 있습니다. 이를 줄이기 위해 검증을 두 겹으로 나눴습니다.

~~~text
바깥쪽 3-fold
  ├─ 94명: 모델 설정 탐색과 학습
  └─ 47명: 바깥쪽 성능 확인

94명 안의 안쪽 3-fold
  ├─ Transformer 설정 탐색
  ├─ TabNet 설정 탐색
  ├─ YDF 설정 탐색
  └─ 세 모델을 섞는 비율 탐색
~~~

바깥쪽 3-fold를 다른 seed로 두 번 반복하여 총 6개 fold를 만들었습니다.

### 9.2 탐색 규모

각 바깥 fold에서 다음 횟수만큼 Optuna 탐색을 수행했습니다.

| 모델 | fold당 trial | 6개 fold 전체 |
| --- | ---: | ---: |
| Transformer | 24 | 144 |
| TabNet | 24 | 144 |
| YDF | 40 | 240 |
| 합계 | 88 | **528** |

각 trial은 다시 안쪽 3-fold에서 평가됐습니다. 최적 설정의 안쪽 OOF 재생성과
바깥쪽 최종 학습까지 합치면 약 **1,656회의 모델 fit**이 수행됐습니다.

### 9.3 설정을 고른 점수

단순 accuracy만 크게 만드는 설정을 선택하지 않았습니다.

~~~text
선택 점수 =
    Macro F1          40%
  + Macro ROC-AUC     25%
  + Accuracy          20%
  + Balanced accuracy 15%
~~~

모든 class를 같은 비중으로 보는 Macro F1과 balanced accuracy를 포함해,
CN만 많이 맞혀도 좋은 모델처럼 보이는 문제를 줄이려 했습니다.

## 10. 실제 6개 fold의 혼합 비율

각 행은 하나의 fold 앙상블입니다. 모델 비율은 세 모델 확률을 섞는 비율이고,
MCI·DEM 배율은 혼합한 뒤 해당 class 확률에 곱한 값입니다. 이후 합이 다시 1이
되도록 정규화합니다.

| Repeat/Fold | Transformer | TabNet | YDF | MCI 배율 | DEM 배율 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0/0 | 0.7 | 0.3 | 0.0 | 0.70 | 1.00 |
| 0/1 | 0.0 | 0.0 | 1.0 | 0.70 | 1.75 |
| 0/2 | 0.1 | 0.3 | 0.6 | 0.70 | 1.75 |
| 1/0 | 0.7 | 0.2 | 0.1 | 1.00 | 1.75 |
| 1/1 | 0.0 | 0.2 | 0.8 | 1.75 | 1.75 |
| 1/2 | 0.5 | 0.0 | 0.5 | 0.70 | 1.20 |

단순 평균 비중은 Transformer 33.3%, TabNet 16.7%, YDF 50.0%입니다. 그러나
fold별 확률 모양과 class 배율이 다르므로 “최종 예측의 절반을 YDF가 결정했다”라고
정확히 해석할 수는 없습니다.

모델 가중치는 0.1 간격의 66개 조합을 비교했습니다. MCI·DEM 배율까지 합치면
각 fold에서 2,376개의 혼합 방법을 안쪽 OOF로 비교했습니다. Validation 점수는
이 비율을 선택하는 데 사용하지 않았습니다.

## 11. Validation 결과

### 11.1 혼동행렬을 쉬운 표로 보기

| 실제 class | 전체 인원 | CN으로 예측 | MCI로 예측 | DEM으로 예측 | 맞힌 인원 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CN | 26 | 25 | 1 | 0 | 25 |
| MCI | 4 | 4 | 0 | 0 | 0 |
| DEM | 3 | 0 | 0 | 3 | 3 |
| 합계 | 33 | 29 | 1 | 3 | **28** |

따라서 accuracy는 다음과 같습니다.

~~~text
28 ÷ 33 = 0.84848 = 84.85%
~~~

### 11.2 전체 지표

| 지표 | 값 | 쉬운 뜻 |
| --- | ---: | --- |
| Accuracy | **0.84848** | 전체 33명 중 28명을 맞힘 |
| Balanced accuracy | **0.65385** | CN·MCI·DEM recall을 같은 비중으로 평균 |
| Macro F1 | **0.63636** | 세 class의 F1을 인원수와 무관하게 평균 |
| Weighted F1 | **0.80716** | 사람이 많은 class에 더 큰 비중을 둔 F1 |
| Macro OvR ROC-AUC | **0.72790** | 각 class를 나머지와 구분해 순위를 매기는 능력의 평균 |
| Log loss | **0.46357** | 틀린 답에 높은 확신을 주면 더 크게 벌점, 낮을수록 좋음 |

### 11.3 class별 결과

| class | Precision | Recall | F1 | OvR AUC |
| --- | ---: | ---: | ---: | ---: |
| CN | 0.8621 | 0.9615 | 0.9091 | 0.6923 |
| MCI | 0.0000 | 0.0000 | 0.0000 | 0.4914 |
| DEM | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

MCI AUC 0.4914는 무작위 순위인 0.5와 거의 같습니다. 저장된 확률을 살펴보면
실제 CN의 평균 CN 확률은 약 0.777이고, 실제 MCI의 평균 CN 확률은 약 0.793입니다.
모델 관점에서 Validation CN과 MCI가 거의 같은 모습이었다는 뜻입니다.

## 12. 0.84848이 높아 보이는 이유

Validation 33명 중 CN이 26명입니다.

~~~text
모두 CN이라고만 답하는 단순 기준:
26 ÷ 33 = 0.78788

이번 앙상블:
28 ÷ 33 = 0.84848

순수한 정답 증가:
2명, 약 6.06%p
~~~

이번 모델은 DEM 3명을 추가로 맞혔지만 CN 1명을 MCI로 잘못 분류했습니다.
결과적으로 “모두 CN” 기준보다 두 명을 더 맞힌 것입니다.

따라서 84.85%만 보고 세 집단을 고르게 잘 구분했다고 말하면 안 됩니다.
Balanced accuracy와 Macro F1이 accuracy보다 낮고, MCI F1이 0이라는 사실을
항상 함께 제시해야 합니다.

Validation은 33명뿐이라 한 사람의 결과가 accuracy를 약 **3.03%p** 바꿉니다.
DEM도 3명뿐이므로 DEM 100%는 아직 매우 불확실합니다.

## 13. 더 보수적인 내부 결과

Nested OOF는 각 Training 사람을 그 사람을 학습하지 않은 모델로 예측한 결과입니다.
각 사람은 두 번 예측되고, 두 확률을 평균했습니다.

| 지표 | Nested OOF | Validation |
| --- | ---: | ---: |
| Accuracy | 0.60993 | 0.84848 |
| Balanced accuracy | 0.47782 | 0.65385 |
| Macro F1 | 0.49126 | 0.63636 |
| Macro ROC-AUC | 0.70644 | 0.72790 |
| Log loss | 0.91710 | 0.46357 |

Nested OOF 혼동행렬은 다음과 같습니다.

| 실제 class | CN 예측 | MCI 예측 | DEM 예측 |
| --- | ---: | ---: | ---: |
| CN 85명 | 70 | 15 | 0 |
| MCI 47명 | 31 | 13 | 3 |
| DEM 9명 | 2 | 4 | 3 |

Training에서도 모두 CN으로 답하면 85/141 = 0.60284입니다. OOF accuracy는 이
기준보다 정답 한 명 정도만 많습니다. 다만 Macro F1은 모두-CN 기준 약 0.251보다
0.491로 높아, MCI와 DEM에 대한 일부 신호는 배웠습니다.

6개 바깥 fold의 accuracy는 **0.4468~0.7021**, Macro AUC는
**0.6034~0.7741**로 많이 흔들렸습니다. 따라서 0.84848을 새 사람에게 그대로
기대할 수 있는 일반 성능으로 보면 안 됩니다.

## 14. MMSE를 어떻게 해석해야 하는가

### 직접 정답 누출은 차단했습니다

DIAG_NM은 모델 입력으로 만들어지지 않았고, forbidden-feature 검사도 적용했습니다.
따라서 모델이 DIAG_NM 문자를 그대로 복사해 0.84848을 만든 것은 아닙니다.

### 그래도 MMSE 의존성은 큽니다

MMSE는 진단하려는 인지 상태를 직접 검사하는 자료입니다. 실제 진단 과정에서
MMSE가 참고됐을 가능성도 데이터만으로 배제할 수 없습니다.

따라서 이 모델의 정확한 용도 표현은 다음과 같습니다.

> MMSE와 생활기록을 함께 사용할 수 있을 때 CN/MCI/DEM 판단을 보조하는 연구용 모델

다음 표현은 부정확합니다.

> 웨어러블만으로 치매를 84.85% 정확도로 진단하는 모델

MMSE 없는 CNBoost와도 입력 조건이 다르므로 숫자를 그대로 비교하면 공정하지
않습니다. 공정한 비교에는 같은 사람·같은 split에서 MMSE만 뺀 ablation 실험이
필요합니다.

## 15. Validation 정보 누출 방지와 남은 주의점

이 실행에서는 다음 순서를 지켰습니다.

1. Validation의 LabelingData 정답 CSV를 열지 않은 상태로 특징과 확률을 생성
2. 예측 파일 저장
3. VALIDATION_PREDICTIONS_FROZEN.json 기록
4. 그 뒤 LabelingData 정답 CSV를 열어 지표 계산

Training과 Validation의 subject hash 교집합도 **0명**입니다.

다만 정확히 말하면 Validation MMSE 원본 CSV 전체를 읽는 과정에서 DIAG_NM 열도
메모리에는 들어왔습니다. 코드는 이 열을 특징으로 추출하지 않고 강제로 제외했지만,
파일 읽기 단계부터 완전히 분리한 것은 아닙니다. 더 엄격한 다음 버전에서는
MMSE CSV를 읽을 때 Q 문항과 TOTAL만 usecols로 지정하는 것이 좋습니다.

또한 앞으로 같은 Validation 점수를 반복해서 보고 모델 설계를 바꾸면 이
Validation도 사실상 개발 데이터가 됩니다. 최종 성능 판단에는 한 번도 보지 않은
새로운 test set이 필요합니다.

## 16. 체크포인트 구성

체크포인트는 [training/models](training/models/)에 저장돼 있습니다.

~~~text
training/models/
├── repeat_00_fold_00/
│   ├── selection.json
│   ├── transformer/
│   │   ├── model.pt
│   │   ├── params.json
│   │   └── preprocessor.joblib
│   ├── tabnet/
│   │   ├── model.zip
│   │   ├── params.json
│   │   └── preprocessor.joblib
│   └── ydf/
│       ├── model/
│       ├── params.json
│       └── preprocessor.joblib
├── repeat_00_fold_01/
├── repeat_00_fold_02/
├── repeat_01_fold_00/
├── repeat_01_fold_01/
└── repeat_01_fold_02/
~~~

전체 구성은 다음과 같습니다.

- Transformer model.pt 6개
- TabNet model.zip 6개
- YDF 모델 디렉터리 6개
- 모델별 preprocessor.joblib 18개
- 모델별 params.json 18개
- fold별 selection.json 6개
- 총 90개 파일, 약 18MB

Transformer와 TabNet 압축 컨테이너 12개의 구조 검사는 모두 통과했습니다.

## 17. 체크포인트를 불러올 때 필요한 순서

현재 저장소에는 18개 모델을 한 번에 불러오는 전용 원클릭 loader는 없습니다.
하지만 저장 형식은 각 라이브러리에서 읽을 수 있는 형태입니다.

### 모델별 로드

- Transformer
  - torch.load로 model.pt를 읽음
  - 저장된 config로 같은 구조를 만듦
  - state_dict를 적용함
- TabNet
  - TabNetClassifier를 만든 뒤 load_model로 model.zip을 읽음
- YDF
  - ydf.load_model로 model 디렉터리를 읽음
- 전처리기
  - joblib.load로 각 preprocessor.joblib을 읽음

joblib이 저장한 class 이름은 preprocessing.FoldPreprocessor입니다. 로드할 때
이 실험 폴더가 Python import 경로에 있어야 합니다.

joblib은 Python pickle 계열 형식이므로 출처를 모르는 파일을 열면 안 됩니다.
이 저장소에 보존된 체크포인트처럼 신뢰할 수 있는 파일만 로드해야 합니다.

### 정확한 최종 예측 순서

~~~text
각 fold에 대해:
  1. 같은 feature_engineering.py로 754개 후보 생성
  2. Transformer 전처리기로 변환 → Transformer 확률
  3. TabNet 전처리기로 변환 → TabNet 확률
  4. YDF 전처리기로 변환 → YDF 확률
  5. selection.json의 세 모델 가중치 적용
  6. selection.json의 MCI·DEM 배율 적용
  7. 각 값을 0.00000001~1 범위로 제한
  8. 세 확률의 합이 1이 되도록 정규화

6개 fold 결과를 동일 비중으로 평균
다시 0.00000001~1 범위 제한과 합 1 정규화
가장 큰 확률의 class를 최종 예측
~~~

이 과정에서 다음 중 하나라도 빠지면 0.84848 예측과 달라질 수 있습니다.

- 하나의 fold
- weight가 0인 모델을 포함한 전체 감사용 체크포인트
- fold별 전처리기
- fold별 class 배율
- class 순서
- 원본 특징 생성 코드

참고로 이 시스템에는 141명 전체로 다시 학습한 단일 final model이 없습니다.
각 바깥 모델은 94명으로 학습됐고, 6개 모델 묶음이 함께 최종 역할을 합니다.

## 18. 실행 환경

| 항목 | 기록된 값 |
| --- | --- |
| GPU | NVIDIA A100-SXM4-40GB |
| CUDA | 12.8 |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| pytorch-tabnet | 4.1.0 |
| YDF | 0.16.1 |
| Optuna | 4.9.0 |
| seed | 20260719 |

체크포인트를 다시 열 때는 가급적 위 버전을 맞추는 것이 안전합니다. 특히 joblib
전처리기는 Python과 scikit-learn/pandas 버전 차이에 민감할 수 있습니다.

## 19. 현재 구현에서 발견된 기술적 한계

### 19.1 Activity state 5 처리

원본 Activity 수열에는 state 5가 있지만, 이 버전은 state 1~4의 비율만 따로
만들고 active 상태도 3과 4만 지정합니다. state 5는 전체 다양성 계산에는 들어가지만
자기 비율이 없고 active 계산에서는 비활동처럼 취급될 수 있습니다.

즉, 일부 활동 의미가 손실되거나 왜곡될 가능성이 있습니다. CNBoost의 수정된
전처리에서는 이 부분을 별도로 보완했습니다.

### 19.2 숫자 수열의 0 제거

심박·RMSSD·MET 같은 일반 숫자 수열 요약 시 0을 제거합니다. 센서의 결측 표현이
0이라면 도움이 되지만, MET에서 실제 휴식이나 매우 낮은 활동을 뜻하는 0까지
제거될 가능성이 있습니다.

### 19.3 작은 표본과 불균형

- Training DEM 9명: 한 명이 DEM recall의 약 11.1%에 해당
- Validation MCI 4명: 한 명이 MCI recall의 25%에 해당
- Validation DEM 3명: 한 명이 DEM recall의 33.3%에 해당

따라서 class별 0% 또는 100%가 매우 쉽게 만들어집니다.

### 19.4 확률 보정 미검증

모델은 class scale과 정규화를 사용하지만, 예측 확률이 실제 위험도와 정확히
일치하는지를 별도 calibration set에서 검증하지 않았습니다. 예를 들어 확률
0.8을 “실제로 80%의 확률”이라고 바로 해석하면 안 됩니다.

## 20. 이 모델을 사용해도 되는 경우와 안 되는 경우

### 연구 목적으로 가능한 경우

- 동일한 데이터 구조에서 기존 결과를 재현할 때
- MMSE+생활기록 결합 모델의 기준선으로 사용할 때
- CN과 뚜렷한 DEM 구분 가능성을 탐색할 때
- 새 CNBoost와 모델 구조를 비교하되 입력 차이를 명시할 때

### 그대로 사용하면 안 되는 경우

- MMSE 없이 웨어러블만으로 예측한다고 설명할 때
- MCI 선별 도구로 사용할 때
- 84.85%를 새 병원·새 기기·새 집단의 보장 성능으로 제시할 때
- 의료진 판단을 대신하는 자동 진단 도구로 사용할 때
- 개별 확률을 임상 위험도로 바로 해석할 때

## 21. 권장 다음 검증

1. CNBoost 결과와 비교할 때 MMSE 포함 여부를 반드시 분리합니다.
2. 같은 split에서 MMSE 포함/제외 ablation을 수행합니다.
3. MCI를 충분히 포함한 새로운 test set을 확보합니다.
4. 같은 Validation을 더 이상 모델 선택에 사용하지 않습니다.
5. Activity state 5와 MET 0 처리 수정 전후를 비교합니다.
6. class별 ROC-AUC, PR-AUC, recall과 calibration을 함께 봅니다.
7. 최종 배포 후보에는 입력 스키마와 체크포인트 해시를 고정합니다.

## 22. 관련 파일

- [최종 결과](training/FINAL_REPORT.json)
- [Validation 상세 결과](training/validation_report.json)
- [Nested OOF 상세 결과](training/nested_cv_report.json)
- [Validation 확률과 정답](training/validation_predictions_evaluated_hashed.csv)
- [Label-free 예측 고정 기록](training/VALIDATION_PREDICTIONS_FROZEN.json)
- [실행 설정](training/run_config.json)
- [실행 환경](training/environment.json)
- [특징 목록](training/feature_manifest.json)
- [Training 특징 감사](training/training_feature_audit.json)
- [Training-only EDA](eda/EDA_REPORT_KO.md)
- [체크포인트](training/models/)
- [학습 파이프라인 소스](../train.py)
- [특징 생성 소스](../feature_engineering.py)
- [전처리 소스](../preprocessing.py)
- [모델 소스](../models.py)
- [Colab 의존성 목록](../requirements_colab.txt)

## 23. 최종 한 문장

> 이 결과는 MMSE·활동·수면을 함께 입력한 18개 모델 앙상블이
> 불균형한 Validation 33명에서 CN과 뚜렷한 DEM을 잘 맞혀 얻은
> accuracy 0.84848이며, MCI 구분 성능은 아직 확보되지 않았습니다.

이 문서는 연구 결과를 과장하지 않고 재현과 다음 실험을 돕기 위한 설명서입니다.
의료 진단을 대신하는 문서가 아닙니다.
