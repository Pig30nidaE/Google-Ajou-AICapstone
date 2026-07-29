# 기존 결과 감사와 새 모델의 설계 근거

최종 갱신: 2026-07-29

이 문서는 새 코드를 만들기 전에 수행한 읽기 전용 감사를 기록한다. 성능 수치는
저장된 OOF 산출물에서 확인한 값이며, 이 폴더의 새 모델을 실행해서 얻은 값이 아니다.

## 1. `GeminiFeaturePipeline`의 실제 결과

감사 대상은
`GeminiFeaturePipeline_result/20260729_133444_utc/FINAL_REPORT.json`이다.
과제는 Training 141명의 **CN(0) 대 MCI+Dem(1)** 이며 CN 85명, 양성 56명이다.

| arm | pooled mean-OOF ROC-AUC | repeat AUC 평균 |
| --- | ---: | ---: |
| wearable BASE, GBDT | 0.5340 | 0.5231 |
| wearable BASE+Gemma, GBDT | 0.5183 | 0.5155 |
| MMSE+BASE, Logistic | 0.6941 | 0.6850 |
| MMSE+BASE+Gemma, Logistic | **0.7181** | **0.7062** |

MMSE+BASE+Gemma와 MMSE+BASE의 pooled OOF 차이는 `+0.02395`였으나 paired
bootstrap 95% 구간은 `[-0.01567, 0.06331]`로 0을 포함한다. 네 가지
BASE 대 BASE+Gemma 비교 모두 신뢰구간이 0을 포함했다. 따라서 현재 결과는
Gemma가 개선을 만들었다는 증거라기보다, 개선 여부가 결정되지 않았다는 결과다.

API/배선은 정상이다.

- 141명 모두 validated cache feature가 존재한다.
- API 실패와 cache miss는 0이다.
- 12개 Gemma feature에 결측이 없다.
- subject ID와 design matrix 행 정렬이 맞다.
- 모든 arm이 동일한 사람 단위 5-fold × 5-repeat split을 사용한다.

학습 중 반복된
`X does not have valid feature names, but LGBMClassifier was fitted with feature names`
경고는 LightGBM의 내부 NumPy 예측과 sklearn feature-name 검사 사이의 경고다.
저장된 행렬의 feature 수·순서가 어긋났다는 증거는 아니며, 낮은 AUC의 원인도 아니다.

## 2. Gemma가 만든 12개 feature의 한계

기존 Gemma feature는 다음과 같은 0–1 주관 점수였다.

1. `routine_regularity`
2. `sleep_timing_variability`
3. `sleep_continuity`
4. `activity_volume_stability`
5. `sustained_exertion`
6. `diurnal_contrast`
7. `long_term_trend_direction`
8. `short_term_volatility`
9. `weekday_weekend_divergence`
10. `cross_domain_coherence`
11. `atypical_day_frequency`
12. `observation_reliability`

문제는 이 값들이 새 정보를 추출하지 못한다는 점이다. Python이 이미 평균, 표준편차,
CV, 분위수, 추세, 시간대 통계를 계산하고, Gemma가 이를 다시 12개 숫자로 손실
압축한다. 수치 anchor와 실행 공식이 없으므로 사람별 독립 호출의 0.6과 0.7이
동일한 척도라고 보장할 수도 없다.

구체적으로:

- `routine_regularity`는 날짜별 24시간 profile 없이 평균 profile만 보고 판단한다.
- `sustained_exertion`은 bout 길이를 요구하지만 기존 payload에는 bout가 없다.
- `cross_domain_coherence`는 상관 입력 없이 모델에 사실상 재계산을 요구한다.
- `atypical_day_frequency`는 전체 날짜가 아니라 최대 24일 표본만 본다.
- `observation_reliability`는 phenotype보다 측정 품질이므로 예측 feature가 아닌
  gate 또는 audit 값에 가깝다.
- 입력이 부족할 때 `null` 대신 0.5를 쓰게 하여 “정보 없음”과 “중간값”을 섞었다.
- 약 12 KB payload에 중복 통계는 많지만 연속 bout, sleep transition,
  nocturnal HR/RMSSD dynamics 같은 핵심 시간 구조는 빠졌다.

이 감사에 따라 새 모델은 **사람별 LLM 점수 생성**을 폐기했다.

## 3. `Binary_Google_DemRankAUC_select1`의 0.92는 비교 기준이 아니다

해당 폴더의 공식 headline은 repeat AUC 평균 `0.919907 ± 0.015149`이고,
20회 OOF 예측을 사람별 평균한 벡터의 AUC는 `0.927469`이다. 서로 다른 두
estimand가 보고서에 함께 있으므로 “0.9275가 공식 nested AUC”라고 읽으면 안 된다.

더 중요한 차이는 과제다.

| 항목 | 현재 Gemini 과제 | DemRankAUC_select1 |
| --- | --- | --- |
| 양성 | MCI+Dem | Dem |
| 음성 | CN | CN+MCI |
| 개발 코호트 | Training 141명 | Training+historical Validation 174명 |
| 양성 수 | 56 | 12 |
| 최고 정보원 | MMSE 중심 | MMSE 중심 |

DemRank의 최고 nested track은 웨어러블/Gemma가 아니라 MMSE 7개였다. 저장된
평균 OOF를 원래 출처별로 나누면 `mmse_core` AUC는 전체 0.9275, 원 Training
141명에서 0.8939, historical Validation 33명에서 1.0이었다. Validation의 Dem
3명은 Training의 Dem보다 MMSE가 훨씬 낮아 case-mix가 쉬웠다. 독립 holdout도 없다.

코드 감사에서는 두 가지 결함도 확인했다.

1. all-zero MMSE 행에서 TOTAL과 문항은 NaN이 되지만 Pandas의 기본
   `sum(skipna=True)` 때문에 domain 합계는 0, recall deficit은 3이 된다.
   “검사 미실시”가 “최중증”처럼 재인코딩된다.
2. outer-test 예측을 그 test batch 안에서 rank-normalize한다. 같은 환자의
   점수가 함께 예측하는 환자 집합에 따라 바뀌며, 환자 한 명만 넣으면 의미 있는
   순위가 나오지 않는 transductive 평가다.

새 모델은 여기서 intraday primitive 계산 아이디어만 재사용한다. 모든 MMSE 파생
열은 유효한 TOTAL이 없으면 함께 NaN으로 만들고, 예측 순위는 **training score로
학습한 empirical CDF**만 사용한다.

## 4. `papers/LLM_API`에서 채택한 원칙

### DeepFeature

[DeepFeature.pdf](../../papers/LLM_API/DeepFeature.pdf)는 LLM이 여러 출처에서
feature description을 만들고, 이를 실행 가능한 feature로 바꾼 뒤 검증하는
흐름을 제안한다. 새 모델은 여기서 **LLM은 의미와 조합을 제안하고, 체계적인 수치
연산은 Python이 담당한다**는 구분을 채택했다.

논문의 반복적 model-feedback은 이 데이터처럼 작은 코호트에서 전체 OOF 결과를
다시 프롬프트에 넣으면 선택 과적합이 된다. 따라서 이 구현은 label/performance
feedback loop를 사용하지 않는다.

### Rubric Representation Learning

[Rubric Representation Learning.pdf](../../papers/LLM_API/Rubric%20Representation%20Learning.pdf)는
모든 사례에 공유되는 global rubric과 구조화된 fact extraction을 사용한다.
새 모델도 모든 사람에게 동일한 JSON program을 적용한다. 다만 labeled examples로
rubric을 합성하는 방식은 사용하지 않는다. 프로그램 생성 시 환자, 라벨, MMSE,
코호트 통계를 전혀 보내지 않는다.

논문의 대상자별 반복 채점·confidence 집계는 이 버전에는 적용하지 않는다.
Gemma가 대상자를 채점하지 않고 Python이 고정 program을 결정론적으로 실행하므로
같은 입력의 호출 간 불일치라는 변수가 존재하지 않기 때문이다.

### FeatLLM

[FeatLLM-ICML2024.pdf](../../papers/LLM_API/FeatLLM-ICML2024.pdf)는 LLM이
명시적인 rule을 만들고 downstream linear model이 rule activation을 사용하는
구조와 feature/sample bagging을 제안한다. 새 모델은 “대화형 판단” 대신 감사
가능한 declarative program을 쓰는 원칙을 채택했다. 반면 labeled example을
프롬프트에 넣는 class-conditioned rule 생성은 누수·과적합 위험 때문에 제외했다.
여러 program을 생성한 뒤 같은 141명 OOF에서 좋은 program만 고르는 방식도 초기
사전 고정 버전에서는 제외했다. 추후 bagging을 검증하려면 program 수와 결합법을
미리 고정하고 전체 결합 자체를 outer fold 안에서 평가해야 한다.

### Explainable cognitive decline detection

[Explainable cognitive decline detection.pdf](../../papers/LLM_API/Explainable%20cognitive%20decline%20detection.pdf)는
구조화된 0–1 LLM side feature를 classical classifier에 결합한다. 이는 기존
파이프라인의 큰 방향을 지지하지만, 새 구현에서는 자의적인 0–1 점수 대신
fold-standardized primitive에 대한 제한된 연산으로 값을 재현 가능하게 만든다.

### Predicting explainable dementia types with LLM

[Predicting explainable dementia types with LLM.pdf](../../papers/LLM_API/Predicting%20explainable%20dementia%20types%20with%20LLM.pdf)는
정의된 concept의 activation을 구조화하고 이를 linear classifier에 넘긴다.
새 모델도 LLM의 직접 진단이 아니라 명시적 concept/composite activation만
downstream 모델에 전달한다.

## 5. 새 모델의 사전 고정 가설

새 파이프라인의 Gemma 출력은 환자별 12점이 아니라 **한 번 생성하는 전역 JSON
feature program**이다.

- 입력: 이름·설명·단위·도메인만 있는 고정 wearable primitive catalogue
- 비입력: 환자값, subject ID, label, diagnosis, MMSE, cohort 통계, 기존 AUC
- 출력: 8–16개 composite와 각 dependency, 방향, 허용 연산, 근거
- 허용 연산: `signed_mean`, `signed_product`, `absolute_gap`
- 금지: 임의 Python, 자유 가중치, threshold, population norm, label rule
- 실행: fold-training에서 impute/winsor/standardize한 값에 Python이 동일 적용
- 결측: fold-training median; 전체 결측/상수 열 제거도 fold-training에서만

시계 시간은 자정에서 23시와 0시가 멀어지는 선형값으로 쓰지 않는다. M10/L5 onset은
미리 sine/cosine unit-circle 좌표로 바꾼 뒤 catalogue와 모델에 전달한다.

모델은 MMSE, wearable primitive, Gemma program을 별도 block으로 학습한다.
각 raw model score는 training reference empirical CDF에만 매핑한다. inner OOF에서
작은 convex weight grid를 고르며 0을 반드시 후보로 둔다. 따라서 Gemma block이
도움이 없으면 가중치 0을 선택할 수 있다.

평가는 동일한 diagnosis-stratified subject split에서 다음 arm을 함께 낸다.

- `mmse_only`
- `wearable_only`
- `program_only`
- `wearable_plus_program`
- `mmse_plus_wearable`
- `full`

1차 estimand는 repeat별 OOF AUC의 평균이다. 반복 OOF를 사람별 평균한 ensemble
AUC는 2차 지표로 분리한다. `full - mmse_only`와
`wearable_plus_program - wearable_only`의 paired subject bootstrap 구간도
같이 보고한다.

## 6. 0.92 목표의 해석

현재 과제의 정식 최고는 약 0.7658이며 wearable-only는 대체로 0.45–0.57이다.
따라서 이 코드만으로 ROC-AUC 0.92 이상을 보장할 근거는 없다. 0.92는 연구 목표로
기록할 수 있지만, 달성 주장은 새 코호트 또는 사전 고정된 nested OOF에서 실제로
관측된 후에만 가능하다.

더 중요한 성공 기준은 다음 둘이다.

1. 동일 split의 `mmse_only` 대비 `full` paired AUC 차이의 신뢰구간이 0보다 큰가.
2. Gemma program weight가 반복적으로 0보다 크게 선택되고,
   `wearable_plus_program`이 `wearable_only`를 안정적으로 개선하는가.

이 기준을 만족하지 않으면 0.92에 미달한 이유를 모델 규모가 작아서라고 해석하지
않고, 현재 웨어러블에 추가적인 분류 신호가 확인되지 않았다고 결론 내린다.
