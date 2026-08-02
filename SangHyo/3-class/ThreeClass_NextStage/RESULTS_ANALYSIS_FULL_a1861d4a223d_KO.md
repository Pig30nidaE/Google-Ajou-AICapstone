# ThreeClass NextStage 실행 결과 분석

- 분석 대상 FULL run: `full_train_only_a1861d4a223d`
- 보조 확인 대상 FAST run: `fast_train_only_1fa63ba40588`
- 비교 대상 기존 run: `full_lifelog_ed5a752a4120`
- 분석일: 2026-07-16
- 주 평가 단위: subject level
- 주 평가 지표: CN/MCI/Dem 3-class Macro F1

## 1. 결론 요약

이번 FULL run은 **실행 및 validation 격리 절차는 정상적으로 완료**되었다. 그러나 성능 관점에서는 현재 기존 모델을 대체할 만한 개선으로 볼 수 없다.

- 주 결과인 repeated nested CV Macro F1은 **0.350 ± 0.071**이다.
- 기존 run의 nested CV Macro F1 **0.358**보다 **0.008 낮다**.
- 새 방법은 MCI F1을 0.250에서 0.323으로 높였지만, CN F1이 0.681에서 0.616으로 낮아졌고 Dem 성능도 안정적으로 개선되지 않았다.
- 역사적으로 재사용된 benchmark에서는 Macro F1이 **0.403**으로, 기존 run의 **0.474**보다 0.071 낮았다. 특히 MCI 4명을 한 명도 맞히지 못했다.
- 고정 후보 중 `mask_tcn_35d`는 nested CV Macro F1 **0.404 ± 0.101**로 가장 높은 가능성을 보였지만, 두 반복의 점수가 0.332와 0.475로 크게 갈렸다. 이 결과를 본 뒤 현재 outer 결과로 TCN을 선택하면 outer fold를 모델 선택에 재사용하는 것이므로, **이번 run의 우승 모델로 사후 승격해서는 안 된다**. 다음 실험의 사전 고정 가설로만 사용할 수 있다.
- outer fold 6개에서 선택된 앙상블 규칙이 모두 달랐고, inner 선택 점수와 outer 성능의 차이가 평균 0.105였다. 현재 가장 큰 병목은 후보 모델의 부족이라기보다 **작은 데이터에서 과도하게 유연한 모델·앙상블 선택 과정의 불안정성**이다.

따라서 현재 권고는 다음과 같다.

> 기존 run을 잠정 기준선으로 유지하고, 다음 실험에서는 후보 수와 선택 자유도를 줄인 뒤 `mask_tcn_35d`를 포함한 소수의 규칙을 미리 고정하여 새로운 outer seed들로 반복 검증한다. 역사적 benchmark는 더 이상 선택 근거로 사용하지 않는다.

## 2. 실행 및 데이터 계약 확인

### 2.1 실행 상태

FULL discovery notebook과 frozen benchmark notebook의 저장 출력에서 exception/error output은 발견되지 않았다.

| 항목 | 확인 결과 |
| --- | --- |
| 실행 장치 | NVIDIA A100-SXM4-80GB |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| TabPFN | 8.1.0 |
| scikit-learn | 1.7.2 |
| discovery 실행 시간 | 2,467.46초, 약 41분 7초 |
| benchmark 실행 시간 | 18.52초 |
| FULL mode | `fast_mode=false` |
| outer 반복 | seed 42, 2024의 2회 |
| fold 수 | 반복당 3-fold |
| 학습 subject | 141명: CN 85, MCI 47, Dem 9 |

FAST run은 outer 반복이 한 번뿐인 실행 점검용 결과이다. Macro F1 0.363을 기록했지만 표준편차가 0으로 표시되는 것은 안정적이라는 뜻이 아니라 **반복이 한 번이라 변동을 계산할 수 없다는 뜻**이다. FAST run에는 완료된 benchmark 결과도 없으므로 성능 결론에는 사용하지 않았다.

### 2.2 누수 및 동결 절차

학습 audit에서 다음 사항이 확인되었다.

- discovery 단계의 `official_benchmark_accessed`는 `false`이다.
- MMSE 값은 feature로 로드되지 않았다.
- 세 modality의 학습 label은 서로 일치한다.
- feature 금지 목록에 subject ID, 진단명, MMSE, 의사 정보, sample order, 절대 날짜, `sleep_period_id`가 명시되어 있다.
- benchmark는 frozen contract 생성 후 별도 notebook에서 실행되었고, 완료 artifact에 frozen contract와 benchmark 입력 hash가 기록되어 있다.
- benchmark는 **historically reused official benchmark**로 명시되어 있으며 독립적인 외부 holdout으로 취급되지 않았다.

이 때문에 이번 결과의 가장 큰 문제는 명백한 label leakage가 아니라, 소표본에서의 선택 변동과 coverage 신호 의존성이다.

### 2.3 예측 계약과 입력 규모

- 목표는 미래 전환 예측이 아니라 **subject별 마지막 activity 날짜를 기준으로 한 동시점 cognitive status 분류**이다.
- class mapping은 CN=0, MCI=1, Dem=2이다.
- tabular view는 `legacy_all` 3,704개, `compact35` 1,031개, `compact_multi` 3,093개 feature이다.
- sequence 입력은 141 × 35일 × 49변수이다.
- 원시 activity와 sleep 행은 각각 9,705개이며, 일 단위 sleep 행은 9,694개이다.

## 3. 주 결과: repeated nested CV

주 보고값은 각 outer 반복에서 모든 subject의 OOF prediction을 합친 뒤 계산한 점수의 평균과 표본 표준편차이다. 반복이 두 번뿐이므로 이 표준편차를 정밀한 신뢰구간으로 해석하면 안 된다.

| 지표 | 새 NextStage | 기존 run | 새−기존 | 해석 |
| --- | ---: | ---: | ---: | --- |
| Macro F1 | 0.350 ± 0.071 | 0.358 | -0.008 | 주 지표 개선 없음 |
| Balanced accuracy | 0.350 ± 0.072 | 0.359 | -0.009 | 개선 없음 |
| Accuracy | 0.489 ± 0.050 | 0.532 | -0.043 | CN 편향 완화의 대가도 포함 |
| Log loss | 0.990 ± 0.002 | 0.958 | +0.032 | 낮을수록 좋으므로 악화 |
| Macro OVR AUROC | 0.542 ± 0.030 | 0.530 | +0.012 | 순위 분리력은 소폭 상승 |
| Macro OVR AUPRC | 0.372 ± 0.026 | 0.359 | +0.013 | 소폭 상승 |

두 outer 반복의 Macro F1은 각각 **0.300, 0.400**으로 0.100 차이가 났다. subject 분할이 바뀌면 결론이 크게 흔들린다는 뜻이다.

학습 fold의 class prior만 예측하는 기준선은 Macro F1 0.251, balanced accuracy 0.333이었다. 새 pipeline은 이 기준선보다 Macro F1은 높지만 log loss는 0.990으로 기준선의 0.847보다 나쁘다. 즉, 소수 클래스를 더 자주 예측해 Macro F1을 높였지만 **확률값의 신뢰도와 calibration은 충분하지 않다**.

### 3.1 클래스별 결과

| 클래스 | 새 Precision | 새 Recall | 새 F1 | 기존 F1 | 변화 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CN, n=85 | 0.604 | 0.629 | 0.616 | 0.681 | -0.065 |
| MCI, n=47 | 0.340 | 0.309 | 0.323 | 0.250 | +0.073 |
| Dem, n=9 | 0.111 | 0.111 | 0.111 | 0.143 | -0.032 |

핵심 변화는 MCI recall이 기존 0.213에서 0.309로 높아진 대신 CN recall이 0.753에서 0.629로 낮아진 것이다. 따라서 새 모델은 기존보다 덜 CN 중심적으로 예측하지만, 그 교환이 전체 Macro F1의 순개선으로 이어지지는 않았다.

Dem F1의 반복 평균은 0.111이고 표준편차는 0.157이다. 첫 반복에서는 Dem 9명 중 0명, 두 번째 반복에서는 2명을 맞혔다. Dem이 9명뿐이라 subject 한두 명이 지표를 크게 바꾸므로, 현재 수치로 Dem 분류 능력이 확립되었다고 말할 수 없다.

반복별 confusion matrix는 다음과 같다. 행은 실제 클래스, 열은 예측 클래스이며 순서는 CN, MCI, Dem이다.

```text
seed 42
[[49, 28, 8],
 [30, 15, 2],
 [ 4,  5, 0]]

seed 2024
[[58, 22, 5],
 [31, 14, 2],
 [ 5,  2, 2]]
```

가장 큰 오류 축은 여전히 CN↔MCI이다. MCI 47명 중 약 30~31명이 CN으로 분류되었다. Dem도 첫 반복에서는 주로 CN 또는 MCI로 흡수되었고, 두 번째 반복에서만 일부 분리되었다.

## 4. 후보 모델과 앙상블 선택의 진단

### 4.1 고정 후보 성능

| 후보 | Nested Macro F1 | SD | Log loss | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mask_tcn_35d` | **0.404** | 0.101 | **0.899** | **0.566** | **0.438** |
| `lda_compact35_k32` | 0.365 | 0.035 | 1.877 | 0.524 | 0.353 |
| `tabpfn3_pairwise_compact35_k64` | 0.351 | 0.075 | 1.008 | 0.539 | 0.355 |
| `tabpfn3_compact_multi_k96_balanced` | 0.349 | 0.019 | 1.096 | 0.547 | 0.365 |
| `tabpfn3_compact35_k64_balanced` | 0.345 | 0.050 | 1.095 | 0.539 | 0.358 |
| `elastic_compact35_k32` | 0.342 | 0.043 | 1.195 | 0.535 | 0.357 |
| `elastic_compact_multi_k48` | 0.311 | 0.004 | 1.196 | 0.548 | 0.359 |
| `cat_compact35_k48` | 0.298 | 0.018 | 1.030 | 0.528 | 0.357 |
| `minirocket_35d` | 0.289 | 0.024 | 0.998 | 0.526 | 0.363 |
| `tabpfn3_compact35_k64_raw` | 0.270 | 0.011 | 0.902 | 0.535 | 0.346 |
| `tabpfn3_compact35_native_raw` | 0.251 | 0.000 | 0.850 | 0.537 | 0.355 |

`mask_tcn_35d`가 점수와 log loss 모두 가장 유망하다. 하지만 반복별 Macro F1은 0.332와 0.475이고, MCI F1 표준편차는 0.204이다. 한 반복에서는 MCI 47명 중 4명, 다른 반복에서는 21명을 맞혔다. 따라서 평균 0.404만 보고 안정적인 우승 모델이라고 결론내리면 안 된다.

`lda_compact35_k32`는 Macro F1은 두 번째이지만 log loss 1.877로 확률 calibration이 매우 나쁘다. `tabpfn3_compact35_native_raw`는 모든 subject를 CN으로 예측하여 Macro F1이 class-prior 기준선 수준에 머물렀다. class balancing이나 pairwise 구성이 TabPFN의 소수 클래스 예측에는 필요했지만, 그 효과도 반복별로 안정적이지 않았다.

### 4.2 adaptive selection이 고정 TCN보다 나빴다

inner CV에서 fold마다 모델과 가중치를 다시 고른 `SELECTED_PIPELINE`은 Macro F1 0.350이었다. 반면 동일 outer prediction에서 고정 후보 `mask_tcn_35d`는 0.404였다.

이 차이는 “현재 결과를 보고 TCN으로 바꾸면 된다”는 뜻이 아니다. TCN의 우수성을 outer 결과에서 발견했기 때문에, 지금 TCN을 선택하면 outer fold가 사실상 개발 데이터가 된다. 올바른 사용법은 다음과 같다.

1. 이번 결과를 근거로 `mask_tcn_35d`를 **다음 실험의 사전 고정 후보**로 등록한다.
2. 아직 사용하지 않은 outer seed 또는 새 외부 cohort에서 다시 평가한다.
3. 새 평가가 끝날 때까지 구조, loss, threshold, class weight를 바꾸지 않는다.

### 4.3 fold별 선택 규칙이 모두 달랐다

6개 outer fold에서 동일한 최종 규칙이 한 번도 반복되지 않았다.

| Outer seed/fold | Inner 선택 모델 | Inner F1 | Outer F1 |
| --- | --- | ---: | ---: |
| 42/0 | CatBoost + balanced multi TabPFN + TCN | 0.524 | 0.276 |
| 42/1 | multi-window Elastic Net 단독 | 0.411 | 0.314 |
| 42/2 | pairwise TabPFN + TCN | 0.405 | 0.294 |
| 2024/0 | raw TabPFN + balanced multi TabPFN + MiniRocket | 0.548 | 0.417 |
| 2024/1 | LDA + Elastic Net + balanced multi TabPFN | 0.479 | 0.416 |
| 2024/2 | balanced multi TabPFN + MiniRocket | 0.329 | 0.348 |

- 6개 중 5개 fold에서 inner 선택 점수가 outer 점수보다 높았다.
- `inner − outer` 차이는 평균 0.105, 중앙값 0.104였다.
- 선택 temperature도 약 0.69부터 상한 2.5까지 크게 달라졌다.
- 선택 pipeline은 기존 legacy candidate보다 fold Macro F1이 4/6 fold에서 높았지만 2/6에서는 낮았고, paired delta는 +0.042 ± 0.077이었다.

후보가 12개이고 여러 앙상블 조합·가중치·temperature까지 inner CV에서 동시에 고르는 현재 검색 공간은 subject 94명, 그중 Dem 6명 정도인 outer-training fold에 비해 너무 유연하다. 이 때문에 inner fold의 우연한 패턴을 선택하는 winner's curse가 발생한 것으로 보인다.

## 5. 최종 OOF와 frozen rule의 의미

전체 학습 데이터에서 final selection OOF로 고정된 규칙은 다음과 같다.

- `tabpfn3_pairwise_compact35_k64`: weight 0.75
- `mask_tcn_35d`: weight 0.25
- temperature: 0.748
- temperature 적용에 따른 OOF log-loss gain: 0.0076
- full-data refit seed: 910000, 910001

이 규칙의 selection OOF Macro F1은 0.398이고 95% conditional bootstrap interval은 약 0.300~0.507이다. confusion matrix는 다음과 같다.

```text
[[56, 24, 5],
 [30, 12, 5],
 [ 3,  3, 3]]
```

그러나 0.398을 주 성능으로 보고하면 안 된다. 이 OOF 결과는 최종 rule 선택에도 사용되었기 때문에 낙관 편향이 포함된다. selection OOF와 nested 주 결과의 차이는 **+0.049**이다. 주 성능은 계속 nested CV의 **0.350 ± 0.071**로 보고해야 한다.

최고 grid rule의 OOF Macro F1은 0.407이었지만, 단순성 규칙에 따라 0.398짜리 rule이 선택되었다. 이 선택 자체는 과적합을 줄이려는 합리적인 장치지만, nested 결과에서 확인된 큰 선택 변동을 해소하지는 못했다.

## 6. 역사적으로 재사용된 benchmark

이 benchmark는 33명(CN 26, MCI 4, Dem 3)뿐이며 이미 과거 실험에 사용되었다. 따라서 다음 표는 frozen inference의 동작과 이동 방향을 설명하기 위한 것이며, 새 모델을 선택하는 독립 증거가 아니다.

| 지표 | 새 NextStage | 기존 run | 새−기존 |
| --- | ---: | ---: | ---: |
| Macro F1 | 0.403 | 0.474 | -0.071 |
| Balanced accuracy | 0.440 | 0.511 | -0.071 |
| Accuracy | 0.576 | 0.576 | 0.000 |
| Log loss | 0.836 | 0.818 | +0.018, 악화 |
| Macro OVR AUROC | 0.609 | 0.612 | -0.004 |
| Macro OVR AUPRC | 0.576 | 0.568 | +0.008 |

새 run의 Macro F1 bootstrap interval은 **0.235~0.552**로 매우 넓다.

```text
새 benchmark confusion matrix
[[17, 6, 3],
 [ 4, 0, 0],
 [ 1, 0, 2]]
```

- CN: F1 0.708, recall 17/26 = 0.654
- MCI: F1 0.000, recall 0/4 = 0.000
- Dem: F1 0.500, recall 2/3 = 0.667

MCI 4명은 모두 CN으로 분류되었다. 반대로 MCI로 예측한 6명은 모두 실제 CN이었다. 즉, benchmark에서 MCI decision region이 완전히 어긋났다. Dem 2/3을 맞힌 결과와 Dem AUROC 0.967은 표면적으로 높지만 표본이 세 명뿐이므로 한 명의 결과가 recall을 0.333씩 바꾼다.

기존 benchmark는 MCI 1/4을 맞혔고 Dem 2/3을 맞혔다. 새 모델은 동일 accuracy를 유지했지만 MCI true positive를 잃으면서 Macro F1과 balanced accuracy가 낮아졌다.

## 7. 데이터 coverage와 shortcut 위험

### 7.1 35일 window는 “35 nights 보유 cohort”가 아니다

학습 데이터에서 마지막 35 calendar day 안에 실제 sleep night 35개를 모두 가진 subject는 16/141명, **11.3%**뿐이다.

| Cohort/class | subject 수 | sleep nights min/median/max | 35 nights 충족률 |
| --- | ---: | --- | ---: |
| Train 전체 | 141 | 3 / 32 / 35 | 11.3% |
| Train CN | 85 | 13 / 32 / 35 | 14.1% |
| Train MCI | 47 | 3 / 31 / 35 | 8.5% |
| Train Dem | 9 | 17 / 31 / 34 | 0.0% |
| Benchmark 전체 | 33 | 1 / 32 / 35 | 18.2% |
| Benchmark CN | 26 | 1 / 32 / 35 | 11.5% |
| Benchmark MCI | 4 | 29 / 35 / 35 | 75.0% |
| Benchmark Dem | 3 | 22 / 29 / 30 | 0.0% |

따라서 이 실험의 35일 calendar window는 참고 논문의 “최소 35 nights” 대상 조건을 재현하지 않는다. 특히 benchmark MCI의 coverage 분포가 CN·Dem과 현저히 다르다. 이는 진단 상태가 아니라 수집 프로토콜이나 장치 착용 정도를 모델이 학습할 위험을 만든다.

최종 TCN은 입력값 외에 `observed_mask`와 `days_since_observed`를 명시적으로 사용한다. 이 정보는 결측을 올바르게 다루는 데 유용하지만 동시에 acquisition-protocol shortcut이 될 수 있다. pairwise TabPFN의 선택 feature에는 명시적인 coverage 이름이 없었지만, 요약 통계 자체가 관측일 수와 결측 처리의 영향을 받으므로 coverage 독립성이 입증된 것은 아니다.

35 nights 완전 관측 subject만 남기는 방법은 Dem이 0명이 되어 사용할 수 없다. 대신 다음 실험에서는 coverage 자체를 제거·통제하는 ablation이 필요하다.

## 8. 선택 feature 해석

pairwise TabPFN의 최종 64개 feature 중:

- sleep 계열이 53개, 82.8%이다.
- activity 계열은 11개, 17.2%이다.
- 기록된 selection round에서 29개가 항상 선택되었고, 48개가 75% 이상 선택되었다.
- 한 feature는 selection frequency가 0인데 최종 평균 ranking 3위로 포함되었다. pairwise rank 합산 방식의 결과일 수 있으므로 최종 ranking과 안정성 빈도를 함께 검토해야 한다.

상위 feature의 주요 패턴은 다음과 같다.

- sleep midpoint와 그 변동성
- lowest-heart-rate clock phase의 변동성
- restless/light sleep의 수준과 변동성
- HR drop 및 HR drop ratio
- awake bouts, sleep-stage entropy와 transition
- activity rest/high activity와 activity-class entropy

이 패턴은 수면 시간대와 일간 변동성이 유력한 설명 변수라는 가설을 지지한다. 다만 현재 분석은 held-out association이며 임상적 원인이나 질환 기전을 입증하지 않는다. 또한 최종 앙상블의 다른 축인 TCN은 feature ranking이 아니라 시계열 패턴과 missingness channel을 사용하므로, TabPFN의 64개 feature만으로 전체 모델을 설명해서는 안 된다.

## 9. 다음 단계 권고

### 우선순위 1 — 기존 run을 잠정 기준선으로 유지

현재 NextStage는 주 nested Macro F1, balanced accuracy, log loss에서 기존 run보다 낮다. 역사적 benchmark에서도 Macro F1이 낮다. 따라서 `full_lifelog_ed5a752a4120`을 현 시점의 잠정 기준선으로 유지하고, NextStage frozen bundle을 교체 모델로 채택하지 않는다.

### 우선순위 2 — 선택 공간을 줄인 사전 고정 재검증

다음 run에서는 이번 outer 결과를 보고 새 조합을 다시 탐색하지 말고, 예를 들어 다음 세 규칙 정도만 사전에 고정하는 것이 좋다.

1. 기존 run의 frozen rule 또는 동등한 재현 기준선
2. `mask_tcn_35d` 단독
3. 현재 frozen pairwise TabPFN 0.75 + TCN 0.25 앙상블

가중치, temperature, feature 수, TCN 구조를 먼저 동결하고 새로운 outer seed를 최소 5회 이상 추가한다. 계산 비용이 허용되면 10회가 더 낫다. 3-fold를 유지하면 각 valid fold에 Dem 3명이 들어가므로, 4-fold 이상으로 늘려 Dem이 2명인 fold를 만드는 것보다 현재 표본에서는 3-fold가 안전하다.

중요하게도 `mask_tcn_35d`의 0.404는 다음 가설을 만든 값이지, 다음 평가의 독립 성능값이 아니다.

### 우선순위 3 — coverage ablation

동일한 새 split에서 최소한 다음을 비교한다.

- TCN 원본: value + `observed_mask` + `days_since_observed`
- explicit coverage 제거: value 중심, 결측 처리 규칙만 유지
- mask만 사용하고 `days_since_observed` 제거
- coverage strata별 OOF 성능: 예를 들어 1~27일, 28~34일, 35일 관측 그룹
- class label을 가린 상태에서 coverage만으로 진단을 예측하는 negative-control 모델

coverage-only 모델의 Macro F1이 의미 있게 높다면 현재 모델 일부가 생체 패턴보다 수집 패턴을 사용하고 있다는 경고다.

### 우선순위 4 — MCI 오류를 주 진단 대상으로 설정

학습 nested에서는 MCI가 개선되었지만 benchmark에서는 0/4로 붕괴했다. 다음 분석은 training OOF에만 한정하여 다음을 확인해야 한다.

- CN으로 오분류된 MCI의 probability margin과 coverage 분포
- 두 outer 반복에서 예측이 뒤집히는 MCI subject 수
- TCN과 pairwise TabPFN이 동시에 틀리는 MCI와 서로 보완하는 MCI
- MCI score calibration과 threshold의 seed별 안정성

이 분석 뒤에만 hierarchical CN-vs-impaired → MCI-vs-Dem 구조를 새 가설로 고려한다. Dem 9명으로 두 번째 단계를 학습해야 하므로, 현 단계에서 바로 복잡한 hierarchical search를 추가하는 것은 권하지 않는다.

### 우선순위 5 — calibration을 별도 통과 조건으로 관리

Macro F1만으로 rule을 고르면 극단적인 소수 클래스 확률이 선택될 수 있다. 다음 run에서는 모든 calibration을 inner fold 안에서만 수행하면서 다음 조건을 함께 본다.

- nested Macro F1과 balanced accuracy
- multiclass log loss
- class별 recall과 prediction count
- selection OOF와 nested 결과의 gap

성능 차이가 반복 변동보다 작다면 더 단순하고 log loss가 낮은 모델을 선택한다.

### 우선순위 6 — 독립 cohort 확보

가장 큰 통계적 한계는 모델보다 표본 수이다. 학습 Dem 9명, benchmark Dem 3명, benchmark MCI 4명으로는 모델 간 0.03~0.07 차이를 신뢰성 있게 확정하기 어렵다. 역사적 benchmark를 반복 사용하지 말고, 가능하면 수집 프로토콜이 다른 독립 cohort를 확보해야 한다. 그 전까지는 높은 AUROC나 Dem recall을 임상 일반화 성능으로 표현하지 않는다.

## 10. 다음 run의 권장 판정 원칙

다음 모델을 채택하려면 다음을 동시에 만족하는 방향이 바람직하다.

- 새 seed의 repeated nested CV에서 기존 run 대비 paired Macro F1 개선이 대부분의 fold에서 같은 방향일 것
- 평균 개선폭이 seed 간 표준편차에 비해 충분히 클 것
- MCI 개선이 CN과 Dem의 붕괴로 만들어진 것이 아닐 것
- log loss가 악화되지 않거나, 악화 이유가 명확하고 별도 calibration으로 재현 가능하게 해결될 것
- explicit coverage signal 제거 후에도 성능이 유지될 것
- selection OOF와 nested 성능 차이가 현재의 0.049보다 작아질 것
- 마지막으로, 독립 cohort에서 frozen inference가 재현될 것

현재 결과는 **새 feature와 sequence 모델의 가능성을 확인한 탐색 결과**로는 가치가 있다. 특히 TCN과 수면 변동성 feature는 다음 실험의 집중 가설이 될 수 있다. 그러나 현 시점의 과학적으로 엄격한 결론은 “기존 기준선을 넘어선 재현 가능한 개선은 아직 확인되지 않았다”이다.

## 11. 근거 artifact

주요 판단은 다음 파일을 직접 대조하여 작성했다.

- `outputs/3class_nextstage/full_train_only_a1861d4a223d/FINAL_TRAINING_REPORT.json`
- `outputs/3class_nextstage/full_train_only_a1861d4a223d/nested_cv_report.json`
- `outputs/3class_nextstage/full_train_only_a1861d4a223d/nested_candidate_metrics.csv`
- `outputs/3class_nextstage/full_train_only_a1861d4a223d/nested_fold_metrics.csv`
- `outputs/3class_nextstage/full_train_only_a1861d4a223d/final_oof_report.json`
- `outputs/3class_nextstage/full_train_only_a1861d4a223d/historical_benchmark_metrics.json`
- `outputs/3class_nextstage/full_train_only_a1861d4a223d/training_data_audit.json`
- `outputs/3class_nextstage/full_train_only_a1861d4a223d/feature_manifest.json`
- `outputs/3class_nextstage/full_train_only_a1861d4a223d/selected_feature_manifest.json`
- `outputs/3class_nextstage/full_train_only_a1861d4a223d/feature_stability/tabpfn3_pairwise_compact35_k64.csv`
- `outputs/3class_subject_ensemble/full_lifelog_ed5a752a4120/nested_cv_report.json`
- `outputs/3class_subject_ensemble/full_lifelog_ed5a752a4120/validation_metrics.json`

Subject 식별 hash와 개별 prediction은 분석 문서에 포함하지 않았다.
