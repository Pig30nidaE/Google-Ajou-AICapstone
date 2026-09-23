# 최종 모델 (V4): 2-model 계층구조 — DEM 분리 모델 + CN/MCI 분리 모델

**기준일**: 2026-09-23
**분석 대상**: 현재 프로젝트 데이터 174명 (Training 141 / Validation 33, 피험자 단위 완전 분리, MMSE 미사용)
**최종 채택 구조**: 원래 4-component(2개 hierarchy를 앙상블) champion을 단순화한 **2-model 구조** — DEM vs 나머지를 가르는 모델 1개 + (DEM 제외) CN vs MCI를 가르는 모델 1개, 총 feature 20개(모델당 10개)

---

## 1. 요약

- 원본 V3 champion(95개 feature, 4개 sub-model을 앙상블, 4,258회 탐색)의 Selection-Validation 성능(Macro AUC 0.976)은 **33명 Validation 정답을 반복 탐색에 직접 사용한 결과이며, 일반화 성능이 아님**이 확인되었다(원본 보고서 스스로도 명시).
- feature/model 단순화(중복 제거 → branch 전문화 → 2-model 축소)를 여러 단계로 시도한 결과, **Selection-Validation 기준 성능은 4-component와 2-model 사이에 통계적으로 유의한 차이가 없었다**(bootstrap 95% CI가 서로 크게 겹침).
- 174명 전체를 프레시하게 재분할한 **honest nested CV(outer 5-fold × inner 3-fold)에서는 4가지 구성(원본/중복제거/4-branch전문화/2-model) 전부 Macro AUC 0.57~0.61로 수렴**했다 — 이는 V2가 이미 확인했던 정직한 추정치(0.543)와 같은 수준이다. 즉 지금까지의 모든 "개선"은 33명이라는 고정 Validation에 반복적으로 맞춰가며 생긴 선택 낙관성이었을 가능성이 크다.
- 다만 지도교수님 피드백에 따라, **본 프로젝트의 현재 목적은 "엄격한 일반화 증명"이 아니라 "어떤 feature/모델 구조가 3-class 구분에 기여하는지"를 보는 것으로 재설정**되었다. 이 문서는 그 목적에 맞춰, **구조가 제일 단순하면서 4-component와 성능 차이가 없는 2-model 구조를 최종 모델로 채택**하고, 그 근거와 feature 기여도를 정리한다.
- 신뢰구간(bootstrap CI)은 계산은 했으나 MCI(n=4)/DEM(n=3)처럼 표본이 극소수인 클래스에서 구간이 통계적으로 무의미(예: [1.0, 1.0], [0.0, 1.0])하게 나와 본 보고서의 핵심 근거로 사용하지 않는다.

---

## 2. 왜 이 구조를 최종으로 선택했는가

| 구성 | feature 수 | 모델 수 | Selection-Validation Macro F1 |
|---|---:|---:|---:|
| 원본 champion (전체 탐색) | 95 | 4 | 0.956 |
| 중복 제거 | 94 | 4 | 0.756 |
| 4-branch, branch별 전문 top10 | 20 | 4 | 0.829 |
| **2-model, top10씩 (최종 채택)** | **20** | **2** | **0.802** |

4-component(4-branch 전문화)와 2-model은 Selection-Validation 성능이 거의 동일하고(F1 0.829 vs 0.802, 95% CI가 겹침), **honest nested CV에서는 두 구성 모두 비슷한 수준(각각 0.577 / 0.569 AUC)으로 수렴**했다. 즉 두 번째 hierarchy를 추가해서 앙상블하는 구조가 주는 실질적 이득은 이 데이터 규모에서는 확인되지 않는다. 모델 수가 절반이고 구조가 훨씬 단순해 임상적 설명이 쉬운 2-model 구조를 최종으로 선택한다.

이 구조는 임의로 고안한 것이 아니라, **원래 champion 안에 이미 내장돼 있던 설계 논리**(DEM 판별 = activity 신호, CN/MCI 판별 = 수면·심박 신호)를 그대로 가져오되, 탐색량과 feature 수만 대폭 줄인 것이다.

---

## 3. 데이터

| Split | CN | MCI | DEM | 합계 |
|---|---:|---:|---:|---:|
| Training | 85 | 47 | 9 | 141 |
| Validation | 26 | 4 | 3 | 33 |
| 전체 | 111 | 51 | 12 | 174 |

- 피험자 단위 완전 분리 (Training/Validation 교집합 0명)
- MMSE 등 인지기능 검사 결과는 predictor로 사용하지 않음 (Activity/Sleep 웨어러블 데이터만 사용)
- 원천: AI-Hub 치매 고위험군 웨어러블 라이프로그, 로컬 원본 CSV로부터 직접 재추출

---

## 4. 모델 구조

```text
Input features (20개)
        │
        ├──────────────────┐
        │                  │
  DEM vs 나머지        (DEM 제외) CN vs MCI
   (activity 10개)         (수면/심박 10개)
        │                  │
      d(0~1)              q(0~1)
        │                  │
        └───────┬──────────┘
                │
   CN  = (1-d) × (1-q)
   MCI = (1-d) × q
   DEM = d
                │
   argmax(log(score) + class offset)
                │
          CN / MCI / DEM
```

### 4.1 DEM 분리 모델

| 항목 | 값 |
|---|---|
| 대상 | DEM vs (CN+MCI) |
| feature 후보 풀 | activity 계열 (수면 feature 접근 불가) |
| feature 선택 | RFE(재귀적 feature 제거), 10개 |
| 모델 | RBF Kernel Ridge (alpha=0.0018, gamma=0.0013) |
| 전처리 | QuantileTransformer(normal) |
| 클래스 불균형 처리 | SMOTE(소수 클래스 합성) |

### 4.2 CN/MCI 분리 모델

| 항목 | 값 |
|---|---|
| 대상 | CN vs MCI (DEM 환자는 학습에서 제외) |
| feature 후보 풀 | 미리 정의된 "compact" 15개 도메인 feature 중 |
| feature 선택 | ANOVA(분산분석), 10개 |
| 모델 | LDA (Linear Discriminant Analysis, shrinkage=0.6) |
| 전처리 | signedlog 변환(부호 유지 log) 후 RobustScaler |
| 클래스 불균형 처리 | SMOTE |

### 4.3 SMOTE 사용 근거

두 branch 모두 SMOTE를 쓴다. 이는 원본 champion의 검색 결과를 그대로 이어받은 값이라, 이번에 별도로 augmentation 없음(class weight만) / augmentation·weight 둘 다 없음과 비교 검증했다(Validation 33명 기준).

| augmentation 설정 | AUC | Macro Recall | Macro F1 | Accuracy |
|---|---:|---:|---:|---:|
| **SMOTE (채택)** | 0.814 | 0.850 | **0.802** | **0.879** |
| augmentation 없음 + class weight | 0.807 | 0.774 | 0.678 | 0.697 |
| augmentation·weight 모두 없음 | 0.875 | 0.774 | 0.643 | 0.697 |

순수 ranking 능력(AUC)만 보면 augmentation을 아예 안 쓴 쪽이 오히려 높지만(0.875), 최종 class 결정(F1·Accuracy) 기준으로는 SMOTE가 뚜렷하게 낫다. 다만 이 비교도 Validation 33명 기준이라 6절의 "Selection-Validation" 한계가 동일하게 적용된다.

### 4.4 최종 결정

두 모델의 원점수(score)를 위 곱셈식으로 합친 뒤, 3개 클래스에 대한 log-score offset(각 클래스별 상수 보정값)을 곱해 최종 argmax로 클래스를 결정한다. offset은 정답이 보이는 상태에서 recall/F1을 최적화하도록 grid search로 고른 값이며(2절 참고), 순수 ranking 지표인 AUC는 offset 적용 전 점수로 계산해 offset의 영향을 받지 않는다.

---

## 5. 사용 feature (20개)

### 5.1 DEM 분리 모델 (activity 10개)

| feature | 의미 |
|---|---|
| `state_code_3_longest_minutes__maximum` | 저강도 활동 최대 연속 지속시간 |
| `state_transition_rate__rolling_range7` | 활동상태 전환율의 7일 변동폭 |
| `state_code_3_longest_minutes__rolling_range7` | 저강도 활동 최대지속시간의 7일 변동폭 |
| `rest__range` | 휴식시간의 기간 내 범위 |
| `met_p10__maximum` | 저활동 MET(하위10%)의 개인 내 최댓값 |
| `met_cosinor_phase_cos__q05` | 일주기 리듬 위상의 5%분위 |
| `met_cosinor_phase_cos__bin_variance4` | 일주기 리듬 위상의 4구간 분산 |
| `met_day_night_difference__kurtosis` | 주야간 MET 차이의 첨도 |
| `met_p10__p90`, `met_p10__q95` | 저활동 MET 분위값 (RFE로 선택되었으나 실제 SHAP 기여도는 0 — 아래 7.3 참고) |

### 5.2 CN/MCI 분리 모델 (수면/심박 중심 10개)

| feature | 의미 |
|---|---|
| `sleep__bedtime_regularity` | 취침시각 규칙성 |
| `paper_hr_drop_ratio__std` | 야간 HR-drop 비율의 일간 표준편차 |
| `paper_hr_drop_ratio__rolling_cv7` | 야간 HR-drop 비율의 7일 변동계수 |
| `steps__std` | 걸음수의 일간 표준편차 |
| `low__std` | 저강도 활동시간의 일간 표준편차 |
| `inactive__std` | 비활동시간의 일간 표준편차 |
| `hourly_profile_stability` | 시간대별 활동 프로파일 안정성 |
| `rmssd__cv` | 심박변이도(RMSSD)의 변동계수 |
| `paper_bedtime_noon_unwrapped__std` | 취침시각(정오 기준)의 일간 표준편차 |
| `paper_wake_hour__std` | 기상시각의 일간 표준편차 |

---

## 6. 성능

### 6.1 Selection-Validation 성능 (33명, 고정 split, offset은 이 33명 정답으로 튜닝)

| Metric | 값 |
|---|---:|
| Macro ROC-AUC | 0.814 |
| Macro Recall | 0.850 |
| Macro F1 | 0.802 |
| Accuracy | 0.879 |
| CN Recall (n=26) | 0.885 |
| MCI Recall (n=4) | 1.000 |
| DEM Recall (n=3) | 0.667 |

**주의**: 이 숫자는 원본 champion보다는 훨씬 적은 탐색(4~5개 후보 중 선택 + offset 1회 grid search)으로 얻었지만, 여전히 이 33명의 정답을 일부 사용한 **Selection-Validation 성능**이며 독립적인 일반화 성능이 아니다.

### 6.2 (참고) Honest Nested CV 성능 — 174명 전체 프레시 재분할, outer 5-fold × inner 3-fold

| Metric | 값 |
|---|---:|
| Macro ROC-AUC | 0.569 |
| Macro Recall | 0.433 |
| Macro F1 | 0.336 |
| Accuracy | 0.368 |
| CN Recall (n=111) | 0.34 |
| MCI Recall (n=51) | 0.37 |
| DEM Recall (n=12) | 0.58 |

이 수치는 174명을 완전히 새로 나누고, feature 선택/offset 튜닝을 outer test와 무관한 inner 데이터로만 수행해 얻은 결과다. V2가 이미 확인했던 정직한 추정치(Macro AUC 0.543)와 같은 수준이며, **이 데이터·feature 조합에서 실제로 기대할 수 있는 성능의 현실적인 하한선**으로 본다.

### 6.3 신뢰구간에 대한 참고

bootstrap 95% CI(subject-level, class-stratified, 2,000회)를 계산했으나, MCI(n=4)·DEM(n=3) recall의 구간이 [1.0, 1.0], [0.0, 1.0]처럼 표본 크기 때문에 통계적으로 해석 불가능한 값으로 나와 핵심 근거로 사용하지 않기로 했다. Macro 수준 지표(AUC 0.680~0.973)도 "이 33명을 다시 뽑았을 때의 흔들림"만 반영하고 "완전히 새로운 사람에게 얼마나 통하는지"는 반영하지 못하므로, 6.2의 nested CV 결과를 함께 봐야 온전한 그림이 된다.

---

## 7. SHAP 기반 feature 기여도 및 전환특성 (CN → MCI → DEM)

최종 모델(2-model, 20개 feature) 고유의 SHAP(permutation explainer, Background=Training 80명 샘플)과, 174명 전체 기준 CN/MCI/DEM 그룹별 원값(raw value) 중앙값을 계산했다.

### 7.1 종합 기여도 (macro mean|SHAP|, 상위 12개)

| feature | branch | macro 기여도 | CN/MCI/DEM 원값 추세 |
|---|---|---:|---|
| `state_code_3_longest_minutes__maximum` | DEM | 0.392 | 단조감소 (130→120→75) |
| `state_transition_rate__rolling_range7` | DEM | 0.309 | U자형 (MCI 최고: 0.092) |
| `rest__range` | DEM | 0.126 | U자형 (MCI 최저: 508) |
| `steps__std` | CN/MCI | 0.094 | U자형 (MCI 최고: 3863) |
| `met_cosinor_phase_cos__bin_variance4` | DEM | 0.070 | U자형 (MCI 최저) |
| `met_day_night_difference__kurtosis` | DEM | 0.065 | U자형 (MCI 최저: 0.027) |
| `met_cosinor_phase_cos__q05` | DEM | 0.059 | U자형 (MCI 최저) |
| `state_code_3_longest_minutes__rolling_range7` | DEM | 0.058 | U자형 (MCI 최고: 50.0) |
| `paper_hr_drop_ratio__rolling_cv7` | CN/MCI | 0.048 | U자형 (MCI 최고: 0.458) |
| `hourly_profile_stability` | CN/MCI | 0.045 | 단조증가 (0.391→0.402→0.403) |
| `low__std` | CN/MCI | 0.041 | 단조감소 (83.4→76.4→45.2) |
| `sleep__bedtime_regularity` | CN/MCI | 0.040 | U자형 (MCI 최고: 0.946) |

### 7.2 해석

- **DEM 분리는 소수(상위 3개)의 activity feature가 거의 전담**한다: 저강도 활동 최대지속시간(단조감소 — 진행될수록 짧아짐), 활동전환율의 주간 변동폭(U자형), 휴식시간 범위(U자형)만으로 전체 기여도의 약 55%를 차지한다.
- **CN/MCI 분리는 여러 수면·활동변동성 feature가 고르게 기여**한다 — 특정 한두 개가 압도하지 않고 걸음수 변동성, HR-drop 변동성, 취침규칙성, 활동프로파일 안정성 등이 비슷한 비중으로 나눠 기여한다.
- **U자형(MCI가 극단값) 패턴이 다수(20개 중 12개)**를 차지한다 — MCI가 "정상과 치매 사이 중간 단계"가 아니라, 다수의 지표에서 **CN·DEM보다 오히려 더 규칙적이거나 더 변동성이 크게** 나타난다. 이는 MCI 단계 특유의 보상 행동 또는 과도기적 불안정성을 반영할 가능성이 있으나, 임상적으로 검증된 기전은 아니다.

### 7.3 주의할 점 — 선택됐지만 기여하지 않는 feature

DEM 모델의 `met_p10__p90`, `met_p10__q95` 두 feature는 RFE가 후보 10개 안에 포함시켰지만 **실제 SHAP 기여도가 정확히 0**이다. 이미 포함된 `met_p10__maximum`과 정보가 겹쳐(비슷한 분포 계열) RBF Ridge 모델이 실질적으로 사용하지 않는 것으로 보인다. 앞서 확인한 "완전 중복 feature 하나가 모델을 흔든다"는 문제와 같은 계열의 현상이며, 향후 feature 목록을 더 정리할 때 우선 제거 후보다.

---

## 8. 한계 및 주의사항 (반드시 함께 보고할 것)

1. **이 README의 6.1 성능(Selection-Validation)은 일반화 성능이 아니다.** feature 선택 방식과 최종 결정 offset이 33명 Validation 정답을 일부 참고해 정해졌다. 원본 champion(4,258회 탐색)보다는 훨씬 적게 참고했지만 0은 아니다.
2. **6.2의 honest nested CV(Macro AUC 0.569)가 이 데이터에서 기대할 수 있는 더 현실적인 성능**이며, 두 숫자를 항상 함께 제시해야 한다.
3. **MCI recall은 구성에 따라 극단적으로 흔들린다** (nested CV 기준 0.08~0.37, 구성별로 큰 차이). 반면 **DEM recall(activity 기반)은 0.58~0.67로 상대적으로 안정적**이다 — DEM 관련 feature/결론은 상대적으로, MCI 관련 결론은 탐색적/가설 수준으로 취급해야 한다.
4. Validation의 MCI(4명)·DEM(3명)은 통계적으로 결론을 내리기에 너무 작다. 신뢰구간이 그 사실을 그대로 보여준다(7.3 참고할 것 없이, 6.3 참고).
5. 20개 feature 중 2개(`met_p10__p90`, `met_p10__q95`)는 선택됐으나 실제 기여도가 0이라 사실상 18개 feature 모델에 가깝다.
6. 본 모델은 MMSE를 사용하지 않는 웨어러블 단독 스크리닝을 목표로 하며, 독립된 외부 코호트에서 검증된 적이 없다. 임상적 진단 도구가 아니라 **feature/구조 기여도를 보여주는 탐색적 분석 결과**로 취급해야 한다.

---

## 9. 재현 정보

- **실행 코드**: 이 폴더의 [`final_model.py`](./final_model.py) — 이 파일을 그대로 실행하면 6.1의 Selection-Validation 수치가 재현된다.
- **의존성**: `final_model.py`는 [`Pig30nidaE/my_lab`](https://github.com/Pig30nidaE/my_lab) 저장소의 `lifelog_v2`/`lifelog_v3` 패키지를 그대로 import한다. 독립 실행 코드가 아니므로, `my_lab`을 clone한 뒤 그 루트에 이 파일을 두고 실행해야 한다:
  ```bash
  git clone https://github.com/Pig30nidaE/my_lab.git
  cp final_model.py my_lab/final_model.py
  cd my_lab
  python final_model.py --data-root /path/to/aihub_data
  ```
- 원본 데이터: AI-Hub 치매 고위험군 웨어러블 라이프로그, 로컬 `1.Training`/`2.Validation` 원본 CSV
- DEM 모델 spec: `task=dem, family=rbf_ridge, view=activity, top_k=10, selection=rfe, transform=quantile, augmentation=smote, alpha=0.0018005071512845635, gamma=0.0013102849275700784, seed=2026`
- CN/MCI 모델 spec: `task=cn_mci, family=lda, view=compact, top_k=10, selection=anova, transform=signedlog, augmentation=smote, shrinkage=0.6, seed=2026`
- 결합식: `CN=(1-d)(1-q), MCI=(1-d)q, DEM=d`, 이후 class offset 적용 argmax
- SHAP: `shap.PermutationExplainer`, Background=Training 80명 샘플, seed=2026
- 이번 실험 전체(중복 제거, branch 전문화, nested CV, bootstrap CI, SMOTE 비교 등)의 상세 스크립트는 `Pig30nidaE/my_lab` 기반으로 작업했으며, 이 폴더에는 최종 채택된 모델 코드와 결과만 남긴다.
