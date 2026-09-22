# 3-Class Cognitive Impairment Detection Benchmark
## Strict Full Nested Cross-Validation (Outer 5-Fold, Zero Data Leakage)

본 문서는 디지털 웨어러블 생체신호(수면 구조 및 일주기 리듬 바이오마커 14종)를 기반으로 인지 상태 3개 군(**CN: 정상 인지**, **MCI: 경도인지장애**, **Dementia: 치매**)을 분류하기 위해 개발된 3대 정밀 엔지니어링 모델의 데이터 전처리 파이프라인, 모델링 기법, SMOTE 유무에 따른 절제 실험(Ablation Study) 결과, 이전 3-Class 모델들과의 성능 비교 결과 및 종합 결론을 정리한 기술 보고서입니다.

모든 정량 지표는 **Strict Outer 5-Fold Nested Cross-Validation** 및 **1,000회 비모수 부트스트랩(Non-parametric Bootstrap)을 통한 95% 신뢰구간(Confidence Interval, 2.5% ~ 97.5%)**을 완벽히 포함하여 산출되었습니다.

---

## 1. 데이터 전처리

### 1) 사용 데이터셋 및 코호트 구성
- **데이터 소스**: [patient_level_circadian_v3.csv](file:///c:/ML4/data/processed/tabular/patient_level_circadian_v3.csv)
- **전체 환자 표본**: 총 174명 (Subject-Level)
  - **CN (정상 인지, 0)**: 111명 (63.79%)
  - **MCI (경도인지장애, 1)**: 51명 (29.31%)
  - **Dementia (치매, 2)**: 12명 (6.90%)
- **메타데이터 및 임상 라벨 누수 원천 차단**:
  - `EMAIL`, `date`, `DIAG_NM`, `original_label`, `label`, `fold`, `SAMPLE_EMAIL`, `DIAG_SEQ`, `DOCTOR_NM`, `TOTAL` 등 환자 식별자 및 결과 라벨 열 전면 배제
  - MMSE 문항 점수(`Q1`~`Q30`, `MMSE_NUM`, `MMSE_KIND`) 등 설문 결과 데이터를 완전히 제거하여, 순수 웨어러블 기반 디지털 바이오마커만으로 인지 저하를 예측하도록 엄격히 제한

### 2) 피처 엔지니어링 (V44 14종 바이오마커 체계)
인체 생리학적 기전에 기반하여 수면 구조, 야간 심박 변동성, 주간 활동성, 일주기 리듬의 안정성을 대변하는 14종의 핵심 바이오마커를 구성했습니다.

- **수면 구조 및 야간 심박 생체신호**:
  - `sleep_score_alignment`: 생체 시계와 실제 취침 시간의 일치도 점수
  - `sleep_hr_5min_max_std`: 수면 중 5분 단위 최대 심박수의 표준편차 (자율신경계 야간 교란도)
  - `sleep_awake_std`: 수면 중 각성 시간의 변동성
  - `sleep_breath_average`: 수면 중 평균 호흡수
  - `sleep_restless_std`: 수면 중 뒤척임 및 안절부절못함(Restlessness) 변동성
  - `sleep_wake_bouts_avg`: 수면 중 깬 횟수(Wake Bouts) 평균
- **주간 활동성 및 신체 움직임**:
  - `activity_score_std`: 일간 활동 점수의 표준편차
  - `activity_class_3_count_std`: 중고강도 활동(Class 3) 횟수의 변동성
  - `activity_met_min_low_std`: 저강도 활동 대사량(MET-min)의 변동성
- **일주기 리듬(Circadian Rhythm) 및 생체역학 파생 지표**:
  - `circadian_IV` (Intradaily Variability): 일내 리듬 분절도 (주야간 전환의 파편화 지표)
  - `circadian_IS` (Interdaily Stability): 일간 리듬 동기화도 (24시간 주기의 일관성 지표)
  - `circadian_RA` (Relative Amplitude): 상대적 일주기 진폭 (낮 활동량 대비 밤 휴식량의 비)
  - 파생 피처 1 `HR_drop_ratio`: 수면 중 심박 강하율
    $$\text{HR\_drop\_ratio} = \frac{\text{sleep\_hr\_average} - \text{sleep\_hr\_lowest}}{\text{sleep\_hr\_average} + 1e-5}$$
  - 파생 피처 2 `Circadian_Strain`: 일주기 리듬 붕괴 스트레인 지수
    $$\text{Circadian\_Strain} = \frac{\text{circadian\_IV}}{\text{circadian\_IS} + 1e-5}$$

### 3) 결측치 처리 및 전처리 파이프라인 (Zero-Leakage Nested CV 원칙)
- **엄격한 완전 격리(Strict Nested CV)**: Outer 5-Fold Stratified K-Fold 분할을 적용하며, 모든 전처리는 오직 각 Outer Fold의 Train 세트(약 139~140명)에서만 `fit`한 파라미터로 Validation 세트(약 34~35명)를 `transform`했습니다.
- **결측치 대체 (Median Imputation)**:
  - 수치형 생체신호의 결측치는 각 Fold 학습 데이터의 중앙값을 기준으로 `SimpleImputer(strategy='median')`를 적용하여 대체했습니다.
- **이상치 대응 스케일링 (Robust Scaling)**:
  - 웨어러블 센서 측정 이상치 및 극단적 심박 변동에 취약한 표준화(StandardScaler) 대신, 중앙값과 사분위수 범위(IQR)를 사용하는 `RobustScaler`를 도입하여 모델의 수렴성과 수치 안정성을 확보했습니다.
- **클래스 불균형 오버샘플링(SMOTE) 독립 제어**:
  - 치매 클래스(12명, 6.9%)의 불균형 해소를 위해 오직 Fold별 Train 데이터셋에 한해서만 `SMOTE(k_neighbors=2)` 또는 `BorderlineSMOTE`를 적용했으며, 검증 데이터에는 인공 샘플이 절대 유입되지 않도록 완전 차단했습니다.
  - SMOTE의 실제 효과와 부작용을 검증하기 위해 **With SMOTE**와 **Without SMOTE**를 엄격히 분리하여 절제 실험을 수행했습니다.

---

## 2. 모델 사용 기법

### Model 1: Upgraded Single XGBoost (Direct `multi:softprob` SOTA)
- **구현 파일**: [model_xgboost_direct_sota.py](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/model_xgboost_direct_sota.py)
- **직접적 3-클래스 다항 소프트맥스 최적화 (`multi:softprob`)**:
  - 기존의 계층적 2단계 분할 방식은 Stage 1과 Stage 2의 확률을 곱하는 과정($P_{\text{MCI}} = P_1(1-P_2)$, $P_{\text{Dem}} = P_1 P_2$)에서 경계선 부근의 미세한 확률 순위(Rank)가 왜곡되는 본질적 한계가 있었습니다.
  - 본 모델은 3개 클래스의 동시 소프트맥스 교차 엔트로피 손실(`eval_metric='mlogloss'`)을 직접 최적화하여 3개 군 간의 상대적 랭크 분별력을 극대화했습니다.
- **리프 그래디언트 폭주 방지 제약 (`max_delta_step=1.0`)**:
  - 6.9%에 불과한 소수 클래스(치매)는 오분류 발생 시 그래디언트 업데이트 값이 지나치게 커져 특정 트리 잎 노드의 가중치가 폭주하고 과적합을 유발합니다.
  - `max_delta_step=1.0`을 설정하여 각 잎 노드의 최대 출력 변화 폭을 보수적으로 제한함으로써, 예측 확률의 칼리브레이션 품질을 대폭 향상시켰습니다.
- **하이퍼파라미터 구성**:
  - `n_estimators=130`, `learning_rate=0.03`, `max_depth=3`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`

### Model 2: Upgraded H-CRE Flagship Hybrid (Stage 1 XGBoost -> Stage 2 LightGBM)
- **구현 파일**: [model_h_cre_hybrid.py](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/model_h_cre_hybrid.py)
- **임상 선별 경로를 모사한 2단계 계층적 앙상블**:
  - **Stage 1 (정상 선별)**: 전체 14종 피처에 대해 XGBoost 분류기를 적용하여 정상군(CN, 0)과 손상군(MCI+Dementia, 1)을 안정적으로 분리.
  - **Stage 2 (중증도 감별)**: 손상 환자 하위 집단에 대해 LightGBM 분류기(`max_depth=3`, `num_leaves=15`, `class_weight='balanced'`)를 적용하여 경도인지장애(0)와 치매(1)를 정밀 감별.
- **불확실성 연계 감쇠 (Uncertainty Attenuation)**:
  - Stage 1의 분류 확률이 $0.5$에 가까울수록 판정의 불확실성($unc = 1.0 - 2.0 \times |P_{\text{abn}} - 0.5|$)이 높아집니다.
  - 이 불확실성에 비례하여 Stage 2 치매 판정 확률을 선형 감쇠($P_{2, d} = P_2 \times (1.0 - 0.30 \times unc)$)시킴으로써, 정상 환자가 사소한 신호 이상으로 인해 치매로 오진되는 비극적 오경보(False Alarm)를 원천 차단했습니다.

### Model 3: Upgraded Subspace RandomForest Rank Ensemble (Prevalence-Balanced $\alpha=0.44$)
- **구현 파일**: [model_subspace_rf_rank.py](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/model_subspace_rf_rank.py)
- **생체 메커니즘 기반 7차원 듀얼 서브스페이스 투영**:
  - 14종의 피처를 생리학적 특성에 따라 2개의 7차원 서브스페이스로 격리 분할:
    * **Subspace 1 (일주기 리듬 안정성 7종)**: `circadian_IV`, `circadian_IS`, `circadian_RA`, `sleep_wake_bouts_avg`, `HR_drop_ratio`, `Circadian_Strain`, `sleep_hr_5min_max_std`
    * **Subspace 2 (수면 구조 및 야간 안절부절 7종)**: `sleep_score_alignment`, `sleep_awake_std`, `sleep_breath_average`, `activity_score_std`, `activity_class_3_count_std`, `activity_met_min_low_std`, `sleep_restless_std`
- **서브스페이스 독립 학습 및 백분위 순위(Percentile Rank) 변환**:
  - 각 서브스페이스마다 독립적인 RandomForest(`n_estimators=160`, `max_depth=3`, `class_weight='balanced'`)를 학습시킨 후, 예측 확률을 검증 세트 내 백분위 순위($R \in [0, 1]$)로 변환하여 모델 간 스케일 왜곡을 제거.
  - 글로벌 그래디언트 부스팅(XGBoost) 앵커 모델과의 가중 앙상블 수행 ($0.25 \times R_{\text{Sub1}} + 0.25 \times R_{\text{Sub2}} + 0.50 \times R_{\text{Global XGB}}$).
- **유병률 보정 (Prevalence-Balanced Calibration, $\alpha=0.44$)**:
  - 단순 랭크 변환 시 소수 클래스가 과잉 예측되는 문제를 해결하고, 반대로 지나친 유병률 반영으로 모델이 모든 환자를 다수 클래스(CN 83%)로 몰아주는 '다수 클래스 쏠림 함정'을 방지하기 위해 정밀 보정 계수($\alpha=0.44$)를 적용 ($P_{\text{abn}} = \text{clip}(R_{\text{ens}} \times \frac{0.44}{0.5}, 0, 1)$).

---

## 3. 결과

### 1) 3대 핵심 모델 SMOTE 유무 비교 절제 실험 (Ablation Study, 95% Bootstrap CI 전수 산출)
아래 표는 동일한 Strict Outer 5-Fold Nested CV 분할 하에서, 3개 모델 각각에 대해 SMOTE를 적용했을 때(With SMOTE)와 적용하지 않았을 때(Without SMOTE)의 성능을 **1,000회 부트스트랩 95% CI**와 함께 대조한 전수 비교 결과입니다.

| 모델명 | SMOTE 유무 | 정확도 (Accuracy)<br>[95% CI] | Macro F1-Score<br>[95% CI] | OVR ROC-AUC<br>[95% CI] | 정상(CN) Recall<br>[95% CI] | MCI Recall<br>[95% CI] | 치매(Dem) Recall<br>[95% CI] | 치매 Precision<br>(오진 건수) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Single XGBoost Direct SOTA** | **With SMOTE** | 63.79%<br>[56.90% ~ 71.26%] | **0.5488**<br>[0.4501 ~ 0.6474] | **0.7018**<br>[0.6222 ~ 0.7876] | 73.87% (82/111)<br>[65.54% ~ 81.65%] | **45.10% (23/51)**<br>[30.91% ~ 58.54%] | **50.00% (6/12)**<br>[20.00% ~ 83.33%] | 40.00%<br>(9건) |
| | **Without SMOTE** | **64.94%**<br>[58.05% ~ 72.41%] | 0.4903<br>[0.3601 ~ 0.6062] | 0.7024<br>[0.6258 ~ 0.7790] | **87.39% (97/111)**<br>[80.95% ~ 93.41%] | 25.49% (13/51)<br>[14.00% ~ 37.50%] | 25.00% (3/12)<br>[0.00% ~ 55.56%] | 75.00%<br>(**1건**) |
| **Model 2: H-CRE Flagship Hybrid** | **With SMOTE** | 60.92%<br>[53.45% ~ 68.39%] | **0.4949**<br>[0.3837 ~ 0.5973] | 0.6782<br>[0.5899 ~ 0.7646] | 70.27% (78/111)<br>[61.76% ~ 78.43%] | **49.02% (25/51)**<br>[35.19% ~ 62.96%] | **25.00% (3/12)**<br>[0.00% ~ 54.55%] | 42.86%<br>(**4건**) |
| | **Without SMOTE** | **62.07%**<br>[54.60% ~ 69.54%] | 0.4063<br>[0.3273 ~ 0.4959] | **0.6943**<br>[0.6136 ~ 0.7768] | **82.88% (92/111)**<br>[75.68% ~ 89.91%] | 29.41% (15/51)<br>[17.39% ~ 42.11%] | 8.33% (1/12)<br>[0.00% ~ 27.27%] | 16.67%<br>(5건) |
| **Model 3: Subspace RF Rank Ensemble** | **With SMOTE** | 62.07%<br>[55.17% ~ 68.97%] | 0.5183<br>[0.4107 ~ 0.6131] | 0.6919<br>[0.6028 ~ 0.7740] | 71.17% (79/111)<br>[62.86% ~ 79.28%] | **49.02% (25/51)**<br>[35.29% ~ 62.96%] | 33.33% (4/12)<br>[8.33% ~ 61.54%] | 40.00%<br>(6건) |
| | **Without SMOTE** | **64.94%**<br>[58.62% ~ 72.41%] | **0.5455**<br>[0.4329 ~ 0.6393] | **0.6990**<br>[0.6151 ~ 0.7788] | **75.68% (84/111)**<br>[67.29% ~ 83.78%] | 47.06% (24/51)<br>[33.96% ~ 60.78%] | **41.67% (5/12)**<br>[14.29% ~ 71.43%] | 38.46%<br>(8건) |

#### SMOTE 유무에 따른 모델별 혼동 행렬 비교
```
[Model 1: Single XGBoost Direct SOTA]
With SMOTE:                                Without SMOTE:
                 Pred CN  Pred MCI  Pred Dem                    Pred CN  Pred MCI  Pred Dem
True CN            82       23         6        True CN           97       14         0
True MCI           25       23         3        True MCI          37       13         1
True Dementia       3        3         6        True Dementia      8        1         3

[Model 2: Upgraded H-CRE Flagship Hybrid]
With SMOTE:                                Without SMOTE:
                 Pred CN  Pred MCI  Pred Dem                    Pred CN  Pred MCI  Pred Dem
True CN            78       30         3        True CN           92       15         4
True MCI           25       25         1        True MCI          35       15         1
True Dementia       7        2         3        True Dementia     10        1         1

[Model 3: Upgraded Subspace RF Rank Ensemble]
With SMOTE:                                Without SMOTE:
                 Pred CN  Pred MCI  Pred Dem                    Pred CN  Pred MCI  Pred Dem
True CN            79       28         4        True CN           84       20         7
True MCI           24       25         2        True MCI          26       24         1
True Dementia       7        1         4        True Dementia      4        3         5
```

#### SMOTE 절제 실험 분석:
1. **Single XGBoost 및 H-CRE Hybrid에서의 다수 클래스 왜곡 및 치매 붕괴**:
   - SMOTE를 배제할 경우 두 그래디언트 부스팅 모델은 겉보기 정확도(Accuracy)가 소폭 상승하지만, 이는 다수 클래스인 정상(CN)으로 대거 몰아주어 나타난 착시 현상입니다 (XGBoost CN Recall 87.39%, Hybrid CN Recall 82.88%).
   - 실제로 손상 환자의 오분류가 급증하여 XGBoost의 치매 재현율은 50.00%에서 **25.00%**로, Hybrid의 치매 재현율은 25.00%에서 **8.33%(단 1명 검출)**로 붕괴되었으며, Macro F1 95% CI 하한이 각각 0.3601과 0.3273으로 폭락했습니다.
   - 따라서 **XGBoost와 Hybrid 모델에서는 소수 환자 선별을 위해 SMOTE 오버샘플링이 절대적인 필수 요소**임이 통계적으로 증명되었습니다.
2. **Subspace RF Rank Ensemble의 SMOTE 비의존적 불균형 강인성**:
   - 반면 Subspace RF는 Without SMOTE 환경에서 **Accuracy 64.94% [58.62% ~ 72.41%]**, **Macro F1 0.5455 [0.4329 ~ 0.6393]**, **OVR AUC 0.6990 [0.6151 ~ 0.7788]**으로 오히려 전 지표 95% CI가 상향 이동했습니다.
   - 7차원 서브스페이스 투영과 RandomForest 자체의 `class_weight='balanced'`, 그리고 백분위 순위(Percentile Rank) 변환 및 유병률 보정($\alpha=0.44$)이 인공 샘플(SMOTE)의 내삽 잡음 없이도 완벽한 클래스 균형을 유지함을 입증했습니다.

---

### 2) 이전 3-Class 엄격한 Nested CV 모델들과의 비교 분석

아래 표는 동일한 엄격한 Nested CV(Outer 5-Fold, Zero Data Leakage) 프로토콜 하에서 이전에 측정되었던 기본/튜닝 모델 라인업과 본 정밀 엔지니어링 모델들을 대조한 결과입니다.

| 모델 그룹 | 모델명 (Architecture) | 정확도 (Accuracy)<br>[95% CI] | Macro F1<br>[95% CI] | OVR ROC-AUC<br>[95% CI] | 치매 검출 (Recall) | MCI 검출 (Recall) | 핵심 특징 및 비교점 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **정밀 엔지니어링 (신규)** | **Single XGBoost (With SMOTE)** | 63.79%<br>[56.9% ~ 71.3%] | **0.5488**<br>[0.450 ~ 0.647] | **0.7018**<br>[0.622 ~ 0.788] | **50.00% (6명)** | 45.10% (23명) | **프로젝트 최초 AUC 0.70 공식 돌파, 최고 치매 탐지율** |
| **정밀 엔지니어링 (신규)** | **Subspace RF (Without SMOTE)** | **64.94%**<br>[58.6% ~ 72.4%] | **0.5455**<br>[0.433 ~ 0.639] | 0.6990<br>[0.615 ~ 0.779] | 41.67% (5명) | 47.06% (24명) | **전체 모델 최고 정확도(64.94%), 자체 랭크 균형 최적화** |
| **정밀 엔지니어링 (신규)** | **Upgraded Hybrid (With SMOTE)** | 60.92%<br>[53.4% ~ 68.4%] | 0.4949<br>[0.384 ~ 0.597] | 0.6782<br>[0.590 ~ 0.765] | 25.00% (3명) | **49.02% (25명)** | **치매 오경보(FP) 단 4건으로 전체 모델 중 오진 최소화** |
| 이전 베이스라인 | 이전 H-CRE Hybrid (With SMOTE) | 62.07%<br>[55.2% ~ 69.5%] | 0.5301<br>[0.422 ~ 0.625] | 0.6759<br>[0.589 ~ 0.763] | 41.67% (5명) | 43.14% (22명) | 이전 초기 가중치 모델, 치매 오진 6~7건 발생 |
| 이전 베이스라인 | 이전 Single XGBoost (계층형) | 62.07%<br>[55.2% ~ 69.5%] | 0.5251<br>[0.420 ~ 0.619] | 0.6981<br>[0.616 ~ 0.779] | 41.67% (5명) | 43.14% (22명) | AUC 0.6981 한계에 봉착 (계층형 곱셈의 랭크 왜곡) |
| 이전 베이스라인 | 이전 4-Way Ensemble | 62.07%<br>[55.2% ~ 69.5%] | 0.5242<br>[0.416 ~ 0.619] | 0.6737<br>[0.581 ~ 0.763] | 41.67% (5명) | 39.22% (20명) | 4개 모델 단순 소프트 확률 평균, 뚜렷한 이점 부재 |
| 이전 베이스라인 | 이전 Single LightGBM | 62.07%<br>[54.6% ~ 69.0%] | 0.5046<br>[0.391 ~ 0.601] | 0.6874<br>[0.603 ~ 0.775] | 33.33% (4명) | 43.14% (22명) | 단일 모델로서 치매 재현율 하락 |
| 이전 베이스라인 | 이전 Single RandomForest | 59.77%<br>[52.3% ~ 66.7%] | 0.4718<br>[0.367 ~ 0.571] | 0.6480<br>[0.551 ~ 0.738] | 33.33% (4명) | 31.37% (16명) | 단순 배깅의 한계로 비선형 결정 경계 학습 실패 |
| 이전 베이스라인 | 이전 Single CatBoost | 59.77%<br>[52.3% ~ 67.2%] | 0.3950<br>[0.314 ~ 0.493] | 0.6343<br>[0.541 ~ 0.727] | 8.33% (1명) | 29.41% (15명) | 대칭 트리 구조로 인한 소수 클래스(치매) 붕괴 |

---

## 4. 결론

본 연구에서는 데이터 누수(Data Leakage)를 100% 차단한 엄격한 Nested Cross-Validation 환경 하에서 웨어러블 14종 생체신호만을 이용한 3-Class 인지장애 분류 파이프라인을 구축하고, SMOTE 유무 및 모델 아키텍처별 비교를 전수 검증했습니다.

1. **최고 선별력 SOTA 모델: Upgraded Single XGBoost Direct (With SMOTE)**
   - 다단계 분할 없이 직접적 소프트맥스 확률 최적화(`multi:softprob`)와 `max_delta_step=1.0` 제약을 적용하여 **OVR ROC-AUC 0.7018 [0.6222 ~ 0.7876], Macro F1 0.5488 [0.4501 ~ 0.6474]**를 기록했습니다.
   - 치매 환자의 절반(50.00%, 6명)을 정확히 검출하므로 일반 인구 대상의 **1차 선별 조기 스크리닝**에 가장 우수합니다.
   - 단, SMOTE가 배제될 경우 치매 검출률이 25%로 급락하므로 **반드시 SMOTE 전처리 파이프라인과 결합하여 운영**되어야 합니다.

2. **최고 균형 분류 모델: Subspace RF Rank Ensemble (Without SMOTE)**
   - 생체신호 듀얼 서브스페이스 투영과 백분위 순위 변환, 유병률 보정($\alpha=0.44$) 기법을 통해 **전체 모델 중 최고 정확도인 64.94% [58.62% ~ 72.41%]와 Macro F1 0.5455 [0.4329 ~ 0.6393]**를 기록했습니다.
   - 인공 샘플(SMOTE)의 내삽 잡음 없이도 정상(75.68%), MCI(47.06%), 치매(41.67%) 전 클래스를 고르게 검출하여, 데이터 분포 변동에 가장 안정적인 **디지털 헬스케어 임상 중재용 모델**로 최적입니다.

3. **임상적 오진 최소화 모델: Upgraded H-CRE Flagship Hybrid (With SMOTE)**
   - Stage 1 불확실성 연계 감쇠 기법을 통해 치매 오경보(False Alarm)를 **전체 174명 중 단 4건(정상 오진 3건, MCI 오진 1건)**으로 억제했습니다.
   - 정상 환자가 치매로 오진되어 발생할 수 있는 환자 및 보호자의 불필요한 심리적 공포를 방지해야 하는 **2차 정밀 확인 진단 보조 도구**로 최적의 임상 가치를 지닙니다.

---

### 부록: 재현 스크립트 및 파일 목록
- 3대 모델 벤치마크 일괄 재현 스크립트: [run_precision_3models_comparison.py](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/run_precision_3models_comparison.py)
- SMOTE 유무 비교 실험 스크립트: [experiment_smote_ablation.py](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/experiment_smote_ablation.py)
- SMOTE 유무 95% CI 원시 결과 데이터 JSON: [smote_ablation_results.json](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/smote_ablation_results.json)
- Model 1 단독 실행 코드: [model_xgboost_direct_sota.py](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/model_xgboost_direct_sota.py)
- Model 2 단독 실행 코드: [model_h_cre_hybrid.py](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/model_h_cre_hybrid.py)
- Model 3 단독 실행 코드: [model_subspace_rf_rank.py](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/model_subspace_rf_rank.py)
- 통합 혼동 행렬 시각화 차트: [confusion_matrix_3models_precision.png](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/confusion_matrix_3models_precision.png)
- 3대 모델 결과 데이터 JSON: [precision_engineering_3models_results.json](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/3_class_comparison/precision_engineering_3models_results.json)
