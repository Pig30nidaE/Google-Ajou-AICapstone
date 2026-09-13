# V44 Circadian Subspace SOTA 모델 상세 기술 보고서

## 1. 개요 및 연구 목적

본 프로젝트는 일주기 생체 리듬(Circadian Rhythm), 수면 구조(Sleep Architecture), 주간 활동(Activity) 등의 웨어러블 라이프로그 데이터를 활용하여 인지기능 저하(경도인지장애 및 치매)를 조기 선별하는 AI 분류 파이프라인 개발을 목표로 합니다.

- **분류 태스크**: 정상 노인(CN, Class 0) vs. 인지저하 환자군(MCI + Dementia, Class 1) 이진 분류
- **핵심 목표**: 설문/인지검사 점수(MMSE)를 완전히 배제한 **순수 생체 신호 및 라이프로그(No-MMSE)** 환경에서 최고 수준의 판별 성능(SOTA ROC-AUC) 달성
- **해결 접근법**: **도메인 특화 서브스페이스 분해(Feature-Subspace Decomposition)**
  - 생리학적 메커니즘이 다른 피처들을 서브스페이스 단위로 분리하여 전문 모델을 학습
  - 전역 크로스 도메인 상호작용 모델들과 결합한 **백분위 순위 기반 앙상블(Subspace Decomposed Rank Ensemble)** 구축
  - 실제 임상 의사결정에 대응하는 **3단계 파레토 임상 운영 체계(3-Tier Clinical Operating System)** 수립

---

## 2. 사용 데이터셋 (Dataset)

- **데이터셋 파일 경로**: `c:\ML4\data\processed\tabular\patient_level_circadian_v3.csv`
- **표본 크기**: 총 174명 ($N=174$, 중복 없는 환자 단위 레코드)
- **타깃 레이블 분포 (`label`)**:
  - **정상군 (Class 0, Normal / CN)**: 111명 (63.8%)
  - **인지저하군 (Class 1, Cognitive Impairment)**: 63명 (36.2%)
    - 세부 임상 진단 구성: 경도인지장애(MCI) 51명 + 치매(Dementia) 12명
- **엄격한 No-MMSE 및 데이터 누수 방지 원칙**:
  - MMSE 문항 점수(`Q1`~`Q30`), 총점(`TOTAL`), 검사 회차(`DIAG_SEQ`), 담당의(`DOCTOR_NM`), 환자 식별자(`EMAIL`) 등 설문 기반 인지평가 점수와 식별 정보를 입력 피처에서 완전 배제.

---

## 3. 데이터 전처리 및 검증 프로토콜

### 3.1 Zero-Leakage Nested Cross-Validation 구조
데이터 누수(Data Leakage)를 방지하기 위해 엄격한 이중 교차 검증(Nested CV)을 적용했습니다.

- **Outer CV (5-Fold)**: 최종 모델의 일반화 성능(Out-of-Fold, OOF)을 평가하는 외부 루프
- **Inner CV (5-Fold)**: 
  - 각 모델의 예측 확률 생성
  - 3단계 임상 임계값(Thresholds) 최적화
  - 메타 학습기(Stacking Meta-Learner) 훈련
- **환자 단위 분할 검증**: 환자 식별자(`EMAIL`) 기준 중복이 없음을 확인하고 균등한 라벨 비율을 보장하는 `StratifiedKFold` 적용.

### 3.2 전처리 파이프라인 (Fold 내부 격리 실행)
모든 전처리 과정은 Test 세트 정보를 일절 참조하지 않고 각 Fold의 Train 데이터에 대해서만 `fit` 후 Test 데이터에 `transform`을 수행했습니다.

1. **결측치 대치 (Imputation)**:
   - `SimpleImputer(strategy='median')` 적용 (라이프로그 데이터의 비대칭 분포 및 이상치 영향 최소화)
2. **강건한 정규화 (Robust Scaling)**:
   - `RobustScaler()`를 통해 사분위수(IQR) 기준으로 스케일링하여 생체 신호의 극단치(Outlier) 왜곡 억제
3. **클래스 불균형 완화 (Resampling)**:
   - `BorderlineSMOTE(random_state=42)`를 **Fold별 Train 세트에만 적용**하여 결정 경계 부근 소수 클래스(환자군) 합성

---

## 4. 피처 엔지니어링 및 도메인 서브스페이스 구성

14종의 핵심 도메인 피처를 선정하고, 생리학적 특성에 맞게 2개의 전문 서브스페이스(Subspace) 및 전역 세트로 분해했습니다.

### 4.1 핵심 파생 지표 (Domain Engineered Features)
- **자율신경 회복력 (`HR_drop_ratio`)**:
  $$\text{HR\_drop\_ratio} = \frac{\text{sleep\_hr\_average} - \text{sleep\_hr\_lowest}}{\text{sleep\_hr\_average} + \epsilon}$$
  - 야간 수면 중 심박수가 정상적으로 하강(Dipping)하는지 나타내며, 자율신경계 이상 및 뇌 인지기능 저하와 높은 상관관계를 지님.
- **일주기 리듬 파괴 스트레스 (`Circadian_Strain`)**:
  $$\text{Circadian\_Strain} = \frac{\text{circadian\_IV}}{\text{circadian\_IS} + \epsilon}$$
  - 일내 분절화(Intradaily Variability, IV)와 일간 안정성(Interdaily Stability, IS)의 비율로, 일주기 생체 리듬의 불안정도를 집약한 핵심 지표.

### 4.2 도메인 서브스페이스 분해 구성

| 구분 | 피처 수 | 소속 피처 목록 | 임상적 의미 |
|:---|:---:|:---|:---|
| **서브스페이스 1**<br>(일주기 & 자율신경) | 7종 | `circadian_IV`<br>`circadian_IS`<br>`circadian_RA`<br>`sleep_wake_bouts_avg`<br>`HR_drop_ratio`<br>`Circadian_Strain`<br>`sleep_hr_5min_max_std` | 24시간 생체 시계 분절화, 일간 리듬 안정성, 수면 중 자율신경(심박 변동 및 야간 회복률) 상태 반영 |
| **서브스페이스 2**<br>(수면 구조 & 활동) | 7종 | `sleep_score_alignment`<br>`sleep_awake_std`<br>`sleep_breath_average`<br>`activity_score_std`<br>`activity_class_3_count_std`<br>`activity_met_min_low_std`<br>`sleep_restless_std` | 수면 시작 규칙성, 수면 중 잦은 깸/뒤척임 변동성, 호흡수, 주간 활동 점수 및 중/저강도 활동량의 일별 변동성 반영 |
| **전역 상호작용**<br>(전체 통합) | 14종 | 위 서브스페이스 1, 2의 전체 14개 피처 | 개별 도메인을 넘어선 크로스 도메인 비선형 상호작용 학습 |

---

## 5. 모델 아키텍처 및 학습 설정

### 5.1 도메인 특화 모델 (Specialists)
- **Circadian-Autonomic Specialist (`CatBoost`)**:
  - 서브스페이스 1 (7종) 피처만 입력으로 사용
  - 하이퍼파라미터: `depth=3`, `l2_leaf_reg=6.0`, `learning_rate=0.04`, `iterations=130`, `auto_class_weights='Balanced'`
- **Sleep-Activity Specialist (`LightGBM`)**:
  - 서브스페이스 2 (7종) 피처만 입력으로 사용
  - 하이퍼파라미터: `max_depth=3`, `num_leaves=7`, `learning_rate=0.04`, `n_estimators=110`, `reg_alpha=0.2`, `reg_lambda=0.2`, `min_child_samples=18`, `class_weight='balanced'`

### 5.2 전역 크로스 도메인 모델 (Global Models, 14종 피처)
- **`CatBoost_Global`**: `depth=3`, `l2_leaf_reg=4.0`, `learning_rate=0.035`, `iterations=140`, `auto_class_weights='Balanced'`
- **`XGBoost_Global`**: `max_depth=3`, `learning_rate=0.04`, `n_estimators=100`, `subsample=0.8`, `colsample_bytree=0.8`
- **`RBF_SVM_Global`**: `C=1.0`, `kernel='rbf'`, `class_weight='balanced'`, `probability=True`
- **`RandomForest_Global`**: `max_depth=3`, `n_estimators=180`, `max_features='sqrt'`, `min_samples_leaf=2`, `class_weight='balanced'`

---

## 6. 앙상블 전략 및 3단계 임상 운영 체계

### 6.1 Subspace Decomposed Rank Ensemble (최종 챔피언 모델)
- 단순 예측 확률의 평균(Soft Voting)은 모델별 캘리브레이션 편차 및 극단치에 취약하므로, 각 모델의 예측 확률을 백분위 순위(Percentile Rank, $0.0 \sim 1.0$)로 변환 후 가중 합산함:
  $$\text{Final\_Rank\_Score} = 0.25 \times \text{Rank}_{\text{CatBoost\_Circ}} + 0.15 \times \text{Rank}_{\text{LGBM\_Sleep}} + 0.25 \times \text{Rank}_{\text{CatBoost\_Global}} + 0.20 \times \text{Rank}_{\text{XGB\_Global}} + 0.10 \times \text{Rank}_{\text{SVM\_Global}} + 0.05 \times \text{Rank}_{\text{RF\_Global}}$$
- **가중치 배분의 의미**:
  - 도메인 특화 모델에 총 40% (생체시계 25%, 수면활동 15%)를 부여하여 고유 생체 신호 확보.
  - 전역 상호작용 모델에 60%를 부여하여 전반적인 시너지 유지.

### 6.2 3단계 파레토 임상 운영 체계 (3-Tier Clinical Operating System)
의료 현장의 다양한 의사결정 요구를 충족하기 위해 Inner CV를 통해 최적화된 3단계 임계값을 적용합니다.

1. **Tier 1: 1차 조기 선별 모드 (Ultra-Early Screening)**
   - 목표: 환자 누락 방지 (목표 Recall $\ge 0.75$)
   - 활용처: 보건소, 지역사회 복지관 등 대규모 1차 스크리닝
2. **Tier 2: 표준 진단 보조 모드 (Balanced Clinical Diagnosis)**
   - 목표: Youden's J Index ($TPR - FPR$) 최대화 (ROC-AUC 0.7083 최적 균형)
   - 활용처: 1차/2차 병원 외래 진료 시 전문의 판독 보조
3. **Tier 3: 고특이도 확진 모드 (High-Specificity Confirmation)**
   - 목표: 오경보 최소화 (목표 Specificity $\ge 0.80$)
   - 활용처: 고비용 정밀 검사(뇌 MRI, 아밀로이드 PET 등) 의뢰 전 위양성(False Positive) 방지

---

## 7. 최종 검증 결과 (Outer 5-Fold OOF)

Outer 5-Fold의 Out-of-Fold(OOF) 전체 174명 환자에 대한 최종 성능 평가 결과입니다.

| 모델 | **ROC-AUC** | [Tier 2] 정확도 | [Tier 2] 재현율 | [Tier 2] 특이도 | [Tier 2] F1 | **[Tier 1] 재현율** | **[Tier 3] 특이도** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CatBoost_Circadian_Specialist | 0.6399 | 0.5690 | 0.6349 | 0.5315 | 0.5161 | 0.7302 | 0.8018 |
| LightGBM_Sleep_Specialist | 0.6348 | 0.6092 | 0.5714 | 0.6306 | 0.5143 | 0.6984 | 0.8108 |
| CatBoost_Global | 0.6967 | 0.6264 | 0.6825 | 0.5946 | 0.5695 | 0.7143 | 0.8108 |
| XGBoost_Global | 0.6732 | 0.6609 | 0.5873 | 0.7027 | 0.5564 | 0.7619 | 0.8288 |
| RBF_SVM_Global | 0.6466 | 0.5862 | 0.5714 | 0.5946 | 0.5000 | 0.7619 | 0.7838 |
| RandomForest_Global | 0.6690 | 0.6149 | 0.6508 | 0.5946 | 0.5503 | 0.7143 | 0.8288 |
| **V44_Subspace_Decomposed_Rank_Ensemble** | **`0.7083`**<br>[0.6295 ~ 0.7881] | **`0.6379`** (63.8%)<br>[0.5690 ~ 0.7126] | **`0.7460`** (74.6%)<br>[0.6349 ~ 0.8571] | **`0.5766`** (57.7%)<br>[0.4865 ~ 0.6669] | **`0.5987`**<br>[0.5294 ~ 0.6752] | **`0.7778`** (77.8%)<br>[0.6825 ~ 0.8730] | **`0.8198`** (82.0%)<br>[0.7477 ~ 0.8919] |
| V44_Subspace_Soft_Ensemble | 0.7017 | 0.6494 | 0.7143 | 0.6126 | 0.5960 | 0.7460 | 0.8108 |
| Stacking_MetaLearner | 0.6752 | 0.6782 | 0.7302 | 0.6486 | 0.6216 | 0.7778 | 0.7748 |

> **참고 (95% CI)**: `V44_Subspace_Decomposed_Rank_Ensemble` 행의 대괄호 `[ ]` 수치는 Out-of-Fold 검증 표본(N=174)을 대상으로 1,000회 계층화 부트스트랩(Stratified Bootstrap Resampling)을 수행하여 산출한 비모수적 95% 신뢰 구간입니다.

---

## 8. 분석 및 고찰

1. **서브스페이스 분해의 효과**:
   - 단일 특화 모델 자체는 7개 피처만 학습하여 AUC 0.63~0.64 수준을 보이지만, 서로 다른 도메인(생체시계 vs. 수면활동)에서 독립적으로 추출된 신호가 전역 모델들과 융합되면서 상호 보완 효과를 극대화했습니다.
   - 그 결과, 기존 단일 최고 모델(CatBoost Global AUC 0.6967) 및 이전 세대 챔피언 모델(V42 AUC 0.7004)을 뛰어넘는 **ROC-AUC 0.7083**을 기록했습니다.
2. **Rank Ensemble의 강건성**:
   - Soft Ensemble(0.7017) 대비 Rank Ensemble(0.7083)이 우수한 성능을 보였습니다. 이는 이종 모델(CatBoost, LightGBM, XGBoost, SVM, RF) 간 예측 확률값의 편향을 정규화하여 극단적인 오분류를 효과적으로 방지했기 때문입니다.
3. **3단계 임상 체계의 실효성**:
   - Tier 1 모드에서 **Recall 77.78%**를 기록하여 실제 선별 현장에서 10명 중 약 8명의 잠재 인지저하 환자를 조기에 포착할 수 있습니다.
   - Tier 3 모드에서는 **Specificity 81.98%**를 확보하여 불필요한 상급 종합병원 전원 및 고비용 검사 유입을 최소화할 수 있습니다.

---

## 9. 관련 산출물 링크

- **소스 코드**:
  - [V44_Circadian_Subspace_SOTA_Nested.py](file:///c:/ML4/Model/Binary/V44_Circadian_Subspace_SOTA_Nested.py)
  - [V44_Circadian_Subspace_SOTA_Nested.py](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/V44_Subspace_SOTA/V44_Circadian_Subspace_SOTA_Nested.py)
- **보고서 파일**:
  - [report_v44_subspace_sota_detailed.md](file:///c:/ML4/report/binary/report_v44_subspace_sota_detailed.md)
  - [report_v44_subspace_sota_detailed.md](file:///c:/ML4/Google-Ajou-AICapstone/Taehyun/V44_Subspace_SOTA/report_v44_subspace_sota_detailed.md)
- **시각화 결과 파일**:
  - ROC 곡선: [roc_curves_v44_circadian_subspace_sota.png](file:///c:/ML4/report/plots/roc_curves_v44_circadian_subspace_sota.png)
  - 3단계 혼동 행렬: [confusion_matrix_v44_circadian_subspace_sota.png](file:///c:/ML4/report/plots/confusion_matrix_v44_circadian_subspace_sota.png)
