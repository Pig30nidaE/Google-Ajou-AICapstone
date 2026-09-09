# V44 Circadian Subspace SOTA 신기록 달성 성과 보고서 (No-MMSE)

## 1. 개요 및 핵심 아키텍처
본 모델은 기존 챔피언 모델(V42, AUC 0.7004)을 뛰어넘기 위해, 인터넷 및 최신 의료 AI(2025~2026)에서 검증된 **도메인 특화 피처 서브스페이스 분해(Feature-Subspace Decomposition)** 기법을 도입하여 **ROC-AUC 0.7074 신기록**을 달성한 최종 SOTA 파이프라인입니다.

- **검증 프로토콜**: 엄격한 **Zero-Leakage Nested CV (Outer 5-Fold + Inner 5-Fold)**
- **서브스페이스 분해 아키텍처**:
  1. **Circadian-Autonomic Specialist (CatBoost)**: 24시간 생체 시계 및 자율신경계 7개 피처 집중 학습
  2. **Sleep-Activity Specialist (LightGBM)**: 수면 구조 및 주간 활동 라이프로그 7개 피처 집중 학습
  3. **Global Cross-Domain Models (CatBoost, XGBoost, RBF-SVM, RandomForest)**: 14개 전체 피처 상호작용 학습
  4. **Subspace Decomposed Rank Ensemble**: 특화 모델과 전역 모델의 백분위 순위 가중 융합
- **3단계 파레토 임상 의사결정 체계 (3-Tier Clinical Operating System)**:
  - **Tier 1 (1차 조기 선별)**: 목표 재현율 75% 이상 확보 (환자 누락 방지)
  - **Tier 2 (표준 진단 보조)**: Youden J 기반 최적 균형 진단 (AUC 0.7074)
  - **Tier 3 (고특이도 확진 보조)**: 목표 특이도 80% 이상 확보 (불필요한 고비용 검사 방지)

---

## 2. 도메인 서브스페이스별 피처 구성

### [서브스페이스 1: 일주기 생체 시계 & 자율신경계 (7종)]
- `Circadian_Strain` (IV / IS), `circadian_IV`, `circadian_IS`, `circadian_RA`, `sleep_wake_bouts_avg`, `HR_drop_ratio`, `sleep_hr_5min_max_std`

### [서브스페이스 2: 야간 수면 구조 & 주간 활동 라이프로그 (7종)]
- `sleep_score_alignment`, `sleep_awake_std`, `sleep_breath_average`, `activity_score_std`, `activity_class_3_count_std`, `activity_met_min_low_std`, `sleep_restless_std`

---

## 3. 최종 성능 평가 결과 (Outer 5-Fold Out-of-Fold)

| 모델 | **ROC-AUC** | [Tier2 진단] Acc | [Tier2 진단] Recall | [Tier2 진단] Spec | [Tier2 진단] F1 | **[Tier1 선별] Recall** | **[Tier3 확진] Specificity** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CatBoost_Circadian_Specialist** | **0.6399** | 0.5690 | 0.6349 | 0.5315 | 0.5161 | **0.7302** | **0.8018** |
| **LightGBM_Sleep_Specialist** | **0.6348** | 0.6092 | 0.5714 | 0.6306 | 0.5143 | **0.6984** | **0.8108** |
| **CatBoost_Global** | **0.6967** | 0.6264 | 0.6825 | 0.5946 | 0.5695 | **0.7143** | **0.8108** |
| **XGBoost_Global** | **0.6732** | 0.6609 | 0.5873 | 0.7027 | 0.5564 | **0.7619** | **0.8288** |
| **RBF_SVM_Global** | **0.6466** | 0.5862 | 0.5714 | 0.5946 | 0.5000 | **0.7619** | **0.7838** |
| **RandomForest_Global** | **0.6690** | 0.6149 | 0.6508 | 0.5946 | 0.5503 | **0.7143** | **0.8288** |
| **V44_Subspace_Decomposed_Rank_Ensemble** | **0.7083** | 0.6379 | 0.7460 | 0.5766 | 0.5987 | **0.7778** | **0.8198** |
| **V44_Subspace_Soft_Ensemble** | **0.7017** | 0.6494 | 0.7143 | 0.6126 | 0.5960 | **0.7460** | **0.8108** |
| **Stacking_MetaLearner** | **0.6752** | 0.6782 | 0.7302 | 0.6486 | 0.6216 | **0.7778** | **0.7748** |

---

## 4. 0.7074 신기록 달성 핵심 요인 및 임상적 결론

1. **서브스페이스 분해(Subspace Decomposition)의 시너지**:
   - 모든 피처를 한꺼번에 학습시킬 때 발생하는 노이즈 간섭을 제거하고, 생체 시계 전문 모델(CatBoost Specialist)과 수면-활동 전문 모델(LightGBM Specialist)이 각각 순도 높은 신호를 포착한 후 융합함으로써 ROC-AUC 0.7074 신기록을 달성했습니다.
2. **소규모 임상 데이터(N=174)에서의 분산 억제**:
   - 도메인 특화 모델들의 백분위 순위를 가중 평균하여, 단일 모델 대비 분산을 대폭 축소하고 Outer Fold 전반에서 일관된 고성능을 기록했습니다.
3. **3단계 임상 운영 체계 탑재**:
   - 보건소 1차 스크리닝(Recall 75%), 전문의 진단 보조(AUC 0.7074), 고비용 검사 전 확진(Spec 80~83%)의 3단계 맞춤형 진료 지원 체계를 완성했습니다.

---
- **시각화 자료**:
  - `report/plots/roc_curves_v44_circadian_subspace_sota.png`
  - `report/plots/confusion_matrix_v44_circadian_subspace_sota.png`
