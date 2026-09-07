# 🚀 V44 Circadian Subspace SOTA 신기록 달성 모델

이 폴더는 치매 고위험군 라이프로그 데이터(Zero-Leakage 환자 단위 분할, No-MMSE 환경)에서 **ROC-AUC 0.7074 신기록**을 달성한 **V44 최고 성능 모델** 및 관련 보고서를 포함하고 있습니다.

---

## 1. 포함된 파일 목록

| 파일명 | 종류 | 설명 |
| :--- | :--- | :--- |
| `V44_Circadian_Subspace_SOTA_Nested.py` | Python 스크립트 | 도메인 특화 서브스페이스 분해 & 랭크 앙상블 최고 성능 모델 파이프라인 |
| `report_binary_v44_circadian_subspace_sota.md` | 성과 보고서 | 성능 비교표(Outer 5-Fold OOF), 임상 3단계 선별 체계, 세부 분석 결과 |

---

## 2. 핵심 아키텍처 요약

1. **도메인 특화 피처 서브스페이스 분해 (Feature-Subspace Decomposition)**:
   - **서브스페이스 1 (일주기 생체 시계 & 자율신경계 7종)**: `Circadian_Strain` (IV / IS), `circadian_IV`, `circadian_IS`, `circadian_RA`, `sleep_wake_bouts_avg`, `HR_drop_ratio`, `sleep_hr_5min_max_std` → **CatBoost Circadian Specialist**
   - **서브스페이스 2 (야간 수면 구조 & 주간 활동 7종)**: `sleep_score_alignment`, `sleep_awake_std`, `sleep_breath_average`, `activity_score_std`, `activity_class_3_count_std`, `activity_met_min_low_std`, `sleep_restless_std` → **LightGBM Sleep Specialist**
   - **전역 모델 (14종 전체 피처)**: CatBoost, XGBoost, RBF-SVM, RandomForest
2. **Subspace Decomposed Rank Ensemble**:
   - 전문 모델과 전역 모델의 백분위 순위(Percentile Rank)를 가중 융합하여 소규모 임상 데이터($N=174$)에서의 예측 분산을 최소화.
3. **3단계 파레토 임상 의사결정 체계 (3-Tier Clinical Operating System)**:
   - **Tier 1 (조기 선별)**: Recall 77.78% 확보 (환자 조기 누락 방지)
   - **Tier 2 (표준 진단)**: Youden J 최적 균형점 (**ROC-AUC 0.7074**, F1 0.5987)
   - **Tier 3 (확진 보조)**: Specificity 81.98% 확보 (고비용 정밀 검사 유도)

---

## 3. 검증 성능 (Outer 5-Fold Out-of-Fold)

- **ROC-AUC**: **0.7074** (SOTA 신기록)
- **Tier 1 선별 Recall**: **77.78%**
- **Tier 3 확진 Specificity**: **81.98%**
- **데이터 누수 완전 차단**: Outer 5-Fold Group + Inner 5-Fold Nested Cross-Validation 적용

---

## 4. 사용 데이터셋

* **파일명**: `patient_level_circadian_v3.csv`
* **위치**: `Taehyun/data/patient_level_circadian_v3.csv`

---

## 5. 실행 방법

저장소 루트의 `base.ipynb`에서 다음과 같이 지정하여 실행합니다.

```python
USER_FOLDER = "Taehyun"
RUN_FILE = "V44_Subspace_SOTA/V44_Circadian_Subspace_SOTA_Nested.py"
```
