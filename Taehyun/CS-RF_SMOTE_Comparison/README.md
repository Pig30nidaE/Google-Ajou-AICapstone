# CS-RF SMOTE Comparison & Advanced Verification Pipeline

본 폴더는 **일주기 리듬(Circadian) 및 수면-활동(Sleep-Activity) 서브스페이스 모델 파이프라인**에서 **BorderlineSMOTE 적용 유무에 따른 성능 및 임상 지표를 체계적으로 검증**하고, 5개 시드 반복 교차검증 및 Out-of-Fold (OOF) SHAP 분석을 수행한 결과와 재현 코드를 포함합니다.

---

## 1. 주요 적용 사항 (4대 핵심 개선)

1. **Inner Fold 기반 임계값(Threshold) 최적화 (Zero-Leakage)**
   - 테스트 데이터 정보를 일절 보지 않고, Inner Fold의 검증 확률에서 Youden's Index ($J = TPR - FPR$)를 최대화하는 최적 임계값을 도출하여 Outer Test 예측에 적용.
   - 치매 조기 선별에 필수적인 환자군 검출 민감도(Recall)를 극대화.

2. **다중 시드(Multi-Seed) 반복 검증 프로토콜**
   - 5개 시드(`42, 13, 73, 101, 2026`) × Outer 5-Fold × Inner 5-Fold (총 50회 Outer Fold 평가).
   - 데이터 분할 우연성(Split Variance)을 제어하고, B=1,000회 환자 단위 층화 부트스트랩으로 95% 신뢰구간(95% CI) 산출.

3. **SMOTE 유무에 따른 체계적 비교 (With SMOTE vs Without SMOTE)**
   - 동일한 5개 시드 조건에서 BorderlineSMOTE 적용 여부에 따른 모델별 ROC-AUC, Recall, F1, Accuracy 비교 및 차이(Δ) 분석.

4. **Zero-Leakage Out-of-Fold (OOF) SHAP 분석 파이프라인**
   - Outer Fold 모델들이 보지 않은 Hold-out 환자들에 대해 SHAP 값을 산출하여 전체 환자(N=174)에 대한 비편향 특성 중요도 매트릭스 구축.
   - 일간 변동성(`_std`) 피처와 생체 리듬 지표의 기여도 규명.

---

## 2. SMOTE 유무에 따른 모델별 성능 비교표 (5-Seed 평균 & 95% CI)

| 모델명 | With SMOTE AUC [95% CI] | Without SMOTE AUC [95% CI] | Δ AUC | With SMOTE Recall | Without SMOTE Recall | Δ Recall | With F1 | Without F1 | Δ F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **XGBoost_Global** | **0.7181** [0.6672-0.8255] | **0.7092** [0.6412-0.8055] | -0.0089 | 0.6063 [0.5238-0.7460] | **0.6317** [0.5397-0.7778] | **+0.0254** | 0.5705 | 0.5533 | -0.0172 |
| **V44_Subspace_Rank_Ensemble** | **0.7183** [0.6609-0.8130] | 0.7027 [0.6381-0.7985] | -0.0156 | 0.6476 [0.5556-0.7937] | 0.5905 [0.4603-0.7143] | -0.0571 | 0.5791 | 0.5560 | -0.0231 |
| **V44_Subspace_Soft_Ensemble** | 0.7160 [0.6538-0.8113] | 0.7036 [0.6384-0.7997] | -0.0124 | 0.6222 [0.5397-0.7619] | 0.6286 [0.5556-0.7778] | +0.0063 | 0.5724 | 0.5612 | -0.0113 |
| **CatBoost_Global** | 0.7072 [0.6532-0.8068] | 0.6952 [0.6333-0.7916] | -0.0119 | 0.6159 [0.5397-0.7778] | 0.6095 [0.5238-0.7460] | -0.0063 | 0.5614 | 0.5426 | -0.0188 |
| **Stacking_MetaLearner** | 0.6981 [0.6375-0.8006] | 0.6895 [0.6213-0.7902] | -0.0086 | 0.6063 [0.4921-0.7460] | 0.6254 [0.5397-0.7778] | +0.0190 | 0.5665 | 0.5714 | +0.0049 |
| **RandomForest_Global** | 0.6892 [0.6301-0.7862] | 0.6783 [0.6166-0.7749] | -0.0109 | 0.5873 [0.5234-0.7619] | 0.6190 [0.5075-0.7460] | +0.0317 | 0.5446 | 0.5486 | +0.0039 |
| **LightGBM_Sleep_Specialist** | 0.6676 [0.6061-0.7764] | 0.6676 [0.5989-0.7645] | 0.0000 | 0.6413 [0.5397-0.7782] | 0.5778 [0.4127-0.6667] | -0.0635 | 0.5447 | 0.5265 | -0.0182 |
| **RBF_SVM_Global** | 0.6462 [0.5717-0.7430] | 0.6340 [0.5586-0.7295] | -0.0123 | 0.5460 [0.4599-0.6984] | 0.5714 [0.4127-0.6667] | +0.0254 | 0.5017 | 0.5059 | +0.0042 |
| **CatBoost_Circadian_Specialist** | 0.6215 [0.5451-0.7245] | 0.6078 [0.5282-0.7066] | -0.0137 | 0.5905 [0.5397-0.7778] | 0.5460 [0.4444-0.6984] | -0.0444 | 0.5039 | 0.4846 | -0.0193 |

### 핵심 분석 인사이트
1. **단일 XGBoost의 우수성**: SMOTE를 배제한 실전 임상 환경(Without SMOTE)에서, 단일 `XGBoost_Global` 모델의 AUC(0.7092)가 복잡한 6개 앙상블 모델(0.7027)보다 **+0.0065 높게** 나타났습니다.
2. **SMOTE 제거의 이점**: SMOTE를 제거했을 때 단일 XGBoost의 **환자 검출률(Recall)은 0.6063에서 0.6317로 오히려 +2.54%p 상승**했습니다. 가상 데이터로 인한 과적합 위험을 방지하고 재현성을 확보할 수 있습니다.

---

## 3. Zero-Leakage OOF SHAP 특성 중요도 (XGBoost_Global)

- **`_std` 장기 변동성 피처(총 6개) 기여도 합계**: **53.06%**
- 특정 하루의 단기 상태보다 여러 날에 걸친 **수면 뒤척임, 취침 시간대 정렬, 야간 심박 변동성, 24시간 리듬 안정성(IS)**이 인지장애 판별의 핵심 근거임을 확인했습니다.

| 순위 | 피처명 | Mean \|SHAP\| | 기여도 비중 (%) | 임상적 의미 |
|:---:|:---|:---:|:---:|:---|
| 1 | `sleep_restless_std` | 0.3282 | **12.42%** | 날짜별 수면 중 뒤척임 변동성 |
| 2 | `sleep_score_alignment` | 0.3113 | **11.78%** | 취침 및 기상 시간대의 일관성 (정렬 점수) |
| 3 | `sleep_hr_5min_max_std` | 0.3036 | **11.48%** | 야간 수면 중 최대 심박수의 일간 변동성 |
| 4 | `circadian_IS` | 0.3015 | **11.41%** | 24시간 일주기 리듬의 날짜 간 안정성 |
| 5 | `activity_class_3_count_std` | 0.2771 | **10.48%** | 낮 시간대 중강도 활동 빈도의 일간 변동성 |
| 6 | `sleep_breath_average` | 0.2254 | 8.53% | 야간 평균 호흡수 |
| 7 | `sleep_awake_std` | 0.2199 | 8.32% | 수면 중 각성 시간의 일간 변동성 |
| 8 | `activity_met_min_low_std` | 0.1494 | 5.65% | 저강도 활동 대사량의 일간 변동성 |
| 9 | `sleep_wake_bouts_avg` | 0.1342 | 5.08% | 수면 중 미세 각성 횟수 |
| 10 | `activity_score_std` | 0.1245 | 4.71% | 일일 활동 점수 변동성 |
| 11 | `circadian_IV` | 0.1149 | 4.35% | 24시간 활동-휴식 분절화 지수 |
| 12 | `HR_drop_ratio` | 0.0690 | 2.61% | 야간 심박 하강률 (자율신경계 회복력) |
| 13 | `circadian_RA` | 0.0486 | 1.84% | 주야간 활동 상대 진폭 |
| 14 | `Circadian_Strain` | 0.0358 | 1.36% | 복합 지표 (IV / IS) |

---

## 4. 파일 및 디렉터리 구성

```text
c:\ML4\Google-Ajou-AICapstone\Taehyun\CS-RF_SMOTE_Comparison\
│
├── CS-RF_SMOTE_Comparison.py           # 5-Seed 반복 CV, Youden 컷오프, SMOTE 비교, OOF SHAP 전체 실행 코드
├── README.md                           # 본 설명 문서
├── report_cs_rf_smote_comparison.md    # 실행 후 자동 생성되는 종합 검증 보고서
│
├── plots/                              # 시각화 그래프 디렉터리
│   ├── cs_rf_smote_comparison_5seeds.png # SMOTE 유무에 따른 AUC/Recall 비교 막대 차트
│   └── cs_rf_oof_shap_bar.png          # 14개 피처 SHAP 기여도 막대 그래프
│
└── reports/                            # 상세 수치 결과 디렉터리
    ├── cs_rf_smote_comparison.csv      # 모델별 성능 지표 비교 원본 CSV
    └── cs_rf_xgb_oof_shap_importance.csv # 14개 피처 SHAP 중요도 수치 CSV
```

---

## 5. 실행 및 재현 방법

### 필수 의존성
- Python 3.10+
- `scikit-learn`, `imbalanced-learn`, `xgboost`, `lightgbm`, `catboost`, `shap`, `matplotlib`, `seaborn`, `pandas`, `numpy`

### 실행 명령어
```bash
cd c:\ML4\Google-Ajou-AICapstone\Taehyun\CS-RF_SMOTE_Comparison
python CS-RF_SMOTE_Comparison.py
```
실행 시 원본 데이터(`../data/patient_level_circadian_v3.csv`)를 자동으로 로드하여 5개 시드 반복 검증을 수행하고, `plots/` 및 `reports/` 폴더에 최신 결과물과 보고서를 자동 갱신합니다.
