# 선행연구 재현 모델 (Paper Reproduction)

이 폴더는 라이프로그 기반 치매/인지기능 저하 예측의 주요 선행연구 3편에 대한 **최종 및 완전 재현(Exact Reproduction & Leakage Proof)** 코드 파일들을 포함하고 있습니다.

모든 코드는 원 논문의 실험 조건을 재현함과 동시에, 데이터 누수(Patient Leakage) 유무 및 Nested CV 적용에 따른 4가지 시나리오(5 Repeats × 5 Folds)를 비교 검증하여 통계적 신뢰도(95% CI)와 혼동 행렬(Confusion Matrix)을 산출합니다.

---

## 1. 포함된 선행연구 재현 파일 목록

| 논문 및 대상 | 재현 스크립트 파일 | 모델 아키텍처 및 특징 |
| :--- | :--- | :--- |
| **최지예 (2025)**<br>*(최현철 외, 2023)* | `Choi_Ensemble.py` | • Pearson 상관계수(r > 0.9) 제거 + RF 중요도 기반 변수 선택<br>• RF + GBM + XGBoost + RBF-SVM 소프트 보팅 앙상블 |
| **천희웅 외 (2025)** | `Cheon_LGBM.py` | • 5-Fold 교차검증 기반 SHAP 절대 중요도 산출<br>• SHAP 상위 40개 특성 전진 선택(Forward Selection) + LightGBM |
| **Kim & Park (2026)** | `KimPark_LR.py` | • 피험자 단위 장기 종단 변동성 지표(Longitudinal Variability) 집계<br>• RFECV 변수 선택 + L2 정규화 로지스틱 회귀(Logistic Regression) |

---

## 2. 검증 시나리오 구성 (공통)

각 스크립트는 다음 4가지 시나리오를 수행하여 원 논문의 재현성 및 일반화 성능을 정밀하게 분석합니다.

1. **Scenario 1 (Leakage + Single CV)**: 환자 분리 없이 무작위 셔플 분할 (원 논문 보고 수치 재현 환경)
2. **Scenario 2 (Leakage + Nested CV)**: 데이터 누수 환경에서 내부 최적화 수행
3. **Scenario 3 (No Leakage + Single CV)**: `GroupKFold`를 통한 엄격한 환자 단위 분리 (Zero-Leakage)
4. **Scenario 4 (No Leakage + Nested CV)**: 환자 분리 + 내부 하이퍼파라미터 튜닝이 결합된 엄격한 일반화 검증

---

## 3. 실행 방법

저장소 루트의 `base.ipynb`에서 `USER_FOLDER`와 `RUN_FILE`을 설정하여 로컬 및 Colab에서 실행할 수 있습니다.

### 예시: 최지예 논문 앙상블 재현 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction/Choi_Ensemble.py"
```

### 예시: 천희웅 논문 LightGBM 재현 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction/Cheon_LGBM.py"
```

### 예시: Kim & Park 논문 로지스틱 회귀 재현 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction/KimPark_LR.py"
```
