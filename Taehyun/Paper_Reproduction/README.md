# 선행연구 재현 모델 (Paper Reproduction)

이 폴더는 라이프로그 기반 치매/인지기능 저하 예측의 주요 선행연구 3편에 대한 **최종 및 완전 재현(Exact Reproduction & Leakage Proof)** 코드 파일들을 포함하고 있습니다.

---

## 1. 포함된 선행연구 재현 파일 목록

| 논문 및 대상 | 재현 스크립트 파일 | 분류 과제 및 아키텍처 |
| :--- | :--- | :--- |
| **최지예 (2025)**<br>*(최영근 외, 2024)* | `Choi_Ensemble.py` | • **3진 분류 (CN vs MCI vs Dem)**<br>• Pearson 상관계수 제거 + RF 중요도 선택<br>• RF + GBM + XGBoost + RBF-SVM + **PyTorch LSTM** 5종 소프트 보팅 앙상블<br>• 누수 허용(Scenario 2) vs 누수 차단(Scenario 3, 4) 비교 |
| **천희웅 외 (2025)** | `Cheon_LGBM.py` | • **이진 분류 (CN vs MCI)**<br>• 논문 실험 순서 3단계 완전 재현<br>  - Step 1: Baseline LGBM 평가 (Table 1)<br>  - Step 2: SHAP 절대 중요도 상위 40개 특성 선택 (Figure 4)<br>  - Step 3: 하이퍼파라미터 튜닝 최종 모델 평가 (Table 2) |
| **Kim & Park (2026)** | `KimPark_LR.py` | • **이진 분류 (CN vs MCI)**<br>• 피험자 단위 장기 종단 변동성 지표(Longitudinal Variability) 집계<br>• RFECV 변수 선택 + L2 정규화 로지스틱 회귀(Logistic Regression)<br>• 4가지 누수 비교 시나리오 평가 |

---

## 2. 사용 데이터셋

* **파일명**:
  * `train_activity.csv` / `val_activity.csv` (일별 활동량 데이터)
  * `train_sleep.csv` / `val_sleep.csv` (일별 수면 데이터)
  * `training_label.csv` (또는 `training_label_activity.csv`) / `val_label.csv` (진단 라벨)
* **위치**: `Taehyun/data/` (서브폴더 미사용)

---

## 3. 실행 방법

저장소 루트의 `base.ipynb`에서 `USER_FOLDER`와 `RUN_FILE`을 설정하여 로컬 및 Colab에서 실행할 수 있습니다.

### 예시: 최지예 논문 5종 앙상블(LSTM 포함) 재현 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction/Choi_Ensemble.py"
```

### 예시: 천희웅 논문 3단계(Table 1~2) 재현 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction/Cheon_LGBM.py"
```

### 예시: Kim & Park 논문 로지스틱 회귀 재현 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction/KimPark_LR.py"
```
