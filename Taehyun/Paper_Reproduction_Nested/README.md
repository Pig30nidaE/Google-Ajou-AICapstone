# 피험자 분할 및 Nested CV 적용 재현 모델 (Paper Reproduction - Nested CV)

이 폴더는 주요 선행연구 3편에 대해 **피험자 분할(Subject-level Group Split / Zero-Leakage)** 및 **중첩 교차검증(Nested Cross-Validation, Optuna 최적화)** 구조를 적용하여 데이터 누수 차단 및 통계적 일반화 성능을 정밀하게 검증하는 코드 파일들을 포함하고 있습니다.

---

## 1. 포함된 파일 목록

| 논문 및 대상 | 재현 스크립트 파일 | 분류 과제 및 주요 기법 |
| :--- | :--- | :--- |
| **천희웅 외 (2025)** | `Cheon_LGBM.py` | • **이진 분류 (CN vs MCI)**<br>• 피험자 단위 분할 (`StratifiedGroupKFold`)<br>• Fold 내 SHAP 전진 선별 (Data Leakage 차단)<br>• 내부 3-Fold Optuna 동적 튜닝 (Nested CV) |
| **최지예 (2025)**<br>*(최영근 외, 2024)* | `Choi_Ensemble.py` | • **3진 분류 (CN vs MCI vs Dem)**<br>• 피험자 단위 분할 (`StratifiedGroupKFold`)<br>• Fold 내 상관계수 필터 + RF 중요도 선별 + SMOTE<br>• RF + GBM + XGBoost + RBF-SVM + **PyTorch LSTM** 5종 소프트 보팅 앙상블 |
| **Kim & Park (2026)** | `KimPark_LR.py` | • **이진 분류 (CN vs MCI)**<br>• 피험자 단위 종단 변동성 지표(Mean, Std, CV, Skewness, Kurtosis) 집계<br>• Fold 내 RFECV 특성 선택 + L2 정규화 로지스틱 회귀<br>• 내부 Fold 규제 파라미터 `C` Optuna 동적 튜닝 |

---

## 2. 4대 검증 시나리오 구성 (공통)

각 스크립트는 5 Repeats × 5 Folds(또는 10 Folds)를 수행하여 다음 4가지 시나리오를 비교 평가하고 신뢰구간(95% CI)과 혼동 행렬(Confusion Matrix)을 산출합니다.

1. **Scenario 1 (Leakage + Single CV)**: 환자 분리 없이 무작위 셔플 분할 + 전체 데이터 기반 특성 선별 (원 논문 재현 환경)
2. **Scenario 2 (Leakage + Nested CV)**: 데이터 누수 환경에서 내부 Optuna 하이퍼파라미터 최적화 수행
3. **Scenario 3 (No Leakage + Single CV)**: `StratifiedGroupKFold`를 통한 엄격한 환자 단위 분리 + Train Fold 내에서만 특성 선별 (Zero-Leakage)
4. **Scenario 4 (No Leakage + Nested CV)**: 환자 분리 + Train Fold 내 특성 선별 + 내부 Fold Optuna 튜닝이 결합된 엄격한 일반화 검증

---

## 3. 사용 데이터셋 (Lifelog Dataset)

본 재현 파이프라인에서 사용하는 데이터셋은 **AI Hub 치매 고위험군 라이프로그 데이터셋**을 기반으로 합니다. 스마트 링/밴드를 통해 수집된 연속 일상 활동량 및 야간 수면 생체 신호 데이터로 구성되어 있습니다.

### 3.1 파일 구성 및 세부 스키마

* **위치**: `Taehyun/data/` (하위 서브폴더 미사용)
* **파일 목록**:
  1. **활동량 데이터 (`train_activity.csv` / `val_activity.csv`)**:
     - 피험자의 일별 주간 신체 활동량 기록 (총 12,000+ 일치)
     - **주요 집계 지표**: 총 걸음 수(`activity_steps`), 소모 칼로리(`activity_cal_total`, `activity_cal_active`), 대사당량(MET: `activity_average_met`), 활동 강도별 체류 시간(`activity_low`, `activity_medium`, `activity_high`, `activity_inactive`, `activity_rest`)
     - **고해상도 시계열 피처**: 5분 단위 활동 강도 시퀀스(`activity_class_5min`), 1분 단위 분당 MET 시계열(`activity_met_1min`)
  2. **수면 데이터 (`train_sleep.csv` / `val_sleep.csv`)**:
     - 피험자의 일별 야간 수면 구조 및 자율신경계 생체 지표
     - **주요 집계 지표**: 총 수면 시간(`sleep_total`), 수면 효율(`sleep_efficiency`), 입면 잠복기(`sleep_onset_latency`), 수면 단계별 지속 시간(`sleep_deep`, `sleep_light`, `sleep_rem`, `sleep_awake`), 최저/평균 심박수(`sleep_hr_lowest`, `sleep_hr_average`), 심박변이도(`sleep_rmssd`)
     - **고해상도 시계열 피처**: 5분 단위 수면 단계 수면도(`sleep_hypnogram_5min`), 5분 단위 연속 심박수(`sleep_hr_5min`), 5분 단위 RMSSD(`sleep_rmssd_5min`)
  3. **진단 라벨 데이터 (`training_label.csv` / `val_label.csv`)**:
     - 피험자 식별자(`SAMPLE_EMAIL` / `EMAIL`)와 임상 전문의의 최종 인지기능 진단 라벨(`DIAG_NM`) 매핑 테이블
     - 원본 라벨: `CN` (정상 인지, Cognitively Normal), `MCI` (경도 인지 장애, Mild Cognitive Impairment), `Dem` (치매, Dementia)

### 3.2 피험자 규모 및 라벨 매핑

* **피험자 규모**: Train 142명 + Validation 34명 = **총 176명**
* **분류 과제별 라벨 정의**:
  - **이진 분류 (Cheon, Kim & Park)**:
    - Label 0: `CN` (정상군)
    - Label 1: `MCI` (경도 인지 장애 고위험군)
  - **3진 분류 (Choi)**:
    - Label 0: `CN` (정상군)
    - Label 1: `MCI` (경도 인지 장애)
    - Label 2: `Dem` (치매)

### 3.3 피험자 분할 및 데이터 누수 방지 원칙

1. **환자 단위 분할 (Patient/Subject-level Split)**:
   - 1명의 피험자당 수십~수백 일치의 종단(Longitudinal) 데이터가 존재합니다.
   - 단순 무작위 셔플(Row-level K-Fold)을 적용할 경우同一 환자의 특정 일자가 Train과 Test에 동시에 포함되는 **환자 정보 누수(Patient Leakage)**가 발생하여 평가 성능이 비정상적으로 과대평가됩니다.
   - 본 폴더의 모델들은 **`StratifiedGroupKFold(groups=EMAIL)`**를 통해 동일 환자의 모든 데이터가 한 Fold에만 온전히 속하도록 엄격히 분리(Zero-Leakage)합니다.
2. **시계열 슬래시 시퀀스 전처리**:
   - 문자열로 압축된 슬래시(`/`) 시계열은 각 Fold의 학습 단계에서 파싱되어 평균, 표준편차, 사분위수(IQR), 수면 단계 전이 횟수(transition count) 등 통계적 요약치로 변환된 후 결측치를 처리합니다.

---

## 4. 실행 방법

저장소 루트의 `base.ipynb`에서 `USER_FOLDER`와 `RUN_FILE`을 설정하여 로컬 및 Colab에서 실행할 수 있습니다.

### 예시: 천희웅 논문 피험자 분할 + Nested CV 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction_Nested/Cheon_LGBM.py"
```

### 예시: 최지예 논문 5종 앙상블(LSTM 포함) 피험자 분할 + Nested CV 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction_Nested/Choi_Ensemble.py"
```

### 예시: Kim & Park 논문 피험자 단위 변동성 + Nested CV 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction_Nested/KimPark_LR.py"
```
