# Taehyun 폴더 안내

이 폴더는 Taehyun 개인 작업 공간입니다. 팀 공통 방식에서는 `base.ipynb`에서 다음처럼 지정해 이 폴더 안의 `.py` 파일을 실행합니다.

```python
USER_FOLDER = "Taehyun"
RUN_FILE = "실행할/파일.py"
```

## 이번 변경 요약

기존 노트북 원본은 삭제하거나 수정하지 않고, 같은 위치에 실행 가능한 `.py` 변환 파일을 추가했습니다.

| 기존 파일 | 추가된 공유용 Python 파일 | 설명 |
| --- | --- | --- |
| `previous/Binary_LGBM_RF.ipynb` | `previous/Binary_LGBM_RF.py` | Binary LGBM/RF 노트북을 Python 스크립트로 변환 |
| `previous/Binary_LGBM_RF_test.ipynb` | `previous/Binary_LGBM_RF_test.py` | Binary LGBM/RF 테스트 노트북을 Python 스크립트로 변환 |

## 기존 구현 유지 여부

기존 XAI Python 패키지는 수정하지 않았습니다.

- `previous/xai/__init__.py`
- `previous/xai/aggregation.py`
- `previous/xai/analyzer.py`
- `previous/xai/detect.py`
- `previous/xai/explainers.py`
- `previous/xai/outputs.py`

기존 `.ipynb` 원본도 로컬에는 그대로 있습니다. 다만 `.ipynb`는 `.gitignore` 대상이므로 Git 공유와 `base.ipynb` 실행은 변환된 `.py` 파일을 기준으로 합니다.

## 주요 실험 및 모델 폴더 안내

### 1. 선행연구 순수 재현 (`Paper_Reproduction/`)
주요 선행연구 3편(최지예 2025, 천희웅 2025, Kim & Park 2026)의 원 논문 방법론 및 실험 절차 순수 재현 코드입니다.

- `Paper_Reproduction/Choi_Ensemble.py`: 최지예(2025) 3진 분류 RF+GBM+XGB+SVM+PyTorch LSTM 5종 보팅 앙상블 재현
- `Paper_Reproduction/Cheon_LGBM.py`: 천희웅(2025) Step 1~3 논문 결과 순서 3단계 완전 재현 (SHAP Top 40 + Tuned LGBM)
- `Paper_Reproduction/KimPark_LR.py`: Kim & Park(2026) 피험자 종단 변동성 지표 + 로지스틱 회귀 재현

### 2. 피험자 분할 & Nested CV 검증 (`Paper_Reproduction_Nested/`)
선행연구 3편에 대해 환자 ID 단위 분할(`StratifiedGroupKFold`, Zero-Leakage) 및 내부 Optuna 튜닝(Nested CV)을 적용하여 4대 시나리오(Leakage vs Zero-Leakage, Single CV vs Nested CV)를 정밀 검증하는 코드입니다.

- `Paper_Reproduction_Nested/Cheon_LGBM.py`: 피험자 분할 + Fold 내 SHAP 선별 + Optuna Nested CV
- `Paper_Reproduction_Nested/Choi_Ensemble.py`: 피험자 분할 + 5종 앙상블(LSTM 포함) + Optuna Nested CV
- `Paper_Reproduction_Nested/KimPark_LR.py`: 피험자 단위 종단 변동성 + Fold 내 RFECV + L2 정규화 LR Nested CV

### 3. V44 최고 성능 모델 (`V44_Subspace_SOTA/`)
도메인 특화 피처 서브스페이스 분해 및 랭크 앙상블 기법으로 **ROC-AUC 0.7074 신기록**을 달성한 최신 SOTA 모델입니다.

- `V44_Subspace_SOTA/V44_Circadian_Subspace_SOTA_Nested.py`: V44 최종 챔피언 파이프라인
- `V44_Subspace_SOTA/report_binary_v44_circadian_subspace_sota.md`: V44 상세 성과 보고서

---

## 실행 예시 (`base.ipynb` 기준)

### 1) V44 최고 성능 모델 실행 (SOTA, ROC-AUC 0.7074)
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "V44_Subspace_SOTA/V44_Circadian_Subspace_SOTA_Nested.py"
```

### 2) 선행연구 순수 재현 모델 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction/Choi_Ensemble.py"      # 최지예 (2025) 재현
# RUN_FILE = "Paper_Reproduction/Cheon_LGBM.py"      # 천희웅 (2025) 재현
# RUN_FILE = "Paper_Reproduction/KimPark_LR.py"      # Kim & Park (2026) 재현
```

### 3) 피험자 분할 & Nested CV 검증 모델 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "Paper_Reproduction_Nested/Cheon_LGBM.py"    # 천희웅 Nested CV 검증
# RUN_FILE = "Paper_Reproduction_Nested/Choi_Ensemble.py" # 최지예 5종 앙상블 Nested CV 검증
# RUN_FILE = "Paper_Reproduction_Nested/KimPark_LR.py"   # Kim & Park Nested CV 검증
```

### 4) 기존 Binary LGBM/RF 실행
```python
USER_FOLDER = "Taehyun"
RUN_FILE = "previous/Binary_LGBM_RF.py"
```

## 데이터와 결과물

데이터, 모델, 결과물은 Git에 올리지 않습니다. 필요한 파일은 각자 로컬(`Taehyun/data/`) 또는 Google Drive에 준비해둡니다.

### 주요 모델별 사용 데이터셋 (`Taehyun/data/` 직하 위치)
* **V44 최고 성능 모델 (`V44_Subspace_SOTA`)**:
  * `patient_level_circadian_v3.csv`
* **선행연구 재현 모델 (`Paper_Reproduction`)**:
  * `train_activity.csv` / `val_activity.csv`
  * `train_sleep.csv` / `val_sleep.csv`
  * `training_label.csv` (또는 `training_label_activity.csv`) / `val_label.csv`

## 앞으로 작업할 때

- 새 실험 코드는 가능하면 `.py` 파일로 작성합니다.
- 자기 폴더인 `Taehyun/` 안에서만 파일을 수정합니다.
- 다른 사용자 폴더는 건드리지 않습니다.
- 공통 XAI 모듈을 바꾸는 경우, 어떤 실험 코드가 영향을 받는지 README나 커밋 메시지에 남깁니다.

