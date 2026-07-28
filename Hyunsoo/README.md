# Hyunsoo 폴더 안내

이 폴더는 Hyunsoo 개인 작업 공간입니다. 팀 공통 방식에서는 `base.ipynb`에서 다음처럼 지정해 이 폴더 안의 `.py` 파일을 실행합니다.

```python
USER_FOLDER = "Hyunsoo"
RUN_FILE = "실행할/파일.py"
```

## 이번 변경 요약

기존 노트북 원본은 삭제하거나 수정하지 않고, 같은 위치에 실행 가능한 `.py` 변환 파일을 추가했습니다.

| 기존 파일 | 추가된 공유용 Python 파일 | 설명 |
| --- | --- | --- |
| `Colab_시작하기.ipynb` | `Colab_시작하기.py` | Colab/Codex 시작용 노트북을 Python 스크립트로 변환 |
| `previous/privious_LSTM_preprocessing/build_lstm_dataset_colab.ipynb` | `previous/privious_LSTM_preprocessing/build_lstm_dataset_colab.py` | LSTM 데이터셋 생성 Colab 노트북을 Python 스크립트로 변환 |
| `previous/privious_TreeModel_preprocessing/build_rf_lgbm_binary_daily_colab.ipynb` | `previous/privious_TreeModel_preprocessing/build_rf_lgbm_binary_daily_colab.py` | RF/LGBM용 binary daily 전처리 Colab 노트북을 Python 스크립트로 변환 |

## 기존 구현 유지 여부

기존 `.py` 파일은 그대로 유지했습니다.

- `previous/privious_LSTM_preprocessing/build_lstm_dataset.py`
- `previous/privious_TreeModel_preprocessing/build_rf_lgbm_binary_daily.py`
- `previous/privious_TreeModel_preprocessing/model/LGBM_RF_Model_ver2.py`

기존 `.ipynb` 원본도 로컬에는 그대로 있습니다. 다만 `.ipynb`는 `.gitignore` 대상이라 Git 공유 기준에서는 새로 추가된 `.py` 파일을 사용합니다.

## 실행 예시

LSTM 전처리 Colab 변환 파일 실행:

```python
USER_FOLDER = "Hyunsoo"
RUN_FILE = "previous/privious_LSTM_preprocessing/build_lstm_dataset_colab.py"
```

Tree model 전처리 Colab 변환 파일 실행:

```python
USER_FOLDER = "Hyunsoo"
RUN_FILE = "previous/privious_TreeModel_preprocessing/build_rf_lgbm_binary_daily_colab.py"
```

## 데이터와 결과물

아래 파일들은 로컬 실행에 필요할 수 있지만 Git에는 올리지 않습니다.

- `*.csv`
- `*.pkl`
- `__pycache__/`
- `.DS_Store`

필요한 데이터 파일은 각자 로컬 또는 Google Drive에 준비해두고 실행합니다.

## 최종 모델: `final_dementia_screening_model.py`

Claude Code와의 세션에서 도출한 최종 확정 모델입니다. 문제를 "CN vs MCI+Dementia"가 아니라
"CN+MCI(정상 취급) vs Dementia 스크리닝"으로 재정의하고, 통계적 이상치(`nia+219@rowan.kr`, 하루
평균 2.8만보를 걷는 Dementia 환자) 1명을 제외한 뒤, leak-free nested CV(SHAP 랭킹과 threshold 모두
outer-train 안에서만 결정)로 찾은 단일 피처 `activity_low_std`(저강도 활동시간의 일별 표준편차) +
로지스틱회귀 모델입니다. 표본이 극히 작은 상황(Dementia 11명)에서는 LightGBM 등 복잡한 모델이나
피처를 여러 개 섞는 조합이 오히려 분산이 커져 성능이 떨어짐을 확인했고, 가장 단순한 모델이 최종
선택되었습니다.

**성능 (173명, leak-free nested CV 20회 반복 평균)**:

| ROC-AUC | Accuracy | Precision | Recall(민감도) | Specificity(특이도) | F1 |
| --- | --- | --- | --- | --- | --- |
| 0.9087 | 0.8514 | 0.2749 | 0.8136 | 0.8540 | 0.4108 |

Precision이 낮은 것은 모델 결함이 아니라 클래스 불균형(양성 11명 : 음성 162명)의 산술적 한계입니다.

실행 예시:

```python
USER_FOLDER = "Hyunsoo"
RUN_FILE = "final_dementia_screening_model.py"
```

원본 데이터는 Google Drive `GoogleAI_contest/aihub_original_data` 아래 AIHub 원본 폴더 구조
(`1.Training/{원천데이터,라벨링데이터}`, `2.Validation/{원천데이터,라벨링데이터}`)가 필요합니다.

이 최종 모델까지 오는 과정에서 시도했던 실험 전체(이전 SOTA 수치가 leakage였음을 발견한 과정, 이상치
탐색, 문제 재정의, 실패한 시도들 포함)는 [`EXPERIMENT_LOG.md`](./EXPERIMENT_LOG.md)에 기록되어 있습니다.
결과만 보지 말고 "왜 이 방식을 골랐는지"가 궁금하다면 이 로그를 참고하세요.

## 앞으로 작업할 때

- 새 실험 코드는 가능하면 `.py` 파일로 작성합니다.
- 자기 폴더인 `Hyunsoo/` 안에서만 파일을 수정합니다.
- 다른 사용자 폴더는 건드리지 않습니다.
- 노트북으로 실험했다면, 공유 전에는 실행할 코드 셀을 `.py` 파일로 옮겨둡니다.

