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

## 앞으로 작업할 때

- 새 실험 코드는 가능하면 `.py` 파일로 작성합니다.
- 자기 폴더인 `Hyunsoo/` 안에서만 파일을 수정합니다.
- 다른 사용자 폴더는 건드리지 않습니다.
- 노트북으로 실험했다면, 공유 전에는 실행할 코드 셀을 `.py` 파일로 옮겨둡니다.

