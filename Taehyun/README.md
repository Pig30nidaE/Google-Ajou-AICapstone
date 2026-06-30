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

## 실행 예시

Binary LGBM/RF 실행:

```python
USER_FOLDER = "Taehyun"
RUN_FILE = "previous/Binary_LGBM_RF.py"
```

Binary LGBM/RF 테스트 실행:

```python
USER_FOLDER = "Taehyun"
RUN_FILE = "previous/Binary_LGBM_RF_test.py"
```

## 데이터와 결과물

데이터, 모델, 결과물은 Git에 올리지 않습니다. 필요한 파일은 각자 로컬 또는 Google Drive에 준비해둡니다.

## 앞으로 작업할 때

- 새 실험 코드는 가능하면 `.py` 파일로 작성합니다.
- 자기 폴더인 `Taehyun/` 안에서만 파일을 수정합니다.
- 다른 사용자 폴더는 건드리지 않습니다.
- 공통 XAI 모듈을 바꾸는 경우, 어떤 실험 코드가 영향을 받는지 README나 커밋 메시지에 남깁니다.

