# SangHyo 폴더 안내

이 폴더는 SangHyo 개인 작업 공간입니다. 팀 공통 방식에서는 `base.ipynb`에서 다음처럼 지정해 이 폴더 안의 `.py` 파일을 실행합니다.

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "실행할/파일.py"
```

## 이번 변경 요약

기존 `.py` 구현은 유지하고, 노트북 원본을 실행 가능한 `.py` 파일로 변환해 추가했습니다.

| 기존 파일 | 추가된 공유용 Python 파일 | 설명 |
| --- | --- | --- |
| `XAI_Paper_Reproduction/XAI_Paper_Reproduction_Colab.ipynb` | `XAI_Paper_Reproduction/XAI_Paper_Reproduction_Colab.py` | XAI paper reproduction Colab 노트북을 Python 스크립트로 변환 |
| `XAI_Paper_Reproduction2/Training/XAI_Paper_Reproduction2_PaperExact_Colab.ipynb` | `XAI_Paper_Reproduction2/Training/XAI_Paper_Reproduction2_PaperExact_Colab.py` | paper exact reproduction Colab 노트북을 Python 스크립트로 변환 |
| `previous/Experiment1.ipynb` | `previous/Experiment1.py` | 기존 Experiment1 노트북 코드 셀을 Python 스크립트로 변환 |
| `previous/Experiment2.ipynb` | `previous/Experiment2.py` | 기존 Experiment2 노트북 코드 셀을 Python 스크립트로 변환 |
| `previous/Experiment3.ipynb` | `previous/Experiment3.py` | 기존 Experiment3 노트북 코드 셀을 Python 스크립트로 변환 |

## 기존 구현 유지 여부

기존 Python 구현은 수정하지 않았습니다.

- `XAI_Paper_Reproduction/scripts/*.py`
- `XAI_Paper_Reproduction/src/xai_paper_reproduction.py`
- `test/*.py`

기존 노트북 원본도 로컬에는 그대로 있습니다. 다만 `.ipynb`는 `.gitignore` 대상이므로 Git 공유와 `base.ipynb` 실행은 변환된 `.py` 파일을 기준으로 합니다.

## 실행 예시

tqdm 테스트 파일 실행:

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "test/tqdm_demo.py"
```

XAI paper reproduction Colab 변환 파일 실행:

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "XAI_Paper_Reproduction/XAI_Paper_Reproduction_Colab.py"
```

Paper exact reproduction Colab 변환 파일 실행:

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "XAI_Paper_Reproduction2/Training/XAI_Paper_Reproduction2_PaperExact_Colab.py"
```

기존 Experiment 실행:

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "previous/Experiment1.py"
```

## 데이터와 결과물

아래 항목은 Git에 올리지 않습니다.

- `outputs/`
- `outputs_paper_exact/`
- `*.csv`
- `*.joblib`
- `*.pkl`
- `.ipynb`
- `requirements.txt`

필요한 파일은 각자 로컬 또는 Google Drive에 준비해둡니다.

## 앞으로 작업할 때

- 새 실험 코드는 가능하면 `.py` 파일로 작성합니다.
- 자기 폴더인 `SangHyo/` 안에서만 파일을 수정합니다.
- 다른 사용자 폴더는 건드리지 않습니다.
- 결과물은 Git에 올리지 않고, 필요한 경우 README에 생성 방법만 설명합니다.

