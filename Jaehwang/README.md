# Jaehwang 폴더 안내

이 폴더는 Jaehwang 개인 작업 공간입니다. 팀 공통 방식에서는 `base.ipynb`에서 다음처럼 지정해 이 폴더 안의 `.py` 파일을 실행합니다.

```python
USER_FOLDER = "Jaehwang"
RUN_FILE = "실행할/파일.py"
```

## 이번 변경 요약

기존에는 `코드 3가지/` 폴더 안에 노트북 사본 파일이 있었습니다. 원본 파일은 수정하지 않고, 같은 실험 번호의 `.py` 파일을 추가했습니다.

| 기존 파일 | 추가된 공유용 Python 파일 | 설명 |
| --- | --- | --- |
| `코드 3가지/Experiment1.ipynb의 사본` | `코드 3가지/Experiment1.py` | Experiment1 노트북 사본의 코드 셀을 Python 스크립트로 변환 |
| `코드 3가지/Experiment2.ipynb의 사본` | `코드 3가지/Experiment2.py` | Experiment2 노트북 사본의 코드 셀을 Python 스크립트로 변환 |
| `코드 3가지/Experiment3.ipynb의 사본` | `코드 3가지/Experiment3.py` | Experiment3 노트북 사본의 코드 셀을 Python 스크립트로 변환 |

## 기존 구현 유지 여부

기존 노트북 사본 파일은 삭제하지 않았습니다. 이번 변경은 Git 공유와 `base.ipynb` 실행을 위해 `.py` 변환 파일을 추가한 것입니다.

`기존 데이터셋/` 안의 데이터 파일도 수정하지 않았습니다. 데이터 파일은 `.gitignore` 기준으로 Git에 올리지 않습니다.

## 실행 예시

Experiment1 실행:

```python
USER_FOLDER = "Jaehwang"
RUN_FILE = "코드 3가지/Experiment1.py"
```

Experiment2 실행:

```python
USER_FOLDER = "Jaehwang"
RUN_FILE = "코드 3가지/Experiment2.py"
```

Experiment3 실행:

```python
USER_FOLDER = "Jaehwang"
RUN_FILE = "코드 3가지/Experiment3.py"
```

## 데이터와 결과물

아래 항목은 각자 로컬 또는 Google Drive에 가지고 있어야 하며 Git에는 올리지 않습니다.

- `기존 데이터셋/*.csv`
- `기존 데이터셋/*.pkl`
- 개인 메모용 `*.txt`

실행 코드에서 데이터 경로가 필요하면 `DATA_ROOT` 또는 `USER_ROOT`를 기준으로 경로를 잡으면 됩니다.

## 앞으로 작업할 때

- 새 실험 코드는 가능하면 `.py` 파일로 작성합니다.
- 자기 폴더인 `Jaehwang/` 안에서만 파일을 수정합니다.
- 다른 사용자 폴더는 건드리지 않습니다.
- 노트북으로 먼저 실험했다면, 공유 전에는 `.py` 파일로 옮겨둡니다.

