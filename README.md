# Google Ajou AI Capstone 코드 공유 가이드

이 저장소는 팀원이 같은 데이터 구조를 기준으로 각자 개인 폴더 안에서만 실험 코드를 관리하고, 공통 `base.ipynb`로 Colab 또는 로컬에서 실행할 수 있도록 구성되어 있습니다.

## 핵심 원칙

- 각 사용자는 자기 폴더만 수정합니다.
  - `Hyunsoo/`
  - `Jaehwang/`
  - `Taehyun/`
  - `SangHyo/`
- 공통 실행 진입점은 루트의 `base.ipynb`입니다.
- `base.ipynb`에서는 `USER_FOLDER`, `RUN_FILE`만 바꾸면 됩니다.
- 필요하다면 base.ipynb를 각자 폴더에 옮긴 후, 수정하여 사용하세요. 이 경우 commit 대상이 되지 않습니다.
- Git에 공유할 코드는 `.py` 파일을 기준으로 합니다.
- 기존 `.ipynb` 원본은 삭제하거나 수정하지 않았습니다. 이번 변경에서는 노트북의 코드 셀을 같은 위치의 `.py` 파일로 변환해 추가했습니다.
- 데이터, 모델, 결과물, 개인 로컬 파일은 `.gitignore` 기준으로 공유하지 않습니다.

## 폴더 구조

```text
Google-Ajou-AICapstone/
  base.ipynb
  Data/                         # 로컬 데이터 폴더, Git에는 올리지 않음
  Hyunsoo/
  Jaehwang/
  Taehyun/
  SangHyo/
```

Colab에서는 데이터가 Google Drive 공유드라이브에 있다고 가정합니다.

```text
/content/drive/Shareddrives/GoogleAI_contest/Data
```

`Data/` 안의 구조는 로컬 `Data/`와 동일해야 합니다.

## base.ipynb 사용 방법

`base.ipynb`에서 사용자가 주로 수정하는 부분은 2번 셀입니다.

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "test/tqdm_demo.py"
```

의미는 다음과 같습니다.

- `USER_FOLDER`: 실행할 개인 폴더명입니다. 예: `"Hyunsoo"`, `"Jaehwang"`, `"Taehyun"`, `"SangHyo"`
- `RUN_FILE`: 개인 폴더를 기준으로 실행할 Python 파일입니다.

예를 들어 다음 설정은:

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "test/tqdm_demo.py"
```

아래 파일을 실행합니다.

```text
SangHyo/test/tqdm_demo.py
```

`RUN_FILE = "test/tqdm_demo"`처럼 `.py`를 빼고 써도, 같은 이름의 `.py` 파일이 있으면 자동으로 찾아줍니다.

## Colab에서 실행하는 방법

편의상 각자 옮길 base파일도 전부 `base.ipynb`로 지칭하겠습니다.

1. GitHub에서 `base.ipynb`를 Colab으로 엽니다.
2. 1번 셀을 실행합니다.
   - Google Drive를 마운트합니다.
   - 기존 `/content/Google-Ajou-AICapstone` 폴더를 삭제합니다.
   - GitHub에서 저장소를 새로 clone합니다.
   - 따라서 매번 GitHub의 최신 코드로 시작합니다.
3. 2번 셀에서 `USER_FOLDER`, `RUN_FILE`을 수정합니다.
4. 3번 셀에서 경로가 맞는지 확인합니다.
5. 4번 셀에서 개인 폴더의 `requirements.txt` 또는 `requirement.txt`가 있으면 설치합니다.
6. 5번 셀에서 지정한 `.py` 파일을 실행합니다.

Colab에서 `/content/Google-Ajou-AICapstone`는 Google Drive가 아닙니다. GitHub 저장소가 Colab 임시 런타임에 clone된 작업 폴더입니다.

Google Drive는 데이터 경로인 `DATA_ROOT`에만 사용합니다.

## 로컬에서 실행하는 방법

로컬에서는 저장소 루트에서 `base.ipynb`를 열고 위에서부터 실행하면 됩니다.

로컬 기본 데이터 경로는 다음과 같습니다.

```text
<repo>/Data
```

데이터 파일은 Git에 올리지 않으므로, 각자 로컬 또는 공유드라이브에서 따로 준비해야 합니다.

## 개인 코드에서 공통 경로 쓰기

`base.ipynb`는 지정한 파일을 일반 Python script처럼 실행합니다. 실행 파일 안에서는 아래 변수를 바로 사용할 수 있습니다.

```python
PROJECT_ROOT  # 저장소 루트
DATA_ROOT     # 공통 데이터 폴더
USER_ROOT     # 개인 폴더
RUN_PATH      # 현재 실행 중인 파일
```

예시:

```python
from pathlib import Path

train_activity = DATA_ROOT / "1.Training/SourceData/1.Gait/train_activity.csv"
print(train_activity)
```

개인 파일을 직접 터미널에서 실행할 수도 있지만, 팀 공통 실행 방식은 `base.ipynb`를 기준으로 맞춥니다.

## Git 사용 방법

작업 전 최신 코드 받기:

```bash
git pull origin main
```

변경 확인:

```bash
git status
```

자기 폴더만 add:

```bash
예: git add Sanghyo/
```

커밋:
커밋 메세지는 각자 알아서 적읍시다, 나중에 로그같은거 찾기 쉬우려면 간략한 정보정도는 언급하면 편할지도?

```bash
git commit -m "Add Sanghyo experiment script"
```

푸시:

```bash
git push origin main
```

다른 사람 폴더는 수정하거나 커밋하지 않습니다. 실수로 다른 사람 폴더가 `git status`에 보이면 커밋 전에 반드시 제외해야 합니다.

## .gitignore 기준

아래 항목은 기본적으로 Git에 올리지 않습니다.

- `.ipynb`
- `.csv`, `.xlsx`, `.xls`
- `.pkl`, `.joblib`, `.pt`, `.pth`, `.h5`
- `outputs/`, `outputs_paper_exact/`
- `__pycache__/`
- `.DS_Store`
- `*.txt`

따라서 기존 노트북, 데이터셋, 결과물, 개인 로컬 파일이 Git에 안 보이는 것은 정상입니다. 이번 변경도 `.gitignore`를 우회해서 데이터나 결과물을 올리는 방식이 아니라, 공유 가능한 `.py` 코드 파일을 추가하는 방식으로 정리했습니다.

## 이번 정리의 목적

기존에는 사용자 폴더 안에 `.ipynb`와 `.py`가 섞여 있었습니다. 이번 정리에서는 기존 구현을 바꾸지 않고, 노트북 코드 셀을 `.py` 파일로 변환해 추가했습니다.

변환 파일에는 다음 주석이 들어 있습니다.

```python
# Auto-generated Python script converted from a Jupyter notebook.
# Source notebook: ...
```

원본 노트북과 변환된 `.py` 파일의 대응 관계는 각 사용자 폴더의 `README.md`에 정리되어 있습니다.

