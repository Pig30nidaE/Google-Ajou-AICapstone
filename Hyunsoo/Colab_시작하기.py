# Auto-generated Python script converted from a Jupyter notebook.
# Source notebook: Hyunsoo/Colab_시작하기.ipynb
# Do not edit this generated file if you need exact notebook parity; edit the source notebook or copy this file first.

# Notebook compatibility helpers. Generated to keep notebook shell/magic cells runnable as Python.
import os as _NOTEBOOK_OS
import subprocess as _NOTEBOOK_SUBPROCESS
from pathlib import Path as _NOTEBOOK_PATH


def _NOTEBOOK_RUN_SHELL(command: str) -> None:
    _NOTEBOOK_SUBPROCESS.run(command, shell=True, check=True)


def _NOTEBOOK_RUN_BASH(script: str) -> None:
    _NOTEBOOK_SUBPROCESS.run(script, shell=True, executable="/bin/bash", check=True)


def _NOTEBOOK_CD(path: str) -> None:
    _NOTEBOOK_OS.chdir(_NOTEBOOK_OS.path.expanduser(path))
    print(_NOTEBOOK_PATH.cwd())


# %% [markdown] cell 1
# # Colab에서 Codex 사용하기
#
# 이 노트북은 GitHub clone 없이 Colab 런타임 안에서 Codex CLI를 실행하는 흐름입니다. API key를 쓰지 않고 `codex login --device-auth`로 ChatGPT 계정에 로그인합니다.

# %% cell 2
# 1. Codex CLI 설치
_NOTEBOOK_RUN_SHELL('curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=1 sh')

# %% cell 3
# 2. PATH 등록 및 설치 확인
import os
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]

_NOTEBOOK_RUN_SHELL('codex --version')

# %% cell 4
# 3. ChatGPT 계정으로 로그인
# 출력되는 URL을 열고, 표시되는 코드를 입력하세요.
# 유료 ChatGPT 플랜을 쓰고 있다면 API key 대신 이 방식을 먼저 사용합니다.
_NOTEBOOK_RUN_SHELL('codex login --device-auth')

# %% cell 5
# 4. GitHub 없이 Colab 로컬 작업 폴더 만들기
# Codex는 기본적으로 Git 저장소 안에서 실행하는 것을 선호하므로,
# 원격 GitHub 없이 로컬 git만 초기화합니다.
_NOTEBOOK_CD('/content')
_NOTEBOOK_RUN_SHELL('mkdir -p codex-workspace')
_NOTEBOOK_CD('/content/codex-workspace')
_NOTEBOOK_RUN_SHELL('git init')

# %% cell 6
# 5. 예제 파일 만들기 또는 기존 파일 업로드 후 이 폴더에 두기
from pathlib import Path

Path("hello.py").write_text(
    "def greet(name):\n"
    "    return f\"Hello, {name}!\"\n"
    "\n"
    "print(greet(\"Colab\"))\n",
    encoding="utf-8",
)

_NOTEBOOK_RUN_SHELL('ls -l hello.py')
_NOTEBOOK_RUN_SHELL('python hello.py')

# %% cell 7
# 6. Codex에게 현재 폴더 분석 맡기기
_NOTEBOOK_RUN_SHELL('codex exec --sandbox workspace-write "현재 폴더 구조를 보고 어떤 파일이 있는지 요약해줘"')

# %% cell 8
# 7. Codex에게 코드 수정/실행 맡기기
_NOTEBOOK_RUN_SHELL('codex exec --sandbox workspace-write "비 개발자도 이해가되게끔 주석 달아줘."')

# %% cell 9

