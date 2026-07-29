# `base.ipynb` 실행 흐름 분석과 `run.py`로의 이관

## 1. `base.ipynb`는 학습 파이프라인이 아니라 런처다

저장소 루트의 `base.ipynb`는 5개 셀로 구성되어 있고, 데이터 로딩·전처리·모델·평가 코드를
포함하지 않는다. 실제 흐름은 다음과 같다.

| 셀 | 하는 일 | 이 파이프라인에 주는 제약 |
| --- | --- | --- |
| 1 | Colab이면 Drive 마운트 → `/content/Google-Ajou-AICapstone`의 기존 checkout 삭제 → `origin/main`을 `--depth 1`로 새로 clone → `PROJECT_ROOT`, `DATA_ROOT` 결정 | **커밋·푸시되지 않은 코드는 실행되지 않는다.** `DATA_ROOT`는 Colab에서 `/content/drive/Shareddrives/GoogleAI_contest/Data`, 로컬에서 `<repo>/Data` |
| 2 | `USER_FOLDER`, `RUN_FILE` 두 값만 수정 | 사용자가 손대는 유일한 셀. 환경변수도 여기에 넣어야 한다 |
| 3 | `USER_ROOT`, `RUN_PATH` 확인 | 실행 파일은 개인 폴더 기준 상대경로 |
| 4 | 개인 폴더의 `requirements.txt`/`requirement.txt`가 있으면 설치 | 이 폴더는 `requirements_colab.txt`라는 이름을 쓰므로 **자동 설치되지 않는다** → `run.py`가 스스로 설치한다 |
| 5 | `runpy.run_path(RUN_PATH, init_globals={PROJECT_ROOT, DATA_ROOT, USER_ROOT, RUN_PATH}, run_name="__main__")`, cwd를 실행 파일 폴더로 이동, 15초 하트비트 출력 | `sys.argv`는 Jupyter 커널의 것이다. **CLI 인자를 쓸 수 없으므로 환경변수로 받아야 한다** |

노트북은 수정하지 않는다. 대신 `run.py`가 위 제약에 맞춰 동작한다.

## 2. 노트북 제약에 대한 `run.py`의 대응

1. **패키지 import**: `runpy`로 실행되면 `__package__`가 비어 상대 import가 깨진다.
   `run.py` 상단에서 저장소 루트를 `sys.path`에 넣고
   `__package__ = "SangHyo.GeminiFeaturePipeline"`를 설정한다
   (`Codex_Dementia_ROCAUC/run.py`에서 검증된 방식을 차용).
2. **인자 전달**: `PROJECT_ROOT`/`DATA_ROOT`/`USER_ROOT`/`RUN_PATH`가 주입되어 있고
   `RUN_PATH`가 이 파일이면 노트북 실행으로 판정하고, 인자를 환경변수 `GFP_ARGS`에서
   읽는다. 없으면 `--stage all --mmse-mode both`로 전체 실행한다.
   셸에서 직접 실행할 때는 Jupyter가 끼워 넣는 `-f kernel-*.json`만 제거하고
   나머지 `sys.argv`를 그대로 파싱한다.
3. **경로**: `DATA_ROOT`가 주입되어 있으면 그것을 최우선으로 쓰고, 없으면
   `PROJECT_ROOT`/현재 경로에서 `1.Training`과 `2.Validation`이 있는 디렉터리를 찾는다.
   `--data-root`나 `GFP_DATA_ROOT`가 있으면 그 값이 이긴다.
4. **의존성**: 노트북 셀 4가 `requirements_colab.txt`를 설치하지 않으므로,
   노트북 실행으로 판정되면 `run.py`가 직접
   `pip install -r requirements_colab.txt`를 실행한 뒤 import 헬스체크를 한다.
5. **출력 스트리밍**: 셀 5의 하트비트는 경과 시간만 출력하므로 단계별 진행 로그는
   `run.py`가 직접 `print(..., flush=True)`로 남긴다.

## 3. 셀 2에 넣어야 하는 코드

```python
# Cell 2 - User inputs
import os

USER_FOLDER = "SangHyo"
RUN_FILE = "GeminiFeaturePipeline/run.py"

# (1) Gemini API 키: 코드나 파일이 아니라 환경변수로만 전달한다.
from google.colab import userdata          # Colab 비밀 저장소를 쓸 때
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")

# (2) 실행 인자 (생략하면 --stage all --mmse-mode both)
os.environ["GFP_ARGS"] = "--stage all --mmse-mode both"

# (3) 선택: 경로/모델을 코드 수정 없이 바꾸고 싶을 때
# os.environ["GFP_OUTPUT_ROOT"] = "/content/drive/MyDrive/GeminiFeaturePipeline_result"
# os.environ["GFP_CACHE_ROOT"]  = "/content/drive/MyDrive/GeminiFeaturePipeline_cache"
# os.environ["GFP_GEMINI_MODEL"] = "gemini-3.6-flash-latest"
# os.environ["GFP_GEMINI_THINKING_LEVEL"] = "minimal"
# os.environ["GFP_GEMINI_MAX_OUTPUT_TOKENS"] = "8192"
```

`userdata`를 쓰지 않는다면 `os.environ["GEMINI_API_KEY"] = getpass.getpass()`처럼
입력받아도 된다. 키를 셀에 문자열로 적어 커밋하지 않는다.

## 4. 노트북에서 옮겨온 것 / 새로 만든 것

**옮겨온 것 (숨은 상태 의존성 제거)**

* 경로 결정 로직: 셀 1의 `PROJECT_ROOT`/`DATA_ROOT` 탐색 → `config.PipelineConfig.resolved_*`
  로 함수화하고, 주입값 > 환경변수 > 설정 파일 > 탐색 순서를 명시했다.
* 의존성 설치: 셀 4 → `run.ensure_dependencies()` (선언된 requirements 파일 기준,
  설치 후 import 검증까지 수행).
* 실행 진입: 셀 5의 `runpy` 호출 → `run.main()` 하나로 수렴. 셀 간 전역 변수 공유
  (`PROJECT_ROOT`가 다음 셀에서 계속 살아 있는 방식) 대신 `PipelineContext` 객체가
  단계 간 상태를 명시적으로 전달한다.

**노트북에 없어서 새로 만든 것**

* 데이터 로딩/정렬/일간 특징(기존 `Binary_Wearable_SequenceFusion_Google`의 규칙 차용)
* Gemini payload, 프롬프트, 스키마, 클라이언트, 캐시
* 사람 단위 split, downstream 모델, 평가, 결과 저장
* 누수 차단 가드와 계약 테스트

## 5. 기존 실행 방식에서 바뀐 점과 이유

| 항목 | 기존 관행 | 이 폴더 | 이유 |
| --- | --- | --- | --- |
| requirements 파일명 | 개인 폴더 `requirements.txt`면 셀 4가 설치 | `requirements_colab.txt` + `run.py` 자체 설치 | 다른 `Binary_*` 폴더와 파일명을 통일하고, 노트북을 수정하지 않기 위해 |
| 결과 경로 | `/content/drive/MyDrive/<실험명>_result/<UTC_RUN_ID>/` | 동일 | `AGENTS.md` 0절 계약 유지 |
| 캐시 | 없음 | `/content/drive/MyDrive/GeminiFeaturePipeline_cache/` (run 디렉터리 **밖**) | Gemini 응답과 파싱된 일간 테이블을 실행 간 재사용해야 재호출 비용이 0이 된다 |
| Validation 33명 | 실험마다 재평가 | 이번 단계에서는 점수화하지 않음 | `AGENTS.md` 2-5의 재사용 금지 계약 |
