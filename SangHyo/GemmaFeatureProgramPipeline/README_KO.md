# GemmaFeatureProgramPipeline

Gemma가 환자를 직접 평가하지 않고, 고정 wearable primitive로부터 실행 가능한
feature program을 **한 번** 만드는 CN 대 MCI+Dem 이진 분류 실험이다.

기존 `GeminiFeaturePipeline`의 환자별 12개 주관 점수 대신 다음 흐름을 사용한다.

```text
논문 요약 + 고정 primitive catalogue
                  |
                  v
       Gemma global JSON program
                  |
        strict schema/DSL validation
                  |
                  v
 fold-local impute/winsor/scale -> Python composite 실행
                  |
                  v
 MMSE / wearable / program block별 inductive rank model
                  |
                  v
 inner-OOF conservative late fusion -> repeated outer OOF
```

상세 감사와 논문별 채택 근거는 [ANALYSIS_KO.md](./ANALYSIS_KO.md)에 있다.

## 핵심 계약

- 과제: Training 141명, CN=0, MCI 또는 Dem=1
- historical Validation 33명의 feature/label을 열지 않음
- Gemma 입력에 patient value, ID, label, diagnosis, MMSE, 기존 AUC가 없음
- Gemma는 임의 코드를 만들지 않고 제한된 JSON DSL만 생성
- imputation, winsorization, scaling, constant-column 제거는 fold-training만 사용
- score rank는 outer/inner training reference CDF만 사용
- test batch 안에서 rank-normalize하지 않음
- 0.92는 목표이지 코드가 보장하는 결과가 아님

## Colab 설정

`base.ipynb` 셀 2:

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "GemmaFeatureProgramPipeline/run.py"

from google.colab import userdata
import os

os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
os.environ["GFPP_GEMINI_MODEL"] = "gemma-4-31b-it"
os.environ["GFPP_ARGS"] = "--stage all --profile full"
```

권장 방식은 프로그램 생성과 학습을 분리하는 것이다.

```python
# 1. 최초 1회: label-free feature program 생성
os.environ["GFPP_ARGS"] = "--stage program"

# 2. 생성된 program/cache를 고정한 뒤 학습
os.environ["GFPP_ARGS"] = "--stage train --offline --profile full"
```

`--regenerate-program`은 기존 결과를 본 뒤 프로그램을 바꾸는 실험 선택을 만들 수
있다. 명시적인 새 사전등록 실험이 아니면 사용하지 않는다.

## 로컬/쉘 진입점

```bash
python SangHyo/GemmaFeatureProgramPipeline/run.py --stage all --profile standard
```

주요 옵션:

- `--stage inspect|program|train|all`
- `--profile smoke|standard|full`
- `--offline`: 캐시된 program만 사용
- `--regenerate-program`: 같은 prompt/model의 캐시도 무시하고 새 program 생성
- `--data-root`, `--output-dir`, `--cache-dir`, `--run-id`
- `--skip-install`: 누락 dependency를 자동 설치하지 않고 즉시 실패

`smoke`는 배선 확인 전용이며 성능으로 보고하지 않는다.

## 주요 산출물

- `RUN_CONFIG.json`
- `LAUNCHER_STATUS.json`
- `DATA_AUDIT.json`
- `FEATURE_PROGRAM.json`
- `FEATURE_PROGRAM_MANIFEST.json`
- `SPLIT_REGISTRY.json`
- `OOF_PREDICTIONS_HASHED.csv`
- `FINAL_REPORT.json`
- 가능한 경우 `deployment/` final bundle

1차 성능은 `FINAL_REPORT.json`의 repeat별 OOF ROC-AUC 평균이다. 사람별 평균
OOF vector AUC는 별도의 secondary estimand다.

## API 설정

기본값:

```yaml
model: gemma-4-31b-it
max_output_tokens: 8192
thinking_level: minimal
thinking_budget: null
```

API key 값은 로그, prompt, cache, 결과 파일에 기록하지 않는다. 환경변수 이름만
설정 파일에 남는다.
