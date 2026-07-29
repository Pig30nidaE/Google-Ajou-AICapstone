# Binary_Gemma_CognitiveFeature_AUC

Hosted Gemma API가 익명 MMSE 점수·문항 패턴을 10개의 연속형 인지 특징으로
변환하고, `CN=0` 대 `MCI 또는 Dem=1`의 ROC-AUC만 최적화하는 실험이다.
실행 진입점은 `run.py` 하나다.

의존성은 `.gitignore`의 `*.txt` 규칙을 피한 `requirements_colab.in`에
고정되어 있으며, `run.py`가 필요할 때 이 파일을 설치한다.

> **성능 상태:** 실제 Gemma API를 사용한 `default`/`max` 정식 run은 아직
> 수행하지 않았다. API key 없이 검증한 synthetic exact-cache smoke는 배선
> 확인용이며 새 ROC-AUC 성능으로 인용하지 않는다.

## 설계

Gemma 기본 모델은 `gemma-4-31b-it`이다. 프롬프트는 CN–MCI 경계가 특히
어렵다는 점과 delayed recall, attention/recall discordance,
orientation-memory gap, ceiling effect, focal-versus-global deficit 원리를
명시한다. 그러나 각 API 요청에는 다음 정보만 들어간다.

Hosted 호출 방식과 모델명은
[Google의 Gemma on Gemini API 공식 문서](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)
기준이다.

- MMSE total
- 6개 domain score
- 30개 문항의 correct 값
- failed-items와 recall-deficit 파생값

소스 CSV도
`SAMPLE_EMAIL`, `TOTAL`, 30개 문항만 `usecols`로 읽는다. 식별자는 API
payload를 만들기 위한 값으로 사용하지 않고 메모리 내 행 정렬에만 쓰인다.
Gemma payload·prompt의 동적 부분·persistent cache에는 subject ID, label,
diagnosis, DIAG/admin field, 절대 날짜, 관측일수·수집량이 없다. API key 값도
저장하지 않는다.

출력 JSON은 다음 10개 고정 key의 `[0,1]` 실수다.

1. `memory_specific_deficit`
2. `orientation_memory_gap`
3. `attention_recall_discordance`
4. `ceiling_adjusted_subtle_error`
5. `temporal_orientation_weakness`
6. `multi_domain_error_burden`
7. `preserved_function_with_focal_failure`
8. `mci_boundary_evidence`
9. `global_severity_evidence`
10. `evidence_reliability`

최종 category/diagnosis를 생성하지 않는다. Hosted Gemma의 structured-output
보장을 전제하지 않고, prompt에 exact JSON 계약을 넣은 뒤 Python에서 key,
type, 범위를 strict validation한다.

## 모델과 평가

- baseline ablation: raw MMSE 39-anchor
- Gemma arm: raw MMSE 39-anchor + 10 Gemma feature
- learner: L2 logistic regression, RBF-SVM, 두 score의 AUC rank blend
- 전처리: 각 CV training fold에서만 median imputation + standard scaling
- score: 각 모델의 train-fold decision score empirical CDF에 test margin을 배치
- split: 동일 repeated subject-level StratifiedKFold
- 선택: **mean repeat OOF ROC-AUC만** 사용

후보·blend 선택은 요청대로 non-nested다. 따라서 직접 누수는 없지만 선택 편향을
포함한 개발 성능이다. `FINAL_REPORT.json`에도 이 한계를 명시한다. Champion은
반드시 Gemma feature를 포함한 arm 중에서만 선택하며 baseline은 ablation이다.

## 실행

```bash
# API 호출 없이 데이터·payload 계약만 검사
python SangHyo/Binary_Gemma_CognitiveFeature_AUC/run.py \
  --stage inspect --profile smoke --data-root Data --output-dir /tmp/bgcfa_inspect

# 최초 feature 추출. GEMINI_API_KEY 필요
python SangHyo/Binary_Gemma_CognitiveFeature_AUC/run.py \
  --stage extract --data-root Data --output-dir /tmp/bgcfa_extract

# 정확한 persistent cache만 사용해 OOF 학습
python SangHyo/Binary_Gemma_CognitiveFeature_AUC/run.py \
  --stage train --offline --profile default --data-root Data \
  --output-dir /tmp/bgcfa_train

# 추출 + 학습
python SangHyo/Binary_Gemma_CognitiveFeature_AUC/run.py \
  --stage all --profile default --data-root Data --output-dir /tmp/bgcfa_all
```

Profile:

- `smoke`: 2-fold × 1-repeat, 작은 후보 grid. 배선 확인 전용
- `default`: 5-fold × 10-repeat
- `max`: 5-fold × 30-repeat, 확장 후보 grid

`--historical-eval`을 추가하면 Validation label을 열기 전에 label-free
feature와 score를 파일로 freeze하고 SHA-256을 기록한 뒤, 마지막에 historical
Validation ROC-AUC를 별도 보고한다. 이 값은 독립 외부검증으로 해석하지 않는다.

Colab `base.ipynb`에서는 다음처럼 사용할 수 있다.

```python
import os
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
os.environ["BGCFA_ARGS"] = "--stage all --profile default"

RUN_FILE = "Binary_Gemma_CognitiveFeature_AUC/run.py"
```

## 산출물

- `RUN_CONFIG.json`
- `DATA_AUDIT.json`
- `API_AUDIT.json`
- `FEATURE_MANIFEST.json`
- `GEMMA_FEATURES_HASHED.csv`
- `SPLIT_REGISTRY.json`
- `OOF_PREDICTIONS_HASHED.csv`
- `FINAL_REPORT.json`
- `LEAKAGE_AUDIT.json`
- `TRAINING_COMPLETE.json`
- `deployment/model.joblib`
- `deployment/deployment.json`
- `LAUNCHER_STATUS.json`

Cache key는 payload hash, model, prompt hash, schema hash, generation hash를 모두
묶는다. Cache 본문에는 payload 원문이나 subject 식별자를 저장하지 않는다.
결과 CSV의 subject token은 실행마다 새로 만든 비공개 random secret으로
SHA-256 처리하며, secret 자체는 저장하지 않아 실행 간 연결이 불가능하다.
