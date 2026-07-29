# GeminiFeaturePipeline

웨어러블 라이프로그를 Gemini에 넣어 **진단 중립적인 구조화 수치 특징 12개**를 받고,
그 특징을 기존 수치 특징과 결합해 **별도의 머신러닝 분류기**가 CN vs MCI+Dem을 예측하는
파이프라인이다. Gemini는 진단하지 않고, 클래스 확률도 내지 않으며, 타깃 레이블과 MMSE를
절대 보지 않는다.

> **현재 상태**: 코드 작성만 완료했다. Gemini API는 아직 한 번도 호출되지 않았고,
> 모델은 아직 학습되지 않았으며, 성능은 아직 측정되지 않았다. 이 README에는 실측 성능
> 수치가 없다. 실제 실행은 사용자가 Google Colab Pro+에서 수행한다.

## 1. 두 가지 파이프라인

| 모드 | Gemini 입력 | downstream 분류기 입력 |
| --- | --- | --- |
| `--mmse-mode without` | 웨어러블 요약만 | BASE(웨어러블 35개) [+ Gemini 12개] |
| `--mmse-mode with` | **동일하게 웨어러블 요약만** | 위 + MMSE 37개 |

MMSE는 기본적으로 **downstream 분류기에만** 추가한다. 두 모드는 같은 subject-wise
split, 같은 평가 코드, 같은 Gemini 특징을 쓰므로 차이는 MMSE 블록 하나뿐이다.
`--mmse-mode both`(기본값)이면 두 모드를 한 번에 실행한다.

비교되는 arm: `{without, with} x {base, base_gemini} x {logreg, gbdt}` = 최대 8개.

## 2. 파일 구성

```
SangHyo/GeminiFeaturePipeline/
├── run.py                # 단일 엔트리 포인트 (CLI + base.ipynb 런처 대응)
├── config.py / config.yaml
├── data.py               # Activity/Sleep -> subject-day 테이블, 라벨/MMSE allow-list 로딩
├── payload.py            # 피험자별 Gemini payload 생성 (통계 + 축약 시계열)
├── guards.py             # 타깃/진단/MMSE 차단 가드
├── schema.py             # Gemini JSON Schema + 특징 메타데이터 + 응답 검증
├── prompts.py            # 시스템 프롬프트 + 사용자 프롬프트 템플릿
├── gemini_client.py      # API 호출, 재시도, rate limit, 캐시, dry-run/offline
├── features.py           # BASE / Gemini / MMSE 블록과 설계 행렬 조립
├── splits.py             # 사람 단위 반복 StratifiedKFold + split registry
├── models.py             # logreg, gbdt (fold-local 전처리 파이프라인)
├── evaluation.py         # 지표, OOF, arm 비교, paired bootstrap
├── pipeline.py           # 단계 오케스트레이션
├── requirements_colab.txt
├── docs/
│   ├── DATA_STRUCTURE_KO.md      # 실제 데이터 구조 분석
│   ├── BASE_NOTEBOOK_FLOW_KO.md  # base.ipynb 흐름 분석과 이관 내역
│   ├── PAPERS_KO.md              # papers/LLM_API 논문 분석
│   └── GEMINI_FEATURES_KO.md     # 특징 12개 정의
└── tests/                # 학습·API 없이 도는 계약 테스트
```

## 3. Colab Pro+ 실행 절차

1. 코드를 커밋·푸시한다. `base.ipynb` 셀 1은 `origin/main`을 새로 clone하므로
   **푸시하지 않은 수정은 실행되지 않는다.**
2. 노트북 셀 1을 실행한다(Drive 마운트 + clone).
3. 셀 2를 아래처럼 바꾼다.

   ```python
   import os
   USER_FOLDER = "SangHyo"
   RUN_FILE = "GeminiFeaturePipeline/run.py"

   from google.colab import userdata
   os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")   # 키는 환경변수로만
   os.environ["GFP_ARGS"] = "--stage all --mmse-mode both"          # 생략 가능(기본값과 동일)
   ```

4. 셀 3, 4, 5를 순서대로 실행한다. 셀 5가 `run.py`를 실행하고, `run.py`가
   `requirements_colab.txt`를 설치한 뒤 전체 파이프라인을 돌린다.
5. 처음에는 비용 없이 배선만 확인하는 것을 권장한다.

   ```python
   os.environ["GFP_ARGS"] = "--stage gemini --dry-run"
   ```

런타임: 표 모델만 쓰므로 **CPU / High-RAM**으로 충분하다. GPU는 필요 없다.

## 4. 셸에서 실행할 때

```bash
python run.py --config config.yaml --stage all --mmse-mode both
```

단계별 실행도 같은 엔트리 포인트를 쓴다.

```bash
python run.py --config config.yaml --stage inspect
```
```bash
python run.py --config config.yaml --stage payload
```
```bash
python run.py --config config.yaml --stage gemini --dry-run
```
```bash
python run.py --config config.yaml --stage gemini
```
```bash
python run.py --config config.yaml --stage train --mmse-mode without
```
```bash
python run.py --config config.yaml --stage all --mmse-mode with
```

자주 쓰는 옵션

| 옵션 | 의미 |
| --- | --- |
| `--dry-run` | payload만 만들고 API를 호출하지 않음(요청 수·크기·저장 경로만 출력) |
| `--offline` | 캐시에 있는 응답만 사용, 캐시 미스는 호출이 아니라 오류 |
| `--retry-failed` | 이전에 실패로 캐시된 피험자만 다시 호출 |
| `--limit-subjects N` | 앞에서 N명만 처리(부분 실행) |
| `--no-gemini` | BASE만으로 학습(Gemini 없이 배선 확인) |
| `--feature-sets base,base_gemini` | 비교 조합 지정 |
| `--models logreg,gbdt` | 모델 지정 |
| `--data-root / --output-dir / --cache-dir` | 경로 재지정 |

`--stage evaluate`는 같은 실행 안에서 `train` 다음에만 의미가 있다(메모리의 arm 결과를
집계한다). 단독으로 호출하면 명시적 오류를 낸다.

## 5. 환경변수

| 변수 | 용도 | 기본값 |
| --- | --- | --- |
| `GEMINI_API_KEY` | **필수(gemini 단계)**. 키 자체는 코드·설정에 절대 넣지 않는다 | 없음 |
| `GFP_ARGS` | base.ipynb 실행 시 CLI 인자 문자열 | `--stage all --mmse-mode both` |
| `GFP_DATA_ROOT` | 데이터 루트 | 노트북 `DATA_ROOT` → `<repo>/Data` |
| `GFP_OUTPUT_ROOT` | 결과 루트 | Colab `/content/drive/MyDrive/GeminiFeaturePipeline_result` |
| `GFP_CACHE_ROOT` | Gemini/일간테이블 캐시 | Colab `/content/drive/MyDrive/GeminiFeaturePipeline_cache` |
| `GFP_GEMINI_MODEL` | 모델명 | `gemini-2.5-flash-lite` |
| `GFP_GEMINI_API_KEY_ENV` | 키를 담은 환경변수 **이름** | `GEMINI_API_KEY` |
| `GFP_GEMINI_MAX_CONCURRENCY`, `GFP_GEMINI_MIN_INTERVAL`, `GFP_GEMINI_MAX_RETRIES`, `GFP_GEMINI_TIMEOUT`, `GFP_GEMINI_LIMIT_SUBJECTS` | 호출 동시성/요청 제한/타임아웃 | config.yaml 값 |
| `GFP_CV_SPLITS`, `GFP_CV_REPEATS`, `GFP_SEED`, `GFP_RUN_ID`, `GFP_N_JOBS`, `GFP_MMSE_MODE` | 실행 파라미터 | config.yaml 값 |

우선순위는 **CLI > 환경변수 > config.yaml > 코드 기본값**이다.

## 6. 생성되는 파일

`<output_root>/<UTC_RUN_ID>/` 아래에 매 실행마다 새로 생성된다.

| 파일 | 내용 |
| --- | --- |
| `LAUNCHER_STATUS.json` | `starting` / `complete` / `failed`, 산출물 목록 |
| `RUN_CONFIG.json` | 설정 스냅샷, python/platform, schema·prompt 해시 (키는 저장하지 않음) |
| `DATA_AUDIT.json` | 분할별 피험자·subject-day 수, 정렬 감사, 라벨 파일 SHA-256, 진단 분포, MMSE 접근 정책 |
| `PAYLOAD_REPORT.json` | payload 수, 바이트 크기 통계, 가드 통과 여부 |
| `payloads/payloads_<split>.json` | 실제 전송 payload (subject_ref 해시 키) |
| `GEMINI_REPORT.json` | 캐시/신규/실패/토큰 사용량, 프롬프트·스키마·generation config 전문 |
| `gemini_features_<split>.csv` | `subject_hash` + 12개 특징 |
| `split_registry.json` | fold별 학습/검증 subject **해시** 목록과 클래스 수 |
| `TRAINING_REPORT.json` | 설계 행렬 구성, 결측 리포트, arm별 fold 기록과 지표 |
| `oof_predictions_hashed.csv` | `subject_hash, y_true, oof__<arm_id>...` |
| `FINAL_REPORT.json` | arm 비교표, BASE vs BASE+Gemini paired 차이, 주의사항 |

Gemini 캐시는 run 디렉터리 **밖**(`<cache_root>/gemini/<model>/<schema_hash>/<request_hash>.json`)에
저장되어 다음 실행에서 재사용된다. 캐시 레코드에는 입력 payload, 원본 응답, 파싱 결과,
시도 횟수, 토큰 사용량, 해시가 모두 들어 있다.

`FINAL_REPORT.json`의 `comparison_table` 컬럼:
`arm_id, mmse_mode, feature_set, model, n_features, roc_auc_pooled_oof,
roc_auc_repeat_mean, roc_auc_repeat_sd, pr_auc, balanced_accuracy,
recall_sensitivity, specificity, mcc`.

## 7. 누수 방지 계약

1. Gemini 요청 직전 payload 전체를 `assert_payload_is_label_free` +
   `assert_payload_is_mmse_free`로 검사한다(키·문자열 값·중첩 구조 모두).
2. payload에는 식별자, 절대 날짜, 진단, MMSE가 구조적으로 존재하지 않는다.
3. 프롬프트와 JSON 스키마에 진단 클래스명·검사명이 들어가면 **import 시점에** 실패한다.
4. Gemini 특징 정의는 라벨을 보지 않고 사전 고정했다(논문의 라벨 기반 rubric 합성은 거부).
5. 캐시 키는 payload+프롬프트+스키마+모델+generation config의 해시이며, 스키마나
   프롬프트가 바뀌면 이전 캐시는 자동 무효화된다.
6. 학습 단위는 사람 1명 = 1행이고, fold마다 subject 교집합이 없음을 단언한다.
7. imputation/scaling은 fold 내부에서만 fit된다(sklearn Pipeline).
8. MMSE 미사용 모드는 설계 행렬 이름에 MMSE 계열이 하나라도 있으면 실패한다.
9. `without`/`with`, `base`/`base_gemini`는 **같은 split plan 객체**를 공유한다.
10. 결과에는 원본 이메일 대신 salt 기반 SHA-256 해시만 저장한다.

## 8. 평가 설계와 그 이유

* 사람 단위 반복 StratifiedKFold(기본 5 fold x 5 repeat). 한 사람이 한 행이므로
  이것이 곧 subject-wise 분할이다.
* **non-nested**를 선택했다. 이번 단계에는 탐색·특징선택·threshold 적합이 전혀 없어
  inner loop가 보호할 대상이 없기 때문이다. 튜닝을 도입하는 순간 nested로 바꿔야 한다.
* 지표: ROC-AUC(주), PR-AUC, recall(민감도), specificity, F1, balanced accuracy, MCC.
  Accuracy는 all-negative baseline과 함께만 보고한다.
* BASE 대비 BASE+Gemini의 차이는 동일 피험자·동일 fold에서 paired bootstrap 구간과 함께
  보고한다. 구간이 0을 포함하면 개선으로 선언하지 않는다.

## 9. 현재 구현의 한계

1. **아직 아무것도 실행되지 않았다.** 성능은 미측정이며, Gemini가 이 payload에서
   유용한 신호를 만들어낼지는 실행 전에는 알 수 없다.
2. 저장소의 기존 결과에 따르면 웨어러블-only는 사람 단위 OOF에서 대체로 0.45~0.57,
   MMSE 포함은 0.67~0.77이다(`SangHyo/AGENTS.md` 3절). Gemini 특징이 이 범위를
   크게 바꿀 것이라고 가정하지 않는다.
3. Gemini 특징은 API 응답이므로 완전한 결정성이 보장되지 않는다. temperature 0.0과
   seed를 고정하고 캐시를 강제해 한 실행 안의 재현성만 확보한다.
4. 스키마를 라벨 없이 사전 고정했기 때문에 이 표본에 최적화되어 있지 않다.
5. 33명 Validation은 점수화하지 않는다. 따라서 외부 일반화 근거는 이번 산출물에 없다.
6. 하이퍼파라미터 튜닝은 없다. `tuning.enabled: true`로 두면 설정 검증 단계에서 막힌다.
7. LightGBM이 없으면 `HistGradientBoostingClassifier`로 대체된다(리포트에 기록됨).
8. Gemini 특징이 없는 피험자가 한 명이라도 있으면 `base_gemini` arm은 실행을 거부한다.
   평균값으로 조용히 채우지 않는다.

## 10. 앞으로의 확장 지점

* **튜닝**: `tuning` 섹션은 인터페이스만 있다. 켜려면 먼저 `evaluation.py`를
  nested CV로 바꾸고, 탐색을 outer 학습 fold 안으로 넣어야 한다.
* **특징 세트 추가**: `features.feature_sets`에 `gemini_only`가 이미 지원된다.
  전통적 고급 특징을 추가하려면 새 블록 빌더를 만들고 `assemble_design_matrix`에
  블록으로 넘기면 된다.
* **MMSE를 Gemini 입력에 넣는 ablation**: 의도적으로 미구현. 넣으려면 payload 빌더에
  별도 경로를 만들고 해당 경로에서만 MMSE 가드를 해제해야 한다(문서화 필수).
* **응답 변동성 점검**: `gemini.repeat_calls`를 늘리면 같은 입력에 대한 반복 호출을
  지원할 수 있다(현재 기본 1, 이번 단계에서는 비권장).
* **Validation 33명 예측**: `data.splits`에 `val`을 추가하면 payload와 Gemini 특징까지는
  생성·캐시된다. 점수화는 별도 결정이 필요하다.

## 11. 사용자에게 필요한 추가 정보

1. Colab에서 사용할 **Gemini 모델명**과 호출 한도(분당 요청 수). 기본값은
   `gemini-2.5-flash-lite`(무료 티어 `gemini-2.5-flash`가 5 req/min 한도에 걸려
   전환, 2026-07-29 실측), 동시성 1, 최소 간격 13초다. 유료 티어라면 올려서 쓴다.
   `gemini-1.5-flash`/`gemini-2.0-flash(-lite)`는 이미 서비스 종료(404)되었고,
   2.5 시리즈도 2026-10-16 종료 예정이니 장기 사용 시 재확인이 필요하다.
2. 비용 추적이 필요하면 `gemini.price_per_million_input_tokens` /
   `price_per_million_output_tokens`에 실제 단가를 넣어야 리포트에 비용 추정이 나온다.
3. Drive의 결과·캐시 경로를 기본값과 다르게 쓸지 여부.
4. 이번 단계에서 33명 Validation 예측 파일을 만들어 둘지 여부(점수화는 하지 않더라도).
5. `Binary_MMSE_MaxAUC`(OOF 0.7658)를 공식 baseline 표에 함께 실을지 여부.

## 12. 재사용한 기존 코드

* `SangHyo/Binary_Wearable_SequenceFusion_Google/data.py`: Activity/Sleep 일자 정렬 규칙,
  슬래시 시계열 파서, 비착용 마스킹, 안전 컬럼 목록의 개념을 차용했다. 변경점은
  상대 day_index/주말 플래그 유지, 채널 축소, 절대 날짜 비노출이다.
* `SangHyo/Binary_Google_ROCAUC_Champion/data.py`: MMSE allow-list와 라벨 사본
  교차검증 방식을 차용했다.
* `SangHyo/Binary_MMSE_MaxAUC/features.py`: MMSE 도메인 구성을 차용했다. 변경점은
  `item_max`를 데이터에서 학습하지 않고 상수 2.0으로 고정하고,
  `num_failed`/`recall_deficit`(이 데이터에서 `TOTAL`의 정확한 아핀 변환)을 제거한 것이다.
* `SangHyo/Codex_Dementia_ROCAUC/run.py`: `runpy` 실행 시 패키지 경로 복구와 Jupyter
  인자 제거 방식을 차용했다.

기존 폴더의 파일은 하나도 수정하거나 삭제하지 않았다.
