# Binary_PooledMaxAUC — ROC-AUC 최대화 실험

과제: `CN = 0` vs `MCI 또는 Dem = 1`

> **상태: 코드만 작성됨. 아직 한 번도 실행되지 않았습니다.**
> 이 문서에는 실측 성능 수치가 없습니다. 모든 성능은 사용자가 Colab에서
> `run.py`를 실행한 뒤 `FINAL_REPORT.json`에서 확인해야 합니다.

## 0. 요청 조건과 그 해석

요청은 두 가지였습니다.

1. `run.py` 단일 엔트리포인트
2. **직접적인** 데이터 누수만 회피 (그 외에는 수단·방법 무관)

그래서 이 실험은 **선택 낙관(selection optimism)을 의도적으로 허용**하고,
**직접 누수는 코드로 강제 차단**합니다. 두 가지를 뒤섞지 않는 것이 이 폴더의
핵심 설계입니다.

| 구분 | 이 실험에서의 처리 |
| --- | --- |
| 같은 사람이 fold 양쪽에 존재 | **금지** (`assert_fold_disjoint`) |
| held-out 라벨이 imputation/scaling/screening/fit에 관여 | **금지** (fold-local + `assert_screening_is_train_local`) |
| 진단·행정·식별 컬럼이 특징으로 유입 | **금지** (`assert_no_forbidden_features`) |
| 관측일수·coverage·non-wear 같은 수집량 proxy | **금지** (정규식 가드) |
| held-out 배치 자체의 순위로 점수 정규화(transductive) | **금지** (학습 fold ECDF 기준) |
| 후보·top_k·앙상블 가중치를 보고 대상 OOF에서 선택 | **허용, 전부 공시** |
| Training 141 + Validation 33 풀링 | **허용, 전부 공시** |

## 1. 0.85 목표에 대한 정직한 전망

**0.85는 보장할 수 없습니다.** 근거는 이 저장소의 실측 기록입니다.

| 기준 | ROC-AUC | 비고 |
| --- | ---: | --- |
| 직접 누수 없는 141명 최고 | 0.7834 | `Binary_Google_YDF_AUC` (non-nested) |
| 직접 누수 없는 141명 차점 | 0.7817 | `Binary_Gemma_CognitiveFeature_AUC` (non-nested) |
| 정식 nested 최고 | 0.7658 | `Binary_MMSE_MaxAUC` |
| non-nested 낙관 폭(실측) | +0.053 ~ +0.084 | `OrdinalStable`, `MaxAUC_Tuned` |
| Taehyun `V41` | 약 0.84 | 174명 풀링 + **전역 SHAP 선택** + **동일-fold Optuna** |

V41의 0.84 중 재현 가능한 정당한 부분은 **174명 풀링뿐**이고, 나머지 두 요소는
직접 누수라 이 실험에서는 쓰지 않습니다. 따라서 현실적 기대는 **0.80~0.84**이며,
0.85는 도달하면 좋은 목표로 설정했을 뿐 코드가 그 수치를 만들어내도록 강제하지
않습니다. `FINAL_REPORT.json`의 `target.reached`가 그대로 사실을 보고합니다.

## 2. AUC를 끌어올리는 정당한 레버 (설계 근거)

1. **174명 풀링** (`data.splits: [train, val]`) — 학습 데이터 +23%. 직접 누수는
   아니지만 남은 독립 hold-out이 사라집니다. `--splits train`으로 141명 모드로
   되돌리면 기존 실험들과 직접 비교 가능해집니다.
2. **뷰(view) 기반 탐색** — MMSE가 신호를 지배하고 넓은 wearable 뱅크는 반복적으로
   실패했다는 근거(`BINARY_EXPERIMENTS_ANALYSIS_KO.md`)를 반영해, 하나의 거대
   행렬 대신 `mmse_core`(39) / `mmse_plus`(+EDA 상호작용) /
   `mmse_wear_small`(+엄선 wearable) / `all`(전체) 중 OOF가 고르게 했습니다.
3. **EDA 기반 MMSE 상호작용** — 보고된 최강 단일 신호는 TOTAL(0.695)이 아니라
   recall+attention 조합(0.755)이었습니다. `recall_plus_q12_5`,
   `orient_minus_recall`, `focal_failure`(천장 근처 총점 + 국소 도메인 실패) 등을
   추가했습니다.
4. **폭넓은 learner 후보** — LR/SVM/ExtraTrees/RF/HGB/LightGBM/CatBoost/XGBoost/
   YDF sparse-oblique. 라이브러리가 없으면 해당 후보만 빠지고 **다른 모델로 조용히
   대체하지 않습니다**(대체하면 저장된 챔피언이 재현 불가).
5. **반복 CV + subject-mean 집계** — 사람별 OOF 점수를 반복 간 평균한 뒤 AUC를
   계산합니다. 저장소가 이미 primary로 쓰는 방식이며 분산을 줄입니다.
6. **rank(ECDF) 블렌드 앙상블** — 학습 fold ECDF 기준으로 정규화한 뒤 simplex
   가중치를 탐색합니다. 서로 다른 스케일의 모델을 결합할 수 있습니다.

## 3. 실행 방법

### Colab (`base.ipynb`) — 셀 2만 수정

```python
import os
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_PooledMaxAUC/run.py"

# 선택: 생략하면 --stage all
os.environ["BPM_ARGS"] = "--stage all --profile default"
```

셀 3·4·5를 실행하면 `run.py`가 `requirements_colab.txt`를 설치한 뒤 전체
파이프라인을 돌립니다. 표 모델만 쓰므로 **CPU / High-RAM**으로 충분하며 GPU는
필요 없습니다.

### 셸

```bash
python run.py --config config.yaml --stage all
```

단계별 실행도 같은 엔트리포인트를 씁니다.

```bash
python run.py --config config.yaml --stage audit
```
```bash
python run.py --config config.yaml --stage features
```
```bash
python run.py --config config.yaml --stage search --profile fast
```

기존 실험과 **직접 비교 가능한** 141명 모드로 돌리려면:

```bash
python run.py --config config.yaml --stage all --splits train
```

### 예상 실행 시간

후보 수 x fold x repeat가 곱해지므로 결코 짧지 않습니다. Colab에서 모든 선택
라이브러리가 설치된 경우 후보는 약 350개이고, `--profile default`(5 fold x 10
repeat = 50 fit/후보)이면 **약 17,500회 fit**입니다.

| profile | repeat | 대략적 소요 | 용도 |
| --- | ---: | --- | --- |
| `fast` | 3 | 30분~1시간 | 배선 확인, 첫 실행 권장 |
| `default` | 10 | 2~4시간 | 정식 결과 |
| `max` | 20 | 5~8시간 | 분산까지 줄이고 싶을 때 |

느리면 `--families logreg,svm_rbf,lightgbm,ydf_oblique`나
`--views mmse_plus,mmse_wear_small`로 탐색 공간을 줄이십시오.

주요 옵션

| 옵션 | 의미 |
| --- | --- |
| `--profile fast\|default\|max` | 반복 수 3 / 10 / 20 |
| `--splits train,val` \| `train` | 174명 풀링(기본) 또는 141명 |
| `--views mmse_core,mmse_plus,...` | 탐색할 특징 뷰 |
| `--families logreg,lightgbm,...` | 후보 learner 계열 |
| `--no-ensemble` | 단일 모델만 비교 |
| `--data-root / --output-dir / --cache-dir` | 경로 재지정 |

## 4. 생성되는 파일

`<output_root>/<UTC_RUN_ID>/` 아래에 매 실행마다 새로 생성됩니다.

| 파일 | 내용 |
| --- | --- |
| `LAUNCHER_STATUS.json` | `starting`/`complete`/`failed` |
| `RUN_CONFIG.json` | 설정 스냅샷, python/platform |
| `DATA_AUDIT.json` | 풀링 코호트 감사, 분할별 파일 SHA-256, 진단 분포 |
| `FEATURE_MANIFEST.json` | 블록·뷰별 특징 수와 전체 컬럼 목록 |
| `CANDIDATE_RESULTS.json` | 모든 후보의 subject-mean / repeat별 AUC, 실패 사유 |
| `FINAL_REPORT.json` | 챔피언, bootstrap CI, 리더보드 top-25, 목표 도달 여부, caveats |
| `LEAKAGE_AUDIT.json` | 직접 누수 검사 결과 + **허용된 낙관 공시 목록** |
| `oof_predictions_hashed.csv` | `subject_hash, split_origin, y_true, oof__*` |

## 5. 성능 해석 규칙 (반드시 함께 읽을 것)

1. 이 실험의 headline은 **개발 점수(development score)**입니다. 후보·top_k·앙상블
   가중치를 보고 대상 OOF에서 골랐으므로 새 코호트 일반화 추정치가 아닙니다.
2. **174명 결과를 141명 실험들(0.7834 등)과 나란히 놓고 "더 높다"고 말할 수
   없습니다.** 표본이 다릅니다. 비교하려면 `--splits train`으로 다시 돌리십시오.
3. MMSE 포함 성능은 진단에 쓰인 인지검사를 다시 사용한 것이라 부분적으로
   순환적입니다. **"웨어러블만으로 스크리닝한 성능"으로 표현하면 안 됩니다.**
4. 다음으로 유효한 확인은 같은 OOF에서 후보를 더 고르는 것이 아니라, 이번에
   선택된 구성을 **고정한 채** 새 split seed 또는 새 피험자에서 평가하는 것입니다.

## 6. 알려진 한계

- 174명 풀링으로 남은 독립 hold-out이 없습니다. 이 실행 이후 33명 Validation은
  더 이상 어떤 의미로도 "미사용 test"가 아닙니다.
- non-nested 선택 낙관이 남아 있고, 저장소 실측 기준 그 폭은 +0.05~+0.08입니다.
  `honest_nested_comparison: true`로 두면 비교 arm을 함께 계산할 수 있게
  인터페이스를 열어두었으나, 현재 구현은 그 훅만 있고 별도 nested 재탐색 로직은
  구현하지 않았습니다(추후 확장 지점).
- Dem 12명은 상대적으로 쉬운 양성이라 전체 AUC를 끌어올립니다. 어려운
  **CN vs MCI** 경계 성능은 별도로 봐야 하며 이번 리포트에는 포함되지 않습니다.
- 후보 수 x fold x repeat가 크므로 `--profile max`는 오래 걸립니다. 먼저
  `--profile fast`로 배선을 확인하는 것을 권장합니다.

## 7. 재사용한 기존 코드

- `SangHyo/Binary_Wearable_SequenceFusion_Google/data.py` — Activity/Sleep 일자
  정렬 규칙, 슬래시 시계열 파서, 비착용 마스킹 (`SangHyo/GeminiFeaturePipeline/data.py`
  의 검증된 포팅을 경유). 변경점: 두 분할 풀링, `split_origin` 유지(감사 전용).
- `SangHyo/Binary_Google_ROCAUC_Champion/data.py` — MMSE allow-list와 라벨 사본
  교차검증 방식.
- `SangHyo/Binary_MMSE_MaxAUC/features.py` — 39개 MMSE 앵커 구성. 변경점:
  `item_max`를 데이터에서 학습하지 않고 상수 2.0으로 고정(문항 코딩이
  1=오답/2=정답이고 `TOTAL == count(item==2)`임을 141/141명에서 검증).
- `SangHyo/Binary_Google_YDF_AUC` — sparse-oblique GBT 하이퍼파라미터와
  training-reference ECDF 앙상블 방식.
- `SangHyo/Codex_Dementia_ROCAUC/run.py` — `runpy` 실행 시 패키지 경로 복구와
  Jupyter 인자 제거 방식.

기존 폴더의 파일은 하나도 수정하지 않았습니다.
