# Binary_Google_YDF_AUC

CN을 0, MCI와 Dem을 1로 합친 사람 단위 이진분류를 **Google
Yggdrasil Decision Forests(YDF)**만으로 학습하는 ROC-AUC 중심 실험입니다.
실행 진입점은 [`run.py`](run.py) 하나입니다.

> **성능 상태:** 이 폴더의 `default`/`max` 정식 run은 아직 완료되지
> 않았습니다. 따라서 이 문서는 새 ROC-AUC나 최고 기록 갱신을 주장하지
> 않습니다. `smoke`도 배선과 저장·재로딩을 확인하는 실행일 뿐 성능 근거로
> 인용할 수 없습니다.

## 1. 이전 실험에서 가져온 판단

전체 `SangHyo/` 결과를 다시 확인했을 때, 정식 subject OOF 최고 앵커는
`Binary_MMSE_MaxAUC`의 약 **0.7658**이었습니다. 반면
`Binary_Google_MaxAUC_Tuned`의 non-nested 탐색에서는 YDF sparse-oblique
단일 branch가 약 **0.7889**까지 관측됐지만, 같은 실험의 nested 결과가
낮아진 점을 고려하면 이 값에는 선택 낙관성이 있습니다.

이 폴더는 그 가설을 다음처럼 분리해 재실행합니다.

- sklearn 모델을 섞지 않고 Google YDF만 사용
- axis-aligned GBT, sparse-oblique GBT, Random Forest를 같은 OOF 틀에서 비교
- 이전 sparse-oblique 최상 설정과 작은 이웃 설정을 후보군에 포함
- 모든 후보·앙상블 선택 기준을 ROC-AUC 하나로 고정
- 사용자가 허용한 대로 최종 후보/가중치 선택은 non-nested로 수행하되,
  그 낙관성을 결과 보고서에 명시

Google YDF의 패키지와 저장 형식은 [공식 YDF 문서](https://ydf.readthedocs.io/en/stable/)
기준이며, 이 실험은 재현성을 위해 `ydf==0.16.1`을 고정합니다.

## 2. 입력과 특징 view

모든 모델 행은 한 사람에 정확히 하나입니다.

| view | 폭 | 구성 |
| --- | ---: | --- |
| `mmse39` | 39 | 기존 MaxAUC 앵커와 같은 MMSE total, 6 domain, 30 item, failed-items, recall-deficit |
| `mmse_all` | 50 | `mmse39`와 안전한 MMSE 파생값 |
| `all151` | 151 | `mmse39` 39개와 Activity/Sleep 측정 특징 112개 |

MMSE SourceData는 pandas `usecols`로 `SAMPLE_EMAIL`, `TOTAL`, 30개 문항만
엽니다. 다음 열은 특징 구성 단계에 들어갈 수 없습니다.

```text
DIAG_NM, DIAG_SEQ, DOCTOR_NM, MMSE_NUM, MMSE_KIND, EMAIL
```

`SAMPLE_EMAIL`과 웨어러블의 `EMAIL`은 사람별 join과 누수 검사에만 쓰며 모델
행렬에는 넣지 않습니다. 진단은 별도 LabelingData 사본에서 열고 Gait/Sleep
두 사본의 일치 여부를 확인한 뒤에만 target으로 변환합니다.

웨어러블 특징은 생리·활동 측정값의 평균, 표준편차, 변동계수, 수면 구조와
24시간 주기 통계입니다. **관측 일수, 행 개수, coverage, missingness,
observation count와 non-wear coverage는 모델 특징으로 쓰지 않습니다.**
기존 112-feature bank에 있던 두 관측 일수 열과 non-wear 열은 실제 활동
구성비 세 개로 대체했습니다. feature name에 이런 collection proxy나
진단·식별 token이 나타나면 즉시 실패합니다.

## 3. 직접 누수 방지 계약

- 사람을 먼저 집계한 뒤 `StratifiedKFold`로 사람 단위 분할
- 한 repeat에서 각 사람은 정확히 한 outer held-out fold에만 등장
- 결측 대치에 의존하지 않는 YDF 입력과 label-aware top-k 선택은 현재
  fold-training 사람만 사용
- correlation pruning도 현재 fold-training에서만 계산
- held-out 사람의 label은 해당 사람을 예측하는 모델·특징 선택에 사용하지 않음
- train/Validation subject ID가 겹치면 실패
- sparse-oblique 키워드를 설치된 YDF가 거부하면 axis-aligned 모델로 낮춰
  재시도하지 않고 실패
- YDF가 없으면 sklearn 또는 다른 estimator로 대체하지 않고 실패
- 점수 척도 정렬은 training-reference ECDF를 저장해 적용하며, held-out
  batch 내부 rank를 다시 계산하지 않음
- 결과에는 원본 ID 대신 실행마다 새 비공개 random secret으로 만든 subject
  token을 기록하고, secret은 저장하지 않아 실행 간 연결을 차단
- Validation label을 열기 전에 예측과 SHA-256 manifest를 먼저 freeze

여기서 막는 것은 사용자가 지정한 **직접적인 데이터 누수**입니다. MMSE가
실제 임상 진단 과정에 사용됐다면 생길 수 있는 incorporation bias는 코드
누수와 다른 연구 설계 한계이며 보고서에서 별도로 밝혀야 합니다.

## 4. 후보 선택과 ROC-AUC

각 candidate의 repeated 5-fold subject OOF raw score와
training-reference ECDF score를 모두 저장합니다. 다음 policy를 ROC-AUC만으로
비교합니다.

- 각 YDF candidate 단일 score
- OOF 상위 2개 동일가중 평균
- OOF 상위 2개 ROC-AUC 최적 simplex 가중 평균
- axis GBT, sparse-oblique GBT, RF family별 OOF 우승자 동일가중 평균
- family 우승자들의 ROC-AUC 최적 simplex 가중 평균
- 전체 YDF candidate의 ROC-AUC 최적 simplex 가중 평균

최종 policy는 같은 141명 OOF의 ROC-AUC를 보고 고르는 **non-nested
selection**입니다. 이는 요청 범위에는 맞지만, 그 OOF AUC는 순수한 사전고정
모델의 불편 추정치가 아닙니다. `run.py`는 모든 개별 후보 점수도 함께 남겨
선택 결과를 역추적할 수 있게 합니다. threshold accuracy, F1 같은 지표는
모델 선택에 사용하지 않습니다.

| profile | outer repeat | seed bag | 후보 수 | 용도 |
| --- | ---: | ---: | ---: | --- |
| `smoke` | 1 | 1 | 3 | 세 YDF family 배선 확인, 비보고용 |
| `default` | 5 | 3 | 9 | 정식 기본 실행 |
| `max` | 10 | 5 | 12 | 더 큰 정식 실행 |

## 5. 실행

의존성은 다음 파일로 고정합니다.

```bash
python -m pip install -r \
  SangHyo/Binary_Google_YDF_AUC/requirements_colab.in
```

저장소 루트 [`base.ipynb`](../../base.ipynb)를 사용할 때는 셀의 실행 파일을
다음처럼 지정합니다. notebook에서 세부 인자를 넘길 때는 `BGYA_ARGS`를
사용합니다.

```python
import os

USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Google_YDF_AUC/run.py"
os.environ["BGYA_ARGS"] = (
    "--stage all --profile default --historical-eval --num-threads 8"
)
```

CLI의 정확한 옵션은 항상 `--help`를 기준으로 확인합니다.

```bash
python SangHyo/Binary_Google_YDF_AUC/run.py --help

# label을 열지 않고 파일·feature schema 계약만 검사
python SangHyo/Binary_Google_YDF_AUC/run.py \
  --stage inspect \
  --profile smoke \
  --data-root Data \
  --skip-install \
  --output-dir <새-결과-폴더>

# Training OOF, AUC-only 선택, 최종 refit
python SangHyo/Binary_Google_YDF_AUC/run.py \
  --stage train \
  --profile default \
  --data-root Data \
  --output-dir <새-결과-폴더>

# Training 뒤 historical Validation도 label-free prediction으로 먼저 freeze한 뒤 평가
python SangHyo/Binary_Google_YDF_AUC/run.py \
  --stage all \
  --profile default \
  --historical-eval \
  --num-threads 8 \
  --data-root Data \
  --output-dir <새-결과-폴더>
```

`inspect`는 Training 특징 source만 열고 label이나 YDF model을 사용하지
않습니다. `train`과 `all`은 Training OOF, AUC-only policy 선택과 최종 refit을
수행합니다. historical Validation은 stage 이름만으로 자동 평가되지 않으며
반드시 `--historical-eval`을 함께 지정해야 합니다. 이때도 예측 CSV와 hash
manifest가 먼저 저장된 뒤 label을 엽니다.

`--num-threads`를 생략하면 최대 8개 CPU thread를 사용합니다. `--seed`는
분할·bag seed의 시작값을 바꾸고, `--skip-install`은 dependency 자동 설치를
막아 현재 환경에 패키지가 없으면 즉시 실패하게 합니다. 환경변수
`BGYA_DATA_ROOT`, `BGYA_OUTPUT_ROOT`로 기본 data/result root도 지정할 수
있습니다.

비어 있지 않은 기존 결과 폴더를 재사용하지 않는 것을 권장합니다. `smoke`의
AUC와 일부 candidate만 끝난 partial run은 최고 성능으로 보고하지 않습니다.

## 6. 결과 판독

정상적인 Training run의 핵심 구조는 다음과 같습니다.

```text
<output>/
├── LAUNCHER_STATUS.json
├── RUN_CONFIG.json
├── DATA_AUDIT.json
├── FEATURE_MANIFEST.json
├── FOLD_FEATURE_SELECTION.json
├── CANDIDATE_RESULTS.json
├── POLICY_RESULTS.json
├── OOF_ALL_CANDIDATES.npz
├── OOF_CHAMPION_REPEATED_HASHED.csv
├── OOF_CHAMPION_SUBJECT_MEAN_HASHED.csv
├── FINAL_REPORT.json
├── LEAKAGE_AUDIT.json
├── TRAINING_COMPLETE.json
└── deployment/
    ├── DEPLOYMENT.json
    ├── ROUNDTRIP.json
    └── components/<candidate>/<bag>/...
```

`--historical-eval`을 지정하면 다음 파일도 생깁니다.

```text
HISTORICAL_VALIDATION_PREDICTIONS_FROZEN.csv
HISTORICAL_VALIDATION_FREEZE.json
HISTORICAL_VALIDATION_REPORT.json
```

정식 결과를 인용하기 전에 다음이 모두 있어야 합니다.

- 실행 profile이 적힌 `RUN_CONFIG.json`과 YDF version이 적힌 최종 보고서
- 사람별 hashed repeated OOF prediction
- 각 fold의 선택 특징, train/held-out 수, bag seed와 overlap 검사 기록
- 모든 candidate와 policy의 subject-mean OOF ROC-AUC
- 선택된 candidate/view/feature/seed-bag 목록
- 최종 YDF checkpoint와 save/load round-trip 검증
- source 접근 allow-list audit
- historical 평가 실행이면 label-free Validation prediction freeze와 그 SHA-256
- 완료 marker

Validation 33명은 과거 여러 실험에서 반복 사용됐으므로 새로운 독립 test가
아니라 historical benchmark입니다. 이 점수로 candidate나 가중치를 다시
고르면 안 됩니다.

## 7. 테스트

```bash
python -m pytest -q \
  SangHyo/Binary_Google_YDF_AUC/tests/test_contracts.py
```

테스트는 source allow-list, 금지 특징명, 관측량 proxy 배제, 사람 단위 fold,
fold-local 특징 선택, YDF-only/fail-closed oblique 계약, reference ECDF,
필수 CLI stage/profile을 확인합니다. `ydf`가 설치된 환경에서는 작은 실제
YDF 모델의 checkpoint save/load 및 예측 동일성도 검사합니다.

프로젝트 전체 데이터·보고 규약은 [`SangHyo/AGENTS.md`](../AGENTS.md)를
따릅니다.
