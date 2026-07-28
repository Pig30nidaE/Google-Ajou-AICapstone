# Binary_Google_ROCAUC_Champion

CN(0)과 MCI+Dem(1)을 사람 단위로 분류하면서 ROC-AUC를 최대화하기 위한
**두 트랙의 누수 방지 실험 코드**입니다.

> **현재 상태:** 구현은 완료됐지만 이 코드의 `default`/`max` 정식 학습 결과와
> 실제 wall time은 아직 없습니다. 따라서 새 ROC-AUC 수치나 기존 최고 기록
> 갱신을 주장하지 않습니다. 아래 숫자는 모두 기존 폴더에서 완료된 과거
> 실험의 관측값입니다.

## 1. 두 트랙

| 트랙 | 허용 입력 | 금지 입력 | 비교 앵커 |
| --- | --- | --- | --- |
| `mmse` | Activity, Sleep, 명시적으로 허용한 MMSE 점수·문항 | MMSE 진단·행정·식별 열 | `Binary_MMSE_MaxAUC`의 규제 LR+SVM |
| `wearable` | Activity와 Sleep만 | `3.CognitiveFunction` 아래의 모든 SourceData·LabelingData | `Binary_Wearable_SequenceFusion_Google`의 sequence Transformer |

두 결과는 별도로 보고합니다. MMSE 포함 성능을 “손목 웨어러블만의 성능”으로
표현하지 않습니다.

`wearable` 트랙의 no-MMSE 계약은 단순히 인지 특징을 마지막에 삭제한다는 뜻이
아닙니다. 해당 트랙은 CognitiveFunction 경로를 **탐색하거나 열 수 없으며**,
접근 시 실행이 실패해야 합니다. `mmse` 트랙도 `DIAG_NM`, `DIAG_SEQ`,
`DOCTOR_NM`, `MMSE_NUM`, `MMSE_KIND`, 이메일 같은 금지 열을 읽지 않습니다.
웨어러블 입력도 검증된 113-channel schema의 SHA-256을 고정합니다. 날짜·시각·
순서·관측량 proxy가 추가되거나 열 순서가 바뀌면 자동 수용하지 않고 실패합니다.
허용 MMSE 원천은 `TOTAL`과 실제 **30개 문항**이며, MaxAUC anchor view는
`TOTAL` 1개, domain score 6개, 문항 30개, failed-items와 recall-deficit을
합친 **39개 특징**입니다.

## 2. 왜 새 설계를 쓰는가

Training은 141명(CN 85, MCI 47, Dem 9)입니다. 345개 이상의 웨어러블 후보를
넓게 탐색하면 inner CV 점수는 올라가지만 새로운 outer fold로 이전되지 않는
현상이 반복됐습니다.

| 기존 정식 실험 | 사람 단위 OOF ROC-AUC | 해석 |
| --- | ---: | --- |
| `Binary_MMSE_MaxAUC` | **0.7658**, 95% CI [0.6840, 0.8457] | 현재 가장 강한 MMSE 앵커 |
| `Binary_Google_OrdinalStable` | 0.7569 | 같은 run의 MMSE-only 0.7494보다 +0.0076 |
| `Binary_Google_MaxAUC_Tuned` | 0.7172 | non-nested 0.8017, optimism +0.0845; 약 10.6시간 |
| `Binary_Wearable_SequenceFusion_Google` 앙상블 | 0.5664 | 웨어러블 고정 앙상블은 약함 |
| 위 실험의 개별 sequence Transformer | **0.6254** | 웨어러블 앵커로 재검증할 제한적 신호 |
| 최신 SOTA DualTrack wearable soft vote | 0.5124 | 95% CI가 0.5 포함 |
| 최신 SOTA DualTrack MMSE stacking | 0.6767 | 복잡한 fusion이 MMSE 앵커보다 낮음 |

하루 행을 사람과 무관하게 나눈 누수 재현 실험에서는 웨어러블-only AUC
0.9526, MMSE 포함 AUC 0.99998이 나왔지만, 정직한 subject OOF에서는 각각
0.5214와 0.6924였습니다. 이 폴더는 이런 day/window 누수를 허용하지 않습니다.

별도 과제인 `Binary_Google_DemScreen`의 CN+MCI vs Dem 결과는 라벨 정의와
난이도가 다르므로 위 기준과 직접 비교하지 않습니다.

## 3. 평가와 선택 절차

기본 profile은 **5 outer folds × 5 repeats**, `max` profile은
**5 outer folds × 10 repeats**의 subject-level repeated nested CV입니다.
두 profile 모두 각 outer-training 안의 inner CV는 계산량을 제한하기 위해
**4 folds × 2 repeats, 총 8개 inner fold**로 고정합니다. 모델 family 선택을
포함해 다음 작업은 모두 현재 outer-training 안의 inner fold에서만 학습합니다.

- 결측 대치, clipping/winsorization, scaling
- label을 이용하는 top-k 특징 선택
- sequence Transformer epoch 선택
- branch score를 공통 순위 척도로 바꾸는 참조 ECDF
- 후보 branch의 포함 여부와 최종 blend/fallback 결정

각 트랙에는 사전 고정한 anchor branch가 있습니다. 실제 후보군은 다음과
같으며, 표 모델의 설정은 탐색하지 않고 코드에 고정돼 있습니다.

| 트랙 | 고정 anchor | 추가 후보 |
| --- | --- | --- |
| `mmse` | `mmse_maxauc_anchor` | `fusion_elastic_top25`, `fusion_rbf_top25`, `fusion_catboost_top25`, optional `fusion_tabpfn_top64` |
| `wearable` | `sequence_transformer_anchor` | `wearable_core_ridge`, `wearable_elastic_top25`, `wearable_rbf_top25`, `wearable_catboost_top25`, optional `wearable_tabpfn_top64` |

각 branch의 raw score는 **inner OOF score로만 학습한 참조 ECDF**에 통과시켜 공통
`[0, 1]` 순위 척도로 바꿉니다. outer-test 자체에서 rank를 다시 계산하지
않습니다. 통과한 branch는 학습된 가중치를 검색하지 않고 동일가중 평균합니다.

선택은 anchor에서 시작하는 greedy forward 절차입니다. 현재 동일가중 blend에
후보 하나를 추가한 정책이 다음 두 조건을 모두 만족할 때만 그 후보를
추가합니다.

1. 두 inner repeat에서 각 사람의 held-out score를 평균한 aggregate inner OOF
   ROC-AUC가 현재 정책보다 최소 **+0.005**
2. 총 8개 inner fold 중 **5개 이상**에서 현재 정책보다 높은 ROC-AUC

첫 단계의 현재 정책은 anchor입니다. fold AUC 동률은 승리로 세지 않습니다.
통과 후보가 여럿이면 AUC gain, fold 승률, branch 이름 순으로 고정된 tie-break를
적용합니다. anchor를 포함해 최대 3개 branch까지만 선택하므로 후보 추가는 최대
2회입니다. 어느 후보도 통과하지 않으면 현재 정책을 유지하며, 첫 단계에서
모두 실패하면 anchor 그대로입니다.

SMOTE와 전체 데이터 선행 oversampling은 사용하지 않습니다. Training의 85:56
비율은 심한 불균형이 아니며, 합성 표본이 필요한지까지 큰 탐색 공간으로 만드는
것보다 규제와 반복 평가를 우선합니다.

자세한 알고리즘과 근거는 [TECHNICAL_REPORT_KO.md](TECHNICAL_REPORT_KO.md)를
참고하십시오.

## 4. Validation을 여는 순서

Validation은 33명(CN 26, MCI+Dem 7)이고 여러 과거 실험에서 이미 사용됐습니다.
따라서 historical benchmark일 뿐, 새 독립 test가 아닙니다.

실행 순서는 다음과 같이 고정합니다.

1. Training label만 사용해 두 트랙의 nested OOF와 최종 refit을 완료
2. Validation label을 열지 않은 상태로 `mmse` 예측 파일 저장
3. Validation label을 열지 않은 상태로 `wearable` 예측 파일 저장
4. 두 파일 모두의 SHA-256, 피험자 수, schema와 설정을 freeze manifest에 기록
5. 두 freeze가 모두 성공했는지 다시 검증
6. 그 뒤에만 Gait/Sleep의 Validation label 사본을 **한 번** 열어 두 트랙 평가

한 트랙이라도 freeze에 실패하면 Validation label을 열지 않습니다. Validation
점수를 본 뒤 특징, seed, branch, gate 또는 threshold를 바꾸는 것도 금지합니다.

## 5. 실행

정식 학습은 사용자가 Google Colab에서만 수행합니다. 저장소 루트
[`base.ipynb`](../../base.ipynb)의 셀 2는 다음과 같이 설정합니다.

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Google_ROCAUC_Champion/run.py"
```

현재 CLI 형식은 다음과 같습니다. `run.py --help`가 정확한 옵션의 source of
truth입니다.

```bash
python SangHyo/Binary_Google_ROCAUC_Champion/run.py \
  --mode full \
  --profile default \
  --tabpfn auto \
  --data-root Data \
  --output-dir <Google-Drive의-새-UTC-run-경로>
```

- `--mode`: 실제 학습을 수행하는 `full`만 허용
- `default`: outer 5 folds × 5 repeats, inner 4 folds × 2 repeats
- `max`: outer 5 folds × 10 repeats, inner 4 folds × 2 repeats
- `--tabpfn auto`: `TABPFN_TOKEN`이 있을 때만 optional branch 포함
- `--tabpfn on|off`: optional branch를 명시적으로 포함하거나 제외
- `--skip-install`: 설치를 건너뛰되 dependency가 없으면 즉시 실패
- `--allow-cpu`: CUDA 강제 검사를 해제하지만 실행 시간이 비현실적으로 길 수 있음

이 구현 작업에서는 정식 학습을 실행하지 않았습니다. 사용자가 Colab에서
`default` 또는 `max`를 완주한 뒤에만 실제 성능과 시간을 판정합니다.
`--output-dir`를 생략하면 기본 경로는
`/content/drive/MyDrive/Binary_Google_ROCAUC_Champion_result/<YYYYMMDD_HHMMSS_utc>`
이며, 비어 있지 않은 출력 폴더는 덮어쓰지 않고 실패합니다.

## 6. 시간·환경 계약

- 전체 두 트랙의 한 run은 **6시간 이내** 완료를 목표로 합니다.
- launcher는 Linux/Colab에서 `SIGALRM`으로 **6시간 hard limit**를 적용합니다.
- 새 구현의 `default`와 `max` 소요 시간은 모두 미실측입니다. 특히 `max`가
  6시간 안에 끝난다는 보장은 없습니다.
- 시간 제한 전에 완료된 repeat만 있는 partial run은 정식 최고 성능으로
  인용하지 않습니다.
- sequence Transformer와 optional TabPFN은 GPU, scikit-learn 표 모델은 CPU와
  RAM도 사용합니다. 기본 실행은 CUDA가 없으면 실패하며, Colab A100+High-RAM을
  권장합니다.
- TabPFN은 optional입니다. 포함 시 움직이는 package default를 쓰지 않고
  `TabPFNClassifier.create_default_for_version(ModelVersion.V2_6)`으로 공식
  **TabPFN v2.6 synthetic-only checkpoint**를 명시합니다. 선택·학습된 TabPFN
  branch의 manifest에는 package/model version, checkpoint 경로와 SHA-256을
  기록합니다. 인증·checkpoint 로딩이 실패하면 다른 버전으로 자동 대체하지
  않고 run이 실패합니다.

## 7. 주요 산출물

현재 코드가 생성하는 주요 파일은 다음과 같습니다.

```text
<result>/<UTC_RUN_ID>/
├── LAUNCHER_STATUS.json
├── data_access_audit.json
├── run_manifest.json
└── training/
    ├── FINAL_REPORT.json
    ├── TRAINING_COMPLETE.json
    ├── deployment_round_trip.json
    ├── mmse/
    │   ├── oof_predictions_hashed.csv
    │   ├── nested_oof_report.json
    │   ├── fold_manifests.json
    │   ├── nested_cv_progress.json
    │   └── deployment/
    │       ├── deployment.json
    │       └── models/
    ├── wearable/
    │   └── 같은 종류의 OOF·fold·deployment 산출물
    ├── validation_predictions_label_free_hashed_mmse.csv
    ├── validation_predictions_label_free_hashed_wearable.csv
    ├── VALIDATION_PREDICTIONS_FROZEN.json
    └── historical_validation_report.json
```

원본 이메일은 결과에 저장하지 않고 안정적인 SHA-256 subject hash만 저장합니다.
각 outer prediction은 해당 사람을 학습하지 않은 모델에서 나왔다는 fold provenance를
함께 남깁니다. 저장한 모델은 같은 실행에서 다시 불러 frozen Validation 입력에
같은 score를 내는지 round-trip 검증합니다(일반 branch 허용오차 `1e-7`,
TabPFN 포함 시 `1e-5`).

## 8. 결과를 읽는 법

Primary는 threshold와 무관한 **subject-level ROC-AUC**입니다. 다음 두 추정량을
혼합하지 않습니다.

- 반복별 outer OOF AUC의 평균과 반복 간 SD
- 각 사람의 반복 OOF score를 평균한 cross-fitted ensemble AUC와 그
  subject-bootstrap 95% CI

후자의 CI를 전자의 평균 AUC에 붙이지 않습니다. anchor와 champion의 차이는
동일 사람을 함께 resample하는 paired subject bootstrap으로 보고합니다.
Balanced Accuracy, MCI+Dem Recall, CN Specificity, PR-AUC와 all-CN Accuracy는
보조 지표입니다. threshold는 ranking 모델이 고정된 뒤 별도 목적에 맞춰
Training OOF에서만 정합니다.

## 9. 한계와 정직한 해석

- 이 프로젝트는 같은 141명 OOF를 보며 여러 모델을 개발했습니다. 새 코드 내부의
  fold 누수를 막아도 **프로젝트 전체의 반복 선택 편향**은 남습니다.
- historical Validation 33명은 새 외부 검증이 아닙니다.
- MMSE가 임상 진단 과정에 사용됐다면 MMSE 트랙에는 predictor가 outcome 정의에
  일부 포함되는 **incorporation bias**가 있을 수 있습니다. 이는 코드 누수와
  별개의 임상적 순환성입니다.
- CN vs MCI+Dem 전체 AUC에는 상대적으로 쉬운 Dem 9명이 포함됩니다. 가능하면
  CN vs MCI AUC를 별도 보조 분석으로 보고합니다.
- 웨어러블-only의 과거 결과는 대부분 0.5 부근입니다. sequence Transformer
  0.6254는 재현해야 할 가설이지, 새 champion의 보장 성능이 아닙니다.
- 어떤 새 수치도 완료 marker, hashed OOF 예측, fold provenance와 freeze
  manifest 없이 최고 기록으로 인정하지 않습니다.

프로젝트 공통 규약은 [`SangHyo/AGENTS.md`](../AGENTS.md)에 정리되어 있습니다.
