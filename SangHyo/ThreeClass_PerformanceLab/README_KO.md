# ThreeClass PerformanceLab

웨어러블 activity/sleep만으로 `CN(0) / MCI(1) / DEM(2)`을 구분하는 새
Training-only 성능 탐색 파이프라인이다. 예측 목표는 미래 전환 예측이 아니라
각 피험자의 마지막 activity 관측 시점에 부여된 **동시점 인지상태**다.

이번 폴더는 기존 실험을 덮어쓰지 않는다. 원본 `Data/`도 읽기 전용이며,
MMSE·진단·ID·절대 날짜는 모델 입력에 포함하지 않는다. 주 성능은 141명의
Training 피험자 `(CN 85, MCI 47, DEM 9)`만으로 계산한 subject-level repeated
nested-CV Macro F1이다.

## 왜 다시 설계했는가

현재까지 가장 정직한 기존 기준선의 nested-CV Macro F1은 약 `0.358`이었다.
최근 `ThreeClass_NextStage`의 adaptive pipeline도 `0.350 +/- 0.071`로 개선되지
않았다. 그 실험에서 calendar 35일의 값·mask·결측 후 경과일을 사용한 고정 TCN이
`0.404 +/- 0.101`로 보였지만, 이미 확인한 outer 결과에서 사후 발견된 값이고
repeat 간 차이도 컸다. 따라서 이 수치는 확립된 성능이 아니라 fresh seed에서만
검증할 고정 comparator 가설이다.

가장 큰 새 가설은 **coverage-invariant observed-event 표현**이다.

- Activity와 sleep을 행 순서로 합치지 않고 modality별로 독립 처리한다.
- 같은 wake-date의 sleep은 가장 긴 main sleep 하나만 유지한다.
- 마지막 최대 28개 실제 관측 event를 event rank로 정렬한다.
- 달력의 빈 날짜 수, 수집 길이, padding mask, days-since-observed는 주 모델에
  주지 않는다.
- scalar와 유효 `CONVERT(...)` 1분/5분 로그에서 수면 단계, HR/HRV, 호흡,
  activity intensity, 일주기, 변동성과 event-rank 추세를 만든다.
- coverage만으로 진단을 예측하는 모델을 별도 negative control로 반드시 실행한다.

세부 사전등록 계약은 [EXPERIMENT_DESIGN_KO.md](EXPERIMENT_DESIGN_KO.md)에 있다.
실행 결과를 본 뒤 같은 run 안에서 feature, seed, class weight, epoch, threshold,
ensemble weight를 바꾸지 않는다.

## 고정 후보

Primary selection 후보는 네 개뿐이다.

1. `event_elastic_v1`: event summary + elastic-net multinomial logistic
2. `event_extra_trees_v1`: event summary + strongly regularized ExtraTrees
3. `event_tcn28_v1`: activity/sleep observed-event 28-step dual TCN
4. `event_elastic_tcn_equal_v1`: elastic과 TCN 확률의 고정 0.5/0.5 평균

`mask_tcn_35d_legacy_v1`, `coverage_only_v1`, class-prior는 비교/감사용이며
winner가 될 수 없다. Class별 threshold·scale, SMOTE, synthetic subject,
adaptive stacking, isotonic calibration은 사용하지 않는다.

최신 foundation model을 무조건 추가하지 않은 것도 의도적이다. TabICLv2 공식
문서는 pretraining 범위가 300개 이상의 training sample이고 300개 미만은 시험하지
않았다고 밝힌다. 이 실험의 outer-train은 약 94명에 불과하다. RealMLP의 공식
benchmark도 주로 1천~50만 행 범위다. 이미 이 코호트에서 TabPFN-3가 안정적 이득을
보이지 않았으므로, 이번 run은 후보 수보다 표현과 선택 안정성을 우선한다.

- [TabICLv2 공식 저장소](https://github.com/soda-inria/tabicl)
- [RealMLP/강한 tabular defaults 논문](https://papers.nips.cc/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract-Conference.html)

## 파일

```text
ThreeClass_PerformanceLab/
├── README_KO.md
├── EXPERIMENT_DESIGN_KO.md
├── EDA_REPORT_KO.md
├── local_eda.py
├── local_feature_smoke.py
├── performance_lab_core.py
├── requirements_colab.txt
├── 01_train_only_discovery_colab.ipynb
├── scripts/
│   └── build_notebook.py
├── tests/
│   └── test_static_contracts.py
└── artifacts/local_eda/
    ├── EDA_REPORT_KO.md
    ├── data_audit.json
    ├── class_feature_summary.csv
    └── feature_contract_smoke.json
```

루트 `EDA_REPORT_KO.md`는 읽기 편한 보고서 사본이며,
`artifacts/local_eda/EDA_REPORT_KO.md`와 동일한 privacy-safe aggregate 내용이다.

`02_frozen_benchmark_colab.ipynb`는 Training-only 결과가 사전등록 GO 조건을
통과하고 frozen artifact가 확정된 뒤에만 만든다. 이렇게 해야 역사적으로
재사용된 benchmark를 또 다른 모델 선택 자료로 쓰는 일을 막을 수 있다.

## Colab Drive 준비

저장소 전체를 Google Drive에 둔다. 기본 예시는 다음과 같다.

```text
/content/drive/MyDrive/GoogleAI_contest/AI_Capstone_Project/
├── Data/
│   └── 1.Training/
└── ThreeClass_PerformanceLab/
```

다른 위치라면 notebook 설정 cell의 세 값만 수정한다.

```python
PROJECT_ROOT_OVERRIDE = None
TRAINING_ROOT_OVERRIDE = None
OUTPUT_BASE_OVERRIDE = None
```

노트북은 로컬 macOS 경로를 하드코딩하지 않는다. `SUBJECT_HASH_KEY`는 Colab
Secrets에 32자 이상의 임의 문자열로 저장하고 notebook access를 허용한다.
이 값은 원본 ID를 keyed hash로 바꾸는 데만 쓰며 출력·artifact에 저장하지 않는다.

## 실행 순서

1. Colab Pro+에서 A100 GPU와 High-RAM을 선택한다.
2. `01_train_only_discovery_colab.ipynb`를 연다.
3. 처음에는 `FAST_MODE=True`로 end-to-end smoke test를 수행한다. Smoke 결과는
   성능 근거가 아니며 별도 output directory에 저장된다.
4. clean runtime에서 `FAST_MODE=False`로 위에서 아래까지 실행한다.
5. 완료 후 아래 파일을 이 대화에 전달한다.

첫 bootstrap cell은 PyTorch/CUDA가 import되기 전에
`CUBLAS_WORKSPACE_CONFIG=:4096:8`을 설정하며, strict deterministic algorithm
상태를 runtime identity에 포함한다. 셀 순서를 바꾸지 않는다.

```text
nested_cv_report.json
nested_cv_config.json
inner_candidate_metrics.csv
inner_fold_split_audit.csv
outer_fold_metrics.csv
outer_repeat_metrics.csv
coverage_negative_control_metrics.json
legacy_mask_tcn_metrics.json
selection_report.json
stop_go_decision.json
environment.json
privacy_audit_pre_freeze.json
privacy_audit.json
FINAL_TRAINING_REPORT.json
TRAINING_COMPLETE.json
fold_assignments_hashed.csv
candidate_outer_predictions_hashed.parquet
```

Full run이 `GO`이면 `frozen_config_before_validation.json`,
`FINAL_TRAINING_REFIT.json`, `final_model_bundle.joblib`,
`selected_preprocessor.joblib`도 함께 전달한다. 후속 단계는
`TRAINING_COMPLETE.json` 존재, 두 privacy audit 통과, `RUN_INVALID_PRIVACY.json`
부재를 모두 확인하기 전에는 frozen artifact를 신뢰하지 않는다.

가능하면 위 개별 파일보다 전체 output 폴더를 압축해 전달하되, 원본 ID나 secret이
포함되지 않았는지 먼저 `privacy_audit.json`을 확인한다. 노트북은 완료된 동일
run을 덮어쓰지 않고 outer fold checkpoint에서 재개한다.

## Full run 평가 계약

- Outer: fresh seed 5개 x stratified 3-fold
- Inner: 각 outer-train에서 fresh seed 2개 x stratified 3-fold
- stochastic model seed: `17011`, `27011`
- TCN epoch: 120 고정, early stopping 없음
- primary: subject-level Macro F1
- 함께 보고: balanced accuracy, accuracy, log loss, class별 precision/recall/F1,
  confusion matrix, OVR AUROC/AUPRC, fold별 class support

DEM은 Training 9명뿐이라 한 명의 예측이 결과를 크게 바꾼다. 높은 단일 점수보다
repeat 방향, MCI/DEM F1, coverage-only 차이, calibration과 분산을 함께 본다.

## MMSE를 사용하지 않는 이유

MMSE source에는 진단의 완전 복제 필드가 있고 문항·총점도 진단 과정과 매우 가까운
동시점 임상검사다. 이를 넣으면 숫자는 쉽게 올라갈 수 있지만 wearable lifelog
분류 성능으로 해석할 수 없다. 이번 코드에는 MMSE 허용 switch 자체를 두지 않는다.
별도 예측 시점에서 MMSE가 합법적으로 제공된다는 명시적 대회 계약이 생기면,
그때 별도 폴더·별도 target 명칭의 임상검사 보조 실험으로만 수행해야 한다.

## 해석 한계

이 notebook이 GO 조건을 통과해도 Training 코호트 내부의 개발 증거일 뿐이다.
Feature family 기획 전에 Training 141명의 class별 aggregate EDA를 보았으므로,
nested-CV 역시 이 사전 고정 설계에 조건부인 내부 추정치다.
공식 benchmark는 이미 여러 실험과 문헌에서 재사용됐으므로 독립 holdout이 아니다.
최종 확증에는 다른 수집 프로토콜의 외부 코호트가 필요하다. Feature importance는
성능 동결 후 OOF/held-out association으로만 계산하며 임상적 원인으로 표현하지
않는다.
