# Binary_Google_DemRankAUC — CN+MCI vs **Dem** subject-level ROC-AUC

> **과제 정의 주의.** 이 폴더의 양성은 **Dem 12명**, 음성은 **CN+MCI 162명**입니다.
> 다른 `Binary_*` 폴더(CN vs MCI+Dem)와 **숫자를 직접 비교하지 마십시오.**
> 비교 대상은 같은 과제를 푼 `Binary_Google_DemScreen` 하나뿐입니다.

> **이 문서에 실측 성능 수치는 없습니다.** 로컬에서 수행한 것은 정적 검사 ·
> 누수 계약 테스트 · 배선 확인(smoke)뿐이고, 정식 학습은 사용자가 Colab에서
> 실행합니다. AGENTS.md 계약대로 **smoke 값은 성능으로 인용하지 않습니다.**

---

## 1. 이전 실험의 핵심 한계

기준선은 `Binary_Google_DemScreen`의 정식 실행
(`20260728_051820_utc/training/FINAL_REPORT.json`, 20 repeat × 5 fold 중첩 CV)입니다.

| arm | OOF ROC-AUC | 비고 |
|---|---:|---|
| `wearable_only__full` (174) | 0.7184 ± 0.0363 | 그 폴더의 headline |
| `wearable_plus_mmse__full` (174) | **0.8284 ± 0.0450** | **이 폴더가 넘어야 할 값** |
| `wearable_only__filtered` (172) | 0.8304 ± 0.0395 | 품질 필터 민감도 |
| `wearable_plus_mmse__filtered` (172) | 0.9225 ± 0.0279 | 품질 필터 민감도 |
| (참고) `mmse_TOTAL` 단일 피처, 학습 없음 | **0.947** | 모델이 아님 |

**핵심 문제: 학습된 151-피처 앙상블(0.8284)이 학습이 필요 없는 단일 컬럼(0.947)보다
0.12 낮습니다.** 모델이 신호를 만들지 못한 게 아니라 **파괴**하고 있었습니다. 원인은
그 폴더의 코드와 리포트에서 특정할 수 있습니다.

1. **fold 내 특징선택이 노이즈 지배.** 양성 ~10명으로 151개 후보를 학습 AUC로
   순위 매겨 top-k를 골랐습니다. `MaxAUC_Tuned`에서 이미 fold별 `top_k`가
   `[70,10,70,10,70,15,0,...]`로 전혀 수렴하지 않는 것이 관측됐습니다.
2. **블렌딩 가중치 과적합.** inner OOF(양성 ~10명) 위에서 4000회 Dirichlet 탐색을
   했습니다. 자유도가 신호보다 큽니다.
3. **log-odds 평균 blending.** ROC-AUC는 **순위**만 봅니다. 척도가 제각각인
   (SVM margin / YDF 잎값 / 확률) 모델을 확률·로그오즈 공간에서 평균하면 *가장
   정보가 많은* 모델이 아니라 *수치 범위가 가장 넓은* 모델이 순위를 지배합니다.
   실제로 `univariate`(inner AUC 0.870)와 `ydf_rf`(0.884)에 가중치 78%가 몰렸습니다.
4. **일별 요약만 사용.** 웨어러블 피처 112개가 전부 하루 단위 집계였습니다.

## 2. 선택한 최종 전략과 근거

위 네 가지를 각각 다룹니다. 사전 등록한 **주 가설은 (4)** 입니다.

### 2-1. 미사용 intraday 시계열 (주 가설)

데이터셋에는 **분/5분 단위 원시 시계열이 이미 들어 있습니다.** 평문
`activity_class_5min` 같은 컬럼은 문자열 `"..."`이라 쓸 수 없지만,
**`CONVERT(... USING utf8)` 컬럼에 실제 값이 있습니다.** 이 저장소의 어떤 실험도
이것을 파싱한 적이 없습니다.

| 컬럼 | 해상도 | 길이 | 내용 |
|---|---|---|---|
| `CONVERT(activity_met_1min USING utf8)` | 1분 | 1440/일 | MET |
| `CONVERT(activity_class_5min USING utf8)` | 5분 | 288/일 | 활동 클래스 0–5 |
| `CONVERT(sleep_hypnogram_5min USING utf8)` | 5분 | 41–180/밤 | 수면 단계 1–4 |
| `CONVERT(sleep_hr_5min USING utf8)` | 5분 | 41–180/밤 | 수면 중 HR |
| `CONVERT(sleep_rmssd_5min USING utf8)` | 5분 | 41–180/밤 | HRV |

여기서 **비모수 일주기 지표**(치매 actigraphy 문헌의 표준 지표)를 계산합니다.

- **IS** interdaily stability — 24시간 패턴의 날짜 간 재현성
- **IV** intradaily variability — 휴식/활동 리듬의 파편화
- **M10 / L5 / RA** — 최활동 10시간, 최비활동 5시간, 상대진폭과 각 onset 시각
- 수면 미세구조 — WASO, 각성 횟수, 단계 전이율, deep/REM bout
- 야간 자율신경 — HR dip, 야간 HR 기울기, RMSSD 평균·변동

`activity_day_start`가 항상 04:00이라 시계 정렬이 코호트 참조 없이 가능합니다.

부수 효과가 하나 더 있습니다. 이 피처들은 **연속형**입니다. 양성 12명에서 depth-2
트리는 서로 다른 잎값을 몇 개밖에 내지 못해 **동점(tie)** 이 대량 발생하고,
ROC-AUC는 동점마다 손해를 봅니다. 연속형 피처는 그 동점을 옳은 방향으로 깹니다.

### 2-2. rank 기반 앙상블 (문제 3)

`ensemble.py`는 `prob_mean` / `logit_mean` / `rank_mean` / `rank_weighted` /
`greedy`(Caruana) / `stack_lr`을 **모두** 제공하고 inner OOF가 고르게 합니다.
rank 정규화 후에는 모든 모델이 척도와 무관하게 같은 양의 순위 정보를 기여합니다.
동점은 평균 순위로 처리합니다.

이전 폴더가 log-odds를 쓴 이유(임계값 전이 안정성)는 임계값에는 맞지만 AUC에는
무관합니다. 그래서 배제하지 않고 **후보로 두고 측정**합니다.

### 2-3. 자유도 축소 (문제 1, 2)

- `rank_mean` 학습기: 피처마다 **부호 1개**만 추정하고 크기는 추정하지 않습니다.
  양성 12명 구간에서 가장 분산이 낮은 선택지이며 순위 공간에서 직접 동작합니다.
- Dirichlet 탐색을 4000 → 800 draw로 줄이고, 자유 파라미터가 0인 `rank_mean`과
  항상 경쟁시킵니다.
- `top_k`를 필수 단계가 아니라 **fold 내에서 고르는 하이퍼파라미터**로 두어
  "전부 쓰고 규제에 맡기기"(`top_k=0`)가 이길 수 있게 했습니다.
- nested 후보군은 **사전 지정 7개 계열**입니다(`config.NESTED_CANDIDATE_FAMILIES`).
  이 코호트 점수를 보고 고른 것이 아닙니다.

### 2-4. 그 밖에 측정하는 것

- **MMSE 전 0 레코드.** `val_mmse.csv`의 `nia+045`(Dem)는 30문항과 TOTAL이 전부
  `0`입니다. 이 검사의 코딩은 1=오답/2=정답이므로 0은 **미실시 기록**이지 0점이
  아닙니다. 그대로 두면 이 사람이 코호트 최저 MMSE가 되고, 하필 Dem이라
  MMSE 기반 순위 모델이 부당하게 유리해집니다. 기본값은 **결측 처리**이며
  (라벨을 보지 않는 계측 규칙), 벤치마크 재현용으로 원래 동작도 유지합니다.
  실측 영향: `-mmse_TOTAL` 단일 피처 AUC 0.9470(원본) / 0.9422(해당자 제외) /
  0.9151(중앙값 대치).
- **`mmse_num_failed` 제거.** 이 도구에서 정확히 `30 - TOTAL`이라 완전 공선입니다
  (그래서 이전 폴더에서 두 값의 AUC가 똑같이 0.947이었습니다).

## 3. 데이터 누수 방지 방식

### 3-1. 구조적으로 불가능하게 만든 것

`features.py`가 산출하는 모든 값은 **한 피험자 자신의 행만의 함수**입니다.
피험자당 35–120개 일별 행이 **1행**으로 줄어들기 때문에, 행 단위 분할이 곧
피험자 단위 분할입니다. `Binary_PaperLGBM_*`이 정량화한 실패 모드(하루 단위
무작위 분할로 AUC 0.52 → 0.95)는 여기서 구조적으로 발생할 수 없습니다.

### 3-2. 주장하지 않고 **검사로 강제**한 것

| 검사 | 무엇을 막는가 |
|---|---|
| `test_features_are_subject_local` | 피험자 부분집합으로 피처를 **재계산**해 남은 행이 bit-identical인지 확인. 코호트 평균·분위수·target encoding이 하나라도 섞이면 실패 |
| `FoldPreprocessor` fit 지문 | fit한 행 집합의 SHA-256을 기록하고, 다른 fold의 행을 transform하려 하면 **AssertionError**. 전체 데이터로 fit 후 fold에 적용하는 실수가 조용히 지나가지 못함 |
| `test_resampling_never_sees_test_rows` | resampler가 받은 행 수가 학습 fold 크기와 정확히 같은지 계수 |
| `assert_no_forbidden` (fail-closed) | `DIAG_NM`·`DIAG_SEQ`·`DOCTOR_NM`·`MMSE_NUM`·`MMSE_KIND`·식별자 및 `diag/label/doctor/email` 부분문자열 차단. 거부 동작 자체도 테스트 |
| `assert_split_integrity` | fold 겹침, 단일 클래스 fold, 양성 0개 test fold, 파티션 불완전 |
| `assert_contract` | 174 / CN 111 / MCI 51 / Dem 12, 중복 ID, 완전 분리 피처(양성 12명에서는 신호보다 누수일 가능성이 큼) |
| 라벨 사본 대조 | Gait·Sleep 라벨 CSV가 일치하지 않으면 중단 |
| `test_quality_flags_do_not_read_labels` | 라벨을 뒤집어도 품질 규칙 결과가 동일한지 |

### 3-3. fold 안에서만 학습되는 것

분할 → 전처리(중앙값 대치·winsorize·표준화) → 특징선택 → 리샘플링 → 학습 → 채점.
여기에 더해 **모델 계열 선택, 블렌딩 방식, 블렌딩 가중치, 운영 임계값**까지
전부 outer-training 안의 inner fold에서 결정됩니다.

### 3-4. 정직하게 남는 편향

- screening 표(`oof_fixed`)에서 최고 행을 사람이 읽고 고르는 것은 **arm-selection
  bias**입니다. 그래서 headline은 **nested 결과만** 사용합니다.
- nested 안에서도 **track(피처 블록) 선택**은 screening을 봤습니다. 그래서
  모든 track의 nested 값을 함께 보고합니다.
- Optuna 단계는 목적함수가 174명 전부를 봅니다. 그 값은 **성능 주장이 아니며**,
  optimism 두 성분(fold 재사용 / 설정 선택)과 함께만 보고합니다.

## 4. 검증 방식

- **repeated stratified subject-level CV.** outer 5-fold × 20 repeat(`full`).
  양성 12명이라 5-fold가 test fold마다 Dem 2–3명을 보장하는 최대 fold 수입니다.
  `safe_k`가 소수 클래스 수로 상한을 걸어 단일 클래스 fold를 원천 차단합니다.
- **repeat이 보고 단위.** 한 repeat 안에서 OOF를 모아 1회 채점하고, 20 repeat의
  평균 ± 표준편차를 보고합니다(DemScreen과 동일 프로토콜 → 직접 비교 가능).
  **fold별 ROC-AUC도 전부 저장**합니다(`fold_metrics.csv`).
- **모든 모델이 동일한 fold 객체**를 씁니다 → paired 비교 가능.
- **불확실성 2종.** repeat 간 표준편차(분할 노이즈)와 subject bootstrap 95% CI
  (표본 노이즈). 양성 12명에서는 **후자가 압도적**입니다.
- **paired bootstrap**으로 headline과 각 baseline의 AUC 차이 CI를 보고합니다.
  **CI가 0을 포함하면 개선으로 선언하지 않습니다.**
- **홀드아웃 테스트셋 없음.** Dem 12명 전부가 학습에 필요합니다. 모든 수치는
  **OOF**이며 독립 외부 검증이 아닙니다. 리포트에 `has_holdout_test_set: false`로
  기록됩니다.

### 증거 등급 (모든 수치에 태그가 붙습니다)

| 태그 | 의미 |
|---|---|
| `oof_nested` | 모델·가중치·임계값을 각 outer fold 안에서 결정. **headline** |
| `oof_fixed` | 고정 설정의 repeated CV. 모델별로는 정직하나 arm-selection bias 잔존 |
| `non_nested_tuned` | Optuna 단계. 성능 주장 아님. optimism과 함께만 |
| `descriptive` | 전체 코호트 통계(단변량 AUC 등). 아무것도 선택하지 않음 |

## 5. 모델 후보

**Google 모델** — 이름만 넣지 않고 실제로 돌립니다.

- **YDF** (Yggdrasil Decision Forests): `ydf_gbt`, `ydf_rf`, `ydf_gbt_oblique`.
  oblique를 유지한 이유는 `MaxAUC_Tuned`에서 단일 학습기 1위였기 때문입니다
  (inner AUC 0.789 vs 축정렬 0.745). 클래스 가중은 sample weight로 적용.
- **TabNet** (`pytorch-tabnet`): `n_d`, `n_a`, `n_steps`, `gamma`,
  `lambda_sparse`, `virtual_batch_size`, lr, `ReduceLROnPlateau` 스케줄러,
  patience/early stopping을 모두 탐색 공간에 노출. **masked unsupervised
  pretraining**(`TabNetPretrainer`)을 지원하되, **해당 fold의 학습 행만으로**
  사전학습합니다. 비지도라도 전체 코호트로 돌리면 검증 피험자의 분포를
  보게 되므로 fold 안에 둡니다.
  전망은 낙관적이지 않습니다 — `Binary_Wearable_TabNet_Google`이 OOF AUC 0.446,
  10개 fold 중 9개가 0.5 미만이었습니다. **거부될 수 있는 후보**로 둡니다.

**트리 baseline** — LightGBM, XGBoost, CatBoost, HistGradientBoosting,
RandomForest, ExtraTrees, BalancedRandomForest, EasyEnsemble.

**그 외** — elastic-net / L2 로지스틱, linear·RBF SVM, MLP,
`rank_mean`(부호만 추정하는 순위 평균), `univariate`(DemScreen 학습기 재현),
`balanced_bag`(균형 부트스트랩 배깅).

### 시계열 모델에 대한 판단 (TSMixer)

**데이터 쪽은 정당합니다.** 피험자당 35–120일(중앙값 66일)의 연속 일별 다변량
기록이 26채널 있습니다. 집계형이 아니라 진짜 시계열이므로 "짧아서 못 쓴다"는
면제 사유에 해당하지 않습니다.

**라벨 쪽이 문제입니다.** 양성이 12명입니다. 이 저장소는 인접 과제에서 이미
측정했습니다 — ConvBiLSTM OOF AUC 0.515–0.526, SequenceFusion 앙상블 0.566.

그래서 **구현하되 기본 비활성**입니다(`--sequence-arm` 또는 `max` 프로파일).
`sequence.py`는 마스킹된 TSMixer(시간축 MLP + 채널축 MLP, attention pooling,
static covariate 결합, 가변 길이 마스크, focal loss 옵션)이며, 채널 정규화
통계는 **fold의 학습 피험자에게서만** 추정합니다. 지면 그것은 의견이 아니라
**측정된 답**입니다.

## 6. 설치

```bash
pip install -r SangHyo/Binary_Google_DemRankAUC/requirements_colab.txt
```

필수는 numpy / pandas / scipy / scikit-learn / joblib / matplotlib 뿐이고,
나머지는 없으면 **자동으로 제외**됩니다(`environment_report()`가 무엇이 빠졌는지
리포트에 기록). Google 모델을 반드시 후보에 넣으려면 최소한 다음이 필요합니다.

```bash
pip install "ydf==0.16.1" "pytorch-tabnet==4.1.0" torch
```

## 7. 실행

Colab `base.ipynb` 셀 2:

```python
USER_FOLDER = "SangHyo"
RUN_FILE    = "Binary_Google_DemRankAUC/run.py"
```

**런타임: CPU (High-RAM).** headline 후보는 전부 트리/선형이라 GPU가 필요 없습니다.
TabNet과 `--sequence-arm`만 GPU를 씁니다.

CLI:

```bash
python SangHyo/Binary_Google_DemRankAUC/run.py --profile full --data-root Data --output-dir ./out
```

| 옵션 | 기본 | 설명 |
|---|---|---|
| `--profile` | `full` | `smoke`(배선 확인) / `standard` / `full`(20 repeat) / `max` |
| `--tracks` | 10개 블록 | 쉼표 구분 피처 블록 |
| `--models` | 설치된 전부 | screening에 쓸 모델 |
| `--cohort` | `both` | `full` / `filtered`(라벨 무관 품질 규칙) / `both` |
| `--resampler` | `class_weight` | `none`/`random_over`/`smote`/`borderline_smote`/`adasyn`/`random_under` |
| `--sequence-arm` | off | TSMixer 활성화(torch 필요) |
| `--no-tune` | off | Optuna 단계 생략 |
| `--drop-suspect` | off | 착용일수·커버리지 피처 제거(adherence artifact ablation) |
| `--hours` | 6.0 | 하드 런타임 상한. 초과 시 graceful 중단 |

### 실측 런타임 (M4 노트북 CPU, 참고용)

nested 후보 7계열 / 15 variant, inner 4-fold 기준입니다.

| 단계 | 측정값 | 20 repeat 환산 |
|---|---:|---:|
| nested `mmse_core` (7 피처) | 45.9 s / 5 fold | 15.3 분 |
| nested `fused_core` (45) | 51.4 s / 5 fold | 17.1 분 |
| nested `wd_full` (556) | 133.7 s / 5 fold | 44.6 분 |
| nested `fused_full` (696) | 155.6 s / 5 fold | 51.9 분 |
| screening `fused_full`, 18개 모델 | 126.8 s / 5 fold | — |

nested track은 최대 3개로 제한됩니다(최고 track + `wd_full` + `fused_core`).
`--cohort both`는 이 전체를 두 번 돌립니다. 6시간 상한에 걸리면
**primary(`full`) 코호트가 먼저 끝나도록 순서가 잡혀 있고**, 남은 arm은
`status: incomplete`로 표시된 뒤 **리포트는 정상 생성**됩니다.
Colab 첫 실행은 `--profile standard`(5 repeat, 약 30분)로 확인하는 것을 권장합니다.

## 8. 출력 파일

```
<output>/
├── LAUNCHER_STATUS.json          # starting/running/complete/failed
└── training/
    ├── FINAL_REPORT.json         # 모든 수치 + 증거 등급 태그
    ├── TRAINING_COMPLETE.json
    ├── run.log
    └── <full|filtered>/
        ├── model_comparison.csv      # 모델 × track 비교표 (oof_fixed)
        ├── fold_metrics.csv          # fold별 ROC-AUC 전부
        ├── oof_predictions_hashed.csv# 피험자 해시 + 모델별 OOF 점수
        ├── headline_roc_curve.csv
        ├── headline_pr_curve.csv
        ├── headline_curves.png
        └── feature_importance_<model>.csv
```

피험자 원본 e-mail은 저장하지 않고 SHA-256 앞 16자리만 남깁니다.

## 9. 로컬에서 실제로 실행해 확인한 것

정식 학습이 아니라 **실행 가능성 검증**입니다.

- `python -m compileall` — 전체 모듈 통과
- `tests/test_contracts.py` — **21/21 통과**. `pytest` 없이도 실행 가능
  (`python -m SangHyo.Binary_Google_DemRankAUC.tests.test_contracts`)
  - 누수 계약 전부 포함(피처 subject-locality, 전처리 fit 범위 거부,
    리샘플링 행 수, 금지 컬럼 거부 동작, split 무결성)
  - `test_every_available_model_fits_and_scores` — 설치된 전 모델이 fit/score
- `run.py --profile smoke` **5회** — full/filtered/both 코호트,
  `class_weight`·`random_over`·`smote` 리샘플러, `--drop-suspect`,
  `--sequence-arm`, track 1–3개, 모델 4개~전체 21개 조합에서 파이프라인이 끝까지
  돌고 위 산출물이 전부 생성됨
- **모델 21종 전부 fit/score 확인** — Google `ydf_gbt`/`ydf_rf`/`ydf_gbt_oblique`,
  Google `tabnet`(**masked pretraining on/off 양쪽**), `lightgbm`/`xgboost`/
  `catboost`/`hist_gb`/`random_forest`/`extra_trees`/`balanced_rf`/`easy_ensemble`,
  `logreg_en`/`logreg_l2`/`svm_rbf`/`svm_linear`/`mlp`,
  `rank_mean`/`univariate`/`bag_*`, 그리고 `tsmixer`
- **TSMixer arm** — `run_repeated_cv` 5-fold 경로를 fold별 정규화 포함해 완주
- Optuna 단계 — optuna 미설치 시 random-search fallback 동작 확인.
  optimism 두 성분이 실제로 0이 아닌 값을 산출
  (측정 예: `hist_gb` fold-reuse +0.050, `logreg_en` selection +0.020)

**검증 중 발견해 고친 실제 버그 2건**(둘 다 TabNet 경로):

1. float64 행렬이 그대로 가속기로 전달돼 Apple MPS에서 `TypeError`. → 입력을
   float32로 캐스팅.
2. `ReduceLROnPlateau`가 torch 2.x + pytorch-tabnet 조합에서
   `step() missing 1 required positional argument: 'metrics'`로 실패. → 버전에
   무관한 `StepLR`로 교체.
3. pretrainer의 eval 로더가 batch_size > holdout 크기일 때 배치 0개를 만들어
   `np.vstack`이 빈 리스트로 실패. → batch를 두 split 중 작은 쪽에 맞추고
   `drop_last=False` 지정.

**정직하게, 로컬에서 확인하지 못한 것:** CUDA 경로(로컬은 MPS/CPU)와 6시간
`full` 프로파일 완주입니다. Colab 첫 실행은 `--profile standard`로 짧게 확인한 뒤
`full`을 권장합니다.

## 10. 양성 12명에서 오는 통계적 한계

- 어떤 점추정치도 **subject bootstrap CI 없이 읽으면 안 됩니다.** DemScreen의
  같은 과제 CI 폭은 0.13–0.29였습니다.
- **AUC 0.05 차이는 이 표본에서 구분되지 않습니다.** paired bootstrap CI가 0을
  포함하면 개선으로 선언하지 않습니다.
- test fold마다 Dem이 2–3명입니다. fold별 AUC는 개별적으로 해석할 수 없고
  분포로만 봐야 합니다.
- 이 폴더의 높은 수치는 **모델이 좋아서가 아니라 과제가 쉬워서**이기도 합니다.
  진단군별 MMSE 평균은 CN 27.7 / MCI 25.8 / **Dem 16.6**입니다.
- MMSE 포함 결과는 **부분적으로 순환적**입니다. MMSE는 이 진단을 내리는 검사
  도구 자체이므로, "손목 웨어러블만으로 치매를 선별한 성능"으로 표현하면 안 됩니다.
  그래서 `wearable_core` / `wearable_full` track을 항상 함께 보고합니다.
- 다음으로 의미 있는 검증은 같은 174명 재평가가 아니라 **새 코호트**입니다.

---

*연구용 분류 실험이며 의료 진단 도구가 아닙니다.*
