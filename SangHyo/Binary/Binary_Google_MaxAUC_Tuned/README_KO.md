# Binary_Google_MaxAUC_Tuned — 성능(ROC-AUC)만 목표로 하는 튜닝 실험

CN(정상) vs MCI+DEM 이진 분류에서, **누수를 막은 채로 ROC-AUC를 최대한 끌어올리는**
것만을 목표로 한 실험입니다. 학습시간 제약을 두지 않고 **하이퍼파라미터 탐색**을
본격적으로 넣었으며, 모델은 **Google YDF(Yggdrasil Decision Forests)**를 중심으로
구성했습니다.

> **목표**: ROC-AUC ≥ 0.80 (누수 없는 subject 단위)
> **직전 최고 기록**: `Binary_MMSE_MaxAUC` = 0.7657 (MMSE 39개 특징, 튜닝 없음)

---

## 1. 0.80을 노리는 3가지 지렛대

직전 실험(0.7657) 대비 이번에 실제로 바꾼 것은 세 가지입니다.

### (1) 특징을 39개 → 151개로 확장
직전 실험은 "웨어러블은 신호를 희석한다"고 보고 **MMSE만** 썼습니다. 그런데 그
실험의 EDA 자체가 반대를 가리키고 있었습니다(mmse+wearable 0.752 > mmse only
0.735). 원인은 웨어러블 요약이 너무 얄팍했기 때문(std 5개 채널뿐)입니다.

이번에는 웨어러블을 제대로 요약합니다:
- 주요 채널 26개에 대해 **mean / std / CV**(변동계수)
- 나머지 채널은 mean
- **수면 구조 비율**: deep/rem/light/awake 비율, 각성 단편화(fragmentation)
- **일주기 규칙성**: 취침·기상·수면중점 시각의 **원형 통계(circular statistics)**
  — 취침시각은 자정을 넘나들기 때문에 일반 표준편차로는 23:50과 00:10을 12시간
  차이로 계산합니다. 원형 SD로 계산해야 "수면 시각 불규칙성"이라는 치매 관련
  지표가 제대로 나옵니다.

**실측 검증**(동일한 고정 로지스틱 회귀, 누수 없는 CV):

| 특징 블록 | ROC-AUC |
|---|---|
| MMSE만 (39개) | 0.7351 |
| 웨어러블만 (112개) | 0.5973 |
| **전체 (151개)** | **0.7721** |

웨어러블 단독은 약하지만(0.60), MMSE와 **합치면 +0.037**을 더합니다. 이번 실험의
전제가 실측으로 확인된 부분입니다.

### (2) Google YDF 중심 모델 풀 + 하이퍼파라미터 탐색
| 모델 | 설명 |
|---|---|
| `ydf_gbt` | **Google YDF** Gradient Boosted Trees |
| `ydf_gbt_oblique` | **YDF sparse oblique 분할** GBT |
| `ydf_rf` | **Google YDF** Random Forest |
| `ydf_rf_oblique` | YDF sparse oblique RF |
| `logreg` | 규제 로지스틱(L1/L2/ElasticNet) |
| `svm` | RBF SVM |

**oblique(사선) 분할**을 넣은 이유가 핵심입니다. 일반 트리는 한 번에 특징 하나만
보고 자르는데(`x3 < 0.7`), YDF의 sparse oblique 분할은 여러 특징의 **희소 선형
결합**으로 자릅니다. 직전 실험에서 트리 계열(HistGBT ~0.73)이 로지스틱 회귀
(~0.757)에 밀렸다는 것은 신호가 대체로 **선형**이라는 뜻이고, oblique 분할은
바로 그 선형 구조를 트리가 잡을 수 있게 해줍니다. Google YDF가 기본 제공하는
기능이며 이번 실험에서 YDF를 쓰는 가장 실질적인 이유입니다.

로지스틱/SVM을 남겨둔 것은 직전 실험의 정직한 승자였기 때문입니다. 앙상블
가중치는 **inner fold ROC-AUC로 결정**되므로, Google 모델도 성능으로 자기 몫을
증명해야 가중치를 받습니다.

### (3) 특징 선택 자체를 하이퍼파라미터로
141명에 151개 특징이면 선택을 어떻게 하느냐가 트리 파라미터보다 중요합니다.
그래서 `top_k`(단변량 AUC 상위 몇 개를 남길지)와 `corr_threshold`(상관 중복
제거 기준)를 **모델 하이퍼파라미터와 함께 탐색**합니다. 상관 0.95 이상인 특징
쌍이 38개나 있어서 중복 제거가 실제로 필요합니다.

---

## 2. 가장 중요한 설계: 튜닝이 성능을 부풀리지 않게 막기

하이퍼파라미터 탐색·특징 선택·앙상블 가중치는 **각각 평가를 오염시킬 수 있는
경로**입니다. 셋 중 하나라도 나중에 점수를 매길 데이터를 미리 보면, 보고되는
ROC-AUC는 부풀려집니다. (옆 폴더에서 논문의 0.9가 만들어진 것과 같은 원리이며,
다만 더 미묘합니다.)

그래서 구조를 이렇게 잡았습니다:

```
바깥 fold (subject 단위, 누수 없음)
 └── 학습 부분만 사용:
      ├── 특징 선택      (inner fold마다 다시 계산)
      ├── 하이퍼파라미터 탐색 (inner fold OOF AUC로 채점)
      └── 앙상블 가중치   (inner fold OOF 확률로 학습)
 └── 테스트 부분: 단 한 번, 최종 예측에만 사용
```

### optimism(낙관 편향) 진단
추가로 **일부러 틀린 방식**도 같이 돌립니다 — 전체 141명으로 튜닝한 뒤 그 같은
데이터의 OOF를 보고하는 방식입니다. 두 값의 차이가 곧 선택 편향의 크기입니다:

```
optimism = (전체로 튜닝한 AUC) − (중첩 CV AUC)
```

이 값을 리포트에 같이 적어두면, 튜닝한 숫자를 튜닝 안 한 숫자처럼 발표하는 일을
구조적으로 막을 수 있습니다. **헤드라인 숫자는 항상 중첩 CV 값**입니다.

> 참고로 로컬 배선 테스트(아주 작은 탐색 예산)에서도 optimism이 **+0.085**로
> 나왔습니다. 튜닝을 하면서 이 진단을 빼면 0.08 정도는 쉽게 과대보고됩니다.

---

## 3. 파이프라인

```
run.py → train.run_experiment(RunConfig)
   │
   ├─ [전처리] features.load_split()        141명 × 151특징 (subject당 1행)
   ├─ [EDA]   eda.run_eda()                 단변량 AUC / 블록 비교 / 중복 / 의심특징 ablation
   ├─ [스플릿+학습] engine.nested_cv()       ← 헤드라인 지표 (중첩 CV, 튜닝은 fold 안에서)
   ├─ [진단]  engine.non_nested_reference()  optimism 계산 + 최종 배포 모델
   ├─ [임계값] 중첩 OOF에서 특이도 0.90/0.95 앵커 + balanced/accuracy
   ├─ [보정]  Platt calibration (중첩 OOF 기준)
   ├─ [배포]  deployment/ (모델별 저장 + 선택된 특징 인덱스 + 가중치)
   └─ [동결]  검증 예측을 라벨 열기 전에 저장 → SHA-256 기록 → 그 후 채점
```

앙상블 결합은 **log-odds 가중 평균**입니다. 랭크 평균이 AUC에는 유리하지만
점수가 "같이 예측한 배치"에 의존하게 되어 임계값이 의미를 잃습니다. log-odds
평균은 절대적·단조 점수를 유지하므로 학습 OOF에서 고른 임계값이 새 데이터에서도
그대로 통합니다.

### 의심 특징(adherence) 처리
`n_days`, `non_wear` 같은 착용 순응도 특징은 진짜 신호(무기력·참여 저하)일 수도
있지만 **모집 시기 차이 같은 프로토콜 아티팩트**일 수도 있습니다. 그래서
`SUSPECT_FEATURES`로 표시하고 ablation을 리포트에 남깁니다.

실측: 단변량 AUC 0.51~0.55(거의 무신호), 빼도 0.7721 → 0.7702로 거의 안 변합니다.
**아티팩트 우려는 없다**는 뜻이지만, 보수적으로 가려면 `MAXAUC_DROP_SUSPECT=1`로
아예 빼고 돌릴 수 있습니다.

---

## 4. 실행 방법 (학습은 사용자가 Colab에서)

### base.ipynb
```
USER_FOLDER = "SangHyo"
RUN_FILE    = "Binary_Google_MaxAUC_Tuned/run.py"
```

### 권장 Colab 환경 — **CPU (고사양/High-RAM), GPU 아님**
Google YDF는 **멀티스레드 CPU 라이브러리**라 GPU를 쓰지 않습니다. A100을 잡으면
그냥 놀게 됩니다. **코어 수가 많은 CPU 런타임**을 고르고 백그라운드 실행을
켜두세요. `ydf`는 `run.py`가 자동 설치합니다(실패하면 sklearn 대체제로 자동 강등).

### 모드
| 모드 | 예상 시간 | 용도 |
|---|---|---|
| `smoke` | ~5분 | 배선 확인용 (지표는 무의미) |
| `standard` | ~2–3시간 | 예산 축소판 |
| **`max`** | **~6–9시간** | **기본값, 성능 우선** |
| `extreme` | ~18–22시간 | 5회 반복 + 약 2배 예산 |

모드는 `--mode` 또는 `MAXAUC_MODE` 환경변수로 지정합니다. 각 모드는 데드라인에
도달하면 **바깥 repeat 경계에서** 멈추므로, 중단되어도 OOF 예측은 항상 완전합니다
(반복 횟수만 줄어들 뿐, 일부만 채점된 코호트가 나오지 않습니다).

### 환경변수
```
MAXAUC_MODE=max            모드 지정
MAXAUC_DROP_SUSPECT=1      착용 순응도 특징 제거 (보수적)
MAXAUC_NO_WEARABLE=1       MMSE만 사용
MAXAUC_SKIP_OPTIMISM=1     optimism 진단 생략 (권장하지 않음)
MAXAUC_KINDS=ydf_gbt,logreg   모델 풀 제한
```

### 재현 (재학습 없이 예측)
```
RUN_FILE = "Binary_Google_MaxAUC_Tuned/predict.py"
```

---

## 5. 산출물

| 경로 | 내용 |
|---|---|
| `training/FINAL_REPORT.json` | 헤드라인 ROC-AUC, optimism 진단, 선택된 모델/가중치/파라미터, 부트스트랩 CI |
| `training/nested_fold_records.json` | fold별 내부 AUC·가중치·선택된 하이퍼파라미터 (튜닝이 fold마다 뭘 골랐는지) |
| `eda/eda_report.json` | 단변량 AUC, 블록 비교, 중복 쌍, 의심특징 ablation |
| `training/oof_predictions_hashed.csv` | subject 해시 + 중첩 OOF 확률/점수 |
| `training/VALIDATION_PREDICTIONS_FROZEN.json` | 동결 시각 + 예측 SHA-256 |
| `training/validation_report.json` | 검증 지표(여러 임계값) |
| `deployment/` | 모델별 저장 + `cols_*.npy` + `calibrator.joblib` + `deployment.json` |

---

## 6. 성능 기대치에 대한 정직한 이야기

**0.80 달성은 보장되지 않습니다.** 근거를 정리하면:

- 고정 로지스틱 회귀 + 확장 특징만으로 이미 **0.772**가 나옵니다(위 표). 여기에
  튜닝·oblique 트리·앙상블이 얼마를 더할지가 관건이고, 현실적으로 **0.78~0.82**
  구간을 예상합니다. 0.80은 그 구간 안에 있지만 확실하지는 않습니다.
- 직전 실험들에서 확인된 **누수 없는 상한은 약 0.75~0.79**였습니다. 이번의 특징
  확장이 그 상한 자체를 조금 올렸다는 것이 위 0.772 수치의 의미입니다.
- **검증셋(33명) ROC-AUC는 학습 OOF보다 항상 낮게 나옵니다**(직전 실험: OOF
  0.766 vs 검증 0.635). MCI 7명 중 몇 명은 MMSE가 정상 범위라 원리적으로 구별이
  불가능하고, 33명 표본에서는 한 명만 뒤바뀌어도 AUC가 크게 흔들립니다.
  **1차 판단은 중첩 CV OOF 수치로** 하세요.
- 논문의 0.9는 day 단위 K-fold 누수의 산물입니다(옆 폴더 `Binary_PaperLGBM_*`에서
  실증: day-random 0.95~1.0 → GroupKFold 0.50~0.69). 이 폴더는 그 방법을 쓰지
  않으므로 0.9와 직접 비교 대상이 아닙니다.

즉 이 실험은 **"정직한 조건에서 낼 수 있는 최대치를 확인하는 것"**이 목적이며,
`optimism` 값이 함께 보고되기 때문에 결과를 과대포장할 수 없게 되어 있습니다.

---

## 7. 파일 구성

| 파일 | 역할 |
|---|---|
| `run.py` | base.ipynb 진입점(의존성/YDF 설치, 경로 해석, 모드별 예산) |
| `features.py` | 151개 특징 생성(MMSE + 웨어러블 + 수면구조 + 원형통계 일주기) |
| `eda.py` | EDA(단변량 AUC, 블록 비교, 중복, 의심특징 ablation) |
| `spaces.py` | 탐색 공간(모델 HP + 특징선택 HP) |
| `learners.py` | YDF(GBT/RF, axis·oblique) + logreg/SVM 래퍼 |
| `engine.py` | 중첩 CV, fold 내부 튜닝, 특징 선택, AUC 최적 앙상블, optimism 진단 |
| `train.py` | 오케스트레이션·리포트·배포 저장·검증 동결 |
| `predict.py` | 배포 번들 로드 → 재학습 없이 예측 재현 |
| `numeric.py` | all-NaN 컬럼에서 경고를 뿜지 않는 중앙값/표준편차 헬퍼 |

라벨 로딩은 감사 완료된
`SangHyo/Binary_Wearable_SequenceFusion_Google/data.py`의 `load_binary_labels`를
재사용하고, 진단 컬럼(`DIAG_NM`/`DIAG_SEQ`/`DOCTOR_NM`/`MMSE_NUM`/`MMSE_KIND`)은
fail-closed로 하드 배제합니다.
