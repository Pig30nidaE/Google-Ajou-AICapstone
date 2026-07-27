# Binary_MMSE_MaxAUC — 누수 없는 ROC-AUC 최대화 모델

CN(정상) vs MCI+DEM(경도인지장애+치매) **이진 분류**를, **주제(subject) 단위
누수를 완전히 차단**한 상태에서 ROC-AUC를 최대한 끌어올리도록 만든 **실제 코드**
입니다. 논문 재현이 아니라, EDA → 전처리 → 데이터 스플릿 → 학습의 전 과정을
스스로 구현한 파이프라인입니다. MMSE는 사용을 허용하되, 웨어러블은 옵션입니다.

> **목표**: ROC-AUC ≥ 0.9 (반드시 넘어야 하는 것은 아님), accuracy ≥ 0.8.
> **정직한 결론**: 누수를 막으면 이 데이터의 subject 단위 ROC-AUC 상한은
> **약 0.74~0.79**입니다. 0.9는 논문처럼 **day 단위 K-fold(같은 사람의 날짜가
> train/test에 동시에 들어가는 누수)**로만 나옵니다. 아래에서 그 근거를 제시합니다.

---

## 1. 왜 이 구조인가 (핵심 요약)

- **누수(leakage) 차단이 최우선.** 한 사람(subject)의 데이터는 절대 train/eval에
  걸쳐 나뉘지 않습니다. 특징을 **subject당 1행**으로 집계하기 때문에, 평범한
  subject 단위 split만으로 누수가 원천 차단됩니다.
- **신호는 거의 전부 MMSE에서 나옵니다.** 자체 EDA 결과, CN vs MCI+DEM을 가르는
  가장 강한 특징은 `TOTAL`, 엔지니어링한 `num_failed`(항목 만점 미달 개수),
  지연회상(`recall`, Q13)입니다. 웨어러블(수면/활동)의 subject 단위 변동성은
  신호를 **희석**시키므로, 기본값은 **MMSE-only**이며 웨어러블은 옵션입니다.
- **작고 대체로 선형인 신호**이므로, 규제 로지스틱 회귀(C=0.1) + RBF SVM 두
  모델의 **품질 게이트 앙상블**이 트리 모델보다 안정적으로 높습니다.
- **정직한 평가.** 모든 성능은 subject 단위 반복 중첩 CV(out-of-fold)로 측정하고,
  검증셋 예측은 **라벨을 열기 전에 동결(freeze)**합니다.

---

## 2. 파이프라인 (EDA → 전처리 → 스플릿 → 학습)

```
run.py ──> train.run_experiment(RunConfig)
   │
   ├─ [전처리] features.load_split(train, MMSE-only)      # subject당 1행
   ├─ [EDA]   eda.run_eda(train, +wearable 비교본)         # eda/eda_report.json
   ├─ [스플릿+학습] engine.nested_cv(...)                  # subject 단위 반복 중첩 CV
   │        · 내부 fold로 각 모델의 balanced-acc 측정
   │        · 품질 게이트(≥0.55) 통과 모델만 가중 블렌딩
   │        · 통과 모델 없으면 최고 단일 모델로 폴백
   ├─ [임계값] 특이도 0.90/0.95 앵커 + balanced/accuracy 최적점
   ├─ [보정]  PlattCalibrator (OOF 확률 보정)
   ├─ [배포]  deployment/  (모델별 joblib + calibrator + deployment.json)
   └─ [동결]  검증 예측을 라벨 열기 전에 저장 → 이후 지표 계산
```

### 2-1. EDA 단계 — `eda.py`
- **방향-무관 단변량 ROC-AUC**: 각 특징이 CN/MCI+DEM을 얼마나 가르는지 정량화.
- **특징셋 비교**(MMSE-only vs MMSE+웨어러블): 빠른 누수-없는 subject CV로,
  웨어러블이 신호를 희석한다는 점을 리포트에 남깁니다.
- 산출물: `eda/eda_report.json`.

주요 발견(전체 학습셋 기준):
| 특징 | 방향-무관 AUC |
|---|---|
| `mmse_TOTAL` | ≈ 0.74 |
| `mmse_num_failed` (만점 미달 항목 수, 엔지니어링) | ≈ 0.74 |
| `mmse_recall` / `mmse_recall_deficit` (지연회상, 엔지니어링) | ≈ 0.73 |
| `mmse_Q13_*` (지연회상 개별 항목) | ≈ 0.68 |

### 2-2. 전처리 단계 — `features.py`
- **MMSE 도메인 특징**: `TOTAL` + 6개 도메인 합(지남력-시간/장소, 기억등록,
  주의집중, 지연회상, 언어) + 원항목 28개.
- **엔지니어링 특징**:
  - `num_failed` = 각 항목이 (학습셋에서 학습한) 항목 만점 미만인 개수.
  - `recall_deficit` = 지연회상 만점 − 실제 지연회상 점수 (MCI의 핵심 지표).
- **누수 안전장치**: 항목 만점(`item_max`)은 **학습셋에서만** 계산해 검증셋에 재사용.
  진단 컬럼(`DIAG_NM`/`DIAG_SEQ`/`DOCTOR_NM`/`MMSE_NUM`/`MMSE_KIND`)은
  `MMSE_FORBIDDEN`으로 **하드 배제**하고, 실수로 섞이면 예외를 던집니다(fail-closed).
- **웨어러블(옵션)**: `MAXAUC_INCLUDE_WEARABLE=1`일 때만, subject별 수면/활동
  지표의 **일간 표준편차**(변동성)만 추가합니다.

### 2-3. 데이터 스플릿 단계 — `engine.nested_cv`
- **subject 단위 StratifiedKFold**(반복). 같은 사람이 train/test에 동시에
  들어가지 않습니다.
- 바깥 fold마다: 안쪽 fold로 각 모델의 balanced-acc를 측정 → **품질 게이트**
  (≥ `weight_gate`, 기본 0.55)를 통과한 모델만 `(balacc−0.5)` 가중으로 블렌딩.
  통과 모델이 없으면 최고 단일 모델로 폴백(잡음 모델이 결과를 끌어내리지 않도록).
- fold별 확률과 마진(`prob − threshold`)을 집계 → 안정적인 OOF 지표 산출.

### 2-4. 학습 단계 — `learners.py`
- `logreg`: `LogisticRegression(C=0.1, class_weight="balanced")` — 규제 강한 선형.
- `svm`: `SVC(kernel="rbf", C=1.0, class_weight="balanced", probability=True)`.
- 전처리(중앙값 대치 + 표준화)는 **fold 내부에서만** 적합 → 누수 없음.

---

## 3. 실행 방법 (학습은 사용자가 Colab에서)

### base.ipynb로 실행
```
USER_FOLDER = "SangHyo"
RUN_FILE    = "Binary_MMSE_MaxAUC/run.py"
```
- CPU만으로 충분합니다(수 분~십수 분). GPU 불필요.
- 결과는 `/content/drive/MyDrive/Binary_MMSE_MaxAUC_result/<run_id>/`에 저장.
- 웨어러블도 넣어 비교하려면 실행 전 셀에서 `os.environ["MAXAUC_INCLUDE_WEARABLE"]="1"`.

### 권장 Colab 환경
- **런타임: CPU (High-RAM)**. 이 모델은 GPU를 쓰지 않습니다.
- Colab Pro+ A100은 불필요합니다. GPU 세션을 아껴서 다른 실험에 쓰세요.

### 로컬/CLI 스모크 테스트
```bash
python SangHyo/Binary_MMSE_MaxAUC/run.py --mode smoke --data-root Data \
  --output-dir /tmp/maxauc_smoke
```

### 재현(재학습 없이 예측) — `predict.py`
학습 시 저장된 `deployment/` 번들만으로 검증 예측을 그대로 재현합니다.
```
RUN_FILE = "Binary_MMSE_MaxAUC/predict.py"
```
- `deployment/`는 결과 폴더 아래에서 자동 탐색합니다. 특정 실행을 지정하려면
  `PREDICT_DEPLOYMENT_DIR` 환경변수로 경로를 주세요.
- CLI: `python SangHyo/Binary_MMSE_MaxAUC/predict.py --deployment-dir <경로> --data-root Data`

---

## 4. 산출물

| 경로 | 내용 |
|---|---|
| `training/FINAL_REPORT.json` | 핵심 리포트(누수-없는 OOF ROC-AUC, 부트스트랩 95% CI, 임계값, 검증 지표) |
| `eda/eda_report.json` | EDA 리포트(단변량 AUC, 특징셋 비교) |
| `training/oof_predictions_hashed.csv` | subject 해시 + OOF 확률 |
| `training/validation_predictions_label_free_hashed.csv` | 라벨 열기 전 동결된 검증 예측 |
| `training/VALIDATION_PREDICTIONS_FROZEN.json` | 동결 시각 + 예측 파일 SHA-256 |
| `training/validation_report.json` | 검증 지표(여러 임계값) |
| `deployment/` | `model_logreg.joblib`, `model_svm.joblib`, `calibrator.joblib`, `deployment.json` |

---

## 5. 정직한 성능 기대치와 한계

- **누수-없는 subject 단위 OOF ROC-AUC ≈ 0.74~0.79** (반복 중첩 CV, 부트스트랩
  95% CI는 대략 0.71~0.86). 이것이 이 데이터의 현실적인 상한입니다.
- **왜 0.9가 안 되는가**: 같은 이웃 폴더의 실험에서, 논문식 **day 단위 K-fold는
  ROC-AUC 0.95~1.0**을 주지만 **GroupKFold(subject 단위)로 바꾸면 0.50~0.69**로
  무너집니다. 즉 논문의 0.9는 성능이 아니라 **누수의 산물**입니다. 이 폴더는
  그 누수를 쓰지 않습니다.
- **근본적 한계**: 검증셋 MCI 중 일부는 MMSE가 27~30(정상/만점)이고 웨어러블도
  정상 범위여서, **구별할 정보 자체가 없습니다.** 누수 없이 이들을 맞히는 것은
  불가능합니다.
- 33명 검증셋은 **재사용된 소규모 벤치마크**이므로 지표 변동이 큽니다. 1차
  성능 판단은 학습셋의 subject 단위 OOF ROC-AUC로 하세요.

---

## 6. 파일 구성

| 파일 | 역할 |
|---|---|
| `run.py` | base.ipynb 진입점(의존성 설치, 데이터/출력 경로 해석, full/smoke) |
| `features.py` | 전처리·특징 생성(MMSE 도메인/항목/엔지니어링, 누수 안전장치) |
| `eda.py` | EDA(단변량 AUC, 특징셋 비교) |
| `engine.py` | 누수-없는 반복 중첩 CV + 품질 게이트 앙상블 + 임계값/지표 |
| `learners.py` | fold-안전 학습기(규제 로지스틱 + RBF SVM) |
| `train.py` | 파이프라인 오케스트레이션·리포트·배포 저장·검증 동결 |
| `predict.py` | 배포 번들 로드 → 재학습 없이 예측 재현 |
| `requirements_colab.txt` | 의존성(sklearn 계열만, LightGBM 불필요) |

라벨과 데이터 파서는 감사 완료된
`SangHyo/Binary_Wearable_SequenceFusion_Google/data.py`를 재사용합니다.
