# Binary_Google_SOTA_DualTrack

`report.docx.pdf`의 V26/V29/V41 앙상블을 **Google YDF 모델로 치환**하고,
**누수 경로를 모두 제거**해서 다시 구현한 CN vs MCI+DEM 이진 분류 실험.

- 누수 감사 상세: [`LEAKAGE_AUDIT_KO.md`](./LEAKAGE_AUDIT_KO.md)
- **학습은 아직 실행하지 않았다.** 아래 수치는 배선 확인용 smoke 결과이며
  성능이 아니다 (`AGENTS.md` 계약 8번).

---

## 1. 실행 방법

`base.ipynb` 셀 2에서:

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Google_SOTA_DualTrack/run.py"
```

### ⚠️ 노트북에서는 환경변수로 설정한다

`base.ipynb`는 `runpy.run_path`로 이 파일을 **커널 안에서** 실행하므로
`sys.argv`에 커널 인자(`-f .../kernel-xxx.json`)가 그대로 남아 있고,
`--mode` 같은 CLI 인자를 넘길 방법이 없다. 그래서 노트북에서는 **셀 2에
아래를 추가**해 설정한다(모르는 인자는 무시하도록 되어 있으므로 커널 인자는
문제를 일으키지 않는다).

```python
import os
os.environ["SOTA_MODE"]  = "full"        # smoke | standard | full
os.environ["SOTA_TRACK"] = "both"        # wearable | mmse_fusion | both
os.environ["SOTA_SMOTE"] = "borderline"  # borderline | plain | none
```

지원 변수: `SOTA_MODE`, `SOTA_TRACK`, `SOTA_SMOTE`, `SOTA_SEED`,
`SOTA_DATA_ROOT`, `SOTA_OUTPUT_ROOT`. 아무것도 설정하지 않으면
`full` + `both` + `borderline`으로 돈다.

`DATA_ROOT`는 `base.ipynb`가 주입하는 값을 그대로 쓴다(`SOTA_DATA_ROOT`로
덮어쓸 수 있음).

### 런타임

**CPU + High-RAM. GPU 불필요.** YDF는 멀티스레드 CPU 라이브러리라 코어 수가
유일하게 의미 있는 자원이고, A100을 잡아도 놀기만 한다. Colab에서 가장 코어가
많은 CPU 런타임을 고르면 된다.

### 모드

| 모드 | 예상 시간 | 내용 |
| --- | --- | --- |
| `smoke` | ~2분 | 배선 확인 전용. **성능 아님** |
| `standard` | ~30분 | repeat 2, 후보 25개, search 8 |
| `full` (기본) | ~2시간 | repeat 3, 후보 40개, search 12 |

soft budget은 5시간 45분이며, 도달하면 **repeat 경계에서** 멈춘다. 따라서
잘려도 OOF는 항상 완결된 상태이고, repeat 수만 줄어든다.

```bash
python -m Binary_Google_SOTA_DualTrack.run --mode full --track both
```

주요 옵션: `--track {wearable,mmse_fusion,both}`,
`--smote {borderline,plain,none}`, `--seed`, `--data-root`, `--output-root`.

### 정적 계약 테스트 (학습 불필요, 2초)

```bash
python -m pytest SangHyo/Binary_Google_SOTA_DualTrack/tests -q
```

---

## 2. 두 개의 트랙

| 트랙 | 입력 | 라벨 독립성 |
| --- | --- | --- |
| `wearable` | Activity + Sleep만 | ✅ 독립 |
| `mmse_fusion` | + MMSE 문항/총점 | ⚠️ **독립 아님** |

`wearable`은 `3.CognitiveFunction` 경로를 **아예 열지 않는다.** 열려고 하면
`LeakageGuardError`로 즉시 중단한다(fail-closed).

`mmse_fusion`은 `DIAG_NM`·`DIAG_SEQ`·`MMSE_NUM`·`MMSE_KIND`를 로드 직후
제거하고 사후 재검증한다. 그래도 **MMSE는 `DIAG_NM`을 부여한 인지검사 자체**라
라벨의 대리변수다. 이 트랙 수치는 "인지검사를 쓰면 얼마나 오르는가"의 **상한**
으로만 읽어야 한다 (감사 문서 6절).

두 트랙은 동일한 fold·동일한 파이프라인을 통과하므로 나란히 비교할 수 있다.

---

## 3. Google 모델 치환

보고서는 LightGBM(Microsoft) · CatBoost(Yandex) · XGBoost(DMLC) ·
RandomForest(scikit-learn)를 섞는다. 이를 Google **YDF**(TensorFlow Decision
Forests의 엔진)로 옮기되, 각 모델의 **귀납 편향**을 맞췄다.

| 보고서 | YDF 치환 | 근거 |
| --- | --- | --- |
| LightGBM | `GradientBoostedTreesLearner`, `growing_strategy="BEST_FIRST_GLOBAL"`, `max_num_nodes=31` | leaf-wise 성장 = LightGBM의 정체성. 보고서의 `num_leaves=31`, `lr=0.05` 그대로 |
| XGBoost | 같은 learner, `growing_strategy="LOCAL"`, `max_depth=6` | level-wise 성장 = XGBoost 방식. 보고서의 `max_depth=6` |
| CatBoost | 같은 learner + `split_axis="SPARSE_OBLIQUE"` | oblique 분기는 노드마다 여러 특징을 조합 → CatBoost가 맡던 "상관 낮은 제3의 의견" 역할 |
| RandomForest | `RandomForestLearner` | 배깅 + OOB |
| **SHAP** | **YDF 내장 variable importance** | `MEAN_DECREASE_IN_AUC` → `SUM_SCORE` → `NUM_AS_ROOT` 순으로 사용. Google 네이티브라 의존성 추가 없음 |
| **Optuna** | **fold-local random search** | YDF 자체 튜너는 내부 홀드아웃을 스스로 잘라내서 fold 경계가 흐려진다. 명시적 inner CV로 채점해 감사 가능하게 유지 |

소프트 보팅 가중치는 보고서 그대로 **.40 / .20 / .20 / .20**을 쓴다(적합하지
않는 고정 상수라 누수가 아니다). 스태킹 메타러너만 4입력 Logistic Regression을
쓰는데, 결합기를 일부러 작게 두어 YDF 베이스 모델이 결과를 지배하게 했다.

> TabNet도 Google(Cloud AI) 모델이지만 넣지 않았다. 141명 규모에서 신경망은
> 이 저장소의 2·3차 실험에서 이미 불안정했고(`AGENTS.md` 2-3절), 보고서 앙상블도
> 전부 트리 기반이라 굳이 축을 흐릴 이유가 없다.

---

## 4. 누수 제거: fold 구조

```
outer StratifiedKFold  (사람 단위, 1인 1행)
  │
  ├─ outer-training fold ───────────────────────────────────┐
  │    fold-local 중앙값 대치 / 상수 컬럼 제거              │
  │    YDF importance 순위          ← 이 fold만 사용        │
  │    전진 선택, inner CV로 채점   ← 이 fold만 사용        │
  │    random search, inner CV로 채점                       │
  │    inner OOF → 스태킹 메타러너 + Youden 임계값          │
  │    SMOTE는 inner-train / outer-train 에만               │
  └──────────────────────────────────────────────────────────┘
  │
  └─ outer-validation fold : 채점만. 어떤 적합에도 쓰지 않음
```

보고서 대비 이동한 것:

| 단계 | 보고서 | 이 실험 |
| --- | --- | --- |
| 특징 순위 | 전체 174명 SHAP | outer-training fold의 YDF importance |
| `K` 선택 | 보고 대상 5-Fold AUC | inner CV |
| 하이퍼파라미터 | 보고 대상 CV로 Optuna 30회 | inner CV로 nested search |
| 스태킹 메타러너 | 전체 OOF로 학습·평가 | inner OOF로 학습, outer엔 적용만 |
| 임계값 | 보고 대상 OOF에서 Youden | inner OOF에서 Youden |
| 홀드아웃 | 없음 (141+33=174 통합) | val 33명 동결 후 1회 평가 |

**사람 1명 = 1행**이므로 같은 사람이 분할 양쪽에 나타날 수 없다. 원 논문의
수치를 부풀렸던 window 단위 분할 문제가 구조적으로 차단된다.

### Validation 동결 프로토콜

1. Training 141명으로 전체 파이프라인 재적합
2. 33명 예측을 CSV로 쓰고 **SHA-256 기록** (`validation_freeze.json`)
3. **그 다음에야** val 라벨을 열어 1회 평가

라벨을 연 뒤 seed·특징·임계값을 고쳐 다시 맞추지 않는다.

---

## 5. 특징 (345개, `wearable` 기준)

보고서의 "약 250개" 집계를 따르되 결측 sentinel 처리를 고쳤다.

- **일별 스칼라** (activity 26 + sleep 30개) × mean/std/median/IQR
- **intraday 배열** (1분 MET, 5분 HR, 5분 RMSSD) × 9종 통계
  - `hr_5min` / `rmssd_5min`의 **`0`은 센서 결측 → NaN** (보고서는 평균에 포함)
- **수면단계 전환** (hypnogram 1=deep/2=light/3=rem/4=awake): 4개 단계 비율 +
  12개 전환 비율 + 전환율 + 파편화 지수
- **활동 클래스 전환** (0=nonwear … 5=high): 동일 구조
- **일주기 위상**: bedtime의 hour sin/cos + 규칙성(원형 평균 결과 길이).
  절대 날짜는 버린다
- **변동계수** `_cv = std/|mean|` (V29의 "불규칙성")

`mmse_fusion`은 여기에 MMSE 27개가 붙어 372개.

### 금지 사항 (코드로 강제)

- 관측 일수(35~120일)는 특징에 넣지 않는다. 전환 변수는 전부 **비율**이다.
- `assert_feature_names_clean`이 `diag` / `label` / `target` / `n_days` / `count`
  등을 **단어 경계 매칭**으로 차단한다. (Oura 실제 채널
  `activity_score_meet_daily_targets`가 부분 문자열 매칭에 걸렸던 적이 있어
  단어 경계로 고쳤고, 회귀 테스트를 남겼다.)

---

## 6. 보고 지표

CN이 다수 클래스라(Training 60.3%, Validation 78.8%) **Accuracy 단독 보고는
금지**다. 모든 리포트가 함께 싣는 값:

Balanced Accuracy · MCI+DEM Recall · CN Specificity · ROC-AUC · Precision · F1 ·
혼동행렬 · 부트스트랩 95% CI · **all-CN baseline**.

outer fold 결합은 확률이 아니라 `score − fold threshold` **margin** 평균을 쓴다
(fold마다 운영점이 다르므로). `AGENTS.md` 2-4절의 방식과 같다.

fold 간 특징 선택 Jaccard 유사도도 보고한다 — 3차 실험에서 ≈0.22로 무너졌던
지표라 계속 노출시킨다.

---

## 7. 출력물

```
<폴더명>_result/<UTC_RUN_ID>/
├── run_manifest.json                    # 설정, 엔진, 요약, 주의사항
├── wearable/
│   ├── data_audit.json                  # 인원/특징 수, 관측일수(특징 아님)
│   ├── eda_report.json                  # Training 전용 EDA
│   ├── nested_cv_report.json            # nested OOF 지표, fold별 기록, 안정성
│   ├── final_model_summary.json         # 최종 선택 특징/파라미터/임계값
│   ├── validation_predictions_frozen.csv
│   ├── validation_freeze.json           # SHA-256 + 동결 시각
│   └── validation_report.json           # 1회 평가 결과
└── mmse_fusion/                         # 동일 구조
```

Drive가 마운트되지 않은 Colab 환경이면 로컬에 잘못 쓰지 않고 즉시 중단한다.

---

## 8. smoke 실행 확인 (2026-07-28, 로컬)

`--mode smoke --track both`, YDF 0.16.1, macOS. **배선 확인일 뿐 성능이 아니다**
(특징 4개, outer fold 3개, search 없음).

- 정적 계약 테스트 24개 통과
- 두 트랙 모두 EDA → nested CV → 동결 → 평가까지 완주, 산출물 14개 생성
- `--smote none`(트리 자체 결측 분기) 경로도 별도 확인 완료
- 코호트 검증: Training 141명 (CN 85 / MCI 47 / Dem 9), Validation 33명,
  ID 중복 0건, Gait/Sleep 라벨 사본 일치
- 결측률 0.002%
- 동결 임계값 ≈0.44–0.46 (보고서 0.471과 근사)

### smoke에서 이미 드러난 것

**inner CV AUC 0.75–0.82 vs nested OOF AUC 0.53.**

이 격차가 감사 문서 1절의 요지를 그대로 보여준다. 보고서가 보고한 0.7818은
inner 쪽 수치에 해당하고, 같은 파이프라인을 정직하게 nested로 돌리면 0.53이다.
`AGENTS.md` 2-3절이 3차 실험에서 관측한 "inner 55.2% → outer 43.1%"와 같은 현상이다.

smoke 기준 `mmse_fusion`은 nested OOF AUC ≈0.60–0.62로 wearable(≈0.53)보다
확실히 높다. 이 격차가 곧 "인지검사 정보가 들어오면 얼마나 오르는가"이며,
동시에 그것이 라벨 대리변수라는 경고이기도 하다.

단변량 상위 신호는 **수면단계 전환**이 차지했다
(`slp_stage_trans_deep_to_light` 0.144, `slp_sleep_score_deep_std` 0.140,
`slp_stage_trans_light_to_deep` 0.137, `slp_sleep_restless_iqr` 0.136).
팀 최종 보고서의 "수면 자율신경·수면단계 지표가 상위 신호"라는 SHAP 결론과
독립적으로 일치한다 — 보고서에서 가져온 전환 변수가 실제로 쓸모 있는 부분이었다는 뜻.

---

## 9. 알려진 한계 (보고서에 반드시 명시할 것)

1. **nested OOF는 정직한 내부 비교값이지 독립 코호트 추정치가 아니다.** 같은
   141명에 대한 이전 실험·EDA를 보면서 특징과 모델 후보를 골랐으므로 실험
   전체 수준의 선택 편향은 남는다 (`AGENTS.md` 계약 9번).
2. **`mmse_fusion`은 라벨 독립이 아니다.** 상한으로만 읽을 것.
3. **Validation은 33명(MCI+DEM 7명)뿐**이라 CI가 매우 넓다. 한 명 차이로 Recall이
   0.14씩 움직인다.
4. **웨어러블만으로 Accuracy 0.90 / Balanced Accuracy 0.80은 사실상 어렵다.**
   팀 최종 보고서가 사람 단위 평가에서 이미 확인한 결론이며, 이 코드가 그 수치를
   보장하지 않는다.
5. SMOTE를 켜면 fold-local 중앙값 대치가 선행된다(합성에는 완전한 행이 필요).
   보고서가 선호한 "트리 자체 결측 분기" 경로를 그대로 쓰려면 `--smote none`.
