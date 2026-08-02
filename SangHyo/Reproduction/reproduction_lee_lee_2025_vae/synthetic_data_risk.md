# synthetic_data_risk.md — VAE 합성자료의 해석 위험

작성일: 2026-08-02

이 문서는 **합성 Dem 자료를 무엇으로 보아서는 안 되는지**를 규정하고,
그 위반을 탐지하기 위한 진단 절차를 정의한다.

---

## 1. 핵심 위험: 12명이 4,000행이 된다

논문 표 5에서 유도된 사실(`report_inconsistencies.md` I-5):

```
Dem 피험자        12명
Dem 실제 기록     785행 → 이상치 제거 후 515행 → train 412행
합성 Dem 생성     4,000행
증강 후 Dem train 4,412행  (실제 412 + 합성 4,000, 9.7배)
```

### 위험 1-A. 표본 크기 착시

증강 후 Dem train은 CN train(5,660)의 78%다. 표면적으로 "불균형이 해소된" 것처럼 보인다.
그러나 **정보의 원천은 여전히 12명**이다. 통계적 유효 표본 크기는 4,412이 아니라
**최대 12(피험자) 또는 412(반복측정 기록)**이다.

이 착시를 언어로 방지하기 위해 본 재현은 다음 표현을 **금지**한다.

| ❌ 금지 표현 | ✅ 대체 표현 |
| --- | --- |
| "Dem 4,412 샘플로 학습" | "Dem 12명 / 실제 412행 + 합성 4,000행으로 학습" |
| "데이터 불균형을 해소" | "train fold의 클래스 행 수 비율을 조정" |
| "치매 환자 데이터를 보충" | "train fold의 Dem 기록 분포로부터 샘플을 생성" |

모든 결과표에 `n_dem_subjects` 열을 **필수**로 출력한다(`evaluation/tables.py`가 강제).

### 위험 1-B. 피험자 다양성이 늘지 않는다

VAE는 12명(train fold에서는 8명)의 기록 분포를 학습한다. `N(0, I)`에서 아무리 많이 샘플링해도
**새로운 피험자의 생리적 패턴은 생성되지 않는다.** 생성물은 학습에 쓰인 소수 피험자의
개인 특성(individual-specific offset)을 재현하며, 이는 새 피험자에 대한 일반화와 무관하다.

이것이 RQ3("VAE 증강이 새로운 독립 피험자에 대한 일반화를 실제로 향상시키는가")가
**실험 A로는 답할 수 없고 B·C가 필요한** 이유다.

---

## 2. 절대 금지 사항 (코드로 강제)

| # | 금지 | 강제 위치 |
| --- | --- | --- |
| 1 | 합성행에 가짜 subject ID 부여 | `augmentation/provenance.py` — subject는 `__SYNTHETIC__` 센티널 고정 |
| 2 | 합성행을 피험자 단위 집계에 포함 | `evaluation/aggregate.py` — `is_synthetic` 행 사전 제거 후 `assert` |
| 3 | 합성행을 validation/test에 추가 | `audit/checks.py::check_no_synthetic_in_eval` |
| 4 | 평가 fold 자료로 VAE 학습 | `audit/checks.py::check_vae_fit_scope` |
| 5 | 합성행 수를 "피험자 수"로 보고 | `evaluation/tables.py` — 피험자 수는 실제 subject의 `nunique`에서만 계산 |
| 6 | 합성행을 early stopping validation에 사용 | `models/base.py` — 내부 val 구성 시 `is_synthetic` 제외 |
| 7 | 합성행으로 scaler fit (실험 B·C) | `preprocessing/pipeline.py` — `scaler_scope: train_real_only` |

각 항목은 `tests/`의 해당 unit test로도 검증된다.

---

## 3. 합성자료 품질 진단 (`src/diagnostics/`)

증강의 효과를 분류 성능만으로 판단하지 않는다. 다음을 **항상** 산출한다.

### 3.1 분포 일치 (`synthetic_quality.py`)

| 진단 | 산출물 | 위험 신호 |
| --- | --- | --- |
| 변수별 평균·표준편차 차이 | `feature_moment_diff.csv` | 표준편차가 실제의 50% 미만 → 다양성 붕괴 |
| 변수별 분위수 차이 (p5/25/50/75/95) | `feature_quantile_diff.csv` | 꼬리가 사라짐 |
| 1D Wasserstein 거리 (변수별) | 같은 파일 | 상위 변수 확인 |
| 상관행렬 Frobenius 차이 | `corr_diff.json` | 큰 값 → 변수 간 구조 미학습 |
| 유효성 위반 건수 | `validity_violations.json` | 음수·범위이탈 다수 → 다양체 이탈 |

### 3.2 암기(memorization) 진단 — **가장 중요**

| 진단 | 정의 | 위험 신호 |
| --- | --- | --- |
| `nn_distance_to_train` | 각 합성행에서 **VAE 학습에 쓴 실제 행**까지의 최근접 거리(표준화 공간 유클리드) | 0에 가까움 → 복제 |
| `nn_distance_to_holdout` | 각 합성행에서 **학습에 쓰지 않은 실제 Dem 행**까지의 최근접 거리 | — |
| `memorization_ratio` | `median(nn_distance_to_train) / median(nn_distance_to_holdout)` | **< 0.5면 암기 경고** |
| `exact_duplicate_rate` | 실제 학습행과 완전 일치하는 합성행 비율 | > 0 이면 즉시 경고 |
| `near_duplicate_rate` | 거리 < ε(기본 0.01·√d)인 비율 | > 5% 경고 |
| `real_real_nn_baseline` | 실제 행끼리의 최근접 거리 중앙값 | 위 값들의 기준선 |

`memorization_ratio < 0.5`이면 `synthetic_quality.json`에 `"warning": "memorization_suspected"`를
기록하고 콘솔에 경고를 출력한다(예외는 던지지 않는다 — 관측 대상이지 오류가 아니다).

> 논문의 "재구성 오차 0.0002"(I-15)와 latent 500(입력 46보다 큼, I-2)의 조합은
> **암기를 강하게 시사한다.** 이 진단이 그것을 정량화한다.

### 3.3 시각화 (`projection.py`)

PCA(항상) 및 UMAP(설치된 경우)로 실제 Dem / 합성 Dem / CN / MCI를 같은 평면에 투영한다.
`projection_pca.png`, `projection_umap.png`.

- 합성점이 실제 Dem 위에 정확히 겹침 → 암기
- 합성점이 실제 Dem 중심에 뭉침 → 다양성 붕괴 (posterior collapse)
- 합성점이 CN/MCI 영역으로 번짐 → 라벨 오염

### 3.4 TSTR / TRTS (`tstr.py`)

| 이름 | 학습 | 평가 | 해석 |
| --- | --- | --- | --- |
| **TSTR** | 합성 Dem + 실제 CN/MCI | **실제 outer test** | 합성자료의 실사용 가치. 주 진단 |
| **TRTS** | 실제 train 전체 | **합성 Dem** | 합성물이 실제 분포 안에 있는지. **보조 진단만** |

TRTS 점수가 높다고 합성자료가 좋은 것이 아니다(암기해도 높게 나온다).
사용자 지시대로 **TRTS는 보조로만** 보고하며, 결과표에 그 취지를 각주로 적는다.

### 3.5 증강 방법 비교

동일한 누수 통제 파이프라인·동일 split에서 다음을 비교한다.

`none` / `class_weight` / `random_oversampling` / `smote` / `vae`

**SMOTE 주의**: `imblearn`의 SMOTE는 이웃 보간으로 합성행을 만든다.
Dem이 fold당 8명·~270행이므로 `k_neighbors`가 기본값(5)보다 커지지 않도록 하고,
**보간 이웃이 같은 피험자에서만 나오는지**를 진단으로 기록한다
(`smote_same_subject_pair_rate`). 이 비율이 높으면 SMOTE도 피험자 내부 보간에 불과하다.

---

## 4. provenance 스키마

모든 합성행은 다음 메타데이터를 갖는다(`augmentation/provenance.py`).

```python
@dataclass(frozen=True)
class SyntheticProvenance:
    is_synthetic: bool              # 항상 True
    source_class: str               # "Dem"
    source_outer_fold: int | None   # 실험 A는 None
    source_inner_fold: int | None
    generator: str                  # "vae" | "smote" | "random_oversampling"
    generator_seed: int
    generator_config_hash: str      # config dict의 SHA-256 앞 16자
    source_subject_hash: str        # 정렬된 source subject 집합의 SHA-256 앞 16자
    n_source_subjects: int          # 예: 8
    n_source_rows: int              # 예: 274
    created_at: str                 # ISO-8601
```

`source_subject_hash`는 **원본 subject ID를 노출하지 않으면서** 어떤 피험자 집합에서
생성되었는지 대조할 수 있게 한다. 감사기가 이 해시로
"outer test 피험자가 생성에 쓰였는가"를 검증한다
(`audit/checks.py::check_synthetic_source_subjects`).

`subject` 필드에는 실제 ID 대신 `"__SYNTHETIC__"` 센티널이 들어간다(§2 금지 1).

산출: `outputs/<run>/synthetic_provenance.parquet` (없으면 CSV).

---

## 5. 결과 해석 규약

결과표와 README는 다음 문장을 **반드시** 포함한다.

> 본 실험의 Dem 클래스는 **독립 피험자 12명**에서 유래한다.
> 합성 Dem 행 N개는 이 12명(각 fold에서는 8명)의 기록 분포에서 생성된 것이며
> 새로운 피험자를 의미하지 않는다.
> 피험자 단위 metric의 분모는 항상 실제 피험자 수다.

또한 다음 delta를 항상 함께 보고한다.

| delta | 의미 |
| --- | --- |
| `Δ(VAE − none)` | 증강 효과 |
| `Δ(실험 B − 실험 A)` | 누수 통제 효과 |
| `Δ(실험 C − 실험 B)` | nested 평가 효과 |
| `Δ(TSTR − 실제 학습)` | 합성자료 단독의 정보량 |

`Δ(VAE − none)`이 실험 A에서는 양수인데 실험 B·C에서 0 또는 음수라면,
그것이 RQ3에 대한 답이다 — 그리고 **그 결과 자체가 본 재현의 산출물**이다.

---

## 6. 이 재현이 주장하지 않을 것

- 합성자료가 실제 환자 데이터를 대체할 수 있다는 주장
- 4,000행 생성으로 "치매 데이터 부족이 해결"되었다는 주장
- 12명 기반 결과로부터 임상 적용 가능성에 대한 결론
- 단일 split·단일 seed 결과로부터의 모델 우열 판정

성능 향상이 관측되더라도, **그 향상이 12명 안에서의 반복측정 구조를 학습한 결과인지
실제 일반화인지**를 §3의 진단과 실험 B·C의 피험자 단위 평가로 구분해 보고한다.
