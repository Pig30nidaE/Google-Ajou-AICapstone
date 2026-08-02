# assumptions.md — 논문 미보고 항목과 본 재현이 채택한 가정

작성일: 2026-08-02

원칙:

1. 논문에 명시된 값은 **그대로** 사용한다.
2. 미보고 항목은 **보수적 기본값**을 쓰고 여기에 기록한다.
3. 미보고 항목을 "원 논문 구현"이라고 부르지 않는다. 설정 이름에 `assumed_` 접두를 붙인다.
4. 모든 가정은 config로 변경 가능해야 한다. 코드에 상수로 박지 않는다.
5. 논문과 데이터가 충돌하면 조용히 고치지 않고 여기와 `report_inconsistencies.md`에 남긴다.

---

## A. 데이터 가정

### A-1. 코호트 = AI-Hub Training + Validation 합본 `verified`

**근거**: 합본이 174명 / 12,183행 / (7,737, 3,661, 785)로 논문 표 3과 **정확히 일치**한다.
가정이 아니라 **실측 확인된 사실**이다. config: `data.use_official_split: false`.

### A-2. activity ↔ sleep 결합은 위치 기반 `assumed_join_positional`

**논문**: 결합 방법 미보고. 날짜 컬럼이 데이터에 **존재하지 않는다.**

**가정**: `EMAIL` 열이 두 파일에서 행 단위로 완전히 동일한 순서임을 실측 확인했으므로,
동일 인덱스끼리 결합한다.

```python
assert (activity["EMAIL"].values == sleep["EMAIL"].values).all()
X = pd.concat([activity[ACT], sleep[SLP]], axis=1)
```

로더가 매번 이 `assert`를 수행하고 실패하면 `SchemaError`를 던진다.
config: `data.join_mode: positional` (대안 없음 — 날짜가 없어 다른 방법이 불가능).

### A-3. 날짜 순서 정보는 사용하지 않는다 `assumed_no_temporal_feature`

논문은 일별 기록을 i.i.d. 행으로 다루며 시계열 구조를 쓰지 않는다.
본 재현도 동일하게 하되, 행 순서를 피험자 내 시간순으로 **가정하지 않는다**.
(피험자 단위 split에서는 시간 정보가 필요 없으므로 무해하다.)

### A-4. 중복 컬럼쌍을 그대로 둔다 `assumed_keep_duplicate_temperature`

`sleep_temperature_delta` == `sleep_temperature_deviation` (12,183행 전체 동일, 실측).
논문이 46개를 사용했다고 명시했으므로 **46개를 그대로 쓴다**.
config: `features.drop_duplicate_columns: false` (기본), `true`는 45개 민감도 변형.
로더는 어느 쪽이든 중복 쌍을 경고 로그로 남긴다.

---

## B. 전처리 가정

### B-1. percentile 이상치의 정확한 의미 `assumed_percentile_semantics`

**논문 §5.1**: "각 특성값의 상위 및 하위 10% 범위를 벗어나는 데이터를 제외"

해석이 최소 4가지로 갈리므로 전부 config로 분리한다.

| config | 의미 |
| --- | --- |
| `scope: global`, `action: drop_row` | 전체 학습자료 분위수, 한 변수라도 벗어나면 행 삭제 **(기본)** |
| `scope: per_class`, `action: drop_row` | 클래스별 분위수, 행 삭제 |
| `scope: global`, `action: clip` | 전체 분위수로 변수별 clipping (행 수 불변) |
| `scope: per_class`, `action: clip` | 클래스별 분위수로 clipping |

**기본값 근거**: "데이터를 제외"라는 표현은 행 삭제를 뜻하고, 클래스별 분위수를 쓴다는
언급이 없으므로 전역이 자연스럽다. **단 이 해석으로는 논문 행 수가 재현되지 않는다**
(잔존율 3.05%, `report_inconsistencies.md` I-1 증거 B). 이는 문서화된 예상 결과다.

### B-2. Isolation Forest 하이퍼파라미터 `assumed_isoforest_params`

**논문**: 알고리즘 이름만 있고 파라미터 전무.

**가정**: `contamination=0.1` (논문의 "10%" 및 전체 잔존율 90.0%와 정합, I-1 증거 A),
`n_estimators=100`, `max_samples="auto"`, `random_state=seed` — 나머지 scikit-learn 기본값.

`contamination`은 실험 A에서 고정 상수이며, 실험 C의 inner CV에서만 `{0.05, 0.1, 0.15}`로 탐색한다.

### B-3. imputer는 median `assumed_median_imputer`

**논문**: 결측 처리 미보고. **실측: 46개 변수 전체에서 결측 0건** → 사실상 no-op.
파이프라인 완결성과 사용자 지시 3(fold 내 fit)을 위해 `SimpleImputer(strategy="median")`를
train fold에서만 fit한다. 실행 시 대체된 셀 수를 로그로 남긴다(기대: 0).

### B-4. StandardScaler fit 범위 `assumed_scaler_scope`

**논문**: 미보고 (I-8).

| 실험 | 기본값 | 근거 |
| --- | --- | --- |
| A | `all_data` | §5.1의 "전처리 된 데이터를 …분할" 흐름 |
| B, C | `train_real_only` | 합성행이 통계량을 지배하지 않도록 (합성 4,000 vs 실제 412) |

실험 A에서 `train_with_synthetic`, `train_real_only`도 override로 실행 가능.

### B-5. 라벨 인코딩은 no-op `verified_noop`

46개 변수 전부 수치형(실측). §4.2의 "범주형 변수는 라벨 인코딩" 단계는 적용 대상이 없다.
구현은 존재하되 대상 컬럼 0개임을 로그로 남긴다.

---

## C. 분할 가정

### C-1. 실험 A의 8:1:1은 행 단위 층화 무작위 분할 `assumed_row_stratified_split`

**논문**: 분할 단위·방법·seed 미보고. 표 5의 정확한 10% 값들은 행 단위와만 정합(I-6).

**가정**:
```python
train, tmp  = train_test_split(rows, test_size=0.2, stratify=y, random_state=seed)
valid, test = train_test_split(tmp,  test_size=0.5, stratify=y_tmp, random_state=seed)
```
`stratify`는 클래스만 사용하고 **피험자는 고려하지 않는다** — 이것이 논문 절차의 재현이다.

표 5와 정확히 같은 정수(707/708 등)가 나오지 않을 수 있다. 실제 분할 결과를
`row_counts.csv`에 논문 표 5와 나란히 출력하고 차이를 보고한다.

### C-2. 실험 B·C의 group CV 구현 `assumed_group_cv_impl`

사용자 지시대로 **StratifiedGroupKFold, group = 피험자 ID**를 기본으로 한다.

```yaml
split:
  method: stratified_group_kfold   # sklearn.model_selection.StratifiedGroupKFold
  n_splits: 3
  groups: subject
```

sklearn 구현은 **행 단위 클래스 비율**을 맞추므로 fold별 Dem 피험자 수가 균등하다는
보장이 없다. 따라서 다음을 **강제**한다.

1. 모든 fold의 train·eval에 CN·MCI·Dem 피험자가 최소 1명씩 존재해야 한다. 아니면 `SplitError`.
2. 대안 구현 `split.method: subject_stratified`를 제공한다.
   피험자 테이블(174행)에 `StratifiedKFold`를 적용해 fold당 Dem 4명을 **보장**한다.
3. `--dry-run`이 두 방식의 fold별 피험자 구성을 표로 출력한다.

**기본값은 지시에 따라 `stratified_group_kfold`**이며, 검증 실패 시 사용자가
`subject_stratified`로 전환하도록 오류 메시지에 명시한다.

### C-3. n_splits = 3, n_repeats = 1 `assumed_fold_counts`

Dem 피험자 12명 → 3-fold에서 fold당 4명. 사용자 지시대로 outer 5-fold는 기본값이 아니다.
`n_repeats`는 실험 B·C 모두 config로 1~10 설정 가능하며 기본 1(계산량 고려).

---

## D. VAE 가정

### D-1. 구조 세부 `assumed_vae_layer_order`

**논문**: "512개와 256개의 뉴런을 가진 두 개의 완전 연결 층", BatchNorm, Dropout 0.3.
층 순서(Linear→BN→ReLU→Dropout인지 Linear→ReLU→BN→Dropout인지)는 미보고.

**가정**: `Linear → BatchNorm1d → ReLU → Dropout(0.3)` (가장 통용되는 순서).
`vae.layer_order`로 변경 가능.

```
Encoder: 46 → [512] → [256] → (mu: 256→L, logvar: 256→L)
Decoder: L → [256] → [512] → 46          # 마지막 층은 선형 출력, 활성화 없음
```

**출력층 활성화**: 미보고. 입력이 표준화된 실수이거나 원 단위 양수이므로
**항등(linear)**을 기본으로 한다. `vae.output_activation: linear | sigmoid`.
`sigmoid`는 min-max 정규화와 함께 쓸 때만 의미가 있으므로 기본값이 아니다.

### D-2. KL 가중치 beta `assumed_beta_1`

**논문**: "재구성 오차와 KL 발산의 **가중합**" — 가중치 값 미보고.

**가정**: `beta = 1.0` (표준 VAE). config `vae.beta`.
재구성 손실은 **변수당 평균 MSE**로 계산하고, KL은 latent 차원 **합**으로 계산한다
(`vae.recon_reduction: mean_per_feature | sum`, `vae.kl_reduction: sum | mean`).
이 조합이 달라지면 beta의 실효값이 수십~수백 배 달라지므로 두 reduction을 모두 로그에 남긴다.

### D-3. epoch / batch size / early stopping `assumed_vae_training_schedule`

**논문**: 전부 미보고.

**가정**: `epochs: 300`, `batch_size: 64`, `early_stopping.patience: 30`,
`early_stopping.monitor: val_total_loss`, `min_delta: 1e-5`.

early stopping용 validation은 **VAE 학습에 쓰는 Dem 기록을 피험자 단위로 분리**해서 만든다
(`vae.val_fraction: 0.2`, `vae.val_split_by: subject`). outer test는 절대 쓰지 않는다.

> ⚠️ train fold의 Dem 피험자가 8명일 때 20%면 1~2명이다. 표본이 극히 작아
> early stopping 자체가 불안정하다. `vae.early_stopping.enabled: false`로 끄고
> 고정 epoch을 쓰는 변형도 제공한다. → `unresolved_questions.md` Q8.

### D-4. VAE fit 범위 `assumed_vae_fit_train_dem_only`

**논문**: 미보고 (I-9).

| 실험 | 기본값 |
| --- | --- |
| A | `train_dem_only` (관대한 해석). `all_dem` 변형도 제공 |
| B, C | `train_dem_only` **고정**, 감사기가 강제 |

전체 클래스 자료로 VAE를 학습하는 변형(`fit_scope: all_classes`)은
**논문에 그런 서술이 없으므로 구현하되 기본 비활성**이며, 사용 시 경고를 출력한다.

### D-5. VAE 입력 공간 `assumed_vae_input_space`

**논문 §4.2**의 순서상 스케일링이 증강 뒤에 오므로 VAE는 **원 단위**에서 학습된 것으로 읽힌다.

| 실험 | 기본값 | 근거 |
| --- | --- | --- |
| A | `raw` | §4.2 서술 순서 |
| B, C | `scaled` | 4자릿수 스케일 차이로 MSE가 대형 변수에 지배됨 |

`scaled`일 때 생성물은 **inverse_transform으로 원 단위로 되돌린 뒤** 유효성 검사를 한다.

### D-6. 생성값 후처리 `assumed_postprocess`

**논문**: 후처리 언급 없음.

| 항목 | 기본값 | 근거 |
| --- | --- | --- |
| `enforce_nonnegative` | **on** | 43개 변수가 실측상 항상 ≥ 0 |
| `clip_to_train_range` | **on** | train fold 관측 [min, max]로 clip |
| `round_integer_valued` | **off** | 논문 미보고. 41개 정수형 변수에 적용하는 변형 제공 |

세 옵션 각각의 **적용 건수를 provenance에 기록**한다. 위반 비율이 높으면 VAE가
데이터 다양체를 벗어나 생성하고 있다는 신호다. → `synthetic_data_risk.md` §3.

### D-7. 합성 행 수 `derived_n_synthetic_4000`

표 5에서 **4,000**으로 유도(I-5). 가정이 아니라 **산술 유도값**이지만
논문이 직접 보고한 값은 아니므로 여기에 기록한다.

- 실험 A: `n_synthetic: 4000` 고정.
- 실험 B: `synthetic_ratio`로 지정(기본 `match_majority`가 아니라 논문과 같은 **9.7배**를
  재현하도록 `ratio_to_real_dem: 9.71`을 기본으로 두되, fold마다 실제 Dem 행 수가 다르므로
  절대 개수가 아닌 배수로 계산). 이 값은 **재선택하지 않는다**(지시: B는 논문 설정 고정).
- 실험 C: inner CV에서 `{0, 1, 3, 5, 10}` 배수 중 선택.

### D-8. 합성행에는 subject ID를 부여하지 않는다 `policy_no_fake_subject`

사용자 지시 15·16에 따라 합성행의 `subject`는 `None`(내부적으로 `__SYNTHETIC__` 센티널)이며
**피험자 단위 집계에 절대 들어가지 않는다**. 감사기와 unit test가 강제한다.

---

## E. 분류기 가정

### E-1. XGBoost 미보고 파라미터 `assumed_xgb_defaults`

논문 보고: `objective=multi:softmax`, `max_depth=6`, `learning_rate=0.1`.

**가정**: `n_estimators=100`, `subsample=1.0`, `colsample_bytree=1.0`,
`reg_lambda=1`, `reg_alpha=0`, `min_child_weight=1`, `tree_method="hist"`.
확률 출력이 필요하므로 **`objective="multi:softprob"`를 사용**한다
(softmax와 학습은 동일하고 출력만 확률). 이 변경을 로그에 명시한다.

### E-2. DNN 미보고 파라미터 `assumed_dnn_schedule`

논문 보고: 512-256-128-64-32, ReLU, L2, BatchNorm, Dropout 0.5, softmax, CE, Adam.

**가정**: `l2: 1e-4`, `lr: 1e-3`, `epochs: 200`, `batch_size: 128`,
`early_stopping.patience: 20` (train 내부 validation에서만).
층 순서는 `Linear → BatchNorm → ReLU → Dropout`.

### E-3. TabNet 미보고 파라미터 `assumed_tabnet_defaults`

논문 보고: `n_d = n_a = 64`, `n_steps = 5`.

**가정**: `gamma=1.3`, `lambda_sparse=1e-3`, `n_independent=2`, `n_shared=2`,
`lr=2e-2`, `max_epochs=200`, `patience=20`, `batch_size=1024`, `virtual_batch_size=128`
(pytorch-tabnet 관례값).

### E-4. 🔴 Wide & Deep의 Wide 입력 `assumed_wide_linear_all_features`

**논문**: Deep 컴포넌트만 서술(256-128-64, ReLU, Dropout 0.3).
**Wide 컴포넌트에 무엇이 들어가는지 전혀 기재가 없다.**

원 Wide&Deep 논문(Cheng et al., 2016)의 wide part는 **범주형 변수의 교차특성**이지만,
이 데이터셋에는 범주형 변수가 없다(I-14).

**가정**: wide = **46개 원 특성에 대한 단일 선형 층**(교차특성 없음).
`w&d.wide_features: all | none | [명시적 리스트]`, `w&d.wide_crosses: []`.

> 사용자 지시대로, **임의의 교차특성을 만들어 원 논문 방식이라고 부르지 않는다.**
> `wide_crosses`가 비어 있지 않으면 로그와 결과표에 `assumed_wide_crosses` 태그가 붙는다.

결합: `logits = wide_linear(x) + deep_mlp(x)` → softmax.
`w&d.combine: sum` (원 논문 방식). 대안 `concat_then_linear` 제공.

### E-5. 공통 학습 가정 `assumed_common_training`

- 손실: cross-entropy, 최적화: Adam (논문 보고).
- **early stopping은 train 내부 validation에서만.** outer test·평가 fold 사용 금지 —
  감사기가 `record_early_stopping()`으로 강제.
- 내부 validation은 **피험자 단위**로 분리한다(`internal_val.split_by: subject`,
  `fraction: 0.2`). 합성행은 내부 validation에 넣지 않는다.
- seed: 논문 미보고 → `seed: 42` 기본, `--seed`로 변경.

### E-6. class_weight `assumed_balanced`

논문은 class weight를 쓰지 않는다(VAE 증강이 그 역할). 비교 baseline으로만 사용하며
`sklearn.utils.class_weight.compute_class_weight("balanced")` 값을 쓴다.

---

## F. 평가 가정

### F-1. 피험자 단위 집계는 확률 **산술평균** `assumed_mean_pooling`

일별 3-class 확률을 피험자별로 산술평균한 뒤 argmax로 예측 클래스를 정한다.
대안(`aggregate.method: median | logit_mean | majority_vote`)을 config로 제공한다.
기록 수가 피험자마다 35~122로 다르지만 **가중치를 주지 않는다**
(피험자 1명 = 1표). 이것이 사용자 지시의 "피험자별 평균"에 해당한다.

### F-2. macro PR-AUC는 OvR average precision의 macro 평균 `assumed_macro_ap`

`sklearn.metrics.average_precision_score(..., average="macro")` (OvR).

### F-3. bootstrap `assumed_bootstrap_2000`

`n_boot: 2000`, **피험자 단위 재표집**(층화 없음), percentile 방식 95% CI.
Dem 피험자가 12명이므로 CI가 매우 넓을 것이 예상된다 — 이는 보고 대상이다.

### F-4. ROC-AUC가 정의되지 않는 fold 처리 `assumed_nan_propagate`

어떤 fold의 평가셋에 특정 클래스 피험자가 0명이면 해당 OvR AUC는 `NaN`으로 두고
macro 평균에서 제외하며, **제외 사실을 결과표에 표시**한다.
(단 실험 B·C의 split 검증이 이 상황을 사전에 차단한다.)

---

## G. 이 재현이 의도적으로 하지 **않는** 것

| 항목 | 이유 |
| --- | --- |
| 논문 수치를 코드에서 보정 | 지시 10. 원문 수치와 계산값을 병기만 한다 |
| 본문/그림 충돌 시 한쪽 선택 | 지시 8·18. 각각 config variant로 구현 |
| 미보고 설정을 "exact"로 명명 | 지시 17 |
| 합성행을 독립 피험자로 집계 | 지시 15·16 |
| 실험 B에서 하이퍼파라미터 재선택 | 지시(B는 논문 설정 고정) |
| Optuna 사용 | 지시 15절 |
| 본 세션에서 학습 실행 | 지시 4·15절 |
