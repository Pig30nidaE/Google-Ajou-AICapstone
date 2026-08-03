# leakage_audit.md — 데이터 누수 통제 설계와 자동검사

작성일: 2026-08-02

이 문서는 **누수를 어떻게 코드로 막는가**와 **실험 A에서 논문 절차의 누수를 어떻게 측정하는가**를
규정한다. 구현은 `src/audit/`, 검증은 `tests/`에 있다.

---

## 1. 두 가지 감사 모드

| 모드 | 사용처 | 동작 |
| --- | --- | --- |
| `enforce` | 실험 B·C | 불변식 위반 시 즉시 `LeakageError` |
| `observe` | **실험 A 전용** | 위반을 기록만 하고 진행 |

**왜 `observe`가 필요한가**: 논문 절차는 설계상 누수를 포함한다(행 단위 분할, split 이전 전처리).
이를 오류로 막으면 재현 자체가 불가능하다. 대신 어떤 불변식이 몇 건 위반되는지
**정량 측정해 보고**하는 것이 실험 A의 산출물이다 (`outputs/A_*/leakage_observation.json`).

`config.validate_config()`가 실험 B·C에서 `audit.mode: observe`를 **차단**하므로
모드를 실수로 바꿀 수 없다.

**예외**: feature 오염 검사(`check_features`)는 모드와 무관하게 **항상 강제**된다.
subject ID나 MMSE가 입력에 들어가는 것은 논문 재현과 무관한 순수 구현 오류이기 때문이다.

---

## 2. 감사기 동작 방식 — "신고 + 검증"

파이프라인의 각 단계가 감사기에 **자기 행위를 신고**하고, 감사기가 불변식을 검증한다.
검사를 사후에 돌리는 것이 아니라 **행위 시점에** 잡는다.

```python
auditor.register_split(fold_id, train_subjects=…, eval_subjects=…,
                       train_row_ids=…, eval_row_ids=…,
                       validation_subjects=…, validation_row_ids=…)
auditor.record_fit("scaler", fold_id, subjects=…, row_ids=…)       # 전처리기
auditor.record_vae_fit(fold_id, subjects=…, labels=…, row_ids=…, expected_label=2)
auditor.record_synthetic(fold_id, source_subjects=…, n_rows=…, target="train")
auditor.record_eval(fold_id, is_synthetic=…, subjects=…)
auditor.record_early_stopping(fold_id, subjects=…, row_ids=…)
auditor.record_selection("latent_dim", fold_id, subjects=…)
auditor.check_features(columns)
```

`register_split`보다 먼저 `record_fit`이 호출되면 `SPLIT_NOT_REGISTERED`가 발생한다 —
즉 **"split보다 전처리를 먼저 했다"가 구조적으로 탐지**된다.

---

## 3. 검사 항목 대조표

사용자 지시 11절의 요구 항목과 구현 위치.

| # | 요구 검사 | 검사 함수 / 코드 | unit test |
| --- | --- | --- | --- |
| 1 | train/test subject ID 교집합 | `check_subject_overlap` → `SUBJECT_OVERLAP` | `test_subject_overlap.py` |
| 2 | train/test 원시 row 교집합 | `check_row_overlap` → `ROW_OVERLAP` | `test_subject_overlap.py` |
| 3 | scaler fit subject 범위 | `check_fit_scope` → `FIT_SCOPE` | `test_preprocessing_scope.py` |
| 4 | outlier detector fit subject 범위 | `check_fit_scope` → `FIT_SCOPE` | `test_preprocessing_scope.py` |
| 5 | VAE fit subject 범위 | `check_vae_fit_scope` → `VAE_FIT_SCOPE` | `test_vae_training_scope.py` |
| 6 | synthetic 생성에 outer test 사용 | `check_synthetic_source_subjects` → `SYNTHETIC_SOURCE_SCOPE` | `test_vae_training_scope.py` |
| 7 | synthetic이 validation/test에 포함 | `check_no_synthetic_in_eval` → `SYNTHETIC_IN_EVAL` | `test_synthetic_test_exclusion.py` |
| 8 | preprocessing이 split 전에 수행 | `check_preprocessing_after_split` → `PREPROCESSING_BEFORE_SPLIT` | `test_preprocessing_scope.py` |
| 9 | MMSE가 입력변수에 포함 | `check_forbidden_features` → `FORBIDDEN_FEATURE` | `test_forbidden_features.py` |
| 10 | 진단 파생변수가 입력에 포함 | `check_forbidden_features` (`DIAG_*`, `Q\d+`) | `test_forbidden_features.py` |
| 11 | test 성능으로 이상치 임계값 선택 | `check_selection_scope` → `SELECTION_SCOPE` | `test_outer_test_isolation.py` |
| 12 | test 성능으로 latent dimension 선택 | `check_selection_scope` → `SELECTION_SCOPE` | `test_outer_test_isolation.py` |
| 13 | synthetic이 독립 subject로 집계 | `check_subject_aggregation_excludes_synthetic` → `SYNTHETIC_AS_SUBJECT` | `test_synthetic_test_exclusion.py` |
| 14 | outer test가 early stopping에 사용 | subject: `EARLY_STOPPING_SCOPE`; 원시행: `EARLY_STOPPING_ROW_SCOPE` | `test_outer_test_isolation.py` |
| 15 | 전처리 fit 원시행 범위 | `check_fit_row_scope` → `FIT_ROW_SCOPE` | `test_preprocessing_scope.py` |
| 16 | VAE fit 원시행 범위 | `check_fit_row_scope` → `VAE_FIT_ROW_SCOPE` | `test_vae_training_scope.py` |

추가로 구현한 검사:

| 검사 | 코드 |
| --- | --- |
| 합성행이 train 이외에 투입 | `SYNTHETIC_TARGET` |
| VAE가 대상 클래스 밖 라벨로 학습 | `VAE_FIT_LABEL` |
| fold 등록 없이 파이프라인 호출 | `SPLIT_NOT_REGISTERED` |
| 명시적 validation과 outer test 행 혼입 | `EARLY_STOPPING_ROW_SCOPE` |

---

## 4. 구조적 차단 — 검사 이전에 애초에 불가능하게

검사만으로는 부족하므로 자료구조 수준에서도 막는다.

| 위험 | 구조적 차단 |
| --- | --- |
| 식별자가 feature에 유입 | `LifelogData`가 `X`(feature)와 `subject`/`y`를 **분리 보유**. `__post_init__`이 `assert_no_forbidden_features` 호출 |
| 합성행이 피험자로 집계 | 합성행 `subject` = `"__SYNTHETIC__"` 센티널 고정. `subjects()`가 항상 제외 |
| 합성 row_id가 실제와 충돌 | 합성행 `row_id`는 **음수**로 채번 (`-1, -2, …`) |
| 합성행이 early stopping val에 유입 | `make_internal_validation`이 `is_synthetic` 행을 val 후보에서 제외 |
| 누수 config를 실수로 사용 | `validate_config`가 실험 B·C에서 `all_data`/`all_dem`/`split.unit: row`를 차단 |
| 실험 B에서 하이퍼파라미터 재선택 | `validate_config`가 `search.enabled: true`를 차단 |
| Dem 없는 fold 생성 | `_validate_fold`가 `SplitError` |

---

## 5. 실험별 통제 수준

### 실험 A (`observe`) — 논문 절차의 누수를 **측정**

| 항목 | 상태 | 관측 결과 (dry-run, A3 기준) |
| --- | --- | --- |
| split 단위 | 행 | train 174명 / test 167명, **중복 167명** |
| 이상치 처리기 fit | 전체 데이터 | test·valid 행이 임계값 결정에 참여 |
| scaler fit | 전체 데이터 | 동일 |
| VAE fit | train Dem만 | 다른 범위는 미구현 상태에서 fail-closed |
| 합성행 투입 | train만 | 평가에는 미투입 (논문과 동일) |

→ `outputs/A_*/leakage_observation.json`에 위반 코드별 건수와 관측치가 저장된다.

### 실험 B (`enforce`) — 순서 강제

```
1. 피험자 1행 테이블 기준 층화 분리  make_group_folds(method=subject_stratified)
2. 이상치 처리기 fit ← train만       FoldPreprocessor.fit()
3. imputer fit       ← train만       FoldPreprocessor.fit()
4. scaler fit        ← train 실제행만 FoldPreprocessor.fit_scaler()
5. VAE fit           ← train Dem만    augment_train_fold()
6. 합성행은 train에만                 append_synthetic()
7. 분류기 학습                        fit_classifier()
8. 평가에는 학습된 변환만 적용         FoldPreprocessor.transform()
9. 평가에 합성행 절대 미추가           record_eval() 검증
```

교정 설계의 기대 구성(seed 42, 3-fold; **아직 재실행하지 않음**):

| fold | train Dem 피험자 | eval Dem 피험자 | train Dem 기록 | VAE fit 기록 |
| --- | ---: | ---: | --- | --- |
| fold_r0_f0 | 8 | 4 | 새 dry-run에서 확인 | train의 실제 Dem 행과 동일해야 함 |
| fold_r0_f1 | 8 | 4 | 새 dry-run에서 확인 | train의 실제 Dem 행과 동일해야 함 |
| fold_r0_f2 | 8 | 4 | 새 dry-run에서 확인 | train의 실제 Dem 행과 동일해야 함 |

모든 fold에서 CN·MCI·Dem 피험자가 train·eval 양쪽에 존재한다.

> ⚠️ **평가 fold에서 행을 삭제하지 않는다.** B는 A와 같은 Isolation Forest를 train에
> 적합하지만, 독립 평가 표본을 사후 선택하지 않도록 eval의 `keep_mask`는 적용하지 않는다.
> percentile 후보도 평가는 `apply_outlier_eval()`로 행 수를 유지한다.

### 실험 C (`enforce`) — 선택을 inner에 가둠

outer test에 절대 사용 금지: 이상치 임계값 선택 / scaler fit / VAE fit /
synthetic ratio 선택 / 모델 선택 / early stopping / threshold 선택.

inner CV는 **별도의 `LeakageAuditor` 인스턴스**를 쓰므로 inner fold의 fit이
outer 감사기 기준으로 검증되지 않는 혼선이 없다. outer 감사기에는
`record_selection("pipeline", outer_fold, subjects=outer_train.subject)`만 신고되며,
outer eval 피험자가 섞이면 즉시 `SELECTION_SCOPE`가 발생한다.

교정 설정상 예상 규모: outer 3-fold × 24 후보 × inner 3-fold + outer 3회 =
**총 219회 모델 적합**. 수정 후 dry-run은 이번 감사에서 실행하지 않았다.

---

## 6. 실행

```bash
python -m pytest tests/ -q
```

현재 상태: 2026-08-03 감사 후 회귀 테스트를 추가했지만, 사용자 지시에 따라 수정 뒤
**테스트를 실행하지 않았다**. 아래 명령은 후속 재실행 시 사용할 검증 명령이다.

```bash
python run.py --config configs/leakage_controlled_non_nested.yaml --dry-run
```

학습 없이 fold 구성·fit 범위·예상 합성행 수·누수 검사를 확인한다.

---

## 7. 알려진 한계

1. **감사기는 신고된 것만 검증한다.** 새 파이프라인 단계를 추가하면서 `record_*` 호출을
   빠뜨리면 그 단계는 감사되지 않는다. 새 단계 추가 시 대응 unit test를 함께 추가해야 한다.
2. **`observe` 모드의 관측치는 위반 목록이지 완전한 누수 정량화가 아니다.**
   예컨대 "test 행이 scaler 평균에 얼마나 기여했는가"는 측정하지 않는다.
3. **실험 A의 이상치 처리는 감사기 fold 등록 이전에 일어난다**(논문 순서 재현).
   fold 등록 뒤 역사적 fit 이벤트로 신고해 `PREPROCESSING_BEFORE_SPLIT`과 원시행 범위
   위반을 기록하고, `observations`에도 `outlier_fit_on_all_data`를 남긴다.
4. **피험자 라벨의 정확성은 검사 범위 밖이다.** 라벨 파일 3종과 MMSE의 `DIAG_NM`이
   174명 전원에서 일치함은 확인했으나, 그 라벨 자체의 타당성은 다루지 않는다.
