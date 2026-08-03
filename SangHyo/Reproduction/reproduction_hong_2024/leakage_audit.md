# 누수 감사 명세

구현: `src/audit/leakage.py`. 계약 테스트: `tests/` 8개 파일 103개 테스트.

모든 검사는 **학습 전에** 실행되고, 위반 시 `LeakageError`를 던져 즉시 중단한다.
경고로 낮출 수 있는 항목은 명시적으로 표시된 것뿐이다.

---

## 1. 검사 목록

### 데이터셋 수준 (`audit_dataset`) — split 이전

| 검사 | 내용 |
| --- | --- |
| `one_row_per_subject_day` | (피험자, 날짜) 중복 없음 |
| `raw_row_id_unique` | 원시 행 id가 고유 |
| `every_subject_labelled` | 일별 표와 피험자 표의 피험자 집합 일치 |
| `label_constant_within_subject` | 한 사람의 라벨이 날짜에 따라 바뀌지 않음 |
| `no_forbidden_features` | 식별자·정답·MMSE가 특징에 없음 |
| `feature_count_matches_paper` | 32개 |
| `features_are_finite` | NaN/Inf 없음 (scaler를 조용히 오염시키는 경로 차단) |

### split 수준 (`audit_sequence_split`) — 매 fold, 매 길이

| # | 검사 | 스펙 §12 대응 |
| ---: | --- | --- |
| 1 | `no_subject_overlap` / `subject_overlap_is_declared` | train/test subject ID 교집합 |
| 2 | `no_raw_row_overlap` | train/test raw row ID 교집합 |
| 3 | `no_shared_subject_dates` | train/test raw date 교집합 |
| 4 | `{train,test}_sequences_are_calendar_consecutive` | 시퀀스가 경계·공백을 넘는지 |
| 5 | `no_identical_windows_across_split` | 동일/거의 동일한 시퀀스가 양쪽에 존재 |
| 6 | `scaler_fitted_on_train_only` | scaler fit에 test가 포함되었는지 |
| 7 | `undersampling_applied_to_train_only` | sampling이 validation/test에 적용되었는지 |
| 8 | `sequence_length_not_chosen_on_test` | test가 길이 선택에 쓰였는지 |
| 9 | `hyperparameters_not_chosen_on_test` | test가 하이퍼파라미터 선택에 쓰였는지 |
| 10 | `early_stopping_not_monitored_on_test` | test가 early stopping에 쓰였는지 |
| 11 | `no_forbidden_features` | MMSE 등이 특징에 포함되었는지 |
| 12 | `subject_id_not_a_feature` | 피험자 ID가 특징인지 |
| 13 | `validation_{disjoint,dates_disjoint}_from_test` | validation이 test와 겹치는지 |

### 중첩 CV 전용 (`audit_outer_test_isolation`)

| 검사 | 내용 |
| --- | --- |
| `outer_test_absent_from_inner_cv` | outer test 피험자가 inner fold 어디에도 없음 |
| `selection_scores_come_from_inner_cv` | 선택 점수의 출처가 inner CV |

### 시간분할 전용 (`audit_temporal_split`)

`temporal_split_spec.md` §7 참조.

---

## 2. 각 검사가 실제로 무엇을 보장하는가

정직하게 적어 둔다. 코드가 잡을 수 있는 것과 없는 것이 다르다.

### (피험자, 날짜) 쌍으로 검사하는 이유

날짜만 비교하면 안 된다. 서로 다른 두 사람이 2020-11-05를 공유하는 것은 정상이다.
누수는 **같은 사람의** 같은 날짜가 양쪽에 있을 때다. 모든 날짜 검사는
`(subject_id, date)` 쌍을 키로 쓴다.

### `no_identical_windows_across_split`

날짜 검사만으로는 잡히지 않는 경우가 있다. 특징값이 완전히 같은 두 윈도우가 서로
다른 (피험자, 날짜)에서 나올 수 있다. 이 검사는 윈도우의 float 바이트를 해시해서
비교하므로 날짜와 무관하게 잡는다. 표본이 20,000개를 넘으면 무작위 부분집합으로
검사한다.

### `scaler_fitted_on_train_only`

`SequenceScaler`는 fit할 때 대상 시퀀스 집합의 지문(sequence_id 정렬 후 SHA-256)을
저장한다. 감사는 이 지문이 훈련 집합의 지문과 같은지 비교한다.

**한계:** 손으로 만든 배열에 fit한 scaler는 잡지 못한다. 그래서 이 패키지에서
scaler를 fit하는 경로는 `SequenceScaler.fit` 하나뿐이고, 정식 경로는
`fit_transform_pair(train, *others)`다. 이 함수는 **첫 번째 인자에만 fit**하므로
호출부에서 순서를 바꾸지 않는 한 잘못될 수 없다.

### `sequence_length_not_chosen_on_test` 등 provenance 검사

이 세 검사(8, 9, 10)는 **출처 선언을 확인**한다. `config_fixed`, `paper_reported`,
`inner_cv` 중 하나여야 하고 그 외의 값이면 실패한다.

**한계:** 사람이 outer test 점수를 보고 config를 고쳐 다시 돌리는 것은 어떤 코드도
막을 수 없다. 그래서 이 항목이 문서에 적혀 있는 것이다. 실험 C의 config는
`sequence.length_selection: inner_cv`가 아니면 로드 자체를 거부한다.

---

## 3. dry-run 실측 결과 (2026-08-02)

`python run.py --config <각 config> --dry-run`

| 실험 | 길이 | 검사 수 | 실패 | 비고 |
| --- | ---: | ---: | ---: | --- |
| `paper_temporal_reconstruction` | 3 | 13 | 0 | |
| | 4 | 13 | 0 | |
| | 5 | 13 | 0 | |
| `strict_same_subject_temporal` | 3 / 4 / 5 | 13 | 0 | embargo 2/3/4일 |
| `fixed_subject_independent` | 3 / 4 / 5 | 13 | 0 | 5-fold 전부 양성 12~13명 |
| `nested_subject_independent` | 3 / 4 / 5 | 13 | 0 | inner isolation 통과 |
| `paper_literal_variant` | 3 | 13 | **2** | 의도된 진단 |
| | 4 | 13 | **2** | |
| | 5 | 13 | **2** | |

데이터셋 수준 검사 7개는 전부 통과.

## 4. `paper_literal_variant`의 실패는 결과다

이 arm은 실패해야 정상이다. 실패하는 두 검사는 `no_raw_row_overlap`과
`no_shared_subject_dates`이며, 그 크기가 곧 측정하려던 값이다.

| 길이 | 경계 교차 윈도우 | 양쪽에 존재하는 (피험자, 날짜) | test 날짜 중 비율 |
| ---: | ---: | ---: | ---: |
| 3일 | 345 (2.9%) | 283 | 32.6% |
| 4일 | 514 (4.4%) | 435 | 48.2% |
| 5일 | 678 (5.9%) | 526 | **62.3%** |

읽는 법: 경계를 넘는 윈도우는 5.9%뿐이지만, 그 윈도우 하나가 test 날짜 여러 개를
train 쪽으로 끌고 간다. 결과적으로 **5일에서는 test 날짜의 62.3%가 이미 train에서
관측된 날짜**가 된다. 윈도우 비율만 보고 "영향이 작다"고 결론 내면 안 된다.

이 arm의 결과 파일에는 `result_kind: "leakage_diagnostic"`과
`all_audits_passed: false`가 기록되고, 비교표의 표 3에서 "성능 주장 불가"로
표시된다.

## 5. 코드가 막지 못하는 것

정직하게 남긴다.

1. **실험 전체의 선택 편향.** 같은 174명으로 여러 arm을 돌리고 그중 좋은 것을
   보고하면, fold 안의 누수를 전부 막아도 편향은 남는다.
2. **사람이 test를 보고 config를 고치는 것.** provenance 검사는 선언을 확인할 뿐이다.
3. **AI-Hub 데이터 자체의 전처리.** 논문은 "already cleaned, with all missing data
   or outliers handled in-house"라고 적었다. 그 in-house 처리가 라벨을 참조했는지
   여부는 이 저장소에서 확인할 수 없다(`unresolved_questions.md` Q-02).
4. **동일 피험자의 시간적 자기상관.** embargo는 완화할 뿐 제거하지 못한다.

## 5-1. 실제 실행에서 드러난 보고 결함 (2026-08-02 수정 완료)

첫 nested 정식 실행(`20260802_150524_utc`)에서 **누수 검사는 30건 전부 통과**했지만,
누수가 아닌 **보고 단계의 결함 세 가지**가 드러났다. 전부 수정했고
`tests/test_reporting_integrity.py`가 회귀를 막는다.

### 결함 1 — 부분집합이 nested 열을 채웠다 (치명적)

nested 실험은 `{model}_Lnested`(정식) 외에 `{model}_L{길이}`(진단용 부분집합)도
만든다. 후자는 inner CV가 그 길이를 고른 fold만 모은 것이다.
`compare.py::_lookup`이 이를 구분하지 않아 표 1의 Nested 열에 들어갔다.

실제로 생성된 표에는 이렇게 찍혔다.

| 표시된 값 | 실제 정체 |
| --- | --- |
| Nested LSTM 3일 = 0.492 | 3 fold / 104명 |
| Nested LSTM 4일 = 0.579 | 1 fold / 35명 |
| **Nested LSTM 5일 = 0.782** | **1 fold / 35명** |
| (표에 없음) | **진짜 nested 추정치 0.533 / 5 fold / 174명** |

읽는 사람이 "nested에서 5일이 0.78"이라고 보고하게 되는 구조였고, 이는 스펙이
금지한 "3·4·5일 중 최고를 test에서 선택"과 결과적으로 같다. RF에서는 더 극단적으로
`random_forest_L3 = 0.806`(1 fold, 35명)이 나왔지만 정식값은 0.519다.

**수정:** `_lookup`이 `is_partial_subset` 블록을 건너뛴다. 정식 nested 값은 표 4로
분리했고, 부분집합은 fold 수·피험자 수와 경고를 붙여 표 5로 격리했다.
`LAUNCHER_STATUS.json`의 `headline_roc_auc`에서도 제외하고
`partial_subsets_not_headline`으로 옮겼다.

### 결함 2 — 퇴화 threshold가 기본으로 선택되었다

15개 fold 중 **6개**가 sensitivity 1.00 / specificity 0.00, 즉 모든 피험자를 양성으로
예측하는 operating point를 썼다.

원인: 모든 점수보다 낮은 threshold는 Youden 지수가 정확히 0이다. chance 수준
모델에서는 실제 후보들이 전부 0보다 낮으므로, 이 퇴화 해가 기본값으로 이긴다.

**수정:** 한 클래스로만 예측하는 후보를 건너뛰고, 남는 후보가 없으면 논문의 고정
0.5로 되돌리며 그 사실을 `threshold_report.fallback_to_fixed`에 기록한다.
ROC-AUC는 threshold와 무관하므로 영향이 없지만, 논문이 보고하는
sensitivity/specificity/F1은 이 수정 전까지 해석할 수 없는 값이었다.

### 결함 3 — threshold를 다른 단위에서 골랐다

threshold를 **시퀀스 단위** inner 점수에서 고른 뒤 **피험자 단위** 평균 확률에
적용했다. LSTM 시퀀스 점수는 45%가 0.001 미만, 25%가 0.999 초과로 심하게 포화되어
있어 극단값이 선택되는데, 피험자 평균은 중앙에 몰리므로 전원이 한 클래스로 간다.
결함 2의 직접적 원인이기도 하다.

**수정:** inner fold의 예측을 피험자 단위로 먼저 집계한 뒤 그 분포에서 threshold를
고른다(`selected_on: inner_cv_subject_level`).

### 부수 수정

pooled 블록은 fold별로 다른 threshold를 하나의 confusion matrix로 합칠 수 없으므로
고정 0.5를 쓰는데, 이 사실이 기록되지 않았다. 이제 `threshold_source`,
`per_fold_thresholds`, `n_degenerate_fold_operating_points`를 함께 남긴다.

## 6. 재현 명령

```bash
python run.py --config configs/paper_temporal_5day.yaml --dry-run
```

```bash
python run.py --config configs/paper_literal_variant.yaml --dry-run
```

```bash
python -m pytest tests/ -q
```
