# leakage_audit.md — 누수 통제 설계와 자동 검사

작성일: 2026-08-02

이 문서는 세 실험이 데이터 누수를 어떻게 막는지, 그리고 그것을 코드가 어떻게
**강제**하는지 기술한다. 검사는 경고가 아니라 **예외**이며, 학습이 시작되기 전에 실행된다.

---

## 0. 이 저장소에서 누수가 왜 중요한가

`SangHyo/AGENTS.md` §3-4에 이미 정량화된 결과가 있다.

| 입력 | 하루 무작위 K-fold(누수) | 정직한 day GroupKFold | 정직한 subject OOF |
| --- | ---: | ---: | ---: |
| 웨어러블-only | **0.9526** | 0.4811 | 0.5214 |
| 웨어러블+MMSE | **0.99998** | 0.6888 | 0.6924 |

같은 특징공학에서 **분할만 바꿔** ROC-AUC가 0.95 → 0.52로 떨어진다.
따라서 이 데이터에서 높은 성능 주장은 **먼저 subject overlap부터 감사**해야 한다.

### 0-1. 다만 원 논문의 문제는 행 단위 누수가 아니다

`reproduction_spec.md` §2에서 복원했듯, 임형준(2025)의 평가집단은 **33명 피험자**이고
분할은 AI-Hub 공식 Training/Validation 경계와 일치한다. 즉 **피험자 독립적**이다.

원 논문의 실제 약점은 다음 네 가지이며, 이 문서의 검사들은 그 각각에 대응한다.

| 원 논문의 약점 | 본 재현의 대응 |
| --- | --- |
| 고정 단일 분할, 양성 7명 | 실험 B·C의 반복 StratifiedGroupKFold |
| 다섯 모델을 test에서 비교한 뒤 최고 채택 | 실험 C의 nested model selection |
| 하이퍼파라미터 탐색 범위 미보고 | 실험 C의 inner CV, outer test 격리 검사 |
| 정규화 적합 범위 미보고 | `scaler_scope` 두 변형 + 전처리 범위 검사 |

---

## 1. 설계 원칙

### 1-1. 분할이 먼저, 표현이 나중

```
❌ 잘못된 순서                    ✅ 본 재현의 순서
   시퀀스 생성                        피험자 분할
   → 분할                             → 각 split 안에서 시퀀스 생성
                                      → train 부분에서만 전처리 fit
```

`src/engine.py`의 `materialise_pair()`가 이 순서를 강제한다. train 표현을 먼저 만들고,
거기서 결정된 shape(특히 `sequence_length`)을 test 표현에 **전달**한다.

> **이 순서가 잡아낸 실제 문제**: `sequence_length: max`를 양쪽에서 독립적으로 계산하면
> train T=120, test T=**122**가 나왔다. 관측일수가 가장 많은 피험자가 test에 있었기
> 때문이다. 이는 텐서 shape 불일치일 뿐 아니라 **평가집단이 전처리 파라미터를 결정**하는
> 누수다. 지금은 train에서 결정한 120을 test에 적용한다.

### 1-2. 전처리는 fold 지역적이며 스스로를 감사한다

`FoldPreprocessor`는 fit할 때 본 피험자 ID를 `fitted_subjects`에 기록한다.
감사기는 이 집합이 (a) 비어 있지 않고, (b) test 피험자와 교집합이 없고,
(c) 현재 fold의 train 부분집합인지 확인한다.

패딩 처리도 여기에 포함된다.

- **fit**: 패딩 시점을 통계에서 제외한다(`mask` 인자). 시퀀스의 60% 이상이 패딩일 수
  있으므로, 포함하면 모든 평균이 0 쪽으로 끌려간다. 논문의 서술 순서
  ("정규화 후 3차원 텐서")와도 이쪽이 일치한다.
- **transform 후**: 패딩 시점을 다시 0으로 되돌린다. 표준화하면 패딩 0이
  `-mean/scale`이 되는데, Conv1d는 이를 실제 관측처럼 합성곱한다. LSTM 계열은
  `pack_padded_sequence`로 어차피 무시하지만 1D-CNN은 그렇지 않다.

### 1-2-1. 패딩 방향 버그 (2026-08-02 수정)

첫 실험 A 실행에서 **딥러닝 3종이 전부 무너진 원인**이 여기 있었다. 기록을 남긴다.

빌더는 `padding: "pre"`가 기본이라 실제 관측을 시퀀스 **뒤쪽**에 놓는다
(`X[i, T-valid:, :] = values`). 그런데 하위 소비자 4곳이 전부 마스크를
`arange(T) < lengths`, 즉 **앞쪽**이 유효라고 계산했다.

| 위치 | 증상 |
| --- | --- |
| `FoldPreprocessor.fit` | 스케일러 통계를 패딩 구간에서 적합 |
| `zero_padding()` | 실제 관측을 0으로 지움 |
| `RecurrentClassifier.forward` | `pack_padded_sequence`가 앞쪽 패딩만 읽음 |
| `Conv1dClassifier.forward` | 풀링 마스크가 패딩 구간을 평균 |

실제 데이터 피해 규모 (Training 141명, T=120):

- 실제 관측 9,705 timestep 중 **6,239개(64.3%) 소실**
- **141명 중 44명은 자기 데이터를 100% 잃음**
- 피험자별 보존율 중앙값 **18.2%**

살아남은 정보의 상당 부분이 **관측일수 자체**였고, 관측일수 단독 AUC는 0.4615다.
실행 결과의 LSTM 0.429 / Bi-LSTM 0.407 / 1D-CNN 0.451이 정확히 그 노이즈 수준에
몰려 있던 이유다.

**수정**: 마스크의 단일 소스를 `Representation.valid_mask()`로 통일하고,
모델에는 `Representation.left_aligned()`가 좌측 정렬한 텐서를 넘긴다. 그 결과
`padding: pre`와 `post`가 **동일한 결과**를 낸다 — 마스킹이 제대로 되면 패딩
방향은 결과에 영향을 주지 않아야 하고, 이제 그 성질이 테스트로 고정돼 있다
(`tests/test_sequence_padding.py`, 13개).

**교훈**: `lengths`만으로 마스크를 재유도하지 말 것. 길이는 유효 구간이 앞에
있다고 암묵적으로 가정한다. 마스크는 boolean 배열로 명시적으로 넘긴다.

### 1-3. 피험자 ID는 group이지 feature가 아니다

`subject_id`, `EMAIL`, `SAMPLE_EMAIL`은 분할 group으로만 쓰이고 특징 행렬에
들어가면 예외가 발생한다. 결과 파일에는 원본 이메일 대신 SHA-256 앞 12자만 저장한다
(`SangHyo/AGENTS.md` §6).

### 1-4. 인지검사는 주 분석에서 fail-closed

`main_lifelog_only`에서는 `TOTAL`, `Q01`–`Q19` 31개 문항, `DIAG_*`가 모두 금지된다.
정확한 allowlist 외에 `mmse`, `snsb`, `diag`를 포함하는 이름도 휴리스틱으로 차단한다.
`secondary_lifelog_plus_cognitive`에서만 인지검사가 허용되며, 그 경우에도 식별자·진단
컬럼은 여전히 금지된다.

---

## 2. 구현된 자동 검사

과제 명세 §9의 항목과 구현 위치를 대응시킨다. 전부 `src/audit/leakage.py`에 있다.

| # | 요구된 검사 | 구현 함수 | 위반 시 |
| --- | --- | --- | --- |
| 1 | train/test subject ID 교집합 | `check_subject_overlap` | `LeakageError` |
| 2 | train/test 원시 row ID 교집합 | `check_row_overlap` | `LeakageError` |
| 3 | 동일 피험자·동일 날짜 중복 | `check_subject_date_duplicates` | 기록(허용) |
| 4 | sequence 구성 원시 날짜의 train/test 중복 | `check_sequence_source_overlap` | `LeakageError` |
| 5 | scaler fit에 포함된 subject 목록 | `check_preprocessing_scope` | `LeakageError` |
| 6 | imputer fit에 포함된 subject 목록 | `check_preprocessing_scope` (동일 객체) | `LeakageError` |
| 7 | feature selector fit 범위 | `check_preprocessing_scope` | `LeakageError` |
| 8 | outer test가 inner tuning에 사용됐는지 | `check_outer_test_isolation` | `LeakageError` |
| 9 | 피험자 ID가 feature에 들어갔는지 | `check_forbidden_features` | `LeakageError` |
| 10 | MMSE·진단 파생변수가 feature에 있는지 | `check_forbidden_features` | `LeakageError` |
| 11 | test label이 threshold 결정에 쓰였는지 | `check_threshold_source` | `LeakageError` |
| 12 | 한 피험자 다중 라벨 | `check_label_consistency` | `LeakageError` |
| 13 | 라벨 없는 일별 행 | `check_label_consistency` | `LeakageError` |

`audit_split()`이 이들을 묶어 **매 fold, 매 모델, 모델 fit 직전**에 실행한다.
`audit_dataset()`은 분할과 무관한 검사를 로드 직후에 실행한다.

### 2-1. config 단계에서 먼저 걸러지는 것

`src/utils/config.py::validate_config()`는 데이터를 만지기 전에 실패한다.

- 실험 B·C에서 행 단위 분할 → 거부
- 실험 B·C에서 `scaler_scope: all_data` → 거부
- 실험 B에서 `tuning.enabled: true` → 거부 (CV 점수로 재선택 금지)
- 실험 C에서 `threshold.policy != inner_cv` → 거부
- 주 분석에서 `include_cognitive_tests: true` → 거부

---

## 3. 의도적 예외 하나: 행 단위 분할 진단

`assumption_variant_random_row_holdout`은 **일부러 누수시킨다.** 원 논문이 분할단위를
명시하지 않았으므로, 만약 행 단위였다면 성능이 얼마나 부풀려지는지 측정하기 위해서다.

이 변형은 다음 세 겹의 표시를 통과해야만 실행된다.

1. config에 `split.leakage_diagnostic_only: true`가 없으면 **config 검증 실패**
2. 실험 A에서만 허용 (B·C는 mode 자체를 거부)
3. 결과 meta에 `leakage_expected: true`,
   `interpret_as: "leakage_diagnostic_not_performance"`가 강제로 기록됨

**이 변형의 숫자를 성능표에 넣지 않는다.** 누수 크기 진단값이다.

같은 이유로 `scaler_scope: all_data`도 실험 A 전용이며,
`preprocessor_scope_all_data` 항목이 감사 로그에 실패로 기록된다.

---

## 4. 실험별 누수 통제 요약

### 실험 A — paper_reported_reconstruction

| 항목 | 설정 |
| --- | --- |
| 분할 | `official_partition` (141 / 33), 피험자 독립 |
| 전처리 | `scaler_scope: train_only` 기본 (`all_data` 변형 제공) |
| 임계값 | 0.5 고정 |
| 모델 선택 | **없음** — 다섯 모델을 모두 보고한다 |
| 남는 한계 | 평가 33명·양성 7명. 단일 분할이라 CI가 매우 넓다 |

원 논문이 다섯 모델 중 최고를 사후 채택한 것과 달리, 본 재현은 **채택하지 않고 전부
보고**한다. 그래야 논문 열과 공정하게 비교된다.

### 실험 B — leakage_controlled_non_nested

| 항목 | 설정 |
| --- | --- |
| 분할 | `StratifiedGroupKFold(5)` × repeats 5 = outer 25개, group = 피험자 |
| 전처리 | fold의 train 부분에서만 fit (강제) |
| 하이퍼파라미터 | 논문 값 고정. **CV 점수로 재선택 금지** |
| 임계값 | 0.5 사전 고정 |
| 평가 | 피험자 단위, 반복별 OOF 통합 |

### 실험 C — nested_subject_independent

| 항목 | 설정 |
| --- | --- |
| Outer | `StratifiedGroupKFold(5)` × repeats 3 = 15개 |
| Inner | `StratifiedGroupKFold(3)`, outer-train 피험자에서만 |
| 선택 대상 | 하이퍼파라미터(모델당 2–3개 후보), 임계값 |
| Outer test 사용 | **금지** — `check_outer_test_isolation`이 매 fold 검증 |
| 저장 항목 | 학습/평가 피험자 ID, inner 최적 설정, 선택 임계값, 피험자별 예측확률·실제 라벨, ROC-AUC / PR-AUC / balanced accuracy / sensitivity / specificity / F1 / Brier |

inner CV는 outer training 피험자만 받는다. `inner_stratified_group_kfold()`에 다른
집합을 넘기면 `check_outer_test_isolation`이 즉시 예외를 던진다.

---

## 5. 정적 테스트

`tests/`의 110개 테스트가 위 계약을 고정한다. 실제 데이터 없이도 동작한다
(합성 코호트 사용). 실행:

```bash
python -m pytest tests/ -q
```

| 파일 | 고정하는 계약 |
| --- | --- |
| `test_subject_overlap.py` | 피험자가 분할 양쪽에 나타나지 않음, config 거부 규칙 |
| `test_row_overlap.py` | 원시 행 중복 없음, 행 단위 변형이 실제로 누수함을 확인 |
| `test_sequence_overlap.py` | 시퀀스가 분할 이후 생성됨, 패딩·절단 규칙 |
| `test_preprocessing_scope.py` | fold-local fit, test 이상치가 스케일러에 영향 없음 |
| `test_forbidden_features.py` | 타깃·식별자·인지검사 컬럼 차단 |
| `test_outer_test_isolation.py` | inner CV가 outer test를 보지 않음, 임계값 출처 |
| `test_paper_arithmetic.py` | 33명·양성 7명 복원이 실제 데이터와 일치 |
| `test_sequence_padding.py` | 패딩이 실제 관측을 덮지 않음, pre/post 결과 동일 |
| `test_sequence_training.py` | early stopping이 학습 안 된 모델을 복원하지 않음 |

주목할 만한 테스트 두 개:

- `test_statistics_come_only_from_training_rows`: test 피험자에게 `1e6` 이상치를
  심어 놓고, 학습 스케일러의 평균이 움직이지 않는지 확인한다.
- `test_row_level_holdout_actually_leaks_subjects`: 진단용 변형이 **실제로**
  피험자를 공유하는지 확인한다. 누수 진단 도구가 누수하지 않으면 무용지물이다.

---

## 6. 남아 있는, 코드로 막을 수 없는 편향

정직하게 기록한다.

1. **실험 전체 선택 편향**: 코드 안의 누수를 막아도, 같은 174명을 반복해서 보며
   설계를 조정한 편향은 남는다.
2. **33명 Validation의 이력**: 실험 A의 평가집단은 이 저장소가 이미 수십 번 관찰한
   historical benchmark다(`SangHyo/AGENTS.md` §2-5). 실험 A는 **논문 재현 대상**일 뿐
   새로운 독립 검증이 아니다.
3. **논문 미보고로 인한 자유도**: 딥러닝 구조와 RF 하이퍼파라미터가 미보고이므로,
   본 재현이 고른 값이 논문보다 좋거나 나쁠 수 있다. 이 차이를 "재현 실패"로 읽으면 안 된다.
4. **새 코호트 부재**: 진짜 일반화 검증은 174명 안의 어떤 재분할도 아니라
   **새로운 피험자**다.
