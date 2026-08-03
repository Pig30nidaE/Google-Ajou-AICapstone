# reproduction_hong_2024

Hong et al. (2024), *Prediction of Cognitive Impairment Using Sleep Lifelog Data
and LSTM Model* (Mathematics 12, 3208) 재현 + 누수 점검 + 신규 피험자 일반화 평가.

**현재 상태:** 2026-08-03의 5일 결과를 감사한 뒤 방법·보고·비식별화 결함을
수정했다. 수정 코드의 학습 재실행은 아직 하지 않았다. 정식 학습은 Colab Pro+에서
수행한다.

---

## 한 문단 요약

논문은 각 피험자의 마지막 1주일을 test로 두고 LSTM 5일 모델에서 AUC 0.92를
보고했다. 이 값은 **이미 학습된 피험자의 미래 기간** 성능(Estimand A)이며, 신규
피험자 스크리닝 성능(Estimand B)이 아니다. 이 패키지는 (1) 논문 방식을 누수 없이
재구현하고, (2) 논문 문언 그대로의 순서가 만드는 누수를 정량화하고, (3) 같은 모델을
피험자 독립 분할과 Nested Group CV로 다시 평가한다. 세 결과를 같은 축에 섞지
않는 것이 이 패키지의 핵심 설계다.

## 문서

| 파일 | 내용 |
| --- | --- |
| [estimand_definition.md](estimand_definition.md) | **먼저 읽을 것.** 두 연구질문의 구분 |
| [reproduction_spec.md](reproduction_spec.md) | 확정된 방법 / 미확인 설정 / 실험 구성 |
| [paper_data_mapping.md](paper_data_mapping.md) | 논문 변수 ↔ 실제 컬럼, 수치 검증 |
| [sequence_generation_spec.md](sequence_generation_spec.md) | 시퀀스 생성 규칙과 실측 개수 |
| [temporal_split_spec.md](temporal_split_spec.md) | 마지막 1주일 분할과 embargo |
| [leakage_audit.md](leakage_audit.md) | 13종 검사와 dry-run 실측 결과 |
| [assumptions.md](assumptions.md) | 미보고 설정 22개와 config 필드 |
| [unresolved_questions.md](unresolved_questions.md) | 확정 불가 항목 13개 |

## 실행

로컬에서는 점검만 한다.

```bash
python run.py --inspect-data
```

```bash
python run.py --config configs/paper_temporal_5day.yaml --dry-run
```

실험 A의 LR/RF/XGBoost는 논문에 맞춰 H2O 3.46.0.1을 사용한다. H2O AutoML에는
SVM family가 없어 SVM은 sklearn으로 명시되며, 결과의 `method_fidelity`에서 이
차이를 확인한다. backend는 자동으로 조용히 대체되지 않는다.

```bash
python -m pytest tests/ -q
```

Colab Pro+에서의 정식 실행은 §"Colab 실행"을 참조.

### 옵션

| 옵션 | 용도 |
| --- | --- |
| `--inspect-data` | 논문 대비 데이터 점검만 |
| `--audit-only` | 데이터셋 수준 누수 감사만 |
| `--dry-run` | 분할·시퀀스·shape·누수검사만, 학습 없음 |
| `--sequence-length {3,4,5}` | 한 길이만 |
| `--fold N` | 한 outer fold만 |
| `--seed N` | seed 덮어쓰기 |
| `--model NAME` | 한 모델만 (반복 가능) |
| `--estimand {A,B}` | config가 다른 estimand면 중단 |
| `--resume` | 완료된 체크포인트 재사용 |
| `--compare` | 비교표 재생성 |

## 실험

| config | Estimand | 답하는 질문 |
| --- | :---: | --- |
| `paper_temporal_{3,4,5}day.yaml` | A | 논문 방식을 누수 없이 구현하면 Table 5가 재현되는가 |
| `paper_literal_variant.yaml` | A + 누수 | 논문 문언 그대로면 누수가 얼마나 생기는가 (**진단 전용**) |
| `strict_same_subject_temporal.yaml` | A | embargo까지 넣으면 얼마나 달라지는가 |
| `fixed_subject_independent.yaml` | B | 같은 모델을 신규 피험자에 쓰면 어떤가 |
| `nested_subject_independent.yaml` | B | 모델 선택 비용까지 포함하면 얼마나 남는가 |

## dry-run에서 이미 확인된 사실

학습 없이 확인된 것들이다. 전부 `--dry-run`으로 재현할 수 있다.

**데이터는 논문과 완전히 일치한다.** 174명 / CN 111·MCI 51·Dem 12 / 12,183건 /
피험자별 35~122건 / 변수 32개 — Table 3의 7개 수치 전부 일치.

**논문이 보고하지 않은 구조적 사실 3가지:**

1. 174명 중 **162명**의 기록에 달력 공백이 있다(총 1,089건). 행 순서만 보고
   이어붙이면 5일 윈도우의 **28.6%**가 실제로는 불연속이다.
2. 같은 날짜에 수면 레코드가 2개인 경우가 24행(11명). 날짜 키를 `bedtime_start`로
   잡으면 충돌이 1,044행으로 늘어난다.
3. `sleep_temperature_delta`와 `sleep_temperature_deviation`이 **완전히 동일한
   열**이다. 논문의 32개 중 실질적으로 다른 정보는 31개다.

**논문 문언 그대로의 순서가 만드는 누수:**

| 길이 | 경계 교차 윈도우 | train과 공유되는 test 날짜 |
| ---: | ---: | --- |
| 3일 | 345 (2.9%) | 338 / 1,036 (**32.6%**) |
| 4일 | 514 (4.4%) | 492 / 1,021 (**48.2%**) |
| 5일 | 678 (5.9%) | 604 / 969 (**62.3%**) |

경계를 넘는 윈도우는 5.9%뿐이지만, 그 윈도우가 test 날짜를 train으로 끌고 가기
때문에 결과적으로 test 날짜의 62.3%가 이미 학습에 노출된다.

**평가군이 길이에 따라 달라진다:**

마지막 1주일 안에 연속 L일을 만들 수 있는 피험자는 3일 156명, 4일 132명,
**5일 111명**이다. 논문의 "5일이 최고"라는 결과에는 모델 차이뿐 아니라 평가군
차이가 섞여 있을 수 있다(`unresolved_questions.md` Q-05).

## 재현 판정 (2026-08-03, 실험 A 3·4·5일 완료)

**판정: 부분 재현.** 판별력(AUC)은 근접했으나 논문의 두 결론은 재현되지 않았다.

| 구분 | 결과 |
| --- | --- |
| AUC 수준 | 근접. 3일 −0.021 / 4일 −0.048 / 5일 −0.071 |
| **중심 주장(LSTM > 비교모델)** | **재현 실패.** 논문 격차 +0.07~+0.29 → 재현 +0.006~+0.19. XGBoost와는 5일에서 +0.016뿐 |
| **길이 추세(3<4<5)** | **재현 실패.** 재현은 0.859 → 0.862 → 0.849로 평탄·하락 |
| Precision 격차 | 평가군 prevalence 차이로 **설명됨** (아래) |
| Sensitivity 격차 | 설명되지 않음. −0.15~−0.25 |
| P@100 | 논문 0.96, 재현 0.88/0.78/0.67. 5일은 이론상 최대가 0.94라 달성 불가 |

### 왜 갈렸는가 — 세 갈래로 분해했다

1. **평가군 구성이 다르다.** 논문 Table 5의 7개 행이 **전부** 양성률 ≈0.50을 과결정한다
   (sens·spec·precision 세 값이 prevalence를 결정한다). 실제 "마지막 1주일" 평가군은
   0.32다. prevalence를 0.50으로 맞춰 같은 예측을 재채점하면 **precision 격차가
   닫힌다**(5일 −0.088 → +0.037). AUC는 prevalence에 불변이라 그대로다.
2. **논문의 작동점은 우리 ROC 곡선 위쪽에 있다.** 논문 sensitivity를 우리 곡선에서
   맞추면 specificity가 5일 기준 0.590으로, 논문의 0.800보다 0.21 낮다. 즉 threshold를
   바꿔도 재현되지 않는 **판별력 차이**다.
3. **비교모델이 LSTM과 같은 정보를 본다.** 비교모델 입력 변환은 논문 미보고(Q-10)이고
   현재는 `flatten`(L×32)이라 LSTM과 동일한 정보량이다. 논문이 "do not reflect
   time-series characteristics"라고 쓴 만큼 저자들은 더 적은 정보를 줬을 가능성이 높다.
   이것이 격차 축소의 유력한 원인이며, `models.representation`을 바꿔 검증할 수 있다.

이 세 진단은 이제 코드가 자동으로 계산한다(`--compare`의 표 6·7·8).

## 실행된 결과 (2026-08-02)

`nested_subject_independent` 1회 완료 — `20260802_150524_utc`, 61분, 누수 검사 30건
전부 통과, 174명 전원 out-of-fold 예측 확보.

| 모델 | 피험자 AUC | 95% CI | Balanced Acc. | inner CV가 고른 길이 |
| --- | ---: | --- | ---: | --- |
| LSTM | 0.533 | [0.441, 0.628] | 0.510 | 3일×3, 4일×1, 5일×1 |
| Random Forest | 0.519 | [0.426, 0.615] | 0.518 | 3일×1, 4일×2, 5일×2 |
| XGBoost | 0.517 | [0.427, 0.611] | 0.536 | 3일×2, 4일×3 |

**세 모델 모두 신뢰구간이 0.5를 포함한다.** 신규 피험자에 대해 수면 라이프로그만으로
인지장애를 구분한다는 근거가 이 실행에는 없다. 이 저장소의 다른 웨어러블-only
실험(사람 단위 OOF 0.45~0.57, `SangHyo/AGENTS.md` §3-1)과 일치하는 결과다.

논문의 0.92(Estimand A, 시퀀스 단위)와 위 0.533(Estimand B, 피험자 단위)을 **같은
축에서 비교하면 안 된다.** 그 차이에는 누수 제거 효과와 질문이 바뀐 효과가 함께
들어 있고, 이를 분리하려면 실험 A와 B1을 마저 실행해야 한다.

> 첫 실행에서 발견된 보고 결함 세 가지(부분집합이 nested 열을 채움, 퇴화 threshold,
> threshold 단위 불일치)는 모두 수정했다. 자세한 내용은
> [leakage_audit.md §5-1](leakage_audit.md) 참조. **수정 이전에 생성된
> `comparison_partial.md`는 사용하지 말고 `--compare`로 다시 만든다.**

## 학습 전 확인할 것

1. `python run.py --inspect-data` — Table 3 대조에서 불일치 0건인가.
2. `python run.py --config <해당 config> --dry-run` — 누수 검사 실패 0건인가.
   (`paper_literal_variant`만 예외. 2건 실패가 정상이며 그것이 측정값이다.)
3. `python -m pytest tests/ -q` — 103개 전부 통과하는가.
4. dry-run의 `예상 학습 횟수`를 보고 시간 예산이 맞는가.
5. GPU 런타임인가(LSTM 포함 시). 표 모델만이면 CPU/High-RAM이 낫다.
6. Drive가 마운트되었는가. 결과는
   `/content/drive/MyDrive/reproduction_hong_2024_result/<UTC_RUN_ID>/`에 쌓인다.

## Colab 실행

저장소 루트 `base.ipynb`의 셀 2를 이렇게 바꾼다.

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Reproduction/reproduction_hong_2024/run.py"
```

`base.ipynb`는 Colab에서 `origin/main`을 새로 clone하므로 **커밋·푸시하지 않은
로컬 수정은 반영되지 않는다.**

권장 실행 순서와 예상 fit 횟수:

```bash
python run.py --config configs/paper_temporal_3day.yaml
```

```bash
python run.py --config configs/paper_temporal_4day.yaml
```

```bash
python run.py --config configs/paper_temporal_5day.yaml
```

```bash
python run.py --config configs/paper_literal_variant.yaml
```

```bash
python run.py --config configs/strict_same_subject_temporal.yaml
```

```bash
python run.py --config configs/fixed_subject_independent.yaml
```

```bash
python run.py --config configs/nested_subject_independent.yaml
```

| 실험 | fit 횟수 | 런타임 |
| --- | ---: | --- |
| `paper_temporal_{3,4,5}day` 각각 | 5 | GPU |
| `paper_literal_variant` | 3 | GPU |
| `strict_same_subject_temporal` | 15 | GPU |
| `fixed_subject_independent` | 75 | GPU |
| `nested_subject_independent` | **555** (inner 540) | GPU, 가장 오래 걸림 |

전부 끝난 뒤 비교표를 만든다.

```bash
python run.py --compare
```

## 결과 읽는 법

1. `LAUNCHER_STATUS.json`의 `status == "complete"`와 `TRAINING_COMPLETE.json`을
   모두 확인하고 두 파일의 `attempt_id`가 같은지 확인한다. 새 시도는 기존 완료
   마커를 `stale_completion_markers/<attempt_id>/`로 이동하므로, `starting` 또는
   `failed`에서 끝난 run은 과거 마커와 관계없이 성능표에서 제외한다.
2. `estimand` 필드를 먼저 본다. A와 B를 같은 표에서 비교하지 않는다.
3. `headline_unit`을 본다. A는 시퀀스 단위, B는 피험자 단위다.
4. `result_kind == "leakage_diagnostic"`인 결과는 성능이 아니다.
5. `thin_subjects`의 `n_subjects_with_evaluable_test_sequence`를 함께 인용한다.
6. `subject_bootstrap_ci`가 0.5를 포함하면 개선으로 선언하지 않는다.

## 구조

```
reproduction_hong_2024/
├── run.py                     # 단일 진입점
├── requirements_colab.txt
├── configs/                   # 7개
└── src/
    ├── engine.py              # 실험 A / A' / B1 / B2 / C 오케스트레이션
    ├── data/                  # schema(32변수), loader, inspect
    ├── sequences/builder.py   # 시퀀스 생성 + provenance
    ├── splits/                # temporal(마지막 1주일), group(StratifiedGroupKFold)
    ├── preprocessing/         # train-only scaler, 비교모델 입력변환
    ├── sampling/              # undersampling + 진단
    ├── models/                # lstm, baselines, h2o_backend, registry
    ├── evaluation/            # metrics(2단위), compare(표 1·2·3)
    ├── explainability/        # Deep SHAP (이번 범위에서는 미실행)
    ├── audit/leakage.py       # 13종 fail-closed 검사
    └── utils/                 # config, io, seeding
└── tests/                     # 8개 파일 103개 테스트
```

## 관련 실험

- `SangHyo/Reproduction/reproduction_lim_2025` — 같은 데이터의 다른 선행연구 재현
- `SangHyo/Binary/Binary_PaperLGBM_NoMMSE` — 같은 데이터에서 하루 단위 무작위
  K-fold(AUC 0.9526)와 사람 단위 OOF(AUC 0.5214)의 차이를 정량화한 실험
- `SangHyo/AGENTS.md` — 이 저장소의 평가·누수 방지 계약
