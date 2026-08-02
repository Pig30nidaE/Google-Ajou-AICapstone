# paper_data_mapping.md — 논문 기재 변수 ↔ 실제 데이터 컬럼 대응

작성일: 2026-08-02
근거: 논문 표 1·2·3(§3.1–3.2)과 `Data/1.Training/SourceData/*`,
`Data/2.Validation/SourceData/*`, `Data/*/LabelingData/*` 실측.

원칙(사용자 지시 3·13): **컬럼명이 없다고 임의로 유사변수로 대체하지 않는다.**
실측 결과 대체가 필요한 변수는 **0개**였다.

---

## 0. 파일 대응

| 논문 기재 | 실제 경로 | 행 수 | 피험자 |
| --- | --- | ---: | ---: |
| 활동 데이터 | `Data/1.Training/SourceData/1.Gait/train_activity.csv` | 9,705 | 141 |
| 활동 데이터 | `Data/2.Validation/SourceData/1.Gait/val_activity.csv` | 2,478 | 33 |
| 수면 데이터 | `Data/1.Training/SourceData/2.Sleep/train_sleep.csv` | 9,705 | 141 |
| 수면 데이터 | `Data/2.Validation/SourceData/2.Sleep/val_sleep.csv` | 2,478 | 33 |
| (미사용) MMSE | `Data/*/SourceData/3.CognitiveFunction/*_mmse.csv` | 141 / 33 | 141 / 33 |
| 라벨 | `Data/*/LabelingData/{1.Gait,2.Sleep,3.CognitiveFunction}/*_label.csv` | 141 / 33 | 141 / 33 |

### 결정적 사실: 논문의 데이터셋 = AI-Hub Training + Validation **합본**

```
활동 행: 9,705 + 2,478 = 12,183   ← 논문 §3.1 "12,183건" 정확 일치
피험자 : 141   +    33 =    174   ← 논문 §3.1 "최종 174명" 정확 일치
```

논문은 AI-Hub가 제공한 Training/Validation 폴더 구분을 **사용하지 않고 합친 뒤
자체적으로 8:1:1로 재분할**했다. 본 재현도 동일하게 합본에서 출발한다
(`data.use_official_split: false`).

> ⚠️ AI-Hub 폴더명의 "Validation"과 논문 표 5의 "Vaild" 열은 **서로 다른 것**이다.
> 코드에서는 전자를 `source_partition`, 후자를 `split`으로 구분해 부른다.

### 라벨 파일 3종의 동일성 (실측)

- 3개 라벨 파일(`1.Gait`, `2.Sleep`, `3.CognitiveFunction`)은 각 partition 내에서 내용이 같다.
- MMSE 파일의 `DIAG_NM`과도 **174명 전원 불일치 0건**.
- 조인 키: 라벨/MMSE는 `SAMPLE_EMAIL`, 활동/수면은 `EMAIL`. 값 집합이 정확히 일치한다.

### 활동 ↔ 수면 행 정렬 (실측)

`train_activity.csv`와 `train_sleep.csv`는 행 수가 같고(9,705)
**`EMAIL` 열이 행 단위로 완전히 동일한 순서**로 정렬되어 있다(val도 동일).
따라서 두 파일은 **위치 기반(index-wise) 결합**이 가능하다.

```python
assert (activity["EMAIL"].values == sleep["EMAIL"].values).all()   # 실측 통과
X = pd.concat([activity[ACT_FEATURES], sleep[SLEEP_FEATURES]], axis=1)
```

논문은 결합 방법을 기재하지 않았다. **날짜 컬럼이 존재하지 않으므로 날짜 조인은
애초에 불가능하다** (`activity_day_start`/`sleep_bedtime_end`에서 파생은 가능하나
논문이 그런 파생을 언급하지 않음). 위치 결합을 기본으로 하고
`data.join_mode: positional`로 명시한다. → `assumptions.md` A-2.

---

## 1. 클래스 분포 검증 (논문 표 3)

| 항목 | 논문 표 3 | 실측 | 판정 |
| --- | ---: | ---: | :---: |
| CN 피험자 | 111 | **111** | ✅ |
| MCI 피험자 | 51 | **51** | ✅ |
| Dem 피험자 | 12 | **12** | ✅ |
| CN 기록 | 7,737 | **7,737** | ✅ |
| MCI 기록 | 3,661 | **3,661** | ✅ |
| Dem 기록 | 785 | **785** | ✅ |

**논문 표 3이 실제 데이터에서 완전히 재현된다.** 대상 코호트 확정에 어떤 가정도 필요 없다.

`DIAG_NM` 고유값 = `{CN, MCI, Dem}`. `AAD`는 존재하지 않는다(→ `report_inconsistencies.md` I-16).

### 피험자별 기록 수

| 클래스 | n | 평균 | 표준편차 | 최소 | 중앙값 | 최대 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CN | 111 | 69.7 | 21.1 | 35 | 66 | 115 |
| MCI | 51 | 71.8 | 19.2 | 37 | 67 | 122 |
| Dem | 12 | 65.4 | 19.3 | 40 | 64 | 107 |

**Dem 12명의 개별 기록 수** (오름차순):
`[40, 44, 50, 51, 55, 61, 67, 74, 74, 75, 87, 107]` (합 785)

이 숫자가 실험 B·C 설계를 지배한다. 3-fold group CV에서 fold당 Dem 피험자는 4명이며,
fold당 Dem 기록 수는 대략 200~300행으로 크게 흔들린다.

---

## 2. 활동 변수 22개 (논문 표 1)

논문 §3.1의 제외 규칙: 이메일 식별자, 5분 활동 로그, 활동 시작/종료 시간, 1분 MET 로그,
비활동 알람 횟수, 미착용 시간.

실제 CSV의 `activity_*` 컬럼은 **28개**이며(논문 §3.1의 "활동성 변수 28개"와 일치),
아래 6개를 제외하면 **정확히 22개**가 남는다.

### 2.1 제외 (6개) — 논문 서술과 1:1 대응

| 논문 서술 | 실제 컬럼 | 확인 |
| --- | --- | :---: |
| 5분 활동 로그 | `activity_class_5min` | ✅ |
| 활동 종료 시간 | `activity_day_end` | ✅ |
| 활동 시작 시간 | `activity_day_start` | ✅ |
| 1분 MET 로그 | `activity_met_1min` | ✅ |
| 비활동 알람 횟수 | `activity_inactivity_alerts` | ✅ |
| 미착용 시간 | `activity_non_wear` | ✅ |

추가로 CSV에는 `CONVERT(activity_class_5min USING utf8)`,
`CONVERT(activity_met_1min USING utf8)` 2개가 더 있다. 이는 위 두 로그 컬럼의 문자열 사본이며
논문의 변수 개수(28)에도 포함되지 않는다. → 제외.

> **주의**: 이 `CONVERT(...)` 컬럼들이 5분/1분 단위 원시 시계열을 담고 있다.
> 본 논문은 일별 요약만 사용하므로 사용하지 않지만, 존재 자체는 기록해 둔다.

### 2.2 사용 (22개) — 논문 표 1과 컬럼명 완전 일치

| # | 논문 표 1 | 실제 컬럼 | 존재 |
| ---: | --- | --- | :---: |
| 1 | activity_average_met | `activity_average_met` | ✅ |
| 2 | activity_cal_active | `activity_cal_active` | ✅ |
| 3 | activity_cal_total | `activity_cal_total` | ✅ |
| 4 | activity_daily_movement | `activity_daily_movement` | ✅ |
| 5 | activity_high | `activity_high` | ✅ |
| 6 | activity_inactive | `activity_inactive` | ✅ |
| 7 | activity_low | `activity_low` | ✅ |
| 8 | activity_medium | `activity_medium` | ✅ |
| 9 | activity_met_min_high | `activity_met_min_high` | ✅ |
| 10 | activity_met_min_inactive | `activity_met_min_inactive` | ✅ |
| 11 | activity_met_min_low | `activity_met_min_low` | ✅ |
| 12 | activity_met_min_medium | `activity_met_min_medium` | ✅ |
| 13 | activity_rest | `activity_rest` | ✅ |
| 14 | activity_score | `activity_score` | ✅ |
| 15 | activity_score_meet_daily_targets | `activity_score_meet_daily_targets` | ✅ |
| 16 | activity_score_move_every_hour | `activity_score_move_every_hour` | ✅ |
| 17 | activity_score_recovery_time | `activity_score_recovery_time` | ✅ |
| 18 | activity_score_stay_active | `activity_score_stay_active` | ✅ |
| 19 | activity_score_training_frequency | `activity_score_training_frequency` | ✅ |
| 20 | activity_score_training_volume | `activity_score_training_volume` | ✅ |
| 21 | activity_steps | `activity_steps` | ✅ |
| 22 | activity_total | `activity_total` | ✅ |

> 프롬프트가 우려한 `activity_low` / `active_low` 명명 차이는 **존재하지 않는다.**
> 실제 컬럼명은 `activity_low`이며 논문 표기와 같다.

---

## 3. 수면 변수 24개 (논문 표 2)

실제 CSV의 `sleep_*` 컬럼은 **32개**다(논문 §3.1은 33개라고 기재 →
`report_inconsistencies.md` I-13). 아래 8개를 제외하면 **정확히 24개**가 남는다.

### 3.1 제외 (8개)

| 논문 서술 | 실제 컬럼 | 확인 |
| --- | --- | :---: |
| 수면 종료 시간 | `sleep_bedtime_end` | ✅ |
| 수면 시작 시간 | `sleep_bedtime_start` | ✅ |
| 5분당 심박동 로그 | `sleep_hr_5min` | ✅ |
| 수면 상태 로그 | `sleep_hypnogram_5min` | ✅ |
| 본 수면 여부 | `sleep_is_longest` | ✅ |
| 수면 식별 아이디 | `sleep_period_id` | ✅ |
| ("등"에 포함) | `sleep_rmssd_5min` | ⚠️ 명시 안 됨 |
| ("등"에 포함) | `sleep_total` | ⚠️ 명시 안 됨 |

논문은 6개만 예시하고 "등"으로 마무리했다. 표 2에 `sleep_rmssd_5min`과 `sleep_total`이
없으므로 이 둘도 제외된 것이 확실하다(`32 − 8 = 24` ✅).

`sleep_rmssd_5min`은 5분 단위 로그이므로 제외 근거가 명확하다.
`sleep_total`(총 수면 시간)은 정량 변수인데도 제외되었다 —
`sleep_duration`(침대 체류 시간)과 중복이기 때문으로 보이나 논문에 근거가 없다.
→ `unresolved_questions.md` Q6.

추가 `CONVERT(...)` 3개(`sleep_hr_5min`, `sleep_hypnogram_5min`, `sleep_rmssd_5min`)도 제외.

### 3.2 사용 (24개) — 논문 표 2와 컬럼명 완전 일치

| # | 논문 표 2 | 실제 컬럼 | 존재 |
| ---: | --- | --- | :---: |
| 1 | sleep_awake | `sleep_awake` | ✅ |
| 2 | sleep_breath_average | `sleep_breath_average` | ✅ |
| 3 | sleep_deep | `sleep_deep` | ✅ |
| 4 | sleep_duration | `sleep_duration` | ✅ |
| 5 | sleep_efficiency | `sleep_efficiency` | ✅ |
| 6 | sleep_hr_average | `sleep_hr_average` | ✅ |
| 7 | sleep_hr_lowest | `sleep_hr_lowest` | ✅ |
| 8 | sleep_light | `sleep_light` | ✅ |
| 9 | sleep_midpoint_at_delta | `sleep_midpoint_at_delta` | ✅ |
| 10 | sleep_midpoint_time | `sleep_midpoint_time` | ✅ |
| 11 | sleep_onset_latency | `sleep_onset_latency` | ✅ |
| 12 | sleep_rem | `sleep_rem` | ✅ |
| 13 | sleep_restless | `sleep_restless` | ✅ |
| 14 | sleep_rmssd | `sleep_rmssd` | ✅ |
| 15 | sleep_score | `sleep_score` | ✅ |
| 16 | sleep_score_alignment | `sleep_score_alignment` | ✅ |
| 17 | sleep_score_deep | `sleep_score_deep` | ✅ |
| 18 | sleep_score_disturbances | `sleep_score_disturbances` | ✅ |
| 19 | sleep_score_efficiency | `sleep_score_efficiency` | ✅ |
| 20 | sleep_score_latency | `sleep_score_latency` | ✅ |
| 21 | sleep_score_rem | `sleep_score_rem` | ✅ |
| 22 | sleep_score_total | `sleep_score_total` | ✅ |
| 23 | sleep_temperature_delta | `sleep_temperature_delta` | ✅ |
| 24 | sleep_temperature_deviation | `sleep_temperature_deviation` | ✅ |

> 프롬프트가 우려한 `sleep_breath_average` 오타, `sleep_hr_lowest` 개명, temperature 컬럼명
> 문제는 **모두 존재하지 않는다.** 24개 전부 논문 표기와 문자 단위로 같다.

---

## 4. 🔴 발견: `sleep_temperature_delta`와 `sleep_temperature_deviation`은 **완전히 동일한 컬럼**

12,183행 전체에서 두 컬럼의 값이 **원소 단위로 100% 일치**한다(실측, `np.array_equal` 통과).
min/max/고유값 수도 동일: `[-1.98, 3.14]`, 286개 고유값.

즉 논문이 "46개 변수"라고 부르는 집합의 **실질 자유도는 45**이며,
완전 공선(collinear)인 변수쌍이 하나 들어 있다.

영향:

| 구성요소 | 영향 |
| --- | --- |
| StandardScaler | 무해 (독립 정규화) |
| XGBoost | 무해하나 두 변수가 중요도를 나눠 가짐 → 해석 왜곡 |
| DNN / W&D / TabNet | 무해 (가중치가 나뉠 뿐) |
| **VAE** | **재구성 손실에서 동일 오차가 2회 계산**되어 이 축의 가중치가 2배 |
| 상관행렬 진단 | corr = 1.0 셀이 생겨 진단 지표를 오염 |

**처리**: `features.drop_duplicate_columns: false`를 **기본값**으로 둔다.
논문이 46개를 썼다고 명시했으므로 재현에서는 46개를 그대로 사용한다.
`true`로 두면 45개를 쓰는 민감도 변형이 되며, 두 설정의 차이를 보고한다.
어느 쪽이든 로더가 중복 쌍을 **항상 로그로 경고**한다.

---

## 5. 변수 프로파일 (VAE 생성값 유효성 검사에 사용)

실측 12,183행 기준.

| 속성 | 값 |
| --- | --- |
| 결측치 | **0건** (46개 변수 전부) → imputer는 사실상 no-op |
| 완전 중복 행 | **0건** |
| dtype | int64 41개, float64 5개 |
| 비정수 변수(5) | `activity_average_met`, `sleep_breath_average`, `sleep_hr_average`, `sleep_temperature_delta`, `sleep_temperature_deviation` |
| **음수가 존재하는 변수(3)** | `sleep_midpoint_at_delta` (502건), `sleep_temperature_delta` (5,985건), `sleep_temperature_deviation` (5,985건) |
| 나머지 43개 변수 | **전부 ≥ 0** → 생성값이 음수면 유효성 위반 |

### 값 범위가 극단적으로 다르다 (VAE 손실 설계에 직결)

| 변수 | 최소 | 최대 |
| --- | ---: | ---: |
| `sleep_duration` | 10,800 | 54,000 |
| `activity_daily_movement` | 0 | 46,659 |
| `activity_steps` | 41 | 44,836 |
| `sleep_awake` | 510 | 27,420 |
| … | | |
| `activity_average_met` | 0.0625 | 4.03 |
| `sleep_temperature_delta` | −1.98 | 3.14 |

최대 스케일 차이가 **4자릿수**다. 원 단위 공간에서 MSE 재구성 손실을 쓰면
`sleep_duration`·`activity_daily_movement` 등 소수 변수만 최적화된다.
논문 §4.2는 StandardScaler를 증강 **이후**에 적용한다고 읽히므로
(→ `report_inconsistencies.md` I-9) 이 문제가 실제로 발생했을 가능성이 있다.

**처리**: `augmentation.vae.input_space: raw | scaled` 두 변형을 모두 구현하고,
재구성 손실을 두 공간 모두에서 로깅한다.

### 0–100 점수형 변수 (12개)

`activity_score`, `activity_score_*` 7개, `sleep_score`, `sleep_score_*` 7개는
관측 범위가 [0, 100] 또는 [1, 100]이다. 특히
`activity_score_meet_daily_targets`(고유값 8개), `activity_score_move_every_hour`(8개),
`activity_score_training_frequency`(6개)는 사실상 **이산 등급 변수**다.
VAE가 연속값을 생성하면 원 데이터에 존재하지 않는 값이 나온다.

**처리**: `augmentation.vae.postprocess`에서
`clip_to_train_range`(기본 on), `enforce_nonnegative`(기본 on),
`round_integer_valued`(기본 **off**, 논문 미보고이므로) 를 config로 제어하고
위반 건수를 provenance에 기록한다. → `synthetic_data_risk.md` §3.

---

## 6. 금지 변수 (사용자 지시 12·13)

입력 feature에 **절대 포함되지 않아야 하는** 컬럼. `src/data/schema.py`의
`FORBIDDEN_FEATURE_PATTERNS`로 정의하고 `tests/test_forbidden_features.py`가 강제한다.

| 컬럼 | 소속 | 금지 사유 |
| --- | --- | --- |
| `EMAIL`, `SAMPLE_EMAIL` | activity/sleep, label/mmse | 피험자 식별자 (지시 12) |
| `DIAG_NM` | label, mmse | 타깃 그 자체 |
| `DIAG_SEQ`, `DOCTOR_NM` | mmse | 진단 절차 파생 |
| `MMSE_NUM`, `MMSE_KIND` | mmse | 인지검사 메타 |
| `TOTAL` | mmse | **MMSE 총점 — 진단범주 구성 변수** (지시 13) |
| `Q01`–`Q19`, `Q11_*`, `Q12_*`, `Q13_*`, `Q14_*`, `Q16_*`, `Q12_TOTAL` | mmse | MMSE 개별 문항 (지시 13) |

MMSE 총점의 클래스별 분포(실측)는 이 금지가 왜 필수인지 보여준다.

| 클래스 | n | 평균 | 표준편차 | 최소 | 최대 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CN | 111 | 27.70 | 1.78 | 20 | 30 |
| MCI | 51 | 25.82 | 3.28 | 17 | 30 |
| Dem | 12 | 16.58 | 8.03 | 0 | 27 |

MMSE 총점 하나만으로도 Dem이 거의 분리된다. **MMSE 파일은 로더가 아예 읽지 않는다**
(`--inspect-data`에서 진단 목적으로만 읽고, 그때도 feature 프레임에 넣지 않는다).

---

## 7. 최종 feature 계약

```
n_features = 46  (활동 22 + 수면 24)
n_rows     = 12,183
n_subjects = 174
classes    = {CN: 0, MCI: 1, Dem: 2}
결측       = 0
실질 자유도 = 45  (temperature 쌍이 동일, §4)
```

`src/data/schema.py`가 이 계약을 상수로 보유하고, 로더가 매 실행마다 검증한다.
어긋나면 `SchemaError`를 던진다 — 조용한 수정은 하지 않는다(사용자 지시 8).
