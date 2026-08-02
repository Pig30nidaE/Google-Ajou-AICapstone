# 논문 변수 ↔ 실제 데이터 매핑

대상 논문: Hong, J.; Seol, Y.; Lee, S.; Yoon, J.; Lee, J.; Park, K.-S.; Ha, J.-W.
*Prediction of Cognitive Impairment Using Sleep Lifelog Data and LSTM Model.*
Mathematics 2024, 12, 3208.

데이터: `Data/1.Training/SourceData/2.Sleep/train_sleep.csv` +
`Data/2.Validation/SourceData/2.Sleep/val_sleep.csv` (AI-Hub 「치매 고위험군
웨어러블 라이프로그」).

> **원칙:** 없는 컬럼을 이름이 비슷한 변수로 조용히 대체하지 않는다. 아래 매핑 중
> 이름이 다른 것은 전부 **수치로 검증**했고, 검증하지 못한 것은 assumption으로
> 표시했다. 검증은 `run.py --inspect-data`로 재현할 수 있다.

---

## 0. 데이터 규모 대조 (전부 일치)

| 항목 | 논문 Table 3 | 실측 | 일치 |
| --- | ---: | ---: | :---: |
| 전체 피험자 | 174 | 174 | O |
| CN (논문 NC) | 111 | 111 | O |
| MCI | 51 | 51 | O |
| Dem (논문 DE) | 12 | 12 | O |
| 수면 레코드 | 12,183 | 12,183 | O |
| 피험자별 관찰 | 35~122일 | 35~122 **행** | O |
| 입력 변수 | 32 | 32 | O |

논문의 "periods ranging from 35 to 122 days"는 **달력 기간이 아니라 레코드 수**와
정확히 일치한다. 실제 달력 span은 36~124일이며 중간에 공백이 있다(§4 참조).

## 1. Table 4와 Table A1의 불일치 해소

논문 안에서 두 표의 이름이 서로 다른 경우가 있다. **Table A1이 실제 컬럼명과
일치**하므로 Table A1을 채택했다.

| Table 4 표기 | Table A1 표기 | 실제 컬럼 | 채택 | 근거 |
| --- | --- | --- | --- | --- |
| `skin_temperature_delta` | `sleep_temperature_delta` | `sleep_temperature_delta` | Table A1 | 실제 컬럼과 일치 |
| `skin_temperature_deviation` | `sleep_temperature_deviation` | `sleep_temperature_deviation` | Table A1 | 실제 컬럼과 일치 |
| `sleep_midpoint_at_delta` | `sleep_midpoint_time_at_delta` | `sleep_midpoint_at_delta` | Table 4 | 실제 컬럼과 일치 |
| `strat4` | `start1-6` | (파생) | Table A1 | `strat4`는 `start4`의 오타 |

내부 특징 이름은 논문 Table 4 표기(`skin_temperature_*`)를 유지하고, 원본 컬럼
매핑만 바꿨다. `src/data/schema.py`의 `PASSTHROUGH_FEATURES`가 그 대응표다.

## 2. 원본 컬럼을 그대로 쓰는 변수 (17개)

| 논문 변수 | AI-Hub 컬럼 | 단위(Table A1) |
| --- | --- | --- |
| `sleep_awake` | `sleep_awake` | seconds |
| `sleep_deep` | `sleep_deep` | seconds |
| `sleep_duration` | `sleep_duration` | seconds |
| `sleep_efficiency` | `sleep_efficiency` | 1–100 |
| `sleep_light` | `sleep_light` | seconds |
| `sleep_rem` | `sleep_rem` | seconds |
| `sleep_midpoint_time` | `sleep_midpoint_time` | time delta |
| `sleep_midpoint_at_delta` | `sleep_midpoint_at_delta` | time delta |
| `sleep_onset_latency` | `sleep_onset_latency` | seconds |
| `sleep_restless` | `sleep_restless` | % |
| `skin_temperature_delta` | `sleep_temperature_delta` | celsius |
| `skin_temperature_deviation` | `sleep_temperature_deviation` | celsius |
| `sleep_total` | `sleep_total` | seconds |
| `sleep_breath_average` | `sleep_breath_average` | breaths/min |
| `sleep_hr_average` | `sleep_hr_average` | bpm |
| `sleep_hr_min` | **`sleep_hr_lowest`** | bpm |

`sleep_hr_min` ← `sleep_hr_lowest`는 이름이 다르므로 수치로 확인했다:
**`sleep_hr_lowest == min(0이 아닌 sleep_hr_5min)`이 1500행 표본에서 일치율
1.000**이다. 이름이 비슷해서 고른 것이 아니라 동일한 값임을 확인해서 골랐다.

## 3. 파생해야 하는 변수 (15개)

### 3-1. 5분 시계열에서 파생 (4개)

**중요:** `sleep_hr_5min`, `sleep_hypnogram_5min`, `sleep_rmssd_5min` 세 컬럼은
모든 행이 `"..."` 자리표시자다(고유값 1개). 실제 시계열은
`CONVERT(<컬럼> USING utf8)` 쌍둥이 컬럼에 슬래시 구분 문자열로 들어 있다.

| 논문 변수 | 계산 | 검증 |
| --- | --- | --- |
| `sleep_hr_max` | `max(0이 아닌 hr_5min)` | 원본 컬럼 없음. `sleep_hr_average`가 같은 계열의 평균과 r=1.0000, 최대오차 0.005로 일치하여 이 계열이 통계변수의 출처임이 확인됨 |
| `sleep_hr_median` | `median(0이 아닌 hr_5min)` | 위와 동일 |
| `sleep_hypnogram_average` | `mean(hypnogram_5min)` | 값 범위 1.40~3.22로 Table A1의 코드북(1=deep, 2=light, 3=REM, 4=awake)과 정합 |
| `rmssd_average` | `mean(0이 아닌 rmssd_5min)` | **모호(A-04)**. 사전집계 컬럼 `sleep_rmssd`와 r=0.997이지만 정확히 일치하는 행은 38.2%뿐 |

`0`은 기기의 "측정값 없음" 표식이므로 hr과 rmssd 통계에서 제외한다. hypnogram에는
0이 관측되지 않는다.

### 3-2. 취침·기상 시각 one-hot (12개)

Table A1: "Whether the start/end of the sleep time is in one of the six time zones
(0–4, 4–8, 8–12, 12–16, 16–20, 20–24 o'clock)".

- `start1`~`start6` ← `sleep_bedtime_start`의 시(hour)를 4로 나눈 몫
- `end1`~`end6` ← `sleep_bedtime_end`의 시(hour)를 4로 나눈 몫

검증: 전 행에서 `start1..6`의 합 = 1, `end1..6`의 합 = 1.

관측 분포(전체 12,171행): `start6`(20–24시) 75.3%, `start1`(0–4시) 14.2%,
`end2`(4–8시) 81.7%, `end3`(8–12시) 10.2%. 야간 수면이 지배적이라는 상식과 맞는다.

### 3-3. 공식이 명시된 변수

Table A1: `sleep_efficiency = (sleep_total / sleep_duration) × 100`.

검증: 원본 `sleep_efficiency` 컬럼과 계산값의 반올림 일치율 **0.999**, 최대오차
0.50(반올림 경계). 원본 컬럼을 그대로 사용한다.

## 4. 논문이 보고하지 않은 구조적 사실

`--inspect-data`가 매번 다시 측정하는 항목이다.

### 4-1. 같은 날짜에 두 개의 수면 레코드

논문은 "Each entry ... represents a single subject's sleep data for one day"라고만
적었다. 실제로는 하루에 두 번의 수면 구간이 기록된 경우가 있다.

| 날짜 키 | 중복 행 수 |
| --- | ---: |
| `sleep_bedtime_end` 기준 | **24행 (11명)** |
| `sleep_bedtime_start` 기준 | 1,044행 |

`bedtime_end`(기상일)를 날짜 키로 채택했다(A-01). Oura의 수면 요약일 관례와도
일치한다. 남은 중복은 `sleep_duration`이 가장 긴 레코드를 남기는 방식으로
12행을 제거해 12,171행을 얻는다(A-02).

### 4-2. 달력 공백

**174명 중 162명**의 기록에 하루 이상의 공백이 있다(총 1,089건). 피험자당 연속구간
중앙값은 6개다. 논문은 시퀀스 구성 시 공백을 어떻게 처리했는지 기술하지 않는다.

이 저장소는 **실제 연속된 calendar day만** 하나의 시퀀스로 묶는다. 행 순서만 보고
이어붙이면 3일 윈도우의 16.6%, 5일 윈도우의 28.6%가 실제로는 공백을 건너뛴다.

가장 긴 연속구간의 최솟값은 5일이므로, 모든 피험자가 5일 시퀀스를 최소 한 개는
만들 수 있다.

### 4-3. 완전히 동일한 두 변수

**`sleep_temperature_delta`와 `sleep_temperature_deviation`은 이 데이터
배포본에서 12,171행 전부 동일한 값이다.** 따라서 논문의 32개 변수 중 실질적으로
서로 다른 정보는 31개다.

Table A1은 두 변수를 "Skin temperature deviation delta"와 "Skin temperature
deviation"으로 다르게 설명하므로, 논문 저자가 받은 배포본에서도 같았는지는 알 수
없다. 논문 재현을 위해 **두 변수를 모두 유지**하되 이 사실을 기록한다. 상관 기반
특징 선택이나 선형모형 해석에서 이 중복이 문제가 될 수 있다.

## 5. 사용하지 않은 원본 컬럼

Table 4에 없으므로 제외한다. 누수는 아니지만 논문의 32개가 아니다.

`sleep_score`, `sleep_score_alignment`, `sleep_score_deep`,
`sleep_score_disturbances`, `sleep_score_efficiency`, `sleep_score_latency`,
`sleep_score_rem`, `sleep_score_total` — Oura가 같은 수면단계에서 파생한 점수다.

`sleep_is_longest`(전 행 1), `sleep_period_id`, `sleep_rmssd`, `sleep_hr_lowest`
— 앞의 둘은 정보가 없거나 행정용, 뒤의 둘은 파생변수의 원천으로만 쓴다.

## 6. 절대 입력이 될 수 없는 변수

`src/data/schema.py`의 `FORBIDDEN_FEATURE_SUBSTRINGS`가 fail-closed로 막는다.
config로도 켤 수 없다.

- 식별자: `EMAIL`, `SAMPLE_EMAIL`, `subject_id`
- 정답: `DIAG_NM`, `DIAG_SEQ`, `label`
- 진단에 사용된 인지검사: `MMSE_*`, `SNSB_*` — 이 실험은 **wearable-only**다.
  `Data/*/SourceData/3.CognitiveFunction/`은 열지 않는다.

## 7. 라벨

논문 §3.2: NC vs {MCI, DE} 이진 통합. AI-Hub 표기로는 `CN` → 0,
`MCI`/`Dem` → 1. 피험자 단위로 111 vs 63.

Gait 라벨 사본과 Sleep 라벨 사본이 일치하는지 로더가 매번 확인하고, 불일치하면
즉시 중단한다(`SangHyo/AGENTS.md` §1).
