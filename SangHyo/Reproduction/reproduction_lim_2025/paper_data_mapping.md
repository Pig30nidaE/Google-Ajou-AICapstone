# paper_data_mapping.md — 논문 기재 변수 ↔ 실제 데이터 컬럼 대응

작성일: 2026-08-02
근거: 학위논문 표 1–11 / 학술지 논문 표 1–8, 그리고
`Data/1.Training/SourceData/*`, `Data/2.Validation/SourceData/*` 실측.

---

## 0. 파일 대응

| 논문 코드 스니펫 | 실제 경로 | 행 수 | 피험자 |
| --- | --- | ---: | ---: |
| `train_activity.csv` | `Data/1.Training/SourceData/1.Gait/train_activity.csv` | 9,705 | 141 |
| `train_sleep.csv` | `Data/1.Training/SourceData/2.Sleep/train_sleep.csv` | 9,705 | 141 |
| `train_mmse.csv` | `Data/1.Training/SourceData/3.CognitiveFunction/train_mmse.csv` | 141 | 141 |
| (암시된 val 블록) | `Data/2.Validation/SourceData/1.Gait/val_activity.csv` | 2,478 | 33 |
| (암시된 val 블록) | `Data/2.Validation/SourceData/2.Sleep/val_sleep.csv` | 2,478 | 33 |
| (암시된 val 블록) | `Data/2.Validation/SourceData/3.CognitiveFunction/val_mmse.csv` | 33 | 33 |
| (미언급) | `Data/*/LabelingData/{1.Gait,2.Sleep,3.CognitiveFunction}/*_label.csv` | 141 / 33 | — |

- 라벨 사본 3개는 각 split 내에서 **완전 동일**하며, MMSE의 `DIAG_NM`과도 **불일치 0건**이다.
- 논문은 라벨을 MMSE 파일의 `DIAG_NIM`에서 가져온다고 서술했다. 실제 컬럼명은 `DIAG_NM`이다.
- 논문 스니펫은 `train_mm.rename(columns={'SAMPLE_EMAIL':'EMAIL'})` 후 EMAIL로 병합한다.
  실측상 `SAMPLE_EMAIL`(MMSE/label) ↔ `EMAIL`(activity/sleep)이며 값 집합이 정확히 일치한다.

---

## 1. 결정적 구조 차이: **날짜 컬럼이 없다**

논문 표 1 (학술지 표 1)은 두 번째 변수로 `date` (요약 날짜, varchar(10))를 제시한다.
**실제 배포 CSV에는 `date` 컬럼이 존재하지 않는다.**

| 논문 기재 | 실제 데이터 | 처리 |
| --- | --- | --- |
| `date` (요약 날짜) | **없음** | activity는 `activity_day_start`, sleep은 `sleep_bedtime_end`에서 파생 |
| `check` (착용 여부) | **없음** | drop 목록에 있으나 no-op |
| `nonwear` (미착용 시간 체크) | **없음** | drop 목록에 있으나 no-op |
| `timezone` (시간 장소 정보) | **없음** | drop 목록에 있으나 no-op |

→ 논문의 drop 목록 5개 중 실제로 무언가를 제거하는 것은
`activity_non_wear`, `activity_inactivity_alerts` **2개뿐**이다.
나머지 3개는 `errors='ignore'` 덕분에 조용히 무시된다.

이 차이는 **시계열 모델에 직접 영향**을 준다. 논문은 `date`로 정렬했다고 암시하지만,
실제로는 타임스탬프 파싱이 필요하며 결합키 정의도 재현자가 선택해야 한다.

### 1-1. 본 재현의 날짜 파생 규칙

```python
activity_date = to_datetime(activity_day_start, utc=True).tz_convert("Asia/Seoul").date()
sleep_date    = to_datetime(sleep_bedtime_end,  utc=True).tz_convert("Asia/Seoul").date()
```

`sleep_bedtime_end`(기상 시각)를 쓰는 이유: `bedtime_start`는 전날 저녁일 수 있어
`activity_day_start`(당일 04:00)와 어긋난다. 기상일 기준이 활동일과 정렬된다.

### 1-2. activity ↔ sleep 결합

실측:
- 두 파일의 **행 수가 피험자별로 완전히 동일**하다 (모든 피험자에서 일치).
- 행 순서의 EMAIL 배열이 **완전히 동일**하다.
- 파생 날짜가 **99.65%** 일치한다.

따라서 두 가지 결합 모드를 제공한다.

| 모드 | 설명 | 결과 |
| --- | --- | --- |
| `positional` (기본) | 피험자 내 행 순서 i ↔ i | 12,183행 전부 보존 |
| `date` | `(subject_id, date)` inner join | train 9,684행 (21행 손실), val 2,478행 |

`date` 모드에서 행이 손실되는 이유는 sleep의 `bedtime_end` 날짜에 중복이 있기 때문이다
(train 11건, val 1건 — 같은 날 새벽·저녁 두 번 잔 경우 등).

---

## 2. 변수 58개 주장의 해부

논문 표 9–11(학술지 표 7–8)은 58행을 제시하며 "총 58개의 변수가 학습에 사용됐다"고 쓴다.
58행 중 1번은 `email`(식별자), 58번은 `DIAG_NIM`(타깃)이므로 **실제 특징은 56개**다.

그 56개를 실제 데이터에 대조하면:

| 구분 | 개수 | 상세 |
| --- | ---: | --- |
| 논문 기재 특징 | 56 | 표 9–11의 2–57번 |
| ① 실제 데이터에 **없음** | 2 | `active_low`, `sleep_temperature_trend_deviation` |
| ② 문자열이라 `numeric_only=True`가 **탈락**시킴 | 6 | `activity_class_5min`, `sleep_bedtime_start`, `sleep_bedtime_end`, `sleep_hr_5min`, `sleep_hypnogram_5min`, `sleep_rmssd_5min` |
| ③ drop 목록으로 제거 | 0 | (drop 목록 변수는 애초에 56개 안에 없음) |
| **④ 최종 생존 = 논문 코드 실행 결과** | **49** | activity 22 + sleep 27 |

### 2-1. ①에 대한 세부

| 논문 기재 | 실제 컬럼 | 판정 |
| --- | --- | --- |
| `active_low` | `activity_low` | **표기 오류.** 의미상 동일 변수, 본 재현은 `activity_low`로 매핑 |
| `sleep_temperature_trend_deviation` | — | **부재.** 실제 sleep에는 `sleep_temperature_delta`, `sleep_temperature_deviation` 2개만 존재 |

참고: 논문 표 4는 `sleep_temperature_deviation`과 `sleep_temperature_trend_deviation`의
한글 설명을 **둘 다 "피부 온도 편차"**로 적고 있어, 원 데이터 사전 자체의 중복 항목을
그대로 옮긴 것으로 보인다.

### 2-2. ②에 대한 세부 — 조용한 탈락

논문 코드는 `groupby('EMAIL').mean(numeric_only=True)`를 쓴다.
`numeric_only=True`는 문자열 컬럼을 **예외 없이 조용히 버린다.**

| 논문이 "사용했다"고 한 변수 | 실제 dtype | 결과 |
| --- | --- | --- |
| `activity_class_5min` | str (`'...'`) | 탈락 |
| `sleep_bedtime_start` | str (ISO8601) | 탈락 |
| `sleep_bedtime_end` | str (ISO8601) | 탈락 |
| `sleep_hr_5min` | str (`'...'`) | 탈락 |
| `sleep_hypnogram_5min` | str (`'...'`) | 탈락 |
| `sleep_rmssd_5min` | str (`'...'`) | 탈락 |

→ 논문이 "5분 단위로 수집" "BLOB 로그"라고 강조한 **고해상도 시계열은 실제 학습에
단 하나도 들어가지 않았다.**

### 2-3. 5분 단위 원자료는 어디 있나

실제 CSV에서 원래 BLOB 컬럼(`activity_class_5min` 등)은 값이 문자 `'...'`로 잘려 있고,
**실제 계열은 별도의 `CONVERT(... USING utf8)` 컬럼에 들어 있다.**

| 잘린 컬럼 | 실제 계열이 있는 컬럼 | 예시값 |
| --- | --- | --- |
| `activity_class_5min` | `CONVERT(activity_class_5min USING utf8)` | `1/1/2/2/2/2/1/1/...` |
| `activity_met_1min` | `CONVERT(activity_met_1min USING utf8)` | `1.2/0.9/1/0.9/...` |
| `sleep_hr_5min` | `CONVERT(sleep_hr_5min USING utf8)` | `63/61/59/58/...` |
| `sleep_hypnogram_5min` | `CONVERT(sleep_hypnogram_5min USING utf8)` | `4/2/2/2/2/1/1/...` |
| `sleep_rmssd_5min` | `CONVERT(sleep_rmssd_5min USING utf8)` | `18/28/25/28/...` |

논문은 이 `CONVERT` 컬럼을 언급하지 않으며, 사용한 흔적도 없다.
**본 재현의 주 분석에서도 사용하지 않는다** (논문 범위 밖).
확장 실험이 필요하면 별도 `assumption_variant`로 분리해야 한다.

---

## 3. 최종 특징 49개 (paper_code_verbatim)

논문 코드를 그대로 실행했을 때 생성되는 컬럼 목록이다. 본 재현의 기본 특징집합이다.

### 3-1. Activity (22개)
```
activity_average_met                activity_score
activity_cal_active                 activity_score_meet_daily_targets
activity_cal_total                  activity_score_move_every_hour
activity_daily_movement             activity_score_recovery_time
activity_high                       activity_score_stay_active
activity_inactive                   activity_score_training_frequency
activity_low                        activity_score_training_volume
activity_medium                     activity_steps
activity_met_min_high               activity_total
activity_met_min_inactive
activity_met_min_low
activity_met_min_medium
```

### 3-2. Sleep (27개)
```
sleep_awake                sleep_restless
sleep_breath_average       sleep_rmssd
sleep_deep                 sleep_score
sleep_duration             sleep_score_alignment
sleep_efficiency           sleep_score_deep
sleep_hr_average           sleep_score_disturbances
sleep_hr_lowest            sleep_score_efficiency
sleep_is_longest    ⚠️      sleep_score_latency
sleep_light                sleep_score_rem
sleep_midpoint_at_delta    sleep_score_total
sleep_midpoint_time        sleep_temperature_delta
sleep_onset_latency        sleep_temperature_deviation
sleep_period_id     ⚠️      sleep_total
sleep_rem
```

⚠️ 주의 대상 2개:

| 변수 | 문제 | 본 재현 처리 |
| --- | --- | --- |
| `sleep_is_longest` | 전 12,183행이 **상수 1** (무분산) | 논문 코드 재현에는 포함, 무분산 경고 기록. `drop_zero_variance: true`로 제거 가능 |
| `sleep_period_id` | 수면 세션 **행정 ID** (0–7). 생리 신호 아님 | 논문 코드 재현에는 포함. 실험 B·C에서는 `drop_administrative: true` 기본 |

`sleep_period_id`는 개인 식별자는 아니지만 의미 없는 인덱스이므로, 프롬프트 §11·§12
정신에 따라 누수 통제 실험에서는 기본 제외한다. 이 선택은 `assumptions.md`에 기록했다.

---

## 4. MMSE 데이터 대응

| 논문 기재 | 실제 컬럼 | 비고 |
| --- | --- | --- |
| `SAMPLE_EMAIL` | `SAMPLE_EMAIL` | 식별자 — **특징 사용 금지** |
| `DIAG_SEQ` (차수) | `DIAG_SEQ` | 1–6, 피험자당 1행뿐 — 진단 절차 메타, **사용 금지** |
| `DIAG_NIM` (진단명) | **`DIAG_NM`** | 타깃 원천 — **특징 사용 금지** |
| `MMSE_KIND` (친절도) | `MMSE_KIND` | 실측 전부 상수 2 — **사용 금지**(행정) |
| — (논문 미기재) | `DOCTOR_NM` | 실측 고유값 1 — **사용 금지**(측정기관) |
| — (논문 미기재) | `MMSE_NUM` | **사용 금지**(행정) |
| `Q01`–`Q19` (하위문항) | `Q01`…`Q19`, `Q11_1..3`, `Q12_1..5`, `Q12_TOTAL`, `Q13_1..3`, `Q14_1..2`, `Q16_1..3` (총 31개) | 인지검사 — 주 분석 **제외**, secondary 전용 |
| `TOTAL` (총점) | `TOTAL` | 인지검사 — 주 분석 **제외**, secondary 전용 |

### 4-1. 문항 코딩

문항 값은 **{1, 2}**이며 논문 표 6–8의 "정답/오답 : 2 / 1" 표기와 일치한다.
즉 **2 = 정답, 1 = 오답**이다. 0/1 이진이 아니므로 합산·역코딩 시 주의가 필요하다.

`TOTAL` = **정답(2) 문항 수**임을 실측 확인했다. `Q12_TOTAL`을 제외한 30개 문항 기준으로
train·validation **양쪽 모두 일치율 100%**다.

`TOTAL` 실측 범위 (Training) 5–30. 진단군별 평균:

| DIAG_NM | n | TOTAL 평균 | min | max |
| --- | ---: | ---: | ---: | ---: |
| CN | 85 | 27.59 | 20 | 30 |
| MCI | 47 | 25.55 | 17 | 29 |
| Dem | 9 | 19.22 | 5 | 27 |

`TOTAL`은 진단에 사용된 검사 점수이므로 주 분석에서 반드시 제외한다.

### 4-1-1. 논문이 언급하지 않은 MMSE 품질 문제 3가지

`--inspect-data`가 자동 검출한다. **secondary 분석에만 영향**을 주지만,
논문의 MMSE 결론(표 15/12)과 직접 관련되므로 기록한다.

**(a) `Q12_TOTAL`은 전 피험자 상수 0이다.**

논문 표 7은 이를 "12번 질문 total 점수"로 기재한다. 실제로는 174명 전원이 0이며,
`Q12_1`–`Q12_5`의 합과도 무관하다(정답 수가 0–5로 분포하는데 `Q12_TOTAL`은 항상 0).
**사용 불가능한 컬럼**이므로 특징에서 제외한다.

**(b) Validation 피험자 1명은 전 문항이 0이다.**

해당 피험자는 **Dem 환자**이며 `TOTAL`도 0이다. MMSE 척도가 1/2인데 0이 나온 것이므로
"0점"이 아니라 **미실시(결측) 표식**으로 읽어야 한다. 오각형 그리기·문장 쓰기까지 포함해
30문항 전부가 0이라는 점이 이를 뒷받침한다.

영향: 이 1명을 0점으로 합산하면 Dem 군의 `TOTAL` 평균이 크게 내려가고 극단 이상치가 된다.
논문이 이 값을 어떻게 처리했는지는 기재가 없다. 본 재현의 secondary 분석은 이를
**결측으로 처리**하며, 0점 유지 변형도 민감도 분석으로 제공한다.

**(c) Training에서 무분산인 문항이 4개 있다.**

`Q11_1`, `Q11_2`, `Q11_3`, `Q14_2`는 Training 141명 전원이 정답(2)이다.
Training만으로 단변량 회귀를 돌리면 이 문항들은 계수가 정의되지 않는다.
논문이 "다중공선성이 있는 변수들을 제거"했다고 한 대상에 이들이 포함되었을 가능성이 있으나,
논문에 제거 목록이 없어 확인할 수 없다(`unresolved_questions.md` Q14).

### 4-2. 논문 표 15의 `Q003`

논문 표 15(학술지 표 12)는 유의 변수로 `Q003`을 제시한다. 실제 컬럼명은 **`Q03`**이며
문항 내용("오늘은 며칠입니까?")이 일치하므로 `Q03`으로 매핑한다.

주의: 논문은 `Q003`을 "지남력"으로 해석하지만, 표 15의 문항 내용은 "오늘이 며칠인가요?"
즉 **시간 지남력**이다. 본 재현은 해석을 옮기지 않고 문항 코드만 매핑한다.

---

## 5. 타깃 정의 매핑

```
DIAG_NM ∈ {CN, MCI, Dem}
label = 0  if DIAG_NM == "CN"
label = 1  if DIAG_NM in {"MCI", "Dem"}
```

논문 서술: "전조증상 또한 치매로 가정하고 이진형 변수(완전 정상군은 0, 치매 전조증상
이상은 1)로 바꿔 구성했다."

실측 검증:

| 집단 | CN | MCI | Dem | label=0 | label=1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training (141) | 85 | 47 | 9 | 85 | 56 |
| Validation (33) | 26 | 4 | 3 | 26 | **7** |
| **합계 (174)** | **111** | **51** | **12** | **111** | **63** |

논문 기재 "63명이 치매 환자로 분류됐다" → **실측과 정확히 일치**. ✅
논문 기재 "치매군이 전체의 36%" → 63/174 = 36.2% → **일치**. ✅

---

## 6. 금지 변수 목록 (fail-closed)

주 분석(`main_lifelog_only`)에서 특징 행렬에 들어오면 **즉시 예외**를 발생시킨다.

### 6-1. 타깃 누수
`DIAG_NM`, `DIAG_NIM`, `DIAG_SEQ`, `label`, `y`, `target`

### 6-2. 인지검사 (주 분석 금지, secondary 전용)
`TOTAL`, `Q12_TOTAL`, 그리고 `Q`로 시작하는 모든 문항 컬럼 31개,
그 외 MMSE·SNSB 파생 일체

### 6-3. 개인 식별·행정
`EMAIL`, `SAMPLE_EMAIL`, `subject_id`, `subject_key`, `DOCTOR_NM`, `MMSE_NUM`,
`MMSE_KIND`, 파일명·경로·측정기관 파생 일체

### 6-4. 실험 B·C 추가 기본 제외
`sleep_period_id` (행정 인덱스), `sleep_is_longest` (무분산)
— `drop_administrative`, `drop_zero_variance` 설정으로 제어

---

## 7. 논문 ↔ 데이터 불일치 종합표

| # | 항목 | 논문 | 실제 데이터 | 심각도 |
| --- | --- | --- | --- | --- |
| 1 | 일별 기록 수 | 12,184 | **12,183** | 중 — 재현 수치 비교 시 명시 필요 |
| 2 | `date` 컬럼 | 존재한다고 기재 | **부재** | **높음** — 시계열 정렬 방법 재현 불가 |
| 3 | `check`/`nonwear`/`timezone` | drop 대상으로 기재 | **부재** | 낮음 — no-op |
| 4 | `active_low` | 기재 | 실제는 `activity_low` | 낮음 — 표기 오류 |
| 5 | `sleep_temperature_trend_deviation` | 기재 | **부재** | 중 — 특징 수 감소 |
| 6 | `DIAG_NIM` | 기재 | 실제는 `DIAG_NM` | 낮음 — 표기 오류 |
| 7 | 사용 변수 58개 | 주장 | 논문 코드 실행 결과 **49개** | **높음** — 입력 차원 불일치 |
| 8 | 5분 단위 BLOB 사용 | 표에 포함 | `numeric_only`가 전부 탈락 | **높음** — 서술과 코드 모순 |
| 9 | 5분 계열 위치 | 미언급 | `CONVERT(...)` 컬럼에 실재 | 중 — 논문이 놓친 자료 |
| 10 | `Q003` | 기재 | 실제는 `Q03` | 낮음 |
| 11 | 표 9 값 예시 | `activity_class_5min` = 3659.73 | `activity_daily_movement` 값과 동일 | 낮음 — 표 정렬 오류 |
| 12 | `sleep_is_longest` | 특징으로 기재 | 전 행 상수 1 (무분산) | 중 |
| 13 | 분할단위 | 미기재 | 피험자 단위로 복원됨 | **높음** — `reproduction_spec.md` §2 |
| 14 | activity↔sleep 결합키 | 미기재 | 날짜 컬럼 없어 파생 필요 | **높음** |
| 15 | MMSE 문항 코딩 | "정답/오답 2/1" | 실측 {1,2} 일치 | 없음 |
| 16 | `Q12_TOTAL` | "12번 질문 total 점수" | 전 피험자 **상수 0**, Q12 합계도 아님 | 중 (secondary) |
| 17 | 전 문항 0인 피험자 | 미언급 | Validation **1명(Dem)**, 미실시 추정 | 중 (secondary) |
| 18 | Training 무분산 문항 | 미언급 | `Q11_1`,`Q11_2`,`Q11_3`,`Q14_2` 전원 정답 | 중 (secondary) |
| 19 | `TOTAL` 정의 | "검사 답변의 총점" | 정답 문항 수와 **100% 일치** 확인 | 없음 |

`run.py --inspect-data`는 이 표를 실제 데이터에서 다시 계산하여
`outputs/inspection/discrepancy_report.md`로 출력한다. 위 값과 달라지면
**데이터 버전이 바뀐 것**이므로 재현 전에 반드시 확인해야 한다.
