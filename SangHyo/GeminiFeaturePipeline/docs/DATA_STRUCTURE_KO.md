# 실제 데이터 구조 분석

작성일 2026-07-29. 아래 수치는 모두 저장소의 `Data/` 파일을 직접 읽어 확인한 값이며,
README나 기존 대화가 아니라 파일 내용과 기존 코드 사용처를 근거로 판단했다.

## 1. 확인한 파일과 역할

| 파일 | 행 x 열 | 피험자 | 역할 |
| --- | ---: | ---: | --- |
| `Data/1.Training/SourceData/1.Gait/train_activity.csv` | 9,705 x 31 | 141 | 하루 1행 활동 요약 + 일중 시계열 |
| `Data/1.Training/SourceData/2.Sleep/train_sleep.csv` | 9,705 x 36 | 141 | 수면 구간 1행 + 수면 시계열 |
| `Data/1.Training/SourceData/3.CognitiveFunction/train_mmse.csv` | 141 x 38 | 141 | MMSE 문항/총점 **+ `DIAG_NM`(진단)** |
| `Data/1.Training/LabelingData/{1.Gait,2.Sleep,3.CognitiveFunction}/training_label.csv` | 141 x 2 | 141 | 진단 라벨, 3개 파일 내용 동일 |
| `Data/2.Validation/...` | 2,478 / 33 | 33 | 같은 구조의 검증 분할 |

* 라벨 3개 사본은 완전히 동일하다. 본 구현은 `1.Gait`, `2.Sleep` 두 사본만 읽고 서로
  일치하는지 확인한 뒤 사용한다(`data.load_diagnoses`).
* `3.CognitiveFunction`의 라벨 사본은 열지 않는다. MMSE 파일도 `usecols` allow-list로만
  읽어 `DIAG_NM`이 메모리에 들어오지 않게 한다.

## 2. 피험자 ID와 조인 키

* SourceData: `EMAIL`, LabelingData/MMSE: `SAMPLE_EMAIL`. 값은 모두
  `nia+NNN@rowan.kr` 형식이며 두 열의 값 집합이 서로 대응한다.
* 근거: (1) 실제 값 확인, (2) 기존 코드가 동일하게 사용
  (`Binary_Wearable_SequenceFusion_Google/data.py`, `Binary_Google_ROCAUC_Champion/data.py`),
  (3) 조인 후 141/33명이 정확히 유지됨.
* 파일 간 조인 키는 `EMAIL == SAMPLE_EMAIL` 하나뿐이고 다른 후보는 없다.

## 3. 행 단위와 피험자 단위의 관계

* Activity/Sleep은 **하루 1행**이고, 한 사람의 관측일수는 35~120일(중앙값 66일)이다.
* 관측 창은 2020-10-17 ~ 2021-02-17이며 사람마다 시작일이 다르다.
* Activity와 Sleep을 정렬해 병합하면 학습 분할에서 9,684 subject-day / 141명이 남는다.
  * 병합 규칙(기존 audited loader에서 그대로 차용):
    `activity_day_start`의 로컬 날짜 == `sleep_bedtime_end`의 로컬 날짜,
    활동 구간 길이 23~25시간, 같은 날짜에 수면이 여러 건이면 가장 긴 수면 1건.
* 따라서 **날짜 행을 무작위로 분할하면 같은 사람의 다른 날이 train/test에 동시에 들어가는
  누수**가 된다. 본 구현의 학습 단위는 항상 사람 1명 = 1행이다.

## 4. 시간 컬럼

* 활동: `activity_day_start`, `activity_day_end` (ISO8601 +09:00).
* 수면: `sleep_bedtime_start`, `sleep_bedtime_end`.
* 모두 로컬(Asia/Seoul)로 변환해 사용한다. **절대 날짜는 Gemini payload에 절대 넣지 않고**
  사람별 상대 `day_index`(첫 관측일=0)와 주중/주말 플래그만 사용한다.

## 5. 일중 시계열이 실제로 들어 있는 위치

`activity_class_5min`, `activity_met_1min`, `sleep_hr_5min`, `sleep_hypnogram_5min`,
`sleep_rmssd_5min` 컬럼은 **전 행이 문자열 `"..."`** 이다. 실제 값은
`CONVERT(<컬럼> USING utf8)` 컬럼에 슬래시 구분 문자열로 들어 있다.

| 시계열 | 길이 | 코드 의미 |
| --- | --- | --- |
| `CONVERT(activity_class_5min ...)` | 288 (5분 x 24h), 일부 285~287 | 0=비착용, 1=rest, 2=inactive, 3=low, 4=medium, 5=high |
| `CONVERT(activity_met_1min ...)` | 1440 (1분) | MET 값 |
| `CONVERT(sleep_hr_5min ...)` | 36~181 가변 | 심박, 0은 결측 |
| `CONVERT(sleep_hypnogram_5min ...)` | 36~180 가변 | 1=deep, 2=light, 3=rem, 4=awake |
| `CONVERT(sleep_rmssd_5min ...)` | 가변 | RMSSD, 0은 결측 |

비착용 코드 0은 MET 마스킹에만 쓰고 특징으로 만들지 않는다.

## 6. 타깃

* `DIAG_NM` ∈ {CN, MCI, Dem}. Training은 CN 85 / MCI 47 / Dem 9,
  Validation은 CN 26 / MCI 4 / Dem 3.
* 기본 이진 타깃은 저장소 표준(`SangHyo/AGENTS.md` 1절)과 동일하게
  **CN=0, MCI+Dem=1** (85 vs 56). `config.yaml`의
  `data.positive_diagnoses`/`negative_diagnoses`로 변경 가능하다.
* `Binary_Google_DemScreen`의 CN+MCI vs Dem은 표본 수와 난이도가 다른 별도 과제이므로
  이 파이프라인의 결과와 직접 비교하지 않는다.

## 7. MMSE 컬럼

* 문항 30개(`Q01`~`Q19` 계열) + `TOTAL` + 행정 메타(`DIAG_SEQ`, `DOCTOR_NM`,
  `MMSE_NUM`, `MMSE_KIND`) + `DIAG_NM`.
* **문항 코딩은 1=오답, 2=정답이다.** 검증: `TOTAL == (문항값==2)의 개수`가 141/141명에서
  정확히 성립한다. 따라서 `item_max`는 데이터에서 학습할 값이 아니라 상수 2.0이다.
* `Q11_1`, `Q11_2`, `Q11_3`, `Q14_2`는 전원 2점(분산 0)이고 `Q12_TOTAL`은 전원 0이다.
  `Q12_TOTAL`은 사용하지 않는다.
* `TOTAL` 분포: 평균 26.38, 표준편차 3.52, 최소 5, 최대 30.
* MMSE는 진단에 사용된 임상 인지검사이므로 강한 예측 신호이자 부분적으로 순환적인
  정보다. MMSE 포함 성능을 "웨어러블만으로 스크리닝한 성능"으로 표현하면 안 된다.

## 8. 결측과 중복

* Activity/Sleep 원본 CSV에는 `NaN` 셀이 없다. 결측은 (a) 병합에서 탈락한 날,
  (b) 시계열 내부의 0 코드(비착용/미측정), (c) 사람별 관측일 사이의 빈 날로 나타난다.
* 같은 subject-day 중복은 병합 단계에서 결정론적으로 1건만 남긴다.
* 사람별 관측 창(span) 36~124일 대비 실제 관측일 35~120일 → coverage는 payload의
  `observation.coverage_ratio`로 Gemini에 명시적으로 전달한다.

## 9. Gemini 입력에 넣는 것과 넣지 않는 것

**넣는 것(모두 Python이 계산한 숫자)**

* 사람별 일간 채널 34개의 기술통계, 추세, 전·후반 차이, 주중/주말 차이
* 24시간 평균 MET 프로파일, 활동강도 비율, 수면단계 비율
* 취침/기상/수면중앙 시각의 원형 통계
* 주간 요약, 균등 간격 표본 시계열, 최대 변화일
* 관측일수·coverage·결측률

**넣지 않는 것**

* `EMAIL` / `SAMPLE_EMAIL` 등 식별자(대신 salt 기반 SHA-256 앞 16자리)
* 절대 날짜, 요일 원본 대신 주중/주말 플래그만 사용
* `DIAG_NM`, 타깃, 진단 파생 변수
* MMSE 전 항목(총점·문항·도메인·파생 포함)
* `activity_non_wear`, `sleep_period_id`, `sleep_is_longest` 등 취득 메타
* 원시 행 전체(288/1440 포인트 원본 시계열은 요약해서만 전달)

## 10. 사용자 확인이 필요한 항목

1. Gemini 모델명 기본값을 `gemini-2.5-flash`로 두었다. 무료 티어가 5 req/min
   한도에 걸리는 것은 확인했지만(재시도 대기시간 준수·동시성 축소로 대응),
   `gemini-2.5-flash-lite`로 바꿔봤더니 신규 키에는 "no longer available to new
   users" 404가 떠서 제외했다(2026-07-29 실측). 유료 티어 사용 여부나 다른
   모델을 쓰기로 정해지면 `config.yaml`의 `gemini.model`,
   `gemini.max_concurrency`, `gemini.price_per_million_*`를 조정해야 한다.
2. Gemini 캐시를 Drive의 어느 경로에 둘지(기본
   `/content/drive/MyDrive/GeminiFeaturePipeline_cache`). 캐시가 날아가면 174건을
   다시 호출해야 한다.
3. 33명 Validation을 이번 단계에서 평가하지 않는 것으로 정했다(재사용 편향 방지).
   최종 보고에 반드시 필요하다면 별도 지시가 필요하다.
