# PREREGISTRATION — B0_Score_NewCohort (봉인 채점기, Phase 3 · NOW-2)

**동결 시각**: 2026-09-05 (KST). **동결 코드**: `B0_Score_NewCohort.ipynb` 코드 셀 6개의 연결 sha256 = `e6251e44213e5d0bbbe91c2a05f0986662f8ad9ea233f4c18098e5924322fecd`
**아티팩트 핀**: `b0_frozen_v1.json` sha256 = `efac45e3d291966757713041dcae2c2a88d53dfe962772fd6e5a7aa98731e12d` (노트북에 하드코딩; 불일치 시 실행 거부)

## 1. 역할
새 코호트의 원시 Oura v1 export(activity/sleep CSV) + steward 의 **label-free manifest**(sid 목록)를 입력으로, 아티팩트 JSON 의 수치만으로 피험자별 점수를 낸다. sklearn 을 쓰지 않는다. **라벨·공변량을 읽지 않는다.**

## 2. 고정 규칙
| 규칙 | 내용 |
|---|---|
| 파일 발견 | 루트 아래 `*activity*.csv`, `*sleep*.csv` 재귀 탐색. 경로에 `LabelingData`, `3.CognitiveFunction`, `label`, `mmse`, `moca`, `cdr` 가 있으면 열지 않음 |
| INPUT_SCHEMA_GUARD | 읽는 컬럼 allowlist: activity 7개(`EMAIL`, `activity_day_start`, `activity_low`, `activity_medium`, `activity_non_wear`, `activity_total`, MET 문자열), sleep 5개. 헤더의 공변량/라벨 토큰 컬럼(age/sex/gender/edu/gait/moca/cdr/mmse/diag/label/dx)은 **존재만 기록**하고 읽지 않음 |
| manifest | `sid` 단일 컬럼. 중복 행 수 기록; 데이터 없는 sid → `no_data`; manifest 밖 sid 는 무시(수 기록) |
| 행 스키마 | 타임스탬프 NaT, `activity_day_start` 시각 ≠ 04:00, MET 토큰 ≠ 1440·비유한·[0, 25] 밖, `activity_low` ∉ [0, 1440] → 해당 피험자 전체를 `schema_fail` (day set 을 바꾸지 않기 위해) |
| 중복 | sleep (sid, date): duration 최장 → 시작 이른 순; activity (sid, date): `activity_day_start` 이른 → `activity_total` 큰 순. AI-Hub 에서 현행 keep-first 와 동일함을 NOW-1 CT9b 로 증명 |
| 창·특징 | (sid, date) 정렬 후 첫 35 조인일. <35 → `insufficient_days`. 창 안 `activity_low` 결측 → `feature_unavailable`(대치 금지). `low_SD`(ddof 0)·`low_MED`. 게이트 없음 |
| 채점 | 아티팩트의 median·mean·scale·coef·intercept 로 softmax → score = 1 − P(CN). 로드 시 test vectors 8개 재현(1e-12) 필수 |
| QC/drift | PSI·KS(low_SD, low_MED, 비착용 평균, gap_days) vs 174명 참조값 — **보고 전용**, 점수 불변, 정지 없음. 단위는 `NewCohort_Design/SCHEMA_CONTRACT.json` 으로 고정 |
| 출력 | `scores_<sha8>.csv`(sid, low_SD, low_MED, n_joined_days, gap_days, score, reason) + `QC_REPORT.json` + `SCORING_RECORD.json`(입력 파일 sha256, manifest sha256, 아티팩트 sha256, 코드 해시, 열람한 라벨 파일 수 = 0) |
| 1회성 | scoring 은 여러 번 가능하나 매번 SCORING_RECORD 가 남는다. 평가기(NOW-3)는 SCHEMA_CONTRACT 를 통과한 **첫 scoring** 만 받는다 |

## 3. 계약 테스트 결과 (2026-09-05)
| # | 내용 | 결과 |
|---|---|---|
| CT5 | AI-Hub 루트 + `manifest_aihub174.csv`(SourceData 만으로 생성) → NOW-1 의 174명 in-sample 점수와 최대 |차| 8.3e-17, 174/174 채점, 라벨 파일 열람 0 | **통과** |
| CT6 | `check_label_blind.py`: 코드 셀에 라벨/MMSE 토큰 없음(차단 목록 정의 1회 제외) | **통과** |
| CT7 | 합성 dirty 코호트(<35일 1, activity_low NaN 1, MET 1400토큰 1, 철회 sid 1, manifest 중복 1, manifest 밖 sid 1, activity/sleep 중복 행 각 1) → `insufficient_days` 1, `feature_unavailable` 1, `schema_fail` 1, `no_data` 1, 중복 1, 무시 1, 중복 해소 1/1 | **통과** |
| 합성 clean | 120명 전원 채점; drift 플래그가 켜지지만 점수 불변(report-only 확인) | 통과 |

## 4. 금지
아티팩트 교체·핀 수정·창 규칙·임계·거부 코드 변경, 결측 대치, drift 에 따른 점수 수정/정지, 라벨·공변량 파일 열람, 평가 전 점수 파일 재생성(재생성 시 새 SCORING_RECORD 가 남고 첫 scoring 규칙이 적용된다).
