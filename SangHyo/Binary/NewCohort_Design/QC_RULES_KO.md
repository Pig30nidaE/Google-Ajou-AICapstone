# QC 규칙 — 새 코호트 웨어러블 export (label-free, report-only)

근거: AI-Hub 174명 EDA(2026-09-04). 봉인 채점기가 자동 적용하며, 점수를 바꾸는 규칙은 하나도 없다(거부 코드만 있다).

| # | 규칙 | 참조값 (174명) | 위반 시 |
|---|---|---|---|
| Q1 | MET 문자열: 끝 `/` 1개 제거 → 정확히 1440 토큰, 전부 유한, 0 ≤ MET ≤ 25 | 12,150/12,150 일 정확히 1440 | 피험자 `schema_fail` |
| Q2 | `activity_day_start` 시각 = 04:00, 타임스탬프 NaT 없음 | 100% | `schema_fail` |
| Q3 | MET 0.0 sentinel: > 60분 연속 0.0 인 날 수 기록 (Oura 는 non_wear 로 세지 않음) | 24/12,150 일 (첫 35일 창 안 14일) | 기록만 (B0 는 게이트 없음) |
| Q4 | 비착용 = MET < 0.5; 일별 착용률·`activity_non_wear` 평균 기록 | 착용률 median 0.999, p1 0.811; nonwear 평균 median 38.9분 | 기록만; PSI > 0.25 → domain-shift 플래그 |
| Q5 | 중복: sleep (sid,date) 최장 duration; activity (sid,date) 이른 day_start·큰 total. 해소 건수 기록 | sleep 12건, activity 0건 | 기록만 |
| Q6 | 창 = 첫 35 조인일; gap_days = span − 35 기록, > 27(p99) 플래그 | median 3, p90 12.7, max 34 | 기록만 |
| Q7 | manifest: sid 단일 컬럼, 중복·no_data·manifest 밖 sid 수 기록 | — | `no_data` 거부 / 무시 |
| Q8 | `activity_low` 결측(창 안) | 0건 | `feature_unavailable` (대치 금지) |
| Q9 | Drift: PSI·KS of low_SD, low_MED, nonwear_mean, gap_days vs 참조 분위수/원값 | PSI 0.000 (자기 참조) | 기록만; 단위는 SCHEMA_CONTRACT 로 고정 |
| Q10 | class_5min 은 읽지 않는다(B0 미사용; 1.4% 절단 알려짐) | — | — |
| Q11 | 공변량/라벨 토큰 컬럼이 export 에 섞여 있으면 존재만 기록하고 읽지 않음 | — | 기록 |
| Q12 | 기기 세대·펌웨어·앱 버전·export API 버전을 피험자별로 별도 파일에 기록 (steward) | AI-Hub: 미기록(2020, Gen2 추정) | 보고서 필수 항목 |
