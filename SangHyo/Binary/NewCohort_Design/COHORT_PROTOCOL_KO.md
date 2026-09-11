# 새 코호트 프로토콜 — Frozen B0 이전(transfer) 검정 + 증강 모델 (Phase 3 · NOW-4)

**상태**: 초안 v1 (2026-09-05). 모집 시작 전에 IRB 문서와 함께 동결하고 외부 타임스탬프(OSF 등)로 봉인한다.
**계약**: CN vs MCI+Dem, 피험자 단위, MMSE 완전 제외(수집은 하되 봉인), 사전등록. 검정력 수치는 전부 `NewCohort_Power.ipynb` → `NewCohort_Power_result/POWER_TABLES.csv` 에서 재현된다.

## 1. 목적과 산출물
1. **Stage A — 이전 검정**: AI-Hub 174명에서 동결한 B0 채점함수(`b0_frozen_v1`)를 새 코호트에 **단 1회** 적용해 AUC_std 의 tier(R/T/F/I)를 판정한다.
2. **Stage B — 증강 모델**: B0 특징 + 나이·성별·교육 (+보행)의 사전등록 모델 하나를 새 코호트 **내부 nested CV** 로 평가하고, 웨어러블 증분 Δ1 = AUC(A1) − AUC(D) 의 경계를 추정한다.
3. 정직한 기대: 참 AUC_std ≈ 0.65 라면 R 확률 0.86, T 확률 0. **0.70 판정은 이 규모(n≈310)에서 불가능**하며(참 0.75 여도 T 확률 0.25~0.5), 그 사실을 보고서 첫 줄에 쓴다.

## 2. 표본 크기와 구성
| 항목 | 값 | 근거 |
|---|---|---|
| 목표 scoreable | **CN 190 / MCI 90 / Dem 30 (n = 310)** | SE_merged 0.033, SE_A_MCI 0.037, SE_A_Dem 0.047 (Hanley–McNeil) |
| 모집 | scoreable 가정 CN 0.90 / MCI 0.80 / Dem 0.65 → **CN 212 / MCI 113 / Dem 47 = 372명** | 12주 반지 착용 순응도; Dem 은 보호자 지원 필수 |
| scoreability 바닥 (군별) | CN ≥ 0.90, MCI ≥ 0.80, Dem ≥ 0.60 | 전체 0.83 이 계획값이므로 "전체 90%" 바닥은 두지 않는다 |
| 필요 n (80% 검정력, 양측 95% 하한) | 하한 > 0.55 @참 0.649: **266** · 하한 ≥ 0.60 @0.68: 388 · 하한 ≥ 0.70 @0.75: **846** · @0.72: 5,692 | POWER_TABLES `required_n` |
| 스크리닝 | 한국 65+ 지역사회 MCI 유병 22~24% → 400~450명 스크리닝으로 MCI 90~105 자연 확보 (KNDES/NaSDEK) | 클리닉 MCI 편중 금지 |
| 연령 | 60~85, **매칭하지 않음**. 기대 나이 격차 d 0.05(연령 제한 표본)~0.64(KBASE)를 사전등록 | 나이는 Stage B 공변량이자 Stage A 층화 secondary |

## 3. 착용·측정
- 기기: **AI-Hub 와 동일한 Oura v1 스키마 export 필수** (`activity_low` 분/일, 04:00 앵커, 1분 MET 1440점 문자열, class 288자, `sleep_bedtime_end` 키). 피험자별 반지 세대·펌웨어·앱 버전·export API 버전 기록, 연구 중 앱 버전 동결. 불일치 시 Stage A 는 "탐색"으로 격하.
- 착용: ≥ 70 조인일, 목표 84일(12주). 주 1회 동기화 확인. Stage A 채점은 **첫 35 조인일**(추정량 동일); 36~70일은 label-free test-retest(기대 Spearman ≈ 0.7)와 Stage B secondary(두 창 평균) 전용.
- 보행(선택, 권장): 9 m 코스 중앙 5 m 통상속도 2회(빠른 쪽 primary, 평균 secondary). dual-task cost 는 수집하지 않는다(Wang 2025: 원속도 대비 무가치). 모든 군에서 ≥ 90% 가용해야 A2 가 K 에 들어간다.

## 4. 변수 (steward 파일; 분석자에게는 manifest 만)
| 블록 | 변수 |
|---|---|
| 라벨 | `DIAG_NM` (CN/MCI/Dem). **primary 진단 알고리즘 1개 고정**: NIA-AA(MCI)/DSM-5(치매) 합의 진단; CDR global 기록(CDR 임계 라벨은 라벨된 민감도). 판정일은 착용 시작 ±6개월 |
| 인구통계 | `age`(년), `sex`, `edu_years`, `edu_cat`(무학/초/중/고/대) — 범주형이 한국 코호트에서 더 강함 |
| 인지검사 | MMSE, MoCA — **수집하되 봉인**(fail-closed 가드; 어떤 웨어러블 모델에도 입력 금지) |
| MCI 아형 | amnestic single/multi, non-amnestic — 라벨을 본 관찰(AI-Hub MCI 25% 고변동)에서 동기됐음을 사전등록에 공시; 기술 층화만 |
| 교란·층화 변수 | GDS-15, 약물(벤조디아제핀·항우울·수면제), BMI, 거주 형태, 이동 보조, 동반질환(파킨슨·뇌졸중·정형외과), 습관 운동, `site`, `enroll_month` |
| 추적 | 12·24개월 재평가(전환·회복) — **검정력 없음**(KLOSCAD 4년 전환 8.7%, 회복 44%; 90명 중 ~4명). 라벨 안정성 기술용 |

## 5. 포함·제외
- 포함: 60~85세, 지역사회 거주 또는 외래, 보조 없이 보행(지팡이 허용·기록), 착용 시작 ±6개월 내 합의 진단, 12주 착용·동기화 가능(보호자 대리 가능), 동의(치매는 대리 동의).
- 제외는 **반지 데이터를 해석 불능으로 만드는 것만**: 손가락 조건(착용 불가), 교대근무, 비보행, AI-Hub 라이프로그 참여 이력. **MMSE 점수 기준 제외 금지**(라벨 도구이며 Dem 꼬리를 절단). 우울·시설 거주·정형외과 제한·약물은 제외 대신 **기록·층화**.
- 순응도 기준은 비인지 항목(보호자 가용성, ADL/IADL, CDR 단계)으로.

## 6. 균형·교란 통제
- |Spearman(enroll_month, y)| < 0.15; 모든 사이트에서 CN 과 양성 동시 모집(클리닉-MCI vs 지역사회-CN 금지); 분기별 전 군 모집(AI-Hub 계절 −17% 드리프트 교훈); 위반 시 월·사이트 층화 AUC 가 co-primary 기술자.
- 모집 순서·시작일·관측기간이 한 축이 되지 않도록 무작위 시작일 배정.

## 7. 데이터 흐름과 1회성
1. 웨어러블 export(라벨·공변량 없음) → 분석자. 라벨·공변량 → steward. 중간 AUC 없음, 대시보드는 웨어러블만.
2. `B0_Score_NewCohort`(봉인 채점기)로 **1회 scoring** → `scores_<sha8>.csv` + SCORING_RECORD (SCHEMA_CONTRACT 통과한 첫 scoring 이 평가 대상).
3. `B0_Evaluate_Once` 를 **steward 가 실행** → tier 판정 → ledger(서명 2인). 2번째 실 평가 = 1회성 주장 무효.
4. 그 다음에만 Stage B(`Binary_NewCohort_Augmented_Nested`) 실행(ledger n_real_evaluations == 1 조건).

## 8. 분석 계획 요약
| Stage | 통계 | 판정 |
|---|---|---|
| A | AUC_std (subject bootstrap 5000), merged·A_MCI·A_Dem 기술, MNAR 경계, 연령층·나이 잔차화 secondary | R/T/F/I (사전등록 §3, `B0_Evaluate_Once/PREREGISTRATION_KO.md`) |
| B | K=2(3): D={age, sex, edu_years}, A1=B0+D, (A2=A1+gait). 새 코호트 내부 nested CV(outer 5×20, inner 4×2, C 그리드 동결), Δ1·Δ2 paired bootstrap, permutation 2000회 선택 재생, Holm | falsifier F-A~F-H(구간 규칙·오경보율 명시). **Δ1 기대: 중앙 시나리오 +0.005~+0.03, MDE 0.07~0.09 → F-A 검정력 13~35%** → F-A 는 확인용이 아니라 Δ1 의 경계 추정 |

## 9. 일정·위험
반지 ~100개 회전·12주 → 372명 ≈ 12~14개월 + IRB 2~4개월 + 판정 지연 · Dem 30~40% 탈락(보호자 지원 예산) · 펌웨어 변경이 모집순서와 상관 · 라벨 불안정(MCI 회복 44%)이 A_MCI 상한 · B0 가 부분적으로 나이 프록시일 가능성(NHATS: 활동 변동성 |r| ≈ 0.4 with age) → 연령층 secondary 와 F-C 가 방어선.
