# NewCohort_Design — 새 코호트 설계 패키지 (Phase 3 · NOW-4/5)

| 파일 | 내용 |
|---|---|
| `NewCohort_Power.ipynb` → `NewCohort_Power_result/POWER_TABLES.csv`, `power_curves.png` | Hanley–McNeil(AI-Hub 값 assert) · 80% 검정력 필요 n · tier 작동 특성(n=310 binormal MC) · 구성/탈락 스윕 · 증강 모델 시나리오(ρ(나이,교육) 음수 포함) · Δ MDE |
| `COHORT_PROTOCOL_KO.md` | 목적·표본·착용·변수·포함/제외·균형·데이터 흐름·분석 계획·위험 |
| `SCHEMA_CONTRACT.json` | 봉인 채점기가 강제하는 입력 스키마(단위는 벤더 문서 기준) |
| `QC_RULES_KO.md` | label-free QC 12개 (거부 코드 외에는 report-only) |
| `Portability_Checks.ipynb` | 174명 label-free 이식성 점검: sleep dedup 규칙 vs `sleep_is_longest`, 첫 35 조인일 vs 달력 35일, (파서·tie-break 동일성은 NOW-1 CT9a/b) |

핵심 수치(POWER_TABLES): 참 AUC_std 0.649 에서 n=310 의 R 확률 0.86, T 확률 0; 하한 ≥ 0.70 을 80% 검정력으로 보려면 참 0.75 에서 n ≈ 846. 증강 Δ1 기대 +0.005~+0.03(중앙), MDE 0.07~0.09.
