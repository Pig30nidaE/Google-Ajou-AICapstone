# PREREGISTRATION — B0_Evaluate_Once (봉인 1회 평가, Phase 3 · NOW-3)

**동결 시각**: 2026-09-05 (KST), 새 코호트 모집 전. **동결 코드**: `B0_Evaluate_Once.ipynb` 코드 셀 5개의 연결 sha256 = `416706604f51416ef977d5303014066ce3a1ebe75996220bb15037add53a8a28`
**채점기**: `B0_Score_NewCohort` (코드 `e6251e44…`), 아티팩트 `b0_frozen_v1.json` (`efac45e3…`). 이 세 해시가 보고서에 함께 기록된다.
**외부 타임스탬프 권장**: 이 파일·아티팩트 sha256·tier 정의를 모집 시작 전에 OSF 등록 또는 steward 에게 서명된 날짜 메시지로 봉인한다.

## 1. 입력과 관리 주체
| 파일 | 보유 | 내용 |
|---|---|---|
| wearable export (activity/sleep CSV) | 분석자 | 라벨·공변량 없음 |
| manifest.csv | steward → 분석자 | sid 목록만 |
| `scores_<sha8>.csv` + `SCORING_RECORD.json` + `QC_REPORT.json` | 분석자 → steward | 봉인 채점기 산출물 (SCHEMA_CONTRACT 통과한 **첫** scoring) |
| 라벨/공변량 파일 (`sid, DIAG_NM, age, sex, edu_years, edu_cat, gait_speed, site, enroll_month, mci_subtype`) | **steward 만** | 이 노트북이 단 한 번 연다. 가능하면 **제3자(steward)가 실행** |

## 2. Endpoint (사전등록)
- **Primary**: **AUC_std** = 0.8095·A_MCI + 0.1905·A_Dem (AI-Hub 51/12 양성 구성으로 재가중; 유병 구성에 불변). subject bootstrap 5000 (seed 20260903) 양측 95% CI. 분해 항등식은 재표집된 n 으로 검사.
- 기술용: merged AUC(구성 의존; AI-Hub sub-AUC 로 기대되는 값을 함께 표기), A_MCI, A_Dem, Hanley–McNeil SE, 단측 95% 하한, PR-AUC.
- Secondary (판정 미사용): A_MCI ≥ 0.58 & 하한 > 0.50; 연령층(60s/70s/80s) AUC; 나이 잔차화 점수 AUC 와 Spearman(score, age); MCI 아형별 A_MCI(기술, **라벨을 본 관찰에서 동기됐음을 공시**); **MNAR 경계**(unscoreable 양성/음성에 최소·최대 점수 대입한 merged 상하한) + tipping point; 균형 검사(|Spearman(enroll_month, y)| < 0.15, 사이트×군).

## 3. Tier (AUC_std 양측 95% CI, 기계적)
| tier | 규칙 | n=310 (190/90/30) 작동 특성 (binormal MC, `NewCohort_Power`) |
|---|---|---|
| **T 목표** | 하한 ≥ 0.70 | 참 0.649 → 0.000 · 0.69 → 0.014 · 0.71 → 0.050 · 0.74 → 0.25 · 0.76 → 0.50 |
| **R 복제** | 하한 > 0.55 (T 제외) | 참 0.649 → **0.86** · 0.612 → 0.48 · 0.56 → 0.05 |
| **F 비이전** | 상한 < 0.60 또는 점 < 0.55 | 참 0.50 → 0.93 · 0.56 → 0.40 |
| **I 미결** | 그 외 | 참 0.56 → 0.55 · 0.612 → 0.49 · 0.649 → 0.14 |
T ⊂ R. **참 AUC_std 가 B0 의 0.649 라면 목표 tier 확률은 0 이다** — 이 실험의 정직한 산출물은 R 여부이지 0.70 판정이 아니다. 참 0.74 이상이어야 T 가 현실적이다(80% 검정력엔 n ≈ 850).

## 4. 1회성 장치
1. `EVALUATION_LEDGER.json`: 실 평가가 1건이라도 있거나 같은 라벨 sha256 이 있으면 **실행 거부** (합성 배선 테스트는 `B0EV_SYNTHETIC=1` + 별도 ledger 경로에서만).
2. 보고서에 점수 파일·라벨 파일·아티팩트·채점기·평가기 코드의 sha256 을 전부 기록. 평가기는 자기 실행 코드 해시를 쓴다.
3. ledger 증가에 서명자 2인(analyst, steward) 필드. 2번째 실 평가는 1회성 주장을 무효화한다(기술적으로 막을 수는 없고 가시화한다).
4. 결과가 무엇이든 채점기·특징·C·부호는 바꾸지 않는다. 증강 단계(NOW-6) 결과는 해석 절만 수정 가능, tier 불변.

## 5. 비교 규칙
AI-Hub nested 0.6491 [0.562, 0.731] 은 **절차**(139명 적합 100개 모델)의 기대치이고, 이 값은 174명 적합 **함수 하나**의 값이다. 두 SE 중 큰 값(≥ 0.044)보다 작은 차이에서는 결론을 내리지 않는다. AI-Hub 포함/제외 기준을 확보하지 못했으므로 이 검정은 "복제"가 아니라 **"이전(transfer) 검정"** 으로 명명한다.

## 6. 배선 테스트 (2026-09-05, 합성 dirty 코호트)
CONSORT·군별 scoreable·AUC_std/merged/A_MCI/A_Dem CI·항등식(1e-16)·tier·secondary·MNAR·균형·보고서·그림·ledger 갱신 전부 동작(0 오류). ledger 에 실 평가 1건을 넣고 재실행 → cell 1 에서 거부(AssertionError) 확인. 합성 숫자는 보고하지 않는다.
