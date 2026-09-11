# PREREGISTRATION — B0_Frozen_Scorer (Phase 3 · NOW-1)

**동결 시각**: 2026-09-05 (KST), 실행 전
**동결 코드**: `B0_Frozen_Scorer.ipynb` 코드 셀 8개의 연결 sha256 = `953e8794525b5fbdc70d1c0f144cb57340bdbc6e291b127ce0ef98baf00ff338` (v2; v1 = `845a5c08…`, §7)
(cell 8 이 실행 시 같은 방식으로 재계산해 `b0_frozen_v1.json.meta.notebook_code_sha256` 과 `FINAL_REPORT.json` 에 기록한다.)
**계약**: CN(0) vs MCI+Dem(1), 174명, MMSE 완전 제외. 이 노트북은 성능을 주장하지 않는다.

## 1. 목적
nested CV 로 검증된 B0 = {`low_SD`, `low_MED`} 를 **하나의 고정 함수**로 내보낸다. 새 코호트 1회 평가(`B0_Evaluate_Once`)의 유일한 채점기다.

## 2. 고정되는 것 (실행 전 확정, 결과와 무관하게 불변)
| 항목 | 값 |
|---|---|
| 특징 정의 | run-2/run-4 cell 2·3·5 와 동일: 조인 프레임 첫 35행, `activity_low` 분의 SD(ddof 0)·중앙값, 착용 게이트 없음 |
| 적합 데이터 | 174명 전체(train+val 풀링), 3-class 라벨 |
| 파이프라인 | SimpleImputer(median) → StandardScaler → LogisticRegression(multinomial, L2, lbfgs, max_iter 5000), classes [0,1,2], score = 1 − P(CN) |
| **C** | **0.1** — B0 arm 의 fold-modal 선택(93/100, run-2·run-3·run-4 동일). 174명 inner-CV(4×2, tol 0.005, 최소 C, seed 7000) 로 다시 고른 값은 **기록만** 한다 |
| 직렬화 | `b0_frozen_v1.json`(정본: 수치 전부 + 특징 코드 원문·sha256 + test vectors + 참조 분포 + 대비 부호) + `b0_frozen_v1.joblib`(편의) |

## 3. 계약 테스트
| # | gate | 통과 기준 |
|---|---|---|
| CT1 | ✓ | 재구축 특징 지문 == `3bc37d120081437d` and run-4 `feature_matrix.csv` 의 low_SD/low_MED 와 최대 |차| ≤ 1e-12 (CSV 텍스트 round-trip 허용; 정확성은 지문 + CT3 가 담당) |
| CT2 | ✓ | 90명 부분집합 재구축이 전 컬럼 비트 동일 |
| CT3 | ✓ | 같은 엔진 nested B0: subject-mean 0.6490776490776491, per-repeat 20개 1e-9 이내, 분해 항등식 1e-9 |
| CT4a | ✓ | 순수 numpy 채점기와 sklearn 점수 최대 |차| ≤ 1e-12 (174명 + test vectors 8) |
| CT4b | ✓ | 별도 프로세스에서 joblib 재로드 → 174명 점수 최대 |차| ≤ 1e-12 |
| CT9a | ✓ | 엄격 파서(끝 `/` 1개 제거·split·정확히 1440·전부 유한) 와 기존 `np.fromstring` 경로가 12,150일 비트 동일; 모든 날 정확히 1440 |
| CT9b | ✓ | 명시적 tie-break 키 (sid, date, ts, activity_total desc / sleep: duration desc, ts_start) 가 현행 규칙과 동일한 12,150 day set·값 |
| CT8 | report-only | 대비 부호 기록: low_SD 의 (MCI−CN, Dem−CN) 두 값과 전역 단조 여부; low_MED 는 부호가 다를 것으로 예상(단조 아님). **gate 아님** |
| CT10 | report-only | C ∈ {0.01, 0.03, 1.0} 점수와의 Spearman. **gate 아님** |

**규칙**: gate 하나라도 실패하면 아티팩트는 무효이며, 원인은 코드 결함(재구축·직렬화)에서만 찾는다. **어떤 CT 결과도 C·그리드·특징 정의·seed·점수 부호를 바꾸지 않는다.** CT8/CT10 은 기록이며 조치 대상이 아니다.

## 4. 산출물
폴더 루트: `b0_frozen_v1.json`, `b0_frozen_v1.joblib`. `B0_Frozen_Scorer_result/<UTC_RUN_ID>/`: `FINAL_REPORT.json`, `CONTRACT_TESTS.json`, `b0_scores_174_insample.csv`(경고: in-sample, 성능 아님).
git: `git add -f` 후 tag `b0-frozen-v1`.

## 5. 사전 기대치 (기록용)
C 감사는 0.1 을 고를 것으로 예상(계획 단계 read-only 계산: {0.003: 0.574, 0.01: 0.585, 0.03: 0.603, 0.1: 0.626}). in-sample merged ≈ 0.68 (낙관; 보고 금지). 대비 부호: low_SD (−, −), low_MED (+, −).

## 6. 공시
계획 단계에서 174명 전체 적합의 계수·C 감사값·in-sample AUC 를 read-only 로 미리 계산해 보았다(§5). 이 노트북은 그것을 **아티팩트로 고정·검증**하는 run 이며, 라벨을 새로 보는 실험이 아니다. 새 코호트 라벨은 이 폴더에서 절대 읽지 않는다.

## 7. 개정 이력
| 버전 | 시점 | 내용 |
|---|---|---|
| v1 | 2026-09-05 00:24 KST | 최초 동결 (코드 sha256 845a5c08…) |
| v2 | 2026-09-05 00:27 KST, 1차 실행 직후 | **v1 실행에서 CT1 만 실패**: run-4 CSV 대비 low_SD 최대 |차| 1.42e-14 (CSV 텍스트 round-trip), 지문·CT3(비트 동일 nested 재현)·CT9·CT4 는 전부 통과. CSV 비교 허용오차를 0.0 → 1e-12 로 정정(계약 테스트 결함이지 특징 드리프트가 아님). 참조 분포에 174명 특징 원값(label-free)을 추가해 KS 검정이 가능하게 함. **C·특징·seed·부호 변경 없음.** 실패한 v1 run 폴더는 삭제하지 않고 남긴다 |
