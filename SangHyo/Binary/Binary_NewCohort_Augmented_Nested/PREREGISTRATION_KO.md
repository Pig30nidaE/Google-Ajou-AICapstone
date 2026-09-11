# PREREGISTRATION — Binary_NewCohort_Augmented_Nested (Stage B: B0 + 인구통계 [+보행], Phase 3 · NOW-6)

**동결 시각**: 2026-09-05 (KST), 새 코호트 모집 전. **동결 코드**: `Binary_NewCohort_Augmented_Nested.ipynb` 코드 셀 6개의 연결 sha256 = `50ff66b26404b94e50a473402b069fd0c135edb93add7715c510f164de1d98c5`
**실행 조건**: `B0_Evaluate_Once/EVALUATION_LEDGER.json` 의 `n_real_evaluations == 1` (Stage A 종료 후). 합성 배선 테스트는 `B0AU_SYNTHETIC=1` 로만.
**계약**: CN vs MCI+Dem, 피험자 단위 nested CV(새 코호트 내부), MMSE/MoCA/CDR 입력 금지(fail-closed 가드), 사전등록. Stage A 의 tier 는 이 결과로 바뀌지 않는다.

## 1. 가설
H1: 동결 B0 특징 {`low_SD`, `low_MED`} 는 나이·성별·교육 위에 out-of-sample 증분 Δ1 을 준다. H0: Δ1 ≤ 0 (웨어러블은 인구통계와 중복).
부가: 보행속도(A2)가 A1 위에 증분 Δ2 를 주는가 — 모든 진단군에서 gait 가용 ≥ 90% 일 때만 K 에 포함.

## 2. Arm (K = 2 또는 3, 실행 전 확정)
| arm | 특징 |
|---|---|
| **D** | age(년), sex(F=1), edu_years |
| **A1** | low_SD, low_MED (봉인 채점기 출력의 특징 열) + D |
| A2 | A1 + gait_speed (조건부) |
진단(K 미포함, 승격 금지): A1′ = frozen_score + D · A1 에서 low_SD 를 fold 내부 age 잔차화 · site/enroll_month 층화 A1 AUC.

## 3. 프로토콜 (run-4 엔진 동일)
median imputer → StandardScaler → multinomial L2 LR(lbfgs, max_iter 5000), score 1 − P(CN); outer StratifiedKFold(5, shuffle, 100+r)×20 on 3-class; inner 4×2 (7000+37r+f+rr), C ∈ {0.003, 0.01, 0.03, 0.1}, tol 0.005 최소 C; subject bootstrap 5000; paired Δ bootstrap 5000; permutation 2000회(축소: outer 5×2, inner 4×1, C ∈ {0.01, 0.1}) 로 **K arm 전부·inner 선택 재생**; Holm over {Δ1, Δ2}.

## 4. Endpoint · Falsifier (구간 규칙, 오경보율 명시)
- Primary **Δ1 = AUC(A1) − AUC(D)** (merged; MCI/Dem 하위도 보고). Secondary Δ2. 절대 AUC·AUC_std 병기.
| # | 규칙 | 오경보/비고 |
|---|---|---|
| F-A | Δ1 CI 하한 > 0 and p_Δ1(Holm) < 0.05 | **검정력 13~35%** (POWER_TABLES delta_mde). 실패는 Δ1 의 경계 추정이지 웨어러블 반증이 아님 |
| F-B | A1 전체 적합에서 low_SD 의 두 대비(MCI−CN, Dem−CN) 모두 음수 | 기전 부호 |
| F-C | 나이 잔차화 arm 의 Δ_MCI(vs D) CI 하한 > −0.02 | 웨어러블 신호가 나이로 환원되는지 |
| F-D | Δ_Dem(A1−D) CI ∋ 0 또는 > 0 | 오경보 ≈ 8% |
| F-F | (K=3) Δ2 CI 하한 > 0 and p_Δ2(Holm) < 0.05 | 사전 기대: 실패(지역사회 보정 gait d 0.14) |
| F-G | \|Spearman(enroll_month, y)\| < 0.15 and 사이트별 A1 AUC 전부 > 0.55 | 위반 시 headline 을 'confounded' 로 표기 |
| F-H | \|AUC(A1′) − AUC(A1)\| < 0.03 | 오경보 ≈ 4%; 재학습된 웨어러블 가중치가 B0 와 다른지 |
- A1 절대 tier(보고): success = CI 하한 ≥ 0.62 & 점 ≥ 0.68 · target = CI 하한 ≥ 0.70 · failure = 점 < 0.62.

## 5. 사전 기대치 (POWER_TABLES, 2026-09-05)
| 시나리오 | D | A1 | Δ1 |
|---|---|---|---|
| 비관(연령 매칭, d_age 0.05) | 0.60 | 0.63 | +0.03 |
| 중앙(d_age 0.4, d_edu 0.26, ρ(age,edu) −0.4, ρ(wear,age) 0.3) | 0.68 | 0.68 | **+0.005** |
| 중앙, 상관 0 | 0.64 | 0.68 | +0.03 |
| 낙관(KBASE d_age 0.64, 범주 교육 0.7) | 0.81 | 0.81 | 0.00 |
CN-vs-MCI 값. **인구통계가 강할수록 웨어러블 증분은 작아진다** — 높은 headline 과 F-A 실패는 같은 결과다. MDE(n=310) 0.07~0.09.

## 6. 금지·공시
인지검사 토큰(mmse/moca/cdr/diag/label) 입력 금지 · arm 추가/교체 · 결과 후 정의 변경 · 진단 arm 승격 · threshold · Stage A tier 수정. 각주: 이 실험은 1회 평가와 같은 피험자·라벨을 재사용하며 B0 의 두 특징을 포함한다.
공시: MCI 아형 층화(수집 항목)는 AI-Hub 에서 라벨을 본 관찰(MCI 25% 고변동)에서 동기됐다.

## 7. 배선 테스트
2026-09-05 합성 clean 코호트(120명, QUICK 2 repeats, permutation 20)로 전 셀 0 오류: K 결정(gait 92% → K=2), Δ1 paired bootstrap, 진단 arm, 층화, permutation 선택 재생, Holm, falsifier 6종, 절대 tier, 저장. 합성 숫자는 보고하지 않는다.
