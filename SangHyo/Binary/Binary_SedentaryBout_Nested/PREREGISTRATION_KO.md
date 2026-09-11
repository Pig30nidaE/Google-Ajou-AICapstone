# PREREGISTRATION — Experiment 2: B0 vs B0 + `longest_w_SD` (깨어있는 창의 최장 정좌 bout 일간 SD)

**실험명**: `Binary_SedentaryBout_Nested`
**동결 시각**: 2026-09-04 13:15 KST, 실행 전 (합성 데이터 배선 테스트 통과 직후, 실데이터 라벨 기반 값은 어떤 것도 계산하지 않은 상태)
**동결 코드**: `Binary_SedentaryBout_Nested.ipynb` 코드 셀 13개의 연결 sha256 =
`2f0cb1a7022098c054cc9454f320c444c5b581bc34f13094b2ff4c3552836e6c`
(노트북 cell 13 이 실행 시 같은 방식으로 재계산해 `FINAL_REPORT.json.prereg_code_sha256` 에 기록한다. 불일치 = 코드가 바뀐 것.)
**계약**: CN(0) vs MCI+Dem(1), 174명, 피험자 단위 분할, nested CV, MMSE 완전 제외. 이 셋은 변경 불가.
**노출 상태**: **confirmatory-record** (§11). blind 가 아니다.

---

## 1. 가설

- H1: 인지저하에서 하루 안의 정좌 분절 구조가 날마다 더 같아진다. 구체적으로 깨어있는 창 안의 **최장 정좌 bout(분)의 35일 SD**(`longest_w_SD`)가
  동결 D블록 `{low_SD, low_MED}` 위에 out-of-sample 증분(Δ)을 준다. (`low_SD` 가설을 분 총량에서 bout 구조로 확장한 **마지막** 기전 가설)
- H0: 증분 없음(또는 희석으로 음수).
- 부가 질문(진단으로만 답한다): 증분이 있다면 (i) 활동 수준(`low_MED`)·비착용·유효일수와 독립인가, (ii) 가족 2순위(평균 bout 길이 SD)에도 있는가.

## 2. 후보 정의 (실행 전 동결, 노트북 cell 3–5 그대로)

```
worn_d(t)    = MET_d(t) ≥ 0.5                       # Oura 비착용 0.1, 착용 바닥 0.9; 0.0 no-data sentinel 도 잡힌다
sed_d(t)     = worn_d(t) and MET_d(t) < 1.5          # 비착용 분은 False → 정좌 run 이 끊긴다 (STRICT3 §8 non-wear 규칙)
onset_d      = 착용분만의 30분 rolling MET 평균 ≥ 1.5 가 처음 성립하는 창의 시작 index (04:00 기준 분; run-2/Exp1 과 동일 규칙)
offset_d     = 그 조건이 마지막으로 성립하는 창의 끝(배타) index
valid_d      = 착용률_d ≥ 0.8 and (offset_d − onset_d) ≥ 240
longest_w_d  = [onset_d, offset_d) 안 sed 연속 run 길이의 최댓값 (분)      (valid_d 인 날만)
longest_w_SD = 첫 35 조인일 중 valid 일들의 longest_w 의 표준편차 (ddof 0, ≥ 3일)
```
`activity_class_5min` 미사용. 24h 로 자르면 최장 bout 이 47% 의 날에서 04:00 경계(수면)에 닿으므로 깨어있는 창이 필수다.
가족 2순위(F3 전용): `meanbout_w_SD` = 같은 창 안 정좌 run 길이의 **평균**의 35일 SD.

### 실행 전 label-free 자격 (2026-09-04 실측, EDA)
| 항목 | `longest_w_SD` | `meanbout_w_SD` |
|---|---|---|
| split-half 신뢰도 (Spearman-Brown) | 0.598 (0.75) | 0.736 (0.85) |
| ρ(low_SD) / ρ(low_MED) | 0.22 / 0.17 | 0.06 / −0.35 |
| ρ(모집월) / ρ(유효일수) | 0.08 / −0.00 | 0.02 / −0.12 |
| ICC(1) of 일별 값 | 0.23 | 0.42 |
headline 은 사용자 우선순위 1번이자 D블록과 가장 직교한 `longest_w_SD` 로 **사용자가 확정**(2026-09-04). `meanbout_w_SD` 는 신뢰도가 더 높지만 수준 축과 결합(−0.35)이 있어 진단으로만 쓴다.

## 3. Arm (K=2, headline 경쟁은 이 둘뿐)

| arm | 특징 | 비고 |
|---|---|---|
| B0 | `low_SD`, `low_MED` | run-2 B0 와 비트 동일. 노트북이 subject-mean 0.6490776490776491 과 per-repeat 20개를 1e-9 내로 assert |
| P | `low_SD`, `low_MED`, `longest_w_SD` | 동일 엔진·fold·seed·그리드 |

## 4. 프로토콜 (run-2 / Experiment 1 계약 재사용, 변경 없음)

```
데이터   : train+val 풀링, activity ⋈ sleep inner join on (sid, sleep_bedtime_end 날짜), 중복 규칙 run-2 동일 → 12,150 행
창       : 조인 프레임 첫 35행 (전원 정확히 35일). B0 는 35일 전부(게이트 없음), 후보는 valid 일만
전처리   : SimpleImputer(median) → StandardScaler → 3-class multinomial L2 LR(lbfgs, max_iter 5000), score = 1 − P(CN)
Outer    : StratifiedKFold(5, shuffle, random_state=100+r) on 3-class, r = 0..19
Inner    : StratifiedKFold(4, shuffle, random_state=7000+37r+f+rr), rr = 0,1; C ∈ {0.003, 0.01, 0.03, 0.1}, tol 0.005 최소 C
집계     : per-repeat OOF AUC(repeat 평균±SD) + 20 repeat 점수 피험자 평균 → subject-mean AUC (헤드라인)
CI       : subject bootstrap 5000 (seed 20260903), paired Δ bootstrap 5000 (seed 20260904)
Permutation: 축소 프로토콜(outer 5×2, inner 4×1, C∈{0.01, 0.1}), 3-class 라벨 피험자 단위 셔플, 2000회; 후보 열 셔플 1000회
```

## 5. Endpoint

- **Primary**: Δ_merged = AUC_P − AUC_B0 (동일 fold, paired subject bootstrap, 95% CI)
- Secondary: Δ_MCI, Δ_Dem, 절대 AUC 3종 + CI + Hanley–McNeil SE, repeat 평균±SD, fold AUC, 선택 C 분포, failed_folds(= 0 이어야 headline 유효), per-repeat Δ 부호, P–B0 OOF 점수 상관, split-half 신뢰도
- Permutation: **p_Δ** = P(Δ_null ≥ Δ_obs), Δ_null = 같은 순열에서 AUC_P − AUC_B0 (헤드라인 p) · p_primary · p_max (B0 포함 max 라 후보 증거 아님)
- co-statistic (판정 미사용): **p_col** = 후보 열만 피험자 간 셔플(라벨·D블록 고정, 1000회)한 Δ null 대비 관측 Δ

## 6. 수용 기준 (Experiment 1 과 동일, 변경 없음)

| tier | 기준 | 후속 |
|---|---|---|
| **T1 유의한 증분** | Δ_merged 95% CI 하한 > 0 **and** p_Δ < 0.05 | 동일 정의로 새 seed 군(random_state 200+r) 복제 1회 후 종료 |
| **T2 신호 부합(복제 필요)** | Δ_merged > 0 and Δ_MCI > 0 and Δ_Dem CI ∋ 0 and p_Δ < 0.10 and per-repeat Δ>0 ≥ 15/20 | 복제 1회. 특징 추가 금지. baseline 은 B0 유지 |
| **T3 기각** | 그 외 | 후보 종결 → **stop rule (§10)** |
| 참고 | B: subject-mean ≥ 0.70 / C: max > null p95 / D: CI 하한 ≥ 0.60 | run-2 형식 유지, 판정에 미사용 |

## 7. 기전·가족 진단 (main 과 동일 20 repeats·동일 fold, K 미포함, headline 승격 금지)

| # | arm | 가르는 것 |
|---|---|---|
| F1 | `longest_w_SD` 단독 | 기술 통계 전용. 단일 열 3-class MNL 은 비단조 특징에서 U자 붕괴하므로(STRICT3 memo) 어떤 판정에도 쓰지 않는다 |
| F2 | P, 후보 열만 fold 내부에서 `low_MED` 에 OLS 잔차화 (D블록 불변 → 기준 = B0ref) | 활동 수준을 빼도 증분이 남는가 |
| F3 | B0 + `meanbout_w_SD` | 가족 2순위. **후보 교체용이 아니라 "가족이 죽었는가" 판정용** |
| F4 | P, 후보 ⟂ `cf_nonwear_mean` (fold 내부) | 비착용 교란 |
| F5 | P, 후보 ⟂ `cf_nvalid` (유효일수, fold 내부) | 관측일수 교란 |
| S1 | `B0_all` = {low_SD, low_MED} 를 전체 조인일·착용률 ≥ 0.8·MET==0.0 run ≤ 60분·피험자별 날짜축 선형 detrend 후 계산 | **기술용 sensitivity, 채택 금지**. §1.2 감쇠 산술의 예측(|Δ| ≤ 0.01) 과 비교만. run 1 에서 전체일 창이 이미 라벨 노출됐음(−0.010)을 공시 |
| 기술 | split-half 신뢰도, 교란 감사(n_days, gap_days, 모집월, 비착용, volume, 유효일수, 창 길이 MED/SD, onset MED, 최장 bout MED) | label-free, 게이트 아님 |

**해석 규칙(기계적, cell 12)**:
(a) T1/T2 이고 F2·F4·F5 의 Δ_MCI(vs B0ref) 모두 > 0 → 구조 신호, 복제 대상.
(b) T1/T2 이나 하나라도 ≤ 0 → 수준/수집 결합 의심, 복제 전 해석 보류.
(c1) T3 이고 F3 Δ_MCI ≤ 0 → 후보·가족 종결 → stop rule.
(c2) T3 이고 F3 Δ_MCI > 0 → headline 기각; 2순위 흔적은 기록만, **추가 실험 없음**(사용자 결정 2026-09-04). 어느 경우도 F/S 수치를 headline 으로 쓰지 않는다.

## 8. 사전 기대치

- 사전 근거 **없음**. 저장소의 정좌 bout 0.62~0.63 은 Training-141 312-way 스크린 최댓값으로 방향무관 null 상한 0.663 안. 정직한 단변량 prior ≈ 0.50~0.55.
- H1 이 참이어도 binormal 결합상 Δ_MCI ≈ +0.01~+0.03, 3번째 특징의 Dem 채널 교란 −0.03~−0.14(세 번 연속 실측) → Δ_merged ≈ −0.02~+0.015.
- paired Δ CI 반폭 ≈ 0.06 → T1 검정력 < 10%. **P(T1) < 5%, P(T2) ≈ 20%, P(T3) ≈ 75%.**
- 성공해도 merged 상한 0.66~0.68 (0.70 은 B0 anchor 기준 직교 단변량 0.634 필요).
- cell 13 의 결정성 해시는 headline 출력 뒤에 실행된다. 배선 테스트에서 일치를 확인했으므로 실제 run 에서 불일치가 나오면 "재실행"이 아니라 "불일치"로 기록한다.

## 9. 금지

결과 후 정의 변경(임계 1.5/0.5/0.8/240, 창 규칙 포함) · 특징 추가 · 추정기/그리드/seed 교체 · threshold 조작 · 진단·S1 arm 승격 ·
0.5 fallback 을 성능으로 보고 · p_max/p_primary 를 후보 증거로 인용 · 스크린 표의 0.62~0.63 을 prior 로 인용 · S1 을 새 baseline 으로 채택.

## 10. Stop rule (사용자 결정 2026-09-04)

T3 → 이 코호트에서 **특징 탐색 종료**. 결론을 STRICT3 §10(신설)과 보고서 초안에 기록: 신호 한계(유일한 살아있는 축의 CN-MCI d′ 0.38~0.45, trait 신뢰도 보정 후 상한 0.68),
표본 한계(SE_Δ 0.033; 0.70 판정에 n ≈ 310 @ 참 AUC 0.75 / ≈ 2,050 @ 0.72), 다음 코호트 사양(신규 피험자, 피험자당 ≥ 70일, 나이·성별·교육, frozen 채점함수 1회 평가).
T2 → 동일 정의로 복제 1회(seed 군 변경) 후 종료. T1 → 복제 + 기전 확정 후 종료. 어느 경우도 후보 정의를 다시 만지지 않고, 선택하지 않은 대안(fold-local pool 스크린, 양방향 normative 채점)을 실행하지 않는다.

---

## 11. 정직 공시 — confirmatory-record 인 이유

1. **구성개념 인접 노출**: 저장소의 3-class EDA 스크린 표 `SangHyo/3-class/ThreeClass_GoogleYDF_CNBoost/eda_outputs/feature_effects.csv`(7,920행, Training-141, label-aware)에
   같은 구성개념의 변형 `activity__event{7,14,28}__activity__class__inactive_longest_run_ratio_within_wear__{std,iqr,mad,…}` 가 있다.
   이 노트북의 정의와 다른 점 4가지: (i) 비율(run/유효 표본) vs **절대 분**, (ii) `class_5min` 의 inactive 토큰 vs **MET < 1.5**, (iii) 24h "within wear" vs **깨어있는 창**, (iv) 7/14/28일 이벤트 창 vs **첫 35일 SD**.
   그럼에도 인접하므로 blind 가 아니라 confirmatory-record 로 보고한다. **이 세션에서는 그 표의 이름 열만 읽었고 AUC 열은 읽지 않았다.** STRICT3 §9.2 에 따르면 그 스크린의 최댓값 0.629 는 방향무관 null 상한 0.663 안이다.
2. **이 정의의 라벨 기반 값은 계산되지 않았다.** 2026-09-04 EDA 는 label-free(신뢰도·상관·교란)만 수행했고, EDA 워크플로의 agent 들은 `LabelingData/` 접근이 금지됐으며 결과에 이 특징의 AUC 는 없다.
   설계 비평 agent 가 "정좌 bout 가족의 label-free 자격" 을 계산했고(신뢰도 0.54~0.87 등) 라벨 값은 보고하지 않았다.
3. **라벨을 본 분석(공시)**: B0 OOF 오류분석(MCI 13/51 이 CN 중앙값 아래, 그들의 `low_SD` 중앙값 107)과 frozen feature 기술통계. 이 관찰은 후보 선택에 쓰이지 않았다(후보는 사용자 우선순위 + label-free 직교성으로 확정).
4. **run 1 노출**: 전체일 창(S1 과 유사)이 run 1 의 30특징 P arm 민감도에서 −0.010 을 기록한 바 있다. S1 은 그래서 기술용이며 채택 규칙이 없다.
5. 프로그램 수준 다중성: 이 실험에서 라벨을 보는 arm 은 main 2 + 진단 6 + 기준 2 = **10개**(K=2 만 headline). 보고서에 p_Δ 와 함께 "17-arm 정도의 비공식 max 는 SE_Δ 0.033 에서 ≈ +0.07" 을 명시한다.

## 12. 개정 이력

| 버전 | 시점 | 내용 |
|---|---|---|
| v1 | 2026-09-04 13:15 KST, 실행 전 | 가설·정의·arm·endpoint·T1/T2/T3·F1~F5/S1·해석 규칙·기대치·금지·stop rule·공시. 배선 테스트(QUICK+SYNTHETIC) 0 오류 통과, 코드 해시 2f0cb1a7… |

## 13. 산출물

`Binary_SedentaryBout_Nested_result/<UTC_RUN_ID>/`: `FINAL_REPORT.json`, `main_arms.csv`, `delta_bootstrap.csv`, `diagnostic_arms.csv`, `fold_aucs.csv`,
`chosen_C.csv`, `confound_audit.csv`, `feature_correlation.csv`, `reliability.csv`, `primary_coefficients.csv`, `feature_matrix.csv`, `daily_metrics_first35.csv`,
`oof_predictions.csv`, `null_max.npy`, `null_delta.npy`, `null_primary.npy`, `null_baseline.npy`, `null_colperm.npy`, `leakage_audit.json`, `PROGRESS.json`, `report.png`.
실행 환경: 로컬 `.venv` (Python 3.12.8, sklearn 1.9.0, numpy 2.5.0, pandas 2.3.3), `jupyter nbconvert --execute --inplace`, 1 BLAS 스레드.
