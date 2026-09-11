# PREREGISTRATION — Experiment 1: B0 vs B0 + `cpl_wake2onset_slope`

**실험명**: `Binary_WakeOnsetCoupling_Nested`
**동결 시각**: 2026-09-04 (KST), 실행 전
**동결 코드**: `Binary_WakeOnsetCoupling_Nested.ipynb` 코드 셀 13개의 연결 sha256 =
`c8ea052ecf371bacd8a4fe8240eee40bec1f17b7792e954e44600c96338c1a8d`
(노트북 cell 13 이 실행 시 같은 방식으로 재계산해 `FINAL_REPORT.json.prereg_code_sha256` 에 기록한다. 불일치 = 코드가 바뀐 것.)
**계약**: CN(0) vs MCI+Dem(1), 174명, 피험자 단위 분할, nested CV, MMSE 완전 제외. 이 셋은 변경 불가.

---

## 1. 가설

- H1: run 2(`Binary_DaytimeVariety_Nested/20260902_140024_utc`)에서 정의된 `cpl_wake2onset_slope`(이하 **v0**)가
  동결 D블록 `{low_SD, low_MED}` 위에 out-of-sample 증분(Δ)을 준다.
- H0: 증분 없음(또는 희석으로 음수).
- 부가 기전 질문(진단으로만 답한다): 증분이 있다면 그것은 (i) 기상→활동개시 결합인가, (ii) 04:00 이전 조기기상 빈도인가.

## 2. 후보 정의 (run-2 cell 4–5 그대로, 변경 없음)

```
onset_d    = 04:00 앵커 분 index. 착용분(MET ≥ 0.5)만으로 30분 rolling 평균 ≥ 1.5 가 처음 성립하는 창의 시작
wake_min_d = (sleep_bedtime_end 시각 − 04:00) mod 1440      ← 04:00 이전 기상은 ~1200~1440 으로 wrap 된다
v0         = 피험자 내부 OLS 기울기 of onset_d on wake_min_d, 유한 쌍 ≥ 8, var(x) > 1e-9
```

### 실행 전에 알려진 정의 결함 (label-free 실측, 공시)
| 실측 | 값 |
|---|---|
| 04:00 이전 기상일 | 520/6,090일 (8.54%), 94/174명 ≥ 1일 |
| Spearman(v0, 04:00 이전 기상 여부) | **−0.83**; (v0, 조기기상일 수) −0.75 |
| wrap 없는 80명 vs 있는 94명의 v0 | 평균 +0.637 vs −0.017 (SD 0.046) |
| 대안 정의와의 Spearman | drop-wrapped 0.21, signed-x 0.35, drop-onset0 0.88 |
| split-half 신뢰도 | v0 0.30, 수정 정의 0.40, 조기기상일 수 0.69 |

→ v0 는 1차적으로 "35일 중 04:00 이전에 깬 적이 있는가"의 준-이진 지표이며, 결합 기울기는 소수 성분이다.
그럼에도 v0 를 headline 으로 두는 이유: 사전 근거(run-2 사후 단변량 0.606)가 v0 위에 있고, 지금 정의를 바꾸는 것은
in-sample 값을 본 뒤의 사후 선택이 되기 때문이다(사용자 결정 A, 2026-09-04). 보고명은 `cpl_wake2onset_slope v0 (wrapped)`.

## 3. Arm (K=2, headline 경쟁은 이 둘뿐)

| arm | 특징 | 비고 |
|---|---|---|
| B0 | `low_SD`, `low_MED` | run-2 B0 와 비트 동일. 노트북이 subject-mean 0.6490776490776491 과 per-repeat 20개를 1e-9 내로 assert |
| P | `low_SD`, `low_MED`, v0 | 동일 엔진·fold·seed·그리드 |

## 4. 프로토콜 (run-2 계약 재사용, 변경 없음)

```
데이터   : train+val 풀링, activity ⋈ sleep inner join on (sid, sleep_bedtime_end 날짜), 중복 규칙 run-2 동일
창       : 조인 프레임 첫 35행 (전원 정확히 35일)
전처리   : SimpleImputer(median) → StandardScaler → 3-class multinomial L2 LR(lbfgs, max_iter 5000), score = 1 − P(CN)
Outer    : StratifiedKFold(5, shuffle, random_state=100+r) on 3-class, r = 0..19
Inner    : StratifiedKFold(4, shuffle, random_state=7000+37r+f+rr), rr = 0,1; C ∈ {0.003, 0.01, 0.03, 0.1}, tol 0.005 최소 C
집계     : per-repeat OOF AUC(repeat 평균±SD) + 20 repeat 점수 피험자 평균 → subject-mean AUC (헤드라인)
CI       : subject bootstrap 5000 (seed 20260903), paired Δ bootstrap 5000 (seed 20260904)
Permutation: 축소 프로토콜(outer 5×2, inner 4×1, C∈{0.01, 0.1}), 3-class 라벨 피험자 단위 셔플, 2000회
```

## 5. Endpoint

- **Primary**: Δ_merged = AUC_P − AUC_B0 (동일 fold, paired subject bootstrap, 95% CI)
- Secondary: Δ_MCI, Δ_Dem, 절대 AUC 3종 + CI + Hanley–McNeil SE, repeat 평균±SD, fold AUC, 선택 C 분포, failed_folds(= 0 이어야 headline 유효), per-repeat Δ 부호, P–B0 OOF 점수 상관
- Permutation: **p_Δ** = P(Δ_null ≥ Δ_obs), Δ_null = 같은 순열에서 AUC_P − AUC_B0 (헤드라인 p) · p_primary · p_max (B0 포함 max 라 후보 증거 아님, 전역 null 확인용)
- co-statistic (판정 미사용): **p_col** = v0 열만 피험자 간 셔플(라벨·D블록 고정, 1000회)한 Δ null 대비 관측 Δ

## 6. 수용 기준 (계획 승인 시점에 동결, 이후 변경 없음)

| tier | 기준 | 후속 |
|---|---|---|
| **T1 유의한 증분** | Δ_merged 95% CI 하한 > 0 **and** p_Δ < 0.05 | 기전 진단으로 해석 확정 후 새 seed 군(random_state 200+r)으로 복제 |
| **T2 신호 부합(복제 필요)** | Δ_merged > 0 and Δ_MCI > 0 and Δ_Dem CI ∋ 0 and p_Δ < 0.10 and per-repeat Δ>0 ≥ 15/20 | 복제 1회. 특징 추가 금지. baseline 은 B0 유지 |
| **T3 기각** | 그 외 | 후보 폐기 → Experiment 2 |
| 참고 | B: subject-mean ≥ 0.70 / C: max > null p95 / D: CI 하한 ≥ 0.60 | run-2 형식 유지, 판정에 미사용 |

## 7. 기전 진단 (main 과 동일 20 repeats·동일 fold, K 미포함, headline 승격 금지)

| # | arm | 가르는 것 |
|---|---|---|
| F1 | v0 단독 | 기술 통계 전용. **주의**: 단일 열 3-class MNL 의 1−P(CN) 은 클래스 평균이 비단조(v0: CN 0.33 / MCI 0.15 / Dem 0.42)이면 U자 점수가 되어 CN-vs-MCI 가 구조적으로 0.5 근처로 붕괴한다(run-2 의 `low_MED` 단독 0.462, `W1_min` 단독 0.426 이 그 사례). 또한 특징이 라벨을 보고 고른 것이라 nested 라도 "out-of-sample 단변량"이 아니다. 어떤 판정에도 쓰지 않는다 |
| F2 | P, v0 를 fold 내부에서 `n_earlywake` 에 OLS 잔차화 (D블록도 같은 nuisance 로 잔차화; 기준 = F2ref B0⟂n_earlywake) | 조기기상 빈도를 빼도 증분이 남는가 |
| F3 | B0 + `n_earlywake` (04:00 이전 기상일 수) | 아티팩트 프록시만으로 증분이 재현되는가 |
| F4 | B0 + `cpl_fixed` (04:00 이전 기상일·onset==0 검열일 제거 후 기울기, 쌍 ≥ 8) | 제대로 측정한 결합의 증분 |
| F5 | P, v0 를 fold 내부에서 `cf_nonwear_mean` 에 잔차화 (기준 = F5ref) | 고전적 교란. nuisance 는 잔차화에만 쓰고 모델 입력 금지 |
| 기술 | split-half 신뢰도, 확장 교란 감사(wake_SD, 기상 중앙시각, 취침, 수면시간, n_earlywake) | label-free, 게이트 아님 |

**해석 규칙(기계적)**: (a) P 증분 + F2 의 CN-vs-MCI 가 F2ref 보다 높음 + F4 Δ_MCI > 0 → 결합 신호 지지.
(b) P 증분 + F2 소멸 + F3 Δ_MCI > 0 → 실체는 조기기상 빈도(수면 타이밍 구성개념); `cpl_*` 로 주장 금지, 후속은 별도 사전등록 판단.
(c) P 증분 없음(T3) → 후보 종결. 어느 경우도 F 계열 수치를 headline 으로 쓰지 않는다.

## 8. 사전 기대치

- 사후 단변량 0.606 은 lean-6 표(실질 4개)의 최댓값 → 선택 보정 p ≈ 0.11~0.16, EB 수축 추정 ≈ 0.53. 사전 근거는 약하다.
- H1 이 참이어도 binormal 결합상 Δ_MCI ≈ +0.01~+0.03, 3번째 특징의 Dem 채널 교란 −0.03~−0.14 → Δ_merged ≈ −0.01~+0.02.
- paired Δ CI 반폭 ≈ 0.06 → T1 검정력 5~10%. 실질 판정은 T2/T3. P(T1) < 5%, P(T2) 25~35%, P(T3) 60~70%.
- 성공해도 merged 상한 0.66~0.68 (0.70 은 CN-vs-MCI ≥ 0.629 필요).
- 알려진 추정기 한계(동결이므로 바꾸지 않음): 3-class MNL 의 1−P(CN) 에서 v0 의 MCI 방향(낮을수록 MCI)과 Dem 방향(높을수록 Dem)은
  부호가 반대라 부분적으로 상쇄된다. run-2 계수(MCI −0.253, Dem +0.102)가 그 증거다. 이는 Δ_MCI 를 작게 만드는 방향의 편향이며,
  T2 가 Δ_MCI > 0 만 요구하는 이유이기도 하다.
- 결정성·실패 fold 게이트 중 cell 13 의 결정성 해시는 headline 출력 뒤에 실행된다. 배선 테스트에서 일치를 확인했으므로 실제 run 에서
  불일치가 나오면 "재실행"이 아니라 "불일치" 로 기록한다 (결과를 본 뒤의 재실행 금지).

## 9. 금지

결과 후 정의 변경 · 특징 추가 · 추정기/그리드/seed 교체 · threshold 조작 · 진단 arm 승격 · 0.5 fallback 을 성능으로 보고 ·
p_max/p_primary 를 후보 증거로 인용 · 0.606 을 선택 보정 없이 인용 · 결과를 "wake→onset coupling" 이름으로 wrap 분해 없이 기록.

## 10. Stop rule

T3 → Experiment 2 (day-to-day sedentary bout dispersion, 1~2특징, 별도 사전등록; §9.2 의 in-sample 0.62~0.63 은 312-way 스크린 최댓값이라 사전 근거로 쓰지 않음).
T2 → 복제 1회 후 판단. T1 → 복제 + 기전 확정. 어느 경우도 이 코호트에서 후보 정의를 다시 만지지 않는다.

---

## 11. 정직 공시 — 이 실험은 blind 가 아니다

1. **in-sample 대안 정의 값 노출**: 계획 단계의 감사 agent 들이 (지시와 달리) 대안 정의의 in-sample 단변량 CN-vs-MCI 를 계산했고 실험자가 그 값을 보았다:
   drop-wrapped 0.52, signed-x/circular 0.535~0.538, drop-onset0 0.606, 조기기상일 수 0.59, 조기기상 여부 0.586.
   이 값으로 어떤 정의도 고르지 않았다(v0 headline 은 사용자 결정, 진단 정의는 기전으로 선택).
2. **nested 결과 노출**: 설계 비평 agent 의 재실행 결과 요약이 계획 승인 **후**, 실행 **전**에 실험자에게 전달됐다:
   Δ_merged ≈ −0.0144 [−0.079, +0.050], Δ_MCI ≈ +0.014 [−0.051, +0.077], Δ_Dem ≈ −0.138 [−0.324, +0.019]
   (사전등록 seed 로 B0 를 4자리까지 재현한 agent 의 값). §6 의 T1/T2/T3 와 §7 의 F1~F5 는 이 값이 도착하기 전
   (계획 승인 시점)에 확정됐고, 도착 후 어떤 판정 기준도 바꾸지 않았다. 따라서 이 노트북은 **사전등록 기준에 따른 확정·기록용 run** 이다.
3. **설계 비평이 제안한 대안 기준은 채택하지 않음**(노출 후 도착): 예) T2 에 Δ_MCI ≥ +0.015, F 단독 arm CN-vs-MCI ≥ 0.55 등.
   비평에서 채택한 것은 판정과 무관한 세 가지뿐: p_col co-statistic 추가, 진단을 main 과 같은 20 repeats 로, P–B0 OOF 상관 보고.
4. 노출된 값이 시사하는 사전 예상: T3 기각 가능성이 높다. 그래도 실행하는 이유는 permutation·기전 분해·산출물이 1차 사료(STRICT3)에
   필요하고, 후보의 종결 근거가 "감사 중 어떤 agent 의 재실행 요약"이 아니라 사전등록된 노트북의 기록이어야 하기 때문이다.

## 12. 개정 이력

| 버전 | 시점 | 내용 |
|---|---|---|
| v1 | 계획 승인 (2026-09-04, 노출 전) | 가설·arm·endpoint·T1/T2/T3·F1~F5·해석 규칙·기대치·금지·stop rule |
| v2 | 실행 전, 설계 비평 도착 후 | §11 공시 추가; p_col co-statistic; 진단 20 repeats; OOF 상관. 판정 기준 변경 없음 |

## 13. 산출물

`Binary_WakeOnsetCoupling_Nested_result/<UTC_RUN_ID>/`: `FINAL_REPORT.json`, `main_arms.csv`, `delta_bootstrap.csv`,
`diagnostic_arms.csv`, `fold_aucs.csv`, `chosen_C.csv`, `confound_audit.csv`, `feature_correlation.csv`, `reliability.csv`,
`primary_coefficients.csv`, `feature_matrix.csv`, `oof_predictions.csv`, `null_max.npy`, `null_delta.npy`, `null_primary.npy`,
`null_baseline.npy`, `null_colperm.npy`, `leakage_audit.json`, `PROGRESS.json`, `report.png`.
실행 환경: 로컬 `.venv` (Python 3.12.8, sklearn 1.9.0, numpy 2.5.0, pandas 2.3.3), `jupyter nbconvert --execute`, 1 BLAS 스레드.
