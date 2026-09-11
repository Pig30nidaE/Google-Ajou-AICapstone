# PREREGISTRATION — Experiment 3 (상한 실험): 합법적 레버 전부를 건 K=4 nested 비교

**실험명**: `Binary_Ceiling_Nested`
**성격**: **상한(ceiling) 측정 / exploratory-record**. 확인 실험이 아니다. STRICT3 §10 의 stop rule(2026-09-04) 이후 **사용자 지시(2026-09-05, "분석 결과를 바탕으로 최대한 0.7 이상의 성능을 뽑아내봐")** 로 1회 실행한다. §10.4 의 두 항목(새 특징 정의, fold-local pool 스크린)을 **이 노트북 1회에 한해** 예외 처리한다(§10). 특징 탐색을 다시 열지 않는다.
**동결 시각**: 2026-09-06 18:40 KST, 실행 전 (합성 데이터 배선 테스트 v2 통과 + 5-렌즈 적대적 감사 45건 반영 직후; 실데이터 라벨 기반 값은 이 노트북으로 어떤 것도 계산하지 않은 상태)
**동결 코드**: `Binary_Ceiling_Nested.ipynb` 코드 셀 13개를 `"\n"` 으로 연결한 sha256 = `97306e3706565d14082b5809d0347cfecd116aab64f35569290d66c7974be6a6`
(노트북 cell 13 이 실행 시 같은 방식으로 재계산해 이 문서의 값과 비교하고 `FINAL_REPORT.json.contract_tests.prereg_code_match` 에 기록한다. 불일치 = 코드가 바뀐 것 → 재실행이 아니라 **불일치로 기록**한다.)
**계약**: CN(0) vs MCI+Dem(1), 174명, 피험자 단위 분할, nested CV, MMSE 완전 제외. 변경 불가.
**노출 상태**: **exploratory-record** (§11). blind 가 아니다. 프로그램 headline 은 여전히 B0 0.6491 이다(STRICT3 §10.2).

---

## 1. 질문과 가설

- **Q**: 엄격 3조건 아래 174명 코호트에서, 합법적인 양의 레버를 **전부 동시에** 걸면 subject-mean merged AUC 가 0.70 에 도달하는가?
- H1: 후보 arm 하나가 점추정 ≥ 0.70 이고 B0 대비 증분이 실재한다 (tier G).
- H0: 도달하지 못한다. 그 경우 **B0 를 포함한 4 arm 의 최댓값**(CI 포함)이 이 코호트의 **방어 가능한 상한**이다.
- 레버는 세 가지뿐이다: (i) **측정** — 첫 35일 대신 전체 조인 유효일 (run 4 S1 = STRICT3 §8 #18; +0.008 로 기각됐던 항목을 **측정 레버**로 다시 쓴다); (ii) **유일하게 음수가 아니었던 추가 특징** `longest_w_SD` (run 4 P, §8 #17, +0.010 n.s.); (iii) **정직한 선택** — 사전 고정 라이브러리 4개 중 fold 안에서만 ≤1개를 고르는 inner-CV 선택.
- 0.70 자체는 어떤 tier 에서도 검정되지 않는다 (n=174 의 SE 0.044 로는 불가능; STRICT3 §4).

## 2. 특징 정의 (실행 전 동결, 노트북 cell 3–5 그대로)

```
조인          : activity ⋈ sleep inner join on (sid, sleep_bedtime_end 날짜); run-2 중복 규칙 → 12,150 행 (174명, 피험자별 ≥ 35일)
worn_d(t)     = MET_d(t) ≥ 0.5 ;  zero_run_d = MET == 0.0 연속 최장 run(분)
ok_all_d      = 착용률_d ≥ 0.8 and zero_run_d ≤ 60                    # 전체일 프레임 유효일 (run 4 S1 과 동일)
B0 (35일)     : low_SD = SD(ddof 0) of activity_low over 첫 35 조인일(게이트 없음), low_MED = median   # 동결, 비트 재현
A1 (전체일)   : low_SD_all = SD(ddof 0) of [activity_low − 날짜축 선형추세] over ok_all 일 (≥ 8일, 아니면 NaN → fold 내부 중앙값 대치)
                low_MED_all = median(activity_low over ok_all 일)                                   # run 4 S1 그대로 (게이트 + detrend 포함)
깨어있는 창   : onset_d = 착용분만의 30분 rolling MET 평균 ≥ 1.5 가 처음 성립하는 창의 시작 index(04:00 기준 분; 0 = 04:00 에 이미 활동 중 = 좌측 검열),
                offset_d = 마지막 성립 창의 끝(배타). valid_all_d = ok_all_d and (offset−onset) ≥ 240
longest_w_d   = [onset, offset) 안 정좌(worn & MET<1.5) 연속 run 의 최댓값(분); 비착용 분은 run 을 끊는다
라이브러리 (전부 across-day 통계, ≥ 3일 아니면 NaN → 파이프라인 내부 중앙값 대치):
  longest_w_SD_all = SD(longest_w_d, valid_all 일)
  onset_SD_all     = SD(onset_d, valid_all 일 중 onset_d > 0 인 날)        # 좌측 검열일 제외 (§8 #16 조기기상 가족과의 중첩 차단). wrap 없음
  clsH_SD_all      = SD(H_d, valid_all & class 길이 288 일)
                     H_d = 깨어있는 창 [onset, offset) 에 해당하는 5분 epoch 에서, 유효 epoch = (MET 착용 분 ≥ 3/5) and (class ≥ 1) 사이의 전이만 센
                           5상태(1..5) 전이행렬의 점유율 가중 엔트로피(bit); 유효 epoch < 12 또는 전이 < 6 → NaN; 한 상태만 있는 날 H = 0
  low_lag1_all     = Pearson r of (res_t, res_{t+1}) over 인접 달력일 쌍 (res = A1 과 같은 detrend 잔차, ok_all 일, 쌍 ≥ 10)
제외             : steps_SD (run 2 교란 감사에 174명 Dem 축 단변량 AUC 0.6219 가 기록됨 → 라벨 노출, §11)
```
동결 상수: NONWEAR_MET 0.5 · WEAR_GATE 0.8 · ZERO_RUN_MAX 60 · SED_MET 1.5 · MIN_WINDOW 240 · CLS_LEN 288 · EPOCH_WORN_MIN 3 · MIN_WORN_EPOCH 12 · MIN_TRANS 6 · MIN_DAYS_DETREND 8 · MIN_PAIRS_LAG1 10 · CLS_PARSE_MIN 0.95 (fail-closed) · NAN_MAX_LIBRARY 0.10 (fail-closed).

`activity_class_5min` 은 어떤 엄격 nested arm 에도 들어간 적이 없는 스트림(STRICT3 §9.2)이다. STRICT3 §1 은 "MET 임계로 클래스 재현 시 82.6% 일치 → 신호로는 쓰지 말 것" 이라 했다. 여기서는 (a) 창을 깨어있는 시간으로 제한하고 (b) MET 착용 마스크와 class≥1 이 일치하는 epoch 만 쓰며 (c) 피험자별 class/MET 착용 불일치율을 교란 감사에 넣어 그 경고를 다룬다. 이 정의의 사전 기대는 §8 #2·#4 와 같은 축의 다른 인코딩이라 CN-vs-MCI 0.50~0.55 다.
`low_lag1_all` 은 상관계수라 scale-free 통계(§8 #3 가족과 형식이 같다)이나, 수준 보정이 아니라 **시간 규칙성**을 겨냥한다. 쌍 수(35~120)에서 표본 SE 0.10~0.20 이라 잡음이 크다 — 정직한 기대는 null.

### 실행 전 label-free 자격
split-half 신뢰도·상관·교란 감사(수집·배치 A / 수준·창 구조 B)는 노트북 cell 5·7 이 실행 시 계산해 보고한다(게이트 아님). 합성 배선 실행에서 코드 경로만 확인했다. 감사 B 에서 후보의 |ρ(수준)| > 0.5 는 "수준 프록시 의심" 으로 **기록**한다(판정 불변).

## 3. Arm (K=4, 헤드라인 경쟁은 이 넷)

| arm | 특징 | 비고 |
|---|---|---|
| B0 | `low_SD`, `low_MED` (35일) | run-2 B0 와 비트 동일. subject-mean 0.6490776490776491 + per-repeat 20개 + sub-AUC 1e-9 assert |
| A1 | `low_SD_all`, `low_MED_all` | run 4 S1 과 같은 정의·같은 fold → subject-mean 0.6576576576576576 + per-repeat 20개 + sub-AUC 1e-9 assert |
| A2 MAXFIX | A1 + `longest_w_SD_all` | 고정 K=3 특징 |
| A3 MAXSEL | A1 + {∅ 또는 라이브러리 1개} | outer-train 의 inner 4×2 CV 평균 AUC 로 (집합, C) 선택. best − 0.005 안의 후보 중 특징 수 최소 → C 최소 → 집합 index 최소 (결정적). 선택은 label permutation 과 column permutation 모두에서 재생 |

동일 엔진: SimpleImputer(median) → StandardScaler → 3-class multinomial L2 LR(lbfgs, max_iter 5000), score = 1 − P(CN). 전처리는 Pipeline 안에서 outer-train 에만 fit.

## 4. 프로토콜 (run-2 / Exp 1 / Exp 2 계약 재사용)

```
Outer    : StratifiedKFold(5, shuffle, random_state=100+r) on 3-class, r = 0..19
Inner    : StratifiedKFold(4, shuffle, random_state=7000+37r+f+rr), rr = 0,1; C ∈ {0.003, 0.01, 0.03, 0.1}, tol 0.005 최소 C
집계     : per-repeat OOF AUC(repeat 평균±SD) + 20 repeat 점수 피험자 평균 → subject-mean AUC (헤드라인)
CI       : subject bootstrap 5000 (seed 20260903); paired Δ bootstrap 5000 (seed 20260904; 점추정 = 관측 차이, CI = percentile);
           승자 편향 bootstrap 5000 (seed 20260905; 같은 재표본에서 후보 3 arm 의 Δ 를 전부 계산해 mean[max_j Δ_j^(b)] − max_j Δ_j)
Permutation: 축소 프로토콜(outer 5×2, inner 4×1, C∈{0.01, 0.1}), 3-class 라벨 피험자 단위 셔플, **2000회**, 4 arm 을 같은 순열에서 계산, A3 선택 재생.
           실패 fold 가 있는 순열은 버린다; 유효 순열 < 95% → p 무효. p = (count+1)/(N+1), Clopper–Pearson 95% MC 구간 병기
Column perm: 라벨·A1 열 고정, A2 는 longest_w_SD_all 열만, A3 는 라이브러리 4열을 같은 순열로 피험자 간 셔플(선택 재생), 각 1000회 → p_col (co-statistic)
결정성   : main 4 arm 전체 재실행 OOF 해시 일치
```

## 5. Endpoint

- **Primary (판정용 단일 통계량)**: BEST = 후보 arm {A1, A2, A3} 중 subject-mean merged AUC 최댓값(기계적 argmax, cell 9). 판정에 쓰는 것은 (a) BEST 점추정, (b) B0 대비 paired Δ_merged 95% CI 하한, (c) **label-permutation p_Δ(BEST) 의 Holm 보정값**(가족 = 후보 3 arm). 이 셋뿐이다.
- **상한 진술용**: CEIL = **B0 를 포함한 4 arm** 의 subject-mean 최댓값 [CI]. 후보 3 arm 이 전부 B0 이하이면 상한 = B0 이고 "레버 ≤ 0" 으로 진술한다. 승자 편향(bootstrap) 을 병기하고 편향 보정 상한 = B0 + max(0, max_j Δ_j − bias) 를 함께 보고한다.
- Secondary(기술, 판정·승격 불가): 각 arm 의 merged/CN-MCI/CN-Dem + CI + Hanley–McNeil SE, repeat 평균±SD, fold AUC, 선택 C 분포, A3 집합 선택 빈도, failed_folds(= 0 이어야 유효), per-repeat Δ 부호, 축소/전체 프로토콜 Δ 부호 일치, **p_col**(A2, A3; "레버가 B0 위에 무엇을 더했는가" null), **p_Δmax**, p_max(B0 포함 K=4; "신호 존재" 검정), arm 별 p, 승자 편향 permutation 휴리스틱, P_boot(AUC_BEST ≤ 0.70), CI 하한 − 0.70, A3 선택 비용 = A1 − A3, Dem 채널 falsifier.
- 진단(승격 금지): D1 = run 4 P 재현(비트 동일 assert, 불일치 = run 무효), D2~D5 = A1 + 라이브러리 각 1개 고정 arm (A3 가 무엇을 골랐는지 설명하는 용도; merged 와 Δ 만 CSV 로, OOF 점수는 내보내지 않음).

## 6. 수용 기준 (단일 규칙, cell 12 에 기계적으로 구현)

| tier | 기준 | 의미 |
|---|---|---|
| **G** | BEST 점추정 ≥ 0.70 **and** (Δ_merged CI 하한 > 0 **and** Holm p_Δ(BEST) < 0.05) | 점추정 ≥ 0.70 + B0 대비 증분 실재. **0.70 자체는 검정되지 않음** |
| **G−** | BEST 점추정 ≥ 0.70 이나 증분 조건 미달 | 0.70 주장 불가 (SE 0.044 에서 참 0.665 라도 ≈ 21% 확률로 일어남) |
| **N+** | 점추정 < 0.70 이나 증분 조건 충족 | 상한 < 0.70 이지만 B0 위의 증분은 실재 (기록; 후속 없음) |
| **N** | 그 외 | 이 코호트의 방어 가능한 상한 = CEIL [CI] |

부가 플래그(판정 불변): permutation null 중심(B0·A3 null 평균 ∈ [0.46, 0.54]) 또는 유효 순열 95% 미달 → tier 에 "[p 해석 불가]" 를 붙인다. Δ_Dem CI 상한 < 0 → "Dem 채널 붕괴" 를 상한 진술에 병기한다. 축소/전체 Δ 부호 불일치는 "증분 미확정" 으로 보고한다(tier 불변).
세 조건은 결과를 보고 고르지 않는다. p_Δmax·p_max·p_col·D 값·부호 일치·승자 편향은 판정에 쓰지 않는다.

## 7. 사전 기대치 (실행 전 산술)

- 분해 항등식 merged = (51·A_MCI + 12·A_Dem)/63. 필요 A_MCI: Dem 0.8348 (B0) → **0.668**, Dem 0.8506 (A1) → **0.665**, Dem 1.00 → 0.629. A1 기준 필요 Δ_MCI = 0.665 − 0.6123 = **+0.052** = Hanley SE_MCI(0.049) 의 1.07배. 네 번의 사전등록 추가에서 관측 Δ_MCI 최대 +0.014.
- B0 = 0.6491, A1 = 0.6577, D1 = 0.6589 는 상수(재현 assert). 움직이는 것은 A2, A3 뿐.
- A2: 사전 Δ(A2 − A1) ~ N(+0.010, 0.035) (run 4 P 의 Δ 와 SE_Δ 0.033~0.035) → **A2 기대 0.66~0.67**. 0.70 도달에 Δ(A2−A1) ≥ +0.042 → z ≈ 0.92 → **P(A2 점추정 ≥ 0.70) ≈ 18%**. G 는 추가로 Δ(A2 − B0) CI 하한 > 0 (관측 Δ ≳ +0.065 → z ≈ 1.5 → P ≈ 7%) 과 Holm p < 0.05 → **P(G) < 5%, P(G−) ≈ 15~20%, P(N+) < 10%, P(N) ≈ 70~75%.**
- A3 ≤ A2 기대 (inner SE ≈ 0.05, 집합 5개 → 선택 잡음 −0.00~−0.02). 예상 선택 빈도: base 또는 base+longest_w_SD_all 다수. 라이브러리 신규 3개의 정직한 단변량 prior 0.50~0.55 → D3~D5 각 Δ_merged −0.02~+0.015.
- 따라서 가장 가능성 높은 산출물은 "상한 0.66~0.67 [0.58, 0.75]" 이며, 0.70 을 못 넘는 이유는 STRICT3 §10 의 신호 한계(CN-vs-MCI)다.

## 8. Falsifier / 무결성 검사 (실행 중 자동, fail-closed)

- B0 재현(per-repeat·sub-AUC 1e-9) · A1 = run 4 S1 재현(per-repeat·sub-AUC 1e-9) · D1 = run 4 P 재현(1e-9). 하나라도 실패 → 계약 파손, Δ 해석 안 함, 결과는 보존
- B0 특징 지문 `3bc37d120081437d` (SYNTHETIC 아니면 assert) · run 4 `feature_matrix.csv` 와 공유 컬럼 5개 최대 |차| ≤ 1e-9 (로컬 실행에서 필수; 파일이 없으면 SKIPPED 로 기록하고 경고)
- 90명 부분집합 재구축 비트 동일(피험자 집계 누수 없음; 일별 규칙은 고정 상수임을 leakage_audit 에 명시) · DM 행 = 첫 35일 창 W 행 · 분해 항등식 |오차| < 1e-9 · failed_folds = 0
- class_5min 파싱률 ≥ 0.95 · 라이브러리 결측 ≤ 10% (정의가 데이터와 맞지 않으면 실행 중단)
- permutation: 실패 fold 순열 제외, 유효 ≥ 95%; B0·A3 null 평균 ∈ [0.46, 0.54]
- 결정성: main 4 arm 재실행 OOF 해시 일치 · 코드셀 sha256 = 이 문서의 값

## 9. 금지

결과 후 정의 변경(§2 동결 상수·창 규칙·detrend·검열 규칙 포함) · 라이브러리 추가/교체/복원(steps 포함) · A3 를 "≤2개 선택" 으로 확장 · 추정기/그리드/seed 교체 · 판정 통계량 교체(p_Δmax·p_max·p_col·D-arm·bootstrap 평균 Δ 로 갈아타기) · 진단 arm(D1~D5) 승격 · A3 선택 빈도 상위 후보나 D 값을 새 고정 arm·후속 사전등록의 근거로 쓰는 것(쓰려면 5-way max 를 세어야 함) · threshold 조작 · 0.5 fallback 을 성능으로 보고 · 실패한 실행 삭제 · B0 동결(`b0_frozen_v1`) 교체.

## 10. 정지 규칙과의 관계 · 결과 후 행동

(a) STRICT3 §10 stop rule(2026-09-04)은 유효하다. (b) 이 run 은 사용자 지시(2026-09-05)에 따른 1회성 상한 측정으로, §10.4 의 두 항목(새 특징 정의 3개 + 재창 1개, fold-local pool 스크린 A3)을 **명시적으로 예외 처리**한다. 예외는 이 노트북 1회 실행에 한정된다. (c) 결과가 G/G−/N+/N 어느 것이든 복제·후속 실험·라이브러리 확장·D 승격·정의 수정을 하지 않는다. (d) 이 run 은 특징 탐색을 재개하지 않는다: 라이브러리 4개는 여기서 닫히고, 다음 단계는 §10.3/§12 (신규 피험자, frozen scorer) 다. (e) 최고 arm 은 arm-selection bias 를 동반하므로(AGENTS.md §2.8) 승자 편향 추정을 항상 함께 보고한다.

- G: 같은 정의로 seed 군(random_state 200+r) 복제 1회 후 종료. 특징 추가 없음. §10.3 새 코호트 사양에 "A2 정의 동결 후 이전(transfer) 검정" 한 줄만 추가.
- G−: 0.70 을 주장하지 않는다. STRICT3 §13 에 "점추정만 초과, 증분 미확정" 으로 기록. 복제·추가 실험 없음.
- N+/N: CEIL [CI] 를 STRICT3 §13 에 **이 코호트의 상한**으로 기록. 이후 이 코호트에서 특징 실험 없음.
- 어느 경우도 B0 동결(`b0_frozen_v1`)은 바꾸지 않는다. A2/A3 가 B0 를 대체하지 않는다(그러려면 새 코호트에서 사전등록된 비교가 필요).

---

## 11. 정직 공시 — exploratory-record 인 이유

| arm / 후보 | 노출 상태 |
|---|---|
| B0 | reproduction (run 2/3/4, 0.6491 기지) |
| A1 | reproduction of run 4 S1 (0.6577 기지; STRICT3 §8 #18 에 "기각" 으로 등재된 정의를 측정 레버로 재사용) |
| D1 | reproduction of run 4 P (0.6589 기지; §8 #17 T3) |
| A2 | 기각된 두 레버(#17 + #18)의 전체일 결합 — 결합 값은 미계산. 사전 근거는 산술(독립 증분 합)뿐 |
| `longest_w_SD_all` | 구성개념 노출(run 4, 35일판 Δ +0.010) |
| `onset_SD_all` | onset 은 run 2/3 결합 특징의 재료였음; onset SD 자체의 라벨 값 미계산 |
| `clsH_SD_all` | class_5min 스트림은 Training-141 대규모 label-aware 탐색(§8 #8 1,077개; §9.2 312개 스크린)에 변형이 존재(이름 열만 확인); 이 정의의 174명 라벨 값 미계산 |
| `low_lag1_all` | 미계산 |
| `steps_SD` (제외) | run 2 교란 감사에 **174명 Dem 축 단변량 AUC 0.6219** 기록 → 라벨 노출 → 라이브러리에서 제외 |

1. **stop rule 뒤의 실행**: STRICT3 §10 은 이 코호트에서 특징 탐색 종료를 선언했다. 이 실험은 새 가설이 아니라 "이미 있는 레버의 합" 을 한 번 재는 상한 측정이며, 사용자 지시로 수행한다. 결과가 N 이면 §10 결론이 강화되고, G/G− 여도 §10.3 의 다음 단계(새 코호트)는 바뀌지 않는다.
2. **다중성**: 라벨을 보는 arm 은 main 4 + 진단 5 + 축소 프로토콜 4 = 13개. 판정은 후보 3 arm 의 max 하나에만 걸고(Holm 3), 진단은 승격 금지이며 OOF 점수를 내보내지 않는다. max-over-3 승자 편향은 bootstrap 으로 추정해 상한 진술에 병기한다(SE_Δ 0.033 에서 비공식 max 편향 ≈ +0.02~0.03).
3. **추정량 불일치**: permutation p 는 축소 프로토콜(2 repeat, C 2개) 추정량에, CI 는 전체 프로토콜(20 repeat, C 4개) 추정량에 대한 것이다(run 4 에서 두 Δ 의 부호가 갈린 전례 있음). 두 Δ 를 나란히 보고하고 부호 불일치는 "증분 미확정" 으로 적는다.
4. **라벨 null 의 의미**: label permutation 은 "어떤 신호라도 있는가" 의 null 이다. "레버가 B0 위에 더했는가" 는 column permutation(p_col) 이 답하며, 둘 다 사전 선언된 역할대로만 쓴다.
5. **감사 에이전트**: 동결 직전 5-렌즈 적대적 감사(자동 에이전트, 2026-09-06)를 수행했다. 에이전트에게 `Data/` 접근을 금지했고 5개 렌즈 모두 "Data/ 미접근" 을 공시했다. 한 렌즈는 run 4 결과 폴더의 `feature_matrix.csv` 첫 두 행(label-free 특징, 원본 이메일 포함)을 읽었다 — 라벨 노출은 아니다. 발견 45건은 §12 에 요약한다.
6. **이전 노출**: run 1 에서 전체일 창의 라벨 노출(−0.010), run 4 에서 S1/P 결과. 모두 STRICT3 에 기록됨.

## 12. 동결 직전 감사 기록 (워크플로 `ceiling-prereg-audit`, 45건 → 중복 제거 후 반영)

| 구분 | 반영 |
|---|---|
| 코드 결함 (수정) | CT-F 경로 `Path("")` 가 `.` 로 해석돼 실데이터 실행이 cell 6 에서 죽는 문제(→ `is_file()` 후보 목록) · 실패 fold 의 `None` 이 A3 집계에서 unpack 오류(→ `("failed", nan)`) · 결과 CSV 에 원본 이메일(→ SHA-256[:16] 해시 ID, AGENTS.md §6) · permutation null 이 0.5 fallback 에 fail-open(→ 실패 순열 제외 + 95% 하한 + null 중심 플래그) · class 파싱 `rstrip("/")` + 파싱률/결측 fail-closed · A1 계약을 per-repeat·sub-AUC 로 강화 · 코드 해시를 이 문서와 자동 비교 · OOF 행렬(20×174) 저장 |
| 정의 변경 (동결 전 허용) | `steps_SD_all` 제외(라벨 노출) · `onset_SD_all` 좌측 검열일 제외 · `clsH_SD_all` 을 깨어있는 창 + MET 착용 마스크로 제한 · `low_lag1_all` 을 A1 과 같은 detrend 잔차에서 계산 · 교란 감사에 수준·창 구조(B) 블록, onset==0 비율, class/MET 불일치율, lag1 쌍 수, 게이트 탈락일 수 추가 · `cf_gap_days_all` 을 게이트 전 조인일로 계산 |
| 통계 규칙 | 단일 판정 통계량 고정(§5·§6) · tier 에 N+ 추가, "CI 하한 > 0.60" 조건 삭제(점추정 ≥ 0.70 이면 자동 충족되는 무의미한 조건) · 상한 = B0 포함 max · 승자 편향을 bootstrap 으로 추정해 병기 · N_PERM 2000 + MC 구간 · p_col 복원 · Δ 점추정을 관측 차이로(bootstrap 평균은 병기) · p_Δmax 라벨을 "기술용" 으로 |
| 문서 | §10 정지 규칙 예외 (a)~(e) · §11 후보별 노출 표 · 라벨 arm 수 13 · A1 정의(게이트+detrend) 명시 · §8 #18 관계 · class_5min §1 경고 인용 |
| 기각(반영 안 함) | "PREREGISTRATION 없음"(감사 시점 이후 작성됨) · "판정 통계량 미고정"(§5 로 해결) · "critB 를 0.65 로 상향"(어차피 0.70 검정이 아니므로 삭제가 정직) · "D2~D5 삭제"(A3 선택 설명에 필요; OOF 미수출·승격 금지로 대체) · "코드 해시를 상수로 내장"(자기참조라 불가; 문서 비교로 대체) |

## 13. 산출물 (실행 시 `Binary_Ceiling_Nested_result/<UTC_RUN_ID>/`)

`FINAL_REPORT.json`(재현 anchor = `arms[*].per_repeat` 20값 + `subject_mean_auc`; 계약·판정·permutation·노출 기록), `main_arms.csv`, `delta_bootstrap.csv`, `diagnostic_arms.csv`(merged·Δ 만), `fold_aucs.csv`(fold 별 pick·inner_dead), `chosen_C.csv`, `a3_selection_frequency.csv`, `confound_audit.csv`, `feature_correlation.csv`, `reliability.csv`, `a2_coefficients.csv`, `feature_matrix_hashed.csv`, `daily_metrics_all_hashed.csv`, `oof_predictions_hashed.csv`(main 4 arm 만), `oof_matrix__{B0,A1,A2,A3}.npy`, `null_*.npy`, `null_colperm__{A2,A3}.npy`, `leakage_audit.json`, `report.png`, `PROGRESS.json`. fold 구성은 저장하지 않으며 sorted(SIDS)+y3+StratifiedKFold(5, shuffle, rs=100+r) 로 재생된다.
