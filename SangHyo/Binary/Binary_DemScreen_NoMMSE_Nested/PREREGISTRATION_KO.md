# 사전등록 — Binary_DemScreen_NoMMSE_Nested

**동결 시각**: 2026-09-07 (실제 `Data/` 라벨로 실행하기 **전**)
**코드셀 sha256**: `81a208cfb708779c282794c25ee23e85a98a557a714f1866a532c360f085010e`  (v2 — §9 참조)
**시드**: 20260907 · **특징 지문**: 실행 시 `FINAL_REPORT.feature_fingerprint` 에 기록

이 문서는 실행 전에 고정된다. 실행 후 어떤 결과가 나오더라도 아래 정의·프로토콜·판정 규칙을 바꾸지 않는다.

---

## 1. 과제와 4가지 계약

CN(111) + MCI(51) = 162 **음성** vs Dem(12) **양성**, 174명 풀링(train 141 + validation 33), 유병률 0.0690.

1. **MMSE 완전 배제** — `3.CognitiveFunction` 경로 미개봉 + 특징명 fail-closed 정규식 가드(`_MUST_PASS` 23개 / `_MUST_FAIL` 19개 자기검증).
2. **피험자 단위 독립 분할** — 피험자 = 1행으로 접은 뒤 분할. 300개 outer fold 전수 `train ∩ test = ∅` 사전 검사.
3. **Nested CV** — 특징 블록·C·학습기·운영 임계값 전부 inner 에서만 결정.
4. **CN+MCI vs Dem 이진분류.**

## 2. 헤드라인과 병기 규칙 (사용자 결정, 2026-09-07)

- **헤드라인 = FULL 174명 전원 · arm `A2_blockselect`.**
- 라벨-블라인드 QC 코호트(일 평균 25,000보 초과 / 착용 7일 미만 제외)는 **나란히 보고**하되
  **더 좋은 쪽을 헤드라인으로 고르지 않는다.**
- 사전등록 arm 6개가 메인 결과. 0.9 를 노리는 탐색은 셀 15 로 분리하고 best-of-K 귀무·winner-bias 보정을 병기한다.

## 3. 특징 정의 (동결)

전부 라벨-프리이며 피험자 내부에서만 계산한다. **admitted day** = 착용률 ≥ 0.80 AND `MET == 0.0` 최장 런 ≤ 60분,
비착용 = `MET < 0.5`.

| 블록 | 특징 |
|---|---|
| A (2) | `low_SD`, `low_MED` |
| B (5) | A + `low_roll7SD`, `low_IQR`, `low_RANGE` |
| C (8) | B + `rest_Q25`, `light_MED`, `restless_Q75` |
| D (11) | C + `metactive_frac_SD`, `cls_lowfrac_SD`, `hyp_onsetep_SD` |
| E (13) | D + `RAabs_SD`, `M10_SD` |
| WD_CORE (13) | DemRankAUC `config.py:213` WD_CORE 정의 그대로 |

**금지**: 날짜 파생(달력일·등록월·day-of-year), 순응도·수집량 프록시(`n_days`, `span`, `coverage`, `gap_*`, `non_wear`),
자기정규화(CV·비율·정규화 엔트로피). 비율형 일주기(IS/IV/RA)는 **탐색 섹션 전용** — 선행에서 34특징 블록 단독 0.3999 로 기각됐다.
블록 E 는 비율이 아니라 **절대 진폭 차(M10 − L5)의 일간 분산**이다.

## 4. 프로토콜 (동결)

- 층화: **3-class y3**(CN/MCI/Dem). outer `StratifiedKFold(5, shuffle, random_state = base + r)`,
  시드군 `base ∈ {100, 500, 900}` × 20 repeats = **60 repeats / 300 outer fold**.
- inner: outer-train 상대 인덱스 위 `StratifiedKFold(4)` × 2 반복 = 후보당 8 fit. inner OOF 는 fold 별 rank-normalize 후 pooling.
- 전처리 순서(전부 outer-train 행에서만 적합): split → nanmedian 대치 → 1/99 winsorize → 표준화 → fit → score.
- C 격자 `{0.003, 0.01, 0.03, 0.1, 0.3, 1.0}`, 동률 허용오차 TOL 0.005 → (특징 수 ↓, C ↓) 결정적 tie-break.
- 불균형 처리는 `class_weight='balanced'` 만. SMOTE 는 동일 예산 **진단 arm** 으로만 존재한다.
- 운영 임계값은 inner OOF 의 Youden J 분위 위치를 outer-test 에 적용한다(threshold leakage 차단).

## 5. arm (K = 6, 사전등록)

| arm | 정의 |
|---|---|
| `A0_fixed_roll7SD` | `-low_roll7SD` 단독, **적합 없음**. 자유모수 0 바닥선 |
| `A1_anchor_A` | 블록 A + L2 LR, C 만 inner |
| `A2_blockselect` | **헤드라인** — inner 가 블록 A~E × C 를 선택 |
| `A3_wdcore13` | WD_CORE 13 + inner 가 {L2 LR, HistGB(depth 2)} 선택 |
| `A4_intraday_D` | 블록 D 고정 + L2 LR |
| `A5_rankmean` | A1~A4 의 rank-space 평균 (자유모수 0) |

진단(승격 금지): non-nested 낙관, SMOTE, day-level 무작위 fold(**INVALID** 표시).

## 6. 판정 게이트 (실행 전 고정)

`A2_blockselect / FULL 174` 에 대해 **네 조건을 동시에** 만족할 때만 "0.9 달성"을 선언한다.

1. 점추정 ≥ 0.90
2. 층화 피험자 부트스트랩 95% CI 하한 ≥ 0.80
3. LOPO 실재 재실행 12회의 **최솟값** ≥ 0.85
4. `p_max` < 0.01

하나라도 실패하면 `0.9 NOT DECLARED` 를 출력하고 **실패한 조건을 그대로 인쇄한다.**
결과가 무엇이든 특징·프로토콜·헤드라인 arm·코호트를 바꾸지 않는다.

## 7. 사전 노출 고지 (중요)

앵커 특징(`activity_low` 분산축)은 **같은 174명 코호트에서 라벨을 보며** SHAP/EDA 스크리닝으로 발견됐다
(`ThreeClass_GoogleYDF_CNBoost/eda_outputs/feature_effects.csv` 7,920행, Hyunsoo 218특징 스윕,
DemRankAUC 210행 모델 스크리닝). 블록 A~E 와 WD_CORE 의 구성도 그 결과를 읽고 정해졌다.
**nested CV 는 fold 내부 선택만 보정하고 이 상속된 노출은 전혀 보정하지 못한다.**
따라서 이 실험의 유효 K 는 선언값 6 보다 훨씬 크며, 어떤 숫자도 "탐색 무편향"이 아니다.
이 문단은 결과 보고서에도 그대로 실린다.

## 8. 정직한 사전 기대

선행 실측(전부 4조건 준수): DemScreen `wearable_only__full` **0.7184**, `wearable_only__filtered` **0.8304**,
DemRankAUC `nested|wd_full` **0.7303**. nested 로 검증된 적 없는 최고치는 `wd_core` + xgboost **0.8644**.

**사전 기대 밴드: FULL 0.80~0.87 / QC 0.86~0.92.**
양성 12명에서 AUC 는 1944쌍 위의 이산량이고 **0.900 은 도달 가능한 값이 아니다**(0.899691 / 0.900206).
95% CI 하한이 0.9 를 넘으려면 양성 ~343명이 필요하다. 따라서 **"0.9 달성"은 점추정 언어로만 가능하고 구간 언어로는 불가능하다.**


---

## 9. 실행 전 노출 기록 (v1 → v2, 전부 공개)

**v1** 코드셀 sha256 `1993d5b30e20bdbd35cfe3a4521e54b1d388aabefc34d4eaac118f08afab69af` 을 동결한 뒤,
합성 데이터와 **실제 `Data/`** 양쪽에서 `DS_QUICK=1` 배선 검증을 실행했다.
QUICK 모드는 repeats 60→2(시드군 1개), boot 5000→300, perm 1000→20, LOPO repeats 5→1 로 축소하므로
**그 숫자는 성능이 아니다.** 그럼에도 실제 라벨을 본 것은 사실이므로 무엇을 봤는지 남긴다.

실제 데이터 QUICK 배선 실행에서 확인된 것:

- 코호트·조인 계약이 전부 통과했다(174명 = CN 111 / MCI 51 / Dem 12, 조인 12,150행, admitted 12,084일).
- 특징 구현이 선행 공표값을 재현했다: `low_roll7SD` 0.8513(STRICT3 부록 A와 정확히 일치),
  `metactive_frac_SD` 0.8364(선행 0.8349), `cls_lowfrac_SD` 0.8328(선행 0.8338), `light_MED` 0.8184(일치),
  `rest_Q25` 0.8133(선행 0.8138). QC 규칙이 제외한 2명의 해시가 DemScreen `FINAL_REPORT.cohort.quality_flagged`
  두 건과 동일했다.
- 축소 프로토콜 헤드라인은 FULL 0.85 대, QC 0.91 대였다.

**이 실행 이후 바꾼 것은 두 가지뿐이며, 둘 다 결과에 의존하지 않는다.**

1. **버그 수정** — 탐색 섹션의 winner's-curse 보정 부호가 반대였다. `E_b[max] − max_obs` 를 **더하고** 있었는데,
   올바른 부트스트랩 선택 낙관 `E_b[ A_b(승자) − A_obs(승자) ]` 로 바꾸고 **빼도록** 고쳤다.
2. **보고 추가** — 사전등록 6개 arm 중 FULL 최고값과 그 CI 를 함께 인쇄하고,
   "헤드라인은 최대가 아니라 사전등록된 추정량"임을 판정 셀이 명시하게 했다.

**바꾸지 않은 것**: 특징 정의, 블록 구성, arm 목록, 헤드라인 arm(`A2_blockselect`), 헤드라인 코호트(FULL 174),
CV 프로토콜, C 격자, 판정 게이트 4조건. 축소 실행에서 `A4_intraday_D` 가 헤드라인보다 높게 나왔지만
**헤드라인을 그쪽으로 옮기지 않았다.**
