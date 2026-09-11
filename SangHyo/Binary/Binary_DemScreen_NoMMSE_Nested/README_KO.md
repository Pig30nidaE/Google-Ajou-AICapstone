# Binary_DemScreen_NoMMSE_Nested

**CN+MCI (162) vs Dementia (12) · 웨어러블 전용 · 피험자 단위 · Nested CV · 단일 노트북**

```
Binary_DemScreen_NoMMSE_Nested.ipynb   ← 유일한 실행 산출물. 18셀, 코드 전부 인라인, 외부 모듈 import 0건
PREREGISTRATION_KO.md                   ← 실행 전 동결 (코드셀 sha256 81a208cf…)
tools/ds_cells.py                       ← 노트북 소스 (셀 마커 형식)
tools/build_nb.py                       ← ds_cells.py → .ipynb 빌더 (sha256 출력)
tools/make_synth_demscreen.py           ← 배선 검증용 합성 코호트 (실제 라벨을 건드리지 않음)
```

---

## 1. 무엇을 하는 노트북인가

요구 조건 네 가지를 **동시에** 지키면서 치매를 선별한다.

| 계약 | 강제 방식 |
|---|---|
| MMSE 완전 배제 | `3.CognitiveFunction` 경로 미개봉 + 특징명 fail-closed 정규식(`_MUST_PASS` 23 / `_MUST_FAIL` 19 자기검증) |
| 피험자 단위 독립 분할 | 피험자 = 1행으로 접은 뒤 분할 + 300 outer fold 전수 `train ∩ test = ∅` 사전 검사 |
| Nested CV | 특징 블록·C·학습기·운영 임계값 전부 inner 에서만 결정, outer-test 는 1회 점수화 |
| CN+MCI vs Dem | 양성 = Dem 12명 (다른 `Binary_*` 폴더의 y 정의와 다르다) |

여기에 **누수 자기검증**(피험자 부분집합으로 특징 재구축 → 전 컬럼 비트 동일)과
**AUC 분해 항등식 검산**(`merged == (111·A_CNvDem + 51·A_MCIvDem)/162`, `<1e-9`)이 assert 로 걸려 있다.

## 2. 실행 방법 (Colab)

`base.ipynb` 진입점 방식이 **아니다.** 노트북 자체를 Colab 에 올려 위에서 아래로 실행한다.

1. Colab 런타임 = **CPU (High-RAM 권장), GPU 불필요.**
2. 노트북 업로드 → 전체 실행. `google.colab` 이 감지되면 Drive 를 마운트하고
   `Shareddrives/GoogleAI_contest/Data` → `MyDrive/GoogleAI_contest/Data` → `MyDrive/Data` 순으로 `DATA_ROOT` 를 찾는다.
3. 결과는 `MyDrive/Binary_DemScreen_NoMMSE_Nested_result/<UTC_RUN_ID>/` 에 새로 저장된다.
4. 진행 상황은 `PROGRESS.json` 과 fold 단위 로그(`[nested:FULL] 125/300 … 남은 예상 4.2분`)로 확인한다.

추가 설치가 필요 없다 — numpy / pandas / scipy / scikit-learn / matplotlib 만 쓴다.

### 환경변수

| 변수 | 뜻 |
|---|---|
| `DS_QUICK=1` | 배선 검증(repeats 60→2, boot→300, perm→20, LOPO repeats→1). **약 1분. 이 숫자는 성능이 아니다.** |
| `DS_SYNTHETIC=1` | 합성 데이터 모드(코호트 계약 assert 완화) |
| `DS_DATA_ROOT` / `DS_OUT_ROOT` | 경로 직접 지정 |

### 예상 소요

로컬 1스레드 실측(축소 실행 기준 외삽): 시계열 파싱 3초 · 특징 2초 · MAIN 2코호트 6분 ·
LOPO 3분 · permutation 1000회 10분 · 탐색 2분 · 진단·그림·결정성 3분 → **약 25분**.
Colab 2 vCPU 에서는 **40~60분**을 예상한다(예산 2.5시간 대비 여유).

## 3. 설계 요약

**특징** — 전부 라벨-프리, 피험자 내부 계산. admitted day = 착용률 ≥ 0.80 AND `MET==0.0` 최장 런 ≤ 60분,
비착용 = `MET < 0.5`. 1분 MET / 5분 class / 야간 5분 hypnogram·HR·RMSSD 는 반드시
`CONVERT(... USING utf8)` 쌍둥이 열에서 파싱한다(plain 열은 `'...'` 플레이스홀더).

| 블록 | 내용 |
|---|---|
| A (2) | `low_SD`, `low_MED` — 앵커 |
| B (5) | + `low_roll7SD`, `low_IQR`, `low_RANGE` — 분산축 |
| C (8) | + `rest_Q25`, `light_MED`, `restless_Q75` — 수준축 |
| D (11) | + `metactive_frac_SD`, `cls_lowfrac_SD`, `hyp_onsetep_SD` — intraday across-day 분산 |
| E (13) | + `RAabs_SD`, `M10_SD` — 일주기 **절대 진폭**의 일간 분산 |
| WD_CORE (13) | DemRankAUC `config.py:213` 정의 그대로 |

**금지**: 날짜 파생, 순응도·수집량 프록시, 자기정규화(CV·비율). 비율형 IS/IV/RA 는
선행에서 34특징 블록 단독 0.3999 로 기각됐으므로 **탐색 섹션에서만** 계산해 논문 보고용으로 표에 싣는다.

**프로토콜** — 3-class y3 층화, outer `StratifiedKFold(5)` × (시드군 3 × 20 repeats) = **300 outer fold**,
inner 4×2. 전처리는 outer-train 에서만 적합(median 대치 → 1/99 winsorize → 표준화).
불균형은 `class_weight='balanced'` 만(SMOTE 는 동일 예산 진단 arm). 운영 임계값은 inner Youden 의 분위 위치.

**arm (K=6)** — `A0` 무학습 바닥선 · `A1` 블록 A · **`A2` inner 가 블록 A~E 선택 (헤드라인)** ·
`A3` WD_CORE 13 (LR vs HistGB) · `A4` 블록 D · `A5` rank-space 평균.
탐색 arm 8개는 셀 15 에 분리하고 best-of-K 귀무·선택 낙관 보정을 병기한다.

## 4. 결과를 어떻게 읽어야 하는가

**헤드라인은 FULL 174명이다.** 라벨-블라인드 QC 코호트(172명)는 나란히 보고하되
더 좋은 쪽을 헤드라인으로 고르지 않는다 — 이득의 거의 전부가 양성 1명 제거에서 나오기 때문이다.

이 과제에서 4조건을 모두 지킨 선행 실측은 **0.7184**(DemScreen `wearable_only__full`) 와
**0.8304**(같은 실행 QC 코호트)다. 저장소의 0.9 이상 기록 세 건은 각각 MMSE 사용 / 소진된 33명 벤치마크 /
전코호트 SHAP 을 읽고 특징을 고른 뒤 지표를 보고 양성 1명을 제외한 post-selection 이라 **비교 대상이 아니다.**

노트북은 다음을 **강제로 함께 출력**한다.

- 60 repeats 분포 — 그 SD 는 **분할 잡음이지 표본 오차가 아니다**(표본 SE 는 Hanley-McNeil 값).
- 층화·비층화 피험자 부트스트랩 5000 CI **와 그 폭**, 리샘플당 양성 수와 서로 다른 원 양성 개체 수.
- Hanley-McNeil Wald / logit CI (부트스트랩과 교차 검증).
- **LOPO 12회 실재 재실행** — 헤드라인은 그 **최솟값**을 함께 인용한다.
- 양성 12명 각각이 앞서는 음성 분율(이 12개의 평균이 곧 AUC).
- permutation 1000회 — 귀무 평균이 ~0.50 인지 **무결성 게이트**로만 읽는다. 이 검정은
  "신호가 전혀 없음"만 기각할 수 있고 참값 0.75 와 0.90 을 구별할 힘은 없다.
- non-nested 낙관, SMOTE 진단, **day-level 무작위 fold 대조군(INVALID 표시)**.

**판정 게이트** — 네 조건(점추정 ≥ 0.90 · 부트스트랩 CI 하한 ≥ 0.80 · LOPO 최소 ≥ 0.85 · `p_max` < 0.01)을
동시에 만족할 때만 "0.9 달성"을 선언하고, 실패하면 실패한 조건을 그대로 인쇄한다.

**양성 12명에서 0.900 은 도달 가능한 값조차 아니다**(1944쌍 격자, 인접값 0.899691 / 0.900206).
95% CI 하한이 0.9 를 넘으려면 양성 ~343명이 필요하다. 즉 **"0.9 달성"은 점추정 언어로만 가능하고
구간 언어로는 불가능하다.**

## 5. 산출물

```
<result>/<UTC_RUN_ID>/
├── FINAL_REPORT.json          # 계약·프로토콜·arm 전체·불확실성·판정·한계
├── RUN_COMPLETE.json          # status / verdict / headline_auc / wall_minutes
├── PROGRESS.json              # fold 단위 진행 (실행 중 다른 창에서 열어볼 수 있음)
├── main_arms.csv  fold_aucs.csv  selection_frequency.csv  bootstrap.csv
├── univariate.csv  confound_audit.csv  fold_audit.csv
├── lopo.csv  positive_contribution.csv
├── explore_arms.csv  explore_univariate.csv
├── feature_matrix_hashed.csv  audit_matrix_hashed.csv  oof_predictions_hashed.csv
├── null_primary.npy  null_max.npy
└── figs/fig1_cohort … fig7_leakage.png
```

피험자 식별자는 전부 SHA-256 앞 16자리로만 저장한다(원본 이메일은 남기지 않는다).
그림의 축·범례는 전부 ASCII 영문이다(Colab 한글 글리프 깨짐 회피). 서사는 마크다운 셀이 담당한다.

## 6. 한계 (FINAL_REPORT.limitations 에도 기록됨)

- **독립 홀드아웃이 없다.** train 141 + validation 33 을 풀링해 CV 한다. 33명 Validation 은
  이미 수십 개 실험에 노출된 역사적 벤치마크다.
- **앵커 특징은 같은 코호트에서 라벨을 보며 발견됐다**(7,920행 EDA 표, 218특징 SHAP 스윕,
  210행 모델 스크리닝). nested CV 는 fold 내부 선택만 보정하고 이 상속된 노출은 보정하지 못한다.
  선언 K=6 은 이번 실행분일 뿐이다.
- **모집 시기 교란**: 등록월 × 클래스 분포가 균등하지 않다. 날짜 파생 특징을 전면 금지하고
  등록월 표를 AUC 옆에 병기한다. `low_roll7SD_detr`(피험자 내 선형추세 제거)가 반증 진단으로 들어 있다.
- **인구통계(나이·성별·교육)가 릴리스에 없다.** 활동 변동성은 연령과 |r| ≈ 0.4 로 알려져 있어
  부분적 연령 프록시 가능성을 배제할 방법이 원천적으로 없다.
- QC 코호트는 라벨-블라인드 규칙의 산물이지만 양성 1명을 제거한다. 헤드라인으로 쓰지 않는다.

## 7. 이 폴더와 섞으면 안 되는 것

`Binary_*` 의 다른 폴더는 대부분 **CN vs MCI+Dem**(양성 63)이며 자체 baseline 이 0.6491~0.7658 이다.
이 폴더는 **CN+MCI vs Dem**(양성 12)로 과제가 다르고 훨씬 쉬우므로,
두 과제의 AUC 를 같은 표에 올리거나 서로 비교하지 않는다.
