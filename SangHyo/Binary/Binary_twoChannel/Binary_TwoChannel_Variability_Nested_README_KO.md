# Binary_TwoChannel_Variability_Nested

**단일 노트북**: [`Binary_TwoChannel_Variability_Nested.ipynb`](Binary_TwoChannel_Variability_Nested.ipynb)
(코드는 전부 셀에 인라인. 별도 모듈·엔트리포인트 없음.)

과제: **CN(111) vs MCI+Dem(63)**, 피험자 174명, **nested CV + 피험자 단위 분할 + MMSE 제외**
상태에서 ROC-AUC ≥ 0.70 도전.

---

## Colab 실행 방법

1. **런타임 유형**: `런타임 → 런타임 유형 변경 → T4 GPU` (Colab Pro/Pro+ 면 L4 권장).
   RAM 은 표준으로 충분하다 (최대 사용 ~200MB).
   GPU 는 **arm B3 (TabPFN) 전용**이다. GPU 가 없어도 나머지 arm 과 모든 진단 arm 이
   CPU 로 완주하고 B3 만 자동으로 건너뛴다.
2. 노트북을 Colab 에 업로드하고 **셀을 위에서부터 순서대로 전부 실행**한다.
   셀 1 에서 Google Drive 마운트 권한을 요청한다.
3. 데이터 경로는 자동 탐색한다.
   1순위 `/content/drive/Shareddrives/GoogleAI_contest/Data`,
   2순위 `/content/drive/MyDrive/GoogleAI_contest/Data`, 그다음 현재 디렉터리 상위의 `Data/`.
   못 찾으면 셀 1 에서 확인한 경로 목록과 함께 즉시 실패한다.
4. 결과는 `/content/drive/MyDrive/Binary_TwoChannel_Variability_Nested_result/<UTC_RUN_ID>/`
   에 저장된다 (`FINAL_REPORT.json`, `main_arms.csv`, `diagnostic_arms.csv`,
   `confound_audit.csv`, `oof_predictions.csv`, `report.png` …).

**추가 설치는 `pip install tabpfn` 하나뿐**이며 셀 1 이 알아서 처리한다.
나머지는 Colab 기본 numpy / pandas / scipy / scikit-learn / matplotlib / joblib 를 쓴다.
**PyCaret · YDF · TabNet 은 쓰지 않는다** (런타임 재시작·핀 충돌 이력 때문).

### 예상 소요시간
| 구간 | 시간 |
|---|---|
| intraday 파싱 + 야간별 지표 | ~2분 |
| 특징 행렬 (174×30) | ~1분 |
| main nested CV (arm 4개 × 5 fold × 10 repeats) | ~5분 |
| 진단 arm 8종 | ~5분 |
| permutation test 1,000회 | 10~35분 (vCPU 수에 따라) |
| bootstrap + 리포트 + 저장 | ~2분 |
| **합계** | **약 25~50분** |

셀 8~10 은 fold 단위로 진행 로그와 ETA 를 출력하므로, 멈춘 것처럼 보이는 구간이 없다.

---

## 실행 전 검증한 것 (실제 데이터 메타데이터 + 합성 데이터 배선 테스트)

**실제 `Data/` 로 확인 (모델 학습은 하지 않음)**

| 확인 항목 | 결과 |
|---|---|
| 라벨 | 174명 = CN 111 / MCI 51 / Dem 12 (Gait·Sleep 사본 일치) |
| activity ⋈ sleep 조인 | 174명 전원 생존, 12,150행 |
| 피험자당 조인 일수 | **min 35** / p5 38 / median 66 / max 122 → **35일 공통 창이 전원에게 정확히 성립, 제외 0명** |
| 중앙 취침시각 | 21.5시 → naive-local 파싱 assert 통과 |
| sleep 중복 subject-day | 12건 → `sleep_duration` 긴 행 유지 규칙으로 처리 |
| 읽는 컬럼 dtype | 전부 수치형. `...` 플레이스홀더는 `_5min` 원본 컬럼에만 있고 우리는 `CONVERT(...)` 만 읽는다 |
| 활동 6분류 합 | 99.3~99.5% 의 날에서 정확히 1440분, `non_wear` 최대 300분 (compositional 성립) |
| intraday 길이 | hypnogram 36~180 epoch, HR/RMSSD 는 hypnogram 길이 또는 +1, 0 = 결측 12.2% |

**합성 데이터(실제 데이터와 동일 스키마)로 전 셀 배선 테스트** — 통과.
그 과정에서 잡아 고친 실제 버그 3건:

1. **fail-closed 가드 오탐** — `S2b_recovery_slope` 안의 `cover` 가 수집량 프록시
   금지어에 걸려 셀 5 에서 중단됐다. 부분문자열이 아니라 토큰 경계 매칭으로 바꾸고,
   오탐/미탐 자기검증(`_MUST_PASS` / `_MUST_FAIL`)을 가드 옆에 붙였다.
2. **joblib `loky` 교착** — permutation 병렬화에서 워커 8개가 전부 0% CPU 로 멈췄다
   (노트북 정의 함수를 자식 프로세스로 보낼 때 발생). **`backend="threading"`** 으로
   교체했고 순차 fallback 도 남겼다.
3. **`plt.show()` 블로킹 / 그림 안 한글 깨짐** — 노트북 안에서만 `show()` 를 호출하도록
   가드하고, 그림 텍스트를 전부 ASCII 로 바꿨다 (Colab 기본 폰트에 CJK 글리프가 없다).

배선 테스트에서 나온 AUC 수치는 **합성 데이터에 인위적으로 신호를 넣은 결과**이므로
성능이 아니며 어디에도 보고하지 않는다. 다만 두 가지 sanity 는 확인됐다:
합성 데이터에 수면 신호를 넣지 않았더니 **수면 블록 ablation 이 0.477** 로 나왔고
(파이프라인이 없는 신호를 만들어내지 않는다), **permutation null 이 0.488 에 중심**을
잡았다 (CV 구조 자체에 편향이 없다). 결정성 재실행 해시도 일치했다.

---

## 이 실험이 새로 검증하는 것

### 1. 집계 통계량 (이 라벨에 한 번도 시도되지 않음)
저장소의 선행 nested 실험은 피험자 집계에 **mean 과 SD 두 통계량만** 썼다
([circnested/features.py:377](Binary_Google_CircadianNested/circnested/features.py:377)).
같은 데이터를 쓴 Kim & Park, *Yonsei Med J* 2026;67(7):524-533 은 이 둘이
**5개 패밀리 중 가장 약한 둘**(mean 0.525, SD/CV/IQR 0.605)이고,
**TS(mean|Δ|·rolling CV) 0.742 / Dist(MAD·kurtosis) 0.701 / EM(median) 0.667** 이
훨씬 강하다는 것을 보였다. 이 노트북은 강한 쪽만 쓴다.

### 2. 두 채널 구조 분리
양성 클래스는 성격이 다른 두 문제의 혼합이고, AUC 는 그 가중평균이다:
`AUC = (51·A_MCI + 12·A_Dem) / 63`.
선행 실험은 전부 하나의 view + 하나의 head 로 병합 라벨을 다뤘다.
설계 리뷰의 실측에서 **수면 특징 40개를 Dem 특징과 한 모델에 섞으면
CN-vs-Dem 이 0.92 → 0.66 으로 붕괴**했다 (Dem 방향은 fold 당 양성 9~10명으로 추정된다).
그래서 D 블록을 2특징으로 고정하고, B2 에서 noisy-OR 로 구조 분리한다.

---

## 정직성 장치 (이 노트북의 핵심)

| 장치 | 셀 | 내용 |
|---|---|---|
| MMSE fail-closed | 2 | `3.CognitiveFunction/` 경로를 만들지 않고, 특징명 정규식 가드 + 자기검증 |
| 누수 자기검증 | 6 | 피험자 90명만으로 특징 행렬 재구축 → **비트 동일성 assert** |
| 특징 동결 | 5 | `assert list(FEAT.columns) == FROZEN_FEATURES` |
| 분해 항등식 | 9 | `(51·A_MCI + 12·A_Dem)/63 == merged` 를 1e-9 오차로 검증 |
| max-over-arms permutation | 10 | arm 별 null 이 아니라 **best-of-K 의 null** 하나 (3-class 라벨 셔플) |
| null 중심 확인 | 10 | primary 단일 arm null 이 0.50 부근이 아니면 경고 |
| 교란 감사 | 7 | `activity_low_std` 의 착용시간·평균분산·창길이 결합 전부 수치화 |
| 결정성 | 12 | repeat 0 재실행 후 OOF 해시 일치 assert |

### 수용 기준 (사전등록) — **셋 다** 충족해야 "달성"
1. Primary merged AUC ≥ 0.70
2. 관측치 > max-over-arms permutation null 의 95백분위
3. subject bootstrap 95% CI 하한 ≥ 0.60

(2)가 없으면 0.70 은 우연과 구분되지 않는다. 이 코호트에서 218-후보 스윕의
null 평균은 **0.706**, 95백분위 **0.765** 였다.

---

## 정직한 기대치

**성공 확률 10~20%.** merged 0.70 은 **Dem head 가 완벽해도 CN-vs-MCI ≥ 0.629**,
현실적 AUC_Dem = 0.90 에서는 **≥ 0.653** 을 요구한다. 팀이 측정한 CN-vs-MCI 범위는
0.498~0.654 이고 중심은 0.58~0.62 다. Dem 쪽에 남은 총 여유는 `(1−0.91)×12/63 = 0.017`뿐이다.

미달해도 남는 것:
1. 이 데이터셋에서 **nested CV + 피험자 단위 + MMSE 제외를 동시에 지킨 최초의 결과**
   (문헌에 nested CV 를 쓴 연구는 존재하지 않는다).
2. 문헌의 0.85~0.95 가 왜 피험자 단위에서 재현되지 않는지에 대한 **분해 기반 설명**.
3. `activity_low_std` 의 성격 규명 — 분산 마커가 아니라 **수준·규모 마커**
   (`low_cv` 의 CN-vs-Dem AUC 는 0.484 로 우연 수준).

## 주의
- `nia+219` 는 Dem 12명 중 1명이자 가장 어려운 양성이다. 제외하면 수치가 오르지만
  **primary 는 174명 전원 유지**하고, 제외판은 진단 arm 으로만 보고한다.
  셀 7 이 라벨 무관 규칙을 적용해 무엇이 제거되는지 출력한다.
- 저장소 `.gitignore` 가 `*.ipynb` 를 제외하므로 커밋하려면 `git add -f` 가 필요하다.
- 진단 arm 의 어떤 숫자도 대표값으로 주장하지 않는다 (permutation K 에도 미포함).
