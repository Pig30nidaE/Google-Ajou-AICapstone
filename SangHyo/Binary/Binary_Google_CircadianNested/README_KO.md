# Binary_Google_CircadianNested

과제: **CN(0) vs MCI+Dem(1)** — 사람(피험자) 단위 이진 분류.

> **상태: 정식 run 1회 완료** — `20260813_060156_utc` (profile `default`,
> Colab CPU, 34.9분, `status: complete`). 결과 요약은 아래 **11절**.
> 한 줄 결론: **사전 가설(circadian 추가 이득)은 기각**됐고, nested OOF
> ROC-AUC는 **0.7426 ± 0.0220**입니다.

---

## 1. 이 실험이 새로 검증하는 것 (사전 고정 가설)

이 저장소의 모든 선행 웨어러블 표현은 **하루 요약(daily summary)** 이었고, 정직한
사람 단위 OOF에서 ROC-AUC 0.45~0.57에 머물렀습니다. 그런데 데이터에는
`CONVERT(... USING utf8)` 열에 **intraday 원시 시계열**(1분 MET, 5분 활동클래스,
5분 수면 hypnogram, 5분 야간 HR/RMSSD)이 실려 있고, 본 과제(CN vs MCI+Dem)에서는
한 번도 사용된 적이 없습니다 (Dem screening 별도 과제에서만 사용됨).

**가설:** actigraphy 문헌이 인지저하와 연관짓는 **circadian 리듬 통계**
(IS·IV·RA·M10·L5), **수면 미세구조**(WASO·분절지수·단계전환), **야간 자율신경
지표**(HR dip·RMSSD)로 구성한 *작고 사전 고정된* 특징 블록(34개)이 MMSE anchor
위에 추가 신호를 준다.

candidate 목록은 이 가설을 짝지어 검증하도록 설계했습니다: 모든 learner가
`mmse` view와 `mmse+circadian` view로 **쌍**을 이루며, 어느 쪽을 쓸지는 **inner
CV만** 결정합니다.

## 2. 데이터

| 구분 | 인원 | CN | MCI | Dem |
|---|---:|---:|---:|---:|
| Training (nested CV 대상) | 141 | 85 | 47 | 9 |
| Validation (동결 1회 채점) | 33 | 26 | 4 | 3 |

- 사람 키: SourceData `EMAIL` / Labeling·MMSE `SAMPLE_EMAIL`. 라벨은 Gait·Sleep
  LabelingData 사본에서만 읽고 두 사본 일치를 assert.
- Activity/Sleep은 하루 1행(사람당 35~120일). **사람 단위로만 분할**하며, 특징은
  전부 "그 사람 자신의 행"만으로 계산되므로(교차 피험자 통계 없음) 특징 구축
  단계에서 누수가 생길 수 없음 — `tests/test_contracts.py`가 부분집합 재구축
  비트 동일성으로 기계적으로 검증.
- MMSE 문항은 1=오답/2=정답 코딩(assert). 검증셋 `nia+045`의 전부-0 행은 "검사
  미시행"이므로 0을 결측으로 처리(라벨 무관 규칙).
- 금지 열(fail-closed): `DIAG_NM, DIAG_SEQ, DOCTOR_NM, MMSE_NUM, MMSE_KIND`,
  이메일, 그리고 수집량 proxy(`n_days/span/cover/gap/nonwear`) 전부.

## 3. 특징 view (사전 고정)

| view | 크기* | 구성 |
|---|---:|---|
| `mmse` | 38→33 | TOTAL + 6 domain 합 + 30 문항 + recall_deficit (`Binary_MMSE_MaxAUC` anchor와 동일 계열; 상수 문항 5개는 degenerate 필터로 제거) |
| `circ` | 34 | 위 1절의 circadian/수면미세구조/야간자율신경 + 취침·기상시각 circular SD |
| `mmse_circ` | 67 | 두 view의 합집합 |

\* 크기는 train 코호트 상수 열 제거 후. 제거 목록은 결과에 기록됨.

## 4. 모델 후보 9개와 선택 규칙

| candidate | view | learner | complexity |
|---|---|---|---:|
| `lr_mmse_c001` | mmse | LR(C=0.01) — anchor | 0 |
| `lr_mmse_c01` | mmse | LR(C=0.1) | 1 |
| `lr_fusion_c001` | mmse_circ | LR(C=0.01) | 2 |
| `gbt_mmse` | mmse | **YDF GBT** | 3 |
| `obl_mmse` | mmse | **YDF sparse-oblique GBT** | 4 |
| `gbt_fusion` | mmse_circ | **YDF GBT** | 5 |
| `obl_fusion` | mmse_circ | **YDF sparse-oblique GBT** | 6 |
| `blend_mmse` | mmse | LR+**YDF oblique** rank blend | 7 |
| `blend_fusion` | mmse_circ | LR+**YDF oblique** rank blend | 8 |

- **선택 규칙(tolerance rule):** inner 평균 AUC 최고 후보 대비 0.005 이내인
  후보 중 complexity가 가장 낮은 것을 선택. 141명 소표본에서 탐색 낙관을 줄이기
  위한 단순성 편향(AGENTS.md의 반복 교훈 반영).
- 트리 하이퍼파라미터는 사전 고정(탐색 없음), LR은 C∈{0.01, 0.1}만 inner CV로 선택.
- rank blend는 **학습 fold ECDF**로 정규화(테스트 배치 rank 사용 금지 — 비변환적).

## 5. Nested CV 프로토콜

```text
Outer: StratifiedKFold(5) × 10 repeats   (사람 단위, CN/MCI/Dem 3-class 층화)
├── Outer-Train (~113명)
│   └── Inner: StratifiedKFold(4) × 2 repeats
│       ├── 9개 candidate 전부 평가 (view 선택 + 모델 선택 + LR C 선택)
│       ├── tolerance rule로 1개 선택
│       └── 선택 후보의 inner OOF에서 임계값(균형정확도 최대) 결정
└── Outer-Test (~28명)
    └── outer-train 전체로 refit한 선택 후보가 1회 채점
```

- 전처리(중앙값 대치·1/99 winsorize·표준화)는 **fold 내부에서만 fit** (sklearn
  Pipeline). YDF는 결측 native 처리.
- 모든 fold에서 후보 9개 각각의 **정직한 per-candidate OOF**도 저장 →
  ① 선택 낙관(optimism) 진단: `max(단일 후보 OOF) − nested OOF`,
  ② 사전 등록된 paired 대비: fusion vs 그 MMSE 쌍 4건 + nested vs anchor.
- 고정 감사 arm: `circadian_ydf`(circ view 단독, YDF GBT) — intraday 특징만으로
  기존 daily-summary 0.5대를 넘는지 진단.
- 불확실성: repeat별 pooled OOF AUC의 mean±sd(**primary**), subject-mean OOF
  AUC + subject bootstrap 95% CI, paired bootstrap 차이 CI, CN vs MCI 보조 AUC.

## 6. Validation(33명) 취급

- 모델·특징·임계값 선택에 **일절 사용하지 않음**.
- 141명 전체에 동일한 inner-CV 선택 절차를 적용해 deployment 모델을 정하고,
  33명 예측 CSV와 SHA-256을 **라벨 파일을 열기 전에** 저장(코드 순서를 정적
  테스트로 고정). 그 후 딱 1회 채점.
- 33명은 여러 실험에서 재사용된 historical benchmark(all-CN 정확도 0.788)이므로
  독립 코호트 성능으로 해석하지 않음(보고서에 disclaimer 자동 포함).

## 7. Google 기술의 역할 (부가기능이 아닌 핵심)

- **Yggdrasil Decision Forests (YDF, Google)** 가 9개 후보 중 6개의 학습 엔진:
  axis-aligned GBT, **sparse-oblique GBT**(Google 연구 기반 oblique split), 그리고
  두 rank blend의 트리 절반. circadian 단독 진단 arm도 YDF.
- 결측 native 처리(대치 불필요) + 소형 tabular에서 검증된 성능이 선택 이유.
- `ydf==0.16.1` 고정, **fallback 금지**: 미설치·API 불일치 시 sklearn 대체 없이
  즉시 실패(대체 실행은 Google 엔진 사용 보고를 거짓으로 만들기 때문).
- 참고: Google TabFM(2026-07 공개 tabular foundation model)도 검토했으나 소스
  설치 + HF 가중치 다운로드가 정식 run의 재현성 리스크라 이번 후보에서 제외.

## 8. 실행 방법

### Colab (`base.ipynb`) — 셀 2만 수정

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary/Binary_Google_CircadianNested/run.py"

# 선택(기본 = default 프로파일):
# import os; os.environ["BGCN_ARGS"] = "--profile max"
```

- **런타임: CPU** (GPU 불필요, High-RAM 불필요). `ydf==0.16.1` 핀은 다른 버전이
  이미 깔려 있어도 강제 설치하며, 미설치 상태에서 실패하면 중단합니다
  (`config.google_technology.version_pin_honored`에 기록).
- 결과: Drive 마운트 시 `/content/drive/MyDrive/Binary_Google_CircadianNested_result/<UTC_RUN_ID>/`,
  아니면 폴더 옆 `Binary_Google_CircadianNested_result/<UTC_RUN_ID>/`.
- 예상 wall time (로컬 M-계열 Mac 실측 기반 추정): default(5×10 repeats) 약
  20분, Colab CPU에서는 **약 30~60분**. `max`(5×20)는 그 2배.

### 로컬

```bash
python run.py --profile smoke    # 배선 확인 전용(성능 아님)
python run.py                    # default
python -m pytest tests/ -q      # 계약 테스트 15개
```

## 9. 산출물

```text
<result>/<UTC_RUN_ID>/
├── LAUNCHER_STATUS.json                 # starting → complete/failed
├── eda/summary.json                     # 분포·view 크기·결측·fingerprint
└── training/
    ├── FINAL_REPORT.json                # ★ 모든 결론의 source of truth
    ├── fold_results.json                # fold별 선택 후보·inner표·임계값·혼동행렬
    ├── oof_predictions_hashed.csv       # 트랙별 subject-mean OOF (이메일은 SHA-256 해시)
    ├── validation_predictions_label_free_hashed.csv
    ├── VALIDATION_PREDICTIONS_FROZEN.json   # 라벨 개봉 전 동결 해시
    ├── validation_report.json
    └── TRAINING_COMPLETE.json
```

## 10. 결과 해석 기준 (11절은 이 기준대로 판독함)

1. **Primary:** `nested_oof.repeat_roc_auc_mean ± sd` — 기존 정식 nested 최고
   `Binary_MMSE_MaxAUC` 0.7658과 비교.
2. `paired_contrasts_subject_mean`의 fusion−mmse 쌍 4건: **CI가 0을 포함하면
   circadian 추가 이득은 "확인 안 됨"으로 보고** (관찰 부호만 기록).
3. `selection_counts`: fusion 후보가 실제로 선택됐는지(가설의 행동적 증거).
4. `selection_optimism.optimism_estimate`: 단일 최고 후보를 OOF로 고르는 것의
   낙관 크기(정직성 진단).
5. `circadian_ydf` 트랙: intraday 단독이 0.5대(기존 daily-summary 기록)를 넘는지.
6. Validation 수치는 disclaimer와 함께만 인용.

## 11. 실측 결과 (`20260813_060156_utc`, profile `default`)

### 11-1. Primary

| 지표 | 값 |
|---|---|
| **nested OOF ROC-AUC (repeat mean ± sd, 10 repeats)** | **0.7426 ± 0.0220** |
| subject-mean OOF ROC-AUC | 0.7519, bootstrap 95% CI **[0.6613, 0.8309]** |
| PR-AUC (prevalence 0.397) | 0.7279 |
| CN vs MCI 보조 AUC (Dem 9명 제외) | 0.7156 |
| fold-local 임계값 기준 균형정확도 / 민감도 / 특이도 | 0.6583 / 0.5554 / 0.7612 |
| inner−outer gap | +0.0145 |
| 선택 낙관(단일 최고 후보 − nested) | +0.0111 |

기존 정식 nested 최고 `Binary_MMSE_MaxAUC` 0.7658 대비 **-0.023**. 두 값의
bootstrap CI가 크게 겹치고 프로토콜(반복 수·후보군)도 다르므로 **순위 역전을
주장하지 않습니다.** 여기서의 의미는 "엄격한 nested 선택을 붙여도 MMSE 앵커
수준을 유지한다"입니다.

### 11-2. 후보별 정직 OOF (선택 아님, 동일 fold에서 각자 refit)

| candidate | repeat mean ± sd | subject-mean | CN vs MCI |
|---|---:|---:|---:|
| `lr_mmse_c001` (anchor) | 0.7617 ± 0.0135 | 0.7630 | 0.7277 |
| `blend_mmse` | 0.7546 ± 0.0188 | 0.7618 | 0.7279 |
| `lr_mmse_c01` | 0.7416 ± 0.0201 | 0.7475 | 0.7094 |
| `blend_fusion` | 0.7365 ± 0.0164 | 0.7401 | 0.7059 |
| `lr_fusion_c001` | 0.7373 ± 0.0169 | 0.7387 | 0.7051 |
| `obl_mmse` (YDF oblique) | 0.7218 ± 0.0268 | 0.7261 | 0.6921 |
| `gbt_mmse` (YDF axis) | 0.7057 ± 0.0229 | 0.7128 | 0.6826 |
| `obl_fusion` | 0.6979 ± 0.0172 | 0.7038 | 0.6663 |
| `gbt_fusion` | 0.6369 ± 0.0291 | 0.6487 | 0.6113 |
| `gbt_circ` (진단 arm) | 0.3999 ± 0.0361 | 0.3834 | 0.3937 |

### 11-3. 가설 검정 — **circadian 추가 이득 없음 (기각)**

사전 등록한 4개 paired 대비가 **전부 음의 부호**이며 CI는 모두 0을 포함합니다.

| 대비 | 차이 | 95% CI |
|---|---:|---|
| `lr_fusion − lr_mmse` | −0.0244 | [−0.0673, +0.0150] |
| `gbt_fusion − gbt_mmse` | −0.0641 | [−0.1395, +0.0130] |
| `obl_fusion − obl_mmse` | −0.0223 | [−0.0920, +0.0440] |
| `blend_fusion − blend_mmse` | −0.0216 | [−0.0692, +0.0237] |

행동적 증거도 일치합니다: 50개 outer fold 중 **45개가 MMSE-only 후보**를
선택했고 fusion 후보는 5개(`blend_fusion` 3, `lr_fusion_c001` 2)뿐이었습니다.
따라서 **intraday circadian 특징은 이 과제(CN vs MCI+Dem)에서 MMSE 위에
추가 신호를 주지 않습니다.**

### 11-4. `gbt_circ` 0.383의 정체 — 신호가 아니라 **용량 과적합**

우연(0.5) 아래이므로 별도 진단을 수행했습니다. **AGENTS.md 계약 6에 따라
점수를 뒤집지 않습니다.**

- **단변량 감사:** circadian 34개의 사람 단위 AUC 평균이 정확히 **0.500**,
  최대 편차 0.116, |편차|>0.10인 특징 3개(우연 기대 1.3개) → 사실상 null.
- **라벨 순열 검정(30회, 동일 outer CV):** 귀무분포 평균 **0.512**,
  95% 구간 [0.411, 0.645] → CV 구조 자체의 음의 편향은 **없음**.
  관측값 0.393은 이 구간 **아래**(0퍼센타일)이므로 단순 우연도 아님.
- **용량 대조:** 동일 특징·동일 fold에서 저용량 LR(C=0.01)은 **0.514**(≈우연),
  YDF GBT(250 trees, depth 4)만 **0.393**.

결론: 141명 · 34개 null 특징에 고용량 부스팅을 적용하면 fold 내부 잡음을
암기하고 held-out에서 체계적으로 역상관하는 예측을 냅니다. 이는 특징의 역방향
마커가 아니라 **표본 대비 모델 용량 문제**입니다. (이 arm은 고정 진단 arm이라
inner CV 튜닝 대상이 아니었고 선택에도 참여하지 않았으므로 primary 결과에는
영향이 없습니다. 다만 진단 arm에 fusion arm과 같은 250-tree 설정을 그대로 쓴
것은 설계상의 아쉬움입니다.)

### 11-5. Validation 33명 (동결 후 1회 채점, 참고용)

deployment 모델은 `lr_mmse_c001`(임계값 0.5921). ROC-AUC **0.5412**,
임계값 기준 TN 26 / FP 0 / FN 5 / TP 2 → 정확도 0.8485(all-CN 0.7879),
균형정확도 0.6429, 특이도 1.000, 민감도 0.2857.

**해석 금지 사항:** 33명은 여러 실험이 반복 사용한 historical benchmark이고
양성이 7명뿐입니다. OOF 0.75와 Val 0.54의 불일치는 이 저장소에서 반복 관측된
현상(예: `GoogleModels` OOF 0.537 / Val 0.758)이며, **모델 순위 근거로 쓰지
않습니다.**

### 11-6. 무결성 감사 (통과)

OOF 141행·검증 33행, 원본 이메일 0건(SHA-256 해시만), 동결 SHA-256이 디스크
파일과 일치하며 라벨은 동결 이후 개봉, train/val 중복 0, 50개 fold가 매 repeat
141명을 정확히 1회 커버, sparse-oblique가 axis GBT와 다른 결과를 내어 실제
실행됨을 확인.

**단, 재현성 이슈 1건:** 이 run은 Colab에 미리 설치돼 있던 **ydf 0.15.0**으로
실행됐습니다(당시 `_ensure_ydf`가 import 성공 시 설치를 건너뛰었음). Google
YDF 정품이며 fallback도 아니지만 README의 핀 0.16.1과 다릅니다. 이후 커밋에서
핀을 강제하고 `FINAL_REPORT > config.google_technology.version_pin_honored`에
기록하도록 수정했으므로, **다음 run은 0.15.0 결과와 직접 비교하지 마십시오.**

### 11-7. 다음 실험에 남기는 결론

1. MMSE 앵커는 여전히 깨지지 않았습니다. wearable 계열은 daily summary에 이어
   **intraday circadian까지** 추가 이득이 없음이 확인됐습니다(이제 두 표현
   모두 소진).
2. 이 코호트에서 부스팅 계열은 MMSE view에서도 규제 LR보다 낮았습니다
   (0.71~0.73 vs 0.76). 141명에서는 **용량이 작은 모델이 유리**하다는 기존
   교훈이 다시 재현됐습니다.
3. 다음 유의미한 진전은 새 모델이 아니라 **새 코호트 또는 더 민감한 인지
   측정**입니다.

## 12. 재현성

seed 20260813에서 전 단계 시드 파생(`derive_seed`, 오버플로 없는 결정적 믹싱).
실행 시작 시 전체 configuration(프로파일·후보·엔진 버전·환경)을 출력하고
`FINAL_REPORT.json > config`에 저장.

---

*연구용 분류 실험이며 의료 진단 도구가 아닙니다. smoke 결과는 어떤 문서에도
성능으로 인용하지 않습니다.*
