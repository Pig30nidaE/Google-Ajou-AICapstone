# Binary_Google_CircadianNested

과제: **CN(0) vs MCI+Dem(1)** — 사람(피험자) 단위 이진 분류.

> **상태: 코드 완성 + 로컬 계약테스트/smoke 배선 확인만 완료. 정식 성능 결과 없음.**
> 이 문서의 어떤 숫자도 이 폴더의 실측 성능이 아닙니다. 정식 결과는 Colab에서
> `run.py`를 실행한 뒤 `FINAL_REPORT.json`에서 확인합니다.

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

- **런타임: CPU** (GPU 불필요, High-RAM 불필요). `ydf==0.16.1`은 없으면 자동
  설치(실패 시 중단).
- 결과: Drive 마운트 시 `/content/drive/MyDrive/Binary_Google_CircadianNested_result/<UTC_RUN_ID>/`,
  아니면 폴더 옆 `Binary_Google_CircadianNested_result/<UTC_RUN_ID>/`.
- 예상 wall time (로컬 M-계열 Mac 실측 기반 추정): default(5×10 repeats) 약
  20분, Colab CPU에서는 **약 30~60분**. `max`(5×20)는 그 2배.

### 로컬

```bash
python run.py --profile smoke    # 배선 확인 전용(성능 아님)
python run.py                    # default
python -m pytest tests/ -q      # 계약 테스트 13개
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

## 10. 결과 해석 기준 (실행 후 확인할 것)

1. **Primary:** `nested_oof.repeat_roc_auc_mean ± sd` — 기존 정식 nested 최고
   `Binary_MMSE_MaxAUC` 0.7658과 비교.
2. `paired_contrasts_subject_mean`의 fusion−mmse 쌍 4건: **CI가 0을 포함하면
   circadian 추가 이득은 "확인 안 됨"으로 보고** (관찰 부호만 기록).
3. `selection_counts`: fusion 후보가 실제로 선택됐는지(가설의 행동적 증거).
4. `selection_optimism.optimism_estimate`: 단일 최고 후보를 OOF로 고르는 것의
   낙관 크기(정직성 진단).
5. `circadian_ydf` 트랙: intraday 단독이 0.5대(기존 daily-summary 기록)를 넘는지.
6. Validation 수치는 disclaimer와 함께만 인용.

## 11. 재현성

seed 20260813에서 전 단계 시드 파생(`derive_seed`, 오버플로 없는 결정적 믹싱).
실행 시작 시 전체 configuration(프로파일·후보·엔진 버전·환경)을 출력하고
`FINAL_REPORT.json > config`에 저장.

---

*연구용 분류 실험이며 의료 진단 도구가 아닙니다. smoke 결과는 어떤 문서에도
성능으로 인용하지 않습니다.*
