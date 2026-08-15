# Binary_Google_SensorFM_Nested

과제: **CN(0) vs MCI+Dem(1)** — 사람(피험자) 단위, **wearable-only** 이진 분류.

> **상태: 코드 완성. 정식 성능 결과 없음. 아직 어떤 학습도 실행하지 않음.**
> 정식 결과는 Colab GPU에서 `run.py` 실행 후 `FINAL_REPORT.json`으로 확인합니다.

---

## 1. 배경: SensorFM이 무엇인가

Google Research / Google DeepMind의 **SensorFM**
(arXiv:2605.22759, *Towards a General Intelligence and Interface for Wearable
Health Data*)은 웨어러블 건강 데이터용 파운데이션 모델입니다.

- **데이터**: 500만 명, 1조 분(minute) 이상의 Fitbit/Pixel Watch 센서 스트림.
  입력은 raw 파형이 아니라 **1분 단위 집계 피처 34종**(HR/HRV 13, SpO2 3,
  수면 단계 5, 가속도 10, 피부온도/EDA 3)을 24시간 창(1440분)으로 자른 것.
- **모델**: ViT-1D **masked autoencoder**. patch = [20분 × 1피처] → 하루
  72×34 토큰. 위치 인코딩은 2D 가산형(피처축 절반은 학습, 시간축 절반은
  사인파 + 8차원 cyclic datetime). 크기는 XXS(10⁵)~B(10⁸) 4단계(Table ED.4).
- **마스킹**: LSM-2(arXiv:2506.05321)의 **AIM(Adaptive & Inherited Masking)**.
  실제 결측(inherited)과 인공 마스크(artificial: 80% random patch / 50%
  temporal block / 50% modality block 중 표본마다 1개)를 합집합으로 처리하고,
  **손실은 "원래 관측되어 있었는데 인공적으로 가린" 셀에만** MSE로 계산.
- **다운스트림(M.3.4)**: encoder 동결 → 사람별로 (결측 아닌) 토큰 임베딩을
  일 단위로 모아 **mean+std 집계 → PCA-50 → 선형(로지스틱) probe**.
  35개 과제 중 34개에서 수공 피처 supervised baseline(M.3.6, "FE")을 이김.
- **평가**: 사람 단위 5-fold CV(“folds are naturally person independent”),
  ROC-AUC는 logit 변환 공간에서 평균.

### 우리가 그대로 가져올 수 없는 것

| 항목 | 논문 | 우리 |
| --- | --- | --- |
| 체크포인트/코드 | **비공개** (2026-08 웹 확인) | 직접 재구현 + from-scratch 학습 |
| 입력 스키마 | Fitbit/Pixel 34피처 | Oura 링 export: 1분 MET, 5분 활동클래스, 5분 수면 HR/RMSSD/hypnogram |
| 사전학습 코호트 | 외부 500만 명 | **없음** — outer fold의 훈련 피험자(~113명)만 사용 |
| 스케일 | B(10⁸)까지 | XXS(1.4×10⁵)/XS(9.3×10⁵)만 현실적 |

따라서 이 실험의 정직한 질문은 "SensorFM을 쓰면 좋아지는가"가 아니라
**"SensorFM 레시피(아키텍처+마스킹+다운스트림 프로토콜)를 우리 규모에서
from scratch로 돌렸을 때, 같은 논문의 FE supervised baseline을 이기는가"**
입니다. 논문 스스로 XXS가 35개 중 33개 과제에서 꼴찌라고 보고하므로
(스케일이 핵심이라는 것이 논문의 주장), **null 결과가 사전 등록된 유력
시나리오**입니다. 기존 wearable-only 정식 OOF는 0.45~0.57 밴드였습니다.

## 2. 우리 데이터 → SensorFM 입력 매핑

`CONVERT(...)` 컬럼의 인트라데이 문자열을 04:00 기준 1440분 × 8채널 격자로
변환합니다 (`sensorfmnested/grids.py`).

| 채널 | 원천 | 해상도 | 결측 규칙 |
| --- | --- | --- | --- |
| `met` | `activity_met_1min` | 1분 | 값>0 & 해당 5분 class≠0(비착용) |
| `act_class` | `activity_class_5min` | 5분→1분 | class 0(비착용)·부재 = 결측 |
| `stage_deep/light/rem/awake` | `sleep_hypnogram_5min` one-hot | 5분→1분 | 수면 구간 밖 = 결측; bedtime_start 기준 배치, 04:00 경계를 넘으면 이웃 창으로 자연 분할 |
| `sleep_hr` | `sleep_hr_5min` | 5분→1분 | 0 = 결측 (Oura 규약) |
| `sleep_rmssd` | `sleep_rmssd_5min` | 5분→1분 | 0 = 결측 |

- 하루 창 결측 >80% 또는 관측 가능 토큰 <8개 → 제외 (논문 M.3.2 규칙).
- patch [20분×1채널] → 하루 72×8 = **576 토큰** (논문과 동일 기하).
- z-score는 **fold-local**(아래 3절), clip ±5 (논문 M.3.2).

## 3. 두 가지 엄격한 검증 — 이 실험에서의 구현

### 3-1. 피험자 단위 분할

- outer/inner 모든 fold를 `StratifiedKFold`를 3-클래스 진단(CN/MCI/Dem)에
  적용해 **사람 단위로** 나눔 (특징 행렬이 사람당 1행이므로 행 분할 = 사람
  분할이고, `assert_fold_partition`이 기계적으로 재검증).
- seed **20260813** + 동일한 fold 생성 코드 → `Binary_Google_TabFM_Nested`,
  `Binary_Google_CircadianNested`와 outer fold가 repeat 단위로 완전히 일치
  (cross-run paired 비교 가능).
- **SSL까지 fold-local**: 논문은 외부 500만 명으로 사전학습하므로 다운스트림
  5-fold에 사전학습-테스트 인물 혼입이 없습니다. 우리는 외부 코호트가 없으므로
  더 엄격하게, **MAE 사전학습을 outer fold마다 outer-train 피험자의 날짜로만**
  수행합니다. outer-test 피험자는 사전학습에 1분도 기여하지 않고, 채널 z-score
  통계·PCA·imputer·scaler·probe·threshold도 전부 fold-local입니다.
  (라벨은 SSL 어디에도 쓰이지 않습니다.)

### 3-2. Nested CV

```text
Outer: StratifiedKFold(5) × 10 repeats  (사람 단위, CN/MCI/Dem 층화)
├── Outer-Train (~113명)
│   ├── [fold-local] MAE 사전학습 (이 fold의 훈련 피험자 날짜만; 그중 10%
│   │    피험자는 사전학습 early-stop 검증용으로 다시 분리)
│   ├── frozen encoder → 사람별 mean+std 임베딩 (M.3.4)
│   └── Inner: StratifiedKFold(4) × 2 → 후보×설정(PCA K, LR C) 평가
│        → 후보별 최적 설정 → tolerance 0.005 단순성 우선 후보 선택
│        → threshold는 선택 후보의 inner OOF에서
└── Outer-Test (~28명) → 선택 후보 1회 채점 (nested 트랙)
     + 모든 후보의 개별 트랙 (선택 낙관도 진단·paired 대비용)
```

- 후보 3개(사전 고정): `fe_paper_lr`(논문 M.3.6 baseline, complexity 0),
  `sensorfm_lr`(SSL 임베딩 probe, 1), `sensorfm_fe_blend`(rank 평균, 2).
- inner CV는 해당 outer fold에서 사전학습된 encoder를 공유합니다. 즉 inner-test
  피험자의 **라벨 없는** 날짜가 encoder 학습에 포함되는데, 이는 표준적인 SSL
  관행이고(라벨은 어디에도 없음), 성능 주장의 방어선인 outer-test 격리는
  침해하지 않습니다. inner CV의 역할은 probe 설정·후보 선택뿐입니다.
- 사전 등록 대비: `nested−fe`, `sensorfm_lr−fe`, `blend−sensorfm_lr`
  (subject paired bootstrap 95% CI).
- 보조 보고: CN vs MCI AUC, non-nested(최고 단일 후보) 대비 optimism,
  repeat 변동, subject bootstrap CI.
- 33명 Validation은 예측 CSV + SHA-256을 **라벨 열기 전에 동결** 후 1회 채점.
  모델 선택에 절대 사용하지 않음 (AGENTS.md 계약 그대로).

## 4. 논문 대비 명시적 편차 (D-목록)

| # | 편차 | 이유 |
| --- | --- | --- |
| D1 | 입력 34피처 → 8채널 (Oura) | 데이터가 다름; 기하(1440분, [20,1] patch)는 유지 |
| D2 | 사전학습을 fold-local from-scratch로 | 체크포인트 비공개 + 누수 방지 (3-1절) |
| D3 | base LR 5e-4→3e-4, batch 4096→256 | 데이터 규모 축소에 따른 안정화; config에 기록 |
| D4 | cyclic datetime 8차원 = sin/cos 4쌍 + 학습 선형층 | 논문이 정확한 구현(Spathis et al. 인용)을 미공개 |
| D5 | ROC-AUC 집계: repeat별 pooled AUC의 산술 mean±sd | 저장소 표준과의 비교 가능성 우선 (논문은 logit-공간 평균) |
| D6 | FE baseline(Table ED.14 20종)에서 Missing Rate 제거, mesor 추가, acrophase는 sin/cos 2개로 인코딩 → 채널당 21종 | 저장소 계약: 수집 과정 proxy 금지 (착용 행동≠생리); 각도 불연속 회피 |
| D7 | probe는 sklearn LogisticRegression (논문: AdamW 400step 로지스틱 헤드) | 동일한 볼록 문제의 수렴해; C는 inner CV로 선택 |
| D8 | 사람 집계 = 일별 토큰 평균 → 날짜 간 mean+std | 논문 M.3.4 문장의 합리적 해석; 코드에 고정 |

재현 충실도 참고: Table ED.4 대비 파라미터 수는 XXS 134,548/138,740(97%),
XS 923,524/933,204(99%), S 7,269,412/7,290,068(100%)로 일치합니다. 잔차는
전부 입력 스키마 차이(채널 embedding 34→8, cyclic projection)에서 나옵니다.

## 5. 실행 방법

```text
base.ipynb Cell 2:
    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Binary/Binary_Google_SensorFM_Nested/run.py"

권장 순서 (Colab GPU 런타임):
    1) os.environ["BGSFM_ARGS"] = "--profile quick"    # 5×2, 40 epoch 예산
       → 시작 수 분 내 "[probe] PROJECTED TOTAL ..." 라인으로 총 소요 확인
    2) os.environ["BGSFM_ARGS"] = "--profile default"  # 5×10, 120 epoch 예산
    (선택) "--profile max" = 동일 프로토콜 XS 용량 재실행; default가 예산 내
    완주하고 용량 검증이 필요할 때만.
```

- 프로파일: `smoke`(배선; 성능 아님) / `quick` / `default` / `max`(XS).
- CPU에서 default/max는 **의도적으로 실패**합니다 (6시간 예산 보호).
- 진행 상황: fold마다 `PROGRESS.json` 갱신 + epoch 10개마다 로그 +
  60초 heartbeat + 첫 fold 전에 실측 스텝 시간 기반 상한 projection.
- 산출물: `/content/drive/MyDrive/Binary_Google_SensorFM_Nested_result/<RUN_ID>/`
  아래 `FINAL_REPORT.json`, `fold_results.json`, `pretrain_records.json`,
  `oof_predictions_hashed.csv`, `deployment_encoder_state.pt`(round-trip 검증
  포함), 동결된 검증 예측과 SHA-256.

로컬 테스트(학습 없음):

```bash
python -m pytest SangHyo/Binary/Binary_Google_SensorFM_Nested/tests -q
```

(numpy/pandas/sklearn/torch가 없는 환경에서는 해당 테스트가 자동 skip.)

## 6. 결과 해석 규칙 (사전 고정)

1. 1차 근거는 **nested 트랙의 repeat별 pooled OOF ROC-AUC mean±sd**.
2. `sensorfm_lr − fe_paper_lr` paired bootstrap CI가 0을 포함하면
   "SSL 이득 없음"으로 보고한다. 점추정 우열만으로 주장하지 않는다.
3. wearable-only 결과이므로 MMSE 계열(0.7658)과 비교하지 않는다. 비교 대상은
   wearable-only 역사(0.45~0.57)와 fe_paper_lr 동시 실행 anchor다.
4. smoke/quick 수치는 성능이 아니다. 33명 Validation은 1회 채점 참고값이다.
5. XXS가 지면 "SensorFM이 안 된다"가 아니라 "**이 레시피는 141명 규모의
   from-scratch 사전학습으로는 FE를 못 이긴다**"까지만 주장한다. 반대로
   이기면, 사전학습 스케일 없이도 재구성 SSL이 이 과제에 유효하다는
   국소적 증거다.
