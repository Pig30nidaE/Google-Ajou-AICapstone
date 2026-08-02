# reproduction_lim_2025 — 임형준(2025) 재현 및 검증설계 강화

> 임형준(2025), 「웨어러블 기기 라이프로그를 활용한 치매 진단 예측 머신러닝 모델 비교」
> (고려대 석사학위논문 / 디지털산업정보학회 논문지 21(2))를 가능한 범위에서 재구현하고,
> 동일 데이터에서 누수를 통제한 검증과 피험자 독립 중첩 교차검증을 추가한다.

**이 작업은 최고 성능을 만드는 프로젝트가 아니다.** 원 논문 결과가 어느 정도 재현되는지,
그리고 검증설계가 엄격해질 때 성능과 모델 순위가 어떻게 변하는지를 평가한다.

**현재 상태: `reported-method reconstruction`** (exact reproduction 아님 — §6 참조)

---

## 1. 30초 요약: 이번 분석에서 밝혀낸 것

### 1-1. 논문의 평가집단은 33명이며, 산술적으로 완전히 복원된다

논문은 "80:20 분할"이라고만 쓰고 **분할단위도 평가단위도 명시하지 않았다.**
그런데 보고된 지표가 정수비로 정확히 분해된다.

| 보고값 | 분수 |
| --- | --- |
| Accuracy 0.787879 (1D-CNN) | **26/33** |
| Accuracy 0.818182 (LSTM, Bi-LSTM) | **27/33** |
| Accuracy 0.6970 (RF, XGBoost) | **23/33** |
| Recall 0.8571 / 0.7143 / 0.1428 | **6/7 · 5/7 · 1/7** |

혼동행렬을 역산하면 보고된 F1과 **다섯 모델 전부** 정확히 맞는다.

| 모델 | TP | FN | FP | TN | F1(계산) | F1(논문) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF / XGBoost | 5 | 2 | 8 | 18 | 10/20 = .500 | 0.5 ✅ |
| LSTM / Bi-LSTM | 1 | 6 | 0 | 26 | 2/8 = .250 | 0.25 ✅ |
| 1D-CNN | 6 | 1 | 6 | 20 | 12/19 = .632 | 0.63 ✅ |

그리고 **AI-Hub Validation 분할은 정확히 33명 · 양성 7명(CN 26 / MCI 4 / Dem 3)이다.**

→ **평가단위는 피험자이고, "80:20"은 AI-Hub 공식 Training(141)/Validation(33) 분할이다**
(141:33 = 81:19). 174명을 `test_size=0.2`로 나누면 35명이 되어 재현되지 않는다.
LSTM/Bi-LSTM의 AUC 분모도 182 = 7×26으로 떨어져 이를 독립적으로 뒷받침한다.

**따라서 원 논문에는 흔히 의심되는 행 단위 피험자 누수가 없다.** 실제 약점은 다른 데
있다 → §1-3.

### 1-2. 논문 기재값과 실제 데이터의 불일치 10건

`python run.py --inspect-data`가 매번 재계산한다. 주요 항목:

| 항목 | 논문 | 실측 | 심각도 |
| --- | --- | --- | --- |
| 일별 기록 수 | 12,184 | **12,183** | 중 |
| 사용 변수 수 | 58개 | **49개** (논문 코드 실행 결과) | **높음** |
| `date` 컬럼 | 표 1에 존재 | **부재** — 타임스탬프에서 파생 필요 | **높음** |
| 5분 BLOB 변수 6개 | "사용됨" | `numeric_only=True`가 **조용히 탈락** | **높음** |
| `active_low` | 기재 | 실제는 `activity_low` | 낮음 |
| `sleep_temperature_trend_deviation` | 기재 | **부재** | 중 |
| `sleep_is_longest` | 특징으로 사용 | 전 행 **상수 1** | 중 |
| `Q12_TOTAL` | "Q12 총점" | 전 피험자 **상수 0**, 합계도 아님 | 중 |
| 전 문항 0인 피험자 | 미언급 | Validation **1명(Dem)** — 미실시 추정 | 중 |

일치한 항목: 피험자 174명 ✅, 양성 63명 ✅, 라벨 정의 ✅, MMSE 코딩 2/1 ✅,
`TOTAL` = 정답 문항 수 ✅.

### 1-3. 원 논문의 실제 방법론적 약점

행 단위 누수가 아니라 다음 네 가지다.

1. **평가 표본이 양성 7명**. 한 명이 바뀌면 Recall이 14.3%p 움직인다.
   1D-CNN(6/7)과 LSTM(1/7)의 차이는 양성 5명 차이일 뿐이다.
2. **모델 선택에 test를 사용**. 다섯 모델을 같은 33명에서 비교한 뒤 최고인 1D-CNN을
   "최종 분류기로 채택"했다. 보고된 AUC 0.810에는 **5개 중 최댓값 선택 편향**이 있다.
3. **하이퍼파라미터 탐색 범위 미보고**. Random Search + 5-Fold CV를 141명 안에서 했는지
   174명 전체에서 했는지 알 수 없다.
4. **정규화 적합 범위 미보고**.

실험 B·C가 이 각각에 대응한다.

---

## 2. 실행 방법

### 2-1. Colab (정식 실행)

저장소 루트 `base.ipynb`의 셀 2만 수정한다.

```python
USER_FOLDER = "SangHyo"
RUN_FILE    = "Reproduction/reproduction_lim_2025/run.py"
```

`base.ipynb`는 `origin/main`을 새로 clone하므로 **커밋·푸시하지 않은 로컬 수정은
반영되지 않는다.**

### 2-2. 명령어

```bash
python run.py --config configs/paper_reproduction.yaml
```

```bash
python run.py --config configs/leakage_controlled_non_nested.yaml
```

```bash
python run.py --config configs/nested_subject_independent.yaml
```

학습 전 확인용:

```bash
python run.py --inspect-data
```

```bash
python run.py --config configs/paper_reproduction.yaml --dry-run
```

세 실험이 끝난 뒤 비교표:

```bash
python run.py --compare
```

### 2-3. 옵션

| 옵션 | 동작 |
| --- | --- |
| `--dry-run` | 경로·컬럼매핑·표본 수·분할 가능성·클래스 분포·모델 입력 shape·누수 검사·예정 단계만 확인. **학습 없음** |
| `--audit-only` | 데이터셋 단위 누수 감사만 |
| `--inspect-data` | 논문 대비 불일치 보고서만 |
| `--compare` | 세 실험의 `FINAL_REPORT.json`으로 비교표 재생성 |
| `--fold N` | outer fold 하나만 |
| `--seed N` | config seed 덮어쓰기 |
| `--resume` | 완료된 (model, repeat, fold) 체크포인트 재사용 |
| `--device auto\|cpu\|cuda` | 기본 auto |
| `--output-dir PATH` | 결과 경로 지정 |

### 2-4. 런타임 권장

| 실험 | 런타임 | 비고 |
| --- | --- | --- |
| A | CPU High-RAM (트리만) / GPU (딥러닝 포함) | fit 5회 |
| B | GPU | fit 125회 (5 모델 × 25 split) |
| C | GPU | **fit 570회** (outer 75 + inner 495) |

실험 C는 처음부터 5개 모델로 돌리지 말고, `models: [random_forest, xgboost]`로
줄여 한 번 돌려 시간을 잰 뒤 확장하는 것을 권한다.

---

## 3. 세 실험

### 실험 A — `paper_reported_reconstruction`

논문 보고 방법을 가능한 범위에서 그대로 재현한다.

- 분할: `official_partition` (141 / 33) — §1-1에서 복원한 것
- RF/XGBoost: `groupby('EMAIL').mean(numeric_only=True)` 피험자 평균 (논문 코드 그대로)
- LSTM/Bi-LSTM/1D-CNN: 피험자당 1시퀀스 (N, T, F)
- XGBoost 하이퍼파라미터: 논문 명시값 그대로
- 임계값 0.5, 다섯 모델 **모두 보고** (논문과 달리 최고를 사후 채택하지 않는다)

대안 변형 (config에서 전환):

| 변형 | 목적 |
| --- | --- |
| `assumption_variant_random_subject_holdout` | "80:20"을 문자 그대로 읽었을 때 |
| `assumption_variant_random_row_holdout` | **누수 크기 정량화 전용.** 성능 주장 금지 |
| `scaler_scope: all_data` | 전체 데이터 정규화의 낙관 편향 측정 |

### 실험 B — `leakage_controlled_non_nested`

하이퍼파라미터를 논문 값으로 고정하고 피험자·전처리 독립성만 확보한다.

- `StratifiedGroupKFold(5)` × repeats 5 = outer 25개, group = 피험자
- 전처리는 fold의 학습부분에서만 fit
- **CV 점수로 모델·하이퍼파라미터를 재선택하지 않는다** (config가 강제)
- 임계값 0.5 사전 고정
- 다섯 모델이 **동일한 outer split**을 공유

### 실험 C — `nested_subject_independent`

모델 선택과 최종 평가를 분리한다.

- Outer `StratifiedGroupKFold(5)` × repeats 3, Inner `StratifiedGroupKFold(3)`
- inner에서만 선택: 하이퍼파라미터(모델당 2–3개 후보), 결정 임계값
- **outer test는 선택에 절대 미사용** — `check_outer_test_isolation`이 매 fold 검증
- fold마다 저장: 학습·평가 피험자 ID, inner 최적 설정, 선택 임계값, 피험자별
  예측확률·실제 라벨, ROC-AUC / PR-AUC / balanced accuracy / sensitivity /
  specificity / F1 / Brier

---

## 4. 폴더 구조

```
reproduction_lim_2025/
├── run.py                        # 단일 진입점
├── README.md                     # 이 문서
├── reproduction_spec.md          # 재현 사양표 + 33명 복원 근거
├── paper_data_mapping.md         # 논문 변수 ↔ 실제 컬럼 대응
├── assumptions.md                # 논문 미보고 항목과 채택한 가정
├── unresolved_questions.md       # 끝내 확정 못 한 15개 질문
├── leakage_audit.md              # 누수 통제 설계와 자동 검사
├── requirements_colab.txt
├── configs/
│   ├── paper_reproduction.yaml
│   ├── leakage_controlled_non_nested.yaml
│   └── nested_subject_independent.yaml
├── src/
│   ├── engine.py                 # 세 실험 오케스트레이션
│   ├── data/       schema.py loader.py inspect.py
│   ├── features/   representations.py
│   ├── splits/     splitters.py
│   ├── models/     tabular.py sequence.py registry.py
│   ├── evaluation/ metrics.py compare.py
│   ├── audit/      leakage.py
│   └── utils/      config.py seeding.py io.py
└── tests/                        # 86개 계약 테스트
```

---

## 5. 결과 파일

```
<output_dir>/
├── LAUNCHER_STATUS.json          # status: starting | complete | failed
├── TRAINING_COMPLETE.json
├── FINAL_REPORT.json             # 성능의 source of truth
├── subject_predictions_hashed.csv
├── comparison_partial.md
├── folds/<model>__r<N>__f<N>.json
└── checkpoints/
```

`LAUNCHER_STATUS.status == "complete"`와 `TRAINING_COMPLETE.json`이 **둘 다** 있어야
정식 결과다 (`SangHyo/AGENTS.md` §6). 원본 이메일은 저장하지 않고 SHA-256 앞 12자만 쓴다.

---

## 6. 왜 exact reproduction이 아닌가

두 논문 어디에도 다음이 **없다**.

- **LSTM / Bi-LSTM / 1D-CNN의 구조 전부**. 층 수, hidden units, filter 수, kernel size,
  dropout, optimizer, learning rate, epoch, batch size 중 **단 하나도** 없다.
  3.3.2절은 LSTM이 *무엇인지*만 설명하고 끝난다.
- **Random Forest 하이퍼파라미터 전부**.
- **시퀀스 길이·패딩 규칙**.
- **random seed**.
- XGBoost `max_depth` (탐색했다고 쓰면서 결과값 누락).

따라서 모든 산출물에 `reproduction_class: "reported-method reconstruction"`이 붙는다.
**본 재현의 딥러닝 성능이 논문과 다르더라도 그것은 재현 실패가 아니라 미보고의 결과다.**

저자에게 아래 4개만 확보하면 재현 정확도가 실질적으로 올라간다.
(1) 딥러닝 구조 코드, (2) RF 하이퍼파라미터, (3) 시퀀스·패딩 코드,
(4) train/test 분할 코드 한 줄.

---

## 7. 학습 전 반드시 확인할 것

1. `python run.py --inspect-data` — 불일치가 **10건**이고 high가
   `date_column_exists`, `paper_feature_names`, `n_features` 3개인지.
   달라졌다면 **데이터 판본이 바뀐 것**이므로 문서부터 갱신한다.
2. `python run.py --config <config> --dry-run` — 모델 입력 shape과 누수 검사 통과 여부.
   실험 A는 train `[141, 49]` / test `[33, 49]`, 시퀀스는 `[141, 120, 49]` / `[33, 120, 49]`.
3. `python -m pytest tests/ -q` — 86개 통과.
4. 실험 C의 fit 예산(570회)이 시간 내에 들어오는지. 안 되면 `repeats`나 `models`를 줄인다.
5. Colab이면 Drive 마운트 확인. 결과는
   `/content/drive/MyDrive/reproduction_lim_2025_result/<UTC_RUN_ID>/`에 저장된다.

---

## 8. 해석 시 주의

- **33명 Validation은 새로운 독립 test가 아니다.** 이 저장소가 이미 수십 번 관찰한
  historical benchmark다(`SangHyo/AGENTS.md` §2-5). 실험 A는 논문 재현 대상일 뿐이다.
- **non-nested 값은 성능 주장이 아니라** nested 대비 과대평가 크기의 진단값이다.
- **`assumption_variant_random_row_holdout`과 `scaler_scope: all_data`의 숫자를
  성능표에 넣지 않는다.** 누수·낙관 편향 측정값이다.
- 인지검사를 포함한 `secondary_lifelog_plus_cognitive` 결과를 **"손목 웨어러블만으로
  스크리닝한 성능"이라고 표현하지 않는다.** MMSE는 진단에 사용된 검사이므로
  부분적으로 순환적이다.
- 이 저장소의 누수 없는 웨어러블-only OOF는 대체로 **ROC-AUC 0.45~0.57**이다
  (`SangHyo/AGENTS.md` §3-1). 실험 B·C가 이 범위로 떨어지는 것은 **예상된 결과**이며,
  논문 재현 실패가 아니라 검증설계 차이의 크기를 보여주는 것이다.

---

## 9. 관련 문서

| 문서 | 내용 |
| --- | --- |
| `reproduction_spec.md` | 재현 사양표, 33명 복원의 전체 근거, 두 논문 차이 |
| `paper_data_mapping.md` | 변수 58 vs 49의 해부, 금지 변수 목록 |
| `assumptions.md` | 가정 A–G, 변경 시 갱신 절차 |
| `unresolved_questions.md` | 미해결 15개 질문, 저자 문의 최소 목록 |
| `leakage_audit.md` | 13개 자동 검사와 구현 위치, 남은 편향 |
| `../../AGENTS.md` | SangHyo 실험 전체 가이드와 기존 결론 |
