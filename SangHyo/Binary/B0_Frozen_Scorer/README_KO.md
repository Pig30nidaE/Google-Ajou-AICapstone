# B0_Frozen_Scorer — 동결 B0 채점함수 아티팩트 (Phase 3 · NOW-1)

**단일 노트북**: [`B0_Frozen_Scorer.ipynb`](B0_Frozen_Scorer.ipynb) (9셀, 코드 전부 인라인)
**사전등록**: [`PREREGISTRATION_KO.md`](PREREGISTRATION_KO.md) (실행 전 동결, 코드셀 sha256)
**아티팩트**: `b0_frozen_v1.json` (정본, 손으로 감사 가능, sklearn 버전 무관) · `b0_frozen_v1.joblib` (편의)

## 무엇인가
174명 AI-Hub 코호트에서 nested CV 로 검증된 B0 = {`low_SD`, `low_MED`} (subject-mean AUC **0.6491** [0.562, 0.731], CN-MCI 0.6054, CN-Dem 0.8348)를
**하나의 고정된 함수**로 내보낸 것이다. 새 독립 코호트에 **단 1회** 적용된다 (`B0_Score_NewCohort` → `B0_Evaluate_Once`).
이 폴더의 어떤 숫자도 성능 주장이 아니다. 성능은 nested 0.6491 하나뿐이다.

## 정의 (변경 불가)
```
window   : activity ⋈ sleep inner join(sid, date) 후 (sid, date) 정렬, 피험자별 첫 35행. 착용 게이트 없음. 갭 허용
low_SD   : 35일 activity_low(분/일) 의 SD (ddof 0)
low_MED  : 35일 activity_low 의 중앙값
pipeline : median imputer → StandardScaler → multinomial L2 LR (lbfgs, max_iter 5000), C = 0.1 (고정)
score    : 1 − P(CN)
```
C = 0.1 은 B0 arm 의 fold-modal 선택(93/100)이다. 174명 inner-CV 로 다시 고른 결과는 **기록만** 하며 C 를 바꾸지 않는다.

## 계약 테스트 (gate = 하나라도 실패하면 아티팩트 무효)
| # | 내용 |
|---|---|
| CT1 | 원시 CSV → 특징 재구축 지문 `3bc37d120081437d` + run-4 `feature_matrix.csv` 와 **정확 일치** |
| CT2 | 90명 부분집합 재구축 비트 동일 (피험자 독립성) |
| CT3 | 같은 엔진으로 nested B0 재실행 → run-2 headline 0.6490776490776491 과 per-repeat 20개 1e-9 일치 |
| CT4a | 순수 numpy 재구현(median → 표준화 → softmax) 과 sklearn 점수 1e-12 일치 |
| CT4b | **별도 프로세스**에서 joblib 재로드 → 174명 점수 1e-12 일치 |
| CT9a | 엄격 파서(끝 `/` 1개 제거 → split → 정확히 1440) 와 기존 `np.fromstring` 경로 12,150일 비트 동일 |
| CT9b | 명시적 tie-break 키 (sid, date, ts, activity_total desc) 가 현행 keep-first 와 동일한 day set·값 |
| CT8 · CT10 | **report-only**: 대비(contrast) 부호, C 민감도. gate 아님. 어떤 결과도 모델을 바꾸지 않는다 |

## 실행
```bash
cd SangHyo/Binary/B0_Frozen_Scorer
B0FS_NB_PATH=$PWD/B0_Frozen_Scorer.ipynb OMP_NUM_THREADS=1 \
  ../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 B0_Frozen_Scorer.ipynb
```
약 1분. 결과: `B0_Frozen_Scorer_result/<UTC_RUN_ID>/` (FINAL_REPORT.json, CONTRACT_TESTS.json, b0_scores_174_insample.csv) + 폴더 루트의 아티팩트 2개.

## 아티팩트 JSON 구조
`meta`(버전·환경·source runs·nested headline·코드셀 sha256·joblib sha256) · `feature_spec`(정의 원문 + 노트북 셀 2–4 소스와 sha256) ·
`preprocess`(imputer median, scaler mean/scale) · `model`(C, coef 3×2, intercept, classes, score_rule, C 감사) · `test_vectors`(8개, NaN 케이스 포함) ·
`contrast_signs` · `reference_distribution`(174명 label-free 분위수 — drift **보고**용) · `diagnostic_in_sample`(경고 표기) · `contract_tests`.

## 이식성 주의 (새 코호트)
- `activity_low` 는 Oura 의 일일 집계라 기기 세대·펌웨어·API 버전이 바뀌면 정의가 달라질 수 있다. **동일 v1 스키마 export 가 필수**이며 불일치 시 1회 평가는 탐색으로 격하된다.
- 35 조인일 미만 피험자는 채점하지 않는다(짧은 창으로 대체 금지). `activity_low` 결측 피험자도 채점하지 않는다(중앙값 대치 금지 — 봉인 채점기 규칙).
- 점수는 `low_SD` 에 대해 단조(낮을수록 양성)이지만 `low_MED` 에 대해서는 전역 단조가 아니다(대비 부호가 다름). 이는 모델의 성질이며 결함이 아니다.
- `.gitignore` 가 `*.ipynb`/`*.csv` 를 제외한다 → `git add -f` + tag `b0-frozen-v1`.
