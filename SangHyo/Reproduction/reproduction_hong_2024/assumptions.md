# 가정 목록

논문이 보고하지 않은 설정을 전부 여기에 모았다. 각 항목은 **config 필드로
노출**되어 있으므로, 다른 선택이 결과를 얼마나 바꾸는지 재실행으로 확인할 수 있다.

영향도: **높음** = 성능이 눈에 띄게 달라질 수 있음 / **중간** = 표본 수가 달라짐 /
**낮음** = 구현 편의.

---

## 데이터 구성

### A-01. 날짜 키를 `sleep_bedtime_end`로 정의 — 영향 **중간**
- 논문: "one day"라고만 서술. 어느 시각 기준인지 없음.
- 선택: 기상 시각(`bedtime_end`)의 날짜.
- 근거: 같은 날짜 충돌이 `bedtime_end` 24행 vs `bedtime_start` 1,044행. 야간 수면이
  자정을 넘기므로 기상일이 자연스러운 "수면일"이다(Oura 관례).
- config: `data.sleep_date_source: bedtime_end | bedtime_start`

### A-02. 같은 날짜 중복은 가장 긴 수면을 남김 — 영향 **낮음**
- 논문: 언급 없음. `sleep_is_longest`는 전 행 1이라 판별에 못 쓴다.
- 선택: `sleep_duration`이 최대인 행 유지 (12,183 → 12,171행, 12행 제거).
- 대안: `latest_bedtime_end`, `first`.
- config: `data.duplicate_policy: longest_duration | latest_bedtime_end | first`

### A-03. AI-Hub 데이터 버전 — 영향 **알 수 없음**
- 논문 Data Availability: 2023-05-01 접근. 버전 표기 없음.
- 이 저장소의 사본으로 Table 3 수치가 **전부 일치**하므로 같은 배포본으로 본다.
- 확인 불가 항목이므로 `unresolved_questions.md` Q-01에도 남긴다.

### A-04. `rmssd_average`를 5분 계열의 평균으로 계산 — 영향 **중간**
- 논문 Table A1: "Average heart rate variability", 계산식 없음.
- 선택: `mean(0이 아닌 CONVERT(sleep_rmssd_5min))`.
- 근거와 한계: 사전집계 컬럼 `sleep_rmssd`와 r=0.997이지만 정확 일치는 38.2%뿐.
  어느 쪽이 논문의 값인지 확정할 수 없다.
- SHAP에서 HRV가 2위 변수로 보고되었으므로 이 선택은 해석에 직접 영향을 준다.
- config: `data.rmssd_source: intraday_mean | sleep_rmssd_column`

---

## 분할

### A-05. "final week"을 마지막 7개 **달력일**로 해석 — 영향 **중간**
- 논문 §4.2: "isolating the final week of data from each subject".
- 두 해석이 이 코호트에서 다른 결과를 낸다.

| 해석 | L=3 test 시퀀스 | L=4 | L=5 |
| --- | ---: | ---: | ---: |
| 마지막 7개 달력일 (채택) | 590 | 426 | 294 |
| 마지막 7개 레코드 | 669 | 477 | 321 |

- 달력일 해석에서는 test 기간이 1~7일로 사람마다 다르다(중앙값 6일).
- config: `split.final_week_mode: calendar_days | record_count`

### A-06. validation 기간 — 영향 **높음**
- 논문은 "best balance between accuracy and **validation loss**"(§4.2)와 early
  stopping 적용(한계 절)을 언급하지만 validation을 **어디서 떼었는지 쓰지 않았다.**
- 선택:
  - 실험 A/B1: test보다 앞선 train 기간의 마지막 14일을 validation으로 사용한다.
    14일은 **논문 보고값이 아닌 민감한 구현 가정**이다. 논문이 명시한 early
    stopping을 실제 적용하면서 test monitor를 피하기 위한 선택이다.
  - 실험 A'·B2·C의 outer refit: 별도 validation이 없으므로 early stopping을 끄고
    고정 epoch를 사용한다. 실험 C의 inner fold는 하이퍼파라미터 선택용 validation이다.
- **어떤 경우에도 test 기간을 early stopping monitor로 쓰지 않는다.**
- `early_stopping: true`인데 독립 validation이 없으면 실행 전 config 검증이 실패한다.
- config: `split.validation_days`

### A-07. undersampling 방식 — 영향 **높음**
- 논문 §4.2: "simple undersampling to balance the data distribution across classes
  in the training set". 단위(시퀀스/피험자), 목표 비율, seed 모두 없음.
- 선택:
  - 실험 A: `random_sequence`(문언에 가장 가까운 해석), 목표 비율 1:1.
  - 실험 B1/B2/C: `subject_balanced`. 피험자별로 같은 **비율**을 남겨 특정
    피험자가 통째로 사라지지 않게 한다.
- 근거: 5일 시퀀스 수가 피험자당 2~118개로 편차가 크다. 무작위 추출은 기록이 적은
  다수 클래스 피험자를 통째로 지울 수 있다.
- 실행 시 sampling 전후의 클래스별 시퀀스 수·피험자 수와 경고를 항상 기록한다.
- config: `sampling.strategy`, `sampling.target_ratio`, `sampling.class_weight`

### A-08. stride = 1 — 영향 **중간**
- 논문: 언급 없음. "grouping continuous data into sequences"만 서술.
- 선택: 1(최대 중첩). 논문의 시퀀스 수를 역산할 단서가 없어 관례를 따랐다.
- 주의: stride 1이면 같은 split 안에서 윈도우끼리 날짜를 공유한다. 이는 정상이며,
  **split을 가로지르는** 공유만 누수다.
- config: `sequence.stride`

---

## 모델

### A-09. 비교모델의 입력 변환 — 영향 **높음**
- 논문: SVM/LR/RF/XGBoost가 "do not reflect time-series characteristics"라고만
  적었다. 3~5일 윈도우를 어떻게 한 행으로 만들었는지 없음.
- 선택: `flatten` (L × 32 → L*32 열).
- 대안: `mean`, `last_day`, `summary`(mean/std/min/max).
- **이 선택을 "논문 방식"이라고 단정하지 않는다.**
- config: `models.representation`

### A-10. LSTM 프레임워크 — 영향 **중간**
- 논문: 명시 없음(수식과 그림은 표준 LSTM). H2O를 쓴 것은 비교모델 쪽이다.
- 선택: PyTorch. 저장소 공통 스택을 따른다.
- 결과: 이 작업은 **exact reproduction이 아니라 reported-method reconstruction**이다.
  Keras 기본값(초기화, 게이트 순서, `recurrent_activation='sigmoid'`)과 PyTorch
  기본값이 다르므로 소수점 단위 일치는 기대할 수 없다.

### A-11. epoch 상한 100 — 영향 **중간**. config: `lstm.max_epochs`
### A-12. early stopping patience 10 — 영향 **중간**. config: `lstm.patience`
### A-13. dropout 0.0 / recurrent dropout 0.0 — 영향 **중간**
- 논문은 dropout을 전혀 언급하지 않는다. 없는 것으로 본다.
- config: `lstm.dropout`(실험 C 탐색 공간), `lstm.recurrent_dropout`

### A-14. batch size 64 — 영향 **중간**. config: 실험 C의 탐색 공간
- 논문 미보고. 표본 크기를 고려한 관례값.

### A-15. learning rate 후보를 {0.001, 0.005, 0.01}로 이산화 — 영향 **낮음**
- 논문 §4.2: "learning rates (0.001 to 0.01)"는 **구간**이지 격자가 아니다.
- 최종 보고값 0.001은 그대로 사용하므로 실험 A/B에는 영향이 없다. 실험 C의 inner
  탐색에만 쓰인다.

### A-16. LSTM 출력은 마지막 timestep의 hidden state — 영향 **중간**
- 논문 Figure 1은 표준 LSTM 셀만 보여 주고, 시퀀스 pooling 여부를 밝히지 않는다.
- 선택: 마지막 hidden state → Dense(64) → ReLU → Dense(1) → sigmoid.
- 근거: "an LSTM layer with 128 units, followed by a dense layer with 64 units, and
  a final dense output layer"라는 서술에 가장 직접적으로 대응한다.

### A-17. H2O AutoML은 선택 backend — 영향 **높음**
- 논문 §4.2는 버전까지 명시했다: "H2O version 3.46.0.1".
- 실험 A의 정식 config는 LR/RF/XGBoost에 H2O 3.46.0.1을 요구한다. 미설치·버전
  불일치 때 sklearn으로 조용히 바꾸지 않고 실패한다. SVM만 아래 제약 때문에
  sklearn을 명시적으로 사용한다. 어느 backend가 돌았는지 결과에 기록한다.
- H2O 경로는 요청한 비교모델별로 family를 고정(GLM/DRF/XGBoost)한 뒤 training
  fold 내부 AutoML로 하이퍼파라미터를 고른다. H2O AutoML에 SVM family가 없어
  `baseline_backend: h2o`로 SVM을 요청하면 조용히 다른 모델을 쓰지 않고 실패한다.
- 실험 C에서 H2O 선택을 쓴다면 반드시 inner CV **안**에 있어야 한다.
- config: `models.baseline_backend`, `models.backend_by_model.<model>`

---

## 평가

### A-18. threshold 0.5 — **논문 확인됨**, 가정 아님
- §4.2와 Table 6에서 명시적으로 확인된다. 실험 A/B1/B2는 0.5 고정.
- 실험 C만 inner CV에서 threshold를 고른다(`threshold.policy: youden`).

### A-19. 피험자 단위 통합은 평균 — 영향 **중간**
- 논문에는 피험자 단위 평가 자체가 없다.
- 선택: 시퀀스 확률의 평균. 중앙값·마지막 시퀀스·다수결을 민감도 분석으로 함께 저장.
- config: 코드 상수(`metrics.AGGREGATIONS`), 전부 자동 계산됨

### A-20. bootstrap은 피험자 단위 2,000회 — 영향 **낮음**
- 시퀀스를 재표집하면 한 사람의 118개 윈도우를 118개의 독립 관측으로 취급하게 되어
  신뢰구간이 지나치게 좁아진다.

### A-21. StratifiedGroupKFold 기본 5-fold, 층화 기준은 이진 라벨 — 영향 **중간**
- 논문에 해당 실험이 없으므로 전부 이 저장소의 설계다.
- Dem 12명이 fold에 고르게 퍼지길 원하면 `split.stratify_on: diagnosis`로 바꾼다.
- config: `split.outer_k`, `split.inner_k`, `split.n_repeats`, `split.stratify_on`

### A-22. SHAP은 outer fold의 held-out 자료에서 계산 — 영향 **높음(해석)**
- 논문 §4.3은 5일 LSTM 한 개를 전체 자료 위에서 Deep SHAP으로 설명했다.
- 이 저장소의 기본값은 `mode: out_of_fold`이며, 논문 방식은 `mode: paper_style`로
  별도 제공하되 "일반화 가능한 변수 중요도가 아니다"라는 주석을 결과에 함께 남긴다.
- config: `explainability.*` (이번 작업 범위에서는 실행하지 않음)

---

## 가정이 아닌 것 — 논문에서 확정된 항목

혼동을 막기 위해 함께 적는다.

- LSTM 128 units, Dense 64 units, Adam, learning rate 0.001 (§4.2)
- grid search 범위: LSTM {64,128,256}, Dense {32,64,128}, lr 0.001–0.01 (§4.2)
- threshold 0.5 (§4.2, Table 6)
- early stopping 적용 (§5 한계)
- 비교모델 하이퍼파라미터는 H2O AutoML 3.46.0.1로 결정 (§4.2)
- test = 각 피험자의 마지막 1주일, 나머지는 train (§4.2)
- undersampling은 **training set에만** 적용 (§4.2)
- 시퀀스 길이 3, 4, 5일 (§4.1, §4.2)
- NC vs {MCI, DE} 이진 통합 (§3.2)
- Deep SHAP을 5일 모델에 적용 (§3.3, §4.3)
- 입력 32개 변수, Table 4/A1에 목록과 정의 (§3.2, Appendix A)
