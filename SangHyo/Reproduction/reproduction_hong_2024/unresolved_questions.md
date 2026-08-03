# 미해결 질문

논문·데이터·코드를 다 확인해도 **확정할 수 없는** 항목이다. 각 항목에 현재 처리
방식과 확인 방법을 함께 적는다.

---

## Q-01. 저자가 사용한 AI-Hub 배포본이 이 저장소의 것과 같은가

- 논문 Data Availability는 2023-05-01 접근만 밝히고 버전 표기가 없다.
- 이 저장소의 사본으로 Table 3의 7개 수치가 **전부 일치**한다(174 / 111 / 51 / 12 /
  12,183 / 35 / 122). 우연의 일치일 가능성은 낮다.
- **현재 처리:** 같은 배포본으로 간주한다(A-03).
- **확인 방법:** 저자에게 배포본 버전 문의. 또는 AI-Hub 변경 이력 확인.

## Q-02. "already cleaned" 전처리가 라벨을 참조했는가

- §4.1: "The dataset we received was already cleaned, with all missing data or
  outliers handled in-house. Consequently, no additional data cleaning was
  required."
- 그 in-house 처리가 진단 라벨을 보고 이뤄졌다면, 어떤 분할 설계도 그 누수를
  되돌릴 수 없다.
- 관측 사실: 32개 변수에 결측이 **0개**다. 이 정도로 깨끗한 웨어러블 데이터는
  흔치 않다. 어떤 규칙으로 채웠는지 알 수 없다.
- **현재 처리:** 결측이 발견되면 로더가 즉시 중단하고, 전체 데이터에 imputer를
  fit하지 않는다. 현재는 결측이 없어 이 경로가 발동하지 않는다.
- **확인 방법:** AI-Hub 데이터 설명서의 전처리 절차 확인.

## Q-03. 논문은 시퀀스를 만든 뒤에 분할했는가, 분할한 뒤에 만들었는가

- §4.2의 서술 순서는 "전체를 시계열로 변환 → 마지막 1주일을 test로 분리"다.
  문자 그대로 읽으면 시퀀스가 먼저다.
- 그러나 이는 서술 순서일 뿐 구현 순서라는 증거가 아니다. 코드가 공개되어 있지 않다.
- **현재 처리:** 기본은 분할 우선(`paper_temporal_reconstruction`), 문언 그대로는
  `paper_literal_variant`로 별도 측정. 두 값을 모두 보고한다.
- **영향:** 크다. 문언대로면 5일에서 test 날짜의 62.3%가 train에 이미 존재한다.
- **확인 방법:** 저자에게 코드 요청.

## Q-04. "final week"은 달력 7일인가 레코드 7개인가

- 이 코호트에서 두 해석이 다른 결과를 낸다(A-05). L=5에서 test 시퀀스 294 vs 321.
- **현재 처리:** `calendar_days`가 기본, `record_count`가 config 대안.
- **확인 방법:** 저자 문의. 또는 논문의 test 시퀀스 수가 보고되었다면 역산 가능하나,
  논문은 test 표본 수를 보고하지 않는다.

## Q-05. 논문의 3일/4일/5일 성능은 같은 평가군에서 나온 값인가

- 이 저장소의 실측: 마지막 1주일 안에 연속 L일을 만들 수 있는 피험자가
  L=3에서 156명, L=4에서 132명, **L=5에서 111명**이다.
- 즉 5일 모델은 착용 순응도가 높은 부분집합에서만 평가된다. 논문이 보고한
  "5일이 가장 좋다"(AUC 0.88 → 0.91 → 0.92)에는 모델 차이와 **평가군 차이**가 섞여
  있을 수 있다.
- 논문은 각 길이의 test 표본 수를 보고하지 않아 확인할 수 없다.
- **현재 처리:** 모든 결과에 `thin_subjects` 블록으로 평가 가능 피험자 수를 함께
  기록한다. 길이 간 비교 시 이 값을 반드시 병기한다.
- **왜 중요한가:** 이 저장소가 재현한 값에서도 같은 현상이 나타날 것이므로, 길이별
  성능 차이를 "긴 시퀀스가 더 좋다"로만 해석하면 안 된다.

## Q-06. 논문의 precision@100 = 0.96은 어떤 단위인가

- Equation 11의 분모는 "Top k **people**"이라고 적혀 있지만, 모델 출력은 시퀀스
  단위다. 그림 5의 캡션도 "Precision@K score of the models (K = 100)"뿐이다.
- test 시퀀스가 294개(5일)인 상황에서 상위 100개가 시퀀스인지 사람인지에 따라
  의미가 완전히 다르다. 사람이라면 test 피험자가 111명이므로 상위 100명은 거의
  전부다.
- **현재 처리:** 시퀀스 단위와 피험자 단위 모두 계산하고, `k_was_truncated`
  플래그로 k가 표본보다 큰 경우를 표시한다.

## Q-07. `sleep_temperature_delta`와 `_deviation`이 원래 같은 값이었는가

- 이 배포본에서는 12,171행 전부 동일하다. Table A1은 두 변수를 다르게 설명한다.
- 저자가 받은 배포본에서도 같았다면, 논문의 32개 변수 중 31개만 서로 다른 정보다.
- **현재 처리:** 논문 재현을 위해 두 변수를 모두 유지하고, `--inspect-data`가 매번
  이 중복을 보고한다.
- **영향:** 성능에는 거의 영향이 없지만, SHAP 변수 중요도가 두 변수로 쪼개져
  각각의 기여가 절반으로 보일 수 있다.

## Q-08. `rmssd_average`의 정확한 계산식

- Table A1은 "Average heart rate variability"만 적었다.
- 5분 계열 평균과 사전집계 `sleep_rmssd` 컬럼이 r=0.997이지만 정확 일치는 38.2%.
- **현재 처리:** 5분 계열 평균이 기본, 컬럼 사용이 대안(A-04).
- **왜 중요한가:** SHAP에서 HRV가 2위 변수로 보고되었다. 계산식이 다르면 그 순위의
  재현 여부에 직접 영향을 준다.

## Q-09. 논문이 사용한 validation 자료는 어디서 왔는가

- §4.2는 "best balance between accuracy and validation loss", §5는 early stopping
  적용을 말한다. 그러나 §4.2의 분할 서술에는 train과 test만 있다.
- 가능성: (a) train에서 일부를 뗐다, (b) test를 validation으로 겸용했다,
  (c) Keras `validation_split`을 썼다.
- (b)라면 보고된 성능은 낙관적이다.
- **현재 처리:** 실험 A/B1은 train 끝 14일을 명시적 validation으로 사용한다.
  이 기간은 논문 미보고 가정으로 결과에 기록한다. A'·B2·C outer refit은 별도
  monitor가 없어 early stopping을 끈다. 어느 경우에도 test를 monitor로 쓰지
  않는다(A-06).

## Q-10. 비교모델(SVM/LR/RF/XGBoost)의 입력 형태

- 논문은 이 모델들이 "do not reflect time-series characteristics"라고만 적었다.
  3~5일 윈도우를 한 행으로 어떻게 만들었는지 없음.
- flatten / mean / last_day / summary 중 어느 것인지에 따라 비교모델 성능이 크게
  달라진다. 논문의 XGBoost AUC 0.81은 이 선택에 의존한다.
- **현재 처리:** `flatten`이 기본, 나머지는 config 대안(A-09). 어느 것도 "논문
  방식"이라고 단정하지 않는다.

## Q-11. H2O AutoML이 무엇까지 선택했는가

- §4.2: "We employed H2O ... to determine the optimal hyperparameters for each
  model based on the length of the time-series data."
- AutoML은 보통 모델군까지 고른다. "각 모델의 하이퍼파라미터만"이라면 알고리즘을
  고정한 채 돌렸다는 뜻인데, 그 설정은 보고되지 않았다.
- AutoML이 자체 CV를 어떤 분할로 돌렸는지도 알 수 없다. 시퀀스 단위 무작위 CV였다면
  같은 피험자가 fold를 넘나들었을 것이다.
- **현재 처리:** 실험 A의 LR/RF/XGBoost는 H2O 3.46.0.1을 필수로 하고 training
  fold만 전달한 뒤 요청 family(GLM/DRF/XGBoost)를 고정한다. H2O AutoML로
  재현할 수 없는 SVM은 sklearn 경로임을 명시한다. 어느 backend·버전·family가
  돌았는지 결과에 기록한다(A-17). 따라서 SVM 행은 여전히 exact-method 재현이 아니다.

## Q-12. 논문 Table 5의 F1이 precision·sensitivity와 정합하는가

- F1은 precision과 sensitivity만으로 결정된다. `verify_paper_arithmetic()`이 매
  dry-run에서 이를 확인한다.
- **현재 결과: 7개 모델 전부 정합**(허용오차 0.015). 표 내부에 모순은 없다.
- 다만 정합한다는 것이 값이 옳다는 뜻은 아니다. 같은 실행에서 나온 값들이라는
  최소한의 확인일 뿐이다.

## Q-13. LSTM의 시퀀스 요약 방식

- Figure 1은 표준 LSTM 셀만 보여 준다. 마지막 hidden state인지, 전체 timestep의
  pooling인지, `return_sequences=True` 후 Flatten인지 알 수 없다.
- **현재 처리:** 마지막 hidden state(A-16). 서술 "an LSTM layer with 128 units,
  followed by a dense layer with 64 units"에 가장 직접적으로 대응한다.

---

## 저자 문의 시 우선순위

1. **Q-03** (시퀀스/분할 순서) — 재현 결과 해석을 가장 크게 바꾼다.
2. **Q-09** (validation 출처) — 보고 성능의 낙관 정도를 결정한다.
3. **Q-05** (길이별 평가군) — "5일이 최고"라는 결론의 근거에 직결된다.
4. **Q-10 / Q-11** (비교모델 설정) — LSTM 우위 주장의 대조군이다.
5. **Q-04** (final week 정의), **Q-08** (rmssd 계산식)
