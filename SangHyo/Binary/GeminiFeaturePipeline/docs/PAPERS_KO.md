# `papers/LLM_API/` 논문 분석과 반영 내역

5편 모두 본문·방법·프롬프트·평가 설정을 확인했다. 아래 "반영"은 코드 주석에도 같은
출처를 남겼다.

## 1. Explainable cognitive decline detection in free dialogues (Applied Intelligence, 2024)

* **구조**: LLM(gpt-3.5-turbo)이 대화에서 진단이 아닌 고수준 특징 26개를 뽑고,
  전통 ML(RF 등)이 최종 분류. "LLM = 특징 추출기, 분류는 별도 모델"이라는 본 프로젝트의
  설계와 동일한 구도다.
* **출력 형식**: 프롬프트가 고정 JSON 스키마를 제시하고 값은 대부분 (0,1) 범위 실수.
  프롬프트 말미에 "ALWAYS RETURN A JSON IN THE GIVEN FORMAT WITHOUT ADDING MORE TEXT
  OR MODIFYING THE FIELD NAMES" 같은 강한 형식 지시를 둔다.
* **집계**: 발화 단위로 LLM을 호출한 뒤 mean/max/min/사분위수로 세션 단위 특징을 만든다.
* **평가**: 10-fold CV, GridSearchCV.
* **반영**: 0.0~1.0 고정 JSON 스키마, "JSON 외 텍스트 금지" 지시,
  프롬프트를 context/task/rules/output 4부로 나눈 구성(`prompts.py`).
* **미반영**: 발화 단위 반복 호출 후 통계 집계. 우리 입력은 대화가 아니라 수치 시계열이라
  Python이 통계를 정확히 계산하는 편이 낫고, 호출 수를 174건으로 억제할 수 있다.

## 2. LLMs can construct powerful representations... (Rubric Representation Learning, 2026)

* **구조**: LLM이 소수 예시를 보고 **global rubric**(모든 샘플에 공통 적용되는 필드
  명세)을 만들고, 그 rubric으로 각 환자 기록을 표준화한 뒤 downstream 모델에 넣는다.
  EHRSHOT 15개 임상 과제에서 count-feature GBM, naive text serialization, 임상
  foundation model을 모두 상회했다.
* **핵심 지시문**: rubric 적용 프롬프트가 "Extract facts only... It must NOT make
  predictions, assign risk levels, or draw conclusions", "Do NOT add commentary,
  predictions, risk assessments, or conclusions"로 못 박는다.
* **장점 논거**: 모든 샘플이 같은 필드 구조를 공유하므로 감사 가능하고, 표 형태로
  변환해 일반 ML 기법을 그대로 쓸 수 있으며, 호출 비용이 O(1)에 가깝다.
* **반영**: (a) 모든 피험자에게 동일한 고정 스키마를 적용, (b) "사실만 추출, 예측·위험도·
  결론 금지" 문구를 시스템 프롬프트에 그대로 채택, (c) 결과를 즉시 표 형태 수치 특징으로
  사용.
* **의도적 미반영**: 이 논문은 rubric을 **라벨 균형 코호트(양성 20 + 음성 20)**를 보여
  합성한다. 본 프로젝트의 `<critical_label_rule>`은 어떤 형태의 라벨 노출도 금지하므로,
  스키마를 라벨로부터 합성하지 않고 도메인 지식과 데이터 구조만으로 사전 고정했다.
  (그 대가로 "이 표본에 최적화된 필드"는 얻지 못한다 — 한계로 문서화.)

## 3. Predicting explainable dementia types with LLM-aided feature engineering (Bioinformatics, 2025)

* **구조**: 임상 교과서에서 개념(feature) 목록을 뽑고, LLM이 임상 노트를 그 개념
  벡터로 변환한 뒤 **선형 분류기**로 예측. 정확도 0.72로 n-gram 로지스틱 0.64와
  GPT-4 직접 예측 0.48을 앞섰다.
* **시사점**: (a) LLM 직접 예측보다 "LLM 특징 + 단순 분류기"가 낫다는 직접 근거,
  (b) 특징 목록을 외부 지식에서 고정해도 충분하다는 근거, (c) 임베딩으로 비용을 97%
  줄인 변형도 제시.
* **반영**: downstream 기본 모델로 규제 로지스틱 회귀를 포함(복잡한 앙상블 대신),
  특징 목록을 외부 지식 기반으로 사전 고정.

## 4. DeepFeature (IMWUT/UbiComp, 2026) — 웨어러블 생체신호 특징 생성

* **구조**: LLM이 과제 설명·전문가 지식·특징 상호작용을 결합해 특징 후보를 생성하고,
  이를 코드로 번역해 실행, 표 특징을 만들어 ML 모델에 투입. 8개 과제에서
  sample-level/subject-level 모두 최고 평균 AUROC.
* **주의점**: 검증 성능 피드백으로 특징을 재선택하는 **iterative refinement**를 쓴다.
  이는 평가 데이터로 특징을 고르는 구조이므로 그대로 쓰면 누수다.
* **반영**: 웨어러블 도메인에서 subject-level 평가를 기본으로 삼는다는 점,
  다중 소스(도메인 지식 + 신호 통계)를 결합한다는 점.
* **의도적 미반영**: 성능 피드백 기반 반복 특징 재선택. 도입한다면 반드시 학습 fold
  내부에서만 수행해야 하며, 이번 단계에서는 구현하지 않고 확장 지점으로만 남긴다.

## 5. FeatLLM (ICML 2024)

* **구조**: few-shot 표 데이터에서 LLM이 클래스 판별 규칙을 만들고, 규칙을 코드로 파싱해
  이진 특징을 생성한 뒤 선형 모델을 학습. 특징 배깅 + 앙상블로 안정화.
* **결정적 제약**: 규칙 생성 프롬프트에 **라벨이 붙은 few-shot 예시**가 들어간다.
  본 프로젝트에서는 금지된다.
* **반영**: "LLM이 만든 특징 + 저복잡도 downstream 모델"이라는 뼈대, 그리고 반복 호출
  결과를 평균해 분산을 줄이는 아이디어(`gemini.repeat_calls` 인터페이스로만 제공,
  기본값 1).
* **미반영**: 라벨 기반 규칙 생성 전부.

## 6. 요약: 이번 구현이 취한 입장

| 논문 아이디어 | 채택 여부 | 이유 |
| --- | --- | --- |
| LLM = 특징 추출기, 별도 분류기 | 채택 | 5편 중 4편이 지지, 프로젝트 요구사항과 일치 |
| 고정 JSON 스키마 + 0~1 실수 | 채택 | 1번 논문 형식, 검증·캐시·표 변환이 쉬움 |
| 모든 샘플 공통 global 스키마 | 채택 | 2번 논문의 감사가능성·비용 논거 |
| "예측·위험도·결론 금지" 지시문 | 채택 | 2번 논문 문구를 그대로 사용 |
| 라벨 예시로 스키마/규칙 합성 | **거부** | `<critical_label_rule>` 위반 |
| 검증 성능 피드백 기반 특징 재선택 | **거부** | 평가 데이터 기반 선택은 누수 |
| LLM 직접 진단/확률 출력 | **거부** | 3번 논문에서도 성능이 가장 낮았고, 요구사항에서 금지 |
