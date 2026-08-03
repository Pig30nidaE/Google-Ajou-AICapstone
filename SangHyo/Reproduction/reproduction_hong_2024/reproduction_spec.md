# 재현 명세

대상: Hong et al. (2024), *Prediction of Cognitive Impairment Using Sleep Lifelog
Data and LSTM Model*, Mathematics 12(20), 3208.

**재현 등급: reported-method reconstruction** (exact reproduction 아님).
근거는 §6.

---

## 1. 논문에서 확정적으로 확인된 방법

인용 위치를 함께 적는다. 이 항목들은 가정이 아니다.

| # | 내용 | 위치 |
| ---: | --- | --- |
| 1 | 과제는 NC vs {MCI, DE} 이진분류 | §3.2 |
| 2 | 입력은 수면 변수 32개, 목록은 Table 4, 정의는 Table A1 | §3.2, Appendix A |
| 3 | 시퀀스 길이 3, 4, 5일 | §4.1, §4.2 |
| 4 | test = 각 피험자의 마지막 1주일, 나머지가 train | §4.2 |
| 5 | undersampling은 **training set에만** 적용 | §4.2 |
| 6 | grid search 범위: LSTM {64,128,256}, Dense {32,64,128}, lr 0.001–0.01 | §4.2 |
| 7 | 최종 구조: LSTM 128 → Dense 64 → Dense 1, Adam, lr 0.001 | §4.2 |
| 8 | 분류 threshold 0.5 | §4.2, Table 6 |
| 9 | early stopping 적용 | §5 (한계) |
| 10 | 비교모델 SVM / LR / RF / XGBoost | §4.2 |
| 11 | 비교모델 하이퍼파라미터는 H2O AutoML **3.46.0.1**로 결정 | §4.2 |
| 12 | SHAP은 **Deep SHAP**, 5일 모델에 적용 | §3.3, §4.3 |
| 13 | 보고 성능은 Table 5, precision@100 = 0.96 | Table 5, Figure 5 |
| 14 | 데이터는 174명 / 12,183건 / CN 111·MCI 51·Dem 12 | Table 3 |
| 15 | 데이터는 이미 정제되어 추가 정제 불필요 | §4.1 |

## 2. 논문에서 확인되지 않은 설정

전체 목록은 `assumptions.md`(A-01 ~ A-22). 영향이 큰 것만 여기 요약한다.

| 항목 | 왜 중요한가 |
| --- | --- |
| **시퀀스 절단과 분할의 선후** | 문언대로면 5일에서 test 날짜의 62.3%가 train에 존재 |
| **validation 자료의 출처** | test를 겸용했다면 보고 성능이 낙관적 |
| **undersampling의 단위와 비율** | 피험자당 시퀀스 수가 2~118개로 편차가 큼 |
| **비교모델의 입력 변환** | LSTM 우위 주장의 대조군을 결정 |
| **stride** | 시퀀스 수와 중첩 정도를 결정 |
| **달력 공백 처리** | 5일 윈도우의 28.6%가 실제로는 불연속 |
| **"final week"의 정의** | 달력 7일 vs 레코드 7개 |
| batch size, epoch, dropout, patience | 학습 동역학 전반 |

## 3. 데이터 대조 결과

`run.py --inspect-data` 실측. **논문 Table 3의 7개 수치가 전부 일치**한다.

| 항목 | 논문 | 실측 |
| --- | ---: | ---: |
| 피험자 | 174 | 174 |
| CN / MCI / Dem | 111 / 51 / 12 | 111 / 51 / 12 |
| 수면 레코드 | 12,183 | 12,183 |
| 피험자별 기록 | 35~122 | 35~122 |
| 입력 변수 | 32 | 32 |

파생변수 공식도 수치로 검증했다(`paper_data_mapping.md` §3). 이름이 다른 매핑
`sleep_hr_min ← sleep_hr_lowest`는 일치율 1.000으로 확인했다.

논문이 보고하지 않은 구조적 사실 3가지를 발견했다:
- 174명 중 **162명**의 기록에 달력 공백이 있다(총 1,089건).
- 같은 날짜에 수면 레코드가 2개인 경우가 24행(11명) 있다.
- `sleep_temperature_delta`와 `_deviation`이 **완전히 동일한 열**이다.

## 4. 실험 구성

| 실험명 | Estimand | 분할 | 선택 | config |
| --- | :---: | --- | --- | --- |
| `paper_temporal_reconstruction` | A | 피험자별 마지막 1주일 + train-side 14일 validation(A-06) | 논문 설정 고정 | `paper_temporal_{3,4,5}day.yaml` |
| `paper_literal_variant` | A + 누수 | 시퀀스 먼저, 그 뒤 분할 | 논문 설정 고정 | `paper_literal_variant.yaml` |
| `strict_same_subject_temporal` | A | 위 + L-1일 embargo | 논문 설정 고정 | `strict_same_subject_temporal.yaml` |
| `fixed_subject_independent` | B | 5-fold StratifiedGroupKFold | 논문 설정 고정 | `fixed_subject_independent.yaml` |
| `nested_subject_independent` | B | outer 5 × inner 3 Group CV | inner CV에서만 | `nested_subject_independent.yaml` |

`paper_literal_variant`의 결과는 **누수 진단값**이며 성능이 아니다.

### 한 번에 하나씩만 바꾸는 사다리

```
A ──(누수 통제)──▶ B1 ──(estimand A→B)──▶ B2 ──(모델 선택 비용)──▶ C
```

- A → B1: 둘 다 estimand A. 차이 = **순수한 누수 제거 효과**
- B1 → B2: 둘 다 누수 통제됨. 차이 = **질문이 바뀐 효과**
- B2 → C: 둘 다 estimand B. 차이 = **모델 선택 비용**

A와 B2를 직접 빼서 "누수 크기"라고 부르면 두 효과가 섞인다.

## 5. 파이프라인

```
AI-Hub CSV (12,183행)
  ↓ 날짜 키 = bedtime_end, (피험자,날짜) 중복 제거 → 12,171행
  ↓ 32개 변수 구성 (원본 17 + 5분계열 파생 4 + 시각 one-hot 12 - 중복조정)
일별 표: 174명 × 32변수, raw_row_id 부여
  ↓ ★ 먼저 분할 (원시 날짜 또는 피험자)
  ↓ ★ 각 split 안에서만 시퀀스 생성 (연속 calendar day만, stride 1)
  ↓ ★ train 후보에서 validation 분리(temporal LSTM, A-06)
  ↓ ★ train에만 undersampling
  ↓ ★ train에만 scaler fit
  ↓ 누수 검사 13종 (실패 시 즉시 중단)
학습 → 시퀀스 단위 + 피험자 단위 평가 → bootstrap CI → 비교표
```

★ 표시가 누수 통제 지점이다.

## 6. 왜 exact reproduction이 아닌가

1. 논문 코드가 공개되지 않았다.
2. 시퀀스 절단과 분할의 선후가 확정되지 않는다(Q-03).
3. validation 자료의 출처가 확정되지 않는다(Q-09).
4. 논문은 프레임워크를 명시하지 않았고 이 저장소는 PyTorch를 쓴다. 실제 저자
   프레임워크와 초기화·게이트 기본값이 달랐을 수 있어 소수점 일치를 보장할 수 없다.
5. 비교모델의 입력 변환이 미보고다(Q-10).
6. H2O AutoML이 무엇까지 선택했는지 미보고다(Q-11).
7. batch size, epoch, dropout, seed가 전부 미보고다.

따라서 **보고된 방법을 재구성한 것**이며, 수치가 Table 5와 다르더라도 그 자체가
논문의 오류를 뜻하지는 않는다. 반대로 수치가 비슷하더라도 같은 구현이라는 증거는
아니다.

## 7. 산출물

| 파일 | 내용 |
| --- | --- |
| `LAUNCHER_STATUS.json` | 실행 상태, estimand, headline AUC와 그 단위 |
| `FINAL_REPORT.json` | 전체 결과 |
| `TRAINING_COMPLETE.json` | 완료 표식 |
| `predictions_hashed.csv` | 시퀀스별 예측 (이메일은 SHA-256 해시) |
| `audit/audit_log.json` | 모든 누수 검사 기록 |
| `comparison.md` | 표 1·2·3과 차이값 해석 |
| `dry_run_report.json` | `--dry-run` 산출물 |

원본 이메일은 어떤 결과 파일에도 저장하지 않는다(`SangHyo/AGENTS.md` §6).

## 8. 이 재현으로 답할 수 있는 질문

- 논문의 검증방식을 누수 없이 다시 구현했을 때 Table 5의 수치가 재현되는가? (A)
- 경계 시퀀스와 전처리 누수를 제거하면 성능이 얼마나 달라지는가? (A vs B1)
- 같은 모델을 **신규 피험자**에 적용하면 어떤 성능이 나오는가? (B2)
- 모델 선택 비용까지 포함하면 얼마나 남는가? (C)
- SHAP이 지목한 변수(호흡수·HRV·REM·깊은수면·뒤척임)가 fold를 넘어 안정적인가?

## 9. 답할 수 없는 질문

- 논문 저자가 실제로 어떤 코드를 돌렸는가.
- 이 코호트 밖의 사람에게 일반화되는가. 174명은 한 코호트이며, 외부 검증은 없다.
- AI-Hub의 in-house 정제가 라벨을 참조했는가(Q-02).
