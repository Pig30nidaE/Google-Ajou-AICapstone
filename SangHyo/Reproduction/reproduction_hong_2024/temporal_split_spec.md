# 시간분할 명세

구현: `src/splits/temporal.py`. 계약 테스트: `tests/test_raw_date_overlap.py`.

이 분할은 **Estimand A**를 추정한다. 같은 피험자가 train과 test 양쪽에 등장하는
것은 버그가 아니라 정의다. `estimand_definition.md` 참조.

---

## 1. 논문의 서술

> "To ensure a rigorous evaluation of the model, test data were constructed by
> isolating the final week of data from each subject within the entire dataset,
> whereas the remaining data were used for model training." (§4.2)

여기서 확정되는 것과 확정되지 않는 것:

| 항목 | 확정 여부 |
| --- | --- |
| test = 피험자별 마지막 1주일 | **확정** |
| train = 그 이전 전부 | **확정** |
| undersampling은 train에만 | **확정** |
| "1주일"이 달력일인지 레코드 수인지 | 미보고 (A-05) |
| validation 기간의 위치 | 미보고 (A-06) |
| 시퀀스 절단과 분할의 선후 | **모호** — literal variant로 별도 측정 |

## 2. 절차 (실험 A)

```
피험자별로:
  1. 날짜 오름차순 정렬
  2. cut = 마지막 날짜 - 6일          # 마지막 7개 달력일
  3. test  = date >= cut
  4. train = date <  cut
전체 피험자에 대해 concat
  ↓
train 프레임 안에서만 시퀀스 생성
test  프레임 안에서만 시퀀스 생성
```

**순서가 핵심이다.** 원시 날짜를 먼저 나누고, 그 다음에 각 구간 안에서만 윈도우를
만든다. 경계를 넘는 윈도우는 제거되는 것이 아니라 애초에 만들어지지 않는다.

## 3. 실험 B1이 추가하는 것

```
피험자별로:
  1. cut = 마지막 날짜 - 6일
  2. test = date >= cut
  3. embargo: cut - (L-1)일 이후의 train 날짜를 버린다
  4. validation = 남은 train 기간의 마지막 14일
  5. train = 그 이전
```

### embargo가 필요한 이유

embargo가 없으면 train의 마지막 윈도우와 test의 첫 윈도우가 **날짜를 공유하지는
않지만 하루 차이로 인접**한다. 수면지표는 하루 단위 자기상관이 강하므로, 인접한
윈도우는 사실상 같은 정보를 담는다.

길이 L 윈도우는 최대 L-1일 뒤를 참조하므로 embargo를 L-1일로 두면 train 윈도우가
닿을 수 있는 가장 마지막 날과 test 윈도우가 닿을 수 있는 가장 이른 날 사이에 실제
간격이 생긴다.

`audit_temporal_split`은 `embargo_days >= L-1`이 아니면 경고를 남긴다.

### embargo의 비용 (실측)

| 길이 | 실험 A train 시퀀스 | 실험 B1 train 시퀀스 | 감소 |
| ---: | ---: | ---: | ---: |
| 3일 | 9,028 | 6,947 | −23.0% |
| 4일 | 8,212 | 6,180 | −24.7% |
| 5일 | 7,494 | 5,513 | −26.4% |

test 시퀀스 수는 변하지 않는다(590 / 426 / 294). embargo와 validation은 train에서만
떼어 간다.

## 4. 분할 결과 (실측, `final_week_mode: calendar_days`)

| 항목 | 값 |
| --- | --- |
| train 행 | 11,128 |
| test 행 | 1,043 |
| train/test 피험자 | 174 / 174 (설계상 동일) |
| test 날짜 범위 | 2020-12-03 ~ 2021-02-17 |
| 피험자별 test 일수 | 최소 1일, 중앙값 6일, 최대 7일 |
| (피험자, 날짜) 중복 | **0** |

test 일수가 1일인 피험자가 존재하는 이유: 마지막 7개 달력일 중 실제 기록이 있는
날만 test가 되기 때문이다. 마지막 기록 직전에 긴 공백이 있으면 test가 얇아진다.

## 5. 평가 가능한 피험자 (중요)

| 길이 | test에서 연속 L일을 만들 수 있는 피험자 |
| ---: | --- |
| 3일 | 156 / 174 |
| 4일 | 132 / 174 |
| 5일 | **111 / 174** |

**5일 모델의 test 시퀀스는 174명 중 111명에게서만 나온다.** 논문의 5일 성능이 가장
높다는 결과를 볼 때 이 사실이 중요하다. 5일 조건은 "마지막 주에 연속 5일을 기록한
사람"만 평가하므로, 착용 순응도가 높은 부분집합으로 평가군이 좁아진다. 3일과 5일의
성능 차이에는 모델 차이뿐 아니라 **평가 대상 집단의 차이**가 섞여 있을 수 있다.

논문은 이 점을 보고하지 않는다. `unresolved_questions.md` Q-05 참조.

## 6. 두 해석의 차이 (A-05)

| 항목 | `calendar_days` (기본) | `record_count` |
| --- | --- | --- |
| 정의 | 마지막 7개 달력일 | 마지막 7개 레코드 |
| test 일수 | 1~7일 (사람마다 다름) | 항상 7일 |
| L=3 test 시퀀스 | 590 | 669 |
| L=4 | 426 | 477 |
| L=5 | 294 | 321 |

`record_count`는 모든 피험자가 같은 수의 test 레코드를 갖지만, 그 7개가 달력상
연속이 아닐 수 있다. 어느 쪽이 논문의 해석인지는 확정할 수 없다.

## 7. 자동 검사

`audit_temporal_split`이 학습 전에 실행하고, 실패하면 즉시 예외를 던진다.

| 검사 | 내용 |
| --- | --- |
| `train_test_days_disjoint` | (피험자, 날짜) 교집합이 0 |
| `validation_days_disjoint` | validation이 train/test와 겹치지 않음 |
| `test_is_strictly_later_than_train` | 피험자별로 train 마지막날 < test 첫날 |
| `embargo_covers_window` | embargo >= L-1 (아니면 경고) |

## 8. 이 분할로 말할 수 있는 것과 없는 것

**말할 수 있다:** 이미 몇 달간 관찰된 사람의 다음 주 자료로 그 사람의 인지상태를
분류하는 성능.

**말할 수 없다:** 처음 보는 사람을 선별하는 성능. 그 값은 실험 B2/C에서 나온다.
