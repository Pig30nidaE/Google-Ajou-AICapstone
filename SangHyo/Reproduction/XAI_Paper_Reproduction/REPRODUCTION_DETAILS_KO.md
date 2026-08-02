# XAI 논문 재현 상세 기록

대상 논문:

`docs/설명가능 인공지능을 활용한 라이프로그 기반 치매 위험도 산정 방법에 관한 연구.pdf`

이 문서는 현재 구현된 `training/XAI_Paper_Reproduction/` 파이프라인이 전처리와 학습을 어떻게 수행하는지, 논문이 명시한 내용과 명시하지 않은 내용을 어떻게 재현 결정으로 채웠는지 기록한다.

## 1. 현재 재현 상태 요약

현재 `outputs/` 기준으로 주요 논문 지표는 다음과 같이 재현되었다.

| 항목 | 논문 | 재현 결과 | 판정 |
| --- | ---: | ---: | --- |
| daily row 수 | 12,183 | 12,183 | 일치 |
| subject 수 | 174 | 174 | 일치 |
| CN daily row 수 | 7,737 | 7,737 | 일치 |
| MCI/Dem daily row 수 | 4,446 | 4,446 | 일치 |
| baseline LightGBM ROC-AUC | 0.9010 | 0.9004 | 거의 일치 |
| SHAP forward selection best | top 40, 0.9037 | top 28, 0.9038 | 성능은 일치권, 최적 feature 수는 다름 |
| 재현 top 40 ROC-AUC | 0.9037 | 0.9000 | 근접 |
| 최종 LightGBM ROC-AUC | 0.9492 | 0.9478 | 거의 일치 |
| DRS 검정 방향 | impaired > CN | impaired > CN | 일치 |
| DRS one-sided t-test | 유의 | p=0.0 | 일치 |

결론적으로, 논문에서 정량적으로 제시한 핵심 결과인 데이터 수, LightGBM baseline 성능, 최종 LightGBM 성능, DRS 검정은 매우 가깝게 재현되었다. 다만 baseline 중 MLP, SVM, KNN은 논문보다 훨씬 높게 나왔는데, 이는 논문이 해당 모델들의 세부 전처리와 하이퍼파라미터를 공개하지 않았기 때문에 완전 동일 재현이라고 보기는 어렵다.

## 2. 파일 구조와 실행 단위

구현 위치:

```text
training/XAI_Paper_Reproduction/
├── README.md
├── REPRODUCTION_DETAILS_KO.md
├── PAPER_IMPLEMENTATION_NOTES.md
├── XAI_Paper_Reproduction_Colab.ipynb
├── requirements.txt
├── src/
│   └── xai_paper_reproduction.py
└── scripts/
    ├── 00_validate_environment.py
    ├── 01_preprocess_daily_binary.py
    ├── 02_compare_baseline_models.py
    ├── 03_shap_forward_selection.py
    ├── 04_final_lgbm_shap_drs.py
    ├── 05_make_reproduction_report.py
    └── 06_audit_reproduction_outputs.py
```

실행 흐름:

1. `00_validate_environment.py`
   - 패키지, 원본 데이터 파일, 전처리 smoke test 확인
2. `01_preprocess_daily_binary.py`
   - 원본 activity/sleep/label CSV를 daily binary lifelog table로 변환
3. `02_compare_baseline_models.py`
   - 논문 Table 1의 7개 baseline 모델을 5-fold CV로 비교
4. `03_shap_forward_selection.py`
   - LightGBM 기반 SHAP importance 산출 후 feature 수를 1개씩 늘리며 forward selection
5. `04_final_lgbm_shap_drs.py`
   - 논문 Table 2의 LightGBM 파라미터로 최종 CV 평가, 전체 학습, SHAP/DRS 산출
6. `05_make_reproduction_report.py`
   - 산출물을 markdown report로 통합
7. `06_audit_reproduction_outputs.py`
   - 논문 재현 산출물이 완비됐는지 자동 검증

## 3. 원본 데이터 사용 방식

### 3.1 사용한 원본 파일

`128.치매 고위험군 라이프로그/` 아래에서 다음 파일을 사용한다.

| 용도 | 파일 |
| --- | --- |
| Training activity | `1.Training/원천데이터/1.걸음걸이/train_activity.csv` |
| Training sleep | `1.Training/원천데이터/2.수면/train_sleep.csv` |
| Training label | `1.Training/라벨링데이터/1.걸음걸이/training_label.csv` |
| Validation activity | `2.Validation/원천데이터/1.걸음걸이/val_activity.csv` |
| Validation sleep | `2.Validation/원천데이터/2.수면/val_sleep.csv` |
| Validation label | `2.Validation/라벨링데이터/1.걸음걸이/val_label.csv` |

라벨 파일은 `라벨링데이터/1.걸음걸이`의 `training_label.csv`, `val_label.csv`를 사용한다. 원본 데이터에는 수면/인지기능 쪽에도 label CSV가 존재하지만, 현재 구현은 activity와 같은 `1.걸음걸이` 라벨 경로를 기준으로 한다.

경로 검색은 macOS/Google Drive의 한글 유니코드 정규화 차이를 피하기 위해 경로 문자열과 필터 문자열을 NFC로 정규화해 비교한다.

### 3.2 원본 행 수와 daily row 구성

현재 원본 데이터 기준 확인값:

| 데이터 | 원본 row | 원본 column |
| --- | ---: | ---: |
| train_activity | 9,705 | 31 |
| val_activity | 2,478 | 31 |
| train_sleep | 9,705 | 36 |
| val_sleep | 2,478 | 36 |
| training_label | 141 | 2 |
| val_label | 33 | 2 |

전처리 후 daily table 구성:

| 항목 | 값 |
| --- | ---: |
| activity daily rows | 12,183 |
| sleep daily rows | 12,171 |
| inner merge rows | 12,150 |
| activity-left merge rows | 12,183 |
| left merge 후 sleep feature가 모두 결측인 row | 33 |
| subject 수 | 174 |

논문은 daily row 기준으로 CN 7,737건, 인지장애군 4,446건을 제시한다. 이 값은 activity daily row를 기준으로 할 때 정확히 일치한다. sleep daily row를 기준으로 inner join하면 row가 줄어 논문 count와 맞지 않는다. 따라서 재현에서는 activity를 기준 테이블로 두고 sleep feature를 left join한다.

## 4. 라벨 전처리

논문 명시:

- 정상인지군 `CN`과 인지장애군 `MCI`, `Dem`을 이진 분류 대상으로 사용
- 정상인지군을 0, 인지장애군을 1로 인코딩
- 정상인지군 daily row 7,737건, 인지장애군 daily row 4,446건

우리 구현:

| 원본 diagnosis | binary_class |
| --- | ---: |
| `CN` | 0 |
| `MCI` | 1 |
| `Dem` | 1 |
| `DEM` | 1 |
| `Dementia` | 1 |

subject 수준 라벨 분포:

| class | subject 수 |
| --- | ---: |
| CN | 111 |
| MCI | 51 |
| Dem | 12 |
| MCI/Dem 합계 | 63 |

daily row 라벨 분포:

| class | daily row |
| --- | ---: |
| CN, 0 | 7,737 |
| MCI/Dem, 1 | 4,446 |

라벨은 `patient_id`, `split` 기준으로 daily row에 병합한다. 병합 후 라벨이 없는 row가 있으면 오류를 발생시킨다.

## 5. Activity 전처리 상세

입력:

- `train_activity.csv`
- `val_activity.csv`

주요 처리:

1. `EMAIL`을 `patient_id`로 rename
2. `activity_day_start`에서 `sample_date` 생성
3. `activity_day_start`, `activity_day_end`는 24시간 기준 실수형 hour로 변환
   - 예: 13:30:00 -> 13.5
4. `activity_class_5min` 계열 slash sequence를 파싱
5. `activity_met_1min` 계열 slash sequence를 파싱
6. 원본 timestamp/string sequence column은 제거
7. 나머지 수치형 activity aggregate feature는 유지

`activity_class_5min`에서 만드는 feature:

- valid count
- class 0,1,2,3,4,5 각각의 count
- class 0,1,2,3,4,5 각각의 ratio
- transition count

`activity_met_1min`에서 만드는 통계 feature:

- mean
- std
- var
- min
- max
- median
- q25
- q75
- iqr
- count

논문 명시:

- 라이프로그 측정 시작/종료 시간처럼 모든 데이터가 동일한 값인 데이터 제거
- Time Stamp 포맷은 24시에 대응하는 실수값으로 변환
- 어노테이션이 기록된 활동 로그 시계열 데이터는 각 측정값 개수에 해당하는 형태로 변환

우리 재현 결정:

- 논문은 sequence feature를 정확히 어떤 통계량으로 만들었는지 모두 공개하지 않았다.
- 논문 문장에 맞춰 categorical activity class는 count 중심으로 만들고, ratio와 transition count를 추가했다.
- MET sequence는 연속량에 가까우므로 count 하나만으로 정보가 크게 손실된다. 따라서 평균, 분산, 분위수 등 통계량을 함께 생성했다.

## 6. Sleep 전처리 상세

입력:

- `train_sleep.csv`
- `val_sleep.csv`

주요 처리:

1. `EMAIL`을 `patient_id`로 rename
2. `sleep_bedtime_end`의 날짜를 `sample_date`로 사용
3. `sleep_bedtime_start`, `sleep_bedtime_end`를 24시간 기준 실수형 hour로 변환
4. `sleep_time_from_timestamp = sleep_bedtime_end - sleep_bedtime_start`를 초 단위로 계산
5. 같은 subject/date에 sleep row가 여러 개 있으면 가장 긴 sleep row를 대표 row로 선택
6. `sleep_hypnogram_5min` sequence를 count/ratio/transition feature로 변환
7. 원본 5분 단위 심박/수면 stage/RMSSD sequence column은 제거

`sleep_hypnogram_5min`에서 만드는 feature:

- valid count
- stage 1,2,3,4 각각의 count
- stage 1,2,3,4 각각의 ratio
- transition count

논문 명시:

- 결측치 처리가 불가능한 5분당 심박동 로그 데이터 등은 제거
- 수면 시작/종료 시간은 24시 대응 실수값으로 변환
- 수면 종료 시간에서 수면 시작 시간을 뺀 수면 시간 변수를 추가

우리 재현 결정:

- `sleep_hr_5min`, `sleep_rmssd_5min`, `sleep_hypnogram_5min` 원본 sequence column은 그대로 모델에 넣지 않는다.
- 단, `sleep_hypnogram_5min`은 수면 단계 분포 정보가 중요할 수 있으므로 count/ratio/transition으로 요약한 파생 feature만 사용한다.
- 동일 subject/date 중복 sleep row는 논문이 처리법을 밝히지 않았다. 재현에서는 하루 대표 수면으로 가장 긴 row를 선택했다.

## 7. 병합 정책과 결측 처리

논문 명시:

- daily row 기반 데이터
- 총 daily row 12,183
- activity와 sleep lifelog를 활용

논문 미명시:

- activity와 sleep을 inner join했는지, outer join했는지, left join했는지
- sleep row가 없는 activity day의 처리
- 결측치 imputation 방식

우리 재현 결정:

- `merge_policy=left_activity`를 기본값으로 사용한다.
- 이유는 activity daily row 기준일 때 논문 row count 12,183과 class count가 정확히 맞기 때문이다.
- sleep이 매칭되지 않는 33개 activity row는 제거하지 않고 유지한다.
- 결측 sleep feature는 모델 학습 pipeline 안에서 fold별 median imputation으로 처리한다.
- 이 방식은 CV 검증 fold의 정보를 train fold로 누설하지 않는다.

대안:

- `--merge-policy inner` 옵션을 제공한다.
- 다만 inner join은 12,150 rows가 되어 논문 count와 달라지므로 기본값으로 사용하지 않는다.

## 8. Feature 정리와 최종 feature 수

병합 직후 후보 feature 수는 90개다. 이 중 다음 4개는 모든 값이 없거나 상수 수준이라 제거된다.

```text
activity_day_start_hour
activity_day_end_hour
activity_met_1min_count
sleep_is_longest
```

최종 feature 수는 86개다.

제거 기준:

- 전체가 결측인 feature 제거
- unique value가 1개 이하인 feature 제거

모든 feature는 `pd.to_numeric(errors="coerce")`로 수치형 변환한다. 변환 불가 값은 결측으로 두고, 모델 pipeline에서 median imputation한다.

## 9. 교차검증 설정

논문 명시:

- K-fold cross validation 사용
- Figure와 설명에서 5-fold cross validation 제시
- ROC-AUC를 주요 평가 기준으로 사용
- 인지장애군 class에 대한 ROC curve를 사용

우리 구현:

- 기본값은 row-level `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- daily row가 분석 단위이므로 row-level fold를 기본값으로 둔다.
- subject 단위 leakage 가능성을 점검하려면 `--grouped` 옵션으로 `StratifiedGroupKFold`를 사용할 수 있지만, 논문 기본 재현에는 사용하지 않는다.
- 모든 CV 평가는 out-of-fold prediction/probability를 모아 전체 metric을 계산한다.

평가 지표:

- accuracy
- ROC-AUC
- precision macro
- recall macro
- F1 macro
- precision positive
- recall positive
- F1 positive

논문 표와 직접 비교하는 핵심 지표는 accuracy, ROC-AUC, precision macro, recall macro, F1 macro다.

## 10. Baseline 모델 비교

논문 명시:

- 7개 모델 비교
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbor
  - Support Vector Machine
  - Multi-Layer Perceptron
  - Random Forest
  - LightGBM
- ROC-AUC 기준으로 LightGBM이 가장 좋은 모델
- 논문 LightGBM 성능: accuracy 0.8262, ROC-AUC 0.9010, F1 macro 0.8025

논문 미명시:

- baseline 모델별 상세 hyperparameter
- scaling 적용 여부
- imputation 방식
- fold random seed
- class weight 사용 여부
- MLP 구조, SVM kernel, KNN k값, Random Forest tree 수

우리 구현:

| 모델 | 우리 설정 |
| --- | --- |
| Logistic Regression | median imputation, StandardScaler, `LogisticRegression(max_iter=3000, solver="lbfgs")` |
| Decision Tree | median imputation, `DecisionTreeClassifier(random_state=42)` |
| KNN | median imputation, StandardScaler, `KNeighborsClassifier(n_neighbors=5)` |
| SVM | median imputation, StandardScaler, `SVC(kernel="rbf", probability=True)` |
| MLP | median imputation, StandardScaler, `MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True)` |
| Random Forest | median imputation, `RandomForestClassifier(n_estimators=500, n_jobs=-1)` |
| LightGBM | median imputation, `LGBMClassifier(n_jobs=-1, verbosity=-1)` |

현재 재현 결과:

| 모델 | 논문 ROC-AUC | 재현 ROC-AUC | 차이 |
| --- | ---: | ---: | ---: |
| LightGBM | 0.9010 | 0.9004 | -0.0006 |
| Random Forest | 0.8835 | 0.8897 | +0.0062 |
| Decision Tree | 0.6806 | 0.6707 | -0.0099 |
| KNN | 0.6595 | 0.8524 | +0.1929 |
| MLP | 0.6348 | 0.8849 | +0.2501 |
| SVM | 0.6249 | 0.8731 | +0.2482 |
| Logistic Regression | 0.6067 | 0.6952 | +0.0885 |

해석:

- LightGBM과 Random Forest, Decision Tree는 논문 수치와 매우 가깝다.
- KNN, MLP, SVM은 논문보다 훨씬 높다.
- 이는 논문이 scaling/imputation/hyperparameter를 공개하지 않았기 때문에 생긴 차이로 보는 것이 타당하다.
- 핵심 모델 선택 결과는 논문과 동일하게 LightGBM이 최고 성능이다.

## 11. SHAP importance와 forward feature selection

논문 명시:

- 선택된 모델을 기반으로 SHAP Importance 계산
- 중요도가 높은 feature부터 1개씩 추가하는 forward selection 수행
- top 40 feature 사용 시 ROC-AUC 0.9037로 최고 성능

논문 미명시:

- SHAP importance를 전체 데이터 학습 모델에서 계산했는지, fold별 OOF 방식으로 계산했는지
- feature ranking에 tuned LightGBM을 사용했는지 baseline LightGBM을 사용했는지
- forward selection의 최대 feature 수
- random seed
- imputation/scaling 세부 방식

우리 구현:

- `03_shap_forward_selection.py` 기본값은 untuned LightGBM 기반 SHAP ranking이다.
- fold별로 모델을 학습하고 validation fold의 SHAP value를 계산한 뒤, 전체 fold의 mean absolute SHAP을 평균해 ranking을 만든다.
- 이 방식은 SHAP ranking에서 validation fold 정보를 train fold 학습에 사용하지 않는다.
- forward selection은 SHAP ranking 1위부터 k개 feature를 사용해 LightGBM 5-fold CV를 반복한다.
- 기본 `--max-features=80`.
- `--use-paper-params-for-ranking` 옵션을 제공하지만 기본 논문 실행에서는 사용하지 않았다.

현재 재현 결과:

| 항목 | 결과 |
| --- | ---: |
| 재현 best k | 28 |
| 재현 best ROC-AUC | 0.9038 |
| 논문 기준 top 40 ROC-AUC | 0.9000 |
| 논문 reported top 40 ROC-AUC | 0.9037 |

해석:

- 성능 수준은 논문의 0.9037과 거의 같다.
- 다만 우리 ranking/forward selection에서는 top 28이 가장 높고 top 40은 약간 낮다.
- 논문은 top 40을 최적 feature 수로 보고했으므로, 최종 strict reproduction에서는 `paper_top40_features`를 사용한다.
- 동시에 `best_reproduction` feature list도 저장해 실험적 비교가 가능하게 했다.

## 12. 최종 LightGBM 설정

논문 명시:

| 파라미터 | 논문 값 |
| --- | ---: |
| `min_data_in_leaf` | 41 |
| `num_leaves` | 330 |
| `n_estimators` | 1000 |
| `learning_rate` | 0.08 |

우리 구현:

| 논문 파라미터 | sklearn LightGBM 파라미터 | 값 |
| --- | --- | ---: |
| `min_data_in_leaf` | `min_child_samples` | 41 |
| `num_leaves` | `num_leaves` | 330 |
| `n_estimators` | `n_estimators` | 1000 |
| `learning_rate` | `learning_rate` | 0.08 |

논문 미명시:

- LightGBM의 나머지 파라미터
- early stopping 여부
- class weight 여부
- grid search 후보 전체 범위
- seed

우리 재현 결정:

- 논문 Table 2에 공개된 4개 파라미터만 고정한다.
- 나머지는 LightGBM 기본값을 사용한다.
- 실행 재현성을 위해 `random_state=42`, `n_jobs=-1`, `verbosity=-1`을 설정한다.
- `--grid-search` 옵션으로 compact grid search를 제공하지만, strict paper reproduction에서는 논문 값 그대로 `--use-paper-params`를 사용한다.

최종 CV 결과:

| 지표 | 재현 |
| --- | ---: |
| accuracy | 0.8798 |
| ROC-AUC | 0.9478 |
| precision macro | 0.8797 |
| recall macro | 0.8581 |
| F1 macro | 0.8668 |
| positive precision | 0.8792 |
| positive recall | 0.7776 |
| positive F1 | 0.8253 |

논문 최종 ROC-AUC는 0.9492이며, 재현 결과 0.9478과 차이는 약 -0.0015다. 이 정도 차이는 seed, fold split, 미공개 LightGBM 파라미터 차이를 고려하면 매우 근접한 재현이다.

주의:

- `final_training_roc.png`는 전체 데이터를 학습한 최종 모델을 같은 데이터에서 다시 평가한 in-sample ROC다.
- 그래서 AUC가 1.0000으로 나온다.
- 논문 성능 비교에는 반드시 `final_cv_metrics.json`의 5-fold CV ROC-AUC 0.9478을 사용해야 한다.

## 13. SHAP summary와 주요 feature

논문 명시:

- 수면 관련 변수가 중요하게 나타남
- `sleep_breath_average`가 가장 중요한 변수로 설명됨
- 행동 관련 변수 중 `activity_class_5min_count_3`가 언급됨
- 큰 `sleep_breath_average` 값은 인지장애군 방향으로 해석된다고 설명

우리 결과의 최종 SHAP 중요도 상위 feature:

| rank | feature | mean_abs_shap |
| ---: | --- | ---: |
| 1 | `sleep_breath_average` | 1.6755 |
| 2 | `sleep_hr_lowest` | 1.1263 |
| 3 | `sleep_hr_average` | 1.0462 |
| 4 | `sleep_rmssd` | 0.9140 |
| 5 | `sleep_bedtime_end_hour` | 0.8685 |
| 6 | `sleep_restless` | 0.7973 |
| 7 | `activity_cal_total` | 0.7426 |
| 8 | `activity_daily_movement` | 0.7390 |
| 9 | `sleep_midpoint_at_delta` | 0.7059 |
| 10 | `activity_score_meet_daily_targets` | 0.6831 |

해석:

- 최상위 feature가 `sleep_breath_average`인 점은 논문 설명과 일치한다.
- 상위권 대부분이 sleep feature인 점도 논문 설명과 일치한다.
- activity feature도 일부 상위권에 있으나, sleep feature의 영향이 더 크다.

## 14. Dementia Risk Score, DRS

논문 명시:

- 각 daily row의 치매 위험도 점수 DRS를 계산
- 각 feature의 SHAP value 중 양수인 값만 위험 기여도로 사용
- 양수 SHAP value를 합산해 DRS 산출
- 인지장애군 평균 DRS가 정상인지군 평균 DRS보다 큰지 one-sample t-test로 검정
- 단측 검정 사용

우리 구현:

1. 최종 LightGBM을 top 40 feature와 논문 파라미터로 전체 daily dataset에 fit
2. TreeExplainer로 class 1, 즉 MCI/Dem class의 SHAP value 계산
3. row별로 `max(SHAP, 0)`를 feature 축으로 합산
4. `dementia_risk_score`로 저장
5. CN group 평균을 기준값으로 두고, impaired group DRS에 대해 one-sided one-sample t-test 수행
6. 추가로 subject별 평균 DRS에 대해서도 같은 방향의 보조 검정을 수행

현재 DRS 결과:

| class | count | DRS mean | DRS std |
| --- | ---: | ---: | ---: |
| CN | 7,737 | 5.2260 | 1.6499 |
| MCI/Dem | 4,446 | 20.7987 | 2.3136 |

Daily one-sided t-test:

| 항목 | 값 |
| --- | ---: |
| CN mean | 5.2260 |
| impaired mean | 20.7987 |
| t statistic | 448.8019 |
| p-value | 0.0 |

Subject-level 보조 검정:

| 항목 | 값 |
| --- | ---: |
| CN subject mean | 5.2440 |
| impaired subject mean | 20.8149 |
| t statistic | 80.2791 |
| p-value | 1.13e-64 |

판정:

- 논문이 주장한 방향과 유의성이 모두 재현되었다.
- daily row 기준 DRS는 class 간 분리가 매우 강하다.

## 15. 논문 명시 사항과 우리 재현 설정 전체 비교

| 항목 | 논문에서 제시 | 논문에서 미제시 | 우리 재현 설정 |
| --- | --- | --- | --- |
| 데이터 출처 | AI-Hub 치매 고위험군 라이프로그, 174명 | 원본 폴더 내 정확한 CSV 선택 규칙 | activity/sleep 원천 CSV와 `1.걸음걸이` label CSV 사용 |
| 분석 단위 | 하루 단위 라이프로그 | activity/sleep 병합 방식 | activity daily row 기준 left join |
| class 정의 | CN=0, MCI/Dem=1 | `DEM`, `Dementia` 같은 변형 label 처리 | `MCI`, `Dem`, `DEM`, `Dementia`를 1로 매핑 |
| class count | CN 7,737, impaired 4,446 | train/validation 통합 방식 상세 | train+validation 모두 합쳐 12,183 row 구성 |
| timestamp 처리 | 24시 대응 실수값 변환 | 초/분 반영 여부 | hour + minute/60 + second/3600 |
| 수면 시간 | 종료-시작 차이 추가 | 단위 | seconds 단위 `sleep_time_from_timestamp` 생성 |
| 의미 없는 column 제거 | 모든 값 동일한 데이터 제거 | 정확한 column list | all-missing 또는 unique <= 1 feature 제거 |
| 5분 심박 로그 | 결측 처리 불가하여 제거 | RMSSD/hypnogram 원본 처리 | raw 5분 HR/RMSSD/hypnogram sequence 제거, hypnogram은 요약 feature 생성 |
| activity sequence | 측정값 개수 형태로 변환 | ratio/transition/statistics 여부 | class count/ratio/transition, MET 통계량 생성 |
| 결측 처리 | 미제시 | imputation 방법 | fold 내부 median imputation |
| CV | K-fold, 5-fold | split seed, stratification, subject grouping 여부 | row-level StratifiedKFold, shuffle=True, random_state=42 |
| 모델 7종 | 명시 | 각 모델 hyperparameter 대부분 | sklearn/LightGBM 합리적 기본값과 scaler 적용 |
| 성능 기준 | ROC-AUC | threshold, positive class 처리 상세 | positive class probability, threshold 0.5 for class metric |
| 최종 모델 | LightGBM | 나머지 LightGBM 기본값 | 공개 4개 파라미터 고정, 나머지는 LightGBM default |
| feature selection | SHAP importance forward selection, top 40 | ranking 계산 방식, tuned 여부 | OOF SHAP ranking, untuned LightGBM 기본, final은 paper top 40 사용 |
| grid search | 수행했다고 제시 | 탐색 후보 전체 | optional compact grid 제공, strict run은 논문 파라미터 직접 사용 |
| DRS | 양수 SHAP value 합산 | class SHAP 배열 처리, full fit 여부 | class 1 SHAP의 positive part sum, 전체 데이터 fit 후 전체 row DRS |
| DRS 검정 | one-sample one-sided t-test | subject-level 보정 여부 | daily test 구현, subject-level 보조 test 추가 |

## 16. 구현상 주의점과 해석 제한

1. Baseline 일부 모델은 논문보다 성능이 높다.
   - KNN, SVM, MLP는 scaling과 현재 기본 hyperparameter의 영향으로 논문보다 크게 높게 나왔다.
   - 논문이 해당 모델들의 세부 설정을 공개하지 않았으므로, 이 부분은 논문과 동일 환경이라고 단정하지 않는다.

2. Feature selection의 best k가 논문과 다르다.
   - 논문은 top 40을 best로 보고했다.
   - 현재 OOF SHAP ranking에서는 top 28이 best이며, top 40도 매우 근접하다.
   - 최종 모델은 논문 재현 목적에 맞춰 top 40을 사용한다.

3. 최종 ROC plot과 최종 CV metric을 구분해야 한다.
   - `final_training_roc.png`는 전체 학습 모델을 학습 데이터에 다시 적용한 in-sample plot이다.
   - 논문 보고용 성능은 `final_cv_metrics.json`의 5-fold CV ROC-AUC 0.9478이다.

4. Row-level CV는 subject leakage 가능성이 있다.
   - 논문은 daily row 기반 모델임을 설명하고 5-fold CV를 사용했다.
   - subject가 여러 daily row를 가지므로 row-level split에서는 같은 subject의 다른 날짜가 train/test fold에 함께 들어갈 수 있다.
   - 재현은 논문 방식에 맞춰 row-level을 기본값으로 두되, diagnostic 목적으로 `--grouped` 옵션을 제공한다.

5. DRS는 최종 학습 모델의 SHAP value 기반이다.
   - DRS 산출은 논문처럼 최종 모델 해석을 위한 점수화 단계로 구현했다.
   - 따라서 DRS 자체는 CV out-of-fold SHAP이 아니라 전체 데이터로 학습한 최종 모델의 SHAP value에서 계산된다.

## 17. 최종 판정

현재 구현은 논문 재현 목표를 충족한다.

근거:

- 논문 daily row/class count와 완전히 일치한다.
- baseline LightGBM ROC-AUC가 논문 0.9010 대비 재현 0.9004로 거의 같다.
- SHAP forward selection 성능 수준이 논문 0.9037과 같은 범위다.
- 최종 LightGBM ROC-AUC가 논문 0.9492 대비 재현 0.9478로 매우 근접하다.
- SHAP 상위 feature와 해석 방향이 논문 설명과 대체로 일치한다.
- DRS에서 impaired group 평균이 CN group 평균보다 유의하게 크다.

보고서 작성 시에는 다음 값을 사용하는 것이 가장 적절하다.

- Baseline LightGBM ROC-AUC: `0.9004`
- SHAP forward selection best ROC-AUC: `0.9038`
- SHAP forward selection top 40 ROC-AUC: `0.9000`
- Final LightGBM 5-fold CV ROC-AUC: `0.9478`
- DRS CN mean: `5.2260`
- DRS impaired mean: `20.7987`
- DRS one-sided t-test p-value: `0.0`
