# XAI Paper Reproduction2 Colab 실행 안내

이 폴더는 기존 `training/XAI_Paper_Reproduction/` 스크립트형 구현을 바탕으로, Colab Pro+에서 노트북만 실행하도록 만든 paper-exact 재현용 버전이다.

## 업로드할 파일

Google Drive에 다음 구조를 만든다.

```text
MyDrive/
└── XAI_Paper_Reproduction2/
    ├── Data/
    │   └── raw/
    └── Training/
        └── XAI_Paper_Reproduction2_PaperExact_Colab.ipynb
```

업로드 항목:

1. 원본 데이터 폴더
   - 권장 Drive 위치: `MyDrive/XAI_Paper_Reproduction2/Data/raw/`
   - 한글 경로 문제가 생기지 않도록, 기존 `128.치매 고위험군 라이프로그/` 폴더명을 `raw`로 바꿔 업로드한다.
   - `raw/` 안에는 `train_activity.csv`, `train_sleep.csv`, `training_label.csv`, `val_activity.csv`, `val_sleep.csv`, `val_label.csv`를 포함한 AI-Hub 원본 하위 폴더 전체가 들어가면 된다.
2. 노트북 파일
   - 로컬 파일: `training/XAI_Paper_Reproduction2/Training/XAI_Paper_Reproduction2_PaperExact_Colab.ipynb`
   - Drive 위치: `MyDrive/XAI_Paper_Reproduction2/Training/XAI_Paper_Reproduction2_PaperExact_Colab.ipynb`

노트북은 다음 데이터 폴더명을 자동 인식한다.

```text
MyDrive/XAI_Paper_Reproduction2/Data/raw/
MyDrive/XAI_Paper_Reproduction2/Data/lifelog_data/
MyDrive/XAI_Paper_Reproduction2/Data/aihub_lifelog/
MyDrive/XAI_Paper_Reproduction2/Data/128_lifelog/
MyDrive/XAI_Paper_Reproduction2/Data/128_dementia_lifelog/
MyDrive/XAI_Paper_Reproduction2/Data/128.치매 고위험군 라이프로그/
```

가장 안전한 방식은 `Data/raw/`를 쓰는 것이다.

## Colab 환경

- 권장 런타임: Colab Pro+ 고RAM 런타임
- GPU는 필수 아님
- Python 3 런타임
- 노트북 첫 부분에서 필요한 패키지를 설치한다.

설치 패키지:

```text
pandas
numpy
scikit-learn
lightgbm
shap
scipy
matplotlib
joblib
tqdm
```

## 실행 방법

1. Colab에서 `XAI_Paper_Reproduction2_PaperExact_Colab.ipynb`를 연다.
2. 런타임 유형은 `고RAM`으로 설정한다. GPU는 필수는 아니며 CPU 런타임으로도 가능하다.
3. 위에서부터 순서대로 실행한다.
4. 중간에 런타임이 끊기면 `런타임 > 런타임 다시 시작` 후 처음부터 다시 실행한다. feature selection과 최종 OOF SHAP 단계가 가장 오래 걸린다.
5. long-running 단계에는 `tqdm` progress bar가 표시된다.
6. 결과는 기본적으로 아래 폴더에 저장된다.

```text
MyDrive/XAI_Paper_Reproduction2/Training/outputs_paper_exact/
```

이 프로젝트의 최종 판단 기준은 Colab Pro+에서 생성된 `outputs_paper_exact/` 산출물이다. 로컬 smoke test나 부분 검산 결과는 노트북 구조 확인용이며, 최종 논문 재현 결과로 사용하지 않는다.

## 주요 산출물

```text
outputs_paper_exact/
├── data/
│   ├── daily_binary_lifelog.csv
│   ├── feature_columns.json
│   └── preprocess_summary.json
├── baselines/
│   ├── model_comparison.csv
│   ├── model_comparison_against_paper.csv
│   └── model_comparison_auc.png
├── feature_selection/
│   ├── shap_importance_full.csv
│   ├── forward_selection_metrics.csv
│   ├── selected_features.json
│   └── forward_selection_auc.png
├── final/
│   ├── final_cv_metrics.json
│   ├── final_lgbm_pipeline.joblib
│   ├── dementia_risk_scores.csv
│   ├── dementia_risk_score_against_paper.csv
│   ├── dementia_risk_score_summary.json
│   ├── oof_shap_values_positive_class.csv
│   ├── full_fit_shap_values_positive_class.csv
│   ├── shap_importance_positive.csv
│   ├── shap_summary_positive.png
│   ├── final_training_roc_in_sample.png
│   └── dementia_risk_score_histogram.png
├── paper_fidelity_summary.json
├── paper_exact_audit.json
└── paper_exact_reproduction_report.md
```

## Colab 실행 후 전달할 파일

Colab 실행이 끝나면 아래 파일을 전달하면 된다. 이 파일들만 있으면 전처리, feature selection, 최종 성능, DRS가 논문 기준을 통과했는지 해석할 수 있다.

```text
MyDrive/XAI_Paper_Reproduction2/Training/outputs_paper_exact/
├── paper_exact_audit.json
├── paper_fidelity_summary.json
├── paper_exact_reproduction_report.md
├── baselines/
│   └── model_comparison_against_paper.csv
├── feature_selection/
│   ├── forward_selection_metrics.csv
│   └── selected_features.json
└── final/
    ├── final_cv_metrics.json
    ├── dementia_risk_score_summary.json
    └── dementia_risk_score_against_paper.csv
```

우선순위는 `paper_exact_audit.json`이 가장 높다. 이 파일의 `passed`가 `true`이면 노트북 내부 기준으로 논문 재현 검증을 통과한 것이다. `false`이면 `checks` 항목 중 실패한 조건을 기준으로 원인을 해석하면 된다.

## 논문 기준 재현 방식

이 노트북은 기존 구현보다 paper-exact 목적을 더 명확히 한다.

- 전처리 row/class count는 논문 값과 반드시 일치하도록 검사한다.
- 최종 feature 수는 논문과 동일하게 top 40을 사용한다.
- 최종 LightGBM은 논문 Table 2 파라미터를 그대로 사용한다.
- 최종 성능 비교는 in-sample ROC가 아니라 5-fold CV ROC-AUC를 기준으로 한다.
- DRS는 5-fold out-of-fold에서 계산된 class 1, 즉 MCI/Dem 방향 SHAP value 중 양수만 합산한다.

## 전처리 상세 기록

### 1. 원본 데이터 선택

노트북은 `Data/raw/` 아래에서 다음 CSV를 자동 탐색한다. 기존 한글 폴더명도 지원하지만, Colab Drive에서 한글 경로 문제가 생기면 최상위 데이터 폴더명을 `raw`로 바꾸는 것을 권장한다. 경로 비교는 macOS와 Google Drive의 한글 유니코드 정규화 차이를 줄이기 위해 NFC 정규화 후 수행한다.

| 용도 | 사용 파일 |
| --- | --- |
| Training activity | `1.Training/원천데이터/1.걸음걸이/train_activity.csv` |
| Training sleep | `1.Training/원천데이터/2.수면/train_sleep.csv` |
| Training label | `1.Training/라벨링데이터/1.걸음걸이/training_label.csv` |
| Validation activity | `2.Validation/원천데이터/1.걸음걸이/val_activity.csv` |
| Validation sleep | `2.Validation/원천데이터/2.수면/val_sleep.csv` |
| Validation label | `2.Validation/라벨링데이터/1.걸음걸이/val_label.csv` |

라벨은 `라벨링데이터/1.걸음걸이`의 파일을 기준으로 사용한다. 원본 폴더에 다른 라벨 파일이 있어도 이 재현에서는 activity/sleep daily row에 붙일 subject-level 진단 라벨만 필요하므로 위 파일만 사용한다.

### 2. 라벨 처리

논문은 정상인지군 `CN`과 인지장애군 `MCI`, `Dem`의 이진 분류를 제시한다. 노트북은 이를 다음처럼 매핑한다.

| 원본 diagnosis | binary class |
| --- | ---: |
| `CN` | 0 |
| `MCI` | 1 |
| `Dem` | 1 |
| `DEM` | 1 |
| `Dementia` | 1 |

전처리 후 반드시 다음 논문 count와 일치해야 한다. 일치하지 않으면 `assert`에서 실행을 중단한다.

| 항목 | 논문 기준 |
| --- | ---: |
| daily row 수 | 12,183 |
| subject 수 | 174 |
| CN daily row 수 | 7,737 |
| MCI/Dem daily row 수 | 4,446 |

### 3. Activity feature 생성

Activity 원본은 `EMAIL`을 `patient_id`로 바꾸고, `activity_day_start`에서 `sample_date`를 만든다. `activity_day_start`, `activity_day_end`는 논문 설명대로 24시간 기준 실수형 hour로 변환한다.

예시는 다음과 같다.

```text
13:30:00 -> 13.5
01:15:30 -> 1 + 15/60 + 30/3600
```

`activity_class_5min` 계열 slash sequence는 각 stage 값의 개수와 비율, transition count로 요약한다.

생성 feature:

- `activity_class_5min_valid_count`
- `activity_class_5min_count_0`부터 `activity_class_5min_count_5`
- `activity_class_5min_ratio_0`부터 `activity_class_5min_ratio_5`
- `activity_class_5min_transition_count`

`activity_met_1min` 계열 slash sequence는 연속량으로 보고 통계량으로 요약한다.

생성 feature:

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

논문은 "어노테이션이 기록된 활동 로그 시계열 데이터는 각 측정값 개수에 해당하는 형태로 변환"한다고만 설명하고, ratio/transition/MET 통계량까지의 전체 규칙은 공개하지 않았다. 따라서 count는 논문 문장에 맞춘 필수 재현이고, ratio/transition/MET 통계량은 원본 sequence 정보를 과도하게 버리지 않기 위한 명시적 재현 설정이다.

### 4. Sleep feature 생성

Sleep 원본은 `EMAIL`을 `patient_id`로 바꾸고, `sleep_bedtime_end`의 날짜를 `sample_date`로 사용한다. `sleep_bedtime_start`, `sleep_bedtime_end`는 24시간 기준 실수형 hour로 변환한다.

논문이 제시한 수면 시간 파생 변수는 다음처럼 만든다.

```text
sleep_time_from_timestamp = sleep_bedtime_end - sleep_bedtime_start
```

단위는 seconds다.

같은 subject/date에 sleep row가 여러 개 있으면 논문이 대표 row 선택 규칙을 공개하지 않았기 때문에, 노트북은 `sleep_time_from_timestamp`가 가장 긴 row를 하루 대표 수면으로 선택한다.

논문은 결측치 처리가 불가능한 5분당 심박동 로그 데이터를 제거했다고 설명한다. 따라서 raw sequence column인 `sleep_hr_5min`, `sleep_rmssd_5min`, raw `sleep_hypnogram_5min`은 모델에 직접 넣지 않는다. 다만 hypnogram의 수면 단계 분포는 유의미할 수 있으므로 stage 1-4의 count/ratio/transition 요약 feature만 생성한다.

생성 feature:

- `sleep_hypnogram_5min_valid_count`
- `sleep_hypnogram_5min_count_1`부터 `sleep_hypnogram_5min_count_4`
- `sleep_hypnogram_5min_ratio_1`부터 `sleep_hypnogram_5min_ratio_4`
- `sleep_hypnogram_5min_transition_count`

### 5. Activity/Sleep 병합

논문 count는 activity daily row를 기준으로 할 때 정확히 맞는다. sleep을 inner join하면 row 수가 줄어 논문 count와 달라진다. 따라서 노트북은 `left_activity` 정책을 고정한다.

```text
daily = activity LEFT JOIN sleep
        ON patient_id, sample_date, split
```

sleep feature가 없는 activity day는 제거하지 않는다. 해당 결측값은 모델 pipeline 안에서 fold별 median imputation으로 채운다. 이 방식은 validation fold의 결측 분포를 train fold imputer 학습에 사용하지 않으므로 CV 정보 누설을 피한다.

### 6. Feature 정리

모든 후보 feature는 `pd.to_numeric(errors="coerce")`로 수치형 변환한다. 변환할 수 없는 값은 결측으로 둔다.

그 다음 다음 조건에 해당하는 feature를 제거한다.

- 전체 값이 결측인 feature
- 결측 제외 unique value가 1개 이하인 feature

기존 동일 로직 산출물 기준으로 병합 직후 후보 feature 90개 중 4개가 제거되어 최종 feature는 86개였다.

대표적으로 제거되는 상수/무효 feature:

```text
activity_day_start_hour
activity_day_end_hour
activity_met_1min_count
```

## 학습 상세 기록

### 1. 공통 CV와 평가 기준

논문은 K-fold, 특히 5-fold cross validation과 ROC-AUC를 제시한다. 노트북은 기본적으로 daily row 단위의 5-fold stratified CV를 사용한다.

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

평가 지표는 out-of-fold prediction/probability를 모두 모은 뒤 계산한다.

- accuracy
- ROC-AUC
- precision macro
- recall macro
- F1 macro
- positive precision
- positive recall
- positive F1

논문 Table과 직접 비교하는 핵심 지표는 accuracy, ROC-AUC, precision macro, recall macro, F1 macro다. ROC-AUC는 class 1, 즉 MCI/Dem 방향 probability로 계산한다. class metric의 threshold는 0.5다.

subject leakage를 더 엄격하게 확인하려면 `StratifiedGroupKFold`가 가능하지만, 논문은 daily row 기반 실험으로 설명되어 있으므로 paper-exact 기본값에는 사용하지 않는다.

### 2. Baseline 7종 모델

논문은 7개 baseline 모델을 비교하고, ROC-AUC 기준 LightGBM을 선택했다. 노트북도 동일한 7종을 모두 실행한다.

| 모델 | 노트북 설정 |
| --- | --- |
| Logistic Regression | median imputation, StandardScaler, `LogisticRegression(max_iter=3000, solver="lbfgs")` |
| Decision Tree | median imputation, `DecisionTreeClassifier(random_state=42)` |
| KNN | median imputation, StandardScaler, `KNeighborsClassifier(n_neighbors=5)` |
| SVM | median imputation, StandardScaler, `SVC(kernel="rbf", probability=True)` |
| MLP | median imputation, StandardScaler, `MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, early_stopping=True)` |
| Random Forest | median imputation, `RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)` |
| LightGBM | median imputation, `LGBMClassifier(random_state=42, n_jobs=-1, verbosity=-1)` |

논문은 각 baseline의 상세 hyperparameter, scaling 여부, imputation 방식, seed를 공개하지 않았다. 따라서 baseline 중 일부 모델의 재현 수치가 논문과 다르게 나올 수 있다. 기존 동일 로직 실행에서는 LightGBM ROC-AUC가 논문 0.9010 대비 0.9004로 거의 일치했고, KNN/SVM/MLP는 논문보다 높게 나왔다.

### 3. SHAP importance와 forward feature selection

논문 제시:

- 선택된 모델인 LightGBM 기반 SHAP importance 계산
- 중요도가 높은 feature부터 하나씩 추가하는 forward feature selection
- top 40 feature에서 ROC-AUC 0.9037로 최고 성능

노트북 설정:

- SHAP ranking은 fold별 LightGBM을 학습하고 validation fold의 SHAP value를 계산하는 OOF 방식으로 만든다.
- fold별 SHAP value의 mean absolute value를 전체 row 기준으로 평균해 feature ranking을 만든다.
- paper-exact 버전에서는 ranking용 LightGBM도 논문 Table 2 파라미터를 사용한다.
- forward selection은 ranking 1위부터 k개 feature를 사용해 5-fold CV를 반복한다.
- `STRICT_PAPER_TOP40=True`가 기본값이므로, local best k가 다르게 나오더라도 최종 모델에는 논문과 동일하게 top 40 feature를 사용한다.

이 설정은 논문이 공개한 "top 40 사용"을 최종 재현 기준으로 고정하기 위한 것이다. 동시에 `forward_selection_metrics.csv`에는 k별 성능을 모두 저장하므로, 실행 환경에서 실제 best k가 40인지도 확인할 수 있다.

### 4. 최종 LightGBM

논문 Table 2에서 공개한 최종 LightGBM 파라미터는 다음 4개다.

| 논문 파라미터 | sklearn LightGBM 파라미터 | 값 |
| --- | --- | ---: |
| `min_data_in_leaf` | `min_child_samples` | 41 |
| `num_leaves` | `num_leaves` | 330 |
| `n_estimators` | `n_estimators` | 1000 |
| `learning_rate` | `learning_rate` | 0.08 |

노트북은 위 4개를 고정하고, 나머지는 LightGBM 기본값을 사용한다. 재현성을 위해 `random_state=42`, `n_jobs=-1`, `verbosity=-1`을 추가한다.

최종 성능은 다음 두 값을 구분해서 기록한다.

- `final_cv_metrics.json`: 5-fold CV 성능, 논문과 비교할 값
- `final_training_roc_in_sample.png`: 전체 데이터로 학습한 모델을 같은 데이터에 다시 적용한 in-sample ROC, 모델 해석용 참고 그림

논문 성능과 비교할 때는 반드시 `final_cv_metrics.json`의 ROC-AUC를 사용해야 한다.

### 5. SHAP과 Dementia Risk Score

논문은 DRS 산출용 SHAP이 전체 데이터에 다시 fit한 모델의 in-sample SHAP인지, 5-fold 검증 과정의 out-of-fold SHAP인지 명시하지 않았다. 기존 full-fit SHAP 방식은 최종 모델이 학습 데이터를 거의 완전히 분리하면서 DRS 평균이 논문보다 크게 벌어졌다. 반면 같은 top-40과 Table 2 파라미터로 5-fold out-of-fold SHAP을 계산하면 논문 DRS 요약값에 훨씬 가깝다.

따라서 paper-exact 노트북은 다음처럼 분리한다.

- 최종 모델 파일과 SHAP summary plot: top 40 feature와 논문 Table 2 파라미터로 전체 daily dataset에 fit한 full-fit LightGBM 사용
- DRS 산출과 DRS 검정: 5-fold out-of-fold LightGBM의 validation fold SHAP 사용

DRS 계산에는 TreeExplainer로 구한 class 1, 즉 MCI/Dem 방향 SHAP value를 사용한다.

DRS는 논문 설명대로 row별 양수 SHAP value만 합산한다.

```text
DRS(row) = sum(max(SHAP_feature_i_for_class_1, 0))
```

논문이 보고한 DRS 요약값은 다음과 같다.

| class | min | max | mean |
| --- | ---: | ---: | ---: |
| CN | 1.06 | 24.99 | 7.59 |
| MCI/Dem | 1.79 | 31.28 | 15.71 |

노트북은 다음을 저장한다.

- 전체 row별 `dementia_risk_score`
- OOF SHAP matrix
- full-fit SHAP matrix
- class별 DRS 요약 통계
- 논문 DRS min/max/mean 대비 delta
- CN 평균을 기준값으로 둔 impaired group one-sided one-sample t-test
- subject별 평균 DRS에 대한 보조 one-sided test
- DRS histogram
- SHAP summary plot

로컬 검산에서 기존 top-40 feature와 Table 2 파라미터를 사용한 OOF SHAP DRS는 CN 평균 7.3072, MCI/Dem 평균 15.5093으로 논문 CN 평균 7.59, MCI/Dem 평균 15.71에 가까웠다. 이 검산 결과를 반영하여 노트북 기본 DRS source를 OOF SHAP으로 설정했다.

## 논문 공개/미공개 항목과 우리 설정

| 항목 | 논문에서 제시 | 논문에서 미제시 | 노트북 재현 설정 |
| --- | --- | --- | --- |
| 데이터 출처 | AI-Hub 치매 고위험군 라이프로그, 174명 | 원본 폴더 내 정확한 CSV 선택 규칙 | activity/sleep 원천 CSV와 `1.걸음걸이` label CSV 사용 |
| 분석 단위 | 하루 단위 lifelog | activity/sleep 병합 방식 | activity daily row 기준 left join |
| class 정의 | CN=0, MCI/Dem=1 | `DEM`, `Dementia` 같은 label 변형 | `MCI`, `Dem`, `DEM`, `Dementia`를 1로 매핑 |
| class count | CN 7,737, impaired 4,446 | train/validation 통합 방식 상세 | train+validation 모두 합쳐 12,183 row 구성 |
| timestamp 처리 | 24시 대응 실수값 변환 | 초 단위 반영 여부 | hour + minute/60 + second/3600 |
| 수면 시간 | 종료-시작 차이 추가 | 단위 | seconds 단위 `sleep_time_from_timestamp` 생성 |
| 무의미 column 제거 | 모든 값 동일한 데이터 제거 | 정확한 column list | all-missing 또는 unique <= 1 feature 제거 |
| 5분 심박 로그 | 결측 처리가 어려워 제거 | RMSSD/hypnogram 처리 | raw HR/RMSSD/hypnogram sequence 제거, hypnogram은 요약 feature 생성 |
| activity sequence | 측정값 개수 형태 변환 | ratio/transition/MET 통계량 여부 | class count/ratio/transition, MET 통계량 생성 |
| 결측 처리 | 미제시 | imputation 방법 | fold 내부 median imputation |
| CV | K-fold, 5-fold | seed, stratification, subject grouping | row-level `StratifiedKFold`, shuffle=True, random_state=42 |
| baseline 모델 | 7종 명시 | 대부분의 hyperparameter | sklearn/LightGBM 기본값 중심, 필요한 모델에 scaler 적용 |
| 성능 기준 | ROC-AUC 중심 | threshold, positive class 처리 | class 1 probability 기준 ROC-AUC, threshold 0.5 |
| 최종 모델 | LightGBM | 공개 4개 외 나머지 파라미터 | Table 2의 4개 값 고정, 나머지는 LightGBM default |
| feature selection | SHAP importance forward selection, top 40 | SHAP ranking 계산 방식, ranking 모델 파라미터, seed | OOF SHAP ranking, ranking에도 Table 2 파라미터 사용, 최종 top 40 고정 |
| grid search | 수행했다고 설명 | 후보 전체 범위 | strict reproduction은 논문 파라미터 직접 사용 |
| DRS | 양수 SHAP value 합산, CN 평균 7.59, MCI/Dem 평균 15.71 | class SHAP 배열 처리, full fit/OOF 여부 | class 1 OOF SHAP positive part sum, 전체 12,183 row DRS |
| DRS 검정 | one-sample one-sided t-test | subject-level 보정 여부 | daily test 구현, subject-level 보조 test 추가 |

## 자동 검증 기준

마지막 audit 셀은 다음 조건을 검사하고 `paper_exact_audit.json`에 저장한다.

- row 수 12,183 일치
- subject 수 174 일치
- class count 7,737/4,446 일치
- baseline 7개 모델 모두 실행
- baseline LightGBM ROC-AUC가 논문 0.9010과 0.01 이내
- forward selection top 40 ROC-AUC가 논문 0.9037과 0.01 이내
- forward selection local best k가 논문처럼 40인지 확인
- 최종 feature 수 40
- 최종 LightGBM 파라미터가 논문 Table 2와 일치
- 최종 5-fold CV ROC-AUC가 논문 0.9492와 0.01 이내
- DRS source가 5-fold out-of-fold SHAP인지 확인
- DRS CN/MCI-Dem 평균이 논문 보고 평균과 0.50 이내
- DRS min/max가 논문 보고 범위와 3.50 이내
- DRS에서 impaired 평균이 CN 평균보다 큼
- DRS one-sided t-test p-value < 0.05
- 전체 12,183 row에 DRS가 계산됨

## 해석 제한과 주의 사항

논문은 다음 정보를 공개하지 않았다.

- fold seed
- baseline 모델별 상세 hyperparameter
- SHAP ranking 계산 방식
- feature engineering 전체 세부 규칙
- LightGBM grid search 후보 전체

따라서 노트북은 논문이 공개한 값은 고정하고, 미공개 부분은 명시적 기본값으로 둔다. `RUN_OPTIONAL_FIDELITY_SEARCH=True`로 바꾸면 top-40 forward-selection 결과를 더 맞추기 위한 seed 탐색을 추가로 실행할 수 있다. 이 탐색은 시간이 오래 걸릴 수 있다.

보고서에서 해석할 때는 다음 구분이 중요하다.

- Baseline 일부 모델, 특히 KNN/SVM/MLP는 논문 미공개 설정 차이로 논문보다 높게 나올 수 있다.
- feature selection의 local best k가 실행 환경에서 40이 아닐 수 있지만, paper-exact 최종 모델은 논문 기준 top 40을 사용한다.
- 최종 성능 비교에는 in-sample ROC 그림이 아니라 5-fold CV metric을 사용한다.
- DRS는 논문 수치에 더 가까운 5-fold out-of-fold SHAP 기반 점수이며, SHAP summary plot은 최종 full-fit 모델 기준이다.
- row-level CV는 daily row 논문 재현에는 맞지만, 같은 subject의 다른 날짜가 train/test fold에 나뉘는 subject leakage 가능성이 있다.
