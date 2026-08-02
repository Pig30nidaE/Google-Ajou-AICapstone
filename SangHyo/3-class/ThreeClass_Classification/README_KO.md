# CN vs MCI vs Dem 3-class subject ensemble

이 폴더는 `Data/` 원본으로 CN/MCI/Dem을 구분하기 위한 새 Colab
파이프라인이다. 실행 노트북은
`3class_subject_ensemble_colab.ipynb`이며, 저장소에서는 학습을 실행하지
않았다. 노트북 JSON과 각 Python 셀의 문법만 정적으로 검증한다.

예측 index는 각 피험자의 마지막 activity 관측일이며 그 직전 최대 28일을
사용한다. 목표는 그 시점의 **동시적 인지상태 분류**이고, 미래 치매 전환이나
발병 시점을 예측하는 예후 모델은 아니다.

## 왜 이 구조인가

현재 데이터는 학습 피험자 141명(CN 85, MCI 47, Dem 9), 공식 benchmark
피험자 33명으로 작고 불균형하다. 반면 각 피험자는 수십 일의 반복 측정값을
갖는다. 날짜나 7일 window를 무작위로 나누면 같은 사람을 학습과 평가에서
동시에 보게 되어 성능이 크게 부풀려진다.

`docs/ML Team project final report submission.pdf`에서도 기존 3-class LSTM은
공식 benchmark에서 모든 피험자를 CN으로 예측해 Accuracy 0.8125를 얻었지만,
Balanced Accuracy는 0.3333, Macro F1은 0.2989였다. 또 1,487개 통합 특성 중
821개에서 train-test 분포 차이(KS p<0.001)가 보고됐다. 따라서 이 노트북은
고차원 LSTM 하나에 의존하지 않고 다음 전략을 사용한다.

1. 연속 7일 조건 때문에 피험자를 버리지 않는다. 각 피험자의 마지막 활동일을
   기준으로 고정된 7/14/28 calendar-day 구간을 만들고 결측일은 coverage로
   명시한다.
2. 5분/1분 로그에서 분포, 분위수, 변동성, 추세, 자기상관, 활동 구간,
   수면단계 전이, 일주기 특성을 만든다.
3. 각 구간의 평균뿐 아니라 표준편차, IQR, 분위수, 추세,
   초기-후기 변화량, 자기상관, 관측 커버리지를 피험자 한 행으로 요약한다.
4. fold 안에서만 결측치 대치, clipping, ANOVA 기반 특성 선택을 수행한다.
5. 보수적/고용량 CatBoost, LightGBM, XGBoost, ExtraTrees, 정규화
   multinomial logistic 모델을 앙상블한다.
6. 직접 3-class 모델과 두 종류의 ordinal/hierarchical 모델을 함께 사용해
   CN 쏠림과 MCI 경계 문제를 완화한다.
7. 앙상블 가중치와 MCI/Dem class-bias는 train OOF 예측에서만 선택한다.
   중첩 CV로 이 선택 과정까지 바깥 fold에서 검증한 다음 구성을 동결한다.
   nested inner OOF, 최종 OOF와 refit 모두 동일한 4-repeat/multi-seed 확률 평균
   계약을 사용한다.
8. `Data/2.Validation` 라벨은 동결 파일을 저장한 뒤 마지막 셀에서 이 run 최초로
   로드한다. 설정·입력 fingerprint별 run 폴더와 완료 marker가 반복 평가/덮어쓰기를
   막는다.

정확한 성능 수치는 실제 실행 전에는 알 수 없다. 이 설계는 현재의 작은
피험자 수에서 가장 가능성이 높은 tabular/ordinal ensemble을 선택한 것이며,
GPU가 크다는 이유만으로 표본 수가 부족한 대형 신경망을 기본 모델로 삼지
않는다. 과거 보고서가 이미 공식 benchmark 결과를 사용했기 때문에 이 cohort는
완전히 미사용된 holdout이 아니다. **모델 선택의 주 근거는 Training nested CV**이고,
공식 benchmark는 역사적으로 재사용된 비교 지표로만 분리 보고한다. 확증에는 새
외부 cohort가 필요하다.

## 런타임 선택

Colab Pro+의 **최신 기본 런타임 + Premium GPU + High-RAM**을 권장한다.
할당 목록에 H100이 표시되면 H100, 그렇지 않으면 A100, 다음으로 L4 또는
사용 가능한 premium GPU를 선택한다. TPU는 사용하지 않는다. CatBoost와 XGBoost는 GPU를
사용하고, LightGBM·ExtraTrees·로지스틱 회귀와 feature engineering은 CPU/RAM을
주로 사용한다.

Colab은 GPU 종류와 가용성을 고정해 공개하지 않고 시점에 따라 바꾼다. 따라서
노트북은 실제 할당 GPU를 시작 시 출력하며, GPU 학습이 실패하면 해당 모델만
CPU로 자동 재시도한다. 과거 런타임을 고정하기보다 최신 런타임을 사용하고,
노트북 설치 셀의 비기본 라이브러리 버전만 고정한다.

공식 안내:

- https://research.google.com/colaboratory/faq.html
- https://research.google.com/colaboratory/runtime-version-faq.html

## Google Drive 준비

Colab의 `drive.mount()`는 일반적인 "공유 문서함" 폴더를 자동으로 경로에
노출하지 않을 수 있다. Google Drive에서 `GoogleAI_contest` 폴더를
**내 드라이브에 바로가기 추가**한 뒤 다음 구조가 보이도록 한다.

```text
/content/drive/MyDrive/GoogleAI_contest/Data/
├── 1.Training/
│   ├── SourceData/
│   └── LabelingData/
└── 2.Validation/
    ├── SourceData/
    └── LabelingData/
```

공유 드라이브 자체에 있는 경우 노트북은
`/content/drive/Shareddrives/GoogleAI_contest/Data`도 탐색한다. 둘 다 다르면
설정 셀의 `DATA_ROOT_OVERRIDE`만 수정한다.

## 실행 방법

1. Colab에서 노트북을 연다.
2. 런타임 유형을 Premium GPU, 가능하면 H100/A100과 High-RAM으로 설정한다.
3. 설치 셀을 실행한 뒤 런타임 재시작 안내가 나오면 한 번 재시작한다.
4. 설정 셀에서 다음 값을 확인한다.

```python
DATA_ROOT_OVERRIDE = None
RESULT_DIR_OVERRIDE = None
FAST_MODE = False
RUN_NESTED_CV = True
ALLOW_MMSE_FEATURES = False
```

5. 위에서 아래로 실행한다. `FAST_MODE=False`가 최종 실행 설정이다.
6. 마지막 공식 benchmark 셀은 동결 후 한 번만 실행한다. 결과 폴더의
   `FINAL_REPORT.json`과 confusion matrix를 확인한다. 성공 시
   `VALIDATION_EVALUATION_COMPLETE.json`이 생기며 같은 run의 재평가를 차단한다.

`FAST_MODE=True`는 코드 경로를 빠르게 점검하기 위한 설정이며 최종 성능
평가용이 아니다. CV 반복 수와 tree 수가 줄어든다.

## MMSE 사용 정책

기본값 `ALLOW_MMSE_FEATURES=False`는 의도적이다. 이 프로젝트 문서의 예측
대상은 wearable lifelog이며, MMSE 문항과 총점은 진단 과정과 매우 가까운
임상 검사다. 이를 입력하면 라이프로그 분류 성능으로 해석할 수 없고 과거
논문 재현에서도 과도한 성능의 원인이 될 수 있다. `DIAG_NM`, `DIAG_SEQ`,
식별자, `sleep_period_id`는 설정과 관계없이 항상 제외된다.

별도의 대회 규정이 **예측 시점에 MMSE 문항·총점을 사용할 수 있음**을
명시하는 경우에만 `ALLOW_MMSE_FEATURES=True`로 바꿀 수 있다. 노트북은 이때도
lifelog-only 후보를 유지하고 MMSE 포함 후보를 train OOF에서 비교한다.
MMSE 허용 결과와 lifelog-only 결과는 서로 다른 실험으로 보고해야 한다. MMSE
branch도 Q 문항과 hash 전 ID만 읽고 진단명·원본 총점은 읽지 않으며, validation의
all-zero 문항 행은 오답이 아니라 미측정 sentinel로 처리한다.

## 누수 방지 계약

- 모델 입력 한 행은 피험자 한 명이다. 동일 피험자가 fold 양쪽에 들어갈 수
  없다.
- validation source feature는 결정론적으로 가공할 수 있지만 validation
  라벨은 해당 run의 final rule 동결 전에는 로드하지 않는다. 이 격리는 새 실행의
  코드 계약이며, cohort 자체가 역사적으로 미사용이라는 뜻은 아니다.
- 모든 대치·clipping·특성 선택·모델 fitting·가중치 선택은 fold train에서만
  수행한다.
- 절대 날짜, 이메일, 파일 순서, 진단명, 진단 순번, 의사명, 수면 record ID는
  특성으로 사용하지 않는다.
- 저장되는 fold 및 prediction 파일에는 원본 이메일 대신 SHA-256 기반
  `subject_hash`만 기록한다.
- primary metric은 subject-level Macro F1이다. Accuracy는 보조 지표다.

## 결과물

기본 출력 base는
`/content/drive/MyDrive/GoogleAI_contest/outputs/3class_subject_ensemble/`이다.
실제 산출물은 그 아래 `full_lifelog_<12자리 config+input hash>/`처럼 설정과 원본
fingerprint가 반영된 run 폴더에 저장된다. `RESULT_DIR_OVERRIDE`도 base 경로다.

- `run_config.json`, `environment.json`: run/input fingerprint와 실행환경
- `data_audit.json`: 스키마·dtype·컬럼별 결측·날짜·중복·modality 불일치·누수 검사
- `data_audit_before_validation.json`, `data_audit_after_validation.json`: 동결 전
  감사 증거와 benchmark label 로드 후 상태를 분리 보존
- `feature_manifest.json`: 원천 제외 컬럼과 최종 feature 목록
- `cache/train_features_*.parquet`, `cache/validation_features_*.parquet`:
  hash ID 기반 cache
- `nested_cv_fold_metrics.csv`, `nested_cv_report.json`: 모델 선택을 포함한 CV
- `nested_cv_checkpoint_*.joblib`: outer fold 단위 재시작 checkpoint
- `candidate_oof_metrics.csv`: 최종 train OOF 후보 비교
- `final_oof_cache_*.joblib`: seed 단위 checkpoint가 포함된 repeated OOF cache
- `frozen_config_before_validation.json`: validation 라벨 로드 전 동결 증거
- `final_model_bundle.joblib`: 전처리기, 선택된 후보 모델, blend rule
- `selected_features.csv`: fold-train에서 학습된 최종 선택 특성
- `validation_predictions_hashed.csv`: PII가 제거된 공식 검증 예측
- `validation_metrics.json`, `FINAL_REPORT.json`: 최종 지표와 한계
- `validation_confusion_and_f1.png`: confusion matrix와 class별 F1
- `selected_feature_drift.csv`: 선택 특성의 train-validation KS 진단
- `VALIDATION_EVALUATION_COMPLETE.json`: 같은 frozen run의 반복 benchmark 평가 차단

`final_model_bundle.joblib`에는 노트북에서 정의한 fold transformer가 포함된다.
새 런타임에서 불러올 때는 설치·설정·feature/model 정의 셀까지 먼저 실행한 뒤
load한다. 저장 직후 같은 런타임에서 reload 검증을 수행한다.

## 결과 해석

Dem 학습 피험자는 9명뿐이다. 공식 benchmark에서도 한 명의 예측이 Dem recall과
Macro F1을 크게 바꾼다. 최종 보고에서는 다음을 함께 확인해야 한다.

- CN/MCI/Dem별 precision, recall, F1, support
- confusion matrix와 특히 MCI가 CN 또는 Dem으로 이동하는 패턴
- nested-CV fold별 평균·표준편차와 selection OOF 성능의 차이
- 전체 지표와 class별 F1/recall의 stratified bootstrap 95% 구간
- train-validation feature drift
- MMSE 허용 여부와 최종 blend에 포함된 모델

높은 Accuracy만 보고 성공으로 판단하지 않는다. 모델 선택은 nested-CV Macro F1과
모든 클래스의 recall을 함께 보고, 역사적으로 재사용된 benchmark 수치는 별도로
해석한다.
