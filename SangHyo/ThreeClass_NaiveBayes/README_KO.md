# CN / MCI / DEM — Gaussian Naive Bayes

이 폴더는 활동·수면 기록만으로 사람 단위 `CN`, `MCI`, `DEM`을 분류하는
Gaussian Naive Bayes 기준선입니다. 기존 모델과 비교 가능한 가볍고 해석하기 쉬운
기준선을 제공하는 것이 목적입니다.

## 모델과 데이터

- 모델: `sklearn.naive_bayes.GaussianNB`
- 클래스 순서: `CN=0`, `MCI=1`, `DEM=2`
- 학습 단위: 하루 단위 행이 아니라 **피험자 1명당 1행**
- 입력: activity, sleep
- 제외: MMSE 원천 값, ID, 진단명, 절대 날짜, 관측 개수·간격·mask 같은 수집 방식 신호

GaussianNB를 택한 이유는 입력이 연속형 실수이고 일부 값은 음수가 될 수 있기
때문입니다. 비음수 count를 가정하는 MultinomialNB/ComplementNB는 이 데이터에
맞지 않습니다.

특징 생성은 검증된 형제 프로젝트
`../ThreeClass_GoogleYDF_CNBoost/feature_engineering.py`를 재사용합니다. 각 피험자의
최근 7/14/28회 실제 관측 activity/sleep에서 생체·행동 요약 특징을 만들며,
인지기능 SourceData 경로는 열지 않습니다. 실행 결과에는 재사용한 코드의 SHA-256도
남아 어느 버전의 특징 코드로 학습했는지 확인할 수 있습니다.

## 누수 방지와 평가

Training 141명은 `CN 85 / MCI 47 / DEM 9`로 불균형합니다. 따라서 accuracy만
보면 모두 CN으로 예측하는 모델도 좋아 보일 수 있습니다.

정식 평가는 다음 계약을 사용합니다.

1. 사람 단위 stratified outer 3-fold를 서로 다른 seed 5개로 반복합니다.
2. 각 outer-training fold 안에서 inner 3-fold로 특징 수, `var_smoothing`,
   empirical/uniform class prior를 고릅니다.
3. median 대치, robust scaling, 상수 특징 제거, 단변량 특징 선택,
   GaussianNB를 하나의 Pipeline으로 묶습니다.
4. 모든 전처리와 특징 선택은 해당 fit fold에서만 학습합니다.
5. 주 결과는 반복별 OOF macro F1, balanced accuracy, macro OVR ROC-AUC입니다.
6. all-CN 및 class-prior 기준선도 함께 기록하며, class prior는 각 outer fit
   fold의 label만으로 계산합니다.
7. Validation은 모델 선택에 사용하지 않습니다. label-free 예측과 해시를 먼저
   동결한 뒤에만 label을 열어 역사적 참고 지표를 계산합니다.

`--fast`는 코드 동작 확인용으로 outer seed 1개와 작은 grid만 사용합니다. 이 결과를
성능 결과로 보고하면 안 됩니다.

## 설치

저장소 루트에서 실행합니다.

```bash
python -m pip install -r SangHyo/ThreeClass_NaiveBayes/requirements_colab.txt
```

Colab에서도 같은 명령을 사용할 수 있습니다. 데이터 루트만 Drive 경로로 바꾸면
됩니다. 루트 `base.ipynb`의 현재 내용은 이 실험을 가리키지 않으므로 아래 CLI를
직접 실행하는 방식이 가장 명확합니다.

## 실행

### 공통 `base.ipynb`에서 실행

`USER_FOLDER`와 `RUN_FILE`을 받는 공통 스켈레톤에서는 다음처럼 지정합니다.

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "ThreeClass_NaiveBayes/run_base.py"
```

지정할 파일은 `train.py`가 아니라 **`run_base.py`**입니다. `train.py`는
`--training-root`, `--output-dir` 같은 CLI 인자가 필요하고, `run_base.py`가
노트북의 `DATA_ROOT`를 이용해 그 인자를 구성합니다. 기본값은 정식 `full` 실행입니다.
스켈레톤에 추가 설정 셀이 있다면 빠른 확인은 아래 값을 함께 지정할 수 있습니다.

```python
NAIVE_BAYES_RUN_MODE = "smoke"  # 정식 실행은 "full"
```

현재 저장소 루트의 `base.ipynb` 내용이 `USER_FOLDER`/`RUN_FILE` 스켈레톤이 아니라
특정 LSTM 경로를 직접 실행하는 버전이라면, 위 두 변수를 사용하는 공통 스켈레톤
버전에서 실행해야 합니다.

### CLI에서 직접 실행

짧은 동작 확인:

```bash
python SangHyo/ThreeClass_NaiveBayes/train.py \
  --training-root Data/1.Training \
  --validation-root Data/2.Validation \
  --output-dir /tmp/threeclass_naive_bayes_fast \
  --fast
```

정식 5-seed 중첩 교차검증:

```bash
python SangHyo/ThreeClass_NaiveBayes/train.py \
  --training-root Data/1.Training \
  --validation-root Data/2.Validation \
  --output-dir /path/to/new/naive_bayes_training_output
```

Validation label을 전혀 열지 않고 예측만 저장하려면
`--skip-validation-labels`를 추가합니다. 결과 덮어쓰기를 막기 위해
`--output-dir`은 없거나 비어 있는 폴더여야 합니다.

Colab Drive 예시:

```bash
python SangHyo/ThreeClass_NaiveBayes/train.py \
  --training-root /content/drive/MyDrive/GoogleAI_contest/Data/1.Training \
  --validation-root /content/drive/MyDrive/GoogleAI_contest/Data/2.Validation \
  --output-dir /content/drive/MyDrive/SangHyo_NaiveBayes_Results/run_01
```

## 주요 출력

- `nested_cv_report.json`: repeat/fold별 OOF 지표와 선택된 설정
- `nested_oof_predictions_hashed.csv`: 원본 ID 없이 저장한 OOF 확률
- `model_manifest.json`: 최종 설정, 선택 특징, 모델 SHA-256
- `models/naive_bayes_pipeline.joblib`: 전처리를 포함한 최종 모델 bundle
- `validation_predictions_label_free_hashed.csv`: label을 열기 전 예측
- `VALIDATION_PREDICTIONS_FROZEN.json`: 예측 동결 시점과 파일 해시
- `validation_report.json`: 선택과 무관한 역사적 Validation 참고 지표
- `FINAL_REPORT.json`: 핵심 결과 요약
- `TRAINING_COMPLETE.json`: 정상 완료 표식

## 검증

```bash
python -m compileall -q SangHyo/ThreeClass_NaiveBayes
python -m unittest discover -s SangHyo/ThreeClass_NaiveBayes/tests -v
python SangHyo/ThreeClass_NaiveBayes/train.py --help
python -m pip check
```

이 코드는 연구·비교 실험용 기준선이며 임상 진단 도구가 아닙니다. 특히 DEM은
Training 9명, Validation 3명뿐이므로 단일 split의 높은 수치를 일반화 성능으로
해석하면 안 됩니다.
