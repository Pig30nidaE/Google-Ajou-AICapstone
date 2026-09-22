# Jaehwang: fixed-domain 3-class wearable XGBoost baseline

웨어러블 특징만으로 CN(0), MCI(1), Dementia(2)를 구분하는 독립 실험이다.
기존 Taehyun/Hyunsoo 파일과 입력 CSV를 수정하지 않는다. 모델 성능에 따라
특징·파라미터·seed를 바꾸는 실험이 아니다.

## 입력과 실행 전 확인

- 입력: 저장소 루트 기준 `Taehyun/data/processed/tabular/patient_level_circadian_v3.csv`
- Target: `original_label`. 기존 `label`은 binary이므로 target으로 사용하지 않는다.
- 현재 입력: 174명, CN 111 / MCI 51 / Dementia 12. EMAIL 결측·중복 없음.
- SHA-256: `1799b855b27950e8d85f0e9fc917d299094fa50cbfbeee1f916f234b2b11215d`
- EMAIL은 피험자 분할/중복 확인에만 사용한다. 결과에는 EMAIL 대신 입력 행 기반
  `subject_index`를 저장한다. 이는 외부 공개용 완전 익명화를 의미하지 않는다.
- 현재 CSV에는 아래 목록 중 `HR_drop_ratio`, `Circadian_Strain`이 없다.
  기본 실행은 누락 목록을 저장한 뒤 실패한다. 누락 특징을 삭제하거나 대체하지 않는다.
- 사용자가 두 공식 사용을 승인했다. 실행 옵션 `--allow-derived`로 승인을 명시한다.
  두 열은 각 outer fold의 `preprocess_fold()` 안에서 train/test 각각 새로 계산한다.
  `HR_drop_ratio=(sleep_hr_average-sleep_hr_lowest)/(sleep_hr_average+1e-5)`;
  `Circadian_Strain=circadian_IV/(circadian_IS+1e-5)`. 다른 특징은 만들어 대체하지 않는다.
  두 공식은 **원천 wearable signal만 이용한 deterministic derived features**다.
  target, MMSE, 진단명, 전체 데이터 통계, fit을 사용하지 않는다.
  원천 열의 결측/inf와 파생 직후·정리 후 결측/inf를
  `results/derived_feature_audit.csv`에 fold별 train/test로 기록한다.
  `role=oof_unique_input`은 test fold들의 감사 횟수를 합산한 전체 입력 174행의 집계다.
  train 기록은 fold 사이에 중복되므로 전체 데이터 집계에 합산하지 않는다.
  분모는 공식대로 epsilon을 더한 값이며, 이 값이 0 또는 비유한 값이면 NaN 처리한다.
  원천 inf 또는 계산 결과 inf도 NaN으로 바꾼 뒤 train-fold median imputation을 적용한다.
- 33행의 IS/IV/RA/wake-bouts가 모두 0이다. 0을 자동 결측 처리하거나 복원하지 않는다.
  따라서 Taehyun의 복원 완료 데이터 결과와 직접 동일한 입력이라고 볼 수 없다.

## 고정 입력 (순서 보존)

```python
DOMAIN_FEATURES = [
    "sleep_score_alignment", "sleep_hr_5min_max_std", "sleep_awake_std",
    "sleep_breath_average", "activity_score_std", "activity_class_3_count_std",
    "activity_met_min_low_std", "sleep_restless_std", "circadian_IV",
    "circadian_IS", "circadian_RA", "sleep_wake_bouts_avg", "HR_drop_ratio",
    "Circadian_Strain",
]
```

`data.py`에서 MMSE, TOTAL, Q번호 문항, DIAG, DOCTOR, label, 식별자 열을
명시적으로 금지한다. 그 외 열도 위 allowlist 밖이면 입력에 포함하지 않는다.
실제 제외 열과 사유를 `excluded_columns.json/csv`로 저장한다.

## 평가와 고정 파라미터

1인 1행이면 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`,
여러 행이면 같은 설정의 `StratifiedGroupKFold`로 EMAIL 그룹을 분리한다.
행별 OOF는 한 번씩 생성하고 반복 행이 있으면 동일 피험자 확률을 평균한 뒤
피험자별 한 번 평가한다. 피험자의 정답이 행 사이에서 다르면 실패한다.
각 fold의 학습/평가에 3개 클래스가 모두 있어야 하며, 없다고 seed를 재탐색하지 않는다.

파생 특징을 먼저 계산한 뒤 각 train fold만으로 중앙값을 fit하여 결측을 대치한다.
test fold에는 동일 공식과 train에서 fit된 imputer의 transform만 적용한다.
원천 결측을 먼저 대치하여 파생값을 만들어 내지 않는다. 전체 결측 train 열은 오류다.
Sample weight는 `N_train_subjects / (3 * N_train_subjects_in_class)`이며,
반복 행이면 해당 피험자의 행 수로 나누고 전체 평균이 1이 되도록 정규화한다.
SMOTE를 사용하지 않는다.

| Parameter | Fixed value |
|---|---|
| objective / num_class / eval_metric | multi:softprob / 3 / mlogloss |
| n_estimators / max_depth / learning_rate | 100 / 2 / 0.04 |
| min_child_weight | 3 |
| subsample / colsample_bytree | 0.8 / 1.0 |
| reg_alpha / reg_lambda | 0.1 / 5.0 |
| tree_method / device / max_bin | hist / cpu / 256 |
| random_state / n_jobs | 42 / 1 |

Nested CV, Optuna, grid/random search, forward selection, confidence penalty,
stacking, voting, 계층 모델, early stopping을 사용하지 않는다. `eval_set`도 전달하지 않는다.
5개 모델은 같은 단일 모델의 fold별 학습 결과이며 앙상블하지 않는다.
모든 OOF 예측을 저장·해시 고정한 뒤 점수를 계산한다. Argmax로 분류한다.

## 환경 및 명령

현재 실행 가능한 Python은 3.13.15다. launcher에 표시된 3.12 경로는 실제 파일이 없다.
이 폴더 전용 `.venv`를 사용한다. 직접 의존성은 `requirements.in`에 고정하고,
설치 검증 후 전체 의존성을 `requirements.lock.txt` 및 각 결과의 `environment.lock.txt`에 기록한다.

저장소 루트 PowerShell에서:

```powershell
python -m venv Jaehwang/three_class_xgb_baseline/.venv
& Jaehwang/three_class_xgb_baseline/.venv/Scripts/python.exe -m pip install -r Jaehwang/three_class_xgb_baseline/requirements.in
& Jaehwang/three_class_xgb_baseline/.venv/Scripts/python.exe Jaehwang/three_class_xgb_baseline/run.py --audit-only
& Jaehwang/three_class_xgb_baseline/.venv/Scripts/python.exe -m unittest discover -s Jaehwang/three_class_xgb_baseline -p test_contracts.py -v
```

두 파생 공식 사용이 명시적으로 허용된 경우에만 학습:

```powershell
& Jaehwang/three_class_xgb_baseline/.venv/Scripts/python.exe Jaehwang/three_class_xgb_baseline/run.py --allow-derived
```

`--data` 및 `--output`을 지정할 수 있다. 출력은 반드시 이 실험 폴더 안이어야 하며,
이미 존재하는 출력 폴더에는 덮어쓰지 않는다. 학습 기본 경로는 `results/`이고,
감사 전용 실행은 `outputs/<UTC run ID>/`다.

## 산출물과 해석

- `input_audit.json`, `excluded_columns.json/csv`: 라벨, 입력 해시, 누락·제외 열, 파생 공식.
- `derived_feature_audit.csv`: 원천 4열 및 파생 2열의 fold별 결측/inf와 전체 입력 집계.
- `manifest.json`, `environment.lock.txt`: 실행 전 고정 설정, 소스 해시, Python 및 패키지 버전.
- `splits.csv`, `leakage_audit.json`: 피험자 분리와 OOF 커버리지 검증.
- `fold_*/model.ubj`, `fold_*/preprocessing.json`: 각 fold 모델과 학습 중앙값·가중치 정보.
- `oof_rows.csv`, `oof_subjects.csv`, `PREDICTIONS_FROZEN.json`: fold별 확률과 예측.
- `metrics.json`, `summary_metrics.json`, `fold_metrics.csv`: Macro F1, balanced accuracy, macro OVR AUC,
  CN/MCI/Dementia별 recall, 보조 accuracy.
- `confusion_matrix.csv/png`, `classification_report.json/csv/txt`, `bootstrap_ci.csv`.
- `runtime_warnings.json`, `input_integrity.json`, `REPORT.md`, `README.md`, `COMPLETE.json`.

CI는 피험자별 고정 OOF 예측에서 클래스별 층화 bootstrap 2,000회(seed 7319)의
2.5/97.5 백분위다. 전체 재학습 불확실성은 포함하지 않는다. Dementia는 12명이며
fold별 2~3명 수준이므로 recall의 변동이 크다. 기존 연구에서 이미 살펴본 데이터와
특징을 사용하므로 새 외부 검증 성능으로 주장하지 않는다. 실행 후 성능을 보고
설정을 바꾸지 않는다.

API 근거: [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html),
[StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html).

## 1차 실행 완료 결과 (2026-09-22)

승인된 두 deterministic derived features를 fold별로 생성하여 위 고정 설정으로
정식 실행 1회를 완료했다. 결과를 보고 설정을 변경하거나 재탐색하지 않았다.
174명 모두 정확히 한 번 OOF 평가를 받았으며 피험자 중복 누수는 0이다.
결과 수치는 fold 평균이 아닌 전체 피험자 OOF를 합친 pooled 지표다.

| 지표 | OOF | 조건부 bootstrap 95% CI |
|---|---:|---:|
| Accuracy | 0.6149 | 0.5460–0.6839 |
| Macro F1 | 0.5481 | 0.4591–0.6338 |
| Balanced Accuracy | 0.5937 | 0.4842–0.6989 |
| Macro OVR ROC-AUC | 0.7246 | 0.6368–0.8057 |
| CN Recall | 0.6486 | 0.5586–0.7297 |
| MCI Recall | 0.5490 | 0.4118–0.6667 |
| Dementia Recall | 0.5833 (7/12) | 0.3333–0.8333 |

혼동행렬의 행은 실제, 열은 예측이며 순서는 CN/MCI/Dementia다.

```text
[[72, 29, 10],
 [20, 28,  3],
 [ 2,  3,  7]]
```

원천 4열의 결측/inf와 파생 2열의 결측/inf는 모두 0이었다.
분모 이상도 없었고, 원본 데이터 SHA-256은 실행 전후 일치했다.
9개 contract test가 통과했으며 저장된 OOF로 지표·혼동행렬을 다시 계산해 일치함을 확인했다.

실행 경고는 Matplotlib/pyparsing의 deprecated API 경고다. Python warning 14건을
`results/runtime_warnings.json`에 기록했으며, 스타일 로딩 중 `parseString`/`resetCache`
로그 8줄도 콘솔에 출력됐다. 학습 오류나 수치 계산 경고는 없었다. PNG 저장도 완료했다.
라이브러리 변경이나 재학습은 하지 않았다.

해석상 주의: Dementia가 12명이라 fold별 recall이 0–1로 변동한다. 일주기 0값
33명은 보존했으며, 이 결과를 기존 복원 완료 데이터의 결과와 동일 조건으로 비교하지 않는다.
전체 산출물은 [results/README.md](results/README.md), 지표와 신뢰구간은
[results/summary_metrics.json](results/summary_metrics.json), 파생 감사는
[results/derived_feature_audit.csv](results/derived_feature_audit.csv)에 있다.

## Git 공유 범위

코드·환경 lock·README와 집계 결과만 공유한다. 피험자별 OOF, 분할 목록,
fold별 모델, 상세 실행 경로가 포함된 감사 파일과 가상환경은 로컬에만 보관한다.
따라서 위 산출물 목록 중 일부는 GitHub에 없으며 원본 데이터로 재실행하면 생성된다.
