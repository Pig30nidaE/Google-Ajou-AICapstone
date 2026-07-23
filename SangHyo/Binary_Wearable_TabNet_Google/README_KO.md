# Binary Wearable TabNet Google

Activity와 Sleep만으로 `CN` 대 `MCI+DEM`을 분류하는 새 실험입니다. 인지검사 원본,
총점, 문항 점수와 파생치는 발견하거나 읽지 않으며 모델 특징에도 들어가지 않습니다.

## 왜 3-class Accuracy 0.848을 그대로 복사하지 않았나

참고 실험의 `0.84848`은 단일 TabNet 결과가 아니라 MMSE 34개 후보를 포함한
Transformer·TabNet·Google YDF 18개 체크포인트 앙상블의 historical Validation
결과였습니다. Validation 33명 중 CN 26명이라는 불균형이 있었고 MCI 4명은 모두
CN으로 분류됐습니다. 해당 실험의 TabNet-only binary 환산 OOF도 강하지 않았습니다.

따라서 이번 실험은 그 결과의 숫자를 성능 근거로 재사용하지 않습니다. 대신 다음
구조만 참고했습니다.

- 한 피험자당 한 행의 tabular 표현
- 모든 피험자에게 동일한 마지막 28개 관측의 median/mean/std/quantile/IQR/MAD/trend 변화량
- CV fold 안에서만 수행하는 대체·winsorization·robust scaling·특징 선택
- TabNet의 AdamW, cosine scheduler, binary class weight
- 여러 outer-fold/seed 체크포인트의 확률 bagging

## 모델과 검증

- 주 모델: Google Research의 TabNet (`pytorch-tabnet==4.1.0`)
- 보조 후보: Google Yggdrasil Decision Forests (`ydf==0.16.1`)
- YDF 혼합 비율은 각 outer training의 inner OOF만으로 결정됩니다. 최종 예측에는
  TabNet이 최소 25% 포함되고 TabNet 100%도 허용됩니다. 거의 같은 점수라면 TabNet
  비중이 큰 조합을 선택합니다.
- Full: subject-level outer 5-fold × 2-repeat, inner 3-fold
- 각 outer fold: TabNet/YDF 각각 4회의 제한된 Optuna 탐색
- 최종 outer 모델: TabNet 3 seeds + YDF 1개
- Primary: repeat-averaged nested OOF, threshold 0.5
- Secondary: Training OOF에서만 선택한 threshold

Accuracy 0.9는 목표이며 코드가 보장하는 결과가 아닙니다. 이미 이전 실험에서 여러
번 확인한 33명 Validation은 historical benchmark일 뿐 새로운 독립 test가 아닙니다.
Accuracy와 함께 balanced accuracy, impaired recall, specificity, ROC-AUC, PR-AUC,
confusion matrix, all-CN baseline과 bootstrap CI를 반드시 확인하십시오.

가변적인 전체 관측 기간과 최근 창을 동시에 사용하면 관측 기간/coverage를 간접적으로
복원할 수 있으므로, 모델 특징은 모든 피험자에게 동일한 마지막 28개 정렬 관측으로
고정했습니다. 전체 sequence 길이는 EDA에만 기록되고 모델에는 전달되지 않습니다.

## Colab 실행

`base.ipynb` Cell 2에서 아래 한 줄만 바꿉니다.

```python
RUN_FILE = "Binary_Wearable_TabNet_Google/run.py"
```

기본값은 반드시 `full`입니다. `smoke`는 코드와 저장 계약을 빠르게 점검할 때만
명시적으로 사용합니다.

```bash
python run.py --mode smoke --output-dir /content/drive/MyDrive/tmp_tabnet_smoke
```

Full은 Colab A100/CUDA가 필요합니다. TabNet은 A100을 사용하고 EDA 및 YDF는 같은
Colab 런타임의 CPU를 사용합니다. 기본 결과 위치는 다음과 같습니다.

```text
/content/drive/MyDrive/Binary_Wearable_TabNet_Google_result/<UTC_RUN_ID>/
```

## 체크포인트

`training/checkpoints/` 아래에 outer fold 10개와 정상 완료 시 `full_refit` bundle이
저장됩니다. 6시간 제한에 가까워져 full refit에 필요한 최소 15분이 남지 않으면,
이미 검증된 outer 10개와 primary cross-fold 예측을 보존하고 secondary full refit만
`FULL_REFIT_SKIPPED.json`으로 명시합니다.
각 bundle에는 다음이 포함됩니다.

- TabNet `model.zip` (fold당 3 seeds)
- YDF model directory
- fold-local preprocessors와 OOF calibrators
- 선택 파라미터와 feature schema/hash manifest
- `roundtrip_verification.json`
- `CHECKPOINT_COMPLETE.json`

저장 직후 모든 TabNet은 새 CPU 객체, YDF는 새 YDF 객체로 다시 로드됩니다. probe
확률이 원 모델과 일치하지 않으면 해당 실험은 실패 처리되어 완료 마커가 생성되지
않습니다.

## 주요 산출물

```text
eda/EDA_REPORT_KO.md
training/nested_cv_report.json
training/nested_oof_predictions_hashed.csv
training/feature_selection_stability.json
training/checkpoint_index.json
training/VALIDATION_PREDICTIONS_FROZEN.json
training/VALIDATION_FULL_REFIT_PREDICTIONS_FROZEN.json  # full refit 완료 시
training/validation_report.json
training/FINAL_REPORT.json
training/TRAINING_COMPLETE.json
```

Validation 확률 CSV는 정답을 읽기 전에 먼저 저장하고 SHA-256으로 동결합니다. 그
파일을 다시 읽은 뒤에만 Gait/Sleep label 복사본을 열어 평가합니다.
