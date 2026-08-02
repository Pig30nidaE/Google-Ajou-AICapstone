# Binary Wearable BalancedFusion Google

Activity와 Sleep만으로 `CN` 대 `MCI+DEM`을 분류하는 후속 실험입니다.
MMSE·인지검사 파일은 찾거나 열지 않고, 특징 이름에도 들어갈 수 없도록 즉시
거부합니다.

목표는 Accuracy `0.90` 이상, Balanced Accuracy `0.75` 이상입니다. 두 수치는
목표이지 코드가 보장하는 값은 아닙니다. 가장 먼저 반복 nested OOF를 보고, 이미 여러
실험에서 사용한 33명 Validation은 보조 확인값으로만 해석해야 합니다.

## 이전 실험에서 바꾼 점

- 후보 1,077개 대신 사전 고정한 56개 Activity/Sleep 생리 채널 사용
- 모든 사람에게 동일한 마지막 35개 정렬 관측만 사용
- 7·14·35 관측의 robust summary를 각 fold 안에서 만들고 최대 24개만 선택
- fold-training의 CN만으로 정상 중앙값/IQR을 만들고 정상범위 이탈 정도 추가
- Google YDF를 사람 요약과 일별 multi-instance 방식으로 각각 학습
- 작은 Elastic-Net, compact TabNet, temporal Transformer도 함께 비교
- 모델마다 inner fold별 OOF percentile을 구하고 그 평균으로 점수 척도를 통일
- class 1을 항상 MCI+DEM으로 검증하고, 반대 방향 모델은 뒤집지 않고 탈락
- inner OOF가 약한 모델은 Google 모델도 최종 가중치 `0` 허용
- 모든 학습 모델이 약하면 억지로 쓰지 않고 상수 prior 기준 모델로 fallback
- Balanced Accuracy 중심으로 blend와 threshold 선택
- 서로 다른 outer fold는 확률이 아니라 `risk - fold threshold` margin을 평균

`SangHyo/previous/`의 Conv1D+BiLSTM 구조는 참고했지만, 저장된 예측 없이 알려진
`0.84`는 재현 가능한 근거가 아니므로 높은 가중치를 강제하지 않았습니다. 현재 저장
결과에서 더 안정적이었던 시간축 아이디어는 작은 Transformer branch로 반영했습니다.

## Colab 실행

`base.ipynb` Cell 2의 `RUN_FILE` 한 줄만 다음처럼 바꿉니다.

```python
RUN_FILE = "Binary_Wearable_BalancedFusion_Google/run.py"
```

기본 실행은 반드시 `full`입니다. 환경변수에 이전 `smoke` 값이 남아 있어도 영향을
받지 않습니다.

Full은 Colab A100/CUDA가 필요합니다. TabNet과 Transformer는 A100을 사용하고,
Google YDF와 EDA는 같은 Colab의 CPU를 사용합니다. 기본 결과 위치는 다음입니다.

```text
/content/drive/MyDrive/Binary_Wearable_BalancedFusion_Google_result/<UTC_RUN_ID>/
```

학습 없이 연결·저장 계약만 빠르게 확인할 때만 smoke를 명시합니다.

```bash
python run.py --mode smoke --output-dir /content/drive/MyDrive/tmp_balanced_smoke
```

## 평가와 누수 방지

- subject-level outer 5-fold × 2 repeats
- 각 outer training 안에서 subject-level inner 3-fold
- 대체·winsorization·scaling·특징 선택·CN 기준·모델 gate·blend·threshold를
  모두 현재 fold의 training 안에서만 학습
- 일별 YDF도 split은 사람 단위이며, 한 사람의 35개 행이 다른 fold로 갈라지지 않음
- 일별 행 가중치의 합은 사람마다 동일하여 기록 행이 많은 사람이 지배하지 않음
- Validation 예측 CSV와 SHA-256을 먼저 저장한 뒤에만 Validation 정답을 열음
- Training/Validation subject ID가 하나라도 겹치면 즉시 중단

Balanced Accuracy는 CN을 맞힌 비율과 MCI+DEM을 맞힌 비율의 평균입니다. CN이 많은
데이터에서 Accuracy만 높고 환자를 거의 못 찾는 문제를 막기 위해 primary 모델 선택
기준으로 사용합니다. 같은 inner OOF 안에서 Balanced Accuracy 0.65 이상을 우선
제약으로 둔 뒤 Accuracy가 가장 높은 보조 threshold도 미리 정해 저장합니다. 따라서
결과 보고서에는 balanced primary와 high-accuracy secondary가 함께 나오며,
Validation 정답을 보고 둘 중 하나를 사후 선택하지 않습니다.

## 체크포인트

`training/checkpoints/repeat_XX_fold_XX/`에 outer 10개 bundle을 저장합니다.
각 bundle에는 다음이 있습니다.

- fold-local value preprocessing, 최대 24개 특징 선택, CN reference
- Elastic-Net
- Google YDF subject model
- Google YDF daily multi-instance model
- compact TabNet
- temporal Transformer
- score 방향, percentile normalizer, blend weight와 threshold
- 파일별 SHA-256 manifest
- CPU 재로딩 확률 일치 검사와 `CHECKPOINT_COMPLETE.json`

이 10개 cross-fold bundle의 margin 평균이 primary 배포 ensemble입니다. 이전
실험처럼 검증되지 않은 새 hyperparameter 조합으로 full-refit을 만들거나, 서로 다른
확률 척도를 그대로 옮기는 방식은 의도적으로 사용하지 않습니다.

## 주요 산출물

```text
eda/EDA_REPORT_KO.md
training/nested_cv_report.json
training/nested_oof_predictions_hashed.csv
training/outer_fold_metrics.csv
training/feature_selection_stability.json
training/checkpoint_index.json
training/ENSEMBLE_DEPLOYMENT.json
training/VALIDATION_PREDICTIONS_FROZEN.json
training/validation_report.json
training/FINAL_REPORT.json
training/TRAINING_COMPLETE.json
```

예상 시간은 A100에서 수 시간 이내이며 5시간 40분 soft limit, 6시간 hard limit을
둡니다. 실제 시간은 Colab CPU 할당과 YDF 실행 속도에 따라 달라질 수 있습니다.

## 실행 로그 참고

`ConvergenceWarning`, TabNet의 “No early stopping”, Transformer의
`enable_nested_tensor` 메시지는 경고이며 실행 중단 원인이 아닙니다. 초기 버전에서
체크포인트 검증 중 TabNet 본체만 CPU로 옮기고 내부 group matrix가 CUDA에 남아
발생했던 mixed-device 오류는, 등록되지 않은 matrix tensor까지 함께 CPU로 이동하고
남은 비-CPU tensor를 검사하도록 수정했습니다.

## 해석상 추가 주의

코드 내부의 outer test 누수는 막았지만, 같은 141명 코호트의 이전 실험과 EDA를
보면서 이번 표현과 모델 후보를 정했다는 실험 전체 수준의 선택 편향은 남습니다.
따라서 새 nested OOF는 정직한 내부 비교값이지만 완전히 처음 보는 코호트의 독립
성능 추정치는 아닙니다. 최종 성능 주장은 별도의 새 대상자 데이터로 확인해야 합니다.
