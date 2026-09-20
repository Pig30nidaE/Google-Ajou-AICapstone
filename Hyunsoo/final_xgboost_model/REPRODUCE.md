# 재현 방법

이 폴더는 보고서에 적힌 **14개 장기 라이프로그 특징 기반 단일 XGBoost** 결과를 다시 실행하기 위한 코드 묶음입니다. 보고서와 저장된 결과표는 기존 최종 결과를 그대로 보존했습니다.

원자료에는 참여자 식별 정보가 포함될 수 있어 GitHub에 올리지 않았습니다. 따라서 완전한 수치 재현에는 연구팀이 보관한 원본 `patient_level_circadian_v3.csv`가 필요합니다.

## 1. 실행 환경

Python 3.12 환경을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. 입력 파일 준비

원본 CSV를 `data/patient_level_circadian_v3.csv`에 둡니다. `data/`의 CSV 파일은 Git에서 자동 제외됩니다.

필수 열은 다음과 같습니다.

- 식별·정답: `EMAIL`, `label`
- 모델 입력: `sleep_score_alignment`, `sleep_hr_5min_max_std`, `sleep_awake_std`, `sleep_breath_average`, `activity_score_std`, `activity_class_3_count_std`, `activity_met_min_low_std`, `sleep_restless_std`, `circadian_IV`, `circadian_IS`, `circadian_RA`, `sleep_wake_bouts_avg`
- 파생 특징 계산용: `sleep_hr_average`, `sleep_hr_lowest`

`EMAIL`은 이메일 주소일 필요가 없으며, 중복과 결측이 없는 비식별 참여자 ID로 바꾸어도 됩니다. `label`은 0과 1을 사용합니다. 코드는 `HR_drop_ratio`와 `Circadian_Strain`을 입력값에서 다시 계산합니다.

기존 실행에 사용한 파일 확인값은 아래와 같습니다.

- 참여자 수: 174명
- label 0: 111명
- label 1: 63명
- SHA-256: `1799b855b27950e8d85f0e9fc917d299094fa50cbfbeee1f916f234b2b11215d`

확인값이 다르면 같은 코드라도 결과가 달라질 수 있습니다.

## 3. 최종 XGBoost와 기존 6개 모델 앙상블 재현

다음 명령은 동일 참여자가 학습과 평가에 동시에 들어가지 않도록 5개 fold를 사용하고, 이를 5개 seed로 반복합니다. 결측값 대치와 스케일 조정은 각 학습 fold에서만 학습합니다.

```bash
python run_experiment.py \
  --data data/patient_level_circadian_v3.csv \
  --output results_original_weights \
  --seeds 42 13 73 101 2026 \
  --bootstrap 2000
```

최종 XGBoost의 지표와 95% 신뢰구간을 다시 계산합니다.

```bash
python fixed_xgb_confidence.py \
  --predictions results_original_weights/predictions.csv \
  --output reproduced_fixed_xgb_confidence.json \
  --metrics-output reproduced_fixed_xgb_metrics.csv \
  --repetitions 10000
```

6개 모델 순위 앙상블과 비교합니다.

```bash
python compare_xgb_vs_ensemble.py \
  --predictions results_original_weights/predictions.csv \
  --repetitions 10000 \
  --output reproduced_xgb_vs_ensemble_metrics.csv \
  --details-output reproduced_xgb_vs_ensemble_details.json \
  --confusion-output reproduced_xgb_vs_ensemble_confusion.csv
```

## 4. SHAP 분석 재현

```bash
python xgb_oof_shap_analysis.py \
  --data data/patient_level_circadian_v3.csv \
  --output reproduced_xgb_shap \
  --bootstrap 10000 \
  --seeds 42 13 73 101 2026
```

SHAP은 평가 대상자를 학습에 포함하지 않은 모델에서 계산됩니다. 주요 결과는 `shap_importance.csv`, `shap_direction_summary.csv`, `shap_importance.png`, `shap_direction.png`입니다.

## 5. 클래스 가중치 실험 재현

```bash
python xgb_class_weight_sensitivity.py \
  --data data/patient_level_circadian_v3.csv \
  --output reproduced_class_weight_sensitivity \
  --bootstrap 10000 \
  --seeds 42 13 73 101 2026
```

## 파일 안내

- `README.md`: 최종모델 제안 보고서
- `SHAP_임상해석.md`: SHAP 결과와 임상적 해석
- `run_experiment.py`: 최종 XGBoost와 기존 앙상블을 같은 분할에서 실행
- `fixed_xgb_confidence.py`: 최종 XGBoost의 지표와 신뢰구간 계산
- `compare_xgb_vs_ensemble.py`: XGBoost와 6개 모델 앙상블 비교
- `xgb_oof_shap_analysis.py`: 평가 데이터 누수를 피한 SHAP 분석
- `xgb_class_weight_sensitivity.py`: 클래스 가중치 변화 실험

생성되는 예측 파일에는 참여자 단위 결과가 들어 있으므로 외부 공개 전 비식별 여부를 다시 확인해야 합니다.
