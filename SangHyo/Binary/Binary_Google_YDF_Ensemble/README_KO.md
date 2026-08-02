# Binary_Google_YDF_Ensemble (모델 2 — 구글 모델)

## 한 줄 요약

구글이 공개한 **Yggdrasil Decision Forests(YDF)** 의 Gradient Boosted Trees와
Random Forest를 주 모델로 써서 `CN` vs `MCI+DEM`을 분류합니다. 특징은 모델 1과
동일하게 **MMSE 영역 점수 + 엄선 웨어러블**을 사용합니다.

## 왜 구글 모델 중 YDF인가

사용자 요청은 "구글 모델(Transformer, TabNet, Yggdrasil 등) 중 **선택**"이었고,
이 데이터에는 **YDF가 최선**입니다.

- 데이터가 **141명으로 매우 작고 표(tabular) 형태**입니다. 이런 조건에서는
  결정 트리 계열이 신경망 계열보다 안정적이고 강합니다.
- 실제로 같은 데이터의 **TabNet 실험은 균형정확도 0.43**으로 무너졌고,
  Transformer/ConvBiLSTM도 사람 단위 평가에서 찍기 수준이었습니다.
- YDF는 **결측치를 그대로 처리**하고 스케일링이 필요 없어, 소수 특징에 대해
  견고한 학습이 가능합니다.

따라서 이 실험은 **YDF GBT + YDF RF** 두 구글 모델을 클래스 가중치로 학습해
품질 게이트로 앙상블합니다. (참고: TabNet은 이 데이터에서 이미 실패했으므로
주 모델에서 제외했습니다.)

## 특징 구성

모델 1과 동일한 EDA 기반 특징:

- **MMSE 영역 점수**(총점·지남력·주의계산·**지연회상**·언어 + 핵심 문항). 진단
  관련 열은 fail-closed 제외, 라벨은 Gait/Sleep 사본에서만 로드.
- **엄선 웨어러블 사람 단위 평균**(치매를 잘 잡는 활동·수면 지표 중심).

YDF는 내부적으로 특징 선택·상호작용을 학습하므로, 별도 스케일링·특징선택
없이 원본 값을 그대로 넣습니다(결측 포함).

## 누수 방지·평가

- 사람 단위 **5-겹 × 5-반복 중첩 교차검증**.
- 품질 게이트(내부 균형정확도 ≥ 0.55인 모델만 앙상블).
- 검증 33명은 예측 동결(SHA-256) 후 1회만 채점. 임계값 3종(0.5/균형/정확도).

## 목표와 정직한 기대치

- 목표: 정확도 **0.90**, 균형정확도 **0.80**.
- 로컬 참고값은 `ydf` 미설치로 **sklearn 폴백(HistGradientBoosting)** 으로 얻은
  값입니다: 중첩검증 AUC ≈0.67, 검증 정확도 ≈0.85. **Colab에서는 실제 YDF가
  설치되어 더 나은 결과가 기대**됩니다(`nested_cv_report.json`의
  `ydf_engine_used`로 실제 엔진 확인 가능).
- 다만 모델 1과 같은 근본 한계(**CN vs MCI의 MMSE 겹침**)로, 중첩검증
  균형정확도 0.80 도달은 보장되지 않습니다.

## 실행 (Colab, base.ipynb)

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Google_YDF_Ensemble/run.py"
```

- `requirements_colab.txt`에 `ydf==0.16.1`이 포함되어 자동 설치됩니다.
- **GPU 불필요**(YDF는 CPU에서 동작). 수 분이면 끝납니다.
- 결과: `/content/drive/MyDrive/Binary_Google_YDF_Ensemble_result/<UTC_실행ID>/`.
- `ydf`가 없으면 자동으로 sklearn로 폴백해 그래도 실행됩니다(엔진은 보고서에
  기록).

## 출처 (구글 모델)

- [Google — Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
- [YDF Gradient Boosted Trees 문서](https://ydf.readthedocs.io/en/latest/py_api/GradientBoostedTreesLearner/)

이 코드는 연구용이며 의료 진단 도구가 아닙니다.
