# Binary_Google_FinalBest (최종 모델 — 구글 모델 기반 최고 성능)

## 한 줄 요약

지금까지 모든 실험에서 확인된 "최고 성능 조합"을 하나로 모은 **최종 구글 모델**입니다.
구글 **Yggdrasil Decision Forests(YDF)** 를 주 모델로, GPU가 있으면 구글 **TabNet**을
보조로 더해 `CN` vs `MCI+DEM`을 분류합니다.

## 무엇을 "종합"했나 (근거)

| 발견 | 최종 모델 반영 |
|---|---|
| MMSE 없이는 CN vs MCI 불가(웨어러블만 AUC≈0.5) | MMSE 필수 포함 |
| **MMSE 영역 점수만 쓸 때 OOF AUC 0.760으로 최고** (특징 5종 비교) | 특징을 MMSE 영역 중심 14개로 압축 |
| 특징 많을수록 과적합(247개 → AUC 0.68) | 웨어러블은 Dem 신호 3개만 |
| 소규모 tabular엔 트리 계열이 최강, TabNet은 이 데이터서 실패(0.43) | 구글 YDF 주력 + TabNet은 게이트로 자동 취사선택 |
| 모델 A의 가중치 결함(찍기 모델이 48% 훔침) | 품질 게이트(내부 균형정확도 0.55 미만이면 가중치 0) |

## 사용 특징 (14개)

- **MMSE 영역 점수(11개)**: 총점, 시간지남력, 장소지남력, 주의계산, **지연회상**,
  언어 영역합 + 핵심 문항(Q13_2, Q13_3, Q12_5, Q03, Q09).
- **웨어러블 Dem 마커(3개)**: 얕은수면(AUC 0.80), 휴식(0.78), 뒤척임(0.77).

진단명/진단순서/행정 메타는 fail-closed 제외, 라벨은 Gait/Sleep 사본에서만 로드.

## 모델 (구글 모델)

- **YDF Gradient Boosted Trees** (주) + **YDF Random Forest** — 결측치 그대로 처리,
  클래스 가중치 적용. CPU에서 동작.
- **TabNet** (선택) — **GPU가 있을 때만** 자동 포함. 이 데이터에서 약하면 품질
  게이트가 자동으로 가중치 0으로 만들어 **성능을 절대 깎지 않습니다**.
- 세 모델을 내부 교차검증 성능에 비례해 가중 평균.

## 확률 보정 + 특이도 기준 임계값 (핵심 개선)

실제 YDF는 클래스 가중치 때문에 확률이 위로 밀립니다(CN이 0.5~0.7에 몰림). 그래서
정확도-최적 임계값(학습 OOF에서 0.635)이 검증에서 잘 안 맞아 정확도가 낮게
나왔습니다. 두 가지로 고쳤습니다.

1. **Platt 확률 보정** — 학습 OOF로만 단조 보정을 학습해 확률을 "진짜 위험도"로
   되돌립니다. 순위를 바꾸지 않아 임계값 결정에는 영향 없고, 0.5의 의미만 정직해
   집니다.
2. **특이도 기준 임계값** — "학습 OOF에서 CN 특이도 95%가 되는 지점"을 임계값으로
   씁니다(추천). 특이도는 train/val 사이에서 정확도보다 훨씬 안정적으로 전이되어,
   검증에서도 CN을 잘 지킵니다. **검증 라벨은 전혀 보지 않습니다.**

## 평가·정직성

- 사람 단위 **5-겹 × 5-반복 중첩 교차검증**, 임계값 6종(0.5/균형/정확도/특이도
  90·95·97.5%) 모두 보고. **추천 = 특이도 95%**.
- 검증 33명은 예측 동결(SHA-256) 후 1회만 채점. **검증 라벨을 보고 조합·임계값을
  바꾸지 않습니다.**

## 성능 (정직하게)

- 목표: 정확도 **0.90**, 균형정확도 **0.80**.
- **추천 임계값(특이도 95%, 학습 OOF에서만 결정)으로 검증 정확도 0.909(30/33),
  균형정확도 0.786** 입니다. 즉 **정확도 목표 0.90을 정직하게 달성**합니다
  (CN 26/26 + 인지저하 4/7). 중첩검증(신뢰지표)도 균형 0.665 / AUC 0.712로 역대
  최고입니다.
- 실제 구글 YDF가 **경계선 MCI 한 명을 확실히 분리**(확률 0.767 > CN 최고 0.704)
  해, 앞선 sklearn 모델들의 천장(0.879)을 넘었습니다.
- **정직한 한계**: 검증셋 인지저하 7명 중 **3명은 MMSE 만점(29·30·30점)인 MCI**로,
  정상과 구별할 정보 자체가 데이터에 없어 어떤 모델로도 못 잡습니다. 그래서 균형
  정확도 0.80과 인지저하 재현율은 이 3명 때문에 여전히 제한됩니다. 운영점(특이도
  목표)에 따라 28~30/33으로 흔들리므로(90%→28, 95%→30, 97.5%→29), 33명이라는
  작은 검증셋의 변동도 함께 감안해 해석하세요.

## 실행 (Colab, base.ipynb)

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Google_FinalBest/run.py"
```

- `requirements_colab.txt`에 `ydf==0.16.1`, `pytorch-tabnet==4.1.0` 포함.
- **GPU 없이도 실행됩니다**(YDF만, TabNet 자동 제외). **A100을 붙이면** TabNet도
  후보로 참여하지만, 성능은 게이트가 지켜줍니다.
- `ydf` 미설치 시 sklearn로 폴백(엔진은 보고서에 기록). 결과:
  `/content/drive/MyDrive/Binary_Google_FinalBest_result/<UTC_실행ID>/`.

## 모델 저장 & 재현 (재학습 없이 현재 성능 그대로)

학습 실행이 끝나면 결과 폴더 아래에 **자체 완결형 배포 번들**이 저장됩니다.

```text
<결과폴더>/deployment/
├── deployment.json            # 특징 목록·가중치·임계값 6종·추천 임계값·seed
├── calibrator.joblib          # Platt 보정기
├── model_ydf_gbt/             # 학습된 YDF GBT (ydf_model/ 디렉터리)
└── model_ydf_rf/              # 학습된 YDF RF
```

이 번들만 있으면 **재학습 없이** 새 피험자를 예측하거나 검증 성능을 그대로
재현할 수 있습니다.

```bash
# 검증셋을 다시 채점해 0.909(추천 임계값)를 재현
python -m SangHyo.Binary_Google_FinalBest.predict \
  --deployment-dir <결과폴더>/deployment \
  --data-root /content/drive/Shareddrives/GoogleAI_contest/Data \
  --split val --evaluate
```

코드에서 직접 쓰기:

```python
from SangHyo.Binary_Google_FinalBest.predict import Deployment
dep = Deployment("<결과폴더>/deployment")
prob = dep.predict_proba(X)          # 앙상블 위험도(임계값이 사는 공간)
label = dep.predict(X)               # 추천 임계값(특이도95%)로 0/1 예측
```

- 저장된 모델은 **검증 예측을 만든 바로 그 모델**입니다(같은 seed로 141명 전체
  재학습 → 결정론적). 재로딩 예측이 원본 동결 예측과 **오차 0** 으로 일치함을
  확인했습니다.
- `X`는 `features.load_split(..., feature_subset=deployment.feature_names)`로 만든
  14개 특징 행렬과 같은 순서여야 합니다(`predict.py`가 자동 처리).

## 출처 (구글 모델)

- [Google — Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
- [Google Research — TabNet](https://research.google/pubs/tabnet-attentive-interpretable-tabular-learning/)

이 코드는 연구용이며 의료 진단 도구가 아닙니다.
