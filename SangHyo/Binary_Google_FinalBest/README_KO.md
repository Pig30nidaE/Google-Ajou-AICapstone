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

## 평가·정직성

- 사람 단위 **5-겹 × 5-반복 중첩 교차검증**, 임계값 3종(0.5/균형/정확도) 보고.
- 검증 33명은 예측 동결(SHA-256) 후 1회만 채점. **검증 라벨을 보고 조합·임계값을
  바꾸지 않습니다.**

## 성능 기대치 (정직하게)

- 목표: 정확도 **0.90**, 균형정확도 **0.80**.
- **정직한 천장은 검증 정확도 0.879(29/33)** 입니다. 앞선 분석에서 전처리 5종을
  다 바꿔도 정확히 0.879에서 멈췄고, 그 이유는 **검증셋 인지저하 7명 중 3명이
  MMSE 만점(29·30·30점)인 MCI** 라 정상과 구별할 정보 자체가 없기 때문입니다.
- 따라서 이 최종 모델의 목표는 "**천장(0.879)을 가장 안정적으로 달성하고, 실제
  구별력(OOF AUC ≈0.74~0.76)을 최대화**"하는 것입니다. 0.90 도달은 그 만점 MCI
  한 명을 잡아야 가능한데, 이는 데이터에 정보가 없어 어떤 모델로도 보장할 수
  없습니다(코드가 아닌 데이터의 한계).

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

## 출처 (구글 모델)

- [Google — Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
- [Google Research — TabNet](https://research.google/pubs/tabnet-attentive-interpretable-tabular-learning/)

이 코드는 연구용이며 의료 진단 도구가 아닙니다.
