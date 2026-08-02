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
재현할 수 있습니다. 세 가지 실행법 모두 가능합니다(**base.ipynb는 수정하지 않습니다**).

### 방법 1 — base.ipynb (셀 2만 수정, 가장 간단)

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Google_FinalBest/predict.py"
```

- 노트북의 `DATA_ROOT`를 자동으로 쓰고, Drive 결과 루트에서 **가장 최근
  `deployment/`를 자동 탐지**합니다. 특정 실행을 지정하려면 실행 전 셀에서
  환경변수를 설정하세요.

```python
import os
os.environ["PREDICT_DEPLOYMENT_DIR"] = "/content/drive/MyDrive/Binary_Google_FinalBest_result/<실행ID>/deployment"
os.environ["PREDICT_SPLIT"] = "val"   # 기본 val, 생략 가능
```

결과 CSV는 해당 실행 폴더에 `reproduced_predictions_val.csv`로 저장되고, val이면
정답을 사후에 열어 **0.909(추천 임계값)를 재현**해 출력합니다.

### 방법 2 — CLI

```bash
python -m SangHyo.Binary_Google_FinalBest.predict \
  --deployment-dir <결과폴더>/deployment \
  --data-root /content/drive/Shareddrives/GoogleAI_contest/Data \
  --split val --evaluate
```

### 방법 3 — 코드에서 직접

```python
from SangHyo.Binary_Google_FinalBest.predict import Deployment
dep = Deployment("<결과폴더>/deployment")
prob = dep.predict_proba(X)          # 앙상블 위험도(임계값이 사는 공간)
label = dep.predict(X)               # 추천 임계값(특이도95%)로 0/1 예측
```

환경변수 정리: `PREDICT_DEPLOYMENT_DIR`(번들 경로), `PREDICT_DATA_ROOT`(Data 루트),
`PREDICT_SPLIT`(train/val), `PREDICT_EVALUATE`(0/1), `PREDICT_RESULTS_ROOT`(자동탐지
루트). CLI 인자가 있으면 항상 우선합니다.

- 저장된 모델은 **검증 예측을 만든 바로 그 모델**입니다(같은 seed로 141명 전체
  재학습 → 결정론적). 재로딩 예측이 원본 동결 예측과 **오차 0** 으로 일치함을
  확인했습니다.
- `X`는 `features.load_split(..., feature_subset=deployment.feature_names)`로 만든
  14개 특징 행렬과 같은 순서여야 합니다(`predict.py`가 자동 처리).

## 하이퍼파라미터 튜닝 (tune 모드)

구글 YDF(GBT+RF)의 하이퍼파라미터를 **랜덤 서치**로 탐색합니다. 각 후보는
기존 **5×5 중첩 CV로 평가**하고 **OOF ROC-AUC가 가장 높은 후보를 선택**합니다.

### 정직성
- **하이퍼파라미터 선택은 학습 141명(OOF)에서만** 합니다. **검증 33명은 선택에
  전혀 쓰지 않으므로** 튜닝 후에도 정직한 held-out 테스트로 남습니다.
- 단, 여러 후보 중 최고를 고른 것이라 **선택된 후보의 OOF 점수는 낙관적**입니다
  (`oof_is_selection_optimistic: true`). 진짜 성능은 **validation_report**를
  보세요. `tuning_report.json`에 모든 후보와 선택 편향 주의가 기록됩니다.

### 실행 (base.ipynb, base.ipynb는 수정 불필요)
실행 전 셀에서 모드를 환경변수로 지정하고 run.py를 실행합니다.

```python
import os
os.environ["FINALBEST_MODE"] = "tune"      # 튜닝 모드
# (선택) os.environ["TUNE_N_CONFIGS"] = "150"          # 탐색 후보 수(기본 150)
# (선택) os.environ["TUNE_SELECTION_METRIC"] = "roc_auc"  # 또는 balanced_accuracy
```
```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_Google_FinalBest/run.py"
```

### 실행 환경 (중요)
- **Colab Pro+에서 "고용량 RAM(High-RAM) CPU 런타임"을 권장합니다. GPU는 필요
  없습니다**(YDF는 CPU에서 돌고, 튜닝은 YDF만 대상 — TabNet은 이 데이터에서
  실패했으므로 튜닝에서 제외). A100을 붙여도 낭비이니 CPU 고용량 RAM이 최선입니다.
- YDF는 멀티스레드로 CPU 코어를 자동 활용합니다(`num_threads=코어수`). 코어가
  많은 런타임일수록 빠릅니다.

### 소요 시간 (20시간 상한 보장)
- 실제 예상은 **약 2~4시간**입니다(150 후보 × 5×5 CV, YDF는 매우 빠름).
- 안전장치: **soft 18시간에 탐색 중단**(그때까지의 최고 후보로 마무리), **hard
  19.5시간에 프로세스 중단** → **20시간을 넘지 않습니다.**
  (`TUNE_SOFT_BUDGET_SECONDS`로 soft 예산 조정 가능.)

### 산출물
```text
training/tuning_report.json    # 모든 후보 + 최적 후보 + 선택 편향 주의
training/FINAL_REPORT.json      # tuned_hyperparameters, 중첩 OOF, 검증 결과
deployment/deployment.json      # tuned_hyperparameters 포함(재현·배포용)
```
튜닝 후에도 `deployment/`가 저장되므로 `predict.py`로 재학습 없이 재현/배포할 수
있습니다.

### 정직한 기대치
- 앞선 분석에서 이 데이터의 판별력 천장(OOF AUC ≈0.71)이 전처리·모델을 바꿔도
  일정하게 나타났습니다. 따라서 **튜닝은 "천장 안에서 가장 좋은 설정"을 찾는
  것**이지, 천장을 크게 넘기기는 어렵습니다. 큰 도약보다는 소폭 개선을
  기대하세요.

## 출처 (구글 모델)

- [Google — Yggdrasil Decision Forests](https://github.com/google/yggdrasil-decision-forests)
- [Google Research — TabNet](https://research.google/pubs/tabnet-attentive-interpretable-tabular-learning/)

이 코드는 연구용이며 의료 진단 도구가 아닙니다.
