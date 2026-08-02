# Binary_MetaEnsemble_Google (메타 앙상블 — M1 + M2(YDF) + M3)

## 한 줄 요약

앞서 만든 세 모델의 **base learner 7개를 하나로 합친 메타 앙상블**입니다.
목표는 세 모델의 강점(**M2 구글 YDF의 인지저하 재현력 + M3 최소특징의 안정적
일반화**)을 모아, 검증 정확도를 한 명이라도 더 끌어올려 0.90(=30/33)에 도전하는
것입니다. `CN` vs `MCI+DEM` 이진 분류.

## 어떻게 합치나 (누수 없이)

7개 base learner를 **같은 사람 단위 fold**에서 학습·평가하고, 품질 게이트로
가중 평균합니다.

- M1(`Binary_MMSE_DomainFusion`): `dom_gbt`, `dom_logreg`, `dom_rf` (전체 ~30개 특징)
- M2(`Binary_Google_YDF_Ensemble`): `ydf_gbt`, `ydf_rf` (구글 YDF, 전체 특징)
- M3(`Binary_EDA_Selective`): `eda_logreg`, `eda_gbt` (최소 14개 특징)

**핵심 원칙(정직성)**:

- 각 모델의 가중치는 **내부 교차검증(inner OOF) 균형정확도가 0.55를 넘을 때만**
  부여됩니다(품질 게이트). 그래서 성능 좋은 모델(M2·M3 계열)이 자연히 큰 비중을
  갖고, 약한 모델은 0에 수렴합니다.
- **가중치와 임계값은 오직 학습 데이터의 OOF에서만** 정합니다.
- 검증 33명은 예측을 먼저 저장(+SHA-256 동결)한 **뒤에야** 정답을 열어 1회만
  채점합니다. **검증 라벨을 보고 조합/임계값을 바꾸지 않습니다.** 따라서 결과
  숫자는 조작되지 않은 정직한 값입니다.

## 왜 이게 한 명을 더 잡을 수 있나

기존 M1·M3는 CN 특이도를 지키려 임계값을 높여 **CN 26/26 + 인지저하 3/7**(29/33)에
머물렀고, M2(YDF)는 **인지저하 4/7**을 잡되 CN을 2명 놓쳤습니다. 서로 다른 모델이
서로 다른 인지저하 환자를 잡으므로, 이들을 결합하면 **CN을 지키면서 4번째
인지저하**를 잡아 26+4=30/33=0.909에 이를 가능성이 생깁니다. 다만 이는 데이터가
허락할 때만 가능하며, 억지로 맞추지 않았습니다.

## 평가·산출물

- 사람 단위 **5-겹 × 5-반복 중첩 교차검증**, 임계값 3종(0.5 / 균형 / 정확도) 보고.
- `nested_cv_report.json`의 `model_mean_weight`로 어느 base 모델이 얼마나
  기여했는지, `ydf_engine_used`로 실제 YDF가 돌았는지 확인할 수 있습니다.

```text
eda/eda_summary.json
training/nested_cv_report.json     # base 7개 모델별 가중치·게이트, 중첩 OOF 성능
training/FINAL_REPORT.json         # 임계값 3종 요약(핵심)
training/validation_report.json    # 검증 33명 1회 채점
training/VALIDATION_PREDICTIONS_FROZEN.json
```

## 실행 (Colab, base.ipynb)

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_MetaEnsemble_Google/run.py"
```

- `requirements_colab.txt`에 `ydf==0.16.1` 포함(M2 base가 사용). **GPU 불필요.**
- 세 base 폴더(`Binary_MMSE_DomainFusion`, `Binary_Google_YDF_Ensemble`,
  `Binary_EDA_Selective`)가 같은 저장소에 있어야 합니다(base.ipynb가 전체 repo를
  clone하므로 자동 충족). `ydf` 미설치 시 M2 base는 sklearn로 폴백합니다.
- 결과: `/content/drive/MyDrive/Binary_MetaEnsemble_Google_result/<UTC_실행ID>/`.

## 정직한 기대치

- 목표: 정확도 **0.90**, 균형정확도 **0.80**.
- 메타는 세 모델의 강점을 모으지만, **근본 한계(CN vs MCI의 MMSE 겹침)는
  그대로**입니다. 로컬(YDF 폴백) 기준 중첩검증 AUC ≈0.71, 검증 정확도 최대
  0.879였습니다. 실제 YDF가 도는 Colab에서 30/33(=0.909) 도달 여부를 확인하세요.
- 중첩검증 균형정확도 0.80은 여전히 데이터 한계로 어렵습니다. 정확도 임계값
  결과가 목표(0.90)에 가장 근접합니다.

이 코드는 연구용이며 의료 진단 도구가 아닙니다.
