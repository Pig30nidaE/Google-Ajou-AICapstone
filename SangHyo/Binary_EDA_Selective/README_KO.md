# Binary_EDA_Selective (모델 3 — EDA 기반 선택 모델)

## 한 줄 요약

`SangHyo/EDA/` 분석과 자체 EDA에서 **실제로 판별력이 확인된 소수(14개) 특징만**
골라, 해석 가능한(interpretable) 단순 모델로 `CN` vs `MCI+DEM`을 분류합니다.
자세한 EDA 근거는 같은 폴더의 [EDA_REPORT_KO.md](EDA_REPORT_KO.md)에 있습니다.

## 설계 철학: "적게, 그러나 확실하게"

이전 실험들의 교훈은 **특징이 많을수록 141명 데이터에서 과적합**된다는 것이었고
(모델 A/B, TabNet 1,077개 특징 실패), EDA는 판별력이 **극소수 특징에 몰려**
있음을 보여줍니다. 그래서 이 모델은:

- **EDA로 검증된 14개 특징만** 사용(아래 목록).
- 단순·강한 규제 모델만 사용: **보정된 L2 로지스틱 회귀** + **얕은 GBT**.
- 별도 특징선택 없음(이미 엄선됨) → 해석과 재현이 쉬움.

로컬 참고값에서 이 모델의 **중첩검증 AUC(0.717)와 신뢰구간이 세 모델 중 가장
좋았습니다** — 적은 특징이 오히려 잘 일반화함을 보여줍니다.

## 사용하는 14개 특징 (EDA 근거)

MMSE(8): `TOTAL`, `recall`(지연회상 영역합), `Q13_2`, `Q13_3`, `Q12_5`,
`orient_time`(시간지남력), `attention`(주의계산), `Q03`.

웨어러블(6, 사람 단위 평균): `sleep_light`(얕은수면), `activity_rest`(휴식),
`sleep_restless`(뒤척임), `activity_score_meet_daily_targets`(하루목표달성),
`sleep_duration`(수면시간), `sleep_score_deep`(깊은수면점수).

선정 기준(방향성 제거 AUC):
- CN vs MCI(어려운 경계): 지연회상 0.704, TOTAL 0.695, Q13_3 0.670 등 → MMSE.
- 치매(Dem) 판별: 얕은수면 0.80, 휴식 0.78, 뒤척임 0.77 등 → 웨어러블.

진단명/진단순서/행정 메타는 특징에 포함되지 않으며(fail-closed), 라벨은
Gait/Sleep 사본에서만 로드합니다.

## 모델 구성·평가

- `logreg`(보정된 L2 로지스틱 회귀) + `gbt_shallow`(깊이 3의 얕은 부스팅),
  클래스 가중치 적용.
- 사람 단위 **5-겹 × 5-반복 중첩 교차검증** + 품질 게이트 + 검증 동결(1회 채점).
- 임계값 3종(0.5 / 균형정확도 / 정확도) 보고.

## 목표와 정직한 기대치

- 목표: 정확도 **0.90**, 균형정확도 **0.80**.
- 로컬 참고값: 중첩검증 AUC ≈**0.717**(세 모델 중 최고), 균형정확도 ≈0.65(0.5)
  ~0.69(정확도 임계값), 검증 33명 정확도 **최대 0.879**. 신뢰구간 AUC
  [0.63, 0.80].
- 근본 한계(**CN vs MCI의 MMSE 겹침**)로 중첩검증 균형정확도 0.80은 아직
  넘지 못합니다. 다만 세 모델 중 **가장 안정적으로 일반화**되는 선택입니다.

## 실행 (Colab, base.ipynb)

```python
USER_FOLDER = "SangHyo"
RUN_FILE = "Binary_EDA_Selective/run.py"
```

- **GPU 불필요, CPU 수 분**. 결과는
  `/content/drive/MyDrive/Binary_EDA_Selective_result/<UTC_실행ID>/`.

## 주요 산출물

```text
EDA_REPORT_KO.md                # EDA 근거(특징 선택 이유·AUC)
eda/eda_summary.json            # 실행 시점 특징별 AUC
training/nested_cv_report.json  # 중첩 OOF 성능
training/FINAL_REPORT.json / validation_report.json / VALIDATION_PREDICTIONS_FROZEN.json
```

이 코드는 연구용이며 의료 진단 도구가 아닙니다.
