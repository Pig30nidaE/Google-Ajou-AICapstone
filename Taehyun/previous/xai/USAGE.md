# `xai` 패키지 사용법

ML4 프로젝트의 SHAP 분석 공용 패키지. **트리 기반(LGBM, RF, XGB)** 과 **Keras 시퀀스 모델(LSTM 등)** 둘 다 같은 인터페이스로 분석 가능.

## 목차

1. [설치 / 의존성](#1-설치--의존성)
2. [빠른 시작](#2-빠른-시작)
3. [트리 모델 예시 (LGBM / RF)](#3-트리-모델-예시-lgbm--rf)
4. [LSTM 모델 예시 (Keras)](#4-lstm-모델-예시-keras)
5. [출력 형태](#5-출력-형태)
6. [API 레퍼런스](#6-api-레퍼런스)
7. [자주 묻는 것](#7-자주-묻는-것)

---

## 1. 설치 / 의존성

- `shap` (pip install shap)
- `numpy`, `pandas`, `matplotlib`
- 모델 라이브러리는 호출자가 이미 설치한 것을 사용 (lightgbm, scikit-learn, tensorflow 등)

import 경로는 프로젝트 루트 기준:

```python
from xai import ShapAnalyzer
```

> 참고: 폴더명을 `shap`이 아니라 `xai`로 둔 이유는 pip `shap` 라이브러리와의 이름 충돌 회피.

---

## 2. 빠른 시작

```python
from xai import ShapAnalyzer

analyzer = ShapAnalyzer(
    model=trained_model,
    feature_names=feature_names,
    n_classes=3,
    class_names=["CN", "MCI", "Dementia"],
)

analyzer.explain(X_test)

# 원하는 출력만 골라서 호출
df = analyzer.to_dataframe()
analyzer.save_plots("outputs/shap", kinds=["summary", "bar"])
raw = analyzer.raw_values()
```

또는 한 줄로:

```python
result = ShapAnalyzer(model, feature_names, n_classes=3).analyze(
    X_test,
    outputs={"dataframe", "plot"},
    plot_dir="outputs/shap",
    plot_kinds=["summary", "bar"],
)
# result["dataframe"], result["plot_paths"]
```

---

## 3. 트리 모델 예시 (LGBM / RF)

`LightGBM_RF_Model.py`에서 학습된 모델을 그대로 넘기면 됨.

```python
from xai import ShapAnalyzer

# 학습 후
analyzer = ShapAnalyzer(
    model=final_lgbm,                    # 또는 final_rf
    feature_names=feature_cols,          # 학습에 쓴 컬럼 리스트
    n_classes=3,
    class_names=["CN", "MCI", "Dementia"],
)

analyzer.explain(X_test)                  # DataFrame 그대로 OK

# 클래스별 long-form DataFrame
df = analyzer.to_dataframe()
df.to_csv("outputs/shap_lgbm_importance.csv", index=False)

# 클래스 평균 단일 랭킹
df_avg = analyzer.to_dataframe(combine_classes=False)

# 플롯 (클래스별 png)
analyzer.save_plots(
    "outputs/shap_lgbm",
    kinds=["summary", "bar", "beeswarm"],
    max_display=20,
)
```

`model_type`은 자동 감지되지만 명시도 가능:

```python
ShapAnalyzer(model=final_lgbm, feature_names=feature_cols, model_type="tree", ...)
```

---

## 4. LSTM 모델 예시 (Keras)

`train_lstm_colab.py` 산출물 기준. 입력은 3D `(N, 7, 71)`.

```python
from xai import ShapAnalyzer

analyzer = ShapAnalyzer(
    model=keras_model,                   # tf.keras.Model
    feature_names=feature_names,         # 길이 71
    model_type="keras_sequence",         # 명시 권장 (자동 감지도 작동)
    n_classes=3,
    class_names=["CN", "MCI", "Dementia"],
    background=X_train_scaled[:200],     # 권장: train에서 200개 정도
    aggregation="abs_mean",              # 시간축 집계 방식
    random_state=42,
)

analyzer.explain(X_test_scaled)          # (N, 7, 71)
df = analyzer.to_dataframe()
analyzer.save_plots("outputs/shap_lstm", kinds=["summary", "bar"])

# 시간축 보존 (3D 원본 SHAP) 필요시
raw_per_class = analyzer.per_class_values()   # 각 (N, 7, 71)
```

### background를 안 넘기면?

자동으로 `X`에서 `background_size`(기본 100)개를 시드(`random_state`) 고정 샘플링.

```python
ShapAnalyzer(model, feature_names, model_type="keras_sequence",
             background_size=200)  # background 인자 생략
```

명시적으로 넘기는 게 재현성 면에서 권장.

### 시간축 집계 모드

| `aggregation` | 의미 | 사용 시점 |
|---|---|---|
| `abs_mean` (기본) | `mean(\|shap\|, axis=time)` | 표준 importance 플롯, 트리와 직접 비교 |
| `mean` | `mean(shap, axis=time)` | 부호 있는 평균 기여도 |
| `sum` | `sum(shap, axis=time)` | 총 영향력 (시퀀스 길이 의존) |
| `last` | `shap[:, -1, :]` | 마지막 시점만 |
| `none` | passthrough | 시간축 분석 직접 하고 싶을 때 |

---

## 5. 출력 형태

세 가지 모두 선택 가능. 각각 독립 호출.

### 5-1. `raw_values()` / `per_class_values()` / `aggregated_values()`

- `raw_values()`: 백엔드가 준 그대로 (버전/모델에 따라 shape 다름).
- `per_class_values()`: `{class_name: ndarray}`. 트리 `(N, F)`, LSTM `(N, T, F)`. 시간축 **보존**.
- `aggregated_values()`: 위에서 시간축을 `self.aggregation`으로 collapse한 `(N, F)`.

### 5-2. `to_dataframe(combine_classes=True)`

long-form DataFrame:

| class | feature | mean_abs_shap | mean_signed_shap |
|---|---|---|---|
| CN | raw_sleep_score_rem | 0.0421 | -0.0103 |
| ... | ... | ... | ... |

`combine_classes=False`면 클래스 평균 단일 랭킹:

| feature | mean_abs_shap |
|---|---|
| raw_sleep_score_rem | 0.0398 |
| ... | ... |

### 5-3. `save_plots(out_dir, kinds=[...])`

`out_dir`에 `shap_<kind>_<class>.png` 형태로 저장. 지원 kinds:

- `"summary"` — 점도형 summary plot
- `"bar"` — 평균 |SHAP| 막대
- `"beeswarm"` — dot plot
- `"violin"` — violin plot

기본 클래스별 분리 저장. `per_class=False`면 클래스 평균 1장.

---

## 6. API 레퍼런스

```
ShapAnalyzer(
    model,
    feature_names: list[str],
    *,
    model_type: str = "auto",            # "tree" | "keras_sequence" | "auto"
    task: str = "multiclass",            # "multiclass" | "binary" | "regression"
    n_classes: int = 3,
    class_names: list[str] | None = None,
    background: np.ndarray | None = None,
    background_size: int = 100,
    aggregation: str = "abs_mean",       # AGGREGATIONS 중
    random_state: int = 42,
)
  .explain(X) -> self
  .raw_values() -> np.ndarray
  .per_class_values() -> dict[str, np.ndarray]
  .aggregated_values() -> dict[str, np.ndarray]
  .to_dataframe(combine_classes: bool = True) -> pd.DataFrame
  .save_plots(out_dir, kinds=("summary","bar"), max_display=20, per_class=True) -> list[Path]
  .analyze(X, outputs={"raw","dataframe","plot"}, plot_dir=None, plot_kinds=...) -> dict
```

상수:
- `AGGREGATIONS = ("abs_mean", "mean", "sum", "last", "none")`

---

## 7. 자주 묻는 것

**Q. 학습 코드에서 어떻게 import?**
프로젝트 루트가 `PYTHONPATH`에 있어야 함. 예: 학습 스크립트 상단에서
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from xai import ShapAnalyzer
```

**Q. 트리 모델인데 `model_type='auto'`가 실패함.**
sklearn-호환 클래스가 아닐 수 있음. `model_type="tree"`로 강제.

**Q. LSTM SHAP이 너무 느림.**
GradientExplainer는 background 크기에 비례해 느려짐. `background_size`를 100~200 정도로 유지. 또한 `X_test`도 전체 대신 200~500개 샘플만 넘기는 게 일반적.

**Q. 클래스 개수가 3이 아님.**
`n_classes`와 `class_names`를 명시.

**Q. 출력 폴더는 자동 생성됨?**
ㅇㅇ. `save_plots(out_dir)`이 없는 경로면 `mkdir -p` 동작.

**Q. 시간축 분석을 직접 하고 싶음.**
`aggregation="none"`으로 두고 `per_class_values()` 호출. `(N, 7, 71)` 배열 반환.

**Q. `analyze()`와 메서드 분리 호출 중 뭘 써야 함?**
한 번 돌리고 끝이면 `analyze()`. 같은 `X`로 여러 출력/플롯을 단계별로 만들 거면 메서드 분리(`explain` 1회 + 출력 N회) 호출이 효율적.
