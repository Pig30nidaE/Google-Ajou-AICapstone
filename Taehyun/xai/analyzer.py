"""ShapAnalyzer -- 모델 의존성 없는 SHAP 실행기

- ML4 프로젝트의 멀티클래스 분류기 대상으로 설계
- 트리(sklearn 호환)와 시퀀스(tf.keras) 모델을 동일 호출 패턴으로 처리

전형적 사용:

    analyzer = ShapAnalyzer(model, feature_names)
    analyzer.explain(X_test)
    df = analyzer.to_dataframe()
    analyzer.save_plots("outputs/shap", kinds=["summary", "bar"])
    raw = analyzer.raw_values()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .aggregation import AGGREGATIONS, aggregate_time_axis
from .detect import detect_model_type
from .explainers import KerasSequenceAdapter, TreeAdapter
from .outputs import importance_dataframe, save_plots


class ShapAnalyzer:
    """학습된 분류기의 SHAP 값을 계산하고 다양한 형태로 제공

    Parameters
    ----------
    model
        학습된 모델. 트리(LGBM/RF/XGB 등) 또는 tf.keras 시퀀스 모델.
    feature_names
        feature 이름 리스트 (순서대로). X의 마지막 축 길이와 일치해야 함.
    model_type
        "tree", "keras_sequence", "auto"(기본). auto면 detect_model_type 호출.
    task
        "multiclass"(기본) / "binary" / "regression". 출력 클래스 슬라이싱 방식 결정.
    n_classes
        멀티클래스 클래스 수. 그 외 task에서는 무시.
    class_names
        클래스 표시명 리스트(길이 n_classes). 미지정 시 ["class_0", ...] 사용.
    background
        Keras 모델용 background 데이터.
        None이면 explain() 시 X에서 background_size개를 random_state로 샘플링.
        트리 모델에서는 무시됨.
    background_size
        background 자동 샘플링 개수 (기본 100).
    aggregation
        시퀀스 SHAP의 시간축 집계 모드. AGGREGATIONS 중 하나. 기본 "abs_mean".
        2D 입력에서는 무시됨.
    random_state
        background 자동 샘플링 시드.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        *,
        model_type: str = "auto",
        task: str = "multiclass",
        n_classes: int = 3,
        class_names: list[str] | None = None,
        background: np.ndarray | None = None,
        background_size: int = 100,
        aggregation: str = "abs_mean",
        random_state: int = 42,
    ):
        # 인자 검증 (잘못된 값은 explain 직전이 아니라 생성 시점에 발견)
        if aggregation not in AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {AGGREGATIONS}, got {aggregation!r}"
            )
        if task not in {"multiclass", "binary", "regression"}:
            raise ValueError(f"Unsupported task: {task!r}")

        self.model = model
        self.feature_names = list(feature_names)

        # auto면 모듈 경로/MRO로 추정. 명시되면 그대로 사용.
        self.model_type = (
            detect_model_type(model) if model_type == "auto" else model_type
        )
        if self.model_type not in {"tree", "keras_sequence"}:
            raise ValueError(f"Unsupported model_type: {self.model_type!r}")

        self.task = task
        self.n_classes = n_classes
        self.class_names = (
            list(class_names)
            if class_names is not None
            else [f"class_{i}" for i in range(n_classes)]
        )
        if len(self.class_names) != n_classes and task == "multiclass":
            raise ValueError("class_names length must equal n_classes")

        self.background = background
        self.background_size = background_size
        self.aggregation = aggregation
        self.random_state = random_state

        # explain() 이후 채워지는 상태
        self._adapter = None
        self._X_explained: np.ndarray | None = None
        self._X_pandas: pd.DataFrame | None = None
        self._raw_shap: np.ndarray | None = None

    def explain(self, X) -> "ShapAnalyzer":
        """X에 대해 SHAP 값 계산. 메서드 체이닝을 위해 self 반환."""
        X_arr, X_pandas = _coerce_input(X, self.feature_names)
        self._X_explained = X_arr
        self._X_pandas = X_pandas

        if self.model_type == "tree":
            self._adapter = TreeAdapter(self.model)
            # 트리는 DataFrame을 그대로 넘기면 feature 이름이 SHAP plot에서 자동 인식됨
            shap_input = X_pandas if X_pandas is not None else X_arr
            self._raw_shap = self._adapter.shap_values(shap_input)
        else:
            # Keras는 background 필수 → 미지정 시 X에서 자동 샘플링
            background = self._resolve_background(X_arr)
            self._adapter = KerasSequenceAdapter(self.model, background)
            self._raw_shap = self._adapter.shap_values(X_arr)

        return self

    def raw_values(self) -> np.ndarray:
        """백엔드가 반환한 SHAP 값 원본

        - 형태는 shap 버전/모델에 따라 다름
        - 정규화된 형태가 필요하면 per_class_values() 사용
        """
        self._require_explained()
        return self._raw_shap

    def per_class_values(self) -> dict[str, np.ndarray]:
        """{class_name: shap_array} 반환. 클래스 축은 제거됨.

        - 트리   : 각 배열 shape (N, F)
        - 시퀀스 : 각 배열 shape (N, T, F)  -- 시간축 보존
        """
        self._require_explained()
        sliced = _slice_classes(self._raw_shap, self.n_classes, self.task)
        return {name: sliced[i] for i, name in enumerate(self.class_names)}

    def aggregated_values(self) -> dict[str, np.ndarray]:
        """per_class_values()에서 시간축을 self.aggregation으로 축약

        시퀀스 모델: (N, T, F) → (N, F)
        트리 모델  : 이미 (N, F)이므로 동일
        """
        return {
            name: aggregate_time_axis(v, self.aggregation)
            for name, v in self.per_class_values().items()
        }

    def to_dataframe(self, combine_classes: bool = True) -> pd.DataFrame:
        """글로벌 feature importance를 DataFrame으로 반환

        Parameters
        ----------
        combine_classes
            True  -- 클래스별 importance를 long-form으로 concat (class 컬럼 포함)
            False -- 클래스 평균을 낸 단일 랭킹 반환
        """
        per_class = self.aggregated_values()
        frames = [
            importance_dataframe(v, self.feature_names, class_label=name)
            for name, v in per_class.items()
        ]
        if combine_classes:
            return pd.concat(frames, ignore_index=True)

        # 클래스 평균 단일 랭킹 (각 클래스의 mean_abs_shap을 feature별 평균)
        merged = (
            pd.concat(frames, ignore_index=True)
            .groupby("feature", as_index=False)
            .agg(mean_abs_shap=("mean_abs_shap", "mean"))
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        return merged

    def save_plots(
        self,
        out_dir: str | Path,
        kinds: Iterable[str] = ("summary", "bar"),
        max_display: int = 20,
        per_class: bool = True,
    ) -> list[Path]:
        """SHAP 플롯을 out_dir에 저장하고 생성된 파일 경로 리스트 반환

        - per_class=True (기본)이고 멀티클래스면 클래스별로 분리 저장
        - per_class=False면 클래스 평균을 사용해 1장으로 저장
        """
        out_dir = Path(out_dir)
        per_class_vals = self.aggregated_values()
        X_2d = self._X_for_plotting()

        written: list[Path] = []
        if per_class and self.task == "multiclass":
            for name, v in per_class_vals.items():
                written.extend(
                    save_plots(
                        shap_2d=v, X_2d=X_2d, feature_names=self.feature_names,
                        out_dir=out_dir, kinds=kinds,
                        class_label=name, max_display=max_display,
                    )
                )
        else:
            # 클래스 평균 = 클래스 축으로 stack 후 평균
            stacked = np.mean(np.stack(list(per_class_vals.values()), axis=0), axis=0)
            written.extend(
                save_plots(
                    shap_2d=stacked, X_2d=X_2d, feature_names=self.feature_names,
                    out_dir=out_dir, kinds=kinds,
                    class_label=None, max_display=max_display,
                )
            )
        return written

    def analyze(
        self,
        X,
        outputs: set[str] | None = None,
        plot_dir: str | Path | None = None,
        plot_kinds: Iterable[str] = ("summary", "bar"),
    ) -> dict[str, Any]:
        """explain + 선택된 출력들을 한 번에 실행하는 편의 메서드

        outputs는 {"raw", "dataframe", "plot"}의 부분집합. 기본은 셋 다.
        "plot"은 plot_dir이 주어진 경우에만 저장.
        """
        outputs = set(outputs) if outputs is not None else {"raw", "dataframe", "plot"}
        self.explain(X)

        result: dict[str, Any] = {}
        if "raw" in outputs:
            result["raw"] = self.raw_values()
            result["per_class"] = self.per_class_values()
        if "dataframe" in outputs:
            result["dataframe"] = self.to_dataframe()
        if "plot" in outputs and plot_dir is not None:
            result["plot_paths"] = self.save_plots(plot_dir, kinds=plot_kinds)
        return result

    def _resolve_background(self, X_arr: np.ndarray) -> np.ndarray:
        """Keras용 background 결정

        - 호출자가 명시했으면 그대로 사용 (재현성 보장 측면에서 권장)
        - None이면 X에서 background_size개를 시드 고정 랜덤 샘플링
        """
        if self.background is not None:
            return np.asarray(self.background)
        rng = np.random.default_rng(self.random_state)
        n = X_arr.shape[0]
        # X 자체가 background_size보다 작은 경우도 대비
        size = min(self.background_size, n)
        idx = rng.choice(n, size=size, replace=False)
        return X_arr[idx]

    def _X_for_plotting(self) -> np.ndarray:
        """플롯에 깔리는 feature 값(2D) 결정

        - 트리 입력은 이미 (N, F)이므로 그대로
        - 시퀀스 입력 (N, T, F)는 시간축 평균으로 (N, F)에 정렬
        """
        X = self._X_explained
        if X.ndim == 3:
            return np.mean(X, axis=1)
        return X

    def _require_explained(self) -> None:
        """explain() 호출 이전에 출력 메서드가 불리면 명확한 에러로 차단"""
        if self._raw_shap is None:
            raise RuntimeError("Call .explain(X) before requesting values.")


def _coerce_input(X, feature_names: list[str]) -> tuple[np.ndarray, pd.DataFrame | None]:
    """입력을 (ndarray, 선택적 DataFrame 뷰) 튜플로 정규화

    - 트리 모델에 DataFrame을 그대로 넘기면 SHAP plot에 feature 이름이 자동 노출
    - 3D 입력은 DataFrame으로 표현 불가 → None 반환
    """
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(), X
    arr = np.asarray(X)
    # 2D이고 feature 수가 맞으면 편의 DataFrame 생성
    if arr.ndim == 2 and arr.shape[1] == len(feature_names):
        df = pd.DataFrame(arr, columns=feature_names)
        return arr, df
    return arr, None


def _slice_classes(raw: np.ndarray, n_classes: int, task: str) -> list[np.ndarray]:
    """shap 백엔드 출력을 길이 n_classes의 리스트로 정규화

    멀티클래스에서 흔한 두 가지 레이아웃 모두 처리:
    - 선행 클래스 축 : (C, N, ...) → [raw[c] for c in range(C)]
    - 후행 클래스 축 : (N, ..., C) → [raw[..., c] for c in range(C)]
    """
    if task != "multiclass":
        return [raw]

    # 선행 축 우선 검사 (C가 마지막 축에도 같은 크기로 오는 모호한 경우 회피)
    if raw.ndim >= 2 and raw.shape[0] == n_classes and raw.shape[-1] != n_classes:
        return [raw[i] for i in range(n_classes)]
    if raw.shape[-1] == n_classes:
        return [raw[..., i] for i in range(n_classes)]
    if raw.shape[0] == n_classes:
        return [raw[i] for i in range(n_classes)]

    raise ValueError(
        f"Cannot locate class axis in SHAP output of shape {raw.shape} for "
        f"n_classes={n_classes}. Pass model_type and n_classes explicitly."
    )
