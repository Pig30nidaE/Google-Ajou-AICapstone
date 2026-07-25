"""시퀀스 SHAP 값의 시간축 집계

- 트리 SHAP   : (N, F)
- 시퀀스 SHAP : (N, T, F)
- 트리와 동일한 (N, F) 형태로 통일하려면 T축 축약 필요

지원 모드:
- "abs_mean" : T축 mean(|shap|)         -- 기본값, 표준 importance 플롯용
- "mean"     : T축 mean (부호 유지)
- "sum"      : T축 sum (부호 유지, 시퀀스 길이에 비례)
- "last"     : t = -1 시점만 사용
- "none"     : 그대로 통과 (3D 반환)
"""

from __future__ import annotations

import numpy as np


# 외부 노출용 모드 튜플 (ShapAnalyzer 검증에서도 사용)
AGGREGATIONS = ("abs_mean", "mean", "sum", "last", "none")


def aggregate_time_axis(shap_values: np.ndarray, mode: str = "abs_mean") -> np.ndarray:
    """3D SHAP 배열의 시간축(axis=1)을 축약해 2D로 반환

    Parameters
    ----------
    shap_values
        시퀀스 모델은 (N, T, F), 트리 모델은 (N, F).
        2D 입력이면 mode와 무관하게 그대로 반환.
    mode
        AGGREGATIONS 중 하나.
    """
    # 트리 출력은 이미 2D → 변환 불필요
    if shap_values.ndim == 2:
        return shap_values

    if shap_values.ndim != 3:
        raise ValueError(
            f"Expected shap_values to be 2D or 3D, got shape {shap_values.shape}"
        )

    if mode == "none":
        return shap_values
    if mode == "abs_mean":
        return np.mean(np.abs(shap_values), axis=1)
    if mode == "mean":
        return np.mean(shap_values, axis=1)
    if mode == "sum":
        return np.sum(shap_values, axis=1)
    if mode == "last":
        return shap_values[:, -1, :]

    raise ValueError(f"Unknown aggregation mode: {mode!r}. Choose from {AGGREGATIONS}.")
