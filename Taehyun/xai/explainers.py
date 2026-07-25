"""SHAP 백엔드 어댑터

각 어댑터:
- 모델과 (필요 시) background 데이터를 받음
- shap_values(X) 호출 시 정규화된 ndarray 반환
- 다중 클래스 출력의 축 정렬은 analyzer 측에서 처리

shap 라이브러리는 모듈 로드 시점이 아니라 어댑터 초기화 시점에 import (지연 로드).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _import_shap():
    """shap 라이브러리 지연 import. 미설치 시 명확한 메시지로 변환."""
    try:
        import shap as shap_lib
    except ImportError as exc:
        raise ImportError(
            "The 'shap' library is required. Install with: pip install shap"
        ) from exc
    return shap_lib


class TreeAdapter:
    """sklearn 호환 트리 분류기용 shap.TreeExplainer 래퍼

    - background 데이터 불필요 (트리 구조에서 직접 계산)
    - 입력은 pandas DataFrame 권장 (feature 이름이 SHAP plot에 자동 표시됨)
    """

    def __init__(self, model: Any):
        shap_lib = _import_shap()
        self.model = model
        self.explainer = shap_lib.TreeExplainer(model)

    def shap_values(self, X) -> np.ndarray:
        raw = self.explainer.shap_values(X)
        return _to_ndarray(raw)

    @property
    def expected_value(self):
        return self.explainer.expected_value


class KerasSequenceAdapter:
    """tf.keras 시퀀스 모델용 shap.GradientExplainer 래퍼

    - GradientExplainer 선택 이유: DeepExplainer는 TF2 버전 호환성이
      취약함. GradientExplainer는 미분 가능한 임의 Keras 모델에서 동작.
    - background는 필수 (analyzer에서 None이면 자동 샘플링 후 전달)
    """

    def __init__(self, model: Any, background: np.ndarray):
        shap_lib = _import_shap()
        if background is None:
            raise ValueError("background must be provided for Keras sequence models.")
        self.model = model
        # GradientExplainer는 float32 텐서 기대 → 명시 캐스팅
        self.background = np.asarray(background, dtype=np.float32)
        self.explainer = shap_lib.GradientExplainer(model, self.background)

    def shap_values(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        raw = self.explainer.shap_values(X)
        return _to_ndarray(raw)


def _to_ndarray(raw) -> np.ndarray:
    """shap 버전/백엔드별 출력 형식 차이를 단일 ndarray로 정규화

    들어올 수 있는 형태:
    - list[ndarray] 길이 C : 멀티클래스 (구버전 shap)  → stack → (C, ...)
    - ndarray (..., C)     : 멀티클래스 (shap >= 0.45) → 그대로 (analyzer가 처리)
    - ndarray              : 단일 출력 (이진/회귀)     → 그대로
    """
    if isinstance(raw, list):
        return np.stack([np.asarray(r) for r in raw], axis=0)

    return np.asarray(raw)
