"""분류기 레지스트리와 학습 진입점."""

from __future__ import annotations

import logging

import numpy as np

from ..audit.leakage import LeakageAuditor
from ..data.loader import LifelogData
from .base import BaseClassifier, make_internal_validation
from .classifiers import (
    DNNClassifier,
    TabNetClassifier,
    WideDeepClassifier,
    XGBoostClassifier,
)

log = logging.getLogger(__name__)

__all__ = ["MODEL_REGISTRY", "make_model", "fit_classifier", "available_models"]

MODEL_REGISTRY: dict[str, type[BaseClassifier]] = {
    "xgboost": XGBoostClassifier,
    "dnn": DNNClassifier,
    "tabnet": TabNetClassifier,
    "wide_deep": WideDeepClassifier,
}

#: 논문 §4.2의 비교 모델 순서.
PAPER_MODEL_ORDER: tuple[str, ...] = ("xgboost", "dnn", "tabnet", "wide_deep")


def available_models() -> list[str]:
    return list(MODEL_REGISTRY)


def make_model(name: str, params: dict | None = None, *, seed: int = 42) -> BaseClassifier:
    key = name.lower().replace("&", "").replace(" ", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r} (가능: {sorted(MODEL_REGISTRY)})")
    return MODEL_REGISTRY[key](params, seed=seed)


def fit_classifier(
    model_name: str,
    train: LifelogData,
    cfg: dict,
    *,
    auditor: LeakageAuditor,
    fold_id: str,
    class_weight: dict[int, float] | None = None,
    seed: int = 42,
) -> BaseClassifier:
    """train fold에서 분류기를 학습한다.

    early stopping용 validation은 train **안에서** 피험자 단위로 떼어내며,
    감사기에 그 범위를 신고한다 (사용자 지시 8절 — outer test로 early stopping 금지).
    """
    params = dict((cfg.get("models") or {}).get(model_name) or {})
    if class_weight:
        params["class_weight"] = class_weight

    iv_cfg = (cfg.get("internal_validation") or {})
    iv = make_internal_validation(
        train.y,
        train.subject,
        train.is_synthetic,
        fraction=float(iv_cfg.get("fraction", 0.2)),
        split_by=iv_cfg.get("split_by", "subject"),
        seed=seed,
    )
    eval_set = None
    if len(iv.val_idx):
        auditor.record_early_stopping(fold_id, subjects=train.subject[iv.val_idx])
        eval_set = (
            train.X.to_numpy()[iv.val_idx],
            train.y[iv.val_idx],
        )

    model = make_model(model_name, params, seed=seed)
    Xtr = train.X.to_numpy()[iv.train_idx]
    ytr = train.y[iv.train_idx]
    log.info(
        "[%s] %s 학습: %d행 (합성 %d행 포함), 내부 val %d행",
        fold_id, model.name, len(ytr),
        int(train.is_synthetic[iv.train_idx].sum()), len(iv.val_idx),
    )
    model.fit(Xtr, ytr, eval_set=eval_set)
    return model
