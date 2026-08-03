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
    validation: LifelogData | None = None,
    seed: int = 42,
) -> BaseClassifier:
    """train fold에서 분류기를 학습한다.

    실험 A가 명시적 valid split을 넘기면 그것을 early stopping에 사용한다. 그 밖의
    실험에서는 train **안에서** 피험자 단위 validation을 떼어낸다. early stopping을
    사용하지 않는 모델은 어떤 학습행도 validation 명목으로 버리지 않는다.
    """
    params = dict((cfg.get("models") or {}).get(model_name) or {})
    if class_weight:
        params["class_weight"] = class_weight

    model = make_model(model_name, params, seed=seed)
    train_idx = np.arange(train.n)
    eval_set = None
    n_val = 0

    # 논문 실험 A는 명시적인 8:1:1 valid split을 갖는다. 그 split을 무시하고
    # train에서 새 validation을 떼면 표 5의 학습행 수와 검증 절차를 재현하지 못한다.
    # B/C는 외부 validation을 넘기지 않으므로 기존처럼 train 내부에서만 분리한다.
    if model.uses_early_stopping and validation is not None:
        if validation.is_synthetic.any():
            raise ValueError("early-stopping validation에는 합성행을 넣을 수 없다")
        auditor.record_early_stopping(
            fold_id, subjects=validation.subject, row_ids=validation.row_id
        )
        eval_set = (validation.X.to_numpy(), validation.y)
        n_val = validation.n
    elif model.uses_early_stopping:
        iv_cfg = (cfg.get("internal_validation") or {})
        iv = make_internal_validation(
            train.y,
            train.subject,
            train.is_synthetic,
            fraction=float(iv_cfg.get("fraction", 0.2)),
            split_by=iv_cfg.get("split_by", "subject"),
            seed=seed,
        )
        train_idx = iv.train_idx
        if len(iv.val_idx):
            auditor.record_early_stopping(
                fold_id,
                subjects=train.subject[iv.val_idx],
                row_ids=train.row_id[iv.val_idx],
            )
            eval_set = (
                train.X.to_numpy()[iv.val_idx],
                train.y[iv.val_idx],
            )
            n_val = len(iv.val_idx)

    Xtr = train.X.to_numpy()[train_idx]
    ytr = train.y[train_idx]
    log.info(
        "[%s] %s 학습: %d행 (합성 %d행 포함), early-stop val %d행",
        fold_id, model.name, len(ytr),
        int(train.is_synthetic[train_idx].sum()), n_val,
    )
    model.fit(Xtr, ytr, eval_set=eval_set)
    model.fit_log = {
        **model.fit_log,
        "n_train_rows_used": int(len(ytr)),
        "n_early_stopping_validation_rows": int(n_val),
        "validation_source": (
            "explicit_valid_split"
            if model.uses_early_stopping and validation is not None
            else "internal_train_split"
            if model.uses_early_stopping and n_val
            else "none"
        ),
        "class_weight_requested": bool(class_weight),
    }
    return model
