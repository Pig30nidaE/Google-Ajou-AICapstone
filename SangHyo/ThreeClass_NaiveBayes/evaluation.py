"""Metrics and model-selection score for the three-class baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

if __package__:
    from .data import CLASS_NAMES
else:
    from data import CLASS_NAMES


CLASS_IDS = np.arange(len(CLASS_NAMES), dtype=np.int64)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"Expected an (n, {len(CLASS_NAMES)}) probability matrix; "
            f"found {values.shape}"
        )
    if not np.isfinite(values).all():
        raise FloatingPointError("Predicted probabilities contain NaN or infinity")
    if np.any(values < 0):
        raise ValueError("Predicted probabilities cannot be negative")
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise FloatingPointError("At least one prediction has no probability mass")
    return values / row_sums


def align_probabilities(
    probabilities: np.ndarray,
    fitted_classes: np.ndarray,
) -> np.ndarray:
    """Place estimator probabilities in the fixed CN/MCI/DEM column order."""

    raw = np.asarray(probabilities, dtype=np.float64)
    classes = np.asarray(fitted_classes, dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != len(classes):
        raise ValueError("Probability columns do not match fitted classes")
    aligned = np.zeros((len(raw), len(CLASS_NAMES)), dtype=np.float64)
    for source_column, class_id in enumerate(classes):
        if class_id not in CLASS_IDS:
            raise ValueError(f"Unexpected fitted class id: {class_id}")
        aligned[:, int(class_id)] = raw[:, source_column]
    return normalize_probabilities(aligned)


def metrics_from_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    p = normalize_probabilities(probabilities)
    predicted = p.argmax(axis=1)
    report = classification_report(
        y,
        predicted,
        labels=CLASS_IDS,
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    per_class_auc: dict[str, float] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        binary = (y == class_id).astype(np.int64)
        try:
            per_class_auc[class_name] = float(
                roc_auc_score(binary, p[:, class_id])
            )
        except ValueError:
            per_class_auc[class_name] = float("nan")
    finite_auc = [value for value in per_class_auc.values() if np.isfinite(value)]
    macro_auc = float(np.mean(finite_auc)) if finite_auc else float("nan")
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y, predicted, average="weighted", zero_division=0)
        ),
        "roc_auc_ovr_macro": macro_auc,
        "cn_vs_rest_auc": per_class_auc["CN"],
        "log_loss": float(log_loss(y, p, labels=CLASS_IDS)),
        "confusion_matrix": confusion_matrix(
            y,
            predicted,
            labels=CLASS_IDS,
        ).tolist(),
        "per_class_auc": per_class_auc,
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "support": {
            name: int(np.sum(y == class_id))
            for class_id, name in enumerate(CLASS_NAMES)
        },
    }


def training_selection_score(estimator, X, y_true: np.ndarray) -> float:
    """Inner-CV score emphasizing class balance and ranking quality."""

    probabilities = align_probabilities(
        estimator.predict_proba(X),
        estimator.classes_,
    )
    metrics = metrics_from_probabilities(np.asarray(y_true), probabilities)
    macro_auc = float(metrics["roc_auc_ovr_macro"])
    cn_auc = float(metrics["cn_vs_rest_auc"])
    macro_auc = macro_auc if np.isfinite(macro_auc) else 0.5
    cn_auc = cn_auc if np.isfinite(cn_auc) else 0.5
    return float(
        0.45 * macro_auc
        + 0.25 * cn_auc
        + 0.20 * float(metrics["balanced_accuracy"])
        + 0.10 * float(metrics["macro_f1"])
    )


def all_cn_baseline(y_true: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    probabilities = np.tile(
        np.asarray([1.0 - 2e-6, 1e-6, 1e-6], dtype=np.float64),
        (len(y), 1),
    )
    return metrics_from_probabilities(y, probabilities)


def class_prior_probabilities(
    y_fit: np.ndarray,
    prediction_count: int,
) -> np.ndarray:
    """Return a prior learned only from the supplied fit-fold labels."""

    labels = np.asarray(y_fit, dtype=np.int64)
    prior = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    if np.count_nonzero(prior) != len(CLASS_NAMES):
        raise ValueError("Class-prior baseline fit fold must contain every class")
    prior /= prior.sum()
    return np.tile(prior, (int(prediction_count), 1))


def summarize_repeat_metrics(repeat_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "cn_vs_rest_auc",
        "log_loss",
    )
    summary: dict[str, Any] = {}
    for key in scalar_keys:
        values = np.asarray([row[key] for row in repeat_metrics], dtype=np.float64)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "values": values.tolist(),
        }
    return summary
