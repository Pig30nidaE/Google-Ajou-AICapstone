"""Subject-level metrics and a deliberately small, fixed blend menu."""

from __future__ import annotations

from collections.abc import Mapping
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

from feature_engineering import CLASS_NAMES
from models import normalize_probabilities


# This small menu is fixed before the new outer-CV results are observed.  It
# replaces the prior 2,376-combination blend/class-scale search that proved
# unstable.  Simpler candidates appear first for the within-0.01 tie rule.
FIXED_BLEND_RECIPES: tuple[tuple[str, dict[str, float]], ...] = (
    ("ydf_multiclass_only", {"ydf_multiclass": 1.0}),
    ("ydf_hierarchical_only", {"ydf_hierarchical": 1.0}),
    ("ydf_random_forest_only", {"ydf_random_forest": 1.0}),
    ("ydf_ovr_only", {"ydf_ovr": 1.0}),
    ("tabnet_only", {"tabnet": 1.0}),
    (
        "direct_hierarchical_equal",
        {"ydf_multiclass": 0.5, "ydf_hierarchical": 0.5},
    ),
    (
        "direct_rf",
        {"ydf_multiclass": 0.7, "ydf_random_forest": 0.3},
    ),
    (
        "direct_hierarchical_rf",
        {
            "ydf_multiclass": 0.4,
            "ydf_hierarchical": 0.4,
            "ydf_random_forest": 0.2,
        },
    ),
    (
        "ydf_four_view",
        {
            "ydf_multiclass": 0.35,
            "ydf_hierarchical": 0.35,
            "ydf_random_forest": 0.15,
            "ydf_ovr": 0.15,
        },
    ),
    (
        "google_all_models",
        {
            "ydf_multiclass": 0.30,
            "ydf_hierarchical": 0.30,
            "ydf_random_forest": 0.15,
            "ydf_ovr": 0.15,
            "tabnet": 0.10,
        },
    ),
)


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
        labels=[0, 1, 2],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    per_class_auc: dict[str, float] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        binary = (y == class_id).astype(np.int64)
        try:
            per_class_auc[class_name] = float(roc_auc_score(binary, p[:, class_id]))
        except ValueError:
            per_class_auc[class_name] = float("nan")
    finite_auc = [value for value in per_class_auc.values() if np.isfinite(value)]
    macro_auc = float(np.mean(finite_auc)) if finite_auc else float("nan")

    non_cn_true = y != 0
    non_cn_predicted = predicted != 0
    non_cn_recall = float(
        np.mean(non_cn_predicted[non_cn_true]) if np.any(non_cn_true) else np.nan
    )
    cn_specificity = non_cn_recall
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": macro_auc,
        "cn_vs_rest_auc": per_class_auc["CN"],
        "per_class_auc": per_class_auc,
        "non_cn_recall": non_cn_recall,
        "cn_specificity": cn_specificity,
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1, 2]).tolist(),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "support": {name: int(np.sum(y == index)) for index, name in enumerate(CLASS_NAMES)},
    }


def selection_score(metrics: Mapping[str, Any]) -> float:
    """Training-only selection score focused on AUC, CN, and subject accuracy."""

    macro_auc = float(metrics["roc_auc_ovr_macro"])
    cn_auc = float(metrics["cn_vs_rest_auc"])
    macro_auc = macro_auc if np.isfinite(macro_auc) else 0.5
    cn_auc = cn_auc if np.isfinite(cn_auc) else 0.5
    score = (
        0.45 * macro_auc
        + 0.30 * cn_auc
        + 0.15 * float(metrics["accuracy"])
        + 0.10 * float(metrics["macro_f1"])
    )
    # Prevent a high-accuracy all-CN solution from winning on this imbalanced cohort.
    recall = float(metrics["non_cn_recall"])
    if np.isfinite(recall) and recall < 0.20:
        score -= 0.50 * (0.20 - recall)
    return float(score)


def apply_recipe(
    model_probabilities: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> np.ndarray:
    missing = sorted(set(weights) - set(model_probabilities))
    if missing:
        raise KeyError(f"Blend recipe requires missing candidate(s): {missing}")
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Blend weights must have a positive sum")
    combined = sum(
        float(weight) * np.asarray(model_probabilities[name], dtype=np.float64)
        for name, weight in weights.items()
    )
    return normalize_probabilities(combined / total)


def select_fixed_blend(
    y_true: np.ndarray,
    model_probabilities: Mapping[str, np.ndarray],
    *,
    simplicity_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Evaluate only the predeclared menu on inner OOF predictions."""

    rows: list[dict[str, Any]] = []
    available = set(model_probabilities)
    for complexity, (name, weights) in enumerate(FIXED_BLEND_RECIPES):
        if not set(weights).issubset(available):
            continue
        probabilities = apply_recipe(model_probabilities, weights)
        metrics = metrics_from_probabilities(y_true, probabilities)
        rows.append(
            {
                "name": name,
                "weights": dict(weights),
                "complexity": complexity,
                "selection_score": selection_score(metrics),
                "metrics": metrics,
            }
        )
    if not rows:
        raise ValueError("No fixed blend recipe is compatible with fitted candidates")
    best_score = max(row["selection_score"] for row in rows)
    eligible = [
        row
        for row in rows
        if row["selection_score"] >= best_score - float(simplicity_tolerance)
    ]
    chosen = min(eligible, key=lambda row: (row["complexity"], -row["selection_score"]))
    return {
        "chosen_name": chosen["name"],
        "weights": chosen["weights"],
        "selection_score": chosen["selection_score"],
        "metrics": chosen["metrics"],
        "simplicity_tolerance": float(simplicity_tolerance),
        "all_fixed_recipes": rows,
        "class_probability_scaling_used": False,
        "adaptive_threshold_used": False,
    }


def class_prior_baseline(y_true: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    prior = np.bincount(y, minlength=3).astype(float)
    prior /= prior.sum()
    probabilities = np.tile(prior, (len(y), 1))
    return metrics_from_probabilities(y, probabilities)


def all_cn_baseline(y_true: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    probabilities = np.tile(np.asarray([1.0 - 2e-6, 1e-6, 1e-6]), (len(y), 1))
    return metrics_from_probabilities(y, probabilities)


def summarize_repeat_metrics(repeat_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "cn_vs_rest_auc",
        "non_cn_recall",
        "log_loss",
    )
    summary: dict[str, Any] = {}
    for key in scalar_keys:
        values = np.asarray([row[key] for row in repeat_metrics], dtype=float)
        summary[key] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
            "values": values.tolist(),
        }
    return summary
