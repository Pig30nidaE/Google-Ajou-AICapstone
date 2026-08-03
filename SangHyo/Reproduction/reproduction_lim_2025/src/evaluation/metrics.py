"""Metrics, subject-level pooling and subject bootstrap CIs.

The primary evaluation unit is the *subject*, so a person with 120 observed days
counts exactly as much as one with 35 (task brief §8-10).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

POOLING_METHODS = ("mean", "median", "majority")


def pool_to_subjects(
    subjects: Sequence[str],
    probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    method: str = "mean",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Collapse per-row predictions to one probability per subject."""
    if method not in POOLING_METHODS:
        raise ValueError(f"method must be one of {POOLING_METHODS}, got {method!r}")
    frame = pd.DataFrame(
        {
            "subject_id": [str(s) for s in subjects],
            "probability": np.asarray(probabilities, dtype=np.float64),
            "label": np.asarray(labels, dtype=np.int64),
        }
    )
    grouped = frame.groupby("subject_id", sort=True)
    labels_per_subject = grouped["label"].nunique()
    if (labels_per_subject > 1).any():
        bad = labels_per_subject[labels_per_subject > 1].index.tolist()
        raise ValueError(f"subjects carry conflicting labels: {bad[:5]}")

    if method == "mean":
        pooled = grouped["probability"].mean()
    elif method == "median":
        pooled = grouped["probability"].median()
    else:
        pooled = grouped["probability"].apply(lambda s: float((s >= threshold).mean()))

    return pd.DataFrame(
        {
            "subject_id": pooled.index,
            "probability": pooled.to_numpy(),
            "label": grouped["label"].first().loc[pooled.index].to_numpy(),
            "n_records": grouped.size().loc[pooled.index].to_numpy(),
        }
    ).reset_index(drop=True)


def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    threshold: float = 0.5,
    predictions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """All primary and secondary metrics for one set of subject-level predictions."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        raise ValueError("no predictions to score")

    y_pred = (
        (y_prob >= threshold).astype(np.int64)
        if predictions is None
        else np.asarray(predictions, dtype=np.int64)
    )
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"predictions shape {y_pred.shape} does not match labels {y_true.shape}"
        )
    both_classes = len(np.unique(y_true)) == 2

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else None
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    precision = float(tp / (tp + fp)) if (tp + fp) else None

    return {
        # primary
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both_classes else None,
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both_classes else None,
        "balanced_accuracy": (
            float(balanced_accuracy_score(y_true, y_pred)) if both_classes else None
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        # secondary
        "accuracy": float((y_pred == y_true).mean()),
        "precision": precision,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "threshold": float(threshold) if predictions is None else None,
        "classification_source": (
            "shared_threshold" if predictions is None else "fold_specific_decisions"
        ),
        "positive_rate_predicted": float(y_pred.mean()),
    }


def bootstrap_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    *,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Percentile bootstrap CI, resampling *subjects* with replacement."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    values: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        sample_true = y_true[idx]
        if len(np.unique(sample_true)) < 2:
            continue
        try:
            values.append(
                float(compute_metrics(sample_true, y_prob[idx], threshold=threshold)[metric])
            )
        except ValueError:
            continue

    if not values:
        return {"metric": metric, "point": None, "lower": None, "upper": None, "n_boot": 0}

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    point = compute_metrics(y_true, y_prob, threshold=threshold)[metric]
    return {
        "metric": metric,
        "point": float(point),
        "lower": float(np.quantile(array, alpha / 2)),
        "upper": float(np.quantile(array, 1 - alpha / 2)),
        "n_boot": int(len(array)),
        "resampling_unit": "subject",
    }


def paired_bootstrap_difference(
    y_true: Sequence[int],
    prob_a: Sequence[float],
    prob_b: Sequence[float],
    *,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """CI for metric(a) - metric(b) on the *same* resampled subjects."""
    y_true = np.asarray(y_true, dtype=np.int64)
    prob_a = np.asarray(prob_a, dtype=np.float64)
    prob_b = np.asarray(prob_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    diffs: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a = compute_metrics(y_true[idx], prob_a[idx])[metric]
        b = compute_metrics(y_true[idx], prob_b[idx])[metric]
        if np.isfinite(a) and np.isfinite(b):
            diffs.append(float(a - b))

    if not diffs:
        return {"metric": metric, "difference": None, "lower": None, "upper": None}

    array = np.asarray(diffs)
    point = (
        compute_metrics(y_true, prob_a)[metric] - compute_metrics(y_true, prob_b)[metric]
    )
    lower = float(np.quantile(array, alpha / 2))
    upper = float(np.quantile(array, 1 - alpha / 2))
    return {
        "metric": metric,
        "difference": float(point),
        "lower": lower,
        "upper": upper,
        "excludes_zero": bool(lower > 0 or upper < 0),
        "n_boot": int(len(array)),
    }


def select_threshold(
    y_true: Sequence[int], y_prob: Sequence[float], *, objective: str = "balanced_accuracy"
) -> float:
    """Pick a threshold from *these* predictions only.

    Callers must pass inner-CV predictions; the nested experiment's outer test
    labels never reach this function (``test_outer_test_isolation.py``).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    candidates = np.unique(np.clip(np.round(y_prob, 4), 0.01, 0.99))
    if candidates.size == 0:
        return 0.5

    best_threshold, best_score = 0.5, -np.inf
    for threshold in candidates:
        score = compute_metrics(y_true, y_prob, threshold=float(threshold))[objective]
        if np.isfinite(score) and score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def calibration_curve_points(
    y_true: Sequence[int], y_prob: Sequence[float], *, n_bins: int = 5
) -> list[dict[str, float]]:
    """Reliability-diagram points, saved rather than plotted."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    points: list[dict[str, float]] = []
    for i in range(int(n_bins)):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if not mask.any():
            continue
        points.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return points
