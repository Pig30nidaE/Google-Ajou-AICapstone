"""Subject-level metrics with explicitly separated repeated-OOF estimands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    objective: str
    objective_value: float
    recall: float


def safe_roc_auc(y: Sequence[int], score: Sequence[float]) -> float:
    from sklearn.metrics import roc_auc_score

    target = np.asarray(y, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError("ROC-AUC requires both CN+MCI and Dem")
    if not np.isfinite(values).all():
        raise ValueError("ROC-AUC score contains non-finite values")
    return float(roc_auc_score(target, values))


def binary_metrics(
    y: Sequence[int],
    score: Sequence[float],
    *,
    threshold: float | None = None,
    prediction: Sequence[int] | None = None,
) -> dict[str, float | int | None | dict[str, int]]:
    """Return the required metrics with Dem fixed as positive class."""

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    target = np.asarray(y, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    if prediction is None:
        resolved_threshold = 0.5 if threshold is None else float(threshold)
        predicted = (values >= resolved_threshold).astype(np.int64)
    else:
        predicted = np.asarray(prediction, dtype=np.int64)
        resolved_threshold = None if threshold is None else float(threshold)
    tn, fp, fn, tp = confusion_matrix(target, predicted, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "roc_auc": safe_roc_auc(target, values),
        "pr_auc": float(average_precision_score(target, values)),
        "dem_recall": float(recall_score(target, predicted, pos_label=1, zero_division=0)),
        "negative_specificity": float(specificity),
        "precision": float(
            precision_score(target, predicted, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(target, predicted, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "mcc": float(matthews_corrcoef(target, predicted)),
        "accuracy": float(accuracy_score(target, predicted)),
        "threshold": resolved_threshold,
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def choose_threshold(
    y: Sequence[int],
    score: Sequence[float],
    *,
    objective: str,
    minimum_recall: float,
) -> ThresholdChoice:
    """Choose an operating point using only cross-fitted training scores."""

    from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

    target = np.asarray(y, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    if objective not in {"mcc", "balanced_accuracy", "f1"}:
        raise ValueError(f"Unknown threshold objective: {objective}")
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.5, 1.0]),
                np.quantile(values, np.linspace(0.01, 0.99, 199)),
            ]
        )
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        prediction = (values >= threshold).astype(np.int64)
        recall = float(np.mean(prediction[target == 1] == 1))
        if recall + 1e-12 < float(minimum_recall):
            continue
        if objective == "mcc":
            value = float(matthews_corrcoef(target, prediction))
        elif objective == "balanced_accuracy":
            value = float(balanced_accuracy_score(target, prediction))
        else:
            value = float(f1_score(target, prediction, zero_division=0))
        # Prefer a higher objective, then higher recall, then a more conservative
        # threshold.  This deterministic tie-break is fixed before evaluation.
        candidate = (value, recall, float(threshold))
        if best is None or candidate > best:
            best = candidate
    if best is None:
        # Extremely defensive fallback: choose the threshold that recalls every
        # training positive. It is still derived exclusively from training OOF.
        threshold = float(np.min(values[target == 1]))
        return ThresholdChoice(threshold, objective, float("nan"), 1.0)
    return ThresholdChoice(
        threshold=best[2],
        objective=objective,
        objective_value=best[0],
        recall=best[1],
    )


def stratified_subject_bootstrap_auc(
    y: Sequence[int],
    score: Sequence[float],
    *,
    iterations: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int | str]:
    """Bootstrap subjects within class for a valid small-positive CI."""

    target = np.asarray(y, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    negative = np.flatnonzero(target == 0)
    positive = np.flatnonzero(target == 1)
    if len(negative) < 2 or len(positive) < 2:
        raise ValueError("Stratified bootstrap requires at least two subjects per class")
    rng = np.random.default_rng(seed)
    samples = np.empty(int(iterations), dtype=np.float64)
    for iteration in range(int(iterations)):
        indices = np.concatenate(
            [
                rng.choice(negative, size=len(negative), replace=True),
                rng.choice(positive, size=len(positive), replace=True),
            ]
        )
        samples[iteration] = safe_roc_auc(target[indices], values[indices])
    alpha = 1.0 - float(confidence)
    return {
        "estimand": "AUC of subject-wise mean repeated cross-fitted score",
        "point": safe_roc_auc(target, values),
        "lower": float(np.quantile(samples, alpha / 2.0)),
        "upper": float(np.quantile(samples, 1.0 - alpha / 2.0)),
        "confidence": float(confidence),
        "iterations": int(iterations),
        "method": "class-stratified subject bootstrap percentile interval",
    }


def summarize_repeat_metrics(
    repeat_metrics: Sequence[Mapping[str, float | int | dict]],
) -> dict[str, dict[str, float] | list[dict]]:
    """Summarize split noise without attaching a different estimator's CI."""

    scalar_keys = (
        "roc_auc",
        "pr_auc",
        "dem_recall",
        "negative_specificity",
        "f1",
        "balanced_accuracy",
        "mcc",
        "accuracy",
    )
    summary: dict[str, dict[str, float] | list[dict]] = {}
    for key in scalar_keys:
        values = np.asarray([float(record[key]) for record in repeat_metrics])
        summary[key] = {
            "mean": float(values.mean()),
            "std_across_repeats": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    summary["per_repeat"] = [dict(record) for record in repeat_metrics]
    return summary


def aggregate_repeated_oof(oof: pd.DataFrame, n_repeats: int) -> pd.DataFrame:
    """Average ranking scores and majority-vote fold-local decisions by subject."""

    required = {
        "subject_hash",
        "y",
        "repeat",
        "score",
        "prediction",
    }
    missing = required - set(oof.columns)
    if missing:
        raise KeyError(f"OOF table lacks columns: {sorted(missing)}")
    counts = oof.groupby("subject_hash").size()
    if not (counts == int(n_repeats)).all():
        raise ValueError(
            "Each subject must have one held-out prediction per repeat; "
            f"observed={sorted(counts.unique().tolist())}"
        )
    label_counts = oof.groupby("subject_hash")["y"].nunique()
    if not (label_counts == 1).all():
        raise ValueError("Subject target changes across OOF repeats")
    aggregated = (
        oof.groupby("subject_hash", sort=False)
        .agg(
            y=("y", "first"),
            score=("score", "mean"),
            prediction_rate=("prediction", "mean"),
            score_std_across_repeats=("score", "std"),
        )
        .reset_index()
    )
    aggregated["prediction"] = (aggregated["prediction_rate"] >= 0.5).astype(int)
    return aggregated


def save_curves(
    y: Sequence[int],
    score: Sequence[float],
    *,
    output_dir: str | Path,
    prefix: str,
) -> dict[str, str]:
    """Save curve coordinates and PNG plots after all model choices are frozen."""

    from sklearn.metrics import precision_recall_curve, roc_curve

    target = np.asarray(y, dtype=np.int64)
    values = np.asarray(score, dtype=np.float64)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    fpr, tpr, roc_threshold = roc_curve(target, values, pos_label=1)
    precision, recall, pr_threshold = precision_recall_curve(
        target, values, pos_label=1
    )
    roc_path = destination / f"{prefix}_roc_curve.csv"
    pr_path = destination / f"{prefix}_pr_curve.csv"
    pd.DataFrame(
        {"fpr": fpr, "tpr": tpr, "threshold": roc_threshold}
    ).to_csv(roc_path, index=False)
    # precision/recall contain one more element than PR thresholds.
    pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.r_[pr_threshold, np.nan],
        }
    ).to_csv(pr_path, index=False)

    paths = {"roc_csv": str(roc_path), "pr_csv": str(pr_path)}
    try:
        import matplotlib.pyplot as plt

        roc_png = destination / f"{prefix}_roc_curve.png"
        pr_png = destination / f"{prefix}_pr_curve.png"
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(fpr, tpr, label=f"AUC={safe_roc_auc(target, values):.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
        axis.set(xlabel="False positive rate", ylabel="True positive rate")
        axis.legend()
        figure.tight_layout()
        figure.savefig(roc_png, dpi=180)
        plt.close(figure)
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(recall, precision)
        axis.axhline(target.mean(), linestyle="--", color="grey")
        axis.set(xlabel="Recall (Dem)", ylabel="Precision (Dem)")
        figure.tight_layout()
        figure.savefig(pr_png, dpi=180)
        plt.close(figure)
        paths.update({"roc_png": str(roc_png), "pr_png": str(pr_png)})
    except ImportError:
        paths["plot_note"] = "matplotlib unavailable; curve coordinates were saved"
    return paths


__all__ = [
    "ThresholdChoice",
    "aggregate_repeated_oof",
    "binary_metrics",
    "choose_threshold",
    "safe_roc_auc",
    "save_curves",
    "stratified_subject_bootstrap_auc",
    "summarize_repeat_metrics",
]
