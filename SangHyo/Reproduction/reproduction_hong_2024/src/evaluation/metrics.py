"""Metrics at two evaluation units, kept deliberately separate.

Sequence level exists to be comparable with Hong et al., who report one row per
window.  Subject level is the primary analysis for the subject-independent
experiments, because a subject with 118 windows would otherwise outvote a subject
with 2.

Nothing here selects a threshold from the data it is scoring.  The threshold is
either the paper's fixed 0.5 or a value chosen on training/inner folds and passed
in.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)

AGGREGATIONS = ("mean", "median", "last", "majority_vote")


def binary_metrics(
    y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> dict[str, Any]:
    """Every headline metric for one (label, score) pair."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return {"n": 0, "note": "no observations"}

    y_pred = (y_score >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else float("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) else float("nan")

    return {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_score)) if both_classes else float("nan"),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if both_classes else float("nan"),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "prevalence": float(y_true.mean()),
    }


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> dict[str, Any]:
    """The paper's precision@K (Equation 11); it reports precision@100 = 0.96."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    requested_k = int(k)
    effective_k = int(min(requested_k, len(y_true)))
    if effective_k == 0:
        return {
            "requested_k": requested_k,
            "k": 0,
            "precision_at_k": float("nan"),
            "max_possible_precision_at_k": float("nan"),
            "n_positive_available": int(y_true.sum()),
            "k_was_truncated": requested_k > 0,
        }
    top = np.argsort(-y_score, kind="stable")[:effective_k]
    return {
        "requested_k": requested_k,
        "k": effective_k,
        "precision_at_k": float(y_true[top].mean()),
        "max_possible_precision_at_k": float(
            min(int(y_true.sum()), effective_k) / effective_k
        ),
        "n_positive_available": int(y_true.sum()),
        "k_was_truncated": bool(effective_k < requested_k),
    }


def aggregate_to_subject(
    subjects: Sequence[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    method: str = "mean",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Collapse sequence scores to one score per subject.

    ``mean`` is the primary; ``median``, ``last`` and ``majority_vote`` are the
    sensitivity analyses the spec asks for.  ``last`` requires the caller to have
    passed sequences in chronological order, which the builder guarantees.
    """
    if method not in AGGREGATIONS:
        raise ValueError(f"method must be one of {AGGREGATIONS}")

    frame = pd.DataFrame(
        {"subject_id": list(subjects), "y_true": np.asarray(y_true).astype(int),
         "y_score": np.asarray(y_score, dtype=float)}
    )
    if frame.empty:
        return pd.DataFrame(columns=["subject_id", "y_true", "y_score", "n_sequences"])

    grouped = frame.groupby("subject_id", sort=True)
    if method == "mean":
        score = grouped["y_score"].mean()
    elif method == "median":
        score = grouped["y_score"].median()
    elif method == "last":
        score = grouped["y_score"].last()
    else:
        score = grouped["y_score"].apply(lambda s: float((s >= threshold).mean()))

    label = grouped["y_true"].first()
    n = grouped.size().rename("n_sequences")
    out = pd.concat([label.rename("y_true"), score.rename("y_score"), n], axis=1)
    return out.reset_index()


def subject_level_metrics(
    subjects: Sequence[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    method: str = "mean",
    threshold: float = 0.5,
) -> dict[str, Any]:
    frame = aggregate_to_subject(
        subjects, y_true, y_score, method=method, threshold=threshold
    )
    metrics = binary_metrics(
        frame["y_true"].to_numpy(), frame["y_score"].to_numpy(), threshold=threshold
    )
    metrics["aggregation"] = method
    metrics["n_subjects"] = int(len(frame))
    return metrics


def subject_precision_at_k(
    subjects: Sequence[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    k: int = 100,
    method: str = "mean",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Precision@K after collapsing every subject to exactly one score."""
    frame = aggregate_to_subject(
        subjects, y_true, y_score, method=method, threshold=threshold
    )
    report = precision_at_k(
        frame["y_true"].to_numpy(), frame["y_score"].to_numpy(), k=k
    )
    report.update(
        {
            "unit": "subject",
            "aggregation": method,
            "n_subjects": int(len(frame)),
        }
    )
    return report


def all_aggregation_metrics(
    subjects: Sequence[str], y_true: np.ndarray, y_score: np.ndarray, *, threshold: float = 0.5
) -> dict[str, Any]:
    return {
        method: subject_level_metrics(
            subjects, y_true, y_score, method=method, threshold=threshold
        )
        for method in AGGREGATIONS
    }


def bootstrap_ci(
    subjects: Sequence[str],
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    metric: str = "roc_auc",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    aggregation: str = "mean",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Subject-level bootstrap CI: subjects are resampled, not sequences.

    Resampling sequences would treat 118 windows from one person as 118
    independent observations and produce an interval far too narrow.
    """
    frame = aggregate_to_subject(
        subjects, y_true, y_score, method=aggregation, threshold=threshold
    )
    if len(frame) < 3 or frame["y_true"].nunique() < 2:
        return {"metric": metric, "point": float("nan"), "note": "too few subjects or one class"}

    labels = frame["y_true"].to_numpy()
    scores = frame["y_score"].to_numpy()
    rng = np.random.default_rng(seed)
    point = binary_metrics(labels, scores, threshold=threshold).get(metric, float("nan"))

    draws = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[idx])) < 2:
            continue
        value = binary_metrics(labels[idx], scores[idx], threshold=threshold).get(metric)
        if value is not None and np.isfinite(value):
            draws.append(value)

    if not draws:
        return {"metric": metric, "point": float(point), "note": "no valid bootstrap draws"}
    lower, upper = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "metric": metric,
        "aggregation": aggregation,
        "point": float(point),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_boot_effective": len(draws),
        "unit": "subject",
    }


def choose_threshold_with_report(
    y_true: np.ndarray, y_score: np.ndarray, *, policy: str = "fixed", fixed: float = 0.5
) -> tuple[float, dict[str, Any]]:
    """Pick an operating point from *training or inner-fold* scores only.

    The paper uses 0.5 (§4.2, Table 6), which is ``policy='fixed'``.  The other
    policies exist for experiment C, where the threshold is one more thing the
    inner CV is allowed to choose.

    **Degenerate operating points are rejected.**  A threshold below every score
    predicts the positive class for everything: sensitivity 1, specificity 0, and
    a Youden index of exactly 0.  On a chance-level model every real candidate
    scores below 0, so that degenerate point would win by default -- which is how
    the 2026-08-02 nested run ended up with 6 of 15 folds reporting
    "sensitivity 1.00 / specificity 0.00".  Candidates that put every subject in
    one class are therefore skipped, and if nothing else qualifies the fixed
    threshold is used and the fallback is recorded.
    """
    info: dict[str, Any] = {"policy": policy, "fallback_to_fixed": False,
                            "n_candidates": 0, "n_degenerate_skipped": 0}
    if policy == "fixed":
        info["threshold"] = float(fixed)
        return float(fixed), info

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        info.update(fallback_to_fixed=True, reason="single class or empty",
                    threshold=float(fixed))
        return float(fixed), info

    candidates = np.unique(np.round(y_score, 4))
    if len(candidates) > 512:
        candidates = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 512)))
    info["n_candidates"] = int(len(candidates))

    if policy == "youden":
        score_of = lambda m: m["sensitivity"] + m["specificity"] - 1  # noqa: E731
    elif policy == "balanced_accuracy":
        score_of = lambda m: m["balanced_accuracy"]  # noqa: E731
    else:
        raise ValueError("policy must be 'fixed', 'youden' or 'balanced_accuracy'")

    best, best_value, degenerate = None, -np.inf, 0
    for t in candidates:
        m = binary_metrics(y_true, y_score, threshold=float(t))
        matrix = m["confusion_matrix"]
        predicts_one_class = (matrix["tp"] + matrix["fp"] == 0) or (
            matrix["tn"] + matrix["fn"] == 0
        )
        if predicts_one_class:
            degenerate += 1
            continue
        value = score_of(m)
        if np.isfinite(value) and value > best_value:
            best, best_value = float(t), value

    info["n_degenerate_skipped"] = degenerate
    if best is None:
        info.update(fallback_to_fixed=True,
                    reason="every candidate collapses to a single predicted class",
                    threshold=float(fixed))
        return float(fixed), info

    info.update(threshold=best, selected_value=float(best_value))
    return best, info


def choose_threshold(
    y_true: np.ndarray, y_score: np.ndarray, *, policy: str = "fixed", fixed: float = 0.5
) -> float:
    """Threshold only; see :func:`choose_threshold_with_report` for the diagnostics."""
    threshold, _ = choose_threshold_with_report(
        y_true, y_score, policy=policy, fixed=fixed
    )
    return threshold


def calibration_curve_points(
    y_true: np.ndarray, y_score: np.ndarray, *, n_bins: int = 10
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return {"bins": [], "n_bins": 0}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(y_score, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = index == b
        if not mask.any():
            continue
        bins.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_predicted": float(y_score[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
            }
        )
    return {"bins": bins, "n_bins": n_bins}


def fold_variability(per_fold: list[dict[str, Any]], metric: str = "roc_auc") -> dict[str, Any]:
    values = [f[metric] for f in per_fold if np.isfinite(f.get(metric, np.nan))]
    if not values:
        return {"metric": metric, "n_folds": 0}
    return {
        "metric": metric,
        "n_folds": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
