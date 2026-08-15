"""Metrics, threshold selection, and subject-level bootstrap uncertainty."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score


def roc_auc_safe(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(scores, dtype=np.float64)))


def pick_threshold_balanced(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximizing balanced accuracy on the given (training-side) scores."""

    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores)
    unique = np.unique(scores[order])
    if unique.size == 1:
        return float(unique[0])
    cuts = (unique[:-1] + unique[1:]) / 2.0
    best_threshold, best_value = float(cuts[0]), -1.0
    positives = max(1, int((y == 1).sum()))
    negatives = max(1, int((y == 0).sum()))
    for cut in cuts:
        predicted = scores >= cut
        sensitivity = float(((y == 1) & predicted).sum()) / positives
        specificity = float(((y == 0) & ~predicted).sum()) / negatives
        value = 0.5 * (sensitivity + specificity)
        if value > best_value + 1e-12:
            best_value, best_threshold = value, float(cut)
    return best_threshold


def thresholded_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    predicted = (np.asarray(scores, dtype=np.float64) >= float(threshold)).astype(int)
    tp = int(((y == 1) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "threshold": float(threshold),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / max(1, len(y)),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "precision": tp / max(1, tp + fp),
        "mcc": float(matthews_corrcoef(y, predicted)) if len(np.unique(y)) > 1 else float("nan"),
    }


def score_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    out = {"roc_auc": roc_auc_safe(y, scores)}
    if len(np.unique(y)) > 1:
        out["pr_auc"] = float(average_precision_score(y, scores))
        out["pr_auc_prevalence_baseline"] = float((y == 1).mean())
    else:  # pragma: no cover - degenerate fold
        out["pr_auc"] = float("nan")
        out["pr_auc_prevalence_baseline"] = float("nan")
    return out


def cn_vs_mci_auc(diag: np.ndarray, scores: np.ndarray) -> float:
    """AUC restricted to CN vs MCI subjects (Dem excluded).

    The 9 Dem subjects separate easily and inflate the headline AUC; this
    secondary number shows the actual early-screening difficulty.
    """

    diag = np.asarray(diag, dtype=object)
    mask = np.isin(diag, ("CN", "MCI"))
    y = (diag[mask] == "MCI").astype(int)
    return roc_auc_safe(y, np.asarray(scores, dtype=np.float64)[mask])


def subject_bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, *, n_boot: int,
                             seed: int) -> dict:
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(int(n_boot)):
        index = rng.integers(0, n, size=n)
        if len(np.unique(y[index])) < 2:
            continue
        draws.append(roc_auc_safe(y[index], scores[index]))
    draws = np.asarray(draws, dtype=np.float64)
    if draws.size == 0:  # pragma: no cover - degenerate cohort
        return {"n_effective_draws": 0, "ci95_low": float("nan"), "ci95_high": float("nan")}
    return {
        "n_effective_draws": int(draws.size),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
    }


def paired_bootstrap_auc_diff(y: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
                              *, n_boot: int, seed: int) -> dict:
    """Subject bootstrap CI of AUC(a) - AUC(b) on the SAME subjects."""

    y = np.asarray(y, dtype=int)
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(int(n_boot)):
        index = rng.integers(0, n, size=n)
        if len(np.unique(y[index])) < 2:
            continue
        draws.append(roc_auc_safe(y[index], a[index]) - roc_auc_safe(y[index], b[index]))
    draws = np.asarray(draws, dtype=np.float64)
    observed = roc_auc_safe(y, a) - roc_auc_safe(y, b)
    if draws.size == 0:  # pragma: no cover
        return {"observed_diff": float(observed), "ci95_low": float("nan"),
                "ci95_high": float("nan"), "n_effective_draws": 0}
    return {
        "observed_diff": float(observed),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "n_effective_draws": int(draws.size),
        "interpretation": "CI containing 0 means no demonstrated improvement",
    }


__all__ = [
    "cn_vs_mci_auc", "paired_bootstrap_auc_diff", "pick_threshold_balanced",
    "roc_auc_safe", "score_metrics", "subject_bootstrap_auc_ci",
    "thresholded_metrics",
]
