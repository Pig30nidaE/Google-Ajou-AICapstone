"""Evaluation metrics.

CN is the majority class (60.3% of Training, 78.8% of Validation), so Accuracy
alone is uninformative -- predicting all-CN scores 0.788 on Validation.  Every
report produced by this experiment therefore carries Balanced Accuracy,
MCI+DEM Recall, CN Specificity, ROC-AUC, the confusion matrix and a bootstrap
interval, always beside the all-CN baseline.
"""

from __future__ import annotations

import numpy as np


def roc_auc(y_true, y_score) -> float:
    """Rank-based ROC-AUC, tie-safe; returns 0.5 for a degenerate label set."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.count_nonzero(y_true == 1))
    n_neg = int(np.count_nonzero(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Average ranks within tied score groups.
    sorted_scores = y_score[order]
    start = 0
    for index in range(1, len(sorted_scores) + 1):
        if index == len(sorted_scores) or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index

    rank_sum = ranks[y_true == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def confusion(y_true, y_pred) -> dict[str, int]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return {
        "tn": int(np.count_nonzero((y_true == 0) & (y_pred == 0))),
        "fp": int(np.count_nonzero((y_true == 0) & (y_pred == 1))),
        "fn": int(np.count_nonzero((y_true == 1) & (y_pred == 0))),
        "tp": int(np.count_nonzero((y_true == 1) & (y_pred == 1))),
    }


def classification_metrics(y_true, y_score, threshold: float) -> dict:
    """Full metric bundle at a given decision threshold."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    y_pred = (y_score >= threshold).astype(np.int64)
    matrix = confusion(y_true, y_pred)
    tn, fp, fn, tp = matrix["tn"], matrix["fp"], matrix["fn"], matrix["tp"]

    recall = tp / (tp + fn) if (tp + fn) else float("nan")          # MCI+DEM sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")     # CN specificity
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and np.isfinite(precision) and np.isfinite(recall)
          and (precision + recall) > 0 else float("nan"))
    balanced = float(np.nanmean([recall, specificity]))

    n_positive = int(np.count_nonzero(y_true == 1))
    majority_accuracy = 1.0 - n_positive / len(y_true) if len(y_true) else float("nan")

    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)) if len(y_true) else float("nan"),
        "balanced_accuracy": balanced,
        "recall_mci_dem": float(recall),
        "specificity_cn": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "roc_auc": roc_auc(y_true, y_score),
        "confusion": matrix,
        "n": int(len(y_true)),
        "n_mci_dem": n_positive,
        "all_cn_baseline_accuracy": float(majority_accuracy),
    }


def youden_threshold(y_true, y_score) -> float:
    """Threshold maximising Youden's J (sensitivity + specificity - 1).

    The report computes this on the very predictions it then scores.  Callers
    here must pass **inner** out-of-fold predictions from the training fold, so
    that the threshold never sees the data it will be applied to.
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return 0.5

    candidates = np.unique(y_score)
    if candidates.size > 1:
        midpoints = (candidates[:-1] + candidates[1:]) / 2.0
        candidates = np.concatenate([[candidates[0] - 1e-6], midpoints,
                                     [candidates[-1] + 1e-6]])

    best_threshold, best_j = 0.5, -np.inf
    for threshold in candidates:
        predicted = (y_score >= threshold).astype(np.int64)
        matrix = confusion(y_true, predicted)
        tp, fn, tn, fp = matrix["tp"], matrix["fn"], matrix["tn"], matrix["fp"]
        sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        j = sensitivity + specificity - 1.0
        if j > best_j:
            best_threshold, best_j = float(threshold), j
    return best_threshold


def bootstrap_interval(
    y_true, y_score, threshold: float, *, n_resamples: int = 2000, seed: int = 0
) -> dict:
    """Percentile bootstrap CI for ROC-AUC and Balanced Accuracy."""

    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    rng = np.random.default_rng(seed)

    aucs, balanced = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc(y_true[idx], y_score[idx]))
        balanced.append(classification_metrics(y_true[idx], y_score[idx], threshold)
                        ["balanced_accuracy"])

    def interval(values):
        if not values:
            return {"lo": float("nan"), "hi": float("nan")}
        return {"lo": float(np.percentile(values, 2.5)),
                "hi": float(np.percentile(values, 97.5))}

    return {"roc_auc_95ci": interval(aucs),
            "balanced_accuracy_95ci": interval(balanced),
            "n_resamples_used": len(aucs)}


def jaccard_stability(selections: list[list[int]]) -> dict:
    """Mean pairwise Jaccard overlap of per-fold feature selections.

    Earlier experiments in this repo found near-random selection overlap
    (~0.22), which is the signature of a feature space too wide for 141 people.
    Reporting it keeps that failure mode visible.
    """

    sets = [set(s) for s in selections if s]
    if len(sets) < 2:
        return {"mean_jaccard": float("nan"), "n_selections": len(sets),
                "always_selected": []}

    scores = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            scores.append(len(sets[i] & sets[j]) / len(union) if union else 0.0)

    common = set.intersection(*sets) if sets else set()
    return {
        "mean_jaccard": float(np.mean(scores)),
        "n_selections": len(sets),
        "always_selected": sorted(int(c) for c in common),
    }
