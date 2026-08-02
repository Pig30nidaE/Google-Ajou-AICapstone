"""Leakage-safe repeated nested cross-validation and metrics.

Design choices follow the honest lessons from the prior experiments and the
team report:

* Splits are subject-level (one row per subject), so a subject never appears in
  both training and evaluation.
* Ensemble weights and the decision threshold are chosen with an inner
  cross-validation on the outer-training subjects only, then applied to the
  untouched outer-test subjects.
* Folds are combined on the ``prob - threshold`` margin rather than raw
  probabilities, so folds with different score scales stay comparable.
* Balanced accuracy is the primary selection metric because CN is the majority
  class (all-CN already scores 0.60 on train, 0.79 on the historical validation).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

LearnerFactory = Callable[[], object]


def binary_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray | None = None) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    recall = tp / (tp + fn) if (tp + fn) else 0.0            # impaired sensitivity
    specificity = tn / (tn + fp) if (tn + fp) else 0.0       # CN specificity
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics = {
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "impaired_recall": recall,
        "cn_specificity": specificity,
        "precision": precision,
        "f1": f1,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if prob is not None and len(np.unique(y)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, prob))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def select_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """Threshold that maximizes balanced accuracy (ties: higher accuracy)."""

    grid = np.unique(np.clip(np.round(prob, 3), 0.02, 0.98))
    if len(grid) == 0:
        return 0.5
    candidates = np.concatenate([[0.5], grid])
    best_threshold, best_key = 0.5, (-1.0, -1.0)
    for threshold in candidates:
        pred = (prob >= threshold).astype(int)
        m = binary_metrics(y, pred)
        key = (m["balanced_accuracy"], m["accuracy"])
        if key > best_key:
            best_key, best_threshold = key, float(threshold)
    return best_threshold


def _weighted_average(probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    stack = np.zeros(len(next(iter(probs.values()))))
    if total <= 0:  # every model looked useless on inner OOF: unweighted mean
        return np.mean(np.vstack(list(probs.values())), axis=0)
    for name, prob in probs.items():
        stack += weights[name] * prob
    return stack / total


def _model_weight(y: np.ndarray, prob: np.ndarray) -> float:
    pred = (prob >= 0.5).astype(int)
    balanced = binary_metrics(y, pred)["balanced_accuracy"]
    return max(0.0, 2.0 * balanced - 1.0)   # 0 at chance, 1 at perfect


def _inner_oof(
    data,
    factories: Sequence[tuple[str, LearnerFactory]],
    train_idx: np.ndarray,
    inner_k: int,
    seed: int,
) -> dict[str, np.ndarray]:
    y = data.y[train_idx]
    oof = {name: np.full(len(train_idx), np.nan) for name, _ in factories}
    inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)
    for itr, ite in inner.split(train_idx, y):
        global_tr = train_idx[itr]
        global_te = train_idx[ite]
        for name, factory in factories:
            learner = factory()
            learner.fit(global_tr)
            oof[name][ite] = learner.predict_proba(global_te)
    return oof


def nested_cv(
    data,
    factories: Sequence[tuple[str, LearnerFactory]],
    *,
    repeats: int,
    outer_k: int,
    inner_k: int,
    seed: int = 20260724,
) -> dict:
    """Run repeated subject-level nested CV and aggregate per-subject scores."""

    n = data.n_subjects
    y = data.y
    prob_sum = np.zeros(n)
    margin_sum = np.zeros(n)
    seen = np.zeros(n)
    fold_metrics: list[dict] = []
    model_inner_balacc: dict[str, list[float]] = {name: [] for name, _ in factories}
    model_weight_log: dict[str, list[float]] = {name: [] for name, _ in factories}

    for repeat in range(repeats):
        outer = StratifiedKFold(
            n_splits=outer_k, shuffle=True, random_state=seed + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(outer.split(np.arange(n), y)):
            inner_seed = seed + 100 * repeat + fold
            oof = _inner_oof(data, factories, train_idx, inner_k, inner_seed)
            weights = {}
            for name, _ in factories:
                balanced = binary_metrics(
                    y[train_idx], (oof[name] >= 0.5).astype(int)
                )["balanced_accuracy"]
                model_inner_balacc[name].append(balanced)
                weights[name] = _model_weight(y[train_idx], oof[name])
                model_weight_log[name].append(weights[name])
            combined_inner = _weighted_average(oof, weights)
            threshold = select_threshold(y[train_idx], combined_inner)

            test_probs = {}
            for name, factory in factories:
                learner = factory()
                learner.fit(train_idx)
                test_probs[name] = learner.predict_proba(test_idx)
            combined_test = _weighted_average(test_probs, weights)

            prob_sum[test_idx] += combined_test
            margin_sum[test_idx] += combined_test - threshold
            seen[test_idx] += 1
            fold_pred = (combined_test >= threshold).astype(int)
            fold_metrics.append(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "threshold": threshold,
                    **binary_metrics(y[test_idx], fold_pred, combined_test),
                }
            )

    if np.any(seen == 0):
        raise AssertionError("Every subject must be evaluated at least once")
    prob = prob_sum / seen
    margin = margin_sum / seen
    pred = (margin > 0).astype(int)
    oof_metrics = binary_metrics(y, pred, prob)
    oof_metrics_half = binary_metrics(y, (prob >= 0.5).astype(int), prob)

    return {
        "oof_prob": prob,
        "oof_margin": margin,
        "oof_pred": pred,
        "oof_metrics_selected_threshold": oof_metrics,
        "oof_metrics_threshold_0.5": oof_metrics_half,
        "fold_metrics": fold_metrics,
        "model_inner_balanced_accuracy": {
            name: float(np.mean(values)) for name, values in model_inner_balacc.items()
        },
        "model_mean_weight": {
            name: float(np.mean(values)) for name, values in model_weight_log.items()
        },
    }


def bootstrap_ci(
    y: np.ndarray,
    prob: np.ndarray,
    margin: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    acc, bal, auc = [], [], []
    for _ in range(n_boot):
        sample = rng.integers(0, n, n)
        ys = y[sample]
        if len(np.unique(ys)) < 2:
            continue
        pred = (margin[sample] > 0).astype(int)
        m = binary_metrics(ys, pred, prob[sample])
        acc.append(m["accuracy"])
        bal.append(m["balanced_accuracy"])
        auc.append(m["roc_auc"])

    def ci(values: list[float]) -> list[float]:
        if not values:
            return [float("nan"), float("nan")]
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {"accuracy": ci(acc), "balanced_accuracy": ci(bal), "roc_auc": ci(auc)}


__all__ = ["binary_metrics", "bootstrap_ci", "nested_cv", "select_threshold"]
