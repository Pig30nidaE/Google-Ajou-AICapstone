"""Leakage-safe repeated nested CV with a quality-gated ensemble.

Improvements over the earlier experiments, from analysing Model A/B:

* **Quality gate** — a model contributes to the blend only if its aggregate
  inner-OOF balanced accuracy clears ``weight_gate`` (default 0.55).  In Model A
  every wearable model was ~chance, yet probability normalisation still handed
  the noisiest one ~48% weight and dragged validation down.  The gate zeroes
  chance-level models; if none qualify in a fold, the single best model is used
  instead of blending noise.
* Subject-level splits, fold-local preprocessing, margin (``prob - threshold``)
  aggregation across folds, and balanced-accuracy-first thresholds are kept.
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
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
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


def select_threshold(y: np.ndarray, prob: np.ndarray, objective: str = "balanced_accuracy") -> float:
    grid = np.unique(np.clip(np.round(prob, 3), 0.02, 0.98))
    candidates = np.concatenate([[0.5], grid]) if len(grid) else np.asarray([0.5])
    best_threshold, best_key = 0.5, (-1.0, -1.0)
    for threshold in candidates:
        m = binary_metrics(y, (prob >= threshold).astype(int))
        if objective == "accuracy":
            key = (m["accuracy"], m["balanced_accuracy"])
        else:
            key = (m["balanced_accuracy"], m["accuracy"])
        if key > best_key:
            best_key, best_threshold = key, float(threshold)
    return best_threshold


def _gated_weights(balaccs: dict[str, float], gate: float) -> tuple[dict[str, float], str]:
    eligible = {n: max(0.0, b - 0.5) for n, b in balaccs.items() if b >= gate}
    if eligible and sum(eligible.values()) > 0:
        return eligible, "gated_blend"
    best = max(balaccs, key=balaccs.get)
    return {best: 1.0}, f"fallback_best:{best}"


def _combine(probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total = sum(weights.values())
    length = len(next(iter(probs.values())))
    if total <= 0:
        return np.mean(np.vstack([probs[n] for n in weights]), axis=0)
    out = np.zeros(length)
    for name, weight in weights.items():
        out += weight * probs[name]
    return out / total


def _inner_oof(data, factories, train_idx, inner_k, seed):
    y = data.y[train_idx]
    oof = {name: np.full(len(train_idx), np.nan) for name, _ in factories}
    inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=seed)
    for itr, ite in inner.split(train_idx, y):
        gtr, gte = train_idx[itr], train_idx[ite]
        for name, factory in factories:
            learner = factory()
            learner.fit(gtr)
            oof[name][ite] = learner.predict_proba(gte)
    return oof


def nested_cv(
    data,
    factories: Sequence[tuple[str, LearnerFactory]],
    *,
    repeats: int,
    outer_k: int,
    inner_k: int,
    weight_gate: float = 0.55,
    seed: int = 20260724,
) -> dict:
    n = data.n_subjects
    y = data.y
    prob_sum = np.zeros(n)
    margin_sum = np.zeros(n)
    seen = np.zeros(n)
    fold_metrics: list[dict] = []
    model_balacc: dict[str, list[float]] = {name: [] for name, _ in factories}
    model_weight: dict[str, list[float]] = {name: [] for name, _ in factories}
    fold_modes: list[str] = []

    for repeat in range(repeats):
        outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(outer.split(np.arange(n), y)):
            inner_seed = seed + 100 * repeat + fold
            oof = _inner_oof(data, factories, train_idx, inner_k, inner_seed)
            balaccs = {
                name: binary_metrics(y[train_idx], (oof[name] >= 0.5).astype(int))["balanced_accuracy"]
                for name, _ in factories
            }
            weights, mode = _gated_weights(balaccs, weight_gate)
            fold_modes.append(mode)
            for name, _ in factories:
                model_balacc[name].append(balaccs[name])
                model_weight[name].append(weights.get(name, 0.0))

            combined_inner = _combine({n_: oof[n_] for n_ in weights}, weights)
            threshold = select_threshold(y[train_idx], combined_inner)

            test_probs = {}
            for name in weights:
                learner = dict(factories)[name]()
                learner.fit(train_idx)
                test_probs[name] = learner.predict_proba(test_idx)
            combined_test = _combine(test_probs, weights)

            prob_sum[test_idx] += combined_test
            margin_sum[test_idx] += combined_test - threshold
            seen[test_idx] += 1
            fold_metrics.append({
                "repeat": repeat, "fold": fold, "threshold": threshold, "mode": mode,
                **binary_metrics(y[test_idx], (combined_test >= threshold).astype(int), combined_test),
            })

    if np.any(seen == 0):
        raise AssertionError("Every subject must be evaluated at least once")
    prob = prob_sum / seen
    margin = margin_sum / seen
    return {
        "oof_prob": prob,
        "oof_margin": margin,
        "oof_pred": (margin > 0).astype(int),
        "oof_metrics_selected_threshold": binary_metrics(y, (margin > 0).astype(int), prob),
        "oof_metrics_threshold_0.5": binary_metrics(y, (prob >= 0.5).astype(int), prob),
        "oof_metrics_accuracy_threshold": binary_metrics(
            y, (prob >= select_threshold(y, prob, "accuracy")).astype(int), prob
        ),
        "accuracy_threshold": select_threshold(y, prob, "accuracy"),
        "fold_metrics": fold_metrics,
        "fold_modes": fold_modes,
        "model_inner_balanced_accuracy": {n: float(np.mean(v)) for n, v in model_balacc.items()},
        "model_mean_weight": {n: float(np.mean(v)) for n, v in model_weight.items()},
        "weight_gate": weight_gate,
    }


def bootstrap_ci(y, prob, margin, *, n_boot=1000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    acc, bal, auc = [], [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        if len(np.unique(y[s])) < 2:
            continue
        m = binary_metrics(y[s], (margin[s] > 0).astype(int), prob[s])
        acc.append(m["accuracy"]); bal.append(m["balanced_accuracy"]); auc.append(m["roc_auc"])

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2

    return {"accuracy": ci(acc), "balanced_accuracy": ci(bal), "roc_auc": ci(auc)}


__all__ = ["binary_metrics", "bootstrap_ci", "nested_cv", "select_threshold"]
