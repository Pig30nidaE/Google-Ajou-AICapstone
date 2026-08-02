"""Repeated nested CV for a 12-positive screening problem.

Everything decided per fold -- feature subset, blend weights **and the operating
threshold** -- is decided on the outer-training part only.  The threshold is
called out because it is the one Hyunsoo's log flags as the mistake that was
easiest to miss: picking Youden's J on the pooled out-of-fold predictions you
then report inflated Precision from 0.44 to 0.72 and F1 from 0.54 to 0.76 while
leaving ROC-AUC untouched.  ROC-AUC is threshold-free and so was never affected,
which is exactly why it is this folder's headline metric and why every
threshold-dependent number here comes from a nested threshold.

Reporting shape
---------------
With 12 positives a single 5-fold split is far too noisy to read, so the unit of
reporting is: **pool the out-of-fold predictions within one repeat, score once,
then repeat R times and report mean ± sd across repeats** (the protocol
Hyunsoo used), plus a bootstrap CI over subjects.  Both are reported because
they answer different questions -- the sd across repeats is split-noise, the
bootstrap CI is sampling-noise, and at n=12 the latter dominates.
"""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from SangHyo.Binary.Binary_Google_MaxAUC_Tuned.engine import (
    binary_metrics,
    blend_scores,
    direction_free_auc,
    optimize_blend,
    safe_auc,
    select_features,
)

from .learners import make_learner
from .spaces import default_params, search_configs


def safe_folds(y: np.ndarray, k: int) -> int:
    """Never ask for more folds than the rarer class has members."""

    minority = int(np.bincount(np.asarray(y, dtype=np.int64), minlength=2).min())
    return int(max(2, min(k, minority)))


def youden_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y, prob)
    best = thresholds[int(np.argmax(tpr - fpr))]
    return float(np.clip(best, 0.0, 1.0))


def _fit_predict(X_tr, y_tr, X_te, kind, params, seed):
    cols = select_features(X_tr, y_tr, top_k=int(params.get("top_k", 0)),
                           corr_threshold=float(params.get("corr_threshold", 1.01)))
    learner = make_learner(kind, params, seed=seed)
    learner.fit(X_tr[:, cols], y_tr)
    return learner.predict_proba(X_te[:, cols]), cols, learner


def inner_oof(X, y, kind, params, *, k, seed):
    """Out-of-fold probabilities within the outer-training part."""

    prob = np.zeros(len(y))
    folds = StratifiedKFold(n_splits=safe_folds(y, k), shuffle=True, random_state=seed)
    for tr, te in folds.split(X, y):
        prob[te] = _fit_predict(X[tr], y[tr], X[te], kind, params, seed)[0]
    return prob


def tune_inner(X, y, kinds, *, inner_k, seed, auc_gate, search):
    """Per-kind configuration + blend weights, fitted on outer-train only."""

    chosen: dict[str, dict] = {}
    for kind in kinds:
        candidates = search_configs(kind) if search else [default_params(kind)]
        best = {"auc": -1.0, "params": candidates[0], "oof": None}
        for params in candidates:
            oof = inner_oof(X, y, kind, params, k=inner_k, seed=seed)
            auc = safe_auc(y, oof)
            if auc > best["auc"]:
                best = {"auc": auc, "params": params, "oof": oof}
        chosen[kind] = best

    eligible = [k for k in kinds if chosen[k]["auc"] >= auc_gate]
    mode = "gated_blend"
    if not eligible:
        eligible = [max(kinds, key=lambda k: chosen[k]["auc"])]
        mode = f"fallback_best:{eligible[0]}"
    matrix = np.column_stack([chosen[k]["oof"] for k in eligible])
    weights, blend_auc = optimize_blend(matrix, y, rng=np.random.default_rng(seed))
    threshold = youden_threshold(y, 1.0 / (1.0 + np.exp(-blend_scores(matrix, weights))))
    return {"chosen": chosen, "eligible": eligible, "weights": weights,
            "inner_auc": blend_auc, "threshold": threshold, "mode": mode}


def repeated_nested_cv(X, y, kinds, *, repeats=20, outer_k=5, inner_k=4, auc_gate=0.55,
                       search=False, seed=20260728,
                       log: Callable[[str], None] = print) -> dict:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = len(y)
    per_repeat: list[dict] = []
    prob_sum = np.zeros(n)
    kind_auc: dict[str, list[float]] = {k: [] for k in kinds}
    weight_log: dict[str, list[float]] = {k: [] for k in kinds}
    univariate_picks: list[int] = []
    started = time.monotonic()

    for repeat in range(repeats):
        outer = StratifiedKFold(n_splits=safe_folds(y, outer_k), shuffle=True,
                                random_state=seed + repeat)
        oof_prob = np.zeros(n)
        oof_pred = np.zeros(n, dtype=int)
        for fold, (tr, te) in enumerate(outer.split(X, y)):
            inner = tune_inner(X[tr], y[tr], kinds, inner_k=inner_k,
                               seed=seed + 100 * repeat + fold, auc_gate=auc_gate,
                               search=search)
            probs = []
            for kind in inner["eligible"]:
                prob, cols, learner = _fit_predict(X[tr], y[tr], X[te], kind,
                                                   inner["chosen"][kind]["params"],
                                                   seed + fold)
                probs.append(prob)
                if kind == "univariate" and learner.chosen_column is not None:
                    # chosen_column indexes the fold's selected subset, not the
                    # full matrix -- map it back so the report names a real feature.
                    univariate_picks.append(int(cols[learner.chosen_column]))
            blended = 1.0 / (1.0 + np.exp(-blend_scores(np.column_stack(probs), inner["weights"])))
            oof_prob[te] = blended
            oof_pred[te] = (blended >= inner["threshold"]).astype(int)
            for kind in kinds:
                kind_auc[kind].append(inner["chosen"][kind]["auc"])
            for kind, weight in zip(inner["eligible"], inner["weights"]):
                weight_log[kind].append(float(weight))

        metrics = binary_metrics(y, oof_pred, oof_prob)
        metrics["repeat"] = repeat
        per_repeat.append(metrics)
        prob_sum += oof_prob
        if (repeat + 1) % 5 == 0 or repeat == repeats - 1:
            done = [m["roc_auc"] for m in per_repeat]
            log(f"    repeat {repeat + 1}/{repeats}: AUC {np.mean(done):.4f} "
                f"(+/-{np.std(done):.4f})  [{time.monotonic() - started:.0f}s]")

    def summarize(key):
        values = [m[key] for m in per_repeat]
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    for kind in kinds:
        weight_log[kind] = weight_log.get(kind) or [0.0]

    return {
        "roc_auc": summarize("roc_auc"),
        "accuracy": summarize("accuracy"),
        "balanced_accuracy": summarize("balanced_accuracy"),
        "precision": summarize("precision"),
        "impaired_recall": summarize("impaired_recall"),
        "cn_specificity": summarize("cn_specificity"),
        "f1": summarize("f1"),
        "mean_confusion": {k: float(np.mean([m["confusion"][k] for m in per_repeat]))
                           for k in ("tn", "fp", "fn", "tp")},
        "mean_oof_prob": (prob_sum / repeats).tolist(),
        "inner_auc_by_kind": {k: float(np.mean(v)) for k, v in kind_auc.items()},
        "mean_blend_weight": {k: float(np.mean(v)) for k, v in weight_log.items()},
        "univariate_pick_counts": _pick_counts(univariate_picks),
        "per_repeat_roc_auc": [m["roc_auc"] for m in per_repeat],
        "repeats": repeats,
        "elapsed_seconds": time.monotonic() - started,
    }


def _pick_counts(picks: Sequence[int]) -> dict:
    if not picks:
        return {}
    values, counts = np.unique(np.asarray(picks), return_counts=True)
    order = np.argsort(-counts)
    return {int(values[i]): int(counts[i]) for i in order[:10]}


def bootstrap_auc_ci(y, prob, *, n_boot=4000, seed=0) -> dict:
    """Subject-level bootstrap CI -- the honest error bar when positives are this few."""

    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(safe_auc(y[idx], prob[idx]))
    if not values:
        return {"point": safe_auc(y, prob), "lo": float("nan"), "hi": float("nan")}
    return {"point": safe_auc(y, prob),
            "lo": float(np.percentile(values, 2.5)),
            "hi": float(np.percentile(values, 97.5)),
            "n_boot": len(values)}


def univariate_table(X, y, feature_names, top=25) -> list:
    ranked = sorted(((name, direction_free_auc(y, X[:, j]))
                     for j, name in enumerate(feature_names)), key=lambda kv: -kv[1])
    return [[name, round(auc, 4)] for name, auc in ranked[:top]]


__all__ = ["bootstrap_auc_ci", "inner_oof", "repeated_nested_cv", "safe_folds",
           "tune_inner", "univariate_table", "youden_threshold"]
