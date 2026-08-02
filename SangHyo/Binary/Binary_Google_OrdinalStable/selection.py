"""Stability selection -- a direct answer to the failure measured in MaxAUC_Tuned.

What went wrong there
---------------------
Feature selection was re-run per fold as "rank by univariate AUC, keep the best
``top_k``", with ``top_k`` itself tuned.  Across the 15 outer folds the tuner
chose::

    [70, 10, 70, 10, 70, 15, 0, 10, 70, 30, 20, 45, 20, 10, 45]

That is not a hyperparameter converging; that is a coin flip.  With 141 subjects
and 151 candidate features, the univariate ranking is unstable enough that each
fold sees a different "best" feature set, and inner-fold AUC ran +0.053 above
outer-fold AUC because the search was fitting that instability.

What this does instead
----------------------
``stability_select`` resamples the *training part of the fold* B times, recomputes
the ranking on each resample, and keeps only features that survive in at least
``threshold`` of them.  A feature that is genuinely informative shows up
regardless of which subjects were drawn; a feature that won by noise does not.
The number of features is then an **outcome** of how much stable signal exists,
not a knob to be tuned -- which removes the degree of freedom that was being
overfitted.

This is the Meinshausen-Bühlmann stability-selection idea applied to a univariate
filter (cheap enough to run inside every fold, unlike a full model-based version).
Hyunsoo's log arrives at the same instinct from the other direction: their SHAP
sweep kept a feature only because it landed in the top-3 in 91% of 100 repeats.
"""

from __future__ import annotations

import numpy as np

from SangHyo.Binary.Binary_Google_MaxAUC_Tuned.engine import direction_free_auc


def _rank_once(X: np.ndarray, y: np.ndarray, rows: np.ndarray, top_k: int) -> np.ndarray:
    scores = np.full(X.shape[1], 0.5)
    for j in range(X.shape[1]):
        column = X[rows, j]
        mask = np.isfinite(column)
        if mask.sum() >= 8 and len(np.unique(y[rows][mask])) > 1 and np.std(column[mask]) > 1e-10:
            scores[j] = direction_free_auc(y[rows], column)
    return np.argsort(-scores)[:top_k]


def stability_select(X: np.ndarray, y: np.ndarray, *, n_bootstrap: int = 60, top_k: int = 25,
                     threshold: float = 0.6, min_features: int = 3, max_features: int = 40,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Features whose ranking survives resampling of the training rows.

    Returns ``(selected_columns, selection_frequency)``.  Called only with the
    training part of a fold, so the held-out part never influences it.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n, p = X.shape
    rng = np.random.default_rng(seed)
    counts = np.zeros(p)

    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]
    take_pos = max(2, int(round(0.632 * len(positives))))
    take_neg = max(2, int(round(0.632 * len(negatives))))

    for _ in range(n_bootstrap):
        # subsample without replacement, stratified: keeps both classes present
        rows = np.concatenate([rng.choice(positives, take_pos, replace=False),
                               rng.choice(negatives, take_neg, replace=False)])
        counts[_rank_once(X, y, rows, top_k)] += 1

    frequency = counts / n_bootstrap
    selected = np.where(frequency >= threshold)[0]

    if len(selected) < min_features:
        selected = np.argsort(-frequency)[:min_features]
    elif len(selected) > max_features:
        selected = selected[np.argsort(-frequency[selected])[:max_features]]
    return np.sort(selected), frequency


def summarize_frequency(frequency: np.ndarray, feature_names, top: int = 20) -> list:
    order = np.argsort(-frequency)[:top]
    return [[feature_names[int(j)], round(float(frequency[j]), 3)] for j in order]


__all__ = ["stability_select", "summarize_frequency"]
