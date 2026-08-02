"""Combining models.  Every weight here is decided on inner OOF predictions.

Why rank averaging is a first-class citizen
-------------------------------------------
ROC-AUC is a function of the *ordering* of scores and nothing else.  The models
in this zoo emit wildly different scales -- an SVM margin, a YDF probability
concentrated in a few leaf values, a rank average already in [0, 1].  Averaging
those probabilities lets whichever model happens to have the widest numeric
range dominate the combined ordering, which is not the same thing as letting the
most *informative* model dominate.

The previous folder blended in log-odds space, arguing that it keeps an absolute
score so a threshold transfers.  That argument is right for thresholds and
irrelevant for AUC.  This module therefore offers both and lets the inner OOF
decide, with rank averaging available as a peer rather than an afterthought.

Combiners
---------
``prob_mean``   plain probability average (the previous default's spirit)
``logit_mean``  log-odds average
``rank_mean``   average of per-model rank-normalised scores
``rank_weighted`` Dirichlet-searched weights over rank-normalised scores
``greedy``      Caruana greedy selection with replacement
``stack_lr``    strongly regularised logistic regression on the OOF score matrix
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

COMBINERS = ("prob_mean", "logit_mean", "rank_mean", "rank_weighted", "greedy", "stack_lr")


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1] by rank, averaging ties.

    Rank normalisation is what makes heterogeneous models comparable: after it,
    every model contributes the same amount of ordering information regardless
    of its native scale.
    """

    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    if n == 0:
        return scores
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Average tied ranks so a model with many ties does not get an arbitrary
    # ordering imposed on it.
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if unique.size < n:
        sums = np.zeros(unique.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks / max(1.0, n - 1.0)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _to_unit(scores: np.ndarray) -> np.ndarray:
    """Min-max a raw score column into [0, 1] so probability-style combiners work
    for margin-emitting models too."""

    scores = np.asarray(scores, dtype=np.float64)
    low, high = float(np.min(scores)), float(np.max(scores))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-12:
        return np.full_like(scores, 0.5)
    return (scores - low) / (high - low)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    if finite.sum() < 2 or len(np.unique(y[finite])) < 2:
        return 0.5
    return float(roc_auc_score(y[finite], score[finite]))


@dataclass
class Blender:
    """A fitted combination rule.  ``fit`` sees inner OOF scores only."""

    kind: str
    weights: np.ndarray | None = None
    members: tuple[int, ...] = ()
    stacker: LogisticRegression | None = None
    inner_auc: float = float("nan")

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if self.kind == "prob_mean":
            return np.column_stack([_to_unit(matrix[:, j]) for j in range(matrix.shape[1])]).mean(axis=1)
        if self.kind == "logit_mean":
            return np.column_stack(
                [_logit(_to_unit(matrix[:, j])) for j in range(matrix.shape[1])]
            ).mean(axis=1)
        ranked = np.column_stack([rank_normalize(matrix[:, j]) for j in range(matrix.shape[1])])
        if self.kind == "rank_mean":
            return ranked.mean(axis=1)
        if self.kind == "rank_weighted":
            return ranked @ self.weights
        if self.kind == "greedy":
            if not self.members:
                return ranked.mean(axis=1)
            return ranked[:, list(self.members)].mean(axis=1)
        if self.kind == "stack_lr":
            return self.stacker.decision_function(ranked)
        raise ValueError(f"Unknown combiner: {self.kind!r}")


def fit_blender(kind: str, matrix: np.ndarray, y: np.ndarray, *, seed: int = 0,
                n_draws: int = 800, greedy_rounds: int = 12) -> Blender:
    """Fit one combination rule on inner OOF scores.

    ``n_draws`` is deliberately far below the previous folder's 4000: with ~10
    positives, searching a simplex harder mostly finds a better fit to the inner
    split's noise.  Every combiner here is also compared against the unweighted
    ``rank_mean``, which has zero free parameters.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    y = np.asarray(y, dtype=np.int64)
    n_models = matrix.shape[1]

    if kind in ("prob_mean", "logit_mean", "rank_mean") or n_models == 1:
        blender = Blender(kind=kind if n_models > 1 else "rank_mean")
        blender.inner_auc = safe_auc(y, blender.apply(matrix))
        return blender

    ranked = np.column_stack([rank_normalize(matrix[:, j]) for j in range(n_models)])

    if kind == "rank_weighted":
        rng = np.random.default_rng(seed)
        candidates = [np.eye(n_models)[i] for i in range(n_models)]
        candidates.append(np.full(n_models, 1.0 / n_models))
        candidates.extend(rng.dirichlet(np.ones(n_models), size=n_draws))
        best_weights, best_auc = candidates[0], -1.0
        for weights in candidates:
            auc = safe_auc(y, ranked @ weights)
            if auc > best_auc:
                best_auc, best_weights = auc, np.asarray(weights, dtype=np.float64)
        best_weights = best_weights / best_weights.sum()
        return Blender(kind=kind, weights=best_weights, inner_auc=float(best_auc))

    if kind == "greedy":
        # Caruana selection with replacement: start from the best single model
        # and keep adding whichever member most improves the running average.
        singles = [safe_auc(y, ranked[:, j]) for j in range(n_models)]
        members = [int(np.argmax(singles))]
        best_auc = max(singles)
        for _ in range(greedy_rounds - 1):
            improved = False
            for candidate in range(n_models):
                trial = members + [candidate]
                auc = safe_auc(y, ranked[:, trial].mean(axis=1))
                if auc > best_auc + 1e-9:
                    best_auc, best_candidate, improved = auc, candidate, True
            if not improved:
                break
            members.append(int(best_candidate))
        return Blender(kind=kind, members=tuple(members), inner_auc=float(best_auc))

    if kind == "stack_lr":
        stacker = LogisticRegression(C=float(0.1), penalty="l2", solver="lbfgs",
                                     class_weight="balanced", max_iter=5000,
                                     random_state=seed)
        stacker.fit(ranked, y)
        blender = Blender(kind=kind, stacker=stacker)
        blender.inner_auc = safe_auc(y, blender.apply(matrix))
        return blender

    raise ValueError(f"Unknown combiner: {kind!r}")


def best_blender(matrix: np.ndarray, y: np.ndarray, *, kinds=COMBINERS, seed: int = 0) -> Blender:
    """Pick the combination rule with the highest inner-OOF AUC.

    Ties resolve toward the earlier (simpler) entry in ``kinds``, which puts the
    parameter-free averages ahead of the searched weights.
    """

    best: Blender | None = None
    for kind in kinds:
        try:
            candidate = fit_blender(kind, matrix, y, seed=seed)
        except Exception:
            continue
        if best is None or candidate.inner_auc > best.inner_auc + 1e-9:
            best = candidate
    if best is None:  # pragma: no cover - only if every combiner raised
        best = Blender(kind="rank_mean")
        best.inner_auc = safe_auc(y, best.apply(matrix))
    return best


def seed_average(score_lists: list[np.ndarray]) -> np.ndarray:
    """Average several seeds of the same model in rank space."""

    if not score_lists:
        raise ValueError("seed_average needs at least one score vector")
    return np.column_stack([rank_normalize(s) for s in score_lists]).mean(axis=1)


__all__ = ["Blender", "COMBINERS", "best_blender", "fit_blender", "rank_normalize",
           "safe_auc", "seed_average"]
