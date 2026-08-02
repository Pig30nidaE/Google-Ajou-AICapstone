"""Fold-local fitted steps: resampling, imputation and feature selection.

Everything here is fitted on a *training* fold only and then applied to the
held-out fold.  That is the difference between this file and the report:

* The report ranks features with SHAP from a model trained on **all 174**
  people, then picks ``K=15`` by maximising the very 5-fold AUC it reports.
  Both the ranking and ``K`` therefore saw the evaluation labels.
* Here the ranking comes from a model trained on the outer-training fold only,
  and ``K`` is chosen on an **inner** CV of that same fold.  The outer fold is
  untouched until scoring.

SMOTE is reimplemented in NumPy rather than pulled from ``imbalanced-learn``:
it is ~40 lines, it removes a dependency, and it makes the fold-locality
auditable in one place.  The report's own SMOTE handling was already correct
(train-fold only) and that behaviour is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .metrics import roc_auc


# ------------------------------------------------------------------- SMOTE ----
def _knn_indices(points: np.ndarray, k: int, seed: int) -> np.ndarray:
    k = max(1, min(k, len(points) - 1))
    finder = NearestNeighbors(n_neighbors=k + 1).fit(points)
    return finder.kneighbors(points, return_distance=False)[:, 1:]


def smote_resample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    kind: str = "borderline",
    k_neighbors: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class up to parity.

    ``kind='plain'`` is the report's V26 SMOTE; ``kind='borderline'`` is its V29
    Borderline-SMOTE, which synthesises only from minority points that sit near
    the decision boundary (at least one majority neighbour, but not *all*
    majority neighbours -- those are treated as noise).

    Input must already be finite; call :meth:`FoldPreprocessor.transform` first.
    """

    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    counts = {c: int(np.count_nonzero(y == c)) for c in (0, 1)}
    minority = min(counts, key=counts.get)
    majority = 1 - minority
    n_needed = counts[majority] - counts[minority]
    minority_idx = np.flatnonzero(y == minority)

    # Not enough minority points to interpolate between: return unchanged.
    if n_needed <= 0 or minority_idx.size < 2:
        return X, y

    # Neighbour search must run in standardised space.  These features mix
    # fractions (0-1) with calories (thousands); on raw scales the large-unit
    # columns would dictate every neighbourhood.  Interpolation itself is
    # affine-invariant, so the synthetic points are built on the raw scale.
    center = X.mean(axis=0)
    spread = X.std(axis=0)
    spread = np.where(spread > 1e-12, spread, 1.0)
    Z = (X - center) / spread

    seeds = minority_idx
    if kind == "borderline" and counts[majority] > 0:
        k = max(1, min(k_neighbors, len(X) - 1))
        neighbours = NearestNeighbors(n_neighbors=k + 1).fit(Z)
        near = neighbours.kneighbors(Z[minority_idx], return_distance=False)[:, 1:]
        majority_fraction = (y[near] == majority).mean(axis=1)
        danger = (majority_fraction >= 0.5) & (majority_fraction < 1.0)
        if np.count_nonzero(danger) >= 2:
            seeds = minority_idx[danger]
        # else: no usable borderline region -> fall back to plain SMOTE seeds.

    minority_points = X[seeds]
    neighbour_idx = _knn_indices(Z[seeds], k_neighbors, seed)

    picks = rng.integers(0, len(seeds), size=n_needed)
    partners = neighbour_idx[picks, rng.integers(0, neighbour_idx.shape[1], size=n_needed)]
    gaps = rng.random((n_needed, 1))
    synthetic = minority_points[picks] + gaps * (minority_points[partners] - minority_points[picks])

    X_out = np.vstack([X, synthetic])
    y_out = np.concatenate([y, np.full(n_needed, minority, dtype=np.int64)])
    order = rng.permutation(len(y_out))
    return X_out[order], y_out[order]


# ------------------------------------------------------------ preprocessing ---
@dataclass
class FoldPreprocessor:
    """Median imputation + variance filter, fitted on one training fold."""

    keep_mask_: np.ndarray = field(default=None, repr=False)
    medians_: np.ndarray = field(default=None, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FoldPreprocessor":
        X = np.asarray(X, dtype=np.float64)
        clean = np.where(np.isfinite(X), X, np.nan)

        with np.errstate(invalid="ignore"):
            medians = np.nanmedian(clean, axis=0)
        self.medians_ = np.where(np.isfinite(medians), medians, 0.0)

        filled = np.where(np.isnan(clean), self.medians_, clean)
        spread = np.nanstd(filled, axis=0)
        # Drop columns that are constant *within this fold*.
        self.keep_mask_ = np.isfinite(spread) & (spread > 1e-12)
        if not self.keep_mask_.any():                      # degenerate fold
            self.keep_mask_ = np.ones(X.shape[1], dtype=bool)
        return self

    def transform(self, X: np.ndarray, *, impute: bool = True) -> np.ndarray:
        """Apply the fold's column filter, optionally imputing missing values.

        ``impute=False`` keeps NaN so YDF can use its native missing-value
        splits -- the report's stated reason for avoiding imputation on trees.
        """

        X = np.asarray(X, dtype=np.float64)
        clean = np.where(np.isfinite(X), X, np.nan)
        if impute:
            clean = np.where(np.isnan(clean), self.medians_, clean)
        return clean[:, self.keep_mask_]


# -------------------------------------------------------- feature selection ---
def rank_features(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 0,
    n_seeds: int = 3,
) -> np.ndarray:
    """Rank features by YDF importance, averaged over a few seeds.

    Google-native replacement for the report's SHAP ranking.  ``X``/``y`` must
    be the training fold only.  Averaging over seeds damps the fold-to-fold
    ranking instability diagnosed in earlier experiments in this repo.
    """

    from .models import GoogleTreeModel, default_params

    params = dict(default_params("gbt_leafwise"), num_trees=200)
    totals = np.zeros(X.shape[1], dtype=np.float64)
    for offset in range(max(1, n_seeds)):
        model = GoogleTreeModel("gbt_leafwise", params, seed=seed + offset).fit(X, y)
        scores = model.importances()
        total = scores.sum()
        if total > 0:
            totals += scores / total           # normalise so seeds weigh equally
    return np.argsort(-totals, kind="stable")


def forward_select(
    X: np.ndarray,
    y: np.ndarray,
    ranking: np.ndarray,
    *,
    max_candidates: int = 40,
    max_selected: int = 15,
    inner_splits: int = 5,
    seed: int = 0,
) -> tuple[list[int], list[dict]]:
    """The report's forward selection, scored on an *inner* CV of this fold.

    Walks the ranked candidates, keeping a feature only when it improves inner
    AUC, and stops at ``max_selected``.  Returns ``(selected, trace)``.
    """

    from .models import GoogleTreeModel, default_params

    candidates = [int(i) for i in ranking[:max_candidates]]
    params = dict(default_params("gbt_leafwise"), num_trees=150)

    n_minority = int(min(np.count_nonzero(y == 0), np.count_nonzero(y == 1)))
    splits = max(2, min(inner_splits, n_minority))
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y))

    def score(columns: list[int]) -> float:
        if not columns:
            return 0.5
        subset = X[:, columns]
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for train_idx, test_idx in folds:
            if len(np.unique(y[train_idx])) < 2:
                continue
            model = GoogleTreeModel("gbt_leafwise", params, seed=seed).fit(
                subset[train_idx], y[train_idx]
            )
            oof[test_idx] = model.predict_proba(subset[test_idx])
        mask = ~np.isnan(oof)
        if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
            return 0.5
        return roc_auc(y[mask], oof[mask])

    selected: list[int] = []
    best = 0.5
    trace: list[dict] = []
    for candidate in candidates:
        if len(selected) >= max_selected:
            break
        trial = selected + [candidate]
        achieved = score(trial)
        improved = achieved > best + 1e-6
        trace.append({"feature": candidate, "inner_auc": round(float(achieved), 5),
                      "kept": bool(improved), "k": len(trial)})
        if improved:
            selected, best = trial, achieved

    if not selected:                              # nothing helped: keep the top rank
        selected = candidates[:1]
        best = score(selected)
    return selected, trace


def random_search(
    X: np.ndarray,
    y: np.ndarray,
    kind: str,
    *,
    budget: int = 12,
    inner_splits: int = 3,
    seed: int = 0,
) -> dict:
    """Fold-local random search over :func:`models.search_space`.

    Google's YDF ships its own tuner, but it optimises against an internal
    holdout carved from the same frame.  Scoring here on an explicit inner CV of
    the outer-training fold keeps the fold boundary auditable, which is the
    property the report's Optuna stage lacked.
    """

    from .models import GoogleTreeModel, default_params, search_space

    rng = np.random.default_rng(seed)
    space = search_space(kind)
    base = default_params(kind)

    n_minority = int(min(np.count_nonzero(y == 0), np.count_nonzero(y == 1)))
    splits = max(2, min(inner_splits, n_minority))
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(y)), y))

    def evaluate(params: dict) -> float:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for train_idx, test_idx in folds:
            if len(np.unique(y[train_idx])) < 2:
                continue
            model = GoogleTreeModel(kind, params, seed=seed).fit(X[train_idx], y[train_idx])
            oof[test_idx] = model.predict_proba(X[test_idx])
        mask = ~np.isnan(oof)
        if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
            return 0.5
        return roc_auc(y[mask], oof[mask])

    best_params, best_score = dict(base), evaluate(base)
    for _ in range(max(0, budget - 1)):
        trial = dict(base)
        for key, choices in space.items():
            trial[key] = choices[int(rng.integers(0, len(choices)))]
        achieved = evaluate(trial)
        if achieved > best_score + 1e-6:
            best_params, best_score = trial, achieved
    return {"params": best_params, "inner_auc": float(best_score)}
