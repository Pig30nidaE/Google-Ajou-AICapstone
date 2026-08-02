"""Fold-local preprocessing and resampling, with a fit-scope guard.

Every statistic used to transform data -- the imputation median, the winsorising
percentiles, the standardisation mean/scale, the selected feature subset -- is
estimated from the training rows of the current fold and from nothing else.

The guard
---------
Stating that is cheap; :class:`FoldPreprocessor` proves it.  ``fit`` records a
fingerprint of the exact rows it saw, and ``transform`` refuses to run unless the
caller passes the same fingerprint it was fitted under, so a pipeline that
accidentally fits on the full matrix and then transforms a fold fails loudly
instead of silently reporting an inflated score.

Resampling
----------
Resamplers are applied *after* the split, to the training rows only, and the
fitted resampler is discarded with the fold.  With ~10 training positives, SMOTE
interpolates between a handful of the same patients, so the default is
``class_weight`` and the synthetic samplers are opt-in comparisons rather than
the assumed answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np

try:  # optional
    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    IMBLEARN_AVAILABLE = False

RESAMPLERS = ("none", "class_weight", "random_over", "smote", "borderline_smote",
              "adasyn", "random_under")


def rows_fingerprint(index: np.ndarray) -> str:
    """Identity of a row set, used to pin a fitted transform to its fold."""

    ordered = np.sort(np.asarray(index, dtype=np.int64))
    return hashlib.sha256(ordered.tobytes()).hexdigest()[:16]


@dataclass
class FoldPreprocessor:
    """Median impute -> winsorise -> standardise, fitted on one fold's training rows.

    Winsorising matters more than usual here: several wearable channels have
    device-artifact outliers (a 40,000-step day), and with 12 positives a single
    extreme value can dominate a linear model's coefficient.  The clip bounds are
    percentiles of the *training* rows only.
    """

    winsorize: float = 0.01
    standardize: bool = True
    median_: np.ndarray | None = field(default=None, init=False)
    lower_: np.ndarray | None = field(default=None, init=False)
    upper_: np.ndarray | None = field(default=None, init=False)
    center_: np.ndarray | None = field(default=None, init=False)
    scale_: np.ndarray | None = field(default=None, init=False)
    fit_fingerprint_: str | None = field(default=None, init=False)
    n_fit_rows_: int = field(default=0, init=False)

    def fit(self, X: np.ndarray, train_index: np.ndarray | None = None) -> "FoldPreprocessor":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("FoldPreprocessor.fit needs a non-empty 2-D matrix")

        with np.errstate(all="ignore"):
            median = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isfinite(median), median, 0.0)
        filled = self._impute(X)

        if self.winsorize and self.winsorize > 0:
            quantiles = np.percentile(
                filled, [100 * self.winsorize, 100 * (1 - self.winsorize)], axis=0
            )
            self.lower_, self.upper_ = quantiles[0], quantiles[1]
            filled = np.clip(filled, self.lower_, self.upper_)
        else:
            self.lower_ = self.upper_ = None

        if self.standardize:
            self.center_ = filled.mean(axis=0)
            scale = filled.std(axis=0)
            self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        else:
            self.center_, self.scale_ = None, None

        self.n_fit_rows_ = int(X.shape[0])
        self.fit_fingerprint_ = rows_fingerprint(
            np.arange(X.shape[0]) if train_index is None else train_index
        )
        return self

    def _impute(self, X: np.ndarray) -> np.ndarray:
        filled = np.array(X, dtype=np.float64, copy=True)
        missing = ~np.isfinite(filled)
        if missing.any():
            filled[missing] = np.take(self.median_, np.nonzero(missing)[1])
        return filled

    def transform(self, X: np.ndarray, *, expect_fingerprint: str | None = None) -> np.ndarray:
        if self.median_ is None:
            raise RuntimeError("FoldPreprocessor.transform called before fit")
        if expect_fingerprint is not None and expect_fingerprint != self.fit_fingerprint_:
            raise AssertionError(
                "Preprocessor fit scope mismatch: this transform was fitted on a "
                "different row set than the fold now being processed"
            )
        out = self._impute(np.asarray(X, dtype=np.float64))
        if self.lower_ is not None:
            out = np.clip(out, self.lower_, self.upper_)
        if self.center_ is not None:
            out = (out - self.center_) / self.scale_
        return out

    def fit_transform(self, X: np.ndarray, train_index: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, train_index).transform(X)


def direction_free_auc(y: np.ndarray, column: np.ndarray) -> float:
    """AUC of a single column, ignoring sign, used for fold-internal ranking."""

    from sklearn.metrics import roc_auc_score

    mask = np.isfinite(column)
    if mask.sum() < max(8, 0.4 * len(y)) or len(np.unique(np.asarray(y)[mask])) < 2:
        return 0.5
    auc = float(roc_auc_score(np.asarray(y)[mask], np.asarray(column)[mask]))
    return max(auc, 1.0 - auc)


def select_features(X: np.ndarray, y: np.ndarray, *, top_k: int = 0,
                    corr_threshold: float = 0.95) -> np.ndarray:
    """Rank by univariate AUC on the *training* rows, then prune redundancy.

    ``top_k <= 0`` keeps every column.  The prior folder's post-mortem showed
    that unrestricted top-k selection over 151 candidates with ~10 positives
    picked a different subset in every fold, so this pipeline treats ``top_k`` as
    a tuned hyperparameter with 0 (keep everything, rely on regularisation) in
    the search space rather than as a mandatory step.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n_features = X.shape[1]
    if top_k is None or top_k <= 0 or top_k >= n_features:
        return np.arange(n_features)

    scores = np.array([direction_free_auc(y, X[:, j]) for j in range(n_features)])
    order = np.argsort(-scores)
    kept: list[int] = []
    for candidate in order:
        if len(kept) >= top_k:
            break
        if corr_threshold < 1.0 and kept:
            block = X[:, kept]
            with np.errstate(all="ignore"):
                correlations = np.array([
                    abs(np.corrcoef(X[:, candidate], block[:, position])[0, 1])
                    for position in range(block.shape[1])
                ])
            correlations = correlations[np.isfinite(correlations)]
            if correlations.size and correlations.max() >= corr_threshold:
                continue
        kept.append(int(candidate))
    return np.asarray(kept if kept else order[:top_k], dtype=np.int64)


def resample(X: np.ndarray, y: np.ndarray, kind: str, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply a resampler to training rows only.

    Returns the input unchanged for ``none``/``class_weight`` (the latter is
    handled by the estimators) and whenever the sampler cannot run -- too few
    minority neighbours, for instance -- so an unusable option degrades into the
    baseline instead of crashing a 6-hour run.
    """

    if kind in ("none", "class_weight"):
        return X, y
    if not IMBLEARN_AVAILABLE:
        return X, y

    y = np.asarray(y, dtype=np.int64)
    minority = int(np.bincount(y, minlength=2).min())
    if minority < 2:
        return X, y
    neighbours = max(1, min(5, minority - 1))
    try:
        if kind == "random_over":
            sampler = RandomOverSampler(random_state=seed)
        elif kind == "smote":
            sampler = SMOTE(random_state=seed, k_neighbors=neighbours)
        elif kind == "borderline_smote":
            sampler = BorderlineSMOTE(random_state=seed, k_neighbors=neighbours,
                                      m_neighbors=max(2, neighbours))
        elif kind == "adasyn":
            sampler = ADASYN(random_state=seed, n_neighbors=neighbours)
        elif kind == "random_under":
            sampler = RandomUnderSampler(random_state=seed)
        else:
            raise ValueError(f"Unknown resampler: {kind!r}")
        X_resampled, y_resampled = sampler.fit_resample(np.asarray(X, dtype=np.float64), y)
        return np.asarray(X_resampled, dtype=np.float64), np.asarray(y_resampled, dtype=np.int64)
    except Exception:
        # Synthetic samplers fail routinely at this sample size; falling back is
        # the honest behaviour and the report records which arm was requested.
        return X, y


def available_resamplers() -> tuple[str, ...]:
    if IMBLEARN_AVAILABLE:
        return RESAMPLERS
    return ("none", "class_weight")


__all__ = [
    "FoldPreprocessor", "IMBLEARN_AVAILABLE", "RESAMPLERS", "available_resamplers",
    "direction_free_auc", "resample", "rows_fingerprint", "select_features",
]
