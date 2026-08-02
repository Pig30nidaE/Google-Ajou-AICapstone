"""Small numeric helpers shared by the feature/EDA/learner layers.

``numpy``'s ``nanmedian``/``nanstd`` emit a ``RuntimeWarning`` for every
all-NaN column they touch.  With ~150 features, repeated CV and a multi-hour
search that turns into tens of thousands of identical warning lines in the
Colab log, which buries the actual progress output.  These helpers compute the
same quantities from the finite entries only, so an all-NaN column simply gets
the documented fallback instead of a warning.

Kept in its own module because ``engine`` imports ``learners``; a shared helper
in either of those would create an import cycle.
"""

from __future__ import annotations

import numpy as np


def column_median(X: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Per-column median over finite entries; ``fill`` where a column has none."""

    X = np.asarray(X, dtype=float)
    finite = np.isfinite(X)
    out = np.full(X.shape[1], float(fill))
    for j in range(X.shape[1]):
        values = X[finite[:, j], j]
        if values.size:
            out[j] = float(np.median(values))
    return out


def column_std(X: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Per-column std over finite entries; ``fill`` where fewer than 2 exist."""

    X = np.asarray(X, dtype=float)
    finite = np.isfinite(X)
    out = np.full(X.shape[1], float(fill))
    for j in range(X.shape[1]):
        values = X[finite[:, j], j]
        if values.size >= 2:
            out[j] = float(np.std(values))
    return out


def impute(X: np.ndarray, median: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(X), X, median)


__all__ = ["column_median", "column_std", "impute"]
