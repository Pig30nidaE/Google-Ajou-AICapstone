"""Interpretable learners over the minimal EDA-selected feature set.

The EDA-selective model deliberately uses only the ~14 features the EDA proved
discriminative, so it favours simple, well-regularized, interpretable models:
an L2 logistic regression (with calibrated probabilities) plus a shallow
gradient-boosting model.  With so few, curated features there is no separate
feature-selection step — imputation and scaling are still fit fold-locally.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .features import SubjectData


class _FoldPrep:
    """Median impute + standardize, fit on training rows only."""

    def fit(self, x: np.ndarray) -> "_FoldPrep":
        self.median_ = np.nanmedian(x, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = self._impute(x)
        self.mean_ = filled.mean(axis=0)
        self.std_ = np.where(filled.std(axis=0) < 1e-8, 1.0, filled.std(axis=0))
        return self

    def _impute(self, x):
        return np.where(np.isfinite(x), x, self.median_)

    def transform(self, x):
        return (self._impute(x) - self.mean_) / self.std_


def _estimator(kind: str):
    if kind == "logreg":
        base = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)
        # Calibrated probabilities improve threshold stability on a small set.
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if kind == "gbt_shallow":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=3,
            min_samples_leaf=12, l2_regularization=1.0,
            class_weight="balanced", random_state=0,
        )
    raise ValueError(kind)


class TabularLearner:
    def __init__(self, data: SubjectData, kind: str) -> None:
        self.data = data
        self.kind = kind

    def fit(self, train_idx: np.ndarray) -> "TabularLearner":
        x, y = self.data.X[train_idx], self.data.y[train_idx]
        self.prep_ = _FoldPrep().fit(x)
        self.model_ = _estimator(self.kind).fit(self.prep_.transform(x), y)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.predict_proba_matrix(self.data.X[idx])

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(self.prep_.transform(np.asarray(X, float)))[:, 1]


__all__ = ["TabularLearner"]
