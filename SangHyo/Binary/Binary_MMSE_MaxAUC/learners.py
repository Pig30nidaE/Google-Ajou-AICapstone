"""Fold-safe learners: regularized logistic regression + RBF SVM.

EDA model search showed these two are the strongest and most stable on the
MMSE feature space (leakage-free subject ROC-AUC ~0.757 / 0.761); tree models
(HistGBT) were weaker (~0.73) because the signal is small and mostly linear.
All preprocessing (median impute + standardize) is fit on ``train_idx`` only.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .features import SubjectData


class _FoldPrep:
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
        return LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000)
    if kind == "svm":
        return SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
                   probability=True, random_state=0)
    raise ValueError(kind)


class Learner:
    """kind in {'logreg', 'svm'}; fold-local prep + calibrated probability."""

    def __init__(self, data: SubjectData, kind: str) -> None:
        self.data = data
        self.kind = kind

    def fit(self, train_idx: np.ndarray) -> "Learner":
        x, y = self.data.X[train_idx], self.data.y[train_idx]
        self.prep_ = _FoldPrep().fit(x)
        self.model_ = _estimator(self.kind).fit(self.prep_.transform(x), y)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.predict_proba_matrix(self.data.X[idx])

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(self.prep_.transform(np.asarray(X, float)))[:, 1]


__all__ = ["Learner"]
