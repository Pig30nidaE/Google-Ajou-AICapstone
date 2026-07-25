"""Fold-safe tabular learners for the MMSE-domain fusion model.

Robust gradient boosting + regularized logistic regression + random forest over
the compact, EDA-grounded feature set.  All preprocessing is fit on the given
``train_idx`` only.  Because the feature set is already small and denoised
(MMSE domains + curated wearable markers), feature selection is light.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

from .features import SubjectData


def _safe_f_classif(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, pvalues = f_classif(x, y)
    return np.nan_to_num(scores, nan=0.0), np.nan_to_num(pvalues, nan=1.0)


class _FoldPrep:
    def __init__(self, max_features: int) -> None:
        self.max_features = max_features

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_FoldPrep":
        self.median_ = np.nanmedian(x, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = self._impute(x)
        self.low_, self.high_ = np.percentile(filled, [1, 99], axis=0)
        clipped = np.clip(filled, self.low_, self.high_)
        self.mean_ = clipped.mean(axis=0)
        self.std_ = np.where(clipped.std(axis=0) < 1e-8, 1.0, clipped.std(axis=0))
        scaled = (clipped - self.mean_) / self.std_
        k = min(self.max_features, scaled.shape[1])
        self.selector_ = SelectKBest(_safe_f_classif, k=k).fit(scaled, y)
        return self

    def _impute(self, x):
        return np.where(np.isfinite(x), x, self.median_)

    def transform(self, x):
        clipped = np.clip(self._impute(x), self.low_, self.high_)
        return self.selector_.transform((clipped - self.mean_) / self.std_)


def _estimator(kind: str):
    if kind == "gbt":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=10, l2_regularization=1.0,
            class_weight="balanced", random_state=0,
        )
    if kind == "logreg":
        return LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=0,
        )
    raise ValueError(kind)


class TabularLearner:
    def __init__(self, data: SubjectData, kind: str, max_features: int = 20) -> None:
        self.data = data
        self.kind = kind
        self.max_features = max_features

    def fit(self, train_idx: np.ndarray) -> "TabularLearner":
        x, y = self.data.X[train_idx], self.data.y[train_idx]
        self.prep_ = _FoldPrep(self.max_features).fit(x, y)
        self.model_ = _estimator(self.kind).fit(self.prep_.transform(x), y)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.predict_proba_matrix(self.data.X[idx])

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(self.prep_.transform(np.asarray(X, float)))[:, 1]


__all__ = ["TabularLearner"]
