"""Fold-safe tabular learners over wearable summaries + MMSE scores.

MMSE features are informative and clean, so a robust tabular ensemble
(gradient boosting + regularized logistic regression + random forest) is the
appropriate, stable choice; no neural network or GPU is required.  All
preprocessing is fit on the given ``train_idx`` only.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

from .datalib import SubjectData


def _safe_f_classif(x: np.ndarray, y: np.ndarray):
    """ANOVA F-scores with constant columns scored 0 (never selected)."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, pvalues = f_classif(x, y)
    return np.nan_to_num(scores, nan=0.0), np.nan_to_num(pvalues, nan=1.0)


class _FoldTabularPrep:
    """Median impute -> 1/99 winsorize -> standardize -> SelectKBest, train-fit."""

    def __init__(self, max_features: int) -> None:
        self.max_features = max_features

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_FoldTabularPrep":
        self.median_ = np.nanmedian(x, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = self._impute(x)
        self.low_ = np.percentile(filled, 1, axis=0)
        self.high_ = np.percentile(filled, 99, axis=0)
        clipped = np.clip(filled, self.low_, self.high_)
        self.mean_ = clipped.mean(axis=0)
        self.std_ = clipped.std(axis=0)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        scaled = (clipped - self.mean_) / self.std_
        k = min(self.max_features, scaled.shape[1])
        self.selector_ = SelectKBest(_safe_f_classif, k=k).fit(scaled, y)
        return self

    def _impute(self, x: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(x), x, self.median_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(self._impute(x), self.low_, self.high_)
        scaled = (clipped - self.mean_) / self.std_
        return self.selector_.transform(scaled)


def _make_estimator(kind: str):
    if kind == "gbt":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=0,
        )
    if kind == "logreg":
        return LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=0,
        )
    raise ValueError(f"Unknown tabular estimator: {kind}")


class TabularLearner:
    def __init__(self, data: SubjectData, kind: str, max_features: int = 40) -> None:
        self.data = data
        self.kind = kind
        self.max_features = max_features

    def fit(self, train_idx: np.ndarray) -> "TabularLearner":
        x = self.data.tabular[train_idx]
        y = self.data.y[train_idx]
        self.prep_ = _FoldTabularPrep(self.max_features).fit(x, y)
        self.model_ = _make_estimator(self.kind).fit(self.prep_.transform(x), y)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.predict_proba_matrix(self.data.tabular[idx])

    def predict_proba_matrix(self, tabular: np.ndarray) -> np.ndarray:
        x = self.prep_.transform(np.asarray(tabular, dtype=np.float64))
        return self.model_.predict_proba(x)[:, 1]


__all__ = ["TabularLearner"]
