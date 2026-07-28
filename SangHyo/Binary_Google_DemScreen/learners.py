"""Learners sized for a 12-positive problem, with Google YDF as the tree engine.

Model choice is dominated by one fact: **there are 12 dementia subjects**, so a
5-fold split trains on about 9 of them.  Two independent results in this repo
say the same thing about that regime -- Hyunsoo's log (LightGBM and multi-feature
combinations lost to a single-feature logistic regression) and this repo's own
``Binary_Google_MaxAUC_Tuned`` run (151 features + 10 h of tuning scored *worse*
than 39 features untuned).  So every model here is deliberately small:

* ``univariate``      - the single best feature chosen inside the fold + logistic
  regression.  This is a faithful re-implementation of Hyunsoo's final model
  inside this folder's evaluation harness, so the comparison is apples-to-apples.
* ``logreg``          - L2 logistic regression on a handful of fold-selected features.
* ``ydf_gbt``         - Google YDF Gradient Boosted Trees, depth-capped and
  strongly regularized (a full-size GBT cannot be justified on 9 positives).
* ``ydf_rf``          - Google YDF Random Forest.
* ``ydf_gbt_oblique`` - YDF sparse-oblique GBT.  Oblique splits were the single
  best individual learner in the MaxAUC_Tuned run (inner AUC 0.789 vs 0.745 for
  axis-aligned), which is why the Google tree family is still worth carrying.

SMOTE is available because Hyunsoo used it, but it is **off by default**: with
~9 training positives, synthetic points are interpolations between a handful of
the same patients, which inflates apparent separation more often than it helps.
It is exposed as a switch so the nested evaluation can answer the question
rather than an opinion.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from SangHyo.Binary_Google_MaxAUC_Tuned.learners import YDFLearner
from SangHyo.Binary_Google_MaxAUC_Tuned.numeric import column_median, impute

YDF_KINDS = ("ydf_gbt", "ydf_rf", "ydf_gbt_oblique")
SK_KINDS = ("univariate", "logreg")
ALL_KINDS = SK_KINDS + YDF_KINDS

try:  # optional; only needed when smote=True
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception:  # pragma: no cover
    SMOTE_AVAILABLE = False


def _resample(X: np.ndarray, y: np.ndarray, seed: int):
    """SMOTE on the training fold only. Returns the input unchanged if unusable."""

    if not SMOTE_AVAILABLE:
        return X, y
    minority = int(np.bincount(y, minlength=2).min())
    if minority < 2:
        return X, y
    try:
        sampler = SMOTE(random_state=seed, k_neighbors=min(5, minority - 1))
        return sampler.fit_resample(X, y)
    except Exception:
        return X, y


class _Scaled:
    """Median-impute + standardize, fit on the training fold only."""

    def fit(self, X: np.ndarray) -> "_Scaled":
        self.median_ = column_median(X)
        filled = impute(X, self.median_)
        self.mean_ = filled.mean(axis=0)
        std = filled.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (impute(np.asarray(X, float), self.median_) - self.mean_) / self.std_


class LinearLearner:
    """kind='logreg' (all given columns) or 'univariate' (best single column)."""

    def __init__(self, kind: str, params: dict, *, seed: int = 0) -> None:
        self.kind = kind
        self.params = dict(params)
        self.seed = seed

    def _pick_column(self, X: np.ndarray, y: np.ndarray) -> int:
        best, best_score = 0, -1.0
        for j in range(X.shape[1]):
            column = X[:, j]
            mask = np.isfinite(column)
            if mask.sum() < 8 or len(np.unique(y[mask])) < 2 or np.std(column[mask]) < 1e-10:
                continue
            auc = roc_auc_score(y[mask], column[mask])
            score = max(auc, 1.0 - auc)
            if score > best_score:
                best, best_score = j, score
        return best

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearLearner":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.columns_ = (np.array([self._pick_column(X, y)]) if self.kind == "univariate"
                         else np.arange(X.shape[1]))
        Xs = X[:, self.columns_]
        self.scaler_ = _Scaled().fit(Xs)
        Xt, yt = self.scaler_.transform(Xs), y
        if self.params.get("smote"):
            Xt, yt = _resample(Xt, yt, self.seed)
        self.model_ = LogisticRegression(
            C=float(self.params.get("C", 1.0)), class_weight="balanced",
            max_iter=5000, random_state=self.seed).fit(Xt, yt)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = np.asarray(X, dtype=np.float64)[:, self.columns_]
        return self.model_.predict_proba(self.scaler_.transform(Xs))[:, 1]

    @property
    def chosen_column(self) -> int | None:
        return int(self.columns_[0]) if self.kind == "univariate" else None


class TreeLearner:
    """Google YDF wrapper; delegates to the audited MaxAUC_Tuned implementation."""

    def __init__(self, kind: str, params: dict, *, seed: int = 0) -> None:
        self.kind = kind
        self.params = dict(params)
        self.seed = seed
        self._inner = YDFLearner(kind, self.params, seed=seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeLearner":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.median_ = None
        if self.params.get("smote"):
            # SMOTE cannot interpolate through NaN, so impute first -- and keep
            # the median so predict-time rows are encoded the same way the trees
            # were trained on. (YDF handles NaN natively, so without SMOTE the
            # raw matrix is passed through untouched.)
            self.median_ = column_median(X)
            X, y = _resample(impute(X, self.median_), y, self.seed)
        self._inner.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.median_ is not None:
            X = impute(X, self.median_)
        return self._inner.predict_proba(X)

    def save(self, path) -> None:
        self._inner.save(path)


def make_learner(kind: str, params: dict, *, seed: int = 0):
    if kind in SK_KINDS:
        return LinearLearner(kind, params, seed=seed)
    if kind in YDF_KINDS:
        return TreeLearner(kind, params, seed=seed)
    raise ValueError(f"Unknown learner kind: {kind}")


__all__ = ["ALL_KINDS", "LinearLearner", "SK_KINDS", "SMOTE_AVAILABLE", "TreeLearner",
           "YDF_KINDS", "make_learner"]
