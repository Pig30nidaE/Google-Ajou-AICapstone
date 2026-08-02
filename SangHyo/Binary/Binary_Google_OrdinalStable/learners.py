"""Learners that can be trained three different ways on the same features.

The score being produced is always the same thing -- P(subject is MCI or Dem) --
but there are three ways to arrive at it, and which one ranks best is an
empirical question this folder measures rather than assumes:

``binary``
    Fit CN vs MCI+Dem directly. The baseline every other folder uses.

``ordinal``
    Fit the full 3-class problem (CN / MCI / Dem) and score with
    ``1 - P(CN)``. The model is told that Dem is a *more severe* state than MCI
    rather than the same state, so the 9 extreme Dem subjects stop dragging the
    CN/MCI boundary around. That boundary is what ROC-AUC on this task actually
    depends on: MMSE means are CN 27.7 / MCI 25.8 / Dem 16.6, so the Dem group
    is separable almost for free while MCI is nearly on top of CN.

``hard_boundary``
    Fit **CN vs MCI only**, dropping Dem from the training rows, then score
    everyone. This spends all model capacity on the difficult comparison; Dem
    subjects still score high at prediction time because they are extreme on the
    same features. Falls back to ``binary`` if a fold ends up without both
    classes.

Google YDF (GBT / RF / sparse-oblique GBT) is the tree engine and supports all
three natively, since YDF handles multi-class labels without a wrapper.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from SangHyo.Binary.Binary_Google_MaxAUC_Tuned.numeric import column_median, impute

STRATEGIES = ("binary", "ordinal", "hard_boundary")
YDF_KINDS = ("ydf_gbt", "ydf_rf", "ydf_gbt_oblique")
SK_KINDS = ("logreg", "svm")
ALL_KINDS = SK_KINDS + YDF_KINDS

try:
    import ydf  # noqa: F401
    YDF_AVAILABLE = True
except Exception:  # pragma: no cover
    YDF_AVAILABLE = False

_LOGREG_USES_L1_RATIO = LogisticRegression().get_params().get("l1_ratio") is not None
_SEVERITY_NAMES = ("CN", "MCI", "Dem")


def _class_weights(y: np.ndarray, names: tuple[str, ...]) -> dict:
    n = len(y)
    k = len(names)
    return {names[c]: (n / (k * max(1, int((y == c).sum())))) for c in range(k)}


# --------------------------------------------------------------------- YDF ----
def _ydf_learner(kind: str, params: dict, seed: int, class_names: tuple[str, ...]):
    import ydf

    common = dict(label="label", label_classes=list(class_names), random_seed=seed,
                  num_threads=int(os.cpu_count() or 4))
    if kind.endswith("_oblique"):
        common["split_axis"] = "SPARSE_OBLIQUE"

    if kind.startswith("ydf_gbt"):
        cls = ydf.GradientBoostedTreesLearner
        core = dict(num_trees=int(params.get("num_trees", 300)),
                    max_depth=int(params.get("max_depth", 3)),
                    min_examples=int(params.get("min_examples", 8)),
                    shrinkage=float(params.get("shrinkage", 0.05)),
                    subsample=float(params.get("subsample", 0.9)),
                    l2_regularization=float(params.get("l2_regularization", 2.0)),
                    validation_ratio=0.0)
    else:
        cls = ydf.RandomForestLearner
        core = dict(num_trees=int(params.get("num_trees", 500)),
                    max_depth=int(params.get("max_depth", 6)),
                    min_examples=int(params.get("min_examples", 5)),
                    num_candidate_attributes_ratio=float(
                        params.get("num_candidate_attributes_ratio", 0.5)))
    try:
        return cls(**core, **common)
    except Exception:
        common.pop("split_axis", None)
        return cls(**core, **common)


class _YDFBackend:
    def fit(self, X, y, kind, params, seed, class_names):
        self.class_names = class_names
        self.columns = [f"f{i}" for i in range(X.shape[1])]
        frame = pd.DataFrame(X, columns=self.columns)
        frame["label"] = [class_names[int(v)] for v in y]
        learner = _ydf_learner(kind, params, seed, class_names)
        learner.class_weights = _class_weights(y, class_names)
        self.model = learner.train(frame, verbose=0)
        self.order = [str(c) for c in self.model.label_classes()]
        return self

    def predict_matrix(self, X) -> np.ndarray:
        raw = np.asarray(self.model.predict(pd.DataFrame(X, columns=self.columns)), dtype=float)
        if raw.ndim == 1:  # YDF returns P(second class) for binary labels
            raw = np.column_stack([1.0 - raw, raw])
        # reorder to the caller's class_names order
        index = [self.order.index(name) for name in self.class_names]
        return raw[:, index]


class _SkBackend:
    def fit(self, X, y, kind, params, seed, class_names):
        self.class_names = class_names
        self.median_ = column_median(X)
        filled = impute(X, self.median_)
        self.mean_ = filled.mean(axis=0)
        std = filled.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)
        Xt = (filled - self.mean_) / self.std_

        if kind == "hgb":
            # Stand-in for YDF when the package is unavailable (local wiring runs).
            # NaN-tolerant and multi-class capable, so it keeps the tree character
            # of the arm instead of silently degrading it to a linear model.
            estimator = HistGradientBoostingClassifier(
                max_iter=int(params.get("num_trees", 300)),
                learning_rate=float(params.get("shrinkage", 0.05)),
                max_depth=int(params.get("max_depth", 3)),
                min_samples_leaf=int(params.get("min_examples", 8)),
                l2_regularization=float(params.get("l2_regularization", 2.0)),
                class_weight="balanced", random_state=seed)
            self.model = estimator.fit(Xt, y)   # same transform predict_matrix applies
            self.classes_ = list(self.model.classes_)
            return self
        if kind == "logreg":
            l1_ratio = float(params.get("l1_ratio", 0.0))
            kwargs = dict(C=float(params.get("C", 0.1)), class_weight="balanced",
                          max_iter=8000, random_state=seed,
                          solver="lbfgs" if l1_ratio == 0.0 else "saga")
            if _LOGREG_USES_L1_RATIO:
                kwargs["l1_ratio"] = l1_ratio
            elif l1_ratio > 0:
                kwargs["penalty"] = "elasticnet" if l1_ratio < 1 else "l1"
                if l1_ratio < 1:
                    kwargs["l1_ratio"] = l1_ratio
            estimator = LogisticRegression(**kwargs)
        else:
            minority = int(np.bincount(y, minlength=len(class_names)).min())
            estimator = CalibratedClassifierCV(
                SVC(C=float(params.get("C", 1.0)), kernel="rbf",
                    gamma=params.get("gamma", "scale"), class_weight="balanced",
                    random_state=seed),
                method="sigmoid", ensemble=False, cv=int(np.clip(minority, 2, 5)))
        self.model = estimator.fit(Xt, y)
        self.classes_ = list(self.model.classes_)
        return self

    def predict_matrix(self, X) -> np.ndarray:
        Xt = (impute(np.asarray(X, float), self.median_) - self.mean_) / self.std_
        proba = self.model.predict_proba(Xt)
        out = np.zeros((len(Xt), len(self.class_names)))
        for position, code in enumerate(self.classes_):
            out[:, int(code)] = proba[:, position]
        return out


class Model:
    """One learner + one training strategy; always scores P(MCI or Dem)."""

    def __init__(self, kind: str, params: dict, *, strategy: str = "binary", seed: int = 0) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        if kind not in ALL_KINDS:
            raise ValueError(f"Unknown learner kind: {kind}")
        self.kind = kind
        self.params = dict(params)
        self.strategy = strategy
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray, severity: np.ndarray) -> "Model":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        severity = np.asarray(severity, dtype=np.int64)

        strategy = self.strategy
        if strategy == "ordinal":
            target, rows, names = severity, np.arange(len(y)), _SEVERITY_NAMES
        elif strategy == "hard_boundary":
            rows = np.where(severity != 2)[0]
            if len(np.unique(y[rows])) < 2:
                strategy, rows = "binary", np.arange(len(y))
            target, names = y, ("CN", "IMPAIRED")
        else:
            target, rows, names = y, np.arange(len(y)), ("CN", "IMPAIRED")

        self.effective_strategy_ = strategy
        self.class_names_ = names
        if self.kind in YDF_KINDS and YDF_AVAILABLE:
            backend, backend_kind = _YDFBackend(), self.kind
        else:
            backend = _SkBackend()
            backend_kind = "hgb" if self.kind in YDF_KINDS else self.kind
        self.backend_ = backend.fit(X[rows], target[rows], backend_kind, self.params,
                                    self.seed, names)
        return self

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """P(impaired): P(class=1) for binary strategies, 1 - P(CN) for ordinal."""

        matrix = self.backend_.predict_matrix(np.asarray(X, dtype=np.float64))
        if self.effective_strategy_ == "ordinal":
            return 1.0 - matrix[:, 0]
        return matrix[:, 1]


def make_model(kind: str, params: dict, *, strategy: str = "binary", seed: int = 0) -> Model:
    return Model(kind, params, strategy=strategy, seed=seed)


__all__ = ["ALL_KINDS", "Model", "SK_KINDS", "STRATEGIES", "YDF_AVAILABLE", "YDF_KINDS",
           "make_model"]
