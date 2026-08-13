"""Model adapters: fold-local sklearn LR pipeline, Google YDF GBTs, rank blend.

Google YDF is the core engine (axis-aligned and sparse-oblique gradient boosted
trees).  There is deliberately NO fallback: if ``ydf`` is missing or cannot
honor the exact sparse-oblique learner, the run fails instead of silently
substituting another library (a fallback run would misreport the Google engine
as used).

Every fitted statistic -- imputation medians, winsorization quantiles,
standardization moments, tree structure, blend ECDFs -- is learned inside
``fit`` on the fold-training subjects only.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES, LR_PARAMS, REQUIRED_YDF_VERSION, YDF_NUM_THREADS, Candidate


class YDFContractError(RuntimeError):
    """Raised when the installed YDF runtime cannot honor the exact model."""


def require_ydf() -> Any:
    if importlib.util.find_spec("ydf") is None:
        raise ModuleNotFoundError(
            "Google YDF is required and no fallback is permitted. "
            f"Install ydf=={REQUIRED_YDF_VERSION} (see requirements_colab.txt)."
        )
    module = importlib.import_module("ydf")
    for attribute in ("GradientBoostedTreesLearner", "load_model"):
        if not hasattr(module, attribute):
            raise YDFContractError(f"Installed ydf lacks required API: {attribute}")
    return module


def ydf_runtime_info() -> dict[str, Any]:
    require_ydf()
    try:
        version = importlib.metadata.version("ydf")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        version = "unknown"
    return {"engine": "google_ydf", "distribution": "ydf", "version": version,
            "fallback_permitted": False}


def balanced_class_weights(y: np.ndarray) -> dict[str, float]:
    target = np.asarray(y, dtype=np.int64)
    counts = np.bincount(target, minlength=2)
    if len(target) == 0 or np.any(counts == 0):
        raise YDFContractError("Both classes are required to fit a model")
    return {
        CLASS_NAMES[index]: float(len(target) / (2.0 * counts[index]))
        for index in range(2)
    }


# ------------------------------------------------------------------ sklearn --
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column at its fold-training 1st/99th percentile."""

    def __init__(self, lower: float = 1.0, upper: float = 99.0) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):  # noqa: N803 - sklearn signature
        values = np.asarray(X, dtype=np.float64)
        self.lower_bounds_ = np.nanpercentile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanpercentile(values, self.upper, axis=0)
        return self

    def transform(self, X):  # noqa: N803
        values = np.asarray(X, dtype=np.float64)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


class LogisticModel:
    """Median impute -> winsorize -> standardize -> L2 logistic regression."""

    def __init__(self, C: float, seed: int) -> None:  # noqa: N803
        self.C = float(C)
        self.seed = int(seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticModel":
        self.feature_names_ = tuple(map(str, X.columns))
        self.pipeline_ = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("winsorize", Winsorizer()),
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(C=self.C, random_state=self.seed, **LR_PARAMS)),
            ]
        )
        self.pipeline_.fit(X.to_numpy(np.float64), np.asarray(y, dtype=int))
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        if tuple(map(str, X.columns)) != self.feature_names_:
            raise ValueError("Prediction schema differs from the fitted schema")
        return self.pipeline_.predict_proba(X.to_numpy(np.float64))[:, 1]


# ---------------------------------------------------------------- Google YDF -
class YDFModel:
    """Adapter around one exact Google YDF gradient-boosted-trees learner."""

    def __init__(self, family: str, params: dict, seed: int) -> None:
        if family not in ("ydf_gbt", "ydf_oblique"):
            raise ValueError(f"Unknown YDF family: {family!r}")
        self.family = family
        self.params = dict(params)
        self.seed = int(seed)

    def _learner(self, y: np.ndarray) -> Any:
        ydf = require_ydf()
        params = self.params
        kwargs: dict[str, Any] = {
            "label": "label",
            "label_classes": list(CLASS_NAMES),
            "class_weights": balanced_class_weights(y),
            "random_seed": self.seed,
            "num_threads": YDF_NUM_THREADS,
            "loss": "BINOMIAL_LOG_LIKELIHOOD",
            "validation_ratio": 0.0,
            "num_trees": int(params["num_trees"]),
            "max_depth": int(params["max_depth"]),
            "min_examples": int(params["min_examples"]),
            "shrinkage": float(params["shrinkage"]),
            "subsample": float(params["subsample"]),
            "num_candidate_attributes_ratio": float(params["num_candidate_attributes_ratio"]),
            "l2_regularization": float(params["l2_regularization"]),
        }
        if self.family == "ydf_oblique":
            kwargs.update(
                {
                    "split_axis": "SPARSE_OBLIQUE",
                    "sparse_oblique_normalization": str(params["sparse_oblique_normalization"]),
                    "sparse_oblique_num_projections_exponent": float(
                        params["sparse_oblique_num_projections_exponent"]
                    ),
                    "sparse_oblique_projection_density_factor": float(
                        params["sparse_oblique_projection_density_factor"]
                    ),
                }
            )
        try:
            return ydf.GradientBoostedTreesLearner(**kwargs)
        except Exception as error:
            raise YDFContractError(
                f"YDF rejected the exact {self.family} learner; "
                "downgrade/fallback is forbidden"
            ) from error

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "YDFModel":
        target = np.asarray(y, dtype=np.int64)
        self.feature_names_ = tuple(map(str, X.columns))
        self.columns_ = tuple(f"f_{i:04d}" for i in range(len(self.feature_names_)))
        frame = pd.DataFrame(
            X.to_numpy(np.float32), columns=list(self.columns_)
        )
        frame["label"] = [CLASS_NAMES[int(v)] for v in target]
        self.model_ = self._learner(target).train(frame, verbose=0)
        classes = tuple(str(v) for v in self.model_.label_classes())
        if set(classes) != set(CLASS_NAMES):
            raise YDFContractError(f"Unexpected YDF label classes: {classes}")
        self.positive_index_ = classes.index("MCI_DEM")
        self.model_classes_ = classes
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        if tuple(map(str, X.columns)) != self.feature_names_:
            raise ValueError("Prediction schema differs from the fitted schema")
        frame = pd.DataFrame(X.to_numpy(np.float32), columns=list(self.columns_))
        raw = np.asarray(self.model_.predict(frame), dtype=np.float64)
        if raw.ndim == 1:
            score = raw if self.positive_index_ == 1 else 1.0 - raw
        elif raw.ndim == 2 and raw.shape[1] == len(self.model_classes_):
            score = raw[:, self.positive_index_]
        else:
            raise YDFContractError(f"Unexpected YDF prediction shape: {raw.shape}")
        score = np.asarray(score, dtype=np.float64)
        if score.shape != (len(X),) or not np.isfinite(score).all():
            raise YDFContractError("YDF emitted invalid binary scores")
        return score


# -------------------------------------------------------------------- blend --
class _TrainECDF:
    """Monotone map through the ECDF of a model's fold-training scores.

    Normalizing each model's scores against its own training-fold distribution
    puts differently scaled scores on a common [0, 1] rank scale WITHOUT
    looking at the held-out batch (no transductive normalization).
    """

    def __init__(self, train_scores: np.ndarray) -> None:
        self.sorted_ = np.sort(np.asarray(train_scores, dtype=np.float64))

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        n = max(1, self.sorted_.size)
        return np.searchsorted(self.sorted_, scores, side="right") / n


class RankBlendModel:
    """Equal-weight blend of the LR and sparse-oblique YDF score ranks."""

    def __init__(self, C: float, oblique_params: dict, seed: int) -> None:  # noqa: N803
        self.lr = LogisticModel(C=C, seed=seed)
        self.ydf = YDFModel("ydf_oblique", oblique_params, seed=seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RankBlendModel":
        self.lr.fit(X, y)
        self.ydf.fit(X, y)
        self.ecdf_lr_ = _TrainECDF(self.lr.predict_score(X))
        self.ecdf_ydf_ = _TrainECDF(self.ydf.predict_score(X))
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return 0.5 * (
            self.ecdf_lr_(self.lr.predict_score(X))
            + self.ecdf_ydf_(self.ydf.predict_score(X))
        )


# ------------------------------------------------------------------ factory --
def build_model(candidate: Candidate, seed: int, *, ydf_trees_override: int | None = None):
    params = dict(candidate.params)
    if candidate.learner == "lr":
        return LogisticModel(C=params["C"], seed=seed)
    if candidate.learner in ("ydf_gbt", "ydf_oblique"):
        if ydf_trees_override is not None:
            params["num_trees"] = int(ydf_trees_override)
        return YDFModel(candidate.learner, params, seed=seed)
    if candidate.learner == "blend_lr_oblique":
        oblique = dict(params["oblique"])
        if ydf_trees_override is not None:
            oblique["num_trees"] = int(ydf_trees_override)
        return RankBlendModel(C=params["C"], oblique_params=oblique, seed=seed)
    raise ValueError(f"Unknown learner: {candidate.learner!r}")


__all__ = [
    "LogisticModel", "RankBlendModel", "Winsorizer", "YDFContractError",
    "YDFModel", "balanced_class_weights", "build_model", "require_ydf",
    "ydf_runtime_info",
]
