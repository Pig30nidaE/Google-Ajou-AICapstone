"""Model adapters: Google TabFM (core), fold-local LR pipeline, rank blend.

TabFM notes
-----------
TabFM is Google Research's pretrained tabular foundation model (July 2026):
an in-context learner where ``fit`` stores the training table as context and
``predict_proba`` is a forward pass -- no dataset-specific training.  The
checkpoint is loaded ONCE per process (``set_shared_model``) and shared by
every ``TabFMClassifier`` instance, which only holds context, not weights.

Two defensive rules, because the wrapped API is young:

1. **Signature-filtered kwargs.**  We *want* ``max_num_rows=256`` (the
   documented default of 100 context rows would silently subsample our
   113-141 training subjects) and ``random_state``.  Only kwargs the installed
   ``TabFMClassifier`` signature accepts are passed; accepted and dropped ones
   are recorded for the report.
2. **Explicit positive-class lookup.**  Probability columns follow
   ``clf.classes_`` (sklearn convention); we never assume column 1 blindly.

Fold locality: the adapter's median imputation (TabFM's NaN handling is not
documented, so we do not rely on it) is fitted on the fold-training rows only,
as are the LR pipeline statistics and the blend ECDFs.

No fallback: if TabFM cannot be imported/loaded, the run fails.  The only
exception is the wiring stub, which activates ONLY when BGTF_WIRING_STUB=1
AND profile=smoke, is labelled NOT-GOOGLE in every artifact, and exists so
machines that cannot install TabFM can still check the pipeline's plumbing.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import LR_PARAMS, TABFM_DESIRED_KWARGS, TABFM_PIP_SPEC, Candidate


class TabFMContractError(RuntimeError):
    """Raised when the installed TabFM runtime cannot honor the experiment."""


# ------------------------------------------------------- shared model state --
_SHARED_MODEL: Any = None
_SHARED_MODEL_INFO: dict[str, Any] = {}
_STUB_ACTIVE = False


def require_tabfm() -> Any:
    if importlib.util.find_spec("tabfm") is None:
        raise ModuleNotFoundError(
            "Google TabFM is required and no fallback is permitted. "
            f"Install '{TABFM_PIP_SPEC}'."
        )
    module = importlib.import_module("tabfm")
    if not hasattr(module, "TabFMClassifier"):
        raise TabFMContractError("Installed tabfm lacks TabFMClassifier")
    return module


def load_shared_model(backend_order: tuple[str, ...] = ("pytorch", "jax")) -> dict[str, Any]:
    """Load the TabFM checkpoint once; later fits reuse it.

    Tries the PyTorch loader module first (torch ships on every Colab image),
    then JAX.  Records which backend actually loaded.
    """

    global _SHARED_MODEL, _SHARED_MODEL_INFO
    if _SHARED_MODEL is not None:
        return _SHARED_MODEL_INFO
    module = require_tabfm()
    errors: list[str] = []
    for backend in backend_order:
        loader_name = f"tabfm_v1_0_0_{backend}"
        loader = getattr(module, loader_name, None)
        if loader is None:
            try:
                loader = importlib.import_module(f"tabfm.{loader_name}")
            except ImportError as error:
                errors.append(f"{loader_name}: {error}")
                continue
        try:
            _SHARED_MODEL = loader.load()
        except Exception as error:  # noqa: BLE001 - report and try next backend
            errors.append(f"{loader_name}.load(): {type(error).__name__}: {error}")
            continue
        try:
            version = importlib.metadata.version("tabfm")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            version = "unknown"
        _SHARED_MODEL_INFO = {
            "engine": "google_tabfm",
            "distribution": "tabfm",
            "version": version,
            "backend_module": loader_name,
            "fallback_permitted": False,
        }
        return _SHARED_MODEL_INFO
    raise TabFMContractError(
        "No TabFM backend could load the checkpoint. Tried: " + " | ".join(errors)
    )


def activate_wiring_stub() -> dict[str, Any]:
    """Deterministic NOT-GOOGLE stand-in for local smoke wiring checks only."""

    global _SHARED_MODEL, _SHARED_MODEL_INFO, _STUB_ACTIVE
    _STUB_ACTIVE = True
    _SHARED_MODEL = "__wiring_stub__"
    _SHARED_MODEL_INFO = {
        "engine": "wiring_stub_NOT_GOOGLE",
        "distribution": "none",
        "version": "stub",
        "backend_module": "stub",
        "fallback_permitted": False,
        "warning": "smoke wiring check only; results must never be reported",
    }
    return _SHARED_MODEL_INFO


def shared_model_info() -> dict[str, Any]:
    if not _SHARED_MODEL_INFO:
        raise TabFMContractError("TabFM model has not been loaded yet")
    return dict(_SHARED_MODEL_INFO)


def _filtered_tabfm_kwargs(classifier_class: Any, seed: int) -> tuple[dict, dict]:
    """Split desired kwargs into (accepted-by-signature, dropped)."""

    desired = dict(TABFM_DESIRED_KWARGS)
    desired["random_state"] = int(seed)
    try:
        parameters = inspect.signature(classifier_class).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-level signature
        return desired, {}
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    if accepts_var_kwargs:
        return desired, {}
    accepted = {k: v for k, v in desired.items() if k in parameters}
    dropped = {k: v for k, v in desired.items() if k not in parameters}
    return accepted, dropped


class _StubClassifier:
    """Nearest-centroid scorer for BGTF_WIRING_STUB=1 smoke runs (NOT GOOGLE)."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_StubClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.array([0, 1])
        scale = np.nanstd(X, axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        self._scale = scale
        self._centroids = {
            label: np.nanmean(X[y == label] / scale, axis=0) for label in (0, 1)
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64) / self._scale
        d0 = np.linalg.norm(X - self._centroids[0], axis=1)
        d1 = np.linalg.norm(X - self._centroids[1], axis=1)
        p1 = 1.0 / (1.0 + np.exp(d1 - d0))
        return np.column_stack([1.0 - p1, p1])


class TabFMModel:
    """Adapter: fold-local median imputation + one TabFM in-context fit."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def _make_classifier(self) -> Any:
        if _SHARED_MODEL is None:
            raise TabFMContractError(
                "call load_shared_model() (or activate_wiring_stub()) first"
            )
        if _STUB_ACTIVE:
            self.kwargs_accepted_, self.kwargs_dropped_ = {}, dict(TABFM_DESIRED_KWARGS)
            return _StubClassifier()
        module = require_tabfm()
        classifier_class = module.TabFMClassifier
        accepted, dropped = _filtered_tabfm_kwargs(classifier_class, self.seed)
        self.kwargs_accepted_, self.kwargs_dropped_ = accepted, dropped
        try:
            return classifier_class(model=_SHARED_MODEL, **accepted)
        except TypeError:
            # Constructor rejected something the signature appeared to allow;
            # retry with the model alone and record that everything dropped.
            self.kwargs_dropped_ = {**accepted, **dropped}
            self.kwargs_accepted_ = {}
            return classifier_class(model=_SHARED_MODEL)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabFMModel":
        self.feature_names_ = tuple(map(str, X.columns))
        values = X.to_numpy(np.float64)
        self.medians_ = np.nanmedian(values, axis=0)
        self.medians_[~np.isfinite(self.medians_)] = 0.0
        values = np.where(np.isfinite(values), values, self.medians_)
        self.classifier_ = self._make_classifier()
        self.classifier_.fit(values, np.asarray(y, dtype=int))
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        if tuple(map(str, X.columns)) != self.feature_names_:
            raise ValueError("Prediction schema differs from the fitted schema")
        values = X.to_numpy(np.float64)
        values = np.where(np.isfinite(values), values, self.medians_)
        probabilities = np.asarray(self.classifier_.predict_proba(values), dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[0] != len(X):
            raise TabFMContractError(
                f"Unexpected predict_proba shape: {probabilities.shape}"
            )
        classes = getattr(self.classifier_, "classes_", None)
        if classes is not None and len(classes) == probabilities.shape[1]:
            matches = np.flatnonzero(np.asarray(classes) == 1)
            if matches.size != 1:
                raise TabFMContractError(f"Positive class not found in {classes!r}")
            column = int(matches[0])
            self.positive_column_source_ = "classes_"
        else:
            column = probabilities.shape[1] - 1
            self.positive_column_source_ = "assumed_last_column"
        score = probabilities[:, column]
        if not np.isfinite(score).all():
            raise TabFMContractError("TabFM emitted non-finite probabilities")
        return score


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
    """Median impute -> winsorize -> standardize -> L2 logistic regression.

    Byte-for-byte the anchor pipeline of Binary_Google_CircadianNested, so the
    anchor track is comparable across the two runs.
    """

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


# -------------------------------------------------------------------- blend --
class _TrainECDF:
    """Monotone map through the ECDF of a model's fold-training scores
    (no transductive normalization against the held-out batch)."""

    def __init__(self, train_scores: np.ndarray) -> None:
        self.sorted_ = np.sort(np.asarray(train_scores, dtype=np.float64))

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        n = max(1, self.sorted_.size)
        return np.searchsorted(self.sorted_, scores, side="right") / n


class RankBlendModel:
    """Equal-weight blend of the LR and TabFM score ranks."""

    def __init__(self, C: float, seed: int) -> None:  # noqa: N803
        self.lr = LogisticModel(C=C, seed=seed)
        self.tabfm = TabFMModel(seed=seed)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "RankBlendModel":
        self.lr.fit(X, y)
        self.tabfm.fit(X, y)
        self.ecdf_lr_ = _TrainECDF(self.lr.predict_score(X))
        self.ecdf_tabfm_ = _TrainECDF(self.tabfm.predict_score(X))
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return 0.5 * (
            self.ecdf_lr_(self.lr.predict_score(X))
            + self.ecdf_tabfm_(self.tabfm.predict_score(X))
        )


# ------------------------------------------------------------------ factory --
def build_model(candidate: Candidate, seed: int):
    params = dict(candidate.params)
    if candidate.learner == "lr":
        return LogisticModel(C=params["C"], seed=seed)
    if candidate.learner == "tabfm":
        return TabFMModel(seed=seed)
    if candidate.learner == "blend_tabfm_lr":
        return RankBlendModel(C=params["C"], seed=seed)
    raise ValueError(f"Unknown learner: {candidate.learner!r}")


__all__ = [
    "LogisticModel", "RankBlendModel", "TabFMContractError", "TabFMModel",
    "Winsorizer", "activate_wiring_stub", "build_model", "load_shared_model",
    "require_tabfm", "shared_model_info",
]
