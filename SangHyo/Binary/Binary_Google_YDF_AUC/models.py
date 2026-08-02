"""Strict Google YDF-only binary model adapters.

There is intentionally no fallback estimator.  Sparse-oblique candidates pass
their exact requested split axis to YDF; an unsupported option is an error, not
an invitation to retry with an axis-aligned learner.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

CLASS_NAMES = ("CN", "MCI_DEM")
YDF_FAMILIES = ("axis_gbt", "sparse_oblique_gbt", "rf")
ENGINE_NAME = "google_ydf"


class YDFContractError(RuntimeError):
    """Raised when the installed YDF runtime cannot honor the exact model."""


class ObliqueContractError(YDFContractError):
    """Raised instead of silently downgrading a sparse-oblique candidate."""


def require_ydf() -> Any:
    """Import Google YDF or fail closed with an actionable error."""

    if importlib.util.find_spec("ydf") is None:
        raise ModuleNotFoundError(
            "Google YDF is required. No sklearn or other fallback is permitted. "
            "Install requirements_colab.in (ydf==0.16.1)."
        )
    module = importlib.import_module("ydf")
    for attribute in (
        "GradientBoostedTreesLearner",
        "RandomForestLearner",
        "load_model",
    ):
        if not hasattr(module, attribute):
            raise YDFContractError(f"Installed ydf lacks required API: {attribute}")
    return module


def ydf_runtime_info() -> dict[str, Any]:
    module = require_ydf()
    try:
        version = importlib.metadata.version("ydf")
    except importlib.metadata.PackageNotFoundError:
        version = str(getattr(module, "__version__", "unknown"))
    return {
        "engine": ENGINE_NAME,
        "distribution": "ydf",
        "version": version,
        "fallback_permitted": False,
    }


def balanced_class_weights(y: np.ndarray) -> dict[str, float]:
    target = np.asarray(y, dtype=np.int64)
    counts = np.bincount(target, minlength=2)
    if len(target) == 0 or np.any(counts == 0):
        raise YDFContractError("Both classes are required to fit a YDF model")
    return {
        CLASS_NAMES[index]: float(len(target) / (2.0 * counts[index]))
        for index in range(2)
    }


def _assert_oblique_constructor_support(learner_class: Any) -> None:
    """Fail before training when the pinned sparse-oblique API is unavailable."""

    try:
        signature = inspect.signature(learner_class)
    except (TypeError, ValueError):
        return
    parameters = signature.parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    required = {
        "split_axis",
        "sparse_oblique_normalization",
        "sparse_oblique_num_projections_exponent",
        "sparse_oblique_projection_density_factor",
    }
    missing = sorted(
        name for name in required if name not in parameters and not accepts_kwargs
    )
    if missing:
        raise ObliqueContractError(
            "YDF sparse-oblique constructor contract unavailable: "
            + ", ".join(missing)
        )


def _oblique_runtime_evidence(learner: Any) -> dict[str, Any]:
    """Inspect an exposed learner config when available.

    Accepting the exact constructor keyword is the minimum contract.  If the
    runtime exposes a hyperparameter mapping, it must also report the requested
    value; otherwise training is stopped.
    """

    evidence: dict[str, Any] = {
        "requested_split_axis": "SPARSE_OBLIQUE",
        "constructor_accepted_exact_kwargs": True,
        "runtime_mapping_checked": False,
    }
    for attribute in ("hyperparameters", "_hyperparameters"):
        value = getattr(learner, attribute, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if not isinstance(value, Mapping) or "split_axis" not in value:
            continue
        observed = str(value["split_axis"]).upper()
        evidence.update(
            {
                "runtime_mapping_checked": True,
                "runtime_split_axis": observed,
                "runtime_mapping_attribute": attribute,
            }
        )
        if "SPARSE_OBLIQUE" not in observed:
            raise ObliqueContractError(
                f"YDF downgraded sparse-oblique split_axis to {observed!r}"
            )
        break
    return evidence


class YDFBinaryModel:
    """Numpy/DataFrame adapter around one exact Google YDF learner."""

    def __init__(
        self,
        family: str,
        params: Mapping[str, Any],
        *,
        seed: int,
        num_threads: int,
    ) -> None:
        if family not in YDF_FAMILIES:
            raise ValueError(f"family must be one of {YDF_FAMILIES}; got {family!r}")
        self.family = str(family)
        self.params = dict(params)
        self.seed = int(seed)
        self.num_threads = max(1, int(num_threads))
        self.engine = ENGINE_NAME

    def _build_learner(self, y: np.ndarray) -> Any:
        ydf = require_ydf()
        common: dict[str, Any] = {
            "label": "label",
            "label_classes": list(CLASS_NAMES),
            "class_weights": balanced_class_weights(y),
            "random_seed": self.seed,
            "num_threads": self.num_threads,
        }
        params = self.params
        if self.family in {"axis_gbt", "sparse_oblique_gbt"}:
            learner_class = ydf.GradientBoostedTreesLearner
            learner_kwargs: dict[str, Any] = {
                "loss": "BINOMIAL_LOG_LIKELIHOOD",
                "validation_ratio": 0.0,
                "num_trees": int(params["num_trees"]),
                "max_depth": int(params["max_depth"]),
                "min_examples": int(params["min_examples"]),
                "shrinkage": float(params["shrinkage"]),
                "subsample": float(params["subsample"]),
                "num_candidate_attributes_ratio": float(
                    params["num_candidate_attributes_ratio"]
                ),
                "l2_regularization": float(params["l2_regularization"]),
            }
            if self.family == "sparse_oblique_gbt":
                _assert_oblique_constructor_support(learner_class)
                learner_kwargs.update(
                    {
                        "split_axis": "SPARSE_OBLIQUE",
                        "sparse_oblique_normalization": str(
                            params["sparse_oblique_normalization"]
                        ),
                        "sparse_oblique_num_projections_exponent": float(
                            params[
                                "sparse_oblique_num_projections_exponent"
                            ]
                        ),
                        "sparse_oblique_projection_density_factor": float(
                            params[
                                "sparse_oblique_projection_density_factor"
                            ]
                        ),
                    }
                )
        else:
            learner_class = ydf.RandomForestLearner
            learner_kwargs = {
                "num_trees": int(params["num_trees"]),
                "max_depth": int(params["max_depth"]),
                "min_examples": int(params["min_examples"]),
                "num_candidate_attributes_ratio": float(
                    params["num_candidate_attributes_ratio"]
                ),
            }
        # Do not catch and retry.  In particular, sparse-oblique kwargs are
        # never removed to manufacture an axis-aligned substitute.
        try:
            learner = learner_class(**learner_kwargs, **common)
        except Exception as error:
            if self.family == "sparse_oblique_gbt":
                raise ObliqueContractError(
                    "YDF rejected the exact sparse-oblique learner; downgrade "
                    "is forbidden"
                ) from error
            raise
        self.oblique_evidence_ = (
            _oblique_runtime_evidence(learner)
            if self.family == "sparse_oblique_gbt"
            else {
                "requested_split_axis": "AXIS_ALIGNED",
                "constructor_accepted_exact_kwargs": True,
            }
        )
        return learner

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> "YDFBinaryModel":
        values = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.int64)
        names = tuple(map(str, feature_names))
        if values.ndim != 2 or values.shape != (len(target), len(names)):
            raise ValueError("YDF fit matrix/schema mismatch")
        if np.isinf(values).any():
            raise ValueError("YDF fit matrix contains infinity")
        if len(names) == 0 or len(names) != len(set(names)):
            raise ValueError("YDF feature names must be non-empty and unique")
        self.feature_names_ = names
        self.columns_ = tuple(
            f"f_{index:04d}" for index in range(values.shape[1])
        )
        frame = pd.DataFrame(values, columns=list(self.columns_))
        frame["label"] = [CLASS_NAMES[int(value)] for value in target]
        learner = self._build_learner(target)
        self.model_ = learner.train(frame, verbose=0)
        classes = tuple(str(value) for value in self.model_.label_classes())
        if set(classes) != set(CLASS_NAMES):
            raise YDFContractError(f"Unexpected YDF label classes: {classes}")
        self.positive_index_ = classes.index("MCI_DEM")
        self.model_classes_ = classes
        return self

    def _frame(self, X: np.ndarray) -> pd.DataFrame:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(self.columns_):
            raise ValueError(
                f"YDF input width {values.shape} differs from fitted "
                f"width {len(self.columns_)}"
            )
        if np.isinf(values).any():
            raise ValueError("YDF prediction matrix contains infinity")
        return pd.DataFrame(values, columns=list(self.columns_))

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model_.predict(self._frame(X)), dtype=np.float64)
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

    def manifest(self) -> dict[str, Any]:
        return {
            "engine": ENGINE_NAME,
            "family": self.family,
            "params": dict(self.params),
            "seed": self.seed,
            "num_threads": self.num_threads,
            "feature_names": list(self.feature_names_),
            "feature_count": len(self.feature_names_),
            "model_classes": list(self.model_classes_),
            "positive_class": "MCI_DEM",
            "positive_index": self.positive_index_,
            "oblique_contract": dict(self.oblique_evidence_),
            "fallback_permitted": False,
            "ydf": ydf_runtime_info(),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
        destination.mkdir(parents=True)
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "model"
            self.model_.save(str(local))
            shutil.copytree(local, destination / "model")
        (destination / "MODEL_META.json").write_text(
            json.dumps(self.manifest(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "YDFBinaryModel":
        root = Path(path)
        metadata = json.loads(
            (root / "MODEL_META.json").read_text(encoding="utf-8")
        )
        if metadata.get("engine") != ENGINE_NAME:
            raise YDFContractError("Checkpoint engine is not Google YDF")
        if metadata.get("fallback_permitted") is not False:
            raise YDFContractError("Checkpoint fallback contract changed")
        if (
            metadata["family"] == "sparse_oblique_gbt"
            and (
                metadata["oblique_contract"]["requested_split_axis"]
                != "SPARSE_OBLIQUE"
                or metadata["oblique_contract"].get(
                    "constructor_accepted_exact_kwargs"
                )
                is not True
                or metadata["oblique_contract"].get(
                    "runtime_mapping_checked"
                )
                is not True
                or "SPARSE_OBLIQUE"
                not in str(
                    metadata["oblique_contract"].get(
                        "runtime_split_axis", ""
                    )
                ).upper()
            )
        ):
            raise ObliqueContractError(
                "Stored oblique runtime evidence is not exact"
            )
        instance = cls(
            metadata["family"],
            metadata["params"],
            seed=int(metadata["seed"]),
            num_threads=int(metadata["num_threads"]),
        )
        instance.feature_names_ = tuple(metadata["feature_names"])
        instance.columns_ = tuple(
            f"f_{index:04d}" for index in range(len(instance.feature_names_))
        )
        instance.model_classes_ = tuple(metadata["model_classes"])
        instance.positive_index_ = int(metadata["positive_index"])
        instance.oblique_evidence_ = dict(metadata["oblique_contract"])
        ydf = require_ydf()
        instance.model_ = ydf.load_model(str(root / "model"))
        observed_classes = tuple(
            str(value) for value in instance.model_.label_classes()
        )
        if observed_classes != instance.model_classes_:
            raise YDFContractError("Reloaded YDF class order changed")
        return instance


__all__ = [
    "CLASS_NAMES",
    "ENGINE_NAME",
    "ObliqueContractError",
    "YDFBinaryModel",
    "YDFContractError",
    "YDF_FAMILIES",
    "balanced_class_weights",
    "require_ydf",
    "ydf_runtime_info",
]
