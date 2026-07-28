"""Fold-local model components used by the nested ROC-AUC experiment.

No class in this module is allowed to receive an outer-test label.  Population
statistics, univariate screening, the MaxAUC quality gate, neural epoch
selection, and empirical score normalization are all fitted from the supplied
training partition only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from SangHyo.Binary_Wearable_SequenceFusion_Google.data import make_fixed_views
from SangHyo.Binary_Wearable_SequenceFusion_Google.models import (
    NeuralConfig,
    NeuralSequenceModel,
    _build_torch_model,
    aggregate_view_probabilities,
    fit_neural_fixed_epochs,
    select_neural_epoch,
)
from SangHyo.Binary_Wearable_SequenceFusion_Google.preprocessing import (
    SequencePreprocessor,
)

MODEL_KINDS = ("ridge", "elastic", "rbf_svm", "catboost", "tabpfn")
_TABPFN_CHECKPOINT_DIGEST_CACHE: dict[str, str] = {}


def _as_float_matrix(X: np.ndarray, role: str) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError(f"{role} must be a non-empty 2-D matrix")
    values = values.copy()
    values[~np.isfinite(values)] = np.nan
    return values


def _as_binary(y: np.ndarray) -> np.ndarray:
    target = np.asarray(y, dtype=np.int64)
    if target.ndim != 1 or set(np.unique(target)) != {0, 1}:
        raise ValueError("Both binary classes are required")
    return target


def continuous_score(model: Any, X: np.ndarray) -> np.ndarray:
    """Extract the positive-class continuous score without thresholding."""

    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(X), dtype=np.float64)
        score = raw[:, 1] if raw.ndim == 2 else raw
    elif hasattr(model, "decision_function"):
        score = np.asarray(model.decision_function(X), dtype=np.float64)
    else:
        raise TypeError("Estimator must expose predict_proba or decision_function")
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if not np.isfinite(score).all():
        raise FloatingPointError("Estimator returned a non-finite score")
    return score


def _cached_file_sha256(path: Path) -> str:
    resolved = str(path.resolve())
    cached = _TABPFN_CHECKPOINT_DIGEST_CACHE.get(resolved)
    if cached is not None:
        return cached
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    _TABPFN_CHECKPOINT_DIGEST_CACHE[resolved] = digest
    return digest


@dataclass
class FoldLocalTablePreprocessor:
    """Median → winsorization → robust scaling fitted on one train fold."""

    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    epsilon: float = 1e-8
    fit_scope: str = "current CV training partition only"

    def fit(
        self, X: np.ndarray, feature_names: Sequence[str]
    ) -> "FoldLocalTablePreprocessor":
        values = _as_float_matrix(X, "X")
        names = tuple(map(str, feature_names))
        if values.shape[1] != len(names):
            raise ValueError("Feature names do not match X")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ValueError("Invalid winsorization quantiles")

        keep = ~np.isnan(values).all(axis=0)
        if not np.any(keep):
            raise ValueError("All candidate features are entirely missing")
        self.input_feature_names_ = names
        self.nonempty_indices_ = np.flatnonzero(keep)
        selected = values[:, keep]
        self.medians_ = np.nanmedian(selected, axis=0)
        filled = np.where(np.isnan(selected), self.medians_[None, :], selected)
        self.lower_ = np.quantile(filled, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(filled, self.upper_quantile, axis=0)
        clipped = np.clip(filled, self.lower_, self.upper_)
        centers = np.median(clipped, axis=0)
        q25, q75 = np.quantile(clipped, [0.25, 0.75], axis=0)
        scales = q75 - q25
        fallback = np.std(clipped, axis=0)
        bad_scale = ~np.isfinite(scales) | (np.abs(scales) < self.epsilon)
        scales[bad_scale] = fallback[bad_scale]
        variable = np.isfinite(scales) & (np.abs(scales) >= self.epsilon)
        if not np.any(variable):
            raise ValueError("All candidate features are constant in this fold")
        self.variable_indices_ = np.flatnonzero(variable)
        self.centers_ = centers[variable]
        self.scales_ = scales[variable]
        self.medians_ = self.medians_[variable]
        self.lower_ = self.lower_[variable]
        self.upper_ = self.upper_[variable]
        self.output_feature_names_ = tuple(
            names[self.nonempty_indices_[index]] for index in self.variable_indices_
        )
        self.nonempty_indices_ = self.nonempty_indices_[self.variable_indices_]
        return self

    def _check(self) -> None:
        if not hasattr(self, "output_feature_names_"):
            raise RuntimeError("Preprocessor is not fitted")

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check()
        values = _as_float_matrix(X, "X")
        if values.shape[1] != len(self.input_feature_names_):
            raise ValueError("Input schema differs from fitted schema")
        selected = values[:, self.nonempty_indices_]
        filled = np.where(np.isnan(selected), self.medians_[None, :], selected)
        clipped = np.clip(filled, self.lower_, self.upper_)
        transformed = (clipped - self.centers_) / self.scales_
        if not np.isfinite(transformed).all():
            raise FloatingPointError("Non-finite value survived table preprocessing")
        return transformed.astype(np.float32)

    def fit_transform(
        self, X: np.ndarray, feature_names: Sequence[str]
    ) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def manifest(self) -> dict[str, Any]:
        self._check()
        return {
            "fit_scope": self.fit_scope,
            "labels_consumed": False,
            "imputation": "median",
            "missingness_indicator_exported": False,
            "winsorization": [self.lower_quantile, self.upper_quantile],
            "scaling": "median/IQR; SD fallback; constants removed",
            "input_features": list(self.input_feature_names_),
            "output_features": list(self.output_feature_names_),
        }


@dataclass
class FoldLocalTabPFNProjector:
    """Drop only unusable columns; preserve raw values and NaNs for TabPFN."""

    epsilon: float = 1e-8
    fit_scope: str = "current CV training partition only"

    def fit(
        self, X: np.ndarray, feature_names: Sequence[str]
    ) -> "FoldLocalTabPFNProjector":
        values = _as_float_matrix(X, "X")
        names = tuple(map(str, feature_names))
        if values.shape[1] != len(names):
            raise ValueError("Feature names do not match X")
        nonempty = ~np.isnan(values).all(axis=0)
        if not np.any(nonempty):
            raise ValueError("All TabPFN candidate features are entirely missing")
        reduced = values[:, nonempty]
        medians = np.nanmedian(reduced, axis=0)
        shadow = np.where(np.isnan(reduced), medians[None, :], reduced)
        variable = np.std(shadow, axis=0) >= float(self.epsilon)
        if not np.any(variable):
            raise ValueError("All TabPFN candidate features are constant")
        self.input_feature_names_ = names
        self.output_indices_ = np.flatnonzero(nonempty)[variable]
        self.output_feature_names_ = tuple(
            names[index] for index in self.output_indices_
        )
        self.selector_medians_ = medians[variable]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "output_indices_"):
            raise RuntimeError("TabPFN projector is not fitted")
        values = _as_float_matrix(X, "X")
        if values.shape[1] != len(self.input_feature_names_):
            raise ValueError("Input schema differs from fitted TabPFN schema")
        return values[:, self.output_indices_].astype(np.float32)

    def fit_transform(
        self, X: np.ndarray, feature_names: Sequence[str]
    ) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    def selector_shadow(self, projected: np.ndarray) -> np.ndarray:
        values = _as_float_matrix(projected, "projected TabPFN X")
        return np.where(
            np.isnan(values), self.selector_medians_[None, :], values
        ).astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        return {
            "fit_scope": self.fit_scope,
            "labels_consumed": False,
            "imputation_for_model": "none; native TabPFN NaN handling",
            "scaling": "none, following official TabPFN guidance",
            "column_operations": "all-NaN and constant removal only",
            "input_features": list(self.input_feature_names_),
            "output_features": list(self.output_feature_names_),
        }


@dataclass
class DirectionFreeAUCSelector:
    """Training-only top-k AUC screen with deterministic correlation pruning."""

    top_k: int = 25
    max_abs_correlation: float = 0.95

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]
    ) -> "DirectionFreeAUCSelector":
        values = _as_float_matrix(X, "X")
        target = _as_binary(y)
        names = tuple(map(str, feature_names))
        if values.shape[1] != len(names):
            raise ValueError("Feature names do not match X")
        merits: list[tuple[float, str, int]] = []
        for index, name in enumerate(names):
            column = values[:, index]
            if np.nanstd(column) < 1e-12:
                auc = 0.5
            else:
                auc = float(roc_auc_score(target, column))
            merits.append((abs(auc - 0.5), name, index))
        ordered = [
            item[2] for item in sorted(merits, key=lambda item: (-item[0], item[1]))
        ]
        selected: list[int] = []
        for index in ordered:
            if len(selected) >= min(int(self.top_k), values.shape[1]):
                break
            if selected:
                correlations = [
                    abs(float(np.corrcoef(values[:, index], values[:, prior])[0, 1]))
                    for prior in selected
                ]
                correlations = [value for value in correlations if np.isfinite(value)]
                if correlations and max(correlations) > float(self.max_abs_correlation):
                    continue
            selected.append(index)
        if not selected:
            raise ValueError("Feature selector produced an empty view")
        self.selected_indices_ = np.asarray(selected, dtype=np.int64)
        self.input_feature_names_ = names
        self.selected_feature_names_ = tuple(names[index] for index in selected)
        self.direction_free_auc_merit_ = {
            name: float(0.5 + merit) for merit, name, _ in merits
        }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("Selector is not fitted")
        values = _as_float_matrix(X, "X")
        if values.shape[1] != len(self.input_feature_names_):
            raise ValueError("Input schema differs from fitted selector schema")
        return values[:, self.selected_indices_].astype(np.float32)


def _make_estimator(kind: str, seed: int) -> Any:
    if kind == "ridge":
        return LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=int(seed),
        )
    if kind == "elastic":
        return LogisticRegression(
            C=0.1,
            penalty="elasticnet",
            l1_ratio=0.25,
            solver="saga",
            class_weight="balanced",
            max_iter=8000,
            random_state=int(seed),
        )
    if kind == "rbf_svm":
        return SVC(
            C=1.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=False,
            random_state=int(seed),
        )
    if kind == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=300,
            depth=3,
            learning_rate=0.03,
            l2_leaf_reg=5.0,
            random_strength=1.0,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=int(seed),
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    if kind == "tabpfn":
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        kwargs: dict[str, Any] = {}
        signature = inspect.signature(TabPFNClassifier)
        if "device" in signature.parameters:
            try:
                import torch

                kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                kwargs["device"] = "cpu"
        if "random_state" in signature.parameters:
            kwargs["random_state"] = int(seed)
        if "n_estimators" in signature.parameters:
            kwargs["n_estimators"] = 8
        # Version 2.6 is explicitly selected because the official repository
        # identifies its default checkpoint as trained purely on synthetic
        # data.  Never use the moving package default ("auto") here.
        return TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_6, **kwargs
        )
    raise ValueError(f"Unknown model kind {kind!r}; choices={MODEL_KINDS}")


@dataclass
class FoldLocalTableModel:
    """Preprocess, optionally screen, and fit one fixed low-search candidate."""

    kind: str
    seed: int
    top_k: int | None = 25
    fit_scope: str = "current CV training partition only"

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]
    ) -> "FoldLocalTableModel":
        target = _as_binary(y)
        if self.kind == "tabpfn":
            self.preprocessor_ = FoldLocalTabPFNProjector(
                fit_scope=self.fit_scope
            )
        else:
            self.preprocessor_ = FoldLocalTablePreprocessor(
                fit_scope=self.fit_scope
            )
        transformed = self.preprocessor_.fit_transform(X, feature_names)
        current_names = self.preprocessor_.output_feature_names_
        if self.top_k is not None and transformed.shape[1] > int(self.top_k):
            selector_input = (
                self.preprocessor_.selector_shadow(transformed)
                if self.kind == "tabpfn"
                else transformed
            )
            self.selector_ = DirectionFreeAUCSelector(top_k=int(self.top_k)).fit(
                selector_input, target, current_names
            )
            transformed = self.selector_.transform(transformed)
            current_names = self.selector_.selected_feature_names_
        else:
            self.selector_ = None
        self.selected_feature_names_ = tuple(current_names)
        self.estimator_ = _make_estimator(self.kind, self.seed)
        self.estimator_.fit(transformed, target)
        if self.kind == "tabpfn":
            checkpoint = Path(str(getattr(self.estimator_, "model_path", "")))
            if not checkpoint.is_file():
                raise RuntimeError(
                    "Pinned TabPFN v2.6 checkpoint path could not be verified"
                )
            self.tabpfn_checkpoint_path_ = str(checkpoint.resolve())
            self.tabpfn_checkpoint_sha256_ = _cached_file_sha256(checkpoint)
            self.tabpfn_package_version_ = importlib.metadata.version("tabpfn")
        return self

    def _transform(self, X: np.ndarray) -> np.ndarray:
        transformed = self.preprocessor_.transform(X)
        if self.selector_ is not None:
            transformed = self.selector_.transform(transformed)
        return transformed

    def score(self, X: np.ndarray) -> np.ndarray:
        return continuous_score(self.estimator_, self._transform(X))

    def manifest(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "seed": int(self.seed),
            "top_k": self.top_k,
            "fit_scope": self.fit_scope,
            "selected_features": list(self.selected_feature_names_),
            "preprocessing": self.preprocessor_.manifest(),
        }
        if self.kind == "tabpfn":
            result["tabpfn_provenance"] = {
                "package_version": self.tabpfn_package_version_,
                "model_version": "v2.6",
                "training_data_contract": (
                    "official v2.6 default; documented as purely synthetic"
                ),
                "model_path": self.tabpfn_checkpoint_path_,
                "checkpoint_sha256": self.tabpfn_checkpoint_sha256_,
                "moving_default_used": False,
            }
        return result

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.kind != "tabpfn":
            joblib.dump(self, output)
            return
        output.mkdir(parents=True, exist_ok=True)
        from tabpfn.model_loading import save_fitted_tabpfn_model

        save_fitted_tabpfn_model(self.estimator_, output / "model.tabpfn_fit")
        state = {
            "kind": self.kind,
            "seed": self.seed,
            "top_k": self.top_k,
            "fit_scope": self.fit_scope,
            "preprocessor": self.preprocessor_,
            "selector": self.selector_,
            "selected_feature_names": self.selected_feature_names_,
            "tabpfn_checkpoint_path": self.tabpfn_checkpoint_path_,
            "tabpfn_checkpoint_sha256": self.tabpfn_checkpoint_sha256_,
            "tabpfn_package_version": self.tabpfn_package_version_,
        }
        joblib.dump(state, output / "adapter.joblib")

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "FoldLocalTableModel":
        source = Path(path)
        if source.is_file():
            loaded = joblib.load(source)
            if not isinstance(loaded, cls):
                raise TypeError("Saved object is not a FoldLocalTableModel")
            return loaded
        from tabpfn.model_loading import load_fitted_tabpfn_model

        state = joblib.load(source / "adapter.joblib")
        result = cls(
            kind=state["kind"],
            seed=int(state["seed"]),
            top_k=state["top_k"],
            fit_scope=state["fit_scope"],
        )
        result.preprocessor_ = state["preprocessor"]
        result.selector_ = state["selector"]
        result.selected_feature_names_ = tuple(state["selected_feature_names"])
        result.tabpfn_checkpoint_path_ = state["tabpfn_checkpoint_path"]
        result.tabpfn_checkpoint_sha256_ = state["tabpfn_checkpoint_sha256"]
        result.tabpfn_package_version_ = state["tabpfn_package_version"]
        result.estimator_ = load_fitted_tabpfn_model(
            source / "model.tabpfn_fit", device=device
        )
        return result


def _maxauc_pipeline(kind: str, seed: int) -> Pipeline:
    if kind == "lr":
        estimator: Any = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=5000,
            solver="liblinear",
            random_state=0,
        )
    elif kind == "svm":
        estimator = SVC(
            C=1.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=0,
        )
    else:
        raise ValueError(kind)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


@dataclass
class MMSEMaxAUCAnchor:
    """Exact low-search LR+RBF-SVM quality-gated MMSE baseline."""

    seed: int
    quality_gate_ba: float = 0.55
    fit_scope: str = "current CV training partition only"

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]
    ) -> "MMSEMaxAUCAnchor":
        values = _as_float_matrix(X, "MMSE X")
        target = _as_binary(y)
        self.feature_names_ = tuple(map(str, feature_names))
        if values.shape[1] != len(self.feature_names_):
            raise ValueError("Feature names do not match MMSE X")
        if not all(name.startswith("mmse__") for name in self.feature_names_):
            raise ValueError("MMSE anchor received a non-MMSE feature")

        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed)
        quality: dict[str, float] = {}
        for kind in ("lr", "svm"):
            oof = np.full(len(target), np.nan, dtype=np.float64)
            for train_index, test_index in splitter.split(values, target):
                candidate = _maxauc_pipeline(kind, self.seed)
                candidate.fit(values[train_index], target[train_index])
                oof[test_index] = continuous_score(candidate, values[test_index])
            quality[kind] = float(
                balanced_accuracy_score(target, (oof >= 0.5).astype(np.int64))
            )
        raw_weights = {
            kind: max(0.0, score - 0.5)
            if score >= float(self.quality_gate_ba)
            else 0.0
            for kind, score in quality.items()
        }
        if sum(raw_weights.values()) <= 0:
            winner = max(quality, key=lambda kind: (quality[kind], kind))
            raw_weights = {kind: float(kind == winner) for kind in quality}
        total_weight = sum(raw_weights.values())
        self.weights_ = {
            kind: float(value / total_weight) for kind, value in raw_weights.items()
        }
        self.quality_balanced_accuracy_ = quality
        self.models_ = {
            kind: _maxauc_pipeline(kind, self.seed).fit(values, target)
            for kind in ("lr", "svm")
        }
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        values = _as_float_matrix(X, "MMSE X")
        score = sum(
            self.weights_[kind] * continuous_score(model, values)
            for kind, model in self.models_.items()
        )
        return np.asarray(score, dtype=np.float64)

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": "mmse_maxauc_anchor",
            "seed": int(self.seed),
            "fit_scope": self.fit_scope,
            "quality_gate_metric": "balanced_accuracy",
            "quality_gate_minimum": float(self.quality_gate_ba),
            "quality_balanced_accuracy": self.quality_balanced_accuracy_,
            "weights": self.weights_,
            "features": list(self.feature_names_),
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)

    @classmethod
    def load(cls, path: str | Path) -> "MMSEMaxAUCAnchor":
        result = joblib.load(path)
        if not isinstance(result, cls):
            raise TypeError("Saved object is not an MMSEMaxAUCAnchor")
        return result


@dataclass(frozen=True)
class ReferenceECDF:
    """Training-reference empirical CDF for rank-compatible model scores."""

    sorted_reference: tuple[float, ...]

    @classmethod
    def fit(cls, training_oof_score: np.ndarray) -> "ReferenceECDF":
        values = np.asarray(training_oof_score, dtype=np.float64).reshape(-1)
        if len(values) < 3 or not np.isfinite(values).all():
            raise ValueError("ECDF reference must contain finite OOF scores")
        return cls(tuple(np.sort(values).tolist()))

    def transform(self, score: np.ndarray) -> np.ndarray:
        reference = np.asarray(self.sorted_reference, dtype=np.float64)
        values = np.asarray(score, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Cannot ECDF-transform non-finite scores")
        left = np.searchsorted(reference, values, side="left")
        right = np.searchsorted(reference, values, side="right")
        percentile = (left + right + 1.0) / (2.0 * (len(reference) + 1.0))
        return np.clip(percentile, 1e-7, 1.0 - 1e-7)


@dataclass
class SequenceTransformerBranch:
    """Audited 28-day/8-view Transformer with train-only epoch selection."""

    seed: int
    sequence_length: int = 28
    n_views: int = 8
    fit_scope: str = "current CV training partition only"

    def fit(
        self,
        sequences: Sequence[np.ndarray],
        y: np.ndarray,
        feature_names: Sequence[str],
        *,
        fast: bool = False,
    ) -> "SequenceTransformerBranch":
        target = _as_binary(y)
        sequences = [np.asarray(sequence, dtype=np.float32) for sequence in sequences]
        if len(sequences) != len(target):
            raise ValueError("Sequences and labels have different lengths")
        self.feature_names_ = tuple(map(str, feature_names))

        split = StratifiedShuffleSplit(
            n_splits=1, test_size=0.20, random_state=self.seed
        )
        epoch_train, epoch_early = next(
            split.split(np.zeros((len(target), 1)), target)
        )
        train_views, train_mapping = make_fixed_views(
            [sequences[index] for index in epoch_train],
            sequence_length=self.sequence_length,
            n_views=self.n_views,
        )
        early_views, early_mapping = make_fixed_views(
            [sequences[index] for index in epoch_early],
            sequence_length=self.sequence_length,
            n_views=self.n_views,
        )
        epoch_preprocessor = SequencePreprocessor(
            view_days=self.sequence_length,
            fit_scope=f"{self.fit_scope}; neural epoch-selection subtrain only",
        ).fit(train_views, self.feature_names_)
        transformed_train = epoch_preprocessor.transform_views(train_views)
        transformed_early = epoch_preprocessor.transform_views(early_views)
        selected_epoch, epoch_history = select_neural_epoch(
            "sequence_transformer",
            transformed_train,
            target[epoch_train][train_mapping],
            transformed_early,
            target[epoch_early],
            early_mapping,
            seed=self.seed,
            fast=fast,
        )

        all_views, all_mapping = make_fixed_views(
            sequences,
            sequence_length=self.sequence_length,
            n_views=self.n_views,
        )
        self.preprocessor_ = SequencePreprocessor(
            view_days=self.sequence_length,
            fit_scope=self.fit_scope,
        ).fit(all_views, self.feature_names_)
        transformed_all = self.preprocessor_.transform_views(all_views)
        self.model_, refit_history = fit_neural_fixed_epochs(
            "sequence_transformer",
            transformed_all,
            target[all_mapping],
            epochs=selected_epoch,
            seed=self.seed,
        )
        self.selected_epoch_ = int(selected_epoch)
        self.epoch_selection_history_ = epoch_history
        self.refit_history_ = refit_history
        return self

    def score(self, sequences: Sequence[np.ndarray]) -> np.ndarray:
        views, mapping = make_fixed_views(
            sequences,
            sequence_length=self.sequence_length,
            n_views=self.n_views,
        )
        transformed = self.preprocessor_.transform_views(views)
        view_score = self.model_.predict_proba(transformed)[:, 1]
        return aggregate_view_probabilities(view_score, mapping, len(sequences))

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": "sequence_transformer",
            "seed": int(self.seed),
            "fit_scope": self.fit_scope,
            "sequence_length": int(self.sequence_length),
            "n_views": int(self.n_views),
            "selected_epoch": int(self.selected_epoch_),
            "neural_config": asdict(self.model_.config),
            "preprocessing": self.preprocessor_.manifest(),
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.preprocessor_, output / "sequence_preprocessor.joblib")
        self.model_.save(output / "sequence_transformer.pt")
        (output / "adapter.json").write_text(
            json.dumps(
                {
                    "seed": int(self.seed),
                    "sequence_length": int(self.sequence_length),
                    "n_views": int(self.n_views),
                    "fit_scope": self.fit_scope,
                    "feature_names": list(self.feature_names_),
                    "selected_epoch": int(self.selected_epoch_),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | None = None
    ) -> "SequenceTransformerBranch":
        import torch

        source = Path(path)
        adapter = json.loads((source / "adapter.json").read_text(encoding="utf-8"))
        result = cls(
            seed=int(adapter["seed"]),
            sequence_length=int(adapter["sequence_length"]),
            n_views=int(adapter["n_views"]),
            fit_scope=adapter["fit_scope"],
        )
        payload = torch.load(
            source / "sequence_transformer.pt",
            map_location="cpu",
            weights_only=True,
        )
        config = NeuralConfig(**payload["config"])
        torch_model = _build_torch_model(config)
        torch_model.load_state_dict(payload["state_dict"])
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        result.model_ = NeuralSequenceModel(torch_model, config, resolved_device)
        result.preprocessor_ = joblib.load(source / "sequence_preprocessor.joblib")
        result.feature_names_ = tuple(adapter["feature_names"])
        result.selected_epoch_ = int(adapter["selected_epoch"])
        result.epoch_selection_history_ = []
        result.refit_history_ = []
        return result


def clone_unfitted(model: Any) -> Any:
    """Clone either a scikit estimator or one of the dataclass branches."""

    try:
        return clone(model)
    except TypeError:
        cls = type(model)
        parameters = {
            field: getattr(model, field)
            for field in getattr(model, "__dataclass_fields__", {})
            if getattr(model.__dataclass_fields__[field], "init", False)
        }
        return cls(**parameters)


__all__ = [
    "DirectionFreeAUCSelector",
    "FoldLocalTableModel",
    "FoldLocalTablePreprocessor",
    "FoldLocalTabPFNProjector",
    "MMSEMaxAUCAnchor",
    "MODEL_KINDS",
    "ReferenceECDF",
    "SequenceTransformerBranch",
    "continuous_score",
]
