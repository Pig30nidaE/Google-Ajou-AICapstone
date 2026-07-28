"""Shared model specification and availability contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.metadata
import importlib.util
from typing import Literal, Sequence


Family = Literal["tabular", "tabnet", "sequence"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: Family
    required_module: str | None = None
    required_distribution: str | None = None
    fixed_feature_suffixes: tuple[str, ...] = ()
    fixed_params: dict = field(default_factory=dict)
    description: str = ""
    google_origin: bool = False

    @property
    def available(self) -> bool:
        return self.required_module is None or (
            importlib.util.find_spec(self.required_module) is not None
        )

    def engine_manifest(self) -> dict[str, str | bool | None]:
        version: str | None = None
        if self.required_distribution and self.available:
            try:
                version = importlib.metadata.version(self.required_distribution)
            except importlib.metadata.PackageNotFoundError:
                version = None
        return {
            "model_name": self.name,
            "family": self.family,
            "available": self.available,
            "required_module": self.required_module,
            "distribution": self.required_distribution,
            "version": version,
            "silent_fallback_allowed": False,
        }


def model_specs() -> tuple[ModelSpec, ...]:
    """Predeclared library compared on shared outer folds."""

    return (
        ModelSpec(
            "univariate_logreg",
            "tabular",
            fixed_feature_suffixes=("activity__scalar__low__std",),
            fixed_params={"C": 0.5, "top_k": 0, "resampler": "none"},
            description=(
                "Prespecified activity-low variability hypothesis; no full-label "
                "feature search."
            ),
        ),
        ModelSpec(
            "elastic_logreg",
            "tabular",
            fixed_params={
                "C": 0.2,
                "l1_ratio": 0.25,
                "top_k": 12,
                "resampler": "none",
            },
            description="Elastic-net logistic regression with fold-local selection.",
        ),
        ModelSpec(
            "rbf_svm",
            "tabular",
            fixed_params={
                "C": 1.0,
                "gamma": "scale",
                "top_k": 12,
                "resampler": "none",
            },
        ),
        ModelSpec(
            "extra_trees",
            "tabular",
            fixed_params={
                "n_estimators": 600,
                "max_depth": 4,
                "min_samples_leaf": 3,
                "max_features": 0.7,
                "top_k": 32,
                "resampler": "none",
            },
        ),
        ModelSpec(
            "random_forest",
            "tabular",
            fixed_params={
                "n_estimators": 700,
                "max_depth": 4,
                "min_samples_leaf": 3,
                "max_features": 0.7,
                "top_k": 32,
                "resampler": "none",
            },
        ),
        ModelSpec(
            "hist_gradient_boosting",
            "tabular",
            fixed_params={
                "learning_rate": 0.04,
                "max_iter": 180,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 8,
                "l2_regularization": 5.0,
                "top_k": 20,
                "resampler": "random_over",
            },
        ),
        ModelSpec(
            "balanced_random_forest",
            "tabular",
            required_module="imblearn",
            required_distribution="imbalanced-learn",
            fixed_params={
                "n_estimators": 700,
                "max_depth": 4,
                "min_samples_leaf": 2,
                "max_features": 0.7,
                "top_k": 32,
                "resampler": "none",
            },
        ),
        ModelSpec(
            "easy_ensemble",
            "tabular",
            required_module="imblearn",
            required_distribution="imbalanced-learn",
            fixed_params={
                "n_estimators": 30,
                "top_k": 20,
                "resampler": "none",
            },
        ),
        ModelSpec(
            "lightgbm",
            "tabular",
            required_module="lightgbm",
            required_distribution="lightgbm",
            fixed_params={
                "n_estimators": 250,
                "learning_rate": 0.025,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 8,
                "reg_alpha": 2.0,
                "reg_lambda": 8.0,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "top_k": 32,
                "resampler": "none",
            },
        ),
        ModelSpec(
            "xgboost",
            "tabular",
            required_module="xgboost",
            required_distribution="xgboost",
            fixed_params={
                "n_estimators": 250,
                "learning_rate": 0.025,
                "max_depth": 3,
                "min_child_weight": 4.0,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "reg_alpha": 2.0,
                "reg_lambda": 8.0,
                "top_k": 32,
                "resampler": "none",
            },
        ),
        ModelSpec(
            "catboost",
            "tabular",
            required_module="catboost",
            required_distribution="catboost",
            fixed_params={
                "iterations": 300,
                "learning_rate": 0.025,
                "depth": 3,
                "l2_leaf_reg": 8.0,
                "random_strength": 1.0,
                "top_k": 32,
                "resampler": "none",
            },
        ),
        ModelSpec(
            "mlp",
            "tabular",
            fixed_params={
                "hidden_layer_sizes": (32, 16),
                "alpha": 0.02,
                "learning_rate_init": 0.001,
                "top_k": 20,
                "resampler": "random_over",
            },
        ),
        ModelSpec(
            "tabnet",
            "tabnet",
            required_module="pytorch_tabnet",
            required_distribution="pytorch-tabnet",
            fixed_params={
                "n_d": 16,
                "n_a": 16,
                "n_steps": 4,
                "gamma": 1.4,
                "lambda_sparse": 0.0001,
                "virtual_batch_size": 16,
                "top_k": 32,
                "pretrain": False,
                "class_weight_mode": "balanced",
            },
            description="Supervised TabNet.",
            google_origin=True,
        ),
        ModelSpec(
            "tabnet_pretrained",
            "tabnet",
            required_module="pytorch_tabnet",
            required_distribution="pytorch-tabnet",
            fixed_params={
                "n_d": 16,
                "n_a": 16,
                "n_steps": 4,
                "gamma": 1.4,
                "lambda_sparse": 0.0001,
                "virtual_batch_size": 16,
                "top_k": 32,
                "pretrain": True,
                "class_weight_mode": "balanced",
            },
            description="Fold-local unsupervised TabNet pretraining then supervised fit.",
            google_origin=True,
        ),
        ModelSpec(
            "tsmixer",
            "sequence",
            required_module="torch",
            required_distribution="torch",
            fixed_params={
                "sequence_length": 28,
                "hidden_size": 64,
                "n_blocks": 2,
                "dropout": 0.30,
                "learning_rate": 0.0007,
                "weight_decay": 0.01,
                "focal_gamma": 1.0,
            },
            description="Compact masked daily-sequence TSMixer classifier.",
            google_origin=True,
        ),
    )


def available_specs(
    requested_names: Sequence[str], *, fail_on_missing: bool
) -> tuple[tuple[ModelSpec, ...], tuple[dict, ...]]:
    registry = {spec.name: spec for spec in model_specs()}
    unknown = sorted(set(requested_names) - set(registry))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    selected: list[ModelSpec] = []
    skipped: list[dict] = []
    for name in requested_names:
        spec = registry[name]
        import_error: Exception | None = None
        if spec.available and spec.required_module is not None:
            try:
                importlib.import_module(spec.required_module)
            except Exception as error:  # native-library/import health preflight
                import_error = error
        if spec.available and import_error is None:
            selected.append(spec)
            continue
        reason = (
            "required optional dependency is not installed"
            if import_error is None
            else (
                "dependency was discoverable but failed import preflight: "
                f"{type(import_error).__name__}: {import_error}"
            )
        )
        record = {
            **spec.engine_manifest(),
            "available": False,
            "reason": reason,
        }
        if fail_on_missing:
            raise ModuleNotFoundError(
                f"{spec.name} requires module {spec.required_module!r}; "
                f"{reason}; silent substitution is forbidden"
            ) from import_error
        skipped.append(record)
    if not selected:
        raise RuntimeError("No requested model is available")
    return tuple(selected), tuple(skipped)


__all__ = ["Family", "ModelSpec", "available_specs", "model_specs"]
