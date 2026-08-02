"""Central configuration for the dementia ROC-AUC experiment.

All values that materially change the estimand, data access, split design, or
model search live here.  The defaults target the known AI Hub layout in this
repository while keeping paths configurable from the CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Track = Literal["wearable", "wearable_protocol", "wearable_mmse"]
Profile = Literal["standard", "max"]


@dataclass(frozen=True)
class DataConfig:
    """Data access and feature-view contract."""

    data_root: str = "Data"
    strict_known_cohort: bool = True
    primary_split: Literal["train"] = "train"
    historical_validation_split: Literal["val"] = "val"
    expected_train_subjects: int = 141
    expected_train_diagnosis_counts: dict[str, int] = field(
        default_factory=lambda: {"CN": 85, "MCI": 47, "Dem": 9}
    )
    expected_validation_subjects: int = 33
    expected_validation_diagnosis_counts: dict[str, int] = field(
        default_factory=lambda: {"CN": 26, "MCI": 4, "Dem": 3}
    )
    target_positive_diagnoses: tuple[str, ...] = ("Dem",)
    target_negative_diagnoses: tuple[str, ...] = ("CN", "MCI")
    tracks: tuple[Track, ...] = (
        "wearable",
        "wearable_protocol",
        "wearable_mmse",
    )
    sequence_min_channel_coverage: float = 0.20
    max_sequence_channels: int = 96

    def resolved_root(self) -> Path:
        return Path(self.data_root).expanduser().resolve()


@dataclass(frozen=True)
class CVConfig:
    """Repeated nested subject-level cross-validation."""

    outer_folds: int = 3
    outer_repeats: int = 5
    inner_folds: int = 3
    inner_repeats: int = 2
    seed: int = 20260728
    minimum_positive_per_validation_fold: int = 1
    bootstrap_iterations: int = 5000
    bootstrap_confidence: float = 0.95
    threshold_objective: Literal["mcc", "balanced_accuracy", "f1"] = "mcc"
    threshold_min_recall: float = 0.50


@dataclass(frozen=True)
class SearchConfig:
    """Nested model screening, tuning, and blending budget."""

    screen_model_names: tuple[str, ...] = (
        "univariate_logreg",
        "elastic_logreg",
        "rbf_svm",
        "extra_trees",
        "random_forest",
        "hist_gradient_boosting",
        "balanced_random_forest",
        "easy_ensemble",
        "lightgbm",
        "xgboost",
        "catboost",
        "mlp",
        "tabnet",
        "tabnet_pretrained",
        "tsmixer",
    )
    top_specs_to_tune: int = 2
    max_ensemble_members: int = 3
    optuna_trials_per_spec: int = 12
    optuna_timeout_seconds_per_spec: int = 600
    blend_weight_trials: int = 256
    minimum_blend_auc_gain: float = 0.0025
    benchmark_all_fixed_models: bool = True
    tune_tabnet_even_if_not_top: bool = False
    tune_tsmixer_even_if_not_top: bool = False
    enable_random_oversampling: bool = True
    enable_smote: bool = True
    enable_adasyn: bool = False
    feature_selection_choices: tuple[int, ...] = (1, 3, 5, 8, 12, 20, 32, 64)


@dataclass(frozen=True)
class NeuralConfig:
    """Resource bounds shared by TabNet and TSMixer."""

    device: Literal["auto", "cpu", "cuda"] = "auto"
    deterministic: bool = True
    max_epochs: int = 160
    patience: int = 25
    batch_size: int = 32
    virtual_batch_size: int = 16
    num_workers: int = 0
    mixed_precision: bool = True
    validation_fraction: float = 0.20


@dataclass(frozen=True)
class RuntimeConfig:
    """Output, parallelism, and execution-safety settings."""

    output_dir: str = "Codex_Dementia_ROCAUC_results"
    n_jobs: int = 1
    fail_on_missing_optional_model: bool = True
    save_fold_models: bool = False
    refit_deployment_model: bool = True
    save_training_fitted_importance: bool = True

    def resolved_output(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete serializable experiment configuration."""

    profile: Profile = "standard"
    data: DataConfig = field(default_factory=DataConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_config(
    *,
    profile: Profile = "standard",
    data_root: str = "Data",
    output_dir: str = "Codex_Dementia_ROCAUC_results",
    n_jobs: int = 1,
    device: Literal["auto", "cpu", "cuda"] = "auto",
) -> ExperimentConfig:
    """Build one of the two declared compute profiles.

    ``standard`` is the reproducible default. ``max`` increases repeats and
    nested search; it also makes TabNet/TSMixer tuning mandatory.  Neither
    profile is executed by constructing this object.
    """

    if profile not in {"standard", "max"}:
        raise ValueError("profile must be 'standard' or 'max'")
    data = DataConfig(data_root=data_root)
    runtime = RuntimeConfig(output_dir=output_dir, n_jobs=max(1, int(n_jobs)))
    neural = NeuralConfig(device=device)
    if profile == "standard":
        return ExperimentConfig(
            profile=profile,
            data=data,
            cv=CVConfig(),
            search=SearchConfig(),
            neural=neural,
            runtime=runtime,
        )
    return ExperimentConfig(
        profile=profile,
        data=data,
        cv=CVConfig(
            outer_folds=3,
            outer_repeats=10,
            inner_folds=3,
            inner_repeats=3,
            bootstrap_iterations=10000,
        ),
        search=SearchConfig(
            top_specs_to_tune=3,
            optuna_trials_per_spec=40,
            optuna_timeout_seconds_per_spec=1800,
            blend_weight_trials=1024,
            tune_tabnet_even_if_not_top=True,
            tune_tsmixer_even_if_not_top=True,
        ),
        neural=NeuralConfig(
            device=device,
            max_epochs=240,
            patience=35,
            batch_size=32,
            virtual_batch_size=16,
        ),
        runtime=runtime,
    )


__all__ = [
    "CVConfig",
    "DataConfig",
    "ExperimentConfig",
    "NeuralConfig",
    "RuntimeConfig",
    "SearchConfig",
    "Track",
    "make_config",
]
