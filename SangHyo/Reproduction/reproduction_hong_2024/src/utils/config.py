"""Config loading and validation.

Every assumption in this reproduction lives in a YAML config, never as a constant
in the model code; ``assumptions.md`` documents each field.  ``validate_config``
is where the estimand contracts are enforced -- a config that would let a test set
influence model selection is rejected before any data is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_NAMES = (
    "paper_temporal_reconstruction",
    "paper_literal_variant",
    "strict_same_subject_temporal",
    "fixed_subject_independent",
    "nested_subject_independent",
)

SPLIT_MODES = (
    "final_week_temporal",
    "final_week_temporal_literal",
    "stratified_group_kfold",
    "nested_stratified_group_kfold",
)

MODEL_NAMES = ("lstm", "logistic_regression", "svm", "random_forest", "xgboost")

BASELINE_REPRESENTATIONS = ("flatten", "mean", "last_day", "summary")
BASELINE_BACKENDS = ("sklearn", "h2o")
BASELINE_MODEL_NAMES = ("logistic_regression", "svm", "random_forest", "xgboost")

#: Experiments whose split intentionally puts the same subject on both sides.
#: These estimate Estimand A and must declare it.
ESTIMAND_A_EXPERIMENTS = (
    "paper_temporal_reconstruction",
    "paper_literal_variant",
    "strict_same_subject_temporal",
)
ESTIMAND_B_EXPERIMENTS = ("fixed_subject_independent", "nested_subject_independent")


class ConfigError(ValueError):
    """Raised when a config violates a contract that must fail before training."""


@dataclass
class Config:
    """A validated experiment configuration."""

    experiment: str
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # -- convenience accessors -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"config is missing required key: {dotted}")
        return value

    @property
    def seed(self) -> int:
        return int(self.get("seed", 42))

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(self.get("models.enabled", ["lstm"]))

    @property
    def split_mode(self) -> str:
        return str(self.require("split.mode"))

    @property
    def sequence_lengths(self) -> tuple[int, ...]:
        return tuple(int(v) for v in self.get("sequence.lengths", [3, 4, 5]))

    @property
    def estimand(self) -> str:
        return "A" if self.experiment in ESTIMAND_A_EXPERIMENTS else "B"

    @property
    def tuning_enabled(self) -> bool:
        return bool(self.get("tuning.enabled", False))

    @property
    def scaler_scope(self) -> str:
        return str(self.get("preprocessing.scaler_scope", "train_only"))

    def baseline_backend_for(self, model_name: str) -> str:
        """Resolve a model-specific backend, falling back to the global default."""
        overrides = self.get("models.backend_by_model", {}) or {}
        return str(overrides.get(model_name, self.get("models.baseline_backend", "sklearn")))

    @property
    def uses_h2o(self) -> bool:
        return any(
            model != "lstm" and self.baseline_backend_for(model) == "h2o"
            for model in self.models
        )


def load_config(path: str | Path) -> Config:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {path}")
    config = Config(experiment=str(raw.get("experiment", "")), raw=raw, path=path)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Fail closed on any contract violation, before any data is touched."""
    if config.experiment not in EXPERIMENT_NAMES:
        raise ConfigError(
            f"experiment must be one of {EXPERIMENT_NAMES}, got {config.experiment!r}"
        )

    mode = config.split_mode
    if mode not in SPLIT_MODES:
        raise ConfigError(f"split.mode must be one of {SPLIT_MODES}, got {mode!r}")

    unknown = set(config.models) - set(MODEL_NAMES)
    if unknown:
        raise ConfigError(f"unknown models: {sorted(unknown)}")

    for length in config.sequence_lengths:
        if length < 1:
            raise ConfigError(f"sequence length must be >= 1, got {length}")

    representation = config.get("models.representation", "flatten")
    if representation not in BASELINE_REPRESENTATIONS:
        raise ConfigError(
            f"models.representation must be one of {BASELINE_REPRESENTATIONS}, "
            f"got {representation!r}"
        )

    default_backend = str(config.get("models.baseline_backend", "sklearn"))
    if default_backend not in BASELINE_BACKENDS:
        raise ConfigError(
            f"models.baseline_backend must be one of {BASELINE_BACKENDS}, "
            f"got {default_backend!r}"
        )
    backend_overrides = config.get("models.backend_by_model", {}) or {}
    if not isinstance(backend_overrides, Mapping):
        raise ConfigError("models.backend_by_model must be a mapping")
    unknown_backend_models = set(backend_overrides) - set(BASELINE_MODEL_NAMES)
    if unknown_backend_models:
        raise ConfigError(
            "models.backend_by_model contains unsupported model keys: "
            f"{sorted(unknown_backend_models)}"
        )
    invalid_backends = {
        name: backend
        for name, backend in backend_overrides.items()
        if str(backend) not in BASELINE_BACKENDS
    }
    if invalid_backends:
        raise ConfigError(f"invalid per-model backends: {invalid_backends}")
    if "svm" in config.models and config.baseline_backend_for("svm") == "h2o":
        raise ConfigError(
            "H2O AutoML has no SVM family. Configure "
            "models.backend_by_model.svm: sklearn and record the method mismatch."
        )
    if config.uses_h2o and config.experiment != "paper_temporal_reconstruction":
        raise ConfigError(
            "the H2O AutoML path is limited to paper_temporal_reconstruction. "
            "In fixed/nested subject experiments it would re-select hyperparameters "
            "inside each outer fit and invalidate their fixed/nested selection contract."
        )

    # --- global safety: never negotiable -------------------------------------
    if config.scaler_scope != "train_only":
        raise ConfigError(
            "preprocessing.scaler_scope must be 'train_only'; there is no experiment "
            "in this package for which fitting a scaler on evaluation data is correct"
        )
    if config.get("features.include_cognitive_tests", False):
        raise ConfigError(
            "MMSE and other cognitive tests were used to make the diagnosis and are "
            "not part of the paper's 32 sleep features; they must never be inputs"
        )
    if config.get("features.include_subject_id", False):
        raise ConfigError("subject id must never be an input variable")
    if float(config.get("lstm.recurrent_dropout", 0.0)) != 0.0:
        raise ConfigError(
            "lstm.recurrent_dropout is not implemented by the current one-layer "
            "PyTorch model; set it to 0.0 instead of silently ignoring it"
        )

    if "lstm" in config.models and bool(config.get("lstm.early_stopping", False)):
        validation_days = int(config.get("split.validation_days", 0))
        supports_explicit_monitor = (
            config.experiment
            in {"paper_temporal_reconstruction", "strict_same_subject_temporal"}
            and validation_days > 0
        )
        if not supports_explicit_monitor:
            raise ConfigError(
                "lstm.early_stopping=true requires an explicit train-side validation "
                "period (split.validation_days > 0). This runner must never silently "
                "disable early stopping or monitor the outer test set."
            )

    declared = str(config.get("estimand", ""))
    if declared and declared != config.estimand:
        raise ConfigError(
            f"config declares estimand {declared!r} but experiment "
            f"{config.experiment!r} estimates {config.estimand!r}"
        )

    # --- per-experiment contracts --------------------------------------------
    if config.experiment == "paper_temporal_reconstruction":
        _require(config, mode == "final_week_temporal",
                 "paper_temporal_reconstruction requires split.mode='final_week_temporal'")
        _require(config, not config.tuning_enabled,
                 "the paper reports one final configuration; set tuning.enabled: false")

    if config.experiment == "paper_literal_variant":
        _require(config, mode == "final_week_temporal_literal",
                 "paper_literal_variant requires split.mode='final_week_temporal_literal'")
        if not config.get("split.leakage_diagnostic_only", False):
            raise ConfigError(
                "paper_literal_variant builds sequences before the split, so windows "
                "cross the train/test boundary. Set split.leakage_diagnostic_only: true "
                "to acknowledge that its numbers are a leakage measurement, not a "
                "performance claim."
            )

    if config.experiment == "strict_same_subject_temporal":
        _require(config, mode == "final_week_temporal",
                 "strict_same_subject_temporal requires split.mode='final_week_temporal'")
        _require(config, not config.tuning_enabled,
                 "B1 keeps the paper's configuration fixed; set tuning.enabled: false")
        if not config.get("split.embargo_matches_sequence_length", False):
            raise ConfigError(
                "strict_same_subject_temporal requires "
                "split.embargo_matches_sequence_length: true so that the embargo is "
                "L-1 days for every sequence length"
            )

    if config.experiment == "fixed_subject_independent":
        _require(config, mode == "stratified_group_kfold",
                 "fixed_subject_independent requires split.mode='stratified_group_kfold'")
        _require(config, not config.tuning_enabled,
                 "B2 fixes the paper's hyperparameters; set tuning.enabled: false")
        _require(config, str(config.get("threshold.policy", "fixed")) == "fixed",
                 "B2 must use the paper's fixed 0.5 threshold, not a selected one")

    if config.experiment == "nested_subject_independent":
        _require(config, mode == "nested_stratified_group_kfold",
                 "nested_subject_independent requires "
                 "split.mode='nested_stratified_group_kfold'")
        if int(config.get("split.inner_k", 0)) < 2:
            raise ConfigError("nested_subject_independent requires split.inner_k >= 2")
        _require(config, config.tuning_enabled,
                 "nested_subject_independent selects inside the inner CV; set "
                 "tuning.enabled: true")
        if str(config.get("sequence.length_selection", "")) != "inner_cv":
            raise ConfigError(
                "sequence.length_selection must be 'inner_cv' for the nested "
                "experiment; picking the best of 3/4/5 on the outer test set is the "
                "exact bias this experiment exists to avoid"
            )

    # Only the nested experiment may select anything from data.
    if config.experiment != "nested_subject_independent":
        for key in ("sequence.length_selection", "threshold.policy"):
            value = str(config.get(key, ""))
            if value in ("outer_test", "test"):
                raise ConfigError(f"{key} must not be selected on the test set")


def _require(config: Config, condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(f"[{config.experiment}] {message}")
