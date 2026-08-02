"""Config loading and validation.

Every assumption in this reproduction lives in a YAML config, never as a constant
in the model code.  ``assumptions.md`` documents each field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

EXPERIMENT_NAMES = (
    "paper_reported_reconstruction",
    "leakage_controlled_non_nested",
    "nested_subject_independent",
)

REPRESENTATIONS = ("tabular_subject_aggregate", "daily_record", "temporal_sequence")

SPLIT_MODES = (
    "official_partition",
    "assumption_variant_random_subject_holdout",
    "assumption_variant_random_row_holdout",
    "stratified_group_kfold",
    "nested_stratified_group_kfold",
)

MODEL_NAMES = ("random_forest", "xgboost", "lstm", "bilstm", "cnn1d")

#: Split modes that put the same subject on both sides of the split.  Allowed only
#: in the paper-reconstruction experiment, and only as a leakage diagnostic.
LEAKY_SPLIT_MODES = ("assumption_variant_random_row_holdout",)


class ConfigError(ValueError):
    """Raised when a config violates a contract that must fail before training."""


@dataclass
class Config:
    """A validated experiment configuration.

    The raw mapping is kept in :attr:`raw` so results can record exactly what was
    requested, including keys this dataclass does not model explicitly.
    """

    experiment: str
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    # -- convenience accessors -------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Read ``a.b.c`` from the raw mapping, returning *default* if absent."""
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
        return tuple(self.get("models", list(MODEL_NAMES)))

    @property
    def split_mode(self) -> str:
        return str(self.require("split.mode"))

    @property
    def scaler_scope(self) -> str:
        return str(self.get("preprocessing.scaler_scope", "train_only"))

    @property
    def is_leaky_split(self) -> bool:
        return self.split_mode in LEAKY_SPLIT_MODES


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config."""
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

    for name in config.models:
        rep = config.get(f"representations.{name}")
        if rep is None:
            raise ConfigError(f"representations.{name} must be set for model {name!r}")
        if rep not in REPRESENTATIONS:
            raise ConfigError(
                f"representations.{name} must be one of {REPRESENTATIONS}, got {rep!r}"
            )

    # --- contracts that differ per experiment --------------------------------
    if config.experiment == "paper_reported_reconstruction":
        # The leaky variant is permitted here *only* as a diagnostic, and it must
        # declare itself as such so no downstream table reads it as performance.
        if config.is_leaky_split and not config.get("split.leakage_diagnostic_only"):
            raise ConfigError(
                f"split.mode={mode!r} duplicates subjects across the split. Set "
                "split.leakage_diagnostic_only: true to acknowledge that its numbers "
                "are a leakage measurement and not a performance claim."
            )
    else:
        if config.is_leaky_split:
            raise ConfigError(
                f"split.mode={mode!r} is a row-level split and is forbidden in "
                f"experiment {config.experiment!r}; subjects must not span folds."
            )
        if config.scaler_scope != "train_only":
            raise ConfigError(
                "preprocessing.scaler_scope must be 'train_only' outside the paper "
                f"reconstruction; got {config.scaler_scope!r}"
            )

    if config.experiment == "leakage_controlled_non_nested":
        if mode != "stratified_group_kfold":
            raise ConfigError(
                "leakage_controlled_non_nested requires split.mode='stratified_group_kfold'"
            )
        if config.get("tuning.enabled", False):
            raise ConfigError(
                "leakage_controlled_non_nested must not re-select models or "
                "hyperparameters from CV scores; set tuning.enabled: false"
            )

    if config.experiment == "nested_subject_independent":
        if mode != "nested_stratified_group_kfold":
            raise ConfigError(
                "nested_subject_independent requires "
                "split.mode='nested_stratified_group_kfold'"
            )
        if int(config.get("split.inner_k", 0)) < 2:
            raise ConfigError("nested_subject_independent requires split.inner_k >= 2")
        if config.get("threshold.policy") != "inner_cv":
            raise ConfigError(
                "nested_subject_independent must select the threshold inside the "
                "inner CV; set threshold.policy: inner_cv"
            )

    # --- global safety -------------------------------------------------------
    if config.get("features.include_cognitive_tests", False):
        analysis = config.get("analysis", "main_lifelog_only")
        if analysis != "secondary_lifelog_plus_cognitive":
            raise ConfigError(
                "cognitive test variables are only allowed when "
                "analysis='secondary_lifelog_plus_cognitive'; the main lifelog "
                "analysis must never see MMSE/SNSB or diagnosis-derived features"
            )

    scope = config.scaler_scope
    if scope not in ("train_only", "all_data"):
        raise ConfigError(
            f"preprocessing.scaler_scope must be 'train_only' or 'all_data', got {scope!r}"
        )
