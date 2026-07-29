"""Configuration: YAML file -> dataclasses, with environment-variable overrides.

Nothing about the execution environment is hard-coded as a requirement.  Every
path, model name, concurrency limit and seed can be set in ``config.yaml``, or
overridden by an environment variable (``GFP_*``), or - for the three globals
that ``base.ipynb`` injects - taken from the notebook (``PROJECT_ROOT``,
``DATA_ROOT``, ``USER_ROOT``).

Resolution order for every knob (last wins):
    dataclass default  ->  config.yaml  ->  environment variable  ->  CLI flag
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

__all__ = [
    "PipelineConfig",
    "RunConfig",
    "PathConfig",
    "DataConfig",
    "GeminiConfig",
    "PayloadConfig",
    "FeatureConfig",
    "CVConfig",
    "ModelConfig",
    "TuningConfig",
    "load_config",
    "config_to_dict",
    "mmse_modes",
]

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = PACKAGE_ROOT / "config.yaml"


# --------------------------------------------------------------------------- #
# dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunConfig:
    experiment_name: str = "GeminiFeaturePipeline"
    run_id: str | None = None
    seed: int = 20260729
    n_jobs: int = 1
    subject_hash_salt: str = "GeminiFeaturePipeline"

    def resolved_run_id(self) -> str:
        return self.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


@dataclass(frozen=True)
class PathConfig:
    data_root: str | None = None
    output_root: str | None = None
    cache_root: str | None = None


@dataclass(frozen=True)
class DataConfig:
    splits: tuple[str, ...] = ("train",)
    positive_diagnoses: tuple[str, ...] = ("MCI", "Dem")
    negative_diagnoses: tuple[str, ...] = ("CN",)
    min_days_per_subject: int = 28
    strict_cohort_contract: bool = True
    expected_subjects: Mapping[str, int] = field(
        default_factory=lambda: {"train": 141, "val": 33}
    )


@dataclass(frozen=True)
class GeminiConfig:
    enabled: bool = True
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.0
    top_p: float = 0.95
    max_output_tokens: int = 2048
    response_seed: int | None = 7
    timeout_seconds: float = 120.0
    max_retries: int = 6
    initial_backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 90.0
    # Defaults are sized for a free-tier API key (observed limit: 5 requests/min
    # for generate_content on gemini-2.5-flash), not a paid one. Sequential calls
    # spaced 13s apart stay under that with margin (~4.6 req/min). A paid key can
    # raise both via config.yaml or GFP_GEMINI_MAX_CONCURRENCY/GFP_GEMINI_MIN_INTERVAL.
    max_concurrency: int = 1
    min_interval_seconds: float = 13.0
    dry_run: bool = False
    offline: bool = False
    retry_failed: bool = False
    limit_subjects: int | None = None
    repeat_calls: int = 1
    price_per_million_input_tokens: float | None = None
    price_per_million_output_tokens: float | None = None


@dataclass(frozen=True)
class PayloadConfig:
    round_digits: int = 4
    max_series_points: int = 24
    weekly_summary: bool = True
    hourly_profile: bool = True
    series_channels: tuple[str, ...] = (
        "act_steps",
        "act_average_met",
        "act_high_minutes",
        "act_inactive_minutes",
        "slp_total_minutes",
        "slp_efficiency",
        "slp_awake_minutes",
        "slp_midsleep_hour",
    )


@dataclass(frozen=True)
class FeatureConfig:
    base_channels: tuple[str, ...] = (
        "act_steps",
        "act_average_met",
        "act_daily_movement",
        "act_high_minutes",
        "act_medium_minutes",
        "act_inactive_minutes",
        "act_intraday_relative_amplitude",
        "act_intraday_transition_rate",
        "slp_total_minutes",
        "slp_efficiency",
        "slp_awake_minutes",
        "slp_deep_ratio",
        "slp_rem_ratio",
        "slp_restless",
        "slp_hr_average",
        "slp_onset_latency_minutes",
    )
    base_stats: tuple[str, ...] = ("mean", "sd")
    include_clock_regularity: bool = True
    feature_sets: tuple[str, ...] = ("base", "base_gemini")
    mmse_item_max: float = 2.0


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    n_repeats: int = 5
    min_positive_per_validation_fold: int = 2


@dataclass(frozen=True)
class ModelConfig:
    enabled: tuple[str, ...] = ("logreg", "gbdt")
    logreg: Mapping[str, Any] = field(
        default_factory=lambda: {"C": 1.0, "max_iter": 2000, "penalty": "l2"}
    )
    gbdt: Mapping[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 7,
            "min_child_samples": 15,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
        }
    )


@dataclass(frozen=True)
class TuningConfig:
    """Present for interface completeness only; MUST stay disabled at this stage."""

    enabled: bool = False
    n_trials: int = 10


@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    payload: PayloadConfig = field(default_factory=PayloadConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    mmse_mode: str = "both"  # without | with | both
    source_config_file: str | None = None

    # ---- resolved locations -------------------------------------------------
    def resolved_data_root(self, injected: Mapping[str, Any] | None = None) -> Path:
        candidate = self.paths.data_root or _injected_path(injected, "DATA_ROOT")
        if candidate:
            return _require_data_root(Path(str(candidate)).expanduser())
        project = _injected_path(injected, "PROJECT_ROOT")
        starts = [Path(str(project))] if project else []
        starts.append(Path.cwd())
        starts.append(PACKAGE_ROOT)
        for start in starts:
            for base in [start, *start.parents]:
                for option in (base / "Data", base):
                    if _looks_like_data_root(option):
                        return option.resolve()
        raise FileNotFoundError(
            "Could not locate a Data root containing 1.Training and 2.Validation. "
            "Set paths.data_root in config.yaml or the GFP_DATA_ROOT env var."
        )

    def resolved_output_root(self) -> Path:
        if self.paths.output_root:
            return Path(str(self.paths.output_root)).expanduser()
        drive = Path("/content/drive/MyDrive")
        if drive.is_dir():
            return drive / f"{self.run.experiment_name}_result"
        return (PACKAGE_ROOT / f"{self.run.experiment_name}_result").resolve()

    def resolved_run_dir(self) -> Path:
        return self.resolved_output_root() / self.run.resolved_run_id()

    def resolved_cache_root(self) -> Path:
        """Gemini cache must survive across runs, so it is *not* under run_dir."""

        if self.paths.cache_root:
            return Path(str(self.paths.cache_root)).expanduser()
        drive = Path("/content/drive/MyDrive")
        if drive.is_dir():
            return drive / f"{self.run.experiment_name}_cache"
        return (PACKAGE_ROOT / f"{self.run.experiment_name}_cache").resolve()

    def with_overrides(self, **kwargs: Any) -> "PipelineConfig":
        return replace(self, **kwargs)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _looks_like_data_root(path: Path) -> bool:
    try:
        return (path / "1.Training").is_dir() and (path / "2.Validation").is_dir()
    except OSError:
        return False


def _require_data_root(path: Path) -> Path:
    for candidate in (path, path / "Data"):
        if _looks_like_data_root(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        f"{path} does not contain 1.Training and 2.Validation"
    )


def _injected_path(injected: Mapping[str, Any] | None, name: str) -> str | None:
    if not injected:
        return None
    value = injected.get(name)
    return None if value is None else str(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        return json.loads(text)
    try:
        import yaml  # PyYAML ships with Colab; also pinned in requirements_colab.txt
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "PyYAML is required to read a .yaml config. "
            "Install it (pip install pyyaml) or pass a .json config instead."
        ) from error
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


def _coerce(value: Any, template: Any) -> Any:
    """Convert a YAML/env value into the type implied by the dataclass default."""

    if template is None:
        return value
    if isinstance(template, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(template, tuple):
        if isinstance(value, str):
            value = [piece.strip() for piece in value.split(",") if piece.strip()]
        return tuple(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, str):
        return str(value)
    return value


# Optional fields default to None, so their intended type cannot be inferred
# from the default value; it is declared here instead.
_OPTIONAL_FIELD_TYPES: dict[tuple[str, str], type] = {
    ("gemini", "response_seed"): int,
    ("gemini", "limit_subjects"): int,
    ("gemini", "price_per_million_input_tokens"): float,
    ("gemini", "price_per_million_output_tokens"): float,
}


def _coerce_optional(section: str, key: str, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "null", "none"}:
        return None
    caster = _OPTIONAL_FIELD_TYPES.get((section, key))
    return caster(value) if caster else value


def _build_section(cls, defaults, overrides: Mapping[str, Any] | None, section: str = ""):
    if not overrides:
        return defaults
    values: dict[str, Any] = {}
    current = asdict(defaults)
    for key, value in overrides.items():
        if key not in current:
            raise KeyError(f"Unknown config key {cls.__name__}.{key}")
        template = getattr(defaults, key)
        if template is None:
            values[key] = _coerce_optional(section, key, value)
        else:
            values[key] = _coerce(value, template)
    return replace(defaults, **values)


_ENV_MAP: dict[str, tuple[str, str]] = {
    "GFP_DATA_ROOT": ("paths", "data_root"),
    "GFP_OUTPUT_ROOT": ("paths", "output_root"),
    "GFP_CACHE_ROOT": ("paths", "cache_root"),
    "GFP_RUN_ID": ("run", "run_id"),
    "GFP_SEED": ("run", "seed"),
    "GFP_N_JOBS": ("run", "n_jobs"),
    "GFP_GEMINI_MODEL": ("gemini", "model"),
    "GFP_GEMINI_API_KEY_ENV": ("gemini", "api_key_env"),
    "GFP_GEMINI_ENABLED": ("gemini", "enabled"),
    "GFP_GEMINI_TEMPERATURE": ("gemini", "temperature"),
    "GFP_GEMINI_MAX_CONCURRENCY": ("gemini", "max_concurrency"),
    "GFP_GEMINI_MIN_INTERVAL": ("gemini", "min_interval_seconds"),
    "GFP_GEMINI_MAX_RETRIES": ("gemini", "max_retries"),
    "GFP_GEMINI_TIMEOUT": ("gemini", "timeout_seconds"),
    "GFP_GEMINI_LIMIT_SUBJECTS": ("gemini", "limit_subjects"),
    "GFP_CV_SPLITS": ("cv", "n_splits"),
    "GFP_CV_REPEATS": ("cv", "n_repeats"),
    "GFP_MMSE_MODE": ("__root__", "mmse_mode"),
}

_SECTIONS = {
    "run": RunConfig,
    "paths": PathConfig,
    "data": DataConfig,
    "gemini": GeminiConfig,
    "payload": PayloadConfig,
    "features": FeatureConfig,
    "cv": CVConfig,
    "models": ModelConfig,
    "tuning": TuningConfig,
}


def _environment_overrides(environ: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for env_name, (section, key) in _ENV_MAP.items():
        if env_name in environ and str(environ[env_name]).strip() != "":
            overrides.setdefault(section, {})[key] = environ[env_name]
    return overrides


def load_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    """Build the effective configuration for one run."""

    environ = os.environ if environ is None else environ
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_FILE
    raw: dict[str, Any] = {}
    if config_path.is_file():
        raw = _read_yaml(config_path)
    elif path is not None:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    env_overrides = _environment_overrides(environ)
    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        merged: dict[str, Any] = dict(raw.get(name) or {})
        merged.update(env_overrides.get(name, {}))
        sections[name] = _build_section(cls, cls(), merged, section=name)

    mmse_mode = str(
        env_overrides.get("__root__", {}).get("mmse_mode", raw.get("mmse_mode", "both"))
    )
    config = PipelineConfig(
        **sections,
        mmse_mode=mmse_mode,
        source_config_file=str(config_path) if config_path.is_file() else None,
    )

    if cli_overrides:
        config = _apply_cli_overrides(config, cli_overrides)
    _validate(config)
    return config


def _apply_cli_overrides(
    config: PipelineConfig, overrides: Mapping[str, Any]
) -> PipelineConfig:
    """CLI flags win over both YAML and environment."""

    per_section: dict[str, dict[str, Any]] = {}
    root: dict[str, Any] = {}
    for dotted, value in overrides.items():
        if value is None:
            continue
        if "." not in dotted:
            root[dotted] = value
            continue
        section, key = dotted.split(".", 1)
        per_section.setdefault(section, {})[key] = value
    updates: dict[str, Any] = {}
    for section, values in per_section.items():
        if section not in _SECTIONS:
            raise KeyError(f"Unknown config section: {section}")
        updates[section] = _build_section(
            _SECTIONS[section], getattr(config, section), values, section=section
        )
    updates.update(root)
    return replace(config, **updates)


def _validate(config: PipelineConfig) -> None:
    if config.mmse_mode not in {"without", "with", "both"}:
        raise ValueError(f"mmse_mode must be without|with|both; got {config.mmse_mode!r}")
    if config.tuning.enabled:
        raise ValueError(
            "tuning.enabled must stay false at this stage: the goal is an initial, "
            "non-tuned Gemini-feature comparison (see README_KO.md 'hyperparameter policy')."
        )
    if config.cv.n_splits < 2:
        raise ValueError("cv.n_splits must be >= 2")
    if config.gemini.repeat_calls < 1:
        raise ValueError("gemini.repeat_calls must be >= 1")
    if config.gemini.temperature < 0:
        raise ValueError("gemini.temperature must be >= 0")
    unknown_sets = set(config.features.feature_sets) - {"base", "base_gemini", "gemini_only"}
    if unknown_sets:
        raise ValueError(f"Unknown feature_sets: {sorted(unknown_sets)}")
    unknown_models = set(config.models.enabled) - {"logreg", "gbdt"}
    if unknown_models:
        raise ValueError(f"Unknown models.enabled: {sorted(unknown_models)}")
    unknown_splits = set(config.data.splits) - {"train", "val"}
    if unknown_splits:
        raise ValueError(f"Unknown data.splits: {sorted(unknown_splits)}")


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    """JSON-serializable snapshot (no secrets: only the *name* of the key env var)."""

    payload = asdict(config)
    for section in payload.values():
        if isinstance(section, dict):
            for key, value in list(section.items()):
                if isinstance(value, tuple):
                    section[key] = list(value)
    return payload


def mmse_modes(config: PipelineConfig) -> Sequence[str]:
    return ("without", "with") if config.mmse_mode == "both" else (config.mmse_mode,)
