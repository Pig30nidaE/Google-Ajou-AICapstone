"""Configuration: YAML -> dataclasses, with environment and CLI overrides.

Resolution order (last wins):
    dataclass default -> config.yaml -> environment (BPM_*) -> CLI flag
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "PipelineConfig",
    "RunConfig",
    "PathConfig",
    "DataConfig",
    "FeatureConfig",
    "CVConfig",
    "ScreeningConfig",
    "CandidateConfig",
    "EnsembleConfig",
    "load_config",
    "config_to_dict",
]

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_FILE = PACKAGE_ROOT / "config.yaml"


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str = "Binary_PooledMaxAUC"
    run_id: str | None = None
    seed: int = 20260730
    n_jobs: int = 1
    subject_hash_salt: str = "Binary_PooledMaxAUC"
    profile: str = "default"  # fast | default | max

    def resolved_run_id(self) -> str:
        return self.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


@dataclass(frozen=True)
class PathConfig:
    data_root: str | None = None
    output_root: str | None = None
    cache_root: str | None = None


@dataclass(frozen=True)
class DataConfig:
    # Pooling Training(141) + Validation(33) is the single largest legitimate
    # lever available: it is not direct leakage, but it does consume the last
    # untouched hold-out, so it is disclosed loudly in every report.
    splits: tuple[str, ...] = ("train", "val")
    positive_diagnoses: tuple[str, ...] = ("MCI", "Dem")
    negative_diagnoses: tuple[str, ...] = ("CN",)
    min_days_per_subject: int = 28
    expected_subjects: Mapping[str, int] = field(
        default_factory=lambda: {"train": 141, "val": 33}
    )
    expected_pooled_diagnoses: Mapping[str, int] = field(
        default_factory=lambda: {"CN": 111, "MCI": 51, "Dem": 12}
    )
    strict_cohort_contract: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    include_intraday: bool = True
    #: Which feature views the candidate search may use.
    views: tuple[str, ...] = ("mmse_core", "mmse_plus", "mmse_wear_small", "all")
    #: Statistics computed per wearable daily channel for the wide `all` view.
    wearable_stats: tuple[str, ...] = (
        "mean",
        "sd",
        "cv",
        "median",
        "iqr",
        "p10",
        "p90",
        "trend_per_week",
        "late_minus_early",
        "weekend_minus_weekday",
    )
    #: Small curated wearable block. Chosen from the repeatedly-reported
    #: Dem markers in AGENTS.md 4-2 / EXPERIMENT_SUMMARY_KO.md 2 (deep-sleep
    #: variability, restless, awake, light-sleep ratio, activity score).
    small_wearable_channels: tuple[str, ...] = (
        "slp_deep_ratio",
        "slp_light_ratio",
        "slp_restless",
        "slp_awake_minutes",
        "slp_efficiency",
        "act_rest_minutes",
        "act_score",
        "act_steps",
    )
    small_wearable_stats: tuple[str, ...] = ("mean", "sd", "cv")
    mmse_item_max: float = 2.0


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    n_repeats: int = 10
    min_positive_per_validation_fold: int = 2


@dataclass(frozen=True)
class ScreeningConfig:
    """Fold-local feature screening. Never sees the held-out fold."""

    enabled: bool = True
    top_k_grid: tuple[int, ...] = (15, 25, 40, 70)
    correlation_threshold: float = 0.95
    winsorize_quantile: float = 0.01


@dataclass(frozen=True)
class CandidateConfig:
    #: Learner families to evaluate. Missing optional libraries are skipped and
    #: reported, never silently substituted.
    families: tuple[str, ...] = (
        "logreg",
        "svm_rbf",
        "extratrees",
        "randomforest",
        "hgb",
        "lightgbm",
        "catboost",
        "xgboost",
        "ydf_oblique",
    )
    logreg_c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    svm_c_grid: tuple[float, ...] = (0.3, 1.0, 3.0)
    svm_gamma_grid: tuple[float, ...] = (0.003, 0.01, 0.03)
    class_weight_balanced: bool = True


@dataclass(frozen=True)
class EnsembleConfig:
    enabled: bool = True
    #: Number of top single candidates entering the weight search.
    n_top_candidates: int = 8
    #: Random simplex draws for the weight search (same procedure as
    #: Binary_Google_YDF_AUC's policy search).
    n_simplex_draws: int = 8192
    #: Also try equal-weight and family-winner blends.
    include_structured_blends: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    #: When true, additionally run a fully nested (inner-CV selection) arm so the
    #: report can show how much of the headline is selection optimism.
    honest_nested_comparison: bool = False
    target_roc_auc: float = 0.85
    source_config_file: str | None = None

    # ---- resolved locations ------------------------------------------------
    def resolved_data_root(self, injected: Mapping[str, Any] | None = None) -> Path:
        candidate = self.paths.data_root or _injected(injected, "DATA_ROOT")
        if candidate:
            return _require_data_root(Path(str(candidate)).expanduser())
        project = _injected(injected, "PROJECT_ROOT")
        starts = [Path(str(project))] if project else []
        starts += [Path.cwd(), PACKAGE_ROOT]
        for start in starts:
            for base in [start, *start.parents]:
                for option in (base / "Data", base):
                    if _looks_like_data_root(option):
                        return option.resolve()
        raise FileNotFoundError(
            "Could not locate a Data root containing 1.Training and 2.Validation. "
            "Set paths.data_root in config.yaml or BPM_DATA_ROOT."
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
        if self.paths.cache_root:
            return Path(str(self.paths.cache_root)).expanduser()
        drive = Path("/content/drive/MyDrive")
        if drive.is_dir():
            return drive / f"{self.run.experiment_name}_cache"
        return (PACKAGE_ROOT / f"{self.run.experiment_name}_cache").resolve()

    def resolved_cv(self) -> CVConfig:
        """Profile shorthand for the repeat count."""

        repeats = {"fast": 3, "default": 10, "max": 20}.get(self.run.profile)
        if repeats is None:
            raise ValueError(f"run.profile must be fast|default|max; got {self.run.profile!r}")
        return replace(self.cv, n_repeats=repeats)


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
    raise FileNotFoundError(f"{path} does not contain 1.Training and 2.Validation")


def _injected(injected: Mapping[str, Any] | None, name: str) -> str | None:
    if not injected:
        return None
    value = injected.get(name)
    return None if value is None else str(value)


def _read_config_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        return json.loads(text)
    try:
        import yaml
    except ModuleNotFoundError as error:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "PyYAML is required for .yaml configs (pip install pyyaml), "
            "or pass a .json config instead."
        ) from error
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return loaded


def _coerce(value: Any, template: Any) -> Any:
    if template is None:
        return value
    if isinstance(template, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(template, tuple):
        if isinstance(value, str):
            value = [piece.strip() for piece in value.split(",") if piece.strip()]
        if template and isinstance(template[0], (int, float)) and not isinstance(template[0], bool):
            caster = type(template[0])
            return tuple(caster(item) for item in value)
        return tuple(value)
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    if isinstance(template, str):
        return str(value)
    return value


_OPTIONAL_TYPES: dict[tuple[str, str], type] = {}


def _build_section(cls, defaults, overrides: Mapping[str, Any] | None, section: str = ""):
    if not overrides:
        return defaults
    current = asdict(defaults)
    values: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in current:
            raise KeyError(f"Unknown config key {cls.__name__}.{key}")
        template = getattr(defaults, key)
        if template is None:
            caster = _OPTIONAL_TYPES.get((section, key))
            if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none"}):
                values[key] = None
            else:
                values[key] = caster(value) if caster else value
        else:
            values[key] = _coerce(value, template)
    return replace(defaults, **values)


_SECTIONS = {
    "run": RunConfig,
    "paths": PathConfig,
    "data": DataConfig,
    "features": FeatureConfig,
    "cv": CVConfig,
    "screening": ScreeningConfig,
    "candidates": CandidateConfig,
    "ensemble": EnsembleConfig,
}

_ENV_MAP: dict[str, tuple[str, str]] = {
    "BPM_DATA_ROOT": ("paths", "data_root"),
    "BPM_OUTPUT_ROOT": ("paths", "output_root"),
    "BPM_CACHE_ROOT": ("paths", "cache_root"),
    "BPM_RUN_ID": ("run", "run_id"),
    "BPM_SEED": ("run", "seed"),
    "BPM_PROFILE": ("run", "profile"),
    "BPM_N_JOBS": ("run", "n_jobs"),
    "BPM_CV_SPLITS": ("cv", "n_splits"),
    "BPM_CV_REPEATS": ("cv", "n_repeats"),
    "BPM_FAMILIES": ("candidates", "families"),
    "BPM_VIEWS": ("features", "views"),
}


def load_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> PipelineConfig:
    environ = os.environ if environ is None else environ
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_FILE
    raw: dict[str, Any] = {}
    if config_path.is_file():
        raw = _read_config_file(config_path)
    elif path is not None:
        raise FileNotFoundError(f"Config file not found: {config_path}")

    env_overrides: dict[str, dict[str, Any]] = {}
    for env_name, (section, key) in _ENV_MAP.items():
        if env_name in environ and str(environ[env_name]).strip():
            env_overrides.setdefault(section, {})[key] = environ[env_name]

    sections: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        merged = dict(raw.get(name) or {})
        merged.update(env_overrides.get(name, {}))
        sections[name] = _build_section(cls, cls(), merged, section=name)

    config = PipelineConfig(
        **sections,
        honest_nested_comparison=bool(raw.get("honest_nested_comparison", False)),
        target_roc_auc=float(raw.get("target_roc_auc", 0.85)),
        source_config_file=str(config_path) if config_path.is_file() else None,
    )
    if cli_overrides:
        config = _apply_cli(config, cli_overrides)
    _validate(config)
    return config


def _apply_cli(config: PipelineConfig, overrides: Mapping[str, Any]) -> PipelineConfig:
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
    if config.run.profile not in {"fast", "default", "max"}:
        raise ValueError(f"run.profile must be fast|default|max; got {config.run.profile!r}")
    if config.cv.n_splits < 2:
        raise ValueError("cv.n_splits must be >= 2")
    unknown_views = set(config.features.views) - {
        "mmse_core",
        "mmse_plus",
        "mmse_wear_small",
        "all",
        "wearable_only",
    }
    if unknown_views:
        raise ValueError(f"Unknown features.views: {sorted(unknown_views)}")
    known_families = {
        "logreg",
        "svm_rbf",
        "extratrees",
        "randomforest",
        "hgb",
        "lightgbm",
        "catboost",
        "xgboost",
        "ydf_oblique",
    }
    unknown_families = set(config.candidates.families) - known_families
    if unknown_families:
        raise ValueError(f"Unknown candidates.families: {sorted(unknown_families)}")
    unknown_splits = set(config.data.splits) - {"train", "val"}
    if unknown_splits:
        raise ValueError(f"Unknown data.splits: {sorted(unknown_splits)}")
    if not config.data.splits:
        raise ValueError("data.splits must not be empty")


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    for section in payload.values():
        if isinstance(section, dict):
            for key, value in list(section.items()):
                if isinstance(value, tuple):
                    section[key] = list(value)
    return payload
