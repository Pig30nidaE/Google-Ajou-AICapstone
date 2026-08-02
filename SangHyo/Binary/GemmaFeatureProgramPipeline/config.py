"""Configuration and path resolution for the feature-program experiment.

Only Training subjects are used for model development.  The repeatedly viewed
33-subject historical Validation set is intentionally outside this pipeline.
Configuration precedence is:

    dataclass defaults < config.yaml < environment < CLI

The CLI layer is applied in :mod:`run`; this module owns the first three.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import EXPERIMENT_NAME

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.yaml"


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str = EXPERIMENT_NAME
    run_id: str | None = None
    seed: int = 20260729
    profile: str = "standard"
    n_bootstrap: int = 4000

    def resolved_run_id(self) -> str:
        return self.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


@dataclass(frozen=True)
class PathConfig:
    data_root: str | None = None
    output_root: str | None = None
    cache_root: str | None = None


@dataclass(frozen=True)
class GemmaConfig:
    enabled: bool = True
    model: str = "gemma-4-31b-it"
    api_key_env: str = "GEMINI_API_KEY"
    temperature: float = 0.0
    max_output_tokens: int = 8192
    thinking_level: str | None = "minimal"
    thinking_budget: int | None = None
    timeout_seconds: float = 180.0
    max_retries: int = 6
    initial_backoff_seconds: float = 2.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 90.0
    offline: bool = False
    regenerate_program: bool = False


@dataclass(frozen=True)
class CVConfig:
    outer_folds: int = 5
    inner_folds: int = 4
    standard_repeats: int = 10
    full_repeats: int = 20
    smoke_repeats: int = 1

    def repeats_for(self, profile: str) -> int:
        mapping = {
            "smoke": self.smoke_repeats,
            "standard": self.standard_repeats,
            "full": self.full_repeats,
        }
        if profile not in mapping:
            raise ValueError(f"Unknown profile {profile!r}; choose from {sorted(mapping)}")
        return int(mapping[profile])


@dataclass(frozen=True)
class PipelineConfig:
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    gemma: GemmaConfig = field(default_factory=GemmaConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    source_config_file: str | None = None

    def resolved_data_root(self, injected: Mapping[str, Any] | None = None) -> Path:
        candidate = self.paths.data_root or _injected(injected, "DATA_ROOT")
        if candidate:
            return _require_data_root(Path(str(candidate)).expanduser())

        project = _injected(injected, "PROJECT_ROOT")
        starts = [Path(str(project))] if project else []
        starts.extend((Path.cwd(), PACKAGE_ROOT))
        for start in starts:
            for parent in (start, *start.parents):
                for option in (parent / "Data", parent):
                    if _looks_like_data_root(option):
                        return option.resolve()
        raise FileNotFoundError(
            "Data root를 찾지 못했습니다. DATA_ROOT 또는 GFPP_DATA_ROOT를 설정하세요."
        )

    def resolved_output_root(self) -> Path:
        if self.paths.output_root:
            return Path(self.paths.output_root).expanduser().resolve()
        drive = Path("/content/drive/MyDrive")
        if drive.is_dir():
            return drive / f"{self.run.experiment_name}_result"
        if _package_is_under_colab_content():
            raise RuntimeError(
                "Colab Drive가 마운트되지 않았습니다. /content/drive/MyDrive를 "
                "마운트하거나 --output-dir을 명시하세요."
            )
        return PACKAGE_ROOT / f"{self.run.experiment_name}_result"

    def resolved_run_dir(self) -> Path:
        return self.resolved_output_root() / self.run.resolved_run_id()

    def resolved_cache_root(self) -> Path:
        if self.paths.cache_root:
            return Path(self.paths.cache_root).expanduser().resolve()
        drive = Path("/content/drive/MyDrive")
        if drive.is_dir():
            return drive / f"{self.run.experiment_name}_cache"
        if _package_is_under_colab_content():
            raise RuntimeError(
                "Colab Drive가 마운트되지 않았습니다. /content/drive/MyDrive를 "
                "마운트하거나 --cache-dir을 명시하세요."
            )
        return PACKAGE_ROOT / f"{self.run.experiment_name}_cache"

    def with_run(self, **changes: Any) -> "PipelineConfig":
        return replace(self, run=replace(self.run, **changes))

    def with_gemma(self, **changes: Any) -> "PipelineConfig":
        return replace(self, gemma=replace(self.gemma, **changes))

    def with_paths(self, **changes: Any) -> "PipelineConfig":
        return replace(self, paths=replace(self.paths, **changes))


def _injected(namespace: Mapping[str, Any] | None, name: str) -> Any:
    return namespace.get(name) if namespace else None


def _looks_like_data_root(path: Path) -> bool:
    return (path / "1.Training").is_dir() and (path / "2.Validation").is_dir()


def _package_is_under_colab_content() -> bool:
    try:
        PACKAGE_ROOT.resolve().relative_to(Path("/content"))
        return True
    except ValueError:
        return False


def _require_data_root(path: Path) -> Path:
    resolved = path.resolve()
    if not _looks_like_data_root(resolved):
        raise FileNotFoundError(
            f"{resolved} 아래에 1.Training/2.Validation 폴더가 없습니다."
        )
    return resolved


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _optional_int(value: Any) -> int | None:
    text = _optional_text(value)
    return None if text is None else int(text)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean, received {value!r}")


def _apply_environment(config: PipelineConfig, environ: Mapping[str, str]) -> PipelineConfig:
    run_changes: dict[str, Any] = {}
    path_changes: dict[str, Any] = {}
    gemma_changes: dict[str, Any] = {}

    scalar_run = {
        "GFPP_RUN_ID": ("run_id", _optional_text),
        "GFPP_PROFILE": ("profile", str),
        "GFPP_SEED": ("seed", int),
        "GFPP_N_BOOTSTRAP": ("n_bootstrap", int),
    }
    scalar_paths = {
        "GFPP_DATA_ROOT": ("data_root", str),
        "GFPP_OUTPUT_ROOT": ("output_root", str),
        "GFPP_CACHE_ROOT": ("cache_root", str),
    }
    scalar_gemma = {
        "GFPP_GEMINI_MODEL": ("model", str),
        "GFPP_GEMINI_API_KEY_ENV": ("api_key_env", str),
        "GFPP_GEMINI_MAX_OUTPUT_TOKENS": ("max_output_tokens", int),
        "GFPP_GEMINI_THINKING_LEVEL": ("thinking_level", _optional_text),
        "GFPP_GEMINI_THINKING_BUDGET": ("thinking_budget", _optional_int),
        "GFPP_GEMINI_OFFLINE": ("offline", _bool),
        "GFPP_REGENERATE_PROGRAM": ("regenerate_program", _bool),
    }
    for source, (target, convert) in scalar_run.items():
        if source in environ:
            run_changes[target] = convert(environ[source])
    for source, (target, convert) in scalar_paths.items():
        if source in environ:
            path_changes[target] = convert(environ[source])
    for source, (target, convert) in scalar_gemma.items():
        if source in environ:
            gemma_changes[target] = convert(environ[source])

    # The prior notebook used GFP_GEMINI_MODEL.  Honour it only when the new,
    # experiment-specific variable is absent so migration is unsurprising.
    if "GFPP_GEMINI_MODEL" not in environ and "GFP_GEMINI_MODEL" in environ:
        gemma_changes["model"] = str(environ["GFP_GEMINI_MODEL"])

    return replace(
        config,
        run=replace(config.run, **run_changes),
        paths=replace(config.paths, **path_changes),
        gemma=replace(config.gemma, **gemma_changes),
    )


def load_config(
    path: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> PipelineConfig:
    source = Path(path) if path else DEFAULT_CONFIG_PATH
    if path is not None and not source.is_file():
        raise FileNotFoundError(f"Configured YAML file does not exist: {source}")
    raw: Mapping[str, Any] = {}
    if source.is_file():
        loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw = _mapping(loaded, "config root")

    run = RunConfig(**dict(_mapping(raw.get("run"), "run")))
    paths = PathConfig(**dict(_mapping(raw.get("paths"), "paths")))
    gemma = GemmaConfig(**dict(_mapping(raw.get("gemma"), "gemma")))
    cv = CVConfig(**dict(_mapping(raw.get("cv"), "cv")))
    config = PipelineConfig(
        run=run,
        paths=paths,
        gemma=gemma,
        cv=cv,
        source_config_file=str(source.resolve()) if source.is_file() else None,
    )
    effective_environment = os.environ if environ is None else environ
    return _apply_environment(config, effective_environment)


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    return asdict(config)


__all__ = [
    "CVConfig",
    "DEFAULT_CONFIG_PATH",
    "GemmaConfig",
    "PathConfig",
    "PipelineConfig",
    "RunConfig",
    "config_to_dict",
    "load_config",
]
