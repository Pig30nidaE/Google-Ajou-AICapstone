"""YAML config 로딩·병합·검증."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Config", "load_config", "ConfigError", "config_hash"]


class ConfigError(ValueError):
    """config가 자기모순이거나 금지 조합을 담고 있을 때."""


class Config(dict):
    """점 표기 접근을 지원하는 dict (``cfg.get_path("vae.latent_dim")``)."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def require(self, path: str) -> Any:
        sentinel = object()
        val = self.get_path(path, sentinel)
        if val is sentinel:
            raise ConfigError(f"config에 필수 항목 {path!r}가 없다")
        return val


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, *, overrides: dict | None = None) -> Config:
    """config를 읽고 ``defaults`` 상속과 override를 적용한 뒤 검증한다.

    ``defaults:`` 키에 다른 yaml 경로를 적으면 그 파일을 먼저 읽어 병합한다 (상대 경로 허용).
    """
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    parent_path = raw.pop("defaults", None)
    merged: dict = {}
    if parent_path:
        parent = load_config(
            (path.parent / parent_path).resolve(), overrides=None
        )
        merged = dict(parent)
    merged = _deep_merge(merged, raw)
    if overrides:
        merged = _deep_merge(merged, overrides)

    cfg = Config(merged)
    cfg.set_path("_meta.config_path", str(path))
    validate_config(cfg)
    return cfg


def config_hash(obj: Any, *, length: int = 16) -> str:
    """config dict의 안정적인 해시 (provenance용)."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


# --------------------------------------------------------------------------------------
_LEAKY_SCOPES = {"all_data", "all_dem", "all_classes"}


def validate_config(cfg: Config) -> None:
    """실험 유형별 금지 조합을 강제한다.

    사용자 지시 14: 전체 데이터에서 이상치 임계값을 결정한 뒤 split하는 구현은
    누수 통제 실험에 사용하지 말라. → config 단계에서 차단한다.
    """
    experiment = cfg.get_path("experiment.name", "")
    leakage_controlled = experiment in {
        "leakage_controlled_non_nested",
        "nested_subject_independent",
    }

    if leakage_controlled:
        bad: list[str] = []
        for key in (
            "preprocessing.fit_scope",
            "preprocessing.scaler_scope",
            "augmentation.vae.fit_scope",
        ):
            val = cfg.get_path(key)
            if val in _LEAKY_SCOPES:
                bad.append(f"{key}={val}")
        if cfg.get_path("split.unit") == "row":
            bad.append("split.unit=row (피험자 단위여야 한다)")
        if bad:
            raise ConfigError(
                f"실험 '{experiment}'는 누수 통제 실험이다. 다음 설정을 사용할 수 없다 "
                f"(사용자 지시 9·14):\n  - " + "\n  - ".join(bad)
            )
        if cfg.get_path("audit.mode") != "enforce":
            raise ConfigError(
                f"실험 '{experiment}'는 audit.mode=enforce여야 한다 "
                f"(현재 {cfg.get_path('audit.mode')!r})"
            )

    if experiment == "leakage_controlled_non_nested" and cfg.get_path("search.enabled", False):
        raise ConfigError(
            "실험 B는 논문 하이퍼파라미터를 고정한다. search.enabled=true를 쓸 수 없다."
        )

    if experiment == "nested_subject_independent":
        n_outer = cfg.get_path("split.outer.n_splits")
        allow_large_k = bool(cfg.get_path("split.outer.allow_large_k", False))
        if n_outer is not None and int(n_outer) >= 5 and not allow_large_k:
            raise ConfigError(
                f"Dem 피험자가 12명뿐이므로 outer {n_outer}-fold를 기본값으로 쓰지 않는다 "
                "(3-fold 권장). 의도적이라면 split.outer.allow_large_k=true를 설정하라."
            )

    method = cfg.get_path("outlier.method")
    if method not in (None, "none", "percentile", "isolation_forest"):
        raise ConfigError(f"알 수 없는 outlier.method: {method!r}")

    aug = cfg.get_path("augmentation.method")
    valid_aug = {"none", "vae", "class_weight", "random_oversampling", "smote"}
    if aug is not None and aug not in valid_aug:
        raise ConfigError(f"알 수 없는 augmentation.method: {aug!r} (가능: {sorted(valid_aug)})")

    vae_fit_scope = cfg.get_path("augmentation.vae.fit_scope")
    if vae_fit_scope not in (None, "train_dem_only"):
        raise ConfigError(
            "현재 VAE 데이터 배선은 augmentation.vae.fit_scope=train_dem_only만 지원한다. "
            f"현재 값: {vae_fit_scope!r}. 지원하지 않는 범위를 조용히 train Dem으로 "
            "대체하면 감사 기록과 실제 학습 범위가 달라지므로 fail-closed 처리한다."
        )

    vae_input_space = cfg.get_path("augmentation.vae.input_space")
    if vae_input_space not in (None, "raw", "scaled"):
        raise ConfigError("augmentation.vae.input_space는 raw 또는 scaled여야 한다")

    recon_reduction = cfg.get_path("augmentation.vae.recon_reduction")
    if recon_reduction not in (None, "mean_per_feature", "sum"):
        raise ConfigError(
            "augmentation.vae.recon_reduction은 mean_per_feature 또는 sum이어야 한다"
        )
    kl_reduction = cfg.get_path("augmentation.vae.kl_reduction")
    if kl_reduction not in (None, "mean", "sum"):
        raise ConfigError("augmentation.vae.kl_reduction은 mean 또는 sum이어야 한다")
