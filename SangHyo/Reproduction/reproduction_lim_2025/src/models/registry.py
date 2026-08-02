"""Model construction from config, plus the small search spaces for the nested CV."""

from __future__ import annotations

from typing import Any

from . import sequence, tabular

TABULAR_MODELS = ("random_forest", "xgboost")
SEQUENCE_MODELS = ("lstm", "bilstm", "cnn1d")
ALL_MODELS = TABULAR_MODELS + SEQUENCE_MODELS

#: Deliberately tiny per-model spaces.  The brief forbids broad search, and the
#: repository's own history (AGENTS.md §4-3) shows large grids losing to simple
#: baselines on this cohort.
INNER_SEARCH_SPACES: dict[str, list[dict[str, Any]]] = {
    "random_forest": [
        {"max_depth": None, "min_samples_leaf": 1},
        {"max_depth": 6, "min_samples_leaf": 2},
        {"max_depth": 4, "min_samples_leaf": 4},
    ],
    "xgboost": [
        {"max_depth": 3},
        {"max_depth": 6},
    ],
    "lstm": [
        {"hidden_size": 32, "dropout": 0.2},
        {"hidden_size": 64, "dropout": 0.3},
    ],
    "bilstm": [
        {"hidden_size": 32, "dropout": 0.2},
        {"hidden_size": 64, "dropout": 0.3},
    ],
    "cnn1d": [
        {"filters": (32, 32), "kernel_size": 3},
        {"filters": (64, 64), "kernel_size": 5},
    ],
}


def is_sequence_model(name: str) -> bool:
    return name in SEQUENCE_MODELS


def build_model(
    name: str,
    *,
    seed: int = 42,
    device: str = "cpu",
    overrides: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
):
    """Build one learner.  Raises before any data is touched if unsupported."""
    if name == "random_forest":
        return tabular.build_random_forest(seed=seed, overrides=overrides)
    if name == "xgboost":
        return tabular.build_xgboost(seed=seed, overrides=overrides)
    if name in SEQUENCE_MODELS:
        return sequence.build_sequence_model(
            name, device=device, overrides=overrides, training=training
        )
    raise ValueError(f"unknown model {name!r}; expected one of {ALL_MODELS}")


def search_space(name: str, *, enabled: bool = True) -> list[dict[str, Any]]:
    """Candidate hyperparameter dicts for the inner CV.

    With ``enabled=False`` this collapses to the single configured setting, which
    is what the non-nested experiment uses (no re-selection from CV scores).
    """
    if not enabled:
        return [{}]
    return [dict(candidate) for candidate in INNER_SEARCH_SPACES.get(name, [{}])]


def dependency_report() -> dict[str, Any]:
    """Which model families can actually run in this environment."""
    try:
        import xgboost  # noqa: F401
        has_xgboost = True
    except ImportError:
        has_xgboost = False
    return {
        "torch_available": sequence.torch_available(),
        "xgboost_available": has_xgboost,
        "runnable_models": [
            name for name in ALL_MODELS
            if (name != "xgboost" or has_xgboost)
            and (name not in SEQUENCE_MODELS or sequence.torch_available())
        ],
    }
