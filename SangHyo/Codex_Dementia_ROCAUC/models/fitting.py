"""Uniform fit/predict dispatch for tabular, TabNet, and TSMixer branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..features import FeatureBundle
from .base import ModelSpec
from .tabular import (
    build_tabular_estimator,
    predict_positive,
    select_spec_columns,
)


@dataclass
class FittedBranch:
    spec: ModelSpec
    params: dict[str, Any]
    estimator: Any
    selected_input_feature_names: tuple[str, ...]
    seed: int

    def predict(self, bundle: FeatureBundle, indices: Sequence[int]) -> np.ndarray:
        positions = np.asarray(indices, dtype=np.int64)
        if self.spec.family in {"tabular", "tabnet"}:
            values, names = select_spec_columns(
                bundle.table.to_numpy(dtype=np.float64),
                bundle.feature_names,
                self.spec,
            )
            if names != self.selected_input_feature_names:
                raise ValueError("Prediction feature schema differs from fitted schema")
            return predict_positive(self.estimator, values[positions])
        if self.spec.family == "sequence":
            if bundle.sequence_feature_names != self.selected_input_feature_names:
                raise ValueError(
                    "Prediction sequence channel schema differs from fitted schema"
                )
            probabilities = np.asarray(
                self.estimator.predict_proba(
                    [bundle.sequences[index] for index in positions]
                ),
                dtype=np.float64,
            )
            return probabilities[:, 1]
        raise ValueError(f"Unknown model family: {self.spec.family}")


@dataclass
class SeedAveragedBranch:
    """Average identically configured refits over prespecified model seeds."""

    spec: ModelSpec
    members: tuple[FittedBranch, ...]
    selected_input_feature_names: tuple[str, ...]
    seeds: tuple[int, ...]

    def predict(self, bundle: FeatureBundle, indices: Sequence[int]) -> np.ndarray:
        if not self.members:
            raise RuntimeError("Seed ensemble contains no fitted branches")
        matrix = np.column_stack(
            [member.predict(bundle, indices) for member in self.members]
        )
        score = matrix.mean(axis=1)
        if not np.isfinite(score).all():
            raise ValueError("Seed ensemble emitted non-finite predictions")
        return np.clip(score, 1e-7, 1.0 - 1e-7)


def fit_branch(
    spec: ModelSpec,
    params: Mapping[str, Any],
    bundle: FeatureBundle,
    y: np.ndarray,
    train_indices: Sequence[int],
    *,
    seed: int,
    config,
) -> FittedBranch:
    positions = np.asarray(train_indices, dtype=np.int64)
    target = np.asarray(y, dtype=np.int64)[positions]
    counts = np.bincount(target, minlength=2)
    if int(counts.min()) < 2:
        raise ValueError(
            f"{spec.name}: fold training needs >=2 per class, got {counts.tolist()}"
        )
    positive_weight = float(counts[0] / counts[1])
    resolved_params = dict(params)
    if spec.family == "tabular":
        values, names = select_spec_columns(
            bundle.table.to_numpy(dtype=np.float64),
            bundle.feature_names,
            spec,
        )
        estimator = build_tabular_estimator(
            spec,
            resolved_params,
            seed=seed,
            n_jobs=int(config.runtime.n_jobs),
            positive_weight=positive_weight,
        )
        estimator.fit(values[positions], target)
        return FittedBranch(spec, resolved_params, estimator, names, seed)
    if spec.family == "tabnet":
        from .tabnet import build_tabnet_estimator

        values, names = select_spec_columns(
            bundle.table.to_numpy(dtype=np.float64),
            bundle.feature_names,
            spec,
        )
        estimator = build_tabnet_estimator(
            spec,
            resolved_params,
            seed=seed,
            neural_config=config.neural,
        )
        estimator.fit(values[positions], target)
        return FittedBranch(spec, resolved_params, estimator, names, seed)
    if spec.family == "sequence":
        from .tsmixer import build_tsmixer_estimator

        estimator = build_tsmixer_estimator(
            resolved_params,
            feature_names=bundle.sequence_feature_names,
            data_config=config.data,
            neural_config=config.neural,
            seed=seed,
        )
        estimator.fit([bundle.sequences[index] for index in positions], target)
        return FittedBranch(
            spec,
            resolved_params,
            estimator,
            bundle.sequence_feature_names,
            seed,
        )
    raise ValueError(f"Unsupported model family: {spec.family}")


def fit_branch_seed_ensemble(
    spec: ModelSpec,
    params: Mapping[str, Any],
    bundle: FeatureBundle,
    y: np.ndarray,
    train_indices: Sequence[int],
    *,
    seed: int,
    n_members: int,
    config,
) -> SeedAveragedBranch:
    """Refit one selected configuration over a fixed seed ensemble."""

    resolved_members = max(1, int(n_members))
    seeds = tuple(int(seed + position * 104729) for position in range(resolved_members))
    branches = tuple(
        fit_branch(
            spec,
            params,
            bundle,
            y,
            train_indices,
            seed=member_seed,
            config=config,
        )
        for member_seed in seeds
    )
    schemas = {branch.selected_input_feature_names for branch in branches}
    if len(schemas) != 1:
        raise ValueError(f"{spec.name}: seed refits resolved different input schemas")
    return SeedAveragedBranch(
        spec=spec,
        members=branches,
        selected_input_feature_names=branches[0].selected_input_feature_names,
        seeds=seeds,
    )


__all__ = [
    "FittedBranch",
    "SeedAveragedBranch",
    "fit_branch",
    "fit_branch_seed_ensemble",
]
