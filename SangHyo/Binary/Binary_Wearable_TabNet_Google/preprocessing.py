"""Fold-local robust preprocessing and stability-based feature selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

from .features import AGGREGATE_STATISTICS, assert_feature_contract


HEAVY_TAILED_COMPONENTS = frozenset(
    {
        "cal_active", "cal_total", "daily_movement", "high", "inactive",
        "inactivity_alerts", "low", "medium", "met_min_high",
        "met_min_inactive", "met_min_low", "met_min_medium", "rest", "steps",
        "total", "awake", "deep", "duration", "light", "onset_latency",
        "rem", "restless", "midpoint_at_delta",
    }
)


def schema_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, names)).encode("utf-8")).hexdigest()


def _heavy_tailed(name: str) -> bool:
    base = str(name).split("__aggregate__", 1)[0]
    component = base.split("__")[-1].lower()
    return component in HEAVY_TAILED_COMPONENTS


@dataclass
class FoldPreprocessor:
    max_features: int = 64
    max_missing_fraction: float = 0.40
    correlation_threshold: float = 0.985
    bootstrap_rounds: int = 24
    minimum_per_modality: int = 12
    seed: int = 20260722
    fit_scope: str = "current CV training subjects only"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FoldPreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FoldPreprocessor requires a pandas DataFrame")
        target = np.asarray(y, dtype=np.int64)
        if len(X) != len(target) or set(np.unique(target)) != {0, 1}:
            raise ValueError("Fold input must contain both binary classes")
        assert_feature_contract(X.columns.tolist())
        self.input_columns_ = list(X.columns)
        frame = X.replace([np.inf, -np.inf], np.nan).copy()
        usable = (frame.isna().mean() <= self.max_missing_fraction) & (
            frame.nunique(dropna=True) > 1
        )
        self.base_columns_ = frame.columns[usable].tolist()
        if not self.base_columns_:
            raise ValueError("No usable feature survived missingness/variance filtering")
        frame = frame[self.base_columns_].astype(np.float64).copy()

        self.log_columns_ = [name for name in self.base_columns_ if _heavy_tailed(name)]
        if self.log_columns_:
            values = frame[self.log_columns_].to_numpy(dtype=np.float64, copy=True)
            finite = np.isfinite(values)
            values[finite] = np.sign(values[finite]) * np.log1p(np.abs(values[finite]))
            frame.loc[:, self.log_columns_] = values
        self.medians_ = frame.median(axis=0).fillna(0.0)
        filled = frame.fillna(self.medians_)
        self.lower_ = filled.quantile(0.01)
        self.upper_ = filled.quantile(0.99)
        clipped = filled.clip(self.lower_, self.upper_, axis=1)
        self.centers_ = clipped.median(axis=0)
        scale = clipped.quantile(0.75) - clipped.quantile(0.25)
        fallback = clipped.std(axis=0)
        scale = scale.mask(~np.isfinite(scale) | (scale.abs() < 1e-8), fallback)
        self.scales_ = scale.mask(~np.isfinite(scale) | (scale.abs() < 1e-8), 1.0)
        standardized = ((clipped - self.centers_) / self.scales_).to_numpy(np.float64)

        rng = np.random.default_rng(self.seed)
        class_indices = [np.flatnonzero(target == class_id) for class_id in (0, 1)]
        rounds = max(1, int(self.bootstrap_rounds))
        effects = np.zeros((rounds, standardized.shape[1]), dtype=np.float64)
        frequency = np.zeros(standardized.shape[1], dtype=np.float64)
        top_k = min(standardized.shape[1], max(2 * int(self.max_features), 64))
        for round_index in range(rounds):
            sampled = np.concatenate(
                [
                    rng.choice(indices, size=max(4, int(np.ceil(0.85 * len(indices)))), replace=True)
                    for indices in class_indices
                ]
            )
            sampled_y = target[sampled]
            negative = standardized[sampled][sampled_y == 0]
            positive = standardized[sampled][sampled_y == 1]
            delta = np.abs(positive.mean(axis=0) - negative.mean(axis=0))
            pooled = np.sqrt(0.5 * (positive.var(axis=0) + negative.var(axis=0)))
            effect = np.divide(delta, pooled, out=np.zeros_like(delta), where=pooled > 1e-8)
            effect = np.nan_to_num(effect, nan=0.0, posinf=1e6, neginf=0.0)
            effects[round_index] = effect
            frequency[np.argsort(-effect, kind="stable")[:top_k]] += 1.0
        frequency /= rounds
        median_effect = np.median(effects, axis=0)
        lower_effect = np.quantile(effects, 0.25, axis=0)
        score = frequency + 0.20 * lower_effect + 0.10 * median_effect
        order = sorted(
            range(len(self.base_columns_)),
            key=lambda index: (-score[index], self.base_columns_[index]),
        )

        centered = standardized - standardized.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.square(centered).sum(axis=0))
        unit = np.divide(centered, norms[None, :], out=np.zeros_like(centered), where=norms[None, :] > 1e-12)
        selected: list[int] = []

        def add(candidate: int) -> bool:
            if candidate in selected or len(selected) >= int(self.max_features):
                return False
            if selected:
                correlations = np.abs(unit[:, selected].T @ unit[:, candidate])
                if float(correlations.max(initial=0.0)) >= self.correlation_threshold:
                    return False
            selected.append(candidate)
            return True

        # Preserve the different distribution/trend views from the reference
        # experiment.  This is a soft quota: correlation pruning still wins.
        statistic_quota = max(1, int(self.max_features) // 32)
        for statistic in AGGREGATE_STATISTICS:
            suffix = f"__aggregate__{statistic}"
            for candidate in order:
                if sum(
                    self.base_columns_[index].endswith(suffix) for index in selected
                ) >= statistic_quota:
                    break
                if self.base_columns_[candidate].endswith(suffix):
                    add(candidate)

        quota = min(int(self.minimum_per_modality), max(1, int(self.max_features) // 3))
        for prefix in ("activity__", "sleep__"):
            for candidate in order:
                if sum(self.base_columns_[i].startswith(prefix) for i in selected) >= quota:
                    break
                if self.base_columns_[candidate].startswith(prefix):
                    add(candidate)
        for candidate in order:
            if len(selected) >= int(self.max_features):
                break
            add(candidate)
        if not selected:
            raise ValueError("Feature selection removed every feature")
        self.selected_indices_ = np.asarray(selected, dtype=np.int64)
        self.selected_columns_ = [self.base_columns_[index] for index in selected]
        self.selection_frequency_ = {
            self.base_columns_[index]: float(frequency[index]) for index in selected
        }
        self.selection_effect_ = {
            self.base_columns_[index]: float(median_effect[index]) for index in selected
        }
        self.achieved_statistic_counts_ = {
            statistic: int(
                sum(
                    self.base_columns_[index].endswith(
                        f"__aggregate__{statistic}"
                    )
                    for index in selected
                )
            )
            for statistic in AGGREGATE_STATISTICS
        }
        return self

    def _prepared_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "selected_columns_"):
            raise RuntimeError("FoldPreprocessor must be fitted before transform")
        if list(X.columns) != self.input_columns_:
            missing = sorted(set(self.input_columns_) - set(X.columns))
            extra = sorted(set(X.columns) - set(self.input_columns_))
            raise ValueError(
                "Input feature schema/order differs from fit; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        frame = (
            X[self.base_columns_]
            .replace([np.inf, -np.inf], np.nan)
            .astype(np.float64)
            .copy()
        )
        if self.log_columns_:
            values = frame[self.log_columns_].to_numpy(dtype=np.float64, copy=True)
            finite = np.isfinite(values)
            values[finite] = np.sign(values[finite]) * np.log1p(np.abs(values[finite]))
            frame.loc[:, self.log_columns_] = values
        return frame

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._prepared_frame(X).fillna(self.medians_)
        frame = frame.clip(self.lower_, self.upper_, axis=1)
        frame = (frame - self.centers_) / self.scales_
        values = frame[self.selected_columns_].to_numpy(np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite value survived fold preprocessing")
        return values

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @property
    def selected_feature_names(self) -> list[str]:
        if not hasattr(self, "selected_columns_"):
            raise RuntimeError("FoldPreprocessor has not been fitted")
        return list(self.selected_columns_)

    def manifest(self) -> dict:
        return {
            "fit_scope": self.fit_scope,
            "labels_used_only_for_fold_local_feature_selection": True,
            "input_schema_sha256": schema_sha256(self.input_columns_),
            "usable_feature_count": len(self.base_columns_),
            "selected_feature_count": len(self.selected_columns_),
            "selected_schema_sha256": schema_sha256(self.selected_columns_),
            "selected_features": self.selected_feature_names,
            "signed_log1p_columns": list(self.log_columns_),
            "imputation": "fold median",
            "winsorization": [0.01, 0.99],
            "scaling": "fold median/IQR with standard deviation fallback",
            "selection": "stratified bootstrap effect stability + correlation pruning",
            "bootstrap_rounds": int(self.bootstrap_rounds),
            "requested_soft_statistic_minimum": max(1, int(self.max_features) // 32),
            "achieved_statistic_counts": dict(self.achieved_statistic_counts_),
            "selection_frequency": dict(self.selection_frequency_),
            "median_bootstrap_effect": dict(self.selection_effect_),
        }


__all__ = ["FoldPreprocessor", "schema_sha256"]
