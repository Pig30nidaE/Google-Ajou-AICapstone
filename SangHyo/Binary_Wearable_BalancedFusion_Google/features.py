"""Fold-local compact summaries and CN-reference features."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

import numpy as np

from .data import VIEW_OBSERVATIONS, assert_wearable_schema


SUMMARY_WINDOWS = (7, 14, 35)
SUMMARY_STATISTICS = ("median", "iqr", "mad", "trimmed_mean", "rank_slope")
HEAVY_TAILED_COMPONENTS = frozenset(
    {
        "cal_active",
        "daily_movement",
        "inactive",
        "steps",
        "awake",
        "deep",
        "onset_latency",
        "rem",
        "restless",
    }
)


def schema_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, names)).encode("utf-8")).hexdigest()


def _component(name: str) -> str:
    return str(name).lower().split("__")[-1]


def _rank_slope(values: np.ndarray) -> np.ndarray:
    n_steps = values.shape[1]
    x = np.linspace(-0.5, 0.5, n_steps, dtype=np.float64)
    denominator = float(x @ x)
    return np.einsum("ntf,t->nf", values - values.mean(axis=1, keepdims=True), x) / denominator


def build_multiscale_summaries(
    transformed_views: np.ndarray,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Create fixed 7/14/35-observation robust summaries without labels."""

    views = np.asarray(transformed_views, dtype=np.float64)
    names = [str(name) for name in feature_names]
    assert_wearable_schema(names)
    if views.ndim != 3 or views.shape[1:] != (VIEW_OBSERVATIONS, len(names)):
        raise ValueError(
            f"Expected [N,{VIEW_OBSERVATIONS},{len(names)}], got {views.shape}"
        )
    if not np.isfinite(views).all():
        raise ValueError("Summary input must already be finite")
    blocks: list[np.ndarray] = []
    output_names: list[str] = []
    for window in SUMMARY_WINDOWS:
        tail = views[:, -window:, :]
        q10, q25, median, q75, q90 = np.quantile(
            tail, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1
        )
        mad = np.median(np.abs(tail - median[:, None, :]), axis=1)
        clipped = np.clip(tail, q10[:, None, :], q90[:, None, :])
        statistics = (
            median,
            q75 - q25,
            mad,
            clipped.mean(axis=1),
            _rank_slope(tail),
        )
        # Feature-major layout makes selected names and arrays easy to audit.
        blocks.append(np.stack(statistics, axis=2).reshape(len(views), -1))
        output_names.extend(
            f"{name}__window{window}__{statistic}"
            for name in names
            for statistic in SUMMARY_STATISTICS
        )
    matrix = np.concatenate(blocks, axis=1).astype(np.float32)
    if matrix.shape[1] != len(output_names) or not np.isfinite(matrix).all():
        raise AssertionError("Invalid multiscale summary matrix")
    return matrix, output_names


@dataclass
class ValuePreprocessor:
    """Label-free robust transform fitted on one fold's fixed views."""

    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    fit_scope: str = "current CV training subjects only"

    def fit(self, views: np.ndarray, feature_names: Sequence[str]) -> "ValuePreprocessor":
        values = np.asarray(views, dtype=np.float64)
        names = [str(name) for name in feature_names]
        assert_wearable_schema(names)
        if values.ndim != 3 or values.shape[1:] != (VIEW_OBSERVATIONS, len(names)):
            raise ValueError("ValuePreprocessor requires fixed [N,35,F] views")
        self.feature_names_ = names
        self.log_indices_ = np.asarray(
            [
                index
                for index, name in enumerate(names)
                if _component(name) in HEAVY_TAILED_COMPONENTS
            ],
            dtype=np.int64,
        )
        prepared = self._signed_log(values)
        flat = prepared.reshape(-1, len(names))
        flat[~np.isfinite(flat)] = np.nan
        if np.isnan(flat).all(axis=0).any():
            bad = np.flatnonzero(np.isnan(flat).all(axis=0))
            raise ValueError(f"All-missing compact features: {[names[i] for i in bad]}")
        self.medians_ = np.nanmedian(flat, axis=0)
        filled = np.where(np.isnan(flat), self.medians_[None, :], flat)
        self.lower_ = np.quantile(filled, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(filled, self.upper_quantile, axis=0)
        clipped = np.clip(filled, self.lower_, self.upper_)
        self.centers_ = np.median(clipped, axis=0)
        q25, q75 = np.quantile(clipped, [0.25, 0.75], axis=0)
        self.scales_ = q75 - q25
        fallback = clipped.std(axis=0)
        bad_scale = ~np.isfinite(self.scales_) | (np.abs(self.scales_) < 1e-8)
        self.scales_[bad_scale] = fallback[bad_scale]
        self.scales_[~np.isfinite(self.scales_) | (np.abs(self.scales_) < 1e-8)] = 1.0
        return self

    def _signed_log(self, values: np.ndarray) -> np.ndarray:
        output = np.asarray(values, dtype=np.float64).copy()
        indices = getattr(self, "log_indices_", np.empty(0, dtype=np.int64))
        if len(indices):
            selected = output[..., indices]
            finite = np.isfinite(selected)
            selected[finite] = np.sign(selected[finite]) * np.log1p(np.abs(selected[finite]))
            output[..., indices] = selected
        return output

    def transform(self, views: np.ndarray) -> np.ndarray:
        if not hasattr(self, "feature_names_"):
            raise RuntimeError("ValuePreprocessor must be fitted first")
        values = np.asarray(views, dtype=np.float64)
        if values.ndim != 3 or values.shape[1:] != (
            VIEW_OBSERVATIONS,
            len(self.feature_names_),
        ):
            raise ValueError("Transform view schema differs from fitted schema")
        prepared = self._signed_log(values)
        prepared[~np.isfinite(prepared)] = np.nan
        filled = np.where(np.isnan(prepared), self.medians_, prepared)
        clipped = np.clip(filled, self.lower_, self.upper_)
        output = (clipped - self.centers_) / self.scales_
        if not np.isfinite(output).all():
            raise FloatingPointError("Non-finite value survived preprocessing")
        return output.astype(np.float32)

    def manifest(self) -> dict:
        return {
            "fit_scope": self.fit_scope,
            "labels_consumed": False,
            "feature_count": len(self.feature_names_),
            "feature_schema_sha256": schema_sha256(self.feature_names_),
            "fixed_view_observations": VIEW_OBSERVATIONS,
            "imputation": "fold-training daily median",
            "winsorization": [self.lower_quantile, self.upper_quantile],
            "scaling": "fold-training median/IQR",
            "signed_log_features": [self.feature_names_[i] for i in self.log_indices_],
        }


@dataclass
class StableFeatureSelector:
    """Small deterministic fold-local selector with direction stability."""

    max_features: int = 24
    bootstrap_rounds: int = 64
    correlation_threshold: float = 0.90
    seed: int = 20260723

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> "StableFeatureSelector":
        values = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.int64)
        names = [str(name) for name in feature_names]
        if values.shape != (len(target), len(names)) or set(np.unique(target)) != {0, 1}:
            raise ValueError("Selector inputs must be aligned and contain both classes")
        if len(names) != len(set(names)):
            raise AssertionError("Summary feature names must be unique")
        centers = np.median(values, axis=0)
        q25, q75 = np.quantile(values, [0.25, 0.75], axis=0)
        scales = q75 - q25
        fallback = values.std(axis=0)
        bad = ~np.isfinite(scales) | (np.abs(scales) < 1e-8)
        scales[bad] = fallback[bad]
        scales[~np.isfinite(scales) | (np.abs(scales) < 1e-8)] = 1.0
        z = (values - centers) / scales
        rng = np.random.default_rng(self.seed)
        class_indices = [np.flatnonzero(target == class_id) for class_id in (0, 1)]
        signed = np.zeros((self.bootstrap_rounds, values.shape[1]), dtype=np.float64)
        for round_id in range(self.bootstrap_rounds):
            sample = [
                rng.choice(index, size=len(index), replace=True) for index in class_indices
            ]
            signed[round_id] = z[sample[1]].mean(axis=0) - z[sample[0]].mean(axis=0)
        median_effect = np.median(signed, axis=0)
        expected_sign = np.sign(median_effect)
        consistency = np.mean(np.sign(signed) == expected_sign[None, :], axis=0)
        lower_abs_effect = np.quantile(np.abs(signed), 0.25, axis=0)
        score = consistency * lower_abs_effect
        eligible = np.flatnonzero(
            np.isfinite(score)
            & (values.std(axis=0) > 1e-8)
            & (consistency >= 0.60)
        )
        if len(eligible) < min(8, self.max_features):
            eligible = np.flatnonzero(np.isfinite(score) & (values.std(axis=0) > 1e-8))
        order = sorted(eligible.tolist(), key=lambda i: (-score[i], names[i]))
        centered = z - z.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.square(centered).sum(axis=0))
        unit = np.divide(
            centered,
            norms[None, :],
            out=np.zeros_like(centered),
            where=norms[None, :] > 1e-12,
        )
        selected: list[int] = []
        used_source_windows: set[tuple[str, str]] = set()
        for candidate in order:
            # Do not let the same raw channel/window occupy the whole tiny model.
            parts = names[candidate].rsplit("__", 1)
            source_window = (parts[0], parts[-1])
            base_window = (parts[0].rsplit("__", 1)[0], parts[0].rsplit("__", 1)[-1])
            if base_window in used_source_windows:
                continue
            if selected:
                correlations = np.abs(unit[:, selected].T @ unit[:, candidate])
                if float(correlations.max(initial=0.0)) >= self.correlation_threshold:
                    continue
            selected.append(candidate)
            used_source_windows.add(base_window)
            if len(selected) >= self.max_features:
                break
        if len(selected) < min(8, self.max_features):
            # Correlation/source constraints are softened only to reach a usable
            # small baseline; ranking remains identical and fold-local.
            for candidate in order:
                if candidate not in selected:
                    selected.append(candidate)
                if len(selected) >= min(8, self.max_features):
                    break
        if not selected:
            raise ValueError("No stable summary feature survived")
        self.feature_names_in_ = names
        self.selected_indices_ = np.asarray(selected, dtype=np.int64)
        self.selected_feature_names_ = [names[i] for i in selected]
        self.consistency_ = {names[i]: float(consistency[i]) for i in selected}
        self.effect_ = {names[i]: float(median_effect[i]) for i in selected}
        self.score_ = {names[i]: float(score[i]) for i in selected}
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        if not hasattr(self, "selected_indices_"):
            raise RuntimeError("StableFeatureSelector must be fitted first")
        if values.ndim != 2 or values.shape[1] != len(self.feature_names_in_):
            raise ValueError("Summary schema differs from fitted selector")
        return values[:, self.selected_indices_]

    def manifest(self) -> dict:
        return {
            "fit_scope": "current CV training subjects and labels only",
            "selection": "bootstrap signed-effect stability + correlation pruning",
            "max_features": self.max_features,
            "selected_feature_count": len(self.selected_feature_names_),
            "selected_features": list(self.selected_feature_names_),
            "bootstrap_rounds": self.bootstrap_rounds,
            "correlation_threshold": self.correlation_threshold,
            "direction_consistency": dict(self.consistency_),
            "median_signed_effect": dict(self.effect_),
            "stability_score": dict(self.score_),
        }


@dataclass
class FoldFeaturePipeline:
    """One fitted fold representation for all model branches."""

    max_features: int = 24
    seed: int = 20260723

    def fit(
        self,
        views: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> "FoldFeaturePipeline":
        target = np.asarray(y, dtype=np.int64)
        self.feature_names_ = [str(name) for name in feature_names]
        self.value_ = ValuePreprocessor().fit(views, self.feature_names_)
        transformed = self.value_.transform(views)
        summary, summary_names = build_multiscale_summaries(
            transformed, self.feature_names_
        )
        self.selector_ = StableFeatureSelector(
            max_features=self.max_features,
            seed=self.seed,
        ).fit(summary, target, summary_names)
        selected = self.selector_.transform(summary).astype(np.float64)
        cn = selected[target == 0]
        if len(cn) < 2:
            raise ValueError("CN-reference features require at least two CN subjects")
        self.cn_center_ = np.median(cn, axis=0)
        q25, q75 = np.quantile(cn, [0.25, 0.75], axis=0)
        self.cn_scale_ = q75 - q25
        fallback = cn.std(axis=0)
        bad = ~np.isfinite(self.cn_scale_) | (np.abs(self.cn_scale_) < 1e-8)
        self.cn_scale_[bad] = fallback[bad]
        self.cn_scale_[
            ~np.isfinite(self.cn_scale_) | (np.abs(self.cn_scale_) < 1e-8)
        ] = 1.0
        return self

    def transform_views(self, views: np.ndarray) -> np.ndarray:
        return self.value_.transform(views)

    def transform_temporal(self, views: np.ndarray) -> np.ndarray:
        """Return raw fold-scaled and within-subject centered daily channels."""

        transformed = self.transform_views(views)
        within = transformed - np.median(transformed, axis=1, keepdims=True)
        output = np.concatenate([transformed, within], axis=2)
        if not np.isfinite(output).all():
            raise FloatingPointError("Invalid temporal representation")
        return output.astype(np.float32)

    def transform_subject(self, views: np.ndarray) -> np.ndarray:
        transformed = self.transform_views(views)
        summary, names = build_multiscale_summaries(transformed, self.feature_names_)
        if names != self.selector_.feature_names_in_:
            raise AssertionError("Summary schema changed after fitting")
        selected = self.selector_.transform(summary).astype(np.float64)
        signed_cn_z = (selected - self.cn_center_) / self.cn_scale_
        absolute_cn_z = np.abs(signed_cn_z)
        output = np.concatenate([selected, absolute_cn_z], axis=1)
        if not np.isfinite(output).all():
            raise FloatingPointError("Invalid subject representation")
        return output.astype(np.float32)

    @property
    def subject_feature_names(self) -> list[str]:
        selected = list(self.selector_.selected_feature_names_)
        return selected + [f"{name}__cn_absolute_z" for name in selected]

    def manifest(self) -> dict:
        return {
            "value_preprocessing": self.value_.manifest(),
            "summary_windows": list(SUMMARY_WINDOWS),
            "summary_statistics": list(SUMMARY_STATISTICS),
            "selection": self.selector_.manifest(),
            "cn_reference": {
                "fit_scope": "CN subjects in current CV training fold only",
                "features": "absolute robust z-deviation from CN median/IQR",
            },
            "subject_output_features": len(self.subject_feature_names),
            "temporal_output_features": len(self.feature_names_) * 2,
        }


__all__ = [
    "FoldFeaturePipeline",
    "StableFeatureSelector",
    "SUMMARY_STATISTICS",
    "SUMMARY_WINDOWS",
    "ValuePreprocessor",
    "build_multiscale_summaries",
    "schema_sha256",
]

