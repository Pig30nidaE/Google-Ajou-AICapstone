"""Fold-local preprocessing for the wearable-only binary experiment.

The raw subject table is deliberately high dimensional.  This transformer
learns imputation, clipping, robust scaling, and a conservative domain-prior
screen from *one training fold only*.  No identifier, calendar/coverage
proxy, diagnosis field, or cognitive/MMSE value is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


FORBIDDEN_TOKENS = (
    "email",
    "subject_id",
    "diag",
    "label",
    "mmse",
    "cognitive",
    "doctor",
    "period_id",
    "observed_count",
    "observed_day",
    "observed_night",
    "valid_count",
    "valid_ratio",
    "raw_length",
    "sequence_length",
    "missing_ratio",
    "coverage",
    "calendar_gap",
    "span_day",
    "nonwear",
    "non_wear",
    "mask",
    "delta_since",
    "absolute_date",
)


def assert_feature_contract(columns: Iterable[str]) -> None:
    offenders = sorted(
        str(column)
        for column in columns
        if any(token in str(column).lower() for token in FORBIDDEN_TOKENS)
    )
    if offenders:
        raise AssertionError(f"Forbidden feature(s) detected: {offenders[:20]}")


def _modality(name: str) -> str:
    base = str(name).removeprefix("cn_abs__")
    if base.startswith("activity"):
        return "activity"
    if base.startswith("sleep"):
        return "sleep"
    return "other"


SUMMARY_PRIORITY = {
    "median": 0,
    "iqr": 1,
    "late_half_minus_early_half": 2,
    "mad": 3,
    "trimmed_mean_10": 4,
    "p10": 5,
    "p90": 6,
    "theil_sen_rank_slope": 7,
}
WINDOW_PRIORITY = {"event28": 0, "event7": 1, "event14": 2}


def _domain_priority(name: str) -> tuple[int, int, str]:
    """Return a label-independent order fixed before any fold is inspected."""

    summary = str(name).rsplit("__", 1)[-1]
    window = next(
        (candidate for candidate in WINDOW_PRIORITY if f"__{candidate}__" in str(name)),
        "event14",
    )
    return (
        SUMMARY_PRIORITY.get(summary, len(SUMMARY_PRIORITY)),
        WINDOW_PRIORITY[window],
        str(name),
    )


def _effect_scores(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Absolute standardized mean difference, bounded for tiny folds."""

    negative = values[target == 0]
    positive = values[target == 1]
    mean_delta = np.abs(positive.mean(axis=0) - negative.mean(axis=0))
    pooled = np.sqrt(0.5 * (positive.var(axis=0) + negative.var(axis=0)))
    return np.divide(
        mean_delta,
        pooled,
        out=np.zeros_like(mean_delta),
        where=pooled > 1e-8,
    )


@dataclass
class FoldPreprocessor:
    """Robust, domain-screened numeric preprocessing learned inside a fold."""

    max_features: int = 24
    max_missing_fraction: float = 0.35
    correlation_threshold: float = 0.975
    bootstrap_rounds: int = 24
    min_features_per_modality: int = 12
    add_cn_deviation: bool = False
    seed: int = 20260722

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FoldPreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FoldPreprocessor expects a pandas DataFrame")
        target = np.asarray(y, dtype=np.int64)
        if len(X) != len(target) or set(np.unique(target)) != {0, 1}:
            raise ValueError("FoldPreprocessor requires aligned binary labels {0, 1}")
        assert_feature_contract(X.columns)

        frame = X.replace([np.inf, -np.inf], np.nan).copy()
        usable = (frame.isna().mean() <= self.max_missing_fraction) & (
            frame.nunique(dropna=True) > 1
        )
        self.base_columns_ = frame.columns[usable].tolist()
        if not self.base_columns_:
            raise ValueError("No usable wearable feature remains")
        frame = frame[self.base_columns_]

        self.medians_ = frame.median(axis=0).fillna(0.0)
        filled = frame.fillna(self.medians_).astype(np.float64)
        self.lower_ = filled.quantile(0.01)
        self.upper_ = filled.quantile(0.99)
        clipped = filled.clip(self.lower_, self.upper_, axis=1)
        self.centers_ = clipped.median(axis=0)
        scale = clipped.quantile(0.75) - clipped.quantile(0.25)
        scale = scale.mask(scale.abs() < 1e-8, clipped.std(axis=0)).fillna(1.0)
        self.scales_ = scale.mask(scale.abs() < 1e-8, 1.0)
        standardized = (clipped - self.centers_) / self.scales_

        values = standardized.to_numpy(dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        class_indices = [np.flatnonzero(target == class_id) for class_id in (0, 1)]
        rounds = max(1, int(self.bootstrap_rounds))
        score_matrix = np.zeros((rounds, values.shape[1]), dtype=np.float64)
        for round_id in range(rounds):
            sampled = np.concatenate(
                [
                    rng.choice(indices, size=max(3, int(np.ceil(0.8 * len(indices)))), replace=True)
                    for indices in class_indices
                ]
            )
            sampled_target = target[sampled]
            score_matrix[round_id] = _effect_scores(values[sampled], sampled_target)
        # A lower-confidence score is harder for a one-off noisy feature to win.
        stable_score = np.quantile(score_matrix, 0.25, axis=0)
        stable_score += 0.25 * np.median(score_matrix, axis=0)
        stable_score = np.nan_to_num(stable_score, nan=0.0, posinf=1e6, neginf=0.0)

        columns = np.asarray(self.base_columns_, dtype=object)
        # With only 141 subjects, selecting the largest label association from
        # thousands of candidates suffers severe winner's curse.  Actual
        # screening is therefore fixed by summary/window priorities.  Fold
        # label effects are calculated only for the audit manifest.
        order = sorted(
            range(len(columns)), key=lambda index: _domain_priority(str(columns[index]))
        )
        centered = values - values.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.square(centered).sum(axis=0, keepdims=True))
        unit = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 1e-12)
        selected: list[int] = []
        selected_set: set[int] = set()

        quota = min(
            int(self.min_features_per_modality),
            max(1, int(self.max_features) // 2),
        )
        for modality in ("activity", "sleep"):
            for candidate in (idx for idx in order if _modality(str(columns[idx])) == modality):
                if sum(_modality(str(columns[idx])) == modality for idx in selected) >= quota:
                    break
                if self._not_redundant(unit, selected, candidate):
                    selected.append(candidate)
                    selected_set.add(candidate)

        for candidate in order:
            if len(selected) >= int(self.max_features):
                break
            if candidate in selected_set:
                continue
            if self._not_redundant(unit, selected, candidate):
                selected.append(candidate)
                selected_set.add(candidate)
        if not selected:
            raise ValueError("Correlation pruning removed every feature")

        self.selected_columns_ = [str(columns[index]) for index in selected]
        self.feature_scores_ = {
            str(columns[index]): float(stable_score[index]) for index in selected
        }
        cn = clipped.loc[target == 0, self.selected_columns_]
        self.cn_centers_ = cn.median(axis=0).fillna(self.centers_[self.selected_columns_])
        cn_scale = cn.quantile(0.75) - cn.quantile(0.25)
        cn_scale = cn_scale.mask(
            cn_scale.abs() < 1e-8, self.scales_[self.selected_columns_]
        ).fillna(1.0)
        self.cn_scales_ = cn_scale.mask(cn_scale.abs() < 1e-8, 1.0)
        signed_names = list(self.selected_columns_)
        deviation_names = [f"cn_abs__{name}" for name in signed_names]
        self.output_feature_names_ = signed_names + (
            deviation_names if self.add_cn_deviation else []
        )
        return self

    def _not_redundant(
        self, unit: np.ndarray, selected: list[int], candidate: int
    ) -> bool:
        if not selected:
            return True
        correlations = np.abs(unit[:, selected].T @ unit[:, candidate])
        return bool(float(correlations.max(initial=0.0)) < self.correlation_threshold)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        assert_feature_contract(X.columns)
        missing = sorted(set(self.base_columns_) - set(X.columns))
        if missing:
            raise ValueError(f"Input is missing fitted features: {missing[:10]}")
        frame = X[self.base_columns_].replace([np.inf, -np.inf], np.nan)
        frame = frame.fillna(self.medians_).astype(np.float64)
        frame = frame.clip(self.lower_, self.upper_, axis=1)
        signed = ((frame - self.centers_) / self.scales_)[self.selected_columns_]
        arrays = [signed.to_numpy(dtype=np.float32)]
        if self.add_cn_deviation:
            deviation = (
                (frame[self.selected_columns_] - self.cn_centers_) / self.cn_scales_
            ).abs()
            arrays.append(deviation.to_numpy(dtype=np.float32))
        result = np.column_stack(arrays)
        if not np.isfinite(result).all():
            raise FloatingPointError("Non-finite value survived preprocessing")
        return result

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _check_fitted(self) -> None:
        if not hasattr(self, "selected_columns_"):
            raise RuntimeError("FoldPreprocessor must be fitted before transform")

    @property
    def selected_feature_names(self) -> list[str]:
        self._check_fitted()
        return list(self.output_feature_names_)

    def manifest(self) -> dict:
        self._check_fitted()
        return {
            "fit_scope": "current training fold only",
            "mmse_features_used": False,
            "identifier_or_calendar_features_used": False,
            "imputation": "training-fold median",
            "clipping": "training-fold 1st/99th percentile",
            "scaling": "training-fold median/IQR",
            "selection": (
                "label-independent domain-prior order after fold-local "
                "missingness/variance/correlation checks"
            ),
            "bootstrap_rounds": int(self.bootstrap_rounds),
            "bootstrap_effects_used_for_selection": False,
            "bootstrap_effects_saved_for_audit_only": True,
            "selected_feature_count_before_cn_deviation": len(self.selected_columns_),
            "model_input_width": len(self.output_feature_names_),
            "selected_features": list(self.output_feature_names_),
            "stable_scores": dict(self.feature_scores_),
        }
