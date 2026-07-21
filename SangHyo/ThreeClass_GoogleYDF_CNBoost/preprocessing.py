"""Fold-local preprocessing with explicit CN-vs-impaired emphasis.

Nothing learned here is shared across validation folds.  Feature screening uses
only the fit-fold labels and balances a task score with a CN-vs-impaired score.
This is intentionally conservative for 141 subjects and 880 raw summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from feature_engineering import feature_family


SelectionTask = Literal["multiclass", "cn_vs_impaired", "mci_vs_dem", "one_vs_rest"]


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=-np.inf, posinf=1e12, neginf=-np.inf)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks / max(1, len(values) - 1)


def _binary_auc_effects(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    positive = np.asarray(target, dtype=np.int64) == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        return np.zeros(matrix.shape[1], dtype=float)
    ranks = pd.DataFrame(matrix).rank(method="average", axis=0).to_numpy(dtype=float)
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    auc = (
        ranks[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2.0
    ) / (n_positive * n_negative)
    return 2.0 * np.abs(auc - 0.5)


def _anova_f_scores(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Vectorized one-way ANOVA F scores without a global learned state."""

    matrix = np.asarray(values, dtype=np.float64)
    labels = np.asarray(target, dtype=np.int64)
    classes = np.unique(labels)
    overall = matrix.mean(axis=0)
    between = np.zeros(matrix.shape[1], dtype=np.float64)
    within = np.zeros(matrix.shape[1], dtype=np.float64)
    for class_id in classes:
        group = matrix[labels == class_id]
        group_mean = group.mean(axis=0)
        between += len(group) * np.square(group_mean - overall)
        within += np.square(group - group_mean).sum(axis=0)
    numerator = between / max(1, len(classes) - 1)
    denominator = within / max(1, len(labels) - len(classes))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


@dataclass
class FoldFeatureSelector:
    """Impute, winsorize, scale, and select inside one training fold only."""

    max_features: int = 96
    max_missing_fraction: float = 0.35
    correlation_threshold: float = 0.975
    cn_focus: float = 0.45
    min_features_per_modality: int = 12

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        task: SelectionTask = "multiclass",
        positive_class: int | None = None,
    ) -> "FoldFeatureSelector":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FoldFeatureSelector requires a pandas DataFrame")
        target = np.asarray(y, dtype=np.int64)
        if len(X) != len(target):
            raise ValueError("Feature and label row counts differ")
        if np.unique(target).size < 2:
            raise ValueError("Feature selection requires at least two classes")

        frame = X.replace([np.inf, -np.inf], np.nan).copy()
        missing_ok = frame.isna().mean() <= float(self.max_missing_fraction)
        variable = frame.nunique(dropna=True) > 1
        self.base_columns_ = frame.columns[missing_ok & variable].tolist()
        if not self.base_columns_:
            raise ValueError("No usable feature remains after missingness/variance filtering")
        frame = frame[self.base_columns_]

        self.medians_ = frame.median(axis=0).fillna(0.0)
        filled = frame.fillna(self.medians_).astype(np.float64)
        self.lower_ = filled.quantile(0.01)
        self.upper_ = filled.quantile(0.99)
        clipped = filled.clip(self.lower_, self.upper_, axis=1)
        self.centers_ = clipped.median(axis=0)
        q25 = clipped.quantile(0.25)
        q75 = clipped.quantile(0.75)
        scales = q75 - q25
        fallback = clipped.std(axis=0)
        scales = scales.mask(scales.abs() < 1e-8, fallback).fillna(1.0)
        self.scales_ = scales.mask(scales.abs() < 1e-8, 1.0)
        standardized = (clipped - self.centers_) / self.scales_

        # CN can be a central "normal range" while MCI and DEM move in opposite
        # directions.  Add absolute distance from a reference fitted only on
        # the current fold's CN subjects.  The MCI-vs-DEM stage has no CN rows,
        # so that stage intentionally keeps only the signed features.
        self.cn_reference_enabled_ = task != "mci_vs_dem"
        if self.cn_reference_enabled_:
            cn_rows = target == 0
            if int(cn_rows.sum()) < 2:
                raise ValueError("CN-reference transformation needs at least two fold CN subjects")
            cn_frame = clipped.loc[cn_rows]
            self.cn_centers_ = cn_frame.median(axis=0).fillna(self.centers_)
            cn_q25 = cn_frame.quantile(0.25)
            cn_q75 = cn_frame.quantile(0.75)
            cn_scales = cn_q75 - cn_q25
            cn_scales = cn_scales.mask(cn_scales.abs() < 1e-8, self.scales_).fillna(1.0)
            self.cn_scales_ = cn_scales.mask(cn_scales.abs() < 1e-8, 1.0)
            cn_absolute = ((clipped - self.cn_centers_) / self.cn_scales_).abs()
            cn_absolute.columns = [f"cn_abs__{column}" for column in cn_absolute.columns]
            standardized = pd.concat([standardized, cn_absolute], axis=1)
        self.model_columns_ = standardized.columns.tolist()

        values = standardized.to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            f_scores = _anova_f_scores(values, target)
        task_rank = _rank01(f_scores)

        if task == "cn_vs_impaired":
            cn_target = target.astype(np.int64)
        elif task == "mci_vs_dem":
            cn_target = target.astype(np.int64)
        elif task == "one_vs_rest":
            if positive_class is None:
                raise ValueError("one_vs_rest selection needs positive_class")
            cn_target = (target == int(positive_class)).astype(np.int64)
        else:
            cn_target = (target != 0).astype(np.int64)

        binary_effect = _binary_auc_effects(values, cn_target)
        binary_rank = _rank01(binary_effect)
        if task == "multiclass":
            focus = float(np.clip(self.cn_focus, 0.0, 1.0))
            combined = (1.0 - focus) * task_rank + focus * binary_rank
        else:
            combined = 0.25 * task_rank + 0.75 * binary_rank

        columns = np.asarray(self.model_columns_, dtype=object)
        ranked_indices = np.argsort(-combined, kind="mergesort").tolist()
        centered_values = values - values.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.square(centered_values).sum(axis=0, keepdims=True))
        unit_columns = np.divide(
            centered_values,
            norms,
            out=np.zeros_like(centered_values),
            where=norms > 1e-12,
        )
        selected_indices: list[int] = []
        selected_set: set[int] = set()

        # Give activity and sleep a minimum opportunity before global ranking.
        # The quota is based on fold-local scores, never on a global EDA list.
        quota = min(int(self.min_features_per_modality), max(1, self.max_features // 4))
        for modality in ("activity", "sleep"):
            modality_indices = [
                index
                for index in ranked_indices
                if str(columns[index]).removeprefix("cn_abs__").startswith(modality)
            ]
            for candidate_index in modality_indices:
                if sum(
                    str(columns[index]).removeprefix("cn_abs__").startswith(modality)
                    for index in selected_indices
                ) >= quota:
                    break
                if self._not_redundant(unit_columns, selected_indices, candidate_index):
                    selected_indices.append(candidate_index)
                    selected_set.add(candidate_index)

        for candidate_index in ranked_indices:
            if len(selected_indices) >= int(self.max_features):
                break
            if candidate_index in selected_set:
                continue
            if self._not_redundant(unit_columns, selected_indices, candidate_index):
                selected_indices.append(candidate_index)
                selected_set.add(candidate_index)
        if not selected_indices:
            raise ValueError("Correlation pruning removed every feature")
        selected = [str(columns[index]) for index in selected_indices]

        score_by_column = {
            str(columns[index]): {
                "combined_rank_score": float(combined[index]),
                "task_f_score": float(np.nan_to_num(f_scores[index])),
                "binary_auc_effect": float(binary_effect[index]),
                "family": feature_family(str(columns[index])),
            }
            for index in range(len(columns))
        }
        self.selected_columns_ = selected
        self.selection_task_ = task
        self.positive_class_ = positive_class
        self.score_by_column_ = score_by_column
        return self

    def _not_redundant(
        self,
        unit_columns: np.ndarray,
        selected_indices: list[int],
        candidate_index: int,
    ) -> bool:
        if not selected_indices:
            return True
        correlations = np.abs(
            unit_columns[:, selected_indices].T @ unit_columns[:, candidate_index]
        )
        maximum = float(np.max(correlations)) if correlations.size else 0.0
        return bool(maximum < float(self.correlation_threshold))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        missing = sorted(set(self.base_columns_) - set(X.columns))
        if missing:
            raise ValueError(f"Input is missing fitted feature(s): {missing[:10]}")
        frame = X[self.base_columns_].replace([np.inf, -np.inf], np.nan)
        frame = frame.fillna(self.medians_).astype(np.float64)
        frame = frame.clip(self.lower_, self.upper_, axis=1)
        standardized = (frame - self.centers_) / self.scales_
        if self.cn_reference_enabled_:
            cn_absolute = ((frame - self.cn_centers_) / self.cn_scales_).abs()
            cn_absolute.columns = [f"cn_abs__{column}" for column in cn_absolute.columns]
            standardized = pd.concat([standardized, cn_absolute], axis=1)
        values = standardized[self.selected_columns_].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite value survived fold preprocessing")
        return values

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        task: SelectionTask = "multiclass",
        positive_class: int | None = None,
    ) -> np.ndarray:
        return self.fit(X, y, task=task, positive_class=positive_class).transform(X)

    def _check_fitted(self) -> None:
        if not hasattr(self, "selected_columns_"):
            raise RuntimeError("FoldFeatureSelector must be fitted before transform")

    @property
    def selected_feature_names(self) -> list[str]:
        self._check_fitted()
        return list(self.selected_columns_)

    def manifest(self) -> dict:
        self._check_fitted()
        return {
            "selection_task": self.selection_task_,
            "positive_class": self.positive_class_,
            "max_features": int(self.max_features),
            "selected_feature_count": int(len(self.selected_columns_)),
            "selected_features": list(self.selected_columns_),
            "selected_feature_scores": {
                name: self.score_by_column_[name] for name in self.selected_columns_
            },
            "preprocessing_fit_scope": "current training fold only",
            "imputation": "training-fold median",
            "clipping": "training-fold 1st/99th percentile",
            "scaling": "training-fold median/IQR",
            "cn_reference_absolute_deviation": bool(self.cn_reference_enabled_),
            "cn_reference_fit_scope": (
                "current training fold CN subjects only"
                if self.cn_reference_enabled_
                else "not used for MCI-vs-DEM stage"
            ),
        }
