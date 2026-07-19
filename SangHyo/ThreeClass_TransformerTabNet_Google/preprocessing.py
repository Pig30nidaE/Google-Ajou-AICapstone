"""Fold-local numeric preprocessing and supervised feature selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif


@dataclass
class FoldPreprocessor:
    """A small-data preprocessor that is fitted on one training fold only.

    MMSE answer features are retained in the clinical-assisted mode.  The
    remaining wearable features are ranked with ANOVA F scores on the fold's
    training subjects, then near-duplicates are pruned by correlation.
    """

    max_features: int = 96
    max_missing_fraction: float = 0.40
    correlation_threshold: float = 0.985

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "FoldPreprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FoldPreprocessor expects a pandas DataFrame")
        frame = X.replace([np.inf, -np.inf], np.nan).copy()
        missing_ok = frame.isna().mean() <= self.max_missing_fraction
        variable = frame.nunique(dropna=True) > 1
        self.base_columns_ = frame.columns[missing_ok & variable].tolist()
        if not self.base_columns_:
            raise ValueError("No usable feature remains after missingness/variance filtering")
        frame = frame[self.base_columns_]

        self.medians_ = frame.median(axis=0).fillna(0.0)
        filled = frame.fillna(self.medians_)
        self.lower_ = filled.quantile(0.01)
        self.upper_ = filled.quantile(0.99)
        clipped = filled.clip(self.lower_, self.upper_, axis=1)
        self.centers_ = clipped.median(axis=0)
        q25 = clipped.quantile(0.25)
        q75 = clipped.quantile(0.75)
        scale = q75 - q25
        fallback = clipped.std(axis=0)
        scale = scale.mask(scale.abs() < 1e-8, fallback).fillna(1.0)
        scale = scale.mask(scale.abs() < 1e-8, 1.0)
        self.scales_ = scale
        standardized = (clipped - self.centers_) / self.scales_

        mandatory = [c for c in standardized if c.startswith("mmse__")]
        candidates = [c for c in standardized if c not in mandatory]
        scores = np.zeros(len(candidates), dtype=float)
        if candidates:
            with np.errstate(divide="ignore", invalid="ignore"):
                scores, _ = f_classif(standardized[candidates].to_numpy(), y)
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=1e12, neginf=-np.inf)
        ranked = [
            candidates[i]
            for i in np.argsort(-scores, kind="mergesort")
        ]

        selected = list(mandatory)
        remaining_slots = max(0, self.max_features - len(selected))
        for col in ranked:
            if remaining_slots <= 0:
                break
            if selected:
                correlations = standardized[selected].corrwith(standardized[col]).abs()
                if correlations.max(skipna=True) >= self.correlation_threshold:
                    continue
            selected.append(col)
            remaining_slots -= 1
        if not selected:
            selected = ranked[: self.max_features]
        self.selected_columns_ = selected
        self.feature_scores_ = {
            col: float(scores[i]) for i, col in enumerate(candidates)
        }
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        missing = sorted(set(self.base_columns_) - set(X.columns))
        if missing:
            raise ValueError(f"Input is missing fitted columns: {missing[:10]}")
        frame = X[self.base_columns_].replace([np.inf, -np.inf], np.nan)
        frame = frame.fillna(self.medians_)
        frame = frame.clip(self.lower_, self.upper_, axis=1)
        frame = (frame - self.centers_) / self.scales_
        values = frame[self.selected_columns_].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite value survived fold preprocessing")
        return values

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _check_fitted(self) -> None:
        if not hasattr(self, "selected_columns_"):
            raise RuntimeError("FoldPreprocessor must be fitted before transform")

    @property
    def selected_feature_names(self) -> list[str]:
        self._check_fitted()
        return list(self.selected_columns_)

