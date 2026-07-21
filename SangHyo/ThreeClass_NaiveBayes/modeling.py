"""Fold-local preprocessing and Gaussian Naive Bayes search."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

if __package__:
    from .data import CLASS_NAMES
    from .evaluation import align_probabilities, training_selection_score
else:
    from data import CLASS_NAMES
    from evaluation import align_probabilities, training_selection_score


def build_pipeline() -> Pipeline:
    """Create one leakage-safe pipeline cloned independently in every fold."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(25.0, 75.0))),
            ("variance", VarianceThreshold(threshold=0.0)),
            ("selector", SelectKBest(score_func=f_classif, k=32)),
            ("model", GaussianNB()),
        ]
    )


def parameter_grid(feature_count: int, *, fast: bool) -> dict[str, list[Any]]:
    if feature_count < 1:
        raise ValueError("At least one input feature is required")
    if fast:
        requested_k = (min(32, feature_count),)
        smoothing = (1e-9,)
    else:
        requested_k = (8, 16, 24, 32, 48, 64, 96)
        smoothing = (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-4)
    k_values = sorted({int(value) for value in requested_k if value <= feature_count})
    if not k_values:
        k_values = [feature_count]
    uniform_prior = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
    return {
        "selector__k": k_values,
        "model__var_smoothing": list(smoothing),
        "model__priors": [None, uniform_prior],
    }


def validate_split_count(y: np.ndarray, n_splits: int, *, context: str) -> None:
    labels = np.asarray(y, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    if len(labels) == 0 or np.count_nonzero(counts) != len(CLASS_NAMES):
        raise ValueError(f"{context} requires all classes {CLASS_NAMES}; found {counts.tolist()}")
    if int(n_splits) < 2:
        raise ValueError(f"{context} n_splits must be at least 2")
    if int(counts.min()) < int(n_splits):
        raise ValueError(
            f"{context} n_splits={n_splits} exceeds the smallest class count "
            f"{int(counts.min())}"
        )


def fit_grid_search(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    inner_folds: int,
    seed: int,
    fast: bool,
    n_jobs: int,
) -> GridSearchCV:
    labels = np.asarray(y, dtype=np.int64)
    validate_split_count(labels, inner_folds, context="Inner CV")
    splitter = StratifiedKFold(
        n_splits=int(inner_folds),
        shuffle=True,
        random_state=int(seed),
    )
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=parameter_grid(X.shape[1], fast=fast),
        scoring=training_selection_score,
        cv=splitter,
        refit=True,
        n_jobs=int(n_jobs),
        error_score="raise",
        return_train_score=False,
    )
    search.fit(X, labels)
    return search


def predict_probabilities(estimator, X: pd.DataFrame) -> np.ndarray:
    return align_probabilities(estimator.predict_proba(X), estimator.classes_)


def selected_feature_names(
    fitted_pipeline: Pipeline,
    input_columns: list[str],
) -> list[str]:
    imputer = fitted_pipeline.named_steps["imputer"]
    variance = fitted_pipeline.named_steps["variance"]
    selector = fitted_pipeline.named_steps["selector"]
    names = np.asarray(imputer.get_feature_names_out(input_columns), dtype=object)
    names = names[np.asarray(variance.get_support(), dtype=bool)]
    names = names[np.asarray(selector.get_support(), dtype=bool)]
    return [str(name) for name in names]


def readable_best_params(best_params: dict[str, Any]) -> dict[str, Any]:
    readable: dict[str, Any] = {}
    for key, value in best_params.items():
        if key == "model__priors":
            readable[key] = "empirical" if value is None else "uniform"
        elif isinstance(value, np.generic):
            readable[key] = value.item()
        else:
            readable[key] = value
    return readable
