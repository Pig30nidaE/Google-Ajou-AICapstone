"""Downstream classifiers.  Gemini never classifies; these estimators do.

Two fixed configurations only (<modeling>, <hyperparameter_policy>):

``logreg``  median imputation -> standardisation -> L2 logistic regression with
            ``class_weight='balanced'``.  This mirrors the current strongest
            honest baseline family in the repository (``Binary_MMSE_MaxAUC``,
            regularised LR/SVM, OOF ROC-AUC 0.7658).
``gbdt``    a small, strongly regularised gradient-boosted tree.  LightGBM when
            available, otherwise scikit-learn's ``HistGradientBoostingClassifier``.
            The fallback is reported in the run report, never silent.

Every step lives inside one scikit-learn ``Pipeline`` so that imputation and
scaling are fitted on the training fold only (<data_leakage_rules> item 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = ["ModelSpec", "available_models", "build_model", "class_weights"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    implementation: str
    params: Mapping[str, Any]


def class_weights(y: np.ndarray) -> dict[int, float]:
    target = np.asarray(y, dtype=np.int64)
    counts = np.bincount(target, minlength=2).astype(float)
    total = float(counts.sum())
    return {
        label: (total / (2.0 * counts[label])) if counts[label] > 0 else 1.0
        for label in (0, 1)
    }


def _lightgbm_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("lightgbm") is not None


def available_models() -> dict[str, str]:
    return {
        "logreg": "sklearn.linear_model.LogisticRegression",
        "gbdt": (
            "lightgbm.LGBMClassifier"
            if _lightgbm_available()
            else "sklearn.ensemble.HistGradientBoostingClassifier"
        ),
    }


def build_model(name: str, params: Mapping[str, Any], *, seed: int) -> tuple[Any, ModelSpec]:
    """Return an unfitted pipeline plus the spec that is written to the report."""

    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        estimator = LogisticRegression(
            C=float(params.get("C", 1.0)),
            penalty=str(params.get("penalty", "l2")),
            max_iter=int(params.get("max_iter", 2000)),
            class_weight="balanced",
            solver="lbfgs",
            random_state=int(seed),
        )
        pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", estimator),
            ]
        )
        return pipeline, ModelSpec("logreg", "sklearn.linear_model.LogisticRegression", dict(params))

    if name == "gbdt":
        if _lightgbm_available():
            from lightgbm import LGBMClassifier

            estimator = LGBMClassifier(
                n_estimators=int(params.get("n_estimators", 300)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                num_leaves=int(params.get("num_leaves", 7)),
                min_child_samples=int(params.get("min_child_samples", 15)),
                subsample=float(params.get("subsample", 0.9)),
                subsample_freq=int(params.get("subsample_freq", 1)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                reg_lambda=float(params.get("reg_lambda", 1.0)),
                class_weight="balanced",
                random_state=int(seed),
                n_jobs=1,
                verbose=-1,
            )
            implementation = "lightgbm.LGBMClassifier"
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier

            estimator = HistGradientBoostingClassifier(
                max_iter=int(params.get("n_estimators", 300)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                max_leaf_nodes=int(params.get("num_leaves", 7)),
                min_samples_leaf=int(params.get("min_child_samples", 15)),
                l2_regularization=float(params.get("reg_lambda", 1.0)),
                class_weight="balanced",
                random_state=int(seed),
            )
            implementation = "sklearn.ensemble.HistGradientBoostingClassifier"
        # Trees handle NaN natively, but the imputer keeps both models on the
        # identical fold-local preprocessing contract.
        pipeline = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("model", estimator)]
        )
        return pipeline, ModelSpec("gbdt", implementation, dict(params))

    raise ValueError(f"Unknown model: {name}")
