"""The four non-temporal comparison models: LR, SVM, RF and XGBoost.

Hong et al. tuned these with H2O AutoML 3.46.0.1 (§4.2, explicitly versioned).
H2O is a heavy JVM dependency, so it is an optional backend: ``h2o.py`` is used
when the config asks for it and the package is importable, and otherwise these
scikit-learn / XGBoost equivalents run instead.  Which backend produced a number
is always recorded in the result, because the two are not interchangeable.

The paper does not say how a 3-to-5-day window was flattened into a row for these
models, so the representation is a config choice (A-09).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

BASELINE_MODELS = ("logistic_regression", "svm", "random_forest", "xgboost")


@dataclass
class BaselineConfig:
    name: str
    params: dict[str, Any]
    seed: int = 42
    class_weight: bool = False

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params),
                "seed": self.seed, "class_weight": self.class_weight}


DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "max_iter": 2000},
    "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
    "random_forest": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    "xgboost": {
        "n_estimators": 300, "max_depth": 4, "learning_rate": 0.1,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
}

#: Deliberately small.  The nested inner CV pays for every candidate, and the
#: cohort is 174 subjects -- a wide search would fit inner-fold noise (see the
#: repository's own MaxAUC_Tuned result, where a 10.6-hour search made things worse).
SEARCH_SPACES: dict[str, list[dict[str, Any]]] = {
    "logistic_regression": [{"C": c, "max_iter": 2000} for c in (0.1, 1.0, 10.0)],
    "svm": [{"C": c, "kernel": "rbf", "gamma": "scale"} for c in (0.1, 1.0, 10.0)],
    "random_forest": [
        {"n_estimators": 300, "max_depth": depth, "min_samples_leaf": leaf}
        for depth, leaf in ((None, 1), (6, 5), (10, 2))
    ],
    "xgboost": [
        {"n_estimators": 300, "max_depth": depth, "learning_rate": lr,
         "subsample": 0.8, "colsample_bytree": 0.8}
        for depth, lr in ((3, 0.1), (4, 0.05), (6, 0.1))
    ],
}


def build_baseline(config: BaselineConfig) -> Any:
    """Instantiate an unfitted estimator with a ``predict_proba``-shaped API."""
    name, params, seed = config.name, dict(config.params), config.seed
    weight = "balanced" if config.class_weight else None

    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=seed, class_weight=weight, **params)

    if name == "svm":
        from sklearn.svm import SVC

        # probability=True is required: every metric in this package is computed
        # from scores, not from hard labels.
        return SVC(probability=True, random_state=seed, class_weight=weight, **params)

    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            random_state=seed, n_jobs=-1, class_weight=weight, **params
        )

    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            **params,
        )

    raise ValueError(f"unknown baseline {name!r}; expected one of {BASELINE_MODELS}")


def fit_baseline(
    config: BaselineConfig, X: np.ndarray, y: np.ndarray
) -> tuple[Any, dict[str, Any]]:
    model = build_baseline(config)
    if len(np.unique(y)) < 2:
        raise ValueError(
            f"baseline {config.name!r} received a single-class training set; "
            "check the split and the undersampling ratio"
        )
    if config.name == "xgboost" and config.class_weight:
        counts = np.bincount(y.astype(int), minlength=2)
        model.set_params(scale_pos_weight=float(counts[0]) / max(counts[1], 1))
    model.fit(X, y)
    return model, {"model": config.name, "n_train_rows": int(len(X)),
                   "n_features": int(X.shape[1]), **config.describe()}


def predict_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if len(X) == 0:
        return np.empty(0, dtype=np.float64)
    proba = model.predict_proba(X)
    return np.asarray(proba)[:, 1].astype(np.float64)


def search_space(name: str, *, enabled: bool = True) -> list[dict[str, Any]]:
    """Candidates for the inner CV; a single default when tuning is off."""
    if not enabled:
        return [dict(DEFAULT_PARAMS[name])]
    return [dict(params) for params in SEARCH_SPACES[name]]


def available_backends() -> dict[str, bool]:
    import importlib.util

    return {
        "sklearn": importlib.util.find_spec("sklearn") is not None,
        "xgboost": importlib.util.find_spec("xgboost") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "h2o": importlib.util.find_spec("h2o") is not None,
        "shap": importlib.util.find_spec("shap") is not None,
    }
