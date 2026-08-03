"""Random Forest and XGBoost.

Random Forest hyperparameters are *not reported anywhere in either paper*
(``unresolved_questions.md`` Q2), so the defaults below are ours.  XGBoost's are
partially reported; the reported string is applied verbatim and the gaps
(``max_depth``, ``objective``, ``random_state``) are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Reported verbatim in Section 3.3.1 of both papers.
PAPER_XGB_REPORTED = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "eval_metric": "logloss",
}

#: Named in the search description but missing from the reported result string.
XGB_ASSUMED = {"max_depth": 6, "objective": "binary:logistic"}

#: No RF hyperparameter table is reported.  ``criterion='entropy'`` is the one
#: value for which the prose gives a clue: Section 3.3.1 says that each tree picks
#: splits by information gain.  The remaining values are conservative defaults.
RF_ASSUMED = {
    "n_estimators": 100,
    "criterion": "entropy",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": None,
}


class ModelDependencyError(ImportError):
    """Raised when a model's library is not installed in this environment."""


@dataclass
class TabularModel:
    """Uniform wrapper so the engine treats every learner the same way."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    reported_params: tuple[str, ...] = ()
    assumed_params: tuple[str, ...] = ()
    estimator: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabularModel":
        X = _as_2d(X)
        self.estimator.fit(X, np.asarray(y).astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability, shape (n,)."""
        X = _as_2d(X)
        proba = self.estimator.predict_proba(X)
        return np.asarray(proba)[:, 1].astype(np.float64)

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": self.name, "params": self.params, "estimator": self.estimator}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TabularModel":
        import joblib

        blob = joblib.load(Path(path))
        return cls(name=blob["name"], params=blob["params"], estimator=blob["estimator"])

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "params_reported_in_paper": list(self.reported_params),
            "params_assumed_by_us": list(self.assumed_params),
        }


def _as_2d(X: np.ndarray) -> np.ndarray:
    """Flatten a sequence tensor if a tree model is given one."""
    X = np.asarray(X, dtype=np.float64)
    return X.reshape(X.shape[0], -1) if X.ndim == 3 else X


def build_random_forest(*, seed: int = 42, overrides: dict[str, Any] | None = None) -> TabularModel:
    from sklearn.ensemble import RandomForestClassifier

    params = dict(RF_ASSUMED)
    params.update(overrides or {})
    params["random_state"] = seed
    params.setdefault("n_jobs", -1)
    return TabularModel(
        name="random_forest",
        params=params,
        reported_params=(),  # the paper reports none
        assumed_params=tuple(sorted(params)),
        estimator=RandomForestClassifier(**params),
    )


def build_xgboost(*, seed: int = 42, overrides: dict[str, Any] | None = None) -> TabularModel:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ModelDependencyError(
            "xgboost is required for the XGBoost arm. Install it via "
            "requirements_colab.txt, or drop 'xgboost' from config.models."
        ) from error

    params = dict(PAPER_XGB_REPORTED)
    params.update(XGB_ASSUMED)
    params.update(overrides or {})
    params["random_state"] = seed
    # `use_label_encoder=False` appears in the paper but was removed in xgboost 2.x;
    # passing it would raise, so it is intentionally not forwarded.
    params.pop("use_label_encoder", None)
    return TabularModel(
        name="xgboost",
        params=params,
        reported_params=tuple(sorted(PAPER_XGB_REPORTED)),
        assumed_params=tuple(sorted(set(XGB_ASSUMED) | {"random_state"})),
        estimator=XGBClassifier(**params),
    )
