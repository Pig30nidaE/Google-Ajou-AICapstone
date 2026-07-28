"""Fold-local preprocessing and strong tabular learners."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import ModelSpec


class FiniteToNan:
    """Scikit-learn transformer replacing infinity without learning statistics."""

    def fit(self, X, y=None):
        del y
        values = np.asarray(X)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=np.float64).copy()
        values[~np.isfinite(values)] = np.nan
        return values

    def get_params(self, deep=True):
        del deep
        return {}

    def set_params(self, **params):
        if params:
            raise ValueError(f"FiniteToNan has no parameters: {params}")
        return self


class QuantileClipper:
    """Winsorization bounds fitted only on the current training fold."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        del y
        values = np.asarray(X, dtype=np.float64)
        self.lower_bounds_ = np.quantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.quantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(
            np.asarray(X, dtype=np.float64),
            self.lower_bounds_,
            self.upper_bounds_,
        )

    def get_params(self, deep=True):
        del deep
        return {"lower": self.lower, "upper": self.upper}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SafeSelectKBest:
    """SelectKBest wrapper that clips k to the fold's actual feature count."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, X, y):
        from sklearn.feature_selection import SelectKBest, f_classif

        values = np.asarray(X)
        resolved = min(max(1, int(self.k)), values.shape[1])
        self.selector_ = SelectKBest(score_func=f_classif, k=resolved).fit(values, y)
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self):
        return self.selector_.get_support()

    def get_params(self, deep=True):
        del deep
        return {"k": self.k}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


def select_spec_columns(
    X: np.ndarray,
    feature_names: tuple[str, ...],
    spec: ModelSpec,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Apply only predeclared name-based anchors before fold-local pipelines."""

    values = np.asarray(X, dtype=np.float64)
    if not spec.fixed_feature_suffixes:
        return values, feature_names
    selected: list[int] = []
    for suffix in spec.fixed_feature_suffixes:
        matches = [
            index
            for index, name in enumerate(feature_names)
            if str(name).endswith(str(suffix))
        ]
        if len(matches) != 1:
            raise KeyError(
                f"{spec.name}: suffix {suffix!r} matched {len(matches)} features"
            )
        selected.append(matches[0])
    return values[:, selected], tuple(feature_names[index] for index in selected)


def _sampler(name: str, seed: int):
    if name == "none":
        return None
    if name == "random_over":
        from imblearn.over_sampling import RandomOverSampler

        return RandomOverSampler(random_state=seed, sampling_strategy=0.5)
    if name == "smote":
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=seed, k_neighbors=2, sampling_strategy=0.5)
    if name == "adasyn":
        from imblearn.over_sampling import ADASYN

        return ADASYN(random_state=seed, n_neighbors=2, sampling_strategy=0.5)
    raise ValueError(f"Unknown fold-local resampler: {name}")


def _classifier(
    spec: ModelSpec,
    params: Mapping[str, Any],
    *,
    seed: int,
    n_jobs: int,
    positive_weight: float,
):
    name = spec.name
    has_external_resampling = str(params.get("resampler", "none")) != "none"
    balanced_class_weight = None if has_external_resampling else "balanced"
    if name in {"univariate_logreg", "elastic_logreg"}:
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=float(params.get("C", 0.2)),
            penalty=("l2" if name == "univariate_logreg" else "elasticnet"),
            l1_ratio=(None if name == "univariate_logreg" else float(params.get("l1_ratio", 0.25))),
            solver="liblinear" if name == "univariate_logreg" else "saga",
            class_weight=balanced_class_weight,
            max_iter=10000,
            random_state=seed,
            n_jobs=n_jobs if name != "univariate_logreg" else None,
        )
    if name == "rbf_svm":
        from sklearn.svm import SVC

        return SVC(
            C=float(params.get("C", 1.0)),
            gamma=params.get("gamma", "scale"),
            kernel="rbf",
            probability=False,
            class_weight=balanced_class_weight,
            random_state=seed,
            cache_size=2048,
        )
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=(
                None if has_external_resampling else "balanced_subsample"
            ),
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            class_weight=(
                None if has_external_resampling else "balanced_subsample"
            ),
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=float(params["learning_rate"]),
            max_iter=int(params["max_iter"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            random_state=seed,
        )
    if name == "balanced_random_forest":
        from imblearn.ensemble import BalancedRandomForestClassifier

        return BalancedRandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"],
            replacement=True,
            sampling_strategy="all",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "easy_ensemble":
        from imblearn.ensemble import EasyEnsembleClassifier

        return EasyEnsembleClassifier(
            n_estimators=int(params["n_estimators"]),
            n_jobs=n_jobs,
            random_state=seed,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            objective="binary",
            verbosity=-1,
            random_state=seed,
            n_jobs=n_jobs,
            scale_pos_weight=(
                1.0 if has_external_resampling else positive_weight
            ),
            subsample_freq=1,
            **{
                key: params[key]
                for key in (
                    "n_estimators",
                    "learning_rate",
                    "num_leaves",
                    "max_depth",
                    "min_child_samples",
                    "reg_alpha",
                    "reg_lambda",
                    "subsample",
                    "colsample_bytree",
                )
            },
        )
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
            scale_pos_weight=(
                1.0 if has_external_resampling else positive_weight
            ),
            **{
                key: params[key]
                for key in (
                    "n_estimators",
                    "learning_rate",
                    "max_depth",
                    "min_child_weight",
                    "subsample",
                    "colsample_bytree",
                    "reg_alpha",
                    "reg_lambda",
                )
            },
        )
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            verbose=False,
            allow_writing_files=False,
            auto_class_weights=(
                None if has_external_resampling else "Balanced"
            ),
            random_seed=seed,
            thread_count=n_jobs,
            eval_metric="AUC",
            **{
                key: params[key]
                for key in (
                    "iterations",
                    "learning_rate",
                    "depth",
                    "l2_leaf_reg",
                    "random_strength",
                )
            },
        )
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return MLPClassifier(
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            alpha=float(params["alpha"]),
            learning_rate_init=float(params["learning_rate_init"]),
            activation="relu",
            early_stopping=False,
            max_iter=800,
            random_state=seed,
        )
    raise ValueError(f"Unsupported tabular model: {name}")


def build_tabular_estimator(
    spec: ModelSpec,
    params: Mapping[str, Any],
    *,
    seed: int,
    n_jobs: int,
    positive_weight: float,
):
    """Construct a pipeline whose every learned step is fold-local."""

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    try:
        from imblearn.pipeline import Pipeline
    except ImportError:
        from sklearn.pipeline import Pipeline
        if str(params.get("resampler", "none")) != "none":
            raise ModuleNotFoundError(
                "imbalanced-learn is required for the selected resampler"
            )

    steps: list[tuple[str, Any]] = [
        ("finite", FiniteToNan()),
        (
            "impute",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        ),
        ("clip", QuantileClipper()),
    ]
    if int(params.get("top_k", 0)) > 0:
        steps.append(("select", SafeSelectKBest(int(params["top_k"]))))
    steps.append(("scale", RobustScaler(quantile_range=(10.0, 90.0))))
    sampler = _sampler(str(params.get("resampler", "none")), seed)
    if sampler is not None:
        steps.append(("resample", sampler))
    steps.append(
        (
            "model",
            _classifier(
                spec,
                params,
                seed=seed,
                n_jobs=n_jobs,
                positive_weight=positive_weight,
            ),
        )
    )
    return Pipeline(steps)


def predict_positive(estimator, X: np.ndarray) -> np.ndarray:
    """Return P(Dem) and verify model class ordering."""

    from scipy.special import expit

    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
        if probabilities.ndim != 2 or 1 not in classes:
            raise ValueError("Classifier does not expose binary probability columns")
        result = probabilities[:, int(np.flatnonzero(classes == 1)[0])]
    elif hasattr(estimator, "decision_function"):
        result = expit(np.asarray(estimator.decision_function(X), dtype=np.float64))
    else:
        raise TypeError("Estimator exposes neither predict_proba nor decision_function")
    if not np.isfinite(result).all():
        raise ValueError("Model emitted non-finite predictions")
    return np.clip(result, 1e-7, 1.0 - 1e-7)


def suggest_tabular_params(
    trial,
    spec: ModelSpec,
    *,
    feature_selection_choices: tuple[int, ...],
    allow_random_over: bool,
    allow_smote: bool,
    allow_adasyn: bool,
) -> dict[str, Any]:
    """Optuna search space sized for nine positive development subjects."""

    params = dict(spec.fixed_params)
    if not spec.fixed_feature_suffixes:
        params["top_k"] = trial.suggest_categorical(
            "top_k", list(feature_selection_choices)
        )
    sampler_choices = ["none"]
    if allow_random_over:
        sampler_choices.append("random_over")
    if allow_smote:
        sampler_choices.append("smote")
    if allow_adasyn:
        sampler_choices.append("adasyn")
    # Balanced ensembles already own their resampling mechanism.
    if spec.name not in {"balanced_random_forest", "easy_ensemble"}:
        params["resampler"] = trial.suggest_categorical(
            "resampler", sampler_choices
        )
    if spec.name in {"univariate_logreg", "elastic_logreg", "rbf_svm"}:
        params["C"] = trial.suggest_float("C", 1e-3, 20.0, log=True)
    if spec.name == "elastic_logreg":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    if spec.name == "rbf_svm":
        params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    elif spec.name in {"extra_trees", "random_forest", "balanced_random_forest"}:
        params["max_depth"] = trial.suggest_int("max_depth", 2, 7)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 2, 10)
        params["max_features"] = trial.suggest_float("max_features", 0.25, 1.0)
    elif spec.name == "hist_gradient_boosting":
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        )
        params["max_leaf_nodes"] = trial.suggest_int("max_leaf_nodes", 3, 15)
        params["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 20)
        params["l2_regularization"] = trial.suggest_float(
            "l2_regularization", 0.1, 30.0, log=True
        )
    elif spec.name in {"lightgbm", "xgboost", "catboost"}:
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", 0.01, 0.12, log=True
        )
        if spec.name == "catboost":
            params["depth"] = trial.suggest_int("depth", 2, 5)
            params["l2_leaf_reg"] = trial.suggest_float(
                "l2_leaf_reg", 1.0, 30.0, log=True
            )
        else:
            params["max_depth"] = trial.suggest_int("max_depth", 2, 5)
            params["reg_alpha"] = trial.suggest_float(
                "reg_alpha", 0.1, 20.0, log=True
            )
            params["reg_lambda"] = trial.suggest_float(
                "reg_lambda", 1.0, 30.0, log=True
            )
    elif spec.name == "mlp":
        width = trial.suggest_categorical("width", [16, 32, 64])
        params["hidden_layer_sizes"] = (width, max(8, width // 2))
        params["alpha"] = trial.suggest_float("alpha", 1e-4, 0.2, log=True)
        params["learning_rate_init"] = trial.suggest_float(
            "learning_rate_init", 1e-4, 5e-3, log=True
        )
    return params


__all__ = [
    "build_tabular_estimator",
    "predict_positive",
    "select_spec_columns",
    "suggest_tabular_params",
]
