"""Candidate learners.

Every learner is a plain ``fit(X, y) -> predict_proba(X)[:, 1]`` object built
fresh per fold.  Optional gradient-boosting libraries are probed once; a missing
library removes its candidates and is reported, never silently swapped for a
different model (a silent substitution would make the saved champion
irreproducible).

Preprocessing is intentionally NOT baked in here: ``engine.py`` fits imputation,
winsorization, scaling and feature screening on the training fold only and hands
this module an already-clean matrix.  Keeping that split is what makes the
fold-local contract auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from typing import Any, Mapping

import numpy as np

__all__ = [
    "Candidate",
    "available_families",
    "build_candidates",
    "fit_predict",
    "needs_scaling",
]


@dataclass(frozen=True)
class Candidate:
    """One (family, hyperparameter, view) combination to evaluate."""

    name: str
    family: str
    view: str
    params: Mapping[str, Any] = field(default_factory=dict)
    top_k: int | None = None

    @property
    def requires_scaling(self) -> bool:
        return needs_scaling(self.family)


#: Distance/gradient based learners need standardized inputs; trees do not.
_SCALED_FAMILIES = frozenset({"logreg", "svm_rbf"})


def needs_scaling(family: str) -> bool:
    return family in _SCALED_FAMILIES


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


_FAMILY_MODULES = {
    "logreg": "sklearn",
    "svm_rbf": "sklearn",
    "extratrees": "sklearn",
    "randomforest": "sklearn",
    "hgb": "sklearn",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
    "xgboost": "xgboost",
    "ydf_oblique": "ydf",
}


def available_families(requested: tuple[str, ...]) -> tuple[list[str], dict[str, str]]:
    """Split requested families into usable ones and a reason map for the rest."""

    usable: list[str] = []
    skipped: dict[str, str] = {}
    for family in requested:
        module = _FAMILY_MODULES.get(family)
        if module is None:
            skipped[family] = "unknown family"
        elif _module_available(module):
            usable.append(family)
        else:
            skipped[family] = f"{module} not installed"
    return usable, skipped


def build_candidates(
    families: tuple[str, ...],
    views: tuple[str, ...],
    *,
    logreg_c_grid: tuple[float, ...],
    svm_c_grid: tuple[float, ...],
    svm_gamma_grid: tuple[float, ...],
    top_k_grid: tuple[int, ...],
) -> list[Candidate]:
    """Enumerate the full candidate space (family x params x view x top_k)."""

    candidates: list[Candidate] = []

    def add(name: str, family: str, view: str, params: Mapping[str, Any], top_k: int | None) -> None:
        suffix = "" if top_k is None else f"__k{top_k}"
        candidates.append(
            Candidate(name=f"{name}__{view}{suffix}", family=family, view=view, params=dict(params), top_k=top_k)
        )

    for view in views:
        # Screening only matters for wide views; narrow MMSE views use all columns.
        k_options: tuple[int | None, ...] = (
            (None,) if view in {"mmse_core", "mmse_plus"} else (None, *top_k_grid)
        )
        for top_k in k_options:
            for family in families:
                if family == "logreg":
                    for c in logreg_c_grid:
                        add(f"lr_c{c:g}", family, view, {"C": float(c)}, top_k)
                elif family == "svm_rbf":
                    for c in svm_c_grid:
                        for gamma in svm_gamma_grid:
                            add(
                                f"svm_c{c:g}_g{gamma:g}",
                                family,
                                view,
                                {"C": float(c), "gamma": float(gamma)},
                                top_k,
                            )
                elif family == "extratrees":
                    add("et_d6", family, view, {"max_depth": 6, "n_estimators": 600}, top_k)
                    add("et_full", family, view, {"max_depth": None, "n_estimators": 600}, top_k)
                elif family == "randomforest":
                    add("rf_d6", family, view, {"max_depth": 6, "n_estimators": 600}, top_k)
                    add("rf_d10", family, view, {"max_depth": 10, "n_estimators": 600}, top_k)
                elif family == "hgb":
                    add("hgb_d3", family, view, {"max_depth": 3, "learning_rate": 0.05}, top_k)
                elif family == "lightgbm":
                    add(
                        "lgbm_leaf7",
                        family,
                        view,
                        {"num_leaves": 7, "learning_rate": 0.05, "n_estimators": 400, "max_depth": 3},
                        top_k,
                    )
                    add(
                        "lgbm_leaf15",
                        family,
                        view,
                        {"num_leaves": 15, "learning_rate": 0.03, "n_estimators": 600, "max_depth": 4},
                        top_k,
                    )
                elif family == "catboost":
                    add("cat_d4", family, view, {"depth": 4, "learning_rate": 0.04, "iterations": 500}, top_k)
                elif family == "xgboost":
                    add(
                        "xgb_d3",
                        family,
                        view,
                        {"max_depth": 3, "learning_rate": 0.04, "n_estimators": 500},
                        top_k,
                    )
                elif family == "ydf_oblique":
                    # The strongest single Google model evidence in the repo
                    # (Binary_Google_YDF_AUC): sparse-oblique GBT.
                    add(
                        "ydf_obl_d3",
                        family,
                        view,
                        {
                            "num_trees": 600,
                            "max_depth": 3,
                            "shrinkage": 0.05,
                            "min_examples": 12,
                            "subsample": 0.7,
                            "l2_regularization": 0.5,
                        },
                        top_k,
                    )
                    add(
                        "ydf_obl_d5",
                        family,
                        view,
                        {
                            "num_trees": 600,
                            "max_depth": 5,
                            "shrinkage": 0.08,
                            "min_examples": 20,
                            "subsample": 0.6,
                            "l2_regularization": 0.0,
                        },
                        top_k,
                    )
    return candidates


def _class_weight_dict(y: np.ndarray) -> dict[int, float]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(float)
    total = float(counts.sum())
    return {c: (total / (2.0 * counts[c])) if counts[c] > 0 else 1.0 for c in (0, 1)}


def fit_predict(
    candidate: Candidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    balanced: bool = True,
    feature_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the training fold and score both folds.

    Returns ``(train_scores, test_scores)``. The training scores are needed as
    the ECDF reference for rank normalization, so that normalization never uses
    the held-out batch's own distribution.
    """

    family = candidate.family
    params = dict(candidate.params)

    if family == "logreg":
        from sklearn.linear_model import LogisticRegression

        # `penalty` is left at its default: it is "l2" on every supported
        # version and passing it explicitly is deprecated from sklearn 1.8.
        model = LogisticRegression(
            C=params.get("C", 1.0),
            solver="lbfgs",
            max_iter=5000,
            class_weight="balanced" if balanced else None,
            random_state=seed,
        )
    elif family == "svm_rbf":
        from sklearn.svm import SVC

        model = SVC(
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
            kernel="rbf",
            probability=True,
            class_weight="balanced" if balanced else None,
            random_state=seed,
        )
    elif family == "extratrees":
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(
            n_estimators=params.get("n_estimators", 600),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=2,
            class_weight="balanced" if balanced else None,
            random_state=seed,
            n_jobs=1,
        )
    elif family == "randomforest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 600),
            max_depth=params.get("max_depth", 6),
            min_samples_leaf=2,
            class_weight="balanced" if balanced else None,
            random_state=seed,
            n_jobs=1,
        )
    elif family == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=params.get("n_estimators", 300),
            min_samples_leaf=10,
            l2_regularization=1.0,
            class_weight="balanced" if balanced else None,
            random_state=seed,
        )
    elif family == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            num_leaves=params.get("num_leaves", 7),
            learning_rate=params.get("learning_rate", 0.05),
            n_estimators=params.get("n_estimators", 400),
            max_depth=params.get("max_depth", 3),
            min_child_samples=10,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            class_weight="balanced" if balanced else None,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    elif family == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            depth=params.get("depth", 4),
            learning_rate=params.get("learning_rate", 0.04),
            iterations=params.get("iterations", 500),
            l2_leaf_reg=4.0,
            auto_class_weights="Balanced" if balanced else None,
            random_state=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
        )
    elif family == "xgboost":
        from xgboost import XGBClassifier

        weights = _class_weight_dict(y_train)
        model = XGBClassifier(
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.04),
            n_estimators=params.get("n_estimators", 500),
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=(weights[1] / weights[0]) if balanced else 1.0,
            eval_metric="auc",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    elif family == "ydf_oblique":
        return _fit_predict_ydf(
            params, X_train, y_train, X_test, seed=seed, feature_names=feature_names
        )
    else:
        raise ValueError(f"Unknown family: {family}")

    model.fit(X_train, y_train)
    return (
        np.asarray(model.predict_proba(X_train)[:, 1], dtype=float),
        np.asarray(model.predict_proba(X_test)[:, 1], dtype=float),
    )


def _fit_predict_ydf(
    params: Mapping[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    feature_names: tuple[str, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Yggdrasil sparse-oblique GBT via its pandas interface."""

    import pandas as pd
    import ydf

    names = list(feature_names) if feature_names else [f"f{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train, columns=names)
    train_df["__label__"] = np.asarray(y_train, dtype=np.int64)
    test_df = pd.DataFrame(X_test, columns=names)

    learner = ydf.GradientBoostedTreesLearner(
        label="__label__",
        task=ydf.Task.CLASSIFICATION,
        num_trees=int(params.get("num_trees", 600)),
        max_depth=int(params.get("max_depth", 3)),
        shrinkage=float(params.get("shrinkage", 0.05)),
        min_examples=int(params.get("min_examples", 12)),
        subsample=float(params.get("subsample", 0.7)),
        l2_regularization=float(params.get("l2_regularization", 0.5)),
        split_axis="SPARSE_OBLIQUE",
        sparse_oblique_normalization="STANDARD_DEVIATION",
        sparse_oblique_num_projections_exponent=1.5,
        sparse_oblique_projection_density_factor=3.0,
        random_seed=int(seed),
    )
    model = learner.train(train_df, verbose=0)
    return (
        np.asarray(model.predict(train_df.drop(columns=["__label__"])), dtype=float),
        np.asarray(model.predict(test_df), dtype=float),
    )
