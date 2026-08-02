"""Google-native replacements for the report's four base learners.

The report ensembles LightGBM (Microsoft), CatBoost (Yandex), XGBoost (DMLC)
and scikit-learn's RandomForest.  This module reproduces the same *inductive
biases* using Google's YDF (Yet-another Decision Forest, the engine behind
TensorFlow Decision Forests):

===================  ==========================================================
report learner       YDF stand-in
===================  ==========================================================
LightGBM             ``GradientBoostedTreesLearner`` with leaf-wise growth
                     (``growing_strategy="BEST_FIRST_GLOBAL"``) -- LightGBM's
                     defining strategy, budgeted by ``max_num_nodes``.
XGBoost              ``GradientBoostedTreesLearner`` with level-wise growth
                     (``growing_strategy="LOCAL"``) budgeted by ``max_depth``.
CatBoost             ``GradientBoostedTreesLearner`` with sparse *oblique*
                     splits.  Oblique splits combine several features per node,
                     giving the decorrelated third opinion CatBoost supplied.
RandomForest         ``RandomForestLearner`` (bagged, out-of-bag scored).
===================  ==========================================================

The report's SHAP attribution is likewise replaced by YDF's built-in variable
importances, so feature ranking is Google-native too and needs no extra
dependency.

If ``ydf`` is unavailable (e.g. a laptop smoke run) each learner degrades to a
NaN-tolerant scikit-learn stand-in.  The engine actually used is recorded in
``engine_`` and surfaced in the run report, so a fallback run can never be
mistaken for a YDF run.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

try:  # pragma: no cover - import probe
    import ydf  # type: ignore

    YDF_AVAILABLE = True
    YDF_VERSION = getattr(ydf, "__version__", "unknown")
except Exception:  # pragma: no cover
    ydf = None  # type: ignore
    YDF_AVAILABLE = False
    YDF_VERSION = None

CLASS_NAMES = ("CN", "MCI_DEM")
POSITIVE_CLASS = CLASS_NAMES[1]

#: Preference order for YDF's importance keys; the first present one is used.
_IMPORTANCE_PREFERENCE = (
    "MEAN_DECREASE_IN_AUC_MCI_DEM_VS_OTHERS",
    "MEAN_DECREASE_IN_AUC_1_VS_OTHERS",
    "MEAN_DECREASE_IN_ACCURACY",
    "SUM_SCORE",
    "NUM_AS_ROOT",
    "NUM_NODES",
)

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[models] {message}")


# ------------------------------------------------------------ hyperparameters --
def default_params(kind: str) -> dict:
    """Report-faithful defaults (V29's tuned LightGBM values where stated)."""

    common = dict(num_trees=300, shrinkage=0.05, subsample=0.8,
                  num_candidate_attributes_ratio=0.6, l2_regularization=1.0,
                  min_examples=20)
    if kind == "gbt_leafwise":          # LightGBM stand-in: num_leaves=31, lr=0.05
        return dict(common, max_num_nodes=31, max_depth=-1)
    if kind == "gbt_depthwise":         # XGBoost stand-in: max_depth=6
        return dict(common, max_depth=6)
    if kind == "gbt_oblique":           # CatBoost stand-in: decorrelated splits
        return dict(common, max_depth=5, num_trees=250)
    if kind == "random_forest":
        return dict(num_trees=500, max_depth=16, min_examples=5,
                    num_candidate_attributes_ratio=0.4)
    raise ValueError(f"Unknown learner kind: {kind}")


#: The report's fixed soft-voting weights: .40 LGBM + .20 Cat + .20 XGB + .20 RF.
SOFT_VOTING_WEIGHTS = {
    "gbt_leafwise": 0.40,
    "gbt_oblique": 0.20,
    "gbt_depthwise": 0.20,
    "random_forest": 0.20,
}
ALL_KINDS = tuple(SOFT_VOTING_WEIGHTS)


def search_space(kind: str) -> dict[str, list]:
    """Small discrete space for the fold-local random search."""

    if kind == "random_forest":
        return {
            "num_trees": [300, 500],
            "max_depth": [12, 16, 24],
            "min_examples": [2, 5, 10],
            "num_candidate_attributes_ratio": [0.3, 0.4, 0.6],
        }
    space = {
        "num_trees": [200, 300, 500],
        "shrinkage": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "num_candidate_attributes_ratio": [0.4, 0.6, 0.8],
        "l2_regularization": [0.1, 1.0, 5.0],
        "min_examples": [5, 10, 20],
    }
    if kind == "gbt_leafwise":
        space["max_num_nodes"] = [15, 31, 63]
    else:
        space["max_depth"] = [3, 4, 6]
    return space


# ------------------------------------------------------------------ YDF build --
def _build_ydf_learner(kind: str, params: dict, seed: int):
    """Construct a YDF learner, degrading gracefully if a knob is unsupported."""

    common = dict(
        label="label", label_classes=list(CLASS_NAMES), random_seed=int(seed),
        num_threads=int(os.cpu_count() or 4),
    )

    if kind == "random_forest":
        cls = ydf.RandomForestLearner
        core = dict(
            num_trees=int(params["num_trees"]),
            max_depth=int(params["max_depth"]),
            min_examples=int(params["min_examples"]),
            num_candidate_attributes_ratio=float(params["num_candidate_attributes_ratio"]),
        )
        optional: dict = {}
    else:
        cls = ydf.GradientBoostedTreesLearner
        core = dict(
            loss="BINOMIAL_LOG_LIKELIHOOD",
            validation_ratio=0.0,          # no internal holdout: folds are ours
            early_stopping="NONE",
            num_trees=int(params["num_trees"]),
            min_examples=int(params["min_examples"]),
            shrinkage=float(params["shrinkage"]),
            subsample=float(params["subsample"]),
            num_candidate_attributes_ratio=float(params["num_candidate_attributes_ratio"]),
            l2_regularization=float(params["l2_regularization"]),
        )
        if kind == "gbt_leafwise":
            optional = dict(growing_strategy="BEST_FIRST_GLOBAL",
                            max_num_nodes=int(params["max_num_nodes"]),
                            max_depth=int(params.get("max_depth", -1)))
        elif kind == "gbt_depthwise":
            optional = dict(growing_strategy="LOCAL", max_depth=int(params["max_depth"]))
        elif kind == "gbt_oblique":
            optional = dict(growing_strategy="LOCAL", max_depth=int(params["max_depth"]),
                            split_axis="SPARSE_OBLIQUE",
                            sparse_oblique_normalization="STANDARD_DEVIATION")
        else:
            raise ValueError(f"Unknown learner kind: {kind}")

    for attempt in (dict(core, **optional, **common), dict(core, **common)):
        try:
            return cls(**attempt)
        except Exception as error:
            _warn_once(
                f"ydf_kwargs_{kind}",
                f"{kind}: YDF rejected optional kwargs ({type(error).__name__}); "
                "retrying with core hyperparameters only.",
            )
    # Last resort: the smallest configuration YDF must accept.
    return cls(label="label", label_classes=list(CLASS_NAMES), random_seed=int(seed))


# ------------------------------------------------------------------- wrapper --
class GoogleTreeModel:
    """Uniform ``fit``/``predict_proba``/``importances`` over YDF or sklearn."""

    def __init__(self, kind: str, params: dict | None = None, *, seed: int = 0) -> None:
        if kind not in ALL_KINDS:
            raise ValueError(f"Unknown learner kind: {kind}")
        self.kind = kind
        self.params = dict(params) if params else default_params(kind)
        self.seed = int(seed)
        self.engine_ = "ydf" if YDF_AVAILABLE else "sklearn_fallback"
        self.columns_: list[str] = []
        self.model_ = None

    # -- fit ------------------------------------------------------------------
    def fit(self, X, y) -> "GoogleTreeModel":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.columns_ = [f"f{i}" for i in range(X.shape[1])]

        if YDF_AVAILABLE:
            frame = pd.DataFrame(X, columns=self.columns_)
            frame["label"] = [CLASS_NAMES[int(v)] for v in y]
            learner = _build_ydf_learner(self.kind, self.params, self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model_ = learner.train(frame, verbose=0)
        else:
            self.model_ = self._fit_sklearn(X, y)
        return self

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

        if self.kind == "random_forest":
            # RandomForestClassifier cannot ingest NaN; impute with the median.
            self._medians_ = np.nanmedian(np.where(np.isfinite(X), X, np.nan), axis=0)
            self._medians_ = np.where(np.isfinite(self._medians_), self._medians_, 0.0)
            model = RandomForestClassifier(
                n_estimators=int(self.params["num_trees"]),
                max_depth=int(self.params["max_depth"]),
                min_samples_leaf=max(1, int(self.params["min_examples"]) // 4),
                max_features=float(self.params["num_candidate_attributes_ratio"]),
                random_state=self.seed, n_jobs=-1,
            )
            return model.fit(self._impute(X), y)

        depth = self.params.get("max_depth", 6)
        model = HistGradientBoostingClassifier(
            max_iter=int(self.params["num_trees"]),
            learning_rate=float(self.params["shrinkage"]),
            max_leaf_nodes=int(self.params.get("max_num_nodes", 31)),
            max_depth=None if int(depth) <= 0 else int(depth),
            min_samples_leaf=int(self.params["min_examples"]),
            l2_regularization=float(self.params["l2_regularization"]),
            early_stopping=False, random_state=self.seed,
        )
        return model.fit(X, y)

    def _impute(self, X: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(X), X, np.nan)
        return np.where(np.isnan(filled), self._medians_, filled)

    # -- predict --------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        """Probability of class 1 (= MCI+DEM), shape ``(n_samples,)``."""

        X = np.asarray(X, dtype=np.float64)
        if self.model_ is None:
            raise RuntimeError("Model is not fitted")

        if YDF_AVAILABLE:
            frame = pd.DataFrame(X, columns=self.columns_)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = np.asarray(self.model_.predict(frame), dtype=np.float64)
            # Binary YDF returns P(positive class) as a 1-D array.
            if raw.ndim == 2:
                raw = raw[:, -1] if raw.shape[1] > 1 else raw.ravel()
            return raw.ravel()

        if self.kind == "random_forest":
            X = self._impute(X)
        return self.model_.predict_proba(X)[:, 1]

    # -- importances ----------------------------------------------------------
    def importances(self) -> np.ndarray:
        """Non-negative per-feature importance, aligned to training columns."""

        n_features = len(self.columns_)
        scores = np.zeros(n_features, dtype=np.float64)
        if self.model_ is None:
            return scores

        if YDF_AVAILABLE:
            try:
                table = self.model_.variable_importances()
            except Exception:
                return scores
            key = next((k for k in _IMPORTANCE_PREFERENCE if k in table), None)
            if key is None:
                key = next(iter(table), None)
            if key is None:
                return scores
            index = {name: i for i, name in enumerate(self.columns_)}
            for importance, name in table[key]:
                position = index.get(str(name))
                if position is not None:
                    scores[position] = float(importance)
            # Permutation importances may be negative; shift to non-negative.
            if scores.min() < 0:
                scores = scores - scores.min()
            return scores

        return np.asarray(self.model_.feature_importances_, dtype=np.float64)


def engine_report() -> dict:
    return {
        "ydf_available": YDF_AVAILABLE,
        "ydf_version": YDF_VERSION,
        "engine": "ydf" if YDF_AVAILABLE else "sklearn_fallback",
        "note": (
            "Google YDF in use."
            if YDF_AVAILABLE
            else "YDF MISSING -- scikit-learn stand-ins used. Wiring check only; "
                 "this is not a Google-model result."
        ),
    }
