"""Learners for the max-AUC experiment -- Google YDF first.

Google Yggdrasil Decision Forests (YDF) is the centrepiece:

* ``ydf_gbt``          - Gradient Boosted Trees (axis-aligned splits)
* ``ydf_gbt_oblique``  - GBT with **sparse oblique** splits.  Oblique splits are
  a YDF speciality: each split is a random sparse *linear combination* of
  numeric features rather than a single-feature threshold.  On small, dense,
  all-numeric tables like this one (141 subjects) they often recover the linear
  structure that plain axis-aligned trees miss -- exactly the regime where the
  MMSE-only logistic regression was previously beating the trees.
* ``ydf_rf`` / ``ydf_rf_oblique`` - Random Forest, same idea.

Two strong non-Google baselines (``logreg``, ``svm``) are kept in the pool
because they were the honest winners in the earlier experiment.  The ensemble
weights are chosen by inner-fold ROC-AUC, so the blend keeps whatever actually
works and drops what does not -- the Google models have to earn their weight.

Interface is deliberately plain ``fit(X, y)`` / ``predict_proba(X)`` on raw
numpy, because the tuner re-selects a different feature subset per candidate.
All preprocessing is fit on the passed training matrix only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .numeric import column_median, impute

CLASS_NAMES = ("CN", "MCI_DEM")
YDF_KINDS = ("ydf_gbt", "ydf_gbt_oblique", "ydf_rf", "ydf_rf_oblique")
SK_KINDS = ("logreg", "svm")
ALL_KINDS = YDF_KINDS + SK_KINDS

try:
    import ydf  # noqa: F401
    YDF_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the runtime
    YDF_AVAILABLE = False

# scikit-learn 1.9 deprecated LogisticRegression's ``penalty`` in favour of a
# continuous ``l1_ratio`` (0 = ridge, 1 = lasso).  Passing both makes 1.9 warn on
# every single fit -- tens of thousands of lines over a full search -- and
# ``penalty`` is scheduled for removal, so pick the API the installed version
# actually wants.  requirements allow scikit-learn <2, so both paths are live.
_LOGREG_USES_L1_RATIO = LogisticRegression().get_params().get("l1_ratio") is not None

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[learners] {message}")


def _balanced_class_weights(y: np.ndarray) -> dict:
    n = len(y)
    counts = {c: int(np.sum(y == c)) for c in (0, 1)}
    return {CLASS_NAMES[c]: (n / (2.0 * counts[c]) if counts[c] else 1.0) for c in (0, 1)}


# --------------------------------------------------------------- YDF ----------
_OBLIQUE_KEYS = ("sparse_oblique_normalization", "sparse_oblique_num_projections_exponent",
                 "sparse_oblique_projection_density_factor")


def _build_ydf_learner(kind: str, params: dict, seed: int):
    """Construct a YDF learner, degrading gracefully if a knob is unsupported."""

    import ydf

    oblique = kind.endswith("_oblique")
    common = dict(label="label", label_classes=list(CLASS_NAMES), random_seed=seed,
                  num_threads=int(os.cpu_count() or 4))
    if oblique:
        common["split_axis"] = "SPARSE_OBLIQUE"
        for key in _OBLIQUE_KEYS:
            if key in params:
                common[key] = params[key]

    if kind.startswith("ydf_gbt"):
        cls = ydf.GradientBoostedTreesLearner
        core = dict(
            loss="BINOMIAL_LOG_LIKELIHOOD", validation_ratio=0.0,
            num_trees=int(params["num_trees"]), max_depth=int(params["max_depth"]),
            min_examples=int(params["min_examples"]), shrinkage=float(params["shrinkage"]),
            subsample=float(params["subsample"]),
            num_candidate_attributes_ratio=float(params["num_candidate_attributes_ratio"]),
            l2_regularization=float(params["l2_regularization"]),
        )
    else:
        cls = ydf.RandomForestLearner
        core = dict(
            num_trees=int(params["num_trees"]), max_depth=int(params["max_depth"]),
            min_examples=int(params["min_examples"]),
            num_candidate_attributes_ratio=float(params["num_candidate_attributes_ratio"]),
        )

    try:
        return cls(**core, **common)
    except Exception as error:  # unsupported knob on this YDF version
        _warn_once(f"ydf_kwargs_{kind}",
                   f"{kind}: YDF rejected optional kwargs ({type(error).__name__}); "
                   "retrying with core hyperparameters only.")
        fallback = {k: v for k, v in common.items() if k not in _OBLIQUE_KEYS}
        try:
            return cls(**core, **fallback)
        except Exception:
            fallback.pop("split_axis", None)
            return cls(**core, **fallback)


class YDFLearner:
    """Google YDF learner; NaN-tolerant sklearn stand-in when ydf is absent."""

    def __init__(self, kind: str, params: dict, *, seed: int = 0) -> None:
        self.kind = kind
        self.params = dict(params)
        self.seed = seed
        self.engine_ = "ydf" if YDF_AVAILABLE else "sklearn_fallback"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "YDFLearner":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_features_ = X.shape[1]
        self.columns_ = [f"f{i}" for i in range(self.n_features_)]
        if YDF_AVAILABLE:
            frame = pd.DataFrame(X, columns=self.columns_)
            frame["label"] = [CLASS_NAMES[int(v)] for v in y]
            learner = _build_ydf_learner(self.kind, self.params, self.seed)
            learner.class_weights = _balanced_class_weights(y)
            self.model_ = learner.train(frame, verbose=0)
            classes = tuple(str(c) for c in self.model_.label_classes())
            self._pos = classes.index("MCI_DEM")
        else:
            p = self.params
            self.model_ = HistGradientBoostingClassifier(
                max_iter=int(p.get("num_trees", 300)),
                learning_rate=float(p.get("shrinkage", 0.05)),
                max_depth=int(p.get("max_depth", 4)),
                min_samples_leaf=int(p.get("min_examples", 5)),
                l2_regularization=float(p.get("l2_regularization", 1.0)),
                class_weight="balanced", random_state=self.seed,
            ).fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if YDF_AVAILABLE:
            raw = np.asarray(self.model_.predict(pd.DataFrame(X, columns=self.columns_)),
                             dtype=np.float64)
            if raw.ndim == 1:
                return raw if self._pos == 1 else 1.0 - raw
            return raw[:, self._pos]
        return self.model_.predict_proba(X)[:, 1]

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {"type": "ydf", "kind": self.kind, "engine": self.engine_,
                "params": self.params, "n_features": int(self.n_features_)}
        if YDF_AVAILABLE:
            # YDF's save() does temp-file work that can fail on the mounted
            # Google Drive (FUSE) filesystem: build locally, then copy over.
            import shutil
            import tempfile

            dest = path / "ydf_model"
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "ydf_model"
                self.model_.save(str(local))
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(local, dest)
            meta["pos_index"] = int(self._pos)
        else:
            joblib.dump(self.model_, path / "sklearn.joblib")
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")


# ------------------------------------------------------------ sklearn ---------
class SkLearner:
    """Median-impute + standardize + (logreg | rbf-SVM). Prep fit on train only."""

    def __init__(self, kind: str, params: dict, *, seed: int = 0) -> None:
        self.kind = kind
        self.params = dict(params)
        self.seed = seed

    def _fit_prep(self, X: np.ndarray) -> None:
        self.median_ = column_median(X)
        filled = impute(X, self.median_)
        self.mean_ = filled.mean(axis=0)
        std = filled.std(axis=0)
        self.std_ = np.where(std < 1e-8, 1.0, std)

    def _prep(self, X: np.ndarray) -> np.ndarray:
        return (impute(X, self.median_) - self.mean_) / self.std_

    def _estimator(self, y: np.ndarray):
        p = self.params
        if self.kind == "logreg":
            penalty = p.get("penalty", "l2")
            l1_ratio = {"l2": 0.0, "l1": 1.0}.get(
                penalty, float(p.get("l1_ratio", 0.5)))
            kwargs = dict(C=float(p.get("C", 0.1)), class_weight="balanced",
                          max_iter=8000, random_state=self.seed,
                          solver="lbfgs" if l1_ratio == 0.0 else "saga")
            if _LOGREG_USES_L1_RATIO:
                kwargs["l1_ratio"] = l1_ratio
            else:
                kwargs["penalty"] = penalty
                if penalty == "elasticnet":
                    kwargs["l1_ratio"] = l1_ratio
            return LogisticRegression(**kwargs)
        if self.kind == "svm":
            base = SVC(C=float(p.get("C", 1.0)), kernel="rbf",
                       gamma=p.get("gamma", "scale"), class_weight="balanced",
                       random_state=self.seed)
            # SVC(probability=True) is deprecated in scikit-learn 1.9 and warns on
            # every fit; CalibratedClassifierCV is the replacement and works on
            # every version this project supports, so use it unconditionally.
            # cv is clipped to the rarer class count so small inner folds stay valid.
            minority = int(np.bincount(np.asarray(y, dtype=np.int64), minlength=2).min())
            return CalibratedClassifierCV(base, method="sigmoid", ensemble=False,
                                          cv=int(np.clip(minority, 2, 5)))
        raise ValueError(self.kind)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SkLearner":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_features_ = X.shape[1]
        self._fit_prep(X)
        self.model_ = self._estimator(y).fit(self._prep(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(self._prep(np.asarray(X, dtype=np.float64)))[:, 1]

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"median": self.median_, "mean": self.mean_, "std": self.std_,
                     "model": self.model_}, path / "sklearn.joblib")
        (path / "meta.json").write_text(
            json.dumps({"type": "sklearn", "kind": self.kind, "params": self.params,
                        "n_features": int(self.n_features_)}, ensure_ascii=False, indent=2),
            encoding="utf-8")


def make_learner(kind: str, params: dict, *, seed: int = 0):
    if kind in YDF_KINDS:
        return YDFLearner(kind, params, seed=seed)
    if kind in SK_KINDS:
        return SkLearner(kind, params, seed=seed)
    raise ValueError(f"Unknown learner kind: {kind}")


def available_kinds(include_ydf_oblique: bool = True) -> tuple[str, ...]:
    kinds = list(SK_KINDS) + ["ydf_gbt", "ydf_rf"]
    if include_ydf_oblique:
        kinds += ["ydf_gbt_oblique", "ydf_rf_oblique"]
    return tuple(kinds)


__all__ = ["ALL_KINDS", "CLASS_NAMES", "SK_KINDS", "SkLearner", "YDFLearner", "YDF_AVAILABLE",
           "YDF_KINDS", "available_kinds", "make_learner"]
