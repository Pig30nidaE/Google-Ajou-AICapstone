"""The model zoo, behind one interface: ``fit(X, y)`` then ``score_samples(X)``.

``score_samples`` returns a continuous score where larger means "more likely
Dem".  ROC-AUC only reads the ordering, so the interface deliberately does not
promise calibrated probabilities -- for models with a ``decision_function`` the
raw margin is preferred, because Platt-scaling a margin at 12 positives adds
variance and compresses ties without changing the ranking that matters.

Families
--------
Google       ``ydf_*`` (Yggdrasil Decision Forests, incl. sparse-oblique splits)
             ``tabnet`` (pytorch-tabnet, with optional masked self-supervised
             pretraining fitted on training-fold rows only)
Tree         LightGBM, XGBoost, CatBoost, HistGradientBoosting, RandomForest,
             ExtraTrees, BalancedRandomForest, EasyEnsemble
Linear/kernel elastic-net and L2 logistic regression, linear and RBF SVM
Neural       MLP
Rank         ``rank_mean`` -- direction-aligned rank average, and ``univariate``,
             the prior folder's single-best-feature learner, kept so the
             comparison against it is run rather than argued

Every optional dependency is probed at import time; ``available_models()``
returns only what can actually run, so a missing wheel degrades the zoo instead
of aborting a six-hour Colab job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


def _probe(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


HAS_YDF = _probe("ydf")
HAS_TABNET = _probe("pytorch_tabnet") and _probe("torch")
HAS_LIGHTGBM = _probe("lightgbm")
HAS_XGBOOST = _probe("xgboost")
HAS_CATBOOST = _probe("catboost")
HAS_IMBLEARN = _probe("imblearn")
HAS_TORCH = _probe("torch")


def torch_device() -> str:
    """CPU/GPU auto-detection, used by the TabNet and sequence arms."""

    if not HAS_TORCH:
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------- interface ---
class Model:
    """Base wrapper.  Subclasses implement ``_fit`` and ``_score``."""

    name = "base"
    supports_importance = False

    def __init__(self, params: dict | None = None, *, seed: int = 0) -> None:
        self.params = dict(params or {})
        self.seed = int(seed)
        self.fitted_ = False
        self.n_features_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray, *, rows: np.ndarray | None = None) -> "Model":
        """``rows`` carries the cohort row indices of ``X``.

        Tabular models ignore it.  The sequence arm needs it to locate the daily
        tensor for exactly these subjects -- passing indices rather than the
        tensor keeps the fold slicing in one place (``engine.fold_fit_predict``)
        instead of duplicating it per model.
        """

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if len(np.unique(y)) < 2:
            raise ValueError(f"{self.name}: single-class training fold")
        if not np.isfinite(X).all():
            raise ValueError(f"{self.name}: received non-finite features; preprocess first")
        self.n_features_ = int(X.shape[1])
        self.rows_ = None if rows is None else np.asarray(rows, dtype=np.int64)
        self._fit(X, y)
        self.fitted_ = True
        return self

    def score_samples(self, X: np.ndarray, *, rows: np.ndarray | None = None) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{self.name}: score_samples before fit")
        X = np.asarray(X, dtype=np.float64)
        self.score_rows_ = None if rows is None else np.asarray(rows, dtype=np.int64)
        scores = np.asarray(self._score(X), dtype=np.float64).ravel()
        if scores.shape[0] != X.shape[0]:
            raise RuntimeError(f"{self.name}: score length mismatch")
        # A model that emits NaN would silently poison an ensemble average.
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _score(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def importance(self) -> np.ndarray | None:
        return None


class SklearnModel(Model):
    """Wraps any scikit-learn-style estimator."""

    def __init__(self, name: str, factory: Callable[..., Any], params: dict | None = None,
                 *, seed: int = 0, prefer_margin: bool = False) -> None:
        super().__init__(params, seed=seed)
        self.name = name
        self._factory = factory
        self._prefer_margin = prefer_margin

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.estimator_ = self._factory(**self.params, seed=self.seed)
        self.estimator_.fit(X, y)

    def _score(self, X: np.ndarray) -> np.ndarray:
        estimator = self.estimator_
        if self._prefer_margin and hasattr(estimator, "decision_function"):
            return estimator.decision_function(X)
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            classes = list(getattr(estimator, "classes_", [0, 1]))
            return proba[:, classes.index(1)] if 1 in classes else proba[:, -1]
        return estimator.decision_function(X)

    @property
    def supports_importance(self) -> bool:  # type: ignore[override]
        estimator = getattr(self, "estimator_", None)
        return hasattr(estimator, "feature_importances_") or hasattr(estimator, "coef_")

    def importance(self) -> np.ndarray | None:
        estimator = getattr(self, "estimator_", None)
        if estimator is None:
            return None
        if hasattr(estimator, "feature_importances_"):
            return np.asarray(estimator.feature_importances_, dtype=np.float64)
        if hasattr(estimator, "coef_"):
            return np.abs(np.asarray(estimator.coef_, dtype=np.float64)).ravel()
        return None


# ------------------------------------------------------------ rank models ----
class RankAggregateModel(Model):
    """Direction-aligned rank average -- one free parameter per feature: its sign.

    Rationale for a 12-positive problem: this estimates only *k* signs and no
    magnitudes, so it has far less variance than any fitted model, and it
    operates directly in rank space, which is what ROC-AUC measures.  Features
    whose training-fold AUC is inside ``[0.5 - margin, 0.5 + margin]`` are
    dropped rather than assigned a coin-flip direction.
    """

    name = "rank_mean"

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        margin = float(self.params.get("margin", 0.05))
        aucs = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
        self.signs_ = np.where(aucs >= 0.5, 1.0, -1.0)
        self.keep_ = np.flatnonzero(np.abs(aucs - 0.5) > margin)
        if self.keep_.size == 0:
            self.keep_ = np.argsort(-np.abs(aucs - 0.5))[: max(1, X.shape[1] // 10)]
        # Reference sample used to map new values onto the training rank scale.
        self.reference_ = np.sort(X[:, self.keep_] * self.signs_[self.keep_], axis=0)

    def _score(self, X: np.ndarray) -> np.ndarray:
        aligned = X[:, self.keep_] * self.signs_[self.keep_]
        ranks = np.empty_like(aligned, dtype=np.float64)
        for column in range(aligned.shape[1]):
            ranks[:, column] = np.searchsorted(self.reference_[:, column],
                                               aligned[:, column], side="left")
        return ranks.mean(axis=1) / max(1, self.reference_.shape[0])

    def importance(self) -> np.ndarray | None:
        weights = np.zeros(self.n_features_, dtype=np.float64)
        weights[self.keep_] = 1.0
        return weights


class UnivariateModel(Model):
    """Best single training-fold feature + logistic regression.

    This is ``Binary_Google_DemScreen``'s ``univariate`` learner, reproduced so
    the new pipeline is compared against it under an identical protocol instead
    of against a remembered number.
    """

    name = "univariate"

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        scores = np.array([abs(roc_auc_score(y, X[:, j]) - 0.5) for j in range(X.shape[1])])
        self.column_ = int(np.argmax(scores))
        self.model_ = LogisticRegression(
            C=float(self.params.get("C", 1.0)), class_weight="balanced",
            max_iter=5000, random_state=self.seed,
        ).fit(X[:, [self.column_]], y)

    def _score(self, X: np.ndarray) -> np.ndarray:
        return self.model_.decision_function(X[:, [self.column_]])

    def importance(self) -> np.ndarray | None:
        weights = np.zeros(self.n_features_, dtype=np.float64)
        weights[self.column_] = 1.0
        return weights


# ------------------------------------------------------------- Google: YDF ---
class YDFModel(Model):
    """Google Yggdrasil Decision Forests.

    Kept small on purpose.  ``Binary_Google_MaxAUC_Tuned`` measured sparse
    *oblique* splits as the strongest single learner (inner AUC 0.789 vs 0.745
    axis-aligned), which is why the oblique variant is carried rather than only
    the default GBT.
    """

    def __init__(self, kind: str, params: dict | None = None, *, seed: int = 0) -> None:
        super().__init__(params, seed=seed)
        self.name = kind
        self.kind = kind

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import pandas as pd
        import ydf

        self.columns_ = [f"f{i}" for i in range(X.shape[1])]
        frame = pd.DataFrame(X, columns=self.columns_)
        frame["label"] = y.astype(np.int64)
        positive_weight = float((y == 0).sum()) / max(1.0, float((y == 1).sum()))
        common = dict(
            label="label",
            task=ydf.Task.CLASSIFICATION,
            random_seed=self.seed,
            # Class weighting via sample weights keeps the 12 positives visible.
            weights="_w",
        )
        frame["_w"] = np.where(y == 1, positive_weight, 1.0)

        if self.kind == "ydf_rf":
            learner = ydf.RandomForestLearner(
                num_trees=int(self.params.get("num_trees", 300)),
                max_depth=int(self.params.get("max_depth", 4)),
                min_examples=int(self.params.get("min_examples", 5)),
                **common,
            )
        else:
            kwargs = dict(
                num_trees=int(self.params.get("num_trees", 150)),
                max_depth=int(self.params.get("max_depth", 3)),
                shrinkage=float(self.params.get("shrinkage", 0.05)),
                min_examples=int(self.params.get("min_examples", 5)),
                l2_regularization=float(self.params.get("l2_regularization", 5.0)),
                early_stopping="NONE",
                **common,
            )
            if self.kind == "ydf_gbt_oblique":
                kwargs["split_axis"] = "SPARSE_OBLIQUE"
                kwargs["sparse_oblique_normalization"] = "STANDARD_DEVIATION"
            learner = ydf.GradientBoostedTreesLearner(**kwargs)
        self.model_ = learner.train(frame, verbose=0)

    def _score(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd

        frame = pd.DataFrame(X, columns=self.columns_)
        return np.asarray(self.model_.predict(frame), dtype=np.float64).ravel()

    def importance(self) -> np.ndarray | None:
        model = getattr(self, "model_", None)
        if model is None:
            return None
        try:
            importances = model.variable_importances().get("SUM_SCORE", [])
        except Exception:  # pragma: no cover - YDF version differences
            return None
        lookup = {str(name): float(value) for value, name in importances}
        return np.array([lookup.get(column, 0.0) for column in self.columns_], dtype=np.float64)


# ---------------------------------------------------------- Google: TabNet ---
class TabNetModel(Model):
    """Google TabNet with optional masked self-supervised pretraining.

    The prompt's TabNet knobs map directly onto the constructor: ``n_d``, ``n_a``,
    ``n_steps``, ``gamma``, ``lambda_sparse``, ``virtual_batch_size``, the
    optimiser learning rate, the scheduler and the early-stopping patience.

    Pretraining uses ``TabNetPretrainer`` on the **training rows of the current
    fold only**.  That is the leakage-relevant point: masked reconstruction is
    unsupervised, but running it over the whole cohort would still let the
    encoder see the held-out subjects' feature distribution, so it is fitted
    inside the fold like everything else.

    Prior evidence is not encouraging -- ``Binary_Wearable_TabNet_Google`` scored
    OOF AUC 0.446 with 9 of 10 folds below 0.5 -- so this is carried as a
    candidate the protocol can reject, not as an expected winner.
    """

    name = "tabnet"

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier

        # pytorch-tabnet hands the raw array straight to ``Tensor.to(device)``,
        # so a float64 matrix reaches the accelerator unchanged.  Apple MPS
        # rejects float64 outright and CUDA merely wastes bandwidth on it.
        X = np.asarray(X, dtype=np.float32)
        device = torch_device()
        n_d = int(self.params.get("n_d", 8))
        width = dict(
            n_d=n_d,
            n_a=int(self.params.get("n_a", n_d)),
            n_steps=int(self.params.get("n_steps", 3)),
            gamma=float(self.params.get("gamma", 1.3)),
            lambda_sparse=float(self.params.get("lambda_sparse", 1e-3)),
            seed=self.seed,
            verbose=0,
            device_name=device,
            optimizer_params=dict(lr=float(self.params.get("lr", 2e-2))),
            # StepLR, not ReduceLROnPlateau: pytorch-tabnet decides whether to
            # pass a metric into ``scheduler.step()`` by probing the scheduler
            # for a private attribute that recent torch releases no longer
            # expose, so the plateau scheduler raises
            # "step() missing 1 required positional argument: 'metrics'" on
            # torch >= 2.x.  A step schedule needs no metric and behaves
            # identically across versions.
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(
                step_size=int(self.params.get("scheduler_step_size", 20)),
                gamma=float(self.params.get("scheduler_gamma", 0.7)),
            ),
        )
        max_epochs = int(self.params.get("max_epochs", 120))
        patience = int(self.params.get("patience", 20))

        # A stratified slice of the training fold is the early-stopping monitor.
        rng = np.random.default_rng(self.seed)
        holdout = _stratified_holdout(y, fraction=0.25, rng=rng)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[holdout] = False
        if train_mask.sum() < 4 or len(np.unique(y[train_mask])) < 2 or len(np.unique(y[holdout])) < 2:
            train_mask = np.ones(len(y), dtype=bool)
            holdout = np.arange(len(y))

        # The batch must fit the *smaller* of the two splits.  pytorch-tabnet
        # builds its eval loader with the same batch size and drops the final
        # partial batch, so a batch larger than the ~35-row holdout leaves the
        # loader with zero batches and the epoch ends in an empty ``np.vstack``.
        smallest = int(min(train_mask.sum(), len(holdout)))
        batch_size = int(max(2, min(smallest, self.params.get("batch_size", 64))))
        virtual_batch = int(max(1, min(batch_size, self.params.get("virtual_batch_size", 16))))

        if self.params.get("pretrain", True):
            from pytorch_tabnet.pretraining import TabNetPretrainer

            pretrainer = TabNetPretrainer(**width)
            pretrainer.fit(
                X_train=X[train_mask],
                eval_set=[X[holdout]],
                pretraining_ratio=float(self.params.get("pretraining_ratio", 0.5)),
                max_epochs=int(self.params.get("pretrain_epochs", 60)),
                patience=patience,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch,
                drop_last=False,
            )
        else:
            pretrainer = None

        self.model_ = TabNetClassifier(**width)
        positive_weight = float((y == 0).sum()) / max(1.0, float((y == 1).sum()))
        self.model_.fit(
            X_train=X[train_mask], y_train=y[train_mask],
            eval_set=[(X[holdout], y[holdout])],
            eval_metric=["auc"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch,
            weights={0: 1.0, 1: positive_weight},
            from_unsupervised=pretrainer,
            drop_last=False,
        )

    def _score(self, X: np.ndarray) -> np.ndarray:
        proba = self.model_.predict_proba(np.asarray(X, dtype=np.float32))
        classes = list(getattr(self.model_, "classes_", [0, 1]))
        index = classes.index(1) if 1 in classes else -1
        return proba[:, index]

    def importance(self) -> np.ndarray | None:
        model = getattr(self, "model_", None)
        importances = getattr(model, "feature_importances_", None)
        return None if importances is None else np.asarray(importances, dtype=np.float64)


def _stratified_holdout(y: np.ndarray, *, fraction: float, rng: np.random.Generator) -> np.ndarray:
    picked: list[int] = []
    for label in (0, 1):
        index = np.flatnonzero(y == label)
        take = max(1, int(round(len(index) * fraction)))
        picked.extend(rng.choice(index, size=min(take, len(index)), replace=False).tolist())
    return np.asarray(sorted(picked), dtype=np.int64)


# ------------------------------------------------------- balanced bagging ----
class BalancedBaggingModel(Model):
    """Average of a base model over balanced bootstrap resamples.

    Each member sees all positives and an equally sized random draw of negatives.
    This is the resampling idea that survives at 12 positives: it never invents
    synthetic patients, and averaging over draws reduces the variance that
    undersampling alone would add.
    """

    name = "balanced_bag"

    def __init__(self, base: str, params: dict | None = None, *, seed: int = 0) -> None:
        super().__init__(params, seed=seed)
        self.base = base
        self.name = f"bag_{base}"

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_members = int(self.params.get("n_members", 15))
        ratio = float(self.params.get("negative_ratio", 3.0))
        base_params = {k: v for k, v in self.params.items()
                       if k not in ("n_members", "negative_ratio")}
        rng = np.random.default_rng(self.seed)
        positives = np.flatnonzero(y == 1)
        negatives = np.flatnonzero(y == 0)
        take = int(min(len(negatives), max(len(positives), round(len(positives) * ratio))))

        self.members_: list[Model] = []
        for member in range(n_members):
            sampled = rng.choice(negatives, size=take, replace=False)
            index = np.concatenate([positives, sampled])
            model = build_model(self.base, base_params, seed=self.seed + member)
            try:
                model.fit(X[index], y[index])
            except Exception:
                continue
            self.members_.append(model)
        if not self.members_:
            fallback = build_model(self.base, base_params, seed=self.seed)
            fallback.fit(X, y)
            self.members_.append(fallback)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # Rank-average the members: they are trained on different negative draws,
        # so their score scales are not comparable but their orderings are.
        from .ensemble import rank_normalize

        stacked = np.column_stack([rank_normalize(m.score_samples(X)) for m in self.members_])
        return stacked.mean(axis=1)


# ------------------------------------------------------------- the registry --
def _logreg_en(*, C: float = 1.0, l1_ratio: float = 0.5, seed: int = 0, **_: Any):
    return LogisticRegression(penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
                              class_weight="balanced", max_iter=8000, random_state=seed)


def _logreg_l2(*, C: float = 1.0, seed: int = 0, **_: Any):
    return LogisticRegression(penalty="l2", solver="lbfgs", C=C, class_weight="balanced",
                              max_iter=8000, random_state=seed)


def _svm_rbf(*, C: float = 1.0, gamma: str | float = "scale", seed: int = 0, **_: Any):
    return SVC(C=C, gamma=gamma, kernel="rbf", class_weight="balanced", random_state=seed)


def _svm_linear(*, C: float = 1.0, seed: int = 0, **_: Any):
    return SVC(C=C, kernel="linear", class_weight="balanced", random_state=seed)


def _random_forest(*, n_estimators: int = 500, max_depth: int | None = 4,
                   min_samples_leaf: int = 2, max_features: str | float = "sqrt",
                   seed: int = 0, **_: Any):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf, max_features=max_features,
                                  class_weight="balanced_subsample", random_state=seed, n_jobs=-1)


def _extra_trees(*, n_estimators: int = 500, max_depth: int | None = None,
                 min_samples_leaf: int = 2, max_features: str | float = "sqrt",
                 seed: int = 0, **_: Any):
    return ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf, max_features=max_features,
                                class_weight="balanced", random_state=seed, n_jobs=-1)


def _hist_gb(*, max_depth: int | None = 3, learning_rate: float = 0.05,
             max_iter: int = 200, l2_regularization: float = 1.0,
             min_samples_leaf: int = 5, seed: int = 0, **_: Any):
    return HistGradientBoostingClassifier(max_depth=max_depth, learning_rate=learning_rate,
                                          max_iter=max_iter, l2_regularization=l2_regularization,
                                          min_samples_leaf=min_samples_leaf,
                                          class_weight="balanced", random_state=seed)


def _mlp(*, hidden: int = 32, alpha: float = 1e-2, learning_rate_init: float = 3e-3,
         seed: int = 0, **_: Any):
    return MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=2000,
                         learning_rate_init=learning_rate_init, random_state=seed,
                         early_stopping=False)


def _lightgbm(*, n_estimators: int = 300, learning_rate: float = 0.05, num_leaves: int = 7,
              max_depth: int = 3, min_child_samples: int = 5, reg_lambda: float = 5.0,
              subsample: float = 0.8, colsample_bytree: float = 0.6, seed: int = 0, **_: Any):
    import lightgbm as lgb

    return lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                              num_leaves=num_leaves, max_depth=max_depth,
                              min_child_samples=min_child_samples, reg_lambda=reg_lambda,
                              subsample=subsample, subsample_freq=1,
                              colsample_bytree=colsample_bytree, class_weight="balanced",
                              random_state=seed, n_jobs=-1, verbosity=-1)


def _xgboost(*, n_estimators: int = 300, learning_rate: float = 0.05, max_depth: int = 3,
             min_child_weight: float = 1.0, reg_lambda: float = 5.0, subsample: float = 0.8,
             colsample_bytree: float = 0.6, scale_pos_weight: float | None = None,
             seed: int = 0, **_: Any):
    import xgboost as xgb

    return xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                             max_depth=max_depth, min_child_weight=min_child_weight,
                             reg_lambda=reg_lambda, subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             scale_pos_weight=scale_pos_weight or 1.0,
                             eval_metric="auc", random_state=seed, n_jobs=-1,
                             tree_method="hist")


def _catboost(*, iterations: int = 400, learning_rate: float = 0.05, depth: int = 3,
              l2_leaf_reg: float = 6.0, seed: int = 0, **_: Any):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, depth=depth,
                              l2_leaf_reg=l2_leaf_reg, auto_class_weights="Balanced",
                              random_seed=seed, verbose=0, allow_writing_files=False)


def _balanced_rf(*, n_estimators: int = 500, max_depth: int | None = 4,
                 min_samples_leaf: int = 2, seed: int = 0, **_: Any):
    from imblearn.ensemble import BalancedRandomForestClassifier

    return BalancedRandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          sampling_strategy="all", replacement=True,
                                          bootstrap=False, random_state=seed, n_jobs=-1)


def _easy_ensemble(*, n_estimators: int = 20, seed: int = 0, **_: Any):
    from imblearn.ensemble import EasyEnsembleClassifier

    return EasyEnsembleClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)


_SKLEARN_REGISTRY: dict[str, tuple[Callable[..., Any], bool, bool]] = {
    # name: (factory, prefer_margin, always_available)
    "logreg_en": (_logreg_en, True, True),
    "logreg_l2": (_logreg_l2, True, True),
    "svm_rbf": (_svm_rbf, True, True),
    "svm_linear": (_svm_linear, True, True),
    "random_forest": (_random_forest, False, True),
    "extra_trees": (_extra_trees, False, True),
    "hist_gb": (_hist_gb, True, True),
    "mlp": (_mlp, False, True),
    "lightgbm": (_lightgbm, False, HAS_LIGHTGBM),
    "xgboost": (_xgboost, False, HAS_XGBOOST),
    "catboost": (_catboost, False, HAS_CATBOOST),
    "balanced_rf": (_balanced_rf, False, HAS_IMBLEARN),
    "easy_ensemble": (_easy_ensemble, False, HAS_IMBLEARN),
}

YDF_MODELS = ("ydf_gbt", "ydf_rf", "ydf_gbt_oblique")
RANK_MODELS = ("rank_mean", "univariate")
GOOGLE_MODELS = YDF_MODELS + ("tabnet",)
TREE_BASELINES = ("lightgbm", "xgboost", "catboost", "hist_gb", "random_forest",
                  "extra_trees", "balanced_rf")


def build_model(name: str, params: dict | None = None, *, seed: int = 0) -> Model:
    params = dict(params or {})
    if name in RANK_MODELS:
        return (RankAggregateModel if name == "rank_mean" else UnivariateModel)(params, seed=seed)
    if name in YDF_MODELS:
        if not HAS_YDF:
            raise ModuleNotFoundError("ydf is not installed; pip install ydf==0.16.1")
        return YDFModel(name, params, seed=seed)
    if name == "tabnet":
        if not HAS_TABNET:
            raise ModuleNotFoundError(
                "pytorch-tabnet/torch are not installed; pip install pytorch-tabnet==4.1.0 torch"
            )
        return TabNetModel(params, seed=seed)
    if name.startswith("bag_"):
        return BalancedBaggingModel(name[len("bag_"):], params, seed=seed)
    if name in _SKLEARN_REGISTRY:
        factory, prefer_margin, available = _SKLEARN_REGISTRY[name]
        if not available:
            raise ModuleNotFoundError(f"{name} requires an optional package that is not installed")
        return SklearnModel(name, factory, params, seed=seed, prefer_margin=prefer_margin)
    raise ValueError(f"Unknown model: {name!r}")


def available_models(*, include_slow: bool = True) -> tuple[str, ...]:
    """Names that can actually be constructed in this environment."""

    names = list(RANK_MODELS)
    names += [name for name, (_, _, ok) in _SKLEARN_REGISTRY.items() if ok]
    if HAS_YDF:
        names += list(YDF_MODELS)
    if HAS_TABNET and include_slow:
        names.append("tabnet")
    names += ["bag_logreg_l2", "bag_hist_gb"]
    return tuple(dict.fromkeys(names))


def environment_report() -> dict:
    """Recorded verbatim in every run report, so a missing model is visible."""

    return {
        "ydf": HAS_YDF,
        "pytorch_tabnet": HAS_TABNET,
        "lightgbm": HAS_LIGHTGBM,
        "xgboost": HAS_XGBOOST,
        "catboost": HAS_CATBOOST,
        "imbalanced_learn": HAS_IMBLEARN,
        "torch": HAS_TORCH,
        "torch_device": torch_device(),
        "available_models": list(available_models()),
    }


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict

    def build(self, seed: int) -> Model:
        return build_model(self.name, self.params, seed=seed)


__all__ = [
    "GOOGLE_MODELS", "HAS_CATBOOST", "HAS_IMBLEARN", "HAS_LIGHTGBM", "HAS_TABNET",
    "HAS_TORCH", "HAS_XGBOOST", "HAS_YDF", "Model", "ModelSpec", "RANK_MODELS",
    "TREE_BASELINES", "YDF_MODELS", "available_models", "build_model",
    "environment_report", "torch_device",
]
