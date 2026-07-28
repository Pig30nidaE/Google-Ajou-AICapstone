"""Supervised and fold-local unsupervised-pretrained TabNet branches."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import ModelSpec
from .tabular import (
    FiniteToNan,
    QuantileClipper,
    SafeSelectKBest,
)


class TabNetBinaryEstimator:
    """Small-sample TabNet with internal early stopping and all-data refit.

    It splits the current fit scope before fitting imputation, clipping,
    supervised selection, or scaling for epoch selection.  After the epoch
    counts are frozen, preprocessing and the model are refit on every subject
    in the enclosing fold-training scope.  When ``pretrain=True``,
    TabNetPretrainer follows the same isolation.
    """

    def __init__(
        self,
        *,
        n_d: int = 16,
        n_a: int = 16,
        n_steps: int = 4,
        gamma: float = 1.4,
        lambda_sparse: float = 1e-4,
        virtual_batch_size: int = 16,
        top_k: int = 32,
        pretrain: bool = False,
        class_weight_mode: str = "balanced",
        learning_rate: float = 0.002,
        max_epochs: int = 160,
        patience: int = 25,
        batch_size: int = 32,
        validation_fraction: float = 0.20,
        device_name: str = "auto",
        seed: int = 0,
        num_workers: int = 0,
    ) -> None:
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.virtual_batch_size = virtual_batch_size
        self.top_k = top_k
        self.pretrain = pretrain
        self.class_weight_mode = class_weight_mode
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.device_name = device_name
        self.seed = seed
        self.num_workers = num_workers

    def get_params(self, deep=True):
        del deep
        return {
            name: getattr(self, name)
            for name in (
                "n_d",
                "n_a",
                "n_steps",
                "gamma",
                "lambda_sparse",
                "virtual_batch_size",
                "top_k",
                "pretrain",
                "class_weight_mode",
                "learning_rate",
                "max_epochs",
                "patience",
                "batch_size",
                "validation_fraction",
                "device_name",
                "seed",
                "num_workers",
            )
        }

    def set_params(self, **params):
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def _device(self) -> str:
        import torch

        if self.device_name == "cpu":
            return "cpu"
        if self.device_name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _new_classifier(self):
        import torch
        from pytorch_tabnet.tab_model import TabNetClassifier

        return TabNetClassifier(
            n_d=int(self.n_d),
            n_a=int(self.n_a),
            n_steps=int(self.n_steps),
            gamma=float(self.gamma),
            lambda_sparse=float(self.lambda_sparse),
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={
                "lr": float(self.learning_rate),
                "weight_decay": 1e-4,
            },
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_params={"mode": "max", "factor": 0.5, "patience": 8},
            mask_type="entmax",
            seed=int(self.seed),
            verbose=0,
            device_name=self._device(),
        )

    def _new_pretrainer(self):
        import torch
        from pytorch_tabnet.pretraining import TabNetPretrainer

        return TabNetPretrainer(
            n_d=int(self.n_d),
            n_a=int(self.n_a),
            n_steps=int(self.n_steps),
            gamma=float(self.gamma),
            lambda_sparse=float(self.lambda_sparse),
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={
                "lr": float(self.learning_rate),
                "weight_decay": 1e-4,
            },
            seed=int(self.seed),
            verbose=0,
            device_name=self._device(),
        )

    def _new_preprocessor(self):
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import RobustScaler

        return Pipeline(
            [
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
                ("select", SafeSelectKBest(int(self.top_k))),
                ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            ]
        )

    def _resolved_batch_sizes(self, n_samples: int) -> tuple[int, int]:
        """Return a valid TabNet batch/virtual-batch pair for small folds.

        Ghost batch normalization expects ``virtual_batch_size`` to be smaller
        than and divide ``batch_size``.  Nested folds can be smaller than the
        configured batch, so resolve both values from the current fit scope.
        """

        if int(n_samples) < 2:
            raise ValueError("TabNet requires at least two training samples")
        batch_size = min(max(2, int(self.batch_size)), int(n_samples))
        requested_virtual = min(
            max(1, int(self.virtual_batch_size)),
            batch_size - 1,
        )
        virtual_batch = requested_virtual
        while virtual_batch > 1 and batch_size % virtual_batch != 0:
            virtual_batch -= 1
        return batch_size, virtual_batch

    def fit(self, X, y):
        from sklearn.model_selection import StratifiedShuffleSplit

        values = np.asarray(X, dtype=np.float64)
        target = np.asarray(y, dtype=np.int64)
        if values.ndim != 2 or len(values) != len(target):
            raise ValueError("TabNet X/y shape mismatch")
        counts = np.bincount(target, minlength=2)
        if int(counts.min()) < 3:
            raise ValueError(
                f"TabNet internal validation requires >=3 per class; counts={counts.tolist()}"
            )
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(self.validation_fraction),
            random_state=int(self.seed),
        )
        train_index, validation_index = next(splitter.split(values, target))
        early_preprocessor = self._new_preprocessor()
        early_train = np.asarray(
            early_preprocessor.fit_transform(
                values[train_index],
                target[train_index],
            ),
            dtype=np.float32,
        )
        early_validation = np.asarray(
            early_preprocessor.transform(values[validation_index]),
            dtype=np.float32,
        )
        batch_size, virtual_batch = self._resolved_batch_sizes(len(train_index))
        common_fit = {
            "batch_size": batch_size,
            "virtual_batch_size": virtual_batch,
            "num_workers": int(self.num_workers),
            "drop_last": False,
        }
        if self.class_weight_mode not in {"balanced", "none"}:
            raise ValueError(
                "TabNet class_weight_mode must be 'balanced' or 'none'"
            )
        fit_weights = 1 if self.class_weight_mode == "balanced" else 0

        early_pretrainer = None
        pretrain_epochs = max(10, int(self.max_epochs) // 2)
        if bool(self.pretrain):
            early_pretrainer = self._new_pretrainer()
            early_pretrainer.fit(
                early_train,
                eval_set=[early_validation],
                eval_name=["valid"],
                max_epochs=pretrain_epochs,
                patience=max(5, int(self.patience) // 2),
                pretraining_ratio=0.8,
                **common_fit,
            )
        early_model = self._new_classifier()
        early_model.fit(
            early_train,
            target[train_index],
            eval_set=[(early_validation, target[validation_index])],
            eval_name=["valid"],
            eval_metric=["auc"],
            max_epochs=int(self.max_epochs),
            patience=int(self.patience),
            weights=fit_weights,
            from_unsupervised=early_pretrainer,
            **common_fit,
        )
        selected_epochs = max(1, int(getattr(early_model, "best_epoch", 0)) + 1)

        # Refit on every fold-training subject using only the epoch count chosen
        # above.  No outer-validation subject enters either stage.
        self.preprocessor_ = self._new_preprocessor()
        final_values = np.asarray(
            self.preprocessor_.fit_transform(values, target),
            dtype=np.float32,
        )
        final_batch_size, final_virtual_batch = self._resolved_batch_sizes(
            len(final_values)
        )
        final_pretrainer = None
        if bool(self.pretrain):
            selected_pretrain_epochs = max(
                1, int(getattr(early_pretrainer, "best_epoch", 0)) + 1
            )
            final_pretrainer = self._new_pretrainer()
            final_pretrainer.fit(
                final_values,
                eval_set=[],
                max_epochs=selected_pretrain_epochs,
                patience=0,
                pretraining_ratio=0.8,
                batch_size=final_batch_size,
                virtual_batch_size=final_virtual_batch,
                num_workers=int(self.num_workers),
                drop_last=False,
            )
        self.model_ = self._new_classifier()
        self.model_.fit(
            final_values,
            target,
            eval_set=[],
            max_epochs=selected_epochs,
            patience=0,
            weights=fit_weights,
            from_unsupervised=final_pretrainer,
            batch_size=final_batch_size,
            virtual_batch_size=final_virtual_batch,
            num_workers=int(self.num_workers),
            drop_last=False,
        )
        self.selected_epochs_ = selected_epochs
        self.classes_ = np.asarray([0, 1], dtype=np.int64)
        self.n_features_in_ = values.shape[1]
        self.n_transformed_features_ = final_values.shape[1]
        return self

    def predict_proba(self, X):
        values = np.asarray(
            self.preprocessor_.transform(np.asarray(X, dtype=np.float64)),
            dtype=np.float32,
        )
        probabilities = np.asarray(
            self.model_.predict_proba(values),
            dtype=np.float64,
        )
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(f"Unexpected TabNet probability shape: {probabilities.shape}")
        return probabilities


def build_tabnet_estimator(
    spec: ModelSpec,
    params: Mapping[str, Any],
    *,
    seed: int,
    neural_config,
):
    return TabNetBinaryEstimator(
        n_d=int(params["n_d"]),
        n_a=int(params["n_a"]),
        n_steps=int(params["n_steps"]),
        gamma=float(params["gamma"]),
        lambda_sparse=float(params["lambda_sparse"]),
        virtual_batch_size=int(
            params.get("virtual_batch_size", neural_config.virtual_batch_size)
        ),
        top_k=int(params.get("top_k", 32)),
        pretrain=bool(params.get("pretrain", spec.name == "tabnet_pretrained")),
        class_weight_mode=str(params.get("class_weight_mode", "balanced")),
        learning_rate=float(params.get("learning_rate", 0.002)),
        max_epochs=int(neural_config.max_epochs),
        patience=int(neural_config.patience),
        batch_size=int(neural_config.batch_size),
        validation_fraction=float(neural_config.validation_fraction),
        device_name=str(neural_config.device),
        seed=int(seed),
        num_workers=int(neural_config.num_workers),
    )


def suggest_tabnet_params(trial, spec: ModelSpec, *, feature_choices: tuple[int, ...]):
    params = dict(spec.fixed_params)
    width = trial.suggest_categorical("feature_width", [8, 16, 24, 32])
    params["n_d"] = width
    params["n_a"] = width
    params["n_steps"] = trial.suggest_int("n_steps", 3, 7)
    params["gamma"] = trial.suggest_float("gamma", 1.0, 2.0)
    params["lambda_sparse"] = trial.suggest_float(
        "lambda_sparse", 1e-6, 1e-2, log=True
    )
    params["virtual_batch_size"] = trial.suggest_categorical(
        "virtual_batch_size", [4, 8, 16, 32]
    )
    params["learning_rate"] = trial.suggest_float(
        "learning_rate", 3e-4, 1e-2, log=True
    )
    params["class_weight_mode"] = trial.suggest_categorical(
        "class_weight_mode", ["balanced", "none"]
    )
    params["top_k"] = trial.suggest_categorical("top_k", list(feature_choices))
    params["pretrain"] = spec.name == "tabnet_pretrained"
    return params


__all__ = [
    "TabNetBinaryEstimator",
    "build_tabnet_estimator",
    "suggest_tabnet_params",
]
