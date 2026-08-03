"""One place that knows how to fit and score every model in this package."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..preprocessing.scaler import represent, representation_feature_names
from ..sequences.builder import SequenceSet
from . import baselines, h2o_backend
from .lstm import LSTMConfig, LSTMSearchSpace, SequenceLSTM

MODEL_NAMES = ("lstm", *baselines.BASELINE_MODELS)

#: Models whose input must be standardised.  Trees are scale-invariant; the
#: distance- and gradient-based ones are not.
NEEDS_SCALING = ("lstm", "logistic_regression", "svm")


def dependency_report() -> dict[str, Any]:
    available = baselines.available_backends()
    runnable = []
    for name in MODEL_NAMES:
        if name == "lstm" and available["torch"]:
            runnable.append(name)
        elif name == "xgboost" and available["xgboost"]:
            runnable.append(name)
        elif name in ("logistic_regression", "svm", "random_forest") and available["sklearn"]:
            runnable.append(name)
    return {
        "available": available,
        "runnable_models": runnable,
        "h2o_version": h2o_backend.installed_version(),
    }


def needs_scaling(model_name: str) -> bool:
    return model_name in NEEDS_SCALING


def search_space(model_name: str, *, enabled: bool = True, limit: int | None = None,
                 seed: int = 42) -> list[dict[str, Any]]:
    """Hyperparameter candidates for the inner CV.

    With ``enabled=False`` this returns exactly one candidate -- the paper's
    reported setting -- which is what experiments A, B1 and B2 use so that no
    selection happens outside a nested loop.
    """
    if model_name == "lstm":
        if not enabled:
            return [{"lstm_units": 128, "dense_units": 64, "learning_rate": 0.001,
                     "dropout": 0.0, "batch_size": 64}]
        return LSTMSearchSpace().candidates(limit=limit, seed=seed)
    return baselines.search_space(model_name, enabled=enabled)


def model_input(
    sequences: SequenceSet, model_name: str, *, representation: str = "flatten"
) -> np.ndarray:
    """The array a given model consumes: 3-D for the LSTM, 2-D for the rest."""
    if model_name == "lstm":
        return sequences.X
    return represent(sequences, representation)


def input_feature_names(
    sequences: SequenceSet, model_name: str, *, representation: str = "flatten"
) -> list[str]:
    if model_name == "lstm":
        return list(sequences.feature_columns)
    return representation_feature_names(
        sequences.feature_columns, sequences.sequence_length, representation
    )


def fit(
    model_name: str,
    train: SequenceSet,
    params: dict[str, Any],
    *,
    validation: SequenceSet | None = None,
    representation: str = "flatten",
    seed: int = 42,
    device: str = "cpu",
    class_weight: bool = False,
    lstm_defaults: dict[str, Any] | None = None,
    baseline_backend: str = "sklearn",
    h2o_config: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit *model_name* on *train* only.

    ``validation`` is used solely for LSTM early stopping and must be a slice of
    training data, never an outer test set -- ``audit/leakage.py`` enforces this
    by fingerprint before any fit is allowed to start.
    """
    X = model_input(train, model_name, representation=representation)

    if model_name == "lstm":
        merged = {**(lstm_defaults or {}), **params}
        config = LSTMConfig(seed=seed, class_weight=class_weight, **merged)
        model = SequenceLSTM(
            config,
            n_features=train.X.shape[-1],
            sequence_length=train.sequence_length,
            device=device,
        )
        validation_arrays = None
        if validation is not None and len(validation):
            validation_arrays = (validation.X, validation.y)
        model.fit(X, train.y, validation=validation_arrays)
        return model, {
            "model": "lstm",
            "backend": "torch",
            "params": config.describe(),
            "training": model.meta,
            "history": model.history,
            "n_train_sequences": len(train),
            "validation_split_name": validation.split_name if validation is not None else None,
        }

    if baseline_backend == "h2o":
        h2o_backend.ensure_available(
            require_exact_version=bool((h2o_config or {}).get("require_exact_version", False))
        )
        h2o_backend.start(
            max_mem_size=str((h2o_config or {}).get("max_mem_size", "4G")),
            nthreads=int((h2o_config or {}).get("nthreads", -1)),
        )
        config = h2o_backend.H2OConfig(
            seed=seed,
            **{
                k: v
                for k, v in (h2o_config or {}).items()
                if k
                not in {"require_exact_version", "max_mem_size", "nthreads"}
            },
        )
        names = input_feature_names(train, model_name, representation=representation)
        model, meta = h2o_backend.fit_automl(
            X, train.y, config, model_name=model_name, feature_names=names
        )
        return model, {"model": model_name, "representation": representation,
                       "n_train_sequences": len(train), **meta}

    config = baselines.BaselineConfig(
        name=model_name, params=params, seed=seed, class_weight=class_weight
    )
    model, meta = baselines.fit_baseline(config, X, train.y)
    return model, {"backend": "sklearn", "representation": representation,
                   "n_train_sequences": len(train), **meta}


def predict(
    model: Any,
    model_name: str,
    sequences: SequenceSet,
    *,
    representation: str = "flatten",
    backend: str = "sklearn",
) -> np.ndarray:
    """Sequence-level positive-class probabilities."""
    if not len(sequences):
        return np.empty(0, dtype=np.float64)
    X = model_input(sequences, model_name, representation=representation)
    if model_name == "lstm":
        return model.predict_proba(X)
    if backend == "h2o":
        names = input_feature_names(sequences, model_name, representation=representation)
        return h2o_backend.predict_proba(model, X, feature_names=names)
    return baselines.predict_proba(model, X)
