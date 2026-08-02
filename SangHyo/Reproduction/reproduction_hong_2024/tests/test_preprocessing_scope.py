"""The scaler must see training sequences and nothing else."""

from __future__ import annotations

import numpy as np
import pytest

from src.audit.leakage import audit_sequence_split
from src.preprocessing.scaler import SequenceScaler, fingerprint, represent
from src.sequences.builder import build_sequences
from src.splits.group import iter_days, stratified_group_splits
from src.utils.config import ConfigError, Config, validate_config


def _pair(data, length: int = 5):
    split = stratified_group_splits(data.subjects, n_splits=4, seed=11)[0]
    train = build_sequences(iter_days(data.daily, split.train_subjects),
                            data.feature_columns, sequence_length=length,
                            split_name="outer_train")
    test = build_sequences(iter_days(data.daily, split.test_subjects),
                           data.feature_columns, sequence_length=length,
                           split_name="outer_test")
    return train, test


def test_scaler_statistics_come_only_from_train(synthetic_data):
    train, test = _pair(synthetic_data)
    scaler = SequenceScaler(method="standard")
    scaled_train, scaled_test = scaler.fit_transform_pair(train, test)

    # The training data must be standardised; the test data is transformed with
    # the *training* statistics, so it is not expected to be exactly centred.
    flat = scaled_train.X.reshape(-1, scaled_train.X.shape[-1])
    np.testing.assert_allclose(flat.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(flat.std(axis=0), 1.0, atol=1e-4)

    reference = SequenceScaler(method="standard").fit(train)
    np.testing.assert_allclose(scaler.mean_, reference.mean_)
    np.testing.assert_allclose(scaler.scale_, reference.scale_)

    test_flat = scaled_test.X.reshape(-1, scaled_test.X.shape[-1])
    assert not np.allclose(test_flat.mean(axis=0), 0.0, atol=1e-4)


def test_scaler_fitted_on_the_wrong_split_is_caught(synthetic_data):
    train, test = _pair(synthetic_data)
    leaky = SequenceScaler(method="standard").fit(test)     # the mistake
    log = audit_sequence_split(
        train, test, context="test", estimand="B",
        scaler=leaky, scaler_fit_source=train,
        sequence_length_source="config_fixed", hyperparameter_source="paper_reported",
    )
    assert "scaler_fitted_on_train_only" in {r["check"] for r in log.failures}


def test_fingerprint_distinguishes_the_splits(synthetic_data):
    train, test = _pair(synthetic_data)
    assert fingerprint(train) != fingerprint(test)
    assert fingerprint(train) == fingerprint(train)


def test_transform_does_not_refit(synthetic_data):
    train, test = _pair(synthetic_data)
    scaler = SequenceScaler(method="standard").fit(train)
    before = scaler.mean_.copy()
    scaler.transform(test)
    np.testing.assert_allclose(scaler.mean_, before)


def test_constant_feature_does_not_divide_by_zero(synthetic_data):
    train, test = _pair(synthetic_data)
    train.X[:, :, 0] = 3.0
    scaler = SequenceScaler(method="standard")
    scaled_train, _ = scaler.fit_transform_pair(train, test)
    assert np.isfinite(scaled_train.X).all()
    assert scaler.meta["n_degenerate_features"] >= 1


def test_baseline_representations_keep_the_row_count(synthetic_data):
    train, _ = _pair(synthetic_data, length=4)
    n, length, n_features = train.X.shape
    assert represent(train, "flatten").shape == (n, length * n_features)
    assert represent(train, "mean").shape == (n, n_features)
    assert represent(train, "last_day").shape == (n, n_features)
    assert represent(train, "summary").shape == (n, 4 * n_features)
    np.testing.assert_allclose(represent(train, "last_day"), train.X[:, -1, :])


def test_config_rejects_a_non_train_scaler_scope():
    config = Config(
        experiment="fixed_subject_independent",
        raw={
            "experiment": "fixed_subject_independent",
            "split": {"mode": "stratified_group_kfold"},
            "preprocessing": {"scaler_scope": "all_data"},
            "models": {"enabled": ["lstm"]},
        },
    )
    with pytest.raises(ConfigError, match="scaler_scope"):
        validate_config(config)
