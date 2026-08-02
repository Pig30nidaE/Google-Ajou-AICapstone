"""Imputation / standardisation must be fitted inside the training fold only."""

from __future__ import annotations

import numpy as np
import pytest

from src.audit.leakage import AuditLog, LeakageError, check_preprocessing_scope
from src.features.representations import (
    FoldPreprocessor,
    PreprocessingScopeError,
    build_tabular_subject_aggregate,
)
from src.splits import splitters


def test_transform_before_fit_raises() -> None:
    pre = FoldPreprocessor()
    with pytest.raises(PreprocessingScopeError, match="before fit"):
        pre.transform(np.zeros((3, 4)))


def test_fit_records_the_subjects_it_saw() -> None:
    pre = FoldPreprocessor()
    pre.fit(np.random.default_rng(0).normal(size=(4, 3)),
            subjects=["b", "a", "a", "c"], feature_names=["f1", "f2", "f3"])
    assert pre.fitted_subjects == ("a", "b", "c")
    assert pre.audit_record()["n_fitted_subjects"] == 3


def test_scope_check_rejects_fitting_on_test_subjects() -> None:
    pre = FoldPreprocessor()
    pre.fit(np.zeros((3, 2)), subjects=["a", "b", "z"], feature_names=["f1", "f2"])
    log = AuditLog()
    with pytest.raises(LeakageError, match="preprocessor_excludes_test"):
        check_preprocessing_scope(pre, ["a", "b", "z"], ["z"], log)


def test_scope_check_rejects_subjects_outside_the_training_fold() -> None:
    pre = FoldPreprocessor()
    pre.fit(np.zeros((3, 2)), subjects=["a", "b", "q"], feature_names=["f1", "f2"])
    log = AuditLog()
    with pytest.raises(LeakageError, match="preprocessor_within_train"):
        check_preprocessing_scope(pre, ["a", "b"], ["c"], log)


def test_scope_check_rejects_an_unfitted_preprocessor() -> None:
    log = AuditLog()
    with pytest.raises(LeakageError, match="preprocessor_fitted"):
        check_preprocessing_scope(FoldPreprocessor(), ["a"], ["b"], log)


def test_correct_fold_local_usage_passes(synthetic_data) -> None:
    split = splitters.random_subject_holdout(synthetic_data, test_size=0.25, seed=2)
    train = build_tabular_subject_aggregate(
        synthetic_data.daily, synthetic_data.feature_columns, subjects=split.train_subjects
    )
    pre = FoldPreprocessor()
    pre.fit(train.X, subjects=split.train_subjects, feature_names=train.feature_names)

    log = AuditLog()
    check_preprocessing_scope(pre, split.train_subjects, split.test_subjects, log)
    assert log.passed


def test_statistics_come_only_from_training_rows(synthetic_data) -> None:
    """A wild outlier among test subjects must not move the fitted mean."""
    split = splitters.random_subject_holdout(synthetic_data, test_size=0.25, seed=4)
    daily = synthetic_data.daily.copy()
    feature = synthetic_data.feature_columns[0]
    daily.loc[daily["subject_id"].isin(split.test_subjects), feature] = 1e6

    train = build_tabular_subject_aggregate(
        daily, synthetic_data.feature_columns, subjects=split.train_subjects
    )
    pre = FoldPreprocessor()
    pre.fit(train.X, subjects=split.train_subjects, feature_names=train.feature_names)
    assert abs(float(pre.mean_[0])) < 100.0, "test-only outlier leaked into the scaler"


def test_all_data_scope_is_forbidden_outside_experiment_a() -> None:
    from src.utils.config import Config, ConfigError, validate_config

    raw = {
        "experiment": "nested_subject_independent",
        "split": {"mode": "nested_stratified_group_kfold", "inner_k": 3},
        "threshold": {"policy": "inner_cv"},
        "preprocessing": {"scaler_scope": "all_data"},
        "models": ["random_forest"],
        "representations": {"random_forest": "tabular_subject_aggregate"},
    }
    with pytest.raises(ConfigError, match="scaler_scope must be 'train_only'"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))


def test_imputation_uses_training_medians_only() -> None:
    train = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    pre = FoldPreprocessor(standardize=False, impute=True)
    pre.fit(train, subjects=["a", "b", "c"], feature_names=["f1", "f2"])
    filled = pre.transform(np.array([[np.nan, np.nan]]))
    assert np.allclose(filled, [[3.0, 30.0]]), "should use the training medians"


def test_zero_variance_column_does_not_produce_nan() -> None:
    train = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    pre = FoldPreprocessor(standardize=True)
    out = pre.fit_transform(train, subjects=["a", "b", "c"], feature_names=["f1", "const"])
    assert np.isfinite(out).all()
    assert np.allclose(out[:, 1], 0.0)


def test_sequence_tensors_are_scaled_per_feature(synthetic_data) -> None:
    from src.features.representations import build_temporal_sequence

    rep = build_temporal_sequence(synthetic_data.daily, synthetic_data.feature_columns)
    pre = FoldPreprocessor()
    scaled = pre.fit_transform(
        rep.X, subjects=rep.subjects, feature_names=rep.feature_names
    )
    assert scaled.shape == rep.X.shape
    assert np.isfinite(scaled).all()
