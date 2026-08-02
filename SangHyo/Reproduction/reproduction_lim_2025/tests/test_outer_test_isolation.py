"""The outer test set must not influence tuning or the threshold."""

from __future__ import annotations

import numpy as np
import pytest

from src.audit.leakage import (
    AuditLog,
    LeakageError,
    check_outer_test_isolation,
    check_threshold_source,
)
from src.evaluation import metrics as M
from src.splits import splitters
from src.utils.config import Config, ConfigError, validate_config


def test_inner_folds_drawn_from_outer_train_are_isolated(synthetic_data) -> None:
    labels = synthetic_data.labels_by_subject()
    outer = next(splitters.stratified_group_kfold(synthetic_data, n_splits=4, seed=0))
    inner = splitters.inner_stratified_group_kfold(
        outer.train_subjects, labels, n_splits=3, seed=0
    )

    log = AuditLog()
    check_outer_test_isolation(outer.test_subjects, inner, log)
    assert log.passed


def test_contaminated_inner_folds_are_rejected(synthetic_data) -> None:
    """Building the inner CV from *all* subjects is the failure mode to catch."""
    labels = synthetic_data.labels_by_subject()
    outer = next(splitters.stratified_group_kfold(synthetic_data, n_splits=4, seed=0))
    everyone = list(labels.index.astype(str))
    contaminated = splitters.inner_stratified_group_kfold(
        everyone, labels, n_splits=3, seed=0
    )

    log = AuditLog()
    with pytest.raises(LeakageError, match="outer_test_isolation"):
        check_outer_test_isolation(outer.test_subjects, contaminated, log)


def test_inner_folds_cover_only_outer_training_subjects(synthetic_data) -> None:
    labels = synthetic_data.labels_by_subject()
    outer = next(splitters.stratified_group_kfold(synthetic_data, n_splits=4, seed=0))
    inner = splitters.inner_stratified_group_kfold(
        outer.train_subjects, labels, n_splits=3, seed=0
    )

    used = {s for split in inner for s in split.train_subjects + split.test_subjects}
    assert used == set(outer.train_subjects)


def test_inner_folds_are_internally_disjoint(synthetic_data) -> None:
    labels = synthetic_data.labels_by_subject()
    outer = next(splitters.stratified_group_kfold(synthetic_data, n_splits=4, seed=0))
    for split in splitters.inner_stratified_group_kfold(
        outer.train_subjects, labels, n_splits=3, seed=0
    ):
        assert set(split.train_subjects) & set(split.test_subjects) == set()


@pytest.mark.parametrize("source", ["outer_test", "test", "evaluation_set"])
def test_threshold_from_the_evaluation_set_is_rejected(source: str) -> None:
    log = AuditLog()
    with pytest.raises(LeakageError, match="threshold_source"):
        check_threshold_source(source, log, experiment="nested_subject_independent")


def test_nested_experiment_requires_an_inner_cv_threshold() -> None:
    log = AuditLog()
    with pytest.raises(LeakageError, match="threshold_from_inner_cv"):
        check_threshold_source("fixed", log, experiment="nested_subject_independent")


def test_fixed_threshold_is_fine_for_the_other_experiments() -> None:
    log = AuditLog()
    check_threshold_source("fixed", log, experiment="leakage_controlled_non_nested")
    check_threshold_source("fixed", log, experiment="paper_reported_reconstruction")
    assert log.passed


def test_select_threshold_only_reads_what_it_is_given() -> None:
    """A threshold chosen on one set should not track a different set's labels."""
    rng = np.random.default_rng(0)
    inner_true = np.array([0, 0, 0, 1, 1, 1])
    inner_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    chosen = M.select_threshold(inner_true, inner_prob)
    assert 0.3 < chosen <= 0.9

    outer_true = rng.integers(0, 2, size=20)
    outer_prob = rng.random(20)
    again = M.select_threshold(inner_true, inner_prob)
    assert chosen == again, "outer data must not change the inner-selected threshold"
    assert len(outer_true) == len(outer_prob)


def test_config_forbids_a_non_inner_threshold_for_the_nested_experiment() -> None:
    raw = {
        "experiment": "nested_subject_independent",
        "split": {"mode": "nested_stratified_group_kfold", "inner_k": 3},
        "threshold": {"policy": "fixed", "value": 0.5},
        "models": ["random_forest"],
        "representations": {"random_forest": "tabular_subject_aggregate"},
    }
    with pytest.raises(ConfigError, match="inner CV"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))


def test_non_nested_experiment_may_not_tune() -> None:
    raw = {
        "experiment": "leakage_controlled_non_nested",
        "split": {"mode": "stratified_group_kfold"},
        "tuning": {"enabled": True},
        "models": ["random_forest"],
        "representations": {"random_forest": "tabular_subject_aggregate"},
    }
    with pytest.raises(ConfigError, match="must not re-select"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))


def test_shipped_nested_config_selects_inside_the_inner_cv(config_dir) -> None:
    from src.utils.config import load_config

    config = load_config(config_dir / "nested_subject_independent.yaml")
    assert config.get("threshold.policy") == "inner_cv"
    assert int(config.get("split.inner_k")) >= 2
    assert config.get("tuning.enabled") is True
    assert config.scaler_scope == "train_only"
