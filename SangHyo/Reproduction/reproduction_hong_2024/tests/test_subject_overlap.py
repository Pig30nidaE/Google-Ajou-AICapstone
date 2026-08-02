"""No subject may appear on both sides of a subject-independent split."""

from __future__ import annotations

import pytest

from src.audit.leakage import LeakageError, audit_sequence_split
from src.data import schema
from src.sequences.builder import build_sequences
from src.splits.group import inner_splits, iter_days, stratified_group_splits


def test_group_kfold_never_shares_a_subject(synthetic_data):
    splits = stratified_group_splits(synthetic_data.subjects, n_splits=4, seed=7)
    assert len(splits) == 4
    for split in splits:
        assert not set(split.train_subjects) & set(split.test_subjects)


def test_every_subject_is_tested_exactly_once(synthetic_data):
    splits = stratified_group_splits(synthetic_data.subjects, n_splits=4, seed=7)
    tested = [s for split in splits for s in split.test_subjects]
    assert sorted(tested) == sorted(synthetic_data.subject_ids())


def test_sequences_inherit_the_subject_partition(synthetic_data):
    split = stratified_group_splits(synthetic_data.subjects, n_splits=4, seed=7)[0]
    train = build_sequences(
        iter_days(synthetic_data.daily, split.train_subjects),
        synthetic_data.feature_columns, sequence_length=5, split_name="outer_train",
    )
    test = build_sequences(
        iter_days(synthetic_data.daily, split.test_subjects),
        synthetic_data.feature_columns, sequence_length=5, split_name="outer_test",
    )
    assert not set(train.subjects.tolist()) & set(test.subjects.tolist())

    log = audit_sequence_split(
        train, test, context="test", estimand="B",
        sequence_length_source="config_fixed", hyperparameter_source="paper_reported",
    )
    log.raise_if_failed()


def test_audit_rejects_a_shared_subject(synthetic_data):
    """A split built the wrong way must fail rather than quietly score well."""
    subjects = list(synthetic_data.subject_ids())
    train_days = iter_days(synthetic_data.daily, tuple(subjects[:12]))
    # Deliberately overlapping: subject 11 is on both sides.
    test_days = iter_days(synthetic_data.daily, tuple(subjects[11:]))

    train = build_sequences(train_days, synthetic_data.feature_columns,
                            sequence_length=3, split_name="outer_train")
    test = build_sequences(test_days, synthetic_data.feature_columns,
                           sequence_length=3, split_name="outer_test")

    log = audit_sequence_split(
        train, test, context="test", estimand="B",
        sequence_length_source="config_fixed", hyperparameter_source="paper_reported",
    )
    assert not log.passed
    failed = {r["check"] for r in log.failures}
    assert "no_subject_overlap" in failed
    with pytest.raises(LeakageError):
        log.raise_if_failed()


def test_inner_folds_never_see_outer_test_subjects(synthetic_data):
    outer = stratified_group_splits(synthetic_data.subjects, n_splits=4, seed=7)[0]
    for inner in inner_splits(synthetic_data.subjects, outer, n_splits=3, seed=7):
        seen = set(inner.train_subjects) | set(inner.test_subjects)
        assert not seen & set(outer.test_subjects)
        assert seen <= set(outer.train_subjects)


def test_subject_id_is_not_a_feature(synthetic_data):
    assert schema.SUBJECT_ID not in synthetic_data.feature_columns
