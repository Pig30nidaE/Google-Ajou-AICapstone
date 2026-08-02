"""Undersampling is a training-set operation only."""

from __future__ import annotations

import numpy as np
import pytest

from src.data import schema
from src.sampling.undersample import class_weights, undersample
from src.sequences.builder import build_sequences
from src.splits.group import iter_days, stratified_group_splits


def _pair(data, length: int = 5):
    split = stratified_group_splits(data.subjects, n_splits=4, seed=5)[0]
    train = build_sequences(iter_days(data.daily, split.train_subjects),
                            data.feature_columns, sequence_length=length,
                            split_name="outer_train")
    test = build_sequences(iter_days(data.daily, split.test_subjects),
                           data.feature_columns, sequence_length=length,
                           split_name="outer_test")
    return train, test


def test_undersampling_balances_the_training_classes(synthetic_data):
    train, _ = _pair(synthetic_data)
    sampled, report = undersample(train, strategy="random_sequence", target_ratio=1.0, seed=1)
    counts = np.bincount(sampled.y, minlength=2)
    assert counts[0] == counts[1]
    assert report["applied"] and report["n_removed"] > 0
    assert len(sampled) < len(train)


def test_minority_sequences_are_never_dropped(synthetic_data):
    train, _ = _pair(synthetic_data)
    sampled, report = undersample(train, seed=1)
    minority = report["minority_class"]
    assert int((sampled.y == minority).sum()) == int((train.y == minority).sum())


def test_undersampling_refuses_a_test_split(synthetic_data):
    _, test = _pair(synthetic_data)
    with pytest.raises(AssertionError, match="only ever allowed on a training split"):
        undersample(test, seed=1)


def test_subject_balanced_strategy_keeps_every_majority_subject(synthetic_data):
    train, _ = _pair(synthetic_data)
    before = set(train.provenance[schema.SUBJECT_ID])
    sampled, report = undersample(train, strategy="subject_balanced", seed=1)
    after = set(sampled.provenance[schema.SUBJECT_ID])
    assert before == after
    assert not any("완전히 제거" in w for w in report["warnings"])
    counts = np.bincount(sampled.y, minlength=2)
    assert abs(int(counts[0]) - int(counts[1])) <= len(before)


def test_random_strategy_reports_its_risks(synthetic_data):
    """The diagnostics must be produced whether or not they trigger here."""
    train, _ = _pair(synthetic_data)
    _, report = undersample(train, strategy="random_sequence", seed=1)
    assert set(report) >= {"before", "after", "warnings", "n_removed", "strategy"}
    assert report["before"]["n_subjects"] >= report["after"]["n_subjects"]
    assert report["after"]["n_positive_subjects"] >= 1
    assert report["after"]["n_negative_subjects"] >= 1


def test_none_strategy_is_a_no_op(synthetic_data):
    train, _ = _pair(synthetic_data)
    sampled, report = undersample(train, strategy="none", seed=1)
    assert len(sampled) == len(train)
    assert report["applied"] is False


def test_sampling_is_deterministic_for_a_seed(synthetic_data):
    train, _ = _pair(synthetic_data)
    first, _ = undersample(train, seed=7)
    second, _ = undersample(train, seed=7)
    third, _ = undersample(train, seed=8)
    assert list(first.provenance["sequence_id"]) == list(second.provenance["sequence_id"])
    assert list(first.provenance["sequence_id"]) != list(third.provenance["sequence_id"])


def test_class_weights_are_the_alternative_to_dropping_data():
    y = np.array([0] * 90 + [1] * 10)
    weights = class_weights(y)
    assert weights[1] > weights[0]
    assert np.isclose(weights[0] * 90 + weights[1] * 10, 100.0)
