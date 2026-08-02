"""No raw daily record, and no identical window, may appear in both splits."""

from __future__ import annotations

import numpy as np

from src.audit.leakage import _duplicate_windows, audit_sequence_split
from src.sequences.builder import build_sequences
from src.splits.group import iter_days, stratified_group_splits
from src.splits.temporal import final_week_split


def test_raw_row_ids_partition_cleanly_in_the_temporal_split(synthetic_data):
    split = final_week_split(synthetic_data.daily)
    for length in (3, 5):
        train = build_sequences(split.train_days, synthetic_data.feature_columns,
                                sequence_length=length, split_name="train")
        test = build_sequences(split.test_days, synthetic_data.feature_columns,
                               sequence_length=length, split_name="test")
        assert not (train.raw_row_ids() & test.raw_row_ids())
        # Every id used must be a real row from the daily table.
        assert (train.raw_row_ids() | test.raw_row_ids()) <= set(
            synthetic_data.daily["raw_row_id"]
        )


def test_raw_row_ids_partition_cleanly_in_the_group_split(synthetic_data):
    split = stratified_group_splits(synthetic_data.subjects, n_splits=4, seed=3)[0]
    train = build_sequences(iter_days(synthetic_data.daily, split.train_subjects),
                            synthetic_data.feature_columns, sequence_length=4,
                            split_name="outer_train")
    test = build_sequences(iter_days(synthetic_data.daily, split.test_subjects),
                           synthetic_data.feature_columns, sequence_length=4,
                           split_name="outer_test")
    assert not (train.raw_row_ids() & test.raw_row_ids())


def test_identical_windows_are_detected(synthetic_data):
    """Copy a window across the split and the duplicate check must see it."""
    split = stratified_group_splits(synthetic_data.subjects, n_splits=4, seed=3)[0]
    train = build_sequences(iter_days(synthetic_data.daily, split.train_subjects),
                            synthetic_data.feature_columns, sequence_length=4,
                            split_name="outer_train")
    test = build_sequences(iter_days(synthetic_data.daily, split.test_subjects),
                           synthetic_data.feature_columns, sequence_length=4,
                           split_name="outer_test")
    assert _duplicate_windows(train, test) == 0

    test.X[0] = train.X[0].copy()
    assert _duplicate_windows(train, test) == 1

    log = audit_sequence_split(
        train, test, context="test", estimand="B",
        sequence_length_source="config_fixed", hyperparameter_source="paper_reported",
    )
    assert "no_identical_windows_across_split" in {r["check"] for r in log.failures}


def test_windows_within_a_subject_overlap_by_design(synthetic_data):
    """Stride-1 windows share days *inside* one split; only across splits is it a leak."""
    sequences = build_sequences(
        synthetic_data.daily, synthetic_data.feature_columns,
        sequence_length=5, stride=1, split_name="all",
    )
    first = set(sequences.provenance["raw_row_ids"].iloc[0])
    second = set(sequences.provenance["raw_row_ids"].iloc[1])
    assert first & second, "consecutive stride-1 windows are expected to overlap"


def test_row_ids_match_the_feature_values(synthetic_data):
    """The stored raw_row_ids must actually be the rows the window contains."""
    sequences = build_sequences(
        synthetic_data.daily, synthetic_data.feature_columns,
        sequence_length=3, split_name="all",
    )
    columns = list(synthetic_data.feature_columns)
    lookup = synthetic_data.daily.set_index("raw_row_id")
    for index in (0, len(sequences) // 2, len(sequences) - 1):
        ids = sequences.provenance["raw_row_ids"].iloc[index]
        expected = lookup.loc[list(ids), columns].to_numpy(dtype=np.float32)
        np.testing.assert_allclose(sequences.X[index], expected, rtol=1e-6)
