"""Sequences must be genuinely consecutive and must not cross a split boundary."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import schema
from src.sequences.builder import (
    build_sequences,
    build_sequences_literal,
    consecutive_runs,
)
from src.splits.temporal import final_week_split, first_test_dates


def test_consecutive_runs_splits_on_a_gap():
    dates = pd.to_datetime(
        ["2020-10-01", "2020-10-02", "2020-10-03", "2020-10-07", "2020-10-08"]
    )
    runs = consecutive_runs(list(dates))
    assert [len(r) for r in runs] == [3, 2]


def test_every_window_spans_exactly_l_calendar_days(synthetic_data):
    for length in (3, 4, 5):
        sequences = build_sequences(
            synthetic_data.daily, synthetic_data.feature_columns,
            sequence_length=length, split_name="all",
        )
        assert len(sequences)
        assert sequences.provenance["is_calendar_consecutive"].all()
        spans = (
            sequences.provenance["end_date"] - sequences.provenance["start_date"]
        ).dt.days
        assert (spans == length - 1).all()


def test_gap_spanning_windows_appear_only_when_asked_for(synthetic_data):
    """The fixture has deliberate gaps, so the two modes must disagree."""
    strict = build_sequences(
        synthetic_data.daily, synthetic_data.feature_columns,
        sequence_length=5, split_name="all", require_consecutive=True,
    )
    naive = build_sequences(
        synthetic_data.daily, synthetic_data.feature_columns,
        sequence_length=5, split_name="all", require_consecutive=False,
    )
    assert len(naive) > len(strict)
    assert (~naive.provenance["is_calendar_consecutive"]).sum() > 0
    assert strict.provenance["is_calendar_consecutive"].all()


def test_no_window_straddles_the_final_week_cut(synthetic_data):
    split = final_week_split(synthetic_data.daily)
    cuts = first_test_dates(synthetic_data.daily)

    for length in (3, 4, 5):
        train = build_sequences(split.train_days, synthetic_data.feature_columns,
                                sequence_length=length, split_name="train")
        test = build_sequences(split.test_days, synthetic_data.feature_columns,
                               sequence_length=length, split_name="test")
        for sequences, expect_test_side in ((train, False), (test, True)):
            for _, row in sequences.provenance.iterrows():
                cut = cuts[row[schema.SUBJECT_ID]]
                flags = [pd.Timestamp(d) >= cut for d in row["raw_dates"]]
                assert len(set(flags)) == 1, "a window sits on both sides of the cut"
                assert flags[0] is expect_test_side


def test_stride_reduces_the_window_count(synthetic_data):
    one = build_sequences(synthetic_data.daily, synthetic_data.feature_columns,
                          sequence_length=5, stride=1, split_name="all")
    two = build_sequences(synthetic_data.daily, synthetic_data.feature_columns,
                          sequence_length=5, stride=2, split_name="all")
    assert len(two) < len(one)


def test_provenance_carries_every_required_field(synthetic_data):
    sequences = build_sequences(
        synthetic_data.daily, synthetic_data.feature_columns,
        sequence_length=4, split_name="train", outer_fold=2, inner_fold=1,
    )
    required = {
        schema.SUBJECT_ID, "start_date", "end_date", "raw_row_ids", "raw_dates",
        "sequence_length", "split_name", "outer_fold", "inner_fold", "sequence_id",
    }
    assert required <= set(sequences.provenance.columns)
    assert (sequences.provenance["outer_fold"] == 2).all()
    assert (sequences.provenance["inner_fold"] == 1).all()
    assert sequences.X.shape == (len(sequences), 4, len(synthetic_data.feature_columns))
    assert sequences.provenance["raw_row_ids"].map(len).eq(4).all()


def test_literal_variant_refuses_to_run_undeclared(synthetic_data):
    cuts = first_test_dates(synthetic_data.daily)
    with pytest.raises(ValueError, match="leakage_diagnostic_only"):
        build_sequences_literal(
            synthetic_data.daily, synthetic_data.feature_columns,
            sequence_length=5, test_start_by_subject=cuts,
        )


def test_literal_variant_measures_the_boundary_crossings(synthetic_data):
    cuts = first_test_dates(synthetic_data.daily)
    _, _, report = build_sequences_literal(
        synthetic_data.daily, synthetic_data.feature_columns,
        sequence_length=5, test_start_by_subject=cuts, leakage_diagnostic_only=True,
    )
    assert report["n_boundary_crossing_sequences"] > 0
    assert report["n_subject_dates_in_both_splits"] > 0
    assert report["mode"] == "paper_literal_variant"


def test_duplicate_dates_are_rejected(synthetic_data):
    doubled = pd.concat(
        [synthetic_data.daily, synthetic_data.daily.iloc[[0]]], ignore_index=True
    ).sort_values([schema.SUBJECT_ID, schema.DATE_COL])
    with pytest.raises(ValueError, match="duplicate dates"):
        build_sequences(doubled, synthetic_data.feature_columns,
                        sequence_length=3, split_name="all")
