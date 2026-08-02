"""The same subject-day must never land in both splits.

Checking bare calendar dates is not enough: two different subjects sharing
2020-11-05 is normal.  The key that matters is (subject, date).
"""

from __future__ import annotations

import pandas as pd

from src.audit.leakage import audit_sequence_split, audit_temporal_split
from src.data import schema
from src.sequences.builder import build_sequences
from src.splits.temporal import assert_no_shared_days, final_week_split, n_dates_in_common


def test_final_week_split_shares_no_subject_day(synthetic_data):
    split = final_week_split(synthetic_data.daily)
    assert_no_shared_days(split)
    assert n_dates_in_common(split.train_days, split.test_days) == 0


def test_test_period_is_strictly_after_train(synthetic_data):
    split = final_week_split(synthetic_data.daily)
    for subject, group in split.train_days.groupby(schema.SUBJECT_ID):
        test_group = split.test_days[split.test_days[schema.SUBJECT_ID] == subject]
        if len(test_group):
            assert group[schema.DATE_COL].max() < test_group[schema.DATE_COL].min()


def test_sequences_do_not_share_raw_dates(synthetic_data):
    split = final_week_split(synthetic_data.daily)
    for length in (3, 4, 5):
        train = build_sequences(split.train_days, synthetic_data.feature_columns,
                                sequence_length=length, split_name="train")
        test = build_sequences(split.test_days, synthetic_data.feature_columns,
                               sequence_length=length, split_name="test")
        assert not (train.subject_date_pairs() & test.subject_date_pairs())
        assert not (train.raw_row_ids() & test.raw_row_ids())


def test_embargo_widens_the_gap_between_train_and_test(synthetic_data):
    plain = final_week_split(synthetic_data.daily, embargo_days=0)
    embargoed = final_week_split(synthetic_data.daily, embargo_days=4)
    assert len(embargoed.train_days) < len(plain.train_days)
    assert embargoed.meta["n_embargoed_rows"] > 0
    assert len(embargoed.test_days) == len(plain.test_days)   # test is untouched

    for subject, group in embargoed.train_days.groupby(schema.SUBJECT_ID):
        test_group = embargoed.test_days[embargoed.test_days[schema.SUBJECT_ID] == subject]
        if len(test_group):
            gap = (test_group[schema.DATE_COL].min() - group[schema.DATE_COL].max()).days
            assert gap > 4


def test_validation_period_is_disjoint_from_both(synthetic_data):
    split = final_week_split(synthetic_data.daily, embargo_days=2, validation_days=5)
    assert split.validation_days is not None and len(split.validation_days)
    assert_no_shared_days(split)
    assert n_dates_in_common(split.validation_days, split.test_days) == 0
    assert n_dates_in_common(split.validation_days, split.train_days) == 0


def test_temporal_audit_reports_a_clean_split(synthetic_data):
    split = final_week_split(synthetic_data.daily, embargo_days=4)
    log = audit_temporal_split(split, sequence_length=5)
    log.raise_if_failed()
    assert log.passed


def test_audit_catches_a_hand_broken_split(synthetic_data):
    """Move one test day back into train and the audit must notice."""
    split = final_week_split(synthetic_data.daily)
    stolen = split.test_days.iloc[[0]]
    broken_train = pd.concat([split.train_days, stolen], ignore_index=True)

    train = build_sequences(broken_train, synthetic_data.feature_columns,
                            sequence_length=3, split_name="train")
    test = build_sequences(split.test_days, synthetic_data.feature_columns,
                           sequence_length=3, split_name="test")
    log = audit_sequence_split(
        train, test, context="test", estimand="A", expect_subject_overlap=True,
        sequence_length_source="config_fixed", hyperparameter_source="paper_reported",
    )
    assert not log.passed
    assert "no_shared_subject_dates" in {r["check"] for r in log.failures}
