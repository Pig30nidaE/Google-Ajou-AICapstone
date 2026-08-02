"""No raw daily record may be used on both sides of a split."""

from __future__ import annotations

import numpy as np
import pytest

from src.audit.leakage import AuditLog, LeakageError, check_row_overlap
from src.data import schema
from src.features.representations import build_daily_record
from src.splits import splitters


def test_check_raises_on_shared_row_id() -> None:
    log = AuditLog()
    with pytest.raises(LeakageError, match="row_overlap"):
        check_row_overlap([1, 2, 3], [3, 4], log)


def test_check_passes_on_disjoint_row_ids() -> None:
    log = AuditLog()
    check_row_overlap([1, 2, 3], [4, 5], log)
    assert log.passed


def test_padding_sentinel_is_not_treated_as_a_shared_row() -> None:
    """``-1`` marks padded calendar slots and must not count as an overlap."""
    log = AuditLog()
    check_row_overlap([-1, 1, 2], [-1, 3, 4], log)
    assert log.passed


def test_subjectwise_daily_split_has_no_row_overlap(synthetic_data) -> None:
    split = splitters.random_subject_holdout(synthetic_data, test_size=0.25, seed=3)
    train = build_daily_record(
        synthetic_data.daily, synthetic_data.feature_columns, subjects=split.train_subjects
    )
    test = build_daily_record(
        synthetic_data.daily, synthetic_data.feature_columns, subjects=split.test_subjects
    )
    assert set(train.row_ids.tolist()) & set(test.row_ids.tolist()) == set()

    log = AuditLog()
    check_row_overlap(train.row_ids, test.row_ids, log)
    assert log.passed


def test_row_level_holdout_actually_leaks_subjects(synthetic_data) -> None:
    """The diagnostic variant must be measurably leaky, and say so."""
    row_split = splitters.random_row_holdout(synthetic_data, test_size=0.2, seed=11)

    assert set(row_split.train_rows) & set(row_split.test_rows) == set(), (
        "rows themselves must still be disjoint"
    )
    assert row_split.meta["n_shared_subjects"] > 0, (
        "the point of this variant is that subjects span the split"
    )
    assert row_split.meta["leakage_expected"] is True
    assert row_split.meta["interpret_as"] == "leakage_diagnostic_not_performance"


def test_row_ids_are_unique_across_the_dataset(synthetic_data) -> None:
    ids = synthetic_data.daily["row_id"].to_numpy()
    assert len(np.unique(ids)) == len(ids)


def test_daily_record_rows_carry_their_subject(synthetic_data) -> None:
    rep = build_daily_record(synthetic_data.daily, synthetic_data.feature_columns)
    assert len(rep.subjects) == len(rep.X)
    assert len(rep.row_ids) == len(rep.X)
    expected = synthetic_data.daily.groupby(schema.SUBJECT_ID).size().sum()
    assert rep.n_units == expected
