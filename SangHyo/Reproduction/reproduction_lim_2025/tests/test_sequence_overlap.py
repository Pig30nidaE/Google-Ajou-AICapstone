"""Sequences must be built after the split, from that split's days only."""

from __future__ import annotations

import numpy as np
import pytest

from src.audit.leakage import AuditLog, LeakageError, check_sequence_source_overlap
from src.features.representations import build_temporal_sequence
from src.splits import splitters


def _sequences(data, subjects, **kwargs):
    return build_temporal_sequence(
        data.daily, data.feature_columns, subjects=subjects, **kwargs
    )


def test_sequences_from_disjoint_subjects_share_no_source_day(synthetic_data) -> None:
    split = splitters.random_subject_holdout(synthetic_data, test_size=0.25, seed=5)
    train = _sequences(synthetic_data, split.train_subjects)
    test = _sequences(synthetic_data, split.test_subjects)

    log = AuditLog()
    check_sequence_source_overlap(train, test, log)
    assert log.passed


def test_overlapping_sequences_are_rejected(synthetic_data) -> None:
    """Building both sides from all subjects is exactly the mistake to catch."""
    everyone = list(synthetic_data.labels_by_subject().index.astype(str))
    train = _sequences(synthetic_data, everyone)
    test = _sequences(synthetic_data, everyone)

    log = AuditLog()
    with pytest.raises(LeakageError, match="sequence_source_overlap"):
        check_sequence_source_overlap(train, test, log)


def test_one_sequence_per_subject(synthetic_data) -> None:
    rep = _sequences(synthetic_data, None)
    assert rep.n_units == synthetic_data.subjects["subject_id"].nunique()
    assert len(set(map(str, rep.subjects))) == rep.n_units


def test_sequence_shape_and_padding(synthetic_data) -> None:
    rep = _sequences(synthetic_data, None, sequence_length="max", padding="pre")
    n, timesteps, n_features = rep.X.shape
    assert n_features == len(synthetic_data.feature_columns)
    assert timesteps == rep.meta["observed_days_max"]
    assert rep.lengths.max() == timesteps
    assert (rep.lengths <= timesteps).all()

    # pre-padding keeps the most recent observation at the end of the sequence
    short = int(np.argmin(rep.lengths))
    pad = timesteps - int(rep.lengths[short])
    if pad:
        assert np.allclose(rep.X[short, :pad, :], 0.0)
        assert not np.allclose(rep.X[short, -1, :], 0.0)


def test_post_padding_puts_valid_steps_first(synthetic_data) -> None:
    rep = _sequences(synthetic_data, None, sequence_length="max", padding="post")
    short = int(np.argmin(rep.lengths))
    valid = int(rep.lengths[short])
    if valid < rep.X.shape[1]:
        assert np.allclose(rep.X[short, valid:, :], 0.0)


def test_truncation_keeps_the_requested_end(synthetic_data) -> None:
    full = _sequences(synthetic_data, None, sequence_length="max")
    last = _sequences(synthetic_data, None, sequence_length=5, truncation="last")
    first = _sequences(synthetic_data, None, sequence_length=5, truncation="first")

    assert last.X.shape[1] == 5 and first.X.shape[1] == 5
    idx = int(np.argmax(full.lengths))
    assert np.allclose(last.X[idx], full.X[idx, -5:, :])
    start = full.X.shape[1] - int(full.lengths[idx])
    assert np.allclose(first.X[idx], full.X[idx, start:start + 5, :])


def test_calendar_gap_handling_is_at_least_as_long(synthetic_data) -> None:
    compressed = _sequences(synthetic_data, None, gap_handling="compress")
    calendar = _sequences(synthetic_data, None, gap_handling="calendar")
    assert calendar.X.shape[1] >= compressed.X.shape[1]
    assert calendar.meta["gap_handling"] == "calendar"


def test_subset_preserves_alignment(synthetic_data) -> None:
    rep = _sequences(synthetic_data, None)
    keep = list(map(str, rep.subjects[:4]))
    subset = rep.subset(rep.mask_for_subjects(keep))
    assert subset.n_units == 4
    assert sorted(map(str, subset.subjects)) == sorted(keep)
    assert len(subset.y) == 4 and len(subset.lengths) == 4
