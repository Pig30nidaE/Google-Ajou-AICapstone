"""Wearable-only data + feature construction (MMSE strictly excluded).

The audited raw parser is reused from ``Binary_Wearable_BalancedFusion_Google``
so the raw JSON/CSV alignment and the fail-closed MMSE exclusion have a single
implementation.  This module adds two label-blind, per-subject representations:

* ``sequences``  : one variable-length ``(days, 56)`` matrix per subject, from
  which fixed equal-count temporal windows are cropped for the Conv1D+BiLSTM.
* ``tabular``    : per-channel robust summaries (median / IQR / trend) over the
  fixed recent window, for the tree/linear tabular ensemble.

No identifier, absolute date, observation count, coverage, mask, or cognitive
value ever enters either representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from SangHyo.Binary.Binary_Wearable_BalancedFusion_Google.data import (
    COMPACT_DAILY_FEATURES,
    VIEW_OBSERVATIONS,
    assert_disjoint_subjects,
    assert_wearable_schema,
    build_subject_dataset,
    load_validation_labels_checked,
)
from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google.data import (
    load_binary_labels,
    make_fixed_views,
)

# 7-day windows, 8 equally spaced crops per subject.  Every subject contributes
# the same number of windows regardless of how many days were recorded, so a
# long recording cannot dominate training (leakage-safe crop contract reused
# from the SequenceFusion experiment).
WINDOW_DAYS = 7
WINDOWS_PER_SUBJECT = 8

# Robust per-channel summaries.  Kept deliberately small (3 x 56 = 168
# candidates) because the TabNet report showed 1,077 candidates on 141 subjects
# overfit badly; fold-local selection trims this further.
SUMMARY_KINDS = ("median", "iqr", "trend")


@dataclass(frozen=True)
class SubjectData:
    """Everything the models need for one split, keyed by subject."""

    subject_ids: np.ndarray            # (n,) original ids (join key only)
    y: np.ndarray | None               # (n,) 0=CN, 1=MCI+DEM, or None
    sequences: list[np.ndarray]        # n items of (days, 56)
    tabular: np.ndarray                # (n, 168) label-blind summaries
    tabular_names: tuple[str, ...]
    channel_names: tuple[str, ...]     # 56 daily channels

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)


def _channel_summaries(window: np.ndarray) -> np.ndarray:
    """Median / IQR / normalized trend for each of the 56 channels.

    ``window`` is ``(days, 56)`` and may contain NaN.  Trend is the ordinary
    least-squares slope of a channel against a 0..1 time axis, divided by the
    channel's own scale so activity and sleep units stay comparable.  All
    summaries use only this subject's own data, so computing them once (outside
    any fold) introduces no cross-subject leakage.
    """

    days, n_channels = window.shape
    out = np.full((n_channels, len(SUMMARY_KINDS)), np.nan, dtype=np.float64)
    time_axis = np.linspace(0.0, 1.0, days) if days > 1 else np.zeros(1)
    for channel in range(n_channels):
        column = window[:, channel].astype(np.float64)
        finite = np.isfinite(column)
        count = int(finite.sum())
        if count == 0:
            continue
        values = column[finite]
        median = float(np.median(values))
        out[channel, 0] = median
        if count >= 2:
            q25, q75 = np.quantile(values, [0.25, 0.75])
            out[channel, 1] = float(q75 - q25)
        if count >= 3:
            t = time_axis[finite]
            t_centered = t - t.mean()
            denom = float(np.sum(t_centered ** 2))
            if denom > 1e-12:
                slope = float(np.sum(t_centered * (values - values.mean())) / denom)
                scale = float(np.median(np.abs(values - median))) or (
                    abs(median) + 1e-6
                )
                out[channel, 2] = slope / scale
    return out.reshape(-1)


def _build_tabular(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(
        f"{channel}__{kind}"
        for channel in COMPACT_DAILY_FEATURES
        for kind in SUMMARY_KINDS
    )
    rows = []
    for sequence in sequences:
        window = np.asarray(sequence, dtype=np.float32)[-VIEW_OBSERVATIONS:]
        rows.append(_channel_summaries(window))
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape[1] != len(names):
        raise AssertionError("Tabular summary width mismatch")
    return matrix, names


def load_split(split_root: str | Path, *, require_labels: bool, expected_split: str) -> SubjectData:
    """Load one split as sequences + tabular summaries, MMSE never opened."""

    dataset = build_subject_dataset(
        split_root, require_labels=require_labels, expected_split=expected_split
    )
    assert_wearable_schema(dataset.feature_names)
    sequences = [np.asarray(seq, dtype=np.float32) for seq in dataset.sequences]
    tabular, tabular_names = _build_tabular(sequences)
    return SubjectData(
        subject_ids=np.asarray(dataset.subject_ids, dtype=str),
        y=None if dataset.y is None else np.asarray(dataset.y, dtype=np.int64),
        sequences=sequences,
        tabular=tabular,
        tabular_names=tabular_names,
        channel_names=tuple(dataset.feature_names),
    )


def make_windows(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(n*8, 7, 56)`` windows and the subject index of every window."""

    return make_fixed_views(
        list(sequences), sequence_length=WINDOW_DAYS, n_views=WINDOWS_PER_SUBJECT
    )


def hash_subject_id(subject_id: str) -> str:
    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "COMPACT_DAILY_FEATURES",
    "SubjectData",
    "VIEW_OBSERVATIONS",
    "WINDOWS_PER_SUBJECT",
    "WINDOW_DAYS",
    "assert_disjoint_subjects",
    "hash_subject_id",
    "load_binary_labels",
    "load_split",
    "load_validation_labels_checked",
    "make_windows",
]
