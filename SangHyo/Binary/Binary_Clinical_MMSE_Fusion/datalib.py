"""Data + feature construction for the wearable + MMSE fusion model.

Per subject the features are:

* wearable summaries : per-channel median / IQR / trend over the fixed recent
  window (same leakage-safe, label-blind construction as the wearable-only
  experiment), reusing the audited parser; plus
* MMSE scores        : the 32 cognitive item/total scores (diagnosis columns
  excluded, see :mod:`mmse`).

The label still comes only from the Gait/Sleep copies via the audited parser.
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
from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google.data import load_binary_labels

from .mmse import load_mmse_features

SUMMARY_KINDS = ("median", "iqr", "trend")


@dataclass(frozen=True)
class SubjectData:
    subject_ids: np.ndarray
    y: np.ndarray | None
    tabular: np.ndarray                # (n, n_wearable + 32)
    tabular_names: tuple[str, ...]
    wearable_names: tuple[str, ...]
    mmse_names: tuple[str, ...]

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)


def _channel_summaries(window: np.ndarray) -> np.ndarray:
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
                scale = float(np.median(np.abs(values - median))) or (abs(median) + 1e-6)
                out[channel, 2] = slope / scale
    return out.reshape(-1)


def _wearable_tabular(sequences: Sequence[np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(
        f"{channel}__{kind}"
        for channel in COMPACT_DAILY_FEATURES
        for kind in SUMMARY_KINDS
    )
    rows = [
        _channel_summaries(np.asarray(seq, dtype=np.float32)[-VIEW_OBSERVATIONS:])
        for seq in sequences
    ]
    matrix = np.asarray(rows, dtype=np.float64)
    return matrix, names


def load_split(split_root: str | Path, *, require_labels: bool, expected_split: str) -> SubjectData:
    dataset = build_subject_dataset(
        split_root, require_labels=require_labels, expected_split=expected_split
    )
    assert_wearable_schema(dataset.feature_names)
    wearable_matrix, wearable_names = _wearable_tabular(dataset.sequences)
    mmse_matrix, mmse_names = load_mmse_features(split_root, dataset.subject_ids)

    tabular = np.hstack([wearable_matrix, mmse_matrix])
    tabular_names = wearable_names + mmse_names
    return SubjectData(
        subject_ids=np.asarray(dataset.subject_ids, dtype=str),
        y=None if dataset.y is None else np.asarray(dataset.y, dtype=np.int64),
        tabular=tabular,
        tabular_names=tabular_names,
        wearable_names=wearable_names,
        mmse_names=mmse_names,
    )


def hash_subject_id(subject_id: str) -> str:
    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "SubjectData",
    "assert_disjoint_subjects",
    "hash_subject_id",
    "load_binary_labels",
    "load_split",
    "load_validation_labels_checked",
]
