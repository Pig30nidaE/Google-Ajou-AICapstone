"""Fail-closed Activity/Sleep-only data contract.

The audited parser from ``Binary_Wearable_TabNet_Google`` is reused so raw
JSON alignment has one implementation.  This module immediately narrows that
119-channel representation to a label-independent, predeclared physiological
allow-list.  Cognitive/MMSE paths are never resolved or opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from SangHyo.Binary.Binary_Wearable_TabNet_Google.data import (
    SubjectSequenceDataset,
    build_subject_dataset as _build_audited_wearable_dataset,
    load_binary_labels,
)


VIEW_OBSERVATIONS = 35

# Declared before looking at any Validation labels.  The list covers activity
# volume/intensity/rhythm and sleep timing/stages/autonomic physiology while
# avoiding IDs, collection coverage, absolute dates, diagnosis and cognition.
COMPACT_DAILY_FEATURES = (
    "activity__circadian__first_harmonic_ratio",
    "activity__circadian__l5",
    "activity__circadian__m10",
    "activity__circadian__peak_cos",
    "activity__circadian__peak_sin",
    "activity__circadian__relative_amplitude",
    "activity__state__transition_rate",
    "activity__met__iqr",
    "activity__met__mean",
    "activity__met__sedentary_ratio",
    "activity__met__vigorous_ratio",
    "activity__scalar__average_met",
    "activity__scalar__cal_active",
    "activity__scalar__daily_movement",
    "activity__scalar__inactive",
    "activity__scalar__score",
    "activity__scalar__score_training_frequency",
    "activity__scalar__score_training_volume",
    "activity__scalar__steps",
    "activity__state__entropy",
    "activity__state__high_ratio",
    "activity__state__inactive_ratio",
    "activity__state__rest_ratio",
    "sleep__clock__bedtime_cos",
    "sleep__clock__bedtime_sin",
    "sleep__clock__midpoint_cos",
    "sleep__clock__midpoint_sin",
    "sleep__clock__wake_cos",
    "sleep__clock__wake_sin",
    "sleep__duration__awake_ratio",
    "sleep__duration__deep_ratio",
    "sleep__duration__light_ratio",
    "sleep__duration__rem_ratio",
    "sleep__hr__drop_ratio",
    "sleep__hr__mean",
    "sleep__hr__std",
    "sleep__hr_rmssd__correlation",
    "sleep__rmssd__mean",
    "sleep__rmssd__std",
    "sleep__scalar__awake",
    "sleep__scalar__breath_average",
    "sleep__scalar__deep",
    "sleep__scalar__efficiency",
    "sleep__scalar__hr_average",
    "sleep__scalar__hr_lowest",
    "sleep__scalar__onset_latency",
    "sleep__scalar__rem",
    "sleep__scalar__restless",
    "sleep__scalar__rmssd",
    "sleep__scalar__score",
    "sleep__scalar__score_deep",
    "sleep__scalar__score_rem",
    "sleep__scalar__score_total",
    "sleep__scalar__temperature_delta",
    "sleep__stage__entropy",
    "sleep__stage__transition_rate",
)

FORBIDDEN_FRAGMENTS = (
    "mmse",
    "cognitive",
    "diagnosis",
    "diag",
    "label",
    "subject_id",
    "email",
    "coverage",
    "count",
    "length",
    "mask",
    "nonwear",
    "non_wear",
    "absolute_date",
    "timestamp",
)


def assert_wearable_schema(names: Sequence[str]) -> None:
    """Reject any schema that is not exactly wearable and non-identifying."""

    values = [str(name) for name in names]
    if not values or len(values) != len(set(values)):
        raise AssertionError("Wearable feature names must be non-empty and unique")
    offenders = [
        name
        for name in values
        if not name.startswith(("activity__", "sleep__"))
        or any(fragment in name.lower() for fragment in FORBIDDEN_FRAGMENTS)
    ]
    if offenders:
        raise AssertionError(f"Forbidden/non-wearable feature names: {offenders[:20]}")


assert_wearable_schema(COMPACT_DAILY_FEATURES)


def build_subject_dataset(
    split_root: str | Path,
    *,
    require_labels: bool,
    expected_split: str,
) -> SubjectSequenceDataset:
    """Load the audited wearable data and retain the fixed compact schema."""

    source = _build_audited_wearable_dataset(
        split_root,
        require_labels=require_labels,
        expected_split=expected_split,
    )
    assert_wearable_schema(source.feature_names)
    source_index = {name: index for index, name in enumerate(source.feature_names)}
    missing = [name for name in COMPACT_DAILY_FEATURES if name not in source_index]
    if missing:
        raise AssertionError(f"Predeclared daily channels are missing: {missing}")
    indices = np.asarray(
        [source_index[name] for name in COMPACT_DAILY_FEATURES], dtype=np.int64
    )
    compact_sequences: list[np.ndarray] = []
    for subject_id, sequence in zip(source.subject_ids, source.sequences):
        values = np.asarray(sequence, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(source.feature_names):
            raise ValueError(f"Invalid sequence shape for subject {subject_id!r}")
        if len(values) < VIEW_OBSERVATIONS:
            raise ValueError(
                f"Subject {subject_id!r} has {len(values)} observations; "
                f"{VIEW_OBSERVATIONS} required"
            )
        compact_sequences.append(values[:, indices].copy())
    audit = {
        **source.audit,
        "source_daily_features": len(source.feature_names),
        "compact_daily_features": len(COMPACT_DAILY_FEATURES),
        "compact_feature_names": list(COMPACT_DAILY_FEATURES),
        "fixed_model_view": (
            f"last {VIEW_OBSERVATIONS} aligned Activity/Sleep observations"
        ),
        "cognitive_source_opened": False,
        "mmse_feature_count": 0,
    }
    return SubjectSequenceDataset(
        subject_ids=np.asarray(source.subject_ids, dtype=str),
        sequences=compact_sequences,
        feature_names=tuple(COMPACT_DAILY_FEATURES),
        y=None if source.y is None else np.asarray(source.y, dtype=np.int64),
        audit=audit,
    )


def make_fixed_views(dataset: SubjectSequenceDataset) -> np.ndarray:
    """Return one equally sized recent-observation view per subject."""

    assert_wearable_schema(dataset.feature_names)
    rows: list[np.ndarray] = []
    for sequence in dataset.sequences:
        values = np.asarray(sequence, dtype=np.float32)
        if values.shape[0] < VIEW_OBSERVATIONS:
            raise ValueError("Every subject must have at least 35 aligned observations")
        rows.append(values[-VIEW_OBSERVATIONS:].copy())
    views = np.stack(rows).astype(np.float32)
    expected = (
        len(dataset.subject_ids),
        VIEW_OBSERVATIONS,
        len(COMPACT_DAILY_FEATURES),
    )
    if views.shape != expected:
        raise AssertionError(f"Fixed-view shape mismatch: {views.shape} != {expected}")
    return views


def assert_disjoint_subjects(
    training_subject_ids: Sequence[str],
    validation_subject_ids: Sequence[str],
) -> None:
    overlap = sorted(set(map(str, training_subject_ids)) & set(map(str, validation_subject_ids)))
    if overlap:
        raise AssertionError(
            f"Training/Validation subject leakage detected ({len(overlap)} subjects)"
        )


def load_validation_labels_checked(
    validation_root: str | Path,
    subject_ids: Sequence[str],
) -> np.ndarray:
    """Load historical Validation labels late and re-check the released contract."""

    labels = load_binary_labels(validation_root)
    expected_ids = [str(value) for value in subject_ids]
    if set(labels.index.astype(str)) != set(expected_ids):
        raise AssertionError("Validation label subjects differ from frozen predictions")
    y = labels.loc[expected_ids].to_numpy(np.int64)
    counts = {class_id: int(np.sum(y == class_id)) for class_id in (0, 1)}
    if len(y) != 33 or counts != {0: 26, 1: 7}:
        raise AssertionError(
            f"Historical Validation label contract changed: n={len(y)}, counts={counts}"
        )
    return y


__all__ = [
    "COMPACT_DAILY_FEATURES",
    "SubjectSequenceDataset",
    "VIEW_OBSERVATIONS",
    "assert_disjoint_subjects",
    "assert_wearable_schema",
    "build_subject_dataset",
    "load_validation_labels_checked",
    "make_fixed_views",
]

