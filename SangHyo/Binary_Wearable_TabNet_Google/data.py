"""Activity/Sleep-only data contract for the binary TabNet experiment.

The parsing and day-alignment implementation is reused from the previously
audited SequenceFusion experiment.  This module deliberately resolves only the
two wearable modalities and the two matching label copies.  Cognitive-source
directories are neither discovered nor opened.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from SangHyo.Binary_Wearable_SequenceFusion_Google import data as wearable_data


SubjectSequenceDataset = wearable_data.SubjectSequenceDataset
load_binary_labels = wearable_data.load_binary_labels


def _extended_daily_features(row: dict[str, Any]) -> dict[str, float]:
    """Add a few predeclared wearable fields omitted by the prior experiment."""

    output = wearable_data._daily_features(row)
    numeric = wearable_data._numeric
    output["activity__scalar__inactivity_alerts"] = numeric(
        row.get("activity_inactivity_alerts")
    )
    restless = numeric(row.get("sleep_restless"))
    duration = numeric(row.get("sleep_duration"))
    output["sleep__scalar__restless"] = restless
    output["sleep__duration__restless_ratio"] = (
        restless / duration
        if np.isfinite(restless) and np.isfinite(duration) and duration > 0
        else np.nan
    )
    output["sleep__scalar__midpoint_at_delta"] = numeric(
        row.get("sleep_midpoint_at_delta")
    )
    midpoint_seconds = numeric(row.get("sleep_midpoint_time"))
    if np.isfinite(midpoint_seconds):
        angle = 2.0 * np.pi * ((midpoint_seconds % 86_400.0) / 86_400.0)
        output["sleep__clock__midpoint_sin"] = float(np.sin(angle))
        output["sleep__clock__midpoint_cos"] = float(np.cos(angle))
    else:
        output["sleep__clock__midpoint_sin"] = np.nan
        output["sleep__clock__midpoint_cos"] = np.nan
    return output


def build_subject_dataset(
    split_root: str | Path,
    *,
    require_labels: bool,
    expected_split: str,
) -> SubjectSequenceDataset:
    """Build one chronological wearable sequence per subject.

    The exact train/validation cardinality and binary label contract are locked
    to the released data.  No collection-length or availability feature is
    emitted, and every retained subject must have at least 28 aligned days.
    """

    split_key = wearable_data._normalise_split(expected_split)
    root = wearable_data.resolve_split_root(split_root, split_key)
    inferred = wearable_data._infer_split_key_from_root(root)
    if inferred != split_key:
        raise AssertionError(f"Expected {split_key}, resolved {inferred} from {root}")
    aligned, source_audit = wearable_data._prepare_sources(root, split_key)
    if aligned.empty:
        raise ValueError("No aligned Activity/Sleep rows were found")

    labels = load_binary_labels(root) if require_labels else None
    aligned_subjects = set(aligned["_subject_id"].astype(str))
    if labels is not None and aligned_subjects != set(labels.index.astype(str)):
        raise ValueError("Wearable subjects and cross-checked label subjects differ")

    subject_ids: list[str] = []
    sequences: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    for subject_id, subject_frame in aligned.groupby("_subject_id", sort=True):
        rows: list[list[float]] = []
        for row in subject_frame.to_dict(orient="records"):
            features = _extended_daily_features(row)
            current_names = tuple(features)
            if feature_names is None:
                feature_names = current_names
                wearable_data._validate_feature_contract(feature_names)
            elif current_names != feature_names:
                raise AssertionError("Daily feature order changed between observations")
            rows.append(list(features.values()))
        matrix = np.asarray(rows, dtype=np.float32)
        matrix[np.isinf(matrix)] = np.nan
        if len(matrix) < wearable_data.DEFAULT_SEQUENCE_LENGTH:
            raise ValueError(
                f"Subject {subject_id!r} has fewer than 28 aligned observations"
            )
        subject_ids.append(str(subject_id))
        sequences.append(matrix)

    if feature_names is None:
        raise ValueError("No wearable features were constructed")
    subject_array = np.asarray(subject_ids, dtype=str)
    y = labels.loc[subject_ids].to_numpy(np.int64) if labels is not None else None
    expected_subjects = {"train": 141, "val": 33}[split_key]
    if len(subject_array) != expected_subjects:
        raise AssertionError(
            f"{split_key} subject contract: expected {expected_subjects}, got {len(subject_array)}"
        )
    if y is not None:
        observed = {class_id: int(np.sum(y == class_id)) for class_id in (0, 1)}
        expected_counts = {
            "train": {0: 85, 1: 56},
            "val": {0: 26, 1: 7},
        }[split_key]
        if observed != expected_counts:
            raise AssertionError(
                f"{split_key} class contract: expected {expected_counts}, got {observed}"
            )

    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=int)
    audit = {
        "split": split_key,
        "labels_loaded": bool(require_labels),
        "retained_subjects": int(len(subject_array)),
        "feature_count": int(len(feature_names)),
        "minimum_retained_observations": int(lengths.min()),
        "median_retained_observations": float(np.median(lengths)),
        "maximum_retained_observations": int(lengths.max()),
        "class_counts": (
            {"CN": int(np.sum(y == 0)), "MCI_DEM": int(np.sum(y == 1))}
            if y is not None
            else None
        ),
        "modalities_opened": ["Activity", "Sleep"],
        "cognitive_source_opened": False,
        "model_input_excludes": [
            "identifiers",
            "diagnosis",
            "cognitive tests",
            "absolute dates",
            "collection counts/coverage/masks",
            "non-wear",
        ],
        **source_audit,
    }
    return SubjectSequenceDataset(
        subject_ids=subject_array,
        sequences=sequences,
        feature_names=feature_names,
        y=y,
        audit=audit,
    )


__all__ = ["SubjectSequenceDataset", "build_subject_dataset", "load_binary_labels"]
