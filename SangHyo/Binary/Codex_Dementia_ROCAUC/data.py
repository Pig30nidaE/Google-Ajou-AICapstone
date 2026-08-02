"""Audited loading for CN+MCI versus dementia subject cohorts.

The raw Activity/Sleep parser is reused from the repository's audited
``Binary_Wearable_SequenceFusion_Google`` implementation.  This package changes
the target definition and adds stricter per-track access assertions:

* ``wearable`` and ``wearable_protocol`` never open CognitiveFunction files.
* ``wearable_mmse`` reads only an explicit MMSE score allow-list.
* diagnosis comes from the two cross-checked Gait/Sleep label copies.

Each call loads exactly one track so a wearable-only run cannot accidentally
retain MMSE values in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import json
from typing import Any, Mapping, Literal

import numpy as np
import pandas as pd

from .config import DataConfig, Track
from .leakage import (
    LeakageError,
    assert_group_label_consistency,
    assert_no_forbidden_features,
    assert_unique_subjects,
)


@dataclass(frozen=True)
class TrackCohort:
    """One row/sequence per unique subject for a single modality contract."""

    track: Track
    subject_ids: np.ndarray
    diagnosis: np.ndarray | None
    y: np.ndarray | None
    split_origin: np.ndarray
    sequences: tuple[np.ndarray, ...]
    daily_feature_names: tuple[str, ...]
    mmse: pd.DataFrame | None
    access_audit: Mapping[str, Any]
    input_fingerprints: Mapping[str, str]

    def __post_init__(self) -> None:
        n = len(self.subject_ids)
        assert_unique_subjects(self.subject_ids)
        if (self.diagnosis is None) != (self.y is None):
            raise LeakageError("Diagnosis and target must both be present or both be hidden")
        if self.diagnosis is not None and self.y is not None:
            if self.diagnosis.shape != (n,) or self.y.shape != (n,):
                raise LeakageError(
                    "Diagnosis/target is not aligned one-to-one with subjects"
                )
            if set(np.unique(self.y)) != {0, 1}:
                raise LeakageError("A labeled cohort must contain both target classes")
            assert_group_label_consistency(self.subject_ids, self.y)
        if self.split_origin.shape != (n,) or len(self.sequences) != n:
            raise LeakageError("Source split/sequences are not aligned with subjects")
        width = len(self.daily_feature_names)
        if width == 0:
            raise LeakageError("Daily wearable schema is empty")
        assert_no_forbidden_features(self.daily_feature_names)
        for position, sequence in enumerate(self.sequences):
            values = np.asarray(sequence)
            if values.ndim != 2 or values.shape[1] != width or values.shape[0] == 0:
                raise LeakageError(
                    f"Subject position {position} has invalid sequence shape {values.shape}"
                )
            if np.isinf(values).any():
                raise LeakageError(f"Subject position {position} contains infinity")
        if self.track == "wearable_mmse":
            if self.mmse is None or not self.mmse.index.equals(
                pd.Index(self.subject_ids, name=self.mmse.index.name)
            ):
                raise LeakageError("MMSE rows are missing or misaligned")
            assert_no_forbidden_features(tuple(map(str, self.mmse.columns)))
        elif self.mmse is not None:
            raise LeakageError(f"{self.track} is forbidden from retaining MMSE")

    @property
    def groups(self) -> np.ndarray:
        return self.subject_ids.copy()

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def n_positive(self) -> int:
        if self.y is None:
            raise LeakageError("Labels are intentionally hidden for this cohort")
        return int(self.y.sum())


_KNOWN_FILES = {
    "train": {
        "directory": "1.Training",
        "activity": ("SourceData", "1.Gait", "train_activity.csv"),
        "sleep": ("SourceData", "2.Sleep", "train_sleep.csv"),
    },
    "val": {
        "directory": "2.Validation",
        "activity": ("SourceData", "1.Gait", "val_activity.csv"),
        "sleep": ("SourceData", "2.Sleep", "val_sleep.csv"),
    },
}


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256_allowed_table(frame: pd.DataFrame | pd.Series) -> str:
    """Hash only the columns already admitted by the modality allow-list."""

    table = frame.to_frame() if isinstance(frame, pd.Series) else frame
    canonical = table.to_csv(
        index=True,
        na_rep="<NA>",
        float_format="%.17g",
        lineterminator="\n",
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fingerprint_inputs(
    data_root: Path,
    *,
    splits: tuple[str, ...],
) -> dict[str, str]:
    output: dict[str, str] = {}
    allowed_keys = {"activity", "sleep"}
    for split in splits:
        layout = _KNOWN_FILES[split]
        split_root = data_root / str(layout["directory"])
        for key in sorted(allowed_keys):
            path = split_root.joinpath(*layout[key])
            if not path.is_file():
                raise FileNotFoundError(path)
            output[f"{split}:{key}:{path.name}"] = _sha256_file(path)
    return output


def _normalize_diagnosis(values: pd.Series) -> pd.Series:
    aliases = {
        "CN": "CN",
        "NORMAL": "CN",
        "MCI": "MCI",
        "DEM": "Dem",
        "DEMENTIA": "Dem",
        "AD": "Dem",
    }
    normalized = values.astype(str).str.strip().str.upper().map(aliases)
    if normalized.isna().any():
        bad = sorted(set(values[normalized.isna()].astype(str)))
        raise LeakageError(f"Unknown diagnoses: {bad}")
    return normalized


def _load_one_split(
    data_config: DataConfig,
    *,
    split: str,
    track: Track,
    require_labels: bool,
) -> tuple[
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[str, ...],
    pd.Series | None,
    pd.DataFrame | None,
    dict[str, Any],
]:
    """Load one source split through audited repository readers."""

    # Heavy scientific dependencies are already required by data construction,
    # but repository imports are delayed so ``run.py --help`` stays lightweight.
    from SangHyo.Binary.Binary_Google_ROCAUC_Champion.data import (
        AccessAudit,
        load_diagnoses,
        load_mmse_allowed,
        load_wearable_sequence,
    )

    audit = AccessAudit()
    wearable = load_wearable_sequence(
        data_config.resolved_root(),
        split,
        require_labels=False,
        audit=audit,
    )
    if wearable.y is not None:
        raise LeakageError(
            "Audited wearable loader unexpectedly returned its legacy target"
        )
    ids = np.asarray(wearable.subject_ids, dtype=str)
    diagnoses: pd.Series | None = None
    if require_labels:
        diagnoses = _normalize_diagnosis(
            load_diagnoses(data_config.resolved_root(), split, audit=audit)
        ).reindex(ids)
        if diagnoses.isna().any():
            raise LeakageError(f"{split}: wearable subject(s) lack diagnosis")

    mmse: pd.DataFrame | None = None
    if track == "wearable_mmse":
        mmse = load_mmse_allowed(
            data_config.resolved_root(),
            split,
            track="mmse",
            audit=audit,
        ).reindex(ids)
        if mmse.isna().all(axis=1).any():
            raise LeakageError(f"{split}: subject with entirely missing MMSE scores")

    audit_payload = audit.to_dict()
    cognitive_reads = audit_payload.get("cognitive_reads", [])
    if track != "wearable_mmse" and cognitive_reads:
        raise LeakageError(
            f"{track} opened CognitiveFunction sources: {cognitive_reads}"
        )
    if track == "wearable_mmse" and not cognitive_reads:
        raise LeakageError("wearable_mmse did not record its explicit MMSE read")

    return (
        ids,
        tuple(np.asarray(sequence, dtype=np.float64) for sequence in wearable.sequences),
        tuple(map(str, wearable.feature_names)),
        diagnoses,
        mmse,
        audit_payload,
    )


def _target_from_diagnosis(
    diagnosis: np.ndarray, data_config: DataConfig
) -> np.ndarray:
    positive = set(data_config.target_positive_diagnoses)
    negative = set(data_config.target_negative_diagnoses)
    observed = set(map(str, diagnosis))
    if not observed <= positive | negative:
        raise LeakageError(
            f"Target mapping does not cover diagnoses: {sorted(observed - positive - negative)}"
        )
    return np.asarray([int(value in positive) for value in diagnosis], dtype=np.int64)


def _load_track_partition(
    data_config: DataConfig,
    track: Track,
    *,
    split: Literal["train", "val"],
    require_labels: bool,
) -> TrackCohort:
    if track not in data_config.tracks:
        raise ValueError(f"Track {track!r} is not enabled in DataConfig.tracks")
    (
        subject_ids,
        sequences,
        feature_names,
        diagnosis_series,
        mmse,
        audit,
    ) = _load_one_split(
        data_config,
        split=split,
        track=track,
        require_labels=require_labels,
    )
    diagnosis = (
        diagnosis_series.to_numpy(dtype=str) if diagnosis_series is not None else None
    )
    y = (
        _target_from_diagnosis(diagnosis, data_config)
        if diagnosis is not None
        else None
    )
    split_origin = np.asarray([split] * len(subject_ids), dtype=str)

    if data_config.strict_known_cohort and require_labels:
        assert diagnosis is not None
        counts = {
            diagnosis_name: int(np.sum(diagnosis == diagnosis_name))
            for diagnosis_name in sorted(set(diagnosis))
        }
        expected_n = (
            data_config.expected_train_subjects
            if split == "train"
            else data_config.expected_validation_subjects
        )
        expected_counts = (
            data_config.expected_train_diagnosis_counts
            if split == "train"
            else data_config.expected_validation_diagnosis_counts
        )
        if len(subject_ids) != expected_n:
            raise LeakageError(
                f"{split}: expected {expected_n} subjects, got {len(subject_ids)}"
            )
        if counts != expected_counts:
            raise LeakageError(
                f"{split} diagnosis-count contract changed: {counts} != {expected_counts}"
            )
    elif data_config.strict_known_cohort:
        expected_n = (
            data_config.expected_train_subjects
            if split == "train"
            else data_config.expected_validation_subjects
        )
        if len(subject_ids) != expected_n:
            raise LeakageError(
                f"{split}: expected {expected_n} unlabeled subjects, got {len(subject_ids)}"
            )

    fingerprints = _fingerprint_inputs(
        data_config.resolved_root(),
        splits=(split,),
    )
    if diagnosis_series is not None:
        fingerprints[f"{split}:diagnosis_allowlist"] = _sha256_allowed_table(
            diagnosis_series
        )
    if mmse is not None:
        fingerprints[f"{split}:mmse_score_allowlist"] = _sha256_allowed_table(
            mmse
        )
    return TrackCohort(
        track=track,
        subject_ids=subject_ids,
        diagnosis=diagnosis,
        y=y,
        split_origin=split_origin,
        sequences=sequences,
        daily_feature_names=feature_names,
        mmse=mmse,
        access_audit={split: audit},
        input_fingerprints=fingerprints,
    )


def load_development_cohort(
    data_config: DataConfig, track: Track
) -> TrackCohort:
    """Load labeled official Training subjects for the primary nested OOF."""

    return _load_track_partition(
        data_config,
        track,
        split=data_config.primary_split,
        require_labels=True,
    )


def load_historical_validation_features(
    data_config: DataConfig, track: Track
) -> TrackCohort:
    """Load Validation predictors without opening either label copy."""

    return _load_track_partition(
        data_config,
        track,
        split=data_config.historical_validation_split,
        require_labels=False,
    )


def load_historical_validation_labels_after_freeze(
    data_config: DataConfig,
    *,
    track: Track,
    subject_ids: np.ndarray,
    freeze_manifest: str | Path,
) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Open historical labels only after label-free predictions are frozen.

    The manifest must contain the exact ordered subject hashes and a SHA-256 of
    the prediction file.  This prevents an accidental evaluate-adjust-evaluate
    loop inside one run.  It cannot undo historical project-level reuse, so the
    resulting metric remains descriptive rather than an independent test.
    """

    from .leakage import hash_subject_id
    from SangHyo.Binary.Binary_Google_ROCAUC_Champion.data import AccessAudit, load_diagnoses

    manifest_path = Path(freeze_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise LeakageError(f"Freeze manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hashes = [hash_subject_id(value) for value in subject_ids]
    if manifest.get("status") != "frozen":
        raise LeakageError("Historical prediction manifest is not frozen")
    if manifest.get("track") != track:
        raise LeakageError("Historical prediction manifest track mismatch")
    if int(manifest.get("n_subjects", -1)) != len(subject_ids):
        raise LeakageError("Historical prediction manifest subject count mismatch")
    if manifest.get("ordered_subject_hashes") != expected_hashes:
        raise LeakageError("Historical prediction subject order/hash mismatch")
    prediction_sha = str(manifest.get("prediction_file_sha256", ""))
    if len(prediction_sha) != 64:
        raise LeakageError("Historical prediction manifest lacks a valid SHA-256")
    deployment_sha = str(manifest.get("deployment_sha256", ""))
    if len(deployment_sha) != 64:
        raise LeakageError("Historical freeze lacks the deployment SHA-256")
    prediction_path = Path(str(manifest.get("prediction_file", ""))).expanduser()
    if not prediction_path.is_file():
        raise LeakageError("Frozen historical prediction file is missing")
    if _sha256_file(prediction_path) != prediction_sha:
        raise LeakageError("Frozen historical prediction SHA-256 no longer matches")

    audit = AccessAudit()
    diagnosis_series = _normalize_diagnosis(
        load_diagnoses(
            data_config.resolved_root(),
            data_config.historical_validation_split,
            audit=audit,
        )
    ).reindex(subject_ids)
    if diagnosis_series.isna().any():
        raise LeakageError("Historical validation label subjects are misaligned")
    diagnosis = diagnosis_series.to_numpy(dtype=str)
    y = _target_from_diagnosis(diagnosis, data_config)
    if data_config.strict_known_cohort:
        counts = {
            name: int(np.sum(diagnosis == name)) for name in sorted(set(diagnosis))
        }
        if counts != data_config.expected_validation_diagnosis_counts:
            raise LeakageError(f"Historical validation counts changed: {counts}")
    return diagnosis, y, audit.to_dict()


__all__ = [
    "TrackCohort",
    "load_development_cohort",
    "load_historical_validation_features",
    "load_historical_validation_labels_after_freeze",
]
