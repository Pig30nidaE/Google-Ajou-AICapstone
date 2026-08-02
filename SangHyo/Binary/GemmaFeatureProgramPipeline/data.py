"""Training-only cohort assembly and leakage contracts.

This module reuses the deterministic, subject-local daily/intraday parsers from
``Binary_Google_DemRankAUC_select1`` but deliberately does *not* reuse that
experiment's cohort or labels:

* only ``1.Training`` is opened;
* the task remains CN=0 versus MCI or Dem=1;
* historical Validation labels are never opened;
* all MMSE-derived fields are missing when the test was not validly
  administered, fixing the all-NaN-domain-sum bug found in the reference code;
* wearable columns are a predeclared catalogue, not a whole-cohort selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from SangHyo.Binary.Binary_Google_DemRankAUC_select1.config import (
    MMSE_DOMAINS,
    MMSE_ITEMS,
)
from SangHyo.Binary.Binary_Google_DemRankAUC_select1.features import (
    build_split_features,
    read_csv,
)

from .catalog import primitive_names

TRAINING_CONTRACT = {"CN": 85, "MCI": 47, "Dem": 9}
DIAGNOSES = ("CN", "MCI", "Dem")
LABEL_PATHS = (
    "1.Training/LabelingData/1.Gait/training_label.csv",
    "1.Training/LabelingData/2.Sleep/training_label.csv",
)
FORBIDDEN_TOKENS = (
    "diag",
    "label",
    "doctor",
    "email",
    "sample",
    "target",
    "class",
)

MMSE_FEATURE_NAMES: tuple[str, ...] = (
    "mmse_TOTAL",
    *(f"mmse_{domain}" for domain in MMSE_DOMAINS),
    *(f"mmse_{item}" for item in MMSE_ITEMS),
)


@dataclass(frozen=True)
class CohortData:
    """One row per Training subject; positive means MCI or Dem."""

    subject_ids: np.ndarray
    diagnosis: np.ndarray
    y: np.ndarray
    mmse: pd.DataFrame
    wearable: pd.DataFrame
    fingerprint: str

    @property
    def n_subjects(self) -> int:
        return int(len(self.subject_ids))

    def audit(self) -> dict[str, object]:
        return {
            "cohort": "1.Training only",
            "task": "CN (0) vs MCI+Dem (1)",
            "n_subjects": self.n_subjects,
            "diagnosis_counts": {
                diagnosis: int(np.sum(self.diagnosis == diagnosis))
                for diagnosis in DIAGNOSES
            },
            "n_positive": int(self.y.sum()),
            "n_mmse_features": int(self.mmse.shape[1]),
            "n_wearable_primitives": int(self.wearable.shape[1]),
            "mmse_missing_rows": int(self.mmse.isna().all(axis=1).sum()),
            "wearable_finite_fraction": float(
                np.isfinite(self.wearable.to_numpy(dtype=np.float64)).mean()
            ),
            "fingerprint": self.fingerprint,
            "historical_validation_opened": False,
        }


def _assert_safe_feature_names(names: Sequence[str], *, block: str) -> None:
    unsafe = [
        str(name)
        for name in names
        if any(token in str(name).lower() for token in FORBIDDEN_TOKENS)
    ]
    if unsafe:
        raise AssertionError(f"{block} block contains forbidden names: {unsafe[:8]}")


def _id_column(frame: pd.DataFrame, path: Path) -> str:
    for candidate in ("SAMPLE_EMAIL", "EMAIL"):
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Label file lacks SAMPLE_EMAIL/EMAIL: {path}")


def _read_label_copy(data_root: Path, relative_path: str) -> pd.Series:
    path = data_root / relative_path
    frame = read_csv(path)
    if "DIAG_NM" not in frame.columns:
        raise KeyError(f"Label file lacks DIAG_NM: {path}")
    id_column = _id_column(frame, path)
    clean = frame[[id_column, "DIAG_NM"]].copy()
    clean["_sid"] = clean[id_column].astype(str).str.strip()
    clean["_diagnosis"] = clean["DIAG_NM"].astype(str).str.strip()
    if clean["_sid"].duplicated().any():
        duplicate_rows = clean.loc[clean["_sid"].duplicated(keep=False)]
        disagree = duplicate_rows.groupby("_sid")["_diagnosis"].nunique()
        if (disagree > 1).any():
            raise AssertionError(f"Conflicting duplicate labels in {path}")
    result = clean.drop_duplicates("_sid").set_index("_sid")["_diagnosis"].sort_index()
    unknown = sorted(set(result.unique()) - set(DIAGNOSES))
    if unknown:
        raise AssertionError(f"Unexpected diagnosis values in {path}: {unknown}")
    return result


def load_training_labels(data_root: str | Path) -> pd.Series:
    """Read and cross-check the two Training label copies only."""

    root = Path(data_root)
    first = _read_label_copy(root, LABEL_PATHS[0])
    second = _read_label_copy(root, LABEL_PATHS[1])
    if not first.index.equals(second.index):
        raise AssertionError("Training Gait/Sleep label subject sets differ")
    if not first.equals(second):
        changed = first.index[first.to_numpy() != second.to_numpy()].tolist()
        raise AssertionError(
            f"Training label copies disagree for {len(changed)} subject(s)"
        )
    observed = {diagnosis: int((first == diagnosis).sum()) for diagnosis in DIAGNOSES}
    if observed != TRAINING_CONTRACT:
        raise AssertionError(f"Training cohort contract changed: {observed}")
    return first


def _mask_invalid_mmse_block(frame: pd.DataFrame) -> pd.DataFrame:
    """Make an unadministered MMSE missing in every derived MMSE column."""

    out = frame.copy()
    mmse_columns = [name for name in out.columns if str(name).startswith("mmse_")]
    if "mmse_TOTAL" not in out.columns:
        raise KeyError("Deterministic feature builder did not produce mmse_TOTAL")
    invalid = ~np.isfinite(pd.to_numeric(out["mmse_TOTAL"], errors="coerce"))
    if invalid.any():
        out.loc[invalid, mmse_columns] = np.nan
    return out


def _add_circular_onset_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode clock-hour onsets without a false discontinuity at midnight."""

    out = frame.copy()
    for window in ("M10", "L5"):
        source = f"wi_met_{window}_onset_h"
        if source not in out.columns:
            raise KeyError(f"Deterministic feature builder did not produce {source}")
        hours = pd.to_numeric(out[source], errors="coerce").astype(float)
        radians = 2.0 * np.pi * hours / 24.0
        out[f"wi_met_{window}_onset_sin"] = np.sin(radians)
        out[f"wi_met_{window}_onset_cos"] = np.cos(radians)
    return out


def _require_columns(frame: pd.DataFrame, names: Sequence[str], *, block: str) -> pd.DataFrame:
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise KeyError(f"{block} primitive schema changed; missing {missing[:12]}")
    selected = frame.loc[:, list(names)].astype(np.float64)
    _assert_safe_feature_names(selected.columns, block=block)
    return selected


def _fingerprint(
    subject_ids: np.ndarray,
    diagnosis: np.ndarray,
    mmse: pd.DataFrame,
    wearable: pd.DataFrame,
) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(map(str, subject_ids)).encode("utf-8"))
    digest.update("\n".join(map(str, diagnosis)).encode("utf-8"))
    for block in (mmse, wearable):
        digest.update("\n".join(map(str, block.columns)).encode("utf-8"))
        digest.update(np.ascontiguousarray(block.to_numpy(dtype=np.float64)).tobytes())
    return digest.hexdigest()


def load_training_cohort(data_root: str | Path) -> CohortData:
    """Build the fixed 141-subject development cohort without opening Validation."""

    root = Path(data_root)
    full = build_split_features(root, "train", mmse_zero_as_missing=True)
    full = _mask_invalid_mmse_block(full)
    full = _add_circular_onset_features(full)
    labels = load_training_labels(root)

    subjects = np.asarray(full.index.astype(str), dtype=str)
    if len(set(subjects.tolist())) != len(subjects):
        raise AssertionError("Duplicate subject id in deterministic feature table")
    missing_labels = sorted(set(subjects) - set(labels.index.astype(str)))
    if missing_labels:
        raise AssertionError(
            f"{len(missing_labels)} feature subject(s) have no Training label"
        )

    diagnosis = labels.reindex(subjects).to_numpy(dtype=str)
    y = np.isin(diagnosis, ("MCI", "Dem")).astype(np.int64)
    mmse = _require_columns(full, MMSE_FEATURE_NAMES, block="MMSE")
    wearable_names = primitive_names()
    wearable = _require_columns(full, wearable_names, block="wearable")
    mmse.index = pd.Index(subjects, name="subject_id")
    wearable.index = pd.Index(subjects, name="subject_id")

    cohort = CohortData(
        subject_ids=subjects,
        diagnosis=diagnosis,
        y=y,
        mmse=mmse,
        wearable=wearable,
        fingerprint=_fingerprint(subjects, diagnosis, mmse, wearable),
    )
    observed = {
        diagnosis_name: int(np.sum(cohort.diagnosis == diagnosis_name))
        for diagnosis_name in DIAGNOSES
    }
    if observed != TRAINING_CONTRACT or cohort.n_subjects != 141 or int(y.sum()) != 56:
        raise AssertionError(
            f"CN vs MCI+Dem contract changed: counts={observed}, n={cohort.n_subjects}"
        )
    return cohort


def hash_subject_id(subject_id: str, *, salt: str = "GemmaFeatureProgramPipeline") -> str:
    payload = f"{salt}\0{subject_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


__all__ = [
    "CohortData",
    "DIAGNOSES",
    "MMSE_FEATURE_NAMES",
    "TRAINING_CONTRACT",
    "hash_subject_id",
    "load_training_cohort",
    "load_training_labels",
]
