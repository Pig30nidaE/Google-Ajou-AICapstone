"""Audited data access for the ROC-AUC champion experiment.

The important property of this module is not convenience but *which bytes may
be read*.  The wearable track delegates to the already-audited
``Binary_Wearable_SequenceFusion_Google`` loader, which knows only Gait and
Sleep.  The MMSE reader uses an explicit ``usecols`` allow-list, so
``DIAG_NM`` and the other diagnosis/administrative fields present in the MMSE
CSV never enter memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google.data import (
    SubjectSequenceDataset,
    build_subject_dataset,
)

TRACKS = ("mmse", "wearable")
SPLITS = ("train", "val")
PERSON_KEY = "SAMPLE_EMAIL"

MMSE_DOMAINS: Mapping[str, tuple[str, ...]] = {
    "orient_time": ("Q01", "Q02", "Q03", "Q04", "Q05"),
    "orient_place": ("Q06", "Q07", "Q08", "Q09", "Q10"),
    "registration": ("Q11_1", "Q11_2", "Q11_3"),
    "attention": ("Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"),
    "recall": ("Q13_1", "Q13_2", "Q13_3"),
    "language": (
        "Q14_1",
        "Q14_2",
        "Q15",
        "Q16_1",
        "Q16_2",
        "Q16_3",
        "Q17",
        "Q18",
        "Q19",
    ),
}
MMSE_ITEMS = tuple(item for items in MMSE_DOMAINS.values() for item in items)
MMSE_ALLOWED_SOURCE_COLUMNS = (PERSON_KEY, "TOTAL", *MMSE_ITEMS)
MMSE_FORBIDDEN_SOURCE_COLUMNS = frozenset(
    {
        "DIAG_NM",
        "DIAG_SEQ",
        "DOCTOR_NM",
        "MMSE_NUM",
        "MMSE_KIND",
        "EMAIL",
    }
)

_LAYOUT = {
    "train": {
        "directory": "1.Training",
        "mmse": "train_mmse.csv",
        "label": "training_label.csv",
        "n": 141,
        "diagnoses": {"CN": 85, "MCI": 47, "Dem": 9},
    },
    "val": {
        "directory": "2.Validation",
        "mmse": "val_mmse.csv",
        "label": "val_label.csv",
        "n": 33,
        "diagnoses": {"CN": 26, "MCI": 4, "Dem": 3},
    },
}
_DIAG_NORMALIZATION = {
    "CN": "CN",
    "NORMAL": "CN",
    "MCI": "MCI",
    "DEM": "Dem",
    "DEMENTIA": "Dem",
    "AD": "Dem",
}


class LeakageContractError(RuntimeError):
    """Raised when a source or schema violates the declared modality contract."""


@dataclass
class AccessAudit:
    """Serializable record of every source read by this package."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        path: Path,
        *,
        purpose: str,
        selected_columns: Iterable[str] | None = None,
    ) -> None:
        self.events.append(
            {
                "path": str(path.resolve()),
                "purpose": str(purpose),
                "selected_columns": (
                    list(map(str, selected_columns))
                    if selected_columns is not None
                    else "delegated audited wearable loader"
                ),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        cognitive = [
            event
            for event in self.events
            if "3.CognitiveFunction" in event["path"]
        ]
        return {
            "events": list(self.events),
            "n_reads": len(self.events),
            "cognitive_reads": cognitive,
        }


def normalise_track(track: str) -> str:
    value = str(track).strip().lower()
    aliases = {"mmse_fusion": "mmse", "no_mmse": "wearable"}
    value = aliases.get(value, value)
    if value not in TRACKS:
        raise ValueError(f"track must be one of {TRACKS}; got {track!r}")
    return value


def normalise_split(split: str) -> str:
    value = str(split).strip().lower()
    aliases = {
        "training": "train",
        "1.training": "train",
        "validation": "val",
        "valid": "val",
        "2.validation": "val",
    }
    value = aliases.get(value, value)
    if value not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
    return value


def resolve_data_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    candidates = (root, root / "Data")
    for candidate in candidates:
        if (candidate / "1.Training").is_dir() and (
            candidate / "2.Validation"
        ).is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Data root with 1.Training and 2.Validation not found below {root}"
    )


def split_root(data_root: str | Path, split: str) -> Path:
    key = normalise_split(split)
    root = resolve_data_root(data_root)
    return (root / _LAYOUT[key]["directory"]).resolve()


def _read_csv_allowlist(
    path: Path,
    columns: Iterable[str],
    *,
    audit: AccessAudit,
    purpose: str,
) -> pd.DataFrame:
    """Read exactly ``columns`` and no other source field."""

    selected = tuple(map(str, columns))
    if not path.is_file():
        raise FileNotFoundError(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            frame = pd.read_csv(
                path,
                encoding=encoding,
                usecols=list(selected),
                low_memory=False,
            )
            audit.record(path, purpose=purpose, selected_columns=selected)
            if tuple(frame.columns) != selected:
                # pandas preserves source order rather than usecols order.
                frame = frame.loc[:, list(selected)]
            return frame
        except UnicodeDecodeError as error:
            last_error = error
    raise RuntimeError(f"Unable to decode CSV: {path}") from last_error


def load_wearable_sequence(
    data_root: str | Path,
    split: str,
    *,
    require_labels: bool,
    audit: AccessAudit,
) -> SubjectSequenceDataset:
    """Load the exact Gait/Sleep sequence representation.

    This function has no track argument on purpose.  A wearable dataset is
    always built first; the caller may explicitly add an MMSE block later.
    """

    key = normalise_split(split)
    root = split_root(data_root, key)
    dataset = build_subject_dataset(
        root,
        require_labels=bool(require_labels),
        expected_split=key,
    )
    audit.record(
        root,
        purpose=f"{key} Activity+Sleep via audited SequenceFusion loader",
    )
    return dataset


def load_mmse_allowed(
    data_root: str | Path,
    split: str,
    *,
    track: str,
    audit: AccessAudit,
) -> pd.DataFrame:
    """Return the MMSE score allow-list indexed by normalized subject ID."""

    resolved_track = normalise_track(track)
    if resolved_track != "mmse":
        raise LeakageContractError(
            "The wearable track is forbidden from opening CognitiveFunction/MMSE"
        )
    key = normalise_split(split)
    path = (
        split_root(data_root, key)
        / "SourceData"
        / "3.CognitiveFunction"
        / _LAYOUT[key]["mmse"]
    )
    frame = _read_csv_allowlist(
        path,
        MMSE_ALLOWED_SOURCE_COLUMNS,
        audit=audit,
        purpose=f"{key} MMSE scores; diagnosis/admin columns excluded by usecols",
    )
    forbidden_loaded = set(frame.columns) & MMSE_FORBIDDEN_SOURCE_COLUMNS
    if forbidden_loaded:
        raise LeakageContractError(
            f"Forbidden MMSE fields were read: {sorted(forbidden_loaded)}"
        )
    if frame[PERSON_KEY].isna().any():
        raise LeakageContractError("MMSE contains missing subject identifiers")
    frame = frame.copy()
    frame[PERSON_KEY] = frame[PERSON_KEY].astype(str).str.strip()
    if frame[PERSON_KEY].duplicated().any():
        raise LeakageContractError("MMSE contains duplicate subject identifiers")
    frame = frame.set_index(PERSON_KEY).sort_index()
    if len(frame) != int(_LAYOUT[key]["n"]):
        raise LeakageContractError(
            f"{key} MMSE subject contract changed: {len(frame)}"
        )
    return frame


def _normalise_label_copy(frame: pd.DataFrame, source: str) -> pd.Series:
    work = frame.copy()
    work[PERSON_KEY] = work[PERSON_KEY].astype(str).str.strip()
    raw = work["DIAG_NM"].astype(str).str.strip().str.upper()
    diagnosis = raw.map(_DIAG_NORMALIZATION)
    if diagnosis.isna().any():
        unknown = sorted(raw[diagnosis.isna()].unique())
        raise LeakageContractError(f"{source}: unknown diagnoses {unknown}")
    result = pd.Series(
        diagnosis.to_numpy(),
        index=work[PERSON_KEY].to_numpy(),
        name="diagnosis",
    )
    if result.index.has_duplicates:
        grouped = result.groupby(level=0).nunique()
        if (grouped > 1).any():
            raise LeakageContractError(f"{source}: conflicting duplicate labels")
        result = result[~result.index.duplicated(keep="first")]
    return result.sort_index()


def load_diagnoses(
    data_root: str | Path,
    split: str,
    *,
    audit: AccessAudit,
) -> pd.Series:
    """Load and cross-check only the Gait and Sleep diagnosis copies.

    Validation callers must invoke this only *after* all label-free prediction
    files have been written and hashed.
    """

    key = normalise_split(split)
    root = split_root(data_root, key)
    filename = _LAYOUT[key]["label"]
    frames: dict[str, pd.Series] = {}
    for modality, directory in (("Gait", "1.Gait"), ("Sleep", "2.Sleep")):
        path = root / "LabelingData" / directory / filename
        frame = _read_csv_allowlist(
            path,
            (PERSON_KEY, "DIAG_NM"),
            audit=audit,
            purpose=f"{key} {modality} diagnosis copy",
        )
        frames[modality] = _normalise_label_copy(frame, modality)
    if not frames["Gait"].equals(frames["Sleep"]):
        raise LeakageContractError("Gait and Sleep diagnosis copies disagree")
    observed = frames["Gait"].value_counts().to_dict()
    expected = _LAYOUT[key]["diagnoses"]
    if observed != expected:
        raise LeakageContractError(
            f"{key} diagnosis contract changed: {observed} != {expected}"
        )
    return frames["Gait"].copy()


def binary_target(diagnosis: pd.Series) -> np.ndarray:
    values = diagnosis.astype(str)
    unknown = sorted(set(values) - {"CN", "MCI", "Dem"})
    if unknown:
        raise LeakageContractError(f"Unexpected diagnoses: {unknown}")
    return values.isin({"MCI", "Dem"}).to_numpy(dtype=np.int64)


def assert_subject_alignment(
    subject_ids: Iterable[str],
    indexed: pd.DataFrame | pd.Series,
    *,
    role: str,
) -> None:
    wanted = list(map(str, subject_ids))
    observed = list(map(str, indexed.index))
    if set(wanted) != set(observed):
        missing = sorted(set(wanted) - set(observed))
        extra = sorted(set(observed) - set(wanted))
        raise LeakageContractError(
            f"{role} subject mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )


def assert_disjoint_subjects(train_ids: Iterable[str], validation_ids: Iterable[str]) -> None:
    overlap = sorted(set(map(str, train_ids)) & set(map(str, validation_ids)))
    if overlap:
        raise LeakageContractError(
            f"Train/Validation subject leakage: {len(overlap)} subjects"
        )


__all__ = [
    "AccessAudit",
    "LeakageContractError",
    "MMSE_ALLOWED_SOURCE_COLUMNS",
    "MMSE_DOMAINS",
    "MMSE_FORBIDDEN_SOURCE_COLUMNS",
    "MMSE_ITEMS",
    "TRACKS",
    "assert_disjoint_subjects",
    "assert_subject_alignment",
    "binary_target",
    "load_diagnoses",
    "load_mmse_allowed",
    "load_wearable_sequence",
    "normalise_split",
    "normalise_track",
    "resolve_data_root",
]
