"""Label loading and cohort contracts.

Verbatim port of the audited loader from ``Binary_Google_TabFM_Nested``
(same SEED, same construction => outer folds are paired across runs).
Labels are read ONLY from the LabelingData copies (Gait and Sleep, which must
agree); the MMSE SourceData file is never opened by this experiment at all
(wearable-only modality contract).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import DIAG_ORDER, LABEL_FILES, POSITIVE_DIAGS, SPLIT_CONTRACT, SPLIT_DIRS


def read_csv(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    last: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:  # pragma: no cover - encoding probe
            last = error
    raise RuntimeError(f"Unable to decode CSV: {path}") from last


def subject_hash(subject_id: str) -> str:
    """Short stable pseudonym; raw emails never appear in saved artifacts."""

    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:12]


def load_labels(data_root: str | Path, split: str) -> pd.DataFrame:
    """Return a frame indexed by subject id with ``diag`` and binary ``y``."""

    root = Path(data_root) / SPLIT_DIRS[split]
    copies = {}
    for name, relative in LABEL_FILES[split].items():
        frame = read_csv(root / relative)
        if not {"SAMPLE_EMAIL", "DIAG_NM"} <= set(frame.columns):
            raise KeyError(f"{relative} lacks SAMPLE_EMAIL/DIAG_NM")
        frame = frame.assign(_sid=frame["SAMPLE_EMAIL"].astype(str).str.strip())
        copies[name] = (
            frame[["_sid", "DIAG_NM"]].drop_duplicates("_sid")
            .sort_values("_sid").reset_index(drop=True)
        )
    gait, sleep = copies["gait"], copies["sleep"]
    if not gait.equals(sleep):
        raise AssertionError(f"Label copies disagree for split={split!r}")

    labels = gait.set_index("_sid")
    unknown = sorted(set(labels["DIAG_NM"]) - set(DIAG_ORDER))
    if unknown:
        raise AssertionError(f"Unknown diagnosis values: {unknown}")

    out = pd.DataFrame(index=labels.index)
    out.index.name = "subject_id"
    out["diag"] = labels["DIAG_NM"].astype(str)
    out["y"] = out["diag"].isin(POSITIVE_DIAGS).astype(int)

    contract = SPLIT_CONTRACT[split]
    counts = out["diag"].value_counts().to_dict()
    observed = {"n": len(out), **{d: int(counts.get(d, 0)) for d in DIAG_ORDER}}
    if observed != contract:
        raise AssertionError(
            f"Cohort contract violated for split={split!r}: expected {contract}, got {observed}"
        )
    return out


def assert_disjoint_splits(train_ids, val_ids) -> None:
    overlap = sorted(set(map(str, train_ids)) & set(map(str, val_ids)))
    if overlap:
        raise AssertionError(f"Subject overlap between train and validation: {overlap[:5]}")


__all__ = ["assert_disjoint_splits", "load_labels", "read_csv", "subject_hash"]
