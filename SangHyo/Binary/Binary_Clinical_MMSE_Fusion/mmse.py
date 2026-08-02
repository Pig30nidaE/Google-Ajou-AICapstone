"""MMSE cognitive-test feature loader with a fail-closed diagnosis guard.

This experiment is the *with-MMSE* counterpart, so it intentionally opens
``SourceData/3.CognitiveFunction``.  It uses only the 32 cognitive item/total
scores.  The diagnosis columns (``DIAG_NM`` is the label itself, ``DIAG_SEQ`` is
a diagnosis-ordered code) and administrative metadata (``MMSE_NUM``,
``MMSE_KIND``, ``DOCTOR_NM``) are hard-excluded and asserted against, so the
label can never leak in through a feature.  Labels themselves still come only
from the Gait/Sleep copies, never from this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# The 32 genuine MMSE cognitive scores (items + section/grand totals).
MMSE_FEATURE_COLUMNS = (
    "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07", "Q08", "Q09", "Q10",
    "Q11_1", "Q11_2", "Q11_3",
    "Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5", "Q12_TOTAL",
    "Q13_1", "Q13_2", "Q13_3",
    "Q14_1", "Q14_2",
    "Q15",
    "Q16_1", "Q16_2", "Q16_3",
    "Q17", "Q18", "Q19",
    "TOTAL",
)

# Never allowed as features: the label, diagnosis-order code, and metadata.
FORBIDDEN_MMSE_COLUMNS = frozenset(
    {"DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND",
     "SAMPLE_EMAIL", "EMAIL"}
)

_SPLIT_FILE = {"train": "train_mmse.csv", "val": "val_mmse.csv"}


def _infer_split(split_root: Path) -> str:
    name = split_root.name.lower()
    if name.startswith("1.") or "training" in name:
        return "train"
    if name.startswith("2.") or "validation" in name:
        return "val"
    raise ValueError(f"Cannot infer train/val from MMSE split root: {split_root}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode MMSE CSV: {path}")


def assert_no_forbidden(names: Sequence[str]) -> None:
    offenders = [name for name in names if name in FORBIDDEN_MMSE_COLUMNS]
    if offenders:
        raise AssertionError(f"Diagnosis/metadata leaked into MMSE features: {offenders}")
    if any("diag" in str(name).lower() for name in names):
        raise AssertionError("A diagnosis-like column reached the MMSE feature list")


assert_no_forbidden(MMSE_FEATURE_COLUMNS)


def load_mmse_features(
    split_root: str | Path,
    subject_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the ``(n, 32)`` MMSE score matrix aligned to ``subject_ids``."""

    root = Path(split_root).expanduser().resolve()
    split = _infer_split(root)
    path = root / "SourceData" / "3.CognitiveFunction" / _SPLIT_FILE[split]
    frame = _read_csv(path)

    id_column = next(
        (c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns), None
    )
    if id_column is None:
        raise KeyError("MMSE file is missing a SAMPLE_EMAIL/EMAIL column")
    frame = frame.copy()
    frame["_subject_id"] = frame[id_column].astype(str).str.strip()

    missing_columns = [c for c in MMSE_FEATURE_COLUMNS if c not in frame.columns]
    if missing_columns:
        raise KeyError(f"MMSE file is missing expected score columns: {missing_columns}")
    assert_no_forbidden(MMSE_FEATURE_COLUMNS)

    indexed = frame.drop_duplicates("_subject_id").set_index("_subject_id")
    wanted = [str(s) for s in subject_ids]
    missing_subjects = [s for s in wanted if s not in indexed.index]
    if missing_subjects:
        raise AssertionError(
            f"{len(missing_subjects)} wearable subjects have no MMSE row "
            f"(e.g. {missing_subjects[:3]})"
        )
    matrix = indexed.loc[wanted, list(MMSE_FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    names = tuple(f"mmse__{c}" for c in MMSE_FEATURE_COLUMNS)
    return matrix, names


__all__ = [
    "FORBIDDEN_MMSE_COLUMNS",
    "MMSE_FEATURE_COLUMNS",
    "assert_no_forbidden",
    "load_mmse_features",
]
