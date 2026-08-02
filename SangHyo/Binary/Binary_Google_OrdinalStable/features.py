"""Training features plus the **3-class severity label** the binary folders discard.

Every other binary folder collapses CN/MCI/Dem into CN vs MCI+Dem before the
model ever sees it.  That throws away the fact that the two positive groups are
not interchangeable: Dem is extreme (MMSE mean 16.6) while MCI sits almost on
top of CN (25.8 vs 27.7).  A binary fit therefore spends much of its capacity on
9 easy, extreme points, even though ROC-AUC on this task is decided almost
entirely by how well CN is ranked against **MCI**.

Keeping the severity label available lets ``engine`` try strategies that use it
(ordinal / hard-boundary) against the plain binary baseline under one identical
nested evaluation, instead of assuming which one is better.

Validation stays label-free here, exactly as in the other folders: labels are
opened only by ``load_validation_labels_checked`` after predictions are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from SangHyo.Binary.Binary_Google_MaxAUC_Tuned import features as _F
from SangHyo.Binary.Binary_Google_MaxAUC_Tuned.features import (  # re-exported for train.py
    assert_disjoint_subjects,
    hash_subject_id,
    load_validation_labels_checked,
)

SEVERITY = {"CN": 0, "MCI": 1, "Dem": 2}
_LABEL_PATH = {"train": ("1.Training", "training_label.csv"),
               "val": ("2.Validation", "val_label.csv")}


@dataclass(frozen=True)
class TrainingData:
    subject_ids: np.ndarray
    y: np.ndarray            # 1 = MCI or Dem  (the task being scored)
    severity: np.ndarray     # 0 = CN, 1 = MCI, 2 = Dem
    X: np.ndarray
    feature_names: tuple[str, ...]
    item_max: dict

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def select(self, names: Sequence[str]) -> "TrainingData":
        index = {n: i for i, n in enumerate(self.feature_names)}
        cols = [index[n] for n in names]
        return TrainingData(self.subject_ids, self.y, self.severity, self.X[:, cols],
                            tuple(names), self.item_max)

    def mmse_only(self) -> "TrainingData":
        return self.select([n for n in self.feature_names if n.startswith("mmse_")])


def _severity_series(data_root: Path, split: str):
    directory, filename = _LABEL_PATH[split]
    frame = _F._read_csv(Path(data_root) / directory / "LabelingData" / "1.Gait" / filename)
    id_col = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns), None)
    if id_col is None:
        raise KeyError(f"Label file lacks SAMPLE_EMAIL/EMAIL: {filename}")
    frame = frame.copy()
    frame["_sid"] = frame[id_col].astype(str).str.strip()
    series = frame.drop_duplicates("_sid").set_index("_sid")["DIAG_NM"].astype(str).str.strip()
    unknown = sorted(set(series.unique()) - set(SEVERITY))
    if unknown:
        raise AssertionError(f"Unexpected DIAG_NM values: {unknown}")
    return series


def load_training(data_root: str | Path, *, include_wearable: bool = True,
                  drop_suspect: bool = False) -> TrainingData:
    data_root = Path(data_root)
    base = _F.load_split(data_root / "1.Training", require_labels=True, split="train",
                         include_wearable=include_wearable, drop_suspect=drop_suspect)
    severity_map = _severity_series(data_root, "train")
    subjects = [str(s) for s in base.subject_ids]
    missing = [s for s in subjects if s not in severity_map.index]
    if missing:
        raise AssertionError(f"{len(missing)} training subjects without DIAG_NM")
    severity = np.array([SEVERITY[severity_map[s]] for s in subjects], dtype=np.int64)

    if not np.array_equal((severity >= 1).astype(np.int64), base.y):
        raise AssertionError("3-class severity disagrees with the audited binary label")
    counts = {name: int((severity == code).sum()) for name, code in SEVERITY.items()}
    if counts != {"CN": 85, "MCI": 47, "Dem": 9}:
        raise AssertionError(f"Training severity contract changed: {counts}")

    return TrainingData(base.subject_ids, base.y, severity, base.X,
                        base.feature_names, base.item_max)


def load_validation(data_root: str | Path, item_max: dict, *, include_wearable: bool = True,
                    drop_suspect: bool = False):
    """Label-free validation features (labels stay closed until the freeze)."""

    return _F.load_split(Path(data_root) / "2.Validation", require_labels=False, split="val",
                         item_max=item_max, include_wearable=include_wearable,
                         drop_suspect=drop_suspect)


__all__ = ["SEVERITY", "TrainingData", "assert_disjoint_subjects", "hash_subject_id",
           "load_training", "load_validation", "load_validation_labels_checked"]
