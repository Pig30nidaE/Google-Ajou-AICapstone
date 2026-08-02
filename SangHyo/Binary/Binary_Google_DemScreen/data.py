"""Pooled 174-subject cohort with 3-class diagnosis, for dementia screening.

Task redefinition (borrowed from Hyunsoo's analysis)
----------------------------------------------------
Every other binary folder here asks **CN vs MCI+Dem**, whose hard boundary is
CN vs MCI -- and that boundary is nearly random (MMSE TOTAL alone: AUC 0.68,
group means CN 27.7 / MCI 25.8).  This folder asks the *other* question:

    positive = Dem (dementia)         n = 12
    negative = CN + MCI               n = 162

which is a genuinely different and much more separable problem (MMSE TOTAL
alone: AUC 0.947, group means CN 27.7 / MCI 25.8 / **Dem 16.6**).  The higher
numbers this folder reports are a property of the question, not of the model --
see README_KO.md, which states this explicitly so the two folders are never
compared as if they were the same task.

Why train and validation are pooled
-----------------------------------
There are only 12 dementia subjects in the entire dataset (9 train + 3 val).
Holding 3 of them out would make both the fit and the estimate meaningless, so
all 174 subjects go into repeated cross-validation and **there is no held-out
test set**.  That is a real limitation, stated here and in every report this
folder writes -- not something to discover later.

Data-quality exclusions
-----------------------
Hyunsoo's script drops one dementia subject (``nia+219@rowan.kr``) whose stated
justification is that removing them improves AUC.  Choosing an exclusion by its
effect on the score is the same class of error as tuning on the test set, so
this folder does not do that.  Instead ``QUALITY_RULES`` states **label-blind**
plausibility criteria up front, and ``train.py`` reports the full cohort *and*
the filtered cohort side by side as a sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from SangHyo.Binary.Binary_Google_MaxAUC_Tuned import features as _F

DIAG_ORDER = ("CN", "MCI", "Dem")
SEVERITY = {"CN": 0, "MCI": 1, "Dem": 2}
_LABEL_FILE = {"train": "training_label.csv", "val": "val_label.csv"}
_SPLIT_DIR = {"train": "1.Training", "val": "2.Validation"}

# Label-blind data-quality rules.  Each maps a feature name to a predicate; a
# subject flagged by any rule is excluded in the *sensitivity* arm only.
QUALITY_RULES = {
    "implausible_sustained_steps": (
        "w_activity_steps__mean",
        lambda v: np.isfinite(v) and v > 25000.0,
        "일 평균 25,000보 초과 — 장기간 지속되기 어려운 값(기기 오사용/기록 오류 가능)",
    ),
    "insufficient_wear": (
        "w_n_activity_days",
        lambda v: np.isfinite(v) and v < 7.0,
        "착용 일수 7일 미만 — 일별 변동성 피처를 신뢰할 수 없음",
    ),
}


@dataclass(frozen=True)
class Cohort:
    subject_ids: np.ndarray
    diagnosis: np.ndarray        # 'CN' | 'MCI' | 'Dem'
    severity: np.ndarray         # 0 | 1 | 2
    y: np.ndarray                # 1 = Dem, 0 = CN+MCI
    X: np.ndarray
    feature_names: tuple[str, ...]
    item_max: dict
    split_of: np.ndarray         # 'train' | 'val', kept for reporting only

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def select(self, names: Sequence[str]) -> "Cohort":
        index = {n: i for i, n in enumerate(self.feature_names)}
        missing = [n for n in names if n not in index]
        if missing:
            raise KeyError(f"Requested features not built: {missing}")
        cols = [index[n] for n in names]
        return Cohort(self.subject_ids, self.diagnosis, self.severity, self.y,
                      self.X[:, cols], tuple(names), self.item_max, self.split_of)

    def subset(self, mask: np.ndarray) -> "Cohort":
        mask = np.asarray(mask, dtype=bool)
        return Cohort(self.subject_ids[mask], self.diagnosis[mask], self.severity[mask],
                      self.y[mask], self.X[mask], self.feature_names, self.item_max,
                      self.split_of[mask])

    def wearable_only(self) -> "Cohort":
        return self.select([n for n in self.feature_names if not n.startswith("mmse_")])


def _labels(data_root: Path, split: str) -> pd.Series:
    path = (Path(data_root) / _SPLIT_DIR[split] / "LabelingData" / "1.Gait" / _LABEL_FILE[split])
    frame = _F._read_csv(path)
    id_col = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns), None)
    if id_col is None:
        raise KeyError(f"Label file lacks SAMPLE_EMAIL/EMAIL: {path}")
    frame = frame.copy()
    frame["_sid"] = frame[id_col].astype(str).str.strip()
    series = frame.drop_duplicates("_sid").set_index("_sid")["DIAG_NM"].astype(str).str.strip()
    unknown = sorted(set(series.unique()) - set(DIAG_ORDER))
    if unknown:
        raise AssertionError(f"Unexpected DIAG_NM values in {path.name}: {unknown}")
    return series


def _split_features(data_root: Path, split: str, item_max: dict | None):
    root = _F._resolve_split_root(Path(data_root) / _SPLIT_DIR[split], split)
    mmse_table = _F._mmse_frame(root, split)
    subjects = sorted(str(s) for s in mmse_table.index)
    mmse_df, resolved = _F._mmse_features(mmse_table.reindex(subjects), item_max)
    wear_df = _F._wearable_features(root, split).reindex(subjects)
    frame = pd.concat([mmse_df, wear_df], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame, resolved


def load_cohort(data_root: str | Path) -> Cohort:
    """All 174 subjects, features + 3-class diagnosis, in one table.

    ``item_max`` (the per-item MMSE maximum used by the engineered composites) is
    learned on the training split only and reused for validation, so pooling the
    two splits does not let validation rows influence their own encoding.
    """

    data_root = Path(data_root)
    train_frame, item_max = _split_features(data_root, "train", None)
    val_frame, _ = _split_features(data_root, "val", item_max)
    if list(train_frame.columns) != list(val_frame.columns):
        raise AssertionError("Train/validation feature schema mismatch")

    frame = pd.concat([train_frame, val_frame], axis=0)
    origin = np.array(["train"] * len(train_frame) + ["val"] * len(val_frame))
    labels = pd.concat([_labels(data_root, "train"), _labels(data_root, "val")])

    subjects = [str(s) for s in frame.index]
    if len(set(subjects)) != len(subjects):
        raise AssertionError("Duplicate subject id across pooled splits")
    missing = [s for s in subjects if s not in labels.index]
    if missing:
        raise AssertionError(f"{len(missing)} subjects without a diagnosis label")

    diagnosis = labels.reindex(subjects).to_numpy(str)
    severity = np.array([SEVERITY[d] for d in diagnosis], dtype=np.int64)
    cohort = Cohort(
        subject_ids=np.asarray(subjects, dtype=str),
        diagnosis=diagnosis,
        severity=severity,
        y=(severity == 2).astype(np.int64),
        X=frame.to_numpy(np.float64),
        feature_names=tuple(frame.columns),
        item_max=item_max,
        split_of=origin,
    )
    _assert_contract(cohort)
    return cohort


def _assert_contract(cohort: Cohort) -> None:
    counts = {d: int((cohort.diagnosis == d).sum()) for d in DIAG_ORDER}
    if cohort.n_subjects != 174 or counts != {"CN": 111, "MCI": 51, "Dem": 12}:
        raise AssertionError(f"Cohort contract changed: n={cohort.n_subjects}, {counts}")
    bad = [n for n in cohort.feature_names if "diag" in n.lower()]
    if bad:
        raise AssertionError(f"Diagnosis leaked into features: {bad}")


def quality_flags(cohort: Cohort) -> dict:
    """Apply the label-blind plausibility rules; never consults ``cohort.y``."""

    index = {n: i for i, n in enumerate(cohort.feature_names)}
    flagged: dict[str, list[str]] = {}
    for rule, (feature, predicate, reason) in QUALITY_RULES.items():
        if feature not in index:
            continue
        column = cohort.X[:, index[feature]]
        for position in np.where([bool(predicate(v)) for v in column])[0]:
            flagged.setdefault(str(cohort.subject_ids[position]), []).append(f"{rule}: {reason}")
    return flagged


def hash_subject_id(subject_id: str) -> str:
    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:16]


__all__ = ["Cohort", "DIAG_ORDER", "QUALITY_RULES", "SEVERITY", "hash_subject_id",
           "load_cohort", "quality_flags"]
