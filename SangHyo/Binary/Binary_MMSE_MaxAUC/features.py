"""EDA-grounded feature construction for the max-ROC-AUC (leakage-free) model.

What the EDA (SangHyo/EDA + additional analysis in eda.py) established:

* CN vs MCI+DEM is driven almost entirely by **MMSE**; the strongest single
  features are ``TOTAL`` (dir-free AUC 0.737), an engineered ``num_failed``
  (# items below the item maximum, 0.737), and delayed-recall (``recall`` 0.729).
* **Wearables dilute the signal** even as day-to-day variability: MMSE-only
  reaches leakage-free subject ROC-AUC ~0.76, but MMSE + wearable drops to ~0.71.
  So the max-AUC model is MMSE-focused; wearable is optional and OFF by default.

Features are per subject (one row) -> a plain subject-level split is leakage-safe
by construction.  Diagnosis columns (``DIAG_NM``/``DIAG_SEQ``) and administrative
metadata are hard-excluded; labels come only from the Gait/Sleep copies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google import data as D

MMSE_DOMAINS = {
    "orient_time": ["Q01", "Q02", "Q03", "Q04", "Q05"],
    "orient_place": ["Q06", "Q07", "Q08", "Q09", "Q10"],
    "registration": ["Q11_1", "Q11_2", "Q11_3"],
    "attention": ["Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"],
    "recall": ["Q13_1", "Q13_2", "Q13_3"],
    "language": ["Q14_1", "Q14_2", "Q15", "Q16_1", "Q16_2", "Q16_3", "Q17", "Q18", "Q19"],
}
MMSE_ITEMS = [c for cols in MMSE_DOMAINS.values() for c in cols]  # 28 raw items
MMSE_FORBIDDEN = frozenset({"DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND"})
_MMSE_FILE = {"train": "train_mmse.csv", "val": "val_mmse.csv"}

# Optional curated wearable *variability* features (best wearable signal in EDA),
# used only when include_wearable=True.  Day-to-day std across a subject's days.
WEARABLE_VAR_CHANNELS = ["sleep_score_deep", "sleep_restless", "sleep_awake",
                         "sleep_efficiency", "activity_score"]


@dataclass(frozen=True)
class SubjectData:
    subject_ids: np.ndarray
    y: np.ndarray | None
    X: np.ndarray
    feature_names: tuple[str, ...]
    item_max: dict           # per-item maximum learned on training (leakage-safe)

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)


def _assert_no_forbidden(names: Sequence[str]) -> None:
    bad = [n for n in names if n in MMSE_FORBIDDEN or "diag" in str(n).lower()]
    if bad:
        raise AssertionError(f"Diagnosis/metadata leaked into features: {bad}")


def _mmse_table(root: Path, split_key: str) -> pd.DataFrame:
    frame = D._read_csv(root / "SourceData" / "3.CognitiveFunction" / _MMSE_FILE[split_key])
    id_col = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns))
    frame = frame.drop(columns=[c for c in frame.columns if c in MMSE_FORBIDDEN]).copy()
    frame["_sid"] = frame[id_col].astype(str).str.strip()
    return frame.drop_duplicates("_sid").set_index("_sid")


def _wearable_variability(root: Path, split_key: str, subjects: Sequence[str]) -> pd.DataFrame:
    aligned, _ = D._prepare_sources(root, split_key)
    rows = {}
    for sid, g in aligned.groupby("_subject_id"):
        r = {}
        for col in WEARABLE_VAR_CHANNELS:
            v = pd.to_numeric(g[col], errors="coerce").to_numpy(float)
            v = v[np.isfinite(v)]
            r[f"w_{col}__std"] = float(np.std(v)) if len(v) > 1 else np.nan
        rows[str(sid)] = r
    return pd.DataFrame(rows).T.reindex([str(s) for s in subjects])


def _build_mmse_features(table: pd.DataFrame, item_max: dict) -> tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(index=table.index)
    out["mmse_TOTAL"] = table["TOTAL"].astype(float)
    for domain, cols in MMSE_DOMAINS.items():
        out[f"mmse_{domain}"] = table[cols].astype(float).sum(axis=1)
    for item in MMSE_ITEMS:
        out[f"mmse_{item}"] = table[item].astype(float)
    # engineered composites (EDA: as strong as TOTAL)
    if item_max is None:
        item_max = {c: float(table[c].astype(float).max()) for c in MMSE_ITEMS}
    below = pd.DataFrame({c: (table[c].astype(float) < item_max[c]).astype(int) for c in MMSE_ITEMS})
    out["mmse_num_failed"] = below.sum(axis=1)
    recall = out["mmse_recall"]
    out["mmse_recall_deficit"] = float(sum(item_max[c] for c in MMSE_DOMAINS["recall"])) - recall
    return out, item_max


def load_split(split_root: str | Path, *, require_labels: bool, split: str,
               include_wearable: bool = False, item_max: dict | None = None) -> SubjectData:
    split_key = D._normalise_split(split)
    root = D.resolve_split_root(split_root, split_key)

    if require_labels:
        labels = D.load_binary_labels(root)
        subjects = [str(s) for s in labels.index]
        y = labels.to_numpy(np.int64)
    else:
        table0 = _mmse_table(root, split_key)
        subjects = sorted(str(s) for s in table0.index)
        y = None

    table = _mmse_table(root, split_key).reindex([str(s) for s in subjects])
    mmse_df, resolved_item_max = _build_mmse_features(table, item_max)
    frames = [mmse_df]
    if include_wearable:
        frames.append(_wearable_variability(root, split_key, subjects))
    X_df = pd.concat(frames, axis=1)
    _assert_no_forbidden(list(X_df.columns))

    return SubjectData(
        subject_ids=np.asarray(subjects, dtype=str),
        y=y,
        X=X_df.to_numpy(np.float64),
        feature_names=tuple(X_df.columns),
        item_max=resolved_item_max,
    )


def load_validation_labels_checked(validation_root, subject_ids):
    root = D.resolve_split_root(validation_root, "val")
    labels = D.load_binary_labels(root)
    wanted = [str(s) for s in subject_ids]
    if set(labels.index.astype(str)) != set(wanted):
        raise AssertionError("Validation label subjects differ from frozen predictions")
    y = labels.loc[wanted].to_numpy(np.int64)
    if len(y) != 33 or {c: int((y == c).sum()) for c in (0, 1)} != {0: 26, 1: 7}:
        raise AssertionError("Validation label contract changed")
    return y


def assert_disjoint_subjects(train_ids, val_ids):
    overlap = sorted(set(map(str, train_ids)) & set(map(str, val_ids)))
    if overlap:
        raise AssertionError(f"Train/Validation subject leakage: {len(overlap)} subjects")


def hash_subject_id(subject_id: str) -> str:
    return hashlib.sha256(str(subject_id).encode()).hexdigest()[:16]


__all__ = ["SubjectData", "assert_disjoint_subjects", "hash_subject_id",
           "load_split", "load_validation_labels_checked"]
