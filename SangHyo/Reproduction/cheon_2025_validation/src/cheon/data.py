"""Raw data loading (READ-ONLY). Never touches 3.CognitiveFunction (MMSE) files.

Paper facts verified here (PAPER_PROTOCOL.md, Cohort / Classification task):
  174 subjects, 12,183 daily records, CN 7,737 vs MCI+Dem 4,446.
Operational assumptions used here (ASSUMPTIONS.md): A02 (pool Training + Validation),
A03 (row-wise pairing of the activity and sleep tables), A04 (label from the Gait label copy,
Sleep copy asserted identical), A05 (hashed subject id).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITY_FILES = {
    "train": "1.Training/SourceData/1.Gait/train_activity.csv",
    "val": "2.Validation/SourceData/1.Gait/val_activity.csv",
}
SLEEP_FILES = {
    "train": "1.Training/SourceData/2.Sleep/train_sleep.csv",
    "val": "2.Validation/SourceData/2.Sleep/val_sleep.csv",
}
LABEL_FILES = {  # Gait copy is the primary; Sleep copy is asserted identical. CognitiveFunction copy is never opened.
    "train": ("1.Training/LabelingData/1.Gait/training_label.csv", "1.Training/LabelingData/2.Sleep/training_label.csv"),
    "val": ("2.Validation/LabelingData/1.Gait/val_label.csv", "2.Validation/LabelingData/2.Sleep/val_label.csv"),
}
FORBIDDEN_PATH_TOKENS = ("CognitiveFunction", "mmse", "MMSE")

EXPECTED = {"n_subjects": 174, "n_records": 12183, "n_cn_records": 7737, "n_impaired_records": 4446}

ID_COL = "EMAIL"
LABEL_COL = "DIAG_NM"


def _check_path(p: Path) -> Path:
    if any(tok in str(p) for tok in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"Refusing to open forbidden cognitive-test file: {p}")
    return p


def _hash_id(s: str) -> str:
    return hashlib.sha256(("cheon2025|" + s).encode("utf-8")).hexdigest()[:16]


def data_fingerprint(data_root: Path) -> str:
    """SHA-256 over the raw activity/sleep/label CSV bytes (sorted paths). No content is exposed."""
    h = hashlib.sha256()
    files = sorted([*ACTIVITY_FILES.values(), *SLEEP_FILES.values(), *(f for pair in LABEL_FILES.values() for f in pair)])
    for rel in files:
        h.update(rel.encode())
        h.update(_check_path(Path(data_root) / rel).read_bytes())
    return h.hexdigest()


def load_daily_records(data_root: str | Path, check_expected: bool = True) -> pd.DataFrame:
    """Return one row per (subject, day): raw activity + sleep columns, `subject_id` (hashed), `y`, `partition`.

    The raw EMAIL column is dropped before returning; only the salted SHA-256 prefix is kept.
    """
    data_root = Path(data_root)
    parts = []
    for part in ("train", "val"):
        act = pd.read_csv(_check_path(data_root / ACTIVITY_FILES[part]))
        slp = pd.read_csv(_check_path(data_root / SLEEP_FILES[part]))
        if len(act) != len(slp):
            raise AssertionError(f"[{part}] activity rows {len(act)} != sleep rows {len(slp)}")
        if not (act[ID_COL].to_numpy() == slp[ID_COL].to_numpy()).all():
            raise AssertionError(f"[{part}] activity/sleep tables are not row-aligned by subject (A03 violated)")
        both = pd.concat([act.reset_index(drop=True), slp.drop(columns=[ID_COL]).reset_index(drop=True)], axis=1)

        gait_lab = pd.read_csv(_check_path(data_root / LABEL_FILES[part][0]))
        sleep_lab = pd.read_csv(_check_path(data_root / LABEL_FILES[part][1]))
        m = gait_lab.merge(sleep_lab, on="SAMPLE_EMAIL", how="outer", suffixes=("_g", "_s"))
        if len(m) != len(gait_lab) or not (m[f"{LABEL_COL}_g"] == m[f"{LABEL_COL}_s"]).all():
            raise AssertionError(f"[{part}] Gait and Sleep label copies differ (A04)")
        lab = gait_lab.rename(columns={"SAMPLE_EMAIL": ID_COL})[[ID_COL, LABEL_COL]]
        if lab[ID_COL].duplicated().any():
            raise AssertionError(f"[{part}] duplicated subject in label file")
        both = both.merge(lab, on=ID_COL, how="left", validate="m:1")
        if both[LABEL_COL].isna().any():
            raise AssertionError(f"[{part}] {int(both[LABEL_COL].isna().sum())} records without a diagnosis label")
        both["partition"] = part
        parts.append(both)

    df = pd.concat(parts, ignore_index=True)
    df["subject_id"] = df[ID_COL].astype(str).map(_hash_id)
    df["y"] = (df[LABEL_COL] != "CN").astype(int)  # CN -> 0, MCI/Dem -> 1 (paper p.164)
    df["diag"] = df[LABEL_COL].astype(str)
    df = df.drop(columns=[ID_COL, LABEL_COL])
    df.insert(0, "record_id", np.arange(len(df)))

    if check_expected:
        got = {
            "n_subjects": int(df["subject_id"].nunique()),
            "n_records": int(len(df)),
            "n_cn_records": int((df["y"] == 0).sum()),
            "n_impaired_records": int((df["y"] == 1).sum()),
        }
        if got != EXPECTED:
            raise AssertionError(f"Cohort does not match the paper's counts: got {got}, expected {EXPECTED}")
    return df


def cohort_summary(df: pd.DataFrame) -> dict:
    subj = df.drop_duplicates("subject_id")
    return {
        "n_subjects": int(subj.shape[0]),
        "n_subjects_by_diag": subj["diag"].value_counts().to_dict(),
        "n_subjects_by_y": subj["y"].value_counts().sort_index().to_dict(),
        "n_records": int(len(df)),
        "n_records_by_diag": df["diag"].value_counts().to_dict(),
        "n_records_by_y": df["y"].value_counts().sort_index().to_dict(),
        "n_records_by_partition": df["partition"].value_counts().to_dict(),
        "days_per_subject": {
            "min": int(df.groupby("subject_id").size().min()),
            "median": float(df.groupby("subject_id").size().median()),
            "max": int(df.groupby("subject_id").size().max()),
        },
    }
