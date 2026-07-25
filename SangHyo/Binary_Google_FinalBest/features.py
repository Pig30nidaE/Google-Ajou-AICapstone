"""EDA-grounded, leakage-safe feature construction (MMSE domains + wearable).

Design comes directly from the team EDA and my own univariate analysis:

* CN vs MCI (the hard boundary; 47 of 56 impaired subjects are MCI) separates
  almost only through **MMSE delayed-recall** items.  Direction-free AUC:
  ``dom_recall`` 0.704, ``TOTAL`` 0.695, ``Q13_3`` 0.670, ``Q13_2`` 0.666.
  Registration (Q11) is constant (everyone 6/6) and is dropped.
* Wearables barely move CN vs MCI (best ~0.60) but clearly flag **Dem**
  (sleep_light 0.80, activity_rest 0.78, sleep_restless 0.77,
  meet_daily_targets 0.74).  A compact set of these is kept.

So every model here fuses domain-engineered MMSE scores with a curated set of
per-subject wearable means.  Diagnosis columns (``DIAG_NM``/``DIAG_SEQ``) and
MMSE administrative metadata are hard-excluded; labels come only from the
Gait/Sleep copies.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from SangHyo.Binary_Wearable_SequenceFusion_Google.data import load_binary_labels

# ----- MMSE domain engineering (values are cognitive scores only) -------------
MMSE_DOMAINS = {
    "orient_time": ["Q01", "Q02", "Q03", "Q04", "Q05"],
    "orient_place": ["Q06", "Q07", "Q08", "Q09", "Q10"],
    "attention": ["Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"],
    "recall": ["Q13_1", "Q13_2", "Q13_3"],          # delayed recall: MCI hallmark
    "language": ["Q14_1", "Q14_2", "Q15", "Q16_1", "Q16_2", "Q16_3", "Q17", "Q18", "Q19"],
}
MMSE_KEY_ITEMS = ["Q13_2", "Q13_3", "Q12_5", "Q03", "Q09"]
MMSE_FORBIDDEN = frozenset(
    {"DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND", "SAMPLE_EMAIL", "EMAIL"}
)

# ----- curated per-subject wearable means (mostly Dem-markers) ----------------
WEARABLE_ACTIVITY_COLS = [
    "activity_score",
    "activity_score_meet_daily_targets",
    "activity_score_training_frequency",
    "activity_score_training_volume",
    "activity_steps",
    "activity_rest",
    "activity_inactive",
    "activity_low",
    "activity_daily_movement",
]
WEARABLE_SLEEP_COLS = [
    "sleep_duration",
    "sleep_awake",
    "sleep_light",
    "sleep_deep",
    "sleep_restless",
    "sleep_efficiency",
    "sleep_score_deep",
    "sleep_breath_average",
    "sleep_hr_average",
    "sleep_hr_lowest",
    "sleep_rmssd",
]

# Final best feature set: MMSE domain scores (best OOF AUC 0.760 in the
# preprocessing-ceiling study) plus the three wearable channels that uniquely
# flag Dementia (sleep_light 0.80, activity_rest 0.78, sleep_restless 0.77).
FINAL_FEATURES = [
    "mmse_TOTAL", "mmse_orient_time", "mmse_orient_place", "mmse_attention",
    "mmse_recall", "mmse_language", "mmse_Q13_2", "mmse_Q13_3", "mmse_Q12_5",
    "mmse_Q03", "mmse_Q09",
    "w_sleep_light", "w_activity_rest", "w_sleep_restless",
]

# Minimal EDA-selected subset (used by the EDA-selective model).
MINIMAL_FEATURES = [
    "mmse_TOTAL", "mmse_recall", "mmse_Q13_2", "mmse_Q13_3", "mmse_Q12_5",
    "mmse_orient_time", "mmse_attention", "mmse_Q03",
    "w_sleep_light", "w_activity_rest", "w_sleep_restless",
    "w_activity_score_meet_daily_targets", "w_sleep_duration", "w_sleep_score_deep",
]

_SPLIT = {
    "train": {"dir": "1.Training", "mmse": "train_mmse.csv", "act": "train_activity.csv",
              "slp": "train_sleep.csv", "n": 141, "counts": {0: 85, 1: 56}},
    "val": {"dir": "2.Validation", "mmse": "val_mmse.csv", "act": "val_activity.csv",
            "slp": "val_sleep.csv", "n": 33, "counts": {0: 26, 1: 7}},
}


@dataclass(frozen=True)
class SubjectData:
    subject_ids: np.ndarray
    y: np.ndarray | None
    X: np.ndarray
    feature_names: tuple[str, ...]
    mmse_names: tuple[str, ...]
    wearable_names: tuple[str, ...]

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    def select(self, names: Sequence[str]) -> "SubjectData":
        index = {n: i for i, n in enumerate(self.feature_names)}
        missing = [n for n in names if n not in index]
        if missing:
            raise KeyError(f"Requested features not built: {missing}")
        cols = [index[n] for n in names]
        keep = tuple(names)
        return SubjectData(
            subject_ids=self.subject_ids,
            y=self.y,
            X=self.X[:, cols],
            feature_names=keep,
            mmse_names=tuple(n for n in keep if n.startswith("mmse_")),
            wearable_names=tuple(n for n in keep if n.startswith("w_")),
        )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode CSV: {path}")


def _resolve_split_root(split_root: str | Path, split: str) -> Path:
    root = Path(split_root).expanduser().resolve()
    expected = _SPLIT[split]["dir"]
    if root.name == expected:
        return root
    if (root / expected).is_dir():
        return root / expected
    if (root / "Data" / expected).is_dir():
        return root / "Data" / expected
    raise FileNotFoundError(f"Could not resolve {expected} under {root}")


def _assert_no_forbidden(names: Sequence[str]) -> None:
    bad = [n for n in names if n in MMSE_FORBIDDEN or "diag" in str(n).lower()]
    if bad:
        raise AssertionError(f"Diagnosis/metadata leaked into features: {bad}")


def _mmse_features(root: Path, split: str, subjects: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    frame = _read_csv(root / "SourceData" / "3.CognitiveFunction" / _SPLIT[split]["mmse"])
    id_col = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in frame.columns), None)
    if id_col is None:
        raise KeyError("MMSE file lacks SAMPLE_EMAIL/EMAIL")
    frame = frame.copy()
    frame["_sid"] = frame[id_col].astype(str).str.strip()
    indexed = frame.drop_duplicates("_sid").set_index("_sid")

    names: list[str] = ["mmse_TOTAL"]
    columns: dict[str, pd.Series] = {"mmse_TOTAL": indexed["TOTAL"].astype(float)}
    for domain, cols in MMSE_DOMAINS.items():
        names.append(f"mmse_{domain}")
        columns[f"mmse_{domain}"] = indexed[cols].astype(float).sum(axis=1)
    for item in MMSE_KEY_ITEMS:
        names.append(f"mmse_{item}")
        columns[f"mmse_{item}"] = indexed[item].astype(float)
    _assert_no_forbidden([n.replace("mmse_", "") for n in names] + names)

    wanted = [str(s) for s in subjects]
    missing = [s for s in wanted if s not in indexed.index]
    if missing:
        raise AssertionError(f"{len(missing)} subjects missing MMSE rows (e.g. {missing[:3]})")
    matrix = np.column_stack([columns[n].reindex(wanted).to_numpy(float) for n in names])
    return matrix, tuple(names)


def _wearable_features(root: Path, split: str, subjects: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    act = _read_csv(root / "SourceData" / "1.Gait" / _SPLIT[split]["act"])
    slp = _read_csv(root / "SourceData" / "2.Sleep" / _SPLIT[split]["slp"])
    for frame, id_default in ((act, "EMAIL"), (slp, "EMAIL")):
        if id_default not in frame.columns:
            raise KeyError(f"Wearable file lacks {id_default}")
    act_mean = (
        act.assign(_sid=act["EMAIL"].astype(str).str.strip())
        .groupby("_sid")[WEARABLE_ACTIVITY_COLS].mean(numeric_only=True)
    )
    slp_mean = (
        slp.assign(_sid=slp["EMAIL"].astype(str).str.strip())
        .groupby("_sid")[WEARABLE_SLEEP_COLS].mean(numeric_only=True)
    )
    names = tuple(f"w_{c}" for c in WEARABLE_ACTIVITY_COLS + WEARABLE_SLEEP_COLS)
    wanted = [str(s) for s in subjects]
    merged = act_mean.join(slp_mean, how="outer").reindex(wanted)
    matrix = merged[WEARABLE_ACTIVITY_COLS + WEARABLE_SLEEP_COLS].to_numpy(float)
    return matrix, names


def load_split(
    split_root: str | Path,
    *,
    require_labels: bool,
    split: str,
    feature_subset: Sequence[str] | None = None,
) -> SubjectData:
    """Build fused MMSE + wearable per-subject features for one split."""

    root = _resolve_split_root(split_root, split)
    if require_labels:
        labels = load_binary_labels(root)
        subjects = [str(s) for s in labels.index]
        y = labels.to_numpy(np.int64)
    else:
        # Validation: never open labels here; take the subject list from MMSE.
        mmse = _read_csv(root / "SourceData" / "3.CognitiveFunction" / _SPLIT[split]["mmse"])
        id_col = next((c for c in ("SAMPLE_EMAIL", "EMAIL") if c in mmse.columns))
        subjects = sorted(mmse[id_col].astype(str).str.strip().unique())
        y = None

    n_expected = _SPLIT[split]["n"]
    if len(subjects) != n_expected:
        raise AssertionError(f"{split}: expected {n_expected} subjects, got {len(subjects)}")
    if y is not None:
        observed = {c: int((y == c).sum()) for c in (0, 1)}
        if observed != _SPLIT[split]["counts"]:
            raise AssertionError(f"{split} class contract failed: {observed}")

    mmse_matrix, mmse_names = _mmse_features(root, split, subjects)
    wear_matrix, wear_names = _wearable_features(root, split, subjects)
    X = np.hstack([mmse_matrix, wear_matrix])
    names = mmse_names + wear_names
    data = SubjectData(
        subject_ids=np.asarray(subjects, dtype=str),
        y=y,
        X=X,
        feature_names=names,
        mmse_names=mmse_names,
        wearable_names=wear_names,
    )
    if feature_subset is not None:
        data = data.select(feature_subset)
    return data


def load_validation_labels_checked(validation_root: str | Path, subject_ids: Sequence[str]) -> np.ndarray:
    root = _resolve_split_root(validation_root, "val")
    labels = load_binary_labels(root)
    wanted = [str(s) for s in subject_ids]
    if set(labels.index.astype(str)) != set(wanted):
        raise AssertionError("Validation label subjects differ from frozen predictions")
    y = labels.loc[wanted].to_numpy(np.int64)
    counts = {c: int((y == c).sum()) for c in (0, 1)}
    if len(y) != 33 or counts != {0: 26, 1: 7}:
        raise AssertionError(f"Validation label contract changed: n={len(y)}, counts={counts}")
    return y


def assert_disjoint_subjects(train_ids: Sequence[str], val_ids: Sequence[str]) -> None:
    overlap = sorted(set(map(str, train_ids)) & set(map(str, val_ids)))
    if overlap:
        raise AssertionError(f"Train/Validation subject leakage: {len(overlap)} subjects")


def hash_subject_id(subject_id: str) -> str:
    return hashlib.sha256(str(subject_id).encode("utf-8")).hexdigest()[:16]


__all__ = [
    "FINAL_FEATURES",
    "MINIMAL_FEATURES",
    "SubjectData",
    "assert_disjoint_subjects",
    "hash_subject_id",
    "load_split",
    "load_validation_labels_checked",
]
