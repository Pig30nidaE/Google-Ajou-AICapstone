"""Load the AI-Hub lifelog CSVs into one tidy daily frame.

The loader stays deliberately close to the paper's own snippet (Table 12 / Table 9)
so that the reconstruction is auditable, and records every place where the paper's
description and the distributed data disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.io import hash_subject
from . import schema

FILE_LAYOUT = {
    "train": {
        "activity": "1.Training/SourceData/1.Gait/train_activity.csv",
        "sleep": "1.Training/SourceData/2.Sleep/train_sleep.csv",
        "mmse": "1.Training/SourceData/3.CognitiveFunction/train_mmse.csv",
        "label_gait": "1.Training/LabelingData/1.Gait/training_label.csv",
        "label_sleep": "1.Training/LabelingData/2.Sleep/training_label.csv",
        "label_cog": "1.Training/LabelingData/3.CognitiveFunction/training_label.csv",
    },
    "validation": {
        "activity": "2.Validation/SourceData/1.Gait/val_activity.csv",
        "sleep": "2.Validation/SourceData/2.Sleep/val_sleep.csv",
        "mmse": "2.Validation/SourceData/3.CognitiveFunction/val_mmse.csv",
        "label_gait": "2.Validation/LabelingData/1.Gait/val_label.csv",
        "label_sleep": "2.Validation/LabelingData/2.Sleep/val_label.csv",
        "label_cog": "2.Validation/LabelingData/3.CognitiveFunction/val_label.csv",
    },
}


class DataContractError(RuntimeError):
    """Raised when the data on disk violates an assumption this code relies on."""


@dataclass
class LifelogData:
    """The loaded dataset plus everything needed to audit it.

    ``daily`` has one row per (subject, observation) with the merged activity and
    sleep features, ``subjects`` has one row per person with the binary label.
    ``cognitive`` is kept separate on purpose: it must never reach the main analysis.
    """

    daily: pd.DataFrame
    subjects: pd.DataFrame
    cognitive: pd.DataFrame
    feature_columns: tuple[str, ...]
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def n_subjects(self) -> int:
        return int(self.subjects[schema.SUBJECT_ID].nunique())

    @property
    def n_rows(self) -> int:
        return int(len(self.daily))

    def labels_by_subject(self) -> pd.Series:
        return self.subjects.set_index(schema.SUBJECT_ID)[schema.LABEL_COL].astype(int)

    def groups(self) -> np.ndarray:
        """Group vector for GroupKFold -- subject ids aligned to ``daily`` rows."""
        return self.daily[schema.SUBJECT_ID].to_numpy()


def resolve_data_root(candidates: list[str | Path]) -> Path:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "1.Training").is_dir() and (path / "2.Validation").is_dir():
            return path
    raise FileNotFoundError(
        "Data root containing 1.Training and 2.Validation not found. Checked: "
        + ", ".join(str(c) for c in candidates if c)
    )


def _read(root: Path, split: str, key: str) -> pd.DataFrame:
    path = root / FILE_LAYOUT[split][key]
    if not path.is_file():
        raise FileNotFoundError(f"missing expected data file: {path}")
    return pd.read_csv(path, low_memory=False)


def _to_date(series: pd.Series) -> pd.Series:
    """Parse mixed-offset ISO8601 timestamps to Asia/Seoul calendar dates."""
    parsed = pd.to_datetime(series, format="mixed", utc=True)
    return parsed.dt.tz_convert("Asia/Seoul").dt.normalize().dt.tz_localize(None)


def _check_label_copies(root: Path, split: str) -> dict[str, Any]:
    frames = {
        key: _read(root, split, key)
        for key in ("label_gait", "label_sleep", "label_cog")
    }
    reference = frames["label_gait"].sort_values(schema.LABEL_PERSON_KEY).reset_index(drop=True)
    identical = {}
    for key, frame in frames.items():
        other = frame.sort_values(schema.LABEL_PERSON_KEY).reset_index(drop=True)
        identical[key] = bool(reference.equals(other))
    if not all(identical.values()):
        raise DataContractError(
            f"[{split}] the three label copies disagree: {identical}. "
            "Resolve the source of truth before training."
        )
    return {"label_copies_identical": True, "n_label_rows": int(len(reference))}


def load_lifelog(
    data_root: str | Path,
    *,
    join_mode: str = "positional",
    sleep_date_source: str = "bedtime_end",
    feature_set: str = "paper_code_verbatim",
    drop_zero_variance: bool = False,
    drop_administrative: bool = False,
) -> LifelogData:
    """Load both splits and merge activity + sleep into one daily frame.

    Parameters
    ----------
    join_mode:
        ``positional`` pairs the i-th activity row with the i-th sleep row inside
        each subject (the files ship 1:1 aligned).  ``date`` inner-joins on
        ``(subject_id, date)`` and loses the rows whose sleep date collides.
    sleep_date_source:
        ``bedtime_end`` (wake time, aligns with the activity day) or
        ``bedtime_start``.
    """
    root = Path(data_root).expanduser().resolve()
    if join_mode not in ("positional", "date"):
        raise ValueError(f"join_mode must be positional/date, got {join_mode!r}")
    if sleep_date_source not in schema.SLEEP_DATE_SOURCES:
        raise ValueError(f"unknown sleep_date_source: {sleep_date_source!r}")

    notes: dict[str, Any] = {
        "join_mode": join_mode,
        "sleep_date_source": sleep_date_source,
        "feature_set": feature_set,
        "per_split": {},
    }

    daily_parts: list[pd.DataFrame] = []
    subject_parts: list[pd.DataFrame] = []
    cognitive_parts: list[pd.DataFrame] = []

    for split in ("train", "validation"):
        notes["per_split"][split] = _check_label_copies(root, split)

        activity = _read(root, split, "activity")
        sleep = _read(root, split, "sleep")
        mmse = _read(root, split, "mmse")

        raw_counts = {
            "activity_rows": int(len(activity)),
            "sleep_rows": int(len(sleep)),
            "mmse_rows": int(len(mmse)),
            "activity_subjects": int(activity[schema.SOURCE_PERSON_KEY].nunique()),
            "sleep_subjects": int(sleep[schema.SOURCE_PERSON_KEY].nunique()),
        }
        notes["per_split"][split].update(raw_counts)

        for frame in (activity, sleep):
            frame[schema.SUBJECT_ID] = frame[schema.SOURCE_PERSON_KEY].map(hash_subject)
        mmse[schema.SUBJECT_ID] = mmse[schema.LABEL_PERSON_KEY].map(hash_subject)

        activity[schema.DATE_COL] = _to_date(activity[schema.ACTIVITY_DATE_SOURCE])
        sleep[schema.DATE_COL] = _to_date(
            sleep[schema.SLEEP_DATE_SOURCES[sleep_date_source]]
        )

        merged = _merge_modalities(activity, sleep, join_mode=join_mode, split=split)
        notes["per_split"][split]["merged_rows"] = int(len(merged))
        notes["per_split"][split]["rows_lost_in_join"] = int(len(activity) - len(merged))

        # --- labels ---------------------------------------------------------
        if mmse[schema.SUBJECT_ID].duplicated().any():
            raise DataContractError(
                f"[{split}] MMSE has more than one row per subject; the binary label "
                "would be ambiguous."
            )
        multi = mmse.groupby(schema.SUBJECT_ID)[schema.DIAGNOSIS_COL].nunique()
        if (multi > 1).any():
            raise DataContractError(
                f"[{split}] subjects with more than one diagnosis: "
                f"{multi[multi > 1].index.tolist()}"
            )

        subjects = mmse[[schema.SUBJECT_ID, schema.DIAGNOSIS_COL]].copy()
        unknown = set(subjects[schema.DIAGNOSIS_COL]) - set(
            schema.POSITIVE_DIAGNOSES + schema.NEGATIVE_DIAGNOSES
        )
        if unknown:
            raise DataContractError(f"[{split}] unexpected diagnosis values: {unknown}")
        subjects[schema.LABEL_COL] = (
            subjects[schema.DIAGNOSIS_COL].isin(schema.POSITIVE_DIAGNOSES).astype(int)
        )
        subjects["split_origin"] = split

        cognitive = mmse.drop(columns=[schema.LABEL_PERSON_KEY]).copy()
        cognitive["split_origin"] = split

        merged = merged.merge(
            subjects[[schema.SUBJECT_ID, schema.LABEL_COL]],
            on=schema.SUBJECT_ID,
            how="inner",
            validate="many_to_one",
        )
        merged["split_origin"] = split

        daily_parts.append(merged)
        subject_parts.append(subjects)
        cognitive_parts.append(cognitive)

    daily = pd.concat(daily_parts, ignore_index=True)
    subjects = pd.concat(subject_parts, ignore_index=True)
    cognitive = pd.concat(cognitive_parts, ignore_index=True)

    overlap = set(subject_parts[0][schema.SUBJECT_ID]) & set(
        subject_parts[1][schema.SUBJECT_ID]
    )
    if overlap:
        raise DataContractError(
            f"{len(overlap)} subjects appear in both the Training and Validation "
            "partitions; the official split would not be subject-disjoint."
        )

    features = _select_features(
        daily,
        feature_set=feature_set,
        drop_zero_variance=drop_zero_variance,
        drop_administrative=drop_administrative,
        notes=notes,
    )

    daily = daily.sort_values([schema.SUBJECT_ID, schema.DATE_COL]).reset_index(drop=True)
    daily["row_id"] = np.arange(len(daily), dtype=np.int64)

    notes["n_subjects"] = int(subjects[schema.SUBJECT_ID].nunique())
    notes["n_daily_rows"] = int(len(daily))
    notes["n_features"] = len(features)
    notes["paper_claimed_daily_rows"] = 12184
    notes["daily_row_discrepancy_vs_paper"] = 12184 - int(len(daily))
    notes["paper_claimed_feature_count"] = 58

    return LifelogData(
        daily=daily,
        subjects=subjects,
        cognitive=cognitive,
        feature_columns=features,
        notes=notes,
    )


def _merge_modalities(
    activity: pd.DataFrame, sleep: pd.DataFrame, *, join_mode: str, split: str
) -> pd.DataFrame:
    """Combine the activity and sleep rows for one split."""
    activity = activity.drop(columns=list(schema.CONVERT_SERIES_COLUMNS), errors="ignore")
    sleep = sleep.drop(columns=list(schema.CONVERT_SERIES_COLUMNS), errors="ignore")

    if join_mode == "positional":
        act_counts = activity.groupby(schema.SUBJECT_ID).size().sort_index()
        slp_counts = sleep.groupby(schema.SUBJECT_ID).size().sort_index()
        if not act_counts.equals(slp_counts):
            raise DataContractError(
                f"[{split}] join_mode='positional' needs identical per-subject row "
                "counts in activity and sleep, but they differ. Use join_mode='date'."
            )
        activity = activity.copy()
        sleep = sleep.copy()
        activity["_pos"] = activity.groupby(schema.SUBJECT_ID).cumcount()
        sleep["_pos"] = sleep.groupby(schema.SUBJECT_ID).cumcount()
        sleep_side = sleep.drop(
            columns=[schema.SOURCE_PERSON_KEY, schema.DATE_COL], errors="ignore"
        )
        merged = activity.merge(
            sleep_side, on=[schema.SUBJECT_ID, "_pos"], how="inner", validate="one_to_one"
        )
        merged = merged.drop(columns=["_pos"])
    else:
        for name, frame in (("activity", activity), ("sleep", sleep)):
            dupes = int(frame.duplicated([schema.SUBJECT_ID, schema.DATE_COL]).sum())
            if dupes:
                frame.drop_duplicates(
                    [schema.SUBJECT_ID, schema.DATE_COL], keep="first", inplace=True
                )
        sleep_side = sleep.drop(columns=[schema.SOURCE_PERSON_KEY], errors="ignore")
        merged = activity.merge(
            sleep_side,
            on=[schema.SUBJECT_ID, schema.DATE_COL],
            how="inner",
            validate="one_to_one",
        )

    return merged.drop(columns=[schema.SOURCE_PERSON_KEY], errors="ignore")


def _select_features(
    daily: pd.DataFrame,
    *,
    feature_set: str,
    drop_zero_variance: bool,
    drop_administrative: bool,
    notes: dict[str, Any],
) -> tuple[str, ...]:
    """Apply the paper's drop list and pick the numeric feature columns."""
    if feature_set not in schema.FEATURE_SETS:
        raise NotImplementedError(
            f"feature_set={feature_set!r} is unknown; expected one of "
            f"{sorted(schema.FEATURE_SETS)}"
        )
    wanted = schema.FEATURE_SETS[feature_set]

    candidates = [c for c in wanted if c in daily.columns]
    missing = [c for c in wanted if c not in daily.columns]
    if missing:
        raise DataContractError(
            f"features expected by feature_set={feature_set!r} are absent from the "
            f"data: {missing}"
        )
    notes["feature_set_size"] = len(candidates)
    if feature_set == "paper_table16_lifelog":
        notes["feature_set_source"] = (
            "Table 16 (thesis) / Table 13 (journal): the only concrete lifelog "
            "variable list either paper prints. Reconstruction hypothesis for the "
            "unstated 'final analysis variable set' of Section 3.2."
        )

    # The paper's drop list, applied verbatim. Three of its five entries are no-ops.
    applied = [c for c in schema.PAPER_DROP_COLUMNS if c in candidates]
    no_ops = [c for c in schema.PAPER_DROP_COLUMNS if c not in daily.columns]
    candidates = [c for c in candidates if c not in applied]
    notes["paper_drop_list_applied"] = applied
    notes["paper_drop_list_no_ops"] = no_ops

    non_numeric = [
        c for c in candidates if not pd.api.types.is_numeric_dtype(daily[c])
    ]
    if non_numeric:
        raise DataContractError(
            f"expected numeric features but got object dtype for {non_numeric}"
        )

    dropped: dict[str, list[str]] = {}
    if drop_zero_variance:
        zero_var = [c for c in candidates if c in schema.ZERO_VARIANCE_FEATURES]
        candidates = [c for c in candidates if c not in zero_var]
        dropped["zero_variance"] = zero_var
    if drop_administrative:
        admin = [c for c in candidates if c in schema.ADMINISTRATIVE_FEATURES]
        candidates = [c for c in candidates if c not in admin]
        dropped["administrative"] = admin
    notes["dropped_by_config"] = dropped

    return tuple(candidates)
