"""Label-safe, MMSE-free subject feature construction.

The raw cognitive-function directory is deliberately absent from every path
resolver in this module.  Activity and sleep are transformed with the audited
``ThreeClass_PerformanceLab`` feature contract: recent observed-event summaries,
intraday physiology/state features, and a strict last-activity prediction index.
No label is used while creating a feature value.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


CLASS_NAMES = ("CN", "MCI", "DEM")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
LABEL_ALIASES = {
    "cn": "CN",
    "normal": "CN",
    "mci": "MCI",
    "dem": "DEM",
    "dementia": "DEM",
}

# These tokens are forbidden in every primary model feature.  Coverage and
# calendar/protocol fields are also blocked because the earlier EDA showed that
# collection protocol itself differs by diagnosis group.
FORBIDDEN_FEATURE_TOKENS = (
    "email",
    "subject_id",
    "diag",
    "mmse",
    "cognitive",
    "doctor",
    "period_id",
    "sample_order",
    "observed_count",
    "observed_day",
    "observed_night",
    "valid_count",
    "valid_ratio",
    "raw_length",
    "sequence_length",
    "missing_ratio",
    "coverage",
    "calendar_gap",
    "span_day",
    "duplicate",
    "nonwear",
    "non_wear",
    "mask",
    "delta_since",
    "absolute_date",
)


def _load_performance_lab_core():
    """Load the repository's audited deterministic feature implementation."""

    core_path = (
        Path(__file__).resolve().parents[1]
        / "ThreeClass_PerformanceLab"
        / "performance_lab_core.py"
    )
    if not core_path.is_file():
        raise FileNotFoundError(
            "Required audited feature core is missing: " f"{core_path}"
        )
    spec = importlib.util.spec_from_file_location("cnboost_performance_lab_core", core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load feature core from {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PERFORMANCE_CORE = _load_performance_lab_core()


@dataclass(frozen=True)
class SplitFiles:
    """Only wearable sources and diagnosis-label copies are addressable."""

    root: Path
    activity: Path
    sleep: Path
    labels: tuple[Path, ...]


@dataclass
class SubjectDataset:
    subject_ids: np.ndarray
    X: pd.DataFrame
    y: np.ndarray | None
    audit: dict


def _one(root: Path, pattern: str, role: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {role} file below {root} with {pattern!r}; "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def discover_split_files(
    split_root: str | Path,
    *,
    require_labels: bool,
) -> SplitFiles:
    """Resolve wearable inputs without ever resolving an MMSE source path."""

    root = Path(split_root).expanduser().resolve()
    activity = _one(root, "SourceData/1.Gait/*activity.csv", "activity")
    sleep = _one(root, "SourceData/2.Sleep/*sleep.csv", "sleep")
    label_paths: tuple[Path, ...] = ()
    if require_labels:
        candidates = tuple(
            sorted(path for path in root.glob("LabelingData/*/*label.csv") if path.is_file())
        )
        if not candidates:
            raise FileNotFoundError(f"No diagnosis label copies found below {root}")
        label_paths = candidates
    return SplitFiles(root=root, activity=activity, sleep=sleep, labels=label_paths)


def normalize_label(value: object) -> str:
    key = str(value).strip().lower()
    if key not in LABEL_ALIASES:
        raise ValueError(f"Unknown diagnosis label: {value!r}")
    return LABEL_ALIASES[key]


def _read_label_copy(path: Path) -> pd.Series:
    frame = pd.read_csv(path, dtype=str)
    id_columns = [
        column
        for column in frame.columns
        if str(column).strip().upper() in {"EMAIL", "SAMPLE_EMAIL"}
    ]
    if len(id_columns) != 1 or "DIAG_NM" not in frame.columns:
        raise ValueError(f"Unexpected label schema in {path}: {list(frame.columns)}")
    labels = frame[[id_columns[0], "DIAG_NM"]].copy()
    labels.columns = ["subject_id", "label"]
    labels["subject_id"] = labels["subject_id"].astype(str).str.strip()
    labels["label"] = labels["label"].map(normalize_label)
    if labels["subject_id"].duplicated().any():
        conflicts = labels.groupby("subject_id")["label"].nunique()
        if (conflicts > 1).any():
            raise ValueError(f"Conflicting labels in {path}")
        labels = labels.drop_duplicates("subject_id", keep="last")
    return labels.set_index("subject_id")["label"].sort_index()


def load_consistent_labels(paths: Sequence[Path]) -> pd.Series:
    if not paths:
        raise ValueError("At least one label copy is required")
    copies = [_read_label_copy(Path(path)) for path in paths]
    reference = copies[0]
    for path, candidate in zip(paths[1:], copies[1:]):
        if not reference.index.equals(candidate.index) or not reference.equals(candidate):
            raise AssertionError(f"Diagnosis label copies disagree: {paths[0]} vs {path}")
    return reference


def assert_no_forbidden_features(columns: Iterable[str]) -> None:
    offenders = sorted(
        str(column)
        for column in columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    )
    if offenders:
        raise AssertionError(f"Forbidden feature(s) detected: {offenders[:20]}")


def _source_subjects(frame: pd.DataFrame, source: str) -> list[str]:
    if "EMAIL" not in frame.columns:
        raise ValueError(f"{source} source does not contain EMAIL")
    subjects = frame["EMAIL"].astype(str).str.strip()
    if (subjects == "").any():
        raise ValueError(f"{source} contains an empty subject identifier")
    return sorted(subjects.unique().tolist())


def _aggregate_observed_window(
    daily: pd.DataFrame,
    subject_ids: Sequence[str],
    feature_columns: Sequence[str],
    *,
    modality: str,
    window: int,
) -> pd.DataFrame:
    """Aggregate the last N observed events without exposing their dates/count."""

    rows: list[dict[str, float]] = []
    for subject_id in subject_ids:
        events = (
            daily.loc[daily["_subject"] == subject_id]
            .sort_values("_event_timestamp", kind="mergesort")
            .tail(window)
        )
        ranks = PERFORMANCE_CORE._event_ranks(len(events))
        row: dict[str, float] = {}
        for column in feature_columns:
            values = pd.to_numeric(events[column], errors="coerce").to_numpy(dtype=float)
            statistics = PERFORMANCE_CORE._event_summary_stats(values, ranks)
            for statistic, value in statistics.items():
                row[f"{modality}__event{window}__{column}__{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(subject_ids, name="_subject"))


def build_multiscale_event_summary(
    activity_raw: pd.DataFrame,
    sleep_raw: pd.DataFrame,
    subject_ids: Sequence[str],
    *,
    windows: tuple[int, ...] = (7, 14, 28),
) -> tuple[pd.DataFrame, dict]:
    """Build 7/14/28 recent-observed-event views with a shared time guard."""

    activity_daily = PERFORMANCE_CORE.make_activity_daily(activity_raw)
    anchor_timestamps = (
        pd.DataFrame(
            {
                "_subject": activity_raw["EMAIL"].astype(str).str.strip(),
                "_end": PERFORMANCE_CORE._parse_kst_timestamps(
                    activity_raw["activity_day_end"], "activity_day_end"
                ),
            }
        )
        .groupby("_subject", sort=True)["_end"]
        .max()
    )
    guarded_sleep = sleep_raw.copy()
    guarded_sleep["_guard_wake"] = PERFORMANCE_CORE._parse_kst_timestamps(
        guarded_sleep["sleep_bedtime_end"], "sleep_bedtime_end"
    )
    guarded_sleep["_guard_anchor"] = (
        guarded_sleep["EMAIL"].astype(str).str.strip().map(anchor_timestamps)
    )
    if guarded_sleep["_guard_anchor"].isna().any():
        raise AssertionError("Sleep subject has no activity prediction timestamp")
    future_sleep_rows = int(
        (guarded_sleep["_guard_wake"] > guarded_sleep["_guard_anchor"]).sum()
    )
    guarded_sleep = guarded_sleep.loc[
        guarded_sleep["_guard_wake"] <= guarded_sleep["_guard_anchor"]
    ].drop(columns=["_guard_wake", "_guard_anchor"])
    sleep_daily, duplicate_table = PERFORMANCE_CORE.make_sleep_daily(guarded_sleep)
    if set(activity_daily["_subject"].astype(str)) != set(subject_ids):
        raise AssertionError("Activity daily subjects differ from the requested cohort")
    if set(sleep_daily["_subject"].astype(str)) != set(subject_ids):
        raise AssertionError("Sleep daily subjects differ from the requested cohort")

    activity_columns = sorted(
        column
        for column in activity_daily.columns
        if column not in {"_subject", "_event_date", "_event_timestamp"}
    )
    sleep_columns = sorted(
        column
        for column in sleep_daily.columns
        if column not in {"_subject", "_event_date", "_event_timestamp"}
    )
    PERFORMANCE_CORE.assert_primary_feature_contract(activity_columns)
    PERFORMANCE_CORE.assert_primary_feature_contract(sleep_columns)
    views: list[pd.DataFrame] = []
    for window in windows:
        if window < 4:
            raise ValueError("Observed-event windows must contain at least four events")
        views.append(
            _aggregate_observed_window(
                activity_daily,
                subject_ids,
                activity_columns,
                modality="activity",
                window=window,
            )
        )
        views.append(
            _aggregate_observed_window(
                sleep_daily,
                subject_ids,
                sleep_columns,
                modality="sleep",
                window=window,
            )
        )
    summary = pd.concat(views, axis=1)
    if not summary.columns.is_unique:
        raise AssertionError("Multiscale feature names are not unique")
    PERFORMANCE_CORE.assert_primary_feature_contract(summary.columns)
    diagnostics = {
        "prediction_index": "last valid activity_day_end timestamp",
        "activity_raw_rows": int(len(activity_raw)),
        "sleep_raw_rows": int(len(sleep_raw)),
        "activity_daily_events": int(len(activity_daily)),
        "sleep_main_events_before_anchor": int(len(sleep_daily)),
        "sleep_post_index_raw_episodes_excluded": future_sleep_rows,
        "duplicate_sleep_subject_date_groups_audited_not_featured": int(
            len(duplicate_table)
        ),
        "activity_daily_biological_features": int(len(activity_columns)),
        "sleep_daily_biological_features": int(len(sleep_columns)),
        "observed_event_windows": list(windows),
        "primary_explicit_coverage_signals": 0,
    }
    return summary, diagnostics


def build_subject_dataset(
    split_root: str | Path,
    *,
    require_labels: bool,
) -> SubjectDataset:
    """Create the fixed MMSE-free subject table for one official split."""

    files = discover_split_files(split_root, require_labels=require_labels)
    activity = pd.read_csv(files.activity, low_memory=False)
    sleep = pd.read_csv(files.sleep, low_memory=False)
    if "EMAIL" in activity.columns:
        activity["EMAIL"] = activity["EMAIL"].astype(str).str.strip()
    if "EMAIL" in sleep.columns:
        sleep["EMAIL"] = sleep["EMAIL"].astype(str).str.strip()
    activity_subjects = _source_subjects(activity, "activity")
    sleep_subjects = _source_subjects(sleep, "sleep")
    if activity_subjects != sleep_subjects:
        raise AssertionError("Activity and sleep subject sets differ")

    labels = load_consistent_labels(files.labels) if require_labels else None
    if labels is not None:
        subject_ids = labels.index.astype(str).tolist()
        if set(subject_ids) != set(activity_subjects):
            raise AssertionError("Wearable source and label subject sets differ")
    else:
        subject_ids = activity_subjects

    X, source_diagnostics = build_multiscale_event_summary(activity, sleep, subject_ids)
    X.index = pd.Index(subject_ids, name="subject_id")
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    assert_no_forbidden_features(X.columns)
    PERFORMANCE_CORE.assert_primary_feature_contract(X.columns)

    y: np.ndarray | None = None
    if labels is not None:
        aligned = labels.reindex(subject_ids)
        if aligned.isna().any():
            raise AssertionError("At least one subject lost its diagnosis during alignment")
        y = aligned.map(CLASS_TO_ID).to_numpy(dtype=np.int64)

    audit = {
        "split_root": str(files.root),
        "subjects": int(len(subject_ids)),
        "class_counts": None
        if y is None
        else {name: int(np.sum(y == class_id)) for class_id, name in enumerate(CLASS_NAMES)},
        "feature_contract": {
            "version": "observed_event_multiscale_7_14_28_v1",
            "observed_event_windows": [7, 14, 28],
            "statistics": list(PERFORMANCE_CORE.SUMMARY_STATS),
            "feature_count": int(X.shape[1]),
            "feature_names": list(X.columns),
            "forbidden_explicit_signals": [
                "observed count",
                "calendar gap",
                "mask",
                "absolute date",
                "non-wear",
            ],
        },
        "source_diagnostics": source_diagnostics,
        "activity_file": str(files.activity),
        "sleep_file": str(files.sleep),
        "label_copy_count": int(len(files.labels)),
        "label_copies_consistent": bool(labels is not None),
        "mmse_source_resolved": False,
        "mmse_source_opened": False,
        "mmse_values_used": False,
        "coverage_or_calendar_protocol_features_used": False,
        "raw_identifier_used_as_feature": False,
        "direct_diagnosis_used_as_feature": False,
        "prediction_index": "last valid activity_day_end timestamp",
    }
    return SubjectDataset(
        subject_ids=np.asarray(subject_ids, dtype=object),
        X=X.reset_index(drop=True),
        y=y,
        audit=audit,
    )


def feature_family(feature_name: str) -> str:
    """Return a compact family label used for EDA and fold-selection audits."""

    name = str(feature_name).lower().removeprefix("cn_abs__")
    modality = "activity" if name.startswith("activity") else "sleep"
    if any(token in name for token in ("iqr", "mad", "std", "entropy", "transition")):
        behavior = "variability"
    elif any(token in name for token in ("slope", "late_half", "early_half")):
        behavior = "change"
    elif any(token in name for token in ("clock", "bedtime", "midpoint")):
        behavior = "circadian"
    elif any(token in name for token in ("hr", "rmssd")):
        behavior = "cardiac"
    elif any(token in name for token in ("stage", "deep", "light", "rem")):
        behavior = "sleep_stage"
    else:
        behavior = "level"
    return f"{modality}:{behavior}"
