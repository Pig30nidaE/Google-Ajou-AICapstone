#!/usr/bin/env python3
"""Privacy-safe, training-only local EDA for three-class status classification.

External dependencies are deliberately limited to pandas and NumPy. The script:

* reads all three Training label copies and the Training activity/sleep sources;
* never reads any Validation label file;
* reads Validation source headers and identifier columns only, solely for schema
  and train/validation subject-overlap audits;
* excludes MMSE values because the source contains target-adjacent diagnosis and
  cognitive-test fields;
* never prints or persists identifiers, hashes, or subject-level rows; and
* writes exactly three aggregate artifacts under ``artifacts/local_eda``.

Run from any directory with a Python environment containing pandas and NumPy:

    python ThreeClass_PerformanceLab/local_eda.py
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAB_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "Data"
OUTPUT_DIR = LAB_ROOT / "artifacts" / "local_eda"
REPORT_COPY_PATH = LAB_ROOT / "EDA_REPORT_KO.md"

TRAIN_LABEL_PATHS = {
    "gait": DATA_ROOT / "1.Training" / "LabelingData" / "1.Gait" / "training_label.csv",
    "sleep": DATA_ROOT / "1.Training" / "LabelingData" / "2.Sleep" / "training_label.csv",
    "cognitive": DATA_ROOT
    / "1.Training"
    / "LabelingData"
    / "3.CognitiveFunction"
    / "training_label.csv",
}
TRAIN_SOURCE_PATHS = {
    "activity": DATA_ROOT / "1.Training" / "SourceData" / "1.Gait" / "train_activity.csv",
    "sleep": DATA_ROOT / "1.Training" / "SourceData" / "2.Sleep" / "train_sleep.csv",
}
TRAIN_MMSE_PATH = (
    DATA_ROOT / "1.Training" / "SourceData" / "3.CognitiveFunction" / "train_mmse.csv"
)
VALIDATION_SOURCE_PATHS = {
    "activity": DATA_ROOT
    / "2.Validation"
    / "SourceData"
    / "1.Gait"
    / "val_activity.csv",
    "sleep": DATA_ROOT
    / "2.Validation"
    / "SourceData"
    / "2.Sleep"
    / "val_sleep.csv",
    "mmse": DATA_ROOT
    / "2.Validation"
    / "SourceData"
    / "3.CognitiveFunction"
    / "val_mmse.csv",
}

EXPECTED_ARTIFACTS = {
    "data_audit.json",
    "class_feature_summary.csv",
    "EDA_REPORT_KO.md",
}

CLASS_ID_TO_NAME = {0: "CN", 1: "MCI", 2: "DEM"}
LABEL_NORMALIZATION = {
    "CN": 0,
    "MCI": 1,
    "DEM": 2,
    "DEMENTIA": 2,
}

ACTIVITY_SEQUENCE_COLUMNS = {
    "activity_class_5min",
    "activity_met_1min",
    "CONVERT(activity_class_5min USING utf8)",
    "CONVERT(activity_met_1min USING utf8)",
}
SLEEP_SEQUENCE_COLUMNS = {
    "sleep_hr_5min",
    "sleep_hypnogram_5min",
    "sleep_rmssd_5min",
    "CONVERT(sleep_hr_5min USING utf8)",
    "CONVERT(sleep_hypnogram_5min USING utf8)",
    "CONVERT(sleep_rmssd_5min USING utf8)",
}


def relative_path(path: Path) -> str:
    """Return a stable project-relative path without exposing a local home path."""

    return str(path.relative_to(PROJECT_ROOT))


def normalize_identifier(series: pd.Series) -> pd.Series:
    """Normalize identifiers in memory; values are never returned in artifacts."""

    out = series.astype("string").str.strip()
    return out.mask(out.eq(""), pd.NA)


def normalize_label(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    key = str(value).strip().upper()
    return float(LABEL_NORMALIZATION[key]) if key in LABEL_NORMALIZATION else np.nan


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA:
        return None
    return value


def finite_array(values: Iterable[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def robust_summary(values: Iterable[Any]) -> dict[str, Any]:
    arr = finite_array(values)
    if arr.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "iqr": None}
    q25, median, q75 = np.quantile(arr, [0.25, 0.50, 0.75])
    return {
        "n": int(arr.size),
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def robust_scale(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if scale > 1e-12:
        return scale
    mad = float(np.median(np.abs(values - np.median(values))) * 1.4826)
    if mad > 1e-12:
        return mad
    std = float(np.std(values, ddof=1))
    return std if std > 1e-12 else np.nan


def cliffs_delta(values: np.ndarray, reference: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    reference = reference[np.isfinite(reference)]
    if values.size == 0 or reference.size == 0:
        return np.nan
    return float(np.sign(values[:, None] - reference[None, :]).mean())


def local_date(series: pd.Series) -> pd.Series:
    """Parse the calendar date as recorded, without UTC date shifting."""

    text = series.astype("string").str.slice(0, 10)
    return pd.to_datetime(text, format="%Y-%m-%d", errors="coerce")


def timestamp_hour(series: pd.Series) -> pd.Series:
    """Extract local wall-clock hour from an ISO-like timestamp string."""

    parts = series.astype("string").str.extract(
        r"(?:T|\s)(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
    )
    hour = pd.to_numeric(parts["hour"], errors="coerce")
    minute = pd.to_numeric(parts["minute"], errors="coerce")
    second = pd.to_numeric(parts["second"], errors="coerce").fillna(0.0)
    out = hour + minute / 60.0 + second / 3600.0
    return out.where(out.between(0.0, 24.0, inclusive="left"))


def timestamp_duration_hours(start: pd.Series, end: pd.Series) -> pd.Series:
    start_dt = pd.to_datetime(start, errors="coerce", utc=True)
    end_dt = pd.to_datetime(end, errors="coerce", utc=True)
    duration = (end_dt - start_dt).dt.total_seconds() / 3600.0
    return duration.where(duration.between(0.0, 48.0, inclusive="both"))


def parse_sequence(value: Any) -> np.ndarray:
    """Parse slash/comma/space separated numeric logs and remove -1 padding."""

    if pd.isna(value):
        return np.array([], dtype=float)
    text = str(value).strip()
    if not text or text == "...":
        return np.array([], dtype=float)
    cleaned = (
        text.replace("...", " ")
        .replace("/", " ")
        .replace(",", " ")
        .replace("[", " ")
        .replace("]", " ")
    )
    arr = np.fromstring(cleaned, sep=" ", dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr[arr != -1]


def run_lengths(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    boundaries = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
    lengths = np.diff(boundaries)
    labels = values[boundaries[:-1]]
    return labels, lengths


def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts[counts > 0]
    if counts.size == 0:
        return np.nan
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def categorical_sequence_features(
    arr: np.ndarray,
    labels: list[int],
    expected_length: int | None,
    kind: str,
) -> dict[str, float]:
    out: dict[str, float] = {"valid_count": float(arr.size)}
    out["valid_ratio"] = (
        float(min(arr.size / expected_length, 1.0)) if expected_length else np.nan
    )
    if arr.size == 0:
        for label in labels:
            out[f"class_{label}_ratio"] = np.nan
        out.update(
            {
                "unknown_ratio": np.nan,
                "transition_rate": np.nan,
                "entropy": np.nan,
                "bout_count": np.nan,
            }
        )
        return out

    rounded = np.rint(arr).astype(int)
    counts = np.asarray([(rounded == label).sum() for label in labels], dtype=float)
    for label, count in zip(labels, counts):
        out[f"class_{label}_ratio"] = float(count / rounded.size)
    out["unknown_ratio"] = float((~np.isin(rounded, labels)).mean())
    out["transition_rate"] = (
        float(np.mean(rounded[1:] != rounded[:-1])) if rounded.size > 1 else 0.0
    )
    out["entropy"] = entropy_from_counts(counts)
    run_labels, lengths = run_lengths(rounded)
    out["bout_count"] = float(lengths.size)

    if kind == "activity_class":
        for state, name in [(0, "nonwear"), (1, "rest"), (2, "inactive")]:
            state_lengths = lengths[run_labels == state]
            out[f"longest_{name}_run"] = (
                float(state_lengths.max()) if state_lengths.size else 0.0
            )
        active_idx = np.flatnonzero(rounded >= 3)
        out["active_time_centroid"] = (
            float(active_idx.mean() / max(rounded.size - 1, 1))
            if active_idx.size
            else np.nan
        )
        midpoint = rounded.size // 2
        out["active_ratio_first_half"] = float(np.mean(rounded[:midpoint] >= 3)) if midpoint else np.nan
        out["active_ratio_second_half"] = (
            float(np.mean(rounded[midpoint:] >= 3)) if midpoint < rounded.size else np.nan
        )
    elif kind == "sleep_hypnogram":
        for state, name in [(1, "deep"), (3, "rem"), (4, "awake")]:
            state_lengths = lengths[run_labels == state]
            out[f"longest_{name}_run"] = (
                float(state_lengths.max()) if state_lengths.size else 0.0
            )
            out[f"{name}_bout_count"] = float(state_lengths.size)
    return out


def numeric_sequence_features(
    arr: np.ndarray,
    expected_length: int | None,
    physiology: bool = False,
    met: bool = False,
) -> dict[str, float]:
    out: dict[str, float] = {"valid_count": float(arr.size)}
    out["valid_ratio"] = (
        float(min(arr.size / expected_length, 1.0)) if expected_length else np.nan
    )
    if arr.size == 0:
        for key in ["mean", "std", "median", "q10", "q25", "q75", "q90", "iqr", "zero_ratio"]:
            out[key] = np.nan
        return out

    q10, q25, median, q75, q90 = np.quantile(arr, [0.10, 0.25, 0.50, 0.75, 0.90])
    out.update(
        {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(median),
            "q10": float(q10),
            "q25": float(q25),
            "q75": float(q75),
            "q90": float(q90),
            "iqr": float(q75 - q25),
            "zero_ratio": float(np.mean(arr == 0)),
        }
    )
    if physiology:
        nonzero = arr[arr > 0]
        out["nonzero_count"] = float(nonzero.size)
        out["nonzero_ratio"] = float(nonzero.size / arr.size)
        if nonzero.size:
            nq10, nq25, nmedian, nq75, nq90 = np.quantile(
                nonzero, [0.10, 0.25, 0.50, 0.75, 0.90]
            )
            out.update(
                {
                    "nonzero_median": float(nmedian),
                    "nonzero_q10": float(nq10),
                    "nonzero_q90": float(nq90),
                    "nonzero_iqr": float(nq75 - nq25),
                }
            )
        else:
            for key in ["nonzero_median", "nonzero_q10", "nonzero_q90", "nonzero_iqr"]:
                out[key] = np.nan
    if met:
        out["ratio_ge_3"] = float(np.mean(arr >= 3.0))
        out["ratio_ge_6"] = float(np.mean(arr >= 6.0))
    return out


def choose_sequence_column(frame: pd.DataFrame, raw_name: str) -> str | None:
    converted = f"CONVERT({raw_name} USING utf8)"
    for column in [converted, raw_name]:
        if column in frame.columns:
            return column
    return None


def parse_sequence_frame(
    frame: pd.DataFrame,
    specs: list[dict[str, Any]],
    modality: str,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, str]]]:
    parsed_parts: list[pd.DataFrame] = []
    audit: dict[str, Any] = {}
    metadata: dict[str, dict[str, str]] = {}

    for spec in specs:
        column = choose_sequence_column(frame, spec["raw_name"])
        if column is None:
            audit[spec["prefix"]] = {"column_present": False}
            continue
        rows: list[dict[str, float]] = []
        lengths: list[int] = []
        empty = 0
        for value in frame[column].tolist():
            arr = parse_sequence(value)
            lengths.append(int(arr.size))
            empty += int(arr.size == 0)
            if spec["kind"] in {"activity_class", "sleep_hypnogram"}:
                feats = categorical_sequence_features(
                    arr,
                    labels=spec["labels"],
                    expected_length=spec.get("expected_length"),
                    kind=spec["kind"],
                )
            else:
                feats = numeric_sequence_features(
                    arr,
                    expected_length=spec.get("expected_length"),
                    physiology=bool(spec.get("physiology", False)),
                    met=bool(spec.get("met", False)),
                )
            rows.append({f"{spec['prefix']}__{key}": value for key, value in feats.items()})

        parsed = pd.DataFrame(rows, index=frame.index)
        parsed_parts.append(parsed)
        for parsed_column in parsed.columns:
            lower = parsed_column.lower()
            coverage_token = any(
                token in lower
                for token in [
                    "valid_count",
                    "valid_ratio",
                    "unknown_ratio",
                    "zero_ratio",
                    "nonzero_count",
                    "nonzero_ratio",
                ]
            )
            metadata[parsed_column] = {
                "modality": modality,
                "feature_group": spec["prefix"],
                "feature_type": "parsed_intraday",
                "analysis_role": "negative_control" if coverage_token else "candidate",
            }

        audit[spec["prefix"]] = {
            "column_present": True,
            "column_used": column,
            "rows": int(len(frame)),
            "empty_or_unparseable_rows": int(empty),
            "valid_length": robust_summary(lengths),
        }

    if not parsed_parts:
        return pd.DataFrame(index=frame.index), audit, metadata
    return pd.concat(parsed_parts, axis=1), audit, metadata


def load_training_labels() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: dict[str, pd.DataFrame] = {}
    copies_audit: dict[str, Any] = {}
    for name, path in TRAIN_LABEL_PATHS.items():
        raw = pd.read_csv(path)
        id_column = "SAMPLE_EMAIL" if "SAMPLE_EMAIL" in raw.columns else "EMAIL"
        if id_column not in raw.columns or "DIAG_NM" not in raw.columns:
            raise RuntimeError(f"Training label schema invalid for {name}; required-column count mismatch")
        normalized = pd.DataFrame(
            {
                "_id": normalize_identifier(raw[id_column]),
                "_class_id": raw["DIAG_NM"].map(normalize_label),
            }
        )
        conflict_count = int(
            normalized.dropna(subset=["_id"])
            .groupby("_id")["_class_id"]
            .nunique(dropna=False)
            .gt(1)
            .sum()
        )
        copies_audit[name] = {
            "path": relative_path(path),
            "rows": int(len(raw)),
            "columns": int(raw.shape[1]),
            "unique_subjects": int(normalized["_id"].nunique(dropna=True)),
            "missing_identifier_rows": int(normalized["_id"].isna().sum()),
            "unknown_or_missing_label_rows": int(normalized["_class_id"].isna().sum()),
            "duplicate_subject_rows_excess": int(normalized["_id"].duplicated().sum()),
            "within_copy_label_conflict_subjects": conflict_count,
        }
        if (
            copies_audit[name]["missing_identifier_rows"]
            or copies_audit[name]["unknown_or_missing_label_rows"]
            or conflict_count
        ):
            raise RuntimeError(f"Training label audit failed for {name}; see aggregate counts")
        frames[name] = normalized.drop_duplicates("_id").reset_index(drop=True)

    reference = frames["gait"].set_index("_id")["_class_id"].sort_index()
    comparisons: dict[str, Any] = {}
    all_sets_equal = True
    all_labels_equal = True
    for name, frame in frames.items():
        current = frame.set_index("_id")["_class_id"].sort_index()
        reference_ids = set(reference.index)
        current_ids = set(current.index)
        intersection = reference_ids & current_ids
        agreement = int(
            sum(float(reference.loc[item]) == float(current.loc[item]) for item in intersection)
        )
        subject_sets_equal = reference_ids == current_ids
        labels_equal = subject_sets_equal and agreement == len(reference)
        comparisons[name] = {
            "subject_sets_equal_to_gait": bool(subject_sets_equal),
            "label_values_equal_to_gait": bool(labels_equal),
            "gait_only_subject_count": int(len(reference_ids - current_ids)),
            "copy_only_subject_count": int(len(current_ids - reference_ids)),
            "overlap_subject_count": int(len(intersection)),
            "agreeing_label_count_on_overlap": agreement,
            "disagreeing_label_count_on_overlap": int(len(intersection) - agreement),
        }
        all_sets_equal &= subject_sets_equal
        all_labels_equal &= labels_equal

    if not all_sets_equal or not all_labels_equal:
        raise RuntimeError("Training label copies disagree; downstream EDA stopped without identifier output")

    master = reference.rename("_class_id").reset_index()
    master["_class_id"] = master["_class_id"].astype(int)
    class_counts = {
        CLASS_ID_TO_NAME[class_id]: int((master["_class_id"] == class_id).sum())
        for class_id in CLASS_ID_TO_NAME
    }
    audit = {
        "canonical_copy": "gait",
        "copies": copies_audit,
        "cross_copy_comparison": comparisons,
        "all_subject_sets_equal": bool(all_sets_equal),
        "all_label_values_equal": bool(all_labels_equal),
        "subject_count": int(len(master)),
        "class_counts": class_counts,
    }
    return master, audit


def top_column_rates(rates: pd.Series, n: int = 10) -> list[dict[str, Any]]:
    ordered = rates.sort_values(ascending=False).head(n)
    return [
        {"column": str(column), "fraction": float(value)}
        for column, value in ordered.items()
        if np.isfinite(value)
    ]


def numeric_scalar_frame(
    frame: pd.DataFrame,
    modality: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], dict[str, Any]]:
    excluded = set(ACTIVITY_SEQUENCE_COLUMNS if modality == "activity" else SLEEP_SEQUENCE_COLUMNS)
    excluded.update({"EMAIL", "SAMPLE_EMAIL", "_id", "_date", "_class_id"})
    excluded.update(
        {
            "activity_day_start",
            "activity_day_end",
            "sleep_bedtime_start",
            "sleep_bedtime_end",
            "sleep_is_longest",
        }
    )
    scalar = pd.DataFrame(index=frame.index)
    metadata: dict[str, dict[str, str]] = {}
    conversion_audit: dict[str, Any] = {}
    minimum_valid = max(5, int(math.ceil(len(frame) * 0.01)))

    for column in frame.columns:
        lower = column.lower()
        if column in excluded or lower.endswith("_id") or "period_id" in lower:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        original_nonmissing = int(frame[column].notna().sum())
        numeric_nonmissing = int(numeric.notna().sum())
        if numeric_nonmissing < minimum_valid:
            continue
        scalar[column] = numeric.astype(float)
        conversion_audit[column] = {
            "original_nonmissing": original_nonmissing,
            "numeric_nonmissing": numeric_nonmissing,
            "numeric_conversion_fraction": (
                float(numeric_nonmissing / original_nonmissing) if original_nonmissing else None
            ),
        }
        role = "negative_control" if "non_wear" in lower or "nonwear" in lower else "candidate"
        metadata[column] = {
            "modality": modality,
            "feature_group": "daily_scalar",
            "feature_type": "scalar",
            "analysis_role": role,
        }

    if modality == "activity":
        scalar["derived_day_start_hour"] = timestamp_hour(frame["activity_day_start"])
        scalar["derived_day_end_hour"] = timestamp_hour(frame["activity_day_end"])
        scalar["derived_record_duration_hours"] = timestamp_duration_hours(
            frame["activity_day_start"], frame["activity_day_end"]
        )
    else:
        scalar["derived_bedtime_start_hour"] = timestamp_hour(frame["sleep_bedtime_start"])
        scalar["derived_bedtime_end_hour"] = timestamp_hour(frame["sleep_bedtime_end"])
        scalar["derived_timestamp_sleep_duration_hours"] = timestamp_duration_hours(
            frame["sleep_bedtime_start"], frame["sleep_bedtime_end"]
        )

    for column in scalar.columns:
        if column.startswith("derived_"):
            metadata[column] = {
                "modality": modality,
                "feature_group": "timestamp_derived",
                "feature_type": "scalar",
                "analysis_role": "candidate",
            }
    return scalar, metadata, conversion_audit


def prepare_daily_source(frame: pd.DataFrame, modality: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    if modality == "activity":
        out["_date"] = local_date(out["activity_day_start"])
        duration = timestamp_duration_hours(out["activity_day_start"], out["activity_day_end"])
        priority = duration.fillna(-np.inf)
    else:
        out["_date"] = local_date(out["sleep_bedtime_end"])
        duration = pd.to_numeric(out.get("sleep_duration"), errors="coerce")
        longest = pd.to_numeric(out.get("sleep_is_longest"), errors="coerce").fillna(0.0)
        priority = longest * 1e12 + duration.fillna(-np.inf)

    valid_date = out["_date"].notna()
    duplicate_excess = int(out.loc[valid_date].duplicated(["_id", "_date"]).sum())
    duplicate_groups = int(
        out.loc[valid_date].groupby(["_id", "_date"], dropna=False).size().gt(1).sum()
    )
    dated = out.loc[valid_date].assign(_priority=priority.loc[valid_date])
    dated = (
        dated.sort_values(["_id", "_date", "_priority"], ascending=[True, True, False])
        .drop_duplicates(["_id", "_date"], keep="first")
        .drop(columns="_priority")
    )
    undated = out.loc[~valid_date]
    daily = pd.concat([dated, undated], axis=0).sort_index()
    audit = {
        "duplicate_subject_date_rows_excess": duplicate_excess,
        "duplicate_subject_date_groups": duplicate_groups,
        "daily_representative_policy": (
            "longest-flag then duration" if modality == "sleep" else "longest timestamp duration"
        ),
        "rows_after_daily_representative_selection": int(len(daily)),
    }
    return daily, audit


def aggregate_daily_features(
    daily: pd.DataFrame,
    daily_features: pd.DataFrame,
    base_metadata: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    if daily_features.empty:
        return pd.DataFrame(), {}
    values = daily_features.copy()
    values.insert(0, "_id", daily["_id"].to_numpy())
    feature_columns = [column for column in values.columns if column != "_id"]
    grouped = values.groupby("_id", sort=False, dropna=True)
    median = grouped[feature_columns].median()
    q25 = grouped[feature_columns].quantile(0.25)
    q75 = grouped[feature_columns].quantile(0.75)
    day_iqr = q75 - q25
    counts = grouped[feature_columns].count()
    sizes = grouped.size()
    missing_fraction = 1.0 - counts.div(sizes, axis=0)

    outputs: list[pd.DataFrame] = []
    metadata: dict[str, dict[str, str]] = {}
    for aggregation, frame in [
        ("subject_median", median),
        ("subject_day_iqr", day_iqr),
        ("subject_missing_fraction", missing_fraction),
    ]:
        renamed = frame.copy()
        renamed.columns = [f"{base_metadata[column]['modality']}__{column}__{aggregation}" for column in frame.columns]
        outputs.append(renamed)
        for source_column, output_column in zip(frame.columns, renamed.columns):
            base = base_metadata[source_column]
            role = (
                "negative_control"
                if aggregation == "subject_missing_fraction"
                else base["analysis_role"]
            )
            metadata[output_column] = {
                **base,
                "aggregation": aggregation,
                "analysis_role": role,
            }
    return pd.concat(outputs, axis=1), metadata


def source_coverage_frame(
    raw: pd.DataFrame,
    scalar: pd.DataFrame,
    modality: str,
    master_index: pd.Index,
) -> pd.DataFrame:
    work = pd.DataFrame({"_id": raw["_id"], "_date": raw["_date"]})
    work["_row_scalar_missing_fraction"] = scalar.isna().mean(axis=1) if not scalar.empty else np.nan
    grouped = work.groupby("_id", sort=False, dropna=True)
    rows = grouped.size().rename(f"protocol_{modality}_row_count")
    days = grouped["_date"].nunique().rename(f"protocol_{modality}_valid_days")
    first = grouped["_date"].min()
    last = grouped["_date"].max()
    span = (last - first).dt.days.add(1).rename(f"protocol_{modality}_span_days")
    coverage = (days / span).rename(f"protocol_{modality}_date_coverage_ratio")
    duplicates = (rows - days).clip(lower=0).rename(f"protocol_{modality}_duplicate_day_rows")
    row_per_day = (rows / days.replace(0, np.nan)).rename(f"protocol_{modality}_rows_per_valid_day")
    missing = grouped["_row_scalar_missing_fraction"].mean().rename(
        f"protocol_{modality}_mean_scalar_missing_fraction"
    )

    valid_first = first.dropna()
    origin = valid_first.min() if not valid_first.empty else pd.NaT
    if pd.isna(origin):
        first_ordinal = pd.Series(np.nan, index=first.index, name=f"protocol_{modality}_first_date_offset_days")
    else:
        first_ordinal = (first - origin).dt.days.rename(f"protocol_{modality}_first_date_offset_days")

    result = pd.concat(
        [rows, days, span, coverage, duplicates, row_per_day, missing, first_ordinal],
        axis=1,
    )
    return result.reindex(master_index)


def cross_modality_coverage(
    activity: pd.DataFrame,
    sleep: pd.DataFrame,
    master_index: pd.Index,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    activity_keys = activity.loc[activity["_date"].notna(), ["_id", "_date"]].drop_duplicates()
    sleep_keys = sleep.loc[sleep["_date"].notna(), ["_id", "_date"]].drop_duplicates()
    merged = activity_keys.merge(sleep_keys, on=["_id", "_date"], how="outer", indicator=True)
    counts = merged.groupby(["_id", "_merge"], observed=False).size().unstack(fill_value=0)
    for category in ["both", "left_only", "right_only"]:
        if category not in counts.columns:
            counts[category] = 0
    matched = counts["both"].astype(float)
    activity_only = counts["left_only"].astype(float)
    sleep_only = counts["right_only"].astype(float)
    union = matched + activity_only + sleep_only
    result = pd.DataFrame(
        {
            "protocol_matched_activity_sleep_days": matched,
            "protocol_activity_only_days": activity_only,
            "protocol_sleep_only_days": sleep_only,
            "protocol_activity_sleep_union_days": union,
            "protocol_matched_day_ratio": matched / union.replace(0, np.nan),
            "protocol_sleep_match_ratio": matched / (matched + sleep_only).replace(0, np.nan),
            "protocol_activity_match_ratio": matched / (matched + activity_only).replace(0, np.nan),
        }
    ).reindex(master_index)
    audit = {
        "activity_unique_subject_dates": int(len(activity_keys)),
        "sleep_unique_subject_dates": int(len(sleep_keys)),
        "matched_subject_dates": int((merged["_merge"] == "both").sum()),
        "activity_only_subject_dates": int((merged["_merge"] == "left_only").sum()),
        "sleep_only_subject_dates": int((merged["_merge"] == "right_only").sum()),
        "union_subject_dates": int(len(merged)),
    }
    return result, audit


def training_source_audit(
    raw: pd.DataFrame,
    daily: pd.DataFrame,
    scalar: pd.DataFrame,
    modality: str,
    master_ids: set[str],
    daily_policy: dict[str, Any],
    sequence_audit: dict[str, Any],
) -> dict[str, Any]:
    source_ids = set(raw["_id"].dropna().astype(str))
    non_identifier = [
        column
        for column in raw.columns
        if column not in {"EMAIL", "SAMPLE_EMAIL"} and not column.startswith("_")
    ]
    missing_rates = raw[non_identifier].isna().mean()
    zero_rates = pd.Series(dtype=float)
    if not scalar.empty:
        zero_rates = scalar.eq(0).mean()
    valid_dates = raw["_date"].dropna()
    return {
        "path": relative_path(TRAIN_SOURCE_PATHS[modality]),
        "rows": int(len(raw)),
        "columns": int(len([column for column in raw.columns if not column.startswith("_")])),
        "unique_subjects": int(len(source_ids)),
        "subjects_with_training_label": int(len(source_ids & master_ids)),
        "source_only_subject_count": int(len(source_ids - master_ids)),
        "label_only_subject_count": int(len(master_ids - source_ids)),
        "rows_without_training_label": int(raw["_class_id"].isna().sum()),
        "exact_duplicate_rows_excess": int(
            raw.drop(columns=[column for column in raw.columns if column.startswith("_")], errors="ignore")
            .duplicated()
            .sum()
        ),
        "date": {
            "valid_rows": int(valid_dates.size),
            "missing_rows": int(raw["_date"].isna().sum()),
            "min": None if valid_dates.empty else str(valid_dates.min().date()),
            "max": None if valid_dates.empty else str(valid_dates.max().date()),
        },
        "daily_selection": daily_policy,
        "missingness": {
            "overall_raw_cell_missing_fraction_excluding_identifier": float(
                raw[non_identifier].isna().to_numpy().mean()
            ),
            "columns_with_any_raw_missing": int(missing_rates.gt(0).sum()),
            "top_raw_missing_columns": top_column_rates(missing_rates),
            "top_numeric_zero_fraction_columns": top_column_rates(zero_rates),
        },
        "scalar_numeric_feature_count": int(scalar.shape[1]),
        "parsed_sequence_audit": sequence_audit,
        "rows_used_for_daily_feature_aggregation": int(len(daily)),
    }


def validation_source_schema_overlap(
    path: Path,
    modality: str,
    training_columns: list[str] | None,
    training_source_ids: set[str],
    training_label_ids: set[str],
) -> tuple[dict[str, Any], set[str]]:
    """Read only a Validation header and its identifier column."""

    header = pd.read_csv(path, nrows=0)
    id_column = "SAMPLE_EMAIL" if "SAMPLE_EMAIL" in header.columns else "EMAIL"
    if id_column not in header.columns:
        raise RuntimeError(f"Validation {modality} source lacks an identifier column")
    id_only = pd.read_csv(path, usecols=[id_column])
    identifiers = set(normalize_identifier(id_only[id_column]).dropna().astype(str))
    columns = list(header.columns)
    target_like = [
        column
        for column in columns
        if any(token in column.upper() for token in ["DIAG", "LABEL", "TARGET", "DOCTOR"])
    ]
    audit = {
        "path": relative_path(path),
        "rows_counted_from_identifier_column_only": int(len(id_only)),
        "columns": columns,
        "column_count": int(len(columns)),
        "identifier_column_read_for_overlap_only": id_column,
        "non_identifier_feature_values_opened": False,
        "unique_subjects": int(len(identifiers)),
        "overlap_with_training_source_subject_count": int(len(identifiers & training_source_ids)),
        "overlap_with_training_label_subject_count": int(len(identifiers & training_label_ids)),
        "target_like_columns_present_in_schema": target_like,
        "schema_exact_match_to_training_source": (
            None if training_columns is None else bool(columns == training_columns)
        ),
        "training_only_columns": (
            [] if training_columns is None else sorted(set(training_columns) - set(columns))
        ),
        "validation_only_columns": (
            [] if training_columns is None else sorted(set(columns) - set(training_columns))
        ),
    }
    return audit, identifiers


def build_class_feature_summary(
    subject_features: pd.DataFrame,
    labels: pd.Series,
    metadata: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in subject_features.columns:
        all_values = pd.to_numeric(subject_features[feature], errors="coerce")
        reference = all_values[labels == 0].to_numpy(dtype=float)
        reference = reference[np.isfinite(reference)]
        reference_median = float(np.median(reference)) if reference.size else np.nan
        reference_scale = robust_scale(reference)
        feature_meta = metadata[feature]
        for class_id, class_name in CLASS_ID_TO_NAME.items():
            class_mask = labels == class_id
            class_total = int(class_mask.sum())
            values = all_values[class_mask].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75])
                class_scale = robust_scale(values)
                finite_scales = np.asarray(
                    [reference_scale, class_scale], dtype=float
                )
                finite_scales = finite_scales[np.isfinite(finite_scales)]
                pooled_scale = (
                    float(np.sqrt(np.mean(np.square(finite_scales))))
                    if finite_scales.size
                    else np.nan
                )
                median_diff = float(median - reference_median) if np.isfinite(reference_median) else np.nan
                robust_smd = (
                    float(median_diff / pooled_scale)
                    if np.isfinite(pooled_scale) and pooled_scale > 1e-12
                    else np.nan
                )
                delta = 0.0 if class_id == 0 else cliffs_delta(values, reference)
            else:
                q25 = median = q75 = median_diff = robust_smd = delta = np.nan
            rows.append(
                {
                    "feature": feature,
                    "modality": feature_meta["modality"],
                    "feature_group": feature_meta["feature_group"],
                    "feature_type": feature_meta["feature_type"],
                    "aggregation": feature_meta["aggregation"],
                    "analysis_role": feature_meta["analysis_role"],
                    "class_id": class_id,
                    "class_name": class_name,
                    "n_subjects_total": class_total,
                    "n_nonmissing": int(values.size),
                    "missing_subject_fraction": (
                        float(1.0 - values.size / class_total) if class_total else np.nan
                    ),
                    "median": float(median) if np.isfinite(median) else np.nan,
                    "q25": float(q25) if np.isfinite(q25) else np.nan,
                    "q75": float(q75) if np.isfinite(q75) else np.nan,
                    "iqr": float(q75 - q25) if np.isfinite(q75) and np.isfinite(q25) else np.nan,
                    "cn_reference_median": reference_median,
                    "median_diff_vs_cn": median_diff,
                    "robust_smd_vs_cn": robust_smd,
                    "cliffs_delta_vs_cn": delta,
                    "abs_cliffs_delta_vs_cn": abs(delta) if np.isfinite(delta) else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    non_cn_max = (
        summary.loc[summary["class_id"].isin([1, 2])]
        .groupby("feature")["abs_cliffs_delta_vs_cn"]
        .max()
    )
    summary["max_abs_cliffs_delta_non_cn"] = summary["feature"].map(non_cn_max)
    return summary.sort_values(["feature", "class_id"]).reset_index(drop=True)


def class_protocol_summary(
    protocol: pd.DataFrame,
    labels: pd.Series,
    selected_features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for class_id, class_name in CLASS_ID_TO_NAME.items():
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "subjects": int((labels == class_id).sum()),
        }
        for feature in selected_features:
            values = protocol.loc[labels == class_id, feature]
            summary = robust_summary(values)
            row[feature] = summary
        rows.append(row)
    return rows


def top_effects(
    summary: pd.DataFrame,
    class_id: int,
    role: str,
    n: int,
) -> pd.DataFrame:
    class_size = int(summary.loc[summary["class_id"] == class_id, "n_subjects_total"].max())
    minimum_nonmissing = max(5, int(math.ceil(class_size * 0.50)))
    subset = summary[
        (summary["class_id"] == class_id)
        & (summary["analysis_role"] == role)
        & (summary["n_nonmissing"] >= minimum_nonmissing)
        & summary["cliffs_delta_vs_cn"].notna()
    ].copy()
    return subset.sort_values("abs_cliffs_delta_vs_cn", ascending=False).head(n)


def negative_control_diagnostics(summary: pd.DataFrame) -> dict[str, Any]:
    subset = summary[
        summary["class_id"].isin([1, 2])
        & summary["analysis_role"].eq("negative_control")
        & summary["cliffs_delta_vs_cn"].notna()
    ].copy()
    strongest = (
        subset.sort_values("abs_cliffs_delta_vs_cn", ascending=False)
        .drop_duplicates("feature")
        .head(12)
    )
    return {
        "interpretation": (
            "Coverage, missingness, valid-length, duplicate-day, and collection-date effects are "
            "negative controls. They can reveal protocol leakage and are not recommended model "
            "features without a pre-specified stability justification."
        ),
        "moderate_or_larger_feature_count_abs_cliffs_delta_ge_0_33": int(
            subset.loc[subset["abs_cliffs_delta_vs_cn"] >= 0.33, "feature"].nunique()
        ),
        "strong_feature_count_abs_cliffs_delta_ge_0_474": int(
            subset.loc[subset["abs_cliffs_delta_vs_cn"] >= 0.474, "feature"].nunique()
        ),
        "strongest": [
            {
                "feature": row.feature,
                "class_name": row.class_name,
                "cliffs_delta_vs_cn": float(row.cliffs_delta_vs_cn),
                "n_nonmissing": int(row.n_nonmissing),
            }
            for row in strongest.itertuples(index=False)
        ],
    }


def format_number(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):.{digits}f}"


def effects_markdown(title: str, frame: pd.DataFrame) -> list[str]:
    lines = [f"### {title}", "", "| feature | class median | CN median | Cliff's delta | robust SMD |", "|---|---:|---:|---:|---:|"]
    if frame.empty:
        lines.append("| 분석 가능한 feature 없음 | NA | NA | NA | NA |")
        return lines
    for row in frame.itertuples(index=False):
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                row.feature,
                format_number(row.median),
                format_number(row.cn_reference_median),
                format_number(row.cliffs_delta_vs_cn),
                format_number(row.robust_smd_vs_cn),
            )
        )
    return lines


def build_report(
    audit: dict[str, Any],
    feature_summary: pd.DataFrame,
) -> str:
    class_counts = audit["training_labels"]["class_counts"]
    activity = audit["training_sources"]["activity"]
    sleep = audit["training_sources"]["sleep"]
    cross = audit["training_cross_modality"]
    protocol = audit["negative_control_diagnostics"]
    coverage_rows = audit["class_coverage_summary"]

    lines = [
        "# Three-Class Performance Lab - Local EDA Report",
        "",
        "이 보고서는 Training 자료만으로 계산한 식별자 비노출 집계입니다. Validation 라벨은 열지 않았고, Validation 원천자료는 헤더와 subject-overlap 확인용 식별자 열만 읽었습니다.",
        "",
        "## 핵심 결과",
        "",
        f"- Training subject: CN {class_counts['CN']}명, MCI {class_counts['MCI']}명, DEM {class_counts['DEM']}명.",
        f"- 세 Training 라벨 사본의 subject 집합과 정규화 라벨은 모두 일치: `{audit['training_labels']['all_label_values_equal']}`.",
        f"- Activity {activity['rows']:,}행/{activity['unique_subjects']}명, Sleep {sleep['rows']:,}행/{sleep['unique_subjects']}명.",
        f"- 날짜 기준 activity-sleep matched {cross['matched_subject_dates']:,}건, activity-only {cross['activity_only_subject_dates']:,}건, sleep-only {cross['sleep_only_subject_dates']:,}건.",
        f"- Protocol/coverage negative control 중 |Cliff's delta| >= 0.33인 feature: {protocol['moderate_or_larger_feature_count_abs_cliffs_delta_ge_0_33']}개. 이 변수들은 성능 후보가 아니라 누수 경고로 해석해야 합니다.",
        "",
        "## 데이터 계약 및 격리",
        "",
        "- Target mapping: CN=0, MCI=1, DEM/Dementia=2.",
        "- Activity 날짜는 `activity_day_start`, sleep 날짜는 수면 종료일(`sleep_bedtime_end`)을 사용했습니다.",
        "- 이 설명용 EDA 집계에서는 같은 subject-date의 sleep을 `sleep_is_longest`와 duration 우선순위로 1건화했습니다. 최종 discovery pipeline은 장비 flag를 feature/선택에 쓰지 않고, prediction index 이전 episode만 남긴 뒤 duration, bedtime start/end의 고정 시간 규칙으로 main sleep을 다시 선택합니다. 원본 중복 수는 별도 audit에만 둡니다.",
        "- 모든 효과크기는 일별 행이 아니라 subject median/IQR로 계산했습니다.",
        "- Validation 라벨은 읽지 않았으며 모델/feature 선택에 사용할 수 없습니다.",
        "",
        "## Modality audit",
        "",
        "| modality | rows | subjects | date range | missing date rows | duplicate subject-date rows | raw missing cell fraction |",
        "|---|---:|---:|---|---:|---:|---:|",
        "| activity | {} | {} | {} - {} | {} | {} | {} |".format(
            format_number(activity["rows"]),
            activity["unique_subjects"],
            activity["date"]["min"],
            activity["date"]["max"],
            activity["date"]["missing_rows"],
            activity["daily_selection"]["duplicate_subject_date_rows_excess"],
            format_number(activity["missingness"]["overall_raw_cell_missing_fraction_excluding_identifier"], 4),
        ),
        "| sleep | {} | {} | {} - {} | {} | {} | {} |".format(
            format_number(sleep["rows"]),
            sleep["unique_subjects"],
            sleep["date"]["min"],
            sleep["date"]["max"],
            sleep["date"]["missing_rows"],
            sleep["daily_selection"]["duplicate_subject_date_rows_excess"],
            format_number(sleep["missingness"]["overall_raw_cell_missing_fraction_excluding_identifier"], 4),
        ),
        "",
        "## Class별 coverage",
        "",
        "| class | activity valid days median [IQR] | sleep valid days median [IQR] | matched-day ratio median [IQR] |",
        "|---|---:|---:|---:|",
    ]
    for row in coverage_rows:
        activity_days = row["protocol_activity_valid_days"]
        sleep_days = row["protocol_sleep_valid_days"]
        matched = row["protocol_matched_day_ratio"]
        lines.append(
            "| {} | {} [{}] | {} [{}] | {} [{}] |".format(
                row["class_name"],
                format_number(activity_days["median"]),
                format_number(activity_days["iqr"]),
                format_number(sleep_days["median"]),
                format_number(sleep_days["iqr"]),
                format_number(matched["median"]),
                format_number(matched["iqr"]),
            )
        )

    lines.extend(["", "## 가장 큰 training-only class 효과", ""])
    lines.extend(effects_markdown("MCI vs CN", top_effects(feature_summary, 1, "candidate", 12)))
    lines.extend([""])
    lines.extend(effects_markdown("DEM vs CN", top_effects(feature_summary, 2, "candidate", 12)))

    lines.extend(
        [
            "",
            "## Coverage/protocol negative controls",
            "",
            "아래 항목은 collection length, 결측, valid sequence length, duplicate-day, first-date 같은 관리·프로토콜 변수입니다. 큰 효과가 있더라도 질병 신호로 간주하거나 바로 모델에 투입하면 안 됩니다.",
            "",
            "| feature | strongest class | Cliff's delta vs CN | n |",
            "|---|---|---:|---:|",
        ]
    )
    for item in protocol["strongest"]:
        lines.append(
            "| `{}` | {} | {} | {} |".format(
                item["feature"],
                item["class_name"],
                format_number(item["cliffs_delta_vs_cn"]),
                item["n_nonmissing"],
            )
        )

    lines.extend(
        [
            "",
            "## MMSE exclusion",
            "",
            "Training/Validation MMSE 값은 feature EDA에서 제외했습니다. MMSE 원천에는 `DIAG_NM`, 임상의/검사 메타데이터 및 진단과 매우 가까운 인지검사 점수가 함께 있어, prediction index와 진단 생성 절차가 확정되기 전 사용하면 target leakage 또는 임상적 순환성이 생길 수 있습니다. Validation MMSE에서는 schema와 overlap 확인을 위한 식별자 열 외 값을 읽지 않았습니다.",
            "",
            "## 모델링 시사점",
            "",
            "1. 완전관측 조건으로 subject를 버리지 않고 141명 모두를 유지하되, primary 입력은 최근 observed event의 생체 요약(`event_summary_v1`)과 coverage 비노출 sequence(`event_sequence28_v1`)로 제한합니다.",
            "2. EDA 효과크기는 해석·감사용이며 이번 run의 고정 4개 primary 후보, feature 집합, hyperparameter를 추가하거나 바꾸는 supervised 선택 근거로 사용하지 않습니다.",
            "3. Observed count, valid length/ratio, calendar gap, missing fraction, padding mask는 primary 입력에서 제외하고 `coverage_only_v1` negative control 또는 고정 `mask_tcn_35d_legacy_v1` comparator에서만 사용합니다.",
            "4. Parsed intraday HR/RMSSD, sleep-stage/activity-state 비율·전이·엔트로피는 허용된 생체 후보로 유지하되, Validation benchmark를 이용한 feature 제거·threshold·ensemble 선택은 금지합니다.",
            "",
            "## 한계",
            "",
            "- DEM Training subject가 9명뿐이므로 큰 효과도 불확실합니다. 효과크기는 탐색적 연관성이지 통계적·임상적 인과가 아닙니다.",
            "- 데이터 수집 프로토콜과 diagnosis timing 정보가 제한되어 chronology 가정은 최종 파이프라인에서 다시 검증해야 합니다.",
            "- `class_feature_summary.csv`는 class별 aggregate만 포함하며 subject-level 행은 저장하지 않습니다.",
            "",
        ]
    )
    return "\n".join(lines)


def assert_privacy_payload(payloads: list[str], identifiers: set[str]) -> None:
    """Fail before writing if any in-memory identifier appears in an artifact."""

    combined = "\n".join(payloads)
    leak_count = sum(bool(identifier) and identifier in combined for identifier in identifiers)
    if leak_count:
        raise RuntimeError(
            f"Privacy guard rejected aggregate artifacts; identifier occurrence count={leak_count}"
        )


def main() -> None:
    for path in [*TRAIN_LABEL_PATHS.values(), *TRAIN_SOURCE_PATHS.values(), *VALIDATION_SOURCE_PATHS.values()]:
        if not path.is_file():
            raise FileNotFoundError(f"Required project input missing: {relative_path(path)}")

    master, label_audit = load_training_labels()
    master_ids = set(master["_id"].astype(str))
    label_series = master.set_index("_id")["_class_id"].astype(int)
    master_index = label_series.index

    activity_raw = pd.read_csv(TRAIN_SOURCE_PATHS["activity"])
    sleep_raw = pd.read_csv(TRAIN_SOURCE_PATHS["sleep"])
    activity_training_columns = list(activity_raw.columns)
    sleep_training_columns = list(sleep_raw.columns)
    for frame, id_column in [(activity_raw, "EMAIL"), (sleep_raw, "EMAIL")]:
        frame["_id"] = normalize_identifier(frame[id_column])
        frame["_class_id"] = frame["_id"].map(label_series)

    activity_raw["_date"] = local_date(activity_raw["activity_day_start"])
    sleep_raw["_date"] = local_date(sleep_raw["sleep_bedtime_end"])
    activity_daily, activity_daily_policy = prepare_daily_source(activity_raw, "activity")
    sleep_daily, sleep_daily_policy = prepare_daily_source(sleep_raw, "sleep")

    activity_scalar_raw, _, _ = numeric_scalar_frame(activity_raw, "activity")
    sleep_scalar_raw, _, _ = numeric_scalar_frame(sleep_raw, "sleep")
    activity_scalar, activity_scalar_metadata, activity_conversion = numeric_scalar_frame(
        activity_daily, "activity"
    )
    sleep_scalar, sleep_scalar_metadata, sleep_conversion = numeric_scalar_frame(
        sleep_daily, "sleep"
    )

    activity_sequence_specs = [
        {
            "raw_name": "activity_class_5min",
            "prefix": "activity_class_5min",
            "kind": "activity_class",
            "labels": [0, 1, 2, 3, 4, 5],
            "expected_length": 288,
        },
        {
            "raw_name": "activity_met_1min",
            "prefix": "activity_met_1min",
            "kind": "numeric",
            "expected_length": 1440,
            "met": True,
        },
    ]
    sleep_sequence_specs = [
        {
            "raw_name": "sleep_hypnogram_5min",
            "prefix": "sleep_hypnogram_5min",
            "kind": "sleep_hypnogram",
            "labels": [1, 2, 3, 4],
        },
        {
            "raw_name": "sleep_hr_5min",
            "prefix": "sleep_hr_5min",
            "kind": "numeric",
            "physiology": True,
        },
        {
            "raw_name": "sleep_rmssd_5min",
            "prefix": "sleep_rmssd_5min",
            "kind": "numeric",
            "physiology": True,
        },
    ]
    activity_parsed, activity_sequence_audit, activity_parsed_metadata = parse_sequence_frame(
        activity_daily, activity_sequence_specs, "activity"
    )
    sleep_parsed, sleep_sequence_audit, sleep_parsed_metadata = parse_sequence_frame(
        sleep_daily, sleep_sequence_specs, "sleep"
    )

    activity_daily_features = pd.concat([activity_scalar, activity_parsed], axis=1)
    sleep_daily_features = pd.concat([sleep_scalar, sleep_parsed], axis=1)
    activity_base_metadata = {**activity_scalar_metadata, **activity_parsed_metadata}
    sleep_base_metadata = {**sleep_scalar_metadata, **sleep_parsed_metadata}

    activity_subject, activity_subject_metadata = aggregate_daily_features(
        activity_daily, activity_daily_features, activity_base_metadata
    )
    sleep_subject, sleep_subject_metadata = aggregate_daily_features(
        sleep_daily, sleep_daily_features, sleep_base_metadata
    )

    activity_protocol = source_coverage_frame(
        activity_raw, activity_scalar_raw, "activity", master_index
    )
    sleep_protocol = source_coverage_frame(sleep_raw, sleep_scalar_raw, "sleep", master_index)
    cross_protocol, cross_audit = cross_modality_coverage(
        activity_raw, sleep_raw, master_index
    )
    protocol = pd.concat([activity_protocol, sleep_protocol, cross_protocol], axis=1)
    protocol_metadata = {
        column: {
            "modality": "protocol",
            "feature_group": "coverage_protocol_negative_control",
            "feature_type": "protocol",
            "aggregation": "subject_level",
            "analysis_role": "negative_control",
        }
        for column in protocol.columns
    }

    subject_features = pd.concat(
        [
            activity_subject.reindex(master_index),
            sleep_subject.reindex(master_index),
            protocol.reindex(master_index),
        ],
        axis=1,
    )
    subject_metadata = {
        **activity_subject_metadata,
        **sleep_subject_metadata,
        **protocol_metadata,
    }
    if subject_features.columns.duplicated().any():
        raise RuntimeError("Engineered feature names are not unique")

    feature_summary = build_class_feature_summary(
        subject_features, label_series.reindex(master_index), subject_metadata
    )

    activity_ids = set(activity_raw["_id"].dropna().astype(str))
    sleep_ids = set(sleep_raw["_id"].dropna().astype(str))
    activity_audit = training_source_audit(
        activity_raw,
        activity_daily,
        activity_scalar,
        "activity",
        master_ids,
        activity_daily_policy,
        activity_sequence_audit,
    )
    sleep_audit = training_source_audit(
        sleep_raw,
        sleep_daily,
        sleep_scalar,
        "sleep",
        master_ids,
        sleep_daily_policy,
        sleep_sequence_audit,
    )
    activity_audit["numeric_conversion_audit"] = activity_conversion
    sleep_audit["numeric_conversion_audit"] = sleep_conversion

    validation_audits: dict[str, Any] = {}
    validation_identifiers: set[str] = set()
    for modality, path in VALIDATION_SOURCE_PATHS.items():
        if modality == "activity":
            train_columns = activity_training_columns
            train_ids = activity_ids
        elif modality == "sleep":
            train_columns = sleep_training_columns
            train_ids = sleep_ids
        else:
            train_columns = None
            train_ids = master_ids
        validation_audit, identifiers = validation_source_schema_overlap(
            path,
            modality,
            train_columns,
            train_ids,
            master_ids,
        )
        validation_audits[modality] = validation_audit
        validation_identifiers |= identifiers

    selected_protocol_features = [
        "protocol_activity_valid_days",
        "protocol_sleep_valid_days",
        "protocol_activity_span_days",
        "protocol_sleep_span_days",
        "protocol_matched_day_ratio",
        "protocol_activity_mean_scalar_missing_fraction",
        "protocol_sleep_mean_scalar_missing_fraction",
    ]
    coverage_summary = class_protocol_summary(
        protocol,
        label_series.reindex(master_index),
        selected_protocol_features,
    )
    negative_controls = negative_control_diagnostics(feature_summary)

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "three_class_cognitive_status_training_only_eda",
        "class_mapping": {"0": "CN", "1": "MCI", "2": "DEM"},
        "privacy_contract": {
            "identifier_values_printed": False,
            "identifier_values_persisted": False,
            "identifier_hashes_created_or_persisted": False,
            "subject_level_rows_persisted": False,
            "artifacts_are_aggregate_only": True,
        },
        "validation_isolation": {
            "validation_label_files_opened": False,
            "validation_label_values_used": False,
            "validation_source_non_identifier_values_opened": False,
            "validation_source_access_scope": "header plus identifier column for schema/overlap only",
        },
        "training_labels": label_audit,
        "training_sources": {
            "activity": activity_audit,
            "sleep": sleep_audit,
        },
        "training_cross_modality": {
            **cross_audit,
            "activity_subject_count": int(len(activity_ids)),
            "sleep_subject_count": int(len(sleep_ids)),
            "activity_sleep_subject_overlap_count": int(len(activity_ids & sleep_ids)),
            "activity_only_subject_count": int(len(activity_ids - sleep_ids)),
            "sleep_only_subject_count": int(len(sleep_ids - activity_ids)),
        },
        "validation_source_schema_and_overlap_only": validation_audits,
        "class_coverage_summary": coverage_summary,
        "negative_control_diagnostics": negative_controls,
        "feature_summary": {
            "engineered_subject_feature_count": int(subject_features.shape[1]),
            "aggregate_rows_in_class_feature_summary": int(len(feature_summary)),
            "candidate_feature_count": int(
                feature_summary.loc[feature_summary["analysis_role"] == "candidate", "feature"].nunique()
            ),
            "negative_control_feature_count": int(
                feature_summary.loc[
                    feature_summary["analysis_role"] == "negative_control", "feature"
                ].nunique()
            ),
            "effect_size_unit": "subject",
            "effect_size_reference": "CN",
            "effect_size_methods": ["Cliff's delta", "median difference", "robust SMD"],
        },
        "mmse_exclusion": {
            "training_mmse_file_exists": bool(TRAIN_MMSE_PATH.is_file()),
            "training_mmse_file_opened": False,
            "training_mmse_values_used": False,
            "validation_mmse_feature_values_opened": False,
            "validation_mmse_identifier_column_opened_for_overlap_only": True,
            "reason": (
                "MMSE sources contain diagnosis/clinical-test fields that are target-adjacent. "
                "They are excluded until prediction-index chronology and diagnostic circularity "
                "are explicitly resolved."
            ),
        },
        "scientific_limitations": [
            "Only nine Training DEM subjects are available; effect sizes are exploratory.",
            "No Validation labels were opened, so this report contains no benchmark performance.",
            "Protocol negative controls are leakage diagnostics, not recommended predictors.",
            "Associations are not clinical causal effects.",
        ],
    }

    report = build_report(audit, feature_summary)
    audit_text = json.dumps(json_ready(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    csv_text = feature_summary.to_csv(index=False, float_format="%.10g")
    all_identifiers = master_ids | activity_ids | sleep_ids | validation_identifiers
    assert_privacy_payload([audit_text, csv_text, report], all_identifiers)

    output_column_names = set(feature_summary.columns)
    forbidden_output_columns = {"EMAIL", "SAMPLE_EMAIL", "patient_id", "subject_id", "hash"}
    if output_column_names & forbidden_output_columns:
        raise RuntimeError("Privacy guard rejected identifier-like output columns")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "data_audit.json").write_text(audit_text, encoding="utf-8")
    (OUTPUT_DIR / "class_feature_summary.csv").write_text(csv_text, encoding="utf-8")
    (OUTPUT_DIR / "EDA_REPORT_KO.md").write_text(report, encoding="utf-8")
    REPORT_COPY_PATH.write_text(report, encoding="utf-8")

    actual_artifacts = {path.name for path in OUTPUT_DIR.iterdir() if path.is_file()}
    unexpected = actual_artifacts - EXPECTED_ARTIFACTS
    missing = EXPECTED_ARTIFACTS - actual_artifacts
    if unexpected or missing:
        raise RuntimeError(
            "Artifact contract failed: "
            f"unexpected_file_count={len(unexpected)}, missing_file_count={len(missing)}"
        )

    print("Privacy-safe local EDA completed.")
    print(relative_path(REPORT_COPY_PATH))
    for name in sorted(EXPECTED_ARTIFACTS):
        print(relative_path(OUTPUT_DIR / name))


if __name__ == "__main__":
    main()
