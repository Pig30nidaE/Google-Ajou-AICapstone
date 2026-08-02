"""Audited loading of the Activity/Sleep lifelog into one row per subject-day.

Source-of-truth facts verified directly against ``Data/`` on 2026-07-29:

* Subject key is ``EMAIL`` in SourceData and ``SAMPLE_EMAIL`` in
  LabelingData/MMSE; both hold the same ``nia+NNN@rowan.kr`` strings.
* ``train_activity.csv`` and ``train_sleep.csv`` each hold 9,705 rows / 141
  subjects; the validation copies hold 2,478 rows / 33 subjects.
* The literal columns ``activity_class_5min``, ``activity_met_1min``,
  ``sleep_hr_5min``, ``sleep_hypnogram_5min`` and ``sleep_rmssd_5min`` contain
  ``"..."`` in **every** row.  The real intraday series live in the
  ``CONVERT(<name> USING utf8)`` columns as slash-separated numbers
  (288 x 5-min activity classes, 1440 x 1-min MET, variable-length sleep series).
* Diagnosis lives in ``LabelingData/{1.Gait,2.Sleep,3.CognitiveFunction}/*_label.csv``
  (identical copies) and *also* inside ``train_mmse.csv`` as ``DIAG_NM``; the MMSE
  reader therefore uses an explicit ``usecols`` allow-list.

Reused with attribution: the daily alignment rules (23-25h activity interval,
longest valid sleep per end-date, activity_day_start local date == sleep
bedtime_end local date) and the slash-series parser come from the audited
``SangHyo/Binary_Wearable_SequenceFusion_Google/data.py``.  Changes made here:
the aligned frame keeps a *relative* day index and weekday flag (needed for the
weekday/weekend and trend statistics that Gemini receives), the daily channel
set is a smaller curated list, and absolute dates never leave this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .guards import LeakageError, assert_names_are_label_free, assert_names_are_mmse_free

__all__ = [
    "ACTIVITY_ALLOWED_COLUMNS",
    "SLEEP_ALLOWED_COLUMNS",
    "DAILY_CHANNELS",
    "HOURLY_MET_COLUMNS",
    "INTENSITY_SHARE_COLUMNS",
    "SLEEP_PHASE_SHARE_COLUMNS",
    "MMSE_DOMAINS",
    "MMSE_ITEMS",
    "DailyDataset",
    "load_daily_dataset",
    "load_diagnoses",
    "binary_target",
    "load_mmse_scores",
    "resolve_split_root",
]

LOCAL_TIMEZONE = "Asia/Seoul"
PERSON_KEY_SOURCE = "EMAIL"
PERSON_KEY_LABEL = "SAMPLE_EMAIL"

ACTIVITY_CLASS_BLOB = "CONVERT(activity_class_5min USING utf8)"
ACTIVITY_MET_BLOB = "CONVERT(activity_met_1min USING utf8)"
SLEEP_HR_BLOB = "CONVERT(sleep_hr_5min USING utf8)"
SLEEP_PHASE_BLOB = "CONVERT(sleep_hypnogram_5min USING utf8)"
SLEEP_RMSSD_BLOB = "CONVERT(sleep_rmssd_5min USING utf8)"

# Explicit read allow-lists.  Acquisition-only fields (activity_non_wear,
# sleep_period_id, sleep_is_longest) are read for validity filtering only and are
# never turned into a channel.
ACTIVITY_ALLOWED_COLUMNS = (
    PERSON_KEY_SOURCE,
    "activity_day_start",
    "activity_day_end",
    "activity_average_met",
    "activity_cal_active",
    "activity_daily_movement",
    "activity_high",
    "activity_medium",
    "activity_low",
    "activity_inactive",
    "activity_rest",
    "activity_score",
    "activity_steps",
    ACTIVITY_CLASS_BLOB,
    ACTIVITY_MET_BLOB,
)
SLEEP_ALLOWED_COLUMNS = (
    PERSON_KEY_SOURCE,
    "sleep_bedtime_start",
    "sleep_bedtime_end",
    "sleep_awake",
    "sleep_breath_average",
    "sleep_deep",
    "sleep_duration",
    "sleep_efficiency",
    "sleep_hr_average",
    "sleep_hr_lowest",
    "sleep_light",
    "sleep_onset_latency",
    "sleep_rem",
    "sleep_restless",
    "sleep_rmssd",
    "sleep_score",
    "sleep_temperature_deviation",
    "sleep_total",
    SLEEP_HR_BLOB,
    SLEEP_PHASE_BLOB,
    SLEEP_RMSSD_BLOB,
)

#: Daily channels offered to the payload builder and to the base feature builder.
DAILY_CHANNELS: tuple[str, ...] = (
    # activity, daily scalars
    "act_steps",
    "act_cal_active",
    "act_daily_movement",
    "act_average_met",
    "act_high_minutes",
    "act_medium_minutes",
    "act_low_minutes",
    "act_inactive_minutes",
    "act_rest_minutes",
    "act_score",
    # activity, derived from the 5-min / 1-min intraday series
    "act_intraday_active_ratio",
    "act_intraday_transition_rate",
    "act_intraday_m10",
    "act_intraday_l5",
    "act_intraday_relative_amplitude",
    # sleep, daily scalars
    "slp_total_minutes",
    "slp_duration_minutes",
    "slp_awake_minutes",
    "slp_onset_latency_minutes",
    "slp_efficiency",
    "slp_deep_ratio",
    "slp_light_ratio",
    "slp_rem_ratio",
    "slp_restless",
    "slp_hr_average",
    "slp_hr_lowest",
    "slp_rmssd",
    "slp_breath_average",
    "slp_temperature_deviation",
    "slp_score",
    # sleep, derived from the 5-min intraday series
    "slp_phase_transition_rate",
    "slp_phase_awake_ratio",
    # clock positions in local decimal hours
    "slp_bedtime_hour",
    "slp_waketime_hour",
    "slp_midsleep_hour",
)

HOURLY_MET_COLUMNS: tuple[str, ...] = tuple(f"hourly_met_h{hour:02d}" for hour in range(24))
INTENSITY_SHARE_COLUMNS: tuple[str, ...] = (
    "intensity_share_rest",
    "intensity_share_inactive",
    "intensity_share_low",
    "intensity_share_medium",
    "intensity_share_high",
)
SLEEP_PHASE_SHARE_COLUMNS: tuple[str, ...] = (
    "phase_share_deep",
    "phase_share_light",
    "phase_share_rem",
    "phase_share_awake",
)

MMSE_DOMAINS: Mapping[str, tuple[str, ...]] = {
    "orient_time": ("Q01", "Q02", "Q03", "Q04", "Q05"),
    "orient_place": ("Q06", "Q07", "Q08", "Q09", "Q10"),
    "registration": ("Q11_1", "Q11_2", "Q11_3"),
    "attention": ("Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"),
    "recall": ("Q13_1", "Q13_2", "Q13_3"),
    "language": ("Q14_1", "Q14_2", "Q15", "Q16_1", "Q16_2", "Q16_3", "Q17", "Q18", "Q19"),
}
MMSE_ITEMS: tuple[str, ...] = tuple(item for items in MMSE_DOMAINS.values() for item in items)
MMSE_ALLOWED_SOURCE_COLUMNS = (PERSON_KEY_LABEL, "TOTAL", *MMSE_ITEMS)
MMSE_FORBIDDEN_SOURCE_COLUMNS = frozenset(
    {"DIAG_NM", "DIAG_SEQ", "DOCTOR_NM", "MMSE_NUM", "MMSE_KIND", "EMAIL", "Q12_TOTAL"}
)

_SPLIT_LAYOUT: Mapping[str, Mapping[str, Any]] = {
    "train": {
        "directory": "1.Training",
        "activity": ("SourceData", "1.Gait", "train_activity.csv"),
        "sleep": ("SourceData", "2.Sleep", "train_sleep.csv"),
        "mmse": ("SourceData", "3.CognitiveFunction", "train_mmse.csv"),
        "label": "training_label.csv",
        "diagnoses": {"CN": 85, "MCI": 47, "Dem": 9},
    },
    "val": {
        "directory": "2.Validation",
        "activity": ("SourceData", "1.Gait", "val_activity.csv"),
        "sleep": ("SourceData", "2.Sleep", "val_sleep.csv"),
        "mmse": ("SourceData", "3.CognitiveFunction", "val_mmse.csv"),
        "label": "val_label.csv",
        "diagnoses": {"CN": 26, "MCI": 4, "Dem": 3},
    },
}
_DIAGNOSIS_ALIASES = {
    "CN": "CN",
    "NORMAL": "CN",
    "MCI": "MCI",
    "DEM": "Dem",
    "DEMENTIA": "Dem",
    "AD": "Dem",
}


@dataclass(frozen=True)
class DailyDataset:
    """One row per subject-day plus the read audit for that split."""

    split: str
    frame: pd.DataFrame
    channels: tuple[str, ...]
    audit: Mapping[str, Any]

    @property
    def subject_ids(self) -> np.ndarray:
        return np.asarray(sorted(self.frame["subject_id"].unique()), dtype=str)

    @property
    def n_subjects(self) -> int:
        return int(self.frame["subject_id"].nunique())


# --------------------------------------------------------------------------- #
# paths and low-level IO
# --------------------------------------------------------------------------- #
def normalise_split(split: str) -> str:
    value = str(split).strip().lower()
    aliases = {"training": "train", "1.training": "train", "validation": "val", "2.validation": "val"}
    value = aliases.get(value, value)
    if value not in _SPLIT_LAYOUT:
        raise ValueError(f"split must be 'train' or 'val'; got {split!r}")
    return value


def resolve_split_root(data_root: str | Path, split: str) -> Path:
    key = normalise_split(split)
    root = Path(data_root).expanduser().resolve()
    directory = str(_SPLIT_LAYOUT[key]["directory"])
    for candidate in (root / directory, root / "Data" / directory, root):
        if candidate.name == directory and candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve {directory} below {root}")


def _read_csv(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read a CSV, optionally restricted to an explicit column allow-list."""

    if not path.is_file():
        raise FileNotFoundError(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            if columns is not None:
                header = pd.read_csv(path, encoding=encoding, nrows=0)
                missing = [name for name in columns if name not in header.columns]
                if missing:
                    raise KeyError(f"{path.name} is missing required columns: {missing}")
            frame = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                usecols=list(columns) if columns is not None else None,
            )
            return frame.loc[:, list(columns)] if columns is not None else frame
        except UnicodeDecodeError as error:
            last_error = error
    raise RuntimeError(f"Unable to decode CSV: {path}") from last_error


def sha256_file(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# numeric helpers (parser reused from Binary_Wearable_SequenceFusion_Google)
# --------------------------------------------------------------------------- #
def _numeric(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def parse_slash_series(value: object) -> np.ndarray:
    """Parse ``"1/2/3/..."`` into a float array; unparsable tokens become NaN."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.empty(0, dtype=np.float64)
    text = str(value).strip()
    if not text or text in {"...", "nan", "None"}:
        return np.empty(0, dtype=np.float64)
    tokens = [token.strip() for token in text.replace(",", "/").split("/")]
    parsed: list[float] = []
    for token in tokens:
        if not token:
            continue
        try:
            parsed.append(float(token))
        except ValueError:
            parsed.append(float("nan"))
    return np.asarray(parsed, dtype=np.float64)


def _series_from_row(row: Mapping[str, Any], blob: str, plain: str) -> np.ndarray:
    value = row.get(blob)
    if value is None or str(value).strip() in {"", "...", "nan", "None"}:
        value = row.get(plain)
    return parse_slash_series(value)


def _transition_rate(values: np.ndarray, valid_codes: Sequence[int]) -> float:
    states = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(states) & np.isin(states, valid_codes)
    adjacent = valid[:-1] & valid[1:]
    if not adjacent.any():
        return float("nan")
    return float(np.mean(states[:-1][adjacent] != states[1:][adjacent]))


def _clock_hour(value: object) -> float:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return float("nan")
    local = timestamp.tz_convert(LOCAL_TIMEZONE)
    return float(local.hour + local.minute / 60.0 + local.second / 3600.0)


def _circular_midpoint(start_hour: float, end_hour: float) -> float:
    if not (np.isfinite(start_hour) and np.isfinite(end_hour)):
        return float("nan")
    span = (end_hour - start_hour) % 24.0
    return float((start_hour + span / 2.0) % 24.0)


# --------------------------------------------------------------------------- #
# per-day feature construction
# --------------------------------------------------------------------------- #
def _activity_intraday(row: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    classes = _series_from_row(row, ACTIVITY_CLASS_BLOB, "activity_class_5min")
    valid_codes = (1, 2, 3, 4, 5)
    valid = classes[np.isfinite(classes) & np.isin(classes, valid_codes)]
    for code, name in ((1, "rest"), (2, "inactive"), (3, "low"), (4, "medium"), (5, "high")):
        output[f"intensity_share_{name}"] = (
            float(np.mean(valid == code)) if valid.size else float("nan")
        )
    output["act_intraday_active_ratio"] = (
        float(np.mean(valid >= 3)) if valid.size else float("nan")
    )
    output["act_intraday_transition_rate"] = _transition_rate(classes, valid_codes)

    met = _series_from_row(row, ACTIVITY_MET_BLOB, "activity_met_1min")
    # Code 0 marks non-wear/acquisition gaps.  It is used only to mask MET
    # minutes; it never becomes a feature (same rule as the audited loader).
    wear_by_minute = np.repeat(np.isin(classes, valid_codes), 5)
    masked = np.full(met.shape, np.nan, dtype=np.float64)
    usable = min(met.size, wear_by_minute.size)
    if usable:
        keep = wear_by_minute[:usable] & np.isfinite(met[:usable])
        positions = np.flatnonzero(keep)
        masked[positions] = met[positions]

    hourly = np.full(24, np.nan, dtype=np.float64)
    start_hour = _clock_hour(row.get("activity_day_start"))
    if masked.size == 1440 and np.isfinite(start_hour):
        block = masked.reshape(24, 60)
        offset = int(round(start_hour)) % 24
        for index in range(24):
            values = block[index][np.isfinite(block[index])]
            if values.size:
                hourly[(offset + index) % 24] = float(np.mean(values))
    for index in range(24):
        output[f"hourly_met_h{index:02d}"] = float(hourly[index])

    if np.isfinite(hourly).sum() >= 20:
        filled = hourly.copy()
        known = np.flatnonzero(np.isfinite(filled))
        if known.size < 24:
            circular_x = np.r_[known - 24, known, known + 24]
            circular_y = np.tile(filled[known], 3)
            filled = np.interp(np.arange(24), circular_x, circular_y)
        doubled = np.r_[filled, filled]
        m10 = max(float(doubled[start : start + 10].mean()) for start in range(24))
        l5 = min(float(doubled[start : start + 5].mean()) for start in range(24))
        output["act_intraday_m10"] = m10
        output["act_intraday_l5"] = l5
        output["act_intraday_relative_amplitude"] = float(
            (m10 - l5) / (m10 + l5 + 1e-8)
        )
    else:
        output["act_intraday_m10"] = float("nan")
        output["act_intraday_l5"] = float("nan")
        output["act_intraday_relative_amplitude"] = float("nan")
    return output


def _sleep_intraday(row: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    phases = _series_from_row(row, SLEEP_PHASE_BLOB, "sleep_hypnogram_5min")
    valid_codes = (1, 2, 3, 4)
    valid = phases[np.isfinite(phases) & np.isin(phases, valid_codes)]
    for code, name in ((1, "deep"), (2, "light"), (3, "rem"), (4, "awake")):
        output[f"phase_share_{name}"] = (
            float(np.mean(valid == code)) if valid.size else float("nan")
        )
    output["slp_phase_transition_rate"] = _transition_rate(phases, valid_codes)
    output["slp_phase_awake_ratio"] = output["phase_share_awake"]
    return output


def _daily_row(row: Mapping[str, Any]) -> dict[str, float]:
    duration = _numeric(row.get("sleep_duration"))
    values: dict[str, float] = {
        "act_steps": _numeric(row.get("activity_steps")),
        "act_cal_active": _numeric(row.get("activity_cal_active")),
        "act_daily_movement": _numeric(row.get("activity_daily_movement")),
        "act_average_met": _numeric(row.get("activity_average_met")),
        "act_high_minutes": _numeric(row.get("activity_high")),
        "act_medium_minutes": _numeric(row.get("activity_medium")),
        "act_low_minutes": _numeric(row.get("activity_low")),
        "act_inactive_minutes": _numeric(row.get("activity_inactive")),
        "act_rest_minutes": _numeric(row.get("activity_rest")),
        "act_score": _numeric(row.get("activity_score")),
        "slp_total_minutes": _numeric(row.get("sleep_total")) / 60.0,
        "slp_duration_minutes": duration / 60.0,
        "slp_awake_minutes": _numeric(row.get("sleep_awake")) / 60.0,
        "slp_onset_latency_minutes": _numeric(row.get("sleep_onset_latency")) / 60.0,
        "slp_efficiency": _numeric(row.get("sleep_efficiency")),
        "slp_restless": _numeric(row.get("sleep_restless")),
        "slp_hr_average": _numeric(row.get("sleep_hr_average")),
        "slp_hr_lowest": _numeric(row.get("sleep_hr_lowest")),
        "slp_rmssd": _numeric(row.get("sleep_rmssd")),
        "slp_breath_average": _numeric(row.get("sleep_breath_average")),
        "slp_temperature_deviation": _numeric(row.get("sleep_temperature_deviation")),
        "slp_score": _numeric(row.get("sleep_score")),
    }
    for source, name in (("sleep_deep", "deep"), ("sleep_light", "light"), ("sleep_rem", "rem")):
        stage_seconds = _numeric(row.get(source))
        values[f"slp_{name}_ratio"] = (
            stage_seconds / duration
            if np.isfinite(stage_seconds) and np.isfinite(duration) and duration > 0
            else float("nan")
        )
    bedtime = _clock_hour(row.get("sleep_bedtime_start"))
    waketime = _clock_hour(row.get("sleep_bedtime_end"))
    values["slp_bedtime_hour"] = bedtime
    values["slp_waketime_hour"] = waketime
    values["slp_midsleep_hour"] = _circular_midpoint(bedtime, waketime)
    values.update(_activity_intraday(row))
    values.update(_sleep_intraday(row))
    return values


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #
def _normalise_subject_ids(values: pd.Series, source: str) -> pd.Series:
    if values.isna().any():
        raise LeakageError(f"Missing subject identifier in {source}")
    normalized = values.astype(str).str.strip()
    invalid = normalized.str.lower().isin({"", "nan", "none", "null", "<na>"})
    if invalid.any():
        raise LeakageError(f"Invalid subject identifier in {source}")
    return normalized


def _local_timestamp(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", utc=True).dt.tz_convert(LOCAL_TIMEZONE)


def _align_sources(split_root: Path, split_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    layout = _SPLIT_LAYOUT[split_key]
    activity_path = split_root.joinpath(*layout["activity"])
    sleep_path = split_root.joinpath(*layout["sleep"])
    activity = _read_csv(activity_path, ACTIVITY_ALLOWED_COLUMNS)
    sleep = _read_csv(sleep_path, SLEEP_ALLOWED_COLUMNS)
    audit: dict[str, Any] = {
        "activity_source_rows": int(len(activity)),
        "sleep_source_rows": int(len(sleep)),
        "activity_file_sha256": sha256_file(activity_path),
        "sleep_file_sha256": sha256_file(sleep_path),
    }

    activity = activity.copy()
    activity["subject_id"] = _normalise_subject_ids(activity[PERSON_KEY_SOURCE], "activity")
    activity["_start"] = _local_timestamp(activity["activity_day_start"])
    activity["_end"] = _local_timestamp(activity["activity_day_end"])
    activity["_date"] = activity["_start"].dt.date
    duration = (activity["_end"] - activity["_start"]).dt.total_seconds()
    valid_interval = (
        activity["_start"].notna()
        & activity["_end"].notna()
        & duration.between(23 * 3600, 25 * 3600)
    )
    audit["activity_rows_dropped_invalid_interval"] = int((~valid_interval).sum())
    activity = activity.loc[valid_interval].copy()
    audit["activity_duplicate_subject_days"] = int(
        activity.duplicated(["subject_id", "_date"]).sum()
    )
    activity = activity.sort_values(
        ["subject_id", "_date", "_start"], kind="mergesort"
    ).drop_duplicates(["subject_id", "_date"], keep="last")

    sleep = sleep.copy()
    sleep["subject_id"] = _normalise_subject_ids(sleep[PERSON_KEY_SOURCE], "sleep")
    sleep["_wake"] = _local_timestamp(sleep["sleep_bedtime_end"])
    sleep["_date"] = sleep["_wake"].dt.date
    sleep["_duration"] = pd.to_numeric(sleep["sleep_duration"], errors="coerce")
    valid_sleep = sleep["_wake"].notna() & sleep["_duration"].gt(0) & sleep["_duration"].le(24 * 3600)
    audit["sleep_rows_dropped_invalid"] = int((~valid_sleep).sum())
    sleep = sleep.loc[valid_sleep].copy()
    audit["sleep_duplicate_subject_days"] = int(sleep.duplicated(["subject_id", "_date"]).sum())
    # Deterministic representative: the longest valid sleep for that wake date.
    sleep = sleep.sort_values(
        ["subject_id", "_date", "_duration", "_wake"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).drop_duplicates(["subject_id", "_date"], keep="first")

    aligned = activity.merge(
        sleep.drop(columns=[PERSON_KEY_SOURCE]),
        on=["subject_id", "_date"],
        how="inner",
        suffixes=("", "__sleep"),
        validate="one_to_one",
    ).sort_values(["subject_id", "_date"], kind="mergesort")
    late_sleep = aligned["_wake"].gt(aligned["_end"])
    if int(late_sleep.sum()):
        raise LeakageError(
            f"{int(late_sleep.sum())} sleep wake timestamps fall outside their activity day"
        )
    audit["aligned_subject_days"] = int(len(aligned))
    audit["aligned_subjects"] = int(aligned["subject_id"].nunique())
    return aligned, audit


def _build_daily_frame(aligned: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in aligned.to_dict(orient="records"):
        values = _daily_row(row)
        date = pd.Timestamp(row["_date"])
        values["subject_id"] = str(row["subject_id"])
        values["_date_ordinal"] = int(date.toordinal())
        values["day_of_week"] = int(date.dayofweek)
        values["is_weekend"] = int(date.dayofweek >= 5)
        records.append(values)
    frame = pd.DataFrame.from_records(records)
    # Relative day index only.  Absolute dates never leave this function.
    frame["day_index"] = (
        frame["_date_ordinal"] - frame.groupby("subject_id")["_date_ordinal"].transform("min")
    ).astype(int)
    frame = frame.drop(columns=["_date_ordinal"])
    ordered = ["subject_id", "day_index", "day_of_week", "is_weekend"]
    remaining = [name for name in frame.columns if name not in ordered]
    return frame.loc[:, ordered + remaining].sort_values(
        ["subject_id", "day_index"], kind="mergesort"
    ).reset_index(drop=True)


def load_daily_dataset(
    data_root: str | Path,
    split: str,
    *,
    min_days_per_subject: int = 28,
    expected_subjects: int | None = None,
    cache_dir: str | Path | None = None,
) -> DailyDataset:
    """Return one row per subject-day with the curated daily channels.

    ``cache_dir`` stores the parsed table keyed by the source-file SHA-256, so
    repeated ``run.py`` stages do not re-parse ~14M intraday values.
    """

    split_key = normalise_split(split)
    split_root = resolve_split_root(data_root, split_key)
    layout = _SPLIT_LAYOUT[split_key]
    fingerprint = hashlib.sha256(
        "|".join(
            [
                sha256_file(split_root.joinpath(*layout["activity"])),
                sha256_file(split_root.joinpath(*layout["sleep"])),
                "daily-v1",
                str(min_days_per_subject),
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]

    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser() / f"daily_{split_key}_{fingerprint}.csv"
        if cache_path.is_file():
            frame = pd.read_csv(cache_path)
            frame["subject_id"] = frame["subject_id"].astype(str)
            audit = {
                "split": split_key,
                "source": "cache",
                "cache_file": str(cache_path),
                "aligned_subject_days": int(len(frame)),
                "aligned_subjects": int(frame["subject_id"].nunique()),
            }
            return DailyDataset(split_key, frame, DAILY_CHANNELS, audit)

    aligned, audit = _align_sources(split_root, split_key)
    frame = _build_daily_frame(aligned)

    counts = frame.groupby("subject_id").size()
    short = counts[counts < int(min_days_per_subject)]
    audit["subjects_below_min_days"] = int(len(short))
    if len(short):
        raise LeakageError(
            f"{len(short)} subject(s) have fewer than {min_days_per_subject} aligned days; "
            "padding or length-based dropping is intentionally not implemented"
        )
    if expected_subjects is not None and int(frame["subject_id"].nunique()) != int(expected_subjects):
        raise LeakageError(
            f"{split_key}: expected {expected_subjects} subjects, "
            f"got {frame['subject_id'].nunique()}"
        )

    missing_channels = [name for name in DAILY_CHANNELS if name not in frame.columns]
    if missing_channels:
        raise LeakageError(f"Daily channel construction is incomplete: {missing_channels}")
    assert_names_are_label_free(DAILY_CHANNELS, context="daily channels")
    assert_names_are_mmse_free(DAILY_CHANNELS, context="daily channels")

    audit.update(
        {
            "split": split_key,
            "source": "parsed",
            "channels": list(DAILY_CHANNELS),
            "days_per_subject_min": int(counts.min()),
            "days_per_subject_median": float(counts.median()),
            "days_per_subject_max": int(counts.max()),
        }
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
        audit["cache_file"] = str(cache_path)
    return DailyDataset(split_key, frame, DAILY_CHANNELS, audit)


# --------------------------------------------------------------------------- #
# labels and MMSE (never used before the Gemini stage has finished)
# --------------------------------------------------------------------------- #
def _normalise_label_copy(frame: pd.DataFrame, source: str) -> pd.Series:
    work = frame.copy()
    work[PERSON_KEY_LABEL] = work[PERSON_KEY_LABEL].astype(str).str.strip()
    raw = work["DIAG_NM"].astype(str).str.strip().str.upper()
    diagnosis = raw.map(_DIAGNOSIS_ALIASES)
    if diagnosis.isna().any():
        raise LeakageError(f"{source}: unknown diagnosis values {sorted(set(raw[diagnosis.isna()]))}")
    series = pd.Series(diagnosis.to_numpy(), index=work[PERSON_KEY_LABEL].to_numpy(), name="diagnosis")
    if series.index.has_duplicates:
        if (series.groupby(level=0).nunique() > 1).any():
            raise LeakageError(f"{source}: conflicting duplicate diagnosis rows")
        series = series[~series.index.duplicated(keep="first")]
    return series.sort_index()


def load_diagnoses(
    data_root: str | Path, split: str, *, strict: bool = True
) -> tuple[pd.Series, dict[str, Any]]:
    """Load and cross-check the Gait and Sleep diagnosis copies."""

    split_key = normalise_split(split)
    root = resolve_split_root(data_root, split_key)
    filename = str(_SPLIT_LAYOUT[split_key]["label"])
    copies: dict[str, pd.Series] = {}
    audit: dict[str, Any] = {"files": []}
    for modality, directory in (("Gait", "1.Gait"), ("Sleep", "2.Sleep")):
        path = root / "LabelingData" / directory / filename
        copies[modality] = _normalise_label_copy(
            _read_csv(path, (PERSON_KEY_LABEL, "DIAG_NM")), modality
        )
        audit["files"].append({"path": str(path), "sha256": sha256_file(path)})
    if not copies["Gait"].equals(copies["Sleep"]):
        raise LeakageError("Gait and Sleep diagnosis copies disagree")
    observed = copies["Gait"].value_counts().to_dict()
    audit["diagnosis_counts"] = {str(key): int(value) for key, value in observed.items()}
    if strict and audit["diagnosis_counts"] != dict(_SPLIT_LAYOUT[split_key]["diagnoses"]):
        raise LeakageError(
            f"{split_key} diagnosis contract changed: {audit['diagnosis_counts']}"
        )
    return copies["Gait"].copy(), audit


def binary_target(
    diagnosis: pd.Series,
    *,
    positive: Iterable[str] = ("MCI", "Dem"),
    negative: Iterable[str] = ("CN",),
) -> pd.Series:
    positive_set, negative_set = set(positive), set(negative)
    unknown = sorted(set(diagnosis.astype(str)) - positive_set - negative_set)
    if unknown:
        raise LeakageError(f"Target mapping does not cover: {unknown}")
    return diagnosis.astype(str).isin(positive_set).astype(np.int64)


def load_mmse_scores(data_root: str | Path, split: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only the MMSE score allow-list; ``DIAG_NM`` never enters memory."""

    split_key = normalise_split(split)
    root = resolve_split_root(data_root, split_key)
    path = root.joinpath(*_SPLIT_LAYOUT[split_key]["mmse"])
    frame = _read_csv(path, MMSE_ALLOWED_SOURCE_COLUMNS)
    forbidden = set(frame.columns) & MMSE_FORBIDDEN_SOURCE_COLUMNS
    if forbidden:
        raise LeakageError(f"Forbidden MMSE source columns were read: {sorted(forbidden)}")
    frame = frame.copy()
    frame[PERSON_KEY_LABEL] = frame[PERSON_KEY_LABEL].astype(str).str.strip()
    if frame[PERSON_KEY_LABEL].duplicated().any():
        raise LeakageError("MMSE table contains duplicate subject identifiers")
    frame = frame.set_index(PERSON_KEY_LABEL).sort_index()
    frame.index.name = "subject_id"
    audit = {
        "path": str(path),
        "sha256": sha256_file(path),
        "selected_columns": list(MMSE_ALLOWED_SOURCE_COLUMNS),
        "n_subjects": int(len(frame)),
    }
    return frame, audit
