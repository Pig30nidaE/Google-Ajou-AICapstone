"""Pooled cohort loading: Activity/Sleep daily table + MMSE + labels for 174 subjects.

Verified source facts (checked directly against ``Data/``):

* Subject key is ``EMAIL`` in SourceData and ``SAMPLE_EMAIL`` in LabelingData/MMSE.
* Training 141 (CN 85 / MCI 47 / Dem 9), Validation 33 (CN 26 / MCI 4 / Dem 3);
  pooled 174 = CN 111 / MCI 51 / Dem 12.
* ``activity_class_5min``, ``activity_met_1min``, ``sleep_hr_5min``,
  ``sleep_hypnogram_5min``, ``sleep_rmssd_5min`` are the literal string ``"..."``
  in every row; the real intraday series live in the
  ``CONVERT(<name> USING utf8)`` columns (288 x 5-min classes, 1440 x 1-min MET,
  variable-length sleep series).
* MMSE items are coded ``1 = incorrect`` / ``2 = correct`` and ``TOTAL`` equals
  the count of items scored 2 (verified exactly on 141/141 training subjects),
  so ``item_max`` is the constant 2.0, not a value learned from the data.

Daily alignment rules (23-25h activity interval, longest valid sleep per wake
date, activity_day_start local date == sleep_bedtime_end local date) and the
slash-series parser are reused from the audited
``SangHyo/Binary_Wearable_SequenceFusion_Google/data.py`` via the validated
port in ``SangHyo/GeminiFeaturePipeline/data.py``.  This module differs by
pooling both splits into one cohort and by keeping a ``split_origin`` marker
(used for auditing only, never as a feature).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .leakage import LeakageError

__all__ = [
    "DAILY_CHANNELS",
    "CLOCK_CHANNELS",
    "MMSE_DOMAINS",
    "MMSE_ITEMS",
    "PooledCohort",
    "load_pooled_cohort",
]

LOCAL_TIMEZONE = "Asia/Seoul"
PERSON_KEY_SOURCE = "EMAIL"
PERSON_KEY_LABEL = "SAMPLE_EMAIL"

ACTIVITY_CLASS_BLOB = "CONVERT(activity_class_5min USING utf8)"
ACTIVITY_MET_BLOB = "CONVERT(activity_met_1min USING utf8)"
SLEEP_HR_BLOB = "CONVERT(sleep_hr_5min USING utf8)"
SLEEP_PHASE_BLOB = "CONVERT(sleep_hypnogram_5min USING utf8)"
SLEEP_RMSSD_BLOB = "CONVERT(sleep_rmssd_5min USING utf8)"

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

#: Daily channels available for per-subject aggregation.
DAILY_CHANNELS: tuple[str, ...] = (
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
    "act_intraday_active_ratio",
    "act_intraday_transition_rate",
    "act_intraday_m10",
    "act_intraday_l5",
    "act_intraday_relative_amplitude",
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
    "slp_phase_transition_rate",
    "slp_phase_awake_ratio",
    "slp_bedtime_hour",
    "slp_waketime_hour",
    "slp_midsleep_hour",
)

#: Clock channels need circular statistics, not linear ones.
CLOCK_CHANNELS: tuple[str, ...] = ("slp_bedtime_hour", "slp_waketime_hour", "slp_midsleep_hour")

#: Intraday-derived channels; dropped when features.include_intraday is false.
INTRADAY_CHANNELS: tuple[str, ...] = (
    "act_intraday_active_ratio",
    "act_intraday_transition_rate",
    "act_intraday_m10",
    "act_intraday_l5",
    "act_intraday_relative_amplitude",
    "slp_phase_transition_rate",
    "slp_phase_awake_ratio",
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
class PooledCohort:
    """One row per subject: labels, MMSE table and the subject-day daily frame."""

    subject_ids: np.ndarray
    y: np.ndarray
    diagnosis: np.ndarray
    split_origin: np.ndarray
    daily: pd.DataFrame
    mmse: pd.DataFrame
    channels: tuple[str, ...]
    audit: Mapping[str, Any]

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def n_positive(self) -> int:
        return int(self.y.sum())


# --------------------------------------------------------------------------- #
# io helpers
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
                path, encoding=encoding, low_memory=False,
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
# intraday parsing
# --------------------------------------------------------------------------- #
def _numeric(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def parse_slash_series(value: object) -> np.ndarray:
    """Parse ``"1/2/3"`` into floats; unparsable tokens become NaN."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.empty(0, dtype=np.float64)
    text = str(value).strip()
    if not text or text in {"...", "nan", "None"}:
        return np.empty(0, dtype=np.float64)
    parsed: list[float] = []
    for token in text.replace(",", "/").split("/"):
        token = token.strip()
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


def _activity_intraday(row: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    classes = _series_from_row(row, ACTIVITY_CLASS_BLOB, "activity_class_5min")
    valid_codes = (1, 2, 3, 4, 5)
    valid = classes[np.isfinite(classes) & np.isin(classes, valid_codes)]
    output["act_intraday_active_ratio"] = float(np.mean(valid >= 3)) if valid.size else float("nan")
    output["act_intraday_transition_rate"] = _transition_rate(classes, valid_codes)

    met = _series_from_row(row, ACTIVITY_MET_BLOB, "activity_met_1min")
    # Code 0 is non-wear: used only to mask MET minutes, never as a feature.
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
    if np.isfinite(hourly).sum() >= 20:
        filled = hourly.copy()
        known = np.flatnonzero(np.isfinite(filled))
        if known.size < 24:
            circular_x = np.r_[known - 24, known, known + 24]
            circular_y = np.tile(filled[known], 3)
            filled = np.interp(np.arange(24), circular_x, circular_y)
        doubled = np.r_[filled, filled]
        m10 = max(float(doubled[s : s + 10].mean()) for s in range(24))
        l5 = min(float(doubled[s : s + 5].mean()) for s in range(24))
        output["act_intraday_m10"] = m10
        output["act_intraday_l5"] = l5
        output["act_intraday_relative_amplitude"] = float((m10 - l5) / (m10 + l5 + 1e-8))
    else:
        output["act_intraday_m10"] = float("nan")
        output["act_intraday_l5"] = float("nan")
        output["act_intraday_relative_amplitude"] = float("nan")
    return output


def _sleep_intraday(row: Mapping[str, Any]) -> dict[str, float]:
    phases = _series_from_row(row, SLEEP_PHASE_BLOB, "sleep_hypnogram_5min")
    valid_codes = (1, 2, 3, 4)
    valid = phases[np.isfinite(phases) & np.isin(phases, valid_codes)]
    awake_ratio = float(np.mean(valid == 4)) if valid.size else float("nan")
    return {
        "slp_phase_transition_rate": _transition_rate(phases, valid_codes),
        "slp_phase_awake_ratio": awake_ratio,
    }


def _daily_row(row: Mapping[str, Any], include_intraday: bool) -> dict[str, float]:
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
    if include_intraday:
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
    if normalized.str.lower().isin({"", "nan", "none", "null", "<na>"}).any():
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
    valid = activity["_start"].notna() & activity["_end"].notna() & duration.between(23 * 3600, 25 * 3600)
    audit["activity_rows_dropped_invalid_interval"] = int((~valid).sum())
    activity = activity.loc[valid].copy()
    activity = activity.sort_values(["subject_id", "_date", "_start"], kind="mergesort").drop_duplicates(
        ["subject_id", "_date"], keep="last"
    )

    sleep = sleep.copy()
    sleep["subject_id"] = _normalise_subject_ids(sleep[PERSON_KEY_SOURCE], "sleep")
    sleep["_wake"] = _local_timestamp(sleep["sleep_bedtime_end"])
    sleep["_date"] = sleep["_wake"].dt.date
    sleep["_duration"] = pd.to_numeric(sleep["sleep_duration"], errors="coerce")
    valid_sleep = sleep["_wake"].notna() & sleep["_duration"].gt(0) & sleep["_duration"].le(24 * 3600)
    audit["sleep_rows_dropped_invalid"] = int((~valid_sleep).sum())
    sleep = sleep.loc[valid_sleep].copy()
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
    audit["aligned_subject_days"] = int(len(aligned))
    audit["aligned_subjects"] = int(aligned["subject_id"].nunique())
    return aligned, audit


def _build_daily_frame(aligned: pd.DataFrame, include_intraday: bool) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in aligned.to_dict(orient="records"):
        values = _daily_row(row, include_intraday)
        date = pd.Timestamp(row["_date"])
        values["subject_id"] = str(row["subject_id"])
        values["_date_ordinal"] = int(date.toordinal())
        values["is_weekend"] = int(date.dayofweek >= 5)
        records.append(values)
    frame = pd.DataFrame.from_records(records)
    # Relative day index only; absolute dates never leave this function.
    frame["day_index"] = (
        frame["_date_ordinal"] - frame.groupby("subject_id")["_date_ordinal"].transform("min")
    ).astype(int)
    frame = frame.drop(columns=["_date_ordinal"])
    ordered = ["subject_id", "day_index", "is_weekend"]
    remaining = [c for c in frame.columns if c not in ordered]
    return (
        frame.loc[:, ordered + remaining]
        .sort_values(["subject_id", "day_index"], kind="mergesort")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# labels and MMSE
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


def _load_diagnoses(data_root: str | Path, split: str, *, strict: bool) -> tuple[pd.Series, dict]:
    split_key = normalise_split(split)
    root = resolve_split_root(data_root, split_key)
    filename = str(_SPLIT_LAYOUT[split_key]["label"])
    copies: dict[str, pd.Series] = {}
    audit: dict[str, Any] = {"files": []}
    for modality, directory in (("Gait", "1.Gait"), ("Sleep", "2.Sleep")):
        path = root / "LabelingData" / directory / filename
        copies[modality] = _normalise_label_copy(_read_csv(path, (PERSON_KEY_LABEL, "DIAG_NM")), modality)
        audit["files"].append({"path": str(path), "sha256": sha256_file(path)})
    if not copies["Gait"].equals(copies["Sleep"]):
        raise LeakageError("Gait and Sleep diagnosis copies disagree")
    observed = {str(k): int(v) for k, v in copies["Gait"].value_counts().to_dict().items()}
    audit["diagnosis_counts"] = observed
    if strict and observed != dict(_SPLIT_LAYOUT[split_key]["diagnoses"]):
        raise LeakageError(f"{split_key} diagnosis contract changed: {observed}")
    return copies["Gait"].copy(), audit


def _load_mmse(data_root: str | Path, split: str) -> tuple[pd.DataFrame, dict]:
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
    return frame, {"path": str(path), "sha256": sha256_file(path), "n_subjects": int(len(frame))}


# --------------------------------------------------------------------------- #
# public loader
# --------------------------------------------------------------------------- #
def load_pooled_cohort(
    data_root: str | Path,
    *,
    splits: Iterable[str] = ("train", "val"),
    positive: Iterable[str] = ("MCI", "Dem"),
    negative: Iterable[str] = ("CN",),
    min_days_per_subject: int = 28,
    include_intraday: bool = True,
    expected_subjects: Mapping[str, int] | None = None,
    expected_pooled_diagnoses: Mapping[str, int] | None = None,
    strict: bool = True,
    cache_dir: str | Path | None = None,
) -> PooledCohort:
    """Load and pool the requested splits into one subject-level cohort."""

    daily_frames: list[pd.DataFrame] = []
    mmse_frames: list[pd.DataFrame] = []
    diagnosis_parts: list[pd.Series] = []
    origin_parts: list[pd.Series] = []
    audit: dict[str, Any] = {"splits": {}, "pooled": {}}

    for split in splits:
        split_key = normalise_split(split)
        split_root = resolve_split_root(data_root, split_key)
        layout = _SPLIT_LAYOUT[split_key]

        fingerprint = hashlib.sha256(
            "|".join(
                [
                    sha256_file(split_root.joinpath(*layout["activity"])),
                    sha256_file(split_root.joinpath(*layout["sleep"])),
                    f"daily-v1-intraday{int(include_intraday)}",
                    str(min_days_per_subject),
                ]
            ).encode("utf-8")
        ).hexdigest()[:16]

        cache_path: Path | None = None
        daily: pd.DataFrame | None = None
        split_audit: dict[str, Any] = {}
        if cache_dir is not None:
            cache_path = Path(cache_dir).expanduser() / f"daily_{split_key}_{fingerprint}.csv"
            if cache_path.is_file():
                daily = pd.read_csv(cache_path)
                daily["subject_id"] = daily["subject_id"].astype(str)
                split_audit = {"source": "cache", "cache_file": str(cache_path)}

        if daily is None:
            aligned, split_audit = _align_sources(split_root, split_key)
            daily = _build_daily_frame(aligned, include_intraday)
            split_audit["source"] = "parsed"
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                daily.to_csv(cache_path, index=False)
                split_audit["cache_file"] = str(cache_path)

        counts = daily.groupby("subject_id").size()
        short = counts[counts < int(min_days_per_subject)]
        if len(short):
            raise LeakageError(
                f"{split_key}: {len(short)} subject(s) have fewer than "
                f"{min_days_per_subject} aligned days"
            )
        if expected_subjects and split_key in expected_subjects:
            expected = int(expected_subjects[split_key])
            if int(daily["subject_id"].nunique()) != expected:
                raise LeakageError(
                    f"{split_key}: expected {expected} subjects, got {daily['subject_id'].nunique()}"
                )

        diagnosis, label_audit = _load_diagnoses(data_root, split_key, strict=strict)
        mmse, mmse_audit = _load_mmse(data_root, split_key)

        subjects = sorted(daily["subject_id"].astype(str).unique())
        missing_labels = sorted(set(subjects) - set(diagnosis.index.astype(str)))
        if missing_labels:
            raise LeakageError(f"{split_key}: {len(missing_labels)} subject(s) lack a diagnosis")
        missing_mmse = sorted(set(subjects) - set(mmse.index.astype(str)))
        if missing_mmse:
            raise LeakageError(f"{split_key}: {len(missing_mmse)} subject(s) lack MMSE rows")

        daily_frames.append(daily)
        mmse_frames.append(mmse.loc[subjects])
        diagnosis_parts.append(diagnosis.loc[subjects])
        origin_parts.append(pd.Series(split_key, index=subjects, name="split_origin"))
        split_audit.update(
            {
                "n_subjects": len(subjects),
                "labels": label_audit,
                "mmse": mmse_audit,
                "days_per_subject_min": int(counts.min()),
                "days_per_subject_median": float(counts.median()),
                "days_per_subject_max": int(counts.max()),
            }
        )
        audit["splits"][split_key] = split_audit

    pooled_daily = pd.concat(daily_frames, ignore_index=True)
    pooled_mmse = pd.concat(mmse_frames, axis=0)
    pooled_diagnosis = pd.concat(diagnosis_parts, axis=0)
    pooled_origin = pd.concat(origin_parts, axis=0)

    if pooled_mmse.index.has_duplicates:
        raise LeakageError("Pooling produced duplicate subject ids across splits")

    subject_ids = np.asarray(sorted(pooled_mmse.index.astype(str)), dtype=str)
    diagnosis = pooled_diagnosis.reindex(subject_ids).to_numpy(dtype=str)
    positive_set, negative_set = set(positive), set(negative)
    unknown = sorted(set(diagnosis) - positive_set - negative_set)
    if unknown:
        raise LeakageError(f"Target mapping does not cover: {unknown}")
    y = np.asarray([int(value in positive_set) for value in diagnosis], dtype=np.int64)

    observed_pooled = {name: int((diagnosis == name).sum()) for name in sorted(set(diagnosis))}
    audit["pooled"] = {
        "n_subjects": int(len(subject_ids)),
        "diagnosis_counts": observed_pooled,
        "class_counts": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        "n_subject_days": int(len(pooled_daily)),
        "positive_definition": sorted(positive_set),
        "negative_definition": sorted(negative_set),
    }
    if strict and expected_pooled_diagnoses:
        if observed_pooled != dict(expected_pooled_diagnoses):
            raise LeakageError(
                f"Pooled diagnosis contract changed: {observed_pooled} != {dict(expected_pooled_diagnoses)}"
            )

    channels = tuple(
        c for c in DAILY_CHANNELS if include_intraday or c not in INTRADAY_CHANNELS
    )
    missing_channels = [c for c in channels if c not in pooled_daily.columns]
    if missing_channels:
        raise LeakageError(f"Daily channel construction incomplete: {missing_channels}")

    return PooledCohort(
        subject_ids=subject_ids,
        y=y,
        diagnosis=diagnosis,
        split_origin=pooled_origin.reindex(subject_ids).to_numpy(dtype=str),
        daily=pooled_daily,
        mmse=pooled_mmse.reindex(subject_ids),
        channels=channels,
        audit=audit,
    )
