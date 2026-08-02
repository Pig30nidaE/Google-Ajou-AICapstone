"""Leakage-conscious Activity/Sleep sequence construction for binary diagnosis.

This module deliberately knows about only the Gait and Sleep source/label copies.
It never scans the data tree and never resolves an unrelated modality directory.
The model-facing representation contains biological/behavioural measurements only:
no identifier, absolute date, acquisition order, observation count, coverage, mask,
padding, period identifier, or non-wear feature is exported.

The main public entry points are :func:`build_subject_dataset`,
:func:`load_binary_labels`, and :func:`make_fixed_views`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


LOCAL_TIMEZONE = "Asia/Seoul"
DEFAULT_SEQUENCE_LENGTH = 28
DEFAULT_N_VIEWS = 8

ACTIVITY_CLASS_BLOB = "CONVERT(activity_class_5min USING utf8)"
ACTIVITY_MET_BLOB = "CONVERT(activity_met_1min USING utf8)"
SLEEP_HR_BLOB = "CONVERT(sleep_hr_5min USING utf8)"
SLEEP_STAGE_BLOB = "CONVERT(sleep_hypnogram_5min USING utf8)"
SLEEP_RMSSD_BLOB = "CONVERT(sleep_rmssd_5min USING utf8)"

ACTIVITY_SEQUENCE_SOURCE_ALTERNATIVES = (
    (ACTIVITY_CLASS_BLOB, "activity_class_5min"),
    (ACTIVITY_MET_BLOB, "activity_met_1min"),
)
SLEEP_SEQUENCE_SOURCE_ALTERNATIVES = (
    (SLEEP_HR_BLOB, "sleep_hr_5min"),
    (SLEEP_STAGE_BLOB, "sleep_hypnogram_5min"),
    (SLEEP_RMSSD_BLOB, "sleep_rmssd_5min"),
)

# These are fixed, label-blind source contracts.  Acquisition-related columns
# such as activity_non_wear, sleep_period_id, and sleep_is_longest are absent.
ACTIVITY_SAFE_SCALARS = (
    "activity_average_met",
    "activity_cal_active",
    "activity_cal_total",
    "activity_daily_movement",
    "activity_high",
    "activity_inactive",
    "activity_low",
    "activity_medium",
    "activity_met_min_high",
    "activity_met_min_inactive",
    "activity_met_min_low",
    "activity_met_min_medium",
    "activity_rest",
    "activity_score",
    "activity_score_meet_daily_targets",
    "activity_score_move_every_hour",
    "activity_score_recovery_time",
    "activity_score_stay_active",
    "activity_score_training_frequency",
    "activity_score_training_volume",
    # Steps is a biological behaviour total, not a source-row/observation count.
    "activity_steps",
    "activity_total",
)

SLEEP_SAFE_SCALARS = (
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
    "sleep_rmssd",
    "sleep_score",
    "sleep_score_alignment",
    "sleep_score_deep",
    "sleep_score_disturbances",
    "sleep_score_efficiency",
    "sleep_score_latency",
    "sleep_score_rem",
    "sleep_score_total",
    "sleep_temperature_delta",
    "sleep_temperature_deviation",
    "sleep_total",
)

_LABEL_MAP = {
    "CN": 0,
    "NORMAL": 0,
    "MCI": 1,
    "DEM": 1,
    "DEMENTIA": 1,
    "AD": 1,
}

_SPLIT_CONFIG: dict[str, dict[str, Any]] = {
    "train": {
        "directory": "1.Training",
        "activity_source": ("SourceData", "1.Gait", "train_activity.csv"),
        "sleep_source": ("SourceData", "2.Sleep", "train_sleep.csv"),
        "label_file": "training_label.csv",
        "expected_subjects": 141,
        "expected_class_counts": {0: 85, 1: 56},
    },
    "val": {
        "directory": "2.Validation",
        "activity_source": ("SourceData", "1.Gait", "val_activity.csv"),
        "sleep_source": ("SourceData", "2.Sleep", "val_sleep.csv"),
        "label_file": "val_label.csv",
        "expected_subjects": 33,
        "expected_class_counts": {0: 26, 1: 7},
    },
}


@dataclass(frozen=True)
class SubjectSequenceDataset:
    """One chronological, variable-length daily sequence per subject.

    ``sequences[i]`` has shape ``(observed_days_i, n_features)``.  Missing
    biological measurements are represented as NaN and must be imputed using
    fold-training statistics by the modelling code.  No sequence contains a
    padding row.
    """

    subject_ids: np.ndarray
    sequences: list[np.ndarray]
    feature_names: tuple[str, ...]
    y: np.ndarray | None
    audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        subject_ids = np.asarray(self.subject_ids)
        if subject_ids.ndim != 1:
            raise ValueError("subject_ids must be one-dimensional")
        if len(subject_ids) != len(self.sequences):
            raise ValueError("subject_ids and sequences have different lengths")
        if len(set(subject_ids.astype(str))) != len(subject_ids):
            raise ValueError("subject_ids must be unique")
        width = len(self.feature_names)
        if width == 0:
            raise ValueError("feature_names must not be empty")
        for index, sequence in enumerate(self.sequences):
            array = np.asarray(sequence)
            if array.ndim != 2 or array.shape[1] != width:
                raise ValueError(
                    f"sequence {index} must have shape (days, {width}); got {array.shape}"
                )
            if array.shape[0] == 0:
                raise ValueError(f"sequence {index} has no observations")
            if np.isinf(array).any():
                raise ValueError(f"sequence {index} contains infinite values")
        if self.y is not None:
            labels = np.asarray(self.y)
            if labels.shape != (len(subject_ids),):
                raise ValueError("y must contain one label per subject")
            if not set(np.unique(labels)).issubset({0, 1}):
                raise ValueError("binary y must contain only 0 and 1")


def _normalise_split(split: str) -> str:
    value = str(split).strip().lower()
    aliases = {
        "training": "train",
        "1.training": "train",
        "validation": "val",
        "valid": "val",
        "2.validation": "val",
    }
    value = aliases.get(value, value)
    if value not in _SPLIT_CONFIG:
        raise ValueError(f"split must be 'train' or 'val'; got {split!r}")
    return value


def resolve_split_root(data_root: str | Path, split: str) -> Path:
    """Resolve only the exact Activity/Sleep split root without tree scanning."""

    key = _normalise_split(split)
    root = Path(data_root).expanduser().resolve()
    expected = _SPLIT_CONFIG[key]["directory"]
    if root.name == expected:
        candidate = root
    elif (root / expected).is_dir():
        candidate = root / expected
    elif (root / "Data" / expected).is_dir():
        candidate = root / "Data" / expected
    else:
        raise FileNotFoundError(
            f"Could not resolve {expected} below {root}; pass the repository root, Data root, "
            "or the split root itself."
        )
    return candidate.resolve()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError as error:
            last_error = error
    raise RuntimeError(f"Unable to decode CSV: {path}") from last_error


def _pick_column(frame: pd.DataFrame, candidates: Sequence[str], purpose: str) -> str:
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(f"Missing {purpose} column; expected one of {list(candidates)}")


def _normalise_subject_ids(values: pd.Series, source_name: str) -> pd.Series:
    """Normalize identifiers while rejecting missing/stringified-null values."""

    if values.isna().any():
        raise ValueError(f"Missing subject identifier in {source_name}")
    normalized = values.astype(str).str.strip()
    invalid = normalized.str.lower().isin({"", "nan", "none", "null", "<na>"})
    if invalid.any():
        raise ValueError(f"Invalid subject identifier in {source_name}")
    return normalized


def _infer_split_key_from_root(split_root: Path) -> str:
    for key, config in _SPLIT_CONFIG.items():
        if split_root.name == config["directory"]:
            return key
    # Allow a caller-provided equivalent directory, but still test only the two
    # explicit safe label filenames below the two permitted modality folders.
    candidates = []
    for key, config in _SPLIT_CONFIG.items():
        filename = config["label_file"]
        if (
            (split_root / "LabelingData" / "1.Gait" / filename).is_file()
            and (split_root / "LabelingData" / "2.Sleep" / filename).is_file()
        ):
            candidates.append(key)
    if len(candidates) != 1:
        raise ValueError(f"Cannot infer train/val layout from split root: {split_root}")
    return candidates[0]


def _normalise_label_copy(frame: pd.DataFrame, source_name: str) -> pd.Series:
    id_column = _pick_column(
        frame,
        ("SAMPLE_EMAIL", "EMAIL", "patient_id", "subject_id"),
        f"subject identifier in {source_name}",
    )
    diagnosis_column = _pick_column(
        frame,
        ("DIAG_NM", "diagnosis", "label", "target"),
        f"diagnosis in {source_name}",
    )
    work = frame[[id_column, diagnosis_column]].copy()
    work.columns = ["subject_id", "diagnosis"]
    work["subject_id"] = _normalise_subject_ids(work["subject_id"], source_name)
    diagnosis = work["diagnosis"].astype(str).str.strip().str.upper()
    work["target"] = diagnosis.map(_LABEL_MAP)
    if work["target"].isna().any():
        raise ValueError(f"Invalid identifier or diagnosis in {source_name} label copy")
    conflicts = work.groupby("subject_id", sort=True)["target"].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"Conflicting duplicate labels in {source_name} label copy")
    output = (
        work.drop_duplicates("subject_id")
        .set_index("subject_id")["target"]
        .astype(np.int64)
        .sort_index()
    )
    output.index.name = "subject_id"
    output.name = "target"
    return output


def load_binary_labels(split_root: str | Path) -> pd.Series:
    """Load and cross-check only the Gait/Sleep label copies.

    Parameters
    ----------
    split_root:
        Exact ``Data/1.Training`` or ``Data/2.Validation`` directory (or an
        equivalent directory with the same two safe label subpaths).

    Returns
    -------
    pandas.Series
        Sorted ``int64`` Series named ``target`` and indexed by ``subject_id``.
        ``CN=0`` and ``MCI/DEM=1``.
    """

    root = Path(split_root).expanduser().resolve()
    key = _infer_split_key_from_root(root)
    filename = _SPLIT_CONFIG[key]["label_file"]
    gait_path = root / "LabelingData" / "1.Gait" / filename
    sleep_path = root / "LabelingData" / "2.Sleep" / filename
    gait = _normalise_label_copy(_read_csv(gait_path), "Gait")
    sleep = _normalise_label_copy(_read_csv(sleep_path), "Sleep")
    if not gait.index.equals(sleep.index):
        raise ValueError(
            "Gait/Sleep label copies contain different subject sets "
            f"(Gait={len(gait)}, Sleep={len(sleep)})"
        )
    if not np.array_equal(gait.to_numpy(), sleep.to_numpy()):
        raise ValueError("Gait/Sleep label copies disagree on at least one diagnosis")
    return gait.copy()


def _local_timestamp(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert(LOCAL_TIMEZONE)


def _numeric(value: object) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return np.nan
    return output if np.isfinite(output) else np.nan


def _parse_slash_sequence(value: object) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.empty(0, dtype=np.float64)
    text = str(value).strip()
    if not text or text in {"...", "nan", "None"}:
        return np.empty(0, dtype=np.float64)
    try:
        return np.fromiter(
            (float(token) for token in text.replace(",", "/").split("/") if token.strip()),
            dtype=np.float64,
        )
    except ValueError:
        parsed: list[float] = []
        for token in text.replace(",", "/").split("/"):
            try:
                parsed.append(float(token.strip()))
            except ValueError:
                parsed.append(np.nan)
        return np.asarray(parsed, dtype=np.float64)


def _sequence_from_row(row: Mapping[str, Any], preferred: str, fallback: str) -> np.ndarray:
    value = row.get(preferred)
    if value is None or str(value).strip() in {"", "...", "nan", "None"}:
        value = row.get(fallback)
    return _parse_slash_sequence(value)


def _entropy(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    _, frequencies = np.unique(finite, return_counts=True)
    probabilities = frequencies / frequencies.sum()
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def _adjacent_state_transition_rate(values: np.ndarray, valid_codes: Sequence[int]) -> float:
    """Transition rate over genuinely adjacent valid samples.

    Invalid/non-wear samples break adjacency instead of being deleted and
    silently joining the states on either side of a gap.
    """

    states = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(states) & np.isin(states, valid_codes)
    adjacent = valid[:-1] & valid[1:]
    if not adjacent.any():
        return np.nan
    return float(np.mean(states[:-1][adjacent] != states[1:][adjacent]))


def _state_longest_run_ratio(
    values: np.ndarray, code: int, valid_codes: Sequence[int]
) -> float:
    """Longest uninterrupted state run divided by the number of valid samples."""

    states = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(states) & np.isin(states, valid_codes)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return np.nan
    best = current = 0
    for state, is_valid in zip(states, valid):
        current = current + 1 if is_valid and state == code else 0
        best = max(best, current)
    return float(best / valid_count)


def _robust_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    names = ("mean", "std", "p10", "median", "p90", "iqr", "mad")
    if len(finite) == 0:
        return {f"{prefix}__{name}": np.nan for name in names}
    p10, q25, median, q75, p90 = np.quantile(finite, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        f"{prefix}__mean": float(np.mean(finite)),
        f"{prefix}__std": float(np.std(finite)),
        f"{prefix}__p10": float(p10),
        f"{prefix}__median": float(median),
        f"{prefix}__p90": float(p90),
        f"{prefix}__iqr": float(q75 - q25),
        f"{prefix}__mad": float(np.median(np.abs(finite - median))),
    }


def _clock_phase(value: object) -> tuple[float, float]:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return np.nan, np.nan
    local = timestamp.tz_convert(LOCAL_TIMEZONE)
    hour = local.hour + local.minute / 60.0 + local.second / 3600.0
    angle = 2.0 * np.pi * hour / 24.0
    return float(np.sin(angle)), float(np.cos(angle))


def _activity_sequence_features(row: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    classes = _sequence_from_row(row, ACTIVITY_CLASS_BLOB, "activity_class_5min")
    valid_codes = (1, 2, 3, 4, 5)
    valid_states = classes[np.isfinite(classes) & np.isin(classes, valid_codes)]
    output["activity__state__entropy"] = _entropy(valid_states)
    output["activity__state__transition_rate"] = _adjacent_state_transition_rate(
        classes, valid_codes
    )
    for code, name in ((1, "rest"), (2, "inactive"), (3, "low"), (4, "medium"), (5, "high")):
        output[f"activity__state__{name}_ratio"] = (
            float(np.mean(valid_states == code)) if len(valid_states) else np.nan
        )
        output[f"activity__state__{name}_longest_run_ratio"] = (
            _state_longest_run_ratio(classes, code, valid_codes)
        )

    met = _sequence_from_row(row, ACTIVITY_MET_BLOB, "activity_met_1min")
    # Code zero is acquisition/non-wear, and never becomes a model feature.  The
    # known wear stream is used only to remove the corresponding MET minutes.
    wear_by_minute = np.repeat(np.isin(classes, [1, 2, 3, 4, 5]), 5)
    masked_met = np.full(len(met), np.nan, dtype=np.float64)
    aligned = min(len(met), len(wear_by_minute))
    if aligned:
        keep = wear_by_minute[:aligned] & np.isfinite(met[:aligned])
        positions = np.flatnonzero(keep)
        masked_met[positions] = met[positions]
    output.update(_robust_stats(masked_met, "activity__met"))
    finite_met = masked_met[np.isfinite(masked_met)]
    output["activity__met__sedentary_ratio"] = (
        float(np.mean(finite_met <= 1.5)) if len(finite_met) else np.nan
    )
    output["activity__met__moderate_ratio"] = (
        float(np.mean((finite_met > 1.5) & (finite_met <= 3.0))) if len(finite_met) else np.nan
    )
    output["activity__met__vigorous_ratio"] = (
        float(np.mean(finite_met > 3.0)) if len(finite_met) else np.nan
    )

    hourly = np.full(24, np.nan, dtype=np.float64)
    if len(masked_met) == 1440:
        matrix = masked_met.reshape(24, 60)
        for hour in range(24):
            values = matrix[hour][np.isfinite(matrix[hour])]
            if len(values):
                hourly[hour] = float(np.mean(values))
        finite_hours = np.flatnonzero(np.isfinite(hourly))
        if len(finite_hours) >= 12:
            circular_x = np.r_[finite_hours - 24, finite_hours, finite_hours + 24]
            circular_y = np.tile(hourly[finite_hours], 3)
            hourly = np.interp(np.arange(24), circular_x, circular_y)
        else:
            hourly[:] = np.nan
    if np.isfinite(hourly).all():
        doubled = np.r_[hourly, hourly]
        m10 = np.asarray([doubled[index : index + 10].mean() for index in range(24)])
        l5 = np.asarray([doubled[index : index + 5].mean() for index in range(24)])
        m10_value, l5_value = float(m10.max()), float(l5.min())
        output["activity__circadian__m10"] = m10_value
        output["activity__circadian__l5"] = l5_value
        output["activity__circadian__relative_amplitude"] = (
            (m10_value - l5_value) / (m10_value + l5_value + 1e-8)
        )
        peak_offset = int(np.argmax(hourly))
        start = pd.to_datetime(row.get("activity_day_start"), errors="coerce", utc=True)
        if pd.isna(start):
            peak_sin = peak_cos = np.nan
        else:
            local = start.tz_convert(LOCAL_TIMEZONE) + pd.Timedelta(hours=peak_offset)
            angle = 2.0 * np.pi * (local.hour + local.minute / 60.0) / 24.0
            peak_sin, peak_cos = float(np.sin(angle)), float(np.cos(angle))
        output["activity__circadian__peak_sin"] = peak_sin
        output["activity__circadian__peak_cos"] = peak_cos
        spectrum = np.fft.rfft(hourly - hourly.mean())
        output["activity__circadian__first_harmonic_ratio"] = float(
            abs(spectrum[1]) / (np.abs(spectrum[1:]).sum() + 1e-8)
        )
    else:
        for name in (
            "m10",
            "l5",
            "relative_amplitude",
            "peak_sin",
            "peak_cos",
            "first_harmonic_ratio",
        ):
            output[f"activity__circadian__{name}"] = np.nan
    return output


def _sleep_sequence_features(row: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    raw_stages = _sequence_from_row(row, SLEEP_STAGE_BLOB, "sleep_hypnogram_5min")
    valid_codes = (1, 2, 3, 4)
    stages = raw_stages[np.isfinite(raw_stages) & np.isin(raw_stages, valid_codes)]
    output["sleep__stage__entropy"] = _entropy(stages)
    output["sleep__stage__transition_rate"] = _adjacent_state_transition_rate(
        raw_stages, valid_codes
    )
    for code, name in ((1, "deep"), (2, "light"), (3, "rem"), (4, "awake")):
        output[f"sleep__stage__{name}_ratio"] = (
            float(np.mean(stages == code)) if len(stages) else np.nan
        )
        output[f"sleep__stage__{name}_longest_run_ratio"] = (
            _state_longest_run_ratio(raw_stages, code, valid_codes)
        )

    hr = _sequence_from_row(row, SLEEP_HR_BLOB, "sleep_hr_5min")
    rmssd = _sequence_from_row(row, SLEEP_RMSSD_BLOB, "sleep_rmssd_5min")
    hr[(~np.isfinite(hr)) | (hr <= 0)] = np.nan
    rmssd[(~np.isfinite(rmssd)) | (rmssd <= 0)] = np.nan
    output.update(_robust_stats(hr, "sleep__hr"))
    output.update(_robust_stats(rmssd, "sleep__rmssd"))
    aligned = min(len(hr), len(rmssd))
    if aligned:
        keep = np.isfinite(hr[:aligned]) & np.isfinite(rmssd[:aligned])
        output["sleep__hr_rmssd__correlation"] = (
            float(np.corrcoef(hr[:aligned][keep], rmssd[:aligned][keep])[0, 1])
            if keep.sum() >= 4
            and np.std(hr[:aligned][keep]) > 0
            and np.std(rmssd[:aligned][keep]) > 0
            else np.nan
        )
    else:
        output["sleep__hr_rmssd__correlation"] = np.nan

    bedtime_sin, bedtime_cos = _clock_phase(row.get("sleep_bedtime_start"))
    wake_sin, wake_cos = _clock_phase(row.get("sleep_bedtime_end"))
    output.update(
        {
            "sleep__clock__bedtime_sin": bedtime_sin,
            "sleep__clock__bedtime_cos": bedtime_cos,
            "sleep__clock__wake_sin": wake_sin,
            "sleep__clock__wake_cos": wake_cos,
        }
    )

    duration = _numeric(row.get("sleep_duration"))
    for source in ("sleep_awake", "sleep_deep", "sleep_light", "sleep_rem", "sleep_total"):
        value = _numeric(row.get(source))
        name = source.removeprefix("sleep_")
        output[f"sleep__duration__{name}_ratio"] = (
            value / duration if np.isfinite(value) and np.isfinite(duration) and duration > 0 else np.nan
        )

    average, lowest = _numeric(row.get("sleep_hr_average")), _numeric(row.get("sleep_hr_lowest"))
    output["sleep__hr__drop"] = average - lowest if np.isfinite(average) and np.isfinite(lowest) else np.nan
    output["sleep__hr__drop_ratio"] = (
        (average - lowest) / average
        if np.isfinite(average) and np.isfinite(lowest) and average > 0
        else np.nan
    )
    positions = np.flatnonzero(np.isfinite(hr))
    if len(positions):
        lowest_position = int(positions[np.argmin(hr[positions])])
        output["sleep__hr__lowest_relative_position"] = lowest_position / max(1, len(hr) - 1)
        start = pd.to_datetime(row.get("sleep_bedtime_start"), errors="coerce", utc=True)
        if pd.isna(start):
            nadir_sin = nadir_cos = np.nan
        else:
            local = start.tz_convert(LOCAL_TIMEZONE) + pd.Timedelta(minutes=5 * lowest_position)
            angle = 2.0 * np.pi * (
                local.hour + local.minute / 60.0 + local.second / 3600.0
            ) / 24.0
            nadir_sin, nadir_cos = float(np.sin(angle)), float(np.cos(angle))
        output["sleep__hr__nadir_clock_sin"] = nadir_sin
        output["sleep__hr__nadir_clock_cos"] = nadir_cos
    else:
        output["sleep__hr__lowest_relative_position"] = np.nan
        output["sleep__hr__nadir_clock_sin"] = np.nan
        output["sleep__hr__nadir_clock_cos"] = np.nan

    def half_difference(values: np.ndarray) -> float:
        if len(values) < 4:
            return np.nan
        left, right = np.array_split(values, 2)
        left, right = left[np.isfinite(left)], right[np.isfinite(right)]
        return float(np.mean(right) - np.mean(left)) if len(left) and len(right) else np.nan

    output["sleep__hr__late_minus_early"] = half_difference(hr)
    output["sleep__rmssd__late_minus_early"] = half_difference(rmssd)
    return output


def _daily_features(row: Mapping[str, Any]) -> dict[str, float]:
    output: dict[str, float] = {}
    for source in ACTIVITY_SAFE_SCALARS:
        output[f"activity__scalar__{source.removeprefix('activity_')}"] = _numeric(row.get(source))
    for source in SLEEP_SAFE_SCALARS:
        output[f"sleep__scalar__{source.removeprefix('sleep_')}"] = _numeric(row.get(source))
    output.update(_activity_sequence_features(row))
    output.update(_sleep_sequence_features(row))
    return output


def _validate_feature_contract(feature_names: Sequence[str]) -> None:
    forbidden = (
        "email",
        "subject_id",
        "patient_id",
        "diagnos",
        "absolute_date",
        "period",
        "order",
        "count",
        "coverage",
        "mask",
        "nonwear",
        "non_wear",
    )
    bad = [name for name in feature_names if any(token in name.lower() for token in forbidden)]
    if bad:
        raise AssertionError(f"Forbidden metadata/acquisition feature names: {bad}")
    if len(set(feature_names)) != len(feature_names):
        raise AssertionError("Duplicate feature names")


def _prepare_sources(split_root: Path, split_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    config = _SPLIT_CONFIG[split_key]
    activity_path = split_root.joinpath(*config["activity_source"])
    sleep_path = split_root.joinpath(*config["sleep_source"])
    activity = _read_csv(activity_path)
    sleep = _read_csv(sleep_path)
    activity_source_rows = int(len(activity))
    sleep_source_rows = int(len(sleep))

    activity_id = _pick_column(activity, ("EMAIL", "SAMPLE_EMAIL"), "Activity subject ID")
    sleep_id = _pick_column(sleep, ("EMAIL", "SAMPLE_EMAIL"), "Sleep subject ID")
    required_activity = {"activity_day_start", "activity_day_end", *ACTIVITY_SAFE_SCALARS}
    required_sleep = {"sleep_bedtime_start", "sleep_bedtime_end", *SLEEP_SAFE_SCALARS}
    missing_activity = sorted(required_activity - set(activity.columns))
    missing_sleep = sorted(required_sleep - set(sleep.columns))
    if missing_activity or missing_sleep:
        raise KeyError(
            f"Missing source columns: activity={missing_activity}, sleep={missing_sleep}"
        )
    missing_sequence_sources = {
        "activity": [
            list(alternatives)
            for alternatives in ACTIVITY_SEQUENCE_SOURCE_ALTERNATIVES
            if not any(column in activity.columns for column in alternatives)
        ],
        "sleep": [
            list(alternatives)
            for alternatives in SLEEP_SEQUENCE_SOURCE_ALTERNATIVES
            if not any(column in sleep.columns for column in alternatives)
        ],
    }
    if any(missing_sequence_sources.values()):
        raise KeyError(
            "Missing sequence source alternatives: "
            f"activity={missing_sequence_sources['activity']}, "
            f"sleep={missing_sequence_sources['sleep']}"
        )

    activity = activity.copy()
    activity["_subject_id"] = _normalise_subject_ids(
        activity[activity_id], "Activity source"
    )
    activity["_timestamp"] = _local_timestamp(activity["activity_day_start"])
    activity["_end_timestamp"] = _local_timestamp(activity["activity_day_end"])
    activity["_local_date"] = activity["_timestamp"].dt.date
    invalid_activity_time = int(
        (activity["_timestamp"].isna() | activity["_end_timestamp"].isna()).sum()
    )
    activity_duration_seconds = (
        activity["_end_timestamp"] - activity["_timestamp"]
    ).dt.total_seconds()
    valid_activity_interval = (
        activity["_timestamp"].notna()
        & activity["_end_timestamp"].notna()
        & activity_duration_seconds.between(23 * 60 * 60, 25 * 60 * 60)
    )
    invalid_activity_interval_rows = int((~valid_activity_interval).sum())
    activity = activity.loc[valid_activity_interval].copy()
    activity_duplicate_days = int(activity.duplicated(["_subject_id", "_local_date"]).sum())
    activity = (
        activity.sort_values(["_subject_id", "_local_date", "_timestamp"], kind="mergesort")
        .drop_duplicates(["_subject_id", "_local_date"], keep="last")
    )

    sleep = sleep.copy()
    sleep["_subject_id"] = _normalise_subject_ids(sleep[sleep_id], "Sleep source")
    sleep["_timestamp"] = _local_timestamp(sleep["sleep_bedtime_end"])
    sleep["_local_date"] = sleep["_timestamp"].dt.date
    sleep["_duration_for_selection"] = pd.to_numeric(sleep["sleep_duration"], errors="coerce")
    valid_sleep = (
        sleep["_timestamp"].notna()
        & sleep["_duration_for_selection"].gt(0)
        & sleep["_duration_for_selection"].le(24 * 60 * 60)
    )
    invalid_sleep_rows = int((~valid_sleep).sum())
    sleep = sleep.loc[valid_sleep].copy()
    sleep_duplicate_days = int(sleep.duplicated(["_subject_id", "_local_date"]).sum())
    # Longest valid sleep is the deterministic representative for that end date.
    sleep = (
        sleep.sort_values(
            ["_subject_id", "_local_date", "_duration_for_selection", "_timestamp"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["_subject_id", "_local_date"], keep="first")
    )

    aligned = activity.merge(
        sleep,
        on=["_subject_id", "_local_date"],
        how="inner",
        suffixes=("", "__sleep_collision"),
        validate="one_to_one",
    )
    aligned = aligned.sort_values(["_subject_id", "_local_date"], kind="mergesort")
    sleep_timestamp_column = "_timestamp__sleep_collision"
    sleep_after_activity_interval = aligned[sleep_timestamp_column].gt(
        aligned["_end_timestamp"]
    )
    invalid_cross_modal_intervals = int(sleep_after_activity_interval.sum())
    if invalid_cross_modal_intervals:
        raise ValueError(
            f"{invalid_cross_modal_intervals} sleep end timestamps occur after the "
            "paired activity-day interval"
        )
    audit = {
        "activity_source_rows": activity_source_rows,
        "sleep_source_rows": sleep_source_rows,
        "invalid_activity_timestamp_rows": invalid_activity_time,
        "invalid_activity_interval_rows": invalid_activity_interval_rows,
        "invalid_sleep_rows": invalid_sleep_rows,
        "invalid_cross_modal_intervals": invalid_cross_modal_intervals,
        "activity_duplicate_subject_days": activity_duplicate_days,
        "sleep_duplicate_subject_days": sleep_duplicate_days,
        "aligned_daily_rows": int(len(aligned)),
        "activity_subjects_after_daily_selection": int(activity["_subject_id"].nunique()),
        "sleep_subjects_after_daily_selection": int(sleep["_subject_id"].nunique()),
        "aligned_subjects": int(aligned["_subject_id"].nunique()),
    }
    return aligned, audit


def load_subject_sequence_dataset(
    data_root: str | Path,
    split: str,
    *,
    include_labels: bool = True,
    min_observations: int = DEFAULT_SEQUENCE_LENGTH,
    strict_contract: bool = False,
) -> SubjectSequenceDataset:
    """Build one variable-length Activity/Sleep sequence per subject.

    ``include_labels=False`` is the safe prediction mode for the historical
    validation split.  Predictions can be written first, and labels loaded only
    afterwards with :func:`load_binary_labels` for a frozen evaluation.
    """

    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    split_key = _normalise_split(split)
    split_root = resolve_split_root(data_root, split_key)
    aligned, source_audit = _prepare_sources(split_root, split_key)
    if aligned.empty:
        raise ValueError("No Activity/Sleep subject-day pairs were aligned")

    labels = load_binary_labels(split_root) if include_labels else None
    aligned_subjects = set(aligned["_subject_id"].astype(str))
    if labels is not None and aligned_subjects != set(labels.index.astype(str)):
        raise ValueError(
            "Aligned Activity/Sleep subjects differ from the cross-checked label subjects "
            f"(aligned={len(aligned_subjects)}, labels={len(labels)})"
        )

    subject_ids: list[str] = []
    sequences: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    short_subjects = 0
    for subject_id, subject_frame in aligned.groupby("_subject_id", sort=True):
        rows: list[list[float]] = []
        for row in subject_frame.to_dict(orient="records"):
            features = _daily_features(row)
            current_names = tuple(features)
            if feature_names is None:
                feature_names = current_names
                _validate_feature_contract(feature_names)
            elif current_names != feature_names:
                raise AssertionError("Daily feature order changed across observations")
            rows.append(list(features.values()))
        matrix = np.asarray(rows, dtype=np.float32)
        matrix[np.isinf(matrix)] = np.nan
        if len(matrix) < min_observations:
            short_subjects += 1
            continue
        subject_ids.append(str(subject_id))
        sequences.append(matrix)

    if short_subjects:
        raise ValueError(
            f"{short_subjects} subjects have fewer than {min_observations} aligned observations; "
            "padding or length-based selection is intentionally disallowed"
        )
    if feature_names is None or not subject_ids:
        raise ValueError("No subject sequences were constructed")

    subject_array = np.asarray(subject_ids, dtype=str)
    y = labels.loc[subject_ids].to_numpy(dtype=np.int64) if labels is not None else None
    lengths = np.asarray([len(sequence) for sequence in sequences], dtype=int)
    class_counts = (
        {str(label): int(np.sum(y == label)) for label in (0, 1)} if y is not None else None
    )
    audit: dict[str, Any] = {
        "split": split_key,
        "labels_loaded": bool(include_labels),
        "label_contract": "Gait/Sleep copies only; CN=0, MCI/DEM=1",
        "date_alignment": (
            "activity_day_start local date == sleep_bedtime_end local date; "
            "23-25h activity interval and sleep_end <= activity_day_end validated"
        ),
        "sleep_selection": "longest valid sleep per subject/end-date",
        "minimum_required_observations": int(min_observations),
        "retained_subjects": int(len(subject_ids)),
        "minimum_retained_observations": int(lengths.min()),
        "median_retained_observations": float(np.median(lengths)),
        "maximum_retained_observations": int(lengths.max()),
        "feature_count": int(len(feature_names)),
        "class_counts": class_counts,
        "model_input_excludes": [
            "identifiers",
            "diagnosis",
            "absolute dates",
            "acquisition period/order",
            "observation counts/coverage/masks",
            "non-wear",
        ],
        **source_audit,
    }

    if strict_contract:
        expected_subjects = int(_SPLIT_CONFIG[split_key]["expected_subjects"])
        if len(subject_ids) != expected_subjects:
            raise AssertionError(
                f"{split_key} subject contract failed: expected {expected_subjects}, got {len(subject_ids)}"
            )
        if y is not None:
            expected_counts = _SPLIT_CONFIG[split_key]["expected_class_counts"]
            observed = {label: int(np.sum(y == label)) for label in (0, 1)}
            if observed != expected_counts:
                raise AssertionError(
                    f"{split_key} class contract failed: expected {expected_counts}, got {observed}"
                )

    return SubjectSequenceDataset(
        subject_ids=subject_array,
        sequences=sequences,
        feature_names=feature_names,
        y=y,
        audit=audit,
    )


def build_subject_dataset(
    split_root: str | Path,
    require_labels: bool,
    expected_split: str | None = None,
) -> SubjectSequenceDataset:
    """Build a split dataset from an exact Training/Validation directory.

    When ``expected_split`` is supplied, both the split identity and the locked
    141/33 subject contract (plus class counts when labels are requested) are
    checked.  Omitting it keeps the same leakage-safe feature contract but skips
    the dataset-release-specific cardinality assertion.
    """

    root = Path(split_root).expanduser().resolve()
    inferred = _infer_split_key_from_root(root)
    if expected_split is not None:
        expected = _normalise_split(expected_split)
        if inferred != expected:
            raise AssertionError(
                f"Expected {expected!r} split but resolved {inferred!r} from {root.name}"
            )
    return load_subject_sequence_dataset(
        root,
        inferred,
        include_labels=bool(require_labels),
        min_observations=DEFAULT_SEQUENCE_LENGTH,
        strict_contract=expected_split is not None,
    )


def make_fixed_views(
    sequences: Sequence[np.ndarray],
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    n_views: int = DEFAULT_N_VIEWS,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the same number of deterministic contiguous crops per subject.

    Crop starts are evenly spread across each subject's observed sequence.  The
    returned mapping contains only subject indices ``0..n_subjects-1`` and is
    used for equal-weight probability aggregation.  No padding, mask,
    observation count, crop position, or sequence length is appended to views.
    A subject with exactly ``sequence_length`` observations receives repeated
    identical crops so that every subject still contributes ``n_views`` rows.
    """

    if sequence_length < 1 or n_views < 1:
        raise ValueError("sequence_length and n_views must be positive")
    sequences = list(sequences)
    if not sequences:
        raise ValueError("At least one subject sequence is required")

    crops: list[np.ndarray] = []
    subject_mapping: list[int] = []
    expected_width: int | None = None
    for subject_index, sequence in enumerate(sequences):
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"sequence {subject_index} is not two-dimensional")
        if expected_width is None:
            expected_width = int(array.shape[1])
        elif array.shape[1] != expected_width:
            raise ValueError("All sequences must have the same feature width")
        maximum_start = len(array) - sequence_length
        if maximum_start < 0:
            raise ValueError(
                f"sequence {subject_index} has {len(array)} observations; {sequence_length} required"
            )
        if n_views == 1:
            crop_starts = np.asarray([maximum_start // 2], dtype=int)
        else:
            crop_starts = np.rint(np.linspace(0, maximum_start, n_views)).astype(int)
        for start in crop_starts:
            crops.append(array[start : start + sequence_length].copy())
            subject_mapping.append(subject_index)
    views = np.stack(crops).astype(np.float32)
    mapping = np.asarray(subject_mapping, dtype=np.int64)
    expected_rows = len(sequences) * n_views
    if views.shape[0] != expected_rows or mapping.shape != (expected_rows,):
        raise AssertionError("Fixed-view cardinality contract failed")
    observed = np.bincount(mapping, minlength=len(sequences))
    if not np.array_equal(observed, np.full(len(sequences), n_views, dtype=int)):
        raise AssertionError("Every subject must receive exactly n_views crops")
    return views, mapping


__all__ = [
    "ACTIVITY_SAFE_SCALARS",
    "DEFAULT_N_VIEWS",
    "DEFAULT_SEQUENCE_LENGTH",
    "SLEEP_SAFE_SCALARS",
    "SubjectSequenceDataset",
    "build_subject_dataset",
    "load_binary_labels",
    "load_subject_sequence_dataset",
    "make_fixed_views",
    "resolve_split_root",
]
