"""Person-level feature construction (the PDF's aggregation stage, corrected).

The report's key structural idea is kept: collapse every person's multi-day
wearable record into **exactly one row**, so that no window of the same person
can straddle a train/test boundary.  On top of that the report parses the
high-resolution intraday arrays (1-min MET, 5-min HR, 5-min RMSSD, 5-min
hypnogram) and adds sleep-stage transition counts and coefficient-of-variation
("irregularity") features.  All of that is reproduced here.

Three corrections were applied relative to the report:

1. ``sleep_hr_5min`` and ``sleep_rmssd_5min`` use ``0`` as a *missing* sentinel
   (the sensor reports 0 when it cannot measure).  The report averages those
   zeros in, which drags every HR/HRV mean toward zero by an amount that depends
   on how much missing data a person has.  Here 0 becomes NaN.
2. No feature may encode *how many days* a person wore the device.  Observation
   count ranges from 35 to 120 days and is a plausible label proxy, so counts
   are computed for the audit log only and never enter the matrix.
3. Absolute dates are dropped.  Bedtime timestamps survive only as circadian
   angles (sin/cos of local hour-of-day), which carry phase but not calendar
   position.

Everything in this module is a pure row-wise transform of a single person's own
data -- nothing is fitted across people -- so it is safe to run once, before any
cross-validation split.  Every *fitted* step lives in :mod:`preprocessing`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import PERSON_KEY, assert_feature_names_clean

# ---------------------------------------------------------------- array specs --
#: ``CONVERT(... USING utf8)`` columns hold the real payload; the same-named
#: plain columns are truncated ``'...'`` placeholders in this dataset.
ACTIVITY_CLASS_COLUMN = "CONVERT(activity_class_5min USING utf8)"
ACTIVITY_MET_COLUMN = "CONVERT(activity_met_1min USING utf8)"
SLEEP_HR_COLUMN = "CONVERT(sleep_hr_5min USING utf8)"
SLEEP_HYPNOGRAM_COLUMN = "CONVERT(sleep_hypnogram_5min USING utf8)"
SLEEP_RMSSD_COLUMN = "CONVERT(sleep_rmssd_5min USING utf8)"

#: Columns that are strings, placeholders or absolute timestamps.  None of these
#: may be treated as a numeric daily scalar.
ACTIVITY_NON_SCALAR = {
    PERSON_KEY, "EMAIL", "activity_day_start", "activity_day_end",
    "activity_class_5min", "activity_met_1min",
    ACTIVITY_CLASS_COLUMN, ACTIVITY_MET_COLUMN,
}
SLEEP_NON_SCALAR = {
    PERSON_KEY, "EMAIL", "sleep_bedtime_start", "sleep_bedtime_end",
    "sleep_hr_5min", "sleep_hypnogram_5min", "sleep_rmssd_5min",
    SLEEP_HR_COLUMN, SLEEP_HYPNOGRAM_COLUMN, SLEEP_RMSSD_COLUMN,
}

#: Oura activity classes.
ACTIVITY_CLASS_LABELS = {
    0: "nonwear", 1: "rest", 2: "inactive", 3: "low", 4: "medium", 5: "high",
}
#: Oura hypnogram stages.
SLEEP_STAGE_LABELS = {1: "deep", 2: "light", 3: "rem", 4: "awake"}


# ------------------------------------------------------------------- parsing --
def parse_slash_array(raw, *, zero_is_missing: bool = False) -> np.ndarray:
    """Parse a ``/``-separated intraday string into a float array.

    Unparseable tokens become NaN.  When ``zero_is_missing`` the sensor's 0
    sentinel is converted to NaN as well.
    """

    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return np.empty(0, dtype=np.float64)
    text = str(raw).strip().strip("/")
    if not text or text == "...":
        return np.empty(0, dtype=np.float64)
    values = pd.to_numeric(pd.Series(text.split("/")), errors="coerce").to_numpy(dtype=np.float64)
    if zero_is_missing:
        values = np.where(values == 0.0, np.nan, values)
    return values


def _finite(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    return values[np.isfinite(values)]


# ---------------------------------------------------------------- statistics --
FULL_STATS = ("mean", "std", "var", "min", "max", "median", "q25", "q75", "iqr")
DAILY_STATS = ("mean", "std", "median", "iqr")


def summarize(values: np.ndarray, prefix: str, stats=FULL_STATS) -> dict[str, float]:
    """Distribution summary of a 1-D array; all-NaN input yields all-NaN output."""

    clean = _finite(np.asarray(values, dtype=np.float64))
    out: dict[str, float] = {}
    if clean.size == 0:
        for stat in stats:
            out[f"{prefix}_{stat}"] = np.nan
        return out

    q25, q75 = np.percentile(clean, [25, 75])
    available = {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if clean.size > 1 else np.nan,
        "var": float(np.var(clean, ddof=1)) if clean.size > 1 else np.nan,
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "median": float(np.median(clean)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }
    for stat in stats:
        out[f"{prefix}_{stat}"] = available[stat]
    return out


def _transition_features(sequence: np.ndarray, labels: dict[int, str], prefix: str) -> dict[str, float]:
    """Stage/class transition *rates* plus a fragmentation index.

    Rates, not raw counts: a raw count grows with recording length and would
    smuggle observation volume into the matrix.
    """

    codes = sorted(labels)
    out = {f"{prefix}_trans_{labels[a]}_to_{labels[b]}": np.nan
           for a in codes for b in codes if a != b}
    out[f"{prefix}_trans_rate"] = np.nan
    out[f"{prefix}_fragmentation"] = np.nan

    clean = _finite(sequence)
    if clean.size < 2:
        return out
    values = clean.astype(np.int64)
    previous, following = values[:-1], values[1:]
    total = float(previous.size)

    changed = previous != following
    out[f"{prefix}_trans_rate"] = float(np.count_nonzero(changed) / total)
    for a in codes:
        for b in codes:
            if a == b:
                continue
            mask = (previous == a) & (following == b)
            out[f"{prefix}_trans_{labels[a]}_to_{labels[b]}"] = float(np.count_nonzero(mask) / total)

    # Mean run length, inverted: high value == highly fragmented.
    run_starts = int(np.count_nonzero(changed)) + 1
    out[f"{prefix}_fragmentation"] = float(run_starts / values.size)
    return out


def _fraction_features(sequence: np.ndarray, labels: dict[int, str], prefix: str) -> dict[str, float]:
    out = {f"{prefix}_frac_{name}": np.nan for name in labels.values()}
    clean = _finite(sequence)
    if clean.size == 0:
        return out
    values = clean.astype(np.int64)
    for code, name in labels.items():
        out[f"{prefix}_frac_{name}"] = float(np.count_nonzero(values == code) / values.size)
    return out


def _circadian_features(timestamps: pd.Series, prefix: str) -> dict[str, float]:
    """Phase-only summary of a timestamp column (no calendar position)."""

    parsed = pd.to_datetime(timestamps, errors="coerce", utc=True, format="ISO8601")
    local = pd.to_datetime(timestamps, errors="coerce", format="ISO8601")
    hours = local.dt.hour + local.dt.minute / 60.0
    hours = hours.dropna().to_numpy(dtype=np.float64)
    out = {
        f"{prefix}_hour_sin_mean": np.nan,
        f"{prefix}_hour_cos_mean": np.nan,
        f"{prefix}_hour_regularity": np.nan,
    }
    if hours.size == 0 or parsed.isna().all():
        return out
    angles = 2.0 * np.pi * hours / 24.0
    sin_mean, cos_mean = float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles)))
    out[f"{prefix}_hour_sin_mean"] = sin_mean
    out[f"{prefix}_hour_cos_mean"] = cos_mean
    # Resultant length of the circular mean: 1.0 == perfectly regular schedule.
    out[f"{prefix}_hour_regularity"] = float(np.hypot(sin_mean, cos_mean))
    return out


# ------------------------------------------------------------ per-person rows --
def _person_activity_features(group: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}

    scalars = [c for c in group.columns if c not in ACTIVITY_NON_SCALAR]
    for column in scalars:
        series = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
        out.update(summarize(series, f"act_{column}", DAILY_STATS))

    met = np.concatenate([parse_slash_array(v) for v in group[ACTIVITY_MET_COLUMN]] or [np.empty(0)])
    out.update(summarize(met, "act_met1min", FULL_STATS))
    # Time above light-activity threshold (MET > 3 == moderate/vigorous).
    finite_met = _finite(met)
    out["act_met1min_frac_above_3"] = (
        float(np.count_nonzero(finite_met > 3.0) / finite_met.size) if finite_met.size else np.nan
    )

    classes = np.concatenate(
        [parse_slash_array(v) for v in group[ACTIVITY_CLASS_COLUMN]] or [np.empty(0)]
    )
    out.update(_fraction_features(classes, ACTIVITY_CLASS_LABELS, "act_class"))
    out.update(_transition_features(classes, ACTIVITY_CLASS_LABELS, "act_class"))

    if "activity_day_start" in group.columns:
        out.update(_circadian_features(group["activity_day_start"], "act_daystart"))
    return out


def _person_sleep_features(group: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}

    scalars = [c for c in group.columns if c not in SLEEP_NON_SCALAR]
    for column in scalars:
        series = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64)
        out.update(summarize(series, f"slp_{column}", DAILY_STATS))

    # 0 is the sensor's "unmeasurable" sentinel for HR and RMSSD.
    heart_rate = np.concatenate(
        [parse_slash_array(v, zero_is_missing=True) for v in group[SLEEP_HR_COLUMN]] or [np.empty(0)]
    )
    out.update(summarize(heart_rate, "slp_hr5min", FULL_STATS))

    rmssd = np.concatenate(
        [parse_slash_array(v, zero_is_missing=True) for v in group[SLEEP_RMSSD_COLUMN]] or [np.empty(0)]
    )
    out.update(summarize(rmssd, "slp_rmssd5min", FULL_STATS))

    hypnogram = np.concatenate(
        [parse_slash_array(v) for v in group[SLEEP_HYPNOGRAM_COLUMN]] or [np.empty(0)]
    )
    out.update(_fraction_features(hypnogram, SLEEP_STAGE_LABELS, "slp_stage"))
    out.update(_transition_features(hypnogram, SLEEP_STAGE_LABELS, "slp_stage"))

    for column, prefix in (("sleep_bedtime_start", "slp_bedstart"),
                           ("sleep_bedtime_end", "slp_bedend")):
        if column in group.columns:
            out.update(_circadian_features(group[column], prefix))
    return out


def _aggregate(frame: pd.DataFrame, builder) -> pd.DataFrame:
    rows, ids = [], []
    for person_id, group in frame.groupby(PERSON_KEY, sort=True):
        ids.append(person_id)
        rows.append(builder(group))
    out = pd.DataFrame(rows, index=pd.Index(ids, name=PERSON_KEY))
    return out.reset_index()


# ------------------------------------------------------ coefficient of variation
def add_coefficient_of_variation(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``<channel>_cv = std / |mean|`` for every channel that has both.

    This is the report's V29 "irregularity" idea.  It is a row-wise ratio of two
    columns of the same person, so it fits nothing across people.
    """

    out = frame.copy()
    new_columns: dict[str, np.ndarray] = {}
    for column in frame.columns:
        if not column.endswith("_std"):
            continue
        mean_column = column[: -len("_std")] + "_mean"
        if mean_column not in frame.columns:
            continue
        std = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        mean = pd.to_numeric(frame[mean_column], errors="coerce").to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(mean) > 1e-9, std / np.abs(mean), np.nan)
        new_columns[column[: -len("_std")] + "_cv"] = ratio
    if new_columns:
        out = pd.concat([out, pd.DataFrame(new_columns, index=frame.index)], axis=1)
    return out


def replace_infinities(frame: pd.DataFrame) -> pd.DataFrame:
    """inf/-inf -> NaN, so YDF's native missing-value splits handle them."""

    numeric = frame.select_dtypes(include=[np.number]).columns
    out = frame.copy()
    out[numeric] = out[numeric].replace([np.inf, -np.inf], np.nan)
    return out


# ------------------------------------------------------------------ public API --
def build_person_features(
    activity: pd.DataFrame,
    sleep: pd.DataFrame,
    mmse: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Collapse day-level tables into one row per person.

    Returns ``(features, audit)``.  ``audit`` carries the per-person observation
    counts, which are reported but deliberately kept **out** of ``features``.
    """

    activity_features = _aggregate(activity, _person_activity_features)
    sleep_features = _aggregate(sleep, _person_sleep_features)

    features = activity_features.merge(sleep_features, on=PERSON_KEY, how="outer")
    if mmse is not None:
        features = features.merge(mmse, on=PERSON_KEY, how="left")

    features = replace_infinities(features)
    features = add_coefficient_of_variation(features)
    features = replace_infinities(features)

    # Drop columns that are entirely missing or perfectly constant across the
    # whole split -- they carry no signal and destabilise scaling.
    payload = features.drop(columns=[PERSON_KEY])
    keep = [c for c in payload.columns
            if payload[c].notna().any() and payload[c].nunique(dropna=True) > 1]
    features = features[[PERSON_KEY] + keep]

    assert_feature_names_clean(features.columns)

    day_counts = activity.groupby(PERSON_KEY).size()
    audit = {
        "n_persons": int(features.shape[0]),
        "n_features": int(features.shape[1] - 1),
        "observation_days": {
            "min": int(day_counts.min()), "median": float(day_counts.median()),
            "max": int(day_counts.max()),
        },
        "dropped_all_nan_or_constant": int(payload.shape[1] - len(keep)),
    }
    return features, audit
