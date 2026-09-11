"""Deterministic, row-wise construction of the 72 features of Cheon et al. (2025), Figure 2 (p.164).

Every feature of a record depends only on that record's own raw values (no dataset-level statistics),
so this step is legitimately applied before any train/test split (prompt §10: "raw deterministic
feature creation -> allowed"). Operational choices: ASSUMPTIONS.md A06-A09.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- Figure 2, left column (activity scalars, 24) --------------------------------------------------
ACTIVITY_SCALARS = [
    "activity_average_met", "activity_cal_active", "activity_cal_total", "activity_daily_movement",
    "activity_high", "activity_inactive", "activity_inactivity_alerts", "activity_low", "activity_medium",
    "activity_met_min_high", "activity_met_min_inactive", "activity_met_min_low", "activity_met_min_medium",
    "activity_non_wear", "activity_rest", "activity_score", "activity_score_meet_daily_targets",
    "activity_score_move_every_hour", "activity_score_recovery_time", "activity_score_stay_active",
    "activity_score_training_frequency", "activity_score_training_volume", "activity_steps", "activity_total",
]
# --- Figure 2, sleep scalars (27) + derived sleep_time, in figure order ----------------------------
SLEEP_BLOCK = [
    "sleep_awake", "sleep_bedtime_end", "sleep_bedtime_start", "sleep_time", "sleep_breath_average",
    "sleep_deep", "sleep_duration", "sleep_efficiency", "sleep_hr_average", "sleep_hr_lowest", "sleep_light",
    "sleep_midpoint_at_delta", "sleep_midpoint_time", "sleep_onset_latency", "sleep_period_id", "sleep_rem",
    "sleep_restless", "sleep_rmssd", "sleep_score", "sleep_score_alignment", "sleep_score_deep",
    "sleep_score_disturbances", "sleep_score_efficiency", "sleep_score_latency", "sleep_score_rem",
    "sleep_score_total", "sleep_temperature_delta", "sleep_total",
]
MET_STATS = ["std", "variance", "kurtosis", "skewness", "mean", "median", "min", "max",
             "autocorrelation", "quantile_25", "quantile_50", "quantile_75"]
MET_FEATURES = [f"activity_met_1min_{s}" for s in MET_STATS]
CLASS_COUNT_FEATURES = [f"activity_class_5min_count_{i}" for i in (1, 2, 3, 4)]
HYPNO_COUNT_FEATURES = [f"sleep_hypnogram_5min_count_{i}" for i in (1, 2, 3, 4)]

PAPER_FEATURES = ACTIVITY_SCALARS + SLEEP_BLOCK + MET_FEATURES + CLASS_COUNT_FEATURES + HYPNO_COUNT_FEATURES
assert len(PAPER_FEATURES) == 72 and len(set(PAPER_FEATURES)) == 72

RAW_SERIES_COLUMNS = {
    "met_1min": "CONVERT(activity_met_1min USING utf8)",
    "class_5min": "CONVERT(activity_class_5min USING utf8)",
    "hypnogram_5min": "CONVERT(sleep_hypnogram_5min USING utf8)",
}
# Columns the paper removed or never listed (kept out of the model input by construction).
NOT_USED_RAW = ["activity_day_start", "activity_day_end", "activity_class_5min", "activity_met_1min",
                "sleep_hr_5min", "sleep_hypnogram_5min", "sleep_rmssd_5min", "sleep_is_longest",
                "sleep_temperature_deviation", "CONVERT(sleep_hr_5min USING utf8)",
                "CONVERT(sleep_rmssd_5min USING utf8)"]
FORBIDDEN_FEATURE_TOKENS = ("mmse", "diag", "q0", "q1", "total_score", "email", "subject", "doctor")


def parse_series(s: str) -> np.ndarray:
    """'1.2/0.9/1/' -> array([1.2, 0.9, 1.0]); a trailing separator is dropped; empty inner tokens -> NaN."""
    toks = str(s).split("/")
    if toks and toks[-1] == "":
        toks = toks[:-1]
    return np.array([float(t) if t != "" else np.nan for t in toks], dtype=float)


def clock_hours(ts: pd.Series) -> pd.Series:
    """Timestamp -> real number on the 24-hour clock (local time, A06): 05:10:28 -> 5.1744."""
    t = pd.to_datetime(ts, utc=True).dt.tz_convert("Asia/Seoul")
    return (t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0).astype(float)


def _met_stats(arr: np.ndarray) -> list[float]:
    s = pd.Series(arr, dtype=float)  # pandas defaults (A08): ddof=1, Fisher kurtosis, lag-1 autocorr, linear quantiles
    return [
        s.std(), s.var(), s.kurt(), s.skew(), s.mean(), s.median(), s.min(), s.max(),
        s.autocorr(lag=1), s.quantile(0.25), s.quantile(0.50), s.quantile(0.75),
    ]


def build_features(raw: pd.DataFrame, sleep_time_definition: str = "clock_difference") -> pd.DataFrame:
    """Return DataFrame[record_id, subject_id, y, diag, partition, <72 paper features>].

    sleep_time_definition (A07): "clock_difference" = converted end clock - converted start clock (literal reading,
    negative when the night crosses midnight); "timestamp_difference" = true duration in hours (== sleep_duration/3600).
    """
    out = pd.DataFrame(index=raw.index)
    for c in ("record_id", "subject_id", "y", "diag", "partition"):
        if c in raw.columns:
            out[c] = raw[c].to_numpy()

    for c in ACTIVITY_SCALARS:
        out[c] = raw[c].astype(float).to_numpy()
    for c in SLEEP_BLOCK:
        if c in ("sleep_bedtime_end", "sleep_bedtime_start", "sleep_time"):
            continue
        out[c] = raw[c].astype(float).to_numpy()

    end_utc = pd.to_datetime(raw["sleep_bedtime_end"], utc=True)
    start_utc = pd.to_datetime(raw["sleep_bedtime_start"], utc=True)
    out["sleep_bedtime_end"] = clock_hours(raw["sleep_bedtime_end"]).to_numpy()
    out["sleep_bedtime_start"] = clock_hours(raw["sleep_bedtime_start"]).to_numpy()
    if sleep_time_definition == "clock_difference":
        out["sleep_time"] = out["sleep_bedtime_end"] - out["sleep_bedtime_start"]  # A07 (literal reading)
    elif sleep_time_definition == "timestamp_difference":
        out["sleep_time"] = ((end_utc - start_utc).dt.total_seconds() / 3600.0).to_numpy()  # hours
    else:
        raise ValueError(f"unknown sleep_time_definition {sleep_time_definition!r}")

    met = raw[RAW_SERIES_COLUMNS["met_1min"]].map(parse_series)
    met_stats = np.array([_met_stats(a) for a in met], dtype=float)
    for j, name in enumerate(MET_FEATURES):
        out[name] = met_stats[:, j]

    cls = raw[RAW_SERIES_COLUMNS["class_5min"]].map(parse_series)
    for i, name in zip((1, 2, 3, 4), CLASS_COUNT_FEATURES):
        out[name] = np.array([int(np.sum(a == i)) for a in cls], dtype=float)

    hyp = raw[RAW_SERIES_COLUMNS["hypnogram_5min"]].map(parse_series)
    for i, name in zip((1, 2, 3, 4), HYPNO_COUNT_FEATURES):
        out[name] = np.array([int(np.sum(a == i)) for a in hyp], dtype=float)

    # order columns: meta first, then the 72 features in Figure 2 order
    meta = [c for c in ("record_id", "subject_id", "y", "diag", "partition") if c in out.columns]
    out = out[meta + PAPER_FEATURES]
    check_no_forbidden(PAPER_FEATURES)  # model inputs only; meta columns (label, hashed id) are never features
    return out


def check_no_forbidden(columns) -> None:
    bad = [c for c in columns if any(tok in c.lower() for tok in FORBIDDEN_FEATURE_TOKENS)
           and c not in ("subject_id",)]
    if bad:
        raise RuntimeError(f"Forbidden (diagnosis/cognitive-test/identifier) columns in feature table: {bad}")


def feature_matrix(feat: pd.DataFrame, features: list[str] | None = None) -> np.ndarray:
    features = PAPER_FEATURES if features is None else list(features)
    return feat[features].to_numpy(dtype=float)
