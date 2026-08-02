"""Leakage-audited core for the CN/MCI/DEM next-stage Colab experiments.

This module deliberately contains no repository-specific paths and never reads
official validation labels. Notebook 01 supplies training data; notebook 02
supplies a frozen benchmark contract.
"""
from __future__ import annotations

import hashlib
import gc
import json
import math
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import kurtosis, trim_mean

CLASS_NAMES = ["CN", "MCI", "Dem"]
CLASS_TO_ID = {"CN": 0, "MCI": 1, "Dem": 2}
ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}
LOOKBACK_WINDOWS = (7, 14, 28)
PRIMARY_LOOKBACK_DAYS = 28
COMPACT_WINDOWS = (35, 50, 70)
SEQUENCE_DAYS = 35

activity_class_blob = "CONVERT(activity_class_5min USING utf8)"
activity_met_blob = "CONVERT(activity_met_1min USING utf8)"
sleep_hr_blob = "CONVERT(sleep_hr_5min USING utf8)"
sleep_stage_blob = "CONVERT(sleep_hypnogram_5min USING utf8)"
sleep_rmssd_blob = "CONVERT(sleep_rmssd_5min USING utf8)"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def normalize_label(value: Any) -> str:
    mapping = {"cn": "CN", "mci": "MCI", "dem": "Dem", "dementia": "Dem"}
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown diagnosis label: {value!r}")
    return mapping[key]


def load_consistent_labels(paths: Sequence[Path]) -> pd.Series:
    """Read one logical label set and require all three modality copies to match."""
    labels = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["SAMPLE_EMAIL", "DIAG_NM"])
        if frame["SAMPLE_EMAIL"].duplicated().any():
            raise AssertionError(f"Duplicate subject in label file: {Path(path).name}")
        frame["DIAG_NM"] = frame["DIAG_NM"].map(normalize_label)
        labels.append(frame.set_index("SAMPLE_EMAIL")["DIAG_NM"].sort_index())
    if not labels or any(not labels[0].equals(other) for other in labels[1:]):
        raise AssertionError("The three modality label mappings are not identical.")
    return labels[0]


def subject_hash(value: Any, salt: str) -> str:
    return hashlib.sha256((salt + "|" + str(value)).encode("utf-8")).hexdigest()[:24]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, ensure_ascii=False, indent=2)
    tmp.replace(path)


def atomic_joblib_dump(payload: Any, path: Path, compress: int = 3) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, tmp, compress=compress)
    tmp.replace(path)

def parse_slash_sequence(value):
    if pd.isna(value):
        return np.empty(0, dtype=np.float32)
    text = str(value).strip().rstrip("/")
    if not text or text == "...":
        return np.empty(0, dtype=np.float32)
    values = []
    for token in text.split("/"):
        try:
            values.append(float(token))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float32)

def finite_values(values, zero_is_missing=False):
    arr = np.asarray(values, dtype=np.float64)
    arr[~np.isfinite(arr)] = np.nan
    if zero_is_missing:
        arr[arr <= 0] = np.nan
    return arr

def safe_slope(values, x=None):
    arr = finite_values(values)
    if x is None:
        x = np.linspace(0.0, 1.0, len(arr), dtype=float)
    else:
        x = np.asarray(x, dtype=float)
    keep = np.isfinite(arr) & np.isfinite(x)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2:
        return np.nan
    return float(np.polyfit(x[keep], arr[keep], 1)[0])

def safe_acf1(values):
    arr = finite_values(values)
    keep = np.isfinite(arr)
    arr = arr[keep]
    if len(arr) < 4 or np.nanstd(arr[:-1]) == 0 or np.nanstd(arr[1:]) == 0:
        return np.nan
    return float(np.corrcoef(arr[:-1], arr[1:])[0, 1])

def longest_run(mask):
    best = current = 0
    for flag in np.asarray(mask, dtype=bool):
        current = current + 1 if flag else 0
        best = max(best, current)
    return int(best)

def categorical_entropy(values):
    arr = finite_values(values)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    _, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())

def sequence_stats(values, prefix, zero_is_missing=False, expected_length=None):
    arr = finite_values(values, zero_is_missing=zero_is_missing)
    valid = arr[np.isfinite(arr)]
    result = {
        f"{prefix}__raw_length": float(len(arr)),
        f"{prefix}__valid_n": float(len(valid)),
        f"{prefix}__valid_ratio": (
            float(len(valid) / expected_length)
            if expected_length
            else float(len(valid) / max(1, len(arr)))
        ),
    }
    if len(valid) == 0:
        return result
    q10, q25, q50, q75, q90 = np.nanquantile(valid, [0.10, 0.25, 0.50, 0.75, 0.90])
    result.update({
        f"{prefix}__mean": float(np.nanmean(valid)),
        f"{prefix}__std": float(np.nanstd(valid)),
        f"{prefix}__min": float(np.nanmin(valid)),
        f"{prefix}__q10": float(q10),
        f"{prefix}__q25": float(q25),
        f"{prefix}__median": float(q50),
        f"{prefix}__q75": float(q75),
        f"{prefix}__q90": float(q90),
        f"{prefix}__max": float(np.nanmax(valid)),
        f"{prefix}__iqr": float(q75 - q25),
        f"{prefix}__cv": float(np.nanstd(valid) / (abs(np.nanmean(valid)) + 1e-6)),
        f"{prefix}__slope": safe_slope(arr),
        f"{prefix}__acf1": safe_acf1(arr),
    })
    return result

def circular_hour_features(value, prefix):
    try:
        ts = pd.Timestamp(value)
        hour = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    except Exception:
        return {f"{prefix}__sin": np.nan, f"{prefix}__cos": np.nan}
    angle = 2.0 * np.pi * hour / 24.0
    return {f"{prefix}__sin": float(np.sin(angle)), f"{prefix}__cos": float(np.cos(angle))}

def rolling_circular_mean(hours, width):
    arr = np.asarray(hours, dtype=float)
    if len(arr) != 24 or not np.isfinite(arr).all():
        return np.nan
    doubled = np.r_[arr, arr]
    return np.asarray([doubled[i:i + width].mean() for i in range(24)])

def activity_sequence_features(row):
    out = {}
    classes = parse_slash_sequence(row[activity_class_blob])
    valid_classes = classes[np.isfinite(classes)]
    out["actseq__class_valid_ratio"] = float(len(valid_classes) / 288.0)
    out["actseq__class_entropy"] = categorical_entropy(valid_classes)
    if len(valid_classes):
        out["actseq__class_transition_rate"] = float(
            np.mean(valid_classes[1:] != valid_classes[:-1])
        ) if len(valid_classes) > 1 else 0.0
        for code, name in {
            0: "nonwear", 1: "rest", 2: "inactive", 3: "low", 4: "medium", 5: "high"
        }.items():
            mask = valid_classes == code
            out[f"actseq__{name}_ratio"] = float(mask.mean())
            out[f"actseq__{name}_longest_run_ratio"] = float(
                longest_run(mask) / max(1, len(valid_classes))
            )

    met = parse_slash_sequence(row[activity_met_blob])
    out.update(sequence_stats(met, "actseq__met", expected_length=1440))
    met_valid = met[np.isfinite(met)]
    if len(met_valid):
        out["actseq__met_sedentary_ratio"] = float(np.mean(met_valid <= 1.5))
        out["actseq__met_moderate_ratio"] = float(np.mean((met_valid > 1.5) & (met_valid <= 3.0)))
        out["actseq__met_vigorous_ratio"] = float(np.mean(met_valid > 3.0))
    if len(met) == 1440 and np.isfinite(met).all():
        hourly = met.reshape(24, 60).mean(axis=1)
        m10 = rolling_circular_mean(hourly, 10)
        l5 = rolling_circular_mean(hourly, 5)
        m10_value = float(np.max(m10))
        l5_value = float(np.min(l5))
        out["actseq__circadian_m10"] = m10_value
        out["actseq__circadian_l5"] = l5_value
        out["actseq__relative_amplitude"] = (m10_value - l5_value) / (m10_value + l5_value + 1e-6)
        peak_hour = int(np.argmax(hourly))
        angle = 2.0 * np.pi * peak_hour / 24.0
        out["actseq__peak_hour_sin"] = float(np.sin(angle))
        out["actseq__peak_hour_cos"] = float(np.cos(angle))
        centered = hourly - hourly.mean()
        fft = np.fft.rfft(centered)
        out["actseq__first_harmonic_ratio"] = float(
            abs(fft[1]) / (np.abs(fft[1:]).sum() + 1e-6)
        )
    return out

def sleep_sequence_features(row):
    out = {}
    stages = parse_slash_sequence(row[sleep_stage_blob])
    valid_stages = stages[np.isfinite(stages)]
    expected = int(math.ceil(float(row["sleep_duration"]) / 300.0))
    out["sleepseq__stage_length_delta"] = float(len(stages) - expected)
    out["sleepseq__stage_entropy"] = categorical_entropy(valid_stages)
    if len(valid_stages):
        out["sleepseq__stage_transition_rate"] = float(
            np.mean(valid_stages[1:] != valid_stages[:-1])
        ) if len(valid_stages) > 1 else 0.0
        for code, name in {1: "deep", 2: "light", 3: "rem", 4: "awake"}.items():
            mask = valid_stages == code
            out[f"sleepseq__{name}_ratio"] = float(mask.mean())
            out[f"sleepseq__{name}_longest_run_ratio"] = float(
                longest_run(mask) / max(1, len(valid_stages))
            )
        awake = valid_stages == 4
        out["sleepseq__awake_bouts"] = float(np.sum(awake & np.r_[True, ~awake[:-1]]))

    hr = parse_slash_sequence(row[sleep_hr_blob])
    rmssd = parse_slash_sequence(row[sleep_rmssd_blob])
    out.update(sequence_stats(hr, "sleepseq__hr", zero_is_missing=True, expected_length=expected))
    out.update(sequence_stats(rmssd, "sleepseq__rmssd", zero_is_missing=True, expected_length=expected))
    out["sleepseq__hr_rmssd_length_delta"] = float(len(hr) - len(rmssd))
    if len(hr) and len(rmssd):
        n = min(len(hr), len(rmssd))
        joint = (hr[:n] > 0) & (rmssd[:n] > 0) & np.isfinite(hr[:n]) & np.isfinite(rmssd[:n])
        out["sleepseq__hr_rmssd_corr"] = (
            float(np.corrcoef(hr[:n][joint], rmssd[:n][joint])[0, 1])
            if joint.sum() >= 4 and np.std(hr[:n][joint]) > 0 and np.std(rmssd[:n][joint]) > 0
            else np.nan
        )
    return out

ACTIVITY_EXCLUDE = {
    "EMAIL", "activity_day_start", "activity_day_end",
    "activity_class_5min", "activity_met_1min",
    activity_class_blob, activity_met_blob,
}
SLEEP_EXCLUDE = {
    "EMAIL", "sleep_bedtime_start", "sleep_bedtime_end", "sleep_midpoint_time",
    "sleep_period_id", "sleep_hr_5min", "sleep_hypnogram_5min", "sleep_rmssd_5min",
    sleep_hr_blob, sleep_stage_blob, sleep_rmssd_blob,
    # 완전 중복인 deviation 대신 delta 하나만 유지
    "sleep_temperature_deviation", "sleep_is_longest",
}

def make_activity_daily(raw):
    dates = raw["activity_day_start"].astype(str).str.slice(0, 10)
    numeric_cols = [c for c in raw.columns if c not in ACTIVITY_EXCLUDE]
    numeric = raw[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric.columns = [f"act__{c}" for c in numeric.columns]
    derived = pd.DataFrame(
        [activity_sequence_features(row) for _, row in raw.iterrows()],
        index=raw.index,
    )
    out = pd.concat([
        pd.DataFrame({
            "subject_id": raw["EMAIL"].astype(str),
            "sample_date": dates,
        }, index=raw.index),
        numeric,
        derived,
    ], axis=1)
    # 동일 subject-date가 생기면 index가 아니라 날짜 key로 결정론적 median 집계
    return out.groupby(["subject_id", "sample_date"], as_index=False).median(numeric_only=True)

def make_sleep_daily(raw):
    work = raw.copy()
    work["sample_date"] = work["sleep_bedtime_end"].astype(str).str.slice(0, 10)
    work["_duration"] = pd.to_numeric(work["sleep_duration"], errors="coerce")
    # 같은 wake-date에는 가장 긴 main sleep 하나만 유지
    work = (
        work.sort_values(["EMAIL", "sample_date", "_duration"], ascending=[True, True, False])
        .drop_duplicates(["EMAIL", "sample_date"], keep="first")
        .reset_index(drop=True)
    )
    numeric_cols = [c for c in work.columns if c not in SLEEP_EXCLUDE | {"sample_date", "_duration"}]
    numeric = work[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric.columns = [f"sleep__{c}" for c in numeric.columns]
    derived_rows = []
    for _, row in work.iterrows():
        feat = sleep_sequence_features(row)
        feat.update(circular_hour_features(row["sleep_bedtime_start"], "sleep__bedtime_start"))
        feat.update(circular_hour_features(row["sleep_bedtime_end"], "sleep__bedtime_end"))
        duration = float(row["sleep_duration"])
        for col in ["sleep_awake", "sleep_deep", "sleep_light", "sleep_rem", "sleep_total"]:
            feat[f"sleep__{col}_ratio_duration"] = (
                float(row[col]) / duration if duration > 0 else np.nan
            )
        derived_rows.append(feat)
    derived = pd.DataFrame(derived_rows, index=work.index)
    out = pd.concat([
        pd.DataFrame({
            "subject_id": work["EMAIL"].astype(str),
            "sample_date": work["sample_date"],
        }, index=work.index),
        numeric,
        derived,
    ], axis=1)
    return out

def aggregate_modality(daily, anchors, modality):
    work = daily.copy()
    work["_date"] = pd.to_datetime(work["sample_date"], errors="coerce")
    value_cols = [c for c in work.columns if c not in {"subject_id", "sample_date", "_date"}]
    rows = {}
    for subject_id, anchor in anchors.items():
        subject = work[work["subject_id"] == subject_id].sort_values("_date")
        features = {}
        for window in LOOKBACK_WINDOWS:
            start = anchor - pd.Timedelta(days=window - 1)
            part = subject[(subject["_date"] >= start) & (subject["_date"] <= anchor)]
            features[f"{modality}__w{window}__observed_day_ratio"] = float(len(part) / window)
            for col in value_cols:
                values = pd.to_numeric(part[col], errors="coerce").to_numpy(dtype=float)
                valid = values[np.isfinite(values)]
                base = f"{modality}__w{window}__{col}"
                features[f"{base}__valid_day_ratio"] = float(len(valid) / window)
                if len(valid) == 0:
                    continue
                q10, q25, q50, q75, q90 = np.quantile(valid, [0.10, 0.25, 0.50, 0.75, 0.90])
                features[f"{base}__mean"] = float(np.mean(valid))
                features[f"{base}__std"] = float(np.std(valid))
                features[f"{base}__median"] = float(q50)
                features[f"{base}__iqr"] = float(q75 - q25)
                features[f"{base}__p10"] = float(q10)
                features[f"{base}__p90"] = float(q90)
                date_x = (part["_date"] - start).dt.days.to_numpy(dtype=float) / max(1, window - 1)
                features[f"{base}__slope"] = safe_slope(values, date_x)
                if window == PRIMARY_LOOKBACK_DAYS:
                    features[f"{base}__acf1"] = safe_acf1(values)
                    midpoint = anchor - pd.Timedelta(days=window // 2)
                    early = pd.to_numeric(part.loc[part["_date"] <= midpoint, col], errors="coerce")
                    late = pd.to_numeric(part.loc[part["_date"] > midpoint, col], errors="coerce")
                    features[f"{base}__late_minus_early"] = (
                        float(late.mean() - early.mean())
                        if late.notna().any() and early.notna().any()
                        else np.nan
                    )
        rows[subject_id] = features
    return pd.DataFrame.from_dict(rows, orient="index")

PAIR_FEATURES = [
    ("sleep__sleep_total", "act__activity_steps"),
    ("sleep__sleep_efficiency", "act__activity_score"),
    ("sleep__sleep_rmssd", "act__activity_average_met"),
    ("sleep__sleep_breath_average", "act__activity_inactive"),
    ("sleep__sleep_restless", "act__activity_total"),
]

def build_paired_features(activity_daily, sleep_daily, anchors):
    paired = activity_daily.merge(
        sleep_daily,
        on=["subject_id", "sample_date"],
        how="inner",
        validate="one_to_one",
    )
    paired["_date"] = pd.to_datetime(paired["sample_date"], errors="coerce")
    rows = {}
    for subject_id, anchor in anchors.items():
        start = anchor - pd.Timedelta(days=PRIMARY_LOOKBACK_DAYS - 1)
        part = paired[
            (paired["subject_id"] == subject_id)
            & (paired["_date"] >= start)
            & (paired["_date"] <= anchor)
        ]
        feat = {
            "fusion__paired_day_ratio": float(len(part) / PRIMARY_LOOKBACK_DAYS),
        }
        for sleep_col, act_col in PAIR_FEATURES:
            if sleep_col not in part or act_col not in part:
                continue
            x = pd.to_numeric(part[sleep_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(part[act_col], errors="coerce").to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            key = f"fusion__corr__{sleep_col}__{act_col}"
            feat[key] = (
                float(np.corrcoef(x[keep], y[keep])[0, 1])
                if keep.sum() >= 5 and np.std(x[keep]) > 0 and np.std(y[keep]) > 0
                else np.nan
            )
        rows[subject_id] = feat
    return pd.DataFrame.from_dict(rows, orient="index")



# ---------------------------------------------------------------------------
# Compact longitudinal-variability branch (35/50/70 calendar days)
# ---------------------------------------------------------------------------

COMPACT_ACTIVITY_COLUMNS = [
    "act__activity_average_met", "act__activity_cal_active",
    "act__activity_daily_movement", "act__activity_high",
    "act__activity_inactive", "act__activity_low", "act__activity_medium",
    "act__activity_non_wear", "act__activity_rest", "act__activity_steps",
    "act__activity_total", "actseq__class_entropy",
    "actseq__class_transition_rate", "actseq__met__std",
    "actseq__met_sedentary_ratio", "actseq__met_moderate_ratio",
    "actseq__met_vigorous_ratio", "actseq__relative_amplitude",
]
COMPACT_SLEEP_COLUMNS = [
    "sleep__sleep_awake", "sleep__sleep_breath_average", "sleep__sleep_deep",
    "sleep__sleep_duration", "sleep__sleep_efficiency", "sleep__sleep_hr_average",
    "sleep__sleep_hr_lowest", "sleep__sleep_light",
    "sleep__sleep_midpoint_at_delta", "sleep__sleep_onset_latency",
    "sleep__sleep_rem", "sleep__sleep_restless", "sleep__sleep_rmssd",
    "sleep__sleep_temperature_delta", "sleep__sleep_total",
    "sleep__sleep_awake_ratio_duration", "sleep__sleep_deep_ratio_duration",
    "sleep__sleep_light_ratio_duration", "sleep__sleep_rem_ratio_duration",
    "sleep__sleep_total_ratio_duration",
    "sleep__daily_sleep_count", "sleep__hr_drop", "sleep__hr_drop_ratio",
    "sleepseq__stage_entropy", "sleepseq__stage_transition_rate",
    "sleepseq__awake_bouts", "sleepseq__hr__std",
    "sleepseq__rmssd__std", "sleep__lowest_hr_position",
    "sleep__lowest_hr_clock_sin", "sleep__lowest_hr_clock_cos",
]


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


# Preserve the previous notebook's exact daily-sleep contract for the legacy anchor.
make_sleep_daily_legacy = make_sleep_daily


def make_sleep_daily(raw):
    """Create one longest main-sleep row per wake date without using period IDs."""
    work = raw.copy()
    work["sample_date"] = work["sleep_bedtime_end"].astype(str).str.slice(0, 10)
    work["_duration"] = pd.to_numeric(work["sleep_duration"], errors="coerce")
    work["_daily_sleep_count"] = (
        work.groupby(["EMAIL", "sample_date"])["EMAIL"].transform("size").astype(float)
    )
    work = (
        work.sort_values(["EMAIL", "sample_date", "_duration"], ascending=[True, True, False])
        .drop_duplicates(["EMAIL", "sample_date"], keep="first")
        .reset_index(drop=True)
    )
    numeric_cols = [
        c for c in work.columns
        if c not in SLEEP_EXCLUDE | {"sample_date", "_duration", "_daily_sleep_count"}
    ]
    numeric = work[numeric_cols].apply(pd.to_numeric, errors="coerce")
    numeric.columns = [f"sleep__{c}" for c in numeric.columns]
    numeric["sleep__daily_sleep_count"] = work["_daily_sleep_count"].to_numpy(dtype=float)

    derived_rows = []
    for _, row in work.iterrows():
        feat = sleep_sequence_features(row)
        feat.update(circular_hour_features(row["sleep_bedtime_start"], "sleep__bedtime_start"))
        feat.update(circular_hour_features(row["sleep_bedtime_end"], "sleep__bedtime_end"))
        duration = _safe_float(row.get("sleep_duration"))
        for col in ["sleep_awake", "sleep_deep", "sleep_light", "sleep_rem", "sleep_total"]:
            value = _safe_float(row.get(col))
            feat[f"sleep__{col}_ratio_duration"] = (
                value / duration if np.isfinite(value) and duration > 0 else np.nan
            )

        hr_average = _safe_float(row.get("sleep_hr_average"))
        hr_lowest = _safe_float(row.get("sleep_hr_lowest"))
        feat["sleep__hr_drop"] = (
            hr_average - hr_lowest
            if np.isfinite(hr_average) and np.isfinite(hr_lowest) else np.nan
        )
        feat["sleep__hr_drop_ratio"] = (
            (hr_average - hr_lowest) / hr_average
            if np.isfinite(hr_average) and np.isfinite(hr_lowest) and hr_average > 0
            else np.nan
        )

        hr = finite_values(parse_slash_sequence(row[sleep_hr_blob]), zero_is_missing=True)
        valid_positions = np.flatnonzero(np.isfinite(hr))
        if valid_positions.size:
            lowest_position = int(valid_positions[np.nanargmin(hr[valid_positions])])
            denominator = max(1, len(hr) - 1)
            feat["sleep__lowest_hr_position"] = lowest_position / denominator
            try:
                lowest_ts = pd.Timestamp(row["sleep_bedtime_start"]) + pd.Timedelta(
                    minutes=5 * lowest_position
                )
                angle = 2.0 * np.pi * (
                    lowest_ts.hour + lowest_ts.minute / 60.0
                ) / 24.0
                feat["sleep__lowest_hr_clock_sin"] = float(np.sin(angle))
                feat["sleep__lowest_hr_clock_cos"] = float(np.cos(angle))
            except Exception:
                feat["sleep__lowest_hr_clock_sin"] = np.nan
                feat["sleep__lowest_hr_clock_cos"] = np.nan
        else:
            feat["sleep__lowest_hr_position"] = np.nan
            feat["sleep__lowest_hr_clock_sin"] = np.nan
            feat["sleep__lowest_hr_clock_cos"] = np.nan
        derived_rows.append(feat)

    derived = pd.DataFrame(derived_rows, index=work.index)
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "subject_id": work["EMAIL"].astype(str),
                    "sample_date": work["sample_date"],
                },
                index=work.index,
            ),
            numeric,
            derived,
        ],
        axis=1,
    )


def robust_histogram_mode(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    if np.nanmax(values) == np.nanmin(values) or len(values) < 4:
        return float(np.nanmedian(values))
    bins = int(np.clip(np.ceil(np.sqrt(len(values))), 3, 10))
    counts, edges = np.histogram(values, bins=bins)
    idx = int(np.argmax(counts))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def compact_variability_stats(values: np.ndarray) -> dict[str, float]:
    """Predeclared M/EM/distribution/dispersion/time-series feature families."""
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    result: dict[str, float] = {
        "valid_day_ratio": float(len(valid) / max(1, len(arr))),
    }
    if not len(valid):
        return result
    q25, q50, q75 = np.quantile(valid, [0.25, 0.50, 0.75])
    mean = float(np.mean(valid))
    median = float(q50)
    mad_median = float(np.median(np.abs(valid - median)))
    result.update(
        {
            "mean": mean,
            "median": median,
            "trimmed_mean_10": float(trim_mean(valid, 0.10)) if len(valid) >= 5 else mean,
            "robust_mode": robust_histogram_mode(valid),
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
            "median_abs_deviation": mad_median,
            "mean_abs_deviation": float(np.mean(np.abs(valid - mean))),
            "kurtosis": (
                float(kurtosis(valid, fisher=True, bias=False))
                if len(valid) >= 4 and np.std(valid) > 0 else np.nan
            ),
            "range": float(np.max(valid) - np.min(valid)),
            "std": float(np.std(valid)),
            "cv": float(np.std(valid) / (abs(mean) + 1e-6)),
            "iqr": float(q75 - q25),
        }
    )
    consecutive = np.abs(np.diff(arr))
    result["short_term_variability"] = (
        float(np.nanmedian(consecutive)) if np.isfinite(consecutive).any() else np.nan
    )
    result["moving_range_mean"] = (
        float(np.nanmean(consecutive)) if np.isfinite(consecutive).any() else np.nan
    )
    series = pd.Series(arr, dtype=float)
    rolling = series.rolling(7, min_periods=3)
    rolling_sd = rolling.std(ddof=0)
    rolling_mean = rolling.mean().abs()
    rolling_cv = rolling_sd / (rolling_mean + 1e-6)
    result["rolling7_sd_mean"] = (
        float(rolling_sd.mean()) if rolling_sd.notna().any() else np.nan
    )
    result["rolling7_cv_mean"] = (
        float(rolling_cv.mean()) if rolling_cv.notna().any() else np.nan
    )
    bin_means = [
        float(np.nanmean(chunk)) if np.isfinite(chunk).any() else np.nan
        for chunk in np.array_split(arr, 4)
    ]
    finite_bins = np.asarray(bin_means, dtype=float)
    result["timebin_mean_std"] = (
        float(np.nanstd(finite_bins)) if np.isfinite(finite_bins).sum() >= 2 else np.nan
    )
    result["timebin_mean_range"] = (
        float(np.nanmax(finite_bins) - np.nanmin(finite_bins))
        if np.isfinite(finite_bins).sum() >= 2 else np.nan
    )
    result["timebin_last_minus_first"] = (
        float(finite_bins[-1] - finite_bins[0])
        if np.isfinite(finite_bins[[0, -1]]).all() else np.nan
    )
    return result


def aggregate_compact_modality(
    daily: pd.DataFrame,
    anchors: pd.Series,
    modality: str,
    candidate_columns: Sequence[str],
    windows: Sequence[int] = COMPACT_WINDOWS,
) -> pd.DataFrame:
    work = daily.copy()
    work["_date"] = pd.to_datetime(work["sample_date"], errors="coerce")
    present_columns = [c for c in candidate_columns if c in work.columns]
    if not present_columns:
        raise AssertionError(f"No compact columns were found for {modality}.")
    rows: dict[str, dict[str, float]] = {}
    for subject_id, anchor in anchors.items():
        subject = work.loc[work["subject_id"] == subject_id].set_index("_date").sort_index()
        features: dict[str, float] = {}
        for window in windows:
            dates = pd.date_range(anchor - pd.Timedelta(days=window - 1), anchor, freq="D")
            grid = subject.reindex(dates)
            features[f"{modality}__w{window}__observed_day_ratio"] = float(
                grid["subject_id"].notna().mean()
            )
            for col in present_columns:
                values = pd.to_numeric(grid[col], errors="coerce").to_numpy(dtype=float)
                stats = compact_variability_stats(values)
                for stat, value in stats.items():
                    features[f"{modality}__w{window}__{col}__{stat}"] = value
        rows[str(subject_id)] = features
    return pd.DataFrame.from_dict(rows, orient="index")


def build_daily_sequence(
    activity_daily: pd.DataFrame,
    sleep_daily: pd.DataFrame,
    anchors: pd.Series,
    days: int = SEQUENCE_DAYS,
) -> tuple[np.ndarray, list[str]]:
    activity_cols = [c for c in COMPACT_ACTIVITY_COLUMNS if c in activity_daily.columns]
    sleep_cols = [c for c in COMPACT_SLEEP_COLUMNS if c in sleep_daily.columns]
    columns = activity_cols + sleep_cols
    joined = activity_daily[["subject_id", "sample_date", *activity_cols]].merge(
        sleep_daily[["subject_id", "sample_date", *sleep_cols]],
        on=["subject_id", "sample_date"],
        how="outer",
        validate="one_to_one",
    )
    joined["_date"] = pd.to_datetime(joined["sample_date"], errors="coerce")
    tensors = []
    for subject_id, anchor in anchors.items():
        dates = pd.date_range(anchor - pd.Timedelta(days=days - 1), anchor, freq="D")
        part = (
            joined.loc[joined["subject_id"] == subject_id]
            .set_index("_date")
            .reindex(dates)
        )
        tensors.append(
            part.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(float)
        )
    tensor = np.stack(tensors).astype(np.float32)
    return tensor, columns


def build_coverage_table(
    activity_daily: pd.DataFrame,
    sleep_daily: pd.DataFrame,
    anchors: pd.Series,
    days: int = SEQUENCE_DAYS,
) -> pd.DataFrame:
    activity_dates = activity_daily.assign(
        _date=pd.to_datetime(activity_daily["sample_date"], errors="coerce")
    )
    sleep_dates = sleep_daily.assign(
        _date=pd.to_datetime(sleep_daily["sample_date"], errors="coerce")
    )
    rows = {}
    for subject_id, anchor in anchors.items():
        start = anchor - pd.Timedelta(days=days - 1)
        activity_set = set(
            activity_dates.loc[
                (activity_dates["subject_id"] == subject_id)
                & activity_dates["_date"].between(start, anchor),
                "_date",
            ].dropna()
        )
        sleep_set = set(
            sleep_dates.loc[
                (sleep_dates["subject_id"] == subject_id)
                & sleep_dates["_date"].between(start, anchor),
                "_date",
            ].dropna()
        )
        rows[str(subject_id)] = {
            "activity_observed_days_35": len(activity_set),
            "sleep_observed_nights_35": len(sleep_set),
            "paired_observed_days_35": len(activity_set & sleep_set),
            "activity_coverage_ratio_35": len(activity_set) / days,
            "sleep_coverage_ratio_35": len(sleep_set) / days,
            "paired_coverage_ratio_35": len(activity_set & sleep_set) / days,
            "meets_35_sleep_nights": int(len(sleep_set) >= days),
        }
    return pd.DataFrame.from_dict(rows, orient="index").reindex(anchors.index)


def build_feature_bundle(activity_raw: pd.DataFrame, sleep_raw: pd.DataFrame) -> dict[str, Any]:
    """Deterministic source-only feature construction; labels are not accepted."""
    activity_daily = make_activity_daily(activity_raw)
    sleep_daily_legacy = make_sleep_daily_legacy(sleep_raw)
    sleep_daily = make_sleep_daily(sleep_raw)
    activity_daily["_date"] = pd.to_datetime(activity_daily["sample_date"], errors="coerce")
    anchors = activity_daily.groupby("subject_id")["_date"].max().sort_index()
    activity_daily = activity_daily.drop(columns="_date")

    legacy = (
        aggregate_modality(activity_daily, anchors, "activity")
        .join(aggregate_modality(sleep_daily_legacy, anchors, "sleep"), how="outer")
        .join(build_paired_features(activity_daily, sleep_daily_legacy, anchors), how="outer")
        .reindex(anchors.index)
    )
    compact35 = (
        aggregate_compact_modality(
            activity_daily, anchors, "activity_compact", COMPACT_ACTIVITY_COLUMNS, (35,)
        )
        .join(
            aggregate_compact_modality(
                sleep_daily, anchors, "sleep_compact", COMPACT_SLEEP_COLUMNS, (35,)
            ),
            how="outer",
        )
        .reindex(anchors.index)
    )
    compact_multi = (
        aggregate_compact_modality(
            activity_daily, anchors, "activity_compact", COMPACT_ACTIVITY_COLUMNS, COMPACT_WINDOWS
        )
        .join(
            aggregate_compact_modality(
                sleep_daily, anchors, "sleep_compact", COMPACT_SLEEP_COLUMNS, COMPACT_WINDOWS
            ),
            how="outer",
        )
        .reindex(anchors.index)
    )
    sequence_values, sequence_columns = build_daily_sequence(
        activity_daily, sleep_daily, anchors, SEQUENCE_DAYS
    )
    coverage = build_coverage_table(activity_daily, sleep_daily, anchors, SEQUENCE_DAYS)
    views = {
        "legacy_all": legacy.apply(pd.to_numeric, errors="coerce"),
        "compact35": compact35.apply(pd.to_numeric, errors="coerce"),
        "compact_multi": compact_multi.apply(pd.to_numeric, errors="coerce"),
    }
    for name, frame in views.items():
        if frame.index.duplicated().any():
            raise AssertionError(f"Duplicate subjects in feature view {name}.")
        if any(
            forbidden in col.lower()
            for col in frame.columns
            for forbidden in ("email", "diag", "mmse", "doctor", "sample_order")
        ):
            raise AssertionError(f"Forbidden feature detected in {name}.")
    return {
        "subject_ids": anchors.index.astype(str).tolist(),
        "anchors": anchors.astype(str).tolist(),
        "views": views,
        "sequence_values": sequence_values,
        "sequence_columns": sequence_columns,
        "coverage": coverage,
        "diagnostics": {
            "subjects": int(len(anchors)),
            "activity_raw_rows": int(len(activity_raw)),
            "sleep_raw_rows": int(len(sleep_raw)),
            "activity_daily_rows": int(len(activity_daily)),
            "sleep_daily_rows": int(len(sleep_daily)),
            "legacy_sleep_daily_rows": int(len(sleep_daily_legacy)),
            "feature_counts": {name: int(frame.shape[1]) for name, frame in views.items()},
            "sequence_shape": list(sequence_values.shape),
        },
    }


def deidentify_feature_bundle(bundle: dict[str, Any], salt: str) -> dict[str, Any]:
    hashed = [subject_hash(value, salt) for value in bundle["subject_ids"]]
    if len(set(hashed)) != len(hashed):
        raise AssertionError("Subject hash collision.")
    output = dict(bundle)
    output["subject_ids"] = hashed
    output.pop("anchors", None)
    output["views"] = {
        name: frame.set_axis(hashed, axis=0).rename_axis("subject_hash")
        for name, frame in bundle["views"].items()
    }
    output["coverage"] = bundle["coverage"].set_axis(hashed, axis=0).rename_axis(
        "subject_hash"
    )
    return output


# ---------------------------------------------------------------------------
# Fold-local feature stability bank and tabular candidates
# ---------------------------------------------------------------------------

@dataclass
class FeatureSlice:
    columns: list[str]
    preserve_nan: bool
    scale: bool
    medians: pd.Series | None = None
    lower: pd.Series | None = None
    upper: pd.Series | None = None
    means: pd.Series | None = None
    stds: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "FeatureSlice":
        clean = frame.reindex(columns=self.columns).replace([np.inf, -np.inf], np.nan)
        if not self.preserve_nan:
            self.medians = clean.median(axis=0).fillna(0.0)
            filled = clean.fillna(self.medians)
            self.lower = filled.quantile(0.005)
            self.upper = filled.quantile(0.995)
            clipped = filled.clip(self.lower, self.upper, axis=1)
            if self.scale:
                self.means = clipped.mean(axis=0)
                self.stds = clipped.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        clean = frame.reindex(columns=self.columns).replace([np.inf, -np.inf], np.nan)
        if self.preserve_nan:
            return clean.to_numpy(dtype=np.float32)
        if self.medians is None or self.lower is None or self.upper is None:
            raise RuntimeError("FeatureSlice has not been fitted.")
        clean = clean.fillna(self.medians).clip(self.lower, self.upper, axis=1)
        if self.scale:
            clean = (clean - self.means) / self.stds
        values = clean.to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise AssertionError("Non-finite classic-model feature after fold transform.")
        return values


class FoldFeatureBank:
    """Boundary-aware deterministic ranking learned on one training fold only."""

    def __init__(self, random_state: int, fast_mode: bool = False):
        self.random_state = int(random_state)
        self.fast_mode = bool(fast_mode)
        self.usable_columns: list[str] = []
        self.ranking: list[str] = []
        self.stability_frequency: pd.Series | None = None
        self.score_table: pd.DataFrame | None = None

    @staticmethod
    def _f_scores(values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        from sklearn.feature_selection import f_classif

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores, _ = f_classif(values, labels)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, frame: pd.DataFrame, labels: np.ndarray) -> "FoldFeatureBank":
        clean = frame.replace([np.inf, -np.inf], np.nan)
        missing = clean.isna().mean(axis=0)
        nunique = clean.nunique(dropna=True)
        self.usable_columns = clean.columns[(missing <= 0.60) & (nunique > 1)].tolist()
        if not self.usable_columns:
            raise AssertionError("No usable features after missingness/constant filtering.")
        clean = clean[self.usable_columns]
        medians = clean.median(axis=0).fillna(0.0)
        values = clean.fillna(medians).to_numpy(dtype=float)
        labels = np.asarray(labels, dtype=int)

        score_sets: dict[str, np.ndarray] = {
            "multiclass": self._f_scores(values, labels),
        }
        for left, right in ((0, 1), (1, 2), (0, 2)):
            keep = np.isin(labels, [left, right])
            score_sets[f"{left}_vs_{right}"] = self._f_scores(values[keep], labels[keep])

        rng = np.random.default_rng(self.random_state)
        repeats = 6 if self.fast_mode else 16
        top_pool = min(values.shape[1], max(32, min(256, values.shape[1] // 4)))
        selected_counts = np.zeros(values.shape[1], dtype=float)
        for _ in range(repeats):
            sampled = []
            for class_id in range(3):
                class_idx = np.flatnonzero(labels == class_id)
                take = max(2, int(np.ceil(0.80 * len(class_idx))))
                sampled.extend(rng.choice(class_idx, size=take, replace=False).tolist())
            bootstrap_scores = self._f_scores(values[np.asarray(sampled)], labels[np.asarray(sampled)])
            top = np.argsort(-bootstrap_scores, kind="mergesort")[:top_pool]
            selected_counts[top] += 1.0
        frequency = selected_counts / repeats

        rank_lists = {
            name: np.argsort(-scores, kind="mergesort").tolist()
            for name, scores in score_sets.items()
        }
        # np.lexsort is ascending: negative frequency/score gives highest first.
        stable_order = np.lexsort(
            (np.arange(values.shape[1]), -score_sets["multiclass"], -frequency)
        ).tolist()
        # Interleave the stability-first multiclass list with all three boundary
        # rankings. Pairwise lists therefore materially affect every top-k slice.
        priority_lists = [
            stable_order,
            rank_lists["0_vs_1"],
            rank_lists["1_vs_2"],
            rank_lists["0_vs_2"],
        ]
        combined: list[int] = []
        seen: set[int] = set()
        for position in range(values.shape[1]):
            for ranking in priority_lists:
                idx = ranking[position]
                if idx not in seen:
                    seen.add(idx)
                    combined.append(idx)

        # Correlation pruning is ranking-only and uses training-fold values.
        kept: list[int] = []
        for idx in combined:
            # No candidate consumes more than 96 ranked features. A 256-feature
            # diverse prefix is sufficient; native TabPFN receives the complete
            # usable set after the unpruned remainder is appended below.
            if len(kept) >= 256:
                break
            if not kept:
                kept.append(idx)
                continue
            candidate = values[:, idx]
            correlated = False
            for other in kept:
                reference = values[:, other]
                if np.std(candidate) == 0 or np.std(reference) == 0:
                    continue
                if abs(np.corrcoef(candidate, reference)[0, 1]) > 0.97:
                    correlated = True
                    break
            if not correlated:
                kept.append(idx)
        # Append pruned features after the diverse prefix so native view remains complete.
        kept_set = set(kept)
        kept.extend(idx for idx in combined if idx not in kept_set)
        self.ranking = [self.usable_columns[idx] for idx in kept]
        self.stability_frequency = pd.Series(
            frequency, index=self.usable_columns, name="selection_frequency"
        )
        self.score_table = pd.DataFrame(
            {name: values_ for name, values_ in score_sets.items()},
            index=self.usable_columns,
        ).assign(selection_frequency=frequency)
        return self

    def make_slice(
        self,
        frame: pd.DataFrame,
        top_k: int | None,
        preserve_nan: bool,
        scale: bool,
    ) -> FeatureSlice:
        columns = self.ranking if top_k is None else self.ranking[: min(top_k, len(self.ranking))]
        return FeatureSlice(columns, preserve_nan=preserve_nan, scale=scale).fit(frame)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    kind: str
    view: str | None = None
    top_k: int | None = None
    balanced: bool = False


CANDIDATE_SPECS = [
    CandidateSpec("lda_compact35_k32", "lda", "compact35", 32),
    CandidateSpec("elastic_compact35_k32", "elastic", "compact35", 32),
    CandidateSpec("elastic_compact_multi_k48", "elastic", "compact_multi", 48),
    CandidateSpec("cat_compact35_k48", "cat", "compact35", 48),
    CandidateSpec("elastic_legacy_all_k64", "elastic", "legacy_all", 64),
    CandidateSpec("tabpfn3_compact35_native_raw", "tabpfn", "compact35", None, False),
    CandidateSpec("tabpfn3_compact35_k64_raw", "tabpfn", "compact35", 64, False),
    CandidateSpec("tabpfn3_compact35_k64_balanced", "tabpfn", "compact35", 64, True),
    CandidateSpec("tabpfn3_compact_multi_k96_balanced", "tabpfn", "compact_multi", 96, True),
    CandidateSpec("tabpfn3_pairwise_compact35_k64", "tabpfn_pairwise", "compact35", 64),
    CandidateSpec("minirocket_35d", "minirocket"),
    CandidateSpec("mask_tcn_35d", "tcn"),
]
CANDIDATE_NAMES = [spec.name for spec in CANDIDATE_SPECS]


class PairwiseTabPFN:
    pairs = ((0, 1), (0, 2), (1, 2))

    def __init__(self, seed: int, n_estimators: int, device: str):
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.device = device
        self.models: list[Any] = []

    def fit(self, values: np.ndarray, labels: np.ndarray) -> "PairwiseTabPFN":
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        self.models = []
        for pair_index, (left, right) in enumerate(self.pairs):
            keep = np.isin(labels, [left, right])
            binary = (labels[keep] == right).astype(int)
            model = TabPFNClassifier.create_default_for_version(
                ModelVersion.V3,
                n_estimators=self.n_estimators,
                balance_probabilities=True,
                device=self.device,
                inference_precision=torch.float32,
                random_state=self.seed + pair_index,
            )
            model.fit(values[keep], binary)
            self.models.append(model)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        log_votes = np.zeros((len(values), 3), dtype=float)
        participation = np.zeros(3, dtype=float)
        for model, (left, right) in zip(self.models, self.pairs):
            probabilities = np.clip(model.predict_proba(values), 1e-7, 1.0)
            log_votes[:, left] += np.log(probabilities[:, 0])
            log_votes[:, right] += np.log(probabilities[:, 1])
            participation[[left, right]] += 1.0
        log_votes /= participation[None, :]
        return softmax(log_votes, axis=1)


def _fit_tabular_model(
    kind: str,
    values: np.ndarray,
    labels: np.ndarray,
    seed: int,
    fast_mode: bool,
    balanced: bool,
    device: str,
) -> Any:
    if kind == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        model = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=np.repeat(1.0 / 3.0, 3)
        )
    elif kind == "elastic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.25,
            C=0.25,
            class_weight="balanced",
            max_iter=5000,
            tol=1e-4,
            random_state=seed,
        )
    elif kind == "cat":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=120 if fast_mode else 700,
            depth=3,
            learning_rate=0.04,
            loss_function="MultiClass",
            auto_class_weights="SqrtBalanced",
            l2_leaf_reg=12.0,
            random_strength=0.5,
            random_seed=seed,
            thread_count=-1,
            task_type="CPU",
            verbose=False,
            allow_writing_files=False,
        )
    elif kind == "tabpfn":
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        model = TabPFNClassifier.create_default_for_version(
            ModelVersion.V3,
            n_estimators=4 if fast_mode else 24,
            balance_probabilities=balanced,
            device=device,
            inference_precision=torch.float32,
            random_state=seed,
        )
    elif kind == "tabpfn_pairwise":
        model = PairwiseTabPFN(seed, 4 if fast_mode else 16, device)
    else:
        raise KeyError(kind)
    return model.fit(values, labels)


# ---------------------------------------------------------------------------
# Sequence branches: fold-normalized MiniRocket and a deliberately small TCN
# ---------------------------------------------------------------------------

class SequenceFoldTransformer:
    def __init__(self):
        self.medians: np.ndarray | None = None
        self.iqrs: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "SequenceFoldTransformer":
        values = np.asarray(values, dtype=float)
        self.medians = np.nanmedian(values, axis=(0, 1))
        q25 = np.nanquantile(values, 0.25, axis=(0, 1))
        q75 = np.nanquantile(values, 0.75, axis=(0, 1))
        self.medians = np.nan_to_num(self.medians, nan=0.0)
        self.iqrs = np.nan_to_num(q75 - q25, nan=1.0)
        self.iqrs[self.iqrs < 1e-6] = 1.0
        return self

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.medians is None or self.iqrs is None:
            raise RuntimeError("SequenceFoldTransformer has not been fitted.")
        values = np.asarray(values, dtype=float)
        observed = np.isfinite(values)
        normalized = (np.where(observed, values, self.medians) - self.medians) / self.iqrs
        delta = np.zeros_like(normalized, dtype=float)
        for time_index in range(1, values.shape[1]):
            delta[:, time_index, :] = np.where(
                observed[:, time_index, :],
                0.0,
                np.minimum(delta[:, time_index - 1, :] + 1.0, values.shape[1]),
            )
        delta /= max(1, values.shape[1])
        channels = np.concatenate(
            [normalized, observed.astype(float), delta], axis=2
        ).astype(np.float32)
        day_mask = observed.any(axis=2).astype(np.float32)
        if not np.isfinite(channels).all():
            raise AssertionError("Non-finite sequence after fold transform.")
        return channels, day_mask


class MiniRocketBundle:
    def __init__(self, seed: int, fast_mode: bool):
        self.seed = int(seed)
        self.fast_mode = bool(fast_mode)
        self.rocket: Any = None
        self.scaler: Any = None
        self.classifier: Any = None

    def fit(self, channels: np.ndarray, labels: np.ndarray) -> "MiniRocketBundle":
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.preprocessing import StandardScaler
        from sktime.transformations.rocket import MiniRocketMultivariate

        panel = np.transpose(channels, (0, 2, 1)).astype(np.float32)
        self.rocket = MiniRocketMultivariate(
            num_kernels=2_000 if self.fast_mode else 10_000,
            random_state=self.seed,
            n_jobs=-1,
        )
        transformed = np.asarray(self.rocket.fit_transform(panel), dtype=np.float32)
        self.scaler = StandardScaler(with_mean=False)
        transformed = self.scaler.fit_transform(transformed)
        self.classifier = RidgeClassifierCV(
            alphas=np.logspace(-3, 3, 13), class_weight="balanced"
        )
        self.classifier.fit(transformed, labels)
        return self

    def predict_proba(self, channels: np.ndarray) -> np.ndarray:
        panel = np.transpose(channels, (0, 2, 1)).astype(np.float32)
        transformed = np.asarray(self.rocket.transform(panel), dtype=np.float32)
        scores = np.asarray(
            self.classifier.decision_function(self.scaler.transform(transformed)), dtype=float
        )
        return softmax(scores, axis=1)


class TinyTCNBundle:
    def __init__(self, seed: int, fast_mode: bool, device: str):
        self.seed = int(seed)
        self.fast_mode = bool(fast_mode)
        self.device = device
        self.input_channels: int | None = None
        self.state_dict: dict[str, Any] | None = None
        self.selected_epoch: int | None = None

    @staticmethod
    def _network(input_channels: int):
        import torch
        from torch import nn

        class ResidualBlock(nn.Module):
            def __init__(self, hidden: int, dilation: int):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(6, hidden),
                    nn.GELU(),
                    nn.Dropout(0.35),
                    nn.Conv1d(hidden, hidden, 3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(6, hidden),
                    nn.GELU(),
                    nn.Dropout(0.35),
                )

            def forward(self, values):
                return values + self.layers(values)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = 24
                self.input = nn.Conv1d(input_channels, hidden, 1)
                self.blocks = nn.Sequential(
                    *[ResidualBlock(hidden, dilation) for dilation in (1, 2, 4, 8)]
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden * 2, 32), nn.GELU(), nn.Dropout(0.35), nn.Linear(32, 3)
                )

            def forward(self, values, day_mask):
                hidden = self.blocks(self.input(values.transpose(1, 2))).transpose(1, 2)
                denominator = day_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                mean_pool = (hidden * day_mask.unsqueeze(-1)).sum(dim=1) / denominator
                positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                last_index = (positions * day_mask.long()).max(dim=1).values
                last_pool = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_index]
                return self.head(torch.cat([mean_pool, last_pool], dim=1))

        return Network()

    def fit(
        self, channels: np.ndarray, day_mask: np.ndarray, labels: np.ndarray
    ) -> "TinyTCNBundle":
        import torch
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedShuffleSplit
        from torch.utils.data import DataLoader, TensorDataset

        set_all_seeds(self.seed)
        self.input_channels = int(channels.shape[2])
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=self.seed)
        train_index, stop_index = next(split.split(np.zeros(len(labels)), labels))
        train_dataset = TensorDataset(
            torch.tensor(channels[train_index]),
            torch.tensor(day_mask[train_index]),
            torch.tensor(labels[train_index], dtype=torch.long),
        )
        loader = DataLoader(
            train_dataset,
            batch_size=min(32, len(train_dataset)),
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        stop_values = torch.tensor(channels[stop_index], device=device)
        stop_mask = torch.tensor(day_mask[stop_index], device=device)
        model = self._network(self.input_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
        counts = np.bincount(labels[train_index], minlength=3).astype(float)
        weights = np.sqrt(len(train_index) / (3.0 * np.maximum(counts, 1.0)))
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device),
            label_smoothing=0.05,
        )
        best_score = -np.inf
        best_state = None
        best_epoch = None
        patience = 12 if self.fast_mode else 30
        stale = 0
        epochs = 60 if self.fast_mode else 300
        for epoch_index in range(epochs):
            model.train()
            for batch_values, batch_mask, batch_labels in loader:
                batch_values = batch_values.to(device)
                batch_mask = batch_mask.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_values, batch_mask), batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                stop_predictions = model(stop_values, stop_mask).argmax(dim=1).cpu().numpy()
            score = f1_score(labels[stop_index], stop_predictions, average="macro", zero_division=0)
            if score > best_score + 1e-6:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                best_epoch = epoch_index + 1
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        if best_state is None or best_epoch is None:
            raise RuntimeError("TCN early stopping did not produce a model state.")
        # The internal split selects only the epoch count. Reset and refit on every
        # candidate-training subject so the held-in stopping subset also contributes
        # gradients; outer-valid subjects remain completely untouched.
        self.selected_epoch = int(best_epoch)
        del model, optimizer, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        refit_seed = self.seed + 100_003
        set_all_seeds(refit_seed)
        full_dataset = TensorDataset(
            torch.tensor(channels),
            torch.tensor(day_mask),
            torch.tensor(labels, dtype=torch.long),
        )
        full_loader = DataLoader(
            full_dataset,
            batch_size=min(32, len(full_dataset)),
            shuffle=True,
            generator=torch.Generator().manual_seed(refit_seed),
        )
        final_model = self._network(self.input_channels).to(device)
        final_optimizer = torch.optim.AdamW(
            final_model.parameters(), lr=8e-4, weight_decay=2e-3
        )
        full_counts = np.bincount(labels, minlength=3).astype(float)
        full_weights = np.sqrt(len(labels) / (3.0 * np.maximum(full_counts, 1.0)))
        final_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(full_weights, dtype=torch.float32, device=device),
            label_smoothing=0.05,
        )
        for _ in range(self.selected_epoch):
            final_model.train()
            for batch_values, batch_mask, batch_labels in full_loader:
                batch_values = batch_values.to(device)
                batch_mask = batch_mask.to(device)
                batch_labels = batch_labels.to(device)
                final_optimizer.zero_grad(set_to_none=True)
                loss = final_criterion(
                    final_model(batch_values, batch_mask), batch_labels
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
                final_optimizer.step()
        self.state_dict = {
            key: value.detach().cpu().clone()
            for key, value in final_model.state_dict().items()
        }
        return self

    def predict_proba(self, channels: np.ndarray, day_mask: np.ndarray) -> np.ndarray:
        import torch

        if self.input_channels is None or self.state_dict is None:
            raise RuntimeError("TinyTCNBundle has not been fitted.")
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        model = self._network(self.input_channels).to(device)
        model.load_state_dict(self.state_dict)
        model.eval()
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(channels), 128):
                logits = model(
                    torch.tensor(channels[start:start + 128], device=device),
                    torch.tensor(day_mask[start:start + 128], device=device),
                )
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probabilities, axis=0)


def fit_predict_candidate(
    spec: CandidateSpec,
    views: Mapping[str, pd.DataFrame],
    sequences: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    predict_index: np.ndarray,
    seed: int,
    fast_mode: bool,
    device: str,
    feature_bank_cache: dict[str, FoldFeatureBank] | None = None,
    feature_bank_seed: int | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    set_all_seeds(seed)
    if spec.kind in {"minirocket", "tcn"}:
        transformer = SequenceFoldTransformer().fit(sequences[train_index])
        train_channels, train_mask = transformer.transform(sequences[train_index])
        predict_channels, predict_mask = transformer.transform(sequences[predict_index])
        if spec.kind == "minirocket":
            model = MiniRocketBundle(seed, fast_mode).fit(train_channels, labels[train_index])
            probabilities = model.predict_proba(predict_channels)
        else:
            model = TinyTCNBundle(seed, fast_mode, device).fit(
                train_channels, train_mask, labels[train_index]
            )
            probabilities = model.predict_proba(predict_channels, predict_mask)
        bundle = {"kind": spec.kind, "sequence_transformer": transformer, "model": model}
    else:
        frame = views[spec.view]
        feature_bank_cache = feature_bank_cache if feature_bank_cache is not None else {}
        if spec.view not in feature_bank_cache:
            feature_bank_cache[spec.view] = FoldFeatureBank(
                seed if feature_bank_seed is None else feature_bank_seed,
                fast_mode,
            ).fit(frame.iloc[train_index], labels[train_index])
        bank = feature_bank_cache[spec.view]
        preserve_nan = spec.kind in {"tabpfn", "tabpfn_pairwise"}
        scale = spec.kind in {"lda", "elastic"}
        feature_slice = bank.make_slice(
            frame.iloc[train_index], spec.top_k, preserve_nan=preserve_nan, scale=scale
        )
        train_values = feature_slice.transform(frame.iloc[train_index])
        predict_values = feature_slice.transform(frame.iloc[predict_index])
        model = _fit_tabular_model(
            spec.kind,
            train_values,
            labels[train_index],
            seed,
            fast_mode,
            spec.balanced,
            device,
        )
        probabilities = model.predict_proba(predict_values)
        bundle = {
            "kind": spec.kind,
            "view": spec.view,
            "feature_slice": feature_slice,
            "model": model,
            "feature_score_table": bank.score_table,
            "feature_ranking": list(bank.ranking),
            "selected_features": list(feature_slice.columns),
        }
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, 1e-7, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if probabilities.shape != (len(predict_index), 3):
        raise AssertionError(f"Unexpected probability shape for {spec.name}: {probabilities.shape}")
    return bundle, probabilities


def predict_fitted_candidate(
    bundle: Mapping[str, Any],
    views: Mapping[str, pd.DataFrame],
    sequences: np.ndarray,
) -> np.ndarray:
    kind = bundle["kind"]
    if kind in {"minirocket", "tcn"}:
        channels, day_mask = bundle["sequence_transformer"].transform(sequences)
        if kind == "minirocket":
            probabilities = bundle["model"].predict_proba(channels)
        else:
            probabilities = bundle["model"].predict_proba(channels, day_mask)
    else:
        values = bundle["feature_slice"].transform(views[bundle["view"]])
        probabilities = bundle["model"].predict_proba(values)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def release_candidate_resources(bundle: Any) -> None:
    """Release ephemeral candidate models between CV fits, especially TabPFN VRAM."""
    def move_tabpfn_to_cpu(value: Any) -> None:
        if isinstance(value, PairwiseTabPFN):
            for child in value.models:
                move_tabpfn_to_cpu(child)
        elif (
            value.__class__.__module__.startswith("tabpfn")
            and hasattr(value, "to")
        ):
            value.to("cpu")
        elif isinstance(value, Mapping):
            for child in value.values():
                move_tabpfn_to_cpu(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                move_tabpfn_to_cpu(child)

    move_tabpfn_to_cpu(bundle)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Metrics, sparse rule selection, repeated nested CV, and frozen artifacts
# ---------------------------------------------------------------------------

def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
    logits = np.log(probabilities) / float(temperature)
    return softmax(logits, axis=1)


def evaluate_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_recall_fscore_support,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2], zero_division=0
    )
    binary = label_binarize(labels, classes=[0, 1, 2])
    per_auc: list[float | None] = []
    per_auprc: list[float | None] = []
    for class_id in range(3):
        try:
            per_auc.append(float(roc_auc_score(binary[:, class_id], probabilities[:, class_id])))
        except ValueError:
            per_auc.append(None)
        try:
            per_auprc.append(
                float(average_precision_score(binary[:, class_id], probabilities[:, class_id]))
            )
        except ValueError:
            per_auprc.append(None)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "macro_ovr_auroc": float(np.mean([x for x in per_auc if x is not None])),
        "macro_ovr_auprc": float(np.mean([x for x in per_auprc if x is not None])),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
        "per_class": {
            CLASS_NAMES[class_id]: {
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
                "ovr_auroc": per_auc[class_id],
                "ovr_auprc": per_auprc[class_id],
            }
            for class_id in range(3)
        },
    }


def summarize_metric_runs(metric_runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not metric_runs:
        raise ValueError("At least one metric run is required.")
    scalar_keys = [
        "macro_f1", "balanced_accuracy", "accuracy", "log_loss",
        "macro_ovr_auroc", "macro_ovr_auprc",
    ]
    summary: dict[str, Any] = {"n_runs": len(metric_runs)}
    for key in scalar_keys:
        values = np.asarray([run[key] for run in metric_runs], dtype=float)
        summary[key] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary[f"{key}_values"] = values.tolist()
    summary["per_class"] = {}
    for class_name in CLASS_NAMES:
        summary["per_class"][class_name] = {}
        for key in ["precision", "recall", "f1", "support", "ovr_auroc", "ovr_auprc"]:
            raw = [run["per_class"][class_name][key] for run in metric_runs]
            values = np.asarray([np.nan if value is None else value for value in raw], dtype=float)
            finite = values[np.isfinite(values)]
            summary["per_class"][class_name][key] = (
                float(np.mean(finite)) if len(finite) else None
            )
            summary["per_class"][class_name][f"{key}_std"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            )
    summary["confusion_matrices"] = [run["confusion_matrix"] for run in metric_runs]
    return summary


def stratified_bootstrap_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    repeats: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    from sklearn.metrics import f1_score

    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(probabilities).argmax(axis=1)
    rng = np.random.default_rng(seed)
    scores = []
    class_indices = [np.flatnonzero(labels == class_id) for class_id in range(3)]
    for _ in range(repeats):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        scores.append(
            f1_score(labels[sampled], predictions[sampled], average="macro", zero_division=0)
        )
    lower, upper = np.quantile(scores, [0.025, 0.975])
    return {"lower_2.5pct": float(lower), "upper_97.5pct": float(upper)}


def candidate_rules(candidate_names: Sequence[str]) -> list[dict[str, Any]]:
    import itertools

    names = list(candidate_names)
    rules: list[dict[str, Any]] = []
    for name in names:
        rules.append({"models": [name], "weights": [1.0]})
    for left, right in itertools.combinations(names, 2):
        rules.append({"models": [left, right], "weights": [0.5, 0.5]})
        rules.append({"models": [left, right], "weights": [0.75, 0.25]})
        rules.append({"models": [left, right], "weights": [0.25, 0.75]})
    for triple in itertools.combinations(names, 3):
        rules.append({"models": list(triple), "weights": [1 / 3, 1 / 3, 1 / 3]})
        for primary in range(3):
            weights = [0.25, 0.25, 0.25]
            weights[primary] = 0.5
            rules.append({"models": list(triple), "weights": weights})
    return rules


def blend_from_rule(
    prediction_map: Mapping[str, np.ndarray], rule: Mapping[str, Any]
) -> np.ndarray:
    blend = np.zeros_like(next(iter(prediction_map.values())), dtype=float)
    for name, weight in zip(rule["models"], rule["weights"]):
        blend += float(weight) * prediction_map[name]
    blend = np.clip(blend, 1e-9, 1.0)
    return blend / blend.sum(axis=1, keepdims=True)


def select_sparse_rule(
    labels: np.ndarray, prediction_map: Mapping[str, np.ndarray]
) -> tuple[dict[str, Any], pd.DataFrame]:
    from sklearn.metrics import f1_score, log_loss

    rows = []
    rules = candidate_rules(list(prediction_map))
    for rule_index, rule in enumerate(rules):
        probabilities = blend_from_rule(prediction_map, rule)
        rows.append(
            {
                "rule_index": rule_index,
                "models": "+".join(rule["models"]),
                "weights": "+".join(f"{weight:.6f}" for weight in rule["weights"]),
                "n_models": len(rule["models"]),
                "macro_f1": f1_score(
                    labels, probabilities.argmax(axis=1), average="macro", zero_division=0
                ),
                "log_loss": log_loss(labels, probabilities, labels=[0, 1, 2]),
            }
        )
    table = pd.DataFrame(rows)
    best_score = float(table["macro_f1"].max())
    eligible = table.loc[table["macro_f1"] >= best_score - 0.01].sort_values(
        ["n_models", "macro_f1", "log_loss", "models"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )
    chosen_row = eligible.iloc[0]
    chosen = dict(rules[int(chosen_row["rule_index"])])
    base = blend_from_rule(prediction_map, chosen)
    base_loss = float(log_loss(labels, base, labels=[0, 1, 2]))
    temperatures = np.geomspace(0.50, 2.50, 41)
    losses = [
        float(log_loss(labels, apply_temperature(base, temperature), labels=[0, 1, 2]))
        for temperature in temperatures
    ]
    best_index = int(np.argmin(losses))
    if base_loss - losses[best_index] >= 0.005:
        chosen["temperature"] = float(temperatures[best_index])
        chosen["temperature_logloss_gain"] = float(base_loss - losses[best_index])
    else:
        chosen["temperature"] = 1.0
        chosen["temperature_logloss_gain"] = 0.0
    chosen["selection_macro_f1"] = float(chosen_row["macro_f1"])
    chosen["best_grid_macro_f1"] = best_score
    return chosen, table


def rule_probabilities(
    prediction_map: Mapping[str, np.ndarray], rule: Mapping[str, Any]
) -> np.ndarray:
    return apply_temperature(blend_from_rule(prediction_map, rule), rule.get("temperature", 1.0))


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def repeated_oof_predictions(
    views: Mapping[str, pd.DataFrame],
    sequences: np.ndarray,
    labels: np.ndarray,
    seeds: Sequence[int],
    n_splits: int,
    fast_mode: bool,
    device: str,
    checkpoint_dir: Path,
    tag: str,
    subject_hashes: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels, dtype=int)
    sums = {name: np.zeros((len(labels), 3), dtype=float) for name in CANDIDATE_NAMES}
    counts = np.zeros(len(labels), dtype=int)
    assignments: list[dict[str, Any]] = []
    checkpoint_dir = Path(checkpoint_dir)
    for repeat_index, seed in enumerate(seeds):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
        for fold_index, (train_index, valid_index) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            path = checkpoint_dir / f"{tag}_seed{seed}_fold{fold_index}.npz"
            if path.exists():
                loaded = np.load(path)
                saved_index = loaded["valid_index"]
                if not np.array_equal(saved_index, valid_index):
                    raise AssertionError(f"Checkpoint split mismatch: {path}")
                fold_predictions = {name: loaded[name] for name in CANDIDATE_NAMES}
            else:
                fold_predictions = {}
                fold_feature_banks: dict[str, FoldFeatureBank] = {}
                fold_bank_seed = int(seed) * 1000 + fold_index * 100 + 91
                for candidate_index, spec in enumerate(CANDIDATE_SPECS):
                    candidate_bundle, probabilities = fit_predict_candidate(
                        spec,
                        views,
                        sequences,
                        labels,
                        train_index,
                        valid_index,
                        int(seed) * 1000 + fold_index * 100 + candidate_index,
                        fast_mode,
                        device,
                        fold_feature_banks,
                        fold_bank_seed,
                    )
                    fold_predictions[spec.name] = probabilities
                    release_candidate_resources(candidate_bundle)
                    del candidate_bundle
                _atomic_npz(path, valid_index=valid_index, **fold_predictions)
            for name, probabilities in fold_predictions.items():
                sums[name][valid_index] += probabilities
            counts[valid_index] += 1
            assignments.append(
                {
                    "repeat": repeat_index,
                    "seed": int(seed),
                    "fold": fold_index,
                    "train_class_counts": np.bincount(labels[train_index], minlength=3).tolist(),
                    "valid_class_counts": np.bincount(labels[valid_index], minlength=3).tolist(),
                    "valid_subject_hashes": (
                        [subject_hashes[index] for index in valid_index]
                        if subject_hashes is not None else None
                    ),
                }
            )
    if not np.all(counts == len(seeds)):
        raise AssertionError(f"Unexpected OOF prediction counts: {Counter(counts)}")
    return {name: values / counts[:, None] for name, values in sums.items()}, assignments


def run_nested_cv(
    views: Mapping[str, pd.DataFrame],
    sequences: np.ndarray,
    labels: np.ndarray,
    subject_hashes: Sequence[str],
    result_dir: Path,
    outer_seeds: Sequence[int],
    inner_seeds: Sequence[int],
    n_splits: int,
    fast_mode: bool,
    device: str,
    deployment_refit_seeds: Sequence[int],
) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels, dtype=int)
    deployment_refit_seeds = [int(seed) for seed in deployment_refit_seeds]
    if not deployment_refit_seeds:
        raise ValueError("At least one deployment refit seed is required for nested CV.")
    result_dir = Path(result_dir)
    nested_dir = result_dir / "checkpoints" / "nested"
    selected_sum = np.zeros((len(labels), 3), dtype=float)
    prior_sum = np.zeros((len(labels), 3), dtype=float)
    candidate_sum = {
        name: np.zeros((len(labels), 3), dtype=float) for name in CANDIDATE_NAMES
    }
    prediction_counts = np.zeros(len(labels), dtype=int)
    selected_by_seed = {
        int(seed): np.zeros((len(labels), 3), dtype=float) for seed in outer_seeds
    }
    prior_by_seed = {
        int(seed): np.zeros((len(labels), 3), dtype=float) for seed in outer_seeds
    }
    candidate_by_seed = {
        int(seed): {
            name: np.zeros((len(labels), 3), dtype=float) for name in CANDIDATE_NAMES
        }
        for seed in outer_seeds
    }
    counts_by_seed = {
        int(seed): np.zeros(len(labels), dtype=int) for seed in outer_seeds
    }
    folds: list[dict[str, Any]] = []
    for outer_seed in outer_seeds:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=int(outer_seed)
        )
        for fold_index, (outer_train, outer_valid) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            fold_path = nested_dir / f"outer_seed{outer_seed}_fold{fold_index}.joblib"
            if fold_path.exists():
                fold_result = joblib.load(fold_path)
                if not np.array_equal(fold_result["outer_valid"], outer_valid):
                    raise AssertionError(f"Nested checkpoint mismatch: {fold_path}")
                if fold_result.get("deployment_refit_seeds") != deployment_refit_seeds:
                    raise AssertionError(f"Nested refit-seed mismatch: {fold_path}")
            else:
                inner_views = {name: frame.iloc[outer_train] for name, frame in views.items()}
                inner_sequences = sequences[outer_train]
                inner_labels = labels[outer_train]
                inner_predictions, inner_assignments = repeated_oof_predictions(
                    inner_views,
                    inner_sequences,
                    inner_labels,
                    inner_seeds,
                    n_splits,
                    fast_mode,
                    device,
                    nested_dir / f"inner_outer_seed{outer_seed}_fold{fold_index}",
                    "inner",
                    [subject_hashes[index] for index in outer_train],
                )
                rule, rule_table = select_sparse_rule(inner_labels, inner_predictions)
                outer_candidate_sums = {
                    name: np.zeros((len(outer_valid), 3), dtype=float)
                    for name in CANDIDATE_NAMES
                }
                outer_feature_banks: dict[str, FoldFeatureBank] = {}
                outer_bank_seed = 900_091
                for deployment_refit_seed in deployment_refit_seeds:
                    for spec in CANDIDATE_SPECS:
                        candidate_bundle, probabilities = fit_predict_candidate(
                            spec,
                            views,
                            sequences,
                            labels,
                            outer_train,
                            outer_valid,
                            deployment_refit_seed,
                            fast_mode,
                            device,
                            outer_feature_banks,
                            outer_bank_seed,
                        )
                        outer_candidate_sums[spec.name] += probabilities
                        release_candidate_resources(candidate_bundle)
                        del candidate_bundle
                outer_candidate_predictions = {
                    name: probabilities / len(deployment_refit_seeds)
                    for name, probabilities in outer_candidate_sums.items()
                }
                selected = rule_probabilities(outer_candidate_predictions, rule)
                train_prior = np.bincount(labels[outer_train], minlength=3).astype(float)
                train_prior /= train_prior.sum()
                prior_probabilities = np.tile(train_prior, (len(outer_valid), 1))
                fold_result = {
                    "outer_seed": int(outer_seed),
                    "fold_index": fold_index,
                    "deployment_refit_seeds": deployment_refit_seeds,
                    "outer_valid": outer_valid,
                    "outer_valid_hashes": [subject_hashes[index] for index in outer_valid],
                    "train_class_counts": np.bincount(labels[outer_train], minlength=3).tolist(),
                    "valid_class_counts": np.bincount(labels[outer_valid], minlength=3).tolist(),
                    "inner_assignments": inner_assignments,
                    "rule": rule,
                    "inner_rule_top20": rule_table.nlargest(20, "macro_f1").to_dict("records"),
                    "selected_probabilities": selected,
                    "prior_probabilities": prior_probabilities,
                    "candidate_probabilities": outer_candidate_predictions,
                    "selected_metrics": evaluate_probabilities(labels[outer_valid], selected),
                    "prior_metrics": evaluate_probabilities(
                        labels[outer_valid], prior_probabilities
                    ),
                    "candidate_metrics": {
                        name: evaluate_probabilities(labels[outer_valid], probabilities)
                        for name, probabilities in outer_candidate_predictions.items()
                    },
                }
                atomic_joblib_dump(fold_result, fold_path)
            selected_sum[outer_valid] += fold_result["selected_probabilities"]
            prior_sum[outer_valid] += fold_result["prior_probabilities"]
            selected_by_seed[int(outer_seed)][outer_valid] = fold_result[
                "selected_probabilities"
            ]
            prior_by_seed[int(outer_seed)][outer_valid] = fold_result[
                "prior_probabilities"
            ]
            for name in CANDIDATE_NAMES:
                candidate_sum[name][outer_valid] += fold_result["candidate_probabilities"][name]
                candidate_by_seed[int(outer_seed)][name][outer_valid] = fold_result[
                    "candidate_probabilities"
                ][name]
            prediction_counts[outer_valid] += 1
            counts_by_seed[int(outer_seed)][outer_valid] += 1
            folds.append(
                {key: value for key, value in fold_result.items() if key not in {
                    "outer_valid", "selected_probabilities", "prior_probabilities",
                    "candidate_probabilities"
                }}
            )
    if not np.all(prediction_counts == len(outer_seeds)):
        raise AssertionError(f"Unexpected nested counts: {Counter(prediction_counts)}")
    for seed, counts in counts_by_seed.items():
        if not np.all(counts == 1):
            raise AssertionError(f"Unexpected per-repeat nested counts for {seed}: {Counter(counts)}")
    selected = selected_sum / prediction_counts[:, None]
    prior = prior_sum / prediction_counts[:, None]
    candidates = {
        name: values / prediction_counts[:, None] for name, values in candidate_sum.items()
    }
    candidate_metrics = {
        name: evaluate_probabilities(labels, probabilities)
        for name, probabilities in candidates.items()
    }
    selected_repeat_metrics = [
        {
            "outer_seed": int(seed),
            **evaluate_probabilities(labels, selected_by_seed[int(seed)]),
        }
        for seed in outer_seeds
    ]
    prior_repeat_metrics = [
        {
            "outer_seed": int(seed),
            **evaluate_probabilities(labels, prior_by_seed[int(seed)]),
        }
        for seed in outer_seeds
    ]
    candidate_repeat_metrics = {
        name: [
            {
                "outer_seed": int(seed),
                **evaluate_probabilities(labels, candidate_by_seed[int(seed)][name]),
            }
            for seed in outer_seeds
        ]
        for name in CANDIDATE_NAMES
    }
    selected_repeat_summary = summarize_metric_runs(selected_repeat_metrics)
    selected_repeat_summary["interpretation"] = (
        "PRIMARY nested estimate: mean and sample SD across complete repeated outer OOF runs"
    )
    prior_repeat_summary = summarize_metric_runs(prior_repeat_metrics)
    candidate_repeat_summary = {
        name: summarize_metric_runs(metrics)
        for name, metrics in candidate_repeat_metrics.items()
    }

    selected_fold_summary = summarize_metric_runs(
        [fold["selected_metrics"] for fold in folds]
    )
    prior_fold_summary = summarize_metric_runs([fold["prior_metrics"] for fold in folds])
    candidate_fold_summary = {
        name: summarize_metric_runs(
            [fold["candidate_metrics"][name] for fold in folds]
        )
        for name in CANDIDATE_NAMES
    }
    legacy_fold_scores = np.asarray(
        [fold["candidate_metrics"]["elastic_legacy_all_k64"]["macro_f1"] for fold in folds]
    )

    def paired_summary(values: np.ndarray) -> dict[str, Any]:
        values = np.asarray(values, dtype=float)
        return {
            "mean_delta": float(np.mean(values)),
            "std_delta": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "positive_folds": int(np.sum(values > 0)),
            "zero_folds": int(np.sum(values == 0)),
            "negative_folds": int(np.sum(values < 0)),
            "n_folds": int(len(values)),
            "deltas": values.tolist(),
        }

    paired_vs_legacy = {
        "SELECTED_PIPELINE": paired_summary(
            np.asarray([fold["selected_metrics"]["macro_f1"] for fold in folds])
            - legacy_fold_scores
        ),
        **{
            name: paired_summary(
                np.asarray([fold["candidate_metrics"][name]["macro_f1"] for fold in folds])
                - legacy_fold_scores
            )
            for name in CANDIDATE_NAMES
        },
    }
    report = {
        "deployment_refit_seeds": deployment_refit_seeds,
        "deployment_match": (
            "Every outer-valid candidate averages the same number of full-training "
            "refit seeds used by the frozen deployment rule."
        ),
        "selected_pipeline": selected_repeat_summary,
        "training_fold_prior_baseline": prior_repeat_summary,
        "repeat_metrics": {
            "selected_pipeline": selected_repeat_metrics,
            "training_fold_prior": prior_repeat_metrics,
            "candidates": candidate_repeat_metrics,
        },
        "outer_fold_summary": {
            "selected_pipeline": selected_fold_summary,
            "training_fold_prior": prior_fold_summary,
            "candidates": candidate_fold_summary,
        },
        "paired_macro_f1_delta_vs_elastic_legacy_all_k64": paired_vs_legacy,
        "averaged_repeated_oof_diagnostic": {
            "selected_pipeline": evaluate_probabilities(labels, selected),
            "training_fold_prior": evaluate_probabilities(labels, prior),
            "candidates": candidate_metrics,
            "warning": (
                "Secondary diagnostic only: probabilities average complete outer-split "
                "pipelines beyond the frozen deployment refit ensemble."
            ),
        },
        "averaged_oof_macro_f1_ci_diagnostic": stratified_bootstrap_ci(labels, selected),
        "candidate_metrics": candidate_repeat_summary,
        "folds": folds,
        "prediction_counts": prediction_counts.tolist(),
        "subject_hashes": list(subject_hashes),
    }
    write_json(result_dir / "nested_cv_report.json", report)
    rows = [
        {
            "candidate": name,
            "metric_basis": "mean_and_sample_sd_across_complete_outer_repeats",
            **{
                key: value
                for key, value in summary.items()
                if not isinstance(value, (dict, list))
            },
        }
        for name, summary in candidate_repeat_summary.items()
    ]
    pd.DataFrame(rows).to_csv(result_dir / "nested_candidate_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"candidate": name, **summary}
            for name, summary in paired_vs_legacy.items()
        ]
    ).to_csv(result_dir / "nested_paired_summary.csv", index=False)
    fold_metric_rows = []
    for fold in folds:
        base = {"outer_seed": fold["outer_seed"], "fold": fold["fold_index"]}
        fold_metric_rows.append(
            {
                **base,
                "candidate": "SELECTED_PIPELINE",
                **{
                    key: value
                    for key, value in fold["selected_metrics"].items()
                    if not isinstance(value, dict)
                },
            }
        )
        fold_metric_rows.append(
            {
                **base,
                "candidate": "TRAIN_FOLD_PRIOR",
                **{
                    key: value
                    for key, value in fold["prior_metrics"].items()
                    if not isinstance(value, dict)
                },
            }
        )
        for name, metrics in fold["candidate_metrics"].items():
            fold_metric_rows.append(
                {
                    **base,
                    "candidate": name,
                    **{
                        key: value
                        for key, value in metrics.items()
                        if not isinstance(value, dict)
                    },
                }
            )
    pd.DataFrame(fold_metric_rows).to_csv(result_dir / "nested_fold_metrics.csv", index=False)
    np.savez_compressed(
        result_dir / "nested_probabilities.npz",
        selected=selected,
        training_fold_prior=prior,
        labels=labels,
        **{
            f"selected_outer_seed_{seed}": selected_by_seed[int(seed)]
            for seed in outer_seeds
        },
        **{
            f"prior_outer_seed_{seed}": prior_by_seed[int(seed)]
            for seed in outer_seeds
        },
        **candidates,
    )
    return report


def fit_final_rule(
    views: Mapping[str, pd.DataFrame],
    sequences: np.ndarray,
    labels: np.ndarray,
    result_dir: Path,
    final_oof_seeds: Sequence[int],
    n_splits: int,
    fast_mode: bool,
    device: str,
    subject_hashes: Sequence[str] | None = None,
    refit_seeds: Sequence[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_dir = Path(result_dir)
    oof_predictions, assignments = repeated_oof_predictions(
        views,
        sequences,
        labels,
        final_oof_seeds,
        n_splits,
        fast_mode,
        device,
        result_dir / "checkpoints" / "final_oof",
        "final_oof",
        subject_hashes,
    )
    rule, rule_table = select_sparse_rule(labels, oof_predictions)
    selected_oof = rule_probabilities(oof_predictions, rule)
    full_index = np.arange(len(labels))
    if refit_seeds is None:
        refit_seeds = [910_000 + index for index in range(len(final_oof_seeds))]
    refit_seeds = [int(seed) for seed in refit_seeds]
    if not refit_seeds:
        raise ValueError("At least one full-data refit seed is required.")
    final_models: dict[str, Any] = {}
    final_feature_banks: dict[str, FoldFeatureBank] = {}
    tabpfn_directory = result_dir / "final_tabpfn_fitted_models"
    for name in rule["models"]:
        spec = next(spec for spec in CANDIDATE_SPECS if spec.name == name)
        final_models[name] = []
        for refit_seed in refit_seeds:
            bundle, _ = fit_predict_candidate(
                spec,
                views,
                sequences,
                labels,
                full_index,
                full_index[:1],
                refit_seed,
                fast_mode,
                device,
                final_feature_banks,
                900_091,
            )
            # TabPFN keeps a large foundation-model copy alive. Persist each full-data
            # refit immediately so several selected components/seeds never accumulate
            # on the GPU. The returned reference remains part of the frozen bundle.
            if spec.kind in {"tabpfn", "tabpfn_pairwise"}:
                bundle = externalize_tabpfn_models(
                    bundle,
                    tabpfn_directory,
                    filename_prefix=f"{name}_seed{refit_seed}",
                )
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            final_models[name].append(bundle)
    report = {
        "rule": rule,
        "full_data_refit_seeds": refit_seeds,
        "deployment_probability_contract": (
            "Each selected component averages full-data refits across the listed seeds; "
            "the frozen sparse rule and temperature are then applied."
        ),
        "metrics": evaluate_probabilities(labels, selected_oof),
        "macro_f1_ci": stratified_bootstrap_ci(labels, selected_oof),
        "candidate_metrics": {
            name: evaluate_probabilities(labels, probabilities)
            for name, probabilities in oof_predictions.items()
        },
        "fold_assignments": assignments,
        "selection_table_top50": rule_table.nlargest(50, "macro_f1").to_dict("records"),
    }
    write_json(result_dir / "final_oof_report.json", report)
    np.savez_compressed(
        result_dir / "final_oof_probabilities.npz",
        selected=selected_oof,
        labels=np.asarray(labels, dtype=int),
        **oof_predictions,
    )
    rule_table.to_csv(result_dir / "final_rule_search.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": name,
                **{
                    key: value
                    for key, value in metrics.items()
                    if not isinstance(value, dict)
                },
            }
            for name, metrics in report["candidate_metrics"].items()
        ]
    ).to_csv(result_dir / "candidate_oof_metrics.csv", index=False)
    return {
        "rule": rule,
        "models": final_models,
        "full_data_refit_seeds": refit_seeds,
    }, report


def predict_frozen_rule(
    frozen_bundle: Mapping[str, Any],
    views: Mapping[str, pd.DataFrame],
    sequences: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    predictions = {
        name: np.mean(
            [
                predict_fitted_candidate(bundle, views, sequences)
                for bundle in (bundles if isinstance(bundles, list) else [bundles])
            ],
            axis=0,
        )
        for name, bundles in frozen_bundle["models"].items()
    }
    return rule_probabilities(predictions, frozen_bundle["rule"]), predictions


def externalize_tabpfn_models(
    payload: Any,
    directory: Path,
    filename_prefix: str = "model",
) -> Any:
    """Replace final TabPFN objects with official fitted-model file references."""
    from tabpfn import TabPFNClassifier
    from tabpfn.model_loading import save_fitted_tabpfn_model

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    counter = {"value": 0}

    def visit(value: Any) -> Any:
        if isinstance(value, TabPFNClassifier):
            index = counter["value"]
            counter["value"] += 1
            path = directory / f"{filename_prefix}_{index:03d}.tabpfn_fit"
            save_fitted_tabpfn_model(value, str(path))
            value.to("cpu")
            return {"__tabpfn_fitted_model__": path.name}
        if isinstance(value, PairwiseTabPFN):
            return {
                "__pairwise_tabpfn__": True,
                "seed": value.seed,
                "n_estimators": value.n_estimators,
                "models": [visit(model) for model in value.models],
            }
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return tuple(visit(item) for item in value)
        return value

    return visit(payload)


def predict_externalized_frozen_rule(
    serialized_bundle: Mapping[str, Any],
    directory: Path,
    device: str,
    views: Mapping[str, pd.DataFrame],
    sequences: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Predict one refit at a time so frozen TabPFN copies do not exhaust VRAM."""
    from tabpfn import TabPFNClassifier

    def release_tabpfn(value: Any) -> None:
        if isinstance(value, TabPFNClassifier):
            value.to("cpu")
        elif isinstance(value, PairwiseTabPFN):
            for child in value.models:
                release_tabpfn(child)
        elif isinstance(value, Mapping):
            for child in value.values():
                release_tabpfn(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                release_tabpfn(child)

    predictions: dict[str, np.ndarray] = {}
    for name, serialized_refits in serialized_bundle["models"].items():
        refits = serialized_refits if isinstance(serialized_refits, list) else [serialized_refits]
        refit_probabilities = []
        for serialized_refit in refits:
            materialized_refit = materialize_tabpfn_models(
                serialized_refit, directory, device
            )
            refit_probabilities.append(
                predict_fitted_candidate(materialized_refit, views, sequences)
            )
            release_tabpfn(materialized_refit)
            del materialized_refit
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        predictions[name] = np.mean(refit_probabilities, axis=0)
    return rule_probabilities(predictions, serialized_bundle["rule"]), predictions


def materialize_tabpfn_models(payload: Any, directory: Path, device: str) -> Any:
    from tabpfn.model_loading import load_fitted_tabpfn_model

    directory = Path(directory)

    def visit(value: Any) -> Any:
        if isinstance(value, dict) and "__tabpfn_fitted_model__" in value:
            return load_fitted_tabpfn_model(
                str(directory / value["__tabpfn_fitted_model__"]), device=device
            )
        if isinstance(value, dict) and value.get("__pairwise_tabpfn__"):
            model = PairwiseTabPFN(value["seed"], value["n_estimators"], device)
            model.models = [visit(item) for item in value["models"]]
            return model
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, tuple):
            return tuple(visit(item) for item in value)
        return value

    return visit(payload)
