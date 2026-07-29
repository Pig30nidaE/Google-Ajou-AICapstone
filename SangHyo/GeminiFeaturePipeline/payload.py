"""Per-subject Gemini input payloads.

Division of labour (per <gemini_input_design>): **Python computes every number**,
Gemini only relates the numbers to each other.  A payload therefore contains

1. exact descriptive statistics per daily channel,
2. an averaged 24-hour intensity profile and the intensity / sleep-phase shares,
3. circular clock statistics for bedtime, wake time and mid-sleep,
4. order-preserving compressed time courses (weekly means, evenly spaced single
   days, and the largest day-over-day changes),

and nothing else.  Raw source rows are never sent.  Identifiers, absolute
dates, diagnosis and MMSE are structurally absent, and the result is checked by
``guards.assert_payload_is_label_free`` / ``assert_payload_is_mmse_free`` before
it can be serialized for the API.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from . import PAYLOAD_VERSION
from .config import PayloadConfig
from .data import (
    DAILY_CHANNELS,
    HOURLY_MET_COLUMNS,
    INTENSITY_SHARE_COLUMNS,
    SLEEP_PHASE_SHARE_COLUMNS,
    DailyDataset,
)
from .guards import (
    assert_payload_is_label_free,
    assert_payload_is_mmse_free,
    hash_subject_id,
)

__all__ = [
    "CLOCK_CHANNELS",
    "build_subject_payload",
    "build_payloads",
    "payload_hash",
    "payload_size_bytes",
]

#: Clock-hour channels are summarised with circular statistics instead of the
#: linear statistics used for every other channel (23:50 and 00:10 are 20 minutes
#: apart, not 23.7 hours).
CLOCK_CHANNELS: tuple[str, ...] = ("slp_bedtime_hour", "slp_waketime_hour", "slp_midsleep_hour")
_LINEAR_CHANNELS: tuple[str, ...] = tuple(
    name for name in DAILY_CHANNELS if name not in CLOCK_CHANNELS
)


def _round(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _mean_of_observed(values: "pd.Series | np.ndarray", digits: int) -> float | None:
    array = np.asarray(values, dtype=float)
    observed = array[np.isfinite(array)]
    return _round(float(np.mean(observed)), digits) if observed.size else None


def _linear_stats(
    values: np.ndarray,
    day_index: np.ndarray,
    is_weekend: np.ndarray,
    *,
    digits: int,
) -> dict[str, Any]:
    finite = np.isfinite(values)
    n_valid = int(finite.sum())
    stats: dict[str, Any] = {
        "n_valid": n_valid,
        "missing_rate": _round(1.0 - n_valid / max(1, values.size), digits),
    }
    keys = (
        "mean",
        "sd",
        "cv",
        "median",
        "iqr",
        "p10",
        "p90",
        "min",
        "max",
        "trend_per_week",
        "late_minus_early",
        "weekend_minus_weekday",
    )
    if n_valid == 0:
        stats.update({key: None for key in keys})
        return stats

    observed = values[finite]
    days = day_index[finite].astype(float)
    mean = float(np.mean(observed))
    sd = float(np.std(observed, ddof=1)) if n_valid > 1 else 0.0
    q10, q25, median, q75, q90 = (
        float(value) for value in np.quantile(observed, [0.10, 0.25, 0.50, 0.75, 0.90])
    )
    stats.update(
        {
            "mean": _round(mean, digits),
            "sd": _round(sd, digits),
            "cv": _round(sd / abs(mean), digits) if abs(mean) > 1e-9 else None,
            "median": _round(median, digits),
            "iqr": _round(q75 - q25, digits),
            "p10": _round(q10, digits),
            "p90": _round(q90, digits),
            "min": _round(float(np.min(observed)), digits),
            "max": _round(float(np.max(observed)), digits),
        }
    )

    centred_days = days - days.mean()
    denominator = float(np.sum(centred_days**2))
    if n_valid >= 3 and denominator > 0:
        slope = float(np.sum(centred_days * (observed - mean)) / denominator)
        stats["trend_per_week"] = _round(slope * 7.0, digits)
    else:
        stats["trend_per_week"] = None

    if n_valid >= 6:
        thirds = max(1, n_valid // 3)
        order = np.argsort(days, kind="mergesort")
        ordered = observed[order]
        stats["late_minus_early"] = _round(
            float(np.mean(ordered[-thirds:]) - np.mean(ordered[:thirds])), digits
        )
    else:
        stats["late_minus_early"] = None

    weekend_mask = is_weekend[finite].astype(bool)
    if weekend_mask.sum() >= 2 and (~weekend_mask).sum() >= 2:
        stats["weekend_minus_weekday"] = _round(
            float(np.mean(observed[weekend_mask]) - np.mean(observed[~weekend_mask])), digits
        )
    else:
        stats["weekend_minus_weekday"] = None
    return stats


def _circular_stats(hours: np.ndarray, *, digits: int) -> dict[str, Any]:
    finite = np.isfinite(hours)
    observed = hours[finite]
    if observed.size == 0:
        return {"n_valid": 0, "mean_hour": None, "circular_sd_hours": None, "range_hours": None}
    angles = 2.0 * np.pi * observed / 24.0
    resultant = complex(float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles))))
    magnitude = abs(resultant)
    mean_hour = (math.degrees(math.atan2(resultant.imag, resultant.real)) / 360.0 * 24.0) % 24.0
    if magnitude <= 1e-12:
        circular_sd = 24.0 / (2.0 * math.pi) * math.sqrt(-2.0 * math.log(1e-12))
    else:
        circular_sd = 24.0 / (2.0 * math.pi) * math.sqrt(-2.0 * math.log(magnitude))
    deviation = _circular_deviation(observed, mean_hour)
    return {
        "n_valid": int(observed.size),
        "mean_hour": _round(mean_hour, digits),
        "circular_sd_hours": _round(circular_sd, digits),
        "range_hours": _round(float(np.max(deviation) - np.min(deviation)), digits),
    }


def _circular_deviation(hours: np.ndarray, reference_hour: float) -> np.ndarray:
    """Signed hours from ``reference_hour``, wrapped into (-12, +12]."""

    deviation = (np.asarray(hours, dtype=float) - float(reference_hour) + 12.0) % 24.0 - 12.0
    return deviation


def _weekly_summary(
    frame: pd.DataFrame, channels: Sequence[str], *, digits: int
) -> dict[str, Any]:
    weeks = (frame["day_index"].to_numpy() // 7).astype(int)
    unique_weeks = sorted(set(weeks.tolist()))
    summary: dict[str, Any] = {
        "week_index": unique_weeks,
        "days_per_week": [int(np.sum(weeks == week)) for week in unique_weeks],
        "channels": {},
    }
    for channel in channels:
        values = frame[channel].to_numpy(dtype=float)
        weekly: list[float | None] = []
        for week in unique_weeks:
            block = values[weeks == week]
            block = block[np.isfinite(block)]
            weekly.append(_round(float(np.mean(block)), digits) if block.size else None)
        summary["channels"][channel] = weekly
    return summary


def _sampled_series(
    frame: pd.DataFrame, channels: Sequence[str], *, max_points: int, digits: int
) -> dict[str, Any]:
    """Evenly spaced individual days (not averages), so volatility stays visible."""

    n_days = len(frame)
    if n_days <= max_points:
        positions = np.arange(n_days)
    else:
        positions = np.unique(np.rint(np.linspace(0, n_days - 1, max_points)).astype(int))
    subset = frame.iloc[positions]
    series: dict[str, Any] = {
        "day_index": [int(value) for value in subset["day_index"].to_numpy()],
        "is_weekend": [int(value) for value in subset["is_weekend"].to_numpy()],
        "channels": {},
    }
    for channel in channels:
        series["channels"][channel] = [
            _round(float(value), digits) for value in subset[channel].to_numpy(dtype=float)
        ]
    return series


def _largest_changes(
    frame: pd.DataFrame, channels: Sequence[str], *, top_k: int, digits: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    days = frame["day_index"].to_numpy()
    for channel in channels:
        values = frame[channel].to_numpy(dtype=float)
        deltas = np.diff(values)
        gaps = np.diff(days)
        usable = np.isfinite(deltas) & (gaps == 1)
        if not usable.any():
            output[channel] = []
            continue
        indices = np.flatnonzero(usable)
        order = indices[np.argsort(-np.abs(deltas[indices]))][:top_k]
        output[channel] = [
            {
                "day_index": int(days[index + 1]),
                "change_from_previous_day": _round(float(deltas[index]), digits),
            }
            for index in sorted(order.tolist())
        ]
    return output


def _clock_series(frame: pd.DataFrame, *, max_points: int, digits: int) -> dict[str, Any]:
    hours = frame["slp_midsleep_hour"].to_numpy(dtype=float)
    finite = np.isfinite(hours)
    if not finite.any():
        return {"day_index": [], "slp_midsleep_deviation_hours": []}
    stats = _circular_stats(hours, digits=digits)
    reference = stats["mean_hour"] if stats["mean_hour"] is not None else 0.0
    deviation = np.where(finite, _circular_deviation(hours, float(reference)), np.nan)
    n_days = len(frame)
    positions = (
        np.arange(n_days)
        if n_days <= max_points
        else np.unique(np.rint(np.linspace(0, n_days - 1, max_points)).astype(int))
    )
    return {
        "reference_mean_hour": stats["mean_hour"],
        "day_index": [int(value) for value in frame["day_index"].to_numpy()[positions]],
        "slp_midsleep_deviation_hours": [
            _round(float(value), digits) for value in deviation[positions]
        ],
    }


def build_subject_payload(
    subject_frame: pd.DataFrame,
    *,
    subject_ref: str,
    payload_config: PayloadConfig,
    channels: Sequence[str] = _LINEAR_CHANNELS,
) -> dict[str, Any]:
    """Build the complete, de-identified payload for one subject."""

    digits = int(payload_config.round_digits)
    frame = subject_frame.sort_values("day_index", kind="mergesort").reset_index(drop=True)
    day_index = frame["day_index"].to_numpy(dtype=int)
    is_weekend = frame["is_weekend"].to_numpy(dtype=int)
    span = int(day_index.max() - day_index.min() + 1) if len(day_index) else 0
    gaps = np.diff(day_index) - 1 if len(day_index) > 1 else np.asarray([0])

    payload: dict[str, Any] = {
        "payload_version": PAYLOAD_VERSION,
        "subject_ref": subject_ref,
        "observation": {
            "n_days": int(len(frame)),
            "window_span_days": span,
            "coverage_ratio": _round(len(frame) / span, digits) if span else None,
            "weekend_days": int(is_weekend.sum()),
            "longest_gap_days": int(gaps.max()) if gaps.size else 0,
            "first_day_index": int(day_index.min()) if len(day_index) else None,
            "last_day_index": int(day_index.max()) if len(day_index) else None,
        },
        "channels": {},
        "clock": {},
    }

    for channel in channels:
        payload["channels"][channel] = _linear_stats(
            frame[channel].to_numpy(dtype=float), day_index, is_weekend, digits=digits
        )
    for channel in CLOCK_CHANNELS:
        payload["clock"][channel] = _circular_stats(
            frame[channel].to_numpy(dtype=float), digits=digits
        )

    if payload_config.hourly_profile:
        hourly_mean: list[float | None] = []
        hourly_sd: list[float | None] = []
        for column in HOURLY_MET_COLUMNS:
            values = frame[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            hourly_mean.append(_round(float(np.mean(values)), digits) if values.size else None)
            hourly_sd.append(
                _round(float(np.std(values, ddof=1)), digits) if values.size > 1 else None
            )
        payload["hourly_profile"] = {
            "description": "mean movement intensity (MET) per local clock hour, index 0 = 00:00",
            "mean_met_by_hour": hourly_mean,
            "sd_met_by_hour": hourly_sd,
        }

    payload["intensity_profile"] = {
        column.replace("intensity_share_", ""): _mean_of_observed(frame[column], digits)
        for column in INTENSITY_SHARE_COLUMNS
    }
    payload["sleep_phase_profile"] = {
        column.replace("phase_share_", ""): _mean_of_observed(frame[column], digits)
        for column in SLEEP_PHASE_SHARE_COLUMNS
    }

    series_channels = [
        name for name in payload_config.series_channels if name in frame.columns and name not in CLOCK_CHANNELS
    ]
    if payload_config.weekly_summary:
        payload["weekly_summary"] = _weekly_summary(frame, series_channels, digits=digits)
    payload["series"] = _sampled_series(
        frame, series_channels, max_points=int(payload_config.max_series_points), digits=digits
    )
    payload["series"]["clock"] = _clock_series(
        frame, max_points=int(payload_config.max_series_points), digits=digits
    )
    payload["largest_daily_changes"] = _largest_changes(
        frame, series_channels[:3], top_k=3, digits=digits
    )

    assert_payload_is_label_free(payload, context=f"payload[{subject_ref}]")
    assert_payload_is_mmse_free(payload, context=f"payload[{subject_ref}]")
    return payload


def build_payloads(
    dataset: DailyDataset,
    *,
    payload_config: PayloadConfig,
    salt: str,
) -> dict[str, dict[str, Any]]:
    """Build one payload per subject, keyed by the raw subject id (kept in memory only)."""

    payloads: dict[str, dict[str, Any]] = {}
    for subject_id, subject_frame in dataset.frame.groupby("subject_id", sort=True):
        subject_ref = hash_subject_id(str(subject_id), salt=salt)
        payloads[str(subject_id)] = build_subject_payload(
            subject_frame, subject_ref=subject_ref, payload_config=payload_config
        )
    return payloads


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def payload_size_bytes(payload: Mapping[str, Any]) -> int:
    return len(canonical_json(payload).encode("utf-8"))
