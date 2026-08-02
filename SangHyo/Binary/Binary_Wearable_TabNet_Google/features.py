"""Label-free subject-level feature bank inspired by the 0.848 ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .data import SubjectSequenceDataset


AGGREGATE_STATISTICS = (
    "median",
    "mean",
    "std",
    "p10",
    "p90",
    "iqr",
    "mad",
    "rank_slope",
    "recent14_minus_previous14_median",
)

CORRELATION_PAIRS = (
    ("activity__scalar__steps", "sleep__scalar__total"),
    ("activity__scalar__inactive", "sleep__scalar__efficiency"),
    ("activity__scalar__average_met", "sleep__scalar__hr_average"),
    ("activity__scalar__score", "sleep__scalar__score"),
    ("activity__state__high_ratio", "sleep__stage__deep_ratio"),
    ("activity__circadian__relative_amplitude", "sleep__scalar__rmssd"),
)

FORBIDDEN_NAME_FRAGMENTS = (
    "mmse",
    "cognitive",
    "email",
    "subject_id",
    "diag",
    "label",
    "coverage",
    "observed_count",
    "sequence_length",
    "absolute_date",
    "non_wear",
    "nonwear",
    "mask",
)


@dataclass(frozen=True)
class SubjectFeatureTable:
    subject_ids: np.ndarray
    X: pd.DataFrame
    y: np.ndarray | None
    audit: dict


def assert_feature_contract(names: Sequence[str]) -> None:
    values = [str(name) for name in names]
    if not values or len(values) != len(set(values)):
        raise AssertionError("Feature names must be non-empty and unique")
    bad = [
        name
        for name in values
        if any(fragment in name.lower() for fragment in FORBIDDEN_NAME_FRAGMENTS)
        or not name.lower().startswith(("activity__", "sleep__", "cross__"))
    ]
    if bad:
        raise AssertionError(f"Forbidden/non-wearable feature names: {bad[:20]}")


def _rank_slope(values: np.ndarray) -> float:
    valid = np.isfinite(values)
    if valid.sum() < 4:
        return np.nan
    y = values[valid].astype(np.float64)
    x = np.linspace(0.0, 1.0, len(y), dtype=np.float64)
    centered_x = x - x.mean()
    denominator = float(centered_x @ centered_x)
    return float(centered_x @ (y - y.mean()) / denominator) if denominator else 0.0


def _summary(values: np.ndarray) -> tuple[float, ...]:
    if len(values) != 28:
        raise ValueError(f"Every subject summary must use exactly 28 observations; got {len(values)}")
    finite = values[np.isfinite(values)]
    if not len(finite):
        return (np.nan,) * len(AGGREGATE_STATISTICS)
    p10, q25, median, q75, p90 = np.quantile(
        finite, [0.10, 0.25, 0.50, 0.75, 0.90]
    )
    recent = values[-14:]
    previous = values[:14]
    recent = recent[np.isfinite(recent)]
    previous = previous[np.isfinite(previous)]
    tail_shift = (
        float(np.median(recent) - np.median(previous))
        if len(recent) and len(previous)
        else np.nan
    )
    return (
        float(median),
        float(np.mean(finite)),
        float(np.std(finite)),
        float(p10),
        float(p90),
        float(q75 - q25),
        float(np.median(np.abs(finite - median))),
        _rank_slope(values),
        tail_shift,
    )


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 8:
        return np.nan
    x, y = left[valid], right[valid]
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def build_subject_feature_table(dataset: SubjectSequenceDataset) -> SubjectFeatureTable:
    """Collapse each subject sequence to exactly one model row."""

    base_names = list(dataset.feature_names)
    assert_feature_contract(base_names)
    index = {name: position for position, name in enumerate(base_names)}
    resolved_pairs = [pair for pair in CORRELATION_PAIRS if set(pair).issubset(index)]
    rows: list[dict[str, float]] = []
    for sequence in dataset.sequences:
        # A fixed, equal-length tail prevents model features from exposing the
        # subject's collection length/coverage.  Sequence length remains EDA-only.
        values = np.asarray(sequence, dtype=np.float64)[-28:]
        if values.shape[0] != 28:
            raise ValueError("Every subject must provide at least 28 aligned observations")
        row: dict[str, float] = {}
        for feature_index, name in enumerate(base_names):
            summary = _summary(values[:, feature_index])
            for statistic, value in zip(AGGREGATE_STATISTICS, summary):
                row[f"{name}__aggregate__{statistic}"] = value
        for left, right in resolved_pairs:
            pair_name = f"cross__{left.removeprefix('activity__')}__x__{right.removeprefix('sleep__')}"
            row[pair_name] = _safe_correlation(values[:, index[left]], values[:, index[right]])
        rows.append(row)
    frame = pd.DataFrame(rows, dtype=np.float32)
    assert_feature_contract(frame.columns.tolist())
    if len(frame) != len(dataset.subject_ids):
        raise AssertionError("Subject aggregation changed row count")
    return SubjectFeatureTable(
        subject_ids=np.asarray(dataset.subject_ids, dtype=str),
        X=frame,
        y=None if dataset.y is None else np.asarray(dataset.y, dtype=np.int64),
        audit={
            "representation": "one row per subject",
            "fixed_observation_window": "last 28 aligned Activity/Sleep observations",
            "base_daily_features": len(base_names),
            "aggregate_statistics": list(AGGREGATE_STATISTICS),
            "resolved_correlation_pairs": [list(pair) for pair in resolved_pairs],
            "engineered_features": int(frame.shape[1]),
            "missing_fraction": float(frame.isna().to_numpy().mean()),
            "observation_count_feature_emitted": False,
            "cognitive_feature_emitted": False,
        },
    )


__all__ = [
    "AGGREGATE_STATISTICS",
    "SubjectFeatureTable",
    "assert_feature_contract",
    "build_subject_feature_table",
]
