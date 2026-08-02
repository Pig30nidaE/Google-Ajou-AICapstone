"""One-row-per-subject feature views for the Google YDF experiment."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import (
    ACTIVITY_MEAN_COLUMNS,
    ACTIVITY_RICH_COLUMNS,
    LeakageContractError,
    MMSE_DOMAINS,
    MMSE_ITEMS,
    SLEEP_MEAN_COLUMNS,
    SLEEP_RICH_COLUMNS,
    SplitSources,
    binary_target,
)

LOCAL_TIMEZONE = "Asia/Seoul"
EXPECTED_MMSE_ALL_FEATURES = 50
EXPECTED_MMSE39_FEATURES = 39
EXPECTED_WEARABLE_FEATURES = 112
EXPECTED_UNION_FEATURES = 162
EXPECTED_ALL151_FEATURES = 151

_FORBIDDEN_FEATURE_TOKENS = frozenset(
    {
        "count",
        "coverage",
        "days",
        "diag",
        "diagnosis",
        "doctor",
        "email",
        "identifier",
        "label",
        "missing",
        "mmse_kind",
        "mmse_num",
        "observation",
        "sample",
        "subject",
        "target",
    }
)


@dataclass(frozen=True)
class SubjectTable:
    """A unique subject table with explicit immutable feature views."""

    split: str
    subject_ids: np.ndarray
    X: np.ndarray
    feature_names: tuple[str, ...]
    views: Mapping[str, tuple[int, ...]]
    y: np.ndarray | None
    diagnoses: np.ndarray | None

    def __post_init__(self) -> None:
        ids = np.asarray(self.subject_ids).astype(str)
        matrix = np.asarray(self.X, dtype=np.float64)
        if ids.ndim != 1 or len(set(ids)) != len(ids):
            raise LeakageContractError(
                "Subject table must contain one unique row per subject"
            )
        if matrix.shape != (len(ids), len(self.feature_names)):
            raise LeakageContractError("Subject table shape/schema mismatch")
        if np.isinf(matrix).any():
            raise LeakageContractError("Feature table contains infinite values")
        assert_feature_names(self.feature_names)
        available = set(range(matrix.shape[1]))
        for view, columns in self.views.items():
            if not columns or not set(columns).issubset(available):
                raise LeakageContractError(f"Invalid feature view: {view}")
        if len(self.views["mmse39"]) != EXPECTED_MMSE39_FEATURES:
            raise LeakageContractError("mmse39 view width changed")
        if len(self.views["mmse_all"]) != EXPECTED_MMSE_ALL_FEATURES:
            raise LeakageContractError("mmse_all view width changed")
        if len(self.views["all151"]) != EXPECTED_ALL151_FEATURES:
            raise LeakageContractError("all151 view width changed")
        if self.y is not None:
            labels = np.asarray(self.y, dtype=np.int64)
            if labels.shape != (len(ids),) or set(np.unique(labels)) != {0, 1}:
                raise LeakageContractError("Training target must contain both classes")

    def view_indices(self, view: str) -> np.ndarray:
        if view not in self.views:
            raise KeyError(f"Unknown view {view!r}; choices={sorted(self.views)}")
        return np.asarray(self.views[view], dtype=np.int64)


def _tokens(name: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(name).lower())
        if token
    }


def assert_feature_names(names: Sequence[str]) -> None:
    values = tuple(map(str, names))
    if not values or len(values) != len(set(values)):
        raise LeakageContractError("Feature names must be non-empty and unique")
    offenders: list[str] = []
    for name in values:
        lowered = name.lower()
        tokens = _tokens(name)
        if tokens & _FORBIDDEN_FEATURE_TOKENS:
            offenders.append(name)
            continue
        if any(
            forbidden in lowered
            for forbidden in (
                "diag_nm",
                "diag_seq",
                "doctor_nm",
                "mmse_kind",
                "mmse_num",
                "non_wear",
                "sample_email",
                "subject_id",
            )
        ):
            offenders.append(name)
    if offenders:
        raise LeakageContractError(
            f"Forbidden direct-leakage feature names: {sorted(offenders)[:12]}"
        )


def _score_item(raw: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(raw, errors="coerce")
    return numeric.map({1.0: 0.0, 2.0: 1.0}).astype(float)


def build_mmse_features(
    mmse: pd.DataFrame,
    subject_ids: Sequence[str],
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Build 50 safe MMSE features and return the 39-feature anchor indices."""

    wanted = list(map(str, subject_ids))
    if set(wanted) != set(map(str, mmse.index)):
        raise LeakageContractError("MMSE subjects do not match the subject table")
    aligned = mmse.reindex(wanted).copy()
    total = pd.to_numeric(aligned["TOTAL"], errors="coerce").astype(float)
    invalid_exam = ~total.between(1.0, 30.0, inclusive="both")
    total.loc[invalid_exam] = np.nan
    scored = pd.DataFrame(
        {item: _score_item(aligned[item]) for item in MMSE_ITEMS},
        index=aligned.index,
    )
    scored.loc[invalid_exam, :] = np.nan

    columns: dict[str, pd.Series] = {
        "mmse__total": total,
        "mmse__total_deficit": 30.0 - total,
    }
    domains: dict[str, pd.Series] = {}
    for domain, items in MMSE_DOMAINS.items():
        values = scored.loc[:, list(items)]
        score = values.sum(axis=1, min_count=len(items))
        domains[domain] = score
        columns[f"mmse__domain__{domain}_score"] = score
        columns[f"mmse__domain__{domain}_fraction"] = score / float(len(items))
    for item in MMSE_ITEMS:
        columns[f"mmse__item__{item.lower()}_correct"] = scored[item]

    reconstructed = scored.sum(axis=1, min_count=len(MMSE_ITEMS))
    columns["mmse__reconstructed_total"] = reconstructed
    columns["mmse__failed_items"] = float(len(MMSE_ITEMS)) - reconstructed
    columns["mmse__recall_deficit"] = 3.0 - domains["recall"]
    columns["mmse__below_24"] = (total < 24.0).where(total.notna()).astype(float)
    columns["mmse__below_27"] = (total < 27.0).where(total.notna()).astype(float)
    orientation = domains["orient_time"] + domains["orient_place"]
    columns["mmse__any_orientation_error"] = (
        orientation < 10.0
    ).where(orientation.notna()).astype(float)

    frame = pd.DataFrame(columns, index=aligned.index, dtype=float)
    anchor_names = {
        "mmse__total",
        "mmse__failed_items",
        "mmse__recall_deficit",
        *(f"mmse__domain__{domain}_score" for domain in MMSE_DOMAINS),
        *(f"mmse__item__{item.lower()}_correct" for item in MMSE_ITEMS),
    }
    anchor = tuple(
        index for index, name in enumerate(frame.columns) if name in anchor_names
    )
    if frame.shape[1] != EXPECTED_MMSE_ALL_FEATURES:
        raise LeakageContractError(
            f"MMSE-all schema changed: {frame.shape[1]} features"
        )
    if len(anchor) != EXPECTED_MMSE39_FEATURES:
        raise LeakageContractError(f"MMSE39 schema changed: {len(anchor)} features")
    return frame, anchor


def _aggregate(
    frame: pd.DataFrame,
    *,
    rich: Sequence[str],
    mean_only: Sequence[str],
) -> pd.DataFrame:
    work = frame.copy()
    work["_sid"] = work["EMAIL"].astype(str)
    for column in (*rich, *mean_only):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    grouped = work.groupby("_sid", sort=True)
    parts: list[pd.DataFrame] = []
    mean = grouped[list(rich)].mean()
    std = grouped[list(rich)].std(ddof=0)
    coefficient = std / mean.abs().replace(0.0, np.nan)
    parts.extend(
        [
            mean.add_prefix("w_").add_suffix("__mean"),
            std.add_prefix("w_").add_suffix("__std"),
            coefficient.add_prefix("w_").add_suffix("__cv"),
            grouped[list(mean_only)].mean().add_prefix("w_").add_suffix("__mean"),
        ]
    )
    return pd.concat(parts, axis=1)


def _activity_composition(activity: pd.DataFrame) -> pd.DataFrame:
    """Two physiological composition summaries replacing collection-day counts."""

    work = activity.copy()
    work["_sid"] = work["EMAIL"].astype(str)
    channels = (
        "activity_rest",
        "activity_inactive",
        "activity_low",
        "activity_medium",
        "activity_high",
    )
    for column in channels:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    full_day = work[list(channels)].sum(axis=1, min_count=len(channels)).replace(
        0.0, np.nan
    )
    active = (
        work["activity_low"] + work["activity_medium"] + work["activity_high"]
    )
    composition = pd.DataFrame(
        {
            "_sid": work["_sid"],
            "w_activity_active_fraction__mean": active / full_day,
            "w_activity_moderate_high_fraction__mean": (
                work["activity_medium"] + work["activity_high"]
            )
            / active.replace(0.0, np.nan),
            "w_activity_high_fraction_of_active__mean": (
                work["activity_high"] / active.replace(0.0, np.nan)
            ),
        },
        index=work.index,
    )
    return composition.groupby("_sid", sort=True).mean(numeric_only=True)


def _clock_seconds(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    try:
        local = parsed.dt.tz_convert(LOCAL_TIMEZONE)
    except (AttributeError, TypeError):
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (
        local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    ).astype(float)


def _circular_stats(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan")
    theta = 2.0 * np.pi * (finite % 86_400.0) / 86_400.0
    cosine = float(np.cos(theta).mean())
    sine = float(np.sin(theta).mean())
    resultant = float(np.hypot(cosine, sine))
    mean_hours = (
        np.arctan2(sine, cosine) % (2.0 * np.pi)
    ) * 24.0 / (2.0 * np.pi)
    if finite.size < 2 or resultant <= 1e-12:
        return float(mean_hours), float("nan")
    circular_sd = (
        np.sqrt(-2.0 * np.log(min(resultant, 1.0)))
        * 24.0
        / (2.0 * np.pi)
    )
    return float(mean_hours), float(circular_sd)


def _circadian(sleep: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame({"_sid": sleep["EMAIL"].astype(str)})
    frame["bedtime"] = _clock_seconds(sleep["sleep_bedtime_start"]).to_numpy()
    frame["waketime"] = _clock_seconds(sleep["sleep_bedtime_end"]).to_numpy()
    frame["midpoint"] = pd.to_numeric(
        sleep["sleep_midpoint_time"], errors="coerce"
    ).to_numpy()
    rows: dict[str, dict[str, float]] = {}
    for subject_id, group in frame.groupby("_sid", sort=True):
        record: dict[str, float] = {}
        for channel in ("bedtime", "waketime", "midpoint"):
            mean_hours, circular_sd = _circular_stats(group[channel].to_numpy())
            record[f"w_circ_{channel}__mean_h"] = mean_hours
            record[f"w_circ_{channel}__circsd_h"] = circular_sd
        rows[str(subject_id)] = record
    return pd.DataFrame.from_dict(rows, orient="index")


def _sleep_architecture(sleep: pd.DataFrame) -> pd.DataFrame:
    work = sleep.copy()
    work["_sid"] = work["EMAIL"].astype(str)
    for column in (
        "sleep_total",
        "sleep_duration",
        "sleep_deep",
        "sleep_rem",
        "sleep_light",
        "sleep_awake",
    ):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    denominator = work["sleep_total"].replace(0.0, np.nan)
    derived = pd.DataFrame(index=work.index)
    for stage in ("sleep_deep", "sleep_rem", "sleep_light", "sleep_awake"):
        derived[f"ratio_{stage}"] = work[stage] / denominator
    derived["fragmentation"] = work["sleep_awake"] / work[
        "sleep_duration"
    ].replace(0.0, np.nan)
    derived["_sid"] = work["_sid"]
    grouped = derived.groupby("_sid", sort=True)
    return pd.concat(
        [
            grouped.mean(numeric_only=True)
            .add_prefix("w_arch_")
            .add_suffix("__mean"),
            grouped["fragmentation"]
            .std(ddof=0)
            .rename("w_arch_fragmentation__std")
            .to_frame(),
        ],
        axis=1,
    )


def build_wearable_features(
    activity: pd.DataFrame,
    sleep: pd.DataFrame,
    subject_ids: Sequence[str],
) -> pd.DataFrame:
    """Build a leakage-safe 112-feature bank derived from the prior run.

    The two prior observation-day counts and non-wear coverage are excluded
    and replaced with activity-composition measurements, preserving the
    audited 151-column MMSE39+wearable candidate view without using collection
    volume.
    """

    wanted = list(map(str, subject_ids))
    activity_features = _aggregate(
        activity,
        rich=ACTIVITY_RICH_COLUMNS,
        mean_only=ACTIVITY_MEAN_COLUMNS,
    )
    sleep_features = _aggregate(
        sleep,
        rich=SLEEP_RICH_COLUMNS,
        mean_only=SLEEP_MEAN_COLUMNS,
    )
    frame = pd.concat(
        [
            activity_features,
            sleep_features,
            _activity_composition(activity),
            _sleep_architecture(sleep),
            _circadian(sleep),
        ],
        axis=1,
    )
    missing = sorted(set(wanted) - set(map(str, frame.index)))
    extra = sorted(set(map(str, frame.index)) - set(wanted))
    if missing or extra:
        raise LeakageContractError(
            f"Wearable subject mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )
    frame = frame.reindex(wanted)
    if frame.shape[1] != EXPECTED_WEARABLE_FEATURES:
        raise LeakageContractError(
            f"Wearable schema changed: {frame.shape[1]} != "
            f"{EXPECTED_WEARABLE_FEATURES}"
        )
    return frame.astype(float)


def build_subject_table(sources: SplitSources) -> SubjectTable:
    """Assemble 39-MMSE, all-MMSE, and 151-candidate immutable views."""

    if sources.diagnoses is None:
        subject_ids = sorted(map(str, sources.mmse.index))
        y = None
        diagnoses = None
    else:
        subject_ids = list(map(str, sources.diagnoses.index))
        if set(subject_ids) != set(map(str, sources.mmse.index)):
            raise LeakageContractError("Diagnosis and MMSE subject sets differ")
        aligned_diagnoses = sources.diagnoses.reindex(subject_ids)
        y = binary_target(aligned_diagnoses).to_numpy(dtype=np.int64)
        diagnoses = aligned_diagnoses.to_numpy(dtype=str)

    mmse, anchor_local = build_mmse_features(sources.mmse, subject_ids)
    wearable = build_wearable_features(
        sources.activity, sources.sleep, subject_ids
    )
    union = pd.concat([mmse, wearable], axis=1)
    if union.shape[1] != EXPECTED_UNION_FEATURES:
        raise LeakageContractError(
            f"Union schema changed: {union.shape[1]} != {EXPECTED_UNION_FEATURES}"
        )
    names = tuple(map(str, union.columns))
    assert_feature_names(names)
    wearable_indices = tuple(range(mmse.shape[1], union.shape[1]))
    all151 = tuple((*anchor_local, *wearable_indices))
    views = {
        "mmse39": tuple(anchor_local),
        "mmse_all": tuple(range(mmse.shape[1])),
        "all151": all151,
    }
    return SubjectTable(
        split=sources.split,
        subject_ids=np.asarray(subject_ids, dtype=str),
        X=union.to_numpy(dtype=np.float32),
        feature_names=names,
        views=views,
        y=y,
        diagnoses=diagnoses,
    )


__all__ = [
    "EXPECTED_ALL151_FEATURES",
    "EXPECTED_MMSE39_FEATURES",
    "EXPECTED_MMSE_ALL_FEATURES",
    "EXPECTED_UNION_FEATURES",
    "EXPECTED_WEARABLE_FEATURES",
    "SubjectTable",
    "assert_feature_names",
    "build_mmse_features",
    "build_subject_table",
    "build_wearable_features",
]
