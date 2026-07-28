"""Label-blind subject feature construction for both champion tracks.

The wearable table bank is deliberately built from the same fixed 28 most
recent observations for every subject.  It never exports sequence length,
coverage, padding, a missingness rate, an absolute date, or a crop position.
All population-level imputation, clipping, scaling, and feature selection are
performed later inside the relevant CV training fold.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import (
    AccessAudit,
    LeakageContractError,
    MMSE_DOMAINS,
    MMSE_ITEMS,
    SubjectSequenceDataset,
    assert_subject_alignment,
    binary_target,
    normalise_track,
)

FIXED_SUMMARY_DAYS = 28
RECENT_SUMMARY_DAYS = 7
SUMMARY_STATISTICS = (
    "median",
    "iqr",
    "mad",
    "p10",
    "p90",
    "normalized_theil_sen",
    "recent7_minus_previous21_median",
)
EXPECTED_WEARABLE_SOURCE_FEATURE_COUNT = 113
EXPECTED_WEARABLE_SOURCE_SCHEMA_SHA256 = (
    "ceadeaf72d7306715df068d3050f5b27f5e9483f0f3c28bef8353014a723011b"
)

# Tokens related to how a record was acquired or labelled are not model inputs.
# ``targets`` is intentionally not banned: activity_score_meet_daily_targets is
# a genuine Oura activity score; the singular ML token ``target`` is banned.
_FORBIDDEN_EXACT_TOKENS = frozenset(
    {
        "id",
        "identifier",
        "sample",
        "email",
        "diag",
        "diagnosis",
        "label",
        "target",
        "period",
        "sequence",
        "length",
        "coverage",
        "count",
        "counts",
        "mask",
        "padding",
        "missing",
        "missingness",
        "nonwear",
        "order",
        "index",
    }
)
_FORBIDDEN_WEARABLE_TIME_TOKENS = frozenset(
    {
        "date",
        "datetime",
        "timestamp",
        "day",
        "weekday",
        "calendar",
        "elapsed",
        "start",
        "end",
        "time",
    }
)
_ALLOWED_WEARABLE_TIME_COMPONENTS = frozenset(
    {
        # This is an Oura biological activity score, not a collection time.
        "activity__scalar__score_recovery_time",
    }
)
_FORBIDDEN_WEARABLE_SUBSTRINGS = (
    "absolute_date",
    "activity_day_start",
    "activity_day_end",
    "period_id",
    "sample_order",
    "observed_count",
    "observed_day",
    "observed_night",
    "valid_count",
    "valid_ratio",
    "raw_length",
    "sequence_length",
    "missing_ratio",
    "calendar_gap",
    "span_day",
    "delta_since",
    "nonwear",
    "non_wear",
)


def _tokens(name: str) -> set[str]:
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def assert_wearable_source_contract(feature_names: Sequence[str]) -> None:
    """Lock the audited 113-channel biological source schema fail-closed."""

    names = tuple(map(str, feature_names))
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    if (
        len(names) != EXPECTED_WEARABLE_SOURCE_FEATURE_COUNT
        or digest != EXPECTED_WEARABLE_SOURCE_SCHEMA_SHA256
    ):
        raise LeakageContractError(
            "Wearable source schema changed; review it before training "
            f"(count={len(names)}, sha256={digest})"
        )
    offenders: list[str] = []
    for name in names:
        lowered = name.lower()
        tokens = _tokens(name)
        allowed_time = any(
            component in lowered
            for component in _ALLOWED_WEARABLE_TIME_COMPONENTS
        )
        forbidden_time = bool(tokens & _FORBIDDEN_WEARABLE_TIME_TOKENS)
        if "time" in tokens and allowed_time:
            forbidden_time = bool(
                (tokens & _FORBIDDEN_WEARABLE_TIME_TOKENS) - {"time"}
            )
        if (
            not lowered.startswith(("activity__", "sleep__"))
            or any(
                proxy in lowered for proxy in _FORBIDDEN_WEARABLE_SUBSTRINGS
            )
            or forbidden_time
        ):
            offenders.append(name)
    if offenders:
        raise LeakageContractError(
            "Wearable acquisition/calendar proxy feature(s): "
            + repr(sorted(offenders)[:25])
        )


def assert_feature_contract(
    feature_names: Sequence[str],
    *,
    track: str,
) -> None:
    """Reject identifiers, outcomes, protocol proxies, and cross-track inputs."""

    resolved_track = normalise_track(track)
    names = [str(name) for name in feature_names]
    if not names or len(names) != len(set(names)):
        raise LeakageContractError("Feature names must be non-empty and unique")
    offenders: list[str] = []
    for name in names:
        lowered = name.lower()
        tokens = _tokens(name)
        if tokens & _FORBIDDEN_EXACT_TOKENS:
            offenders.append(name)
            continue
        if "non_wear" in lowered or "n_days" in lowered:
            offenders.append(name)
            continue
        if not (
            lowered.startswith("activity__")
            or lowered.startswith("sleep__")
            or lowered.startswith("mmse__")
        ):
            offenders.append(name)
            continue
        if lowered.startswith(("activity__", "sleep__")):
            allowed_time = any(
                component in lowered
                for component in _ALLOWED_WEARABLE_TIME_COMPONENTS
            )
            forbidden_time = bool(tokens & _FORBIDDEN_WEARABLE_TIME_TOKENS)
            if "time" in tokens and allowed_time:
                forbidden_time = bool(
                    (tokens & _FORBIDDEN_WEARABLE_TIME_TOKENS) - {"time"}
                )
            if (
                any(
                    proxy in lowered
                    for proxy in _FORBIDDEN_WEARABLE_SUBSTRINGS
                )
                or forbidden_time
            ):
                offenders.append(name)
                continue
        if resolved_track == "wearable" and lowered.startswith("mmse__"):
            offenders.append(name)
    if offenders:
        raise LeakageContractError(
            "Forbidden or cross-track model feature(s): "
            + repr(sorted(offenders)[:25])
        )


@dataclass(frozen=True)
class ChampionDataset:
    """One row and, separately, one raw daily sequence per subject."""

    track: str
    subject_ids: np.ndarray
    X: np.ndarray
    feature_names: tuple[str, ...]
    views: Mapping[str, tuple[int, ...]]
    sequences: tuple[np.ndarray, ...]
    sequence_feature_names: tuple[str, ...]
    y: np.ndarray | None
    diagnoses: np.ndarray | None
    audit: Mapping[str, object]

    def __post_init__(self) -> None:
        resolved_track = normalise_track(self.track)
        ids = np.asarray(self.subject_ids).astype(str)
        values = np.asarray(self.X)
        if ids.ndim != 1 or len(set(ids)) != len(ids):
            raise ValueError("subject_ids must be a unique one-dimensional vector")
        if values.shape != (len(ids), len(self.feature_names)):
            raise ValueError("X shape does not match subjects and feature names")
        if np.isinf(values).any():
            raise ValueError("X must not contain infinite values")
        if len(self.sequences) != len(ids):
            raise ValueError("sequences and subject_ids have different lengths")
        assert_wearable_source_contract(self.sequence_feature_names)
        if self.y is not None:
            labels = np.asarray(self.y)
            if labels.shape != (len(ids),) or not set(np.unique(labels)).issubset(
                {0, 1}
            ):
                raise ValueError("y must be one binary label per subject")
        assert_feature_contract(self.feature_names, track=resolved_track)
        all_indices = set(range(values.shape[1]))
        for name, indices in self.views.items():
            if not str(name) or not set(indices).issubset(all_indices):
                raise ValueError(f"Invalid feature view {name!r}")

    def view_indices(self, name: str) -> np.ndarray:
        if name not in self.views:
            raise KeyError(f"Unknown feature view {name!r}; choices={sorted(self.views)}")
        return np.asarray(self.views[name], dtype=np.int64)

    def view_matrix(self, name: str) -> np.ndarray:
        return np.asarray(self.X[:, self.view_indices(name)], dtype=np.float64)

    def subset(self, indices: Sequence[int]) -> "ChampionDataset":
        selected = np.asarray(indices, dtype=np.int64)
        diagnoses = (
            None if self.diagnoses is None else np.asarray(self.diagnoses)[selected]
        )
        labels = None if self.y is None else np.asarray(self.y)[selected]
        return ChampionDataset(
            track=self.track,
            subject_ids=np.asarray(self.subject_ids)[selected],
            X=np.asarray(self.X)[selected],
            feature_names=self.feature_names,
            views=self.views,
            sequences=tuple(self.sequences[index] for index in selected),
            sequence_feature_names=self.sequence_feature_names,
            y=labels,
            diagnoses=diagnoses,
            audit=self.audit,
        )


def _fixed_recent_window(sequence: np.ndarray, days: int) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < int(days):
        raise LeakageContractError(
            f"Every wearable sequence must contain at least {days} observations"
        )
    # The most recent equal-sized biological window is used for everyone.  Its
    # start/end timestamp and the discarded sequence length are never emitted.
    return values[-int(days) :, :].copy()


def _nan_quantile(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.nanquantile(values, quantiles, axis=1)


def _normalized_theil_sen_nan(values: np.ndarray) -> np.ndarray:
    n_subjects, n_steps, n_features = values.shape
    ranks = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    slopes: list[np.ndarray] = []
    for left in range(n_steps - 1):
        denominator = ranks[left + 1 :] - ranks[left]
        delta = values[:, left + 1 :, :] - values[:, left : left + 1, :]
        slopes.append(delta / denominator[None, :, None])
    joined = np.concatenate(slopes, axis=1)
    with np.errstate(all="ignore"):
        result = np.nanmedian(joined, axis=1)
    if result.shape != (n_subjects, n_features):
        raise AssertionError("Unexpected Theil-Sen result shape")
    return result


def summarize_wearable_sequences(
    sequences: Sequence[np.ndarray],
    feature_names: Sequence[str],
    *,
    days: int = FIXED_SUMMARY_DAYS,
    recent_days: int = RECENT_SUMMARY_DAYS,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Create fixed, per-subject raw summaries without consulting labels."""

    names = tuple(map(str, feature_names))
    assert_wearable_source_contract(names)
    if not 1 <= int(recent_days) < int(days):
        raise ValueError("recent_days must leave at least one previous observation")
    windows = np.stack(
        [_fixed_recent_window(sequence, int(days)) for sequence in sequences]
    )
    if windows.shape[2] != len(names):
        raise ValueError("Sequence width and feature-name width differ")
    q10, q25, median, q75, q90 = _nan_quantile(
        windows, (0.10, 0.25, 0.50, 0.75, 0.90)
    )
    with np.errstate(all="ignore"):
        mad = np.nanmedian(np.abs(windows - median[:, None, :]), axis=1)
        slope = _normalized_theil_sen_nan(windows)
        recent = np.nanmedian(windows[:, -int(recent_days) :, :], axis=1)
        previous = np.nanmedian(windows[:, : -int(recent_days), :], axis=1)
    statistics = (median, q75 - q25, mad, q10, q90, slope, recent - previous)
    values = np.stack(statistics, axis=2).reshape(len(sequences), -1)
    output_names = tuple(
        f"{name}__summary__{statistic}"
        for name in names
        for statistic in SUMMARY_STATISTICS
    )
    if values.shape != (len(sequences), len(output_names)):
        raise AssertionError("Wearable summary shape contract failed")
    return values.astype(np.float32), output_names


def _score_item(raw: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(raw, errors="coerce")
    # Source encoding: 2=correct, 1=incorrect.  Other values are not silently
    # interpreted, and are left for fold-local imputation.
    return numeric.map({1.0: 0.0, 2.0: 1.0}).astype(float)


def build_mmse_features(
    mmse: pd.DataFrame,
    subject_ids: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    """Build auditable clinical score features from the source allow-list."""

    assert_subject_alignment(subject_ids, mmse, role="MMSE")
    aligned = mmse.reindex(list(map(str, subject_ids))).copy()
    total = pd.to_numeric(aligned["TOTAL"], errors="coerce").astype(float)
    # The validation export contains one all-zero placeholder examination.
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
    core_names = {"mmse__total", "mmse__total_deficit"}
    domain_scores: dict[str, pd.Series] = {}
    for domain, items in MMSE_DOMAINS.items():
        values = scored.loc[:, list(items)]
        score = values.sum(axis=1, min_count=len(items))
        domain_scores[domain] = score
        score_name = f"mmse__domain__{domain}_score"
        fraction_name = f"mmse__domain__{domain}_fraction"
        columns[score_name] = score
        columns[fraction_name] = score / float(len(items))
        core_names.update({score_name, fraction_name})

    for item in MMSE_ITEMS:
        columns[f"mmse__item__{item.lower()}_correct"] = scored[item]

    reconstructed = scored.sum(axis=1, min_count=len(MMSE_ITEMS))
    columns["mmse__reconstructed_total"] = reconstructed
    columns["mmse__failed_items"] = float(len(MMSE_ITEMS)) - reconstructed
    columns["mmse__recall_deficit"] = 3.0 - domain_scores["recall"]
    columns["mmse__below_24"] = (total < 24.0).where(total.notna()).astype(float)
    columns["mmse__below_27"] = (total < 27.0).where(total.notna()).astype(float)
    orientation = domain_scores["orient_time"] + domain_scores["orient_place"]
    columns["mmse__any_orientation_error"] = (
        orientation < 10.0
    ).where(orientation.notna()).astype(float)
    core_names.update(
        {
            "mmse__reconstructed_total",
            "mmse__failed_items",
            "mmse__recall_deficit",
            "mmse__below_24",
            "mmse__below_27",
            "mmse__any_orientation_error",
        }
    )

    frame = pd.DataFrame(columns, index=aligned.index, dtype=float)
    names = tuple(frame.columns)
    core_indices = tuple(index for index, name in enumerate(names) if name in core_names)
    # Affine re-encoding of the previous 39-column MaxAUC champion:
    # TOTAL + 6 domains + 30 items + failed-items + recall-deficit.  Standard
    # scaling makes raw {1,2} and correctness {0,1} exactly equivalent.
    anchor_names = {
        "mmse__total",
        "mmse__failed_items",
        "mmse__recall_deficit",
        *(
            f"mmse__domain__{domain}_score"
            for domain in MMSE_DOMAINS
        ),
        *(f"mmse__item__{item.lower()}_correct" for item in MMSE_ITEMS),
    }
    anchor_indices = tuple(
        index for index, name in enumerate(names) if name in anchor_names
    )
    if len(anchor_indices) != 39:
        raise AssertionError(
            f"MaxAUC anchor schema must contain 39 columns; got {len(anchor_indices)}"
        )
    return frame.to_numpy(dtype=np.float32), names, core_indices, anchor_indices


def _is_wearable_core(name: str) -> bool:
    lowered = name.lower()
    source_markers = (
        "activity__scalar__low",
        "activity__scalar__inactive",
        "activity__scalar__high",
        "activity__scalar__total",
        "activity__scalar__steps",
        "activity__scalar__met_min_low",
        "activity__scalar__rest",
        "activity__scalar__score_meet_daily_targets",
        "activity__state__transition_rate",
        "sleep__scalar__light",
        "sleep__stage__restless_ratio",
        "sleep__scalar__score",
        "sleep__scalar__score_deep",
        "sleep__scalar__score_rem",
        "sleep__scalar__hr_lowest",
        "sleep__scalar__rmssd",
        "sleep__rmssd__",
        "sleep__stage__transition_rate",
        "sleep__scalar__duration",
        "sleep__scalar__total",
    )
    stable_statistics = (
        "__median",
        "__iqr",
        "__mad",
        "__p10",
        "__p90",
        "__normalized_theil_sen",
        "__recent7_minus_previous21_median",
    )
    return any(marker in lowered for marker in source_markers) and lowered.endswith(
        stable_statistics
    )


def build_champion_dataset(
    wearable: SubjectSequenceDataset,
    *,
    track: str,
    audit: AccessAudit,
    mmse: pd.DataFrame | None = None,
    diagnoses: pd.Series | None = None,
) -> ChampionDataset:
    """Assemble the selected track while preserving an explicit modality wall."""

    resolved_track = normalise_track(track)
    subject_ids = np.asarray(wearable.subject_ids).astype(str)
    wearable_values, wearable_names = summarize_wearable_sequences(
        wearable.sequences, wearable.feature_names
    )
    blocks = [wearable_values]
    names = list(wearable_names)
    wearable_indices = tuple(range(len(names)))
    wearable_core = tuple(
        index for index, name in enumerate(names) if _is_wearable_core(name)
    )
    if not wearable_core:
        raise LeakageContractError("Predeclared wearable-core view is empty")

    views: dict[str, tuple[int, ...]] = {
        "wearable_all": wearable_indices,
        "wearable_core": wearable_core,
    }
    if resolved_track == "mmse":
        if mmse is None:
            raise LeakageContractError("The MMSE track requires the score allow-list")
        mmse_values, mmse_names, mmse_core_local, mmse_anchor_local = build_mmse_features(
            mmse, subject_ids
        )
        offset = len(names)
        blocks.append(mmse_values)
        names.extend(mmse_names)
        mmse_all = tuple(range(offset, offset + len(mmse_names)))
        mmse_core = tuple(offset + index for index in mmse_core_local)
        mmse_anchor = tuple(offset + index for index in mmse_anchor_local)
        views.update(
            {
                "mmse_all": mmse_all,
                "mmse_core": mmse_core,
                "mmse_anchor": mmse_anchor,
                "fusion_all": tuple(range(len(names))),
                "fusion_core": tuple((*wearable_core, *mmse_all)),
            }
        )
    elif mmse is not None:
        raise LeakageContractError("MMSE data was supplied to the wearable track")

    if diagnoses is not None:
        assert_subject_alignment(subject_ids, diagnoses, role="diagnosis")
        aligned_diagnoses = diagnoses.reindex(subject_ids)
        y = binary_target(aligned_diagnoses)
        diagnosis_values: np.ndarray | None = aligned_diagnoses.to_numpy(dtype=str)
        if wearable.y is not None and not np.array_equal(y, wearable.y):
            raise LeakageContractError("Wearable and explicit diagnoses disagree")
    else:
        y = None if wearable.y is None else np.asarray(wearable.y, dtype=np.int64)
        diagnosis_values = None

    matrix = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
    assert_feature_contract(names, track=resolved_track)
    return ChampionDataset(
        track=resolved_track,
        subject_ids=subject_ids,
        X=matrix,
        feature_names=tuple(names),
        views=views,
        sequences=tuple(np.asarray(sequence, dtype=np.float32) for sequence in wearable.sequences),
        sequence_feature_names=tuple(map(str, wearable.feature_names)),
        y=y,
        diagnoses=diagnosis_values,
        audit=audit.to_dict(),
    )


__all__ = [
    "ChampionDataset",
    "FIXED_SUMMARY_DAYS",
    "RECENT_SUMMARY_DAYS",
    "SUMMARY_STATISTICS",
    "assert_feature_contract",
    "assert_wearable_source_contract",
    "build_champion_dataset",
    "build_mmse_features",
    "summarize_wearable_sequences",
]
