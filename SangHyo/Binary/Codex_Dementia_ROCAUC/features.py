"""Label-free, subject-local feature construction.

No function in this module accepts a target vector.  Daily summaries are
computed independently inside each subject, so they may be materialized before
cross-validation.  Any operation that learns across subjects (imputation,
scaling, supervised selection, correlation filtering, winsorization) is left
to the fold-local model pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

import numpy as np
import pandas as pd

from .data import TrackCohort
from .leakage import LeakageError, assert_no_forbidden_features


MMSE_DOMAINS: dict[str, tuple[str, ...]] = {
    "orientation_time": ("Q01", "Q02", "Q03", "Q04", "Q05"),
    "orientation_place": ("Q06", "Q07", "Q08", "Q09", "Q10"),
    "registration": ("Q11_1", "Q11_2", "Q11_3"),
    "attention": ("Q12_1", "Q12_2", "Q12_3", "Q12_4", "Q12_5"),
    "recall": ("Q13_1", "Q13_2", "Q13_3"),
    "language": (
        "Q14_1",
        "Q14_2",
        "Q15",
        "Q16_1",
        "Q16_2",
        "Q16_3",
        "Q17",
        "Q18",
        "Q19",
    ),
}
MMSE_ITEMS = tuple(item for items in MMSE_DOMAINS.values() for item in items)
MMSE_ALLOWED = ("TOTAL", *MMSE_ITEMS)


@dataclass(frozen=True)
class FeatureBundle:
    """Model-ready subject-level table plus an untouched raw sequence branch."""

    subject_ids: np.ndarray
    table: pd.DataFrame
    sequences: tuple[np.ndarray, ...]
    sequence_feature_names: tuple[str, ...]
    feature_families: dict[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        if not self.table.index.equals(pd.Index(self.subject_ids, name="subject_id")):
            raise LeakageError("Feature table index is not aligned with subject IDs")
        if len(self.sequences) != len(self.subject_ids):
            raise LeakageError("Sequence branch is not aligned with subject IDs")
        assert_no_forbidden_features(tuple(map(str, self.table.columns)))
        values = self.table.to_numpy(dtype=np.float64)
        if np.isinf(values).any():
            raise LeakageError("Subject feature table contains infinity")

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(map(str, self.table.columns))


def _safe_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    return normalized or "unnamed"


def _longest_true_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _rank_slope(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if int(finite.sum()) < 3:
        return np.nan
    x = np.linspace(0.0, 1.0, len(values))[finite]
    y = values[finite]
    x_centered = x - x.mean()
    denominator = float(np.square(x_centered).sum())
    if denominator <= 1e-12:
        return np.nan
    return float(np.dot(x_centered, y - y.mean()) / denominator)


def _autocorrelation(values: np.ndarray, lag: int = 1) -> float:
    if len(values) <= lag:
        return np.nan
    left, right = values[:-lag], values[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 4:
        return np.nan
    left, right = left[valid], right[valid]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _quantile_entropy(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 5 or np.ptp(finite) <= 1e-12:
        return np.nan
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 9)))
    if len(edges) < 3:
        return np.nan
    counts, _ = np.histogram(finite, bins=edges)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    return float(entropy / np.log(len(probabilities))) if len(probabilities) > 1 else 0.0


def _spectral_peak_ratio(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if int(finite.sum()) < 8:
        return np.nan
    positions = np.flatnonzero(finite)
    filled = np.interp(np.arange(len(values)), positions, values[finite])
    centered = filled - filled.mean()
    if np.std(centered) <= 1e-12:
        return np.nan
    power = np.square(np.abs(np.fft.rfft(centered)))[1:]
    total = float(power.sum())
    return float(power.max() / total) if total > 1e-12 else np.nan


def _rolling_variability(values: np.ndarray, window: int = 7) -> float:
    if len(values) < 3:
        return np.nan
    minimum = min(3, window)
    result = (
        pd.Series(values, dtype=float)
        .rolling(window=min(window, len(values)), min_periods=minimum)
        .std(ddof=0)
        .mean()
    )
    return float(result) if np.isfinite(result) else np.nan


def _biological_statistics(values: np.ndarray) -> dict[str, float]:
    """Rich statistics for one biological channel within one subject."""

    vector = np.asarray(values, dtype=np.float64)
    vector[~np.isfinite(vector)] = np.nan
    finite = vector[np.isfinite(vector)]
    names = (
        "mean",
        "median",
        "std",
        "min",
        "max",
        "range",
        "p10",
        "p25",
        "p75",
        "p90",
        "iqr",
        "cv",
        "skew",
        "kurtosis",
        "first_last_delta",
        "rank_slope",
        "recent7_minus_all",
        "recent7_to_all_ratio",
        "autocorr1",
        "entropy",
        "spectral_peak_ratio",
        "rolling7_variability",
    )
    if len(finite) == 0:
        return {name: np.nan for name in names}
    q10, q25, median, q75, q90 = np.quantile(
        finite, [0.10, 0.25, 0.50, 0.75, 0.90]
    )
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=0))
    centered = finite - mean
    standardized = centered / std if std > 1e-12 else np.zeros_like(centered)
    recent = vector[-min(7, len(vector)) :]
    recent_mean = float(np.nanmean(recent)) if np.isfinite(recent).any() else np.nan
    first_position, last_position = np.flatnonzero(np.isfinite(vector))[[0, -1]]
    denominator = abs(mean)
    return {
        "mean": mean,
        "median": float(median),
        "std": std,
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "range": float(np.ptp(finite)),
        "p10": float(q10),
        "p25": float(q25),
        "p75": float(q75),
        "p90": float(q90),
        "iqr": float(q75 - q25),
        "cv": float(std / denominator) if denominator > 1e-12 else np.nan,
        "skew": float(np.mean(np.power(standardized, 3))) if len(finite) >= 3 else np.nan,
        "kurtosis": (
            float(np.mean(np.power(standardized, 4)) - 3.0)
            if len(finite) >= 4
            else np.nan
        ),
        "first_last_delta": float(vector[last_position] - vector[first_position]),
        "rank_slope": _rank_slope(vector),
        "recent7_minus_all": recent_mean - mean if np.isfinite(recent_mean) else np.nan,
        "recent7_to_all_ratio": (
            recent_mean / mean
            if np.isfinite(recent_mean) and abs(mean) > 1e-12
            else np.nan
        ),
        "autocorr1": _autocorrelation(vector),
        "entropy": _quantile_entropy(vector),
        "spectral_peak_ratio": _spectral_peak_ratio(vector),
        "rolling7_variability": _rolling_variability(vector),
    }


def _protocol_statistics(values: np.ndarray) -> dict[str, float]:
    vector = np.asarray(values, dtype=np.float64)
    missing = ~np.isfinite(vector)
    return {
        "missing_fraction": float(missing.mean()),
        "valid_days": float((~missing).sum()),
        "longest_missing_run": float(_longest_true_run(missing)),
    }


def aggregate_wearable_sequences(
    subject_ids: Sequence[str],
    sequences: Sequence[np.ndarray],
    daily_feature_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create biological and protocol-sensitivity tables.

    The function intentionally has no ``y`` parameter.  Absolute dates and
    enrollment order are not available from the audited daily representation,
    preventing calendar-wave artifacts.  Chronological within-subject order is
    retained for trend, recent-window, autocorrelation, and spectral features.
    """

    names = tuple(_safe_name(name) for name in daily_feature_names)
    if len(set(names)) != len(names):
        raise LeakageError("Daily feature names collide after normalization")
    biological_rows: list[dict[str, float]] = []
    protocol_rows: list[dict[str, float]] = []
    for sequence in sequences:
        values = np.asarray(sequence, dtype=np.float64)
        biological: dict[str, float] = {}
        protocol: dict[str, float] = {
            "protocol__sequence_length": float(values.shape[0])
        }
        for column, source_name in enumerate(names):
            for statistic, value in _biological_statistics(values[:, column]).items():
                biological[f"wearable__{source_name}__{statistic}"] = value
            for statistic, value in _protocol_statistics(values[:, column]).items():
                protocol[f"protocol__{source_name}__{statistic}"] = value
        biological_rows.append(biological)
        protocol_rows.append(protocol)
    index = pd.Index(np.asarray(subject_ids, dtype=str), name="subject_id")
    biology_frame = pd.DataFrame(biological_rows, index=index, dtype=float)
    protocol_frame = pd.DataFrame(protocol_rows, index=index, dtype=float)
    return biology_frame, protocol_frame


def _mmse_features(mmse: pd.DataFrame) -> pd.DataFrame:
    """Build prespecified MMSE features without reading diagnosis metadata."""

    missing = [column for column in MMSE_ALLOWED if column not in mmse.columns]
    if missing:
        raise LeakageError(f"MMSE allow-list is incomplete: {missing}")
    numeric = mmse.loc[:, list(MMSE_ALLOWED)].apply(pd.to_numeric, errors="coerce")
    output = pd.DataFrame(index=mmse.index)
    total = numeric["TOTAL"].where(numeric["TOTAL"].between(1.0, 30.0))
    invalid_exam = total.isna()
    # Source encoding is 2=correct and 1=incorrect. Unexpected values are kept
    # missing for fold-local imputation rather than silently reinterpreted.
    scored = numeric.loc[:, list(MMSE_ITEMS)].apply(
        lambda column: column.map({1.0: 0.0, 2.0: 1.0})
    )
    scored.loc[invalid_exam, :] = np.nan
    output["mmse__total"] = total
    # Standard MMSE maximum is fixed clinically; it is not learned from subjects.
    output["mmse__total_deficit"] = 30.0 - total
    for item in MMSE_ITEMS:
        output[f"mmse__item__{item.lower()}_correct"] = scored[item]
    domain_scores: dict[str, pd.Series] = {}
    for domain, items in MMSE_DOMAINS.items():
        domain_score = scored.loc[:, list(items)].sum(
            axis=1, min_count=len(items)
        )
        domain_scores[domain] = domain_score
        output[f"mmse__domain__{domain}"] = domain_score
        output[f"mmse__domain_fraction__{domain}"] = domain_score / float(
            len(items)
        )
        output[f"mmse__domain_missing__{domain}"] = scored.loc[
            :, list(items)
        ].isna().mean(axis=1)
    reconstructed = scored.sum(axis=1, min_count=len(MMSE_ITEMS))
    output["mmse__reconstructed_total"] = reconstructed
    output["mmse__failed_items"] = float(len(MMSE_ITEMS)) - reconstructed
    output["mmse__recall_deficit"] = 3.0 - domain_scores["recall"]
    output["mmse__below_24"] = (total < 24.0).where(total.notna()).astype(float)
    output["mmse__below_27"] = (total < 27.0).where(total.notna()).astype(float)
    orientation = (
        domain_scores["orientation_time"] + domain_scores["orientation_place"]
    )
    output["mmse__any_orientation_error"] = (
        orientation < 10.0
    ).where(orientation.notna()).astype(float)
    return output.astype(float)


def _predeclared_interactions(table: pd.DataFrame) -> pd.DataFrame:
    """Small clinical interaction bank, based only on semantic column names."""

    output = pd.DataFrame(index=table.index)

    def find(suffix: str) -> str | None:
        matches = [name for name in table.columns if str(name).endswith(suffix)]
        return str(matches[0]) if len(matches) == 1 else None

    def ratio(name: str, numerator_suffix: str, denominator_suffix: str) -> None:
        numerator = find(numerator_suffix)
        denominator = find(denominator_suffix)
        if numerator is None or denominator is None:
            return
        den = table[denominator].abs().replace(0.0, np.nan)
        output[name] = table[numerator] / den

    ratio(
        "interaction__sleep_deep_to_total_mean",
        "sleep__duration__deep_ratio__mean",
        "sleep__duration__total_ratio__mean",
    )
    ratio(
        "interaction__activity_high_to_inactive_mean",
        "activity__scalar__high__mean",
        "activity__scalar__inactive__mean",
    )
    ratio(
        "interaction__activity_low_variability_to_mean",
        "activity__scalar__low__std",
        "activity__scalar__low__mean",
    )
    if {
        "mmse__domain__recall",
        "mmse__domain__orientation_time",
    } <= set(table.columns):
        output["interaction__mmse_recall_x_orientation"] = (
            table["mmse__domain__recall"]
            * table["mmse__domain__orientation_time"]
        )
    return output.astype(float)


def build_feature_bundle(cohort: TrackCohort) -> FeatureBundle:
    """Build the fixed table view for one track."""

    biology, protocol = aggregate_wearable_sequences(
        cohort.subject_ids,
        cohort.sequences,
        cohort.daily_feature_names,
    )
    parts = [biology]
    families: dict[str, tuple[str, ...]] = {
        "wearable_biology": tuple(biology.columns),
    }
    if cohort.track == "wearable_protocol":
        parts.append(protocol)
        families["protocol_sensitivity"] = tuple(protocol.columns)
    if cohort.track == "wearable_mmse":
        if cohort.mmse is None:
            raise LeakageError("wearable_mmse track has no MMSE table")
        mmse = _mmse_features(cohort.mmse)
        mmse.index = biology.index
        parts.append(mmse)
        families["mmse"] = tuple(mmse.columns)
    table = pd.concat(parts, axis=1)
    interactions = _predeclared_interactions(table)
    if not interactions.empty:
        table = pd.concat([table, interactions], axis=1)
        families["interactions"] = tuple(interactions.columns)
    if table.columns.duplicated().any():
        duplicates = table.columns[table.columns.duplicated()].tolist()
        raise LeakageError(f"Duplicate engineered feature names: {duplicates[:10]}")
    table = table.replace([np.inf, -np.inf], np.nan).astype(np.float64)
    assert_no_forbidden_features(tuple(map(str, table.columns)))
    return FeatureBundle(
        subject_ids=cohort.subject_ids.copy(),
        table=table,
        sequences=cohort.sequences,
        sequence_feature_names=cohort.daily_feature_names,
        feature_families=families,
    )


__all__ = [
    "FeatureBundle",
    "MMSE_ALLOWED",
    "aggregate_wearable_sequences",
    "build_feature_bundle",
]
