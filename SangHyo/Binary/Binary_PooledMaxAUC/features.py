"""Subject-level feature bank and the "views" the candidate search chooses from.

All features are computed from each subject's own data only, so building the
bank is label-free and can safely happen once before cross-validation.  What
must stay fold-local is *screening and preprocessing* (see ``engine.py``), not
this construction.

View design follows the evidence in ``BINARY_EXPERIMENTS_ANALYSIS_KO.md``:
MMSE dominates, and large wearable banks consistently hurt (1,077-feature
wearable search reached 0.5370; MMSE 39 alone reached 0.7658).  So instead of
one giant matrix, the search picks among nested views and lets the OOF decide.

MMSE coding note (verified on all 141 training subjects): items are
``1 = incorrect`` / ``2 = correct`` and ``TOTAL == count(item == 2)``.  Hence
``num_failed = 30 - TOTAL`` and ``recall_deficit = 6 - recall`` are exact affine
transforms of existing columns.  They are still included in ``mmse_core`` because
the 39-feature block is the published anchor of ``Binary_MMSE_MaxAUC`` and trees
are indifferent to the collinearity, but the redundancy is documented rather
than presented as new information.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data import CLOCK_CHANNELS, MMSE_DOMAINS, MMSE_ITEMS, PooledCohort
from .leakage import LeakageError, assert_no_forbidden_features

__all__ = ["FeatureBank", "build_feature_bank"]

MMSE_PREFIX = "mmse__"
WEAR_PREFIX = "wear__"


@dataclass(frozen=True)
class FeatureBank:
    """All engineered features plus the named column subsets ("views")."""

    subject_ids: np.ndarray
    frame: pd.DataFrame
    views: Mapping[str, tuple[str, ...]]
    audit: Mapping[str, Any]

    def matrix(self, view: str) -> tuple[np.ndarray, tuple[str, ...]]:
        if view not in self.views:
            raise KeyError(f"Unknown view {view!r}; have {sorted(self.views)}")
        columns = list(self.views[view])
        return self.frame.loc[:, columns].to_numpy(dtype=np.float64), tuple(columns)


# --------------------------------------------------------------------------- #
# per-channel aggregation
# --------------------------------------------------------------------------- #
def _linear_stats(
    values: np.ndarray, day_index: np.ndarray, is_weekend: np.ndarray, stats: Sequence[str]
) -> dict[str, float]:
    finite = np.isfinite(values)
    observed = values[finite]
    out: dict[str, float] = {}
    if observed.size == 0:
        return {stat: float("nan") for stat in stats}

    mean = float(np.mean(observed))
    sd = float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0
    for stat in stats:
        if stat == "mean":
            out[stat] = mean
        elif stat == "sd":
            out[stat] = sd
        elif stat == "cv":
            out[stat] = sd / abs(mean) if abs(mean) > 1e-9 else float("nan")
        elif stat == "median":
            out[stat] = float(np.median(observed))
        elif stat == "iqr":
            q25, q75 = np.quantile(observed, [0.25, 0.75])
            out[stat] = float(q75 - q25)
        elif stat == "p10":
            out[stat] = float(np.quantile(observed, 0.10))
        elif stat == "p90":
            out[stat] = float(np.quantile(observed, 0.90))
        elif stat == "min":
            out[stat] = float(np.min(observed))
        elif stat == "max":
            out[stat] = float(np.max(observed))
        elif stat == "trend_per_week":
            days = day_index[finite].astype(float)
            centred = days - days.mean()
            denominator = float(np.sum(centred**2))
            out[stat] = (
                float(np.sum(centred * (observed - mean)) / denominator) * 7.0
                if observed.size >= 3 and denominator > 0
                else float("nan")
            )
        elif stat == "late_minus_early":
            if observed.size >= 6:
                order = np.argsort(day_index[finite], kind="mergesort")
                ordered = observed[order]
                third = max(1, observed.size // 3)
                out[stat] = float(np.mean(ordered[-third:]) - np.mean(ordered[:third]))
            else:
                out[stat] = float("nan")
        elif stat == "weekend_minus_weekday":
            mask = is_weekend[finite].astype(bool)
            out[stat] = (
                float(np.mean(observed[mask]) - np.mean(observed[~mask]))
                if mask.sum() >= 2 and (~mask).sum() >= 2
                else float("nan")
            )
        else:
            raise LeakageError(f"Unknown wearable statistic: {stat}")
    return out


def _circular_sd_hours(values: np.ndarray) -> float:
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    if observed.size < 2:
        return float("nan")
    angles = 2.0 * np.pi * observed / 24.0
    magnitude = abs(complex(float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles)))))
    magnitude = min(max(magnitude, 1e-12), 1.0)
    return 24.0 / (2.0 * math.pi) * math.sqrt(-2.0 * math.log(magnitude))


def _build_wearable(cohort: PooledCohort, config: FeatureConfig) -> pd.DataFrame:
    """Per-subject aggregates of every daily channel."""

    linear_channels = [c for c in cohort.channels if c not in CLOCK_CHANNELS]
    rows: dict[str, dict[str, float]] = {}
    for subject_id, frame in cohort.daily.groupby("subject_id", sort=True):
        day_index = frame["day_index"].to_numpy(dtype=int)
        is_weekend = frame["is_weekend"].to_numpy(dtype=int)
        values: dict[str, float] = {}
        for channel in linear_channels:
            stats = _linear_stats(
                frame[channel].to_numpy(dtype=float),
                day_index,
                is_weekend,
                config.wearable_stats,
            )
            for stat, value in stats.items():
                values[f"{WEAR_PREFIX}{channel}__{stat}"] = value
        # Clock channels: circular spread only (a linear sd across midnight is
        # meaningless, e.g. 23:50 and 00:10 are 20 minutes apart, not 23.7h).
        for channel in CLOCK_CHANNELS:
            if channel in cohort.daily.columns:
                values[f"{WEAR_PREFIX}{channel}__circular_sd"] = _circular_sd_hours(
                    frame[channel].to_numpy(dtype=float)
                )
        rows[str(subject_id)] = values
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame.index.name = "subject_id"
    return frame


# --------------------------------------------------------------------------- #
# MMSE blocks
# --------------------------------------------------------------------------- #
def _build_mmse_core(mmse: pd.DataFrame, item_max: float) -> pd.DataFrame:
    """The 39-feature anchor used by Binary_MMSE_MaxAUC."""

    missing = [c for c in ("TOTAL", *MMSE_ITEMS) if c not in mmse.columns]
    if missing:
        raise LeakageError(f"MMSE table missing allow-listed columns: {missing}")

    frame = pd.DataFrame(index=mmse.index.astype(str))
    frame[f"{MMSE_PREFIX}total"] = mmse["TOTAL"].astype(float)
    for domain, items in MMSE_DOMAINS.items():
        frame[f"{MMSE_PREFIX}domain_{domain}"] = mmse[list(items)].astype(float).sum(axis=1)
    for item in MMSE_ITEMS:
        frame[f"{MMSE_PREFIX}item_{item.lower()}"] = mmse[item].astype(float)

    observed_max = float(mmse[list(MMSE_ITEMS)].to_numpy(dtype=float).max())
    if observed_max > float(item_max):
        raise LeakageError(
            f"MMSE item scores exceed declared item_max={item_max} (observed {observed_max})"
        )
    below = pd.DataFrame(
        {c: (mmse[c].astype(float) < float(item_max)).astype(int) for c in MMSE_ITEMS}
    )
    # Exact affine transforms of TOTAL / recall in this dataset; kept only for
    # parity with the published 39-feature anchor.
    frame[f"{MMSE_PREFIX}num_failed"] = below.sum(axis=1).astype(float)
    frame[f"{MMSE_PREFIX}recall_deficit"] = (
        float(len(MMSE_DOMAINS["recall"]) * item_max) - frame[f"{MMSE_PREFIX}domain_recall"]
    )
    frame.index.name = "subject_id"
    return frame.sort_index()


def _build_mmse_engineered(core: pd.DataFrame, mmse: pd.DataFrame) -> pd.DataFrame:
    """Interactions motivated by the EDA in EXPERIMENT_SUMMARY_KO.md section 2.

    The strongest single reported signal there is not TOTAL but the
    recall + attention combination (direction-free AUC 0.755 vs TOTAL 0.695),
    and the CN/MCI boundary is driven by delayed recall (Q13).
    """

    frame = pd.DataFrame(index=core.index)
    recall = core[f"{MMSE_PREFIX}domain_recall"]
    attention = core[f"{MMSE_PREFIX}domain_attention"]
    orient_time = core[f"{MMSE_PREFIX}domain_orient_time"]
    orient_place = core[f"{MMSE_PREFIX}domain_orient_place"]
    language = core[f"{MMSE_PREFIX}domain_language"]
    total = core[f"{MMSE_PREFIX}total"]

    frame[f"{MMSE_PREFIX}recall_plus_q12_5"] = recall + mmse["Q12_5"].astype(float)
    frame[f"{MMSE_PREFIX}recall_plus_attention"] = recall + attention
    frame[f"{MMSE_PREFIX}orient_minus_recall"] = (orient_time + orient_place) - 2.0 * recall
    frame[f"{MMSE_PREFIX}language_minus_recall"] = language / 3.0 - recall
    frame[f"{MMSE_PREFIX}recall_share_of_total"] = recall / total.replace(0, np.nan)
    frame[f"{MMSE_PREFIX}attention_share_of_total"] = attention / total.replace(0, np.nan)

    domains = [f"{MMSE_PREFIX}domain_{name}" for name in MMSE_DOMAINS]
    # Normalize each domain to its own maximum so the spread is comparable.
    domain_max = {name: float(len(items) * 2) for name, items in MMSE_DOMAINS.items()}
    normalized = pd.DataFrame(
        {
            name: core[f"{MMSE_PREFIX}domain_{name}"] / domain_max[name]
            for name in MMSE_DOMAINS
        }
    )
    frame[f"{MMSE_PREFIX}domain_min_norm"] = normalized.min(axis=1)
    frame[f"{MMSE_PREFIX}domain_range_norm"] = normalized.max(axis=1) - normalized.min(axis=1)
    frame[f"{MMSE_PREFIX}domain_sd_norm"] = normalized.std(axis=1)
    frame[f"{MMSE_PREFIX}n_imperfect_domains"] = (normalized < 1.0).sum(axis=1).astype(float)
    frame[f"{MMSE_PREFIX}ceiling_flag"] = (total >= 29).astype(float)
    # Focal failure: near-ceiling TOTAL but a specific weak domain. This is the
    # exact pattern the analysis flagged as the hard CN-vs-MCI boundary.
    frame[f"{MMSE_PREFIX}focal_failure"] = (total >= 27).astype(float) * (
        1.0 - normalized.min(axis=1)
    )
    _ = domains  # kept for readability of intent
    return frame


# --------------------------------------------------------------------------- #
# cross-block interactions
# --------------------------------------------------------------------------- #
def _build_cross_interactions(
    mmse_all: pd.DataFrame, wearable: pd.DataFrame, config: FeatureConfig
) -> pd.DataFrame:
    """A deliberately small MMSE x wearable-variability block.

    Rationale: every previous attempt to widen the wearable side failed, so this
    stays limited to the few wearable channels the repository has repeatedly
    identified as Dem markers, crossed only with the recall signal.
    """

    frame = pd.DataFrame(index=mmse_all.index)
    recall_norm = mmse_all[f"{MMSE_PREFIX}domain_recall"] / 6.0
    for channel in config.small_wearable_channels[:4]:
        column = f"{WEAR_PREFIX}{channel}__cv"
        if column in wearable.columns:
            variability = wearable[column].reindex(frame.index)
            frame[f"x__recall_x_{channel}_cv"] = (1.0 - recall_norm) * variability
    return frame


# --------------------------------------------------------------------------- #
# public builder
# --------------------------------------------------------------------------- #
def build_feature_bank(cohort: PooledCohort, config: FeatureConfig) -> FeatureBank:
    mmse_core = _build_mmse_core(cohort.mmse, config.mmse_item_max)
    mmse_engineered = _build_mmse_engineered(mmse_core, cohort.mmse)
    wearable = _build_wearable(cohort, config)

    subject_ids = np.asarray([str(s) for s in cohort.subject_ids], dtype=str)
    index = pd.Index(subject_ids, name="subject_id")
    mmse_core = mmse_core.reindex(index)
    mmse_engineered = mmse_engineered.reindex(index)
    wearable = wearable.reindex(index)
    cross = _build_cross_interactions(
        pd.concat([mmse_core, mmse_engineered], axis=1), wearable, config
    )

    frame = pd.concat([mmse_core, mmse_engineered, wearable, cross], axis=1)
    if frame.columns.has_duplicates:
        duplicated = sorted(frame.columns[frame.columns.duplicated()].tolist())
        raise LeakageError(f"Duplicate feature names across blocks: {duplicated}")
    if len(frame) != len(subject_ids):
        raise LeakageError("Feature merge changed the subject count")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    assert_no_forbidden_features(frame.columns, context="feature bank")

    core_cols = tuple(mmse_core.columns)
    plus_cols = core_cols + tuple(mmse_engineered.columns)
    small_wear_cols = tuple(
        f"{WEAR_PREFIX}{channel}__{stat}"
        for channel in config.small_wearable_channels
        for stat in config.small_wearable_stats
        if f"{WEAR_PREFIX}{channel}__{stat}" in frame.columns
    )
    wear_all_cols = tuple(wearable.columns)
    cross_cols = tuple(cross.columns)

    views: dict[str, tuple[str, ...]] = {
        "mmse_core": core_cols,
        "mmse_plus": plus_cols,
        "mmse_wear_small": plus_cols + small_wear_cols + cross_cols,
        "all": tuple(frame.columns),
        "wearable_only": wear_all_cols,
    }
    views = {name: cols for name, cols in views.items() if name in set(config.views)}
    if not views:
        raise LeakageError("No feature views enabled")
    for name, columns in views.items():
        if not columns:
            raise LeakageError(f"View {name!r} is empty")
        assert_no_forbidden_features(columns, context=f"view[{name}]")

    audit = {
        "n_subjects": int(len(subject_ids)),
        "n_features_total": int(frame.shape[1]),
        "block_sizes": {
            "mmse_core": len(core_cols),
            "mmse_engineered": int(mmse_engineered.shape[1]),
            "wearable": int(wearable.shape[1]),
            "cross_interactions": int(cross.shape[1]),
        },
        "view_sizes": {name: len(cols) for name, cols in views.items()},
        "overall_missing_rate": float(np.isnan(frame.to_numpy(dtype=float)).mean()),
        "notes": [
            "mmse__num_failed = 30 - mmse__total and mmse__recall_deficit = 6 - "
            "domain_recall are exact affine transforms in this dataset; kept only "
            "for parity with the published 39-feature anchor.",
            "Clock channels use circular SD; linear SD across midnight is invalid.",
            "Acquisition proxies (observation counts, coverage, non-wear) are "
            "excluded by the leakage guard, not merely unused.",
        ],
    }
    return FeatureBank(subject_ids=subject_ids, frame=frame, views=views, audit=audit)
