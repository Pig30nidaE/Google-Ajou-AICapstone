"""Subject-level design matrices: BASE, Gemini and the optional MMSE block.

Three independent blocks are built and only combined at the very end:

``base``    per-subject mean/sd of the curated wearable channels plus circular
            clock regularity.  Deliberately small (about 35 columns for 141
            subjects) because every large wearable feature bank in this
            repository collapsed - see ``SangHyo/AGENTS.md`` sections 3-1/4-3.
``gemini``  the 12 validated numbers returned by the feature-extraction stage.
``mmse``    the cognitive-test allow-list, added **only** in ``mmse_mode=with``
            and **only** to the downstream classifier, never to a Gemini payload.

Verified data facts used here (checked against ``Data/`` on 2026-07-29):
MMSE items are coded 1 = incorrect and 2 = correct, and ``TOTAL`` equals the
number of items scored 2 for all 141/141 training subjects.  Therefore
``item_max`` is a constant 2.0 rather than a value learned from the data (no
fold-dependent statistic), and the ``num_failed`` / ``recall_deficit``
composites used by ``Binary_MMSE_MaxAUC`` are omitted here because in this
dataset they are exact affine transforms of ``TOTAL`` and of the recall domain
sum, so they add columns without adding information.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import FeatureConfig
from .data import DAILY_CHANNELS, MMSE_DOMAINS, MMSE_ITEMS, DailyDataset
from .guards import (
    LeakageError,
    assert_names_are_label_free,
    assert_names_are_mmse_free,
)
from .payload import CLOCK_CHANNELS
from .schema import DESIGN_MATRIX_PREFIX, FEATURE_NAMES

__all__ = [
    "BASE_PREFIX",
    "MMSE_PREFIX",
    "DesignMatrix",
    "build_base_features",
    "build_gemini_features",
    "build_mmse_features",
    "assemble_design_matrix",
]

BASE_PREFIX = "base__"
MMSE_PREFIX = "mmse__"


@dataclass(frozen=True)
class DesignMatrix:
    subject_ids: np.ndarray
    X: np.ndarray
    feature_names: tuple[str, ...]
    feature_set: str
    mmse_mode: str
    blocks: Mapping[str, tuple[str, ...]]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _circular_sd_hours(values: np.ndarray) -> float:
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    if observed.size < 2:
        return float("nan")
    angles = 2.0 * np.pi * observed / 24.0
    magnitude = abs(complex(float(np.mean(np.cos(angles))), float(np.mean(np.sin(angles)))))
    magnitude = min(max(magnitude, 1e-12), 1.0)
    return 24.0 / (2.0 * math.pi) * math.sqrt(-2.0 * math.log(magnitude))


def build_base_features(
    dataset: DailyDataset, feature_config: FeatureConfig
) -> pd.DataFrame:
    """Per-subject wearable summary features (no label, no MMSE, no identifier)."""

    unknown = sorted(set(feature_config.base_channels) - set(DAILY_CHANNELS))
    if unknown:
        raise LeakageError(f"features.base_channels contains unknown channels: {unknown}")
    unknown_stats = sorted(set(feature_config.base_stats) - {"mean", "sd", "median", "iqr", "cv"})
    if unknown_stats:
        raise LeakageError(f"features.base_stats contains unknown statistics: {unknown_stats}")

    rows: dict[str, dict[str, float]] = {}
    for subject_id, frame in dataset.frame.groupby("subject_id", sort=True):
        values: dict[str, float] = {}
        for channel in feature_config.base_channels:
            observed = frame[channel].to_numpy(dtype=float)
            observed = observed[np.isfinite(observed)]
            for statistic in feature_config.base_stats:
                key = f"{BASE_PREFIX}{channel}__{statistic}"
                if observed.size == 0:
                    values[key] = float("nan")
                    continue
                if statistic == "mean":
                    values[key] = float(np.mean(observed))
                elif statistic == "sd":
                    values[key] = float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0
                elif statistic == "median":
                    values[key] = float(np.median(observed))
                elif statistic == "iqr":
                    q25, q75 = np.quantile(observed, [0.25, 0.75])
                    values[key] = float(q75 - q25)
                elif statistic == "cv":
                    mean = float(np.mean(observed))
                    sd = float(np.std(observed, ddof=1)) if observed.size > 1 else 0.0
                    values[key] = sd / abs(mean) if abs(mean) > 1e-9 else float("nan")
        if feature_config.include_clock_regularity:
            for channel in CLOCK_CHANNELS:
                values[f"{BASE_PREFIX}{channel}__circular_sd"] = _circular_sd_hours(
                    frame[channel].to_numpy(dtype=float)
                )
        rows[str(subject_id)] = values

    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame.index.name = "subject_id"
    assert_names_are_label_free(frame.columns, context="base features")
    assert_names_are_mmse_free(frame.columns, context="base features")
    return frame


def build_gemini_features(
    features_by_subject: Mapping[str, Mapping[str, float]]
) -> pd.DataFrame:
    """Wide table of the validated Gemini fields, one row per subject."""

    rows: dict[str, dict[str, float]] = {}
    for subject_id, values in features_by_subject.items():
        missing = [name for name in FEATURE_NAMES if name not in values]
        if missing:
            raise LeakageError(f"Gemini features for {subject_id} are incomplete: {missing}")
        rows[str(subject_id)] = {
            f"{DESIGN_MATRIX_PREFIX}{name}": float(values[name]) for name in FEATURE_NAMES
        }
    frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    frame.index.name = "subject_id"
    assert_names_are_label_free(frame.columns, context="gemini features")
    assert_names_are_mmse_free(frame.columns, context="gemini features")
    return frame


def build_mmse_features(mmse_table: pd.DataFrame, *, item_max: float = 2.0) -> pd.DataFrame:
    """Cognitive-test block for the ``with`` pipeline only."""

    missing = [item for item in ("TOTAL", *MMSE_ITEMS) if item not in mmse_table.columns]
    if missing:
        raise LeakageError(f"MMSE table is missing allow-listed columns: {missing}")
    frame = pd.DataFrame(index=mmse_table.index.astype(str))
    frame[f"{MMSE_PREFIX}total"] = mmse_table["TOTAL"].astype(float)
    for domain, items in MMSE_DOMAINS.items():
        frame[f"{MMSE_PREFIX}domain_{domain}"] = mmse_table[list(items)].astype(float).sum(axis=1)
    for item in MMSE_ITEMS:
        frame[f"{MMSE_PREFIX}item_{item.lower()}"] = mmse_table[item].astype(float)
    observed_max = float(mmse_table[list(MMSE_ITEMS)].to_numpy(dtype=float).max())
    if observed_max > float(item_max):
        raise LeakageError(
            f"MMSE item scores exceed the declared item_max={item_max} (observed {observed_max}); "
            "the fixed scoring assumption in config.yaml no longer holds"
        )
    frame.index.name = "subject_id"
    assert_names_are_label_free(frame.columns, context="mmse features")
    return frame.sort_index()


def _align(frame: pd.DataFrame, subjects: Sequence[str], *, block: str) -> pd.DataFrame:
    aligned = frame.reindex([str(subject) for subject in subjects])
    if aligned.isna().all(axis=1).any():
        missing = [
            subject
            for subject, empty in zip(subjects, aligned.isna().all(axis=1).to_numpy())
            if empty
        ]
        raise LeakageError(
            f"{block} block has no row for {len(missing)} subject(s); "
            "feature merging must not change the subject set"
        )
    return aligned


def assemble_design_matrix(
    *,
    subjects: Sequence[str],
    base: pd.DataFrame,
    gemini: pd.DataFrame | None,
    mmse: pd.DataFrame | None,
    feature_set: str,
    mmse_mode: str,
) -> DesignMatrix:
    """Combine the requested blocks and enforce the mode contract."""

    if feature_set not in {"base", "base_gemini", "gemini_only"}:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    if mmse_mode not in {"without", "with"}:
        raise ValueError(f"Unknown mmse_mode: {mmse_mode}")

    subjects = [str(subject) for subject in subjects]
    blocks: dict[str, tuple[str, ...]] = {}
    frames: list[pd.DataFrame] = []

    if feature_set in {"base", "base_gemini"}:
        aligned = _align(base, subjects, block="base")
        blocks["base"] = tuple(aligned.columns)
        frames.append(aligned)
    if feature_set in {"base_gemini", "gemini_only"}:
        if gemini is None:
            raise LeakageError(f"feature_set={feature_set} requires Gemini features")
        aligned = _align(gemini, subjects, block="gemini")
        blocks["gemini"] = tuple(aligned.columns)
        frames.append(aligned)
    if mmse_mode == "with":
        if mmse is None:
            raise LeakageError("mmse_mode=with requires the MMSE block")
        aligned = _align(mmse, subjects, block="mmse")
        blocks["mmse"] = tuple(aligned.columns)
        frames.append(aligned)

    matrix = pd.concat(frames, axis=1)
    if matrix.columns.has_duplicates:
        duplicated = sorted(matrix.columns[matrix.columns.duplicated()].tolist())
        raise LeakageError(f"Duplicate feature names across blocks: {duplicated}")
    if len(matrix) != len(subjects):
        raise LeakageError("Subject count changed while merging feature blocks")

    names = tuple(str(name) for name in matrix.columns)
    assert_names_are_label_free(names, context=f"design matrix[{feature_set}/{mmse_mode}]")
    if mmse_mode == "without":
        assert_names_are_mmse_free(
            names, context="design matrix[mmse_mode=without]"
        )
    return DesignMatrix(
        subject_ids=np.asarray(subjects, dtype=str),
        X=matrix.to_numpy(dtype=np.float64),
        feature_names=names,
        feature_set=feature_set,
        mmse_mode=mmse_mode,
        blocks=blocks,
    )


def describe_blocks(matrix: DesignMatrix) -> dict[str, Any]:
    return {
        "feature_set": matrix.feature_set,
        "mmse_mode": matrix.mmse_mode,
        "n_subjects": int(len(matrix.subject_ids)),
        "n_features": matrix.n_features,
        "block_sizes": {name: len(columns) for name, columns in matrix.blocks.items()},
        "feature_names": list(matrix.feature_names),
    }


def missing_value_report(matrix: DesignMatrix) -> dict[str, Any]:
    missing = np.isnan(matrix.X)
    per_feature = missing.mean(axis=0)
    worst: Iterable[tuple[str, float]] = sorted(
        zip(matrix.feature_names, per_feature.tolist()), key=lambda item: -item[1]
    )
    return {
        "overall_missing_rate": float(missing.mean()),
        "features_with_missing": int((per_feature > 0).sum()),
        "worst_features": [
            {"feature": name, "missing_rate": round(float(rate), 4)}
            for name, rate in list(worst)[:10]
            if rate > 0
        ],
    }
