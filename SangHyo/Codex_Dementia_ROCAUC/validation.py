"""Zero-fit code/split contracts and read-only data audit helpers."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .data import load_development_cohort
from .features import aggregate_wearable_sequences
from .leakage import (
    assert_disjoint_groups,
    assert_no_forbidden_features,
    assert_unique_subjects,
)
from .models.base import model_specs
from .splits import build_repeated_group_plan, split_plan_payload


def validate_code_without_fit(package_root: str | Path) -> dict[str, Any]:
    """Validate source contracts without calling any estimator ``fit`` method."""

    root = Path(package_root).resolve()
    python_files = sorted(root.rglob("*.py"))
    if not python_files:
        raise FileNotFoundError(f"No Python files below {root}")
    syntax_files = []
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        syntax_files.append(str(path))

    synthetic_ids = np.asarray([f"s{i:02d}" for i in range(18)])
    synthetic_y = np.asarray([0] * 12 + [1] * 6, dtype=np.int64)
    plan = build_repeated_group_plan(
        synthetic_y,
        synthetic_ids,
        n_splits=3,
        n_repeats=2,
        seed=17,
        minimum_positive_validation=1,
        layer="zero_fit_synthetic",
    )
    payload = split_plan_payload(
        plan,
        subject_ids=synthetic_ids,
        y=synthetic_y,
    )
    sequences = tuple(
        np.column_stack(
            [
                np.linspace(index, index + 1.0, 35),
                np.sin(np.linspace(0.0, 3.0, 35) + index),
            ]
        )
        for index in range(len(synthetic_ids))
    )
    biological, protocol = aggregate_wearable_sequences(
        synthetic_ids,
        sequences,
        ("activity__scalar__low", "sleep__scalar__duration"),
    )
    assert_unique_subjects(synthetic_ids)
    assert_no_forbidden_features(tuple(biological.columns))
    assert_no_forbidden_features(tuple(protocol.columns))
    assert_disjoint_groups(
        synthetic_ids[:9],
        synthetic_ids[9:],
        context="zero-fit synthetic disjointness",
    )
    return {
        "status": "passed",
        "training_calls_executed": 0,
        "syntax_files": syntax_files,
        "synthetic_split_records": len(payload["records"]),
        "synthetic_biological_features": biological.shape[1],
        "synthetic_protocol_features": protocol.shape[1],
        "registered_models": [spec.engine_manifest() for spec in model_specs()],
    }


def audit_data_without_training(config: ExperimentConfig) -> dict[str, Any]:
    """Read schema/IDs/features for enabled tracks; never instantiate a model."""

    tracks = {}
    for track in config.data.tracks:
        cohort = load_development_cohort(config.data, track)
        tracks[track] = {
            "n_subjects": cohort.n_subjects,
            "n_positive_dem": cohort.n_positive,
            "n_negative_cn_mci": int((cohort.y == 0).sum()),
            "sequence_length_min": int(
                min(sequence.shape[0] for sequence in cohort.sequences)
            ),
            "sequence_length_median": float(
                np.median([sequence.shape[0] for sequence in cohort.sequences])
            ),
            "sequence_length_max": int(
                max(sequence.shape[0] for sequence in cohort.sequences)
            ),
            "daily_schema_width": len(cohort.daily_feature_names),
            "input_fingerprints": cohort.input_fingerprints,
            "access_audit": cohort.access_audit,
        }
    return {
        "status": "passed",
        "training_calls_executed": 0,
        "tracks": tracks,
    }


__all__ = ["audit_data_without_training", "validate_code_without_fit"]

