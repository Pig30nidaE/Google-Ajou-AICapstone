"""Result artifacts must be valid for strict JSON consumers."""

from __future__ import annotations

import json

import numpy as np

from src.evaluation.metrics import compute_metrics
from src.utils.io import CheckpointStore, sha256_file, write_json


def test_write_json_replaces_every_non_finite_number(tmp_path) -> None:
    path = write_json(
        tmp_path / "strict.json",
        {
            "python_nan": float("nan"),
            "numpy_inf": np.float64("inf"),
            "nested": [1.0, float("-inf")],
        },
    )
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert json.loads(raw) == {
        "python_nan": None,
        "numpy_inf": None,
        "nested": [1.0, None],
    }


def test_undefined_precision_is_none_not_nan() -> None:
    metrics = compute_metrics([0, 0, 1], [0.2, 0.3, 0.4], threshold=0.5)
    assert metrics["precision"] is None


def test_resume_rejects_stale_fingerprint_or_changed_model(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    model_path = store.model_path("cnn1d", 0, 0, ".pt")
    model_path.write_bytes(b"checkpoint-v2")
    store.save(
        "cnn1d",
        0,
        0,
        {
            "artifact_schema_version": 2,
            "run_fingerprint": "current-run",
            "fold_record": {
                "model": "cnn1d",
                "repeat": 0,
                "fold": 0,
                "audit_passed": True,
                "subject_metrics": {"roc_auc": 0.5},
                "input_shape": [3, 2],
                "artifact_schema_version": 2,
                "run_fingerprint": "current-run",
            },
            "subject_predictions": [
                {"subject_id": "hashed", "probability": 0.4, "label": 0}
            ],
            "subject_metrics": {"roc_auc": 0.5},
            "model_summary": {"name": "cnn1d"},
            "audit": {"all_passed": True},
            "input_shape": [3, 2],
            "preprocessor_state": {
                "standardize": True,
                "impute": True,
                "feature_names": ["a", "b"],
                "medians": [0.0, 0.0],
                "mean": [0.0, 0.0],
                "scale": [1.0, 1.0],
                "fitted_subjects": ["train"],
            },
            "representation_meta": {
                "kind": "temporal_sequence",
                "train": {},
                "test": {},
                "feature_names": ["a", "b"],
            },
            "train_subjects": ["train"],
            "test_subjects": ["hashed"],
            "threshold": 0.5,
            "threshold_source": "fixed",
            "model_checkpoint": {
                "filename": model_path.name,
                "sha256": sha256_file(model_path),
            },
        },
    )

    assert store.is_complete(
        "cnn1d", 0, 0,
        expected_schema_version=2,
        expected_run_fingerprint="current-run",
        require_model=True,
    )
    assert not store.is_complete(
        "cnn1d", 0, 0,
        expected_schema_version=2,
        expected_run_fingerprint="stale-run",
        require_model=True,
    )

    model_path.write_bytes(b"tampered")
    assert not store.is_complete(
        "cnn1d", 0, 0,
        expected_schema_version=2,
        expected_run_fingerprint="current-run",
        require_model=True,
    )
