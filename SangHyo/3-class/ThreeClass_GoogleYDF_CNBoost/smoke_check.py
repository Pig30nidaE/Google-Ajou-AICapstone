"""Feature/preprocessing contract check that does not train a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from feature_engineering import build_subject_dataset
from preprocessing import FoldFeatureSelector


def run(training_root: str, validation_root: str, output_json: str | None) -> dict:
    training = build_subject_dataset(training_root, require_labels=True)
    validation = build_subject_dataset(validation_root, require_labels=False)
    if training.y is None:
        raise AssertionError("Training labels are absent")
    if len(training.y) != 141 or np.bincount(training.y, minlength=3).tolist() != [85, 47, 9]:
        raise AssertionError("Training cohort contract changed")
    if list(training.X.columns) != list(validation.X.columns):
        raise AssertionError("Training/validation feature schemas differ")
    selector = FoldFeatureSelector(max_features=48, correlation_threshold=0.95)
    transformed_training = selector.fit_transform(
        training.X,
        training.y,
        task="multiclass",
    )
    transformed_validation = selector.transform(validation.X)
    if not np.isfinite(transformed_training).all() or not np.isfinite(transformed_validation).all():
        raise FloatingPointError("Non-finite value in smoke-check transform")
    payload = {
        "model_training_executed": False,
        "training_subjects": int(len(training.subject_ids)),
        "validation_subjects": int(len(validation.subject_ids)),
        "raw_feature_count": int(training.X.shape[1]),
        "selected_feature_count": int(transformed_training.shape[1]),
        "training_shape": list(transformed_training.shape),
        "validation_shape": list(transformed_validation.shape),
        "mmse_source_opened": False,
        "mmse_values_used": False,
        "validation_labels_opened": False,
        "finite_values": True,
        "selector_manifest": selector.manifest(),
    }
    if output_json:
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--output-json")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments.training_root, arguments.validation_root, arguments.output_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))

