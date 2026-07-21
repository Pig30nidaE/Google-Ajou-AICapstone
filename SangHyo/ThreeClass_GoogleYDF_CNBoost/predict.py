"""Label-free inference from a completed full-training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from evaluation import apply_recipe
from feature_engineering import CLASS_NAMES, build_subject_dataset
from models import FittedCandidate


def _hash_subject(value: object) -> str:
    return hashlib.sha256(f"google_ydf_cnboost_mmse_free_v1::{value}".encode()).hexdigest()[:24]


def run(model_dir: str, data_root: str, output_file: str) -> None:
    checkpoint = Path(model_dir).expanduser().resolve()
    complete = checkpoint / "FULL_CHECKPOINT_COMPLETE.json"
    selection_path = checkpoint / "full_selection.json"
    if not complete.is_file() or not selection_path.is_file():
        raise FileNotFoundError(f"Incomplete full-training checkpoint: {checkpoint}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    weights = selection["training_only_oof_blend"]["weights"]
    data = build_subject_dataset(data_root, require_labels=False)
    probabilities_by_candidate = {
        candidate_name: FittedCandidate.load(checkpoint / candidate_name).predict_proba(data.X)
        for candidate_name in weights
    }
    probabilities = apply_recipe(probabilities_by_candidate, weights)
    frame = pd.DataFrame(
        {
            "subject_hash": [_hash_subject(value) for value in data.subject_ids],
            "predicted_class": [CLASS_NAMES[index] for index in probabilities.argmax(axis=1)],
            "p_cn": probabilities[:, 0],
            "p_mci": probabilities[:, 1],
            "p_dem": probabilities[:, 2],
        }
    )
    destination = Path(output_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prediction file: {destination}")
    frame.to_csv(destination, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.model_dir, arguments.data_root, arguments.output_file)

