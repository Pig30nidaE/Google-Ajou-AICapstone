"""Artifact integrity, atomic JSON, and label-free prediction freezing."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .leakage import hash_subject_id


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    """Atomically replace a JSON artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        default=json_default,
        allow_nan=False,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def environment_manifest(model_manifests: Sequence[Mapping[str, Any]]) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "models": list(model_manifests),
    }


def freeze_label_free_predictions(
    *,
    output_csv: str | Path,
    manifest_path: str | Path,
    track: str,
    subject_ids: Sequence[str],
    scores: Sequence[float],
    predictions: Sequence[int],
    deployment_sha256: str,
) -> dict:
    """Persist and hash predictions before historical labels may be opened."""

    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(subject_ids, dtype=str)
    score_values = np.asarray(scores, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if score_values.shape != ids.shape or predicted.shape != ids.shape:
        raise ValueError("Historical prediction arrays are misaligned")
    if (
        not np.isfinite(score_values).all()
        or np.any((score_values < 0.0) | (score_values > 1.0))
        or not set(np.unique(predicted)) <= {0, 1}
    ):
        raise ValueError("Historical predictions are invalid")
    hashes = [hash_subject_id(value) for value in ids]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Historical subject hash collision")
    pd.DataFrame(
        {
            "subject_hash": hashes,
            "score_dem": score_values,
            "prediction_dem": predicted,
        }
    ).to_csv(destination, index=False)
    prediction_sha = sha256_file(destination)
    payload = {
        "status": "frozen",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "track": track,
        "n_subjects": len(ids),
        "ordered_subject_hashes": hashes,
        "prediction_file": str(destination.resolve()),
        "prediction_file_sha256": prediction_sha,
        "deployment_sha256": deployment_sha256,
        "labels_opened_before_freeze": False,
        "historical_validation_is_independent_test": False,
    }
    write_json(manifest_path, payload)
    # Verify bytes and JSON after close, before returning permission to load labels.
    reread = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if reread != payload or sha256_file(destination) != prediction_sha:
        raise IOError("Prediction freeze round-trip verification failed")
    return payload


__all__ = [
    "environment_manifest",
    "freeze_label_free_predictions",
    "json_default",
    "sha256_file",
    "write_json",
]
