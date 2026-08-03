"""Output paths, JSON writing, checkpoint bookkeeping and subject hashing."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

EXPERIMENT_FOLDER_NAME = "reproduction_hong_2024"
DEFAULT_COLAB_RESULTS_ROOT = Path(
    f"/content/drive/MyDrive/{EXPERIMENT_FOLDER_NAME}_result"
)
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def hash_subject(email: str, *, length: int = 12) -> str:
    """Map a raw EMAIL to a stable pseudonymous id.

    ``SangHyo/AGENTS.md`` forbids writing raw emails into results, so every
    artifact this package produces uses these hashes instead.
    """
    digest = hashlib.sha256(str(email).encode("utf-8")).hexdigest()
    return digest[:length]


def hash_sequence_id(sequence_id: str, *, length: int = 16) -> str:
    """Return a stable, namespaced hash for an internal sequence identifier.

    Internal ``sequence_id`` values deliberately carry the raw subject key so
    that split/scaler provenance can be audited.  They must therefore never be
    written verbatim to an artifact.  A separate namespace avoids making a
    sequence hash equal to a subject hash when their source strings happen to
    match.
    """
    digest = hashlib.sha256(f"sequence:{sequence_id}".encode("utf-8")).hexdigest()
    return digest[:length]


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def resolve_output_dir(explicit: str | Path | None, *, allow_local: bool = False) -> Path:
    """Resolve where a run writes its artifacts.

    On Colab this is a fresh ``<UTC_RUN_ID>`` directory on Drive, per the
    repository convention.  ``allow_local`` is for ``--dry-run`` / ``--audit-only``
    / ``--inspect-data``, which produce no performance numbers and may write
    beside the code.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()
    if DEFAULT_COLAB_RESULTS_ROOT.parent.is_dir():
        return (DEFAULT_COLAB_RESULTS_ROOT / utc_run_id()).resolve()
    if allow_local:
        return (Path(__file__).resolve().parents[2] / "outputs" / utc_run_id()).resolve()
    raise FileNotFoundError(
        "Google Drive is not mounted at /content/drive/MyDrive. Run base.ipynb Cell 1 "
        "first, or pass --output-dir explicitly for a local test."
    )


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            value = float(o)
            return value if np.isfinite(value) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)


def _json_safe(value: Any) -> Any:
    """Make values strict-JSON-safe and redact raw email subject identifiers."""
    if isinstance(value, Mapping):
        return {
            EMAIL_PATTERN.sub(
                lambda match: f"<subject_hash:{hash_subject(match.group(0))}>",
                str(key),
            ): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, str):
        return EMAIL_PATTERN.sub(
            lambda match: f"<subject_hash:{hash_subject(match.group(0))}>", value
        )
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        indent=2,
        cls=_NumpyEncoder,
        allow_nan=False,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_status(output_dir: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write ``LAUNCHER_STATUS.json`` -- the file the team's result contract reads."""
    return write_json(Path(output_dir) / "LAUNCHER_STATUS.json", payload)


class CheckpointStore:
    """Per-(model, sequence length, repeat, fold) bookkeeping for ``--resume``.

    Only completed units are recorded, so an interrupted run never resumes from a
    half-written fold.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, length: int, repeat: int, fold: int) -> str:
        return f"{model}__L{length}__r{repeat}__f{fold}"

    def marker(self, model: str, length: int, repeat: int, fold: int) -> Path:
        return self.root / f"{self._key(model, length, repeat, fold)}.json"

    def predictions_path(self, model: str, length: int, repeat: int, fold: int) -> Path:
        return self.root / f"{self._key(model, length, repeat, fold)}.predictions.csv"

    def model_path(self, model: str, length: int, repeat: int, fold: int, suffix: str) -> Path:
        return self.root / f"{self._key(model, length, repeat, fold)}{suffix}"

    def is_complete(self, model: str, length: int, repeat: int, fold: int) -> bool:
        # A result block without its predictions cannot support metric
        # round-tripping or a complete resumed FINAL_REPORT.  Old checkpoints
        # that only have JSON are intentionally treated as incomplete.
        marker = self.marker(model, length, repeat, fold)
        predictions = self.predictions_path(model, length, repeat, fold)
        if not marker.is_file() or not predictions.is_file():
            return False
        try:
            payload = read_json(marker)
            expected = payload.get("checkpoint_predictions")
            if not isinstance(expected, Mapping):
                return False
            self.load_predictions(
                model, length, repeat, fold, expected_metadata=expected
            )
            return True
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def load(self, model: str, length: int, repeat: int, fold: int) -> dict[str, Any]:
        return read_json(self.marker(model, length, repeat, fold))

    def save(
        self, model: str, length: int, repeat: int, fold: int, payload: Mapping[str, Any]
    ) -> Path:
        return write_json(self.marker(model, length, repeat, fold), payload)

    def save_predictions(
        self, model: str, length: int, repeat: int, fold: int, frame: Any
    ) -> Path:
        path = self.predictions_path(model, length, repeat, fold)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            frame.to_csv(temporary, index=False)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def predictions_metadata(
        self, model: str, length: int, repeat: int, fold: int, frame: Any
    ) -> dict[str, Any]:
        path = self.predictions_path(model, length, repeat, fold)
        if not path.is_file():
            raise FileNotFoundError(path)
        identity: dict[str, list[Any]] = {}
        for column in ("model", "sequence_length", "repeat", "outer_fold"):
            if column not in frame:
                continue
            values = frame[column].dropna().unique().tolist()
            identity[column] = sorted(
                (item.item() if isinstance(item, np.generic) else item for item in values),
                key=str,
            )
        return {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_rows": int(len(frame)),
            "columns": [str(column) for column in frame.columns],
            "identity": identity,
        }

    def load_predictions(
        self,
        model: str,
        length: int,
        repeat: int,
        fold: int,
        *,
        expected_metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        import pandas as pd

        path = self.predictions_path(model, length, repeat, fold)
        frame = pd.read_csv(path)
        required_columns = {
            "model", "subject_hash", "sequence_hash", "sequence_length", "y_true", "y_score"
        }
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"checkpoint sidecar is missing required columns: {sorted(missing_columns)}"
            )
        actual = self.predictions_metadata(model, length, repeat, fold, frame)
        identity = actual["identity"]
        if identity.get("model") != [model]:
            raise ValueError(
                f"checkpoint sidecar model identity {identity.get('model')} != {[model]}"
            )
        if length > 0 and identity.get("sequence_length") not in (None, [length]):
            raise ValueError(
                "checkpoint sidecar sequence length does not match its key: "
                f"{identity.get('sequence_length')} != {[length]}"
            )
        if identity.get("repeat") not in (None, [], [repeat]):
            raise ValueError(
                f"checkpoint sidecar repeat {identity.get('repeat')} != {[repeat]}"
            )
        if identity.get("outer_fold") not in (None, [], [fold]):
            raise ValueError(
                f"checkpoint sidecar fold {identity.get('outer_fold')} != {[fold]}"
            )
        if expected_metadata is None:
            return frame
        expected = dict(expected_metadata)
        if actual != expected:
            raise ValueError(
                "checkpoint prediction sidecar failed integrity validation: "
                f"expected={expected}, actual={actual}"
            )
        return frame
