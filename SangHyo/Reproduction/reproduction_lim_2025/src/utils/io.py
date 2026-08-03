"""Output paths, JSON writing, checkpoint bookkeeping and subject hashing."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

EXPERIMENT_FOLDER_NAME = "reproduction_lim_2025"
DEFAULT_COLAB_RESULTS_ROOT = Path(
    f"/content/drive/MyDrive/{EXPERIMENT_FOLDER_NAME}_result"
)


def hash_subject(email: str, *, length: int = 12) -> str:
    """Map a raw EMAIL to a stable pseudonymous id.

    ``SangHyo/AGENTS.md`` forbids writing raw emails into results, so every
    artifact this package produces uses these hashes instead.
    """
    digest = hashlib.sha256(str(email).encode("utf-8")).hexdigest()
    return digest[:length]


def sha256_file(path: str | Path) -> str:
    """Content identity used for data/model checkpoint provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def resolve_output_dir(explicit: str | Path | None, *, allow_local: bool = False) -> Path:
    """Resolve where a run writes its artifacts.

    On Colab this is a fresh ``<UTC_RUN_ID>`` directory on Drive, per the
    repository convention.  ``allow_local`` is for ``--dry-run`` / ``--audit-only``,
    which produce no performance numbers and may write beside the code.
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
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            value = float(o)
            return value if np.isfinite(value) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, (datetime,)):
            return o.isoformat()
        return super().default(o)


def _json_safe(value: Any) -> Any:
    """Recursively convert payloads to strict RFC-compatible JSON values.

    ``JSONEncoder.default`` never sees built-in ``float('nan')`` values, so an
    encoder alone allowed literal ``NaN`` into result files.  Those files happen
    to load in Python's permissive parser but fail in strict consumers such as
    ``JSON.parse``.  Normalise first and then require ``allow_nan=False``.
    """
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        indent=2,
        cls=_NumpyEncoder,
        allow_nan=False,
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(serialised)
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
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
    """Per-(repeat, fold, model) checkpoint bookkeeping for ``--resume``.

    Only completed units are recorded, so an interrupted run never resumes from a
    half-written fold.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, repeat: int, fold: int) -> str:
        return f"{model}__r{repeat}__f{fold}"

    def marker(self, model: str, repeat: int, fold: int) -> Path:
        return self.root / f"{self._key(model, repeat, fold)}.json"

    def model_path(self, model: str, repeat: int, fold: int, suffix: str) -> Path:
        return self.root / f"{self._key(model, repeat, fold)}{suffix}"

    def is_complete(
        self,
        model: str,
        repeat: int,
        fold: int,
        *,
        expected_schema_version: int | None = None,
        expected_run_fingerprint: str | None = None,
        require_model: bool = False,
    ) -> bool:
        """Only accept a complete, current, content-verified checkpoint marker."""
        marker = self.marker(model, repeat, fold)
        if not marker.is_file():
            return False
        try:
            payload = read_json(marker)
            if not isinstance(payload, Mapping):
                return False
            required = {
                "fold_record", "subject_predictions", "subject_metrics",
                "model_summary", "audit", "input_shape", "preprocessor_state",
                "representation_meta", "train_subjects", "test_subjects",
                "threshold", "threshold_source",
            }
            if not required.issubset(payload):
                return False
            if not (
                isinstance(payload["fold_record"], Mapping)
                and isinstance(payload["subject_predictions"], list)
                and isinstance(payload["subject_metrics"], Mapping)
                and isinstance(payload["model_summary"], Mapping)
                and isinstance(payload["audit"], Mapping)
                and isinstance(payload["input_shape"], list)
                and isinstance(payload["preprocessor_state"], Mapping)
                and isinstance(payload["representation_meta"], Mapping)
                and isinstance(payload["train_subjects"], list)
                and isinstance(payload["test_subjects"], list)
            ):
                return False
            if not (
                payload["subject_predictions"]
                and payload["subject_metrics"]
                and payload["model_summary"]
                and payload["audit"]
                and payload["input_shape"]
                and payload["preprocessor_state"]
                and payload["representation_meta"]
                and payload["train_subjects"]
                and payload["test_subjects"]
            ):
                return False
            if not {
                "standardize", "impute", "feature_names", "medians", "mean",
                "scale", "fitted_subjects",
            }.issubset(payload["preprocessor_state"]):
                return False
            if not {
                "kind", "train", "test", "feature_names",
            }.issubset(payload["representation_meta"]):
                return False
            first_prediction = payload["subject_predictions"][0]
            if not (
                isinstance(first_prediction, Mapping)
                and {"subject_id", "probability", "label"}.issubset(first_prediction)
            ):
                return False
            fold_record = payload["fold_record"]
            if not {
                "model", "repeat", "fold", "audit_passed", "subject_metrics",
                "input_shape", "artifact_schema_version", "run_fingerprint",
            }.issubset(fold_record):
                return False
            if (
                fold_record.get("model") != model
                or int(fold_record.get("repeat", -1)) != int(repeat)
                or int(fold_record.get("fold", -1)) != int(fold)
                or not isinstance(fold_record.get("subject_metrics"), Mapping)
                or not isinstance(fold_record.get("input_shape"), list)
                or fold_record.get("artifact_schema_version")
                != payload.get("artifact_schema_version")
                or fold_record.get("run_fingerprint")
                != payload.get("run_fingerprint")
            ):
                return False
            if (
                expected_schema_version is not None
                and int(payload.get("artifact_schema_version", -1))
                != int(expected_schema_version)
            ):
                return False
            if (
                expected_run_fingerprint is not None
                and payload.get("run_fingerprint") != expected_run_fingerprint
            ):
                return False
            model_meta = payload.get("model_checkpoint")
            if require_model:
                if not isinstance(model_meta, Mapping):
                    return False
                filename = str(model_meta.get("filename", ""))
                if not filename or Path(filename).name != filename:
                    return False
                model_path = self.root / filename
                if not model_path.is_file():
                    return False
                if sha256_file(model_path) != model_meta.get("sha256"):
                    return False
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return True

    def load(self, model: str, repeat: int, fold: int) -> dict[str, Any]:
        return read_json(self.marker(model, repeat, fold))

    def save(self, model: str, repeat: int, fold: int, payload: Mapping[str, Any]) -> Path:
        return write_json(self.marker(model, repeat, fold), payload)
