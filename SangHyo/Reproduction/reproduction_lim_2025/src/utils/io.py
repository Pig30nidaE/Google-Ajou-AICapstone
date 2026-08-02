"""Output paths, JSON writing, checkpoint bookkeeping and subject hashing."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
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


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, cls=_NumpyEncoder),
        encoding="utf-8",
    )
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

    def is_complete(self, model: str, repeat: int, fold: int) -> bool:
        return self.marker(model, repeat, fold).is_file()

    def load(self, model: str, repeat: int, fold: int) -> dict[str, Any]:
        return read_json(self.marker(model, repeat, fold))

    def save(self, model: str, repeat: int, fold: int, payload: Mapping[str, Any]) -> Path:
        return write_json(self.marker(model, repeat, fold), payload)
