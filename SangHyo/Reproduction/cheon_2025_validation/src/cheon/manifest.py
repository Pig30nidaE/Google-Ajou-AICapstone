"""Run manifest: timestamps, versions, git state, fingerprints. Never includes subject identifiers."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.name).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def git_info(repo_root: Path) -> dict:
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain")) if run("status", "--porcelain") is not None else None}


def package_versions() -> dict:
    out = {}
    for name in ("lightgbm", "shap", "sklearn", "pandas", "numpy", "scipy", "yaml"):
        try:
            mod = __import__(name)
            out[name if name != "sklearn" else "scikit-learn"] = getattr(mod, "__version__", "?")
        except Exception as e:  # pragma: no cover
            out[name] = f"unavailable: {e}"
    return out


def environment_text() -> str:
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        freeze = "(pip freeze unavailable)\n"
    hw = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
          "cpu_count": os.cpu_count(), "python": sys.version}
    return "# environment\n" + "\n".join(f"{k}: {v}" for k, v in hw.items()) + "\n\n# pip freeze\n" + freeze


def build_manifest(*, experiment: str, config: dict, config_text: str, repo_root: Path, code_paths: list[Path],
                   data_fp: str, seeds: list[int], start: str, end: str | None = None, extra: dict | None = None) -> dict:
    t0 = datetime.fromisoformat(start)
    t1 = datetime.fromisoformat(end) if end else None
    return {
        "experiment": experiment,
        "start_utc": start, "end_utc": end,
        "runtime_seconds": (t1 - t0).total_seconds() if t1 else None,
        "git": git_info(repo_root),
        "python_version": sys.version,
        "packages": package_versions(),
        "platform": {"platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count()},
        "seeds": seeds,
        "data_fingerprint_sha256": data_fp,
        "code_fingerprint_sha256": code_fingerprint(code_paths),
        "config_fingerprint_sha256": _sha256_bytes(config_text.encode()),
        "config": config,
        **(extra or {}),
    }


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default))


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
