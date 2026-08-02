"""base.ipynb entry point -- dementia screening (CN+MCI vs Dem) with Google YDF.

    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Binary_Google_DemScreen/run.py"

Recommended Colab runtime: **CPU, High-RAM. No GPU** (YDF is a multi-threaded CPU
library; every model here is tiny by design).

Modes
-----
``smoke``     ~1 min    wiring check, 2 repeats, metrics meaningless
``standard``  ~10 min   10 repeats
``full``      ~25 min   20 repeats (default; matches Hyunsoo's protocol)

Environment switches::

    DEMSCREEN_SEARCH=1   enable the small opt-in grid (off by default -- see spaces.py)
    DEMSCREEN_SMOTE=1    enable SMOTE inside training folds (off by default)
    DEMSCREEN_KINDS=a,b  restrict the learner pool
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

EXPERIMENT_NAME = "Binary_Google_DemScreen"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
DEFAULT_COLAB_RESULTS_ROOT = Path(f"/content/drive/MyDrive/{EXPERIMENT_NAME}_result")

ALL_KINDS = ("univariate", "logreg", "ydf_gbt", "ydf_rf", "ydf_gbt_oblique")

MODE_SETTINGS: dict[str, dict[str, Any]] = {
    "smoke": dict(repeats=2, outer_k=5, inner_k=3),
    "standard": dict(repeats=10, outer_k=5, inner_k=4),
    "full": dict(repeats=20, outer_k=5, inner_k=4),
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


def _resolve_data_root(namespace: Mapping[str, Any], explicit: str | None) -> Path:
    candidates: list[Path] = []
    for value in (explicit, namespace.get("DATA_ROOT"), os.environ.get("SANGHYO_DATA_ROOT")):
        if value:
            candidates.append(Path(os.fspath(value)).expanduser())
    project = namespace.get("PROJECT_ROOT")
    if project:
        candidates.append(Path(os.fspath(project)) / "Data")
    candidates.extend([
        REPOSITORY_ROOT / "Data",
        Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
        Path("/content/drive/MyDrive/GoogleAI_contest/Data"),
    ])
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "1.Training").is_dir() and (resolved / "2.Validation").is_dir():
            return resolved
    raise FileNotFoundError("Data root with 1.Training and 2.Validation not found. Checked: "
                            + ", ".join(map(str, candidates)))


def _ensure_dependencies(skip_install: bool, want_smote: bool) -> None:
    required = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
                "scikit-learn": "sklearn", "joblib": "joblib"}
    missing = [d for d, m in required.items() if importlib.util.find_spec(m) is None]
    if missing and not skip_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "-r", str(REQUIREMENTS_FILE)], check=True)
        importlib.invalidate_caches()
    elif missing:
        raise ModuleNotFoundError("Missing dependencies: " + ", ".join(missing))

    for package, module in (("ydf", "ydf"),) + ((("imbalanced-learn", "imblearn"),) if want_smote else ()):
        if importlib.util.find_spec(module) is None and not skip_install:
            print(f"[run] installing {package}...", flush=True)
            result = subprocess.run([sys.executable, "-m", "pip", "install",
                                     "--disable-pip-version-check", package], check=False)
            importlib.invalidate_caches()
            if result.returncode != 0:
                print(f"[run] WARNING: {package} install failed; continuing with fallback.",
                      flush=True)


def _resolve_output_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    override = os.environ.get("SANGHYO_OUTPUT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if not DEFAULT_COLAB_RESULTS_ROOT.parent.is_dir():
        raise FileNotFoundError("Google Drive not mounted at /content/drive/MyDrive. Run "
                                "base.ipynb Cell 1 first, or pass --output-dir for a local test.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    return (DEFAULT_COLAB_RESULTS_ROOT / run_id).resolve()


def _write_status(output: Path, payload: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "LAUNCHER_STATUS.json").write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_kinds() -> tuple[str, ...]:
    raw = os.environ.get("DEMSCREEN_KINDS", "").strip()
    if not raw:
        return ALL_KINDS
    chosen = tuple(k.strip() for k in raw.split(",") if k.strip())
    unknown = [k for k in chosen if k not in ALL_KINDS]
    if unknown:
        raise ValueError(f"Unknown learner kinds in DEMSCREEN_KINDS: {unknown}")
    return chosen


def run_pipeline(*, namespace=None, data_root=None, output_dir=None, mode=None,
                 skip_install=False) -> dict:
    namespace = globals() if namespace is None else namespace
    run_mode = (mode or os.environ.get("DEMSCREEN_MODE") or "full").strip().lower()
    if run_mode not in MODE_SETTINGS:
        raise ValueError(f"mode must be one of {sorted(MODE_SETTINGS)}")

    want_smote = _env_flag("DEMSCREEN_SMOTE")
    _ensure_dependencies(skip_install, want_smote)

    resolved_data = _resolve_data_root(namespace, data_root)
    output = _resolve_output_dir(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    kinds = _resolve_kinds()
    started = time.monotonic()
    _write_status(output, {"status": "starting", "experiment": EXPERIMENT_NAME, "mode": run_mode,
                           "kinds": list(kinds), "data_root": str(resolved_data),
                           "output_dir": str(output)})

    from SangHyo.Binary.Binary_Google_DemScreen.train import RunConfig, run_experiment

    config = RunConfig(data_root=str(resolved_data), output_dir=str(output), run_mode=run_mode,
                       kinds=kinds, search=_env_flag("DEMSCREEN_SEARCH"), smote=want_smote,
                       **MODE_SETTINGS[run_mode])
    try:
        result = run_experiment(config)
    except Exception as error:
        _write_status(output, {"status": "failed", "mode": run_mode,
                               "error_type": type(error).__name__, "error": str(error),
                               "elapsed_seconds": time.monotonic() - started})
        raise

    _write_status(output, {"status": "complete", "mode": run_mode,
                           "elapsed_seconds": time.monotonic() - started,
                           "headline_roc_auc": result["headline_roc_auc"],
                           "final_report": str(output / "training" / "FINAL_REPORT.json")})
    print(f"\nComplete ({run_mode}) in {(time.monotonic() - started) / 60:.1f} min. "
          f"Report: {output / 'training' / 'FINAL_REPORT.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=tuple(MODE_SETTINGS), default=None)
    parser.add_argument("--skip-install", action="store_true")
    arguments, _unknown = parser.parse_known_args()
    run_pipeline(namespace=globals(), data_root=arguments.data_root,
                 output_dir=arguments.output_dir, mode=arguments.mode,
                 skip_install=arguments.skip_install)


if __name__ == "__main__":
    main()
