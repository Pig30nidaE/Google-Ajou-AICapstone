"""base.ipynb entry point — MMSE-focused, leakage-free, max ROC-AUC.

    USER_FOLDER = "SangHyo"
    RUN_FILE = "Binary_MMSE_MaxAUC/run.py"

CPU-only. Runs EDA -> preprocessing -> subject-level split -> training and reports
leakage-free subject-level ROC-AUC as the primary metric. Wearable features are
optional (env MAXAUC_INCLUDE_WEARABLE=1); EDA shows MMSE-only is the max-AUC choice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping

EXPERIMENT_NAME = "Binary_MMSE_MaxAUC"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
DEFAULT_COLAB_RESULTS_ROOT = Path(f"/content/drive/MyDrive/{EXPERIMENT_NAME}_result")
HARD_RUNTIME_SECONDS = 21_600


class HardDeadlineExceeded(TimeoutError):
    pass


def _alarm_handler(signum, frame):  # pragma: no cover
    del signum, frame
    raise HardDeadlineExceeded("Six-hour hard runtime limit reached")


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


def _ensure_dependencies(skip_install: bool) -> None:
    required = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
                "scikit-learn": "sklearn", "matplotlib": "matplotlib", "joblib": "joblib"}
    missing = [d for d, m in required.items() if importlib.util.find_spec(m) is None]
    if missing and not skip_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "-r", str(REQUIREMENTS_FILE)], check=True)
        importlib.invalidate_caches()
    elif missing:
        raise ModuleNotFoundError("Missing dependencies: " + ", ".join(missing))


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


def run_pipeline(*, namespace=None, data_root=None, output_dir=None, mode=None,
                 skip_install=False, evaluate_validation=True) -> dict:
    namespace = globals() if namespace is None else namespace
    run_mode = (mode or "full").strip().lower()
    if run_mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")
    _ensure_dependencies(skip_install)

    resolved_data = _resolve_data_root(namespace, data_root)
    output = _resolve_output_dir(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))

    include_wearable = os.environ.get("MAXAUC_INCLUDE_WEARABLE", "0").strip().lower() in {"1", "true", "yes"}
    started = time.monotonic()
    _write_status(output, {"status": "starting", "experiment": EXPERIMENT_NAME, "mode": run_mode,
                           "include_wearable": include_wearable, "data_root": str(resolved_data),
                           "output_dir": str(output)})

    from SangHyo.Binary.Binary_MMSE_MaxAUC.train import RunConfig, run_experiment

    settings = dict(repeats=1, folds=3, inner_folds=2) if run_mode == "smoke" else dict(repeats=5, folds=5, inner_folds=3)
    config = RunConfig(
        training_root=str(resolved_data / "1.Training"),
        validation_root=str(resolved_data / "2.Validation"),
        output_dir=str(output), run_mode=run_mode, include_wearable=include_wearable,
        evaluate_validation=evaluate_validation, **settings,
    )
    try:
        result = run_experiment(config)
    except Exception as error:
        _write_status(output, {"status": "failed", "mode": run_mode,
                               "error_type": type(error).__name__, "error": str(error),
                               "elapsed_seconds": time.monotonic() - started})
        raise

    final_report = output / "training" / "FINAL_REPORT.json"
    _write_status(output, {"status": "complete", "mode": run_mode,
                           "elapsed_seconds": time.monotonic() - started,
                           "final_report": str(final_report)})
    print(f"\nComplete ({run_mode}). Final report: {final_report}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    arguments, _unknown = parser.parse_known_args()

    alarm_enabled = arguments.mode.strip().lower() == "full" and hasattr(signal, "SIGALRM")
    previous = None
    if alarm_enabled:
        previous = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(HARD_RUNTIME_SECONDS)
    try:
        run_pipeline(namespace=globals(), data_root=arguments.data_root,
                     output_dir=arguments.output_dir, mode=arguments.mode,
                     skip_install=arguments.skip_install,
                     evaluate_validation=not arguments.skip_validation)
    finally:
        if alarm_enabled:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
