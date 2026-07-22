"""Single launcher used by the repository's ``base.ipynb`` notebook."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping


EXPERIMENT_NAME = "Binary_Wearable_SequenceFusion_Google"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
SOFT_RUNTIME_SECONDS = 20_700  # 5 h 45 m
HARD_RUNTIME_SECONDS = 21_600  # 6 h


class HardDeadlineExceeded(TimeoutError):
    pass


def _alarm_handler(signum, frame):  # pragma: no cover - platform signal path
    del signum, frame
    raise HardDeadlineExceeded("The six-hour hard runtime limit was reached")


def _resolve_data_root(
    namespace: Mapping[str, Any], explicit: str | None
) -> Path:
    candidates: list[Path] = []
    for value in (
        explicit,
        namespace.get("DATA_ROOT"),
        os.environ.get("BINARY_SEQUENCE_DATA_ROOT"),
    ):
        if value:
            candidates.append(Path(os.fspath(value)).expanduser())
    project = namespace.get("PROJECT_ROOT")
    if project:
        candidates.append(Path(os.fspath(project)) / "Data")
    candidates.extend(
        [
            EXPERIMENT_ROOT.parents[1] / "Data",
            Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
            Path("/content/drive/MyDrive/GoogleAI_contest/Data"),
        ]
    )
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "1.Training").is_dir() and (
            resolved / "2.Validation"
        ).is_dir():
            return resolved
    raise FileNotFoundError(
        "Data root containing 1.Training and 2.Validation was not found. Checked: "
        + ", ".join(str(item) for item in candidates)
    )


def _ensure_dependencies(skip_install: bool) -> None:
    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "joblib": "joblib",
        "ydf": "ydf",
    }
    missing = [
        distribution
        for distribution, module in modules.items()
        if importlib.util.find_spec(module) is None
    ]
    try:
        ydf_version = importlib.metadata.version("ydf")
    except importlib.metadata.PackageNotFoundError:
        ydf_version = None
    if ydf_version != "0.16.1" and "ydf" not in missing:
        missing.append("ydf")
    if missing:
        if skip_install:
            raise ModuleNotFoundError("Missing dependencies: " + ", ".join(missing))
        print("Installing experiment requirements because these are missing:", missing)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(REQUIREMENTS_FILE),
            ],
            check=True,
        )
        importlib.invalidate_caches()
        still_missing = [
            distribution
            for distribution, module in modules.items()
            if importlib.util.find_spec(module) is None
        ]
        try:
            installed_ydf = importlib.metadata.version("ydf")
        except importlib.metadata.PackageNotFoundError:
            installed_ydf = None
        if installed_ydf != "0.16.1":
            still_missing.append("ydf==0.16.1")
        if still_missing:
            raise ModuleNotFoundError(
                "Dependency installation finished but these are unavailable: "
                + ", ".join(still_missing)
            )
    if importlib.util.find_spec("torch") is None:
        raise ModuleNotFoundError(
            "PyTorch is required. Colab already supplies the correct A100 CUDA build; "
            "select an A100 runtime and restart before running this file."
        )


def _write_launcher_status(output: Path, payload: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "LAUNCHER_STATUS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_pipeline(
    *,
    namespace: Mapping[str, Any] | None = None,
    data_root: str | None = None,
    output_dir: str | None = None,
    mode: str | None = None,
    skip_install: bool = False,
    evaluate_historical_validation: bool = True,
) -> dict[str, Any]:
    """Resolve paths and run EDA through frozen historical evaluation."""

    namespace = globals() if namespace is None else namespace
    run_mode = (mode or os.environ.get("BINARY_SEQUENCE_RUN_MODE", "full")).lower()
    if run_mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")
    _ensure_dependencies(skip_install)

    import torch

    if run_mode == "full" and not torch.cuda.is_available():
        if os.environ.get("BINARY_ALLOW_CPU_FULL") != "1":
            raise RuntimeError(
                "Full mode requires a CUDA GPU. Use the requested Colab A100 runtime, "
                "or set BINARY_ALLOW_CPU_FULL=1 only if a much slower CPU run is intended."
            )
    resolved_data = _resolve_data_root(namespace, data_root)
    if output_dir:
        output = Path(output_dir).expanduser().resolve()
    elif os.environ.get("BINARY_SEQUENCE_OUTPUT_DIR"):
        output = Path(os.environ["BINARY_SEQUENCE_OUTPUT_DIR"]).expanduser().resolve()
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
        output = EXPERIMENT_ROOT / "outputs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))
    started = time.monotonic()
    _write_launcher_status(
        output,
        {
            "status": "starting",
            "experiment": EXPERIMENT_NAME,
            "mode": run_mode,
            "data_root": str(resolved_data),
            "output_dir": str(output),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "note": "Neural models use A100; Google YDF uses the Colab CPU cores.",
        },
    )
    if __package__:
        from .train import RunConfig, run_experiment
    else:
        from train import RunConfig, run_experiment

    config = RunConfig(
        training_root=str(resolved_data / "1.Training"),
        validation_root=str(resolved_data / "2.Validation"),
        output_dir=str(output),
        run_mode=run_mode,
        max_runtime_seconds=1_800 if run_mode == "smoke" else SOFT_RUNTIME_SECONDS,
        evaluate_historical_validation=evaluate_historical_validation,
    )
    try:
        result = run_experiment(config)
    except Exception as exc:
        _write_launcher_status(
            output,
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    _write_launcher_status(
        output,
        {
            "status": "complete",
            "elapsed_seconds": time.monotonic() - started,
            "final_report": str(output / "training" / "FINAL_REPORT.json"),
        },
    )
    print("\nComplete. Final report:", output / "training" / "FINAL_REPORT.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--mode", choices=("full", "smoke"), default=os.environ.get("BINARY_SEQUENCE_RUN_MODE", "full")
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--no-evaluate-validation", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring notebook/kernel arguments:", unknown)
    alarm_enabled = hasattr(signal, "SIGALRM") and args.mode == "full"
    if alarm_enabled:
        signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(HARD_RUNTIME_SECONDS)
    try:
        run_pipeline(
            namespace=globals(),
            data_root=args.data_root,
            output_dir=args.output_dir,
            mode=args.mode,
            skip_install=args.skip_install,
            evaluate_historical_validation=not args.no_evaluate_validation,
        )
    finally:
        if alarm_enabled:
            signal.alarm(0)


if __name__ == "__main__":
    main()
