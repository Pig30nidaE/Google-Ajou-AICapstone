"""Single Colab/base.ipynb launcher; full A100 mode is the default."""

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


EXPERIMENT_NAME = "Binary_Wearable_BalancedFusion_Google"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
DEFAULT_COLAB_RESULTS_ROOT = Path(
    "/content/drive/MyDrive/Binary_Wearable_BalancedFusion_Google_result"
)
SOFT_RUNTIME_SECONDS = 20_400
HARD_RUNTIME_SECONDS = 21_600


class HardDeadlineExceeded(TimeoutError):
    pass


def _alarm_handler(signum, frame):  # pragma: no cover - Linux/Colab path
    del signum, frame
    raise HardDeadlineExceeded("The six-hour hard runtime limit was reached")


def _resolve_data_root(namespace: Mapping[str, Any], explicit: str | None) -> Path:
    candidates: list[Path] = []
    for value in (
        explicit,
        namespace.get("DATA_ROOT"),
        os.environ.get("BALANCED_FUSION_DATA_ROOT"),
    ):
        if value:
            candidates.append(Path(os.fspath(value)).expanduser())
    project = namespace.get("PROJECT_ROOT")
    if project:
        candidates.append(Path(os.fspath(project)) / "Data")
    candidates.extend(
        [
            REPOSITORY_ROOT / "Data",
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
        + ", ".join(map(str, candidates))
    )


def _ensure_dependencies(skip_install: bool) -> None:
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:  # pip always vendors packaging, including fresh Colab
        from pip._vendor.packaging.specifiers import SpecifierSet

    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "joblib": "joblib",
        "pytorch-tabnet": "pytorch_tabnet",
        "ydf": "ydf",
    }
    required_versions = {
        "numpy": ">=1.26,<3",
        "pandas": ">=2.1,<3",
        "scipy": ">=1.11,<2",
        "scikit-learn": ">=1.4,<2",
        "matplotlib": ">=3.8,<4",
        "joblib": ">=1.3,<2",
        "pytorch-tabnet": "==4.1.0",
        "ydf": "==0.16.1",
    }
    missing = [
        distribution
        for distribution, module in required.items()
        if importlib.util.find_spec(module) is None
    ]
    for distribution, specifier in required_versions.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if (
            installed is None
            or installed not in SpecifierSet(specifier)
        ) and distribution not in missing:
            missing.append(f"{distribution}{specifier}")
    if missing:
        if skip_install:
            raise ModuleNotFoundError(
                "Missing/incompatible dependencies: " + ", ".join(missing)
            )
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
    incompatible = []
    for distribution, specifier in required_versions.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed is None or installed not in SpecifierSet(specifier):
            incompatible.append(
                f"{distribution} installed={installed!r}, required={specifier}"
            )
    if incompatible:
        raise ModuleNotFoundError(
            "Dependency verification failed: " + "; ".join(incompatible)
        )
    if importlib.util.find_spec("torch") is None:
        raise ModuleNotFoundError(
            "PyTorch is required. Colab supplies an accelerator-matched build; "
            "this experiment intentionally does not replace it."
        )


def _resolve_output_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    override = os.environ.get("BALANCED_FUSION_OUTPUT_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if not DEFAULT_COLAB_RESULTS_ROOT.parent.is_dir():
        raise FileNotFoundError(
            "Google Drive is not mounted at /content/drive/MyDrive. Run "
            "base.ipynb Cell 1 first or pass --output-dir for a local smoke test."
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    return (DEFAULT_COLAB_RESULTS_ROOT / run_id).resolve()


def _write_status(output: Path, payload: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "LAUNCHER_STATUS.json").write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8"
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
    namespace = globals() if namespace is None else namespace
    # Never inherit a stale smoke environment from a reused Colab runtime.
    run_mode = (mode or "full").strip().lower()
    if run_mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")
    _ensure_dependencies(skip_install)
    import torch

    if run_mode == "full" and not torch.cuda.is_available():
        raise RuntimeError(
            "Full mode requires CUDA. Select an A100 Colab runtime. "
            "Only explicit --mode smoke supports CPU."
        )
    resolved_data = _resolve_data_root(namespace, data_root)
    output = _resolve_output_dir(output_dir)
    if output.exists():
        raise FileExistsError(f"Output directory must be new: {output}")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))
    started = time.monotonic()
    _write_status(
        output,
        {
            "status": "starting",
            "experiment": EXPERIMENT_NAME,
            "mode": run_mode,
            "smoke": run_mode == "smoke",
            "data_root": str(resolved_data),
            "output_dir": str(output),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "compute_note": (
                "TabNet and Transformer use A100; Google YDF and EDA use CPU."
            ),
        },
    )
    from SangHyo.Binary.Binary_Wearable_BalancedFusion_Google.train import (
        RunConfig,
        run_experiment,
    )

    if run_mode == "smoke":
        settings = {
            "outer_folds": 2,
            "outer_repeats": 1,
            "inner_folds": 2,
            "max_features": 8,
            "max_runtime_seconds": 1_800,
        }
    else:
        settings = {
            "outer_folds": 5,
            "outer_repeats": 2,
            "inner_folds": 3,
            "max_features": 24,
            "max_runtime_seconds": SOFT_RUNTIME_SECONDS,
        }
    config = RunConfig(
        training_root=str(resolved_data / "1.Training"),
        validation_root=str(resolved_data / "2.Validation"),
        output_dir=str(output),
        run_mode=run_mode,
        evaluate_historical_validation=evaluate_historical_validation,
        **settings,
    )
    try:
        result = run_experiment(config)
    except Exception as error:
        _write_status(
            output,
            {
                "status": "failed",
                "mode": run_mode,
                "smoke": run_mode == "smoke",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    final_report = output / "training" / "FINAL_REPORT.json"
    _write_status(
        output,
        {
            "status": "complete",
            "mode": run_mode,
            "smoke": run_mode == "smoke",
            "elapsed_seconds": time.monotonic() - started,
            "final_report": str(final_report),
            "results_are_on_google_drive": str(output).startswith(
                "/content/drive/MyDrive/"
            ),
        },
    )
    print(
        f"\nComplete ({run_mode}, smoke={run_mode == 'smoke'}). "
        f"Final report: {final_report}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument(
        "--skip-historical-validation",
        action="store_true",
        help="freeze predictions but do not open/evaluate historical Validation labels",
    )
    # base.ipynb executes this file with runpy inside ipykernel, whose own
    # ``-f <kernel.json>`` remains in sys.argv.  Ignore those notebook-owned
    # arguments while preserving this launcher's explicit options.
    arguments, _notebook_arguments = parser.parse_known_args()
    alarm_enabled = (
        arguments.mode.strip().lower() == "full"
        and hasattr(signal, "SIGALRM")
    )
    previous = None
    if alarm_enabled:
        previous = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(HARD_RUNTIME_SECONDS)
    try:
        run_pipeline(
            namespace=globals(),
            data_root=arguments.data_root,
            output_dir=arguments.output_dir,
            mode=arguments.mode,
            skip_install=arguments.skip_install,
            evaluate_historical_validation=not arguments.skip_historical_validation,
        )
    finally:
        if alarm_enabled:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    main()
