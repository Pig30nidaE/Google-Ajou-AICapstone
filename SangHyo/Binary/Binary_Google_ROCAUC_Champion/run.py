"""Colab/base.ipynb launcher for the two-track ROC-AUC champion protocol.

There is no smoke-performance mode.  Static contracts are covered by tests;
the only training profiles are the predeclared ``default`` and ``max`` runs.
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

if __package__ in (None, ""):
    # Add the repository root so absolute ``SangHyo.*`` imports resolve.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "SangHyo.Binary_Google_ROCAUC_Champion"

EXPERIMENT_NAME = "Binary_Google_ROCAUC_Champion"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
DEFAULT_RESULTS_ROOT = Path(f"/content/drive/MyDrive/{EXPERIMENT_NAME}_result")
HARD_RUNTIME_SECONDS = 6 * 60 * 60


class HardDeadlineExceeded(TimeoutError):
    """Raised by the launcher before a run can silently exceed six hours."""


def _alarm_handler(signum, frame):  # pragma: no cover - Colab/Linux signal path
    del signum, frame
    raise HardDeadlineExceeded("The predeclared six-hour hard limit was reached")


def _write_status(output: Path, payload: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "LAUNCHER_STATUS.json").write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_data_root(
    namespace: Mapping[str, Any],
    explicit: str | None,
) -> Path:
    candidates: list[Path] = []
    for value in (
        explicit,
        namespace.get("DATA_ROOT"),
        os.environ.get("BINARY_ROCAUC_DATA_ROOT"),
    ):
        if value:
            candidates.append(Path(os.fspath(value)).expanduser())
    project_root = namespace.get("PROJECT_ROOT")
    if project_root:
        candidates.append(Path(os.fspath(project_root)) / "Data")
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
        "Data root with 1.Training and 2.Validation was not found. Checked: "
        + ", ".join(map(str, candidates))
    )


def _resolve_output_dir(explicit: str | None) -> Path:
    if explicit:
        output = Path(explicit).expanduser().resolve()
    elif os.environ.get("BINARY_ROCAUC_OUTPUT_DIR"):
        output = Path(os.environ["BINARY_ROCAUC_OUTPUT_DIR"]).expanduser().resolve()
    else:
        if not DEFAULT_RESULTS_ROOT.parent.is_dir():
            raise FileNotFoundError(
                "Google Drive is not mounted at /content/drive/MyDrive. Mount "
                "Drive or pass --output-dir."
            )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
        output = (DEFAULT_RESULTS_ROOT / run_id).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {output}"
        )
    return output


def _ensure_dependencies(*, skip_install: bool, require_tabpfn: bool) -> None:
    modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "joblib": "joblib",
        "catboost": "catboost",
        "torch": "torch",
    }
    if require_tabpfn:
        modules["tabpfn"] = "tabpfn"
    missing = [
        distribution
        for distribution, module in modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return
    if skip_install:
        raise ModuleNotFoundError("Missing dependencies: " + ", ".join(missing))
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
    remaining = [
        distribution
        for distribution, module in modules.items()
        if importlib.util.find_spec(module) is None
    ]
    if remaining:
        raise ModuleNotFoundError(
            "Dependency installation completed but modules remain missing: "
            + ", ".join(remaining)
        )


def _resolve_tabpfn(mode: str) -> tuple[bool, str]:
    value = str(mode).lower()
    if value == "off":
        return False, "explicitly disabled"
    if value == "on":
        return True, "explicitly enabled; pinned v2.6 synthetic checkpoint"
    if value != "auto":
        raise ValueError("tabpfn must be auto, on, or off")
    enabled = bool(os.environ.get("TABPFN_TOKEN"))
    reason = (
        "enabled because TABPFN_TOKEN is present"
        if enabled
        else "skipped because TABPFN_TOKEN is absent; use --tabpfn on after authentication"
    )
    return enabled, reason


def run_pipeline(
    *,
    namespace: Mapping[str, Any] | None = None,
    data_root: str | None = None,
    output_dir: str | None = None,
    profile: str = "default",
    tabpfn: str = "auto",
    skip_install: bool = False,
    allow_cpu: bool = False,
) -> dict[str, Any]:
    namespace = globals() if namespace is None else namespace
    include_tabpfn, tabpfn_reason = _resolve_tabpfn(tabpfn)
    output = _resolve_output_dir(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _write_status(
        output,
        {
            "status": "starting",
            "experiment": EXPERIMENT_NAME,
            "profile": profile,
            "tabpfn": {
                "requested": tabpfn,
                "included": include_tabpfn,
                "reason": tabpfn_reason,
            },
        },
    )
    try:
        _ensure_dependencies(
            skip_install=skip_install, require_tabpfn=include_tabpfn
        )
        import torch

        if not torch.cuda.is_available() and not allow_cpu:
            raise RuntimeError(
                "A CUDA runtime is required for the full Sequence Transformer "
                "protocol. Select a Colab A100 GPU, or pass --allow-cpu only "
                "if you accept a potentially impractical runtime."
            )
        resolved_data = _resolve_data_root(namespace, data_root)
        _write_status(
            output,
            {
                "status": "running",
                "experiment": EXPERIMENT_NAME,
                "profile": profile,
                "data_root": str(resolved_data),
                "output_dir": str(output),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
                "tabpfn": {
                    "requested": tabpfn,
                    "included": include_tabpfn,
                    "reason": tabpfn_reason,
                    "model_version": "v2.6 synthetic" if include_tabpfn else None,
                },
                "hard_runtime_seconds": HARD_RUNTIME_SECONDS,
            },
        )
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(HARD_RUNTIME_SECONDS)
        # Import only after dependency installation; this keeps ``--help`` and
        # a fresh Colab launcher usable before scikit-learn is installed.
        from .train import RunConfig, run_experiment

        result = run_experiment(
            RunConfig(
                data_root=str(resolved_data),
                output_dir=str(output),
                profile=profile,
                include_tabpfn=include_tabpfn,
                hard_runtime_seconds=HARD_RUNTIME_SECONDS,
            )
        )
    except Exception as error:
        _write_status(
            output,
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": float(time.monotonic() - started),
                "validation_score_must_not_drive_changes": True,
            },
        )
        raise
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    _write_status(
        output,
        {
            "status": "complete",
            "elapsed_seconds": float(time.monotonic() - started),
            "final_report": str(output / "training" / "FINAL_REPORT.json"),
        },
    )
    print("Complete:", output / "training" / "FINAL_REPORT.json")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the leakage-safe MMSE and wearable ROC-AUC protocols. "
            "This launcher performs real training only."
        )
    )
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument("--profile", choices=("default", "max"), default="default")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--tabpfn", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def _without_jupyter_kernel_args(argv: list[str]) -> list[str]:
    """Remove only Jupyter's injected ``-f <kernel-*.json>`` argument.

    ``base.ipynb`` executes this file with :func:`runpy.run_path`, so the
    notebook kernel's own command-line arguments remain in ``sys.argv``.
    Unknown user arguments must still fail; therefore ``parse_known_args`` is
    intentionally not used.
    """

    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if token in {"-f", "--f"} and index + 1 < len(argv):
            connection_file = Path(str(argv[index + 1]))
            if (
                connection_file.name.startswith("kernel-")
                and connection_file.suffix == ".json"
            ):
                index += 2
                continue
        if token.startswith(("-f=", "--f=")):
            connection_file = Path(token.split("=", 1)[1])
            if (
                connection_file.name.startswith("kernel-")
                and connection_file.suffix == ".json"
            ):
                index += 1
                continue
        cleaned.append(token)
        index += 1
    return cleaned


def main(argv: list[str] | None = None) -> dict[str, Any]:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    arguments = _parser().parse_args(_without_jupyter_kernel_args(raw_argv))
    return run_pipeline(
        data_root=arguments.data_root,
        output_dir=arguments.output_dir,
        profile=arguments.profile,
        tabpfn=arguments.tabpfn,
        skip_install=arguments.skip_install,
        allow_cpu=arguments.allow_cpu,
    )


if __name__ == "__main__":
    main()
