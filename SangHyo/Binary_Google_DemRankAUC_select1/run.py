"""Colab / base.ipynb launcher.

    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Binary_Google_DemRankAUC/run.py"

Runtime: **CPU (High-RAM)** is the right choice.  Every headline model is a
tree ensemble or a regularised linear model; only ``--sequence-arm`` and TabNet
use a GPU, and neither is in the default candidate set.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "SangHyo.Binary_Google_DemRankAUC"

EXPERIMENT_NAME = "Binary_Google_DemRankAUC"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
DEFAULT_RESULTS_ROOT = Path(f"/content/drive/MyDrive/{EXPERIMENT_NAME}_result")

REQUIRED_MODULES = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
                    "scikit-learn": "sklearn", "joblib": "joblib", "matplotlib": "matplotlib"}


def _write_status(output: Path, payload: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "LAUNCHER_STATUS.json").write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _resolve_data_root(namespace: Mapping[str, Any], explicit: str | None) -> Path:
    candidates: list[Path] = []
    for value in (explicit, namespace.get("DATA_ROOT"),
                  os.environ.get("DEMRANKAUC_DATA_ROOT")):
        if value:
            candidates.append(Path(os.fspath(value)).expanduser())
    project_root = namespace.get("PROJECT_ROOT")
    if project_root:
        candidates.append(Path(os.fspath(project_root)) / "Data")
    candidates += [
        EXPERIMENT_ROOT.parents[1] / "Data",
        Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
        Path("/content/drive/MyDrive/GoogleAI_contest/Data"),
    ]
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if (resolved / "1.Training").is_dir() and (resolved / "2.Validation").is_dir():
            return resolved
    raise FileNotFoundError(
        "Data root with 1.Training and 2.Validation was not found. Checked: "
        + ", ".join(map(str, candidates))
    )


def _resolve_output_dir(explicit: str | None) -> Path:
    if explicit:
        output = Path(explicit).expanduser().resolve()
    elif os.environ.get("DEMRANKAUC_OUTPUT_DIR"):
        output = Path(os.environ["DEMRANKAUC_OUTPUT_DIR"]).expanduser().resolve()
    else:
        if not DEFAULT_RESULTS_ROOT.parent.is_dir():
            raise FileNotFoundError(
                "Google Drive is not mounted at /content/drive/MyDrive. Mount Drive "
                "or pass --output-dir."
            )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
        output = (DEFAULT_RESULTS_ROOT / run_id).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {output}")
    return output


def _ensure_dependencies(*, skip_install: bool) -> list[str]:
    missing = [dist for dist, module in REQUIRED_MODULES.items()
               if importlib.util.find_spec(module) is None]
    if not missing:
        return []
    if skip_install:
        raise ModuleNotFoundError("Missing dependencies: " + ", ".join(missing))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "-r", str(REQUIREMENTS_FILE)],
        check=True,
    )
    importlib.invalidate_caches()
    remaining = [dist for dist, module in REQUIRED_MODULES.items()
                 if importlib.util.find_spec(module) is None]
    if remaining:
        raise ModuleNotFoundError("Install completed but modules remain missing: "
                                  + ", ".join(remaining))
    return missing


def run_pipeline(*, namespace: Mapping[str, Any] | None = None, data_root: str | None = None,
                 output_dir: str | None = None, profile: str = "full",
                 tracks: str | None = None, models: str | None = None,
                 cohort: str = "both", resampler: str = "class_weight",
                 sequence_arm: bool = False, no_tune: bool = False,
                 drop_suspect: bool = False, skip_install: bool = False,
                 hours: float = 6.0) -> dict[str, Any]:
    namespace = globals() if namespace is None else namespace
    output = _resolve_output_dir(output_dir)
    started = time.monotonic()
    _write_status(output, {"status": "starting", "experiment": EXPERIMENT_NAME,
                           "profile": profile, "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        installed = _ensure_dependencies(skip_install=skip_install)
        resolved_data = _resolve_data_root(namespace, data_root)

        from .config import DEFAULT_TRACKS, RunConfig
        from .models import environment_report

        _write_status(output, {
            "status": "running", "experiment": EXPERIMENT_NAME, "profile": profile,
            "data_root": str(resolved_data), "output_dir": str(output),
            "installed_now": installed, "environment": environment_report(),
        })

        config = RunConfig(
            data_root=str(resolved_data),
            output_dir=str(output),
            profile=profile,
            tracks=tuple(t.strip() for t in tracks.split(",")) if tracks else DEFAULT_TRACKS,
            models=tuple(m.strip() for m in models.split(",")) if models else None,
            cohort=cohort,
            drop_suspect=drop_suspect,
            resamplers=(resampler,),
            use_sequence_arm=sequence_arm,
            tune=not no_tune,
            hard_runtime_seconds=int(hours * 3600),
        )
        from .train import run_experiment

        result = run_experiment(config)
    except Exception as error:
        _write_status(output, {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": float(time.monotonic() - started),
        })
        raise
    _write_status(output, {
        "status": "complete",
        "elapsed_seconds": float(time.monotonic() - started),
        "final_report": str(Path(output) / "training" / "FINAL_REPORT.json"),
        "headline_roc_auc": result.get("headline", {}).get("roc_auc_mean"),
    })
    print("Complete:", Path(output) / "training" / "FINAL_REPORT.json")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe CN+MCI vs Dem subject-level ROC-AUC pipeline."
    )
    parser.add_argument("--profile", choices=("smoke", "standard", "full", "max"), default="full")
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--tracks", help="comma-separated feature blocks")
    parser.add_argument("--models", help="comma-separated model names")
    parser.add_argument("--cohort", choices=("full", "filtered", "both"), default="both")
    parser.add_argument("--resampler", default="class_weight")
    parser.add_argument("--sequence-arm", action="store_true",
                        help="enable the TSMixer daily-sequence arm (needs torch)")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--drop-suspect", action="store_true",
                        help="drop wear-time/coverage features (adherence-artifact ablation)")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--hours", type=float, default=6.0)
    return parser


def _without_jupyter_kernel_args(argv: list[str]) -> list[str]:
    """Strip only Jupyter's injected ``-f <kernel-*.json>``.

    ``base.ipynb`` runs this file with ``runpy.run_path``, so the notebook
    kernel's own arguments are still in ``sys.argv``.  Genuinely unknown user
    arguments must still fail, so ``parse_known_args`` is deliberately not used.
    """

    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if token in {"-f", "--f"} and index + 1 < len(argv):
            connection = Path(str(argv[index + 1]))
            if connection.name.startswith("kernel-") and connection.suffix == ".json":
                index += 2
                continue
        if token.startswith(("-f=", "--f=")):
            connection = Path(token.split("=", 1)[1])
            if connection.name.startswith("kernel-") and connection.suffix == ".json":
                index += 1
                continue
        cleaned.append(token)
        index += 1
    return cleaned


def main(argv: list[str] | None = None) -> dict[str, Any]:
    raw = list(sys.argv[1:] if argv is None else argv)
    arguments = _parser().parse_args(_without_jupyter_kernel_args(raw))
    return run_pipeline(
        data_root=arguments.data_root, output_dir=arguments.output_dir,
        profile=arguments.profile, tracks=arguments.tracks, models=arguments.models,
        cohort=arguments.cohort, resampler=arguments.resampler,
        sequence_arm=arguments.sequence_arm, no_tune=arguments.no_tune,
        drop_suspect=arguments.drop_suspect, skip_install=arguments.skip_install,
        hours=arguments.hours,
    )


if __name__ == "__main__":
    main()
