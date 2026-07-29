"""Strict CLI and unmodified ``base.ipynb`` entry point.

Training requires a literal acknowledgement token so code validation and data
audit can never start an expensive model fit accidentally.  The only automatic
training path is a strongly identified ``base.ipynb`` ``runpy`` invocation.
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
from typing import Any, Mapping

if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parent
    repository_root = package_root.parents[1]
    import_path = str(repository_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)
    __package__ = "SangHyo.Codex_Dementia_ROCAUC"


EXPERIMENT_NAME = "Codex_Dementia_ROCAUC"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_CORE = EXPERIMENT_ROOT / "requirements_core.txt"
REQUIREMENTS_COLAB = EXPERIMENT_ROOT / "requirements_colab.txt"
DEFAULT_RESULTS_ROOT = Path(
    f"/content/drive/MyDrive/{EXPERIMENT_NAME}_result"
)
TRAINING_ACKNOWLEDGEMENT = "I_UNDERSTAND_THIS_RUNS_TRAINING"

_CORE_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn",
    "optuna": "optuna",
    "matplotlib": "matplotlib",
    "cloudpickle": "cloudpickle",
    "joblib": "joblib",
}
_MODEL_MODULES = {
    "torch": "torch",
    "pytorch-tabnet": "pytorch_tabnet",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
}


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="Data")
    parser.add_argument("--output-dir", default="Codex_Dementia_ROCAUC_results")
    parser.add_argument("--profile", choices=("standard", "max"), default="standard")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-install", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CN+MCI versus Dem subject-level ROC-AUC pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate-code",
        help="syntax/synthetic split/leakage checks; executes zero model fits",
    )
    validate.add_argument(
        "--output",
        default=None,
        help="optional JSON path for the zero-fit validation report",
    )
    validate.add_argument("--skip-install", action="store_true")
    audit = subparsers.add_parser(
        "audit-data",
        help="read-only schema/ID audit; executes zero model fits",
    )
    _common(audit)
    make_splits = subparsers.add_parser(
        "make-splits",
        help="save the primary subject split registry; executes zero model fits",
    )
    _common(make_splits)
    train = subparsers.add_parser("train", help="execute the full training protocol")
    _common(train)
    train.add_argument(
        "--execute-training",
        required=True,
        metavar="ACK",
        help=f"must equal {TRAINING_ACKNOWLEDGEMENT}",
    )
    return parser


def _is_kernel_connection_file(value: str) -> bool:
    path = Path(str(value))
    return path.name.startswith("kernel-") and path.suffix == ".json"


def _without_jupyter_kernel_args(argv: list[str]) -> list[str]:
    """Remove only Jupyter's injected kernel connection-file arguments."""

    cleaned: list[str] = []
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if (
            token in {"-f", "--f"}
            and index + 1 < len(argv)
            and _is_kernel_connection_file(str(argv[index + 1]))
        ):
            index += 2
            continue
        if token.startswith(("-f=", "--f=")) and _is_kernel_connection_file(
            token.split("=", 1)[1]
        ):
            index += 1
            continue
        if _is_kernel_connection_file(token):
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned


def _is_unmodified_base_run(namespace: Mapping[str, Any]) -> bool:
    """Return whether ``base.ipynb`` is executing this exact file via runpy."""

    required = ("PROJECT_ROOT", "DATA_ROOT", "USER_ROOT", "RUN_PATH")
    if any(namespace.get(name) is None for name in required):
        return False
    try:
        run_path = Path(os.fspath(namespace["RUN_PATH"])).resolve()
        project_root = Path(os.fspath(namespace["PROJECT_ROOT"])).resolve()
        user_root = Path(os.fspath(namespace["USER_ROOT"])).resolve()
    except (OSError, TypeError, ValueError):
        return False
    allowed_user_roots = {
        (project_root / "SangHyo").resolve(),
        EXPERIMENT_ROOT,
    }
    return (
        run_path == Path(__file__).resolve()
        and (project_root / "base.ipynb").is_file()
        and EXPERIMENT_ROOT
        == (project_root / "SangHyo" / EXPERIMENT_NAME).resolve()
        and user_root in allowed_user_roots
    )


def _base_training_argv(namespace: Mapping[str, Any]) -> list[str]:
    """Build the explicit, safe standard-training argv for original base.ipynb."""

    data_root = Path(os.fspath(namespace["DATA_ROOT"])).expanduser().resolve()
    if not (data_root / "1.Training").is_dir() or not (
        data_root / "2.Validation"
    ).is_dir():
        raise FileNotFoundError(
            "Injected DATA_ROOT must contain 1.Training and 2.Validation: "
            f"{data_root}"
        )

    if not DEFAULT_RESULTS_ROOT.parent.is_dir():
        raise FileNotFoundError(
            "Google Drive is not mounted at /content/drive/MyDrive. "
            "Run original base.ipynb Cell 1 first."
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_utc")
    output = (DEFAULT_RESULTS_ROOT / run_id).resolve()

    argv = [
        "train",
        "--profile",
        "standard",
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output),
        "--device",
        "auto",
        "--n-jobs",
        "4",
        "--execute-training",
        TRAINING_ACKNOWLEDGEMENT,
    ]
    print(
        "[launcher] original base.ipynb detected; starting explicit "
        "standard training",
        flush=True,
    )
    print(f"[launcher] data_root  = {data_root}", flush=True)
    print(f"[launcher] output_dir = {output}", flush=True)
    return argv


def _ensure_dependencies(
    *,
    include_models: bool,
    skip_install: bool,
    enforce_requirements: bool = False,
) -> None:
    """Install missing declared dependencies before importing the pipeline."""

    required = dict(_CORE_MODULES)
    requirements_file = REQUIREMENTS_CORE
    if include_models:
        required.update(_MODEL_MODULES)
        requirements_file = REQUIREMENTS_COLAB

    missing = [
        distribution
        for distribution, module in required.items()
        if importlib.util.find_spec(module) is None
    ]
    if (missing or enforce_requirements) and skip_install:
        raise ModuleNotFoundError(
            "Dependency installation is required but --skip-install was set"
            + (": " + ", ".join(missing) if missing else "")
        )
    if missing or enforce_requirements:
        print(
            "[launcher] checking/installing declared dependencies"
            + (": " + ", ".join(missing) if missing else ""),
            flush=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                str(requirements_file),
            ],
            check=True,
        )
        importlib.invalidate_caches()

    failures: dict[str, str] = {}
    for distribution, module in required.items():
        try:
            importlib.import_module(module)
        except Exception as error:  # import health preflight, not model fallback
            failures[distribution] = f"{type(error).__name__}: {error}"
    if failures:
        details = "; ".join(
            f"{name} ({message})" for name, message in failures.items()
        )
        raise ModuleNotFoundError(
            "Dependency import preflight failed after installation check: "
            + details
        )


def _config(args) -> object:
    from .config import make_config

    return make_config(
        profile=args.profile,
        data_root=args.data_root,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        device=args.device,
    )


def main(
    argv: list[str] | None = None,
    *,
    namespace: Mapping[str, Any] | None = None,
) -> int:
    namespace = globals() if namespace is None else namespace
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    cleaned_argv = _without_jupyter_kernel_args(raw_argv)
    base_launch = _is_unmodified_base_run(namespace)
    if base_launch:
        cleaned_argv = _base_training_argv(namespace)

    parser = build_parser()
    args = parser.parse_args(cleaned_argv)
    if (
        args.command == "train"
        and args.execute_training != TRAINING_ACKNOWLEDGEMENT
    ):
        parser.error(
            "--execute-training must exactly equal "
            f"{TRAINING_ACKNOWLEDGEMENT}"
        )
    _ensure_dependencies(
        include_models=args.command == "train",
        skip_install=bool(args.skip_install),
        enforce_requirements=base_launch,
    )

    if args.command == "validate-code":
        from .artifacts import write_json
        from .validation import validate_code_without_fit

        report = validate_code_without_fit(Path(__file__).resolve().parent)
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "audit-data":
        from .artifacts import write_json
        from .validation import audit_data_without_training

        config = _config(args)
        report = audit_data_without_training(config)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "DATA_AUDIT_ZERO_FIT.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "make-splits":
        from .artifacts import write_json
        from .data import load_development_cohort
        from .splits import (
            build_repeated_group_plan,
            save_split_plan,
        )

        config = _config(args)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        reports = {}
        for track in config.data.tracks:
            cohort = load_development_cohort(config.data, track)
            plan = build_repeated_group_plan(
                cohort.y,
                cohort.groups,
                n_splits=config.cv.outer_folds,
                n_repeats=config.cv.outer_repeats,
                seed=config.cv.seed,
                minimum_positive_validation=(
                    config.cv.minimum_positive_per_validation_fold
                ),
                layer=f"{track}_outer",
            )
            path = output / track / "split_registry_outer.json"
            save_split_plan(
                path,
                plan,
                subject_ids=cohort.subject_ids,
                y=cohort.y,
            )
            reports[track] = str(path.resolve())
        write_json(
            output / "SPLIT_PLAN_ZERO_FIT.json",
            {"training_calls_executed": 0, "tracks": reports},
        )
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "train":
        config = _config(args)
        output = config.runtime.resolved_output()
        if output.exists() and any(output.iterdir()):
            parser.error(f"refusing to train into non-empty output directory: {output}")
        from .train import run_experiment

        run_experiment(config)
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    main(namespace=globals())
