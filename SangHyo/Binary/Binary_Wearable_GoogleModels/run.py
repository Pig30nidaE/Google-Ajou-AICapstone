"""Single-file launcher for the binary CN vs. MCI+DEM experiment.

``base.ipynb`` executes this file with ``runpy`` and injects repository paths
as globals.  The launcher deliberately uses only the Python standard library
until the experiment requirements have been checked (and, if necessary,
installed).  It then imports and runs the complete EDA -> training -> frozen
label-free validation prediction -> one-time validation evaluation pipeline.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Iterator, Mapping, Sequence


EXPERIMENT_NAME = "Binary_Wearable_GoogleModels"
REQUIREMENTS_FILE = Path(__file__).resolve().with_name("requirements_colab.txt")

# The training code receives a 5 h 45 min soft budget, leaving 15 minutes for
# model/report finalization.  SIGALRM provides a six-hour launcher guard on
# Linux/macOS (including Colab).
FULL_SOFT_RUNTIME_SECONDS = 20_700
SMOKE_SOFT_RUNTIME_SECONDS = 1_800
HARD_RUNTIME_SECONDS = 21_600

_IMPORT_NAMES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "joblib": "joblib",
    "ydf": "ydf",
    "pytorch-tabnet": "pytorch_tabnet",
}
_VERSION_BOUNDS = {
    "numpy": ((1, 26), (3, 0)),
    "pandas": ((2, 1), (4, 0)),
    "scipy": ((1, 11), (2, 0)),
    "scikit-learn": ((1, 5), (2, 0)),
    "matplotlib": ((3, 8), (4, 0)),
    "joblib": ((1, 3), (2, 0)),
    "pytorch-tabnet": ((4, 1), (5, 0)),
}


class HardDeadlineExceeded(TimeoutError):
    """Raised when the complete launcher reaches its six-hour ceiling."""


def _path(value: object | None, *, name: str) -> Path | None:
    """Convert a notebook/CLI path value without depending on its input type."""

    if value is None or value == "":
        return None
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be path-like, got {type(value).__name__}") from exc
    return Path(raw).expanduser().resolve()


def _has_data_splits(path: Path) -> bool:
    return (path / "1.Training").is_dir() and (path / "2.Validation").is_dir()


def _resolve_data_root(
    *,
    cli_value: str | None,
    namespace: Mapping[str, Any],
    project_root: Path,
) -> Path:
    explicit = _path(cli_value, name="--data-root")
    if explicit is not None:
        if not _has_data_splits(explicit):
            raise FileNotFoundError(
                f"--data-root must contain 1.Training and 2.Validation: {explicit}"
            )
        return explicit

    candidates = [
        _path(namespace.get("DATA_ROOT"), name="DATA_ROOT"),
        _path(os.environ.get("BINARY_DATA_ROOT"), name="BINARY_DATA_ROOT"),
        project_root / "Data",
        Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
        Path("/content/drive/MyDrive/GoogleAI_contest/Data"),
    ]
    for candidate in candidates:
        if candidate is not None and _has_data_splits(candidate):
            return candidate.resolve()
    checked = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "Could not find a data root containing both 1.Training and 2.Validation. "
        f"Checked: {checked}"
    )


def _resolve_notebook_paths(
    namespace: Mapping[str, Any], cli_data_root: str | None
) -> dict[str, Path]:
    """Resolve globals injected by base.ipynb, with safe local fallbacks."""

    experiment_root = Path(__file__).resolve().parent

    injected_project = _path(namespace.get("PROJECT_ROOT"), name="PROJECT_ROOT")
    project_root = (
        injected_project
        if injected_project is not None and injected_project.is_dir()
        else experiment_root.parents[1]
    ).resolve()

    injected_user = _path(namespace.get("USER_ROOT"), name="USER_ROOT")
    user_root = (
        injected_user
        if injected_user is not None and injected_user.is_dir()
        else experiment_root.parent
    ).resolve()

    injected_run = _path(namespace.get("RUN_PATH"), name="RUN_PATH")
    run_path = (
        injected_run
        if injected_run is not None and injected_run.is_file()
        else Path(__file__).resolve()
    )

    data_root = _resolve_data_root(
        cli_value=cli_data_root,
        namespace=namespace,
        project_root=project_root,
    )
    return {
        "project_root": project_root,
        "data_root": data_root,
        "user_root": user_root,
        "experiment_root": experiment_root,
        "run_path": run_path,
    }


def _distribution_name(requirement: str) -> str:
    for marker in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
        requirement = requirement.split(marker, 1)[0]
    return requirement.strip().lower()


def _requirement_specs() -> list[str]:
    if not REQUIREMENTS_FILE.is_file():
        raise FileNotFoundError(f"Requirements file is missing: {REQUIREMENTS_FILE}")
    specs = []
    for raw_line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            specs.append(line)
    return specs


def _numeric_version(version: str, width: int = 4) -> tuple[int, ...]:
    parts = [int(value) for value in re.findall(r"\d+", version.split("+", 1)[0])]
    return tuple((parts + [0] * width)[:width])


def _version_in_bounds(distribution: str, version: str) -> bool:
    if distribution == "ydf":
        return version == "0.16.1"
    bounds = _VERSION_BOUNDS.get(distribution)
    if bounds is None:
        return True
    observed = _numeric_version(version)
    lower = tuple((*bounds[0], *([0] * (len(observed) - len(bounds[0])))))
    upper = tuple((*bounds[1], *([0] * (len(observed) - len(bounds[1])))))
    return lower <= observed < upper


def _missing_requirements(*, mode: str) -> list[str]:
    """Return absent requirements; also enforce the YDF compatibility pin."""

    missing = []
    for spec in _requirement_specs():
        distribution = _distribution_name(spec)
        if mode == "smoke" and distribution == "pytorch-tabnet":
            # Smoke mode exercises the complete data/CV/freeze path with the
            # two CPU YDF models and deliberately avoids a large Torch install.
            continue
        import_name = _IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))
        try:
            found = importlib.util.find_spec(import_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            found = False
        if not found:
            missing.append(spec)
            continue
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(spec)
            continue
        if not _version_in_bounds(distribution, installed_version):
            missing.append(spec)
            continue
        if (
            mode == "full"
            and distribution == "pytorch-tabnet"
            and importlib.util.find_spec("torch") is None
        ):
            # Re-requesting pytorch-tabnet makes pip repair its missing Torch
            # dependency without pinning an accelerator-incompatible build.
            missing.append(spec)
    return missing


def ensure_requirements(*, skip_install: bool, mode: str) -> None:
    """Install the requirements file only when an import is actually absent."""

    missing = _missing_requirements(mode=mode)
    if not missing:
        print("Dependencies : already satisfied (pip skipped)")
        return

    rendered = ", ".join(missing)
    if skip_install:
        raise ModuleNotFoundError(
            f"Missing/incompatible dependencies with --skip-install: {rendered}"
        )

    print(f"Dependencies : installing because these are missing/incompatible: {rendered}")
    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    if mode == "full":
        install_command += ["-r", str(REQUIREMENTS_FILE)]
    else:
        install_command += missing
    subprocess.run(install_command, check=True)
    importlib.invalidate_caches()
    still_missing = _missing_requirements(mode=mode)
    if still_missing:
        raise ModuleNotFoundError(
            "Dependency installation completed but imports are still unavailable: "
            + ", ".join(still_missing)
        )


def _run_mode(args: argparse.Namespace) -> str:
    mode = os.environ.get("BINARY_RUN_MODE", "full").strip().lower()
    if args.smoke:
        mode = "smoke"
    if mode not in {"full", "smoke"}:
        raise ValueError("BINARY_RUN_MODE must be either 'full' or 'smoke'.")
    return mode


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _prepare_output_dir(cli_value: str | None, experiment_root: Path) -> Path:
    if cli_value:
        output_dir = Path(cli_value).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"--output-dir must be new or empty to avoid overwriting results: {output_dir}"
            )
        return output_dir

    local_root = experiment_root / "outputs"
    drive_root = Path("/content/drive/MyDrive/Binary_Wearable_GoogleModels_Results")
    results_root = drive_root if Path("/content/drive/MyDrive").is_dir() else local_root
    try:
        results_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if results_root == local_root:
            raise
        print(f"Drive output unavailable ({exc}); falling back to {local_root}")
        results_root = local_root
        results_root.mkdir(parents=True, exist_ok=True)

    base_name = _timestamp()
    for suffix in range(1_000):
        name = base_name if suffix == 0 else f"{base_name}_{suffix:02d}"
        output_dir = results_root / name
        try:
            output_dir.mkdir(parents=False, exist_ok=False)
            return output_dir.resolve()
        except FileExistsError:
            continue
    raise FileExistsError(f"Could not allocate a unique output directory under {results_root}")


def _git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unavailable"


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in _IMPORT_NAMES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    try:
        versions["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        versions["torch"] = "not-installed"
    return versions


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _environment_payload(
    *, paths: Mapping[str, Path], output_dir: Path, mode: str
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "started_at": datetime.now().astimezone().isoformat(),
        "git_commit": _git_commit(paths["project_root"]),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "working_directory": str(Path.cwd().resolve()),
        "project_root": str(paths["project_root"]),
        "data_root": str(paths["data_root"]),
        "user_root": str(paths["user_root"]),
        "experiment_root": str(paths["experiment_root"]),
        "run_path": str(paths["run_path"]),
        "output_dir": str(output_dir),
        "run_mode": mode,
        "soft_runtime_seconds": (
            SMOKE_SOFT_RUNTIME_SECONDS if mode == "smoke" else FULL_SOFT_RUNTIME_SECONDS
        ),
        "hard_runtime_seconds": HARD_RUNTIME_SECONDS,
        "package_versions": _installed_versions(),
    }


def _print_environment(payload: Mapping[str, Any]) -> None:
    print("\n=== Binary wearable experiment environment ===")
    for key in (
        "git_commit",
        "python",
        "platform",
        "project_root",
        "data_root",
        "user_root",
        "run_path",
        "run_mode",
        "output_dir",
        "soft_runtime_seconds",
        "hard_runtime_seconds",
    ):
        value = str(payload[key]).replace("\n", " ")
        print(f"{key:24}: {value}")
    print("Pipeline                 : EDA -> training -> freeze predictions -> one evaluation")
    print("================================================\n")


@contextmanager
def _hard_deadline(seconds: int) -> Iterator[None]:
    """Enforce a wall-clock ceiling where SIGALRM is safely available."""

    supported = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not supported:
        print("Warning: OS hard-deadline guard is unavailable on this platform/thread.")
        yield
        return

    def _raise_timeout(signum: int, frame: object) -> None:
        del signum, frame
        raise HardDeadlineExceeded(
            f"The launcher exceeded its hard {seconds}-second (6-hour) limit."
        )

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            elapsed = time.monotonic() - started
            remaining = max(0.001, previous_timer[0] - elapsed)
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small CPU-friendly smoke test (also available via BINARY_RUN_MODE=smoke).",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Never invoke pip; fail clearly if an experiment dependency is missing.",
    )
    parser.add_argument(
        "--output-dir",
        help="Exact new/empty output directory (default: timestamped local or Drive folder).",
    )
    parser.add_argument(
        "--data-root",
        help="Directory containing 1.Training and 2.Validation.",
    )
    return parser


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        # Jupyter/Colab kernel flags remain in sys.argv when base.ipynb uses
        # runpy.run_path.  They are unrelated to this experiment.
        print(f"Ignoring notebook/host arguments: {' '.join(unknown)}")
    return args


def main(
    namespace: Mapping[str, Any] | None = None,
    argv: Sequence[str] | None = None,
) -> Path:
    """Resolve the notebook context and execute the complete binary pipeline."""

    values = dict(namespace or {})
    args = _parse_args(argv)
    mode = _run_mode(args)
    paths = _resolve_notebook_paths(values, args.data_root)
    output_dir = _prepare_output_dir(args.output_dir, paths["experiment_root"])
    started = time.monotonic()

    try:
        with _hard_deadline(HARD_RUNTIME_SECONDS):
            ensure_requirements(skip_install=args.skip_install, mode=mode)

            # base.ipynb uses runpy rather than a normal module import, so its
            # execution directory is not guaranteed to be at sys.path[0].
            project_path = str(paths["project_root"])
            if project_path in sys.path:
                sys.path.remove(project_path)
            sys.path.insert(0, project_path)

            # Heavy third-party imports begin inside train.py only after the
            # dependency gate above has completed.
            from SangHyo.Binary.Binary_Wearable_GoogleModels.train import (
                RunConfig,
                run_pipeline,
            )

            environment = _environment_payload(
                paths=paths,
                output_dir=output_dir,
                mode=mode,
            )
            _write_json(output_dir / "launcher_environment.json", environment)
            _print_environment(environment)

            if mode == "smoke":
                config = RunConfig(
                    training_root=str(paths["data_root"] / "1.Training"),
                    validation_root=str(paths["data_root"] / "2.Validation"),
                    output_dir=str(output_dir),
                    run_mode="smoke",
                    seed=20260722,
                    outer_folds=3,
                    outer_repeats=1,
                    inner_folds=2,
                    max_features=24,
                    max_runtime_seconds=SMOKE_SOFT_RUNTIME_SECONDS,
                    include_neural=False,
                )
            else:
                config = RunConfig(
                    training_root=str(paths["data_root"] / "1.Training"),
                    validation_root=str(paths["data_root"] / "2.Validation"),
                    output_dir=str(output_dir),
                    run_mode="full",
                    seed=20260722,
                    outer_folds=5,
                    outer_repeats=2,
                    inner_folds=3,
                    max_features=24,
                    max_runtime_seconds=FULL_SOFT_RUNTIME_SECONDS,
                    include_neural=True,
                )

            result = run_pipeline(config)

        elapsed = time.monotonic() - started
        _write_json(
            output_dir / "LAUNCHER_COMPLETE.json",
            {
                "success": True,
                "completed_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": elapsed,
                "output_dir": str(output_dir),
                "pipeline_result": result,
            },
        )
        print(f"Completed in {elapsed / 60.0:.1f} minutes. Results: {output_dir}")
        return output_dir
    except BaseException as exc:
        _write_json(
            output_dir / "LAUNCHER_FAILED.json",
            {
                "success": False,
                "failed_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "output_dir": str(output_dir),
            },
        )
        raise


if __name__ == "__main__":
    main(globals())
