"""Required entry point for Colab ``base.ipynb`` and shell execution.

This file is intentionally the only launcher.  It adapts to the globals that
``base.ipynb`` injects and reads notebook arguments from ``GFPP_ARGS`` because
Jupyter owns ``sys.argv``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    _experiment_root = Path(__file__).resolve().parent
    _repository_root = _experiment_root.parents[1]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))
    __package__ = f"SangHyo.{_experiment_root.name}"

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
ARGS_ENV_VARIABLE = "GFPP_ARGS"
STAGE_CHOICES = ("all", "inspect", "program", "train")

_BASE_DISTRIBUTIONS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
    "joblib": "joblib",
}
_API_DISTRIBUTIONS = {"google-genai": "google.genai"}
_VERSION_BOUNDS = {
    "numpy": ((1, 26), (3, 0)),
    "pandas": ((2, 1), (3, 0)),
    "scipy": ((1, 11), (2, 0)),
    "scikit-learn": ((1, 4), (2, 0)),
    "PyYAML": ((6, 0), (7, 0)),
    "joblib": ((1, 3), (2, 0)),
    "google-genai": ((1, 68), (2, 0)),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Global Gemma feature-program + nested subject-level "
            "CN versus MCI+Dem evaluation"
        ),
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--stage", choices=STAGE_CHOICES, default="all")
    parser.add_argument("--profile", choices=("smoke", "standard", "full"), default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--gemma-model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="program stage must use the exact cached program",
    )
    parser.add_argument(
        "--regenerate-program",
        action="store_true",
        help="explicitly replace the exact program cache identity",
    )
    parser.add_argument("--skip-install", action="store_true")
    return parser


def _is_kernel_file(value: str) -> bool:
    path = Path(str(value))
    return path.name.startswith("kernel-") and path.suffix == ".json"


def strip_jupyter_arguments(argv: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        token = str(values[index])
        if token in {"-f", "--f"} and index + 1 < len(values) and _is_kernel_file(values[index + 1]):
            index += 2
            continue
        if token.startswith(("-f=", "--f=")) and _is_kernel_file(token.split("=", 1)[1]):
            index += 1
            continue
        if _is_kernel_file(token):
            index += 1
            continue
        cleaned.append(token)
        index += 1
    return cleaned


def is_base_notebook_launch(namespace: Mapping[str, Any]) -> bool:
    required = ("PROJECT_ROOT", "DATA_ROOT", "USER_ROOT", "RUN_PATH")
    if any(namespace.get(name) is None for name in required):
        return False
    try:
        return Path(os.fspath(namespace["RUN_PATH"])).resolve() == Path(__file__).resolve()
    except (OSError, TypeError, ValueError):
        return False


def notebook_argv(environ: Mapping[str, str]) -> list[str]:
    raw = str(environ.get(ARGS_ENV_VARIABLE, "")).strip()
    return shlex.split(raw) if raw else ["--stage", "all", "--profile", "standard"]


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _distribution_version_acceptable(distribution: str) -> bool:
    bounds = _VERSION_BOUNDS.get(distribution)
    if bounds is None:
        return True
    try:
        raw = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    numeric = tuple(int(piece) for piece in re.findall(r"\d+", raw)[:3])
    if not numeric:
        return False
    lower, upper = bounds
    return numeric >= lower and numeric < upper


def _requirement_lines() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = line
        for marker in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            name = name.split(marker, 1)[0]
        mapping[name.lower()] = line
    return mapping


def ensure_dependencies(*, include_api: bool, skip_install: bool) -> None:
    required = dict(_BASE_DISTRIBUTIONS)
    if include_api:
        required.update(_API_DISTRIBUTIONS)
    missing = [
        distribution
        for distribution, module in required.items()
        if not _module_available(module)
        or not _distribution_version_acceptable(distribution)
    ]
    if missing and skip_install:
        raise ModuleNotFoundError(
            "--skip-install was set but dependencies are missing: " + ", ".join(missing)
        )
    if missing:
        declared = _requirement_lines()
        targets = [declared.get(name.lower(), name) for name in missing]
        print(f"[launcher] installing/upgrading: {', '.join(missing)}", flush=True)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                *targets,
            ],
            capture_output=True,
            text=True,
        )
        if completed.stdout:
            print(completed.stdout, flush=True)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr, flush=True)
        if completed.returncode != 0:
            tail = "\n".join(completed.stderr.strip().splitlines()[-25:])
            raise RuntimeError(f"Dependency installation failed:\n{tail}")
        importlib.invalidate_caches()
    failures: dict[str, str] = {}
    for distribution, module in required.items():
        try:
            importlib.import_module(module)
            if not _distribution_version_acceptable(distribution):
                raise RuntimeError("installed version is outside requirements_colab.txt")
        except Exception as error:  # noqa: BLE001 - dependency preflight
            failures[distribution] = f"{type(error).__name__}: {error}"
    if failures:
        raise ModuleNotFoundError(f"Dependency preflight failed: {failures}")


def _apply_cli(config: Any, args: argparse.Namespace) -> Any:
    run_changes = {
        key: value
        for key, value in {
            "run_id": args.run_id,
            "profile": args.profile,
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
        }.items()
        if value is not None
    }
    path_changes = {
        key: value
        for key, value in {
            "data_root": args.data_root,
            "output_root": args.output_dir,
            "cache_root": args.cache_dir,
        }.items()
        if value is not None
    }
    gemma_changes = {
        key: value
        for key, value in {
            "model": args.gemma_model,
            "api_key_env": args.api_key_env,
        }.items()
        if value is not None
    }
    if args.offline:
        gemma_changes["offline"] = True
    if args.regenerate_program:
        gemma_changes["regenerate_program"] = True
    if args.offline and args.regenerate_program:
        raise ValueError("--offline and --regenerate-program cannot be used together")
    if run_changes:
        config = config.with_run(**run_changes)
    if path_changes:
        config = config.with_paths(**path_changes)
    if gemma_changes:
        config = config.with_gemma(**gemma_changes)
    return config


def main(
    argv: Sequence[str] | None = None,
    *,
    namespace: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    namespace = globals() if namespace is None else namespace
    environ = os.environ if environ is None else environ
    notebook_launch = is_base_notebook_launch(namespace)
    if argv is not None:
        arguments = list(argv)
    elif notebook_launch:
        arguments = notebook_argv(environ)
        print(
            f"[launcher] base.ipynb launch detected; arguments = {' '.join(arguments)}",
            flush=True,
        )
    else:
        arguments = strip_jupyter_arguments(sys.argv[1:])

    args = build_parser().parse_args(arguments)
    ensure_dependencies(include_api=False, skip_install=bool(args.skip_install))

    from .config import load_config

    config = _apply_cli(load_config(args.config, environ=environ), args)
    if config.gemma.offline and config.gemma.regenerate_program:
        raise ValueError("offline and regenerate_program cannot both be enabled")
    needs_api = args.stage in {"program", "all"} and not config.gemma.offline
    if needs_api:
        ensure_dependencies(include_api=True, skip_install=bool(args.skip_install))

    from .pipeline import make_context, run_all, run_stage, write_status

    injected = {
        name: namespace.get(name)
        for name in ("PROJECT_ROOT", "DATA_ROOT", "USER_ROOT")
    }
    context = make_context(config, injected=injected)
    write_status(
        context,
        "starting",
        stage=args.stage,
        profile=config.run.profile,
    )
    try:
        if args.stage == "all":
            run_all(context)
        else:
            run_stage(context, args.stage)
        write_status(context, "complete", stage=args.stage)
    except BaseException as error:  # noqa: BLE001 - artifact must reflect failure
        write_status(
            context,
            "failed",
            stage=args.stage,
            error_type=type(error).__name__,
            error=(
                "Failure message omitted from the persisted artifact to prevent "
                "raw subject identifiers or API details from being written. "
                "See the live notebook traceback."
            ),
        )
        raise
    print(f"[launcher] done. artifacts in: {context.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(namespace=globals()))
