"""Single entry point.

Two ways in, both landing in :func:`main`:

1. **Colab via the repository's ``base.ipynb``** - Cell 2 sets
   ``RUN_FILE = "Binary_PooledMaxAUC/run.py"`` and Cell 5 executes this file with
   ``runpy``, injecting ``PROJECT_ROOT``/``DATA_ROOT``/``USER_ROOT``/``RUN_PATH``.
   ``sys.argv`` then belongs to the Jupyter kernel, so arguments are read from the
   ``BPM_ARGS`` environment variable instead.
2. **Shell** - ``python run.py --config config.yaml --stage all``.

``base.ipynb`` is never modified; this file adapts to it.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:  # runpy / direct execution
    _package_root = Path(__file__).resolve().parent
    _repository_root = _package_root.parents[1]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))
    __package__ = f"SangHyo.{_package_root.name}"

from .config import PipelineConfig, load_config  # noqa: E402
from .pipeline import STAGES, make_context, run_all, run_stage, write_status  # noqa: E402

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
ARGS_ENV_VARIABLE = "BPM_ARGS"

#: Hard requirements. Optional boosting libraries are handled by models.py and
#: their absence removes candidates rather than breaking the run.
_REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Maximum-ROC-AUC CN vs MCI+Dem search on the pooled 174-subject cohort. "
            "Avoids direct leakage only; selection optimism is permitted and disclosed."
        ),
    )
    parser.add_argument("--config", default=None, help="YAML/JSON config (default: config.yaml)")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--profile", choices=("fast", "default", "max"), default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--splits",
        default=None,
        help="comma separated: train,val (default) or train (141-subject, comparable to other experiments)",
    )
    parser.add_argument("--views", default=None, help="comma separated feature views")
    parser.add_argument("--families", default=None, help="comma separated learner families")
    parser.add_argument("--cv-splits", type=int, default=None)
    parser.add_argument("--cv-repeats", type=int, default=None)
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "paths.data_root": args.data_root,
        "paths.output_root": args.output_dir,
        "paths.cache_root": args.cache_dir,
        "run.run_id": args.run_id,
        "run.seed": args.seed,
        "run.n_jobs": args.n_jobs,
        "run.profile": args.profile,
        "cv.n_splits": args.cv_splits,
        "cv.n_repeats": args.cv_repeats,
    }
    if args.splits:
        overrides["data.splits"] = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if args.views:
        overrides["features.views"] = tuple(v.strip() for v in args.views.split(",") if v.strip())
    if args.families:
        overrides["candidates.families"] = tuple(
            f.strip() for f in args.families.split(",") if f.strip()
        )
    if args.no_ensemble:
        overrides["ensemble.enabled"] = False
    return {k: v for k, v in overrides.items() if v is not None}


# --------------------------------------------------------------------------- #
# base.ipynb interoperability
# --------------------------------------------------------------------------- #
def _is_kernel_connection_file(value: str) -> bool:
    path = Path(str(value))
    return path.name.startswith("kernel-") and path.suffix == ".json"


def strip_jupyter_arguments(argv: Sequence[str]) -> list[str]:
    cleaned: list[str] = []
    index = 0
    argv = list(argv)
    while index < len(argv):
        token = str(argv[index])
        if token in {"-f", "--f"} and index + 1 < len(argv) and _is_kernel_connection_file(argv[index + 1]):
            index += 2
            continue
        if token.startswith(("-f=", "--f=")) and _is_kernel_connection_file(token.split("=", 1)[1]):
            index += 1
            continue
        if _is_kernel_connection_file(token):
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
        run_path = Path(os.fspath(namespace["RUN_PATH"])).resolve()
        project_root = Path(os.fspath(namespace["PROJECT_ROOT"])).resolve()
    except (OSError, TypeError, ValueError):
        return False
    return run_path == Path(__file__).resolve() and (project_root / "base.ipynb").is_file()


def notebook_argv(environ: Mapping[str, str] | None = None) -> list[str]:
    environ = os.environ if environ is None else environ
    raw = str(environ.get(ARGS_ENV_VARIABLE, "")).strip()
    return shlex.split(raw) if raw else ["--stage", "all"]


# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #
def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _parse_requirements(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        name = line
        for separator in ("==", ">=", "<=", "~=", "!=", ">", "<", "["):
            name = name.split(separator, 1)[0]
        mapping[name.strip().lower()] = line
    return mapping


def ensure_dependencies(*, skip_install: bool, install_optional: bool = False) -> dict[str, str]:
    """Install only what is missing. Optional boosting libs are best-effort."""

    missing = [dist for dist, module in _REQUIRED_MODULES.items() if not _module_available(module)]
    if missing and skip_install:
        raise ModuleNotFoundError(
            "Required dependencies are missing but --skip-install was given: " + ", ".join(missing)
        )
    lines = _parse_requirements(REQUIREMENTS_FILE) if REQUIREMENTS_FILE.is_file() else {}
    if missing:
        targets = [lines.get(name.lower(), name) for name in missing]
        print(f"[launcher] installing required dependencies: {', '.join(missing)}", flush=True)
        _pip_install(targets, fatal=True)

    if install_optional and not skip_install:
        optional = {
            "lightgbm": "lightgbm",
            "catboost": "catboost",
            "xgboost": "xgboost",
            "ydf": "ydf",
        }
        absent = [dist for dist, module in optional.items() if not _module_available(module)]
        if absent:
            targets = [lines.get(name.lower(), name) for name in absent]
            print(f"[launcher] installing optional learners: {', '.join(absent)}", flush=True)
            # Best-effort: a missing booster only removes candidates.
            _pip_install(targets, fatal=False)

    failures: dict[str, str] = {}
    for dist, module in _REQUIRED_MODULES.items():
        try:
            importlib.import_module(module)
        except Exception as error:  # noqa: BLE001 - import preflight
            failures[dist] = f"{type(error).__name__}: {error}"
    if failures:
        raise ModuleNotFoundError(f"Dependency preflight failed: {failures}")
    return {name: "ok" for name in _REQUIRED_MODULES}


def _pip_install(targets: Sequence[str], *, fatal: bool) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *targets],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-20:])
        message = f"pip install failed for {list(targets)} (exit {result.returncode}):\n{tail}"
        if fatal:
            raise RuntimeError(message)
        print(f"[launcher] WARNING: {message}", file=sys.stderr, flush=True)
    importlib.invalidate_caches()


# --------------------------------------------------------------------------- #
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
        print(f"[launcher] base.ipynb launch detected; arguments = {' '.join(arguments)}", flush=True)
    else:
        arguments = strip_jupyter_arguments(sys.argv[1:])

    args = build_parser().parse_args(arguments)
    ensure_dependencies(
        skip_install=bool(args.skip_install),
        install_optional=args.stage in {"search", "all"},
    )

    config: PipelineConfig = load_config(
        args.config, environ=environ, cli_overrides=_cli_overrides(args)
    )
    injected = {key: namespace.get(key) for key in ("PROJECT_ROOT", "DATA_ROOT", "USER_ROOT")}
    context = make_context(config, injected=injected)
    write_status(context, "starting", stage=args.stage)

    try:
        if args.stage == "all":
            run_all(context)
        else:
            run_stage(context, args.stage)
            write_status(context, "complete", stages=[args.stage])
    except BaseException as error:  # noqa: BLE001 - status must reflect reality
        write_status(context, "failed", stage=args.stage, error=f"{type(error).__name__}: {error}"[:1000])
        raise
    print(f"[launcher] done. artifacts in: {context.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(namespace=globals()))
