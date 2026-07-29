"""Single entry point for the whole pipeline.

Two ways to start it, both landing in :func:`main`:

1. **Colab / repository ``base.ipynb``** - Cell 2 selects this file
   (``RUN_FILE = "GeminiFeaturePipeline/run.py"``) and Cell 5 executes it with
   ``runpy``, injecting ``PROJECT_ROOT``, ``DATA_ROOT``, ``USER_ROOT`` and
   ``RUN_PATH``.  ``sys.argv`` then belongs to the Jupyter kernel, so arguments
   are read from the ``GFP_ARGS`` environment variable instead (see README_KO).
2. **Shell** - ``python run.py --config config.yaml --stage all --mmse-mode both``.

``base.ipynb`` itself is never modified: this file adapts to it.
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

if __package__ in {None, ""}:  # ``runpy``/direct execution: make the package importable
    _package_root = Path(__file__).resolve().parent
    _repository_root = _package_root.parents[1]
    if str(_repository_root) not in sys.path:
        sys.path.insert(0, str(_repository_root))
    __package__ = f"SangHyo.{_package_root.name}"

from .config import PipelineConfig, load_config  # noqa: E402
from .pipeline import STAGES, make_context, run_all, run_stage, write_status  # noqa: E402

EXPERIMENT_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
ARGS_ENV_VARIABLE = "GFP_ARGS"

_REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "PyYAML": "yaml",
}
_GEMINI_MODULES = {"google-genai": "google.genai"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Gemini structured-feature extraction + downstream subject-level "
            "classification (CN vs MCI+Dem). Gemini never sees a label."
        ),
    )
    parser.add_argument("--config", default=None, help="path to a YAML/JSON config (default: config.yaml)")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--mmse-mode", choices=("without", "with", "both"), default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None, help="results root; a new run id is created inside it")
    parser.add_argument("--cache-dir", default=None, help="persistent Gemini/daily-table cache root")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)

    gemini = parser.add_argument_group("gemini")
    gemini.add_argument("--gemini-model", default=None)
    gemini.add_argument("--api-key-env", default=None, help="name of the env var holding the key")
    gemini.add_argument("--dry-run", action="store_true", help="build payloads only; never call the API")
    gemini.add_argument("--offline", action="store_true", help="use cached answers only")
    gemini.add_argument("--retry-failed", action="store_true", help="re-issue previously failed subjects")
    gemini.add_argument("--limit-subjects", type=int, default=None)
    gemini.add_argument("--max-concurrency", type=int, default=None)
    gemini.add_argument("--no-gemini", action="store_true", help="BASE-only run, no feature extraction")

    parser.add_argument("--feature-sets", default=None, help="comma separated: base,base_gemini")
    parser.add_argument("--models", default=None, help="comma separated: logreg,gbdt")
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
        "gemini.model": args.gemini_model,
        "gemini.api_key_env": args.api_key_env,
        "gemini.limit_subjects": args.limit_subjects,
        "gemini.max_concurrency": args.max_concurrency,
        "mmse_mode": args.mmse_mode,
    }
    if args.dry_run:
        overrides["gemini.dry_run"] = True
    if args.offline:
        overrides["gemini.offline"] = True
    if args.retry_failed:
        overrides["gemini.retry_failed"] = True
    if args.no_gemini:
        overrides["gemini.enabled"] = False
        overrides["features.feature_sets"] = ("base",)
    if args.feature_sets:
        overrides["features.feature_sets"] = tuple(
            piece.strip() for piece in args.feature_sets.split(",") if piece.strip()
        )
    if args.models:
        overrides["models.enabled"] = tuple(
            piece.strip() for piece in args.models.split(",") if piece.strip()
        )
    return {key: value for key, value in overrides.items() if value is not None}


# --------------------------------------------------------------------------- #
# base.ipynb interoperability
# --------------------------------------------------------------------------- #
def _is_kernel_connection_file(value: str) -> bool:
    path = Path(str(value))
    return path.name.startswith("kernel-") and path.suffix == ".json"


def strip_jupyter_arguments(argv: Sequence[str]) -> list[str]:
    """Drop only the connection-file arguments Jupyter injects into ``sys.argv``."""

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
    """True when the repository's unmodified ``base.ipynb`` is executing this file."""

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
    """Arguments for a notebook launch: ``GFP_ARGS`` if set, else the full pipeline."""

    environ = os.environ if environ is None else environ
    raw = str(environ.get(ARGS_ENV_VARIABLE, "")).strip()
    if raw:
        return shlex.split(raw)
    return ["--stage", "all", "--mmse-mode", "both"]


# --------------------------------------------------------------------------- #
# dependencies
# --------------------------------------------------------------------------- #
def _module_available(module: str) -> bool:
    """``find_spec`` on a dotted name raises when the parent package is absent."""

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _parse_requirements(path: Path) -> dict[str, str]:
    """Map normalized distribution name -> its exact requirement line.

    Used so an install can be scoped to a handful of distributions (e.g. just
    ``google-genai``) while still applying the version constraints declared in
    ``requirements_colab.txt``, instead of re-resolving the whole file.
    """

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


def ensure_dependencies(*, include_gemini: bool, skip_install: bool) -> dict[str, str]:
    """Install only what the current stage actually needs, and only if missing.

    A ``--dry-run`` never calls the Gemini API, so it must not require
    ``google-genai``/``lightgbm`` to be installable at all; forcing a full
    ``requirements_colab.txt`` install on every notebook launch previously
    broke dry-run whenever those extras hit a resolver conflict in the Colab
    image, even though dry-run never needed them.
    """

    required = dict(_REQUIRED_MODULES)
    if include_gemini:
        required.update(_GEMINI_MODULES)
    missing = [
        distribution
        for distribution, module in required.items()
        if not _module_available(module)
    ]
    if missing and skip_install:
        raise ModuleNotFoundError(
            "Dependencies must be installed but --skip-install was given: " + ", ".join(missing)
        )
    if missing:
        lines = _parse_requirements(REQUIREMENTS_FILE)
        targets = [lines.get(name.lower(), name) for name in missing]
        print(f"[launcher] installing missing dependencies: {', '.join(missing)}", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *targets],
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-25:])
            raise RuntimeError(
                f"pip install failed for {targets} (exit {result.returncode}). "
                f"Last lines of stderr:\n{tail}"
            )
        importlib.invalidate_caches()
    failures: dict[str, str] = {}
    for distribution, module in required.items():
        try:
            importlib.import_module(module)
        except Exception as error:  # noqa: BLE001 - import preflight, not a fallback
            failures[distribution] = f"{type(error).__name__}: {error}"
    if failures:
        raise ModuleNotFoundError(f"Dependency preflight failed: {failures}")
    return {name: "ok" for name in required}


# --------------------------------------------------------------------------- #
# main
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
    stage = args.stage
    needs_api = stage in {"gemini", "all"} and not args.dry_run and not args.offline and not args.no_gemini
    ensure_dependencies(include_gemini=needs_api, skip_install=bool(args.skip_install))

    config: PipelineConfig = load_config(
        args.config, environ=environ, cli_overrides=_cli_overrides(args)
    )
    injected = {key: namespace.get(key) for key in ("PROJECT_ROOT", "DATA_ROOT", "USER_ROOT")}
    context = make_context(config, injected=injected)
    write_status(context, "starting", stage=stage, mmse_mode=config.mmse_mode)

    try:
        if stage == "all":
            run_all(context)
        else:
            run_stage(context, stage)
            write_status(context, "complete", stages=[stage])
    except BaseException as error:  # noqa: BLE001 - status must reflect reality
        write_status(context, "failed", stage=stage, error=f"{type(error).__name__}: {error}"[:1000])
        raise
    print(f"[launcher] done. artifacts in: {context.run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(namespace=globals()))
