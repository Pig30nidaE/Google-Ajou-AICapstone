"""base.ipynb entry point -- performance-first, leakage-free, heavily tuned.

    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Binary_Google_MaxAUC_Tuned/run.py"

Recommended Colab runtime
-------------------------
**CPU, High-RAM.  No GPU.**  Google YDF is a multi-threaded CPU library, so the
useful knob is core count, not an accelerator -- an A100 session would sit idle.
Pick the highest-core CPU runtime available and enable background execution.

Modes (``--mode`` or ``MAXAUC_MODE``)
------------------------------------
``smoke``     ~5 min       wiring check only, metrics are meaningless
``standard``  ~2-3 h       full pipeline, reduced search budget
``max``       ~6-9 h       default; the performance-first configuration
``extreme``   ~18-22 h     5 outer repeats, ~2x search budget

Each mode stops at an outer-repeat boundary if its deadline is reached, so the
out-of-fold predictions are always complete -- a truncated run reports fewer
repeats, never a partially-scored cohort.

Optional environment switches::

    MAXAUC_DROP_SUSPECT=1   drop adherence/wear-time features (conservative run)
    MAXAUC_NO_WEARABLE=1    MMSE-only feature matrix
    MAXAUC_SKIP_OPTIMISM=1  skip the non-nested diagnostic (not recommended)
    MAXAUC_KINDS=a,b,c      restrict the learner pool
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
import time
from typing import Any, Mapping

EXPERIMENT_NAME = "Binary_Google_MaxAUC_Tuned"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
DEFAULT_COLAB_RESULTS_ROOT = Path(f"/content/drive/MyDrive/{EXPERIMENT_NAME}_result")

ALL_KINDS = ("logreg", "svm", "ydf_gbt", "ydf_rf", "ydf_gbt_oblique", "ydf_rf_oblique")

MODE_SETTINGS: dict[str, dict[str, Any]] = {
    "smoke": dict(
        repeats=1, outer_k=3, inner_k=3, screen_repeats=1, final_repeats=1,
        budgets={"logreg": 4, "svm": 3, "ydf_gbt": 3, "ydf_rf": 3,
                 "ydf_gbt_oblique": 3, "ydf_rf_oblique": 3},
        deadline_seconds=None, run_optimism_check=True,
    ),
    "standard": dict(
        repeats=2, outer_k=5, inner_k=5, screen_repeats=1, final_repeats=2,
        budgets={"logreg": 24, "svm": 18, "ydf_gbt": 30, "ydf_rf": 24,
                 "ydf_gbt_oblique": 30, "ydf_rf_oblique": 24},
        deadline_seconds=6 * 3600, run_optimism_check=True,
    ),
    "max": dict(
        repeats=3, outer_k=5, inner_k=5, screen_repeats=1, final_repeats=2,
        budgets={"logreg": 30, "svm": 24, "ydf_gbt": 45, "ydf_rf": 30,
                 "ydf_gbt_oblique": 45, "ydf_rf_oblique": 30},
        deadline_seconds=14 * 3600, run_optimism_check=True,
    ),
    "extreme": dict(
        repeats=5, outer_k=5, inner_k=5, screen_repeats=2, final_repeats=3,
        budgets={"logreg": 60, "svm": 45, "ydf_gbt": 90, "ydf_rf": 60,
                 "ydf_gbt_oblique": 90, "ydf_rf_oblique": 60},
        deadline_seconds=22 * 3600, run_optimism_check=True,
    ),
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


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
                "scikit-learn": "sklearn", "joblib": "joblib"}
    missing = [dist for dist, module in required.items() if importlib.util.find_spec(module) is None]
    if missing and not skip_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "-r", str(REQUIREMENTS_FILE)], check=True)
        importlib.invalidate_caches()
    elif missing:
        raise ModuleNotFoundError("Missing dependencies: " + ", ".join(missing))

    # Google YDF is the point of this experiment: install it if it is absent.
    if importlib.util.find_spec("ydf") is None and not skip_install:
        print("[run] installing Google YDF (ydf)...", flush=True)
        result = subprocess.run([sys.executable, "-m", "pip", "install",
                                 "--disable-pip-version-check", "ydf"], check=False)
        importlib.invalidate_caches()
        if result.returncode != 0:
            print("[run] WARNING: ydf install failed; the YDF learners will fall back to "
                  "a NaN-tolerant sklearn stand-in.", flush=True)


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


def _resolve_kinds() -> tuple[str, ...]:
    raw = os.environ.get("MAXAUC_KINDS", "").strip()
    if not raw:
        return ALL_KINDS
    chosen = tuple(k.strip() for k in raw.split(",") if k.strip())
    unknown = [k for k in chosen if k not in ALL_KINDS]
    if unknown:
        raise ValueError(f"Unknown learner kinds in MAXAUC_KINDS: {unknown}")
    return chosen


def run_pipeline(*, namespace=None, data_root=None, output_dir=None, mode=None,
                 skip_install=False, evaluate_validation=True) -> dict:
    namespace = globals() if namespace is None else namespace
    run_mode = (mode or os.environ.get("MAXAUC_MODE") or "max").strip().lower()
    if run_mode not in MODE_SETTINGS:
        raise ValueError(f"mode must be one of {sorted(MODE_SETTINGS)}")
    _ensure_dependencies(skip_install)

    resolved_data = _resolve_data_root(namespace, data_root)
    output = _resolve_output_dir(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib_cache"))

    settings = dict(MODE_SETTINGS[run_mode])
    if _env_flag("MAXAUC_SKIP_OPTIMISM"):
        settings["run_optimism_check"] = False
    kinds = _resolve_kinds()
    settings["budgets"] = {k: v for k, v in settings["budgets"].items() if k in kinds}

    started = time.monotonic()
    _write_status(output, {"status": "starting", "experiment": EXPERIMENT_NAME, "mode": run_mode,
                           "kinds": list(kinds), "data_root": str(resolved_data),
                           "output_dir": str(output)})

    from SangHyo.Binary_Google_MaxAUC_Tuned.train import RunConfig, run_experiment

    config = RunConfig(
        training_root=str(resolved_data / "1.Training"),
        validation_root=str(resolved_data / "2.Validation"),
        output_dir=str(output),
        run_mode=run_mode,
        kinds=kinds,
        drop_suspect=_env_flag("MAXAUC_DROP_SUSPECT"),
        include_wearable=not _env_flag("MAXAUC_NO_WEARABLE"),
        evaluate_validation=evaluate_validation,
        **settings,
    )
    try:
        result = run_experiment(config)
    except Exception as error:
        _write_status(output, {"status": "failed", "mode": run_mode,
                               "error_type": type(error).__name__, "error": str(error),
                               "elapsed_seconds": time.monotonic() - started})
        raise

    _write_status(output, {"status": "complete", "mode": run_mode,
                           "elapsed_seconds": time.monotonic() - started,
                           "headline_roc_auc": result["headline_roc_auc"],
                           "final_report": str(output / "training" / "FINAL_REPORT.json")})
    print(f"\nComplete ({run_mode}) in {(time.monotonic() - started) / 60:.1f} min. "
          f"Report: {output / 'training' / 'FINAL_REPORT.json'}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--mode", choices=tuple(MODE_SETTINGS), default=None)
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    arguments, _unknown = parser.parse_known_args()
    run_pipeline(namespace=globals(), data_root=arguments.data_root,
                 output_dir=arguments.output_dir, mode=arguments.mode,
                 skip_install=arguments.skip_install,
                 evaluate_validation=not arguments.skip_validation)


if __name__ == "__main__":
    main()
