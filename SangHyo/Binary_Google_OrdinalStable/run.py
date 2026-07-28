"""base.ipynb entry point -- ROC-AUC improvement ablation (ordinal + stability).

    USER_FOLDER = "SangHyo"
    RUN_FILE    = "Binary_Google_OrdinalStable/run.py"

Recommended Colab runtime: **CPU, High-RAM. No GPU** (Google YDF is a
multi-threaded CPU library).

Modes
-----
``smoke``     ~3 min     1 repeat, 2 arms, wiring check only
``standard``  ~30 min    2 repeats, all 6 arms
``full``      ~1.5 h     3 repeats, all 6 arms (default)
``deep``      ~4 h       6 repeats -- use to confirm a winner, not to discover one

Environment switches::

    ORDSTABLE_STRATEGIES=binary,ordinal      restrict strategies
    ORDSTABLE_SELECTIONS=stability           restrict selection methods
    ORDSTABLE_KINDS=logreg,ydf_gbt_oblique   restrict learners
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

EXPERIMENT_NAME = "Binary_Google_OrdinalStable"
EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
REQUIREMENTS_FILE = EXPERIMENT_ROOT / "requirements_colab.txt"
DEFAULT_COLAB_RESULTS_ROOT = Path(f"/content/drive/MyDrive/{EXPERIMENT_NAME}_result")

ALL_KINDS = ("logreg", "svm", "ydf_gbt", "ydf_rf", "ydf_gbt_oblique")
ALL_STRATEGIES = ("binary", "ordinal", "hard_boundary")
ALL_SELECTIONS = ("fold_topk", "stability")

MODE_SETTINGS: dict[str, dict[str, Any]] = {
    "smoke": dict(repeats=1, outer_k=3, inner_k=3, top_k=15, top_m=1,
                  strategies=("binary", "ordinal"), selections=("stability",)),
    "standard": dict(repeats=2, outer_k=5, inner_k=5, top_k=25, top_m=2,
                     strategies=ALL_STRATEGIES, selections=ALL_SELECTIONS),
    "full": dict(repeats=3, outer_k=5, inner_k=5, top_k=25, top_m=2,
                 strategies=ALL_STRATEGIES, selections=ALL_SELECTIONS),
    "deep": dict(repeats=6, outer_k=5, inner_k=5, top_k=25, top_m=2,
                 strategies=ALL_STRATEGIES, selections=ALL_SELECTIONS),
}


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
    missing = [d for d, m in required.items() if importlib.util.find_spec(m) is None]
    if missing and not skip_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "-r", str(REQUIREMENTS_FILE)], check=True)
        importlib.invalidate_caches()
    elif missing:
        raise ModuleNotFoundError("Missing dependencies: " + ", ".join(missing))
    if importlib.util.find_spec("ydf") is None and not skip_install:
        print("[run] installing Google YDF (ydf)...", flush=True)
        result = subprocess.run([sys.executable, "-m", "pip", "install",
                                 "--disable-pip-version-check", "ydf"], check=False)
        importlib.invalidate_caches()
        if result.returncode != 0:
            print("[run] WARNING: ydf install failed; tree arms fall back to HistGBT.",
                  flush=True)


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


def _restrict(env_name: str, allowed: tuple[str, ...], default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    chosen = tuple(v.strip() for v in raw.split(",") if v.strip())
    unknown = [v for v in chosen if v not in allowed]
    if unknown:
        raise ValueError(f"Unknown values in {env_name}: {unknown}")
    return chosen


def run_pipeline(*, namespace=None, data_root=None, output_dir=None, mode=None,
                 skip_install=False, evaluate_validation=True) -> dict:
    namespace = globals() if namespace is None else namespace
    run_mode = (mode or os.environ.get("ORDSTABLE_MODE") or "full").strip().lower()
    if run_mode not in MODE_SETTINGS:
        raise ValueError(f"mode must be one of {sorted(MODE_SETTINGS)}")
    _ensure_dependencies(skip_install)

    resolved_data = _resolve_data_root(namespace, data_root)
    output = _resolve_output_dir(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new/empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    settings = dict(MODE_SETTINGS[run_mode])
    settings["strategies"] = _restrict("ORDSTABLE_STRATEGIES", ALL_STRATEGIES,
                                       settings["strategies"])
    settings["selections"] = _restrict("ORDSTABLE_SELECTIONS", ALL_SELECTIONS,
                                       settings["selections"])
    kinds = _restrict("ORDSTABLE_KINDS", ALL_KINDS, ALL_KINDS)

    started = time.monotonic()
    _write_status(output, {"status": "starting", "experiment": EXPERIMENT_NAME, "mode": run_mode,
                           "kinds": list(kinds), "strategies": list(settings["strategies"]),
                           "selections": list(settings["selections"]),
                           "data_root": str(resolved_data), "output_dir": str(output)})

    from SangHyo.Binary_Google_OrdinalStable.train import RunConfig, run_experiment

    config = RunConfig(data_root=str(resolved_data), output_dir=str(output), run_mode=run_mode,
                       kinds=kinds, evaluate_validation=evaluate_validation, **settings)
    try:
        result = run_experiment(config)
    except Exception as error:
        _write_status(output, {"status": "failed", "mode": run_mode,
                               "error_type": type(error).__name__, "error": str(error),
                               "elapsed_seconds": time.monotonic() - started})
        raise

    _write_status(output, {"status": "complete", "mode": run_mode,
                           "elapsed_seconds": time.monotonic() - started,
                           "best_arm": result["best_arm"],
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
