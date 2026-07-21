"""Entry point for the repository's USER_FOLDER/RUN_FILE base notebook.

The training CLI intentionally requires explicit paths.  This small adapter
resolves the paths supplied by ``base.ipynb`` (or safe local/Colab defaults),
creates a unique output directory, and calls the same tested training code.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from .train import DEFAULT_OUTER_SEEDS, RunConfig, run
else:
    from train import DEFAULT_OUTER_SEEDS, RunConfig, run


EXPERIMENT_NAME = "ThreeClass_NaiveBayes"


def _path(value: object | None) -> Path | None:
    if value is None:
        return None
    return Path(str(value)).expanduser().resolve()


def _first_existing(candidates: list[Path | None], *, description: str) -> Path:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    rendered = [str(candidate) for candidate in candidates if candidate is not None]
    raise FileNotFoundError(
        f"Could not resolve {description}. Checked: {rendered}"
    )


def resolve_base_settings(namespace: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Resolve notebook globals without requiring them for direct local runs."""

    values = dict(namespace or {})
    experiment_root = Path(__file__).resolve().parent
    project_root = _first_existing(
        [
            _path(values.get("PROJECT_ROOT")),
            experiment_root.parents[1],
        ],
        description="PROJECT_ROOT",
    )
    data_root = _first_existing(
        [
            _path(values.get("DATA_ROOT")),
            project_root / "Data",
            Path("/content/drive/Shareddrives/GoogleAI_contest/Data"),
            Path("/content/drive/MyDrive/GoogleAI_contest/Data"),
        ],
        description="DATA_ROOT",
    )
    training_root = data_root / "1.Training"
    validation_root = data_root / "2.Validation"
    missing_splits = [
        str(path) for path in (training_root, validation_root) if not path.is_dir()
    ]
    if missing_splits:
        raise FileNotFoundError(
            "Training/Validation split directory is missing: " + ", ".join(missing_splits)
        )

    mode = str(
        values.get(
            "NAIVE_BAYES_RUN_MODE",
            os.environ.get("NAIVE_BAYES_RUN_MODE", values.get("RUN_MODE", "full")),
        )
    ).strip().lower()
    if mode == "fast":
        mode = "smoke"
    if mode not in {"smoke", "full"}:
        raise ValueError("NAIVE_BAYES_RUN_MODE must be 'smoke' or 'full'.")

    results_override = _path(values.get("NAIVE_BAYES_RESULTS_ROOT"))
    if results_override is not None:
        results_root = results_override
    elif Path("/content/drive/MyDrive").is_dir():
        results_root = Path("/content/drive/MyDrive/SangHyo_NaiveBayes_Results")
    else:
        results_root = experiment_root / "training_outputs"
    results_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = results_root / f"{run_id}_{mode}_gaussian_nb"
    seed = int(values.get("SEED", 20260721))
    n_jobs = int(values.get("NAIVE_BAYES_N_JOBS", 1))
    skip_validation_labels = bool(values.get("NAIVE_BAYES_SKIP_VALIDATION_LABELS", False))
    return {
        "project_root": project_root,
        "data_root": data_root,
        "training_root": training_root,
        "validation_root": validation_root,
        "results_root": results_root,
        "output_dir": output_dir,
        "mode": mode,
        "seed": seed,
        "n_jobs": n_jobs,
        "skip_validation_labels": skip_validation_labels,
    }


def main(namespace: Mapping[str, Any] | None = None) -> Path:
    settings = resolve_base_settings(namespace)
    fast = settings["mode"] == "smoke"
    outer_seeds = DEFAULT_OUTER_SEEDS[:1] if fast else DEFAULT_OUTER_SEEDS
    config = RunConfig(
        training_root=str(settings["training_root"]),
        validation_root=str(settings["validation_root"]),
        output_dir=str(settings["output_dir"]),
        outer_folds=3,
        outer_seeds=outer_seeds,
        inner_folds=3,
        seed=settings["seed"],
        n_jobs=settings["n_jobs"],
        fast=fast,
        evaluate_validation_labels=not settings["skip_validation_labels"],
    )
    print(f"Project root : {settings['project_root']}")
    print(f"Data root    : {settings['data_root']}")
    print(f"Run mode     : {settings['mode']}")
    print(f"Output       : {settings['output_dir']}")
    run(config)
    return Path(config.output_dir)


if __name__ == "__main__":
    main(globals())
