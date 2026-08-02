"""Train and evaluate a Gaussian Naive Bayes CN/MCI/DEM baseline.

All learned preprocessing is fitted inside subject-level folds.  Activity and
sleep are used as features; cognitive-function source values are never opened.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

if __package__:
    from .data import (
        CLASS_NAMES,
        build_subject_dataset,
        load_aligned_labels,
        shared_code_manifest,
    )
    from .evaluation import (
        all_cn_baseline,
        class_prior_probabilities,
        metrics_from_probabilities,
        summarize_repeat_metrics,
    )
    from .modeling import (
        fit_grid_search,
        predict_probabilities,
        readable_best_params,
        selected_feature_names,
        validate_split_count,
    )
else:
    from data import (
        CLASS_NAMES,
        build_subject_dataset,
        load_aligned_labels,
        shared_code_manifest,
    )
    from evaluation import (
        all_cn_baseline,
        class_prior_probabilities,
        metrics_from_probabilities,
        summarize_repeat_metrics,
    )
    from modeling import (
        fit_grid_search,
        predict_probabilities,
        readable_best_params,
        selected_feature_names,
        validate_split_count,
    )


DESIGN_VERSION = "threeclass_gaussian_naive_bayes_v1"
DEFAULT_OUTER_SEEDS = (137, 1009, 2027, 4099, 8191)


@dataclass(frozen=True)
class RunConfig:
    training_root: str
    validation_root: str | None
    output_dir: str
    outer_folds: int
    outer_seeds: tuple[int, ...]
    inner_folds: int
    seed: int
    n_jobs: int
    fast: bool
    evaluate_validation_labels: bool


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def subject_hash(subject_id: object) -> str:
    payload = f"{DESIGN_VERSION}::{subject_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_info() -> dict[str, Any]:
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
        },
    }


def code_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    local = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.glob("*.py"))
    }
    return {**local, **shared_code_manifest()}


def prepare_output_dir(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def assert_disjoint_subjects(
    training_subject_ids: np.ndarray,
    validation_subject_ids: np.ndarray,
) -> None:
    """Reject a holdout that contains a Training subject."""

    training_keys = {
        str(value).strip().casefold() for value in training_subject_ids
    }
    validation_keys = {
        str(value).strip().casefold() for value in validation_subject_ids
    }
    overlap = training_keys & validation_keys
    if overlap:
        raise AssertionError(
            "Training and Validation subject sets overlap; refusing evaluation "
            f"for {len(overlap)} subject(s)"
        )


def _prediction_frame(
    subject_ids: np.ndarray,
    probabilities: np.ndarray,
    *,
    y_true: np.ndarray | None = None,
    repeat: int | None = None,
    folds: np.ndarray | None = None,
) -> pd.DataFrame:
    predicted = probabilities.argmax(axis=1)
    payload: dict[str, Any] = {
        "subject_hash": [subject_hash(value) for value in subject_ids],
        "predicted_class": [CLASS_NAMES[index] for index in predicted],
    }
    for class_id, class_name in enumerate(CLASS_NAMES):
        payload[f"probability_{class_name}"] = probabilities[:, class_id]
    if y_true is not None:
        payload["true_class"] = [CLASS_NAMES[index] for index in y_true]
    if repeat is not None:
        payload["repeat"] = np.full(len(subject_ids), int(repeat), dtype=np.int64)
    if folds is not None:
        payload["outer_fold"] = np.asarray(folds, dtype=np.int64)
    return pd.DataFrame(payload)


def run_nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    subject_ids: np.ndarray,
    config: RunConfig,
    output: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    validate_split_count(y, config.outer_folds, context="Outer CV")
    fold_reports: list[dict[str, Any]] = []
    repeat_reports: list[dict[str, Any]] = []
    repeat_probabilities: list[np.ndarray] = []
    prior_repeat_probabilities: list[np.ndarray] = []
    prior_repeat_metrics: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for repeat_index, outer_seed in enumerate(config.outer_seeds):
        splitter = StratifiedKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=outer_seed,
        )
        probabilities = np.full((len(y), len(CLASS_NAMES)), np.nan, dtype=np.float64)
        prior_probabilities = np.full_like(probabilities, np.nan)
        fold_assignment = np.full(len(y), -1, dtype=np.int64)
        for fold_index, (fit_indices, holdout_indices) in enumerate(splitter.split(X, y)):
            inner_seed = int(outer_seed + 10_003 * (fold_index + 1))
            search = fit_grid_search(
                X.iloc[fit_indices],
                y[fit_indices],
                inner_folds=config.inner_folds,
                seed=inner_seed,
                fast=config.fast,
                n_jobs=config.n_jobs,
            )
            holdout_probabilities = predict_probabilities(
                search.best_estimator_,
                X.iloc[holdout_indices],
            )
            probabilities[holdout_indices] = holdout_probabilities
            prior_probabilities[holdout_indices] = class_prior_probabilities(
                y[fit_indices],
                len(holdout_indices),
            )
            fold_assignment[holdout_indices] = fold_index
            fold_reports.append(
                {
                    "repeat": repeat_index,
                    "outer_seed": outer_seed,
                    "fold": fold_index,
                    "fit_subjects": int(len(fit_indices)),
                    "holdout_subjects": int(len(holdout_indices)),
                    "fit_class_counts": np.bincount(
                        y[fit_indices], minlength=len(CLASS_NAMES)
                    ).tolist(),
                    "holdout_class_counts": np.bincount(
                        y[holdout_indices], minlength=len(CLASS_NAMES)
                    ).tolist(),
                    "inner_seed": inner_seed,
                    "inner_best_score": float(search.best_score_),
                    "best_params": readable_best_params(search.best_params_),
                    "selected_features": selected_feature_names(
                        search.best_estimator_,
                        list(X.columns),
                    ),
                    "holdout_metrics": metrics_from_probabilities(
                        y[holdout_indices],
                        holdout_probabilities,
                    ),
                }
            )
        if (
            not np.isfinite(probabilities).all()
            or not np.isfinite(prior_probabilities).all()
            or np.any(fold_assignment < 0)
        ):
            raise AssertionError("Outer CV did not predict every subject exactly once")
        repeat_metrics = metrics_from_probabilities(y, probabilities)
        repeat_reports.append(
            {
                "repeat": repeat_index,
                "outer_seed": outer_seed,
                "metrics": repeat_metrics,
            }
        )
        repeat_probabilities.append(probabilities)
        prior_repeat_probabilities.append(prior_probabilities)
        prior_repeat_metrics.append(
            metrics_from_probabilities(y, prior_probabilities)
        )
        prediction_frames.append(
            _prediction_frame(
                subject_ids,
                probabilities,
                y_true=y,
                repeat=repeat_index,
                folds=fold_assignment,
            )
        )

    repeat_average = np.mean(np.stack(repeat_probabilities, axis=0), axis=0)
    prior_repeat_average = np.mean(
        np.stack(prior_repeat_probabilities, axis=0),
        axis=0,
    )
    report = {
        "design_version": DESIGN_VERSION,
        "class_order": list(CLASS_NAMES),
        "outer_folds": config.outer_folds,
        "outer_seeds": list(config.outer_seeds),
        "inner_folds": config.inner_folds,
        "fast_mode": config.fast,
        "repeat_metrics": repeat_reports,
        "repeat_summary": summarize_repeat_metrics(
            [item["metrics"] for item in repeat_reports]
        ),
        "repeat_averaged_oof_metrics_supplemental": metrics_from_probabilities(
            y,
            repeat_average,
        ),
        "folds": fold_reports,
        "baselines": {
            "all_cn": all_cn_baseline(y),
            "class_prior_outer_fit_only": {
                "repeat_metrics": prior_repeat_metrics,
                "repeat_summary": summarize_repeat_metrics(prior_repeat_metrics),
                "repeat_averaged_oof_metrics_supplemental": metrics_from_probabilities(
                    y,
                    prior_repeat_average,
                ),
            },
        },
    }
    write_json(output / "nested_cv_report.json", report)
    write_csv(
        output / "nested_oof_predictions_hashed.csv",
        pd.concat(prediction_frames, ignore_index=True),
    )
    return report, repeat_average


def fit_full_model(
    X: pd.DataFrame,
    y: np.ndarray,
    config: RunConfig,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    search = fit_grid_search(
        X,
        y,
        inner_folds=config.inner_folds,
        seed=config.seed,
        fast=config.fast,
        n_jobs=config.n_jobs,
    )
    bundle = {
        "design_version": DESIGN_VERSION,
        "class_names": CLASS_NAMES,
        "feature_names": list(X.columns),
        "pipeline": search.best_estimator_,
    }
    model_path = output / "models" / "naive_bayes_pipeline.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_model = model_path.with_name(model_path.name + ".tmp")
    joblib.dump(bundle, temporary_model)
    temporary_model.replace(model_path)
    manifest = {
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "class_order": list(CLASS_NAMES),
        "inner_seed": config.seed,
        "inner_best_score": float(search.best_score_),
        "best_params": readable_best_params(search.best_params_),
        "selected_features": selected_feature_names(
            search.best_estimator_,
            list(X.columns),
        ),
        "pipeline_steps": [name for name, _ in search.best_estimator_.steps],
    }
    write_json(output / "model_manifest.json", manifest)
    return bundle, manifest


def evaluate_validation(
    validation_root: str,
    bundle: dict[str, Any],
    training_subject_ids: np.ndarray,
    config: RunConfig,
    output: Path,
) -> dict[str, Any]:
    validation = build_subject_dataset(validation_root, require_labels=False)
    assert_disjoint_subjects(training_subject_ids, validation.subject_ids)
    write_json(output / "validation_feature_audit_label_free.json", validation.audit)
    required_columns = list(bundle["feature_names"])
    found_columns = list(validation.X.columns)
    missing = sorted(set(required_columns) - set(found_columns))
    extra = sorted(set(found_columns) - set(required_columns))
    if missing or extra:
        raise ValueError(
            "Validation feature schema differs from Training: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    validation_X = validation.X.loc[:, required_columns]
    probabilities = predict_probabilities(bundle["pipeline"], validation_X)
    label_free_frame = _prediction_frame(validation.subject_ids, probabilities)
    frozen_path = output / "validation_predictions_label_free_hashed.csv"
    write_csv(frozen_path, label_free_frame)
    write_json(
        output / "VALIDATION_PREDICTIONS_FROZEN.json",
        {
            "frozen": True,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "prediction_file": str(frozen_path),
            "prediction_sha256": sha256_file(frozen_path),
            "subjects": int(len(validation.subject_ids)),
            "labels_accessed_before_freeze": False,
        },
    )

    # Validation-label access is intentionally below the frozen prediction marker.
    if not config.evaluate_validation_labels:
        return {
            "evaluated": False,
            "prediction_file": str(frozen_path),
            "warning": "Validation labels were intentionally not opened.",
        }
    validation_y = load_aligned_labels(validation_root, validation.subject_ids)
    metrics = metrics_from_probabilities(validation_y, probabilities)
    write_csv(
        output / "validation_predictions_evaluated_hashed.csv",
        _prediction_frame(
            validation.subject_ids,
            probabilities,
            y_true=validation_y,
        ),
    )
    report = {
        "evaluated": True,
        "metrics": metrics,
        "warning": (
            "The 33-subject Validation split has been used by earlier experiments. "
            "Treat it as a historical benchmark, never as a tuning source or fresh test."
        ),
    }
    write_json(output / "validation_report.json", report)
    return report


def run(config: RunConfig) -> None:
    output = prepare_output_dir(config.output_dir)
    write_json(output / "run_config.json", asdict(config))
    write_json(output / "environment.json", runtime_info())
    write_json(output / "code_manifest.json", code_manifest())

    training = build_subject_dataset(config.training_root, require_labels=True)
    if training.y is None:
        raise AssertionError("Training labels were not loaded")
    y = np.asarray(training.y, dtype=np.int64)
    write_json(output / "training_feature_audit.json", training.audit)
    write_json(
        output / "feature_manifest.json",
        {
            "feature_count": int(training.X.shape[1]),
            "feature_names": list(training.X.columns),
            "class_order": list(CLASS_NAMES),
            "subjects": int(len(training.subject_ids)),
            "class_counts": {
                name: int(np.sum(y == class_id))
                for class_id, name in enumerate(CLASS_NAMES)
            },
            "mmse_source_opened": False,
            "identifier_used_as_feature": False,
            "diagnosis_used_as_feature": False,
        },
    )

    nested_report, repeat_average = run_nested_cv(
        training.X,
        y,
        training.subject_ids,
        config,
        output,
    )
    bundle, model_manifest = fit_full_model(training.X, y, config, output)
    validation_report = None
    if config.validation_root is not None:
        validation_report = evaluate_validation(
            config.validation_root,
            bundle,
            training.subject_ids,
            config,
            output,
        )

    final_report = {
        "design_version": DESIGN_VERSION,
        "task": "MMSE-free subject-level CN/MCI/DEM classification",
        "model": "GaussianNB",
        "class_order": list(CLASS_NAMES),
        "primary_nested_repeat_summary": nested_report["repeat_summary"],
        "repeat_averaged_oof_metrics_supplemental": metrics_from_probabilities(
            y,
            repeat_average,
        ),
        "baselines": nested_report["baselines"],
        "full_model": model_manifest,
        "validation_historical": validation_report,
        "mmse_source_opened": False,
        "direct_label_leakage_blocked": True,
        "all_learned_preprocessing_is_fold_local": True,
        "fast_mode_not_for_performance_reporting": config.fast,
    }
    write_json(output / "FINAL_REPORT.json", final_report)
    write_json(
        output / "TRAINING_COMPLETE.json",
        {
            "success": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output),
        },
    )
    primary = nested_report["repeat_summary"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "macro_f1_mean": primary["macro_f1"]["mean"],
                "balanced_accuracy_mean": primary["balanced_accuracy"]["mean"],
                "roc_auc_ovr_macro_mean": primary["roc_auc_ovr_macro"]["mean"],
                "fast_mode": config.fast,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--validation-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument(
        "--outer-seeds",
        default=",".join(str(value) for value in DEFAULT_OUTER_SEEDS),
        help="Comma-separated repeat seeds (default: five preregistered seeds).",
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--skip-validation-labels",
        action="store_true",
        help="Write frozen Validation predictions without opening label files.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="One repeat and a tiny grid for code checks; not a performance run.",
    )
    args = parser.parse_args()
    try:
        outer_seeds = tuple(
            int(value.strip())
            for value in args.outer_seeds.split(",")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--outer-seeds must be comma-separated integers") from exc
    if not outer_seeds:
        raise ValueError("At least one outer seed is required")
    if len(set(outer_seeds)) != len(outer_seeds):
        raise ValueError("Outer seeds must be unique")
    if args.fast:
        outer_seeds = outer_seeds[:1]
    if args.n_jobs == 0 or args.n_jobs < -1:
        raise ValueError("--n-jobs must be -1 or a positive integer")
    return RunConfig(
        training_root=args.training_root,
        validation_root=args.validation_root,
        output_dir=args.output_dir,
        outer_folds=args.outer_folds,
        outer_seeds=outer_seeds,
        inner_folds=args.inner_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        fast=args.fast,
        evaluate_validation_labels=not args.skip_validation_labels,
    )


if __name__ == "__main__":
    run(parse_args())
