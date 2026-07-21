"""Nested subject-level training for the MMSE-free Google model ensemble.

This script is prepared for Colab A100 / High-RAM but is not executed while the
repository is authored.  Google YDF is CPU-oriented; the A100 is used by the
Google Research TabNet diversity candidate.
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

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from evaluation import (
    all_cn_baseline,
    apply_recipe,
    class_prior_baseline,
    metrics_from_probabilities,
    select_fixed_blend,
    selection_score,
    summarize_repeat_metrics,
)
from feature_engineering import (
    CLASS_NAMES,
    build_subject_dataset,
    discover_split_files,
    load_consistent_labels,
)
from models import (
    CANDIDATE_NAMES,
    FittedCandidate,
    fit_candidate,
    normalize_probabilities,
    set_global_seed,
    suggest_parameters,
)


DESIGN_VERSION = "google_ydf_cnboost_mmse_free_v1"
FRESH_OUTER_SEEDS = (137, 1009, 2027, 4099, 8191)
EXPECTED_TRAINING_COUNTS = (85, 47, 9)
HISTORICAL_LIFELOG_ONLY = {
    "accuracy": 0.5319148936170213,
    "macro_f1": 0.3579,
    "roc_auc_ovr_macro": 0.5300,
    "note": "Earlier 2-repeat x 3-fold lifelog-only development baseline; reference only.",
}
TARGETS = {
    "accuracy": 0.80,
    "roc_auc_ovr_macro": 0.80,
    "cn_vs_rest_auc": 0.85,
    "macro_f1": 0.65,
}


@dataclass(frozen=True)
class RunConfig:
    training_root: str
    validation_root: str | None
    output_dir: str
    outer_folds: int
    outer_seeds: tuple[int, ...]
    inner_folds: int
    trials_ydf_multiclass: int
    trials_ydf_hierarchical: int
    trials_ydf_random_forest: int
    trials_ydf_ovr: int
    trials_tabnet: int
    skip_tabnet: bool
    fit_full_checkpoint: bool
    seed: int
    fast: bool


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


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_indices(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def subject_hash(subject_id: object) -> str:
    payload = f"{DESIGN_VERSION}::{subject_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_info() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("ydf", "pytorch-tabnet", "optuna", "numpy", "pandas", "scikit-learn")
        },
    }
    try:
        import torch

        payload["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        payload["torch"] = None
    return payload


def code_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    files = (
        "feature_engineering.py",
        "preprocessing.py",
        "models.py",
        "evaluation.py",
        "train.py",
    )
    manifest = {name: sha256_file(root / name) for name in files}
    shared_core = root.parent / "ThreeClass_PerformanceLab" / "performance_lab_core.py"
    manifest["../ThreeClass_PerformanceLab/performance_lab_core.py"] = sha256_file(shared_core)
    return manifest


def input_manifest(split_root: str, *, include_labels: bool) -> dict[str, Any]:
    files = discover_split_files(split_root, require_labels=include_labels)
    paths = [files.activity, files.sleep, *files.labels]
    return {
        "split_root": str(files.root),
        "mmse_source_resolved": False,
        "files": [
            {
                "role": (
                    "activity"
                    if path == files.activity
                    else "sleep"
                    if path == files.sleep
                    else "diagnosis_label_copy"
                ),
                "path": str(path),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in paths
        ],
    }


def active_candidates(config: RunConfig) -> tuple[str, ...]:
    if config.skip_tabnet:
        return tuple(name for name in CANDIDATE_NAMES if name != "tabnet")
    return CANDIDATE_NAMES


def trial_count(config: RunConfig, candidate_name: str) -> int:
    return {
        "ydf_multiclass": config.trials_ydf_multiclass,
        "ydf_hierarchical": config.trials_ydf_hierarchical,
        "ydf_random_forest": config.trials_ydf_random_forest,
        "ydf_ovr": config.trials_ydf_ovr,
        "tabnet": config.trials_tabnet,
    }[candidate_name]


def validate_fold_contract(y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> None:
    for fold_id, (fit_indices, valid_indices) in enumerate(splits):
        if np.intersect1d(fit_indices, valid_indices).size:
            raise AssertionError(f"Subject overlap in fold {fold_id}")
        train_counts = np.bincount(y[fit_indices], minlength=3)
        valid_counts = np.bincount(y[valid_indices], minlength=3)
        if np.any(train_counts == 0) or np.any(valid_counts == 0):
            raise AssertionError(
                f"Every fold must contain all classes; train={train_counts}, valid={valid_counts}"
            )


def tune_candidate(
    candidate_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    config: RunConfig,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import optuna

    validate_fold_contract(y, splits)

    def objective(trial: Any) -> float:
        params = suggest_parameters(candidate_name, trial, fast=config.fast)
        oof = np.zeros((len(y), 3), dtype=np.float64)
        seen = np.zeros(len(y), dtype=np.int8)
        fold_metrics: list[dict[str, Any]] = []
        for fold_id, (fit_indices, valid_indices) in enumerate(splits):
            fitted = fit_candidate(
                candidate_name,
                X.iloc[fit_indices].reset_index(drop=True),
                y[fit_indices],
                params,
                seed + trial.number * 1009 + fold_id * 53,
            )
            probabilities = fitted.predict_proba(X.iloc[valid_indices])
            oof[valid_indices] = probabilities
            seen[valid_indices] += 1
            fold_metrics.append(metrics_from_probabilities(y[valid_indices], probabilities))
        if not np.all(seen == 1):
            raise AssertionError(f"Inner OOF coverage failed for {candidate_name}")
        metrics = metrics_from_probabilities(y, oof)
        score = selection_score(metrics)
        trial.set_user_attr("resolved_params", _jsonable(params))
        trial.set_user_attr("oof_metrics", _jsonable(metrics))
        trial.set_user_attr("fold_metrics", _jsonable(fold_metrics))
        return score

    sampler = optuna.samplers.TPESampler(seed=int(seed), multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=max(1, trial_count(config, candidate_name)),
        n_jobs=1,
        gc_after_trial=True,
        catch=(FloatingPointError, RuntimeError, ValueError),
        show_progress_bar=True,
    )
    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        raise RuntimeError(f"All tuning trials failed for {candidate_name}")
    best = dict(study.best_trial.user_attrs["resolved_params"])
    records = [
        {
            "candidate": candidate_name,
            "number": trial.number,
            "state": str(trial.state),
            "value": trial.value,
            "params": trial.user_attrs.get("resolved_params", trial.params),
            "oof_metrics": trial.user_attrs.get("oof_metrics"),
        }
        for trial in study.trials
    ]
    return best, records


def collect_oof(
    candidate_name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> np.ndarray:
    probabilities = np.zeros((len(y), 3), dtype=np.float64)
    seen = np.zeros(len(y), dtype=np.int8)
    for fold_id, (fit_indices, valid_indices) in enumerate(splits):
        fitted = fit_candidate(
            candidate_name,
            X.iloc[fit_indices].reset_index(drop=True),
            y[fit_indices],
            params,
            seed + fold_id * 97,
        )
        probabilities[valid_indices] = fitted.predict_proba(X.iloc[valid_indices])
        seen[valid_indices] += 1
    if not np.all(seen == 1):
        raise AssertionError(f"OOF coverage failed for {candidate_name}: {seen.tolist()}")
    return normalize_probabilities(probabilities)


def save_candidate_checkpoint(
    candidate: FittedCandidate,
    path: Path,
    identity: dict[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite candidate checkpoint: {path}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(
            f"An incomplete checkpoint already exists and needs inspection: {temporary}"
        )
    temporary.mkdir(parents=True, exist_ok=False)
    candidate.save(temporary)
    write_json(temporary / "checkpoint_identity.json", identity)
    write_json(
        temporary / "CHECKPOINT_COMPLETE.json",
        {
            "complete": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidate": candidate.name,
        },
    )
    temporary.replace(path)


def target_check(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        key: bool(np.isfinite(metrics[key]) and metrics[key] >= threshold)
        for key, threshold in TARGETS.items()
    }
    return {
        "thresholds": TARGETS,
        "checks": checks,
        "all_targets_met": bool(all(checks.values())),
        "note": "Targets are goals, not guaranteed outcomes.",
    }


def prediction_frame(
    subject_ids: np.ndarray,
    probabilities: np.ndarray,
    y_true: np.ndarray | None = None,
) -> pd.DataFrame:
    p = normalize_probabilities(probabilities)
    frame = pd.DataFrame(
        {
            "subject_hash": [subject_hash(value) for value in subject_ids],
            "predicted_class": [CLASS_NAMES[index] for index in p.argmax(axis=1)],
            "p_cn": p[:, 0],
            "p_mci": p[:, 1],
            "p_dem": p[:, 2],
        }
    )
    if y_true is not None:
        frame.insert(1, "true_class", [CLASS_NAMES[int(value)] for value in y_true])
    return frame


def fit_full_training_checkpoint(
    X: pd.DataFrame,
    y: np.ndarray,
    candidates: tuple[str, ...],
    config: RunConfig,
    output: Path,
    external_X: pd.DataFrame | None,
) -> dict[str, Any]:
    """Tune on Training-only inner OOF, then refit deployable full-data checkpoints."""

    full_dir = output / "models" / "full_training_refit"
    full_dir.mkdir(parents=True, exist_ok=False)
    cv = StratifiedKFold(
        n_splits=config.inner_folds,
        shuffle=True,
        random_state=config.seed + 700_001,
    )
    splits = list(cv.split(X, y))
    validate_fold_contract(y, splits)
    best_params: dict[str, dict[str, Any]] = {}
    oof_by_candidate: dict[str, np.ndarray] = {}
    trial_records: list[dict[str, Any]] = []
    for candidate_index, candidate_name in enumerate(candidates):
        candidate_seed = config.seed + 800_011 + candidate_index * 10_007
        params, records = tune_candidate(
            candidate_name,
            X,
            y,
            splits,
            config=config,
            seed=candidate_seed,
        )
        best_params[candidate_name] = params
        trial_records.extend(records)
        oof_by_candidate[candidate_name] = collect_oof(
            candidate_name,
            params,
            X,
            y,
            splits,
            seed=candidate_seed + 900_001,
        )
    blend = select_fixed_blend(y, oof_by_candidate)
    external_by_candidate: dict[str, np.ndarray] = {}
    for candidate_index, candidate_name in enumerate(candidates):
        fitted = fit_candidate(
            candidate_name,
            X.reset_index(drop=True),
            y,
            best_params[candidate_name],
            config.seed + 1_000_003 + candidate_index * 10_007,
        )
        save_candidate_checkpoint(
            fitted,
            full_dir / candidate_name,
            {
                "scope": "full Training refit",
                "candidate": candidate_name,
                "subjects": int(len(y)),
                "class_counts": np.bincount(y, minlength=3).tolist(),
                "design_version": DESIGN_VERSION,
                "code_manifest": code_manifest(),
                "feature_manifest_sha256": sha256_file(output / "feature_manifest.json"),
                "training_input_manifest_sha256": sha256_file(
                    output / "training_input_manifest.json"
                ),
                "ydf_version": package_version("ydf"),
                "pytorch_tabnet_version": package_version("pytorch-tabnet"),
            },
        )
        if external_X is not None:
            external_by_candidate[candidate_name] = fitted.predict_proba(external_X)
    full_external = (
        apply_recipe(external_by_candidate, blend["weights"])
        if external_by_candidate
        else None
    )
    selection = {
        "best_params": best_params,
        "training_only_oof_blend": blend,
        "training_only_oof_candidate_metrics": {
            name: metrics_from_probabilities(y, probabilities)
            for name, probabilities in oof_by_candidate.items()
        },
        "trial_records": trial_records,
        "external_probabilities": full_external,
    }
    write_json(
        full_dir / "full_selection.json",
        {key: value for key, value in selection.items() if key != "external_probabilities"},
    )
    write_json(
        full_dir / "FULL_CHECKPOINT_COMPLETE.json",
        {
            "complete": True,
            "candidates": list(candidates),
            "blend": blend,
        },
    )
    return selection


def run(config: RunConfig) -> None:
    output = Path(config.output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            "Output directory must be new so completed predictions/checkpoints cannot be overwritten: "
            f"{output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "run_config.json", asdict(config))
    write_json(output / "environment.json", runtime_info())
    write_json(output / "code_manifest.json", code_manifest())
    write_json(
        output / "training_input_manifest.json",
        input_manifest(config.training_root, include_labels=True),
    )
    set_global_seed(config.seed)

    train = build_subject_dataset(config.training_root, require_labels=True)
    if train.y is None:
        raise AssertionError("Training labels are unexpectedly absent")
    X = train.X
    y = train.y
    counts = tuple(np.bincount(y, minlength=3).tolist())
    if len(y) != 141 or counts != EXPECTED_TRAINING_COUNTS:
        raise AssertionError(
            f"Training cohort changed: n={len(y)}, class counts={counts}; "
            f"expected n=141, {EXPECTED_TRAINING_COUNTS}"
        )
    if config.outer_folds != 3 or config.inner_folds != 3:
        raise ValueError("The nine DEM subjects require the locked 3x3 nested-CV contract")
    write_json(output / "training_feature_audit.json", train.audit)
    write_json(
        output / "feature_manifest.json",
        {
            "design_version": DESIGN_VERSION,
            "feature_count": int(X.shape[1]),
            "feature_names": list(X.columns),
            "mmse_source_opened": False,
            "mmse_values_used": False,
            "coverage_or_calendar_protocol_features_used": False,
            "feature_builder": "ThreeClass_PerformanceLab event_summary_v1",
        },
    )

    external = None
    if config.validation_root:
        # No validation label path is resolved here.  Labels are opened only after
        # predictions and the frozen marker are on disk.
        external = build_subject_dataset(config.validation_root, require_labels=False)
        write_json(
            output / "validation_source_input_manifest_label_free.json",
            input_manifest(config.validation_root, include_labels=False),
        )
        if list(external.X.columns) != list(X.columns):
            missing = sorted(set(X.columns) - set(external.X.columns))
            extra = sorted(set(external.X.columns) - set(X.columns))
            raise AssertionError(
                f"Training/validation feature schema mismatch; missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )
        write_json(output / "validation_feature_audit_label_free.json", external.audit)

    candidates = active_candidates(config)
    repeat_predictions: list[np.ndarray] = []
    outer_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    external_sum = (
        np.zeros((len(external.subject_ids), 3), dtype=np.float64)
        if external is not None
        else None
    )
    external_model_count = 0

    outer_seeds = config.outer_seeds[:1] if config.fast else config.outer_seeds
    for repeat_index, outer_seed in enumerate(outer_seeds):
        outer_cv = StratifiedKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=int(outer_seed),
        )
        outer_splits = list(outer_cv.split(X, y))
        validate_fold_contract(y, outer_splits)
        repeat_oof = np.zeros((len(y), 3), dtype=np.float64)
        repeat_seen = np.zeros(len(y), dtype=np.int8)
        for outer_fold, (outer_train_indices, outer_valid_indices) in enumerate(outer_splits):
            fold_tag = f"repeat_{repeat_index:02d}_seed_{outer_seed}_fold_{outer_fold:02d}"
            fold_dir = output / "models" / fold_tag
            fold_dir.mkdir(parents=True, exist_ok=False)
            X_outer_train = X.iloc[outer_train_indices].reset_index(drop=True)
            y_outer_train = y[outer_train_indices]
            X_outer_valid = X.iloc[outer_valid_indices]
            y_outer_valid = y[outer_valid_indices]

            inner_seed = int(outer_seed) + outer_fold * 1009 + 17
            inner_cv = StratifiedKFold(
                n_splits=config.inner_folds,
                shuffle=True,
                random_state=inner_seed,
            )
            inner_splits = list(inner_cv.split(X_outer_train, y_outer_train))
            validate_fold_contract(y_outer_train, inner_splits)
            best_params: dict[str, dict[str, Any]] = {}
            inner_oof: dict[str, np.ndarray] = {}
            for candidate_index, candidate_name in enumerate(candidates):
                model_seed = inner_seed + candidate_index * 10_007
                params, trial_records = tune_candidate(
                    candidate_name,
                    X_outer_train,
                    y_outer_train,
                    inner_splits,
                    config=config,
                    seed=model_seed,
                )
                best_params[candidate_name] = params
                for record in trial_records:
                    record.update(
                        {
                            "repeat": repeat_index,
                            "outer_seed": outer_seed,
                            "outer_fold": outer_fold,
                        }
                    )
                tuning_rows.extend(trial_records)
                inner_oof[candidate_name] = collect_oof(
                    candidate_name,
                    params,
                    X_outer_train,
                    y_outer_train,
                    inner_splits,
                    seed=model_seed + 900_001,
                )

            blend = select_fixed_blend(y_outer_train, inner_oof)
            write_json(
                fold_dir / "selection.json",
                {
                    "best_params": best_params,
                    "inner_oof_candidate_metrics": {
                        name: metrics_from_probabilities(y_outer_train, probabilities)
                        for name, probabilities in inner_oof.items()
                    },
                    "inner_oof_fixed_blend": blend,
                },
            )

            outer_by_candidate: dict[str, np.ndarray] = {}
            external_by_candidate: dict[str, np.ndarray] = {}
            per_candidate_metrics: dict[str, Any] = {}
            for candidate_index, candidate_name in enumerate(candidates):
                fitted = fit_candidate(
                    candidate_name,
                    X_outer_train,
                    y_outer_train,
                    best_params[candidate_name],
                    int(outer_seed) + outer_fold * 1009 + candidate_index * 53,
                )
                probabilities = fitted.predict_proba(X_outer_valid)
                outer_by_candidate[candidate_name] = probabilities
                per_candidate_metrics[candidate_name] = metrics_from_probabilities(
                    y_outer_valid, probabilities
                )
                checkpoint_identity = {
                    "design_version": DESIGN_VERSION,
                    "repeat": repeat_index,
                    "outer_seed": outer_seed,
                    "outer_fold": outer_fold,
                    "candidate": candidate_name,
                    "train_index_sha256": hash_indices(outer_train_indices),
                    "valid_index_sha256": hash_indices(outer_valid_indices),
                    "class_order": list(CLASS_NAMES),
                    "code_manifest": code_manifest(),
                    "feature_manifest_sha256": sha256_file(
                        output / "feature_manifest.json"
                    ),
                    "training_input_manifest_sha256": sha256_file(
                        output / "training_input_manifest.json"
                    ),
                    "ydf_version": package_version("ydf"),
                    "pytorch_tabnet_version": package_version("pytorch-tabnet"),
                }
                save_candidate_checkpoint(
                    fitted,
                    fold_dir / candidate_name,
                    checkpoint_identity,
                )
                if external is not None:
                    external_by_candidate[candidate_name] = fitted.predict_proba(external.X)

            outer_probabilities = apply_recipe(outer_by_candidate, blend["weights"])
            repeat_oof[outer_valid_indices] = outer_probabilities
            repeat_seen[outer_valid_indices] += 1
            fold_metrics = metrics_from_probabilities(y_outer_valid, outer_probabilities)
            row = {
                "repeat": repeat_index,
                "outer_seed": outer_seed,
                "outer_fold": outer_fold,
                "n_train": int(len(outer_train_indices)),
                "n_valid": int(len(outer_valid_indices)),
                "train_class_counts": np.bincount(y_outer_train, minlength=3).tolist(),
                "valid_class_counts": np.bincount(y_outer_valid, minlength=3).tolist(),
                "blend": blend,
                "metrics": fold_metrics,
                "per_candidate_metrics": per_candidate_metrics,
            }
            outer_rows.append(row)
            write_json(fold_dir / "fold_report.json", row)
            write_json(
                fold_dir / "FOLD_COMPLETE.json",
                {"complete": True, "checkpoint_candidates": list(candidates)},
            )
            if external is not None and external_sum is not None:
                external_sum += apply_recipe(external_by_candidate, blend["weights"])
                external_model_count += 1

            write_json(output / "nested_progress.json", {"completed_folds": outer_rows})
            write_json(output / "tuning_trials.json", tuning_rows)
            pd.DataFrame(
                [
                    {
                        "repeat": item["repeat"],
                        "outer_seed": item["outer_seed"],
                        "outer_fold": item["outer_fold"],
                        "n_train": item["n_train"],
                        "n_valid": item["n_valid"],
                        "blend": item["blend"]["chosen_name"],
                        **{
                            key: item["metrics"][key]
                            for key in (
                                "accuracy",
                                "balanced_accuracy",
                                "macro_f1",
                                "roc_auc_ovr_macro",
                                "cn_vs_rest_auc",
                                "non_cn_recall",
                                "log_loss",
                            )
                        },
                    }
                    for item in outer_rows
                ]
            ).to_csv(output / "outer_fold_metrics.csv", index=False)

        if not np.all(repeat_seen == 1):
            raise AssertionError(
                f"Outer repeat {repeat_index} OOF coverage mismatch: {repeat_seen.tolist()}"
            )
        repeat_predictions.append(normalize_probabilities(repeat_oof))

    repeat_metrics = [metrics_from_probabilities(y, values) for values in repeat_predictions]
    repeat_average_oof = normalize_probabilities(np.mean(repeat_predictions, axis=0))
    repeat_average_metrics = metrics_from_probabilities(y, repeat_average_oof)
    oof_frame = prediction_frame(train.subject_ids, repeat_average_oof, y)
    oof_frame.to_csv(output / "nested_repeat_averaged_oof_hashed.csv", index=False)
    for repeat_index, probabilities in enumerate(repeat_predictions):
        prediction_frame(train.subject_ids, probabilities, y).to_csv(
            output / f"nested_oof_repeat_{repeat_index:02d}_hashed.csv",
            index=False,
        )
    write_json(output / "tuning_trials.json", tuning_rows)
    nested_report = {
        "primary_repeat_metrics": repeat_metrics,
        "primary_repeat_summary": summarize_repeat_metrics(repeat_metrics),
        "repeat_averaged_oof_supplemental": repeat_average_metrics,
        "target_check_repeat_averaged_oof": target_check(repeat_average_metrics),
        "class_prior_baseline": class_prior_baseline(y),
        "all_cn_baseline": all_cn_baseline(y),
        "historical_lifelog_only_reference": HISTORICAL_LIFELOG_ONLY,
        "outer_folds": outer_rows,
        "interpretation": (
            "Repeat means/std are primary. Repeat-averaged OOF is supplemental. "
            "This is development evidence from 141 subjects, including only nine DEM subjects."
        ),
    }
    write_json(output / "nested_cv_report.json", nested_report)

    full_selection = None
    if config.fit_full_checkpoint:
        full_selection = fit_full_training_checkpoint(
            X,
            y,
            candidates,
            config,
            output,
            external.X if external is not None else None,
        )

    validation_report = None
    if external is not None and external_sum is not None:
        expected_models = len(outer_seeds) * config.outer_folds
        if external_model_count != expected_models:
            raise AssertionError(
                f"Validation fold-ensemble count={external_model_count}, expected={expected_models}"
            )
        external_probabilities = normalize_probabilities(external_sum / external_model_count)
        label_free_path = output / "validation_predictions_label_free_hashed.csv"
        prediction_frame(external.subject_ids, external_probabilities).to_csv(
            label_free_path, index=False
        )
        if full_selection is not None and full_selection["external_probabilities"] is not None:
            prediction_frame(
                external.subject_ids,
                full_selection["external_probabilities"],
            ).to_csv(output / "validation_predictions_full_refit_label_free_hashed.csv", index=False)
        write_json(
            output / "VALIDATION_PREDICTIONS_FROZEN.json",
            {
                "prediction_file": label_free_path.name,
                "rows": int(len(external.subject_ids)),
                "labels_opened_before_prediction_write": False,
                "selection_or_tuning_used_validation_labels": False,
                "benchmark_role": "historical, already reused; never a model-selection source",
            },
        )

        # The first validation-label access occurs only below the frozen marker.
        validation_files = discover_split_files(config.validation_root, require_labels=True)
        validation_labels = load_consistent_labels(validation_files.labels)
        aligned_labels = validation_labels.reindex(external.subject_ids)
        if aligned_labels.isna().any():
            raise AssertionError("Validation source subject without a diagnosis label")
        aligned_y = aligned_labels.map({name: index for index, name in enumerate(CLASS_NAMES)})
        aligned_y_array = aligned_y.to_numpy(dtype=np.int64)
        validation_metrics = metrics_from_probabilities(aligned_y_array, external_probabilities)
        validation_report = {
            "metrics": validation_metrics,
            "target_check": target_check(validation_metrics),
            "warning": (
                "This 33-subject validation set has been evaluated in previous experiments. "
                "It is a historical benchmark, not fresh evidence and not a selection source."
            ),
        }
        prediction_frame(external.subject_ids, external_probabilities, aligned_y_array).to_csv(
            output / "validation_predictions_evaluated_hashed.csv", index=False
        )
        write_json(output / "validation_report.json", validation_report)

    final_report = {
        "design_version": DESIGN_VERSION,
        "task": "MMSE-free subject-level CN/MCI/DEM classification",
        "primary_nested_repeat_summary": nested_report["primary_repeat_summary"],
        "nested_repeat_averaged_oof_supplemental": repeat_average_metrics,
        "validation_historical": validation_report,
        "candidates": list(candidates),
        "google_first": True,
        "mmse_source_opened": False,
        "mmse_values_used": False,
        "direct_label_leakage_blocked": True,
        "coverage_shortcut_blocked": True,
        "full_training_checkpoint_written": bool(config.fit_full_checkpoint),
        "target_check_nested_supplemental": target_check(repeat_average_metrics),
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


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--validation-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument(
        "--outer-seeds",
        default=",".join(str(value) for value in FRESH_OUTER_SEEDS),
        help="Comma-separated fresh seeds; default is the predeclared five-seed set.",
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--trials-ydf-multiclass", type=int, default=36)
    parser.add_argument("--trials-ydf-hierarchical", type=int, default=36)
    parser.add_argument("--trials-ydf-random-forest", type=int, default=24)
    parser.add_argument("--trials-ydf-ovr", type=int, default=24)
    parser.add_argument("--trials-tabnet", type=int, default=20)
    parser.add_argument("--skip-tabnet", action="store_true")
    parser.add_argument("--no-full-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    try:
        outer_seeds = tuple(int(value.strip()) for value in args.outer_seeds.split(",") if value.strip())
    except ValueError as exc:
        raise ValueError("--outer-seeds must be comma-separated integers") from exc
    if not outer_seeds:
        raise ValueError("At least one outer seed is required")
    if len(set(outer_seeds)) != len(outer_seeds):
        raise ValueError("Outer seeds must be unique")
    if args.fast:
        args.trials_ydf_multiclass = min(args.trials_ydf_multiclass, 1)
        args.trials_ydf_hierarchical = min(args.trials_ydf_hierarchical, 1)
        args.trials_ydf_random_forest = min(args.trials_ydf_random_forest, 1)
        args.trials_ydf_ovr = min(args.trials_ydf_ovr, 1)
        args.trials_tabnet = min(args.trials_tabnet, 1)
    return RunConfig(
        training_root=args.training_root,
        validation_root=args.validation_root,
        output_dir=args.output_dir,
        outer_folds=args.outer_folds,
        outer_seeds=outer_seeds,
        inner_folds=args.inner_folds,
        trials_ydf_multiclass=args.trials_ydf_multiclass,
        trials_ydf_hierarchical=args.trials_ydf_hierarchical,
        trials_ydf_random_forest=args.trials_ydf_random_forest,
        trials_ydf_ovr=args.trials_ydf_ovr,
        trials_tabnet=args.trials_tabnet,
        skip_tabnet=args.skip_tabnet,
        fit_full_checkpoint=not args.no_full_checkpoint,
        seed=args.seed,
        fast=args.fast,
    )


if __name__ == "__main__":
    run(parse_args())
