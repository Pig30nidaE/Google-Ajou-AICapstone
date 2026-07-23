"""Nested subject-level CV, checkpoint bagging, and frozen evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .data import build_subject_dataset, load_binary_labels
from .eda import run_eda
from .features import SubjectFeatureTable, build_subject_feature_table
from .models import (
    CLASS_NAMES,
    GOOGLE_MODEL_EVIDENCE,
    MODEL_NAMES,
    PlattCalibrator,
    TabNetAdapter,
    YDFAdapter,
    fit_tabnet,
    fit_ydf,
    set_global_seed,
    suggest_parameters,
)
from .preprocessing import FoldPreprocessor, schema_sha256


TARGET_ACCURACY = 0.90
EXPECTED_DAILY_FEATURES = 119
EXPECTED_SUBJECT_FEATURES = 1_077
EXPECTED_CORRELATION_PAIRS = 6
TUNING_RESERVE_SECONDS = 1_800
CALIBRATION_RESERVE_SECONDS = 1_200
OUTER_COMPLETION_RESERVE_SECONDS = 600
FULL_REFIT_MINIMUM_SECONDS = 900
SMOKE_RESERVE_SECONDS = 60
HISTORICAL_VALIDATION_WARNING = (
    "The 33-person Validation split has been inspected by earlier experiments. "
    "It is a historical benchmark, not a fresh independent test set."
)


@dataclass(frozen=True)
class RunConfig:
    training_root: str
    validation_root: str
    output_dir: str
    run_mode: str = "full"
    seed: int = 20260723
    outer_folds: int = 5
    outer_repeats: int = 2
    inner_folds: int = 3
    trials_tabnet: int = 4
    trials_ydf: int = 4
    tabnet_seeds: int = 3
    max_runtime_seconds: int = 20_400
    evaluate_historical_validation: bool = True


class SoftTimeBudgetExceeded(TimeoutError):
    """Raised only by this module's cooperative pre-step deadline checks."""


def _reserve_for_mode(full_reserve_seconds: float, *, smoke: bool) -> float:
    """Keep production safety margins without exhausting the 30-minute smoke budget."""

    return float(min(full_reserve_seconds, SMOKE_RESERVE_SECONDS)) if smoke else float(
        full_reserve_seconds
    )


def _ensure_time_budget(
    deadline: float,
    *,
    reserve_seconds: float,
    next_step: str,
) -> None:
    remaining = float(deadline) - time.monotonic()
    if remaining <= float(reserve_seconds):
        raise SoftTimeBudgetExceeded(
            f"Not enough soft-budget time before {next_step}: remaining={remaining:.1f}s, "
            f"required reserve={reserve_seconds:.1f}s"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_hash(subject_id: str) -> str:
    return hashlib.sha256(
        f"binary-wearable-tabnet-v1::{subject_id}".encode("utf-8")
    ).hexdigest()[:20]


def runtime_info() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "pytorch-tabnet",
        "ydf",
        "torch",
    ):
        try:
            payload[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            payload[distribution] = None
    try:
        import torch

        payload.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        payload["cuda_available"] = False
    return payload


def code_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files: dict[str, Any] = {}
    for path in sorted(root.glob("*.py")):
        files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    shared_root = root.parent / "Binary_Wearable_SequenceFusion_Google"
    shared_dependencies = {}
    for filename in ("data.py", "eda.py"):
        path = shared_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Audited wearable dependency is missing: {path}")
        shared_dependencies[str(path.relative_to(root.parents[1]))] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return {
        "experiment_root": str(root),
        "files": files,
        "transitive_shared_dependencies": shared_dependencies,
    }


def _safe_metric(function, *args, **kwargs) -> float:
    try:
        return float(function(*args, **kwargs))
    except ValueError:
        return float("nan")


def binary_metrics(
    y_true: np.ndarray,
    p_impaired: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    target = np.asarray(y_true, dtype=np.int64)
    probability = np.clip(np.asarray(p_impaired, dtype=np.float64), 1e-7, 1 - 1e-7)
    if target.shape != probability.shape:
        raise ValueError("Targets and probabilities have different shapes")
    prediction = (probability >= float(threshold)).astype(np.int64)
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]
    report = classification_report(
        target,
        prediction,
        labels=[0, 1],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "f1_impaired": float(f1_score(target, prediction, zero_division=0)),
        "precision_impaired": float(precision_score(target, prediction, zero_division=0)),
        "recall_impaired": float(recall_score(target, prediction, zero_division=0)),
        "specificity_cn": float(tn / max(1, tn + fp)),
        "roc_auc": _safe_metric(roc_auc_score, target, probability),
        "pr_auc": _safe_metric(average_precision_score, target, probability),
        "log_loss": _safe_metric(
            log_loss,
            target,
            np.column_stack([1.0 - probability, probability]),
            labels=[0, 1],
        ),
        "brier_score": float(brier_score_loss(target, probability)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "support": {"CN": int(np.sum(target == 0)), "MCI_DEM": int(np.sum(target == 1))},
        "per_class": {name: report[name] for name in CLASS_NAMES},
    }


def selection_score(metrics: Mapping[str, Any]) -> float:
    auc = float(metrics["roc_auc"])
    if not np.isfinite(auc):
        auc = 0.5
    return float(
        0.35 * float(metrics["accuracy"])
        + 0.25 * float(metrics["balanced_accuracy"])
        + 0.25 * auc
        + 0.15 * float(metrics["f1_impaired"])
    )


def _target_check(metrics: Mapping[str, Any]) -> dict[str, Any]:
    accuracy = float(metrics["accuracy"])
    return {
        "target_accuracy": TARGET_ACCURACY,
        "accuracy": accuracy,
        "met": bool(np.isfinite(accuracy) and accuracy >= TARGET_ACCURACY),
        "note": "0.90 is a goal, not a guaranteed result.",
    }


def _all_cn_baseline(y: np.ndarray) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int64)
    return {
        "accuracy": float(np.mean(target == 0)),
        "correct": int(np.sum(target == 0)),
        "total": int(len(target)),
        "warning": "Accuracy alone can look high on the imbalanced Validation split.",
    }


def _release_model_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _fit_candidate(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    params: dict[str, Any],
    *,
    seed: int,
    device_name: str,
    smoke: bool,
) -> tuple[FoldPreprocessor, Any]:
    preprocessor = FoldPreprocessor(
        max_features=int(params["max_features"]),
        bootstrap_rounds=4 if smoke else 20,
        minimum_per_modality=6 if smoke else 12,
        seed=int(seed) + 31,
    )
    transformed = preprocessor.fit_transform(X, y)
    if model_name == "tabnet":
        model = fit_tabnet(
            transformed,
            y,
            params,
            seed=seed,
            device_name=device_name,
        )
    elif model_name == "ydf":
        model = fit_ydf(transformed, y, params, seed=seed)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return preprocessor, model


def _positive_probability(model: Any, transformed: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(transformed), dtype=np.float64)[:, 1]


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    trials: int,
    seed: int,
    device_name: str,
    smoke: bool,
    deadline: float,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    import optuna

    tuning_reserve = _reserve_for_mode(TUNING_RESERVE_SECONDS, smoke=smoke)

    def objective(trial) -> float:
        params = suggest_parameters(model_name, trial, smoke=smoke)
        probability = np.zeros(len(y), dtype=np.float64)
        seen = np.zeros(len(y), dtype=np.int8)
        for fold_id, (fit_indices, valid_indices) in enumerate(splits):
            _ensure_time_budget(
                deadline,
                reserve_seconds=tuning_reserve,
                next_step=f"{model_name} tuning trial={trial.number} inner_fold={fold_id}",
            )
            preprocessor = fitted = transformed = None
            try:
                preprocessor, fitted = _fit_candidate(
                    model_name,
                    X.iloc[fit_indices].reset_index(drop=True),
                    y[fit_indices],
                    params,
                    seed=seed + trial.number * 1009 + fold_id * 97,
                    device_name=device_name,
                    smoke=smoke,
                )
                transformed = preprocessor.transform(X.iloc[valid_indices])
                probability[valid_indices] = _positive_probability(fitted, transformed)
                seen[valid_indices] += 1
            finally:
                preprocessor = fitted = transformed = None
                _release_model_memory()
        if not np.all(seen == 1):
            raise AssertionError(f"Inner coverage failed for {model_name}: {seen.tolist()}")
        metrics = binary_metrics(y, probability)
        score = selection_score(metrics)
        trial.set_user_attr("resolved_params", _jsonable(params))
        trial.set_user_attr("metrics", _jsonable(metrics))
        return score

    _ensure_time_budget(
        deadline,
        reserve_seconds=tuning_reserve,
        next_step=f"{model_name} tuning study",
    )
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=True,
        n_startup_trials=min(2, max(1, int(trials))),
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=max(1, int(trials)),
        timeout=max(
            1.0,
            deadline - time.monotonic() - tuning_reserve,
        ),
        n_jobs=1,
        gc_after_trial=True,
        catch=(FloatingPointError, RuntimeError, ValueError),
        show_progress_bar=True,
    )
    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        raise RuntimeError(f"All {model_name} tuning trials failed")
    best = study.best_trial
    best_params = dict(best.user_attrs["resolved_params"])
    best_summary = {
        "score": float(best.value),
        "trial": int(best.number),
        "metrics": best.user_attrs["metrics"],
    }
    records = [
        {
            "model": model_name,
            "number": int(trial.number),
            "state": str(trial.state),
            "value": trial.value,
            "params": trial.user_attrs.get("resolved_params", trial.params),
            "metrics": trial.user_attrs.get("metrics"),
        }
        for trial in study.trials
    ]
    return best_params, best_summary, records


def collect_inner_oof(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
    *,
    seed: int,
    device_name: str,
    smoke: bool,
    ensemble_seeds: int,
    deadline: float,
) -> np.ndarray:
    if ensemble_seeds < 1:
        raise ValueError("ensemble_seeds must be positive")
    probability = np.zeros(len(y), dtype=np.float64)
    seen = np.zeros(len(y), dtype=np.int8)
    calibration_reserve = _reserve_for_mode(
        CALIBRATION_RESERVE_SECONDS, smoke=smoke
    )
    for fold_id, (fit_indices, valid_indices) in enumerate(splits):
        _ensure_time_budget(
            deadline,
            reserve_seconds=calibration_reserve,
            next_step=f"{model_name} calibration OOF inner_fold={fold_id}",
        )
        fold_seed = seed + fold_id * 193
        preprocessor = transformed_fit = transformed = fitted = None
        seed_probabilities: list[np.ndarray] = []
        fitted_models: list[Any] = []
        try:
            preprocessor = FoldPreprocessor(
                max_features=int(params["max_features"]),
                bootstrap_rounds=4 if smoke else 20,
                minimum_per_modality=6 if smoke else 12,
                seed=fold_seed + 31,
            )
            transformed_fit = preprocessor.fit_transform(
                X.iloc[fit_indices].reset_index(drop=True), y[fit_indices]
            )
            transformed = preprocessor.transform(X.iloc[valid_indices])
            if model_name == "tabnet":
                for seed_index in range(int(ensemble_seeds)):
                    _ensure_time_budget(
                        deadline,
                        reserve_seconds=calibration_reserve,
                        next_step=(
                            f"TabNet calibration OOF inner_fold={fold_id} seed={seed_index}"
                        ),
                    )
                    fitted = fit_tabnet(
                        transformed_fit,
                        y[fit_indices],
                        params,
                        seed=fold_seed + seed_index * 10_007,
                        device_name=device_name,
                    )
                    fitted_models.append(fitted)
                    seed_probabilities.append(_positive_probability(fitted, transformed))
            elif model_name == "ydf":
                if ensemble_seeds != 1:
                    raise ValueError("YDF inner OOF uses exactly one fitted model")
                fitted = fit_ydf(
                    transformed_fit,
                    y[fit_indices],
                    params,
                    seed=fold_seed,
                )
                fitted_models.append(fitted)
                seed_probabilities.append(_positive_probability(fitted, transformed))
            else:
                raise ValueError(f"Unknown model: {model_name}")
            probability[valid_indices] = np.mean(seed_probabilities, axis=0)
            seen[valid_indices] += 1
        finally:
            fitted = preprocessor = transformed = transformed_fit = None
            fitted_models.clear()
            seed_probabilities.clear()
            _release_model_memory()
    if not np.all(seen == 1):
        raise AssertionError(f"Inner OOF coverage failed for {model_name}: {seen.tolist()}")
    return probability


def select_calibration_and_blend(
    y: np.ndarray,
    raw_probabilities: Mapping[str, np.ndarray],
) -> tuple[dict[str, PlattCalibrator], dict[str, Any]]:
    calibrators = {
        name: PlattCalibrator().fit(raw_probabilities[name], y) for name in MODEL_NAMES
    }
    calibrated = {
        name: calibrators[name].transform(raw_probabilities[name]) for name in MODEL_NAMES
    }
    rows: list[dict[str, Any]] = []
    # The experiment is explicitly TabNet-led.  YDF may add diversity, but a
    # selected primary prediction always retains at least 25% TabNet weight.
    for tabnet_weight in (0.25, 0.50, 0.75, 1.00):
        probability = (
            tabnet_weight * calibrated["tabnet"]
            + (1.0 - tabnet_weight) * calibrated["ydf"]
        )
        metrics = binary_metrics(y, probability)
        rows.append(
            {
                "tabnet_weight": float(tabnet_weight),
                "ydf_weight": float(1.0 - tabnet_weight),
                "score": selection_score(metrics),
                "metrics": metrics,
            }
        )
    maximum = max(row["score"] for row in rows)
    near_best = [row for row in rows if row["score"] >= maximum - 0.005]
    chosen = max(
        near_best,
        key=lambda row: (
            row["tabnet_weight"],
            -row["metrics"]["brier_score"],
        ),
    )
    return calibrators, {
        "selection_scope": "current outer-training inner OOF only",
        "near_best_tolerance": 0.005,
        "chosen": chosen,
        "grid": rows,
        "calibrators": {name: calibrators[name].reason for name in MODEL_NAMES},
        "per_model_raw_metrics": {
            name: binary_metrics(y, raw_probabilities[name]) for name in MODEL_NAMES
        },
        "per_model_calibrated_metrics": {
            name: binary_metrics(y, calibrated[name]) for name in MODEL_NAMES
        },
    }


def apply_blend(
    calibrated_probabilities: Mapping[str, np.ndarray],
    blend: Mapping[str, Any],
) -> np.ndarray:
    chosen = blend["chosen"]
    return np.clip(
        float(chosen["tabnet_weight"]) * calibrated_probabilities["tabnet"]
        + float(chosen["ydf_weight"]) * calibrated_probabilities["ydf"],
        1e-7,
        1 - 1e-7,
    )


def _checkpoint_file_manifest(root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root))
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return files


def verify_checkpoint_tree(
    root: Path,
    *,
    expected_outer_bundles: int,
    tabnet_seeds_per_bundle: int,
    require_full_refit: bool,
) -> dict[str, Any]:
    """Fail closed unless every expected model bundle is complete and indexed."""

    outer = sorted(path for path in root.glob("repeat_*_fold_*") if path.is_dir())
    full = root / "full_refit"
    full_exists = full.is_dir()
    partial_bundles = sorted(path.name for path in root.glob("*.partial"))
    if partial_bundles:
        raise AssertionError(
            f"Incomplete checkpoint directories remain: {partial_bundles}"
        )
    bundles = [*outer, *([full] if full_exists else [])]
    if len(outer) != int(expected_outer_bundles) or (
        require_full_refit and not full_exists
    ):
        raise AssertionError(
            f"Checkpoint bundle count mismatch: outer={len(outer)}, full={full_exists}"
        )
    rows = []
    for bundle in bundles:
        complete = bundle / "CHECKPOINT_COMPLETE.json"
        manifest = bundle / "checkpoint_manifest.json"
        tabnet_files = sorted(bundle.glob("tabnet_seed_*/model.zip"))
        ydf_metadata = bundle / "ydf/adapter.json"
        if not complete.is_file() or not manifest.is_file() or not ydf_metadata.is_file():
            raise AssertionError(f"Incomplete checkpoint bundle: {bundle}")
        if len(tabnet_files) != int(tabnet_seeds_per_bundle):
            raise AssertionError(
                f"{bundle.name} has {len(tabnet_files)} TabNet files; "
                f"expected {tabnet_seeds_per_bundle}"
            )
        complete_payload = json.loads(complete.read_text(encoding="utf-8"))
        if complete_payload.get("complete") is not True or complete_payload.get(
            "roundtrip_verified"
        ) is not True:
            raise AssertionError(f"Checkpoint completion marker is invalid: {bundle}")
        actual_manifest_sha = sha256_file(manifest)
        if complete_payload.get("manifest_sha256") != actual_manifest_sha:
            raise AssertionError(f"Checkpoint manifest hash mismatch: {bundle}")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        expected_files = set(manifest_payload["files_before_manifest"]) | {
            "checkpoint_manifest.json",
            "CHECKPOINT_COMPLETE.json",
        }
        actual_files = {
            str(stored.relative_to(bundle))
            for stored in bundle.rglob("*")
            if stored.is_file()
        }
        if actual_files != expected_files:
            raise AssertionError(
                f"Checkpoint file set mismatch in {bundle}: "
                f"missing={sorted(expected_files - actual_files)}, "
                f"unexpected={sorted(actual_files - expected_files)}"
            )
        for relative, expected in manifest_payload["files_before_manifest"].items():
            stored = bundle / relative
            if not stored.is_file():
                raise AssertionError(f"Checkpoint file is missing: {stored}")
            if stored.stat().st_size != int(expected["bytes"]):
                raise AssertionError(f"Checkpoint file size mismatch: {stored}")
            if sha256_file(stored) != expected["sha256"]:
                raise AssertionError(f"Checkpoint file hash mismatch: {stored}")
        rows.append(
            {
                "bundle": bundle.name,
                "scope": "full_refit" if bundle == full else "outer_fold",
                "tabnet_model_zips": [str(path.relative_to(bundle)) for path in tabnet_files],
                "ydf_adapter": str(ydf_metadata.relative_to(bundle)),
                "complete_sha256": sha256_file(complete),
                "manifest_sha256": actual_manifest_sha,
                "roundtrip_verified": True,
            }
        )
    return {
        "verified": True,
        "outer_bundle_count": len(outer),
        "full_refit_bundle_count": int(full_exists),
        "full_refit_required": bool(require_full_refit),
        "tabnet_seeds_per_bundle": int(tabnet_seeds_per_bundle),
        "bundles": rows,
    }


def save_checkpoint_bundle(
    path: Path,
    *,
    preprocessors: Mapping[str, FoldPreprocessor],
    tabnet_models: list[TabNetAdapter],
    ydf_model: YDFAdapter,
    calibrators: Mapping[str, PlattCalibrator],
    selection: Mapping[str, Any],
    probe: pd.DataFrame,
) -> dict[str, Any]:
    """Atomically save and CPU-reload every fitted checkpoint."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Incomplete checkpoint requires inspection: {temporary}")
    temporary.mkdir(parents=True, exist_ok=False)
    joblib.dump(dict(preprocessors), temporary / "preprocessors.joblib")
    joblib.dump(dict(calibrators), temporary / "calibrators.joblib")
    write_json(temporary / "selection.json", selection)

    tabnet_probe = preprocessors["tabnet"].transform(probe)
    expected_tabnet = [model.predict_proba(tabnet_probe) for model in tabnet_models]
    for seed_index, model in enumerate(tabnet_models):
        model.save(temporary / f"tabnet_seed_{seed_index:02d}" / "model")
    ydf_probe = preprocessors["ydf"].transform(probe)
    expected_ydf = ydf_model.predict_proba(ydf_probe)
    ydf_model.save(temporary / "ydf")

    loaded_preprocessors = joblib.load(temporary / "preprocessors.joblib")
    loaded_calibrators = joblib.load(temporary / "calibrators.joblib")
    reloaded_tabnet_probe = loaded_preprocessors["tabnet"].transform(probe)
    maximum_difference = 0.0
    for seed_index, expected in enumerate(expected_tabnet):
        loaded = TabNetAdapter.load(
            temporary / f"tabnet_seed_{seed_index:02d}" / "model.zip",
            device_name="cpu",
        )
        observed = loaded.predict_proba(reloaded_tabnet_probe)
        maximum_difference = max(maximum_difference, float(np.max(np.abs(expected - observed))))
        if not np.allclose(expected, observed, rtol=2e-4, atol=2e-5):
            raise AssertionError(f"TabNet seed {seed_index} checkpoint round-trip mismatch")
        del loaded
    loaded_ydf = YDFAdapter.load(temporary / "ydf")
    observed_ydf = loaded_ydf.predict_proba(loaded_preprocessors["ydf"].transform(probe))
    maximum_difference = max(maximum_difference, float(np.max(np.abs(expected_ydf - observed_ydf))))
    if not np.allclose(expected_ydf, observed_ydf, rtol=1e-8, atol=1e-8):
        raise AssertionError("YDF checkpoint round-trip mismatch")
    for name in MODEL_NAMES:
        before = calibrators[name].transform(np.asarray([0.2, 0.5, 0.8]))
        after = loaded_calibrators[name].transform(np.asarray([0.2, 0.5, 0.8]))
        if not np.allclose(before, after, rtol=0.0, atol=1e-12):
            raise AssertionError(f"{name} calibrator checkpoint mismatch")

    verification = {
        "fresh_cpu_reload": True,
        "tabnet_models_checked": len(tabnet_models),
        "ydf_models_checked": 1,
        "probe_subjects": int(len(probe)),
        "maximum_absolute_probability_difference": maximum_difference,
    }
    write_json(temporary / "roundtrip_verification.json", verification)
    files = _checkpoint_file_manifest(temporary)
    write_json(
        temporary / "checkpoint_manifest.json",
        {
            "files_before_manifest": files,
            "feature_schemas": {
                name: schema_sha256(preprocessor.selected_feature_names)
                for name, preprocessor in preprocessors.items()
            },
        },
    )
    write_json(
        temporary / "CHECKPOINT_COMPLETE.json",
        {
            "complete": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": sha256_file(temporary / "checkpoint_manifest.json"),
            "roundtrip_verified": True,
        },
    )
    temporary.replace(path)
    _release_model_memory()
    return verification


def _select_threshold(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    rows = []
    for threshold in np.round(np.arange(0.25, 0.751, 0.01), 2):
        metrics = binary_metrics(y, probability, float(threshold))
        score = (
            0.60 * metrics["accuracy"]
            + 0.25 * metrics["balanced_accuracy"]
            + 0.15 * metrics["f1_impaired"]
            - 0.005 * abs(float(threshold) - 0.5)
        )
        rows.append({"threshold": float(threshold), "score": float(score), "metrics": metrics})
    eligible = [
        row
        for row in rows
        if row["metrics"]["recall_impaired"] >= 0.40
        and row["metrics"]["specificity_cn"] >= 0.60
    ]
    candidates = eligible or rows
    chosen = max(
        candidates,
        key=lambda row: (
            row["score"],
            row["metrics"]["balanced_accuracy"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return {
        "threshold": float(chosen["threshold"]),
        "chosen": chosen,
        "grid": rows,
        "selection_scope": "repeat-averaged Training outer OOF only",
        "primary_metric_policy": "OOF at 0.5 remains primary; selected threshold is secondary",
        "constraints_met": bool(eligible),
    }


def _bootstrap_confidence_intervals(
    y: np.ndarray,
    probability: np.ndarray,
    *,
    seed: int,
    rounds: int = 1000,
) -> dict[str, Any]:
    target = np.asarray(y, dtype=np.int64)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(target == class_id) for class_id in (0, 1)]
    values = {name: [] for name in ("accuracy", "balanced_accuracy", "roc_auc")}
    for _ in range(int(rounds)):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        metrics = binary_metrics(target[sampled], probability[sampled])
        for name in values:
            values[name].append(metrics[name])
    return {
        name: {
            "lower_95": float(np.nanquantile(metric_values, 0.025)),
            "median": float(np.nanmedian(metric_values)),
            "upper_95": float(np.nanquantile(metric_values, 0.975)),
        }
        for name, metric_values in values.items()
    }


def _feature_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        sets = [
            set(row["features"])
            for row in rows
            if row["model"] == model_name
        ]
        jaccards = []
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                union = sets[left] | sets[right]
                jaccards.append(len(sets[left] & sets[right]) / max(1, len(union)))
        frequencies: dict[str, int] = {}
        for selected in sets:
            for name in selected:
                frequencies[name] = frequencies.get(name, 0) + 1
        result[model_name] = {
            "folds": len(sets),
            "mean_pairwise_jaccard": float(np.mean(jaccards)) if jaccards else None,
            "median_pairwise_jaccard": float(np.median(jaccards)) if jaccards else None,
            "features_in_every_fold": sorted(
                name for name, count in frequencies.items() if count == len(sets)
            ),
            "top_frequency": sorted(
                (
                    {"feature": name, "fold_count": count}
                    for name, count in frequencies.items()
                ),
                key=lambda item: (-item["fold_count"], item["feature"]),
            )[:50],
        }
    return result


def _fit_outer_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    X_external: pd.DataFrame,
    best_params: Mapping[str, dict[str, Any]],
    *,
    seed: int,
    tabnet_seeds: int,
    device_name: str,
    smoke: bool,
    deadline: float,
    reserve_seconds: float,
) -> tuple[
    dict[str, FoldPreprocessor],
    list[TabNetAdapter],
    YDFAdapter,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    preprocessors: dict[str, FoldPreprocessor] = {}
    valid_raw: dict[str, np.ndarray] = {}
    external_raw: dict[str, np.ndarray] = {}

    tab_preprocessor = FoldPreprocessor(
        max_features=int(best_params["tabnet"]["max_features"]),
        bootstrap_rounds=4 if smoke else 24,
        minimum_per_modality=6 if smoke else 12,
        seed=seed + 11,
    )
    tab_train = tab_preprocessor.fit_transform(X_train, y_train)
    tab_valid = tab_preprocessor.transform(X_valid)
    tab_external = tab_preprocessor.transform(X_external)
    tabnet_models = []
    tab_valid_probabilities = []
    tab_external_probabilities = []
    for seed_index in range(int(tabnet_seeds)):
        _ensure_time_budget(
            deadline,
            reserve_seconds=reserve_seconds,
            next_step=f"final TabNet seed={seed_index}",
        )
        fitted = fit_tabnet(
            tab_train,
            y_train,
            best_params["tabnet"],
            seed=seed + seed_index * 10_007,
            device_name=device_name,
        )
        tabnet_models.append(fitted)
        tab_valid_probabilities.append(_positive_probability(fitted, tab_valid))
        tab_external_probabilities.append(_positive_probability(fitted, tab_external))
    preprocessors["tabnet"] = tab_preprocessor
    valid_raw["tabnet"] = np.mean(tab_valid_probabilities, axis=0)
    external_raw["tabnet"] = np.mean(tab_external_probabilities, axis=0)

    ydf_preprocessor = FoldPreprocessor(
        max_features=int(best_params["ydf"]["max_features"]),
        bootstrap_rounds=4 if smoke else 24,
        minimum_per_modality=6 if smoke else 12,
        seed=seed + 29,
    )
    ydf_train = ydf_preprocessor.fit_transform(X_train, y_train)
    _ensure_time_budget(
        deadline,
        reserve_seconds=reserve_seconds,
        next_step="final YDF fit",
    )
    ydf_model = fit_ydf(ydf_train, y_train, best_params["ydf"], seed=seed + 43)
    preprocessors["ydf"] = ydf_preprocessor
    valid_raw["ydf"] = _positive_probability(
        ydf_model, ydf_preprocessor.transform(X_valid)
    )
    external_raw["ydf"] = _positive_probability(
        ydf_model, ydf_preprocessor.transform(X_external)
    )
    return preprocessors, tabnet_models, ydf_model, valid_raw, external_raw


def _consensus_parameters(
    fold_records: list[dict[str, Any]], model_name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate outer-fold winners without selecting one extreme best fold."""

    candidates = [dict(record["best_params"][model_name]) for record in fold_records]
    if not candidates:
        raise ValueError(f"No fold parameters exist for {model_name}")
    categorical_keys = {
        "max_features",
        "n_d",
        "epochs",
        "batch_size",
        "virtual_batch_size",
        "mask_type",
        "num_trees",
        "shrinkage",
        "subsample",
        "num_candidate_attributes_ratio",
        "num_threads",
    }
    log_scale_keys = {"lambda_sparse", "lr", "weight_decay", "l2_regularization"}
    resolved: dict[str, Any] = {}
    strategy: dict[str, str] = {}
    for key in candidates[0]:
        values = [candidate[key] for candidate in candidates]
        if key in categorical_keys or isinstance(values[0], (str, bool)):
            counts = Counter(values)
            maximum = max(counts.values())
            tied_values = [value for value, count in counts.items() if count == maximum]
            tied = sorted(
                tied_values,
                key=(
                    (lambda value: float(value))
                    if all(isinstance(value, (int, float, np.number)) for value in tied_values)
                    else (lambda value: str(value))
                ),
            )
            resolved[key] = tied[len(tied) // 2]
            strategy[key] = "mode; deterministic middle tie-break"
        elif key in log_scale_keys:
            positive = np.asarray(values, dtype=float)
            if np.any(positive <= 0):
                raise ValueError(f"Log-scale parameter {key} contains a non-positive value")
            resolved[key] = float(np.exp(np.median(np.log(positive))))
            strategy[key] = "geometric median"
        elif isinstance(values[0], (int, np.integer)):
            resolved[key] = int(round(float(np.median(values))))
            strategy[key] = "rounded median"
        else:
            resolved[key] = float(np.median(values))
            strategy[key] = "median"
    return resolved, {
        "source_fold_count": len(candidates),
        "aggregation": strategy,
        "source_parameters": candidates,
    }


def _fit_full_refit(
    X: pd.DataFrame,
    y: np.ndarray,
    X_external: pd.DataFrame,
    fold_records: list[dict[str, Any]],
    raw_oof: Mapping[str, np.ndarray],
    *,
    checkpoint_dir: Path,
    config: RunConfig,
    device_name: str,
    smoke: bool,
    deadline: float,
    operating_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    best_params: dict[str, dict[str, Any]] = {}
    sources: dict[str, Any] = {}
    for model_name in MODEL_NAMES:
        best_params[model_name], sources[model_name] = _consensus_parameters(
            fold_records, model_name
        )
    calibrators, blend = select_calibration_and_blend(y, raw_oof)
    fit_seed_base = config.seed + 7_000_001
    preprocessors, tabnet_models, ydf_model, _, external_raw = _fit_outer_models(
        X,
        y,
        X.iloc[: min(12, len(X))],
        X_external,
        best_params,
        seed=fit_seed_base,
        tabnet_seeds=config.tabnet_seeds,
        device_name=device_name,
        smoke=smoke,
        deadline=deadline,
        reserve_seconds=_reserve_for_mode(300.0, smoke=smoke),
    )
    calibrated_external = {
        name: calibrators[name].transform(external_raw[name]) for name in MODEL_NAMES
    }
    external_probability = apply_blend(calibrated_external, blend)
    selection = {
        "scope": "full 141-subject refit for deployment/secondary prediction",
        "parameter_sources": sources,
        "best_params": best_params,
        "fit_seeds": {
            "tabnet": [
                fit_seed_base + seed_index * 10_007
                for seed_index in range(config.tabnet_seeds)
            ],
            "ydf": fit_seed_base + 43,
        },
        "training_outer_oof_calibration_and_blend": blend,
        "training_oof_selected_operating_threshold": float(operating_threshold),
        "warning": "This refit has no independent OOF metric; cross-fold bagging is the primary evaluation model.",
    }
    save_checkpoint_bundle(
        checkpoint_dir,
        preprocessors=preprocessors,
        tabnet_models=tabnet_models,
        ydf_model=ydf_model,
        calibrators=calibrators,
        selection=selection,
        probe=X.iloc[: min(8, len(X))],
    )
    return external_probability, selection


def run_experiment(config: RunConfig) -> dict[str, Any]:
    if config.run_mode not in {"full", "smoke"}:
        raise ValueError("run_mode must be 'full' or 'smoke'")
    smoke = config.run_mode == "smoke"
    if config.outer_folds < 2 or config.inner_folds < 2:
        raise ValueError("Nested CV requires at least two outer and inner folds")
    if config.outer_repeats < 1 or config.tabnet_seeds < 1:
        raise ValueError("outer_repeats and tabnet_seeds must be positive")
    output_root = Path(config.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    training_dir = output_root / "training"
    if training_dir.exists():
        raise FileExistsError(f"Training output already exists: {training_dir}")
    training_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = training_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + float(config.max_runtime_seconds)
    write_json(training_dir / "run_config.json", asdict(config))
    write_json(training_dir / "environment.json", runtime_info())
    write_json(training_dir / "code_manifest.json", code_manifest())
    set_global_seed(config.seed)

    train_sequence = build_subject_dataset(
        config.training_root,
        require_labels=True,
        expected_split="train",
    )
    train_table = build_subject_feature_table(train_sequence)
    if train_table.y is None:
        raise AssertionError("Training labels are unexpectedly absent")
    y = train_table.y
    X = train_table.X
    if (
        len(train_sequence.feature_names) != EXPECTED_DAILY_FEATURES
        or X.shape[1] != EXPECTED_SUBJECT_FEATURES
        or len(train_table.audit["resolved_correlation_pairs"])
        != EXPECTED_CORRELATION_PAIRS
    ):
        raise AssertionError(
            "Audited wearable feature contract changed: "
            f"daily={len(train_sequence.feature_names)}, subject={X.shape[1]}, "
            f"correlations={len(train_table.audit['resolved_correlation_pairs'])}"
        )
    write_json(training_dir / "training_data_audit.json", train_sequence.audit)
    write_json(training_dir / "training_feature_audit.json", train_table.audit)
    write_json(
        training_dir / "feature_manifest.json",
        {
            "feature_count": X.shape[1],
            "feature_names": X.columns.tolist(),
            "schema_sha256": schema_sha256(X.columns.tolist()),
            "modalities": ["Activity", "Sleep"],
            "cognitive_features": False,
        },
    )
    run_eda(train_sequence, train_table, output_root / "eda")

    external_sequence = build_subject_dataset(
        config.validation_root,
        require_labels=False,
        expected_split="val",
    )
    external_table = build_subject_feature_table(external_sequence)
    if list(external_table.X.columns) != list(X.columns):
        raise AssertionError("Training/Validation feature schema or order differs")
    overlapping_subjects = set(train_table.subject_ids) & set(
        external_table.subject_ids
    )
    if overlapping_subjects:
        raise AssertionError(
            "Training/Validation subject leakage detected: "
            f"overlap_count={len(overlapping_subjects)}"
        )
    external_X = external_table.X
    write_json(
        training_dir / "validation_data_audit_label_free.json",
        external_sequence.audit,
    )
    write_json(
        training_dir / "validation_feature_audit_label_free.json",
        external_table.audit,
    )

    device_name = "cpu" if smoke else "cuda"
    oof_blend_sum = np.zeros(len(y), dtype=np.float64)
    oof_raw_sum = {name: np.zeros(len(y), dtype=np.float64) for name in MODEL_NAMES}
    oof_count = np.zeros(len(y), dtype=np.int16)
    external_blend_sum = np.zeros(len(external_X), dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    tuning_records: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []

    for repeat in range(config.outer_repeats):
        outer_seed = config.seed + repeat * 100_003
        outer_cv = StratifiedKFold(
            n_splits=config.outer_folds,
            shuffle=True,
            random_state=outer_seed,
        )
        for fold, (train_indices, valid_indices) in enumerate(outer_cv.split(X, y)):
            _ensure_time_budget(
                deadline,
                reserve_seconds=_reserve_for_mode(
                    TUNING_RESERVE_SECONDS, smoke=smoke
                ),
                next_step=f"outer repeat={repeat} fold={fold}",
            )
            fold_tag = f"repeat_{repeat:02d}_fold_{fold:02d}"
            print(f"\n=== {fold_tag} ({len(train_indices)} train / {len(valid_indices)} valid) ===")
            X_outer = X.iloc[train_indices].reset_index(drop=True)
            y_outer = y[train_indices]
            X_valid = X.iloc[valid_indices]
            y_valid = y[valid_indices]
            inner_seed = outer_seed + fold * 1009 + 17
            inner_cv = StratifiedKFold(
                n_splits=config.inner_folds,
                shuffle=True,
                random_state=inner_seed,
            )
            inner_splits = list(inner_cv.split(X_outer, y_outer))
            best_params: dict[str, dict[str, Any]] = {}
            tuning_summary: dict[str, Any] = {}
            inner_raw: dict[str, np.ndarray] = {}
            for model_index, model_name in enumerate(MODEL_NAMES):
                trials = config.trials_tabnet if model_name == "tabnet" else config.trials_ydf
                model_seed = inner_seed + model_index * 10_007
                params, summary, records = tune_model(
                    model_name,
                    X_outer,
                    y_outer,
                    inner_splits,
                    trials=trials,
                    seed=model_seed,
                    device_name=device_name,
                    smoke=smoke,
                    deadline=deadline,
                )
                best_params[model_name] = params
                tuning_summary[model_name] = summary
                for record in records:
                    record.update({"repeat": repeat, "outer_fold": fold})
                tuning_records.extend(records)
                inner_raw[model_name] = collect_inner_oof(
                    model_name,
                    X_outer,
                    y_outer,
                    inner_splits,
                    params,
                    seed=model_seed + 900_001,
                    device_name=device_name,
                    smoke=smoke,
                    ensemble_seeds=(
                        config.tabnet_seeds if model_name == "tabnet" else 1
                    ),
                    deadline=deadline,
                )
            calibrators, blend = select_calibration_and_blend(y_outer, inner_raw)
            fit_seed_base = outer_seed + fold * 1009 + 2_000_003
            preprocessors, tabnet_models, ydf_model, valid_raw, external_raw = _fit_outer_models(
                X_outer,
                y_outer,
                X_valid,
                external_X,
                best_params,
                seed=fit_seed_base,
                tabnet_seeds=config.tabnet_seeds,
                device_name=device_name,
                smoke=smoke,
                deadline=deadline,
                reserve_seconds=_reserve_for_mode(
                    OUTER_COMPLETION_RESERVE_SECONDS, smoke=smoke
                ),
            )
            valid_calibrated = {
                name: calibrators[name].transform(valid_raw[name]) for name in MODEL_NAMES
            }
            external_calibrated = {
                name: calibrators[name].transform(external_raw[name]) for name in MODEL_NAMES
            }
            p_valid = apply_blend(valid_calibrated, blend)
            p_external = apply_blend(external_calibrated, blend)
            metrics = binary_metrics(y_valid, p_valid)
            oof_blend_sum[valid_indices] += p_valid
            for model_name in MODEL_NAMES:
                oof_raw_sum[model_name][valid_indices] += valid_raw[model_name]
            oof_count[valid_indices] += 1
            external_blend_sum += p_external
            selection = {
                "scope": "one outer fold",
                "repeat": repeat,
                "fold": fold,
                "class_counts": np.bincount(y_outer, minlength=2).tolist(),
                "best_params": best_params,
                "fit_seeds": {
                    "tabnet": [
                        fit_seed_base + seed_index * 10_007
                        for seed_index in range(config.tabnet_seeds)
                    ],
                    "ydf": fit_seed_base + 43,
                },
                "tuning": tuning_summary,
                "inner_calibration_and_blend": blend,
                "preprocessors": {
                    name: preprocessor.manifest()
                    for name, preprocessor in preprocessors.items()
                },
            }
            verification = save_checkpoint_bundle(
                checkpoints / fold_tag,
                preprocessors=preprocessors,
                tabnet_models=tabnet_models,
                ydf_model=ydf_model,
                calibrators=calibrators,
                selection=selection,
                probe=X_valid.iloc[: min(8, len(X_valid))],
            )
            for model_name, preprocessor in preprocessors.items():
                selected_feature_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "model": model_name,
                        "features": preprocessor.selected_feature_names,
                    }
                )
            fold_record = {
                "repeat": repeat,
                "fold": fold,
                "n_train": len(train_indices),
                "n_valid": len(valid_indices),
                "best_params": best_params,
                "tuning": tuning_summary,
                "inner_calibration_and_blend": blend,
                "outer_metrics_at_0_5": metrics,
                "per_model_outer_raw_metrics": {
                    name: binary_metrics(y_valid, valid_raw[name]) for name in MODEL_NAMES
                },
                "checkpoint_roundtrip": verification,
            }
            fold_records.append(fold_record)
            write_json(training_dir / "nested_progress.json", {"completed_folds": fold_records})
            write_json(training_dir / "tuning_trials_partial.json", tuning_records)
            pd.DataFrame(
                [
                    {
                        "repeat": row["repeat"],
                        "fold": row["fold"],
                        "n_train": row["n_train"],
                        "n_valid": row["n_valid"],
                        **{
                            name: row["outer_metrics_at_0_5"][name]
                            for name in (
                                "accuracy",
                                "balanced_accuracy",
                                "f1_impaired",
                                "roc_auc",
                                "log_loss",
                            )
                        },
                    }
                    for row in fold_records
                ]
            ).to_csv(training_dir / "outer_fold_metrics.csv", index=False)
            del tabnet_models, ydf_model, preprocessors
            _release_model_memory()

    if not np.all(oof_count == config.outer_repeats):
        raise AssertionError(f"Outer OOF coverage mismatch: {oof_count.tolist()}")
    total_folds = config.outer_folds * config.outer_repeats
    oof = np.clip(oof_blend_sum / oof_count, 1e-7, 1 - 1e-7)
    raw_oof = {
        name: np.clip(oof_raw_sum[name] / oof_count, 1e-7, 1 - 1e-7)
        for name in MODEL_NAMES
    }
    oof_primary = binary_metrics(y, oof, 0.5)
    threshold_selection = _select_threshold(y, oof)
    oof_selected = binary_metrics(y, oof, threshold_selection["threshold"])
    confidence_intervals = _bootstrap_confidence_intervals(
        y,
        oof,
        seed=config.seed + 8_000_003,
        rounds=200 if smoke else 1000,
    )
    pd.DataFrame(
        {
            "subject_hash": [subject_hash(value) for value in train_table.subject_ids],
            "true_class": [CLASS_NAMES[int(value)] for value in y],
            "p_mci_dem": oof,
            "predicted_at_0_5": [CLASS_NAMES[int(value)] for value in (oof >= 0.5)],
            "repeat_prediction_count": oof_count,
            "p_tabnet_raw": raw_oof["tabnet"],
            "p_ydf_raw": raw_oof["ydf"],
        }
    ).to_csv(training_dir / "nested_oof_predictions_hashed.csv", index=False)
    pd.DataFrame(
        [
            {
                "repeat": row["repeat"],
                "fold": row["fold"],
                "model": row["model"],
                "rank": rank,
                "feature": feature,
            }
            for row in selected_feature_rows
            for rank, feature in enumerate(row["features"])
        ]
    ).to_csv(training_dir / "selected_features_by_fold.csv", index=False)
    feature_stability = _feature_stability(selected_feature_rows)
    write_json(training_dir / "feature_selection_stability.json", feature_stability)
    write_json(training_dir / "tuning_trials.json", tuning_records)
    nested_report = {
        "primary_repeat_averaged_oof_at_0_5": oof_primary,
        "secondary_oof_at_training_selected_threshold": oof_selected,
        "threshold_selection": threshold_selection,
        "subject_bootstrap_95_ci_at_0_5": confidence_intervals,
        "all_cn_baseline": _all_cn_baseline(y),
        "target_check": _target_check(oof_primary),
        "folds": fold_records,
        "cv_contract": {
            "unit": "subject",
            "outer_folds": config.outer_folds,
            "outer_repeats": config.outer_repeats,
            "inner_folds": config.inner_folds,
            "oof_prediction_count_per_subject": config.outer_repeats,
        },
    }
    write_json(training_dir / "nested_cv_report.json", nested_report)

    # Freeze the primary cross-fold prediction before the optional secondary
    # full refit.  A time-budget skip can therefore never erase the primary result.
    external_crossfold = np.clip(external_blend_sum / total_folds, 1e-7, 1 - 1e-7)
    frozen_path = training_dir / "validation_predictions_label_free_hashed.csv"
    frozen_frame = pd.DataFrame(
        {
            "subject_hash": [subject_hash(value) for value in external_table.subject_ids],
            "p_mci_dem_crossfold": external_crossfold,
            "predicted_crossfold_at_0_5": [
                CLASS_NAMES[int(value)] for value in (external_crossfold >= 0.5)
            ],
        }
    )
    frozen_frame.to_csv(frozen_path, index=False)
    frozen_sha = sha256_file(frozen_path)
    write_json(
        training_dir / "VALIDATION_PREDICTIONS_FROZEN.json",
        {
            "prediction_file": frozen_path.name,
            "sha256": frozen_sha,
            "rows": len(frozen_frame),
            "labels_opened_before_prediction_write": False,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )

    full_external: np.ndarray | None = None
    full_selection: dict[str, Any] = {
        "status": "skipped",
        "reason": "insufficient soft-budget time",
    }
    full_frozen_path: Path | None = None
    full_frozen_sha: str | None = None
    try:
        _ensure_time_budget(
            deadline,
            reserve_seconds=_reserve_for_mode(
                FULL_REFIT_MINIMUM_SECONDS, smoke=smoke
            ),
            next_step="secondary full-data refit",
        )
        full_external, full_selection = _fit_full_refit(
            X,
            y,
            external_X,
            fold_records,
            raw_oof,
            checkpoint_dir=checkpoints / "full_refit",
            config=config,
            device_name=device_name,
            smoke=smoke,
            deadline=deadline,
            operating_threshold=threshold_selection["threshold"],
        )
        full_selection = {"status": "complete", **full_selection}
        full_frozen_path = (
            training_dir / "validation_full_refit_predictions_label_free_hashed.csv"
        )
        pd.DataFrame(
            {
                "subject_hash": [
                    subject_hash(value) for value in external_table.subject_ids
                ],
                "p_mci_dem_full_refit": full_external,
                "predicted_full_refit_at_0_5": [
                    CLASS_NAMES[int(value)] for value in (full_external >= 0.5)
                ],
            }
        ).to_csv(full_frozen_path, index=False)
        full_frozen_sha = sha256_file(full_frozen_path)
        write_json(
            training_dir / "VALIDATION_FULL_REFIT_PREDICTIONS_FROZEN.json",
            {
                "prediction_file": full_frozen_path.name,
                "sha256": full_frozen_sha,
                "rows": len(full_external),
                "labels_opened_before_prediction_write": False,
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
    except SoftTimeBudgetExceeded as error:
        full_selection = {
            "status": "skipped",
            "reason": str(error),
            "primary_crossfold_predictions_already_frozen": True,
        }
        write_json(training_dir / "FULL_REFIT_SKIPPED.json", full_selection)

    checkpoint_index = verify_checkpoint_tree(
        checkpoints,
        expected_outer_bundles=total_folds,
        tabnet_seeds_per_bundle=config.tabnet_seeds,
        require_full_refit=full_external is not None,
    )
    write_json(training_dir / "checkpoint_index.json", checkpoint_index)

    validation_report = None
    if config.evaluate_historical_validation:
        if sha256_file(frozen_path) != frozen_sha:
            raise AssertionError("Frozen Validation prediction file changed before evaluation")
        reloaded = pd.read_csv(frozen_path)
        if len(reloaded) != len(external_table.subject_ids):
            raise AssertionError("Frozen Validation prediction row count changed")
        validation_labels = load_binary_labels(config.validation_root)
        expected_validation_subjects = set(external_table.subject_ids.astype(str))
        observed_validation_subjects = set(validation_labels.index.astype(str))
        if observed_validation_subjects != expected_validation_subjects:
            raise AssertionError(
                "Validation label subjects differ from the frozen label-free rows"
            )
        aligned_y = validation_labels.loc[external_table.subject_ids].to_numpy(np.int64)
        observed_validation_counts = {
            class_id: int(np.sum(aligned_y == class_id)) for class_id in (0, 1)
        }
        if observed_validation_counts != {0: 26, 1: 7}:
            raise AssertionError(
                "Validation binary class contract changed: "
                f"{observed_validation_counts}"
            )
        crossfold_probability = reloaded["p_mci_dem_crossfold"].to_numpy(float)
        validation_report = {
            "primary_crossfold_bagging_at_0_5": binary_metrics(
                aligned_y, crossfold_probability, 0.5
            ),
            "secondary_crossfold_at_training_selected_threshold": binary_metrics(
                aligned_y, crossfold_probability, threshold_selection["threshold"]
            ),
            "secondary_full_refit_at_0_5": None,
            "all_cn_baseline": _all_cn_baseline(aligned_y),
            "warning": HISTORICAL_VALIDATION_WARNING,
            "frozen_prediction_sha256_verified": True,
        }
        full_reloaded = None
        if full_frozen_path is not None and full_frozen_sha is not None:
            if sha256_file(full_frozen_path) != full_frozen_sha:
                raise AssertionError("Frozen full-refit Validation prediction changed")
            full_reloaded = pd.read_csv(full_frozen_path)
            full_probability = full_reloaded["p_mci_dem_full_refit"].to_numpy(float)
            validation_report["secondary_full_refit_at_0_5"] = binary_metrics(
                aligned_y, full_probability, 0.5
            )
        validation_report["target_check"] = _target_check(
            validation_report["primary_crossfold_bagging_at_0_5"]
        )
        evaluated = reloaded.copy()
        evaluated.insert(1, "true_class", [CLASS_NAMES[int(value)] for value in aligned_y])
        if full_reloaded is not None:
            evaluated["p_mci_dem_full_refit_secondary"] = full_reloaded[
                "p_mci_dem_full_refit"
            ].to_numpy(float)
        evaluated.to_csv(
            training_dir / "validation_predictions_evaluated_hashed.csv", index=False
        )
        write_json(training_dir / "validation_report.json", validation_report)

    final_report = {
        "experiment": "Binary_Wearable_TabNet_Google",
        "run_mode": config.run_mode,
        "binary_target": "CN vs MCI+DEM",
        "input_modalities": ["Activity", "Sleep"],
        "cognitive_test_used": False,
        "models": list(MODEL_NAMES),
        "google_model_evidence": GOOGLE_MODEL_EVIDENCE,
        "nested_oof_primary": oof_primary,
        "historical_validation": validation_report,
        "target_check_nested_oof": _target_check(oof_primary),
        "historical_validation_warning": HISTORICAL_VALIDATION_WARNING,
        "checkpoint_contract": {
            "outer_checkpoint_bundles": total_folds,
            "tabnet_models_per_outer_bundle": config.tabnet_seeds,
            "full_refit_bundle": bool(full_external is not None),
            "fresh_cpu_roundtrip_verified": True,
            "root": str(checkpoints),
            "index_file": str(training_dir / "checkpoint_index.json"),
            "tree_verified": checkpoint_index["verified"],
        },
        "full_refit_selection": full_selection,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(training_dir / "FINAL_REPORT.json", final_report)
    write_json(
        training_dir / "TRAINING_COMPLETE.json",
        {
            "success": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": config.run_mode,
            "final_report": str(training_dir / "FINAL_REPORT.json"),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return final_report


__all__ = ["RunConfig", "binary_metrics", "run_experiment"]
