"""Leakage-aware subject-level training for CN versus MCI/DEM.

The pipeline performs Training-only EDA, repeated outer subject CV, inner-OOF
model/threshold selection, and one frozen historical Validation evaluation.
MMSE and collection-protocol shortcuts are excluded by the data contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
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

if __package__:  # Collision-safe package import when launched from base.ipynb.
    from .data import build_binary_dataset
    from .eda import run_eda
    from .models import (
        CLASS_NAMES,
        GOOGLE_MODEL_EVIDENCE,
        MODEL_NAMES,
        fit_model,
        normalize_probabilities,
        set_global_seed,
    )
    from .preprocessing import FoldPreprocessor, assert_feature_contract
else:  # Direct development import with this folder on sys.path.
    from data import build_binary_dataset  # type: ignore
    from eda import run_eda  # type: ignore
    from models import (  # type: ignore
        CLASS_NAMES,
        GOOGLE_MODEL_EVIDENCE,
        MODEL_NAMES,
        fit_model,
        normalize_probabilities,
        set_global_seed,
    )
    from preprocessing import FoldPreprocessor, assert_feature_contract  # type: ignore


TARGET_ACCURACY = 0.90
CV_UNIT = "subject"
VALIDATION_PREDICTIONS_FILE = "validation_predictions_label_free_hashed.csv"
VALIDATION_FREEZE_MARKER = "VALIDATION_PREDICTIONS_FROZEN.json"


@dataclass(frozen=True)
class RunConfig:
    training_root: str
    validation_root: str | None
    output_dir: str
    run_mode: str = "full"
    seed: int = 20260722
    outer_folds: int = 5
    outer_repeats: int = 2
    inner_folds: int = 3
    max_features: int = 24
    max_runtime_seconds: int = 20_700
    include_neural: bool = True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
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
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def subject_hash(subject_id: str) -> str:
    return hashlib.sha256(
        f"wearable-binary-report-v1::{subject_id}".encode("utf-8")
    ).hexdigest()[:20]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_metric(callable_metric, *args, **kwargs) -> float:
    try:
        return float(callable_metric(*args, **kwargs))
    except ValueError:
        return float("nan")


def binary_metrics(
    y_true: np.ndarray,
    p_impaired: np.ndarray,
    *,
    threshold: float = 0.5,
    predicted: np.ndarray | None = None,
) -> dict[str, Any]:
    target = np.asarray(y_true, dtype=np.int64)
    probability = np.clip(np.asarray(p_impaired, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    decision = (
        np.asarray(predicted, dtype=np.int64)
        if predicted is not None
        else (probability >= float(threshold)).astype(np.int64)
    )
    matrix = confusion_matrix(target, decision, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    report = classification_report(
        target,
        decision,
        labels=[0, 1],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(target, decision)),
        "balanced_accuracy": float(balanced_accuracy_score(target, decision)),
        "f1_impaired": float(f1_score(target, decision, zero_division=0)),
        "precision_impaired": float(precision_score(target, decision, zero_division=0)),
        "recall_impaired": float(recall_score(target, decision, zero_division=0)),
        "specificity_cn": float(tn / max(1, tn + fp)),
        "roc_auc": _safe_metric(roc_auc_score, target, probability),
        "pr_auc": _safe_metric(average_precision_score, target, probability),
        "log_loss": _safe_metric(
            log_loss, target, np.column_stack([1.0 - probability, probability]), labels=[0, 1]
        ),
        "brier_score": float(brier_score_loss(target, probability)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "support": {"CN": int((target == 0).sum()), "MCI_DEM": int((target == 1).sum())},
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def _selection_score(metrics: Mapping[str, Any], threshold: float) -> float:
    auc = float(metrics["roc_auc"])
    auc = 0.5 if not np.isfinite(auc) else auc
    pr_auc = float(metrics["pr_auc"])
    pr_auc = 0.0 if not np.isfinite(pr_auc) else pr_auc
    score = (
        0.35 * float(metrics["accuracy"])
        + 0.25 * float(metrics["balanced_accuracy"])
        + 0.15 * float(metrics["f1_impaired"])
        + 0.15 * auc
        + 0.10 * pr_auc
        - 0.05 * abs(float(threshold) - 0.5)
    )
    if float(metrics["recall_impaired"]) < 0.20:
        score -= 0.10
    return float(score)


def _weight_candidates(model_names: tuple[str, ...]) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for name in model_names:
        candidates.append({candidate: float(candidate == name) for candidate in model_names})
    for left_index, left in enumerate(model_names):
        for right in model_names[left_index + 1 :]:
            candidates.append(
                {name: (0.5 if name in {left, right} else 0.0) for name in model_names}
            )
    if len(model_names) >= 3:
        ydf_names = [name for name in model_names if name.startswith("ydf_")]
        if len(ydf_names) == 2:
            candidates.append(
                {
                    name: (0.35 if name in ydf_names else 0.30 / (len(model_names) - 2))
                    for name in model_names
                }
            )
    candidates.append({name: 1.0 / len(model_names) for name in model_names})
    unique: dict[tuple[float, ...], dict[str, float]] = {}
    for candidate in candidates:
        key = tuple(round(candidate[name], 8) for name in model_names)
        unique[key] = candidate
    return list(unique.values())


def apply_blend(
    model_probabilities: Mapping[str, np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    probability = sum(
        float(weights[name]) * np.asarray(model_probabilities[name], dtype=np.float64)[:, 1]
        for name in weights
    )
    return np.clip(probability, 1e-8, 1.0 - 1e-8)


def optimize_inner_blend(
    y_true: np.ndarray, model_probabilities: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    names = tuple(model_probabilities)
    # This small grid is fixed in advance.  A dense threshold search on only
    # 141 subjects was empirically unstable and could overstate OOF accuracy.
    thresholds = np.asarray([0.45, 0.50, 0.55, 0.60, 0.65], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for weights in _weight_candidates(names):
        probability = apply_blend(model_probabilities, weights)
        best_for_weights: dict[str, Any] | None = None
        for threshold in thresholds:
            metrics = binary_metrics(y_true, probability, threshold=float(threshold))
            row = {
                "weights": weights,
                "threshold": float(threshold),
                "metrics": metrics,
                "selection_score": _selection_score(metrics, float(threshold)),
                "complexity": int(sum(value > 0 for value in weights.values())),
            }
            if best_for_weights is None or (
                row["selection_score"], -row["complexity"], -abs(threshold - 0.5)
            ) > (
                best_for_weights["selection_score"],
                -best_for_weights["complexity"],
                -abs(best_for_weights["threshold"] - 0.5),
            ):
                best_for_weights = row
        assert best_for_weights is not None
        rows.append(best_for_weights)
    best_score = max(float(row["selection_score"]) for row in rows)
    # Prefer a simpler blend if it is effectively tied on inner OOF data.
    eligible = [row for row in rows if row["selection_score"] >= best_score - 0.005]
    chosen = min(
        eligible,
        key=lambda row: (
            row["complexity"],
            abs(float(row["threshold"]) - 0.5),
            -float(row["selection_score"]),
        ),
    )
    return {"chosen": chosen, "candidates": rows, "selection_scope": "inner OOF only"}


def _active_models(config: RunConfig) -> tuple[str, ...]:
    return MODEL_NAMES if config.include_neural else tuple(
        name for name in MODEL_NAMES if name.startswith("ydf_")
    )


def _new_preprocessor(config: RunConfig, seed: int) -> FoldPreprocessor:
    fast = config.run_mode == "smoke"
    return FoldPreprocessor(
        max_features=config.max_features,
        bootstrap_rounds=4 if fast else 24,
        min_features_per_modality=max(4, min(12, config.max_features // 2)),
        add_cn_deviation=False,
        seed=seed,
    )


def _check_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(
            f"Runtime budget reached before {stage}; progress files were preserved. "
            "Reduce folds/models or resume in a fresh run."
        )


def _runtime_info() -> dict[str, Any]:
    info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "pandas", "sklearn", "ydf", "torch", "pytorch_tabnet"):
        try:
            module = __import__(module_name)
            info[module_name] = getattr(module, "__version__", "installed")
        except ImportError:
            info[module_name] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        info["cuda_available"] = False
        info["cuda_device"] = None
    return info


def _assert_subject_contract(dataset, expected_counts: tuple[int, int] | None) -> None:
    if dataset.y is None:
        raise AssertionError("A labeled dataset was required")
    if len(dataset.subject_ids) != len(set(map(str, dataset.subject_ids))):
        raise AssertionError("Duplicate subject IDs violate subject-level CV")
    assert_feature_contract(dataset.X.columns)
    counts = tuple(int(value) for value in np.bincount(dataset.y, minlength=2))
    if expected_counts is not None and counts != expected_counts:
        raise AssertionError(
            f"Cohort differs from the audited contract: {counts} != {expected_counts}"
        )


def _collect_inner_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_names: tuple[str, ...],
    *,
    config: RunConfig,
    seed: int,
    deadline: float,
) -> dict[str, np.ndarray]:
    probabilities = {
        name: np.zeros((len(y), 2), dtype=np.float64) for name in model_names
    }
    seen = np.zeros(len(y), dtype=np.int16)
    fast = config.run_mode == "smoke"
    for inner_fold, (fit_idx, valid_idx) in enumerate(splits):
        preprocessor = _new_preprocessor(config, seed + inner_fold * 101)
        X_fit = preprocessor.fit_transform(X.iloc[fit_idx], y[fit_idx])
        X_valid = preprocessor.transform(X.iloc[valid_idx])
        feature_names = preprocessor.selected_feature_names
        for model_index, model_name in enumerate(model_names):
            _check_deadline(deadline, f"inner fold {inner_fold} / {model_name}")
            fitted, _ = fit_model(
                model_name,
                X_fit,
                y[fit_idx],
                feature_names,
                seed=seed + inner_fold * 10_007 + model_index * 503,
                fast=fast,
            )
            probabilities[model_name][valid_idx] = fitted.predict_proba(X_valid)
        seen[valid_idx] += 1
    if not np.all(seen == 1):
        raise AssertionError(f"Inner OOF coverage failure: {seen.tolist()}")
    return probabilities


def _save_model(fitted: Any, model_name: str, root: Path) -> None:
    if model_name.startswith("ydf_"):
        fitted.save(root / model_name)
    elif model_name == "tabnet":
        fitted.save(root / "tabnet_model")
    elif model_name == "transformer":
        fitted.save(root / "transformer_model.pt")
    else:
        raise ValueError(f"Unknown model for checkpoint: {model_name}")


def freeze_validation_predictions(
    output_dir: Path,
    subject_ids: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> Path:
    """Persist label-free predictions before any Validation label is opened."""

    frame = pd.DataFrame(
        {
            "subject_hash": [subject_hash(str(value)) for value in subject_ids],
            "predicted_class": [
                CLASS_NAMES[int(value)] for value in (probability >= threshold).astype(int)
            ],
            "p_cn": 1.0 - probability,
            "p_impaired": probability,
            "threshold_from_training_inner_oof": float(threshold),
        }
    )
    prediction_path = output_dir / VALIDATION_PREDICTIONS_FILE
    frame.to_csv(prediction_path, index=False)
    prediction_sha256 = file_sha256(prediction_path)
    write_json(
        output_dir / VALIDATION_FREEZE_MARKER,
        {
            "prediction_file": prediction_path.name,
            "prediction_sha256": prediction_sha256,
            "rows": len(frame),
            "columns": frame.columns.tolist(),
            "labels_opened_before_prediction_write": False,
            "threshold_or_weights_tuned_on_validation": False,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return prediction_path


def evaluate_frozen_validation(
    validation_root: str,
    output_dir: Path,
    expected_subject_ids: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Open historical Validation labels only after the freeze marker exists."""

    marker = output_dir / VALIDATION_FREEZE_MARKER
    prediction_path = output_dir / VALIDATION_PREDICTIONS_FILE
    if not marker.is_file() or not prediction_path.is_file():
        raise RuntimeError("Validation predictions must be frozen before evaluation")
    marker_payload = json.loads(marker.read_text(encoding="utf-8"))
    if marker_payload.get("prediction_file") != prediction_path.name:
        raise RuntimeError("Validation freeze marker points to an unexpected file")
    observed_sha256 = file_sha256(prediction_path)
    if marker_payload.get("prediction_sha256") != observed_sha256:
        raise RuntimeError("Frozen Validation prediction SHA-256 does not match")
    frozen = pd.read_csv(prediction_path)
    expected_hashes = [subject_hash(str(value)) for value in expected_subject_ids]
    if frozen["subject_hash"].astype(str).tolist() != expected_hashes:
        raise RuntimeError("Frozen Validation subject order/hash differs from prediction order")
    frozen_probability = frozen["p_impaired"].to_numpy(dtype=np.float64)
    frozen_thresholds = frozen["threshold_from_training_inner_oof"].to_numpy(
        dtype=np.float64
    )
    if not np.allclose(frozen_probability, probability, rtol=1e-12, atol=1e-12):
        raise RuntimeError("In-memory and frozen Validation probabilities differ")
    if not np.allclose(frozen_thresholds, float(threshold), rtol=0.0, atol=1e-12):
        raise RuntimeError("In-memory and frozen Validation thresholds differ")
    frozen_threshold = float(frozen_thresholds[0])
    frozen_decision = (frozen_probability >= frozen_threshold).astype(np.int64)
    expected_classes = [CLASS_NAMES[int(value)] for value in frozen_decision]
    if frozen["predicted_class"].astype(str).tolist() != expected_classes:
        raise RuntimeError("Frozen Validation class labels differ from its probabilities")
    labeled = build_binary_dataset(validation_root, require_labels=True)
    _assert_subject_contract(labeled, expected_counts=(26, 7))
    label_by_subject = {
        str(subject): int(label)
        for subject, label in zip(labeled.subject_ids, labeled.y, strict=True)
    }
    aligned = np.asarray(
        [label_by_subject[str(subject)] for subject in expected_subject_ids], dtype=np.int64
    )
    metrics = binary_metrics(
        aligned,
        frozen_probability,
        threshold=frozen_threshold,
    )
    evaluated = pd.DataFrame(
        {
            "subject_hash": [subject_hash(str(value)) for value in expected_subject_ids],
            "true_class": [CLASS_NAMES[int(value)] for value in aligned],
            "predicted_class": [
                CLASS_NAMES[int(value)] for value in frozen_decision
            ],
            "p_cn": 1.0 - frozen_probability,
            "p_impaired": frozen_probability,
            "threshold_from_training_inner_oof": frozen_threshold,
        }
    )
    evaluated.to_csv(output_dir / "validation_predictions_evaluated_hashed.csv", index=False)
    report = {
        "metrics": metrics,
        "target_accuracy": TARGET_ACCURACY,
        "target_met": bool(metrics["accuracy"] >= TARGET_ACCURACY),
        "minimum_correct_for_target": 30,
        "validation_subjects": 33,
        "frozen_prediction_sha256": observed_sha256,
        "evaluation_read_probabilities_from_frozen_csv": True,
        "selection_or_tuning_used_validation_labels": False,
        "benchmark_role": "historical holdout already reused by this repository; not a fresh test",
    }
    write_json(output_dir / "validation_report.json", report)
    return report


def run_pipeline(config: RunConfig) -> dict[str, Any]:
    if config.run_mode not in {"full", "smoke"}:
        raise ValueError("run_mode must be 'full' or 'smoke'")
    if CV_UNIT != "subject":
        raise AssertionError("Only subject-level CV is permitted")
    start = time.monotonic()
    deadline = start + int(config.max_runtime_seconds)
    run_root = Path(config.output_dir).expanduser().resolve()
    training_output = run_root / "training"
    eda_output = run_root / "eda"
    if (training_output / "TRAINING_COMPLETE.json").exists():
        raise FileExistsError(f"Completed run will not be overwritten: {run_root}")
    training_output.mkdir(parents=True, exist_ok=True)
    write_json(training_output / "run_config.json", asdict(config))
    write_json(training_output / "environment.json", _runtime_info())
    write_json(training_output / "model_origins.json", GOOGLE_MODEL_EVIDENCE)
    set_global_seed(config.seed)

    # Training-only EDA is completed before any historical Validation label is opened.
    train_data = build_binary_dataset(config.training_root, require_labels=True)
    _assert_subject_contract(train_data, expected_counts=(85, 56))
    run_eda(Path(config.training_root), eda_output, dataset=train_data)
    X = train_data.X
    y = np.asarray(train_data.y, dtype=np.int64)
    write_json(training_output / "training_feature_audit.json", train_data.audit)
    write_json(
        training_output / "feature_manifest.json",
        {
            "cv_unit": CV_UNIT,
            "subjects": len(y),
            "raw_engineered_feature_count": X.shape[1],
            "feature_names": X.columns.tolist(),
            "mmse_source_opened": False,
            "mmse_values_used": False,
            "identifier_or_protocol_features_used": False,
        },
    )

    external = None
    external_probability_sum = np.zeros(0, dtype=np.float64)
    external_threshold_sum = 0.0
    external_fold_count = 0
    if config.validation_root:
        # Crucially label-free: data.discover_split_files cannot resolve a label here.
        external = build_binary_dataset(config.validation_root, require_labels=False)
        if external.y is not None:
            raise AssertionError("Label-free Validation unexpectedly contains labels")
        if len(external.subject_ids) != len(set(map(str, external.subject_ids))):
            raise AssertionError("Duplicate Validation subjects detected")
        if set(map(str, train_data.subject_ids)) & set(map(str, external.subject_ids)):
            raise AssertionError("Training and Validation subject sets overlap")
        if list(external.X.columns) != list(X.columns):
            raise AssertionError("Training/Validation wearable feature schemas differ")
        external_probability_sum = np.zeros(len(external.subject_ids), dtype=np.float64)
        write_json(
            training_output / "validation_feature_audit_label_free.json", external.audit
        )

    model_names = _active_models(config)
    if not model_names:
        raise ValueError("At least one model is required")
    repeat_probabilities = np.zeros((config.outer_repeats, len(y)), dtype=np.float64)
    repeat_predictions = np.zeros((config.outer_repeats, len(y)), dtype=np.int8)
    repeat_thresholds = np.full(
        (config.outer_repeats, len(y)), np.nan, dtype=np.float64
    )
    repeat_seen = np.zeros((config.outer_repeats, len(y)), dtype=np.int8)
    outer_rows: list[dict[str, Any]] = []

    for repeat in range(config.outer_repeats):
        outer_seed = config.seed + repeat * 100_003
        outer_cv = StratifiedKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=outer_seed
        )
        for outer_fold, (fit_idx, valid_idx) in enumerate(outer_cv.split(X, y)):
            _check_deadline(deadline, f"outer repeat {repeat} fold {outer_fold}")
            fold_started = time.monotonic()
            fold_tag = f"repeat_{repeat:02d}_fold_{outer_fold:02d}"
            fold_dir = training_output / "models" / fold_tag
            fold_dir.mkdir(parents=True, exist_ok=True)
            X_outer = X.iloc[fit_idx].reset_index(drop=True)
            y_outer = y[fit_idx]
            inner_cv = StratifiedKFold(
                n_splits=config.inner_folds,
                shuffle=True,
                random_state=outer_seed + outer_fold * 1009 + 17,
            )
            inner_splits = list(inner_cv.split(X_outer, y_outer))
            inner_probabilities = _collect_inner_oof(
                X_outer,
                y_outer,
                inner_splits,
                model_names,
                config=config,
                seed=outer_seed + outer_fold * 10_007,
                deadline=deadline,
            )
            selection = optimize_inner_blend(y_outer, inner_probabilities)
            chosen = selection["chosen"]

            preprocessor = _new_preprocessor(
                config, outer_seed + outer_fold * 1009 + 701
            )
            X_fit = preprocessor.fit_transform(X.iloc[fit_idx], y_outer)
            X_valid = preprocessor.transform(X.iloc[valid_idx])
            X_external = preprocessor.transform(external.X) if external is not None else None
            joblib.dump(preprocessor, fold_dir / "preprocessor.joblib")
            write_json(fold_dir / "preprocessor_manifest.json", preprocessor.manifest())
            outer_probabilities: dict[str, np.ndarray] = {}
            external_probabilities: dict[str, np.ndarray] = {}
            per_model_metrics: dict[str, Any] = {}
            params_by_model: dict[str, Any] = {}
            for model_index, model_name in enumerate(model_names):
                _check_deadline(deadline, f"outer {fold_tag} / {model_name}")
                fitted, params = fit_model(
                    model_name,
                    X_fit,
                    y_outer,
                    preprocessor.selected_feature_names,
                    seed=outer_seed + outer_fold * 1009 + model_index * 53,
                    fast=config.run_mode == "smoke",
                )
                p_valid = fitted.predict_proba(X_valid)
                outer_probabilities[model_name] = p_valid
                per_model_metrics[model_name] = binary_metrics(
                    y[valid_idx], p_valid[:, 1], threshold=0.5
                )
                params_by_model[model_name] = params
                if X_external is not None:
                    external_probabilities[model_name] = fitted.predict_proba(X_external)
                _save_model(fitted, model_name, fold_dir)

            p_outer = apply_blend(outer_probabilities, chosen["weights"])
            threshold = float(chosen["threshold"])
            decision = (p_outer >= threshold).astype(np.int8)
            metrics = binary_metrics(y[valid_idx], p_outer, threshold=threshold)
            repeat_probabilities[repeat, valid_idx] = p_outer
            repeat_predictions[repeat, valid_idx] = decision
            repeat_thresholds[repeat, valid_idx] = threshold
            repeat_seen[repeat, valid_idx] += 1
            row = {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "n_train": len(fit_idx),
                "n_valid": len(valid_idx),
                "metrics": metrics,
                "inner_selection": chosen,
                "per_model_metrics": per_model_metrics,
                "params": params_by_model,
                "elapsed_seconds": time.monotonic() - fold_started,
            }
            outer_rows.append(row)
            write_json(fold_dir / "selection.json", selection)
            write_json(fold_dir / "fold_report.json", row)

            if external is not None:
                external_probability_sum += apply_blend(
                    external_probabilities, chosen["weights"]
                )
                external_threshold_sum += threshold
                external_fold_count += 1
            pd.DataFrame(
                [
                    {
                        "repeat": item["repeat"],
                        "outer_fold": item["outer_fold"],
                        "n_train": item["n_train"],
                        "n_valid": item["n_valid"],
                        **{
                            key: item["metrics"][key]
                            for key in (
                                "accuracy",
                                "balanced_accuracy",
                                "f1_impaired",
                                "roc_auc",
                                "pr_auc",
                                "log_loss",
                                "threshold",
                            )
                        },
                    }
                    for item in outer_rows
                ]
            ).to_csv(training_output / "outer_fold_metrics.csv", index=False)
            write_json(
                training_output / "nested_progress.json",
                {
                    "completed_outer_folds": len(outer_rows),
                    "expected_outer_folds": config.outer_repeats * config.outer_folds,
                    "elapsed_seconds": time.monotonic() - start,
                    "runtime_budget_seconds": config.max_runtime_seconds,
                },
            )

    if not np.all(repeat_seen == 1):
        raise AssertionError("Every subject must appear exactly once per outer repeat")
    if not np.isfinite(repeat_thresholds).all():
        raise AssertionError("Every OOF subject must have its fold-specific threshold")
    repeat_reports = []
    for repeat in range(config.outer_repeats):
        report = binary_metrics(
            y,
            repeat_probabilities[repeat],
            threshold=0.5,
            predicted=repeat_predictions[repeat],
        )
        report["threshold"] = None
        report["threshold_policy"] = "fold-specific threshold selected on inner OOF"
        report["fold_threshold_summary"] = {
            "min": float(repeat_thresholds[repeat].min()),
            "mean": float(repeat_thresholds[repeat].mean()),
            "max": float(repeat_thresholds[repeat].max()),
        }
        repeat_reports.append(report)
    mean_probability = repeat_probabilities.mean(axis=0)
    majority_prediction = (repeat_predictions.mean(axis=0) >= 0.5).astype(np.int8)
    aggregate_metrics = binary_metrics(
        y, mean_probability, threshold=0.5, predicted=majority_prediction
    )
    aggregate_metrics["threshold"] = None
    aggregate_metrics["threshold_policy"] = (
        "majority vote of repeat/fold decisions using inner-OOF thresholds"
    )
    repeat_summary = {
        key: {
            "mean": float(np.nanmean([report[key] for report in repeat_reports])),
            "std": float(np.nanstd([report[key] for report in repeat_reports])),
        }
        for key in (
            "accuracy",
            "balanced_accuracy",
            "f1_impaired",
            "roc_auc",
            "pr_auc",
            "log_loss",
        )
    }
    oof_frame = pd.DataFrame(
        {
            "subject_hash": [subject_hash(str(value)) for value in train_data.subject_ids],
            "true_class": [CLASS_NAMES[int(value)] for value in y],
            "predicted_class": [CLASS_NAMES[int(value)] for value in majority_prediction],
            "p_cn": 1.0 - mean_probability,
            "p_impaired": mean_probability,
        }
    )
    for repeat in range(config.outer_repeats):
        oof_frame[f"threshold_repeat_{repeat:02d}"] = repeat_thresholds[repeat]
        oof_frame[f"predicted_class_repeat_{repeat:02d}"] = [
            CLASS_NAMES[int(value)] for value in repeat_predictions[repeat]
        ]
    oof_frame.to_csv(training_output / "nested_oof_predictions_hashed.csv", index=False)
    nested_report = {
        "cv_unit": CV_UNIT,
        "outer_scheme": f"{config.outer_folds}-fold x {config.outer_repeats} repeats",
        "inner_scheme": f"{config.inner_folds}-fold for blend and threshold selection",
        "repeat_metrics": repeat_reports,
        "repeat_metric_summary": repeat_summary,
        "repeat_averaged_probability_metrics": aggregate_metrics,
        "target_accuracy": TARGET_ACCURACY,
        "target_met_by_repeat_mean": bool(
            repeat_summary["accuracy"]["mean"] >= TARGET_ACCURACY
        ),
        "all_preprocessing_and_selection_fold_local": True,
    }
    write_json(training_output / "nested_cv_report.json", nested_report)

    validation_report = None
    if external is not None:
        expected_folds = config.outer_folds * config.outer_repeats
        if external_fold_count != expected_folds:
            raise AssertionError("Validation ensemble missed an outer fold")
        p_external = np.clip(
            external_probability_sum / external_fold_count, 1e-8, 1.0 - 1e-8
        )
        validation_threshold = float(external_threshold_sum / external_fold_count)
        freeze_validation_predictions(
            training_output,
            external.subject_ids,
            p_external,
            validation_threshold,
        )
        validation_report = evaluate_frozen_validation(
            str(config.validation_root),
            training_output,
            external.subject_ids,
            p_external,
            validation_threshold,
        )

    final_report = {
        "task": "CN (0) vs MCI + DEM (1)",
        "feature_sources": ["Activity", "Sleep"],
        "mmse_source_opened_or_used": False,
        "cv_unit": CV_UNIT,
        "models": list(model_names),
        "google_model_evidence": GOOGLE_MODEL_EVIDENCE,
        "nested_cv": nested_report,
        "validation": validation_report,
        "accuracy_target": TARGET_ACCURACY,
        "accuracy_target_is_guaranteed": False,
        "elapsed_seconds": time.monotonic() - start,
        "runtime_budget_seconds": config.max_runtime_seconds,
    }
    write_json(training_output / "FINAL_REPORT.json", final_report)
    write_json(
        training_output / "TRAINING_COMPLETE.json",
        {
            "success": True,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - start,
            "output_dir": str(run_root),
        },
    )
    return final_report


if __name__ == "__main__":
    raise SystemExit(
        "Use run.py (or base.ipynb RUN_FILE) so paths and dependency checks are applied."
    )
