"""Nested-CV training entry point for the three-model CN/MCI/DEM ensemble.

This file is intentionally not executed while preparing the repository.  Run it
from ``SangHyo/base_sanghyo.ipynb`` on the requested A100 / High-RAM runtime.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from feature_engineering import (
    CLASS_NAMES,
    build_subject_dataset,
    discover_split_files,
    load_labels,
)
from models import MODEL_NAMES, fit_model, set_global_seed, suggest_parameters
from preprocessing import FoldPreprocessor


TARGETS = {
    "accuracy": 0.80,
    "macro_f1": 0.70,
    "roc_auc_ovr_macro": 0.85,
}


@dataclass(frozen=True)
class RunConfig:
    training_root: str
    validation_root: str | None
    output_dir: str
    feature_mode: str
    outer_folds: int
    outer_repeats: int
    inner_folds: int
    trials_transformer: int
    trials_tabnet: int
    trials_ydf: int
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
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def subject_hash(subject_id: str) -> str:
    # This is a reporting pseudonym, never a model feature.
    return hashlib.sha256(f"cn-mci-dem-report-v1::{subject_id}".encode()).hexdigest()[:20]


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    p = np.clip(p, 1e-8, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def metrics_from_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    p = normalize_probabilities(probabilities)
    predicted = p.argmax(axis=1)
    report = classification_report(
        y_true,
        predicted,
        labels=[0, 1, 2],
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    try:
        auc = float(roc_auc_score(y_true, p, labels=[0, 1, 2], multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": auc,
        "log_loss": float(log_loss(y_true, p, labels=[0, 1, 2])),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1, 2]).tolist(),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "support": {name: int(np.sum(y_true == i)) for i, name in enumerate(CLASS_NAMES)},
    }


def composite_score(metrics: dict[str, Any]) -> float:
    auc = metrics["roc_auc_ovr_macro"]
    auc = 0.5 if not np.isfinite(auc) else auc
    return float(
        0.40 * metrics["macro_f1"]
        + 0.25 * auc
        + 0.20 * metrics["accuracy"]
        + 0.15 * metrics["balanced_accuracy"]
    )


def apply_class_scales(probabilities: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return normalize_probabilities(probabilities * np.asarray(scales, dtype=float)[None, :])


def optimize_inner_blend(
    y: np.ndarray,
    model_probabilities: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Choose ensemble weights and class scales using inner OOF predictions only."""
    names = list(MODEL_NAMES)
    scale_values = (0.70, 0.85, 1.0, 1.20, 1.45, 1.75)
    best: dict[str, Any] | None = None
    # 0.10 simplex grid: 66 model-weight combinations.
    for transformer_units in range(11):
        for tabnet_units in range(11 - transformer_units):
            ydf_units = 10 - transformer_units - tabnet_units
            weights = np.array([transformer_units, tabnet_units, ydf_units], dtype=float) / 10.0
            blended = sum(weights[i] * model_probabilities[name] for i, name in enumerate(names))
            for mci_scale in scale_values:
                for dem_scale in scale_values:
                    scales = np.array([1.0, mci_scale, dem_scale], dtype=float)
                    adjusted = apply_class_scales(blended, scales)
                    metrics = metrics_from_probabilities(y, adjusted)
                    score = composite_score(metrics)
                    candidate = {
                        "score": score,
                        "log_loss": metrics["log_loss"],
                        "weights": {name: float(weights[i]) for i, name in enumerate(names)},
                        "class_scales": scales.tolist(),
                        "metrics": metrics,
                    }
                    if best is None or (score, -metrics["log_loss"]) > (
                        best["score"], -best["log_loss"]
                    ):
                        best = candidate
    assert best is not None
    return best


def apply_blend(model_probabilities: dict[str, np.ndarray], blend: dict[str, Any]) -> np.ndarray:
    combined = sum(
        float(blend["weights"][name]) * model_probabilities[name]
        for name in MODEL_NAMES
    )
    return apply_class_scales(combined, np.asarray(blend["class_scales"], dtype=float))


def _trial_count(config: RunConfig, model_name: str) -> int:
    return {
        "transformer": config.trials_transformer,
        "tabnet": config.trials_tabnet,
        "ydf": config.trials_ydf,
    }[model_name]


def tune_model(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    config: RunConfig,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import optuna

    def objective(trial):
        params = suggest_parameters(model_name, trial, fast=config.fast)
        fold_metrics = []
        for fold_id, (fit_idx, valid_idx) in enumerate(splits):
            preprocessor = FoldPreprocessor(max_features=int(params["max_features"]))
            X_fit = preprocessor.fit_transform(X.iloc[fit_idx], y[fit_idx])
            X_valid = preprocessor.transform(X.iloc[valid_idx])
            model = fit_model(
                model_name,
                X_fit,
                y[fit_idx],
                params,
                seed=seed + trial.number * 1009 + fold_id * 53,
            )
            p_valid = model.predict_proba(X_valid)
            fold_metrics.append(metrics_from_probabilities(y[valid_idx], p_valid))
        means = {
            key: float(np.nanmean([m[key] for m in fold_metrics]))
            for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro", "log_loss")
        }
        score = composite_score(means)
        trial.set_user_attr("resolved_params", _jsonable(params))
        trial.set_user_attr("fold_metrics", _jsonable(fold_metrics))
        trial.set_user_attr("mean_metrics", _jsonable(means))
        return score

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=max(1, _trial_count(config, model_name)),
        n_jobs=1,
        gc_after_trial=True,
        catch=(FloatingPointError, RuntimeError, ValueError),
        show_progress_bar=True,
    )
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        raise RuntimeError(f"All {model_name} tuning trials failed")
    best_params = dict(study.best_trial.user_attrs["resolved_params"])
    trial_records = [
        {
            "model": model_name,
            "number": t.number,
            "state": str(t.state),
            "value": t.value,
            "params": t.user_attrs.get("resolved_params", t.params),
            "mean_metrics": t.user_attrs.get("mean_metrics"),
        }
        for t in study.trials
    ]
    return best_params, trial_records


def collect_inner_oof(
    model_name: str,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> np.ndarray:
    probabilities = np.zeros((len(y), 3), dtype=np.float64)
    seen = np.zeros(len(y), dtype=np.int16)
    for fold_id, (fit_idx, valid_idx) in enumerate(splits):
        preprocessor = FoldPreprocessor(max_features=int(params["max_features"]))
        X_fit = preprocessor.fit_transform(X.iloc[fit_idx], y[fit_idx])
        X_valid = preprocessor.transform(X.iloc[valid_idx])
        model = fit_model(model_name, X_fit, y[fit_idx], params, seed + fold_id * 97)
        probabilities[valid_idx] = model.predict_proba(X_valid)
        seen[valid_idx] += 1
    if not np.all(seen == 1):
        raise AssertionError(f"Inner OOF coverage error for {model_name}: {seen}")
    return probabilities


def target_check(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {name: bool(metrics[name] >= threshold) for name, threshold in TARGETS.items()}
    return {
        "thresholds": TARGETS,
        "checks": checks,
        "all_targets_met": bool(all(checks.values())),
        "note": "목표값은 희망 기준이며 데이터가 보장하는 값이 아닙니다.",
    }


def runtime_info() -> dict[str, Any]:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        payload.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except ImportError:
        payload["torch"] = None
    return payload


def run(config: RunConfig) -> None:
    output = Path(config.output_dir).expanduser().resolve()
    if (output / "TRAINING_COMPLETE.json").exists():
        raise FileExistsError(f"Completed run will not be overwritten: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run_config.json", asdict(config))
    write_json(output / "environment.json", runtime_info())
    set_global_seed(config.seed)

    train_data = build_subject_dataset(
        config.training_root,
        feature_mode=config.feature_mode,
        require_labels=True,
    )
    if train_data.y is None:
        raise AssertionError("Training target is unexpectedly missing")
    X = train_data.X
    y = train_data.y
    counts = np.bincount(y, minlength=3)
    if len(y) != 141 or counts.tolist() != [85, 47, 9]:
        raise AssertionError(
            f"Training cohort differs from the audited contract: n={len(y)}, counts={counts.tolist()}"
        )
    if config.outer_folds > int(counts.min()) or config.inner_folds > 3:
        raise ValueError("Fold count is too high for the nine DEM subjects; use 3x3 nested CV")
    write_json(output / "training_feature_audit.json", train_data.audit)
    write_json(
        output / "feature_manifest.json",
        {
            "feature_mode": config.feature_mode,
            "raw_engineered_feature_count": X.shape[1],
            "feature_names": X.columns.tolist(),
            "direct_diagnosis_columns_used": False,
        },
    )

    external = None
    external_accumulator = np.zeros((0, 3), dtype=np.float64)
    external_model_count = 0
    if config.validation_root:
        # Labels are intentionally not opened here. Predictions are materialized first.
        external = build_subject_dataset(
            config.validation_root,
            feature_mode=config.feature_mode,
            require_labels=False,
        )
        missing_external = sorted(set(X.columns) - set(external.X.columns))
        extra_external = sorted(set(external.X.columns) - set(X.columns))
        if missing_external or extra_external:
            raise AssertionError(
                f"Train/validation feature schema mismatch; missing={missing_external[:5]}, "
                f"extra={extra_external[:5]}"
            )
        external.X = external.X[X.columns]
        external_accumulator = np.zeros((len(external.subject_ids), 3), dtype=np.float64)
        write_json(output / "validation_feature_audit_label_free.json", external.audit)

    oof_sum = np.zeros((len(y), 3), dtype=np.float64)
    oof_count = np.zeros(len(y), dtype=np.int16)
    outer_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    selected_feature_rows: list[dict[str, Any]] = []

    for repeat in range(config.outer_repeats):
        outer_seed = config.seed + repeat * 100_003
        outer_cv = StratifiedKFold(
            n_splits=config.outer_folds, shuffle=True, random_state=outer_seed
        )
        for outer_fold, (outer_train_idx, outer_valid_idx) in enumerate(outer_cv.split(X, y)):
            fold_tag = f"repeat_{repeat:02d}_fold_{outer_fold:02d}"
            fold_dir = output / "models" / fold_tag
            X_outer_train = X.iloc[outer_train_idx].reset_index(drop=True)
            y_outer_train = y[outer_train_idx]
            X_outer_valid = X.iloc[outer_valid_idx]
            y_outer_valid = y[outer_valid_idx]
            inner_seed = outer_seed + outer_fold * 1009 + 17
            inner_cv = StratifiedKFold(
                n_splits=config.inner_folds, shuffle=True, random_state=inner_seed
            )
            inner_splits = list(inner_cv.split(X_outer_train, y_outer_train))

            best_params: dict[str, dict[str, Any]] = {}
            inner_model_probabilities: dict[str, np.ndarray] = {}
            for model_index, model_name in enumerate(MODEL_NAMES):
                model_seed = inner_seed + model_index * 10_007
                params, trial_records = tune_model(
                    model_name,
                    X_outer_train,
                    y_outer_train,
                    inner_splits,
                    config=config,
                    seed=model_seed,
                )
                best_params[model_name] = params
                for record in trial_records:
                    record.update({"repeat": repeat, "outer_fold": outer_fold})
                tuning_rows.extend(trial_records)
                inner_model_probabilities[model_name] = collect_inner_oof(
                    model_name,
                    params,
                    X_outer_train,
                    y_outer_train,
                    inner_splits,
                    seed=model_seed + 900_001,
                )

            blend = optimize_inner_blend(y_outer_train, inner_model_probabilities)
            write_json(
                fold_dir / "selection.json",
                {"best_params": best_params, "inner_blend": blend},
            )

            outer_model_probabilities: dict[str, np.ndarray] = {}
            external_model_probabilities: dict[str, np.ndarray] = {}
            per_model_metrics: dict[str, Any] = {}
            for model_index, model_name in enumerate(MODEL_NAMES):
                params = best_params[model_name]
                preprocessor = FoldPreprocessor(max_features=int(params["max_features"]))
                X_fit = preprocessor.fit_transform(X.iloc[outer_train_idx], y_outer_train)
                X_valid = preprocessor.transform(X_outer_valid)
                fitted = fit_model(
                    model_name,
                    X_fit,
                    y_outer_train,
                    params,
                    seed=outer_seed + outer_fold * 1009 + model_index * 53,
                )
                p_valid = fitted.predict_proba(X_valid)
                outer_model_probabilities[model_name] = p_valid
                per_model_metrics[model_name] = metrics_from_probabilities(y_outer_valid, p_valid)

                model_dir = fold_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(preprocessor, model_dir / "preprocessor.joblib")
                if model_name == "transformer":
                    fitted.save(model_dir / "model.pt")
                elif model_name == "tabnet":
                    fitted.save(model_dir / "model")
                else:
                    fitted.save(model_dir / "model")
                write_json(model_dir / "params.json", params)
                for rank, feature_name in enumerate(preprocessor.selected_feature_names):
                    selected_feature_rows.append(
                        {
                            "repeat": repeat,
                            "outer_fold": outer_fold,
                            "model": model_name,
                            "rank": rank,
                            "feature": feature_name,
                        }
                    )

                if external is not None:
                    X_external = preprocessor.transform(external.X)
                    external_model_probabilities[model_name] = fitted.predict_proba(X_external)

            p_outer = apply_blend(outer_model_probabilities, blend)
            fold_metrics = metrics_from_probabilities(y_outer_valid, p_outer)
            oof_sum[outer_valid_idx] += p_outer
            oof_count[outer_valid_idx] += 1
            outer_rows.append(
                {
                    "repeat": repeat,
                    "outer_fold": outer_fold,
                    "n_train": len(outer_train_idx),
                    "n_valid": len(outer_valid_idx),
                    **{key: fold_metrics[key] for key in (
                        "accuracy", "balanced_accuracy", "macro_f1", "roc_auc_ovr_macro", "log_loss"
                    )},
                    "inner_blend": blend,
                    "per_model_metrics": per_model_metrics,
                    "confusion_matrix": fold_metrics["confusion_matrix"],
                }
            )
            if external is not None:
                external_accumulator += apply_blend(external_model_probabilities, blend)
                external_model_count += 1

            pd.DataFrame(outer_rows).drop(
                columns=["inner_blend", "per_model_metrics", "confusion_matrix"], errors="ignore"
            ).to_csv(output / "outer_fold_metrics.csv", index=False)
            write_json(output / "nested_progress.json", {"completed_folds": outer_rows})

    if not np.all(oof_count == config.outer_repeats):
        raise AssertionError(f"Outer OOF coverage mismatch: {oof_count.tolist()}")
    oof = normalize_probabilities(oof_sum / oof_count[:, None])
    oof_metrics = metrics_from_probabilities(y, oof)
    oof_frame = pd.DataFrame(
        {
            "subject_hash": [subject_hash(s) for s in train_data.subject_ids],
            "true_class": [CLASS_NAMES[int(v)] for v in y],
            "predicted_class": [CLASS_NAMES[int(v)] for v in oof.argmax(axis=1)],
            "p_cn": oof[:, 0],
            "p_mci": oof[:, 1],
            "p_dem": oof[:, 2],
        }
    )
    oof_frame.to_csv(output / "nested_oof_predictions_hashed.csv", index=False)
    pd.DataFrame(outer_rows).drop(
        columns=["inner_blend", "per_model_metrics", "confusion_matrix"], errors="ignore"
    ).to_csv(output / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(selected_feature_rows).to_csv(output / "selected_features_by_fold.csv", index=False)
    write_json(output / "tuning_trials.json", tuning_rows)
    write_json(
        output / "nested_cv_report.json",
        {
            "metrics_from_repeat_averaged_oof": oof_metrics,
            "target_check": target_check(oof_metrics),
            "outer_folds": outer_rows,
            "interpretation": (
                "Nested OOF is an internal development estimate. It is not a promise of external "
                "clinical performance, especially with only nine DEM training subjects."
            ),
        },
    )

    validation_report = None
    if external is not None:
        if external_model_count != config.outer_repeats * config.outer_folds:
            raise AssertionError("Validation ensemble did not receive every outer model")
        p_external = normalize_probabilities(external_accumulator / external_model_count)
        validation_predictions = pd.DataFrame(
            {
                "subject_hash": [subject_hash(s) for s in external.subject_ids],
                "predicted_class": [CLASS_NAMES[int(v)] for v in p_external.argmax(axis=1)],
                "p_cn": p_external[:, 0],
                "p_mci": p_external[:, 1],
                "p_dem": p_external[:, 2],
            }
        )
        # This write occurs before the validation label file is opened below.
        prediction_path = output / "validation_predictions_label_free_hashed.csv"
        validation_predictions.to_csv(prediction_path, index=False)
        write_json(
            output / "VALIDATION_PREDICTIONS_FROZEN.json",
            {
                "prediction_file": prediction_path.name,
                "rows": len(validation_predictions),
                "labels_opened_before_prediction_write": False,
            },
        )

        validation_files = discover_split_files(config.validation_root, require_label=True)
        validation_labels = load_labels(validation_files.label).set_index("subject_id")["target"]
        aligned_y = pd.Series(external.subject_ids).map(validation_labels).to_numpy()
        if pd.isna(aligned_y).any():
            raise AssertionError("Some validation source subjects have no label")
        aligned_y = aligned_y.astype(np.int64)
        validation_report = metrics_from_probabilities(aligned_y, p_external)
        validation_predictions.insert(
            1, "true_class", [CLASS_NAMES[int(v)] for v in aligned_y]
        )
        validation_predictions.to_csv(output / "validation_predictions_evaluated_hashed.csv", index=False)
        write_json(
            output / "validation_report.json",
            {
                "metrics": validation_report,
                "target_check": target_check(validation_report),
                "warning": "Validation is evaluated once after label-free predictions are frozen.",
            },
        )

    write_json(
        output / "FINAL_REPORT.json",
        {
            "nested_oof": oof_metrics,
            "validation": validation_report,
            "target_check_nested_oof": target_check(oof_metrics),
            "models": list(MODEL_NAMES),
            "feature_mode": config.feature_mode,
            "direct_label_leakage_blocked": True,
        },
    )
    write_json(
        output / "TRAINING_COMPLETE.json",
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "success": True,
            "output_dir": str(output),
        },
    )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--validation-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-mode",
        choices=["clinical_plus_lifelog", "wearable_only"],
        default="clinical_plus_lifelog",
    )
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--outer-repeats", type=int, default=2)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--trials-transformer", type=int, default=24)
    parser.add_argument("--trials-tabnet", type=int, default=24)
    parser.add_argument("--trials-ydf", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    if args.fast:
        args.outer_repeats = 1
        args.trials_transformer = min(args.trials_transformer, 1)
        args.trials_tabnet = min(args.trials_tabnet, 1)
        args.trials_ydf = min(args.trials_ydf, 1)
    return RunConfig(**vars(args))


if __name__ == "__main__":
    run(parse_args())

