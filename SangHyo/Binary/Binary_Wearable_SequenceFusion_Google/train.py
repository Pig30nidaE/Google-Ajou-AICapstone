"""Subject-level repeated CV and frozen historical validation evaluation."""

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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

if __package__:
    from .data import (
        SubjectSequenceDataset,
        build_subject_dataset,
        load_binary_labels,
        make_fixed_views,
    )
    from .eda import run_training_eda
    from .models import (
        CLASS_NAMES,
        ENSEMBLE_WEIGHTS,
        GOOGLE_MODEL_EVIDENCE,
        MODEL_NAMES,
        aggregate_view_probabilities,
        blend_probabilities,
        fit_neural_fixed_epochs,
        fit_ydf,
        select_neural_epoch,
        set_global_seed,
    )
    from .preprocessing import (
        SequencePreprocessor,
        StableSummarySelector,
        build_subject_summary_features,
    )
else:
    from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google.data import (  # type: ignore
        SubjectSequenceDataset,
        build_subject_dataset,
        load_binary_labels,
        make_fixed_views,
    )
    from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google.eda import run_training_eda  # type: ignore
    from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google.models import (  # type: ignore
        CLASS_NAMES,
        ENSEMBLE_WEIGHTS,
        GOOGLE_MODEL_EVIDENCE,
        MODEL_NAMES,
        aggregate_view_probabilities,
        blend_probabilities,
        fit_neural_fixed_epochs,
        fit_ydf,
        select_neural_epoch,
        set_global_seed,
    )
    from SangHyo.Binary.Binary_Wearable_SequenceFusion_Google.preprocessing import (  # type: ignore
        SequencePreprocessor,
        StableSummarySelector,
        build_subject_summary_features,
    )


TARGET_ACCURACY = 0.90
CV_UNIT = "subject"
HISTORICAL_VALIDATION_WARNING = (
    "The 33-person Validation split has already been inspected by earlier experiments. "
    "It is a historical benchmark, not a fresh untouched test set."
)


@dataclass(frozen=True)
class RunConfig:
    training_root: str
    validation_root: str
    output_dir: str
    run_mode: str = "full"
    seed: int = 20260722
    sequence_length: int = 28
    views_per_subject: int = 8
    outer_folds: int = 5
    outer_repeats: int = 2
    max_runtime_seconds: int = 20_700
    evaluate_historical_validation: bool = True


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
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_hash(subject_id: str) -> str:
    return hashlib.sha256(
        f"binary-sequence-fusion-v1::{subject_id}".encode("utf-8")
    ).hexdigest()[:20]


def _safe_metric(metric, *args, **kwargs) -> float:
    try:
        return float(metric(*args, **kwargs))
    except ValueError:
        return float("nan")


def binary_metrics(
    y_true: np.ndarray, p_impaired: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    target = np.asarray(y_true, dtype=np.int64)
    probability = np.clip(np.asarray(p_impaired, dtype=np.float64), 1e-7, 1 - 1e-7)
    prediction = (probability >= threshold).astype(np.int64)
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    tn, fp, fn, tp = [int(item) for item in matrix.ravel()]
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
        "precision_impaired": float(
            precision_score(target, prediction, zero_division=0)
        ),
        "recall_impaired": float(recall_score(target, prediction, zero_division=0)),
        "specificity_cn": float(tn / max(1, tn + fp)),
        "roc_auc": _safe_metric(roc_auc_score, target, probability),
        "pr_auc": _safe_metric(average_precision_score, target, probability),
        "log_loss": _safe_metric(
            log_loss,
            target,
            np.column_stack([1 - probability, probability]),
            labels=[0, 1],
        ),
        "brier_score": float(brier_score_loss(target, probability)),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "support": {
            "CN": int((target == 0).sum()),
            "MCI_DEM": int((target == 1).sum()),
        },
        "per_class": {name: report[name] for name in CLASS_NAMES},
    }


def _select_threshold(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    """Choose one coarse threshold from Training OOF predictions only."""

    rows: list[dict[str, Any]] = []
    for threshold in np.round(np.arange(0.35, 0.751, 0.05), 2):
        metrics = binary_metrics(y, probability, float(threshold))
        score = (
            0.55 * metrics["accuracy"]
            + 0.30 * metrics["balanced_accuracy"]
            + 0.15 * metrics["f1_impaired"]
            - 0.01 * abs(float(threshold) - 0.5)
        )
        rows.append(
            {
                "threshold": float(threshold),
                "selection_score": float(score),
                "minimum_class_recall": float(
                    min(metrics["recall_impaired"], metrics["specificity_cn"])
                ),
                **metrics,
            }
        )
    eligible = [
        row
        for row in rows
        if row["recall_impaired"] >= 0.40 and row["specificity_cn"] >= 0.60
    ]
    if eligible:
        chosen = max(
            eligible,
            key=lambda row: (
                row["selection_score"],
                row["balanced_accuracy"],
                -abs(row["threshold"] - 0.5),
            ),
        )
        feasibility = "met recall_impaired>=0.40 and specificity_cn>=0.60"
    else:
        # If probabilities have no useful operating point, do not trade nearly
        # all of one class for the other.  Maximize the worse class recall.
        chosen = max(
            rows,
            key=lambda row: (
                row["minimum_class_recall"],
                row["balanced_accuracy"],
                row["accuracy"],
                -abs(row["threshold"] - 0.5),
            ),
        )
        feasibility = "no threshold met both constraints; maximin class recall fallback"
    return {
        "threshold": float(chosen["threshold"]),
        "chosen": chosen,
        "grid": rows,
        "selection_scope": "pooled repeated Training OOF predictions only",
        "constraint_policy": feasibility,
        "caution": (
            "Metrics at the selected threshold reuse OOF labels for threshold choice; "
            "OOF @0.5 remains the primary untuned CV estimate."
        ),
    }


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "sklearn", "ydf", "torch"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "installed")
        except ImportError:
            info[name] = None
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


def _assert_dataset_contract(
    dataset: SubjectSequenceDataset, expected_counts: tuple[int, int]
) -> None:
    if dataset.y is None:
        raise AssertionError("A labeled Training dataset is required")
    if len(dataset.subject_ids) != len(set(map(str, dataset.subject_ids))):
        raise AssertionError("Subject IDs must be unique")
    counts = tuple(int(item) for item in np.bincount(dataset.y, minlength=2))
    if counts != expected_counts:
        raise AssertionError(f"Unexpected class counts: {counts} != {expected_counts}")
    if any(len(sequence) < 28 for sequence in dataset.sequences):
        raise AssertionError("Every subject needs at least 28 aligned wearable events")


def _view_subset(
    views: np.ndarray,
    view_subject: np.ndarray,
    subject_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    wanted = np.asarray(subject_indices, dtype=np.int64)
    remap = {int(original): local for local, original in enumerate(wanted.tolist())}
    keep = np.isin(view_subject, wanted)
    mapping = np.asarray([remap[int(item)] for item in view_subject[keep]], dtype=np.int64)
    counts = np.bincount(mapping, minlength=len(wanted))
    if len(set(counts.tolist())) != 1:
        raise AssertionError(f"Views are not subject-balanced: {counts.tolist()}")
    return views[keep], mapping


def _aggregate_rows(
    rows: np.ndarray, view_subject: np.ndarray, n_subjects: int
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float64)
    result = np.zeros((n_subjects, values.shape[1]), dtype=np.float64)
    counts = np.zeros(n_subjects, dtype=np.int64)
    np.add.at(result, view_subject, values)
    np.add.at(counts, view_subject, 1)
    if np.any(counts == 0) or len(set(counts.tolist())) != 1:
        raise AssertionError(f"Unequal subject rows: {counts.tolist()}")
    return (result / counts[:, None]).astype(np.float32)


def _summary_for_views(
    transformed_views: np.ndarray,
    view_subject: np.ndarray,
    n_subjects: int,
    feature_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    view_summary, names = build_subject_summary_features(
        transformed_views, feature_names, recent_days=7
    )
    return _aggregate_rows(view_summary, view_subject, n_subjects), names


def _deadline(deadline: float, stage: str) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError(
            f"Soft runtime budget reached before {stage}. Partial progress is saved."
        )


def _fold_seed(base: int, repeat: int, fold: int, offset: int = 0) -> int:
    return int(base + repeat * 10_003 + fold * 701 + offset)


def _fit_predict_fold(
    *,
    config: RunConfig,
    repeat: int,
    fold: int,
    train_idx: np.ndarray,
    outer_idx: np.ndarray,
    train_dataset: SubjectSequenceDataset,
    train_views: np.ndarray,
    train_view_subject: np.ndarray,
    validation_dataset: SubjectSequenceDataset,
    validation_views: np.ndarray,
    validation_view_subject: np.ndarray,
    fold_dir: Path,
    deadline: float,
) -> dict[str, Any]:
    assert train_dataset.y is not None
    seed = _fold_seed(config.seed, repeat, fold)
    fast = config.run_mode == "smoke"
    fold_dir.mkdir(parents=True, exist_ok=True)

    fit_raw, fit_map = _view_subset(train_views, train_view_subject, train_idx)
    outer_raw, outer_map = _view_subset(train_views, train_view_subject, outer_idx)
    preprocessor = SequencePreprocessor(
        view_days=config.sequence_length,
        fit_scope="current outer-fold training subject views only",
    )
    preprocessor.fit(fit_raw, train_dataset.feature_names)
    fit_views = preprocessor.transform_views(fit_raw)
    outer_views = preprocessor.transform_views(outer_raw)
    validation_transformed = preprocessor.transform_views(validation_views)

    fit_summary, summary_names = _summary_for_views(
        fit_views, fit_map, len(train_idx), train_dataset.feature_names
    )
    outer_summary, _ = _summary_for_views(
        outer_views, outer_map, len(outer_idx), train_dataset.feature_names
    )
    validation_summary, _ = _summary_for_views(
        validation_transformed,
        validation_view_subject,
        len(validation_dataset.subject_ids),
        train_dataset.feature_names,
    )
    y_fit_subject = train_dataset.y[train_idx]
    y_outer = train_dataset.y[outer_idx]
    summary_selector = StableSummarySelector(
        max_features=64 if fast else 160,
        bootstrap_rounds=4 if fast else 32,
        minimum_per_modality=16 if fast else 40,
        minimum_per_statistic=3 if fast else 8,
        seed=seed + 19,
    )
    fit_summary = summary_selector.fit_transform(
        fit_summary, y_fit_subject, summary_names
    )
    outer_summary = summary_selector.transform(outer_summary)
    validation_summary = summary_selector.transform(validation_summary)
    summary_names = summary_selector.selected_feature_names
    fold_probabilities: dict[str, np.ndarray] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    model_audit: dict[str, Any] = {}

    for model_offset, model_name in enumerate(("ydf_gbt", "ydf_rf")):
        _deadline(deadline, f"repeat {repeat} fold {fold} {model_name}")
        model = fit_ydf(
            model_name,
            fit_summary,
            y_fit_subject,
            summary_names,
            seed=seed + model_offset * 31,
            fast=fast,
        )
        fold_probabilities[model_name] = model.predict_proba(outer_summary)[:, 1]
        validation_probabilities[model_name] = model.predict_proba(
            validation_summary
        )[:, 1]
        if not fast:
            model.save(fold_dir / model_name)
        model_audit[model_name] = {"input": "subject_summary", "features": len(summary_names)}

    early_splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.20, random_state=seed + 83
    )
    core_pos, early_pos = next(
        early_splitter.split(np.zeros(len(train_idx)), y_fit_subject)
    )
    core_subjects = train_idx[core_pos]
    early_subjects = train_idx[early_pos]
    core_views_raw, core_map = _view_subset(
        train_views, train_view_subject, core_subjects
    )
    early_views_raw, early_map = _view_subset(
        train_views, train_view_subject, early_subjects
    )
    # Epoch selection has its own core-only value transformer.  Even the
    # unsupervised median/quantiles must not see the internal early-stop
    # subjects.  The outer-fold transformer above is used only for the final
    # refit and outer/Validation prediction.
    epoch_preprocessor = SequencePreprocessor(
        view_days=config.sequence_length,
        fit_scope="internal core-subject views only",
    )
    epoch_preprocessor.fit(core_views_raw, train_dataset.feature_names)
    core_views = epoch_preprocessor.transform_views(core_views_raw)
    early_views = epoch_preprocessor.transform_views(early_views_raw)
    core_y_view = train_dataset.y[core_subjects][core_map]
    early_y_subject = train_dataset.y[early_subjects]
    fit_y_view = y_fit_subject[fit_map]

    for model_offset, model_name in enumerate(
        ("conv_bilstm", "sequence_transformer"), start=2
    ):
        _deadline(deadline, f"repeat {repeat} fold {fold} {model_name}")
        model_seed = seed + model_offset * 31
        best_epoch, selection_history = select_neural_epoch(
            model_name,
            core_views,
            core_y_view,
            early_views,
            early_y_subject,
            early_map,
            seed=model_seed,
            fast=fast,
        )
        model, refit_history = fit_neural_fixed_epochs(
            model_name,
            fit_views,
            fit_y_view,
            epochs=best_epoch,
            seed=model_seed + 1,
        )
        outer_view_probability = model.predict_proba(outer_views)[:, 1]
        validation_view_probability = model.predict_proba(
            validation_transformed
        )[:, 1]
        fold_probabilities[model_name] = aggregate_view_probabilities(
            outer_view_probability, outer_map, len(outer_idx)
        )
        validation_probabilities[model_name] = aggregate_view_probabilities(
            validation_view_probability,
            validation_view_subject,
            len(validation_dataset.subject_ids),
        )
        if not fast:
            model.save(fold_dir / f"{model_name}.pt")
        write_json(fold_dir / f"{model_name}_epoch_selection.json", selection_history)
        write_json(fold_dir / f"{model_name}_refit_history.json", refit_history)
        model_audit[model_name] = {
            "input": "eight fixed 28-event views",
            "best_epoch_from_internal_subject_holdout": int(best_epoch),
            "core_subjects": int(len(core_subjects)),
            "early_stop_subjects": int(len(early_subjects)),
            "epoch_preprocessor_fit_scope": "internal core subjects only",
        }

    blended_outer = blend_probabilities(fold_probabilities)
    blended_validation = blend_probabilities(validation_probabilities)
    report = {
        "repeat": repeat,
        "fold": fold,
        "seed": seed,
        "cv_unit": CV_UNIT,
        "train_subjects": int(len(train_idx)),
        "outer_subjects": int(len(outer_idx)),
        "subject_overlap": 0,
        "class_counts_train": np.bincount(y_fit_subject, minlength=2),
        "class_counts_outer": np.bincount(y_outer, minlength=2),
        "preprocessing": preprocessor.manifest(),
        "summary_selection": summary_selector.manifest(),
        "models": model_audit,
        "metrics_at_0_5": {
            name: binary_metrics(y_outer, probability, 0.5)
            for name, probability in {**fold_probabilities, "ensemble": blended_outer}.items()
        },
    }
    write_json(fold_dir / "fold_report.json", report)
    joblib.dump(preprocessor, fold_dir / "preprocessor.joblib")
    joblib.dump(summary_selector, fold_dir / "summary_selector.joblib")
    return {
        "outer_indices": outer_idx,
        "outer_probabilities": {**fold_probabilities, "ensemble": blended_outer},
        "validation_probabilities": {
            **validation_probabilities,
            "ensemble": blended_validation,
        },
        "report": report,
    }


def _final_refit_validation_predictions(
    *,
    config: RunConfig,
    train_dataset: SubjectSequenceDataset,
    train_views: np.ndarray,
    train_view_subject: np.ndarray,
    validation_dataset: SubjectSequenceDataset,
    validation_views: np.ndarray,
    validation_view_subject: np.ndarray,
    fold_reports: list[dict[str, Any]],
    models_dir: Path,
    deadline: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Refit the frozen recipe on all 141 Training subjects before prediction."""

    assert train_dataset.y is not None
    fast = config.run_mode == "smoke"
    selected_epochs = {
        model_name: max(
            1,
            int(
                round(
                    np.median(
                        [
                            report["models"][model_name][
                                "best_epoch_from_internal_subject_holdout"
                            ]
                            for report in fold_reports
                        ]
                    )
                )
            ),
        )
        for model_name in ("conv_bilstm", "sequence_transformer")
    }
    preprocessor = SequencePreprocessor(
        view_days=config.sequence_length,
        fit_scope="all 141 Training-subject views for frozen final refit",
    )
    preprocessor.fit(train_views, train_dataset.feature_names)
    train_transformed = preprocessor.transform_views(train_views)
    validation_transformed = preprocessor.transform_views(validation_views)
    train_summary_base, summary_names = _summary_for_views(
        train_transformed,
        train_view_subject,
        len(train_dataset.subject_ids),
        train_dataset.feature_names,
    )
    validation_summary_base, _ = _summary_for_views(
        validation_transformed,
        validation_view_subject,
        len(validation_dataset.subject_ids),
        train_dataset.feature_names,
    )
    final_root = models_dir / "final_full_training_refits"
    final_root.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, final_root / "preprocessor.joblib")
    write_json(final_root / "preprocessor_manifest.json", preprocessor.manifest())
    seed_count = 1 if fast else 2
    per_model: dict[str, list[np.ndarray]] = {name: [] for name in MODEL_NAMES}
    per_seed_ensemble: list[np.ndarray] = []
    seed_reports: list[dict[str, Any]] = []
    y_view = train_dataset.y[train_view_subject]
    for seed_index in range(seed_count):
        _deadline(deadline, f"final full-training refit seed {seed_index}")
        seed = config.seed + 50_021 + seed_index * 10_007
        seed_dir = final_root / f"seed_{seed_index:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        selector = StableSummarySelector(
            max_features=64 if fast else 160,
            bootstrap_rounds=4 if fast else 32,
            minimum_per_modality=16 if fast else 40,
            minimum_per_statistic=3 if fast else 8,
            seed=seed + 19,
            fit_scope="all 141 Training subjects for frozen final refit",
        )
        train_summary = selector.fit_transform(
            train_summary_base, train_dataset.y, summary_names
        )
        validation_summary = selector.transform(validation_summary_base)
        selected_names = selector.selected_feature_names
        joblib.dump(selector, seed_dir / "summary_selector.joblib")
        write_json(seed_dir / "summary_selector_manifest.json", selector.manifest())
        probabilities: dict[str, np.ndarray] = {}
        for model_offset, model_name in enumerate(("ydf_gbt", "ydf_rf")):
            _deadline(deadline, f"final seed {seed_index} {model_name}")
            model = fit_ydf(
                model_name,
                train_summary,
                train_dataset.y,
                selected_names,
                seed=seed + model_offset * 31,
                fast=fast,
            )
            probabilities[model_name] = model.predict_proba(validation_summary)[:, 1]
            model.save(seed_dir / model_name)
        neural_histories: dict[str, Any] = {}
        for model_offset, model_name in enumerate(
            ("conv_bilstm", "sequence_transformer"), start=2
        ):
            _deadline(deadline, f"final seed {seed_index} {model_name}")
            model, history = fit_neural_fixed_epochs(
                model_name,
                train_transformed,
                y_view,
                epochs=selected_epochs[model_name],
                seed=seed + model_offset * 31,
            )
            view_probability = model.predict_proba(validation_transformed)[:, 1]
            probabilities[model_name] = aggregate_view_probabilities(
                view_probability,
                validation_view_subject,
                len(validation_dataset.subject_ids),
            )
            model.save(seed_dir / f"{model_name}.pt")
            neural_histories[model_name] = history
            write_json(seed_dir / f"{model_name}_history.json", history)
        ensemble = blend_probabilities(probabilities)
        per_seed_ensemble.append(ensemble)
        for name, probability in probabilities.items():
            per_model[name].append(probability)
        seed_report = {
            "seed_index": seed_index,
            "seed": seed,
            "training_subjects": len(train_dataset.subject_ids),
            "selected_epochs": selected_epochs,
            "summary_feature_count": len(selected_names),
            "probability_only_before_historical_labels": True,
        }
        seed_reports.append(seed_report)
        write_json(seed_dir / "refit_report.json", seed_report)
    averaged = {
        name: np.mean(np.stack(values), axis=0) for name, values in per_model.items()
    }
    averaged["ensemble"] = np.mean(np.stack(per_seed_ensemble), axis=0)
    report = {
        "recipe_frozen_from": "Training repeated outer OOF only",
        "training_subjects_per_refit": len(train_dataset.subject_ids),
        "refit_seeds": seed_count,
        "selected_epochs_median_across_outer_folds": selected_epochs,
        "fixed_ensemble_weights": ENSEMBLE_WEIGHTS,
        "validation_labels_loaded": False,
        "seeds": seed_reports,
    }
    write_json(final_root / "FINAL_REFIT_REPORT.json", report)
    return averaged, report


def run_experiment(config: RunConfig) -> dict[str, Any]:
    """Run EDA, repeated subject CV, then freeze/evaluate historical Validation."""

    started = time.monotonic()
    deadline = started + config.max_runtime_seconds
    output = Path(config.output_dir).expanduser().resolve()
    training_dir = output / "training"
    models_dir = training_dir / "models"
    output.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)
    write_json(training_dir / "run_config.json", asdict(config))
    write_json(training_dir / "environment.json", _runtime_info())
    write_json(training_dir / "model_origins.json", GOOGLE_MODEL_EVIDENCE)

    set_global_seed(config.seed)
    train_dataset = build_subject_dataset(
        config.training_root, require_labels=True, expected_split="training"
    )
    _assert_dataset_contract(train_dataset, (85, 56))
    # Label-free construction is deliberately completed before Validation
    # label files are opened later in this function.
    validation_dataset = build_subject_dataset(
        config.validation_root, require_labels=False, expected_split="validation"
    )
    if len(validation_dataset.subject_ids) != 33:
        raise AssertionError("Historical Validation must contain 33 subjects")
    if set(train_dataset.subject_ids) & set(validation_dataset.subject_ids):
        raise AssertionError("Training/Validation subject overlap")
    if train_dataset.feature_names != validation_dataset.feature_names:
        raise AssertionError("Training/Validation daily feature contract differs")

    run_training_eda(train_dataset, output / "eda")
    train_views, train_view_subject = make_fixed_views(
        train_dataset.sequences,
        sequence_length=config.sequence_length,
        n_views=config.views_per_subject,
    )
    validation_views, validation_view_subject = make_fixed_views(
        validation_dataset.sequences,
        sequence_length=config.sequence_length,
        n_views=config.views_per_subject,
    )
    expected_train_views = len(train_dataset.subject_ids) * config.views_per_subject
    expected_validation_views = (
        len(validation_dataset.subject_ids) * config.views_per_subject
    )
    if len(train_views) != expected_train_views or len(validation_views) != expected_validation_views:
        raise AssertionError("Fixed-view construction is not subject balanced")

    assert train_dataset.y is not None
    repeats = 1 if config.run_mode == "smoke" else config.outer_repeats
    folds = 3 if config.run_mode == "smoke" else config.outer_folds
    n_subjects = len(train_dataset.subject_ids)
    oof = {
        name: np.full((repeats, n_subjects), np.nan, dtype=np.float64)
        for name in (*MODEL_NAMES, "ensemble")
    }
    validation_fold_probabilities: dict[str, list[np.ndarray]] = {
        name: [] for name in (*MODEL_NAMES, "ensemble")
    }
    fold_reports: list[dict[str, Any]] = []
    for repeat in range(repeats):
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=config.seed + repeat * 1009,
        )
        for fold, (train_idx, outer_idx) in enumerate(
            splitter.split(np.zeros(n_subjects), train_dataset.y)
        ):
            _deadline(deadline, f"repeat {repeat} fold {fold}")
            if set(train_idx) & set(outer_idx):
                raise AssertionError("Subject index overlap inside outer CV")
            result = _fit_predict_fold(
                config=config,
                repeat=repeat,
                fold=fold,
                train_idx=train_idx,
                outer_idx=outer_idx,
                train_dataset=train_dataset,
                train_views=train_views,
                train_view_subject=train_view_subject,
                validation_dataset=validation_dataset,
                validation_views=validation_views,
                validation_view_subject=validation_view_subject,
                fold_dir=models_dir / f"repeat_{repeat:02d}_fold_{fold:02d}",
                deadline=deadline,
            )
            for name, probability in result["outer_probabilities"].items():
                oof[name][repeat, outer_idx] = probability
            for name, probability in result["validation_probabilities"].items():
                validation_fold_probabilities[name].append(probability)
            fold_reports.append(result["report"])
            write_json(
                training_dir / "progress.json",
                {
                    "status": "running",
                    "completed_folds": len(fold_reports),
                    "total_folds": repeats * folds,
                    "last_repeat": repeat,
                    "last_fold": fold,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )

    for name, values in oof.items():
        if not np.isfinite(values).all():
            raise AssertionError(f"Incomplete OOF probabilities for {name}")
    averaged_oof = {name: values.mean(axis=0) for name, values in oof.items()}
    threshold_selection = _select_threshold(train_dataset.y, averaged_oof["ensemble"])
    selected_threshold = float(threshold_selection["threshold"])
    oof_report = {
        "primary_metric_policy": "OOF @ fixed threshold 0.5",
        "primary_ensemble_at_0_5": binary_metrics(
            train_dataset.y, averaged_oof["ensemble"], 0.5
        ),
        "selected_threshold_secondary": binary_metrics(
            train_dataset.y, averaged_oof["ensemble"], selected_threshold
        ),
        "individual_models_at_0_5": {
            name: binary_metrics(train_dataset.y, averaged_oof[name], 0.5)
            for name in MODEL_NAMES
        },
        "repeat_metrics_at_0_5": [
            binary_metrics(train_dataset.y, oof["ensemble"][repeat], 0.5)
            for repeat in range(repeats)
        ],
        "threshold_selection": threshold_selection,
        "fixed_ensemble_weights": ENSEMBLE_WEIGHTS,
        "n_subjects": n_subjects,
        "folds": folds,
        "repeats": repeats,
    }
    write_json(training_dir / "oof_report.json", oof_report)
    oof_frame = pd.DataFrame(
        {
            "subject_hash": [subject_hash(item) for item in train_dataset.subject_ids],
            "target": train_dataset.y,
            **{f"p_{name}": averaged_oof[name] for name in averaged_oof},
            "pred_ensemble_at_0_5": (averaged_oof["ensemble"] >= 0.5).astype(int),
            "pred_ensemble_selected": (
                averaged_oof["ensemble"] >= selected_threshold
            ).astype(int),
        }
    )
    oof_frame.to_csv(training_dir / "oof_predictions_hashed.csv", index=False)

    cv_fold_validation_average = {
        name: np.mean(np.stack(values), axis=0)
        for name, values in validation_fold_probabilities.items()
    }
    validation_average, final_refit_report = _final_refit_validation_predictions(
        config=config,
        train_dataset=train_dataset,
        train_views=train_views,
        train_view_subject=train_view_subject,
        validation_dataset=validation_dataset,
        validation_views=validation_views,
        validation_view_subject=validation_view_subject,
        fold_reports=fold_reports,
        models_dir=models_dir,
        deadline=deadline,
    )
    cv_refit_mad = {
        name: float(
            np.mean(np.abs(cv_fold_validation_average[name] - validation_average[name]))
        )
        for name in validation_average
    }
    threshold_transfer_warning = (
        "Training OOF @0.5 is primary. The selected OOF threshold is secondary because "
        "full-training refit probabilities are not distribution-identical to cross-fitted OOF."
    )
    if cv_refit_mad["ensemble"] >= 0.03:
        threshold_transfer_warning += (
            f" Ensemble CV-fold versus final-refit MAD is {cv_refit_mad['ensemble']:.4f}; "
            "interpret the transferred threshold especially cautiously."
        )
    label_free_path = training_dir / "validation_predictions_label_free_hashed.csv"
    label_free = pd.DataFrame(
        {
            "subject_hash": [
                subject_hash(item) for item in validation_dataset.subject_ids
            ],
            **{f"p_{name}": validation_average[name] for name in validation_average},
            "threshold_from_training_oof": selected_threshold,
            "predicted_class": (
                validation_average["ensemble"] >= selected_threshold
            ).astype(int),
        }
    )
    label_free.to_csv(label_free_path, index=False)
    freeze = {
        "status": "frozen_before_loading_historical_validation_labels",
        "path": str(label_free_path),
        "sha256": file_sha256(label_free_path),
        "rows": len(label_free),
        "threshold": selected_threshold,
        "validation_role": "historical benchmark",
        "prediction_source": "full 141-subject refit seed ensemble",
        "threshold_transfer_warning": threshold_transfer_warning,
        "warning": HISTORICAL_VALIDATION_WARNING,
    }
    write_json(training_dir / "VALIDATION_PREDICTIONS_FROZEN.json", freeze)

    validation_report: dict[str, Any] | None = None
    if config.evaluate_historical_validation:
        # Evaluation consumes the frozen bytes, not the in-memory arrays that
        # produced them.  This makes the metrics cryptographically traceable to
        # the label-free artifact in the freeze manifest.
        if file_sha256(label_free_path) != freeze["sha256"]:
            raise AssertionError("Frozen Validation prediction hash changed before evaluation")
        frozen_predictions = pd.read_csv(label_free_path)
        expected_hashes = [subject_hash(item) for item in validation_dataset.subject_ids]
        if frozen_predictions["subject_hash"].astype(str).tolist() != expected_hashes:
            raise AssertionError("Frozen Validation subject order/hash differs from data order")
        frozen_thresholds = frozen_predictions[
            "threshold_from_training_oof"
        ].to_numpy(dtype=float)
        if not np.allclose(frozen_thresholds, selected_threshold, rtol=0.0, atol=1e-12):
            raise AssertionError("Frozen Validation threshold differs from Training OOF choice")
        frozen_probability = {
            name: frozen_predictions[f"p_{name}"].to_numpy(dtype=float)
            for name in (*MODEL_NAMES, "ensemble")
        }
        labels = load_binary_labels(config.validation_root)
        missing = sorted(set(map(str, validation_dataset.subject_ids)) - set(labels.index))
        if missing:
            raise AssertionError(f"Missing historical labels for {len(missing)} subjects")
        y_validation = np.asarray(
            [int(labels.loc[str(item)]) for item in validation_dataset.subject_ids],
            dtype=np.int64,
        )
        if tuple(np.bincount(y_validation, minlength=2)) != (26, 7):
            raise AssertionError("Unexpected historical Validation class counts")
        validation_report = {
            "role": "historical_reused_benchmark",
            "warning": HISTORICAL_VALIDATION_WARNING,
            "primary_threshold_policy": "0.5 is primary; Training-OOF-selected threshold is secondary",
            "threshold_transfer_warning": threshold_transfer_warning,
            "metrics_at_training_selected_threshold": binary_metrics(
                y_validation, frozen_probability["ensemble"], selected_threshold
            ),
            "metrics_at_0_5": binary_metrics(
                y_validation, frozen_probability["ensemble"], 0.5
            ),
            "individual_models_at_0_5": {
                name: binary_metrics(y_validation, frozen_probability[name], 0.5)
                for name in MODEL_NAMES
            },
            "frozen_prediction_sha256": freeze["sha256"],
            "target_accuracy": TARGET_ACCURACY,
            "target_requires_correct": "at least 30 of 33",
        }
        write_json(training_dir / "historical_validation_report.json", validation_report)
        evaluated = frozen_predictions.copy()
        evaluated.insert(1, "target", y_validation)
        evaluated.to_csv(
            training_dir / "historical_validation_predictions_evaluated_hashed.csv",
            index=False,
        )

    elapsed = time.monotonic() - started
    final_report = {
        "experiment": "Binary_Wearable_SequenceFusion_Google",
        "status": "complete",
        "task": "CN versus MCI + DEM",
        "mmse_used": False,
        "cognitive_source_opened": False,
        "target_accuracy": TARGET_ACCURACY,
        "performance_claim_policy": (
            "Target, not guarantee. A fresh untouched external test is required for a 0.90 claim."
        ),
        "previous_result_analysis": {
            "old_binary_nested_oof_accuracy": 0.532,
            "old_historical_validation_accuracy": 0.727,
            "old_all_cn_validation_baseline": 26 / 33,
            "old_preprocessing_failure": (
                "The same alphabetically early event28-median 24 features were selected "
                "in all folds; multiscale variability and trend information were discarded."
            ),
            "conv_bilstm_0_84_caution": (
                "No saved metrics artifact exists in SangHyo/previous. If it was 27/32 "
                "subject accuracy, its all-CN baseline was already 26/32."
            ),
        },
        "design_changes": {
            "subject_balancing": "exactly eight fixed 28-event views per subject",
            "sequence_preprocessing": "fold-only median, winsorization, robust scaling; no mask/count",
            "summary_bank": "level, variability, tails, trend, recent shift",
            "models": list(MODEL_NAMES),
            "fixed_ensemble_weights": ENSEMBLE_WEIGHTS,
            "selection": "neural epoch uses internal subject holdout; no Validation tuning",
        },
        "training_oof": oof_report,
        "final_full_training_refit": final_refit_report,
        "cv_fold_validation_probability_audit": {
            "purpose": "label-free sensitivity audit; not the reported historical score",
            "mean_absolute_difference_from_final_refit": cv_refit_mad,
            "threshold_transfer_warning": threshold_transfer_warning,
        },
        "historical_validation": validation_report,
        "freeze": freeze,
        "data_audit": {
            "training": train_dataset.audit,
            "validation_label_free": validation_dataset.audit,
        },
        "elapsed_seconds": elapsed,
        "runtime_budget_seconds": config.max_runtime_seconds,
    }
    write_json(training_dir / "FINAL_REPORT.json", final_report)
    write_json(
        training_dir / "TRAINING_COMPLETE.json",
        {
            "status": "complete",
            "elapsed_seconds": elapsed,
            "final_report": str(training_dir / "FINAL_REPORT.json"),
        },
    )
    return final_report
