"""End-to-end wearable-only experiment: EDA, nested CV, freeze, evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .datalib import (
    SubjectData,
    assert_disjoint_subjects,
    hash_subject_id,
    load_split,
    load_validation_labels_checked,
)
from .engine import binary_metrics, bootstrap_ci, nested_cv, select_threshold
from .learners import (
    TORCH_AVAILABLE,
    SequenceLearner,
    TabularLearner,
    cuda_available,
)


@dataclass
class RunConfig:
    training_root: str
    validation_root: str
    output_dir: str
    run_mode: str = "full"
    repeats: int = 2
    outer_folds: int = 5
    inner_folds: int = 3
    max_features: int = 30
    epochs: int = 60
    use_sequence_model: bool = True
    evaluate_validation: bool = True
    seed: int = 20260724
    extra: dict[str, Any] = field(default_factory=dict)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eda(data: SubjectData) -> dict:
    y = data.y
    aucs = []
    for j, name in enumerate(data.tabular_names):
        column = data.tabular[:, j]
        finite = np.isfinite(column)
        if finite.sum() < len(y) * 0.5 or len(np.unique(y[finite])) < 2:
            continue
        try:
            auc = roc_auc_score(y[finite], column[finite])
        except ValueError:
            continue
        aucs.append((name, float(max(auc, 1 - auc))))
    aucs.sort(key=lambda item: item[1], reverse=True)
    return {
        "n_subjects": int(data.n_subjects),
        "class_counts": {"CN": int(np.sum(y == 0)), "MCI_DEM": int(np.sum(y == 1))},
        "all_cn_accuracy": float(np.mean(y == 0)),
        "n_daily_channels": len(data.channel_names),
        "n_tabular_features": len(data.tabular_names),
        "observation_days": {
            "min": int(min(len(s) for s in data.sequences)),
            "median": float(np.median([len(s) for s in data.sequences])),
            "max": int(max(len(s) for s in data.sequences)),
        },
        "top_single_feature_direction_free_auc": aucs[:15],
        "cognitive_test_used": False,
        "mmse_feature_count": 0,
    }


def _build_factories(data: SubjectData, config: RunConfig):
    factories = [
        ("gbt", lambda: TabularLearner(data, "gbt", config.max_features)),
        ("logreg", lambda: TabularLearner(data, "logreg", config.max_features)),
        ("rf", lambda: TabularLearner(data, "rf", config.max_features)),
    ]
    # Run the neural model only with a real GPU (or in smoke mode for wiring).
    # CPU training of 40+ nets would blow the runtime budget, so we fall back to
    # the tabular ensemble instead, matching the launcher's warning.
    sequence_active = (
        config.use_sequence_model
        and TORCH_AVAILABLE
        and (cuda_available() or config.run_mode == "smoke")
    )
    if sequence_active:
        factories.append(
            ("conv_bilstm", lambda: SequenceLearner(data, epochs=config.epochs, seed=config.seed))
        )
    return factories, sequence_active


def _final_weights(mean_weights: dict[str, float]) -> dict[str, float]:
    total = sum(mean_weights.values())
    if total <= 0:
        return {name: 1.0 / len(mean_weights) for name in mean_weights}
    return {name: value / total for name, value in mean_weights.items()}


def run_experiment(config: RunConfig) -> dict:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    train = load_split(config.training_root, require_labels=True, expected_split="train")
    factories, sequence_active = _build_factories(train, config)

    eda = _eda(train)
    _write_json(output / "eda" / "eda_summary.json", eda)

    cv = nested_cv(
        train,
        factories,
        repeats=config.repeats,
        outer_k=config.outer_folds,
        inner_k=config.inner_folds,
        seed=config.seed,
    )
    ci = bootstrap_ci(train.y, cv["oof_prob"], cv["oof_margin"], seed=config.seed)

    # Persist honest nested-OOF predictions (hashed ids) and per-fold metrics.
    oof_frame = pd.DataFrame(
        {
            "subject_hash": [hash_subject_id(s) for s in train.subject_ids],
            "y_true": train.y,
            "oof_prob": cv["oof_prob"],
            "oof_margin": cv["oof_margin"],
            "oof_pred": cv["oof_pred"],
        }
    )
    (output / "training").mkdir(parents=True, exist_ok=True)
    oof_frame.to_csv(output / "training" / "oof_predictions_hashed.csv", index=False)
    pd.DataFrame(cv["fold_metrics"]).to_csv(
        output / "training" / "fold_metrics.csv", index=False
    )
    nested_report = {
        "run_mode": config.run_mode,
        "repeats": config.repeats,
        "outer_folds": config.outer_folds,
        "inner_folds": config.inner_folds,
        "sequence_model_active": sequence_active,
        "torch_available": TORCH_AVAILABLE,
        "models": [name for name, _ in factories],
        "oof_metrics_selected_threshold": cv["oof_metrics_selected_threshold"],
        "oof_metrics_threshold_0.5": cv["oof_metrics_threshold_0.5"],
        "bootstrap_95ci": ci,
        "model_inner_balanced_accuracy": cv["model_inner_balanced_accuracy"],
        "model_mean_weight": cv["model_mean_weight"],
    }
    _write_json(output / "training" / "nested_cv_report.json", nested_report)

    final_weights = _final_weights(cv["model_mean_weight"])
    final_threshold = select_threshold(train.y, cv["oof_prob"])

    result: dict[str, Any] = {
        "eda": eda,
        "nested_cv": nested_report,
        "final_weights": final_weights,
        "final_threshold": final_threshold,
    }

    if config.evaluate_validation:
        result["validation"] = _freeze_and_evaluate(
            config, train, factories, final_weights, final_threshold, output
        )

    final_report = {
        "experiment": "Binary_Wearable_ConvBiLSTM_NoMMSE",
        "run_mode": config.run_mode,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "subject-level nested-OOF balanced accuracy",
        "targets": {"accuracy": 0.90, "balanced_accuracy": 0.80},
        "nested_oof_selected_threshold": cv["oof_metrics_selected_threshold"],
        "nested_oof_threshold_0.5": cv["oof_metrics_threshold_0.5"],
        "bootstrap_95ci": ci,
        "final_weights": final_weights,
        "final_threshold": final_threshold,
        "validation": result.get("validation"),
        "cognitive_test_used": False,
    }
    _write_json(output / "training" / "FINAL_REPORT.json", final_report)
    _write_json(
        output / "training" / "TRAINING_COMPLETE.json",
        {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()},
    )
    return result


def _freeze_and_evaluate(
    config: RunConfig,
    train: SubjectData,
    factories,
    final_weights: dict[str, float],
    final_threshold: float,
    output: Path,
) -> dict:
    validation = load_split(
        config.validation_root, require_labels=False, expected_split="val"
    )
    assert_disjoint_subjects(train.subject_ids, validation.subject_ids)

    all_idx = np.arange(train.n_subjects)
    val_probs: dict[str, np.ndarray] = {}
    for name, factory in factories:
        learner = factory()
        learner.fit(all_idx)
        if isinstance(learner, SequenceLearner):
            val_probs[name] = learner.predict_proba_sequences(validation.sequences)
        else:
            val_probs[name] = learner.predict_proba_matrix(validation.tabular)

    combined = np.zeros(validation.n_subjects)
    for name, prob in val_probs.items():
        combined += final_weights[name] * prob
    combined /= sum(final_weights.values())

    # Freeze label-free predictions with a hash BEFORE any label is opened.
    frozen_frame = pd.DataFrame(
        {
            "subject_hash": [hash_subject_id(s) for s in validation.subject_ids],
            "prob_impaired": combined,
            "pred_selected_threshold": (combined >= final_threshold).astype(int),
            "pred_threshold_0.5": (combined >= 0.5).astype(int),
        }
    )
    frozen_csv = output / "training" / "validation_predictions_label_free_hashed.csv"
    frozen_frame.to_csv(frozen_csv, index=False)
    frozen_meta = {
        "n_subjects": int(validation.n_subjects),
        "final_threshold": final_threshold,
        "prediction_csv_sha256": _sha256(frozen_csv),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Predictions frozen before validation labels were opened.",
    }
    _write_json(output / "training" / "VALIDATION_PREDICTIONS_FROZEN.json", frozen_meta)

    # Only now open the Gait/Sleep label copies for a single evaluation.
    y_val = load_validation_labels_checked(config.validation_root, validation.subject_ids)
    report = {
        "frozen": frozen_meta,
        "historical_benchmark_note": (
            "These 33 subjects were reused across prior experiments; treat as a "
            "historical benchmark, not a fresh independent test."
        ),
        "all_cn_accuracy": float(np.mean(y_val == 0)),
        "metrics_selected_threshold": binary_metrics(
            y_val, (combined >= final_threshold).astype(int), combined
        ),
        "metrics_threshold_0.5": binary_metrics(
            y_val, (combined >= 0.5).astype(int), combined
        ),
    }
    _write_json(output / "training" / "validation_report.json", report)
    return report


__all__ = ["RunConfig", "run_experiment"]
