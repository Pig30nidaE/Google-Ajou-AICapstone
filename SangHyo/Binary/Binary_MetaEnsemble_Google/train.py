"""Meta-ensemble that stacks the three models' base learners.

It reuses the exact base learners already validated in the three experiments:

* M1 (Binary_MMSE_DomainFusion): gbt, logreg, rf over the full ~30 features.
* M2 (Binary_Google_YDF_Ensemble): Google YDF gbt, rf over the full features.
* M3 (Binary_EDA_Selective): logreg, gbt_shallow over the minimal 14 features.

All seven are trained/evaluated on the **same subject-level folds** and combined
with the same quality gate: a learner contributes only if its inner-OOF balanced
accuracy clears the gate (so M2's recall and M3's stability naturally dominate,
while weak learners get zero weight).  Weights and thresholds are chosen on
training out-of-fold predictions only; validation labels are opened once, after
predictions are frozen.  Nothing is tuned to the validation set.
"""

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

from SangHyo.Binary.Binary_EDA_Selective.learners import TabularLearner as EDALearner
from SangHyo.Binary.Binary_Google_YDF_Ensemble.learners import YDF_AVAILABLE, YDFLearner
from SangHyo.Binary.Binary_MMSE_DomainFusion.learners import TabularLearner as DomainLearner

from .engine import binary_metrics, bootstrap_ci, nested_cv, select_threshold
from .features import (
    MINIMAL_FEATURES,
    SubjectData,
    assert_disjoint_subjects,
    hash_subject_id,
    load_split,
    load_validation_labels_checked,
)

EXPERIMENT_NAME = "Binary_MetaEnsemble_Google"

# name -> (family, kind); "eda_*" models use the minimal feature view.
BASE_MODELS = {
    "dom_gbt": ("dom", "gbt"),
    "dom_logreg": ("dom", "logreg"),
    "dom_rf": ("dom", "rf"),
    "ydf_gbt": ("ydf", "gbt"),
    "ydf_rf": ("ydf", "rf"),
    "eda_logreg": ("eda", "logreg"),
    "eda_gbt": ("eda", "gbt_shallow"),
}


@dataclass
class RunConfig:
    training_root: str
    validation_root: str
    output_dir: str
    run_mode: str = "full"
    repeats: int = 5
    outer_folds: int = 5
    inner_folds: int = 3
    max_features: int = 20
    weight_gate: float = 0.55
    evaluate_validation: bool = True
    seed: int = 20260724
    extra: dict[str, Any] = field(default_factory=dict)


def _make_learner(name, data_full, data_min, seed, max_features):
    family, kind = BASE_MODELS[name]
    if family == "dom":
        return DomainLearner(data_full, kind, max_features)
    if family == "ydf":
        return YDFLearner(data_full, kind, seed=seed)
    if family == "eda":
        return EDALearner(data_min, kind)
    raise ValueError(name)


def _uses_minimal(name: str) -> bool:
    return BASE_MODELS[name][0] == "eda"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _eda(data: SubjectData) -> dict:
    y = data.y
    return {
        "n_subjects": int(data.n_subjects),
        "class_counts": {"CN": int((y == 0).sum()), "MCI_DEM": int((y == 1).sum())},
        "all_cn_accuracy": float(np.mean(y == 0)),
        "n_features_full": len(data.feature_names),
        "n_features_minimal": len(MINIMAL_FEATURES),
        "base_models": list(BASE_MODELS),
        "cognitive_test_used": True,
    }


def _final_weights(model_mean_weight, model_balacc) -> dict[str, float]:
    eligible = {n: w for n, w in model_mean_weight.items() if w > 0}
    if not eligible:
        best = max(model_balacc, key=model_balacc.get)
        return {best: 1.0}
    total = sum(eligible.values())
    return {n: w / total for n, w in eligible.items()}


def run_experiment(config: RunConfig) -> dict:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    train_full = load_split(config.training_root, require_labels=True, split="train")
    train_min = load_split(config.training_root, require_labels=True, split="train",
                           feature_subset=MINIMAL_FEATURES)
    if not np.array_equal(train_full.subject_ids, train_min.subject_ids):
        raise AssertionError("Full/minimal views must share subject order")

    factories = [
        (name, (lambda name=name: _make_learner(name, train_full, train_min, config.seed, config.max_features)))
        for name in BASE_MODELS
    ]

    eda = _eda(train_full)
    _write_json(output / "eda" / "eda_summary.json", eda)

    cv = nested_cv(
        train_full, factories,
        repeats=config.repeats, outer_k=config.outer_folds, inner_k=config.inner_folds,
        weight_gate=config.weight_gate, seed=config.seed,
    )
    ci = bootstrap_ci(train_full.y, cv["oof_prob"], cv["oof_margin"], seed=config.seed)

    (output / "training").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "subject_hash": [hash_subject_id(s) for s in train_full.subject_ids],
        "y_true": train_full.y, "oof_prob": cv["oof_prob"],
        "oof_margin": cv["oof_margin"], "oof_pred": cv["oof_pred"],
    }).to_csv(output / "training" / "oof_predictions_hashed.csv", index=False)
    pd.DataFrame(cv["fold_metrics"]).to_csv(output / "training" / "fold_metrics.csv", index=False)

    nested_report = {
        "run_mode": config.run_mode, "repeats": config.repeats,
        "outer_folds": config.outer_folds, "inner_folds": config.inner_folds,
        "weight_gate": config.weight_gate, "base_models": list(BASE_MODELS),
        "ydf_engine_used": bool(YDF_AVAILABLE),
        "cognitive_test_used": True,
        "oof_metrics_selected_threshold": cv["oof_metrics_selected_threshold"],
        "oof_metrics_threshold_0.5": cv["oof_metrics_threshold_0.5"],
        "oof_metrics_accuracy_threshold": cv["oof_metrics_accuracy_threshold"],
        "bootstrap_95ci": ci,
        "model_inner_balanced_accuracy": cv["model_inner_balanced_accuracy"],
        "model_mean_weight": cv["model_mean_weight"],
    }
    _write_json(output / "training" / "nested_cv_report.json", nested_report)

    final_weights = _final_weights(cv["model_mean_weight"], cv["model_inner_balanced_accuracy"])
    thr_balanced = select_threshold(train_full.y, cv["oof_prob"], "balanced_accuracy")
    thr_accuracy = select_threshold(train_full.y, cv["oof_prob"], "accuracy")

    validation = None
    if config.evaluate_validation:
        validation = _freeze_and_evaluate(config, train_full, train_min, final_weights,
                                          thr_balanced, thr_accuracy, output)

    final_report = {
        "experiment": EXPERIMENT_NAME, "run_mode": config.run_mode,
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "targets": {"accuracy": 0.90, "balanced_accuracy": 0.80},
        "ydf_engine_used": bool(YDF_AVAILABLE),
        "nested_oof_threshold_0.5": cv["oof_metrics_threshold_0.5"],
        "nested_oof_balanced_threshold": cv["oof_metrics_selected_threshold"],
        "nested_oof_accuracy_threshold": cv["oof_metrics_accuracy_threshold"],
        "bootstrap_95ci": ci, "final_weights": final_weights,
        "final_threshold_balanced": thr_balanced, "final_threshold_accuracy": thr_accuracy,
        "validation": validation, "cognitive_test_used": True,
    }
    _write_json(output / "training" / "FINAL_REPORT.json", final_report)
    _write_json(output / "training" / "TRAINING_COMPLETE.json",
                {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()})
    return {"eda": eda, "nested_cv": nested_report, "validation": validation}


def _freeze_and_evaluate(config, train_full, train_min, final_weights, thr_balanced, thr_accuracy, output):
    val_full = load_split(config.validation_root, require_labels=False, split="val")
    val_min = load_split(config.validation_root, require_labels=False, split="val",
                         feature_subset=MINIMAL_FEATURES)
    assert_disjoint_subjects(train_full.subject_ids, val_full.subject_ids)
    if not np.array_equal(val_full.subject_ids, val_min.subject_ids):
        raise AssertionError("Validation full/minimal views must share subject order")

    all_idx = np.arange(train_full.n_subjects)
    probs = {}
    for name in final_weights:
        learner = _make_learner(name, train_full, train_min, config.seed, config.max_features).fit(all_idx)
        X = val_min.X if _uses_minimal(name) else val_full.X
        probs[name] = learner.predict_proba_matrix(X)
    combined = np.zeros(val_full.n_subjects)
    for name, w in final_weights.items():
        combined += w * probs[name]
    combined /= sum(final_weights.values())

    frozen = pd.DataFrame({
        "subject_hash": [hash_subject_id(s) for s in val_full.subject_ids],
        "prob_impaired": combined,
        "pred_threshold_0.5": (combined >= 0.5).astype(int),
        "pred_balanced_threshold": (combined >= thr_balanced).astype(int),
        "pred_accuracy_threshold": (combined >= thr_accuracy).astype(int),
    })
    frozen_csv = output / "training" / "validation_predictions_label_free_hashed.csv"
    frozen.to_csv(frozen_csv, index=False)
    frozen_meta = {
        "n_subjects": int(val_full.n_subjects),
        "final_threshold_balanced": thr_balanced, "final_threshold_accuracy": thr_accuracy,
        "prediction_csv_sha256": _sha256(frozen_csv),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Predictions frozen before validation labels were opened.",
    }
    _write_json(output / "training" / "VALIDATION_PREDICTIONS_FROZEN.json", frozen_meta)

    y_val = load_validation_labels_checked(config.validation_root, val_full.subject_ids)
    report = {
        "frozen": frozen_meta,
        "historical_benchmark_note": "33 subjects reused across experiments; historical benchmark, not a fresh test.",
        "all_cn_accuracy": float(np.mean(y_val == 0)),
        "metrics_threshold_0.5": binary_metrics(y_val, (combined >= 0.5).astype(int), combined),
        "metrics_balanced_threshold": binary_metrics(y_val, (combined >= thr_balanced).astype(int), combined),
        "metrics_accuracy_threshold": binary_metrics(y_val, (combined >= thr_accuracy).astype(int), combined),
    }
    _write_json(output / "training" / "validation_report.json", report)
    return report


__all__ = ["RunConfig", "run_experiment", "EXPERIMENT_NAME"]
