"""Final best experiment: Google YDF (+optional TabNet) on the strongest features.

Design synthesises everything learned:
* Feature set = MMSE domain scores (best OOF AUC 0.760 in the preprocessing
  study) + 3 Dementia-flagging wearable channels.  Richer sets only added noise.
* Models = Google YDF GBT + RF (always) and Google TabNet (only with a GPU),
  combined with the quality gate that already fixed Model A's weighting flaw.
* Honest evaluation: subject-level repeated nested CV, three thresholds, and a
  validation freeze — no tuning to the validation labels.
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

from .engine import binary_metrics, bootstrap_ci, nested_cv, select_threshold
from .features import (
    FINAL_FEATURES,
    SubjectData,
    assert_disjoint_subjects,
    hash_subject_id,
    load_split,
    load_validation_labels_checked,
)
from .learners import YDF_AVAILABLE, TabNetLearner, YDFLearner, tabnet_available

EXPERIMENT_NAME = "Binary_Google_FinalBest"


@dataclass
class RunConfig:
    training_root: str
    validation_root: str
    output_dir: str
    run_mode: str = "full"
    repeats: int = 5
    outer_folds: int = 5
    inner_folds: int = 3
    weight_gate: float = 0.55
    use_tabnet: bool = True
    tabnet_epochs: int = 120
    evaluate_validation: bool = True
    seed: int = 20260724
    extra: dict[str, Any] = field(default_factory=dict)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_learner(name, data, seed, tabnet_epochs):
    if name == "ydf_gbt":
        return YDFLearner(data, "gbt", seed=seed)
    if name == "ydf_rf":
        return YDFLearner(data, "rf", seed=seed)
    if name == "tabnet":
        return TabNetLearner(data, seed=seed, epochs=tabnet_epochs)
    raise ValueError(name)


def _eda(data: SubjectData) -> dict:
    y = data.y
    aucs = []
    for j, name in enumerate(data.feature_names):
        col = data.X[:, j]
        m = np.isfinite(col)
        if m.sum() < len(y) * 0.5 or len(np.unique(y[m])) < 2:
            continue
        try:
            a = roc_auc_score(y[m], col[m])
        except ValueError:
            continue
        aucs.append((name, float(max(a, 1 - a))))
    aucs.sort(key=lambda t: t[1], reverse=True)
    return {
        "n_subjects": int(data.n_subjects),
        "class_counts": {"CN": int((y == 0).sum()), "MCI_DEM": int((y == 1).sum())},
        "all_cn_accuracy": float(np.mean(y == 0)),
        "n_features": len(data.feature_names),
        "feature_names": list(data.feature_names),
        "cognitive_test_used": True,
        "top_feature_direction_free_auc": aucs[:15],
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

    train = load_split(config.training_root, require_labels=True, split="train",
                       feature_subset=FINAL_FEATURES)

    model_names = ["ydf_gbt", "ydf_rf"]
    tabnet_on = bool(config.use_tabnet and tabnet_available())
    if tabnet_on:
        model_names.append("tabnet")
    factories = [
        (nm, (lambda nm=nm: _make_learner(nm, train, config.seed, config.tabnet_epochs)))
        for nm in model_names
    ]

    eda = _eda(train)
    _write_json(output / "eda" / "eda_summary.json", eda)

    cv = nested_cv(train, factories, repeats=config.repeats, outer_k=config.outer_folds,
                   inner_k=config.inner_folds, weight_gate=config.weight_gate, seed=config.seed)
    ci = bootstrap_ci(train.y, cv["oof_prob"], cv["oof_margin"], seed=config.seed)

    (output / "training").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "subject_hash": [hash_subject_id(s) for s in train.subject_ids],
        "y_true": train.y, "oof_prob": cv["oof_prob"],
        "oof_margin": cv["oof_margin"], "oof_pred": cv["oof_pred"],
    }).to_csv(output / "training" / "oof_predictions_hashed.csv", index=False)
    pd.DataFrame(cv["fold_metrics"]).to_csv(output / "training" / "fold_metrics.csv", index=False)

    nested_report = {
        "run_mode": config.run_mode, "repeats": config.repeats,
        "outer_folds": config.outer_folds, "inner_folds": config.inner_folds,
        "weight_gate": config.weight_gate, "models": model_names,
        "google_models": "Yggdrasil Decision Forests (GBT+RF)" + (" + TabNet" if tabnet_on else ""),
        "ydf_engine_used": bool(YDF_AVAILABLE), "tabnet_used": tabnet_on,
        "feature_set": FINAL_FEATURES, "cognitive_test_used": True,
        "oof_metrics_selected_threshold": cv["oof_metrics_selected_threshold"],
        "oof_metrics_threshold_0.5": cv["oof_metrics_threshold_0.5"],
        "oof_metrics_accuracy_threshold": cv["oof_metrics_accuracy_threshold"],
        "bootstrap_95ci": ci,
        "model_inner_balanced_accuracy": cv["model_inner_balanced_accuracy"],
        "model_mean_weight": cv["model_mean_weight"],
    }
    _write_json(output / "training" / "nested_cv_report.json", nested_report)

    final_weights = _final_weights(cv["model_mean_weight"], cv["model_inner_balanced_accuracy"])
    thr_balanced = select_threshold(train.y, cv["oof_prob"], "balanced_accuracy")
    thr_accuracy = select_threshold(train.y, cv["oof_prob"], "accuracy")

    validation = None
    if config.evaluate_validation:
        validation = _freeze_and_evaluate(config, train, final_weights, thr_balanced, thr_accuracy, output)

    final_report = {
        "experiment": EXPERIMENT_NAME, "run_mode": config.run_mode,
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "targets": {"accuracy": 0.90, "balanced_accuracy": 0.80},
        "ydf_engine_used": bool(YDF_AVAILABLE), "tabnet_used": tabnet_on,
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


def _freeze_and_evaluate(config, train, final_weights, thr_balanced, thr_accuracy, output):
    validation = load_split(config.validation_root, require_labels=False, split="val",
                            feature_subset=FINAL_FEATURES)
    assert_disjoint_subjects(train.subject_ids, validation.subject_ids)

    all_idx = np.arange(train.n_subjects)
    probs = {}
    for name in final_weights:
        learner = _make_learner(name, train, config.seed, config.tabnet_epochs).fit(all_idx)
        probs[name] = learner.predict_proba_matrix(validation.X)
    combined = np.zeros(validation.n_subjects)
    for name, w in final_weights.items():
        combined += w * probs[name]
    combined /= sum(final_weights.values())

    frozen = pd.DataFrame({
        "subject_hash": [hash_subject_id(s) for s in validation.subject_ids],
        "prob_impaired": combined,
        "pred_threshold_0.5": (combined >= 0.5).astype(int),
        "pred_balanced_threshold": (combined >= thr_balanced).astype(int),
        "pred_accuracy_threshold": (combined >= thr_accuracy).astype(int),
    })
    frozen_csv = output / "training" / "validation_predictions_label_free_hashed.csv"
    frozen.to_csv(frozen_csv, index=False)
    frozen_meta = {
        "n_subjects": int(validation.n_subjects),
        "final_threshold_balanced": thr_balanced, "final_threshold_accuracy": thr_accuracy,
        "prediction_csv_sha256": _sha256(frozen_csv),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Predictions frozen before validation labels were opened.",
    }
    _write_json(output / "training" / "VALIDATION_PREDICTIONS_FROZEN.json", frozen_meta)

    y_val = load_validation_labels_checked(config.validation_root, validation.subject_ids)
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
