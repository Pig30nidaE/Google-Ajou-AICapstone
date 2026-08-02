"""EDA -> preprocessing -> split -> training, maximizing leakage-free ROC-AUC."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .eda import run_eda
from .engine import binary_metrics, bootstrap_ci, nested_cv, select_threshold, select_threshold_specificity
from .features import (
    SubjectData,
    assert_disjoint_subjects,
    hash_subject_id,
    load_split,
    load_validation_labels_checked,
)
from .learners import Learner

EXPERIMENT_NAME = "Binary_MMSE_MaxAUC"
SPECIFICITY_TARGETS = (0.90, 0.95)


class PlattCalibrator:
    def fit(self, prob, y):
        p = np.clip(np.asarray(prob, float), 1e-5, 1 - 1e-5)
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1.0).fit(logits, np.asarray(y, int))
        slope = float(lr.coef_[0, 0])
        self.model = lr if (np.isfinite(slope) and slope > 0) else None
        return self

    def transform(self, prob):
        p = np.clip(np.asarray(prob, float), 1e-5, 1 - 1e-5)
        if self.model is None:
            return p
        return np.clip(self.model.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1], 1e-7, 1 - 1e-7)


@dataclass
class RunConfig:
    training_root: str
    validation_root: str
    output_dir: str
    run_mode: str = "full"
    repeats: int = 5
    folds: int = 5
    inner_folds: int = 3
    weight_gate: float = 0.55
    include_wearable: bool = False
    evaluate_validation: bool = True
    seed: int = 20260724
    extra: dict[str, Any] = field(default_factory=dict)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _factories(data: SubjectData):
    return [("logreg", lambda: Learner(data, "logreg")), ("svm", lambda: Learner(data, "svm"))]


def _final_weights(mean_weight, balacc):
    eligible = {n: w for n, w in mean_weight.items() if w > 0}
    if not eligible:
        return {max(balacc, key=balacc.get): 1.0}
    total = sum(eligible.values())
    return {n: w / total for n, w in eligible.items()}


def run_experiment(config: RunConfig) -> dict:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    # ---- preprocessing (EDA-grounded features) ----
    train = load_split(config.training_root, require_labels=True, split="train",
                       include_wearable=config.include_wearable)
    # ---- EDA stage (saved report) ----
    with_wear = None
    if not config.include_wearable:
        with_wear = load_split(config.training_root, require_labels=True, split="train",
                               include_wearable=True)
    eda_report = run_eda(train, with_wear)
    _write_json(output / "eda" / "eda_report.json", eda_report)

    # ---- split + training: leakage-free subject-level repeated nested CV ----
    factories = _factories(train)
    cv = nested_cv(train, factories, repeats=config.repeats, outer_k=config.folds,
                   inner_k=config.inner_folds, weight_gate=config.weight_gate, seed=config.seed)
    ci = bootstrap_ci(train.y, cv["oof_prob"], cv["oof_margin"], seed=config.seed)
    oof_auc = cv["oof_metrics_threshold_0.5"]["roc_auc"]

    (output / "training").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject_hash": [hash_subject_id(s) for s in train.subject_ids],
                 "y_true": train.y, "oof_prob": cv["oof_prob"]}).to_csv(
        output / "training" / "oof_predictions_hashed.csv", index=False)

    final_weights = _final_weights(cv["model_mean_weight"], cv["model_inner_balanced_accuracy"])
    thresholds = {
        "threshold_0.5": 0.5,
        "balanced": select_threshold(train.y, cv["oof_prob"], "balanced_accuracy"),
        "accuracy": select_threshold(train.y, cv["oof_prob"], "accuracy"),
    }
    for tgt in SPECIFICITY_TARGETS:
        thresholds[f"specificity_{tgt}"] = select_threshold_specificity(train.y, cv["oof_prob"], tgt)
    recommended = "specificity_0.95"
    calibrator = PlattCalibrator().fit(cv["oof_prob"], train.y)

    all_idx = np.arange(train.n_subjects)
    fitted = {name: factory().fit(all_idx) for name, factory in factories}
    _save_deployment(output, fitted, final_weights, calibrator, thresholds, recommended, train, config)

    validation = None
    if config.evaluate_validation:
        validation = _freeze_and_evaluate(config, train, fitted, final_weights, thresholds,
                                          recommended, calibrator, output)

    report = {
        "experiment": EXPERIMENT_NAME, "run_mode": config.run_mode,
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "leakage-free subject-level ROC-AUC",
        "target": {"roc_auc": 0.90, "accuracy": 0.80},
        "include_wearable": config.include_wearable,
        "eda": eda_report,
        "leakage_free_subject_oof_roc_auc": oof_auc,
        "bootstrap_95ci_roc_auc": ci["roc_auc"],
        "nested_oof_threshold_0.5": cv["oof_metrics_threshold_0.5"],
        "nested_oof_balanced_threshold": cv["oof_metrics_selected_threshold"],
        "nested_oof_accuracy_threshold": cv["oof_metrics_accuracy_threshold"],
        "model_inner_balanced_accuracy": cv["model_inner_balanced_accuracy"],
        "final_weights": final_weights, "thresholds_from_training_oof": thresholds,
        "recommended_threshold": recommended, "validation": validation,
        "honesty_note": ("ROC-AUC is the primary metric. All evaluation is subject-level "
                         "(leakage-free). MMSE-only is the max-AUC configuration; wearables "
                         "dilute it (see eda.feature_set_cv_roc_auc)."),
    }
    _write_json(output / "training" / "FINAL_REPORT.json", report)
    _write_json(output / "training" / "TRAINING_COMPLETE.json",
                {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()})
    return report


def _save_deployment(output, fitted, final_weights, calibrator, thresholds, recommended, train, config):
    dep = Path(output) / "deployment"
    dep.mkdir(parents=True, exist_ok=True)
    for name, learner in fitted.items():
        joblib.dump({"kind": learner.kind, "median": learner.prep_.median_,
                     "mean": learner.prep_.mean_, "std": learner.prep_.std_,
                     "model": learner.model_}, dep / f"model_{name}.joblib")
    joblib.dump(calibrator, dep / "calibrator.joblib")
    _write_json(dep / "deployment.json", {
        "experiment": EXPERIMENT_NAME, "feature_names": list(train.feature_names),
        "include_wearable": config.include_wearable, "item_max": train.item_max,
        "final_weights": final_weights, "thresholds_from_training_oof": thresholds,
        "recommended_threshold": recommended, "class_mapping": {"0": "CN", "1": "MCI_DEM"},
        "seed": config.seed,
        "note": "Load with predict.py to reproduce validation predictions without retraining.",
    })
    return dep


def _freeze_and_evaluate(config, train, fitted, final_weights, thresholds, recommended, calibrator, output):
    validation = load_split(config.validation_root, require_labels=False, split="val",
                            include_wearable=config.include_wearable, item_max=train.item_max)
    assert_disjoint_subjects(train.subject_ids, validation.subject_ids)
    probs = {name: fitted[name].predict_proba_matrix(validation.X) for name in final_weights}
    combined = sum(final_weights[n] * probs[n] for n in final_weights) / sum(final_weights.values())

    frozen = pd.DataFrame({"subject_hash": [hash_subject_id(s) for s in validation.subject_ids],
                           "prob_impaired": combined})
    frozen_csv = output / "training" / "validation_predictions_label_free_hashed.csv"
    frozen.to_csv(frozen_csv, index=False)
    frozen_meta = {"n_subjects": int(validation.n_subjects),
                   "prediction_csv_sha256": hashlib.sha256(frozen_csv.read_bytes()).hexdigest(),
                   "frozen_utc": datetime.now(timezone.utc).isoformat(),
                   "note": "Predictions frozen before validation labels were opened."}
    _write_json(output / "training" / "VALIDATION_PREDICTIONS_FROZEN.json", frozen_meta)

    y_val = load_validation_labels_checked(config.validation_root, validation.subject_ids)
    metrics_by_threshold = {name: binary_metrics(y_val, (combined >= t).astype(int), combined)
                            for name, t in thresholds.items()}
    report = {
        "frozen": frozen_meta,
        "historical_benchmark_note": "33-subject reused benchmark, not a fresh test.",
        "roc_auc": metrics_by_threshold["threshold_0.5"]["roc_auc"],
        "all_cn_accuracy": float(np.mean(y_val == 0)),
        "recommended_threshold": recommended,
        "metrics_at_recommended": metrics_by_threshold[recommended],
        "metrics_by_threshold": metrics_by_threshold,
    }
    _write_json(output / "training" / "validation_report.json", report)
    return report


__all__ = ["RunConfig", "run_experiment", "EXPERIMENT_NAME"]
