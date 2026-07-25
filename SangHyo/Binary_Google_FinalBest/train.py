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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .engine import (
    binary_metrics,
    bootstrap_ci,
    nested_cv,
    select_threshold,
    select_threshold_specificity,
)

SPECIFICITY_TARGETS = (0.90, 0.95, 0.975)


class PlattCalibrator:
    """Monotonic (Platt) probability calibration fit on training OOF only.

    Class-weighted YDF probabilities are inflated (CN clusters near 0.5-0.7);
    calibration maps them back so a probability reads as an honest risk.  Being
    monotonic it never changes the ranking, so specificity-anchored thresholds
    are unaffected; it only makes a fixed 0.5 cut meaningful.  Falls back to
    identity if the fitted slope is not positive.
    """

    def fit(self, prob: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        p = np.clip(np.asarray(prob, float), 1e-5, 1 - 1e-5)
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        lr = LogisticRegression(C=1.0, solver="lbfgs").fit(logits, np.asarray(y, int))
        slope = float(lr.coef_[0, 0])
        self.model = lr if (np.isfinite(slope) and slope > 0) else None
        return self

    def transform(self, prob: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(prob, float), 1e-5, 1 - 1e-5)
        if self.model is None:
            return p
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        return np.clip(self.model.predict_proba(logits)[:, 1], 1e-7, 1 - 1e-7)
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

    # Thresholds are all chosen on training OOF only.  The specificity-anchored
    # thresholds transfer across the split much better than accuracy-optimal;
    # spec@0.95 is the recommended operating point.
    thresholds = {
        "threshold_0.5": 0.5,
        "balanced": select_threshold(train.y, cv["oof_prob"], "balanced_accuracy"),
        "accuracy": select_threshold(train.y, cv["oof_prob"], "accuracy"),
    }
    for target in SPECIFICITY_TARGETS:
        thresholds[f"specificity_{target}"] = select_threshold_specificity(
            train.y, cv["oof_prob"], target
        )
    recommended = "specificity_0.95"

    # Platt calibration fit on the honest nested OOF (for interpretable risks).
    calibrator = PlattCalibrator().fit(cv["oof_prob"], train.y)
    oof_cal = calibrator.transform(cv["oof_prob"])
    oof_calibrated_metrics = binary_metrics(train.y, (oof_cal >= 0.5).astype(int), oof_cal)

    # Refit the deployment models on all training subjects ONCE (deterministic
    # with a fixed seed), then save a self-contained bundle.  The same fitted
    # models are reused for the validation freeze, so the saved model reproduces
    # the reported validation predictions exactly.
    all_idx = np.arange(train.n_subjects)
    fitted = {name: _make_learner(name, train, config.seed, config.tabnet_epochs).fit(all_idx)
              for name in final_weights}
    # Saving the deployment bundle must never crash the whole run — if it fails
    # (e.g. a filesystem quirk on the mounted Drive), we still report metrics and
    # record why the save failed so it can be diagnosed.
    deployment_dir = None
    deployment_error = None
    try:
        deployment_dir = _save_deployment(
            output, fitted, final_weights, calibrator, thresholds, recommended,
            train.feature_names, config,
        )
    except Exception as error:  # noqa: BLE001 - report and continue
        import traceback
        deployment_error = f"{type(error).__name__}: {error}"
        print(f"[warn] deployment 저장 실패(무시하고 계속): {deployment_error}")
        traceback.print_exc()

    validation = None
    if config.evaluate_validation:
        validation = _freeze_and_evaluate(
            config, train, fitted, final_weights, thresholds, recommended, calibrator, output
        )

    final_report = {
        "experiment": EXPERIMENT_NAME, "run_mode": config.run_mode,
        "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "targets": {"accuracy": 0.90, "balanced_accuracy": 0.80},
        "ydf_engine_used": bool(YDF_AVAILABLE), "tabnet_used": tabnet_on,
        "nested_oof_threshold_0.5": cv["oof_metrics_threshold_0.5"],
        "nested_oof_balanced_threshold": cv["oof_metrics_selected_threshold"],
        "nested_oof_accuracy_threshold": cv["oof_metrics_accuracy_threshold"],
        "bootstrap_95ci": ci, "final_weights": final_weights,
        "thresholds_from_training_oof": thresholds,
        "recommended_threshold": recommended,
        "nested_oof_calibrated_threshold_0.5": oof_calibrated_metrics,
        "deployment_dir": (str(deployment_dir) if deployment_dir else None),
        "deployment_error": deployment_error,
        "validation": validation, "cognitive_test_used": True,
    }
    _write_json(output / "training" / "FINAL_REPORT.json", final_report)
    _write_json(output / "training" / "TRAINING_COMPLETE.json",
                {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat()})
    return {"eda": eda, "nested_cv": nested_report, "validation": validation}


def _save_deployment(output, fitted, final_weights, calibrator, thresholds, recommended,
                     feature_names, config):
    """Save a self-contained bundle that reproduces the model without retraining."""

    import joblib

    dep = Path(output) / "deployment"
    dep.mkdir(parents=True, exist_ok=True)
    for name, learner in fitted.items():
        learner.save(dep / f"model_{name}")
    joblib.dump(calibrator, dep / "calibrator.joblib")
    _write_json(dep / "deployment.json", {
        "experiment": EXPERIMENT_NAME,
        "feature_names": list(feature_names),
        "final_weights": final_weights,
        "thresholds_from_training_oof": thresholds,
        "recommended_threshold": recommended,
        "class_mapping": {"0": "CN", "1": "MCI_DEM"},
        "seed": config.seed,
        "note": "Load with predict.py to reproduce validation predictions without retraining.",
    })
    return dep


def _freeze_and_evaluate(config, train, fitted, final_weights, thresholds, recommended, calibrator, output):
    validation = load_split(config.validation_root, require_labels=False, split="val",
                            feature_subset=FINAL_FEATURES)
    assert_disjoint_subjects(train.subject_ids, validation.subject_ids)

    probs = {name: fitted[name].predict_proba_matrix(validation.X) for name in final_weights}
    combined = np.zeros(validation.n_subjects)
    for name, w in final_weights.items():
        combined += w * probs[name]
    combined /= sum(final_weights.values())
    combined_cal = calibrator.transform(combined)

    frozen = pd.DataFrame({
        "subject_hash": [hash_subject_id(s) for s in validation.subject_ids],
        "prob_impaired": combined,
        "prob_impaired_calibrated": combined_cal,
        **{f"pred_{name}": (combined >= t).astype(int) for name, t in thresholds.items()},
    })
    frozen_csv = output / "training" / "validation_predictions_label_free_hashed.csv"
    frozen.to_csv(frozen_csv, index=False)
    frozen_meta = {
        "n_subjects": int(validation.n_subjects),
        "thresholds_from_training_oof": thresholds,
        "recommended_threshold": recommended,
        "prediction_csv_sha256": _sha256(frozen_csv),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Predictions frozen before validation labels were opened.",
    }
    _write_json(output / "training" / "VALIDATION_PREDICTIONS_FROZEN.json", frozen_meta)

    y_val = load_validation_labels_checked(config.validation_root, validation.subject_ids)
    metrics_by_threshold = {
        name: binary_metrics(y_val, (combined >= t).astype(int), combined)
        for name, t in thresholds.items()
    }
    report = {
        "frozen": frozen_meta,
        "historical_benchmark_note": "33 subjects reused across experiments; historical benchmark, not a fresh test.",
        "all_cn_accuracy": float(np.mean(y_val == 0)),
        "recommended_threshold": recommended,
        "metrics_at_recommended": metrics_by_threshold[recommended],
        "metrics_by_threshold": metrics_by_threshold,
        "metrics_calibrated_0.5": binary_metrics(y_val, (combined_cal >= 0.5).astype(int), combined_cal),
    }
    _write_json(output / "training" / "validation_report.json", report)
    return report


__all__ = ["RunConfig", "run_experiment", "EXPERIMENT_NAME"]
