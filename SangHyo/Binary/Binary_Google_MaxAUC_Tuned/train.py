"""Orchestration: EDA -> preprocessing -> leakage-free split -> tuned training.

Stage order and what each stage is allowed to see::

    EDA            training subjects only
    nested CV      tuning inside outer-train only  -> HEADLINE honest ROC-AUC
    non-nested     tunes on all 141 (deliberately optimistic)
                   -> optimism diagnostic + the fitted final model
    thresholds     chosen on nested (honest) OOF predictions
    validation     predictions frozen and hashed *before* labels are opened

The headline number is ``nested.pooled_oof_roc_auc``.  ``optimism`` is reported
next to it so a tuned result can never be quietly presented as if no tuning
happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .eda import run_eda
from .engine import (
    binary_metrics,
    blend_scores,
    bootstrap_ci,
    nested_cv,
    non_nested_reference,
    safe_auc,
    select_features,
    select_threshold,
    select_threshold_specificity,
)
from .features import (
    assert_disjoint_subjects,
    hash_subject_id,
    load_split,
    load_validation_labels_checked,
)
from .learners import YDF_AVAILABLE, make_learner

EXPERIMENT_NAME = "Binary_Google_MaxAUC_Tuned"
SPECIFICITY_TARGETS = (0.90, 0.95)
TARGET_ROC_AUC = 0.80


def _log(message: str) -> None:
    print(message, flush=True)


class PlattCalibrator:
    """Maps blended scores onto calibrated probabilities (fit on nested OOF)."""

    def fit(self, prob, y):
        p = np.clip(np.asarray(prob, float), 1e-5, 1 - 1e-5)
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        model = LogisticRegression(C=1.0).fit(logits, np.asarray(y, int))
        slope = float(model.coef_[0, 0])
        self.model = model if (np.isfinite(slope) and slope > 0) else None
        return self

    def transform(self, prob):
        p = np.clip(np.asarray(prob, float), 1e-5, 1 - 1e-5)
        if self.model is None:
            return p
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        return np.clip(self.model.predict_proba(logits)[:, 1], 1e-7, 1 - 1e-7)


@dataclass
class RunConfig:
    training_root: str
    validation_root: str
    output_dir: str
    run_mode: str = "max"
    kinds: tuple[str, ...] = ("logreg", "svm", "ydf_gbt", "ydf_rf",
                              "ydf_gbt_oblique", "ydf_rf_oblique")
    budgets: dict[str, int] = field(default_factory=dict)
    repeats: int = 3
    outer_k: int = 5
    inner_k: int = 5
    screen_repeats: int = 1
    final_repeats: int = 2
    auc_gate: float = 0.55
    drop_suspect: bool = False
    include_wearable: bool = True
    evaluate_validation: bool = True
    run_optimism_check: bool = True
    deadline_seconds: float | None = None
    seed: int = 20260727
    extra: dict[str, Any] = field(default_factory=dict)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _estimate_fits(config: RunConfig) -> int:
    per_kind = sum(
        int(config.budgets.get(k, 24)) * config.inner_k * config.screen_repeats
        + max(1, int(round(int(config.budgets.get(k, 24)) * 0.25))) * config.inner_k * config.final_repeats
        for k in config.kinds
    )
    outer = config.repeats * config.outer_k * (per_kind + len(config.kinds))
    return outer + (per_kind if config.run_optimism_check else 0)


def run_experiment(config: RunConfig) -> dict:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    wall_started = time.monotonic()

    _log(f"[{EXPERIMENT_NAME}] mode={config.run_mode}  ydf={'yes' if YDF_AVAILABLE else 'NO (sklearn fallback)'}")
    _log(f"  kinds: {', '.join(config.kinds)}")
    _log(f"  approx. model fits: {_estimate_fits(config):,}")

    # ---- preprocessing -------------------------------------------------------
    train = load_split(config.training_root, require_labels=True, split="train",
                       include_wearable=config.include_wearable,
                       drop_suspect=config.drop_suspect)
    _log(f"  training matrix: {train.n_subjects} subjects x {train.n_features} features")

    # ---- EDA -----------------------------------------------------------------
    _log("[1/5] EDA")
    eda_report = run_eda(train)
    _write_json(output / "eda" / "eda_report.json", eda_report)
    _log(f"      block AUC (fixed logreg): {eda_report['feature_block_cv_roc_auc']}")

    # ---- nested CV: the honest number ----------------------------------------
    _log("[2/5] nested CV with inner tuning (headline metric)")
    nested = nested_cv(train.X, train.y, config.kinds, repeats=config.repeats,
                       outer_k=config.outer_k, inner_k=config.inner_k, budgets=config.budgets,
                       screen_repeats=config.screen_repeats, final_repeats=config.final_repeats,
                       auc_gate=config.auc_gate, seed=config.seed,
                       deadline_seconds=config.deadline_seconds, log=_log)
    honest_auc = nested["pooled_oof_roc_auc"]
    _log(f"      pooled OOF ROC-AUC = {honest_auc:.4f}  "
         f"(per-fold {nested['mean_fold_roc_auc']:.4f} +/- {nested['std_fold_roc_auc']:.4f})")

    # ---- thresholds + calibration, all from the nested OOF -------------------
    thresholds = {
        "threshold_0.5": 0.5,
        "balanced": select_threshold(train.y, nested["oof_prob"], "balanced_accuracy"),
        "accuracy": select_threshold(train.y, nested["oof_prob"], "accuracy"),
    }
    for target in SPECIFICITY_TARGETS:
        thresholds[f"specificity_{target}"] = select_threshold_specificity(
            train.y, nested["oof_prob"], target)
    recommended = "specificity_0.95"
    calibrator = PlattCalibrator().fit(nested["oof_prob"], train.y)

    oof_pred = (nested["oof_prob"] >= thresholds[recommended]).astype(int)
    ci = bootstrap_ci(train.y, nested["oof_prob"], oof_pred, seed=config.seed)
    (output / "training").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject_hash": [hash_subject_id(s) for s in train.subject_ids],
                  "y_true": train.y, "oof_prob": nested["oof_prob"],
                  "oof_score": nested["oof_score"]}).to_csv(
        output / "training" / "oof_predictions_hashed.csv", index=False)
    _write_json(output / "training" / "nested_fold_records.json", nested["fold_records"])

    # ---- final model (tuned on all 141) + optimism diagnostic ----------------
    _log("[3/5] final tuning on all training subjects (also the optimism check)")
    reference = non_nested_reference(train.X, train.y, config.kinds, budgets=config.budgets,
                                     inner_k=config.inner_k, screen_repeats=config.screen_repeats,
                                     final_repeats=config.final_repeats, auc_gate=config.auc_gate,
                                     seed=config.seed, log=_log)
    optimism = reference["roc_auc"] - honest_auc
    _log(f"      non-nested (optimistic) AUC = {reference['roc_auc']:.4f} -> optimism = {optimism:+.4f}")

    _log("[4/5] fitting deployment models")
    fitted = {}
    for kind in reference["eligible"]:
        params = reference["best_params"][kind]
        cols = select_features(train.X, train.y, top_k=int(params.get("top_k", 0)),
                               corr_threshold=float(params.get("corr_threshold", 1.01)))
        learner = make_learner(kind, params, seed=config.seed)
        learner.fit(train.X[:, cols], train.y)
        fitted[kind] = {"learner": learner, "cols": cols, "params": params}
    deployment_error = None
    try:
        _save_deployment(output, fitted, reference, thresholds, recommended, calibrator, train, config)
    except Exception as error:  # Drive/FUSE issues must not lose the run
        deployment_error = f"{type(error).__name__}: {error}"
        _log(f"      [warn] deployment save failed: {deployment_error}")

    # ---- validation ----------------------------------------------------------
    validation = None
    if config.evaluate_validation:
        _log("[5/5] validation (predictions frozen before labels are opened)")
        validation = _freeze_and_evaluate(config, train, fitted, reference, thresholds,
                                          recommended, output)
        _log(f"      validation ROC-AUC = {validation['roc_auc']:.4f}")

    report = {
        "experiment": EXPERIMENT_NAME,
        "run_mode": config.run_mode,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - wall_started,
        "ydf_available": YDF_AVAILABLE,
        "primary_metric": "leakage-free subject-level ROC-AUC (nested CV, tuning inside outer fold)",
        "target": {"roc_auc": TARGET_ROC_AUC},
        "headline_roc_auc": honest_auc,
        "target_met": bool(honest_auc >= TARGET_ROC_AUC),
        "nested": {
            "pooled_oof_roc_auc": honest_auc,
            "mean_fold_roc_auc": nested["mean_fold_roc_auc"],
            "std_fold_roc_auc": nested["std_fold_roc_auc"],
            "repeats_completed": nested["repeats_completed"],
            "elapsed_seconds": nested["elapsed_seconds"],
            "bootstrap_95ci": ci,
            "metrics_at_recommended": binary_metrics(train.y, oof_pred, nested["oof_prob"]),
        },
        "optimism_check": {
            "non_nested_roc_auc": reference["roc_auc"],
            "nested_roc_auc": honest_auc,
            "optimism": optimism,
            "note": ("optimism = how much higher the number would look if the same data "
                     "were used to tune and to report. Only the nested value is honest."),
        },
        "selected_models": {
            "eligible": reference["eligible"],
            "weights": reference["weights"],
            "inner_auc_by_kind": reference["auc_by_kind"],
            "best_params": reference["best_params"],
        },
        "eda": eda_report,
        "thresholds_from_nested_oof": thresholds,
        "recommended_threshold": recommended,
        "validation": validation,
        "deployment_error": deployment_error,
        "config": {
            "kinds": list(config.kinds), "budgets": config.budgets, "repeats": config.repeats,
            "outer_k": config.outer_k, "inner_k": config.inner_k,
            "screen_repeats": config.screen_repeats, "final_repeats": config.final_repeats,
            "auc_gate": config.auc_gate, "drop_suspect": config.drop_suspect,
            "include_wearable": config.include_wearable, "seed": config.seed,
        },
        "honesty_note": (
            "Every hyperparameter, feature subset and ensemble weight in the headline "
            "number was chosen using outer-training subjects only. The 33-subject "
            "validation set is a small reused benchmark, not a fresh test set."
        ),
    }
    _write_json(output / "training" / "FINAL_REPORT.json", report)
    _write_json(output / "training" / "TRAINING_COMPLETE.json",
                {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                 "headline_roc_auc": honest_auc})
    _log(f"\nHeadline leakage-free ROC-AUC: {honest_auc:.4f} "
         f"(target {TARGET_ROC_AUC}: {'MET' if honest_auc >= TARGET_ROC_AUC else 'not met'})")
    return report


def _save_deployment(output, fitted, reference, thresholds, recommended, calibrator, train, config):
    dep = Path(output) / "deployment"
    dep.mkdir(parents=True, exist_ok=True)
    for kind, bundle in fitted.items():
        bundle["learner"].save(dep / f"model_{kind}")
        np.save(dep / f"cols_{kind}.npy", bundle["cols"])
    joblib.dump(calibrator, dep / "calibrator.joblib")
    _write_json(dep / "deployment.json", {
        "experiment": EXPERIMENT_NAME,
        "feature_names": list(train.feature_names),
        "item_max": train.item_max,
        "include_wearable": config.include_wearable,
        "drop_suspect": config.drop_suspect,
        "eligible": list(reference["eligible"]),
        "weights": reference["weights"],
        "best_params": reference["best_params"],
        "thresholds_from_nested_oof": thresholds,
        "recommended_threshold": recommended,
        "class_mapping": {"0": "CN", "1": "MCI_DEM"},
        "blend": "weighted mean of log-odds, then sigmoid",
        "seed": config.seed,
        "note": "Load with predict.py to reproduce predictions without retraining.",
    })
    return dep


def _blend_from_fitted(fitted, reference, X):
    order = list(reference["eligible"])
    matrix = np.column_stack([
        fitted[k]["learner"].predict_proba(X[:, fitted[k]["cols"]]) for k in order
    ])
    weights = np.array([reference["weights"][k] for k in order], dtype=float)
    score = blend_scores(matrix, weights)
    return 1.0 / (1.0 + np.exp(-score))


def _freeze_and_evaluate(config, train, fitted, reference, thresholds, recommended, output):
    validation = load_split(config.validation_root, require_labels=False, split="val",
                            item_max=train.item_max, include_wearable=config.include_wearable,
                            drop_suspect=config.drop_suspect)
    assert_disjoint_subjects(train.subject_ids, validation.subject_ids)
    if list(validation.feature_names) != list(train.feature_names):
        raise AssertionError("Validation feature schema differs from training")

    prob = _blend_from_fitted(fitted, reference, validation.X)
    frozen = pd.DataFrame({"subject_hash": [hash_subject_id(s) for s in validation.subject_ids],
                           "prob_impaired": prob})
    frozen_csv = output / "training" / "validation_predictions_label_free_hashed.csv"
    frozen.to_csv(frozen_csv, index=False)
    frozen_meta = {
        "n_subjects": int(validation.n_subjects),
        "prediction_csv_sha256": hashlib.sha256(frozen_csv.read_bytes()).hexdigest(),
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Predictions frozen before validation labels were opened.",
    }
    _write_json(output / "training" / "VALIDATION_PREDICTIONS_FROZEN.json", frozen_meta)

    y_val = load_validation_labels_checked(config.validation_root, validation.subject_ids)
    metrics_by_threshold = {name: binary_metrics(y_val, (prob >= t).astype(int), prob)
                            for name, t in thresholds.items()}
    report = {
        "frozen": frozen_meta,
        "historical_benchmark_note": "33-subject reused benchmark, not a fresh test set.",
        "roc_auc": safe_auc(y_val, prob),
        "all_cn_accuracy": float(np.mean(y_val == 0)),
        "recommended_threshold": recommended,
        "metrics_at_recommended": metrics_by_threshold[recommended],
        "metrics_by_threshold": metrics_by_threshold,
    }
    _write_json(output / "training" / "validation_report.json", report)
    return report


__all__ = ["EXPERIMENT_NAME", "PlattCalibrator", "RunConfig", "TARGET_ROC_AUC", "run_experiment"]
