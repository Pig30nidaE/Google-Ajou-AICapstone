"""Factorial ablation: which of the proposed fixes actually raises ROC-AUC?

Runs 3 training strategies x 2 selection methods under one identical nested
evaluation, plus the MMSE-only baseline that scored 0.7657 in
``Binary_MMSE_MaxAUC``.  Every arm reports the same pooled subject-level OOF
ROC-AUC, so the comparison is like-for-like and the answer is measured, not
argued.

A caveat this file writes into its own report: **picking the best of N arms is
itself a selection step.**  With 6 arms whose estimates carry a standard error of
roughly 0.03-0.04 at this sample size, the winner's headline is expected to sit
above its own true value by a noticeable margin.  ``arm_selection_caveat``
records the observed spread so the best number is never read as if it had been
the only thing tried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .engine import (
    GRIDS,
    binary_metrics,
    choose_columns,
    fit_pool,
    nested_cv,
    non_nested_reference,
    safe_auc,
)
from .features import (
    assert_disjoint_subjects,
    hash_subject_id,
    load_training,
    load_validation,
    load_validation_labels_checked,
)
from .learners import ALL_KINDS, STRATEGIES, YDF_AVAILABLE, make_model
from .selection import stability_select, summarize_frequency

EXPERIMENT_NAME = "Binary_Google_OrdinalStable"
REFERENCE_SCORES = {
    "Binary_MMSE_MaxAUC (MMSE only, untuned)": 0.7657,
    "Binary_Google_MaxAUC_Tuned (151 feats, heavy search)": 0.7172,
}


def _log(message: str) -> None:
    print(message, flush=True)


@dataclass
class RunConfig:
    data_root: str
    output_dir: str
    run_mode: str = "full"
    kinds: tuple[str, ...] = ALL_KINDS
    strategies: tuple[str, ...] = STRATEGIES
    selections: tuple[str, ...] = ("fold_topk", "stability")
    repeats: int = 3
    outer_k: int = 5
    inner_k: int = 5
    top_k: int = 25
    top_m: int = 2
    auc_gate: float = 0.55
    evaluate_validation: bool = True
    seed: int = 20260728
    extra: dict[str, Any] = field(default_factory=dict)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")


def run_experiment(config: RunConfig) -> dict:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    wall = time.monotonic()

    _log(f"[{EXPERIMENT_NAME}] mode={config.run_mode}  "
         f"ydf={'yes' if YDF_AVAILABLE else 'NO (HistGBT stand-in)'}")

    data = load_training(config.data_root)
    _log(f"  training: {data.n_subjects} subjects x {data.n_features} features "
         f"(CN {int((data.severity == 0).sum())}, MCI {int((data.severity == 1).sum())}, "
         f"Dem {int((data.severity == 2).sum())})")

    # --- stability-selection diagnostic on the full training set (reporting only) ---
    _cols, frequency = stability_select(data.X, data.y, top_k=config.top_k, seed=config.seed)
    stability_report = {
        "n_features_surviving": int(len(_cols)),
        "most_stable_features": summarize_frequency(frequency, data.feature_names),
    }
    _write_json(output / "eda" / "stability_report.json", stability_report)
    _log(f"  stability selection keeps {len(_cols)} / {data.n_features} features")

    # --- baseline: reproduce the current best configuration -------------------
    _log("[1/3] baseline arm (MMSE only, binary, fold_topk)")
    mmse = data.mmse_only()
    arms: dict[str, dict] = {}
    arms["baseline_mmse_only"] = nested_cv(
        mmse, config.kinds, strategy="binary", selection="fold_topk", repeats=config.repeats,
        outer_k=config.outer_k, inner_k=config.inner_k, top_k=config.top_k, top_m=config.top_m,
        auc_gate=config.auc_gate, seed=config.seed, log=_log)
    arms["baseline_mmse_only"]["strategy"] = "binary"
    arms["baseline_mmse_only"]["selection"] = "fold_topk"
    arms["baseline_mmse_only"]["feature_set"] = "mmse_only"
    _log(f"      -> {arms['baseline_mmse_only']['pooled_oof_roc_auc']:.4f}")

    # --- factorial ablation on the full feature set ---------------------------
    _log("[2/3] ablation: strategy x selection")
    for strategy in config.strategies:
        for selection in config.selections:
            name = f"{strategy}__{selection}"
            _log(f"    {name}")
            result = nested_cv(data, config.kinds, strategy=strategy, selection=selection,
                               repeats=config.repeats, outer_k=config.outer_k,
                               inner_k=config.inner_k, top_k=config.top_k, top_m=config.top_m,
                               auc_gate=config.auc_gate, seed=config.seed, log=_log)
            result.update({"strategy": strategy, "selection": selection,
                           "feature_set": "all_151"})
            arms[name] = result
            _log(f"      -> {result['pooled_oof_roc_auc']:.4f} "
                 f"(fold sd {result['std_fold_roc_auc']:.3f}, "
                 f"inner-outer {result['mean_inner_minus_outer']:+.3f}, "
                 f"{result['mean_n_selected']:.0f} feats)")

    ranked = sorted(arms.items(), key=lambda kv: -kv[1]["pooled_oof_roc_auc"])
    best_name, best_arm = ranked[0]
    scores = [a["pooled_oof_roc_auc"] for a in arms.values()]

    # --- optimism + validation for the winning arm ----------------------------
    _log(f"[3/3] winner = {best_name} ({best_arm['pooled_oof_roc_auc']:.4f})")
    winner_data = mmse if best_arm["feature_set"] == "mmse_only" else data
    reference = non_nested_reference(winner_data, config.kinds, strategy=best_arm["strategy"],
                                     selection=best_arm["selection"], inner_k=config.inner_k,
                                     top_k=config.top_k, top_m=config.top_m,
                                     auc_gate=config.auc_gate, seed=config.seed)
    optimism = reference["roc_auc"] - best_arm["pooled_oof_roc_auc"]
    _log(f"      non-nested {reference['roc_auc']:.4f} -> optimism {optimism:+.4f}")

    (output / "training").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"subject_hash": [hash_subject_id(s) for s in data.subject_ids],
                  "y_true": data.y, "severity": data.severity,
                  "oof_score": best_arm["oof_score"]}).to_csv(
        output / "training" / "oof_predictions_hashed.csv", index=False)

    validation = None
    if config.evaluate_validation:
        validation = _freeze_and_evaluate(config, data, winner_data, best_arm, output)
        _log(f"      validation ROC-AUC {validation['roc_auc']:.4f}")

    report = {
        "experiment": EXPERIMENT_NAME,
        "run_mode": config.run_mode,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - wall,
        "ydf_available": YDF_AVAILABLE,
        "primary_metric": "leakage-free subject-level pooled OOF ROC-AUC",
        "reference_scores_from_previous_folders": REFERENCE_SCORES,
        "best_arm": best_name,
        "headline_roc_auc": best_arm["pooled_oof_roc_auc"],
        "improvement_vs_best_previous": best_arm["pooled_oof_roc_auc"] - max(REFERENCE_SCORES.values()),
        "arm_ranking": [[name, round(arm["pooled_oof_roc_auc"], 4)] for name, arm in ranked],
        "arms": arms,
        "optimism_check": {
            "arm": best_name,
            "non_nested_roc_auc": reference["roc_auc"],
            "nested_roc_auc": best_arm["pooled_oof_roc_auc"],
            "optimism": optimism,
        },
        "arm_selection_caveat": {
            "n_arms": len(arms),
            "spread_of_arm_scores": {"min": float(np.min(scores)), "max": float(np.max(scores)),
                                     "std": float(np.std(scores))},
            "note": ("The headline is the best of these arms, so it carries a "
                     "selection-over-arms bias on top of the per-arm optimism. Treat a "
                     "win smaller than the spread above as inconclusive, and confirm any "
                     "real winner by re-running that arm alone with more repeats."),
        },
        "stability_selection": stability_report,
        "validation": validation,
        "config": {
            "kinds": list(config.kinds), "strategies": list(config.strategies),
            "selections": list(config.selections), "repeats": config.repeats,
            "outer_k": config.outer_k, "inner_k": config.inner_k, "top_k": config.top_k,
            "top_m": config.top_m, "auc_gate": config.auc_gate, "seed": config.seed,
            "grids": GRIDS,
        },
        "honesty_note": (
            "Feature selection, configuration choice and blend weights are all fitted "
            "on outer-training subjects only. The 33-subject validation set is a small "
            "reused benchmark, not a fresh test set."
        ),
    }
    _write_json(output / "training" / "FINAL_REPORT.json", report)
    _write_json(output / "training" / "TRAINING_COMPLETE.json",
                {"status": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                 "headline_roc_auc": best_arm["pooled_oof_roc_auc"], "best_arm": best_name})

    _log(f"\nBest arm: {best_name}  ROC-AUC {best_arm['pooled_oof_roc_auc']:.4f}")
    for name, score in REFERENCE_SCORES.items():
        _log(f"   vs {name}: {best_arm['pooled_oof_roc_auc'] - score:+.4f}")
    return report


def _freeze_and_evaluate(config, data, winner_data, best_arm, output):
    validation = load_validation(config.data_root, data.item_max)
    assert_disjoint_subjects(data.subject_ids, validation.subject_ids)

    if best_arm["feature_set"] == "mmse_only":
        keep = [n for n in winner_data.feature_names]
        index = {n: i for i, n in enumerate(validation.feature_names)}
        X_val = validation.X[:, [index[n] for n in keep]]
    else:
        if list(validation.feature_names) != list(data.feature_names):
            raise AssertionError("Validation feature schema differs from training")
        X_val = validation.X

    cols = choose_columns(winner_data.X, winner_data.y, best_arm["selection"],
                          top_k=config.top_k, seed=config.seed)
    pool = fit_pool(winner_data.X, winner_data.y, winner_data.severity, cols, config.kinds,
                    best_arm["strategy"], folds=config.inner_k, seed=config.seed,
                    auc_gate=config.auc_gate, top_m=config.top_m)

    scores = []
    for kind in pool["eligible"]:
        per_config = []
        for params in pool["tuned"][kind]["params_list"]:
            model = make_model(kind, params, strategy=best_arm["strategy"], seed=config.seed)
            model.fit(winner_data.X[:, cols], winner_data.y, winner_data.severity)
            per_config.append(model.predict_score(X_val[:, cols]))
        scores.append(np.mean(per_config, axis=0))
    weights = np.asarray(pool["weights"], dtype=float)
    combined = np.average(np.column_stack(scores), axis=1, weights=weights)

    frozen_csv = output / "training" / "validation_predictions_label_free_hashed.csv"
    pd.DataFrame({"subject_hash": [hash_subject_id(s) for s in validation.subject_ids],
                  "score": combined}).to_csv(frozen_csv, index=False)
    frozen_meta = {"n_subjects": int(len(validation.subject_ids)),
                   "prediction_csv_sha256": hashlib.sha256(frozen_csv.read_bytes()).hexdigest(),
                   "frozen_utc": datetime.now(timezone.utc).isoformat(),
                   "note": "Predictions frozen before validation labels were opened."}
    _write_json(output / "training" / "VALIDATION_PREDICTIONS_FROZEN.json", frozen_meta)

    y_val = load_validation_labels_checked(Path(config.data_root) / "2.Validation",
                                           validation.subject_ids)
    threshold = float(np.quantile(combined, 1.0 - float(np.mean(winner_data.y))))
    metrics = binary_metrics(y_val, (combined >= threshold).astype(int), combined)
    report = {"frozen": frozen_meta,
              "historical_benchmark_note": "33-subject reused benchmark, not a fresh test set.",
              "roc_auc": safe_auc(y_val, combined),
              "all_cn_accuracy": float(np.mean(y_val == 0)),
              "metrics_at_prevalence_threshold": metrics}
    _write_json(output / "training" / "validation_report.json", report)
    return report


__all__ = ["EXPERIMENT_NAME", "REFERENCE_SCORES", "RunConfig", "run_experiment"]
