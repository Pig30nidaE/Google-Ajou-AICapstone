"""Top-level three-phase experiment orchestration.

Phase A trains/evaluates on official Training subjects only.
Phase B writes and hashes label-free historical Validation predictions for
*every* track.
Phase C opens historical labels and computes descriptive metrics.  No result
from Phase C is returned to model selection or refitting code.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .artifacts import (
    environment_manifest,
    freeze_label_free_predictions,
    sha256_file,
    write_json,
)
from .config import ExperimentConfig
from .data import (
    TrackCohort,
    load_development_cohort,
    load_historical_validation_features,
    load_historical_validation_labels_after_freeze,
)
from .engine import (
    DeploymentEnsemble,
    refit_deployment,
    run_primary_nested_oof,
)
from .features import build_feature_bundle
from .leakage import assert_disjoint_groups, hash_subject_id
from .metrics import binary_metrics, save_curves
from .models.base import available_specs, model_specs


HISTORICAL_BASELINES = {
    "source": (
        "SangHyo/Binary_Google_DemScreen/"
        "Binary_Google_DemScreen_result/20260728_051820_utc/training/"
        "FINAL_REPORT.json"
    ),
    "task": "CN+MCI versus Dem",
    "cohort": "Training+Validation pooled 174; not independent holdout",
    "wearable_full_repeat_oof_auc_mean": 0.7184027777777777,
    "wearable_full_subject_mean_oof_auc": 0.7803497942386831,
    "wearable_mmse_full_repeat_oof_auc_mean": 0.8283822016460907,
    "comparison_warning": (
        "The new primary uses Training-only 141 subjects, so no numeric "
        "improvement delta is scientifically valid against pooled-174 baselines."
    ),
}


def _historical_evaluation(
    *,
    config: ExperimentConfig,
    track: str,
    inference_cohort: TrackCohort,
    freeze_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    diagnosis, y, label_audit = load_historical_validation_labels_after_freeze(
        config.data,
        track=track,
        subject_ids=inference_cohort.subject_ids,
        freeze_manifest=freeze_manifest_path,
    )
    freeze_manifest = json.loads(
        freeze_manifest_path.read_text(encoding="utf-8")
    )
    prediction_path = Path(freeze_manifest["prediction_file"])
    if sha256_file(prediction_path) != freeze_manifest["prediction_file_sha256"]:
        raise RuntimeError("Frozen historical predictions changed before evaluation")
    frozen = pd.read_csv(prediction_path)
    expected_hashes = [
        hash_subject_id(value) for value in inference_cohort.subject_ids
    ]
    if frozen["subject_hash"].astype(str).tolist() != expected_hashes:
        raise RuntimeError("Frozen historical prediction order changed")
    score = frozen["score_dem"].to_numpy(dtype=float)
    prediction = frozen["prediction_dem"].to_numpy(dtype=int)
    metrics = binary_metrics(y, score, prediction=prediction)
    curves = save_curves(
        y,
        score,
        output_dir=output_dir / "historical_validation_curves",
        prefix=track,
    )
    labeled_predictions = pd.DataFrame(
        {
            "subject_hash": [
                hash_subject_id(value) for value in inference_cohort.subject_ids
            ],
            "diagnosis": diagnosis,
            "y_dem": y,
            "score_dem_frozen": score,
            "prediction_dem_frozen": prediction,
        }
    )
    labeled_path = output_dir / "historical_validation_labeled_predictions.csv"
    labeled_predictions.to_csv(labeled_path, index=False)
    report = {
        "track": track,
        "evaluation": (
            "historical Validation descriptive evaluation after prediction freeze"
        ),
        "is_independent_external_test": False,
        "labels_were_reused_in_prior_project_experiments": True,
        "metrics": metrics,
        "n_subjects": len(y),
        "n_positive_dem": int(y.sum()),
        "n_negative_cn_mci": int((y == 0).sum()),
        "freeze_manifest": str(freeze_manifest_path.resolve()),
        "freeze_manifest_sha256": sha256_file(freeze_manifest_path),
        "label_access_audit": label_audit,
        "curve_files": curves,
        "labeled_prediction_file": str(labeled_path.resolve()),
    }
    write_json(output_dir / "HISTORICAL_VALIDATION_REPORT.json", report)
    return report


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Execute the complete protocol. This function performs model training."""

    root = config.runtime.resolved_output()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {root}")
    # Fail dependency import health before creating a partially initialized run.
    available_specs(
        config.search.screen_model_names,
        fail_on_missing=config.runtime.fail_on_missing_optional_model,
    )
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    write_json(root / "CONFIG.json", config.to_dict())
    write_json(
        root / "ENVIRONMENT.json",
        environment_manifest([spec.engine_manifest() for spec in model_specs()]),
    )
    write_json(
        root / "RUN_STATUS.json",
        {
            "status": "phase_a_training_only_primary",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "profile": config.profile,
        },
    )

    development: dict[str, TrackCohort] = {}
    deployments: dict[str, DeploymentEnsemble] = {}
    deployment_manifests: dict[str, dict[str, Any]] = {}
    primary_reports: dict[str, dict[str, Any]] = {}

    # Phase A: Validation source and label files are not touched.
    for track in config.data.tracks:
        track_root = root / track
        cohort = load_development_cohort(config.data, track)
        development[track] = cohort
        result = run_primary_nested_oof(
            cohort,
            config=config,
            output_dir=track_root / "primary_oof",
        )
        primary_reports[track] = result.report
        if config.runtime.refit_deployment_model:
            deployment, manifest = refit_deployment(
                cohort,
                config=config,
                output_dir=track_root / "deployment",
            )
            deployments[track] = deployment
            deployment_manifests[track] = manifest
        else:
            raise RuntimeError(
                "Historical label-free prediction requires refit_deployment_model=True"
            )

    # Phase B: build historical predictors and freeze all tracks before opening
    # either Validation label copy for any track.
    write_json(
        root / "RUN_STATUS.json",
        {
            "status": "phase_b_freezing_all_label_free_predictions",
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    inference: dict[str, TrackCohort] = {}
    freeze_paths: dict[str, Path] = {}
    freeze_manifests: dict[str, dict[str, Any]] = {}
    for track in config.data.tracks:
        deployment_manifest = deployment_manifests[track]
        deployment_path = Path(deployment_manifest["model_path"])
        if sha256_file(deployment_path) != deployment_manifest["model_sha256"]:
            raise RuntimeError(
                f"{track}: deployment artifact changed before historical inference"
            )
        cohort = load_historical_validation_features(config.data, track)
        inference[track] = cohort
        assert_disjoint_groups(
            development[track].subject_ids,
            cohort.subject_ids,
            context=f"{track} official Training versus historical Validation",
        )
        bundle = build_feature_bundle(cohort)
        score = deployments[track].predict_score(bundle)
        prediction = (
            score >= float(deployments[track].threshold)
        ).astype(np.int64)
        track_root = root / track
        freeze_path = track_root / "HISTORICAL_PREDICTIONS_FROZEN.json"
        freeze_paths[track] = freeze_path
        freeze_manifests[track] = freeze_label_free_predictions(
            output_csv=track_root / "historical_validation_predictions_label_free.csv",
            manifest_path=freeze_path,
            track=track,
            subject_ids=cohort.subject_ids,
            scores=score,
            predictions=prediction,
            deployment_sha256=deployment_manifest["model_sha256"],
        )
    if set(freeze_manifests) != set(config.data.tracks) or not all(
        manifest["status"] == "frozen" for manifest in freeze_manifests.values()
    ):
        raise RuntimeError("Not every track froze successfully; labels remain closed")
    write_json(
        root / "ALL_TRACKS_FROZEN.json",
        {
            "status": "all_tracks_frozen",
            "tracks": freeze_manifests,
            "labels_opened": False,
        },
    )

    # Phase C: purely descriptive historical evaluation; no object from here is
    # passed back into training or deployment selection.
    write_json(
        root / "RUN_STATUS.json",
        {
            "status": "phase_c_historical_descriptive_evaluation",
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    historical_reports = {
        track: _historical_evaluation(
            config=config,
            track=track,
            inference_cohort=inference[track],
            freeze_manifest_path=freeze_paths[track],
            output_dir=root / track,
        )
        for track in config.data.tracks
    }
    final = {
        "status": "complete",
        "task": "CN+MCI (0) versus Dem (1)",
        "primary_metric": "subject-level ROC-AUC",
        "profile": config.profile,
        "primary_reports": primary_reports,
        "historical_validation_reports": historical_reports,
        "historical_prior_context": HISTORICAL_BASELINES,
        "primary_result_type": (
            "Training-only repeated nested OOF; not independent external test"
        ),
        "historical_result_type": (
            "prediction-frozen historical Validation; repeatedly reused, not independent"
        ),
        "numeric_improvement_over_prior_pooled_result": None,
        "numeric_improvement_note": HISTORICAL_BASELINES["comparison_warning"],
        "elapsed_seconds": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "FINAL_REPORT.json", final)
    write_json(
        root / "RUN_STATUS.json",
        {
            "status": "complete",
            "final_report_sha256": sha256_file(root / "FINAL_REPORT.json"),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return final


__all__ = ["HISTORICAL_BASELINES", "run_experiment"]
