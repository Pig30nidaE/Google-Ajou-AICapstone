"""Stage orchestration and artifact writing.

The program stage is label-free and can run before the cohort is assembled.
The train stage is cache-only by design: it never changes the Gemma program
after seeing an evaluation result.  ``all`` performs the two stages in that
order, so the newly generated program is immediately frozen in the persistent
cache before any label is opened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from . import EXPERIMENT_NAME, PACKAGE_VERSION
from .catalog import PRIMITIVE_CATALOG
from .config import PipelineConfig, config_to_dict
from .data import CohortData, hash_subject_id, load_training_cohort
from .evaluation import EvaluationResult, evaluate_nested
from .modeling import fit_final_bundle
from .program_client import ProgramClientConfig, ProgramResult, generate_global_program
from .program_schema import program_feature_names

STAGES = ("all", "inspect", "program", "train")


@dataclass
class PipelineContext:
    config: PipelineConfig
    injected: Mapping[str, Any]
    data_root: Path
    run_dir: Path
    cache_root: Path

    def log(self, message: str) -> None:
        print(message, flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_status(
    context: PipelineContext,
    status: str,
    **details: Any,
) -> None:
    payload = {
        "experiment": EXPERIMENT_NAME,
        "package_version": PACKAGE_VERSION,
        "status": status,
        "updated_utc": _utc_now(),
        "run_dir": str(context.run_dir),
        **details,
    }
    write_json(context.run_dir / "LAUNCHER_STATUS.json", payload)


def make_context(
    config: PipelineConfig,
    *,
    injected: Mapping[str, Any] | None = None,
) -> PipelineContext:
    namespace = dict(injected or {})
    context = PipelineContext(
        config=config,
        injected=namespace,
        data_root=config.resolved_data_root(namespace),
        run_dir=config.resolved_run_dir(),
        cache_root=config.resolved_cache_root(),
    )
    context.run_dir.mkdir(parents=True, exist_ok=False)
    context.cache_root.mkdir(parents=True, exist_ok=True)
    write_json(context.run_dir / "RUN_CONFIG.json", config_to_dict(config))
    context.log(f"[setup] experiment : {EXPERIMENT_NAME}")
    context.log(f"[setup] data root  : {context.data_root}")
    context.log(f"[setup] run dir    : {context.run_dir}")
    context.log(f"[setup] cache root : {context.cache_root}")
    return context


def _program_client_config(
    config: PipelineConfig,
    *,
    offline: bool,
    allow_regenerate: bool,
) -> ProgramClientConfig:
    gemma = config.gemma
    accepted = {
        "model": gemma.model,
        "api_key_env": gemma.api_key_env,
        "temperature": gemma.temperature,
        "max_output_tokens": gemma.max_output_tokens,
        "thinking_level": gemma.thinking_level,
        "thinking_budget": gemma.thinking_budget,
        "timeout_seconds": gemma.timeout_seconds,
        "max_retries": gemma.max_retries,
        "initial_backoff_seconds": gemma.initial_backoff_seconds,
        "backoff_multiplier": gemma.backoff_multiplier,
        "max_backoff_seconds": gemma.max_backoff_seconds,
        "offline": offline,
    }
    # ``regenerate`` was added after the initial client contract.  Keep the
    # orchestration compatible with either client revision without silently
    # discarding a user's request.
    fields = getattr(ProgramClientConfig, "__dataclass_fields__", {})
    regenerate = bool(config.gemma.regenerate_program and allow_regenerate)
    if "regenerate_program" in fields:
        accepted["regenerate_program"] = regenerate
    elif "regenerate" in fields:
        accepted["regenerate"] = regenerate
    elif regenerate:
        raise RuntimeError(
            "The installed program client does not support --regenerate-program"
        )
    return ProgramClientConfig(**accepted)


def _program_result(
    context: PipelineContext,
    *,
    offline: bool,
    allow_regenerate: bool = True,
) -> ProgramResult:
    return generate_global_program(
        _program_client_config(
            context.config,
            offline=offline,
            allow_regenerate=allow_regenerate,
        ),
        cache_root=context.cache_root,
        catalog=PRIMITIVE_CATALOG,
        logger=context.log,
    )


def _write_program_artifacts(context: PipelineContext, result: ProgramResult) -> None:
    write_json(context.run_dir / "FEATURE_PROGRAM.json", result.program)
    write_json(
        context.run_dir / "FEATURE_PROGRAM_MANIFEST.json",
        {
            **result.manifest,
            "from_cache": bool(result.from_cache),
            "persistent_cache_path": str(result.cache_path),
            "n_program_features": len(program_feature_names(result.program)),
            "patient_rows_sent_to_api": 0,
            "labels_sent_to_api": 0,
            "mmse_values_sent_to_api": 0,
        },
    )


def stage_inspect(context: PipelineContext) -> dict[str, Any]:
    context.log("[stage] inspect: Training-only cohort and leakage contracts")
    cohort = load_training_cohort(context.data_root)
    audit = cohort.audit()
    write_json(context.run_dir / "DATA_AUDIT.json", audit)
    return audit


def stage_program(context: PipelineContext) -> dict[str, Any]:
    if not context.config.gemma.enabled:
        raise RuntimeError("Gemma is disabled; a feature program cannot be generated")
    context.log(
        "[stage] program: one global label-free program, "
        f"model={context.config.gemma.model}"
    )
    result = _program_result(context, offline=bool(context.config.gemma.offline))
    _write_program_artifacts(context, result)
    summary = {
        "program_hash": result.manifest["program_hash"],
        "n_features": len(program_feature_names(result.program)),
        "from_cache": bool(result.from_cache),
    }
    context.log(
        f"[program] features={summary['n_features']} "
        f"from_cache={summary['from_cache']}"
    )
    return summary


def _hashed_oof(cohort: CohortData, oof: pd.DataFrame) -> pd.DataFrame:
    frame = oof.copy()
    if "subject_index" in frame.columns:
        positions = pd.to_numeric(frame["subject_index"], errors="raise").astype(int)
        if (positions < 0).any() or (positions >= cohort.n_subjects).any():
            raise AssertionError("OOF subject_index escaped the cohort bounds")
        frame.insert(
            0,
            "subject_hash",
            [
                hash_subject_id(str(cohort.subject_ids[position]))
                for position in positions.to_numpy()
            ],
        )
    elif "subject_id" in frame.columns:
        frame["subject_hash"] = frame["subject_id"].map(hash_subject_id)
    else:
        raise AssertionError("Evaluation OOF table lacks subject_index/subject_id")
    frame = frame.drop(
        columns=[
            name
            for name in ("subject_id", "subject_index")
            if name in frame.columns
        ]
    )
    return frame


def _evaluation_payload(
    context: PipelineContext,
    cohort: CohortData,
    program_result: ProgramResult,
    evaluation: EvaluationResult,
) -> dict[str, Any]:
    report = dict(evaluation.report)
    caveats = [
        "The historical 33-subject Validation set is not an independent target here and was not opened.",
        "ROC-AUC 0.92 is an aspiration, not an expected or guaranteed result.",
        "Repeated OOF on one 141-subject cohort does not replace external validation.",
        "The feature program is fixed before labels are opened; changing it after seeing these results defines a new experiment.",
    ]
    if context.config.run.profile == "smoke":
        caveats.insert(
            0,
            "This is a smoke wiring run and must not be reported as model performance.",
        )
    return {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": _utc_now(),
        "task": {
            "positive": ["MCI", "Dem"],
            "negative": ["CN"],
            "cohort": "1.Training only",
            "n_subjects": cohort.n_subjects,
            "n_positive": int(cohort.y.sum()),
            "historical_validation_used": False,
        },
        "target_roc_auc": {
            "value": 0.92,
            "status": "research aspiration; never assumed or guaranteed",
            "comparison_baseline": "same-split mmse_only arm",
        },
        "program": {
            "model": context.config.gemma.model,
            "program_hash": program_result.manifest["program_hash"],
            "n_features": len(program_feature_names(program_result.program)),
            "global_program_definition_count": 1,
            "api_attempts": int(program_result.manifest["attempt"]),
            "patient_rows_sent_to_api": 0,
        },
        "validation_design": {
            "outer_folds": context.config.cv.outer_folds,
            "inner_folds": context.config.cv.inner_folds,
            "repeats": context.config.cv.repeats_for(context.config.run.profile),
            "outer_stratification": "CN/MCI/Dem diagnosis; target remains binary",
            "selection_scope": "inner OOF only",
            "rank_mapping": "training-reference empirical CDF only",
            "primary_estimand": "mean of repeat-level outer OOF ROC-AUC",
            "secondary_estimand": "ROC-AUC of subject-wise mean repeated OOF scores",
        },
        **report,
        "caveats": caveats,
    }


def _save_deployment(
    context: PipelineContext,
    cohort: CohortData,
    program_result: ProgramResult,
    evaluation: EvaluationResult,
) -> dict[str, Any]:
    """Fit the fixed final architecture and verify an immediate CPU round-trip.

    These predictions are never used as performance estimates.  Their only
    purpose is to prove that the deployment object can reproduce its own score
    after serialization.
    """

    selected_weight = evaluation.report["selection"]["modal_full_weight"]
    bundle = fit_final_bundle(
        cohort,
        program_result.program,
        selected_weight,
        seed=context.config.run.seed,
    )
    deployment_dir = context.run_dir / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=False)
    bundle_path = deployment_dir / "model.joblib"
    probe_rows = min(5, cohort.n_subjects)
    before = bundle.predict_score(
        cohort.mmse.iloc[:probe_rows],
        cohort.wearable.iloc[:probe_rows],
    )
    joblib.dump(bundle, bundle_path)
    reloaded = joblib.load(bundle_path)
    after = reloaded.predict_score(
        cohort.mmse.iloc[:probe_rows],
        cohort.wearable.iloc[:probe_rows],
    )
    if not np.allclose(before, after, rtol=0.0, atol=1e-12):
        raise AssertionError("Deployment checkpoint changed scores after joblib round-trip")
    metadata = {
        **bundle.to_metadata(),
        "program_hash": program_result.manifest["program_hash"],
        "checkpoint": str(bundle_path),
        "round_trip_verified": True,
        "round_trip_probe_rows": int(probe_rows),
        "performance_interpretation": (
            "The final refit is for future inference only; in-sample refit scores "
            "are not saved or reported as performance."
        ),
    }
    write_json(deployment_dir / "deployment.json", metadata)
    return metadata


def stage_train(context: PipelineContext) -> dict[str, Any]:
    context.log("[stage] train: cached program + repeated nested subject OOF")
    # Train is deliberately cache-only even when the general config says live.
    program_result = _program_result(
        context,
        offline=True,
        allow_regenerate=False,
    )
    _write_program_artifacts(context, program_result)
    cohort = load_training_cohort(context.data_root)
    write_json(context.run_dir / "DATA_AUDIT.json", cohort.audit())
    repeats = context.config.cv.repeats_for(context.config.run.profile)
    evaluation = evaluate_nested(
        cohort,
        program_result.program,
        outer_folds=context.config.cv.outer_folds,
        inner_folds=context.config.cv.inner_folds,
        repeats=repeats,
        seed=context.config.run.seed,
        n_bootstrap=context.config.run.n_bootstrap,
        logger=context.log,
    )
    hashed_oof = _hashed_oof(cohort, evaluation.oof)
    hashed_oof.to_csv(
        context.run_dir / "OOF_PREDICTIONS_HASHED.csv",
        index=False,
    )
    write_json(context.run_dir / "SPLIT_REGISTRY.json", evaluation.split_registry)
    deployment = _save_deployment(
        context,
        cohort,
        program_result,
        evaluation,
    )
    final_report = _evaluation_payload(
        context,
        cohort,
        program_result,
        evaluation,
    )
    final_report["deployment"] = deployment
    final_report["artifacts"] = {
        "run_config": str(context.run_dir / "RUN_CONFIG.json"),
        "launcher_status": str(context.run_dir / "LAUNCHER_STATUS.json"),
        "data_audit": str(context.run_dir / "DATA_AUDIT.json"),
        "feature_program": str(context.run_dir / "FEATURE_PROGRAM.json"),
        "feature_program_manifest": str(
            context.run_dir / "FEATURE_PROGRAM_MANIFEST.json"
        ),
        "split_registry": str(context.run_dir / "SPLIT_REGISTRY.json"),
        "oof_predictions": str(context.run_dir / "OOF_PREDICTIONS_HASHED.csv"),
        "checkpoint": str(context.run_dir / "deployment" / "model.joblib"),
    }
    write_json(context.run_dir / "FINAL_REPORT.json", final_report)
    return final_report


def run_stage(context: PipelineContext, stage: str) -> dict[str, Any]:
    if stage == "inspect":
        return stage_inspect(context)
    if stage == "program":
        return stage_program(context)
    if stage == "train":
        return stage_train(context)
    raise ValueError(f"Unknown stage {stage!r}; expected inspect|program|train")


def run_all(context: PipelineContext) -> dict[str, Any]:
    program = stage_program(context)
    training = stage_train(context)
    return {"program": program, "train": training}


__all__ = [
    "PipelineContext",
    "STAGES",
    "make_context",
    "run_all",
    "run_stage",
    "stage_inspect",
    "stage_program",
    "stage_train",
    "write_json",
    "write_status",
]
