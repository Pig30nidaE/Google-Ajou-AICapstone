"""Stage orchestration.  ``run.py`` is a thin CLI on top of this module.

Stage order (identical to the order in which the artifacts appear in the run
directory)::

    inspect  -> data audit, schema/forbidden-column checks, cohort contract
    payload  -> per-subject Gemini payloads + guard results + size report
    gemini   -> cached/dry-run/offline/live feature extraction
    train    -> design matrices, one shared split plan, all arms
    evaluate -> comparison table, paired differences, FINAL_REPORT.json

The development cohort is the official Training split (141 subjects).  The
33-subject historical Validation split is deliberately *not* scored here: it has
been observed by dozens of previous experiments, so an extra look at it would be
benchmark overfitting rather than an independent test (``SangHyo/AGENTS.md``
2-5).  Listing ``val`` in ``data.splits`` only generates and caches its Gemini
features for later use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from . import EXPERIMENT_NAME, PAYLOAD_VERSION, SCHEMA_VERSION
from .config import PipelineConfig, config_to_dict, mmse_modes
from .data import (
    DAILY_CHANNELS,
    MMSE_ALLOWED_SOURCE_COLUMNS,
    MMSE_FORBIDDEN_SOURCE_COLUMNS,
    binary_target,
    load_daily_dataset,
    load_diagnoses,
    load_mmse_scores,
)
from .evaluation import (
    ArmResult,
    evaluate_arm,
    paired_arm_comparison,
    write_json,
    write_oof_csv,
)
from .features import (
    assemble_design_matrix,
    build_base_features,
    build_gemini_features,
    build_mmse_features,
    describe_blocks,
    missing_value_report,
)
from .gemini_client import GeminiFeatureExtractor
from .guards import hash_subject_id
from .models import available_models
from .payload import build_payloads, payload_size_bytes
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, prompt_hash
from .schema import FEATURE_NAMES, feature_instructions, response_schema, schema_hash
from .splits import build_split_plan, save_split_plan

__all__ = ["PipelineContext", "STAGES", "make_context", "run_stage", "run_all", "write_status"]

DEVELOPMENT_SPLIT = "train"
STAGES = ("inspect", "payload", "gemini", "train", "evaluate", "all")


@dataclass
class PipelineContext:
    config: PipelineConfig
    run_dir: Path
    data_root: Path
    cache_root: Path
    logger: Any = print
    artifacts: dict[str, str] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)

    def record(self, name: str, path: Path) -> Path:
        self.artifacts[name] = str(path)
        return path

    def log(self, message: str) -> None:
        self.logger(message)


def write_status(context: PipelineContext, status: str, **extra: Any) -> Path:
    """``LAUNCHER_STATUS.json`` follows the repository convention in AGENTS.md 6."""

    return context.record(
        "launcher_status",
        write_json(
            context.run_dir / "LAUNCHER_STATUS.json",
            {
                "experiment": EXPERIMENT_NAME,
                "status": status,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": dict(context.artifacts),
                **extra,
            },
        ),
    )


def make_context(config: PipelineConfig, *, injected: Mapping[str, Any] | None = None, logger=print) -> PipelineContext:
    data_root = config.resolved_data_root(injected)
    run_dir = config.resolved_run_dir()
    cache_root = config.resolved_cache_root()
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    context = PipelineContext(
        config=config, run_dir=run_dir, data_root=data_root, cache_root=cache_root, logger=logger
    )
    context.log(f"[setup] experiment  : {EXPERIMENT_NAME}")
    context.log(f"[setup] data root   : {data_root}")
    context.log(f"[setup] run dir     : {run_dir}")
    context.log(f"[setup] cache root  : {cache_root}")
    context.record(
        "run_config",
        write_json(
            run_dir / "RUN_CONFIG.json",
            {
                "experiment": EXPERIMENT_NAME,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "data_root": str(data_root),
                "cache_root": str(cache_root),
                "payload_version": PAYLOAD_VERSION,
                "schema_version": SCHEMA_VERSION,
                "schema_hash": schema_hash(),
                "prompt_hash": prompt_hash(),
                "config": config_to_dict(config),
            },
        ),
    )
    return context


# --------------------------------------------------------------------------- #
# shared loaders (cached in context.state so stages can be chained cheaply)
# --------------------------------------------------------------------------- #
def _daily(context: PipelineContext, split: str):
    key = f"daily::{split}"
    if key not in context.state:
        expected = context.config.data.expected_subjects.get(split)
        context.state[key] = load_daily_dataset(
            context.data_root,
            split,
            min_days_per_subject=context.config.data.min_days_per_subject,
            expected_subjects=expected if context.config.data.strict_cohort_contract else None,
            cache_dir=context.cache_root / "daily_tables",
        )
    return context.state[key]


def _payloads(context: PipelineContext, split: str) -> dict[str, dict[str, Any]]:
    key = f"payloads::{split}"
    if key not in context.state:
        context.state[key] = build_payloads(
            _daily(context, split),
            payload_config=context.config.payload,
            salt=context.config.run.subject_hash_salt,
        )
    return context.state[key]


def _cohort(context: PipelineContext) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Subject order and binary target for the development split."""

    if "cohort" not in context.state:
        dataset = _daily(context, DEVELOPMENT_SPLIT)
        subjects = dataset.subject_ids
        diagnosis, audit = load_diagnoses(
            context.data_root,
            DEVELOPMENT_SPLIT,
            strict=context.config.data.strict_cohort_contract,
        )
        aligned = diagnosis.reindex([str(subject) for subject in subjects])
        if aligned.isna().any():
            raise ValueError("Some wearable subjects have no diagnosis record")
        y = binary_target(
            aligned,
            positive=context.config.data.positive_diagnoses,
            negative=context.config.data.negative_diagnoses,
        ).to_numpy(dtype=np.int64)
        context.state["cohort"] = (subjects, y, audit)
    return context.state["cohort"]


def _extractor(context: PipelineContext, *, force_offline: bool = False) -> GeminiFeatureExtractor:
    gemini_config = context.config.gemini
    if force_offline and not gemini_config.offline:
        from dataclasses import replace

        gemini_config = replace(gemini_config, offline=True, dry_run=False, retry_failed=False)
    return GeminiFeatureExtractor(
        gemini_config, cache_root=context.cache_root, logger=context.logger
    )


# --------------------------------------------------------------------------- #
# stage: inspect
# --------------------------------------------------------------------------- #
def stage_inspect(context: PipelineContext) -> dict[str, Any]:
    context.log("[stage] inspect: reading sources and checking the modality contract")
    report: dict[str, Any] = {
        "stage": "inspect",
        "training_fits_executed": 0,
        "gemini_calls_executed": 0,
        "splits": {},
        "gemini_contract": {
            "schema_version": SCHEMA_VERSION,
            "schema_hash": schema_hash(),
            "prompt_hash": prompt_hash(),
            "feature_names": list(FEATURE_NAMES),
        },
        "mmse_access_policy": {
            "allowed_source_columns": list(MMSE_ALLOWED_SOURCE_COLUMNS),
            "forbidden_source_columns": sorted(MMSE_FORBIDDEN_SOURCE_COLUMNS),
            "used_by": "downstream classifier only, mmse_mode=with",
        },
    }
    for split in context.config.data.splits:
        dataset = _daily(context, split)
        report["splits"][split] = {
            "n_subjects": dataset.n_subjects,
            "n_subject_days": int(len(dataset.frame)),
            "daily_channels": list(DAILY_CHANNELS),
            "audit": dict(dataset.audit),
        }
        context.log(
            f"[inspect] {split}: {dataset.n_subjects} subjects, "
            f"{len(dataset.frame)} subject-days, {len(DAILY_CHANNELS)} channels"
        )

    subjects, y, label_audit = _cohort(context)
    report["development_cohort"] = {
        "split": DEVELOPMENT_SPLIT,
        "n_subjects": int(len(subjects)),
        "positive_definition": list(context.config.data.positive_diagnoses),
        "negative_definition": list(context.config.data.negative_diagnoses),
        "class_counts": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        "diagnosis_counts": label_audit.get("diagnosis_counts"),
        "label_sources": label_audit.get("files"),
    }
    context.record("data_audit", write_json(context.run_dir / "DATA_AUDIT.json", report))
    return report


# --------------------------------------------------------------------------- #
# stage: payload
# --------------------------------------------------------------------------- #
def stage_payload(context: PipelineContext) -> dict[str, Any]:
    context.log("[stage] payload: building de-identified Gemini payloads")
    report: dict[str, Any] = {"stage": "payload", "gemini_calls_executed": 0, "splits": {}}
    payload_dir = context.run_dir / "payloads"
    payload_dir.mkdir(parents=True, exist_ok=True)
    for split in context.config.data.splits:
        payloads = _payloads(context, split)
        sizes = [payload_size_bytes(payload) for payload in payloads.values()]
        by_reference = {
            str(payload["subject_ref"]): payload for payload in payloads.values()
        }
        if len(by_reference) != len(payloads):
            raise ValueError("Subject reference hash collision")
        path = payload_dir / f"payloads_{split}.json"
        write_json(path, by_reference)
        context.record(f"payloads_{split}", path)
        report["splits"][split] = {
            "n_payloads": len(payloads),
            "payload_bytes_total": int(sum(sizes)),
            "payload_bytes_median": int(sorted(sizes)[len(sizes) // 2]) if sizes else 0,
            "payload_bytes_max": int(max(sizes)) if sizes else 0,
            "guards": "label-free and MMSE-free assertions passed for every payload",
            "file": str(path),
        }
        context.log(
            f"[payload] {split}: {len(payloads)} payloads, "
            f"median {report['splits'][split]['payload_bytes_median']} bytes"
        )
    context.record("payload_report", write_json(context.run_dir / "PAYLOAD_REPORT.json", report))
    return report


# --------------------------------------------------------------------------- #
# stage: gemini
# --------------------------------------------------------------------------- #
def stage_gemini(context: PipelineContext) -> dict[str, Any]:
    gemini_config = context.config.gemini
    mode = (
        "dry_run"
        if gemini_config.dry_run
        else "offline"
        if gemini_config.offline
        else "live"
    )
    context.log(f"[stage] gemini: feature extraction, mode={mode}, model={gemini_config.model}")
    report: dict[str, Any] = {"stage": "gemini", "mode": mode, "splits": {}}

    if not gemini_config.enabled:
        context.log("[gemini] disabled in config; only BASE feature sets will be usable")
        report["note"] = "gemini.enabled=false"
        context.record("gemini_report", write_json(context.run_dir / "GEMINI_REPORT.json", report))
        return report

    extractor = _extractor(context)
    for split in context.config.data.splits:
        payloads = _payloads(context, split)
        order = sorted(payloads)
        if gemini_config.dry_run:
            report["splits"][split] = extractor.dry_run_report(payloads, subject_order=order)
            context.log(
                f"[gemini] dry-run {split}: would send "
                f"{report['splits'][split]['requests_that_would_be_sent']} request(s), "
                f"{report['splits'][split]['already_cached']} already cached"
            )
            continue

        results, summary = extractor.extract(payloads, subject_order=order)
        usable = {
            subject: result.features
            for subject, result in results.items()
            if result.features is not None
        }
        context.state[f"gemini_features::{split}"] = usable
        report["splits"][split] = {
            **summary.to_dict(),
            "usable_subjects": len(usable),
            "cache_directory": str(extractor.cache_dir),
        }
        if usable:
            _write_gemini_features_csv(context, split, usable)
        context.log(
            f"[gemini] {split}: cached={summary.cached} fresh={summary.fresh} "
            f"failed={summary.failed} cache_miss={summary.cache_miss} "
            f"tokens_in={summary.prompt_tokens} tokens_out={summary.output_tokens}"
        )

    report["contract"] = {
        "schema_hash": schema_hash(),
        "prompt_hash": prompt_hash(),
        "model": gemini_config.model,
        "generation_config": extractor.generation_config(),
        "response_schema": response_schema(),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt_template": USER_PROMPT_TEMPLATE,
        "feature_instructions": feature_instructions(),
    }
    context.record("gemini_report", write_json(context.run_dir / "GEMINI_REPORT.json", report))
    return report


def _write_gemini_features_csv(
    context: PipelineContext, split: str, features: Mapping[str, Mapping[str, float]]
) -> Path:
    import pandas as pd

    salt = context.config.run.subject_hash_salt
    frame = pd.DataFrame.from_dict(
        {hash_subject_id(subject, salt=salt): values for subject, values in features.items()},
        orient="index",
    )
    frame.index.name = "subject_hash"
    path = context.run_dir / f"gemini_features_{split}.csv"
    frame.sort_index().to_csv(path)
    return context.record(f"gemini_features_{split}", path)


def _gemini_features_for_training(context: PipelineContext) -> dict[str, dict[str, float]] | None:
    """Use this run's answers when present, otherwise read the persistent cache."""

    key = f"gemini_features::{DEVELOPMENT_SPLIT}"
    if key in context.state and context.state[key]:
        return dict(context.state[key])
    if not context.config.gemini.enabled:
        return None
    payloads = _payloads(context, DEVELOPMENT_SPLIT)
    extractor = _extractor(context, force_offline=True)
    results, summary = extractor.extract(payloads, subject_order=sorted(payloads))
    usable = {
        subject: result.features for subject, result in results.items() if result.features is not None
    }
    context.log(
        f"[train] Gemini features from cache: {len(usable)}/{len(payloads)} subjects "
        f"(cache misses: {summary.cache_miss})"
    )
    return usable or None


# --------------------------------------------------------------------------- #
# stage: train
# --------------------------------------------------------------------------- #
def stage_train(context: PipelineContext) -> dict[str, Any]:
    config = context.config
    context.log("[stage] train: building design matrices and running the shared split plan")
    subjects, y, _ = _cohort(context)
    dataset = _daily(context, DEVELOPMENT_SPLIT)

    base = build_base_features(dataset, config.features)
    gemini_features = _gemini_features_for_training(context)
    gemini = None
    if gemini_features:
        missing = sorted(set(map(str, subjects)) - set(gemini_features))
        if missing:
            raise RuntimeError(
                f"{len(missing)} subject(s) have no validated Gemini features. "
                "Run `--stage gemini` (optionally with --retry-failed) before training, "
                "or drop 'base_gemini' from features.feature_sets."
            )
        gemini = build_gemini_features(gemini_features)
    elif any(name != "base" for name in config.features.feature_sets):
        raise RuntimeError(
            "No Gemini features are available but feature_sets requests a Gemini arm. "
            "Run `--stage gemini` first, or set features.feature_sets: [base]."
        )

    mmse = None
    if "with" in mmse_modes(config):
        table, mmse_audit = load_mmse_scores(context.data_root, DEVELOPMENT_SPLIT)
        mmse = build_mmse_features(table, item_max=config.features.mmse_item_max)
        context.log(f"[train] MMSE block: {mmse.shape[1]} columns from {mmse_audit['path']}")

    plan = build_split_plan(
        y,
        subjects,
        n_splits=config.cv.n_splits,
        n_repeats=config.cv.n_repeats,
        seed=config.run.seed,
        min_positive_per_validation_fold=config.cv.min_positive_per_validation_fold,
    )
    context.record(
        "split_registry",
        save_split_plan(
            context.run_dir / "split_registry.json",
            plan,
            subject_ids=subjects,
            y=y,
            salt=config.run.subject_hash_salt,
        ),
    )
    context.log(
        f"[train] split plan {plan.plan_hash}: {plan.n_splits} folds x {plan.n_repeats} repeats "
        f"over {plan.n_subjects} subjects"
    )

    results: list[ArmResult] = []
    matrices: dict[str, Any] = {}
    for mode in mmse_modes(config):
        for feature_set in config.features.feature_sets:
            matrix = assemble_design_matrix(
                subjects=subjects,
                base=base,
                gemini=gemini,
                mmse=mmse,
                feature_set=feature_set,
                mmse_mode=mode,
            )
            matrices[f"{mode}__{feature_set}"] = {
                **describe_blocks(matrix),
                "missing_values": missing_value_report(matrix),
            }
            for model_name in config.models.enabled:
                arm_id = f"{mode}__{feature_set}__{model_name}"
                results.append(
                    evaluate_arm(
                        matrix,
                        y,
                        plan,
                        model_name=model_name,
                        model_params=getattr(config.models, model_name),
                        seed=config.run.seed,
                        arm_id=arm_id,
                        logger=context.logger,
                    )
                )

    context.state["arm_results"] = results
    context.state["design_matrices"] = matrices
    context.state["split_plan"] = plan

    report = {
        "stage": "train",
        "development_split": DEVELOPMENT_SPLIT,
        "n_subjects": int(len(subjects)),
        "class_counts": {"negative": int((y == 0).sum()), "positive": int((y == 1).sum())},
        "split_plan": {
            "plan_hash": plan.plan_hash,
            "n_splits": plan.n_splits,
            "n_repeats": plan.n_repeats,
            "seed": plan.seed,
            "shared_by_all_arms": True,
        },
        "model_implementations": available_models(),
        "design_matrices": matrices,
        "arms": [result.to_dict() for result in results],
    }
    context.record("training_report", write_json(context.run_dir / "TRAINING_REPORT.json", report))
    context.record(
        "oof_predictions",
        write_oof_csv(
            context.run_dir / "oof_predictions_hashed.csv",
            subject_ids=subjects,
            y=y,
            results=results,
            salt=config.run.subject_hash_salt,
        ),
    )
    return report


# --------------------------------------------------------------------------- #
# stage: evaluate
# --------------------------------------------------------------------------- #
def stage_evaluate(context: PipelineContext) -> dict[str, Any]:
    context.log("[stage] evaluate: comparison table and final report")
    results: Sequence[ArmResult] = context.state.get("arm_results", ())
    if not results:
        raise RuntimeError(
            "No arm results in memory. Run `--stage train` (or `--stage all`) in the same "
            "invocation; TRAINING_REPORT.json of an earlier run is a read-only artifact."
        )
    subjects, y, _ = _cohort(context)

    table = [
        {
            "arm_id": result.arm_id,
            "mmse_mode": result.mmse_mode,
            "feature_set": result.feature_set,
            "model": result.model,
            "n_features": result.n_features,
            "roc_auc_pooled_oof": round(float(result.metrics["roc_auc"]), 6),
            "roc_auc_repeat_mean": round(float(result.metrics["roc_auc_repeat_mean"]), 6),
            "roc_auc_repeat_sd": round(float(result.metrics["roc_auc_repeat_sd"]), 6),
            "pr_auc": round(float(result.metrics["pr_auc"]), 6),
            "balanced_accuracy": round(float(result.metrics["balanced_accuracy"]), 6),
            "recall_sensitivity": round(float(result.metrics["recall_sensitivity"]), 6),
            "specificity": round(float(result.metrics["specificity"]), 6),
            "mcc": round(float(result.metrics["mcc"]), 6),
        }
        for result in results
    ]
    table.sort(key=lambda row: -row["roc_auc_pooled_oof"])

    by_id = {result.arm_id: result for result in results}
    comparisons = []
    for result in results:
        if result.feature_set != "base_gemini":
            continue
        reference_id = f"{result.mmse_mode}__base__{result.model}"
        if reference_id in by_id:
            comparisons.append(paired_arm_comparison(y, by_id[reference_id], result))

    report = {
        "stage": "evaluate",
        "experiment": EXPERIMENT_NAME,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task": {
            "positive": list(context.config.data.positive_diagnoses),
            "negative": list(context.config.data.negative_diagnoses),
            "n_subjects": int(len(subjects)),
            "n_positive": int((y == 1).sum()),
            "all_negative_baseline_accuracy": round(float((y == 0).mean()), 6),
        },
        "validation_design": (
            f"repeated subject-level StratifiedKFold, {context.config.cv.n_splits} folds x "
            f"{context.config.cv.n_repeats} repeats, one fixed configuration per model, no tuning"
        ),
        "gemini_role": "diagnosis-neutral feature extractor; it never sees a label and never predicts a class",
        "comparison_table": table,
        "paired_differences_base_vs_base_gemini": comparisons,
        "artifacts": dict(context.artifacts),
        "caveats": [
            "Non-nested CV is valid here only because no search, selection or threshold fitting occurs.",
            "The 33-subject historical Validation split was not scored in this run.",
            "Repeated-split spread (roc_auc_repeat_sd) is split noise, not an independent-cohort estimate.",
        ],
    }
    context.record("final_report", write_json(context.run_dir / "FINAL_REPORT.json", report))
    for row in table:
        context.log(
            f"[result] {row['arm_id']:<40} ROC-AUC={row['roc_auc_pooled_oof']:.4f} "
            f"(repeat sd {row['roc_auc_repeat_sd']:.4f}, features={row['n_features']})"
        )
    return report


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
_STAGE_FUNCTIONS = {
    "inspect": stage_inspect,
    "payload": stage_payload,
    "gemini": stage_gemini,
    "train": stage_train,
    "evaluate": stage_evaluate,
}


def run_stage(context: PipelineContext, stage: str) -> dict[str, Any]:
    if stage not in _STAGE_FUNCTIONS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    return _STAGE_FUNCTIONS[stage](context)


def run_all(context: PipelineContext) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for stage in ("inspect", "payload", "gemini", "train", "evaluate"):
        if stage in {"train", "evaluate"} and context.config.gemini.dry_run:
            context.log(f"[stage] {stage}: skipped because gemini.dry_run is enabled")
            continue
        reports[stage] = run_stage(context, stage)
    write_status(context, "complete", stages=list(reports))
    return reports
