"""Stage orchestration: audit -> features -> search -> report."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from . import EXPERIMENT_NAME, FEATURE_CONTRACT_VERSION
from .config import PipelineConfig, config_to_dict
from .data import load_pooled_cohort
from .engine import (
    CandidateResult,
    binary_metrics,
    build_split_plan,
    evaluate_candidate,
    roc_auc,
    search_ensemble,
)
from .features import build_feature_bank
from .leakage import LeakageAudit, hash_subject_id
from .models import available_families, build_candidates

__all__ = ["STAGES", "PipelineContext", "make_context", "run_stage", "run_all", "write_status"]

STAGES = ("audit", "features", "search", "report", "all")


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return destination


@dataclass
class PipelineContext:
    config: PipelineConfig
    run_dir: Path
    data_root: Path
    cache_root: Path
    logger: Any = print
    artifacts: dict[str, str] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    audit: LeakageAudit = field(default_factory=LeakageAudit)

    def record(self, name: str, path: Path) -> Path:
        self.artifacts[name] = str(path)
        return path

    def log(self, message: str) -> None:
        self.logger(message)


def write_status(context: PipelineContext, status: str, **extra: Any) -> Path:
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


def make_context(
    config: PipelineConfig, *, injected: Mapping[str, Any] | None = None, logger=print
) -> PipelineContext:
    data_root = config.resolved_data_root(injected)
    run_dir = config.resolved_run_dir()
    cache_root = config.resolved_cache_root()
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    context = PipelineContext(
        config=config, run_dir=run_dir, data_root=data_root, cache_root=cache_root, logger=logger
    )
    context.log(f"[setup] experiment : {EXPERIMENT_NAME}")
    context.log(f"[setup] data root  : {data_root}")
    context.log(f"[setup] run dir    : {run_dir}")
    context.log(f"[setup] profile    : {config.run.profile}")

    # Disclose every permitted optimism source up front, so it is impossible to
    # read the final number without also reading these.
    context.audit.disclose(
        "Cohort pools Training(141) + Validation(33) = 174 subjects; no untouched "
        "hold-out remains, and scores are NOT comparable to the 141-subject "
        "numbers used by other SangHyo experiments."
    )
    context.audit.disclose(
        "Candidates, screening size (top_k) and ensemble weights are selected on "
        "the same repeated OOF that is reported (non-nested selection optimism)."
    )
    context.audit.disclose(
        "MMSE is a clinical cognitive test used in the diagnostic process; "
        "MMSE-inclusive performance is partly circular and must never be "
        "described as wearable-only screening performance."
    )

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
                "feature_contract": FEATURE_CONTRACT_VERSION,
                "config": config_to_dict(config),
            },
        ),
    )
    return context


# --------------------------------------------------------------------------- #
def _cohort(context: PipelineContext):
    if "cohort" not in context.state:
        config = context.config
        context.state["cohort"] = load_pooled_cohort(
            context.data_root,
            splits=config.data.splits,
            positive=config.data.positive_diagnoses,
            negative=config.data.negative_diagnoses,
            min_days_per_subject=config.data.min_days_per_subject,
            include_intraday=config.features.include_intraday,
            expected_subjects=config.data.expected_subjects,
            expected_pooled_diagnoses=(
                config.data.expected_pooled_diagnoses
                if set(config.data.splits) == {"train", "val"}
                else None
            ),
            strict=config.data.strict_cohort_contract,
            cache_dir=context.cache_root / "daily_tables",
        )
    return context.state["cohort"]


def _bank(context: PipelineContext):
    if "bank" not in context.state:
        context.state["bank"] = build_feature_bank(_cohort(context), context.config.features)
    return context.state["bank"]


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_audit(context: PipelineContext) -> dict[str, Any]:
    context.log("[stage] audit: loading pooled cohort and checking contracts")
    cohort = _cohort(context)
    context.audit.record(
        "subject_ids_unique", len(set(cohort.subject_ids.tolist())) == len(cohort.subject_ids)
    )
    context.audit.record("both_classes_present", len(set(cohort.y.tolist())) == 2)
    report = {
        "stage": "audit",
        "model_fits_executed": 0,
        "cohort": dict(cohort.audit),
        "n_subjects": cohort.n_subjects,
        "n_positive": cohort.n_positive,
        "split_origin_counts": {
            name: int((cohort.split_origin == name).sum())
            for name in sorted(set(cohort.split_origin.tolist()))
        },
    }
    context.log(
        f"[audit] pooled {cohort.n_subjects} subjects "
        f"({cohort.n_positive} positive / {cohort.n_subjects - cohort.n_positive} negative)"
    )
    context.record("data_audit", write_json(context.run_dir / "DATA_AUDIT.json", report))
    return report


def stage_features(context: PipelineContext) -> dict[str, Any]:
    context.log("[stage] features: building the subject-level feature bank")
    bank = _bank(context)
    report = {"stage": "features", "model_fits_executed": 0, **dict(bank.audit)}
    for name, columns in bank.views.items():
        context.log(f"[features] view {name:<16} -> {len(columns)} columns")
    context.record("feature_manifest", write_json(context.run_dir / "FEATURE_MANIFEST.json", {
        **report,
        "views": {name: list(cols) for name, cols in bank.views.items()},
    }))
    return report


def stage_search(context: PipelineContext) -> dict[str, Any]:
    config = context.config
    cohort = _cohort(context)
    bank = _bank(context)
    cv = config.resolved_cv()

    families, skipped = available_families(config.candidates.families)
    if skipped:
        context.log(f"[search] skipped families (library missing): {skipped}")
    if not families:
        raise RuntimeError("No learner family is available; install requirements_colab.txt")

    views = tuple(bank.views)
    candidates = build_candidates(
        tuple(families),
        views,
        logreg_c_grid=config.candidates.logreg_c_grid,
        svm_c_grid=config.candidates.svm_c_grid,
        svm_gamma_grid=config.candidates.svm_gamma_grid,
        top_k_grid=config.screening.top_k_grid if config.screening.enabled else (),
    )
    context.log(
        f"[search] {len(candidates)} candidates x {cv.n_splits} folds x {cv.n_repeats} repeats"
    )

    plan = build_split_plan(
        cohort.y,
        cohort.subject_ids,
        n_splits=cv.n_splits,
        n_repeats=cv.n_repeats,
        seed=config.run.seed,
        min_positive_per_validation_fold=cv.min_positive_per_validation_fold,
    )
    context.audit.record("all_folds_subject_disjoint", True, f"plan {plan.plan_hash}")
    context.log(f"[search] split plan {plan.plan_hash} (shared by every candidate)")

    matrices = {name: bank.matrix(name) for name in views}
    results: list[CandidateResult] = []
    started = time.monotonic()
    for index, candidate in enumerate(candidates, start=1):
        X, names = matrices[candidate.view]
        result = evaluate_candidate(
            candidate,
            X,
            cohort.y,
            names,
            plan,
            seed=config.run.seed,
            winsorize_quantile=config.screening.winsorize_quantile,
            correlation_threshold=config.screening.correlation_threshold,
            balanced=config.candidates.class_weight_balanced,
            audit=context.audit,
        )
        results.append(result)
        if result.error:
            context.log(f"[search] {index:>3}/{len(candidates)} {candidate.name}: FAILED {result.error}")
        elif index % 5 == 0 or index == len(candidates):
            elapsed = time.monotonic() - started
            context.log(
                f"[search] {index:>3}/{len(candidates)} {candidate.name}: "
                f"subject-mean AUC={result.subject_mean_auc:.4f} ({elapsed:.0f}s elapsed)"
            )

    usable = [r for r in results if r.error is None]
    if not usable:
        raise RuntimeError("Every candidate failed; see CANDIDATE_RESULTS.json")
    usable.sort(key=lambda r: -r.subject_mean_auc)

    ensemble = (
        search_ensemble(
            results,
            cohort.y,
            n_top=config.ensemble.n_top_candidates,
            n_draws=config.ensemble.n_simplex_draws,
            seed=config.run.seed,
            include_structured=config.ensemble.include_structured_blends,
            progress=context.logger,
        )
        if config.ensemble.enabled
        else {"enabled": False, "reason": "disabled in config"}
    )

    context.state["plan"] = plan
    context.state["results"] = results
    context.state["ensemble"] = ensemble

    write_json(
        context.run_dir / "CANDIDATE_RESULTS.json",
        {
            "n_candidates": len(results),
            "n_failed": len(results) - len(usable),
            "skipped_families": skipped,
            "results": [r.to_dict() for r in results],
        },
    )
    context.record("candidate_results", context.run_dir / "CANDIDATE_RESULTS.json")
    return {
        "stage": "search",
        "n_candidates": len(results),
        "best_single": usable[0].to_dict(),
        "ensemble": {k: v for k, v in ensemble.items() if k != "blended_scores"},
    }


def stage_report(context: PipelineContext) -> dict[str, Any]:
    config = context.config
    cohort = _cohort(context)
    results: Sequence[CandidateResult] = context.state.get("results", ())
    if not results:
        raise RuntimeError("No search results in memory; run `--stage search` (or `all`) first")
    ensemble = context.state.get("ensemble", {"enabled": False})
    plan = context.state["plan"]

    usable = sorted(
        [r for r in results if r.error is None], key=lambda r: -r.subject_mean_auc
    )
    best_single = usable[0]

    if ensemble.get("enabled") and ensemble.get("subject_mean_oof_roc_auc", 0) > best_single.subject_mean_auc:
        champion_scores = np.asarray(ensemble["blended_scores"], dtype=float)
        champion_name = f"ensemble[{ensemble['selection_kind']}]"
        champion_auc = float(ensemble["subject_mean_oof_roc_auc"])
    else:
        champion_scores = best_single.subject_mean_oof
        champion_name = best_single.candidate.name
        champion_auc = best_single.subject_mean_auc

    metrics = binary_metrics(cohort.y, champion_scores, threshold=float(np.median(champion_scores)))
    interval = _subject_bootstrap(cohort.y, champion_scores, seed=config.run.seed)

    leaderboard = [
        {
            "rank": i + 1,
            "name": r.candidate.name,
            "family": r.candidate.family,
            "view": r.candidate.view,
            "top_k": r.candidate.top_k,
            "subject_mean_oof_roc_auc": round(r.subject_mean_auc, 6),
            "mean_repeat_roc_auc": round(r.mean_repeat_auc, 6),
            "sd_repeat_roc_auc": round(r.sd_repeat_auc, 6),
        }
        for i, r in enumerate(usable[:25])
    ]

    target = float(config.target_roc_auc)
    report = {
        "experiment": EXPERIMENT_NAME,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "task": {
            "positive": list(config.data.positive_diagnoses),
            "negative": list(config.data.negative_diagnoses),
            "cohort_splits": list(config.data.splits),
            "n_subjects": int(cohort.n_subjects),
            "n_positive": int(cohort.n_positive),
            "diagnosis_counts": cohort.audit["pooled"]["diagnosis_counts"],
        },
        "champion": {
            "name": champion_name,
            "subject_mean_oof_roc_auc": champion_auc,
            "subject_bootstrap_95_ci": interval,
            "metrics_at_median_threshold": metrics,
            "ensemble": {k: v for k, v in ensemble.items() if k != "blended_scores"},
        },
        "target": {
            "value": target,
            "reached": bool(champion_auc >= target),
            "gap": round(champion_auc - target, 6),
            "note": (
                "The target is an aspiration, not a guarantee. The strongest "
                "directly-comparable prior results without direct leakage are "
                "0.7834 (Binary_Google_YDF_AUC) and 0.7817 "
                "(Binary_Gemma_CognitiveFeature_AUC) on the 141-subject cohort."
            ),
        },
        "leaderboard_top25": leaderboard,
        "cv": {
            "plan_hash": plan.plan_hash,
            "n_splits": plan.n_splits,
            "n_repeats": plan.n_repeats,
            "seed": plan.seed,
            "shared_by_all_candidates": True,
            "nested": False,
        },
        "metric_contract": {
            "primary": "subject-mean repeated OOF ROC-AUC",
            "secondary": "mean of per-repeat pooled ROC-AUC",
            "nested": False,
            "selection_optimism": "present and disclosed; see LEAKAGE_AUDIT.json",
        },
        "honest_comparison": context.state.get("honest_comparison"),
        "artifacts": dict(context.artifacts),
        "caveats": [
            "Pooled 174-subject cohort: the historical 33-subject Validation set is "
            "inside training, so no untouched hold-out remains in this run.",
            "Non-nested selection: candidates, top_k and ensemble weights were "
            "chosen on the reported OOF. Repository evidence puts this optimism at "
            "roughly +0.05 to +0.08 ROC-AUC (Binary_Google_MaxAUC_Tuned: nested "
            "0.7172 vs non-nested 0.8017).",
            "MMSE-inclusive scores are partly circular and are not wearable-only "
            "screening performance.",
            "The only valid next confirmation is re-running this frozen "
            "configuration on new subjects or a new split seed without re-selecting.",
        ],
    }

    context.record("final_report", write_json(context.run_dir / "FINAL_REPORT.json", report))
    context.record(
        "leakage_audit", write_json(context.run_dir / "LEAKAGE_AUDIT.json", context.audit.to_dict())
    )
    _write_oof_csv(context, cohort, usable, champion_scores, champion_name)

    context.log("")
    context.log(f"[result] champion            : {champion_name}")
    context.log(f"[result] subject-mean OOF AUC: {champion_auc:.6f}")
    context.log(f"[result] bootstrap 95% CI    : [{interval[0]:.4f}, {interval[1]:.4f}]")
    context.log(f"[result] target {target:.2f} reached : {champion_auc >= target}")
    context.log("[result] NOTE: non-nested development score on a pooled cohort; see caveats.")
    return report


def _subject_bootstrap(y: np.ndarray, scores: np.ndarray, *, seed: int, n: int = 4000) -> list[float]:
    rng = np.random.default_rng(int(seed))
    target = np.asarray(y, dtype=np.int64)
    values = np.asarray(scores, dtype=float)
    indices = np.arange(len(target))
    draws: list[float] = []
    for _ in range(int(n)):
        sample = rng.choice(indices, size=len(indices), replace=True)
        if len(set(target[sample].tolist())) < 2:
            continue
        draws.append(roc_auc(target[sample], values[sample]))
    if not draws:
        return [float("nan"), float("nan")]
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def _write_oof_csv(
    context: PipelineContext,
    cohort,
    usable: Sequence[CandidateResult],
    champion_scores: np.ndarray,
    champion_name: str,
) -> None:
    import pandas as pd

    salt = context.config.run.subject_hash_salt
    frame = pd.DataFrame(
        {
            "subject_hash": [hash_subject_id(s, salt=salt) for s in cohort.subject_ids],
            "split_origin": cohort.split_origin,
            "y_true": cohort.y,
            "oof__champion": champion_scores,
        }
    )
    for result in usable[:10]:
        frame[f"oof__{result.candidate.name}"] = result.subject_mean_oof
    frame.attrs["champion"] = champion_name
    path = context.run_dir / "oof_predictions_hashed.csv"
    frame.to_csv(path, index=False)
    context.record("oof_predictions", path)


_STAGE_FUNCTIONS = {
    "audit": stage_audit,
    "features": stage_features,
    "search": stage_search,
    "report": stage_report,
}


def run_stage(context: PipelineContext, stage: str) -> dict[str, Any]:
    if stage not in _STAGE_FUNCTIONS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {STAGES}")
    return _STAGE_FUNCTIONS[stage](context)


def run_all(context: PipelineContext) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for stage in ("audit", "features", "search", "report"):
        reports[stage] = run_stage(context, stage)
    write_status(context, "complete", stages=list(reports))
    return reports
