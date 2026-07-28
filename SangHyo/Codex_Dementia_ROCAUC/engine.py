"""Repeated nested CV, honest per-model benchmarks, and final refit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .artifacts import sha256_file, write_json
from .config import ExperimentConfig
from .data import TrackCohort
from .ensemble import BlendPolicy, fit_blend_policy, stacking_applicability
from .features import FeatureBundle, build_feature_bundle
from .leakage import (
    LeakageError,
    assert_disjoint_groups,
    assert_prediction_coverage,
    hash_subject_id,
)
from .metrics import (
    aggregate_repeated_oof,
    binary_metrics,
    choose_threshold,
    safe_roc_auc,
    save_curves,
    stratified_subject_bootstrap_auc,
    summarize_repeat_metrics,
)
from .models.base import ModelSpec, available_specs
from .models.fitting import (
    FittedBranch,
    SeedAveragedBranch,
    fit_branch,
    fit_branch_seed_ensemble,
)
from .optimization import (
    InnerEvaluation,
    fixed_inner_screen,
    model_evaluation_seed,
    tune_inner_optuna,
)
from .splits import (
    SplitPlan,
    build_inner_plan,
    build_repeated_group_plan,
    save_split_plan,
)


@dataclass
class SelectedStrategy:
    specs: dict[str, ModelSpec]
    params_by_model: dict[str, dict[str, Any]]
    evaluations: dict[str, InnerEvaluation]
    policy: BlendPolicy
    threshold: float
    threshold_record: dict[str, Any]
    screen_summary: dict[str, dict[str, Any]]
    tuning_summary: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "params_by_model": self.params_by_model,
            "blend_policy": self.policy.to_dict(),
            "threshold": self.threshold_record,
            "screening": self.screen_summary,
            "tuning": self.tuning_summary,
        }


@dataclass
class DeploymentEnsemble:
    track: str
    branches: dict[str, SeedAveragedBranch]
    policy: BlendPolicy
    threshold: float
    feature_names: tuple[str, ...]
    sequence_feature_names: tuple[str, ...]
    label_definition: dict[str, str]

    def predict_score(self, bundle: FeatureBundle) -> np.ndarray:
        if bundle.feature_names != self.feature_names:
            raise ValueError("Deployment tabular feature schema mismatch")
        if bundle.sequence_feature_names != self.sequence_feature_names:
            raise ValueError("Deployment sequence feature schema mismatch")
        indices = np.arange(len(bundle.subject_ids), dtype=np.int64)
        component = {
            name: branch.predict(bundle, indices)
            for name, branch in self.branches.items()
        }
        return self.policy.predict(component)

    def predict(self, bundle: FeatureBundle) -> np.ndarray:
        return (self.predict_score(bundle) >= float(self.threshold)).astype(np.int64)


@dataclass(frozen=True)
class PrimaryRunResult:
    track: str
    report: dict[str, Any]
    oof_path: str
    aggregate_oof_path: str
    model_comparison_path: str


def _dump_pickle(path: Path, payload: Any) -> None:
    import cloudpickle

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        cloudpickle.dump(payload, handle)


def load_deployment(path: str | Path) -> DeploymentEnsemble:
    import cloudpickle

    with Path(path).open("rb") as handle:
        model = cloudpickle.load(handle)
    if not isinstance(model, DeploymentEnsemble):
        raise TypeError("Deployment artifact has an unexpected type")
    return model


def _sequence_filter(
    specs: Sequence[ModelSpec],
    bundle: FeatureBundle,
) -> tuple[tuple[ModelSpec, ...], list[dict[str, Any]]]:
    selected: list[ModelSpec] = []
    skipped: list[dict[str, Any]] = []
    for spec in specs:
        if spec.family != "sequence":
            selected.append(spec)
            continue
        from .models.tsmixer import (
            TSMIXER_SEQUENCE_LENGTH_CHOICES,
            sequence_applicability,
        )

        applicable, reason = sequence_applicability(
            bundle.sequences,
            sequence_length=max(TSMIXER_SEQUENCE_LENGTH_CHOICES),
        )
        if applicable:
            selected.append(spec)
        else:
            skipped.append(
                {
                    **spec.engine_manifest(),
                    "reason": reason,
                    "model_executed": False,
                }
            )
    return tuple(selected), skipped


def _select_strategy(
    specs: Sequence[ModelSpec],
    bundle: FeatureBundle,
    cohort: TrackCohort,
    outer_train_indices: np.ndarray,
    inner_plan: SplitPlan,
    *,
    seed: int,
    config: ExperimentConfig,
) -> SelectedStrategy:
    if cohort.y is None:
        raise LeakageError("Strategy selection requires development labels")
    y, groups = cohort.y, cohort.groups
    screening = fixed_inner_screen(
        specs,
        bundle,
        y,
        groups,
        outer_train_indices,
        inner_plan,
        seed=seed,
        config=config,
    )
    ranked = sorted(
        specs,
        key=lambda spec: (
            -screening[spec.name].aggregate_auc,
            screening[spec.name].fold_auc_std,
            spec.name,
        ),
    )
    tune_names = {
        spec.name for spec in ranked[: int(config.search.top_specs_to_tune)]
    }
    if config.search.tune_tabnet_even_if_not_top:
        tune_names.update(
            spec.name for spec in specs if spec.family == "tabnet"
        )
    if config.search.tune_tsmixer_even_if_not_top:
        tune_names.update(
            spec.name for spec in specs if spec.family == "sequence"
        )

    evaluations = dict(screening)
    params_by_model = {
        spec.name: dict(spec.fixed_params) for spec in specs
    }
    tuning_summary: dict[str, dict[str, Any]] = {}
    for position, spec in enumerate(specs):
        if spec.name not in tune_names:
            continue
        tuning = tune_inner_optuna(
            spec,
            bundle,
            y,
            groups,
            outer_train_indices,
            inner_plan,
            seed=model_evaluation_seed(seed, position),
            config=config,
        )
        fixed_evaluation = screening[spec.name]
        repeat_wins = int(
            np.sum(
                np.asarray(tuning.evaluation.repeat_auc)
                > np.asarray(fixed_evaluation.repeat_auc) + 1e-12
            )
        )
        required_repeat_wins = len(fixed_evaluation.repeat_auc) // 2 + 1
        retained_tuned = (
            tuning.evaluation.aggregate_auc
            > fixed_evaluation.aggregate_auc + 1e-12
            and repeat_wins >= required_repeat_wins
        )
        if retained_tuned:
            evaluations[spec.name] = tuning.evaluation
            params_by_model[spec.name] = dict(tuning.evaluation.params)
        tuning_summary[spec.name] = {
            "study_name": tuning.study_name,
            "best_trial_number": tuning.best_trial_number,
            "best_value": tuning.best_value,
            "tuned_evaluation": tuning.evaluation.summary(),
            "fixed_evaluation": fixed_evaluation.summary(),
            "retained_configuration": (
                "tuned" if retained_tuned else "predeclared_fixed"
            ),
            "tuned_repeat_wins": repeat_wins,
            "required_repeat_wins": required_repeat_wins,
            "trials": list(tuning.trials),
        }

    anchor = "univariate_logreg"
    if anchor not in evaluations:
        raise LeakageError("Prespecified univariate anchor is unavailable")
    candidate_names = [anchor]
    for spec in sorted(
        specs,
        key=lambda value: (
            -evaluations[value.name].aggregate_auc,
            evaluations[value.name].fold_auc_std,
            value.name,
        ),
    ):
        if spec.name not in candidate_names:
            candidate_names.append(spec.name)
        if len(candidate_names) >= max(5, config.search.max_ensemble_members + 2):
            break
    local_y = y[np.asarray(outer_train_indices, dtype=np.int64)]
    policy = fit_blend_policy(
        local_y,
        {
            name: evaluations[name].aggregate_oof_score
            for name in candidate_names
        },
        anchor=anchor,
        max_members=config.search.max_ensemble_members,
        minimum_auc_gain=config.search.minimum_blend_auc_gain,
        weight_trials=config.search.blend_weight_trials,
        seed=seed + 880301,
        repeat_oof_by_model={
            name: evaluations[name].repeat_oof_score
            for name in candidate_names
        },
    )
    inner_blend = policy.predict(
        {
            name: evaluations[name].aggregate_oof_score
            for name in policy.members
        }
    )
    threshold_choice = choose_threshold(
        local_y,
        inner_blend,
        objective=config.cv.threshold_objective,
        minimum_recall=config.cv.threshold_min_recall,
    )
    return SelectedStrategy(
        specs={spec.name: spec for spec in specs},
        params_by_model=params_by_model,
        evaluations=evaluations,
        policy=policy,
        threshold=threshold_choice.threshold,
        threshold_record={
            "threshold": threshold_choice.threshold,
            "objective": threshold_choice.objective,
            "objective_value": threshold_choice.objective_value,
            "inner_oof_recall": threshold_choice.recall,
            "fit_scope": "current outer-training subjects' inner OOF only",
            "interpretation": (
                "selection-only apparent metric; model/tuning/blend/threshold "
                "choices reuse inner labels, so only outer OOF estimates performance"
            ),
        },
        screen_summary={
            name: evaluation.summary() for name, evaluation in screening.items()
        },
        tuning_summary=tuning_summary,
    )


def _extract_training_importance(
    branch: FittedBranch,
) -> list[dict[str, Any]]:
    """Best-effort training-fitted importance; never reads validation labels."""

    if branch.spec.family == "sequence":
        return []
    estimator = branch.estimator
    if hasattr(estimator, "named_steps"):
        try:
            model = estimator.named_steps["model"]
        except KeyError:
            return []
    else:
        model = estimator
    raw = None
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float)).reshape(-1)
    elif hasattr(model, "model_") and hasattr(model.model_, "feature_importances_"):
        raw = np.asarray(model.model_.feature_importances_, dtype=float)
    if raw is None or raw.ndim != 1:
        return []
    # After missing indicators and supervised selection, exact semantic mapping
    # can differ by fold. Persist transformed positions instead of inventing names.
    order = np.argsort(-raw, kind="stable")[: min(50, len(raw))]
    return [
        {
            "model": branch.spec.name,
            "transformed_feature_position": int(index),
            "importance": float(raw[index]),
            "semantic_mapping_note": (
                "position after fold-local imputation indicators and selection"
            ),
        }
        for index in order
        if np.isfinite(raw[index])
    ]


def _model_comparison(
    benchmark: pd.DataFrame,
    *,
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, frame in benchmark.groupby("model_name", sort=True):
        repeat_auc: list[float] = []
        repeat_pr: list[float] = []
        for repeat in range(repeats):
            subset = frame[frame["repeat"] == repeat]
            if len(subset) == 0:
                raise RuntimeError(f"{model_name} lacks benchmark repeat {repeat}")
            metrics = binary_metrics(
                subset["y"].to_numpy(),
                subset["score"].to_numpy(),
                threshold=0.5,
            )
            repeat_auc.append(float(metrics["roc_auc"]))
            repeat_pr.append(float(metrics["pr_auc"]))
        subject_mean = (
            frame.groupby("subject_hash", sort=False)
            .agg(y=("y", "first"), score=("score", "mean"))
            .reset_index()
        )
        rows.append(
            {
                "model_name": model_name,
                "repeat_oof_roc_auc_mean": float(np.mean(repeat_auc)),
                "repeat_oof_roc_auc_std": float(np.std(repeat_auc, ddof=0)),
                "repeat_oof_pr_auc_mean": float(np.mean(repeat_pr)),
                "subject_mean_repeated_oof_roc_auc": safe_roc_auc(
                    subject_mean["y"], subject_mean["score"]
                ),
                "n_repeats": repeats,
                "evaluation": "fixed-config subject-level outer OOF",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["repeat_oof_roc_auc_mean", "model_name"],
        ascending=[False, True],
    )


def run_primary_nested_oof(
    cohort: TrackCohort,
    *,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> PrimaryRunResult:
    """Run the primary Training-only repeated nested subject OOF experiment."""

    if cohort.y is None or cohort.diagnosis is None:
        raise LeakageError("Primary nested OOF requires labeled development cohort")
    started = time.monotonic()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = build_feature_bundle(cohort)
    specs, dependency_skips = available_specs(
        config.search.screen_model_names,
        fail_on_missing=config.runtime.fail_on_missing_optional_model,
    )
    specs, sequence_skips = _sequence_filter(specs, bundle)
    skipped = [*dependency_skips, *sequence_skips]
    if "univariate_logreg" not in {spec.name for spec in specs}:
        raise LeakageError("The prespecified anchor cannot be skipped")

    outer_plan = build_repeated_group_plan(
        cohort.y,
        cohort.groups,
        n_splits=config.cv.outer_folds,
        n_repeats=config.cv.outer_repeats,
        seed=config.cv.seed,
        minimum_positive_validation=config.cv.minimum_positive_per_validation_fold,
        layer=f"{cohort.track}_outer",
    )
    save_split_plan(
        output / "split_registry_outer.json",
        outer_plan,
        subject_ids=cohort.subject_ids,
        y=cohort.y,
    )
    specs_by_name = {spec.name: spec for spec in specs}
    spec_positions = {spec.name: position for position, spec in enumerate(specs)}
    ensemble_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    fold_manifests: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for outer_position, outer in enumerate(outer_plan.records):
        inner = build_inner_plan(
            outer.train_indices,
            cohort.y,
            cohort.groups,
            n_splits=config.cv.inner_folds,
            n_repeats=config.cv.inner_repeats,
            seed=config.cv.seed + outer.repeat * 50021 + outer.fold * 809,
            minimum_positive_validation=config.cv.minimum_positive_per_validation_fold,
            layer=f"{outer.split_id}_inner",
        )
        save_split_plan(
            output / "inner_splits" / f"{outer.split_id}.json",
            inner,
            subject_ids=cohort.subject_ids[outer.train_indices],
            y=cohort.y[outer.train_indices],
        )
        strategy = _select_strategy(
            specs,
            bundle,
            cohort,
            outer.train_indices,
            inner,
            seed=config.cv.seed + outer_position * 10000019,
            config=config,
        )
        component_scores: dict[str, np.ndarray] = {}
        selected_branches: dict[str, SeedAveragedBranch] = {}
        for name in strategy.policy.members:
            branch = fit_branch_seed_ensemble(
                specs_by_name[name],
                strategy.params_by_model[name],
                bundle,
                cohort.y,
                outer.train_indices,
                seed=outer.seed + (spec_positions[name] + 1) * 1543,
                n_members=config.cv.inner_repeats,
                config=config,
            )
            score = branch.predict(bundle, outer.validation_indices)
            component_scores[name] = score
            selected_branches[name] = branch
            if config.runtime.save_training_fitted_importance:
                for seed_member, fitted_member in enumerate(branch.members):
                    records = _extract_training_importance(fitted_member)
                    for importance in records:
                        importance.update(
                            {
                                "split_id": outer.split_id,
                                "repeat": outer.repeat,
                                "fold": outer.fold,
                                "seed_member": seed_member,
                                "seed": fitted_member.seed,
                            }
                        )
                    importance_rows.extend(records)
        blended = strategy.policy.predict(component_scores)
        prediction = (blended >= strategy.threshold).astype(np.int64)
        for local_position, global_position in enumerate(outer.validation_indices):
            ensemble_rows.append(
                {
                    "track": cohort.track,
                    "subject_hash": hash_subject_id(
                        cohort.subject_ids[global_position]
                    ),
                    "diagnosis": str(cohort.diagnosis[global_position]),
                    "y": int(cohort.y[global_position]),
                    "repeat": outer.repeat,
                    "fold": outer.fold,
                    "split_id": outer.split_id,
                    "score": float(blended[local_position]),
                    "prediction": int(prediction[local_position]),
                    "fold_local_threshold": float(strategy.threshold),
                    "selected_models": "|".join(strategy.policy.members),
                    "blend_mode": strategy.policy.mode,
                }
            )

        if config.search.benchmark_all_fixed_models:
            for benchmark_position, spec in enumerate(specs):
                branch = fit_branch(
                    spec,
                    spec.fixed_params,
                    bundle,
                    cohort.y,
                    outer.train_indices,
                    seed=outer.seed + 500000 + benchmark_position * 4099,
                    config=config,
                )
                scores = branch.predict(bundle, outer.validation_indices)
                for local_position, global_position in enumerate(
                    outer.validation_indices
                ):
                    benchmark_rows.append(
                        {
                            "track": cohort.track,
                            "model_name": spec.name,
                            "subject_hash": hash_subject_id(
                                cohort.subject_ids[global_position]
                            ),
                            "y": int(cohort.y[global_position]),
                            "repeat": outer.repeat,
                            "fold": outer.fold,
                            "split_id": outer.split_id,
                            "score": float(scores[local_position]),
                            "configuration": "predeclared_fixed",
                        }
                    )

        fold_manifest = {
            "split_id": outer.split_id,
            "repeat": outer.repeat,
            "fold": outer.fold,
            "train_subject_hashes": [
                hash_subject_id(value)
                for value in cohort.subject_ids[outer.train_indices]
            ],
            "validation_subject_hashes": [
                hash_subject_id(value)
                for value in cohort.subject_ids[outer.validation_indices]
            ],
            "strategy": strategy.to_dict(),
            "selected_refit_seed_ensemble_members": config.cv.inner_repeats,
            "outer_metrics": binary_metrics(
                cohort.y[outer.validation_indices],
                blended,
                threshold=strategy.threshold,
                prediction=prediction,
            ),
        }
        fold_manifests.append(fold_manifest)
        write_json(
            output / "folds" / f"{outer.split_id}.json",
            fold_manifest,
        )
        if config.runtime.save_fold_models:
            _dump_pickle(
                output / "fold_models" / f"{outer.split_id}.pkl",
                {
                    "branches": selected_branches,
                    "policy": strategy.policy,
                    "threshold": strategy.threshold,
                },
            )
        write_json(
            output / "progress.json",
            {
                "status": "running",
                "completed_outer_folds": outer_position + 1,
                "total_outer_folds": len(outer_plan.records),
                "elapsed_seconds": time.monotonic() - started,
            },
        )

    oof = pd.DataFrame(ensemble_rows)
    expected_subject_hashes = {
        hash_subject_id(value) for value in cohort.subject_ids
    }
    observed_subject_hashes = set(oof["subject_hash"].astype(str))
    if (
        len(expected_subject_hashes) != cohort.n_subjects
        or observed_subject_hashes != expected_subject_hashes
    ):
        raise LeakageError(
            f"{cohort.track}: primary OOF subject set/hash collision mismatch"
        )
    counts = oof.groupby("subject_hash").size().to_numpy()
    assert_prediction_coverage(
        counts,
        expected_repeats=config.cv.outer_repeats,
        context=f"{cohort.track} primary outer OOF",
    )
    oof_path = output / "oof_predictions_long.csv"
    oof.to_csv(oof_path, index=False)
    repeat_metrics: list[dict[str, Any]] = []
    for repeat in range(config.cv.outer_repeats):
        subset = oof[oof["repeat"] == repeat]
        record = binary_metrics(
            subset["y"],
            subset["score"],
            prediction=subset["prediction"],
        )
        record["repeat"] = repeat
        repeat_metrics.append(record)
    aggregate = aggregate_repeated_oof(oof, config.cv.outer_repeats)
    aggregate_path = output / "oof_predictions_subject_mean.csv"
    aggregate.to_csv(aggregate_path, index=False)
    aggregate_metrics = binary_metrics(
        aggregate["y"],
        aggregate["score"],
        prediction=aggregate["prediction"],
    )
    bootstrap = stratified_subject_bootstrap_auc(
        aggregate["y"],
        aggregate["score"],
        iterations=config.cv.bootstrap_iterations,
        confidence=config.cv.bootstrap_confidence,
        seed=config.cv.seed,
    )
    curve_paths = save_curves(
        aggregate["y"],
        aggregate["score"],
        output_dir=output / "curves",
        prefix=cohort.track,
    )

    benchmark = pd.DataFrame(benchmark_rows)
    if benchmark.empty:
        comparison = pd.DataFrame(
            columns=[
                "model_name",
                "repeat_oof_roc_auc_mean",
                "repeat_oof_roc_auc_std",
                "repeat_oof_pr_auc_mean",
                "subject_mean_repeated_oof_roc_auc",
                "n_repeats",
                "evaluation",
            ]
        )
    else:
        for model_name, frame in benchmark.groupby("model_name", sort=True):
            if set(frame["subject_hash"].astype(str)) != expected_subject_hashes:
                raise LeakageError(
                    f"{cohort.track}/{model_name}: benchmark OOF subject set mismatch"
                )
            assert_prediction_coverage(
                frame.groupby("subject_hash").size().to_numpy(),
                expected_repeats=config.cv.outer_repeats,
                context=f"{cohort.track}/{model_name} fixed benchmark OOF",
            )
        benchmark.to_csv(output / "fixed_model_oof_long.csv", index=False)
        comparison = _model_comparison(
            benchmark,
            repeats=config.cv.outer_repeats,
        )
    comparison_path = output / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    importance_path = output / "training_fitted_importance.csv"
    pd.DataFrame(
        importance_rows,
        columns=(
            None
            if importance_rows
            else [
                "model",
                "transformed_feature_position",
                "importance",
                "semantic_mapping_note",
                "split_id",
                "repeat",
                "fold",
                "seed_member",
                "seed",
            ]
        ),
    ).to_csv(importance_path, index=False)
    write_json(output / "fold_manifests.json", fold_manifests)

    report = {
        "track": cohort.track,
        "task": "CN+MCI (0) versus Dem (1)",
        "primary_evaluation": (
            "official Training split only; repeated nested subject-level OOF"
        ),
        "is_independent_external_test": False,
        "n_subjects": cohort.n_subjects,
        "n_positive_dem": cohort.n_positive,
        "n_negative_cn_mci": int((cohort.y == 0).sum()),
        "feature_count": len(bundle.feature_names),
        "sequence_schema_width": len(bundle.sequence_feature_names),
        "outer_cv": {
            "folds": config.cv.outer_folds,
            "repeats": config.cv.outer_repeats,
        },
        "inner_cv": {
            "folds": config.cv.inner_folds,
            "repeats": config.cv.inner_repeats,
        },
        "selected_refit_seed_ensemble_members": config.cv.inner_repeats,
        "repeat_level_estimand": {
            "description": (
                "score each complete outer-OOF repeat, then summarize split noise"
            ),
            "metrics": summarize_repeat_metrics(repeat_metrics),
        },
        "subject_mean_repeated_oof_estimand": {
            "description": (
                "average each subject's cross-fitted scores across repeats, then score"
            ),
            "metrics": aggregate_metrics,
            "bootstrap_roc_auc": bootstrap,
        },
        "estimands_not_mixed": True,
        "fixed_model_comparison_file": str(comparison_path),
        "skipped_models": skipped,
        "stacking": stacking_applicability(cohort.y),
        "timesfm": {
            "executed": False,
            "reason": (
                "TimesFM is a forecasting model; this repository has no validated "
                "frozen-representation contract for classification, and twelve/ "
                "nine positives do not justify representation tuning."
            ),
        },
        "curve_files": curve_paths,
        "input_fingerprints": cohort.input_fingerprints,
        "access_audit": cohort.access_audit,
        "elapsed_seconds": time.monotonic() - started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "limitations": [
            "Primary development OOF has only nine Dem subjects.",
            "Repeated folds quantify split noise, not new-subject sampling uncertainty.",
            "The project has repeatedly inspected these subjects in prior experiments, "
            "so project-level model-selection bias remains.",
            "wearable_mmse is a separate reference track with diagnostic incorporation bias.",
            "Inner-CV tuning/blend/threshold metrics are selection-only; only the "
            "untouched outer OOF estimates the complete adaptive procedure.",
        ],
    }
    write_json(output / "PRIMARY_REPORT.json", report)
    write_json(
        output / "PRIMARY_COMPLETE.json",
        {
            "status": "complete",
            "track": cohort.track,
            "oof_sha256": sha256_file(oof_path),
            "aggregate_oof_sha256": sha256_file(aggregate_path),
            "report_sha256": sha256_file(output / "PRIMARY_REPORT.json"),
        },
    )
    return PrimaryRunResult(
        track=cohort.track,
        report=report,
        oof_path=str(oof_path),
        aggregate_oof_path=str(aggregate_path),
        model_comparison_path=str(comparison_path),
    )


def refit_deployment(
    cohort: TrackCohort,
    *,
    config: ExperimentConfig,
    output_dir: str | Path,
) -> tuple[DeploymentEnsemble, dict[str, Any]]:
    """Rerun the same selection procedure on all development subjects and refit."""

    if cohort.y is None:
        raise LeakageError("Deployment refit requires development labels")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = build_feature_bundle(cohort)
    specs, dependency_skips = available_specs(
        config.search.screen_model_names,
        fail_on_missing=config.runtime.fail_on_missing_optional_model,
    )
    specs, sequence_skips = _sequence_filter(specs, bundle)
    inner = build_repeated_group_plan(
        cohort.y,
        cohort.groups,
        n_splits=config.cv.inner_folds,
        n_repeats=config.cv.inner_repeats,
        seed=config.cv.seed + 70000001,
        minimum_positive_validation=config.cv.minimum_positive_per_validation_fold,
        layer=f"{cohort.track}_final_inner",
    )
    save_split_plan(
        output / "deployment_inner_splits.json",
        inner,
        subject_ids=cohort.subject_ids,
        y=cohort.y,
    )
    all_indices = np.arange(cohort.n_subjects, dtype=np.int64)
    strategy = _select_strategy(
        specs,
        bundle,
        cohort,
        all_indices,
        inner,
        seed=config.cv.seed + 90000011,
        config=config,
    )
    specs_by_name = {spec.name: spec for spec in specs}
    spec_positions = {spec.name: position for position, spec in enumerate(specs)}
    branches = {
        name: fit_branch_seed_ensemble(
            specs_by_name[name],
            strategy.params_by_model[name],
            bundle,
            cohort.y,
            all_indices,
            seed=(
                config.cv.seed
                + 95000021
                + (spec_positions[name] + 1) * 109
            ),
            n_members=config.cv.inner_repeats,
            config=config,
        )
        for name in strategy.policy.members
    }
    deployment = DeploymentEnsemble(
        track=cohort.track,
        branches=branches,
        policy=strategy.policy,
        threshold=strategy.threshold,
        feature_names=bundle.feature_names,
        sequence_feature_names=bundle.sequence_feature_names,
        label_definition={"negative": "CN+MCI", "positive": "Dem"},
    )
    model_path = output / "deployment.pkl"
    _dump_pickle(model_path, deployment)
    reloaded = load_deployment(model_path)
    sample_positions = np.arange(min(7, cohort.n_subjects), dtype=np.int64)
    # Build a complete bundle because DeploymentEnsemble validates schemas, then
    # compare a deterministic prefix after serialization.
    before = deployment.predict_score(bundle)[sample_positions]
    after = reloaded.predict_score(bundle)[sample_positions]
    if not np.allclose(before, after, rtol=1e-6, atol=1e-7):
        raise IOError("Deployment serialization round-trip changed predictions")
    manifest = {
        "track": cohort.track,
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "strategy": strategy.to_dict(),
        "round_trip": {
            "passed": True,
            "n_checked": len(sample_positions),
            "rtol": 1e-6,
            "atol": 1e-7,
        },
        "skipped_models": [*dependency_skips, *sequence_skips],
        "trained_subjects": cohort.n_subjects,
        "trained_positive_dem": cohort.n_positive,
        "seed_ensemble_members_per_model": config.cv.inner_repeats,
    }
    write_json(output / "deployment_manifest.json", manifest)
    return deployment, manifest


__all__ = [
    "DeploymentEnsemble",
    "PrimaryRunResult",
    "load_deployment",
    "refit_deployment",
    "run_primary_nested_oof",
]
