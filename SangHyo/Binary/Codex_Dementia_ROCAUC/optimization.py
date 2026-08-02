"""Shared-split inner screening and Optuna tuning by subject-level OOF AUC."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .features import FeatureBundle
from .leakage import (
    assert_disjoint_groups,
    assert_fold_local_fit,
    assert_prediction_coverage,
)
from .metrics import safe_roc_auc
from .models.base import ModelSpec
from .models.fitting import fit_branch
from .splits import SplitPlan


@dataclass(frozen=True)
class InnerEvaluation:
    model_name: str
    params: dict[str, Any]
    aggregate_oof_score: np.ndarray
    aggregate_auc: float
    fold_auc: tuple[float, ...]
    repeat_auc: tuple[float, ...]
    repeat_oof_score: np.ndarray
    prediction_counts: np.ndarray
    fold_records: tuple[dict, ...]

    @property
    def fold_auc_std(self) -> float:
        return float(np.std(self.fold_auc, ddof=0))

    def summary(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "params": self.params,
            "aggregate_inner_oof_auc": self.aggregate_auc,
            "mean_fold_auc": float(np.mean(self.fold_auc)),
            "fold_auc_std": self.fold_auc_std,
            "repeat_auc": list(self.repeat_auc),
            "fold_records": list(self.fold_records),
        }


@dataclass(frozen=True)
class TuningResult:
    evaluation: InnerEvaluation
    study_name: str
    best_trial_number: int | None
    best_value: float
    trials: tuple[dict, ...]


def model_evaluation_seed(base_seed: int, model_position: int) -> int:
    """Give a model the same stochastic stream for fixed and tuned settings."""

    return int(base_seed + (int(model_position) + 1) * 2000003)


def evaluate_inner_oof(
    spec: ModelSpec,
    params: Mapping[str, Any],
    bundle: FeatureBundle,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: Sequence[int],
    inner_plan: SplitPlan,
    *,
    seed: int,
    config,
    trial=None,
) -> InnerEvaluation:
    """Cross-fit one fixed configuration entirely inside outer-training."""

    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    target = np.asarray(y, dtype=np.int64)
    subject_groups = np.asarray(groups, dtype=str)
    local_y = target[outer_train]
    prediction_sum = np.zeros(len(outer_train), dtype=np.float64)
    prediction_count = np.zeros(len(outer_train), dtype=np.int64)
    repeat_prediction = np.zeros(
        (inner_plan.n_repeats, len(outer_train)), dtype=np.float64
    )
    repeat_covered = np.zeros(
        (inner_plan.n_repeats, len(outer_train)), dtype=np.int64
    )
    fold_auc: list[float] = []
    fold_records: list[dict] = []
    for record in inner_plan.records:
        global_train = outer_train[record.train_indices]
        global_validation = outer_train[record.validation_indices]
        assert_disjoint_groups(
            subject_groups[global_train],
            subject_groups[global_validation],
            context=f"{record.split_id}/{spec.name}",
        )
        fitted = fit_branch(
            spec,
            params,
            bundle,
            target,
            global_train,
            seed=int(seed + record.repeat * 10007 + record.fold * 101),
            config=config,
        )
        assert_fold_local_fit(
            subject_groups[global_train],
            subject_groups[global_validation],
            component=f"{record.split_id}/{spec.name}",
        )
        score = fitted.predict(bundle, global_validation)
        auc = safe_roc_auc(target[global_validation], score)
        prediction_sum[record.validation_indices] += score
        prediction_count[record.validation_indices] += 1
        repeat_prediction[record.repeat, record.validation_indices] = score
        repeat_covered[record.repeat, record.validation_indices] += 1
        fold_auc.append(auc)
        fold_records.append(
            {
                "split_id": record.split_id,
                "repeat": record.repeat,
                "fold": record.fold,
                "auc": auc,
                "n_train": len(global_train),
                "n_validation": len(global_validation),
                "n_positive_train": int(target[global_train].sum()),
                "n_positive_validation": int(target[global_validation].sum()),
                "seed": fitted.seed,
            }
        )
        if trial is not None and np.all(repeat_covered[record.repeat] == 1):
            completed_repeats = np.flatnonzero(
                np.all(repeat_covered == 1, axis=1)
            )
            partial_aggregate = repeat_prediction[completed_repeats].mean(axis=0)
            trial.report(
                safe_roc_auc(local_y, partial_aggregate),
                step=int(record.repeat),
            )
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned(
                    f"Pruned {spec.name} after repeat {record.repeat}"
                )
    assert_prediction_coverage(
        prediction_count,
        expected_repeats=inner_plan.n_repeats,
        context=f"inner OOF/{spec.name}",
    )
    aggregate = prediction_sum / prediction_count
    repeat_auc: list[float] = []
    for repeat in range(inner_plan.n_repeats):
        if not np.all(repeat_covered[repeat] == 1):
            raise RuntimeError(f"Inner repeat {repeat} does not cover every subject")
        repeat_auc.append(safe_roc_auc(local_y, repeat_prediction[repeat]))
    return InnerEvaluation(
        model_name=spec.name,
        params=dict(params),
        aggregate_oof_score=aggregate,
        aggregate_auc=safe_roc_auc(local_y, aggregate),
        fold_auc=tuple(map(float, fold_auc)),
        repeat_auc=tuple(repeat_auc),
        repeat_oof_score=repeat_prediction,
        prediction_counts=prediction_count,
        fold_records=tuple(fold_records),
    )


def _suggest_params(trial, spec: ModelSpec, config) -> dict[str, Any]:
    if spec.family == "tabular":
        from .models.tabular import suggest_tabular_params

        return suggest_tabular_params(
            trial,
            spec,
            feature_selection_choices=config.search.feature_selection_choices,
            allow_random_over=config.search.enable_random_oversampling,
            allow_smote=config.search.enable_smote,
            allow_adasyn=config.search.enable_adasyn,
        )
    if spec.family == "tabnet":
        from .models.tabnet import suggest_tabnet_params

        return suggest_tabnet_params(
            trial,
            spec,
            feature_choices=config.search.feature_selection_choices,
        )
    if spec.family == "sequence":
        from .models.tsmixer import suggest_tsmixer_params

        return suggest_tsmixer_params(trial, spec.fixed_params)
    raise ValueError(spec.family)


def tune_inner_optuna(
    spec: ModelSpec,
    bundle: FeatureBundle,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: Sequence[int],
    inner_plan: SplitPlan,
    *,
    seed: int,
    config,
) -> TuningResult:
    """Tune a model with the same inner split registry used by every family."""

    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study_name = f"{spec.name}_{inner_plan.layer}"
    sampler = optuna.samplers.TPESampler(
        seed=int(seed),
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(5, config.search.optuna_trials_per_spec // 6),
        n_warmup_steps=0,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
    )

    def objective(trial):
        params = _suggest_params(trial, spec, config)
        evaluation = evaluate_inner_oof(
            spec,
            params,
            bundle,
            y,
            groups,
            outer_train_indices,
            inner_plan,
            seed=seed,
            config=config,
            trial=trial,
        )
        trial.set_user_attr("resolved_params_json", json.dumps(params, default=str))
        trial.set_user_attr("fold_auc_std", evaluation.fold_auc_std)
        trial.set_user_attr("mean_fold_auc", float(np.mean(evaluation.fold_auc)))
        return evaluation.aggregate_auc

    study.optimize(
        objective,
        n_trials=int(config.search.optuna_trials_per_spec),
        timeout=int(config.search.optuna_timeout_seconds_per_spec),
        n_jobs=1,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        fallback = evaluate_inner_oof(
            spec,
            spec.fixed_params,
            bundle,
            y,
            groups,
            outer_train_indices,
            inner_plan,
            seed=seed,
            config=config,
        )
        trials = tuple(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "duration_seconds": (
                    trial.duration.total_seconds() if trial.duration else None
                ),
            }
            for trial in study.trials
        )
        return TuningResult(
            evaluation=fallback,
            study_name=study_name,
            best_trial_number=None,
            best_value=fallback.aggregate_auc,
            trials=trials,
        )
    best_params = _suggest_params(study.best_trial, spec, config)
    # Re-evaluate deterministically to retain the exact OOF vector for blending.
    best_evaluation = evaluate_inner_oof(
        spec,
        best_params,
        bundle,
        y,
        groups,
        outer_train_indices,
        inner_plan,
        seed=seed,
        config=config,
    )
    trials: list[dict] = []
    for trial in study.trials:
        trials.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
                "duration_seconds": (
                    trial.duration.total_seconds() if trial.duration else None
                ),
            }
        )
    return TuningResult(
        evaluation=best_evaluation,
        study_name=study_name,
        best_trial_number=study.best_trial.number,
        best_value=float(study.best_value),
        trials=tuple(trials),
    )


def fixed_inner_screen(
    specs: Sequence[ModelSpec],
    bundle: FeatureBundle,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_indices: Sequence[int],
    inner_plan: SplitPlan,
    *,
    seed: int,
    config,
) -> dict[str, InnerEvaluation]:
    """Broad cheap screening before targeted Bayesian optimization."""

    output: dict[str, InnerEvaluation] = {}
    for position, spec in enumerate(specs):
        output[spec.name] = evaluate_inner_oof(
            spec,
            spec.fixed_params,
            bundle,
            y,
            groups,
            outer_train_indices,
            inner_plan,
            seed=model_evaluation_seed(seed, position),
            config=config,
        )
    return output


__all__ = [
    "InnerEvaluation",
    "TuningResult",
    "evaluate_inner_oof",
    "fixed_inner_screen",
    "model_evaluation_seed",
    "tune_inner_optuna",
]
