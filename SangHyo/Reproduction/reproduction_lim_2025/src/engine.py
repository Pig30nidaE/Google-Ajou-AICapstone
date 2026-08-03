"""Orchestration for the three experiments.

Ordering rule that the whole file exists to enforce: **split first, then build the
representation, then fit preprocessing on the training part only.**  Sequences are
assembled inside each split so a window can never span the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .audit import leakage
from .data import schema
from .evaluation import metrics as M
from .features.representations import (
    FoldPreprocessor,
    build_representation,
    fit_transform_pair,
)
from .models import registry
from .splits import splitters
from .utils.io import CheckpointStore, write_json
from .utils.seeding import fold_seed, seed_everything

PRIMARY_METRIC = "roc_auc"


@dataclass
class FoldResult:
    model: str
    repeat: int
    fold: int
    subject_predictions: pd.DataFrame
    subject_metrics: dict[str, Any]
    record_metrics: dict[str, Any] | None = None
    selection: dict[str, Any] | None = None
    threshold: float = 0.5
    threshold_source: str = "fixed"
    train_subjects: tuple[str, ...] = ()
    test_subjects: tuple[str, ...] = ()
    model_summary: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    input_shape: tuple[int, ...] = ()


def _representation_kwargs(config, model: str) -> dict[str, Any]:
    """Representation options, with per-model overrides layered on the defaults."""
    base = dict(config.get("representation_options", {}) or {})
    base.update(dict(config.get(f"representation_overrides.{model}", {}) or {}))
    return base


def _materialise(
    data, config, model: str, subjects: Sequence[str], *, overrides: dict[str, Any] | None = None
):
    """Build one model's representation for exactly the given subjects."""
    kind = config.get(f"representations.{model}")
    kwargs = _representation_kwargs(config, model)
    kwargs.update(overrides or {})
    return build_representation(
        kind, data.daily, data.feature_columns, subjects=subjects, **kwargs
    )


def materialise_pair(data, config, model: str, train_subjects, test_subjects):
    """Build the train and test representations with a shared, train-derived shape.

    ``sequence_length: max`` must resolve on the *training* subjects only.  Letting
    each side resolve its own length both mismatches the tensors (a held-out
    subject with more days would silently widen T) and lets the evaluation set
    decide a preprocessing parameter.
    """
    train_rep = _materialise(data, config, model, train_subjects)
    overrides: dict[str, Any] | None = None
    if train_rep.kind == "temporal_sequence":
        overrides = {"sequence_length": int(train_rep.meta["sequence_length"])}
    test_rep = _materialise(data, config, model, test_subjects, overrides=overrides)
    return train_rep, test_rep


def _needs_scaling(model: str, config) -> bool:
    """Tree models are scale-invariant; the paper only normalises the deep inputs."""
    if config.get("preprocessing.standardize_tree_models", False):
        return True
    return registry.is_sequence_model(model)


def _fit_predict(
    model_name: str,
    train_rep,
    test_rep,
    *,
    seed: int,
    device: str,
    overrides: dict[str, Any] | None,
    training: dict[str, Any] | None,
) -> tuple[np.ndarray, Any]:
    """Fit on the training representation and score the test representation."""
    model = registry.build_model(
        model_name, seed=seed, device=device, overrides=overrides, training=training
    )
    if registry.is_sequence_model(model_name):
        # pack_padded_sequence and the Conv1d pooling mask both require the valid
        # steps at the front, so hand them a left-aligned view regardless of the
        # configured padding side.
        train_X, train_lengths = train_rep.left_aligned()
        test_X, test_lengths = test_rep.left_aligned()
        model.fit(train_X, train_rep.y, lengths=train_lengths, seed=seed)
        probabilities = model.predict_proba(test_X, lengths=test_lengths)
    else:
        model.fit(train_rep.X, train_rep.y)
        probabilities = model.predict_proba(test_rep.X)
    return np.asarray(probabilities, dtype=np.float64), model


def _score_fold(
    test_rep,
    probabilities: np.ndarray,
    *,
    config,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any] | None]:
    """Pool predictions to subjects and compute both evaluation levels."""
    pooling = str(config.get("evaluation.pooling", "mean"))
    pooled = M.pool_to_subjects(
        test_rep.subjects, probabilities, test_rep.y, method=pooling, threshold=threshold
    )
    subject_metrics = M.compute_metrics(
        pooled["label"], pooled["probability"], threshold=threshold
    )
    subject_metrics["pooling"] = pooling
    subject_metrics["evaluation_unit"] = "subject"

    record_metrics = None
    if test_rep.kind == "daily_record":
        record_metrics = M.compute_metrics(test_rep.y, probabilities, threshold=threshold)
        record_metrics["evaluation_unit"] = "daily_record"
        record_metrics["note"] = "보조 결과. 주 평가는 피험자 단위다."

    if config.get("evaluation.sensitivity_pooling", True) and pooling == "mean":
        alternatives: dict[str, Any] = {}
        for method in ("median", "majority"):
            other = M.pool_to_subjects(
                test_rep.subjects, probabilities, test_rep.y,
                method=method, threshold=threshold,
            )
            alternatives[method] = M.compute_metrics(
                other["label"], other["probability"], threshold=threshold
            )
        subject_metrics["pooling_sensitivity"] = alternatives

    return pooled, subject_metrics, record_metrics


def _inner_select(
    data, config, model_name: str, split, *, seed: int, device: str
) -> dict[str, Any]:
    """Model/hyperparameter/threshold selection inside the outer training subjects."""
    labels = data.labels_by_subject()
    inner_splits = splitters.inner_stratified_group_kfold(
        split.train_subjects,
        labels,
        n_splits=int(config.get("split.inner_k", 3)),
        seed=seed,
    )

    # Proof that the outer test never enters tuning -- raises if it does.
    audit_log = leakage.AuditLog()
    leakage.check_outer_test_isolation(split.test_subjects, inner_splits, audit_log)

    candidates = registry.search_space(
        model_name, enabled=bool(config.get("tuning.enabled", False))
    )
    training_cfg = dict(config.get("training", {}) or {})

    scored: list[dict[str, Any]] = []
    for index, overrides in enumerate(candidates):
        oof_prob: list[np.ndarray] = []
        oof_true: list[np.ndarray] = []
        for inner in inner_splits:
            train_rep, test_rep = materialise_pair(
                data, config, model_name, inner.train_subjects, inner.test_subjects
            )
            pre = FoldPreprocessor(standardize=_needs_scaling(model_name, config))
            fit_transform_pair(pre, train_rep, test_rep)
            leakage.check_preprocessing_scope(
                pre, inner.train_subjects, inner.test_subjects, audit_log, tag="_inner"
            )

            probabilities, _ = _fit_predict(
                model_name, train_rep, test_rep,
                seed=seed + index, device=device,
                overrides=overrides, training=training_cfg,
            )
            pooled = M.pool_to_subjects(
                test_rep.subjects, probabilities, test_rep.y,
                method=str(config.get("evaluation.pooling", "mean")),
            )
            oof_prob.append(pooled["probability"].to_numpy())
            oof_true.append(pooled["label"].to_numpy())

        y_prob = np.concatenate(oof_prob)
        y_true = np.concatenate(oof_true)
        score = M.compute_metrics(y_true, y_prob)[PRIMARY_METRIC]
        scored.append(
            {
                "overrides": overrides,
                "inner_score": None if not np.isfinite(score) else float(score),
                "inner_oof_probabilities": y_prob,
                "inner_oof_labels": y_true,
            }
        )

    ranked = [c for c in scored if c["inner_score"] is not None]
    best = max(ranked, key=lambda c: c["inner_score"]) if ranked else scored[0]

    threshold = 0.5
    if config.get("threshold.policy") == "inner_cv" and best.get("inner_oof_labels") is not None:
        threshold = M.select_threshold(
            best["inner_oof_labels"],
            best["inner_oof_probabilities"],
            objective=str(config.get("threshold.objective", "balanced_accuracy")),
        )

    return {
        "selected_overrides": best["overrides"],
        "inner_score": best.get("inner_score"),
        "threshold": float(threshold),
        "threshold_source": "inner_cv",
        "n_inner_folds": len(inner_splits),
        "n_candidates": len(candidates),
        "candidate_scores": [
            {"overrides": c["overrides"], "inner_score": c["inner_score"]} for c in scored
        ],
        "inner_audit": audit_log.summary(),
        "inner_fold_subjects": [
            {"train": list(s.train_subjects), "test": list(s.test_subjects)}
            for s in inner_splits
        ],
        "_inner_splits": inner_splits,
    }


def run_fold(
    data, config, model_name: str, split, *, device: str, checkpoints: CheckpointStore | None = None
) -> FoldResult:
    """Train and evaluate one model on one split, with audits before fitting."""
    seed = fold_seed(config.seed, split.repeat, split.fold)
    seed_everything(seed)

    selection: dict[str, Any] | None = None
    overrides: dict[str, Any] | None = None
    threshold = float(config.get("threshold.value", 0.5))
    threshold_source = str(config.get("threshold.policy", "fixed"))
    inner_splits: Sequence[Any] | None = None

    if config.experiment == "nested_subject_independent":
        selection = _inner_select(
            data, config, model_name, split, seed=seed, device=device
        )
        overrides = selection["selected_overrides"]
        threshold = selection["threshold"]
        threshold_source = selection["threshold_source"]
        inner_splits = selection.pop("_inner_splits")

    # --- split first, then represent -----------------------------------------
    # `overrides` are model hyperparameters, not representation options, so they
    # deliberately do not reach the builder here.
    train_rep, test_rep = materialise_pair(
        data, config, model_name, split.train_subjects, split.test_subjects
    )

    # --- preprocessing fitted on the training part only ----------------------
    preprocessor = FoldPreprocessor(standardize=_needs_scaling(model_name, config))
    if config.scaler_scope == "all_data":
        # Only reachable in the paper reconstruction, and only to measure the bias
        # that fitting on everything would introduce.
        fit_subjects = tuple(sorted(set(split.train_subjects) | set(split.test_subjects)))
        combined = _materialise(
            data, config, model_name, fit_subjects,
            overrides=(
                {"sequence_length": int(train_rep.meta["sequence_length"])}
                if train_rep.kind == "temporal_sequence" else None
            ),
        )
        fit_transform_pair(
            preprocessor, train_rep, test_rep,
            fit_subjects=fit_subjects, fit_on=combined,
        )
    else:
        fit_transform_pair(preprocessor, train_rep, test_rep)

    # --- audit BEFORE fitting -------------------------------------------------
    audit_log = leakage.audit_split(
        split,
        train_rep=train_rep,
        test_rep=test_rep,
        preprocessor=None if config.scaler_scope == "all_data" else preprocessor,
        experiment=config.experiment,
        threshold_source=threshold_source,
        include_cognitive=bool(config.get("features.include_cognitive_tests", False)),
        inner_splits=inner_splits,
    )
    if config.scaler_scope == "all_data":
        audit_log.record(
            "preprocessor_scope_all_data",
            False,
            {
                "declared": "assumption_variant: scaler fitted on train+test",
                "interpret_as": "optimism measurement, not a performance claim",
            },
        )

    probabilities, model = _fit_predict(
        model_name, train_rep, test_rep,
        seed=seed, device=device,
        overrides=overrides, training=dict(config.get("training", {}) or {}),
    )

    pooled, subject_metrics, record_metrics = _score_fold(
        test_rep, probabilities, config=config, threshold=threshold
    )

    if checkpoints is not None and config.get("checkpointing.save_models", False):
        suffix = ".pt" if registry.is_sequence_model(model_name) else ".joblib"
        try:
            model.save(checkpoints.model_path(model_name, split.repeat, split.fold, suffix))
        except Exception as error:  # checkpointing must never sink a completed fold
            audit_log.record("checkpoint_save", False, {"error": str(error)})

    return FoldResult(
        model=model_name,
        repeat=split.repeat,
        fold=split.fold,
        subject_predictions=pooled,
        subject_metrics=subject_metrics,
        record_metrics=record_metrics,
        selection=selection,
        threshold=threshold,
        threshold_source=threshold_source,
        train_subjects=tuple(split.train_subjects),
        test_subjects=tuple(split.test_subjects),
        model_summary=model.summary(),
        audit=audit_log.summary(),
        input_shape=train_rep.input_shape,
    )


def aggregate_model(model_name: str, folds: Sequence[FoldResult], *, seed: int = 42) -> dict[str, Any]:
    """Pool out-of-fold subject predictions and summarise across folds."""
    if not folds:
        return {"model": model_name, "n_folds": 0, "subject_level": None}

    frames = []
    for fold in folds:
        frame = fold.subject_predictions.copy()
        frame["repeat"] = fold.repeat
        frame["fold"] = fold.fold
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)

    # Average duplicate predictions when repeats put a subject in several test folds.
    pooled = (
        oof.groupby(["repeat", "subject_id"], as_index=False)
        .agg(probability=("probability", "mean"), label=("label", "first"))
    )
    per_repeat: list[dict[str, Any]] = []
    for repeat, group in pooled.groupby("repeat"):
        try:
            per_repeat.append(
                {
                    "repeat": int(repeat),
                    **M.compute_metrics(group["label"], group["probability"],
                                        threshold=float(folds[0].threshold)),
                }
            )
        except ValueError:
            continue

    overall = (
        pooled.groupby("subject_id", as_index=False)
        .agg(probability=("probability", "mean"), label=("label", "first"))
    )
    subject_level = M.compute_metrics(
        overall["label"], overall["probability"], threshold=float(folds[0].threshold)
    )
    subject_level["evaluation_unit"] = "subject"

    fold_scores = [
        f.subject_metrics.get(PRIMARY_METRIC) for f in folds
        if f.subject_metrics.get(PRIMARY_METRIC) is not None
    ]
    fold_scores = [s for s in fold_scores if np.isfinite(s)]

    return {
        "model": model_name,
        "n_folds": len(folds),
        "subject_level": subject_level,
        "bootstrap_ci": M.bootstrap_ci(
            overall["label"], overall["probability"], metric=PRIMARY_METRIC, seed=seed
        ),
        "calibration_curve": M.calibration_curve_points(
            overall["label"], overall["probability"]
        ),
        "fold_scores": fold_scores,
        "fold_score_std": float(np.std(fold_scores)) if len(fold_scores) > 1 else None,
        "fold_score_mean": float(np.mean(fold_scores)) if fold_scores else None,
        "per_repeat": per_repeat,
        "repeat_score_std": (
            float(np.std([r[PRIMARY_METRIC] for r in per_repeat
                          if np.isfinite(r.get(PRIMARY_METRIC, np.nan))]))
            if len(per_repeat) > 1 else None
        ),
        "threshold": float(folds[0].threshold),
        "threshold_source": folds[0].threshold_source,
        "model_summary": folds[0].model_summary,
        "input_shape": list(folds[0].input_shape),
        "selections": [f.selection for f in folds if f.selection],
    }


def run_experiment(
    data, config, *, output_dir: Path, device: str = "cpu", only_fold: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Run every configured model over every split and write the artifacts."""
    output_dir = Path(output_dir)
    (output_dir / "folds").mkdir(parents=True, exist_ok=True)
    checkpoints = CheckpointStore(output_dir / "checkpoints")

    dataset_audit = leakage.audit_dataset(data)
    splits = splitters.build_outer_splits(data, config)
    if only_fold is not None:
        splits = [s for s in splits if s.fold == only_fold]
        if not splits:
            raise ValueError(f"--fold {only_fold} matched no split")

    labels = data.labels_by_subject()
    results: dict[str, list[FoldResult]] = {model: [] for model in config.models}
    fold_records: list[dict[str, Any]] = []

    for split in splits:
        for model_name in config.models:
            if resume and checkpoints.is_complete(model_name, split.repeat, split.fold):
                cached = checkpoints.load(model_name, split.repeat, split.fold)
                frame = pd.DataFrame(cached["subject_predictions"])
                results[model_name].append(
                    FoldResult(
                        model=model_name, repeat=split.repeat, fold=split.fold,
                        subject_predictions=frame,
                        subject_metrics=cached["subject_metrics"],
                        record_metrics=cached.get("record_metrics"),
                        selection=cached.get("selection"),
                        threshold=cached.get("threshold", 0.5),
                        threshold_source=cached.get("threshold_source", "fixed"),
                        train_subjects=tuple(cached.get("train_subjects", ())),
                        test_subjects=tuple(cached.get("test_subjects", ())),
                        model_summary=cached.get("model_summary", {}),
                        audit=cached.get("audit", {}),
                        input_shape=tuple(cached.get("input_shape", ())),
                    )
                )
                fold_records.append(cached["fold_record"])
                continue

            result = run_fold(
                data, config, model_name, split, device=device, checkpoints=checkpoints
            )
            results[model_name].append(result)

            record = {
                "model": model_name,
                "repeat": split.repeat,
                "fold": split.fold,
                "split": split.describe(labels),
                "n_train_subjects": len(result.train_subjects),
                "n_test_subjects": len(result.test_subjects),
                "train_subjects": list(result.train_subjects),
                "test_subjects": list(result.test_subjects),
                "threshold": result.threshold,
                "threshold_source": result.threshold_source,
                "input_shape": list(result.input_shape),
                "subject_metrics": result.subject_metrics,
                "record_metrics": result.record_metrics,
                "selection": _strip_arrays(result.selection),
                "audit_passed": result.audit.get("all_passed"),
                "audit_failures": result.audit.get("failures"),
                "training_diagnostics": result.model_summary.get("training_diagnostics"),
            }
            fold_records.append(record)

            write_json(
                output_dir / "folds" / f"{model_name}__r{split.repeat}__f{split.fold}.json",
                {
                    **record,
                    "subject_predictions": result.subject_predictions.to_dict("records"),
                    "audit": result.audit,
                    "model_summary": result.model_summary,
                },
            )
            checkpoints.save(
                model_name, split.repeat, split.fold,
                {
                    "fold_record": record,
                    "subject_predictions": result.subject_predictions.to_dict("records"),
                    "subject_metrics": result.subject_metrics,
                    "record_metrics": result.record_metrics,
                    "selection": _strip_arrays(result.selection),
                    "threshold": result.threshold,
                    "threshold_source": result.threshold_source,
                    "train_subjects": list(result.train_subjects),
                    "test_subjects": list(result.test_subjects),
                    "model_summary": result.model_summary,
                    "audit": result.audit,
                    "input_shape": list(result.input_shape),
                },
            )

    models_block = {
        model: aggregate_model(model, folds, seed=config.seed)
        for model, folds in results.items() if folds
    }

    predictions = []
    for model, folds in results.items():
        for fold in folds:
            frame = fold.subject_predictions.copy()
            frame.insert(0, "model", model)
            frame.insert(1, "repeat", fold.repeat)
            frame.insert(2, "fold", fold.fold)
            predictions.append(frame)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(
            output_dir / "subject_predictions_hashed.csv", index=False
        )

    # A model whose early stopping restored the epoch-0 weights is untrained; its
    # metrics must never be read as a performance result.
    degenerate = sorted({
        record["model"] for record in fold_records
        if (record.get("training_diagnostics") or {}).get("degenerate_training")
    })

    return {
        "experiment": config.experiment,
        "reproduction_class": (
            "reported-method reconstruction"
            if config.experiment == "paper_reported_reconstruction" else "extension"
        ),
        "degenerate_training_models": degenerate,
        "degenerate_training_note": (
            "이 모델들은 early stopping이 epoch 0 가중치를 복원했다. 사실상 학습되지 "
            "않은 상태이므로 성능 결과로 인용하지 않는다."
        ) if degenerate else None,
        "config": config.raw,
        "config_path": str(config.path) if config.path else None,
        "seed": config.seed,
        "device": device,
        "data_notes": data.notes,
        "dataset_audit": dataset_audit.summary(),
        "n_splits": len(splits),
        "splits": [s.describe(labels) for s in splits],
        "models": models_block,
        "folds": fold_records,
        "all_audits_passed": all(
            r.get("audit_passed", True) for r in fold_records
        ) and dataset_audit.passed,
    }


def _strip_arrays(selection: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop the raw inner-CV prediction arrays before serialising."""
    if not selection:
        return selection
    return {
        k: v for k, v in selection.items()
        if not k.startswith("_") and not isinstance(v, np.ndarray)
    }
