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
    zero_padding,
)
from .models import registry
from .splits import splitters
from .utils.io import CheckpointStore, sha256_file, write_json
from .utils.seeding import fold_seed, seed_everything

PRIMARY_METRIC = "roc_auc"
ARTIFACT_SCHEMA_VERSION = 2


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
    preprocessor_state: dict[str, Any] = field(default_factory=dict)
    representation_meta: dict[str, Any] = field(default_factory=dict)
    checkpoint_meta: dict[str, Any] = field(default_factory=dict)


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
    # Seed before construction, not only inside ``fit``.  Torch initialises the
    # module during fit/build, so otherwise paired inner-CV candidates receive
    # different ambient RNG states despite being passed the same seed.
    seed_everything(seed)
    model = registry.build_model(
        model_name, seed=seed, device=device, overrides=overrides, training=training
    )
    if registry.is_sequence_model(model_name):
        if model_name in ("lstm", "bilstm"):
            # pack_padded_sequence requires valid steps at the front.
            train_X, train_lengths = train_rep.left_aligned()
            model.fit(
                train_X, train_rep.y, lengths=train_lengths,
                padding_side="post", seed=seed,
            )
        else:
            # A flattened CNN is position-sensitive.  Preserve the configured
            # pre/post padding instead of silently converting it to post-padding.
            model.fit(
                train_rep.X, train_rep.y, lengths=train_rep.lengths,
                padding_side=train_rep.padding_side, seed=seed,
            )
        probabilities = _predict_fitted(model_name, model, test_rep)
    else:
        model.fit(train_rep.X, train_rep.y)
        probabilities = model.predict_proba(test_rep.X)
    return np.asarray(probabilities, dtype=np.float64), model


def _predict_fitted(model_name: str, model, representation) -> np.ndarray:
    """Score a fitted model using the same representation contract as training."""
    if model_name in ("lstm", "bilstm"):
        X, lengths = representation.left_aligned()
        return np.asarray(
            model.predict_proba(X, lengths=lengths, padding_side="post"),
            dtype=np.float64,
        )
    if model_name == "cnn1d":
        return np.asarray(
            model.predict_proba(
                representation.X,
                lengths=representation.lengths,
                padding_side=representation.padding_side,
            ),
            dtype=np.float64,
        )
    return np.asarray(model.predict_proba(representation.X), dtype=np.float64)


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
    for overrides in candidates:
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

            # Pair candidates on the same stochastic seed within an inner fold.
            # Adding the candidate index here would confound hyperparameters with
            # a different neural initialisation / bootstrap draw.
            inner_seed = fold_seed(seed, 0, inner.fold)
            probabilities, _ = _fit_predict(
                model_name, train_rep, test_rep,
                seed=inner_seed, device=device,
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
    data,
    config,
    model_name: str,
    split,
    *,
    device: str,
    checkpoints: CheckpointStore | None = None,
) -> FoldResult:
    """Train and evaluate one model on one split, with audits before fitting."""
    seed = fold_seed(config.seed, split.repeat, split.fold)
    seed_everything(seed)

    selection: dict[str, Any] | None = None
    # Architecture/hyperparameter overrides declared in the config, e.g. the
    # single-conv-block CNN the paper actually describes.  The nested experiment
    # replaces these with its inner-CV choice below.
    overrides: dict[str, Any] | None = dict(
        config.get(f"model_overrides.{model_name}", {}) or {}
    ) or None
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
    raw_test_X = test_rep.X.copy()

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

    checkpoint_meta: dict[str, Any] = {}
    if checkpoints is not None and config.get("checkpointing.save_models", False):
        suffix = ".pt" if registry.is_sequence_model(model_name) else ".joblib"
        model_path = checkpoints.model_path(
            model_name, split.repeat, split.fold, suffix
        )
        roundtrip_error: Exception | None = None
        same = False
        max_abs: float | None = None
        try:
            model.save(model_path)
            reloaded = registry.load_model(model_name, model_path, device="cpu")
            restored_preprocessor = FoldPreprocessor.from_state_record(
                preprocessor.state_record(),
                expected_feature_names=test_rep.feature_names,
            )
            roundtrip_rep = test_rep.subset(
                np.ones(test_rep.n_units, dtype=bool)
            )
            roundtrip_rep.X = restored_preprocessor.transform(raw_test_X)
            zero_padding(roundtrip_rep)
            roundtrip = _predict_fitted(model_name, reloaded, roundtrip_rep)
            same = bool(
                roundtrip.shape == probabilities.shape
                and np.isfinite(roundtrip).all()
                and np.allclose(roundtrip, probabilities, rtol=1e-5, atol=1e-7)
            )
            max_abs = (
                float(np.max(np.abs(roundtrip - probabilities)))
                if roundtrip.shape == probabilities.shape and len(roundtrip) else None
            )
        except Exception as error:
            # Do not put ``enforce`` inside the try block: its own LeakageError
            # would otherwise be caught and mislabeled as a save/load exception.
            roundtrip_error = error
        audit_log.enforce(
            "checkpoint_roundtrip",
            same and roundtrip_error is None,
            (
                "model checkpoint could not be saved and reloaded"
                if roundtrip_error is not None
                else "a freshly reloaded CPU checkpoint changed the evaluation scores"
            ),
            {
                "path": str(model_path),
                "max_abs_probability_difference": max_abs,
                "n_predictions": int(len(probabilities)),
                "error": None if roundtrip_error is None else str(roundtrip_error),
            },
        )
        checkpoint_meta = {
            "filename": model_path.name,
            "sha256": sha256_file(model_path),
        }

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
        preprocessor_state=preprocessor.state_record(),
        representation_meta={
            "kind": train_rep.kind,
            "train": dict(train_rep.meta),
            "test": dict(test_rep.meta),
            "feature_names": list(train_rep.feature_names),
        },
        checkpoint_meta=checkpoint_meta,
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
        frame["decision_threshold"] = float(fold.threshold)
        frame["prediction"] = (
            frame["probability"].to_numpy(dtype=float) >= float(fold.threshold)
        ).astype(int)
        frames.append(frame)
    oof = pd.concat(frames, ignore_index=True)
    duplicate_assignments = oof.duplicated(["repeat", "subject_id"], keep=False)
    if duplicate_assignments.any():
        examples = (
            oof.loc[duplicate_assignments, ["repeat", "subject_id"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "a subject was assigned to multiple test folds in one repeat; "
            f"examples={examples}"
        )

    # Average duplicate predictions when repeats put a subject in several test folds.
    pooled = (
        oof.groupby(["repeat", "subject_id"], as_index=False)
        .agg(
            probability=("probability", "mean"),
            label=("label", "first"),
            prediction=("prediction", "first"),
            decision_threshold=("decision_threshold", "first"),
        )
    )
    per_repeat: list[dict[str, Any]] = []
    for repeat, group in pooled.groupby("repeat"):
        repeat_thresholds = group["decision_threshold"].unique()
        score_kwargs = (
            {"threshold": float(repeat_thresholds[0])}
            if len(repeat_thresholds) == 1
            else {"predictions": group["prediction"]}
        )
        per_repeat.append(
            {
                "repeat": int(repeat),
                **M.compute_metrics(group["label"], group["probability"],
                                    **score_kwargs),
            }
        )

    overall = (
        pooled.groupby("subject_id", as_index=False)
        .agg(
            probability=("probability", "mean"),
            label=("label", "first"),
            decision_vote=("prediction", "mean"),
        )
    )
    thresholds = sorted({float(fold.threshold) for fold in folds})
    if len(thresholds) == 1:
        subject_level = M.compute_metrics(
            overall["label"], overall["probability"], threshold=thresholds[0]
        )
    else:
        subject_level = M.compute_metrics(
            overall["label"], overall["probability"],
            predictions=(overall["decision_vote"] >= 0.5).astype(int),
        )
    subject_level["evaluation_unit"] = "subject"

    fold_scores = [
        f.subject_metrics.get(PRIMARY_METRIC) for f in folds
        if f.subject_metrics.get(PRIMARY_METRIC) is not None
    ]
    fold_scores = [s for s in fold_scores if np.isfinite(s)]

    repeat_scores = [
        float(record[PRIMARY_METRIC])
        for record in per_repeat
        if record.get(PRIMARY_METRIC) is not None
        and np.isfinite(float(record[PRIMARY_METRIC]))
    ]

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
            float(np.std(repeat_scores)) if len(repeat_scores) > 1 else None
        ),
        "threshold": thresholds[0] if len(thresholds) == 1 else None,
        "thresholds_by_fold": [
            {
                "repeat": int(fold.repeat), "fold": int(fold.fold),
                "threshold": float(fold.threshold),
            }
            for fold in folds
        ],
        "threshold_source": (
            folds[0].threshold_source if len(thresholds) == 1
            else "fold_specific_inner_cv"
        ),
        "model_summary": folds[0].model_summary,
        "input_shape": list(folds[0].input_shape),
        "selections": [f.selection for f in folds if f.selection],
    }


def run_experiment(
    data, config, *, output_dir: Path, device: str = "cpu", only_fold: int | None = None,
    resume: bool = False, run_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Run every configured model over every split and write the artifacts."""
    if resume and not run_fingerprint:
        raise ValueError(
            "--resume requires a non-empty run_fingerprint; stale checkpoints "
            "must never be reused without code/config/data/device identity"
        )
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
            if resume and checkpoints.is_complete(
                model_name,
                split.repeat,
                split.fold,
                expected_schema_version=ARTIFACT_SCHEMA_VERSION,
                expected_run_fingerprint=run_fingerprint,
                require_model=bool(config.get("checkpointing.save_models", False)),
            ):
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
                        preprocessor_state=cached.get("preprocessor_state", {}),
                        representation_meta=cached.get("representation_meta", {}),
                        checkpoint_meta=cached.get("model_checkpoint", {}),
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
                "preprocessor_state": result.preprocessor_state,
                "representation_meta": result.representation_meta,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "model_checkpoint": result.checkpoint_meta or None,
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
                    "preprocessor_state": result.preprocessor_state,
                    "representation_meta": result.representation_meta,
                    "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                    "run_fingerprint": run_fingerprint,
                    "model_checkpoint": result.checkpoint_meta or None,
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

    # A model with no finite optimiser-driven parameter change is untrained; its
    # metrics must never be read as a performance result.
    degenerate = sorted({
        record["model"] for record in fold_records
        if (record.get("training_diagnostics") or {}).get("degenerate_training")
    })
    collapsed = sorted({
        model for model, block in models_block.items()
        if block.get("subject_level")
        and block["subject_level"].get("positive_rate_predicted") in (0.0, 1.0)
    })

    return {
        "experiment": config.experiment,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "reproduction_class": (
            "reported-method reconstruction"
            if config.experiment == "paper_reported_reconstruction" else "extension"
        ),
        "degenerate_training_models": degenerate,
        "degenerate_training_note": (
            "이 모델들은 유한한 optimizer update/parameter change를 남기지 못했다. "
            "사실상 학습되지 않은 상태이므로 성능 결과로 인용하지 않는다."
        ) if degenerate else None,
        "collapsed_classification_models": collapsed,
        "collapsed_classification_note": (
            "평가에 적용된 임계값에서 모든 피험자를 한 클래스로 예측했다. Accuracy가 "
            "다수 클래스 비율과 같아도 분류 성능 재현으로 보지 않는다."
        ) if collapsed else None,
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
            r.get("audit_passed") is True for r in fold_records
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
