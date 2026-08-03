"""Orchestration for the four experiments.

Each experiment is a different answer to "which rows may the model see", and the
shape of this file follows that: every runner starts by dividing something
(days or subjects), and only then calls the sequence builder on each side
separately.  There is no code path in which windows exist before a split does,
except the explicitly-flagged literal variant.

Runners
-------
``run_paper_temporal``        experiment A  -- estimand A, paper's final week
``run_paper_literal``         experiment A' -- estimand A + leakage, diagnostic only
``run_strict_temporal``       experiment B1 -- estimand A, embargoed
``run_fixed_subject``         experiment B2 -- estimand B, hyperparameters fixed
``run_nested_subject``        experiment C  -- estimand B, selection inside inner CV
"""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .audit import leakage
from .data import schema
from .data.loader import LifelogData
from .evaluation import metrics
from .models import registry
from .preprocessing.scaler import SequenceScaler
from .sampling.undersample import undersample
from .sequences.builder import SequenceSet, build_sequences, build_sequences_literal
from .splits import group as group_splits
from .splits import temporal as temporal_splits
from .utils.config import Config
from .utils.io import CheckpointStore, hash_sequence_id, hash_subject, write_json
from .utils.seeding import fold_seed, seed_everything


# --- shared helpers -----------------------------------------------------------

def _sampling_options(config: Config) -> dict[str, Any]:
    return {
        "strategy": str(config.get("sampling.strategy", "random_sequence")),
        "target_ratio": float(config.get("sampling.target_ratio", 1.0)),
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("numpy", "pandas", "scikit-learn", "torch", "xgboost", "h2o"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _run_signature(data: LifelogData, config: Config, *, device: str) -> str:
    """Bind resumable checkpoints to code, resolved config and loaded data."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(config.raw, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
    )
    digest.update(
        json.dumps(
            {"device": device, "dependencies": _dependency_versions()},
            sort_keys=True,
        ).encode("utf-8")
    )
    columns = [
        schema.SUBJECT_ID,
        schema.DATE_COL,
        schema.RAW_ROW_ID,
        schema.LABEL_COL,
        *data.feature_columns,
    ]
    row_hashes = pd.util.hash_pandas_object(data.daily[columns], index=True).to_numpy()
    digest.update(row_hashes.tobytes())
    source_root = Path(__file__).resolve().parent
    for source in sorted(source_root.rglob("*.py")):
        digest.update(str(source.relative_to(source_root)).encode("utf-8"))
        digest.update(source.read_bytes())
    runner_source = source_root.parent / "run.py"
    if runner_source.is_file():
        digest.update(b"run.py")
        digest.update(runner_source.read_bytes())
    return digest.hexdigest()


def _lstm_defaults(config: Config) -> dict[str, Any]:
    return {
        "max_epochs": int(config.get("lstm.max_epochs", 100)),
        "patience": int(config.get("lstm.patience", 10)),
        "early_stopping": bool(config.get("lstm.early_stopping", True)),
        "recurrent_dropout": float(config.get("lstm.recurrent_dropout", 0.0)),
        "threshold": float(config.get("threshold.value", 0.5)),
    }


def _prepare(
    train: SequenceSet,
    evaluation: SequenceSet,
    config: Config,
    *,
    validation: SequenceSet | None,
    model_name: str,
    seed: int,
) -> tuple[SequenceSet, SequenceSet, SequenceSet | None, SequenceScaler, dict[str, Any]]:
    """Undersample the training split, then fit the scaler on it and nothing else."""
    sampled, sampling_report = undersample(train, seed=seed, **_sampling_options(config))
    sampling_report["split_applied_to"] = "train"

    method = "standard" if registry.needs_scaling(model_name) else "none"
    scaler = SequenceScaler(method=method)
    to_scale = [evaluation] + ([validation] if validation is not None else [])
    scaled = scaler.fit_transform_pair(sampled, *to_scale)
    scaled_train, scaled_eval = scaled[0], scaled[1]
    scaled_validation = scaled[2] if validation is not None else None
    return scaled_train, scaled_eval, scaled_validation, scaler, sampling_report


def _score_block(
    sequences: SequenceSet,
    scores: np.ndarray,
    *,
    threshold: float,
    seed: int,
    include_subject: bool = True,
) -> dict[str, Any]:
    """Sequence-level and subject-level metrics for one evaluation set."""
    block: dict[str, Any] = {
        "sequence_level": metrics.binary_metrics(sequences.y, scores, threshold=threshold),
        "precision_at_k": metrics.precision_at_k(sequences.y, scores, k=100),
    }
    if include_subject and len(sequences):
        subjects = sequences.subjects
        block["subject_level"] = metrics.subject_level_metrics(
            subjects, sequences.y, scores, method="mean", threshold=threshold
        )
        block["subject_level_sensitivity_analysis"] = metrics.all_aggregation_metrics(
            subjects, sequences.y, scores, threshold=threshold
        )
        block["subject_precision_at_k"] = metrics.subject_precision_at_k(
            subjects, sequences.y, scores, k=100, method="mean", threshold=threshold
        )
        block["subject_bootstrap_ci"] = metrics.bootstrap_ci(
            subjects,
            sequences.y,
            scores,
            seed=seed,
            aggregation="mean",
            threshold=threshold,
        )
    return block


def _predictions_frame(sequences: SequenceSet, scores: np.ndarray, **extra: Any) -> pd.DataFrame:
    """Hashed, per-sequence predictions -- the artifact every claim must rest on."""
    frame = pd.DataFrame(
        {
            "subject_hash": [hash_subject(s) for s in sequences.subjects],
            "sequence_hash": [
                hash_sequence_id(value) for value in sequences.provenance["sequence_id"]
            ],
            "start_date": sequences.provenance["start_date"],
            "end_date": sequences.provenance["end_date"],
            "sequence_length": sequences.sequence_length,
            "split_name": sequences.provenance["split_name"],
            "outer_fold": sequences.provenance["outer_fold"],
            "inner_fold": sequences.provenance["inner_fold"],
            "y_true": sequences.y,
            "y_score": scores,
        }
    )
    for key, value in extra.items():
        frame[key] = value
    _assert_predictions_deidentified(frame)
    return frame


def _assert_predictions_deidentified(frame: pd.DataFrame) -> None:
    """Fail before writing if an output table still carries an internal identifier."""
    forbidden_columns = {"subject_id", "sequence_id", "email", "sample_email"}
    leaked_columns = forbidden_columns & {str(column).lower() for column in frame.columns}
    if leaked_columns:
        raise ValueError(
            f"prediction artifact contains raw-identifier columns: {sorted(leaked_columns)}"
        )

    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].astype(str)
        if values.str.contains(r"[^\s@]+@[^\s@]+", regex=True, na=False).any():
            raise ValueError(
                f"prediction artifact column {column!r} contains an email-like identifier"
            )


def _assert_checkpoint_metrics(block: dict[str, Any], frame: pd.DataFrame) -> None:
    """Round-trip checkpoint scores so a valid CSV hash cannot mask a stale block."""
    expected = block.get("sequence_level") or {}
    threshold = float(expected.get("threshold", block.get("threshold", 0.5)))
    actual = metrics.binary_metrics(
        frame["y_true"].to_numpy(), frame["y_score"].to_numpy(), threshold=threshold
    )
    for name in (
        "n", "n_positive", "n_negative", "roc_auc", "pr_auc", "balanced_accuracy",
        "sensitivity", "specificity", "f1", "brier", "accuracy", "precision",
    ):
        left, right = expected.get(name), actual.get(name)
        if left is None and (
            right is None
            or isinstance(right, (int, float)) and not np.isfinite(float(right))
        ):
            continue
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if np.isclose(float(left), float(right), rtol=1e-10, atol=1e-12, equal_nan=True):
                continue
        elif left == right:
            continue
        raise ValueError(
            f"checkpoint metric {name!r} does not round-trip from predictions: "
            f"stored={left}, recomputed={right}"
        )


def _method_fidelity(
    model_name: str, fit_meta: dict[str, Any], config: Config
) -> dict[str, Any]:
    """Record method matches separately from numerical similarity."""
    if model_name == "lstm":
        training = fit_meta.get("training") or {}
        params = fit_meta.get("params") or {}
        applied = bool(params.get("early_stopping") and training.get("used_validation"))
        return {
            "paper_backend": "not_reported",
            "actual_backend": fit_meta.get("backend"),
            "paper_reports_early_stopping": True,
            "early_stopping_requested": bool(params.get("early_stopping")),
            "early_stopping_applied": applied,
            "early_stopping_matches_reported_method": applied,
            "paper_validation_source_reported": False,
            "actual_validation_source": fit_meta.get("validation_split_name"),
            "actual_validation_days": int(config.get("split.validation_days", 0)),
            "validation_assumption_id": "A-06",
            "note": (
                "논문은 early-stopping validation의 출처를 보고하지 않았다. "
                "수치 유사성과 방법 충실도를 별도로 판정한다."
            ),
        }

    backend = fit_meta.get("backend")
    version = fit_meta.get("h2o_version")
    return {
        "paper_backend": "h2o_automl_3.46.0.1",
        "actual_backend": backend,
        "actual_h2o_version": version,
        "backend_matches_reported_method": bool(
            backend == "h2o" and version == "3.46.0.1"
        ),
        "hyperparameters_reported_by_paper": False,
        "note": (
            "논문은 baseline 최종 hyperparameter와 window representation을 보고하지 않았다."
        ),
    }


def _fit_and_score(
    model_name: str,
    params: dict[str, Any],
    train: SequenceSet,
    evaluation: SequenceSet,
    config: Config,
    *,
    validation: SequenceSet | None,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    representation = str(config.get("models.representation", "flatten"))
    backend = config.baseline_backend_for(model_name)
    model, fit_meta = registry.fit(
        model_name,
        train,
        params,
        validation=validation,
        representation=representation,
        seed=seed,
        device=device,
        class_weight=bool(config.get("sampling.class_weight", False)),
        lstm_defaults=_lstm_defaults(config),
        baseline_backend=backend,
        h2o_config=config.get("models.h2o", {}) or {},
    )
    scores = registry.predict(
        model, model_name, evaluation, representation=representation,
        backend=fit_meta.get("backend", backend),
    )
    return scores, fit_meta


# --- experiment A: the paper's temporal split ---------------------------------

def run_paper_temporal(
    data: LifelogData,
    config: Config,
    *,
    output_dir: Path,
    device: str = "cpu",
    strict: bool = False,
    resume: bool = False,
    only_length: int | None = None,
) -> dict[str, Any]:
    """Experiments A and B1: hold out each subject's final week.

    ``strict=True`` adds the L-1 day embargo and a training-side validation
    period, which is the only difference between B1 and A.
    """
    seed_everything(config.seed)
    checkpoints = CheckpointStore(output_dir / "checkpoints")
    results: dict[str, Any] = {}
    dataset_audit = leakage.audit_dataset(data)
    dataset_audit.raise_if_failed()
    audits: list[dict[str, Any]] = [dataset_audit.summary()]
    run_signature = _run_signature(data, config, device=device)
    predictions: list[pd.DataFrame] = []
    lengths = [only_length] if only_length else list(config.sequence_lengths)

    for length in lengths:
        embargo = (length - 1) if strict else int(config.get("split.embargo_days", 0))
        split = temporal_splits.final_week_split(
            data.daily,
            final_week_mode=str(config.get("split.final_week_mode", "calendar_days")),
            final_week_length=int(config.get("split.final_week_length", 7)),
            embargo_days=embargo,
            validation_days=int(config.get("split.validation_days", 0)),
            name=config.experiment,
        )
        temporal_splits.assert_no_shared_days(split)
        split_audit = leakage.audit_temporal_split(split, sequence_length=length)
        split_audit.raise_if_failed()
        audits.append(split_audit.summary())

        stride = int(config.get("sequence.stride", 1))
        require_consecutive = bool(config.get("sequence.require_consecutive", True))
        train = build_sequences(
            split.train_days, data.feature_columns, sequence_length=length,
            stride=stride, split_name="train", require_consecutive=require_consecutive,
        )
        test = build_sequences(
            split.test_days, data.feature_columns, sequence_length=length,
            stride=stride, split_name="test", require_consecutive=require_consecutive,
        )
        validation = None
        if split.validation_days is not None and len(split.validation_days):
            validation = build_sequences(
                split.validation_days, data.feature_columns, sequence_length=length,
                stride=stride, split_name="validation",
                require_consecutive=require_consecutive,
            )

        for model_name in config.models:
            key = f"{model_name}_L{length}"
            if resume and checkpoints.is_complete(model_name, length, 0, 0):
                loaded = checkpoints.load(model_name, length, 0, 0)
                if loaded.get("checkpoint_signature") != run_signature:
                    raise ValueError(
                        f"checkpoint {key} was created from different code/config/data"
                    )
                loaded_audit = loaded.get("leakage_audit")
                if not loaded_audit:
                    raise ValueError(
                        f"checkpoint {key} has no leakage_audit; refusing an incomplete resume"
                    )
                results[key] = loaded
                audits.append(loaded_audit)
                resumed_frame = checkpoints.load_predictions(
                    model_name,
                    length,
                    0,
                    0,
                    expected_metadata=loaded["checkpoint_predictions"],
                )
                _assert_checkpoint_metrics(loaded, resumed_frame)
                predictions.append(resumed_frame)
                continue

            seed = fold_seed(config.seed, length, 0)
            scaled_train, scaled_test, scaled_validation, scaler, sampling = _prepare(
                train, test, config, validation=validation, model_name=model_name, seed=seed
            )
            hyperparameter_source = (
                "paper_reported"
                if model_name == "lstm"
                else "train_internal_cv"
                if config.baseline_backend_for(model_name) == "h2o"
                else "config_fixed_unreported"
            )
            early_stopping_source = (
                "validation_period"
                if model_name == "lstm"
                and bool(config.get("lstm.early_stopping", True))
                and scaled_validation is not None
                else "not_applicable" if model_name != "lstm" else "none"
            )
            audit = leakage.audit_sequence_split(
                scaled_train, scaled_test,
                context=f"{config.experiment}/{key}",
                estimand="A",
                validation=scaled_validation,
                scaler=scaler,
                scaler_fit_source=scaled_train,
                sampling_report=sampling,
                sequence_length_source="config_fixed",
                hyperparameter_source=hyperparameter_source,
                early_stopping_source=early_stopping_source,
                expect_subject_overlap=True,
            )
            audit.raise_if_failed()
            audits.append(audit.summary())

            params = registry.search_space(model_name, enabled=False)[0]
            scores, fit_meta = _fit_and_score(
                model_name, params, scaled_train, scaled_test, config,
                validation=scaled_validation, seed=seed, device=device,
            )
            threshold = float(config.get("threshold.value", 0.5))
            block = {
                "model": model_name,
                "sequence_length": length,
                "estimand": "A",
                "estimand_note": (
                    "이미 학습된 피험자의 미래 기간 분류 성능이다. 신규 피험자 "
                    "일반화 성능이 아니다."
                ),
                "split": split.describe(),
                "thin_subjects": temporal_splits.summarise_thin_subjects(split, length),
                "train": scaled_train.describe(),
                "test": scaled_test.describe(),
                "sampling": sampling,
                "scaler": scaler.describe(),
                "fit": fit_meta,
                "method_fidelity": _method_fidelity(model_name, fit_meta, config),
                "validation": (
                    scaled_validation.describe() if scaled_validation is not None else None
                ),
                "leakage_audit": audit.summary(),
                "checkpoint_signature": run_signature,
                **_score_block(scaled_test, scores, threshold=threshold, seed=seed),
            }
            results[key] = block
            frame = _predictions_frame(scaled_test, scores, model=model_name)
            checkpoints.save_predictions(model_name, length, 0, 0, frame)
            block["checkpoint_predictions"] = checkpoints.predictions_metadata(
                model_name, length, 0, 0, frame
            )
            checkpoints.save(model_name, length, 0, 0, block)
            predictions.append(frame)

    return _finalise(
        config,
        results,
        audits,
        predictions,
        output_dir,
        data=data,
        estimand="A",
        device=device,
        evaluated_lengths=lengths,
    )


def run_paper_literal(
    data: LifelogData,
    config: Config,
    *,
    output_dir: Path,
    device: str = "cpu",
    resume: bool = False,
    only_length: int | None = None,
) -> dict[str, Any]:
    """The literal reading of §4.2: window the whole record, then split.

    This produces numbers that are *not* performance estimates.  Everything it
    writes is tagged ``leakage_diagnostic``.
    """
    seed_everything(config.seed)
    results: dict[str, Any] = {}
    dataset_audit = leakage.audit_dataset(data)
    dataset_audit.raise_if_failed()
    audits: list[dict[str, Any]] = [dataset_audit.summary()]
    predictions: list[pd.DataFrame] = []
    lengths = [only_length] if only_length else list(config.sequence_lengths)

    cuts = temporal_splits.first_test_dates(
        data.daily,
        final_week_mode=str(config.get("split.final_week_mode", "calendar_days")),
        final_week_length=int(config.get("split.final_week_length", 7)),
    )

    for length in lengths:
        train, test, literal_report = build_sequences_literal(
            data.daily,
            data.feature_columns,
            sequence_length=length,
            stride=int(config.get("sequence.stride", 1)),
            test_start_by_subject=cuts,
            require_consecutive=bool(config.get("sequence.require_consecutive", False)),
            leakage_diagnostic_only=True,
        )
        audits.append(literal_report)

        for model_name in config.models:
            key = f"{model_name}_L{length}"
            seed = fold_seed(config.seed, length, 0)
            scaled_train, scaled_test, _, scaler, sampling = _prepare(
                train, test, config, validation=None, model_name=model_name, seed=seed
            )
            # Boundary crossing is the point of this arm, so those checks are
            # downgraded to warnings -- but everything else still fails closed.
            hyperparameter_source = (
                "paper_reported" if model_name == "lstm" else "config_fixed_unreported"
            )
            audit = leakage.audit_sequence_split(
                scaled_train, scaled_test,
                context=f"{config.experiment}/{key}",
                estimand="A",
                scaler=scaler,
                scaler_fit_source=scaled_train,
                sampling_report=sampling,
                sequence_length_source="config_fixed",
                hyperparameter_source=hyperparameter_source,
                early_stopping_source="not_applicable" if model_name != "lstm" else "none",
                expect_subject_overlap=True,
                allow_boundary_crossing=True,
                allow_literal_overlap_diagnostic=True,
            )
            audit.raise_if_failed()
            audits.append(audit.summary())

            params = registry.search_space(model_name, enabled=False)[0]
            scores, fit_meta = _fit_and_score(
                model_name, params, scaled_train, scaled_test, config,
                validation=None, seed=seed, device=device,
            )
            threshold = float(config.get("threshold.value", 0.5))
            results[key] = {
                "model": model_name,
                "sequence_length": length,
                "estimand": "A_with_leakage",
                "result_kind": "leakage_diagnostic",
                "estimand_note": (
                    "경계를 가로지르는 윈도우 때문에 test 날짜가 train에 포함된다. "
                    "이 수치는 누수 크기 진단이며 성능 주장에 사용할 수 없다."
                ),
                "literal_split_report": literal_report,
                "train": scaled_train.describe(),
                "test": scaled_test.describe(),
                "sampling": sampling,
                "fit": fit_meta,
                "method_fidelity": _method_fidelity(model_name, fit_meta, config),
                "leakage_audit": audit.summary(),
                **_score_block(scaled_test, scores, threshold=threshold, seed=seed),
            }
            predictions.append(_predictions_frame(scaled_test, scores, model=model_name))

    report = _finalise(
        config,
        results,
        audits,
        predictions,
        output_dir,
        data=data,
        estimand="A",
        device=device,
        evaluated_lengths=lengths,
    )
    report["result_kind"] = "leakage_diagnostic"
    return report


# --- experiments B2 and C: subject-independent --------------------------------

def run_fixed_subject(
    data: LifelogData,
    config: Config,
    *,
    output_dir: Path,
    device: str = "cpu",
    resume: bool = False,
    only_fold: int | None = None,
    only_length: int | None = None,
) -> dict[str, Any]:
    """Experiment B2: the paper's configuration, evaluated on unseen subjects."""
    return _run_group_cv(
        data, config, output_dir=output_dir, device=device, resume=resume,
        only_fold=only_fold, only_length=only_length, nested=False,
    )


def run_nested_subject(
    data: LifelogData,
    config: Config,
    *,
    output_dir: Path,
    device: str = "cpu",
    resume: bool = False,
    only_fold: int | None = None,
    only_length: int | None = None,
) -> dict[str, Any]:
    """Experiment C: model, hyperparameters *and* sequence length chosen inside."""
    return _run_group_cv(
        data, config, output_dir=output_dir, device=device, resume=resume,
        only_fold=only_fold, only_length=only_length, nested=True,
    )


def _run_group_cv(
    data: LifelogData,
    config: Config,
    *,
    output_dir: Path,
    device: str,
    resume: bool,
    only_fold: int | None,
    only_length: int | None,
    nested: bool,
) -> dict[str, Any]:
    seed_everything(config.seed)
    checkpoints = CheckpointStore(output_dir / "checkpoints")
    dataset_audit = leakage.audit_dataset(data)
    dataset_audit.raise_if_failed()
    audits: list[dict[str, Any]] = [dataset_audit.summary()]
    run_signature = _run_signature(data, config, device=device)
    predictions: list[pd.DataFrame] = []

    outer = group_splits.stratified_group_splits(
        data.subjects,
        n_splits=int(config.get("split.outer_k", 5)),
        n_repeats=int(config.get("split.n_repeats", 1)),
        seed=config.seed,
        name=config.experiment,
        stratify_on=str(config.get("split.stratify_on", "label")),
    )
    if only_fold is not None:
        outer = [s for s in outer if s.fold == only_fold]

    labels = data.labels_by_subject()
    viability = group_splits.check_split_viability(outer, labels)
    if not viability["viable"]:
        raise ValueError(f"at least one outer fold lacks both classes: {viability}")

    lengths = [only_length] if only_length else list(config.sequence_lengths)
    stride = int(config.get("sequence.stride", 1))
    require_consecutive = bool(config.get("sequence.require_consecutive", True))
    threshold_policy = str(config.get("threshold.policy", "fixed"))
    fixed_threshold = float(config.get("threshold.value", 0.5))

    # Nested picks one length per fold; non-nested reports every length separately.
    length_grid = [None] if nested else lengths
    results: dict[str, Any] = {}

    for reported_length in length_grid:
        per_model_folds: dict[str, list[dict[str, Any]]] = {m: [] for m in config.models}
        per_model_scores: dict[str, list[pd.DataFrame]] = {m: [] for m in config.models}
        selections: list[dict[str, Any]] = []

        for split in outer:
            outer_train_days = group_splits.iter_days(data.daily, split.train_subjects)
            outer_test_days = group_splits.iter_days(data.daily, split.test_subjects)

            for model_name in config.models:
                if resume and checkpoints.is_complete(
                    model_name, reported_length or 0, split.repeat, split.fold
                ):
                    loaded = checkpoints.load(
                        model_name, reported_length or 0, split.repeat, split.fold
                    )
                    if loaded.get("checkpoint_signature") != run_signature:
                        raise ValueError(
                            "checkpoint was created from different code/config/data"
                        )
                    loaded_audit = loaded.get("leakage_audit")
                    if not loaded_audit:
                        raise ValueError(
                            "checkpoint has no leakage_audit; refusing an incomplete resume"
                        )
                    audits.append(loaded_audit)
                    if nested:
                        loaded_isolation = loaded.get("inner_isolation_audit")
                        if not loaded_isolation:
                            raise ValueError(
                                "nested checkpoint has no inner_isolation_audit; "
                                "refusing an incomplete resume"
                            )
                        audits.append(loaded_isolation)
                        selections.append(
                            {
                                "model": model_name,
                                "repeat": split.repeat,
                                "fold": split.fold,
                                "sequence_length": loaded["sequence_length"],
                            }
                        )
                    frame = checkpoints.load_predictions(
                        model_name,
                        reported_length or 0,
                        split.repeat,
                        split.fold,
                        expected_metadata=loaded["checkpoint_predictions"],
                    )
                    _assert_predictions_deidentified(frame)
                    _assert_checkpoint_metrics(loaded, frame)
                    per_model_folds[model_name].append(loaded)
                    per_model_scores[model_name].append(frame)
                    predictions.append(frame)
                    continue

                seed = fold_seed(config.seed, split.repeat, split.fold, reported_length or 0)

                isolation_summary = None
                if nested:
                    choice = _inner_selection(
                        data, config, split, model_name,
                        lengths=lengths, stride=stride, seed=seed, device=device,
                    )
                    length = choice["sequence_length"]
                    params = choice["params"]
                    inner_threshold = choice["threshold"]
                    inner_threshold_report = choice.get("threshold_report", {})
                    selections.append({"model": model_name, "repeat": split.repeat,
                                       "fold": split.fold, **choice})
                    isolation = leakage.audit_outer_test_isolation(
                        split.test_subjects,
                        inner_splits=choice["inner_splits"],
                        selection_scores_source="inner_cv",
                        context=f"nested/{model_name}/r{split.repeat}f{split.fold}",
                    )
                    isolation.raise_if_failed()
                    isolation_summary = isolation.summary()
                    audits.append(isolation_summary)
                else:
                    length = reported_length
                    params = registry.search_space(model_name, enabled=False)[0]
                    inner_threshold = fixed_threshold
                    inner_threshold_report = {"policy": "fixed", "fallback_to_fixed": False}

                train = build_sequences(
                    outer_train_days, data.feature_columns, sequence_length=length,
                    stride=stride, split_name="outer_train",
                    require_consecutive=require_consecutive,
                    outer_fold=split.fold,
                )
                test = build_sequences(
                    outer_test_days, data.feature_columns, sequence_length=length,
                    stride=stride, split_name="outer_test",
                    require_consecutive=require_consecutive,
                    outer_fold=split.fold,
                )
                if not len(train) or not len(test):
                    continue

                scaled_train, scaled_test, _, scaler, sampling = _prepare(
                    train, test, config, validation=None, model_name=model_name, seed=seed
                )
                hyperparameter_source = (
                    "inner_cv"
                    if nested
                    else "paper_reported"
                    if model_name == "lstm"
                    else "config_fixed_unreported"
                )
                early_stopping_source = (
                    "none" if model_name == "lstm" else "not_applicable"
                )
                audit = leakage.audit_sequence_split(
                    scaled_train, scaled_test,
                    context=f"{config.experiment}/{model_name}_L{length}/r{split.repeat}f{split.fold}",
                    estimand="B",
                    scaler=scaler,
                    scaler_fit_source=scaled_train,
                    sampling_report=sampling,
                    sequence_length_source="inner_cv" if nested else "config_fixed",
                    hyperparameter_source=hyperparameter_source,
                    early_stopping_source=early_stopping_source,
                    expect_subject_overlap=False,
                )
                audit.raise_if_failed()
                audits.append(audit.summary())

                scores, fit_meta = _fit_and_score(
                    model_name, params, scaled_train, scaled_test, config,
                    validation=None, seed=seed, device=device,
                )
                threshold = inner_threshold if threshold_policy != "fixed" else fixed_threshold
                fold_block = {
                    "model": model_name,
                    "repeat": split.repeat,
                    "fold": split.fold,
                    "sequence_length": length,
                    "n_test_subjects": len(split.test_subjects),
                    "params": params,
                    "threshold": threshold,
                    "threshold_source": (
                        "inner_cv_subject_level" if (nested and threshold_policy != "fixed")
                        else "paper_fixed_0.5"
                    ),
                    "threshold_report": inner_threshold_report,
                    "sampling": sampling,
                    "fit": fit_meta,
                    "method_fidelity": _method_fidelity(model_name, fit_meta, config),
                    "leakage_audit": audit.summary(),
                    "inner_isolation_audit": isolation_summary,
                    "checkpoint_signature": run_signature,
                    **_score_block(scaled_test, scores, threshold=threshold, seed=seed),
                }
                per_model_folds[model_name].append(fold_block)
                frame = _predictions_frame(scaled_test, scores, model=model_name)
                frame["repeat"] = split.repeat
                checkpoints.save_predictions(
                    model_name, reported_length or 0, split.repeat, split.fold, frame
                )
                fold_block["checkpoint_predictions"] = checkpoints.predictions_metadata(
                    model_name,
                    reported_length or 0,
                    split.repeat,
                    split.fold,
                    frame,
                )
                checkpoints.save(
                    model_name, reported_length or 0, split.repeat, split.fold, fold_block
                )
                per_model_scores[model_name].append(frame)
                predictions.append(frame)

        for model_name in config.models:
            folds = per_model_folds[model_name]
            if not folds:
                continue
            key = f"{model_name}_L{reported_length}" if reported_length else f"{model_name}_Lnested"
            results[key] = _pool_folds(
                model_name, folds, per_model_scores[model_name],
                threshold=fixed_threshold, seed=config.seed,
                selections=[s for s in selections if s["model"] == model_name],
                nested=nested,
            )
            # The nested arm reports one row per length actually chosen, so the
            # comparison table can still find lstm_L3 / L4 / L5 keys.
            if nested:
                for length in sorted({f["sequence_length"] for f in folds}):
                    subset = [f for f in folds if f["sequence_length"] == length]
                    results[f"{model_name}_L{length}"] = _pool_folds(
                        model_name, subset,
                        [df for df in per_model_scores[model_name]
                         if int(df["sequence_length"].iloc[0]) == length],
                        threshold=fixed_threshold, seed=config.seed,
                        selections=[s for s in selections
                                    if s["model"] == model_name
                                    and s["sequence_length"] == length],
                        nested=True,
                        partial=True,
                    )

    report = _finalise(
        config,
        results,
        audits,
        predictions,
        output_dir,
        data=data,
        estimand="B",
        device=device,
        evaluated_lengths=lengths,
    )
    report["outer_split_viability"] = viability
    return report


def _inner_selection(
    data: LifelogData,
    config: Config,
    outer: group_splits.SubjectSplit,
    model_name: str,
    *,
    lengths: list[int],
    stride: int,
    seed: int,
    device: str,
) -> dict[str, Any]:
    """Choose sequence length, hyperparameters and threshold on inner folds only.

    The outer test subjects are never passed to ``inner_splits``, so nothing
    scored here has seen them.
    """
    inner = group_splits.inner_splits(
        data.subjects, outer,
        n_splits=int(config.get("split.inner_k", 3)),
        seed=config.seed,
        stratify_on=str(config.get("split.stratify_on", "label")),
    )
    limit = config.get("tuning.max_candidates")
    candidates = registry.search_space(
        model_name, enabled=True, limit=int(limit) if limit else None, seed=seed
    )
    selection_metric = str(config.get("tuning.metric", "roc_auc"))
    threshold_policy = str(config.get("threshold.policy", "fixed"))

    best = {"score": -np.inf, "params": candidates[0], "sequence_length": lengths[0],
            "threshold": float(config.get("threshold.value", 0.5)),
            "threshold_report": {"policy": "fixed", "fallback_to_fixed": False}}
    trace: list[dict[str, Any]] = []

    for length in lengths:
        for params in candidates:
            fold_scores, pooled_true, pooled_score = [], [], []
            # The threshold has to be chosen on the unit it will be applied to.
            # Subject scores are means of sequence scores, so they are far less
            # saturated than the sequence scores; a threshold tuned on sequences
            # and applied to subjects sends every subject to one class.
            subject_true, subject_score = [], []
            for inner_split in inner:
                train_days = group_splits.iter_days(data.daily, inner_split.train_subjects)
                valid_days = group_splits.iter_days(data.daily, inner_split.test_subjects)
                train = build_sequences(
                    train_days, data.feature_columns, sequence_length=length,
                    stride=stride, split_name="inner_train",
                    require_consecutive=bool(
                        config.get("sequence.require_consecutive", True)
                    ),
                    outer_fold=outer.fold, inner_fold=inner_split.fold,
                )
                valid = build_sequences(
                    valid_days, data.feature_columns, sequence_length=length,
                    stride=stride, split_name="inner_valid",
                    require_consecutive=bool(
                        config.get("sequence.require_consecutive", True)
                    ),
                    outer_fold=outer.fold, inner_fold=inner_split.fold,
                )
                if not len(train) or not len(valid):
                    continue
                scaled_train, scaled_valid, _, _, _ = _prepare(
                    train, valid, config, validation=None,
                    model_name=model_name, seed=seed + inner_split.fold,
                )
                scores, _ = _fit_and_score(
                    model_name, params, scaled_train, scaled_valid, config,
                    validation=None, seed=seed + inner_split.fold, device=device,
                )
                subject_metrics = metrics.subject_level_metrics(
                    scaled_valid.subjects, scaled_valid.y, scores, method="mean"
                )
                fold_scores.append(subject_metrics.get(selection_metric, np.nan))
                pooled_true.append(scaled_valid.y)
                pooled_score.append(scores)

                aggregated = metrics.aggregate_to_subject(
                    scaled_valid.subjects, scaled_valid.y, scores, method="mean"
                )
                subject_true.append(aggregated["y_true"].to_numpy())
                subject_score.append(aggregated["y_score"].to_numpy())

            mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
            trace.append({"sequence_length": length, "params": params,
                          "inner_score": mean_score})
            if np.isfinite(mean_score) and mean_score > best["score"]:
                threshold = float(config.get("threshold.value", 0.5))
                report = {"policy": "fixed", "fallback_to_fixed": False,
                          "selected_on": "n/a"}
                if threshold_policy != "fixed" and subject_true:
                    threshold, report = metrics.choose_threshold_with_report(
                        np.concatenate(subject_true), np.concatenate(subject_score),
                        policy=threshold_policy, fixed=threshold,
                    )
                    report["selected_on"] = "inner_cv_subject_level"
                best = {"score": mean_score, "params": params,
                        "sequence_length": length, "threshold": threshold,
                        "threshold_report": report}

    return {**best, "inner_splits": inner, "n_candidates_evaluated": len(trace),
            "selection_metric": selection_metric, "trace": trace}


def _pool_folds(
    model_name: str,
    folds: list[dict[str, Any]],
    score_frames: list[pd.DataFrame],
    *,
    threshold: float,
    seed: int,
    selections: list[dict[str, Any]],
    nested: bool,
    partial: bool = False,
) -> dict[str, Any]:
    """Pool out-of-fold predictions into one subject-level estimate.

    The pooled threshold is the fixed 0.5, **not** the per-fold inner-CV choice:
    each fold selected its own operating point, and averaging different operating
    points into one confusion matrix would describe no classifier that exists.
    The per-fold thresholds stay in ``per_fold`` and are summarised here, so the
    difference is visible rather than silent.  ROC-AUC and PR-AUC are
    threshold-free and unaffected.
    """
    fold_thresholds = [f.get("threshold") for f in folds if f.get("threshold") is not None]
    degenerate_folds = [
        {"repeat": f.get("repeat"), "fold": f.get("fold"), "threshold": f.get("threshold")}
        for f in folds
        if (f.get("subject_level") or {}).get("specificity") == 0.0
        or (f.get("subject_level") or {}).get("sensitivity") == 0.0
    ]
    block: dict[str, Any] = {
        "model": model_name,
        "estimand": "B",
        "estimand_note": (
            "학습에서 한 번도 관찰되지 않은 신규 피험자에 대한 성능이다. "
            "논문의 동일 피험자 시간분할 값과 같은 의미로 비교하지 않는다."
        ),
        "n_folds": len(folds),
        "per_fold": folds,
        "sequence_length": sorted({f["sequence_length"] for f in folds}),
        "threshold_source": "paper_fixed_0.5",
        "threshold_note": (
            "pooled 지표는 논문의 고정 0.5를 쓴다. fold별로 선택된 threshold는 "
            "per_fold에 남아 있으며 서로 다르므로 하나의 confusion matrix로 "
            "합치지 않는다. ROC-AUC와 PR-AUC는 threshold와 무관하다."
        ),
        "per_fold_thresholds": fold_thresholds,
        "n_degenerate_fold_operating_points": len(degenerate_folds),
        "degenerate_fold_operating_points": degenerate_folds,
        "fold_variability": {
            unit: metrics.fold_variability(
                [f[unit] for f in folds if unit in f], metric="roc_auc"
            )
            for unit in ("sequence_level", "subject_level")
        },
        "is_partial_subset": partial,
    }
    if degenerate_folds:
        block["warnings"] = [
            f"{len(degenerate_folds)}개 fold가 모든 피험자를 한 클래스로 예측하는 "
            "operating point를 사용했다. 해당 fold의 sensitivity/specificity/F1은 "
            "해석하지 않는다."
        ]
    if nested:
        block["selection"] = {
            "n_selections": len(selections),
            "chosen_sequence_lengths": (
                pd.Series([s["sequence_length"] for s in selections]).value_counts().to_dict()
                if selections else {}
            ),
            "note": "시퀀스 길이는 inner CV에서만 선택했다. outer test는 사용하지 않았다.",
        }

    if score_frames:
        pooled = pd.concat(score_frames, ignore_index=True)
        block["sequence_level"] = metrics.binary_metrics(
            pooled["y_true"].to_numpy(), pooled["y_score"].to_numpy(), threshold=threshold
        )
        block["subject_level"] = metrics.subject_level_metrics(
            pooled["subject_hash"], pooled["y_true"], pooled["y_score"],
            method="mean", threshold=threshold,
        )
        block["subject_level_sensitivity_analysis"] = metrics.all_aggregation_metrics(
            pooled["subject_hash"], pooled["y_true"], pooled["y_score"], threshold=threshold
        )
        block["subject_bootstrap_ci"] = metrics.bootstrap_ci(
            pooled["subject_hash"], pooled["y_true"], pooled["y_score"], seed=seed
        )
        block["calibration"] = metrics.calibration_curve_points(
            pooled["y_true"].to_numpy(), pooled["y_score"].to_numpy()
        )
        block["precision_at_k"] = metrics.precision_at_k(
            pooled["y_true"].to_numpy(), pooled["y_score"].to_numpy(), k=100
        )
        block["subject_precision_at_k"] = metrics.subject_precision_at_k(
            pooled["subject_hash"],
            pooled["y_true"].to_numpy(),
            pooled["y_score"].to_numpy(),
            k=100,
            method="mean",
            threshold=threshold,
        )
    return block


# --- shared tail --------------------------------------------------------------

def _finalise(
    config: Config,
    results: dict[str, Any],
    audits: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    output_dir: Path,
    *,
    data: LifelogData,
    estimand: str,
    device: str,
    evaluated_lengths: list[int],
) -> dict[str, Any]:
    if not results:
        raise ValueError("no result blocks were produced; refusing to mark the run complete")
    if not audits:
        raise ValueError("no audit records were produced; refusing to mark the run complete")
    if not predictions:
        raise ValueError(
            "result blocks exist without prediction records; refusing an unverifiable report"
        )

    frame = pd.concat(predictions, ignore_index=True)
    _assert_predictions_deidentified(frame)
    key_columns = ["model", "sequence_hash"]
    if "repeat" in frame.columns:
        key_columns.append("repeat")
    if frame.duplicated(key_columns).any():
        n_duplicate = int(frame.duplicated(key_columns).sum())
        raise ValueError(
            f"prediction artifact has {n_duplicate} duplicate rows by {key_columns}"
        )

    path = output_dir / "predictions_hashed.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()

    prediction_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    all_passed = bool(audits) and all(a.get("all_passed", False) for a in audits)
    warning_records = [
        {"context": audit.get("context"), **check}
        for audit in audits
        for check in audit.get("checks", [])
        if not check.get("passed", False) and check.get("severity") == "warning"
    ]
    if not all_passed and config.experiment != "paper_literal_variant":
        raise ValueError(
            "one or more audit records did not pass; refusing a formal performance report"
        )
    write_json(output_dir / "audit" / "audit_log.json", {"audits": audits})

    executed_lengths: set[int] = set()
    for block in results.values():
        value = block.get("sequence_length")
        values = value if isinstance(value, list) else [value]
        executed_lengths.update(int(item) for item in values if item is not None)
    configured_lengths = set(config.sequence_lengths)
    evaluated_length_set = {int(length) for length in evaluated_lengths}
    configured_models = set(config.models)
    executed_models = {
        str(block.get("model"))
        for block in results.values()
        if block.get("model") is not None
    }
    expected_outer = int(config.get("split.outer_k", 1)) * int(
        config.get("split.n_repeats", 1)
    )
    if config.experiment == "nested_subject_independent":
        expected_result_keys = {f"{model}_Lnested" for model in configured_models}
    else:
        expected_result_keys = {
            f"{model}_L{length}"
            for model in configured_models
            for length in evaluated_length_set
        }
    missing_result_keys = sorted(expected_result_keys - set(results))
    formal_coverage = {
        key: int(results[key].get("n_folds", expected_outer))
        for key in sorted(expected_result_keys & set(results))
    }
    observed_outer = min(formal_coverage.values(), default=0)
    is_partial_run = (
        evaluated_length_set != configured_lengths
        or executed_models != configured_models
        or bool(missing_result_keys)
        or observed_outer < expected_outer
    )

    return {
        "experiment": config.experiment,
        "config_path": str(config.path),
        "resolved_config": config.raw,
        "data": {**data.describe(), "notes": data.notes},
        "estimand": estimand,
        "seed": config.seed,
        "run_signature": _run_signature(data, config, device=device),
        "runtime_signature_context": {
            "device": device,
            "dependencies": _dependency_versions(),
        },
        "models": list(config.models),
        "sequence_lengths": sorted(executed_lengths),
        "run_scope": {
            "configured_models": sorted(configured_models),
            "executed_models": sorted(executed_models),
            "configured_sequence_lengths": sorted(configured_lengths),
            "evaluated_sequence_lengths": sorted(evaluated_length_set),
            "reported_sequence_lengths": sorted(executed_lengths),
            "expected_outer_splits": expected_outer,
            "minimum_observed_outer_splits": observed_outer,
            "outer_coverage_by_result": formal_coverage,
            "expected_result_keys": sorted(expected_result_keys),
            "missing_result_keys": missing_result_keys,
            "is_partial_run": bool(is_partial_run),
        },
        "results": results,
        "all_audits_passed": bool(all_passed),
        "n_audit_records": len(audits),
        "n_audit_warnings": len(warning_records),
        "audit_warnings": warning_records,
        "artifacts": {
            "predictions": {
                "path": str(path),
                "n_rows": int(len(frame)),
                "sha256": prediction_sha256,
                "identifier_columns": ["subject_hash", "sequence_hash"],
            }
        },
    }


RUNNERS: dict[str, Callable[..., dict[str, Any]]] = {
    "paper_temporal_reconstruction": lambda data, config, **kw: run_paper_temporal(
        data, config, strict=False, **kw
    ),
    "paper_literal_variant": run_paper_literal,
    "strict_same_subject_temporal": lambda data, config, **kw: run_paper_temporal(
        data, config, strict=True, **kw
    ),
    "fixed_subject_independent": run_fixed_subject,
    "nested_subject_independent": run_nested_subject,
}


def run_experiment(
    data: LifelogData,
    config: Config,
    *,
    output_dir: Path,
    device: str = "cpu",
    only_fold: int | None = None,
    only_length: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    runner = RUNNERS.get(config.experiment)
    if runner is None:
        raise ValueError(f"no runner for experiment {config.experiment!r}")
    kwargs: dict[str, Any] = {
        "output_dir": output_dir, "device": device, "resume": resume,
        "only_length": only_length,
    }
    if config.experiment in ("fixed_subject_independent", "nested_subject_independent"):
        kwargs["only_fold"] = only_fold
    if not config.uses_h2o:
        return runner(data, config, **kwargs)

    # H2O owns a JVM process.  Keep it alive across every model/fold in this run,
    # then release it even when fitting or report finalisation fails.
    from .models import h2o_backend

    try:
        return runner(data, config, **kwargs)
    finally:
        h2o_backend.shutdown()
