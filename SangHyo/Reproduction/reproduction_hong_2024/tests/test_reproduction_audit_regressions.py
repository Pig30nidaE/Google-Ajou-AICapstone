"""Regression contracts added after auditing the 2026-08-03 reproduction.

These tests are intentionally small and do not fit a model.  They cover the
reporting defects that allowed a numerically close AUC to look like a complete
reproduction even though early stopping was inactive, P@100 was unattainable,
and the prediction artifact still contained raw email addresses.
"""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from run import _archive_stale_completion_markers, run_pipeline
from src.audit.leakage import audit_sequence_split, audit_temporal_split
from src.engine import _assert_predictions_deidentified, _finalise, _predictions_frame
from src.evaluation import metrics
from src.evaluation.compare import build_comparison, load_reports, render_comparison_markdown
from src.models.lstm import LSTMConfig, SequenceLSTM
from src.sequences.builder import build_sequences
from src.splits.temporal import final_week_split
from src.utils.config import Config, ConfigError, load_config, validate_config
from src.utils.io import CheckpointStore, write_json


def test_prediction_export_uses_only_hashed_identifiers(synthetic_data):
    split = final_week_split(synthetic_data.daily)
    sequences = build_sequences(
        split.test_days,
        synthetic_data.feature_columns,
        sequence_length=3,
        split_name="test",
    )
    raw_sequence_ids = set(sequences.provenance["sequence_id"].astype(str))

    frame = _predictions_frame(
        sequences,
        np.linspace(0.0, 1.0, len(sequences)),
        model="lstm",
    )

    assert "subject_id" not in frame
    assert "sequence_id" not in frame
    assert {"subject_hash", "sequence_hash"} <= set(frame)
    assert not (set(frame["sequence_hash"].astype(str)) & raw_sequence_ids)
    assert not frame.astype(str).apply(lambda column: column.str.contains("@").any()).any()


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"sequence_id": ["test|L5|nia+005@rowan.kr|2021-02-11"]}),
        pd.DataFrame({"comment": ["subject nia+005@rowan.kr"]}),
    ],
)
def test_prediction_privacy_guard_rejects_raw_identifiers(frame):
    with pytest.raises(ValueError, match="identifier|email-like"):
        _assert_predictions_deidentified(frame)


def test_json_writer_replaces_nonfinite_values_and_emits_strict_json(tmp_path):
    path = write_json(
        tmp_path / "report.json",
        {"history": {"val_loss": [np.nan, np.float32(np.inf), 0.25]}},
    )

    text = path.read_text(encoding="utf-8")

    def reject_constant(value: str):
        raise AssertionError(f"non-standard JSON constant: {value}")

    parsed = json.loads(text, parse_constant=reject_constant)
    assert parsed["history"]["val_loss"] == [None, None, 0.25]


def test_json_writer_redacts_embedded_email_identifiers(tmp_path):
    path = write_json(
        tmp_path / "failure.json",
        {"error": "bad sequence test|L5|nia+005@rowan.kr|2021-02-11"},
    )
    text = path.read_text(encoding="utf-8")

    assert "nia+005@rowan.kr" not in text
    assert "<subject_hash:" in text


def test_checkpoint_is_complete_only_with_prediction_sidecar(tmp_path):
    store = CheckpointStore(tmp_path / "checkpoints")
    payload = {"checkpoint_signature": "abc"}
    store.save("lstm", 5, 0, 0, payload)
    assert not store.is_complete("lstm", 5, 0, 0)

    frame = pd.DataFrame(
        {
            "model": ["lstm"],
            "subject_hash": ["subjecthash"],
            "sequence_hash": ["deadbeef"],
            "sequence_length": [5],
            "y_true": [1],
            "y_score": [0.9],
        }
    )
    store.save_predictions(
        "lstm",
        5,
        0,
        0,
        frame,
    )
    assert not store.is_complete("lstm", 5, 0, 0)

    payload["checkpoint_predictions"] = store.predictions_metadata(
        "lstm", 5, 0, 0, frame
    )
    store.save("lstm", 5, 0, 0, payload)
    assert store.is_complete("lstm", 5, 0, 0)

    store.predictions_path("lstm", 5, 0, 0).write_text(
        "model,subject_hash,sequence_hash,sequence_length,y_true,y_score\n"
        "lstm,subjecthash,tampered,5,1,0.9\n",
        encoding="utf-8",
    )
    assert not store.is_complete("lstm", 5, 0, 0)


def test_new_attempt_recoverably_invalidates_all_old_completion_markers(tmp_path):
    for marker in (
        "TRAINING_COMPLETE.json",
        "PARTIAL_RUN_COMPLETE.json",
        "DIAGNOSTIC_COMPLETE.json",
    ):
        write_json(tmp_path / marker, {"attempt_id": "old"})

    archived = _archive_stale_completion_markers(tmp_path, "new-attempt")

    assert len(archived) == 3
    assert not any((tmp_path / marker).exists() for marker in (
        "TRAINING_COMPLETE.json",
        "PARTIAL_RUN_COMPLETE.json",
        "DIAGNOSTIC_COMPLETE.json",
    ))
    assert all((tmp_path / relative).is_file() for relative in archived)


def test_preflight_failure_closes_attempt_and_cannot_inherit_old_completion(tmp_path):
    write_json(tmp_path / "TRAINING_COMPLETE.json", {"attempt_id": "old"})
    write_json(
        tmp_path / "LAUNCHER_STATUS.json",
        {"status": "complete", "attempt_id": "old"},
    )

    with pytest.raises(SystemExit, match="--config is required"):
        run_pipeline(argv=["--output-dir", str(tmp_path), "--skip-install"])

    status = json.loads(
        (tmp_path / "LAUNCHER_STATUS.json").read_text(encoding="utf-8")
    )
    assert status["status"] == "failed"
    assert status["error_type"] == "SystemExit"
    assert status["attempt_id"] != "old"
    assert not (tmp_path / "TRAINING_COMPLETE.json").exists()


def test_literal_overlap_allowance_does_not_suppress_unrelated_fatal_audits(
    synthetic_data,
):
    sequences = build_sequences(
        synthetic_data.daily,
        synthetic_data.feature_columns,
        sequence_length=3,
        split_name="literal",
    )
    audit = audit_sequence_split(
        sequences,
        sequences,
        context="literal-diagnostic",
        estimand="A",
        sampling_report={"split_applied_to": "test"},
        sequence_length_source="config_fixed",
        hyperparameter_source="config_fixed_unreported",
        early_stopping_source="none",
        expect_subject_overlap=True,
        allow_boundary_crossing=True,
        allow_literal_overlap_diagnostic=True,
    )

    warning_checks = {record["check"] for record in audit.warnings}
    fatal_checks = {record["check"] for record in audit.failures}
    assert {"no_raw_row_overlap", "no_shared_subject_dates"} <= warning_checks
    assert "undersampling_applied_to_train_only" in fatal_checks
    with pytest.raises(AssertionError, match="undersampling_applied_to_train_only"):
        audit.raise_if_failed()


def test_precision_at_100_records_attainability_and_effective_k():
    y_true = np.array([1] * 94 + [0] * 200)
    y_score = np.linspace(1.0, 0.0, len(y_true))
    report = metrics.precision_at_k(y_true, y_score, k=100)

    assert report["requested_k"] == 100
    assert report["k"] == 100
    assert report["n_positive_available"] == 94
    assert report["max_possible_precision_at_k"] == 0.94


def test_subject_precision_at_100_cannot_be_mistaken_for_sequence_precision():
    subjects = [f"subject-{i:03d}" for i in range(111)]
    y_true = np.array([1] * 36 + [0] * 75)
    y_score = np.linspace(1.0, 0.0, len(y_true))
    report = metrics.subject_precision_at_k(subjects, y_true, y_score, k=100)

    assert report["unit"] == "subject"
    assert report["n_subjects"] == 111
    assert report["max_possible_precision_at_k"] == 0.36


def _paper_temporal_block(model: str, *, auc: float) -> dict:
    return {
        "model": model,
        "sequence_length": 5,
        "sequence_level": {
            "n": 294,
            "n_positive": 94,
            "prevalence": 94 / 294,
            "sensitivity": 0.7766,
            "specificity": 0.88,
            "roc_auc": auc,
            "accuracy": 0.8469,
            "precision": 0.7526,
            "f1": 0.7644,
        },
        "subject_level": {"n_subjects": 111},
        "precision_at_k": {
            "precision_at_k": 0.73,
            "max_possible_precision_at_k": 0.94,
        },
        "subject_precision_at_k": {
            "precision_at_k": 0.36,
            "max_possible_precision_at_k": 0.36,
        },
        "fit": {
            "backend": "torch" if model == "lstm" else "sklearn",
            "representation": None if model == "lstm" else "flatten",
            "params": {"early_stopping": model == "lstm"},
            "training": {"used_validation": False},
        },
        "method_fidelity": {"early_stopping_matches_reported_method": False},
    }


def test_comparison_checks_all_metrics_baselines_and_p_at_100():
    report = {
        "results": {
            "lstm_L5": _paper_temporal_block("lstm", auc=0.91146),
            "logistic_regression_L5": _paper_temporal_block(
                "logistic_regression", auc=0.71266
            ),
        }
    }
    rows = build_comparison({"paper_temporal_reconstruction": report})[
        "table6_paper_metric_reproduction"
    ]
    by_model = {row["model"]: row for row in rows}

    assert set(by_model) == {"lstm", "logistic_regression"}
    assert set(by_model["lstm"]["metrics"]) == {
        "sensitivity", "specificity", "roc_auc", "accuracy", "precision", "f1"
    }
    assert by_model["lstm"]["all_table5_metrics_within_abs_0_03"] is False
    assert by_model["lstm"]["paper_precision_at_100_attainable_on_sequence_set"] is False
    assert by_model["lstm"]["early_stopping_applied"] is False
    assert by_model["logistic_regression"]["metrics"]["roc_auc"]["paper"] == 0.63

    rendered = render_comparison_markdown(
        build_comparison({"paper_temporal_reconstruction": report})
    )
    assert "subject P@100 0.360, 이론상 최대 0.360" in rendered


def test_cli_partial_report_never_populates_formal_comparison_tables():
    report = {
        "run_scope": {"is_partial_run": True},
        "results": {"lstm_L5": _paper_temporal_block("lstm", auc=0.99)},
    }
    comparison = build_comparison({"paper_temporal_reconstruction": report})

    row5 = next(
        row for row in comparison["table1_by_sequence_length"]
        if row["sequence_length"] == 5
    )
    assert row5["paper_temporal_reconstruction"] is None
    assert comparison["table6_paper_metric_reproduction"] == []


def test_partial_nested_report_never_becomes_the_formal_table4_estimate():
    report = {
        "run_scope": {"is_partial_run": True},
        "results": {
            "lstm_Lnested": {
                "model": "lstm",
                "sequence_length": [5],
                "n_folds": 1,
                "subject_level": {"roc_auc": 0.99, "n_subjects": 35},
            }
        },
    }
    comparison = build_comparison({"nested_subject_independent": report})
    assert comparison["table4_nested_selected"] == []


def _mergeable_report(length: int, *, seed: int = 42) -> dict:
    raw_config = {
        "experiment": "paper_temporal_reconstruction",
        "seed": seed,
        "sequence": {"lengths": [length], "stride": 1},
        "models": {
            "enabled": ["lstm"],
            "baseline_backend": "sklearn",
        },
    }
    block = _paper_temporal_block("lstm", auc=0.9)
    block["sequence_length"] = length
    return {
        "experiment": "paper_temporal_reconstruction",
        "estimand": "A",
        "seed": seed,
        "data": {"n_subjects": 174, "n_daily_rows": 12171},
        "models": ["lstm"],
        "sequence_lengths": [length],
        "resolved_config": raw_config,
        "runtime_signature_context": {
            "device": "cpu",
            "dependencies": {"torch": "2.x"},
        },
        "results": {f"lstm_L{length}": block},
        "all_audits_passed": True,
        "run_scope": {"is_partial_run": False},
    }


def test_per_length_reports_merge_only_when_all_three_are_compatible(tmp_path):
    paths = []
    for length in (3, 4, 5):
        path = tmp_path / f"L{length}.json"
        write_json(path, _mergeable_report(length))
        paths.append(path)

    merged = load_reports({"paper_temporal_reconstruction": paths})[
        "paper_temporal_reconstruction"
    ]
    assert set(merged["results"]) == {"lstm_L3", "lstm_L4", "lstm_L5"}
    assert merged["run_scope"]["is_partial_run"] is False
    assert "artifacts" not in merged and len(merged["source_metadata"]) == 3


def test_incomplete_or_incompatible_per_length_merge_is_not_formal(tmp_path):
    paths = []
    for length in (3, 4):
        path = tmp_path / f"L{length}.json"
        write_json(path, _mergeable_report(length))
        paths.append(path)
    merged = load_reports({"paper_temporal_reconstruction": paths})[
        "paper_temporal_reconstruction"
    ]
    assert merged["run_scope"]["is_partial_run"] is True

    bad = tmp_path / "bad.json"
    write_json(bad, _mergeable_report(5, seed=7))
    with pytest.raises(ValueError, match="seed differs"):
        load_reports({"paper_temporal_reconstruction": [paths[0], bad]})


def test_paper_temporal_early_stopping_requires_train_side_validation(config_dir):
    valid = load_config(config_dir / "paper_temporal_5day.yaml")
    assert valid.get("split.validation_days") == 14
    assert valid.get("lstm.early_stopping") is True

    broken_raw = deepcopy(valid.raw)
    broken_raw["split"]["validation_days"] = 0
    broken = Config(experiment=valid.experiment, raw=broken_raw, path=valid.path)
    with pytest.raises(ConfigError, match="explicit train-side validation"):
        validate_config(broken)


def test_lstm_refuses_to_silently_ignore_requested_early_stopping():
    model = SequenceLSTM(
        LSTMConfig(early_stopping=True), n_features=2, sequence_length=3
    )
    with pytest.raises(ValueError, match="explicit train-side validation"):
        model.fit(
            np.zeros((2, 3, 2), dtype=np.float32),
            np.array([0, 1]),
            validation=None,
        )


def test_unimplemented_recurrent_dropout_is_rejected(config_dir):
    valid = load_config(config_dir / "paper_temporal_5day.yaml")
    broken_raw = deepcopy(valid.raw)
    broken_raw["lstm"]["recurrent_dropout"] = 0.2
    broken = Config(experiment=valid.experiment, raw=broken_raw, path=valid.path)
    with pytest.raises(ConfigError, match="recurrent_dropout is not implemented"):
        validate_config(broken)


def test_model_specific_backends_are_fail_closed(config_dir):
    valid = load_config(config_dir / "paper_temporal_5day.yaml")
    assert valid.baseline_backend_for("logistic_regression") == "h2o"
    assert valid.baseline_backend_for("random_forest") == "h2o"
    assert valid.baseline_backend_for("xgboost") == "h2o"
    assert valid.baseline_backend_for("svm") == "sklearn"
    assert valid.uses_h2o

    svm_raw = deepcopy(valid.raw)
    svm_raw["models"]["backend_by_model"]["svm"] = "h2o"
    with pytest.raises(ConfigError, match="no SVM family"):
        validate_config(Config(valid.experiment, svm_raw, valid.path))

    typo_raw = deepcopy(valid.raw)
    typo_raw["models"]["baseline_backend"] = "sklean"
    with pytest.raises(ConfigError, match="baseline_backend"):
        validate_config(Config(valid.experiment, typo_raw, valid.path))


def test_every_shipped_config_still_validates(config_dir):
    for path in sorted(config_dir.glob("*.yaml")):
        load_config(path)


def test_finalise_marks_missing_model_length_matrix_as_partial(synthetic_data, tmp_path):
    config = Config(
        "fixed_subject_independent",
        {
            "experiment": "fixed_subject_independent",
            "seed": 42,
            "sequence": {"lengths": [3, 4]},
            "split": {"mode": "stratified_group_kfold", "outer_k": 1},
            "models": {"enabled": ["lstm"], "baseline_backend": "sklearn"},
        },
    )
    report = _finalise(
        config,
        {"lstm_L3": {"model": "lstm", "sequence_length": 3, "n_folds": 1}},
        [{"context": "test", "checks": [], "all_passed": True}],
        [pd.DataFrame({"model": ["lstm"], "sequence_hash": ["abc"]})],
        tmp_path,
        data=synthetic_data,
        estimand="B",
        device="cpu",
        evaluated_lengths=[3, 4],
    )
    assert report["run_scope"]["is_partial_run"] is True
    assert report["run_scope"]["missing_result_keys"] == ["lstm_L4"]


def test_nested_selected_lengths_do_not_make_a_full_inner_grid_partial(
    synthetic_data, tmp_path
):
    config = Config(
        "nested_subject_independent",
        {
            "experiment": "nested_subject_independent",
            "seed": 42,
            "sequence": {"lengths": [3, 4, 5]},
            "split": {
                "mode": "nested_stratified_group_kfold",
                "outer_k": 5,
                "n_repeats": 1,
            },
            "models": {"enabled": ["lstm"], "baseline_backend": "sklearn"},
        },
    )
    report = _finalise(
        config,
        {
            "lstm_Lnested": {
                "model": "lstm",
                "sequence_length": [3, 4],
                "n_folds": 5,
            }
        },
        [{"context": "test", "checks": [], "all_passed": True}],
        [pd.DataFrame({"model": ["lstm"], "sequence_hash": ["abc"]})],
        tmp_path,
        data=synthetic_data,
        estimand="B",
        device="cpu",
        evaluated_lengths=[3, 4, 5],
    )
    assert report["run_scope"]["is_partial_run"] is False


def test_zero_embargo_is_visible_as_a_warning_not_a_false_pass(synthetic_data):
    split = final_week_split(synthetic_data.daily, embargo_days=0)
    audit = audit_temporal_split(split, sequence_length=5)

    assert audit.passed  # warning-only: split-first construction still has no shared rows
    assert audit.summary()["n_warnings"] == 1
    assert audit.warnings[0]["check"] == "embargo_covers_window"
