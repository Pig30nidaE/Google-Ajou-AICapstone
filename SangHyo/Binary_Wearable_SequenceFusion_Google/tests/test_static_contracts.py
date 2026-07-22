"""Static and small numeric contracts for the SequenceFusion experiment."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import re

import numpy as np
import pytest

from SangHyo.Binary_Wearable_SequenceFusion_Google.data import (
    ACTIVITY_CLASS_BLOB,
    _activity_sequence_features,
)
from SangHyo.Binary_Wearable_SequenceFusion_Google.preprocessing import (
    DEFAULT_VIEW_DAYS,
    SUMMARY_STATISTICS,
    SequencePreprocessor,
    assert_sequence_feature_contract,
    build_subject_summary_features,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _synthetic_views(subjects: int = 4) -> tuple[np.ndarray, list[str]]:
    names = [
        "activity__raw__activity_steps",
        "activity__met__vigorous_ratio",
        "sleep__duration__deep_ratio",
    ]
    values = np.arange(
        subjects * DEFAULT_VIEW_DAYS * len(names), dtype=np.float64
    ).reshape(subjects, DEFAULT_VIEW_DAYS, len(names))
    values[..., 1:] /= 100.0
    return values, names


def test_preprocessor_fit_api_has_no_target_and_requires_equal_views() -> None:
    signature = inspect.signature(SequencePreprocessor.fit)
    assert "y" not in signature.parameters
    assert "target" not in signature.parameters

    views, names = _synthetic_views()
    fitted = SequencePreprocessor().fit(views, names)
    assert fitted.transform_views(views).shape == views.shape
    with pytest.raises(ValueError, match="same fixed view|fixed view length"):
        SequencePreprocessor().fit(views[:, :-1, :], names)
    with pytest.raises(ValueError, match="fixed view length"):
        fitted.transform_views(views[:, :-1, :])


def test_fold_local_value_statistics_and_manifest_contract() -> None:
    names = ["activity__raw__activity_steps", "sleep__duration__deep_ratio"]
    fold_views = np.zeros((3, DEFAULT_VIEW_DAYS, 2), dtype=np.float64)
    fold_views[:, :, 0] = np.arange(DEFAULT_VIEW_DAYS, dtype=float)[None, :]
    fitted = SequencePreprocessor().fit(fold_views, names)
    manifest = fitted.manifest()

    expected_log_median = np.median(np.log1p(np.arange(DEFAULT_VIEW_DAYS)))
    assert fitted.medians_[0] == pytest.approx(expected_log_median)
    assert manifest["fit_scope"] == "current fold training subjects only"
    assert manifest["fit_scope"] in manifest["imputation"]
    assert manifest["winsorization"]["fit_scope"] == manifest["fit_scope"]
    assert manifest["fit_scope"] in manifest["scaling"]
    assert manifest["labels_consumed_by_preprocessor"] is False
    assert manifest["fixed_view_length_required"] is True
    assert manifest["subject_balance_verified_by_preprocessor"] is False
    assert manifest["variable_sequence_transform_emits_mask"] is False
    assert manifest["variable_sequence_transform_emits_padding"] is False
    assert manifest["signed_log1p_predeclared_features"] == [names[0]]

    # Values outside the fit fold cannot change any fitted statistic.
    held_out = np.full((1, DEFAULT_VIEW_DAYS, 2), 1e12, dtype=np.float64)
    before = fitted.medians_.copy()
    transformed = fitted.transform_views(held_out)
    assert np.array_equal(before, fitted.medians_)
    assert np.isfinite(transformed).all()


def test_variable_sequences_are_transformed_without_padding_or_masks() -> None:
    views, names = _synthetic_views()
    fitted = SequencePreprocessor().fit(views, names)
    sequences = [views[0, :5], views[1, :17]]
    transformed = fitted.transform_sequences(sequences)
    assert [item.shape for item in transformed] == [(5, 3), (17, 3)]
    assert all(np.isfinite(item).all() for item in transformed)


def test_invalid_activity_samples_break_runs_and_transitions() -> None:
    features = _activity_sequence_features(
        {
            ACTIVITY_CLASS_BLOB: "1/0/1/2",
            "activity_met_1min": "",
            "activity_day_start": "2026-01-01T04:00:00+09:00",
        }
    )
    # The zero is not exposed as a feature and must not join the two rest runs.
    assert features["activity__state__rest_ratio"] == pytest.approx(2 / 3)
    assert features["activity__state__rest_longest_run_ratio"] == pytest.approx(1 / 3)
    # Only the genuinely adjacent 1->2 pair contributes to the transition rate.
    assert features["activity__state__transition_rate"] == pytest.approx(1.0)


def test_semantic_summary_bank_is_fixed_traceable_and_label_free() -> None:
    names = ["activity__raw__activity_steps", "sleep__duration__deep_ratio"]
    windows = np.zeros((2, DEFAULT_VIEW_DAYS, 2), dtype=np.float64)
    windows[0, :, 0] = np.arange(DEFAULT_VIEW_DAYS, dtype=float)
    windows[0, :, 1] = 2.0
    windows[1] = windows[0] * -1.0
    values, output_names = build_subject_summary_features(windows, names)

    assert values.shape == (2, len(names) * len(SUMMARY_STATISTICS))
    assert len(output_names) == len(set(output_names))
    assert output_names[0] == f"{names[0]}__summary__median"
    assert output_names[-1].endswith("recent7_minus_previous21_median")
    # Recent median 24 minus previous-21 median 10 on a 0..27 sequence.
    shift_index = output_names.index(
        f"{names[0]}__summary__recent7_minus_previous21_median"
    )
    assert values[0, shift_index] == pytest.approx(14.0)
    assert values[1, shift_index] == pytest.approx(-14.0)
    assert all("label" not in name and "diag" not in name for name in output_names)


@pytest.mark.parametrize(
    "feature_name",
    [
        "activity__subject_id",
        "sleep__observed_count",
        "activity__absolute_date",
        "sleep__missing_mask",
        "activity__non_wear",
        "sleep__mmse",
        "sleep__cognitive_score",
        "other__steps",
    ],
)
def test_forbidden_features_fail_closed(feature_name: str) -> None:
    with pytest.raises(AssertionError):
        assert_sequence_feature_contract([feature_name])


def test_production_code_cannot_resolve_or_open_cognitive_score_sources() -> None:
    production_files = [
        path
        for path in PACKAGE_ROOT.glob("*.py")
        if path.name not in {"__init__.py"}
    ]
    assert production_files, "SequenceFusion production modules are missing"
    combined = "\n".join(path.read_text(encoding="utf-8") for path in production_files)
    forbidden_path_patterns = (
        r"SourceData[/\\]3[.]CognitiveFunction",
        r"LabelingData[/\\]3[.]CognitiveFunction",
        r"(?:train|validation)_mmse[.]csv",
        r"read_csv\s*\([^)]*(?:mmse|cognitive)",
    )
    for pattern in forbidden_path_patterns:
        assert re.search(pattern, combined, flags=re.IGNORECASE | re.DOTALL) is None

    # The wearable-only feature contracts must explicitly reject both names.
    preprocessing_source = (PACKAGE_ROOT / "preprocessing.py").read_text(encoding="utf-8")
    assert '"mmse"' in preprocessing_source
    assert '"cognitive"' in preprocessing_source


def test_expected_google_and_reference_model_names_are_registered() -> None:
    models = importlib.import_module(
        "SangHyo.Binary_Wearable_SequenceFusion_Google.models"
    )
    registered = set(getattr(models, "MODEL_NAMES"))
    assert {"ydf_gbt", "sequence_transformer", "conv_bilstm"}.issubset(registered)

    origins = getattr(models, "GOOGLE_MODEL_EVIDENCE")
    assert origins["ydf_gbt"]["origin"].lower() == "google"
    assert "google" in origins["sequence_transformer"]["origin"].lower()


def test_subject_probability_aggregation_requires_equal_view_counts() -> None:
    models = importlib.import_module(
        "SangHyo.Binary_Wearable_SequenceFusion_Google.models"
    )
    probabilities = np.asarray([0.1, 0.3, 0.7, 0.9], dtype=np.float64)
    equal = models.aggregate_view_probabilities(
        probabilities, np.asarray([0, 0, 1, 1]), n_subjects=2
    )
    assert np.allclose(equal, [0.2, 0.8])
    with pytest.raises(AssertionError, match="Unequal/missing subject views"):
        models.aggregate_view_probabilities(
            probabilities, np.asarray([0, 0, 0, 1]), n_subjects=2
        )


def test_run_py_is_the_only_executable_python_entrypoint() -> None:
    run_file = PACKAGE_ROOT / "run.py"
    assert run_file.is_file()
    entrypoints = []
    for path in PACKAGE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"if\s+__name__\s*==\s*[\"']__main__[\"']", source):
            entrypoints.append(path.name)
    assert entrypoints == ["run.py"]
    assert "run_pipeline" in run_file.read_text(encoding="utf-8")


def test_readme_warns_historical_validation_is_not_a_fresh_test() -> None:
    readme = PACKAGE_ROOT / "README_KO.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8").lower()
    assert "validation" in text
    assert "재사용" in text or "historical" in text
    assert "새로운 독립" in text or "fresh test" in text or "새 독립" in text


def test_base_notebook_needs_only_one_run_file_path() -> None:
    # The experiment itself has one launcher; users point base.ipynb RUN_FILE
    # at this file and do not invoke individual model/preprocessing scripts.
    expected = "SangHyo/Binary_Wearable_SequenceFusion_Google/run.py"
    assert (REPOSITORY_ROOT / expected).is_file()
    assert expected.endswith("/run.py")


def test_base_notebook_fresh_clones_main_in_colab() -> None:
    notebook = json.loads((REPOSITORY_ROOT / "base.ipynb").read_text(encoding="utf-8"))
    setup_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "63dc880f")
    source = "".join(setup_cell["source"])
    assert "shutil.rmtree(clone_path)" in source
    assert '"--depth", "1", "--branch", "main"' in source
    assert '"--single-branch"' in source
    assert "sys.modules.pop(module_name, None)" in source
    assert '"rev-parse", "--short", "HEAD"' in source


def test_default_results_root_is_versioned_on_colab_mydrive() -> None:
    launcher = (PACKAGE_ROOT / "run.py").read_text(encoding="utf-8")
    expected = (
        "/content/drive/MyDrive/"
        "Binary_Wearable_SequenceFusion_Google_result"
    )
    assert expected in launcher
    assert "DEFAULT_COLAB_RESULTS_ROOT / run_id" in launcher
    assert "Google Drive is not mounted" in launcher
