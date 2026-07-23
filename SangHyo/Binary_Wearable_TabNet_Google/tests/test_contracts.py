"""Static, numeric, and small model contracts for the TabNet experiment."""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from SangHyo.Binary_Wearable_TabNet_Google.data import SubjectSequenceDataset
from SangHyo.Binary_Wearable_TabNet_Google.features import (
    AGGREGATE_STATISTICS,
    assert_feature_contract,
    build_subject_feature_table,
)
from SangHyo.Binary_Wearable_TabNet_Google.models import balanced_class_weights
from SangHyo.Binary_Wearable_TabNet_Google.preprocessing import FoldPreprocessor


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _dataset(sequences: list[np.ndarray]) -> SubjectSequenceDataset:
    return SubjectSequenceDataset(
        subject_ids=np.asarray([f"s{index}" for index in range(len(sequences))]),
        sequences=sequences,
        feature_names=("activity__scalar__steps", "sleep__scalar__efficiency"),
        y=np.asarray([index % 2 for index in range(len(sequences))], dtype=np.int64),
        audit={"split": "train"},
    )


def test_subject_features_use_exactly_the_same_last_28_observations() -> None:
    tail = np.arange(56, dtype=np.float32).reshape(28, 2)
    short = tail.copy()
    long = np.vstack([np.full((73, 2), 1e9, dtype=np.float32), tail])
    table = build_subject_feature_table(_dataset([short, long]))

    assert table.X.shape == (2, 2 * len(AGGREGATE_STATISTICS))
    assert np.allclose(table.X.iloc[0], table.X.iloc[1], equal_nan=True)
    assert table.audit["fixed_observation_window"].startswith("last 28")
    assert table.audit["observation_count_feature_emitted"] is False


@pytest.mark.parametrize(
    "name",
    [
        "activity__mmse__total",
        "sleep__cognitive_score",
        "activity__subject_id",
        "sleep__coverage",
        "activity__sequence_length",
        "sleep__absolute_date",
        "other__steps",
    ],
)
def test_feature_contract_fails_closed(name: str) -> None:
    with pytest.raises(AssertionError):
        assert_feature_contract([name])


def test_fold_preprocessing_is_local_finite_and_schema_strict() -> None:
    columns = [
        "activity__scalar__steps__aggregate__median",
        "activity__scalar__score__aggregate__mean",
        "sleep__scalar__efficiency__aggregate__median",
        "sleep__scalar__total__aggregate__iqr",
    ]
    frame = pd.DataFrame(
        [
            [1.0, 80.0, 90.0, 10.0],
            [2.0, 81.0, 91.0, 11.0],
            [8.0, 70.0, 75.0, 20.0],
            [9.0, 71.0, 74.0, np.nan],
        ],
        columns=columns,
    )
    target = np.asarray([0, 0, 1, 1])
    fitted = FoldPreprocessor(
        max_features=4,
        bootstrap_rounds=2,
        minimum_per_modality=1,
        seed=1,
    ).fit(frame, target)
    before = fitted.medians_.copy()
    transformed = fitted.transform(pd.DataFrame([[1e12] * 4], columns=columns))
    assert np.array_equal(before.to_numpy(), fitted.medians_.to_numpy())
    assert np.isfinite(transformed).all()
    with pytest.raises(ValueError, match="schema/order"):
        fitted.transform(frame[columns[::-1]])
    manifest = fitted.manifest()
    assert manifest["fit_scope"] == "current CV training subjects only"


def test_binary_class_weights_have_exactly_two_keys_and_mean_one() -> None:
    y = np.asarray([0] * 8 + [1] * 4)
    weights = balanced_class_weights(y, power=0.5)
    assert weights.shape == (2,)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[1] > weights[0]


def test_production_never_resolves_or_reads_cognitive_source_paths() -> None:
    production = [path for path in PACKAGE_ROOT.glob("*.py") if path.name != "__init__.py"]
    # Include the two shared audited dependencies imported by this package.
    production += [
        REPOSITORY_ROOT / "SangHyo/Binary_Wearable_SequenceFusion_Google/data.py",
        REPOSITORY_ROOT / "SangHyo/Binary_Wearable_SequenceFusion_Google/eda.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in production)
    forbidden_path_patterns = (
        r"SourceData[/\\]3[.]CognitiveFunction",
        r"LabelingData[/\\]3[.]CognitiveFunction",
        r"(?:train|val|validation)_mmse[.]csv",
        r"read_csv\s*\([^)]*(?:mmse|cognitive)",
    )
    for pattern in forbidden_path_patterns:
        assert re.search(pattern, combined, flags=re.IGNORECASE | re.DOTALL) is None
    data_source = (PACKAGE_ROOT / "data.py").read_text(encoding="utf-8")
    assert '"Activity", "Sleep"' in data_source
    assert '"cognitive_source_opened": False' in data_source


def test_only_run_py_is_an_executable_entrypoint_and_full_is_default() -> None:
    entrypoints = []
    for path in PACKAGE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if re.search(r"if\s+__name__\s*==\s*[\"']__main__[\"']", source):
            entrypoints.append(path.name)
    assert entrypoints == ["run.py"]
    launcher = (PACKAGE_ROOT / "run.py").read_text(encoding="utf-8")
    assert 'run_mode = (mode or "full").strip().lower()' in launcher
    assert 'default="full"' in launcher
    assert "BINARY_TABNET_RUN_MODE" not in launcher
    assert 'run_mode == "smoke"' in launcher
    assert "/content/drive/MyDrive/Binary_Wearable_TabNet_Google_result" in launcher
    assert "DEFAULT_COLAB_RESULTS_ROOT / run_id" in launcher


def test_checkpoint_and_validation_freeze_contracts_are_present() -> None:
    training = (PACKAGE_ROOT / "train.py").read_text(encoding="utf-8")
    assert "TabNetAdapter.load" in training
    assert "YDFAdapter.load" in training
    assert "roundtrip_verification.json" in training
    assert "CHECKPOINT_COMPLETE.json" in training
    assert "verify_checkpoint_tree" in training
    assert "partial_bundles" in training
    assert "actual_files != expected_files" in training
    assert "checkpoint_index.json" in training
    assert 'checkpoints / "full_refit"' in training
    freeze_write = training.index("frozen_frame.to_csv")
    freeze_hash = training.index("VALIDATION_PREDICTIONS_FROZEN.json")
    label_open = training.index("validation_labels = load_binary_labels")
    assert freeze_write < freeze_hash < label_open
    assert "sha256_file(frozen_path) != frozen_sha" in training


def test_nested_cv_and_model_registration_contracts() -> None:
    launcher = (PACKAGE_ROOT / "run.py").read_text(encoding="utf-8")
    assert '"outer_folds": 5' in launcher
    assert '"outer_repeats": 2' in launcher
    assert '"inner_folds": 3' in launcher
    assert '"tabnet_seeds": 3' in launcher
    assert '"trials_tabnet": 4' in launcher
    assert '"trials_ydf": 4' in launcher
    training = (PACKAGE_ROOT / "train.py").read_text(encoding="utf-8")
    assert "config.tabnet_seeds if model_name == \"tabnet\" else 1" in training
    models = importlib.import_module("SangHyo.Binary_Wearable_TabNet_Google.models")
    assert tuple(models.MODEL_NAMES) == ("tabnet", "ydf")
    assert "google" in models.GOOGLE_MODEL_EVIDENCE["tabnet"]["origin"].lower()
    assert "google" in models.GOOGLE_MODEL_EVIDENCE["ydf"]["origin"].lower()


def test_smoke_deadline_reserve_fits_its_total_budget() -> None:
    launcher = (PACKAGE_ROOT / "run.py").read_text(encoding="utf-8")
    training = importlib.import_module("SangHyo.Binary_Wearable_TabNet_Google.train")
    assert '"max_runtime_seconds": 1_800' in launcher
    assert training._reserve_for_mode(1_800, smoke=True) < 1_800
    assert training._reserve_for_mode(1_800, smoke=False) == 1_800


def test_selected_blend_always_contains_tabnet() -> None:
    from SangHyo.Binary_Wearable_TabNet_Google.train import (
        select_calibration_and_blend,
    )

    y = np.asarray([0, 1] * 10, dtype=np.int64)
    raw = {
        "tabnet": np.linspace(0.1, 0.9, len(y)),
        "ydf": np.where(y == 1, 0.8, 0.2).astype(float),
    }
    _, blend = select_calibration_and_blend(y, raw)
    assert blend["chosen"]["tabnet_weight"] >= 0.25
    assert all(row["tabnet_weight"] >= 0.25 for row in blend["grid"])


def test_base_notebook_requires_only_the_new_run_file_path() -> None:
    user_folder = "SangHyo"
    run_file = "Binary_Wearable_TabNet_Google/run.py"
    assert (REPOSITORY_ROOT / user_folder / run_file).is_file()
    notebook = json.loads((REPOSITORY_ROOT / "base.ipynb").read_text(encoding="utf-8"))
    notebook_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "RUN_PATH = USER_ROOT / RUN_PATH" in notebook_source
    assert "runpy.run_path(" in notebook_source
    assert 'run_name="__main__"' in notebook_source
    assert '"DATA_ROOT": DATA_ROOT' in notebook_source


def test_production_split_and_feature_shape_guards_are_present() -> None:
    training = (PACKAGE_ROOT / "train.py").read_text(encoding="utf-8")
    assert "EXPECTED_DAILY_FEATURES = 119" in training
    assert "EXPECTED_SUBJECT_FEATURES = 1_077" in training
    assert "overlapping_subjects" in training
    assert "observed_validation_subjects != expected_validation_subjects" in training
    assert "observed_validation_counts != {0: 26, 1: 7}" in training


def test_tabnet_fit_declares_binary_weight_keys() -> None:
    models = importlib.import_module("SangHyo.Binary_Wearable_TabNet_Google.models")
    source = inspect.getsource(models.fit_tabnet)
    assert "range(2)" in source
    assert "set(weight_map) != {0, 1}" in source


def test_small_tabnet_checkpoint_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_tabnet")
    from SangHyo.Binary_Wearable_TabNet_Google.models import TabNetAdapter, fit_tabnet

    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    y = np.asarray([0, 1] * 20, dtype=np.int64)
    params = {
        "n_d": 8,
        "n_steps": 3,
        "gamma": 1.2,
        "lambda_sparse": 1e-5,
        "mask_type": "entmax",
        "lr": 0.01,
        "weight_decay": 1e-5,
        "epochs": 2,
        "batch_size": 16,
        "virtual_batch_size": 8,
        "class_weight_power": 0.25,
    }
    fitted = fit_tabnet(X, y, params, seed=11, device_name="cpu")
    expected = fitted.predict_proba(X[:6])
    saved = fitted.save(tmp_path / "model")
    loaded = TabNetAdapter.load(saved, device_name="cpu")
    assert np.allclose(expected, loaded.predict_proba(X[:6]), rtol=1e-5, atol=1e-6)


def test_small_ydf_checkpoint_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("ydf")
    from SangHyo.Binary_Wearable_TabNet_Google.models import YDFAdapter, fit_ydf

    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 6)).astype(np.float32)
    y = np.asarray([0, 1] * 30, dtype=np.int64)
    params = {
        "num_trees": 20,
        "max_depth": 3,
        "min_examples": 2,
        "shrinkage": 0.08,
        "subsample": 0.9,
        "num_candidate_attributes_ratio": 0.6,
        "l2_regularization": 0.01,
        "class_weight_power": 0.25,
        "num_threads": 2,
    }
    fitted = fit_ydf(X, y, params, seed=13)
    expected = fitted.predict_proba(X[:8])
    fitted.save(tmp_path / "ydf")
    loaded = YDFAdapter.load(tmp_path / "ydf")
    assert np.allclose(expected, loaded.predict_proba(X[:8]), rtol=0.0, atol=1e-10)
