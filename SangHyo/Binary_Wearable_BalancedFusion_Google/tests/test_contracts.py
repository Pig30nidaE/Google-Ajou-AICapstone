"""Non-training contracts for the BalancedFusion experiment."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

from SangHyo.Binary_Wearable_BalancedFusion_Google.data import (
    COMPACT_DAILY_FEATURES,
    SubjectSequenceDataset,
    VIEW_OBSERVATIONS,
    assert_disjoint_subjects,
    assert_wearable_schema,
    make_fixed_views,
)
from SangHyo.Binary_Wearable_BalancedFusion_Google.features import (
    FoldFeaturePipeline,
    SUMMARY_STATISTICS,
    SUMMARY_WINDOWS,
    ValuePreprocessor,
    build_multiscale_summaries,
)
from SangHyo.Binary_Wearable_BalancedFusion_Google.models import (
    MODEL_NAMES,
    TabNetAdapter,
    balanced_class_weights,
)
from SangHyo.Binary_Wearable_BalancedFusion_Google.train import (
    RankNormalizer,
    binary_metrics,
    select_ensemble,
    select_threshold,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _dataset(sequences: list[np.ndarray]) -> SubjectSequenceDataset:
    return SubjectSequenceDataset(
        subject_ids=np.asarray([f"s{index}" for index in range(len(sequences))]),
        sequences=sequences,
        feature_names=tuple(COMPACT_DAILY_FEATURES),
        y=np.asarray([index % 2 for index in range(len(sequences))]),
        audit={"split": "train"},
    )


def test_compact_schema_is_predeclared_wearable_only() -> None:
    assert len(COMPACT_DAILY_FEATURES) == 56
    assert len(COMPACT_DAILY_FEATURES) == len(set(COMPACT_DAILY_FEATURES))
    assert_wearable_schema(COMPACT_DAILY_FEATURES)
    assert all(
        name.startswith(("activity__", "sleep__"))
        for name in COMPACT_DAILY_FEATURES
    )


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
def test_schema_rejects_cognition_identity_and_protocol_proxy(name: str) -> None:
    with pytest.raises(AssertionError):
        assert_wearable_schema([name])


def test_fixed_view_uses_exactly_the_same_last_35_observations() -> None:
    width = len(COMPACT_DAILY_FEATURES)
    tail = np.arange(VIEW_OBSERVATIONS * width, dtype=np.float32).reshape(
        VIEW_OBSERVATIONS, width
    )
    long = np.vstack([np.full((71, width), 1e9, dtype=np.float32), tail])
    views = make_fixed_views(_dataset([tail.copy(), long]))
    assert views.shape == (2, VIEW_OBSERVATIONS, width)
    assert np.array_equal(views[0], views[1])


def test_training_validation_subject_overlap_fails_closed() -> None:
    assert_disjoint_subjects(["a", "b"], ["c", "d"])
    with pytest.raises(AssertionError, match="leakage"):
        assert_disjoint_subjects(["a", "b"], ["b", "c"])


def test_value_transform_and_multiscale_summary_are_finite_and_fixed() -> None:
    rng = np.random.default_rng(3)
    views = rng.normal(size=(12, VIEW_OBSERVATIONS, len(COMPACT_DAILY_FEATURES)))
    views[0, 0, 0] = np.nan
    fitted = ValuePreprocessor().fit(views[:8], COMPACT_DAILY_FEATURES)
    transformed = fitted.transform(views[8:])
    summary, names = build_multiscale_summaries(
        transformed, COMPACT_DAILY_FEATURES
    )
    expected = (
        len(COMPACT_DAILY_FEATURES)
        * len(SUMMARY_WINDOWS)
        * len(SUMMARY_STATISTICS)
    )
    assert summary.shape == (4, expected)
    assert len(names) == expected
    assert np.isfinite(summary).all()
    assert fitted.manifest()["labels_consumed"] is False


def test_fold_pipeline_is_small_local_and_has_cn_reference() -> None:
    rng = np.random.default_rng(7)
    n = 36
    y = np.asarray([0] * 18 + [1] * 18)
    views = rng.normal(
        size=(n, VIEW_OBSERVATIONS, len(COMPACT_DAILY_FEATURES))
    ).astype(np.float32)
    views[y == 1, :, 0] += 0.8
    fitted = FoldFeaturePipeline(max_features=24, seed=11).fit(
        views, y, COMPACT_DAILY_FEATURES
    )
    subject = fitted.transform_subject(views)
    temporal = fitted.transform_temporal(views)
    assert 8 <= len(fitted.selector_.selected_feature_names_) <= 24
    assert subject.shape[1] == 2 * len(fitted.selector_.selected_feature_names_)
    assert temporal.shape == (
        n,
        VIEW_OBSERVATIONS,
        2 * len(COMPACT_DAILY_FEATURES),
    )
    assert np.isfinite(subject).all() and np.isfinite(temporal).all()
    manifest = fitted.manifest()
    assert "CN subjects in current CV training fold only" in manifest[
        "cn_reference"
    ]["fit_scope"]


def test_balanced_weights_and_threshold_prioritize_both_classes() -> None:
    y = np.asarray([0] * 12 + [1] * 6)
    weights = balanced_class_weights(y)
    assert weights.shape == (2,)
    assert weights[1] > weights[0]
    risk = np.r_[np.linspace(0.05, 0.55, 12), np.linspace(0.45, 0.95, 6)]
    choice = select_threshold(y, risk)
    metrics = binary_metrics(y, risk, choice["chosen_threshold"])
    assert choice["selection_scope"].endswith("inner OOF only")
    assert metrics["recall_impaired"] > 0
    assert metrics["specificity_cn"] > 0


def test_rank_normalizer_is_monotonic_and_bounded() -> None:
    fitted = RankNormalizer().fit(np.asarray([0.1, 0.2, 0.4, 0.8]))
    transformed = fitted.transform(np.asarray([-1.0, 0.15, 0.9]))
    assert np.all(np.diff(transformed) > 0)
    assert np.all((transformed > 0) & (transformed < 1))


def test_rank_normalizer_uses_midrank_for_ties() -> None:
    fitted = RankNormalizer().fit(
        np.asarray([0.4, 0.4, 0.4, 0.4]),
        np.asarray([0, 0, 1, 1]),
    )
    assert np.isclose(fitted.transform(np.asarray([0.4]))[0], 0.5)


def test_tabnet_cpu_move_includes_unregistered_group_matrices() -> None:
    import torch
    from torch import nn

    class MatrixHolder(nn.Module):
        def __init__(self, attribute: str) -> None:
            super().__init__()
            setattr(self, attribute, torch.ones(2, 2))

    class FakeNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = MatrixHolder("group_attention_matrix")
            self.embedder = MatrixHolder("embedding_group_matrix")
            self.linear = nn.Linear(2, 2)

    class FakeTabNet:
        def __init__(self) -> None:
            self.network = FakeNetwork()
            self.group_matrix = torch.ones(2, 2)
            self.device = torch.device("cuda")
            self.device_name = "cuda"
            self.pin_memory = True

    fitted = TabNetAdapter(FakeTabNet()).move_to_cpu_for_inference()
    assert fitted.model.device.type == "cpu"
    assert fitted.model.device_name == "cpu"
    assert fitted.model.pin_memory is False
    assert fitted.model.group_matrix.device.type == "cpu"
    assert fitted.model.network.encoder.group_attention_matrix.device.type == "cpu"
    assert fitted.model.network.embedder.embedding_group_matrix.device.type == "cpu"


def test_tabnet_reloaded_score_uses_persisted_mapper_without_classes() -> None:
    class ReloadedLikeTabNet:
        preds_mapper = {"0": 0, "1": 1}

        @staticmethod
        def predict_proba(X: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray([[0.7, 0.3]]), (len(X), 1))

    model = ReloadedLikeTabNet()
    assert not hasattr(model, "classes_")
    score = TabNetAdapter(model).predict_score(np.zeros((3, 2)))
    assert np.allclose(score, np.asarray([0.3, 0.3, 0.3]))


def test_tabnet_rejects_reversed_persisted_class_mapping() -> None:
    class ReversedTabNet:
        preds_mapper = {"0": 1, "1": 0}

        @staticmethod
        def predict_proba(X: np.ndarray) -> np.ndarray:
            return np.tile(np.asarray([[0.3, 0.7]]), (len(X), 1))

    with pytest.raises(ValueError, match="Unexpected TabNet class mapping"):
        TabNetAdapter(ReversedTabNet()).predict_score(np.zeros((2, 2)))


def test_weak_google_models_are_allowed_zero_weight() -> None:
    y = np.asarray([0, 1] * 18)
    folds = np.tile(np.asarray([0, 1, 2]), 12)
    strong = np.where(y == 1, 0.9, 0.1).astype(float)
    rng = np.random.default_rng(21)
    raw = {
        "elastic_net": strong,
        "ydf_subject": rng.uniform(0.45, 0.55, len(y)),
        "ydf_daily": rng.uniform(0.45, 0.55, len(y)),
        "tabnet": rng.uniform(0.45, 0.55, len(y)),
        "transformer": rng.uniform(0.45, 0.55, len(y)),
    }
    _, selection = select_ensemble(y, raw, folds)
    weights = selection["chosen"]["weights"]
    assert weights["elastic_net"] > 0
    assert any(weights[name] == 0 for name in ("ydf_subject", "ydf_daily", "tabnet"))
    assert selection["google_model_minimum_weight"] == 0.0


def test_constant_prior_is_available_when_every_branch_is_uninformative() -> None:
    y = np.asarray([0, 1] * 18)
    folds = np.tile(np.asarray([0, 1, 2]), 12)
    raw = {name: np.full(len(y), 0.5) for name in MODEL_NAMES}
    _, selection = select_ensemble(y, raw, folds)
    assert selection["quality_gate_passed"] == []
    assert selection["fallback_used"] is True
    assert selection["chosen"]["weights"]["prior"] == 1.0


def test_model_registry_and_full_default_contract() -> None:
    assert tuple(MODEL_NAMES) == (
        "elastic_net",
        "ydf_subject",
        "ydf_daily",
        "tabnet",
        "transformer",
    )
    launcher = (PACKAGE_ROOT / "run.py").read_text(encoding="utf-8")
    assert 'run_mode = (mode or "full").strip().lower()' in launcher
    assert 'default="full"' in launcher
    assert '"outer_folds": 5' in launcher
    assert '"outer_repeats": 2' in launcher
    assert '"inner_folds": 3' in launcher
    assert "parse_known_args()" in launcher
    assert '"scikit-learn": ">=1.4,<2"' in launcher
    assert "SpecifierSet" in launcher
    assert "BINARY_TABNET_RUN_MODE" not in launcher
    assert "/content/drive/MyDrive/Binary_Wearable_BalancedFusion_Google_result" in launcher


def test_only_run_py_is_an_executable_entrypoint() -> None:
    entrypoints = []
    for path in PACKAGE_ROOT.glob("*.py"):
        if re.search(
            r"if\s+__name__\s*==\s*[\"']__main__[\"']",
            path.read_text(encoding="utf-8"),
        ):
            entrypoints.append(path.name)
    assert entrypoints == ["run.py"]
    run_file = "Binary_Wearable_BalancedFusion_Google/run.py"
    assert (REPOSITORY_ROOT / "SangHyo" / run_file).is_file()
    notebook = json.loads((REPOSITORY_ROOT / "base.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "RUN_PATH = USER_ROOT / RUN_PATH" in source
    assert "runpy.run_path(" in source
    assert '"DATA_ROOT": DATA_ROOT' in source


def test_launcher_ignores_ipykernel_arguments_without_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = importlib.import_module(
        "SangHyo.Binary_Wearable_BalancedFusion_Google.run"
    )
    observed = {}

    def fake_run_pipeline(**kwargs):
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(launcher, "run_pipeline", fake_run_pipeline)
    monkeypatch.setattr(
        "sys.argv",
        ["run.py", "-f", "/tmp/fake-kernel.json", "--mode", "smoke"],
    )
    launcher.main()
    assert observed["mode"] == "smoke"


def test_validation_freeze_precedes_label_open_and_checkpoints_reload() -> None:
    training = (PACKAGE_ROOT / "train.py").read_text(encoding="utf-8")
    models_source = (PACKAGE_ROOT / "models.py").read_text(encoding="utf-8")
    freeze_csv = training.index("frozen_frame.to_csv")
    freeze_marker = training.index("VALIDATION_PREDICTIONS_FROZEN.json")
    label_open = training.index("validation_labels = load_validation_labels_checked")
    assert freeze_csv < freeze_marker < label_open
    for adapter in (
        "ElasticNetAdapter.load",
        "YDFSubjectAdapter.load",
        "YDFDailyAdapter.load",
        "TabNetAdapter.load",
        "TransformerAdapter.load",
    ):
        assert adapter in training
    assert "roundtrip_verification.json" in training
    assert "CHECKPOINT_COMPLETE.json" in training
    assert "verify_checkpoint_tree" in training
    assert 'models["tabnet"].move_to_cpu_for_inference()' in training
    assert "reloaded_selection = json.loads" in training
    assert "reloaded_paths = json.loads" in training
    assert "maximum_training_duration_seconds=20.0 if fast else 75.0" in models_source
    assert "maximum_training_duration_seconds=20.0 if fast else 90.0" in models_source
    assert "validation_ratio=0.0" in models_source


def test_production_does_not_resolve_cognitive_or_mmse_files() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PACKAGE_ROOT.glob("*.py")
        if path.name != "__init__.py"
    )
    forbidden_paths = (
        r"SourceData[/\\]3[.]CognitiveFunction",
        r"LabelingData[/\\]3[.]CognitiveFunction",
        r"(?:train|val|validation)_mmse[.]csv",
        r"read_csv\s*\([^)]*(?:mmse|cognitive)",
    )
    for pattern in forbidden_paths:
        assert re.search(pattern, sources, flags=re.IGNORECASE | re.DOTALL) is None
    assert '"cognitive_source_opened": False' in (
        PACKAGE_ROOT / "data.py"
    ).read_text(encoding="utf-8")
