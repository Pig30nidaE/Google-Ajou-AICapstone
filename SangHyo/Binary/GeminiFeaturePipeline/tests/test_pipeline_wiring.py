"""Stage/mode wiring, split contracts and configuration injection.

Static or synthetic only: no model is fitted, no API is called, and the real
``Data/`` tree is never opened.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from SangHyo.Binary.GeminiFeaturePipeline import config as config_module, pipeline
from SangHyo.Binary.GeminiFeaturePipeline import run
from SangHyo.Binary.GeminiFeaturePipeline.features import (
    assemble_design_matrix,
    build_gemini_features,
    build_mmse_features,
)
from SangHyo.Binary.GeminiFeaturePipeline.data import MMSE_ITEMS
from SangHyo.Binary.GeminiFeaturePipeline.guards import LeakageError, assert_disjoint_subjects
from SangHyo.Binary.GeminiFeaturePipeline.schema import FEATURE_NAMES
from SangHyo.Binary.GeminiFeaturePipeline.splits import build_split_plan

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


# --- 10. every stage and mode reaches a real function ----------------------- #
def test_every_cli_stage_maps_to_a_function():
    parser = run.build_parser()
    stage_action = next(action for action in parser._actions if action.dest == "stage")
    assert set(stage_action.choices) == set(pipeline.STAGES)
    for stage in pipeline.STAGES:
        if stage == "all":
            continue
        assert callable(pipeline._STAGE_FUNCTIONS[stage])


def test_models_stage_discovers_availability_without_generating(tmp_path):
    """The `models` diagnostic must read the key's real model list, not the docs."""

    from SangHyo.Binary.GeminiFeaturePipeline.config import GeminiConfig
    from SangHyo.Binary.GeminiFeaturePipeline.gemini_client import GeminiFeatureExtractor

    class _FakeModel:
        def __init__(self, name, actions):
            self.name = name
            self.display_name = name
            self.supported_actions = actions
            self.input_token_limit = 1000
            self.output_token_limit = 100

    class _FakeModels:
        def list(self):
            return [
                _FakeModel("models/gemini-2.5-flash", ["generateContent"]),
                _FakeModel("models/gemini-3.1-flash-lite", ["generateContent"]),
                _FakeModel("models/text-embedding-004", ["embedContent"]),
            ]

    class _FakeClient:
        models = _FakeModels()

    extractor = GeminiFeatureExtractor(GeminiConfig(model="gemini-2.5-flash"), cache_root=tmp_path)
    extractor._client = _FakeClient()
    report = extractor.list_available_models()

    # embedding-only models must not be offered as generation targets
    assert report["generate_content_models"] == ["gemini-2.5-flash", "gemini-3.1-flash-lite"]
    assert report["configured_model_is_available"] is True
    assert report["n_models_visible"] == 3

    unavailable = GeminiFeatureExtractor(
        GeminiConfig(model="gemini-2.5-flash-lite"), cache_root=tmp_path
    )
    unavailable._client = _FakeClient()
    assert unavailable.list_available_models()["configured_model_is_available"] is False


def test_models_stage_requires_the_sdk_even_in_dry_run():
    """`models` always calls the API, so dry-run must not skip its dependency."""

    parser = run.build_parser()
    for argv in (["--stage", "models"], ["--stage", "models", "--dry-run"]):
        args = parser.parse_args(argv)
        needs_api = args.stage == "models" or (
            args.stage in {"gemini", "all"}
            and not args.dry_run
            and not args.offline
            and not args.no_gemini
        )
        assert needs_api is True


def test_every_mmse_mode_is_reachable():
    parser = run.build_parser()
    action = next(action for action in parser._actions if action.dest == "mmse_mode")
    assert set(action.choices) == {"without", "with", "both"}
    assert config_module.mmse_modes(config_module.PipelineConfig(mmse_mode="both")) == (
        "without",
        "with",
    )
    assert config_module.mmse_modes(config_module.PipelineConfig(mmse_mode="with")) == ("with",)


def test_entrypoint_main_block_is_minimal():
    """`if __name__ == '__main__':` must only delegate, never implement logic."""

    tree = ast.parse((PACKAGE_ROOT / "run.py").read_text(encoding="utf-8"))
    main_blocks = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and ast.dump(node.test).find("__main__") != -1
    ]
    assert len(main_blocks) == 1
    assert len(main_blocks[0].body) == 1


def test_notebook_launch_arguments():
    assert run.notebook_argv({}) == ["--stage", "all", "--mmse-mode", "both"]
    assert run.notebook_argv({"GFP_ARGS": "--stage gemini --dry-run"}) == [
        "--stage",
        "gemini",
        "--dry-run",
    ]
    assert run.strip_jupyter_arguments(["-f", "/x/kernel-1234.json", "--stage", "train"]) == [
        "--stage",
        "train",
    ]


# --- 11. paths and secrets come from configuration, not from the source ----- #
def test_paths_are_injected_not_hard_coded():
    configured = config_module.load_config(
        None,
        environ={
            "GFP_OUTPUT_ROOT": "/tmp/out",
            "GFP_CACHE_ROOT": "/tmp/cache",
            "GFP_GEMINI_MODEL": "gemini-2.5-pro",
            "GFP_CV_SPLITS": "4",
        },
    )
    assert str(configured.resolved_output_root()) == "/tmp/out"
    assert str(configured.resolved_cache_root()) == "/tmp/cache"
    assert configured.gemini.model == "gemini-2.5-pro"
    assert configured.cv.n_splits == 4

    allowed = {"config.py", "config.yaml", "README_KO.md", "pipeline.py"}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if path.name in allowed or "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        assert "/content/drive" not in text, f"{path.name} hard-codes a Drive path"


def test_cli_overrides_beat_environment():
    configured = config_module.load_config(
        None,
        environ={"GFP_GEMINI_MODEL": "gemini-2.5-flash"},
        cli_overrides={"gemini.model": "gemini-3-pro", "mmse_mode": "with"},
    )
    assert configured.gemini.model == "gemini-3-pro"
    assert configured.mmse_mode == "with"


def test_thinking_knobs_are_configurable_and_mutually_exclusive():
    configured = config_module.load_config(
        None,
        environ={
            "GFP_GEMINI_MAX_OUTPUT_TOKENS": "4096",
            "GFP_GEMINI_THINKING_LEVEL": "null",
            "GFP_GEMINI_THINKING_BUDGET": "0",
        },
    )
    assert configured.gemini.max_output_tokens == 4096
    assert configured.gemini.thinking_level is None
    assert configured.gemini.thinking_budget == 0

    with pytest.raises(ValueError, match="only one"):
        config_module.load_config(
            None,
            environ={"GFP_GEMINI_THINKING_BUDGET": "0"},
        )


def test_tuning_must_stay_disabled():
    with pytest.raises(ValueError, match="tuning.enabled"):
        config_module.load_config(None, cli_overrides={"tuning.enabled": True})


# --- 4. train/validation subject separation --------------------------------- #
def test_split_plan_never_shares_a_subject_between_folds():
    rng = np.random.default_rng(0)
    y = np.asarray([0] * 85 + [1] * 56)
    rng.shuffle(y)
    subjects = [f"s{index:03d}" for index in range(len(y))]
    plan = build_split_plan(y, subjects, n_splits=5, n_repeats=3, seed=1, min_positive_per_validation_fold=2)
    assert plan.n_repeats == 3 and len(plan.records) == 15
    for record in plan.records:
        train = {subjects[index] for index in record.train_indices}
        validation = {subjects[index] for index in record.validation_indices}
        assert not (train & validation)
        assert y[record.validation_indices].sum() >= 2
    for repeat in range(plan.n_repeats):
        covered = np.concatenate([r.validation_indices for r in plan.for_repeat(repeat)])
        assert sorted(covered.tolist()) == list(range(len(y)))


def test_overlapping_subjects_are_rejected():
    with pytest.raises(LeakageError):
        assert_disjoint_subjects(["a", "b"], ["b", "c"], context="test")


def test_more_folds_than_positives_is_rejected():
    y = np.asarray([0] * 20 + [1] * 3)
    subjects = [f"s{index}" for index in range(len(y))]
    with pytest.raises(LeakageError, match="exceed the minority"):
        build_split_plan(y, subjects, n_splits=5, n_repeats=1, seed=1)


# --- 8. merging must not change the subject set ----------------------------- #
def _blocks(subjects):
    base = pd.DataFrame(
        {"base__act_steps__mean": np.arange(len(subjects), dtype=float)},
        index=[str(s) for s in subjects],
    )
    base.index.name = "subject_id"
    gemini = build_gemini_features(
        {str(s): {name: 0.5 for name in FEATURE_NAMES} for s in subjects}
    )
    mmse_table = pd.DataFrame(
        {"TOTAL": [28.0] * len(subjects), **{item: [2.0] * len(subjects) for item in MMSE_ITEMS}},
        index=[str(s) for s in subjects],
    )
    mmse_table.index.name = "subject_id"
    return base, gemini, build_mmse_features(mmse_table)


def test_subject_count_is_preserved_when_blocks_are_merged():
    subjects = [f"s{index}" for index in range(10)]
    base, gemini, mmse = _blocks(subjects)
    matrix = assemble_design_matrix(
        subjects=subjects, base=base, gemini=gemini, mmse=mmse,
        feature_set="base_gemini", mmse_mode="with",
    )
    assert matrix.X.shape[0] == len(subjects)
    assert matrix.n_features == 1 + len(FEATURE_NAMES) + mmse.shape[1]
    assert set(matrix.blocks) == {"base", "gemini", "mmse"}


def test_missing_subject_in_a_block_is_an_error():
    subjects = [f"s{index}" for index in range(10)]
    base, gemini, mmse = _blocks(subjects)
    with pytest.raises(LeakageError, match="has no row"):
        assemble_design_matrix(
            subjects=subjects + ["s99"], base=base, gemini=gemini, mmse=None,
            feature_set="base", mmse_mode="without",
        )


# --- 3/11. the no-MMSE arm must be structurally MMSE-free ------------------- #
def test_without_mode_rejects_the_mmse_block():
    subjects = [f"s{index}" for index in range(10)]
    base, gemini, mmse = _blocks(subjects)
    contaminated = pd.concat([base, mmse], axis=1)
    with pytest.raises(LeakageError, match="MMSE"):
        assemble_design_matrix(
            subjects=subjects, base=contaminated, gemini=gemini, mmse=None,
            feature_set="base_gemini", mmse_mode="without",
        )


def test_with_and_without_share_the_same_split_plan():
    y = np.asarray([0] * 12 + [1] * 8)
    subjects = [f"s{index:02d}" for index in range(len(y))]
    first = build_split_plan(y, subjects, n_splits=4, n_repeats=2, seed=7)
    second = build_split_plan(y, subjects, n_splits=4, n_repeats=2, seed=7)
    assert first.plan_hash == second.plan_hash
