"""Nothing selected inside the nested loop may have seen the outer test set."""

from __future__ import annotations

import pytest

from src.audit.leakage import LeakageError, audit_outer_test_isolation
from src.splits.group import SubjectSplit, inner_splits, stratified_group_splits
from src.utils.config import Config, ConfigError, load_config, validate_config


def test_inner_cv_is_drawn_from_outer_train_only(synthetic_data):
    outer = stratified_group_splits(synthetic_data.subjects, n_splits=5, seed=2)[0]
    inner = inner_splits(synthetic_data.subjects, outer, n_splits=3, seed=2)

    log = audit_outer_test_isolation(
        outer.test_subjects, inner_splits=inner, selection_scores_source="inner_cv"
    )
    log.raise_if_failed()
    assert log.passed


def test_a_contaminated_inner_fold_is_rejected(synthetic_data):
    outer = stratified_group_splits(synthetic_data.subjects, n_splits=5, seed=2)[0]
    contaminated = SubjectSplit(
        train_subjects=tuple(outer.train_subjects[:5]) + (outer.test_subjects[0],),
        test_subjects=tuple(outer.train_subjects[5:10]),
        name="contaminated",
    )
    log = audit_outer_test_isolation(
        outer.test_subjects, inner_splits=[contaminated],
        selection_scores_source="inner_cv",
    )
    assert not log.passed
    with pytest.raises(LeakageError):
        log.raise_if_failed()


def test_selection_scores_must_come_from_inner_cv(synthetic_data):
    outer = stratified_group_splits(synthetic_data.subjects, n_splits=5, seed=2)[0]
    inner = inner_splits(synthetic_data.subjects, outer, n_splits=3, seed=2)
    log = audit_outer_test_isolation(
        outer.test_subjects, inner_splits=inner, selection_scores_source="outer_test"
    )
    assert "selection_scores_come_from_inner_cv" in {r["check"] for r in log.failures}


def test_a_leaked_outer_split_cannot_even_be_constructed(synthetic_data):
    """The guard fires at construction, so ``inner_splits`` can never receive one.

    This is why the nested runner has no "is the outer split clean?" branch: an
    outer split whose train side contains a test subject does not exist.
    """
    outer = stratified_group_splits(synthetic_data.subjects, n_splits=5, seed=2)[0]
    with pytest.raises(AssertionError, match="both sides"):
        SubjectSplit(
            train_subjects=tuple(synthetic_data.subject_ids()),   # includes the test subjects
            test_subjects=outer.test_subjects,
            name="broken",
        )


def test_split_construction_rejects_overlap():
    with pytest.raises(AssertionError, match="both sides"):
        SubjectSplit(train_subjects=("a", "b"), test_subjects=("b", "c"))


def test_inner_splits_only_ever_return_outer_train_subjects(synthetic_data):
    """The second guard: whatever comes back is a subset of the outer train side."""
    outer = stratified_group_splits(synthetic_data.subjects, n_splits=5, seed=2)[0]
    for inner in inner_splits(synthetic_data.subjects, outer, n_splits=3, seed=2):
        assert set(inner.train_subjects) | set(inner.test_subjects) <= set(
            outer.train_subjects
        )


def test_nested_config_must_select_length_inside_inner_cv():
    raw = {
        "experiment": "nested_subject_independent",
        "split": {"mode": "nested_stratified_group_kfold", "inner_k": 3},
        "sequence": {"lengths": [3, 4, 5], "length_selection": "outer_test"},
        "tuning": {"enabled": True},
        "models": {"enabled": ["lstm"]},
    }
    with pytest.raises(ConfigError, match="length_selection"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))


def test_non_nested_configs_must_not_tune(config_dir):
    for name in ("fixed_subject_independent", "strict_same_subject_temporal",
                 "paper_temporal_5day"):
        config = load_config(config_dir / f"{name}.yaml")
        assert config.tuning_enabled is False


def test_shipped_nested_config_is_valid_and_tunes_inside(config_dir):
    config = load_config(config_dir / "nested_subject_independent.yaml")
    assert config.tuning_enabled is True
    assert config.get("sequence.length_selection") == "inner_cv"
    assert int(config.get("split.inner_k")) >= 2
    assert config.estimand == "B"
