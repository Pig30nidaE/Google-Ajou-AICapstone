"""No subject may appear on both sides of any split."""

from __future__ import annotations

import pytest

from src.audit.leakage import AuditLog, LeakageError, check_subject_overlap
from src.splits import splitters
from src.utils.config import Config


def test_check_raises_on_shared_subject() -> None:
    log = AuditLog()
    with pytest.raises(LeakageError, match="subject_overlap"):
        check_subject_overlap(["a", "b", "c"], ["c", "d"], log)


def test_check_passes_on_disjoint_subjects() -> None:
    log = AuditLog()
    check_subject_overlap(["a", "b"], ["c", "d"], log)
    assert log.passed


def test_subject_split_rejects_overlap_at_construction() -> None:
    with pytest.raises(ValueError, match="both sides"):
        splitters.SubjectSplit(train_subjects=("a", "b"), test_subjects=("b",))


def test_official_partition_is_subject_disjoint(synthetic_data) -> None:
    split = splitters.official_partition(synthetic_data)
    assert set(split.train_subjects) & set(split.test_subjects) == set()
    assert len(split.train_subjects) > 0 and len(split.test_subjects) > 0


def test_group_kfold_folds_are_subject_disjoint(synthetic_data) -> None:
    folds = list(
        splitters.stratified_group_kfold(synthetic_data, n_splits=4, repeats=2, seed=1)
    )
    assert len(folds) == 8
    for split in folds:
        assert set(split.train_subjects) & set(split.test_subjects) == set()


def test_every_subject_is_tested_once_per_repeat(synthetic_data) -> None:
    all_subjects = set(synthetic_data.labels_by_subject().index.astype(str))
    for repeat in range(2):
        folds = [
            s for s in splitters.stratified_group_kfold(
                synthetic_data, n_splits=4, repeats=2, seed=1
            )
            if s.repeat == repeat
        ]
        tested = [s for split in folds for s in split.test_subjects]
        assert sorted(tested) == sorted(all_subjects), "each subject tested exactly once"


def test_random_subject_holdout_is_disjoint(synthetic_data) -> None:
    split = splitters.random_subject_holdout(synthetic_data, test_size=0.25, seed=7)
    assert set(split.train_subjects) & set(split.test_subjects) == set()
    assert split.name.startswith("assumption_variant")


def test_row_level_split_is_rejected_outside_experiment_a() -> None:
    """A row split duplicates subjects, so only experiment A may declare it."""
    from src.utils.config import ConfigError, validate_config

    raw = {
        "experiment": "leakage_controlled_non_nested",
        "split": {"mode": "assumption_variant_random_row_holdout"},
        "models": ["random_forest"],
        "representations": {"random_forest": "daily_record"},
    }
    with pytest.raises(ConfigError, match="row-level split"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))


def test_row_level_split_needs_explicit_diagnostic_flag() -> None:
    from src.utils.config import ConfigError, validate_config

    raw = {
        "experiment": "paper_reported_reconstruction",
        "split": {"mode": "assumption_variant_random_row_holdout"},
        "models": ["random_forest"],
        "representations": {"random_forest": "daily_record"},
    }
    with pytest.raises(ConfigError, match="leakage_diagnostic_only"):
        validate_config(Config(experiment=raw["experiment"], raw=raw))

    raw["split"]["leakage_diagnostic_only"] = True
    validate_config(Config(experiment=raw["experiment"], raw=raw))  # now allowed
