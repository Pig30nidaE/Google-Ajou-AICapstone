"""train/test 피험자·원시 row 교집합 검사."""

from __future__ import annotations

import numpy as np
import pytest

from src.audit.checks import check_row_overlap, check_subject_overlap
from src.audit.leakage import LeakageError
from src.splits.group_cv import Fold, describe_folds, make_group_folds
from src.splits.row_level import paper_row_split


def test_subject_overlap_detected():
    v = check_subject_overlap(["a", "b", "c"], ["c", "d"])
    assert len(v) == 1 and v[0].code == "SUBJECT_OVERLAP"
    assert v[0].detail["n_shared"] == 1


def test_subject_overlap_clean():
    assert check_subject_overlap(["a", "b"], ["c", "d"]) == []


def test_row_overlap_detected():
    v = check_row_overlap([1, 2, 3], [3, 4])
    assert len(v) == 1 and v[0].code == "ROW_OVERLAP"


def test_group_folds_have_no_subject_overlap(fake_data):
    folds = make_group_folds(fake_data, n_splits=3, seed=1)
    assert len(folds) == 3
    for f in folds:
        tr, ev = f.subjects(fake_data)
        assert check_subject_overlap(tr, ev) == []
        assert check_row_overlap(fake_data.row_id[f.train_idx], fake_data.row_id[f.eval_idx]) == []


def test_group_folds_cover_all_rows_exactly_once(fake_data):
    folds = make_group_folds(fake_data, n_splits=3, seed=1)
    covered = np.concatenate([f.eval_idx for f in folds])
    assert sorted(covered.tolist()) == list(range(fake_data.n))


def test_every_fold_contains_all_three_classes(fake_data):
    folds = make_group_folds(fake_data, n_splits=3, seed=7)
    desc = describe_folds(fake_data, folds)
    for side in ("train", "eval"):
        for cls in ("CN", "MCI", "Dem"):
            assert (desc[f"{side}_{cls}_subjects"] > 0).all(), f"{side}/{cls} 누락 fold 존재"


def test_subject_stratified_method_also_disjoint(fake_data):
    folds = make_group_folds(fake_data, method="subject_stratified", n_splits=3, seed=3)
    for f in folds:
        tr, ev = f.subjects(fake_data)
        assert check_subject_overlap(tr, ev) == []


def test_auditor_raises_on_subject_overlap(enforcing_auditor, fake_data):
    with pytest.raises(LeakageError, match="SUBJECT_OVERLAP"):
        enforcing_auditor.register_split(
            "f0",
            train_subjects=["a", "b"],
            eval_subjects=["b", "c"],
            train_row_ids=[0, 1],
            eval_row_ids=[2, 3],
        )


def test_row_level_split_leaks_subjects_by_design(fake_data):
    """실험 A의 행 단위 분할은 피험자 누수를 **의도적으로** 포함한다."""
    split = paper_row_split(fake_data, seed=0)
    shared = set(fake_data.subject[split.train_idx]) & set(fake_data.subject[split.test_idx])
    assert shared, "행 단위 분할인데 피험자 중복이 없다면 구현이 잘못된 것이다"


def test_observe_mode_measures_overlap_without_raising(observing_auditor, fake_data):
    split = paper_row_split(fake_data, seed=0)
    observing_auditor.register_split(
        "paper",
        train_subjects=fake_data.subject[split.train_idx],
        eval_subjects=fake_data.subject[split.test_idx],
        train_row_ids=fake_data.row_id[split.train_idx],
        eval_row_ids=fake_data.row_id[split.test_idx],
        require_disjoint_subjects=False,
    )
    obs = [o for o in observing_auditor.observations if o["kind"] == "subject_overlap_measured"]
    assert obs and obs[0]["n_shared_subjects"] > 0
    assert observing_auditor.violations == []
