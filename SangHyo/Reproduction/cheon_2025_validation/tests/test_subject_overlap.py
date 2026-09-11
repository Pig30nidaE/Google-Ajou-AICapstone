"""G/N outer train-test overlap = 0; N inner train-validation overlap = 0; outer-test subjects never inside inner."""
import numpy as np
import pandas as pd
import pytest

from conftest import EXP_DIR, make_experiment, synthetic
from cheon.splits import assert_subject_disjoint, record_folds, record_overlap_subjects, subject_folds


def test_subject_folds_are_disjoint_and_cover_every_subject():
    X, y, g = synthetic()
    folds = subject_folds(y, g, 5, seed=1)
    seen = []
    for tr, te in folds:
        info = assert_subject_disjoint(g, tr, te)
        assert info["subject_overlap"] == 0
        seen += list(set(g[te]))
    assert sorted(seen) == sorted(set(g))  # each subject in exactly one test fold


def test_nested_inner_folds_exclude_outer_test_subjects():
    X, y, g = synthetic()
    for f, (otr, ote) in enumerate(subject_folds(y, g, 4, seed=3)):
        test_subjects = set(g[ote])
        for itr, iva in subject_folds(y[otr], g[otr], 3, seed=100 + f):
            itr_g, iva_g = otr[itr], otr[iva]
            assert_subject_disjoint(g, itr_g, iva_g)
            assert not (set(g[itr_g]) | set(g[iva_g])) & test_subjects


def test_record_folds_do_overlap_subjects_by_construction():
    """R's record-level split is expected to share subjects across folds (documented leak, A15/A10)."""
    X, y, g = synthetic()
    tr, te = record_folds(y, 5, seed=0)[0]
    assert record_overlap_subjects(g, tr, te) > 0


def test_assert_subject_disjoint_raises_on_overlap():
    g = np.array(["a", "a", "b", "b"])
    with pytest.raises(AssertionError):
        assert_subject_disjoint(g, np.array([0, 2]), np.array([1, 3]))


def test_pipeline_G_and_N_log_zero_overlap(tmp_path, monkeypatch):
    import cheon.pipeline as pl
    monkeypatch.setattr(pl, "paper_anchored_grid", lambda: [{}, {"num_leaves": 8, "n_estimators": 30}])
    exG = make_experiment(tmp_path / "g")
    exG.run_subject_independent()
    s = pd.read_csv(exG.out / "split_summary.csv")
    assert (s["subject_overlap"] == 0).all() and len(s) == 3
    exN = make_experiment(tmp_path / "n", experiment="nested_cv")
    exN.run_nested()
    s = pd.read_csv(exN.out / "split_summary.csv")
    assert (s["subject_overlap"] == 0).all()
    assert set(s["level"]) == {"outer", "inner"}
    fm = pd.read_csv(exN.out / "outer_fold_metrics.csv")
    assert (fm["subject_overlap"] == 0).all()


@pytest.mark.parametrize("name", ["subject_independent", "nested_cv"])
def test_real_run_artifacts_have_zero_overlap(name):
    p = EXP_DIR / "results" / name / "split_summary.csv"
    if not p.exists():
        pytest.skip("full run not present yet")
    s = pd.read_csv(p)
    assert (s["subject_overlap"] == 0).all()
    fm = pd.read_csv(EXP_DIR / "results" / name / "fold_metrics.csv")
    assert (fm["subject_overlap"] == 0).all()
