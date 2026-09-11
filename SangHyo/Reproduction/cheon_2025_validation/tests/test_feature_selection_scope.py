"""G: SHAP ranking sees the training fold only. N: inner rankings see inner-train only (never outer-test)."""
import numpy as np
import pandas as pd

import cheon.pipeline as pl
from conftest import make_experiment


class Spy:
    def __init__(self, ex):
        self.ex, self.calls = ex, []
        self.orig = pl.rank_features
        self.row_index = {ex.X[i].tobytes(): i for i in range(len(ex.X))}

    def __call__(self, X_fit, y_fit, feature_names, params, seed, n_threads, X_explain=None):
        idx = np.array([self.row_index[X_fit[i].tobytes()] for i in range(len(X_fit))])
        self.calls.append(idx)
        return self.orig(X_fit, y_fit, feature_names, params, seed, n_threads, X_explain)


def test_G_ranking_uses_training_fold_only(tmp_path, monkeypatch):
    ex = make_experiment(tmp_path)
    spy = Spy(ex)
    monkeypatch.setattr(pl, "rank_features", spy)
    ex.run_subject_independent()
    folds = ex.folds_for(0)
    assert len(spy.calls) == len(folds)
    for (tr, te), idx in zip(folds, spy.calls):
        assert set(idx) == set(tr)
        assert not set(ex.groups[idx]) & set(ex.groups[te])


def test_N_inner_rankings_never_see_outer_test_subjects(tmp_path, monkeypatch):
    ex = make_experiment(tmp_path, experiment="nested_cv")
    monkeypatch.setattr(pl, "paper_anchored_grid", lambda: [{}, {"num_leaves": 8, "n_estimators": 30}])
    spy = Spy(ex)
    monkeypatch.setattr(pl, "rank_features", spy)
    ex.run_nested()
    outer = ex.folds_for(0)
    n_inner = ex.cfg["inner_splits"]
    assert len(spy.calls) == len(outer) * (n_inner + 1)  # per outer fold: n_inner inner rankings + 1 outer-train ranking
    c = 0
    for otr, ote in outer:
        test_subjects = set(ex.groups[ote])
        for _ in range(n_inner + 1):
            idx = spy.calls[c]; c += 1
            assert set(idx) <= set(otr)
            assert not set(ex.groups[idx]) & test_subjects


def test_R_ranking_is_dataset_wide_by_design(tmp_path, monkeypatch):
    """Documents A15: the paper-faithful arm ranks on all records (this is the leak G/N remove)."""
    ex = make_experiment(tmp_path, experiment="reproduction", split_unit="record",
                         feature_selection={"scope": "dataset", "max_k": 4, "k_fixed": None})
    spy = Spy(ex)
    monkeypatch.setattr(pl, "rank_features", spy)
    ex.run_reproduction()
    assert all(len(idx) == len(ex.X) for idx in spy.calls)


def test_N_selected_k_and_params_come_from_inner_folds(tmp_path, monkeypatch):
    ex = make_experiment(tmp_path, experiment="nested_cv")
    monkeypatch.setattr(pl, "paper_anchored_grid", lambda: [{}, {"num_leaves": 8, "n_estimators": 30}])
    ex.run_nested()
    inner = pd.read_csv(ex.out / "inner_selection.csv")
    hp = pd.read_csv(ex.out / "selected_hyperparameters_by_fold.csv")
    for _, row in hp.iterrows():
        ks = inner[(inner.outer_fold == row.outer_fold) & (inner.stage == "k_selection")].groupby("k")["auc"].mean()
        assert int(ks.idxmax()) == int(row.k_selected)
        hs = inner[(inner.outer_fold == row.outer_fold) & (inner.stage == "hp_selection")].groupby("grid_index")["auc"].mean()
        assert abs(float(hs.max()) - float(row.inner_mean_auc_at_selected_params)) < 1e-9  # CSV float round-trip
