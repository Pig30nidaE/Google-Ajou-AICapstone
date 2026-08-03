"""2026-08-03 결과 감사에서 발견된 증강 배선 회귀 테스트."""

from __future__ import annotations

import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from src.audit.leakage import LeakageAuditor
from src.augmentation import generators
from src.models.classifiers import TabNetClassifier, XGBoostClassifier
from src.splits.group_cv import make_group_folds


def _train_fold_with_auditor(data):
    fold = make_group_folds(
        data, method="subject_stratified", n_splits=3, seed=0
    )[0]
    auditor = LeakageAuditor(mode="enforce", name="wiring-test")
    auditor.register_split(
        fold.fold_id,
        train_subjects=data.subject[fold.train_idx],
        eval_subjects=data.subject[fold.eval_idx],
        train_row_ids=data.row_id[fold.train_idx],
        eval_row_ids=data.row_id[fold.eval_idx],
    )
    return data.take(fold.train_idx), fold, auditor


def test_generation_count_match_majority_is_not_noop():
    n = generators._resolve_n_synthetic(
        {"match_majority": True},
        n_real_minority=10,
        class_counts={0: 100, 1: 60, 2: 10},
        target=2,
    )
    assert n == 90


def test_generative_method_with_zero_rows_fails_closed(fake_data):
    train, fold, auditor = _train_fold_with_auditor(fake_data)
    with pytest.raises(ValueError, match="no-op"):
        generators.augment_train_fold(
            train,
            {"method": "smote", "target_class": "Dem", "smote": {"k_neighbors": 5}},
            auditor=auditor,
            fold_id=fold.fold_id,
        )


def test_vae_fit_is_recorded_before_generation(fake_data, monkeypatch):
    train, fold, auditor = _train_fold_with_auditor(fake_data)

    def fake_generate(real, _cfg, _sub_cfg, *, n_synthetic, **_kwargs):
        X = real.X.iloc[:n_synthetic].reset_index(drop=True)
        return X, X.copy(), {"method": "vae", "n_synthetic": n_synthetic}

    monkeypatch.setattr(generators, "_vae_generate", fake_generate)
    out = generators.augment_train_fold(
        train,
        {
            "method": "vae",
            "target_class": "Dem",
            "vae": {
                "latent_dim": 50,
                "n_synthetic": 1,
                "fit_scope": "train_dem_only",
            },
        },
        auditor=auditor,
        fold_id=fold.fold_id,
    )
    assert out.n_synthetic == 1
    events = auditor.folds[fold.fold_id].events
    assert sum(event["kind"] == "vae_fit" for event in events) == 1


def test_unsupported_vae_fit_scope_fails_closed(fake_data, monkeypatch):
    train, fold, auditor = _train_fold_with_auditor(fake_data)
    monkeypatch.setattr(
        generators,
        "_vae_generate",
        lambda *_args, **_kwargs: pytest.fail("지원하지 않는 scope에서 VAE를 호출했다"),
    )
    with pytest.raises(ValueError, match="train_dem_only"):
        generators.augment_train_fold(
            train,
            {
                "method": "vae",
                "target_class": "Dem",
                "vae": {"latent_dim": 50, "n_synthetic": 1, "fit_scope": "all_dem"},
            },
            auditor=auditor,
            fold_id=fold.fold_id,
        )


def test_xgboost_receives_class_weight_as_sample_weight(monkeypatch):
    captured = {}

    class FakeXGB:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _X, _y, *, sample_weight=None, **_kwargs):
            captured["sample_weight"] = np.asarray(sample_weight)

    module = ModuleType("xgboost")
    module.XGBClassifier = FakeXGB
    monkeypatch.setitem(sys.modules, "xgboost", module)

    model = XGBoostClassifier({"class_weight": {0: 1.0, 1: 2.0, 2: 3.0}})
    model.fit(np.zeros((3, 2)), np.array([0, 1, 2]))
    assert np.array_equal(captured["sample_weight"], np.array([1.0, 2.0, 3.0]))


def test_tabnet_receives_class_weight(monkeypatch):
    captured = {}

    class FakeTabNet:
        def __init__(self, **_kwargs):
            pass

        def fit(self, _X, _y, **kwargs):
            captured["weights"] = kwargs["weights"]

    package = ModuleType("pytorch_tabnet")
    tab_model = ModuleType("pytorch_tabnet.tab_model")
    tab_model.TabNetClassifier = FakeTabNet
    package.tab_model = tab_model
    monkeypatch.setitem(sys.modules, "pytorch_tabnet", package)
    monkeypatch.setitem(sys.modules, "pytorch_tabnet.tab_model", tab_model)

    weights = {0: 1.0, 1: 2.0, 2: 3.0}
    model = TabNetClassifier({"class_weight": weights})
    model.fit(pd.DataFrame(np.zeros((3, 2))), np.array([0, 1, 2]))
    assert captured["weights"] == weights
