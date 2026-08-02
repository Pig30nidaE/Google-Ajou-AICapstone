"""scaler·imputer·이상치 처리기가 train fold 밖 자료를 보지 않는지 검증한다."""

from __future__ import annotations

import numpy as np
import pytest

from src.audit.checks import check_fit_scope, check_preprocessing_after_split
from src.audit.leakage import LeakageAuditor, LeakageError
from src.preprocessing.outliers import make_outlier_handler
from src.preprocessing.pipeline import FoldPreprocessor
from src.splits.group_cv import make_group_folds
from src.utils.config import Config, ConfigError, validate_config


def _register(auditor, data, fold):
    auditor.register_split(
        fold.fold_id,
        train_subjects=data.subject[fold.train_idx],
        eval_subjects=data.subject[fold.eval_idx],
        train_row_ids=data.row_id[fold.train_idx],
        eval_row_ids=data.row_id[fold.eval_idx],
    )


def test_fit_scope_violation_detected():
    v = check_fit_scope("scaler", ["a", "z"], ["a", "b"])
    assert len(v) == 1 and v[0].code == "FIT_SCOPE"
    assert v[0].detail["n_outside"] == 1


def test_preprocessing_before_split_detected():
    v = check_preprocessing_after_split(False, "scaler")
    assert len(v) == 1 and v[0].code == "PREPROCESSING_BEFORE_SPLIT"


def test_pipeline_fits_only_on_train_fold(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    train = fake_data.take(fold.train_idx)

    pre = FoldPreprocessor(
        {"outlier": {"method": "percentile", "percentile": {"q": 0.1, "action": "clip"}},
         "preprocessing": {"scaler_scope": "train_real_only"}},
        auditor=enforcing_auditor, fold_id=fold.fold_id,
    )
    pre.fit(train)
    pre.fit_scaler(pre.apply_outlier(train))
    assert enforcing_auditor.violations == []

    events = enforcing_auditor.folds[fold.fold_id].events
    components = {e["component"] for e in events if e["kind"] == "fit"}
    assert {"outlier_detector", "imputer", "scaler"} <= components


def test_fitting_on_full_data_raises_in_enforce_mode(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    pre = FoldPreprocessor({}, auditor=enforcing_auditor, fold_id=fold.fold_id)
    with pytest.raises(LeakageError, match="FIT_SCOPE"):
        pre.fit(fake_data)   # 전체 데이터로 fit → eval 피험자가 섞인다


def test_fit_before_split_registration_raises(fake_data, enforcing_auditor):
    pre = FoldPreprocessor({}, auditor=enforcing_auditor, fold_id="never_registered")
    with pytest.raises(LeakageError, match="SPLIT_NOT_REGISTERED"):
        pre.fit(fake_data)


def test_scaler_statistics_come_from_train_only(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    train = fake_data.take(fold.train_idx)
    pre = FoldPreprocessor(
        {"preprocessing": {"scaler_scope": "train_real_only"}},
        auditor=enforcing_auditor, fold_id=fold.fold_id,
    )
    pre.fit(train)
    pre.fit_scaler(train)
    assert np.allclose(pre.scaler.mean_, train.X.to_numpy().mean(axis=0))
    # 평가 fold는 변환만 받으므로 평균이 0이 아니어야 한다.
    ev = pre.transform(fake_data.take(fold.eval_idx))
    assert not np.allclose(ev.X.to_numpy().mean(axis=0), 0.0, atol=1e-6)


def test_percentile_bounds_come_from_fit_data_only(fake_data):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    train = fake_data.take(fold.train_idx)
    h = make_outlier_handler({"method": "percentile", "percentile": {"q": 0.1}})
    h.fit(train.X, train.y)
    lo, hi = h._bounds[None]
    assert np.allclose(lo.to_numpy(), train.X.quantile(0.1).to_numpy())
    assert np.allclose(hi.to_numpy(), train.X.quantile(0.9).to_numpy())


def test_clip_action_preserves_row_count(fake_data):
    h = make_outlier_handler(
        {"method": "percentile", "percentile": {"q": 0.1, "action": "clip"}}
    )
    h.fit(fake_data.X, fake_data.y)
    res = h.transform(fake_data.X, fake_data.y)
    assert res.keep_mask.all() and res.n_dropped == 0
    assert res.n_clipped_cells > 0


def test_config_blocks_leaky_scopes_for_controlled_experiments():
    cfg = Config(
        {
            "experiment": {"name": "leakage_controlled_non_nested"},
            "preprocessing": {"scaler_scope": "all_data"},
            "audit": {"mode": "enforce"},
        }
    )
    with pytest.raises(ConfigError, match="scaler_scope"):
        validate_config(cfg)


def test_config_blocks_row_split_for_controlled_experiments():
    cfg = Config(
        {
            "experiment": {"name": "nested_subject_independent"},
            "split": {"unit": "row"},
            "audit": {"mode": "enforce"},
        }
    )
    with pytest.raises(ConfigError, match="split.unit=row"):
        validate_config(cfg)


def test_config_requires_enforce_mode_for_controlled_experiments():
    cfg = Config(
        {
            "experiment": {"name": "leakage_controlled_non_nested"},
            "audit": {"mode": "observe"},
        }
    )
    with pytest.raises(ConfigError, match="enforce"):
        validate_config(cfg)
