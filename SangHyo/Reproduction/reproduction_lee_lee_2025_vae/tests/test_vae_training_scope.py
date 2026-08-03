"""VAE가 현재 fold의 실제 Dem 자료로만 학습되는지 검증한다."""

from __future__ import annotations

import numpy as np
import pytest

from src.audit.checks import check_vae_fit_scope
from src.audit.leakage import LeakageError
from src.augmentation.provenance import SyntheticProvenance, subject_set_hash
from src.data.schema import CLASS_TO_CODE
from src.splits.group_cv import make_group_folds


def _register(auditor, data, fold):
    auditor.register_split(
        fold.fold_id,
        train_subjects=data.subject[fold.train_idx],
        eval_subjects=data.subject[fold.eval_idx],
        train_row_ids=data.row_id[fold.train_idx],
        eval_row_ids=data.row_id[fold.eval_idx],
    )


def test_vae_fit_outside_train_detected():
    v = check_vae_fit_scope(["a", "zz"], [2, 2], ["a", "b"], expected_label=2)
    assert any(x.code == "VAE_FIT_SCOPE" for x in v)


def test_vae_fit_wrong_label_detected():
    v = check_vae_fit_scope(["a"], [2, 0], ["a", "b"], expected_label=2)
    assert any(x.code == "VAE_FIT_LABEL" for x in v)


def test_vae_fit_clean():
    assert check_vae_fit_scope(["a"], [2, 2], ["a", "b"], expected_label=2) == []


def test_auditor_accepts_train_only_dem_fit(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    train = fake_data.take(fold.train_idx)
    dem = train.take(train.y == CLASS_TO_CODE["Dem"])
    assert dem.n > 0
    enforcing_auditor.record_vae_fit(
        fold.fold_id, subjects=dem.subject, labels=dem.y,
        row_ids=dem.row_id,
        expected_label=CLASS_TO_CODE["Dem"], n_rows=dem.n,
    )
    assert enforcing_auditor.violations == []


def test_auditor_rejects_vae_fit_on_eval_row_even_with_train_subject_label(
    fake_data, enforcing_auditor
):
    """행 단위 split의 subject 중복이 있어도 원시 eval 행 사용을 놓치지 않는다."""
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    train = fake_data.take(fold.train_idx)
    dem = train.take(train.y == CLASS_TO_CODE["Dem"])
    with pytest.raises(LeakageError, match="VAE_FIT_ROW_SCOPE"):
        enforcing_auditor.record_vae_fit(
            fold.fold_id,
            subjects=np.concatenate([dem.subject, dem.subject[:1]]),
            labels=np.full(dem.n + 1, CLASS_TO_CODE["Dem"]),
            row_ids=np.concatenate([dem.row_id, fake_data.row_id[fold.eval_idx][:1]]),
            expected_label=CLASS_TO_CODE["Dem"],
            n_rows=dem.n + 1,
        )


def test_auditor_rejects_vae_fit_on_all_dem(fake_data, enforcing_auditor):
    """전체 Dem(=평가 fold 포함)으로 VAE를 학습하면 즉시 예외."""
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    all_dem = fake_data.take(fake_data.y == CLASS_TO_CODE["Dem"])
    with pytest.raises(LeakageError, match="VAE_FIT_SCOPE"):
        enforcing_auditor.record_vae_fit(
            fold.fold_id, subjects=all_dem.subject, labels=all_dem.y,
            row_ids=all_dem.row_id,
            expected_label=CLASS_TO_CODE["Dem"], n_rows=all_dem.n,
        )


def test_auditor_rejects_vae_fit_on_all_classes(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    train = fake_data.take(fold.train_idx)
    with pytest.raises(LeakageError, match="VAE_FIT_LABEL"):
        enforcing_auditor.record_vae_fit(
            fold.fold_id, subjects=train.subject, labels=train.y,
            row_ids=train.row_id,
            expected_label=CLASS_TO_CODE["Dem"], n_rows=train.n,
        )


def test_synthetic_source_subjects_must_be_train_only(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    with pytest.raises(LeakageError, match="SYNTHETIC_SOURCE_SCOPE"):
        enforcing_auditor.record_synthetic(
            fold.fold_id, source_subjects=fake_data.subject[fold.eval_idx], n_rows=100
        )


def test_synthetic_cannot_target_eval(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    with pytest.raises(LeakageError, match="SYNTHETIC_TARGET"):
        enforcing_auditor.record_synthetic(
            fold.fold_id, source_subjects=fake_data.subject[fold.train_idx],
            n_rows=10, target="test",
        )


def test_provenance_records_source_subject_hash():
    prov = SyntheticProvenance.create(
        source_class="Dem", generator="vae", generator_seed=1,
        generator_config_hash="abc123", source_subjects=["s1", "s2", "s1"],
        n_source_rows=40, source_outer_fold="outer_r0_f0",
    )
    assert prov.is_synthetic is True
    assert prov.n_source_subjects == 2
    assert prov.source_subject_hash == subject_set_hash(["s2", "s1"])
    frame = prov.to_frame(5)
    assert len(frame) == 5 and frame["source_outer_fold"].nunique() == 1


def test_provenance_hash_is_order_independent():
    assert subject_set_hash(["b", "a"]) == subject_set_hash(["a", "b", "a"])
    assert subject_set_hash(["a"]) != subject_set_hash(["b"])


def test_vae_config_requires_explicit_latent_dim():
    from src.augmentation.vae import VAEConfig

    with pytest.raises(ValueError, match="latent_dim"):
        VAEConfig.from_dict({"epochs": 10}, input_dim=46)


def test_vae_config_carries_paper_reported_defaults():
    from src.augmentation.vae import VAEConfig

    cfg = VAEConfig.from_dict({"latent_dim": 500}, input_dim=46)
    assert cfg.encoder_hidden == (512, 256)
    assert cfg.decoder_hidden == (256, 512)
    assert cfg.dropout == 0.3
    assert cfg.learning_rate == 1e-4
    assert cfg.batch_norm is True
    assert cfg.recon_reduction == "mean_per_feature"
    assert cfg.kl_reduction == "mean"
