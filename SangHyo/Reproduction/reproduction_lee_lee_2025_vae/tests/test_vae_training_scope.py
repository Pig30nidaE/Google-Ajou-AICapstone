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
    # 손실 축척은 논문 미보고 항목이다. 2026-08-03 감사에서 mean/mean 조합이 생성 분산을
    # 붕괴시킨 것이 확인되어 표준 ELBO(sum/sum)로 교정했다 — 아래 전용 테스트가 이를 고정한다.
    assert cfg.recon_reduction == "sum"
    assert cfg.kl_reduction == "sum"


# ── 2026-08-03 생성 붕괴 회귀 테스트 ──────────────────────────────────────────
def test_default_loss_is_standard_elbo():
    """recon=mean_per_feature + kl=mean 조합은 실효 beta를 1/latent_dim으로 떨어뜨려
    생성 분산을 붕괴시켰다 (실행 20260803_041244_full: 표준편차 비 0.30).
    기본값이 다시 그 조합으로 돌아가지 않도록 고정한다."""
    from src.augmentation.vae import VAEConfig

    cfg = VAEConfig.from_dict({"latent_dim": 500}, input_dim=46)
    assert cfg.recon_reduction == "sum"
    assert cfg.kl_reduction == "sum"
    assert not (cfg.recon_reduction == "mean_per_feature" and cfg.kl_reduction == "mean")


def test_shipped_configs_do_not_use_collapsing_loss_combo():
    """실제로 실행되는 config가 붕괴 조합을 쓰지 않는지 검사한다."""
    import yaml
    from pathlib import Path

    from src.utils.config import load_config

    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(cfg_dir.glob("*.yaml")):
        cfg = load_config(path)
        if cfg.get_path("augmentation.method") != "vae":
            continue
        recon = cfg.get_path("augmentation.vae.recon_reduction")
        kl = cfg.get_path("augmentation.vae.kl_reduction")
        assert not (recon == "mean_per_feature" and kl == "mean"), (
            f"{path.name}: recon=mean_per_feature + kl=mean 은 실효 beta=1/latent_dim이라 "
            "생성 분산이 붕괴한다"
        )


def test_generation_fidelity_flags_variance_collapse():
    """분산이 죽은 합성자료를 진단이 잡아내는지."""
    import numpy as np
    import pandas as pd

    from src.augmentation.generators import _generation_fidelity
    from src.data.schema import PAPER_FEATURES

    rng = np.random.default_rng(0)
    cols = list(PAPER_FEATURES)
    real = pd.DataFrame(rng.normal(0, 1, size=(200, len(cols))), columns=cols)
    collapsed = pd.DataFrame(rng.normal(0, 0.2, size=(500, len(cols))), columns=cols)
    healthy = pd.DataFrame(rng.normal(0, 1.0, size=(500, len(cols))), columns=cols)

    bad = _generation_fidelity(collapsed, real)
    assert bad["variance_collapse_suspected"] is True
    assert bad["std_ratio_median"] < 0.5
    assert bad["n_features_below_half"] == len(cols)

    good = _generation_fidelity(healthy, real)
    assert good["variance_collapse_suspected"] is False


def test_tabnet_early_stops_on_logloss_not_accuracy():
    """accuracy로 early stopping하면 CN 64% 데이터에서 '전부 CN' 해에 멈춘다
    (실행 20260803_041244_full: TabNet balanced accuracy 정확히 0.3333)."""
    from src.models.classifiers import TabNetClassifier

    # 내장 'logloss'는 sklearn log_loss를 labels 없이 호출해, validation에 클래스가
    # 빠지면 죽는다 (TSTR은 실제 Dem을 전부 제거하므로 그 상황이 설계상 발생).
    # 그래서 labels를 명시하는 자체 metric을 쓴다.
    from src.models.classifiers import _TABNET_LOGLOSS_NAME

    assert TabNetClassifier.DEFAULTS["eval_metric"] == [_TABNET_LOGLOSS_NAME]
    assert "logloss" not in TabNetClassifier.DEFAULTS["eval_metric"]
