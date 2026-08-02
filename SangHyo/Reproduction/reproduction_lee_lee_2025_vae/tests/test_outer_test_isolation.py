"""outer test가 early stopping·모델 선택·임계값 선택에 쓰이지 않는지 검증한다.

사용자 지시 5절(실험 C)·8절·11절.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from src.audit.checks import check_early_stopping_scope, check_selection_scope
from src.audit.leakage import LeakageError
from src.models.base import make_internal_validation
from src.splits.group_cv import make_group_folds


def _register(auditor, data, fold):
    auditor.register_split(
        fold.fold_id,
        train_subjects=data.subject[fold.train_idx],
        eval_subjects=data.subject[fold.eval_idx],
        train_row_ids=data.row_id[fold.train_idx],
        eval_row_ids=data.row_id[fold.eval_idx],
    )


def test_early_stopping_outside_train_detected():
    v = check_early_stopping_scope(["a", "outer_test_subj"], ["a", "b"])
    assert len(v) == 1 and v[0].code == "EARLY_STOPPING_SCOPE"


def test_selection_outside_train_detected():
    v = check_selection_scope("latent_dim", ["zz"], ["a", "b"])
    assert len(v) == 1 and v[0].code == "SELECTION_SCOPE"
    assert "latent_dim" in v[0].message


def test_auditor_rejects_early_stopping_on_outer_test(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    with pytest.raises(LeakageError, match="EARLY_STOPPING_SCOPE"):
        enforcing_auditor.record_early_stopping(
            fold.fold_id, subjects=fake_data.subject[fold.eval_idx]
        )


def test_auditor_accepts_early_stopping_inside_train(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    enforcing_auditor.record_early_stopping(
        fold.fold_id, subjects=fake_data.subject[fold.train_idx][:10]
    )
    assert enforcing_auditor.violations == []


@pytest.mark.parametrize(
    "what", ["outlier_threshold", "latent_dim", "synthetic_ratio", "classifier", "decision_threshold"]
)
def test_auditor_rejects_selection_using_outer_test(fake_data, enforcing_auditor, what):
    """사용자 지시 5절의 'outer test에 절대 사용하지 말 것' 전 항목."""
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    with pytest.raises(LeakageError, match="SELECTION_SCOPE"):
        enforcing_auditor.record_selection(
            what, fold.fold_id, subjects=fake_data.subject[fold.eval_idx]
        )


def test_internal_validation_is_subject_disjoint(fake_data):
    iv = make_internal_validation(
        fake_data.y, fake_data.subject, fake_data.is_synthetic,
        fraction=0.2, split_by="subject", seed=0,
    )
    tr_subs = set(fake_data.subject[iv.train_idx])
    val_subs = set(fake_data.subject[iv.val_idx])
    assert val_subs and not (tr_subs & val_subs)


def test_internal_validation_excludes_synthetic_rows(fake_data):
    from src.augmentation.provenance import SyntheticProvenance

    prov = SyntheticProvenance.create(
        source_class="Dem", generator="vae", generator_seed=0,
        generator_config_hash="x", source_subjects=fake_data.subject[:2], n_source_rows=2,
    )
    aug = fake_data.append_synthetic(fake_data.X.iloc[:40].reset_index(drop=True), 2, prov.to_frame(40))
    iv = make_internal_validation(
        aug.y, aug.subject, aug.is_synthetic, fraction=0.2, split_by="subject", seed=0
    )
    assert not aug.is_synthetic[iv.val_idx].any(), "합성행이 early stopping validation에 들어갔다"


def test_internal_validation_indices_partition_all_rows(fake_data):
    iv = make_internal_validation(
        fake_data.y, fake_data.subject, fake_data.is_synthetic, fraction=0.2, seed=0
    )
    assert sorted(np.concatenate([iv.train_idx, iv.val_idx]).tolist()) == list(range(fake_data.n))


def test_nested_inner_folds_never_touch_outer_eval():
    """inner fold의 어떤 행도 outer eval에 속하지 않는다.

    실제 코호트 비율(CN 111 / MCI 51 / Dem 12)을 축소해 재현한다. Dem 12명은
    outer 3-fold에서 train 8명이 되고 inner 3-fold에서 fold당 2~3명이 된다.
    """
    from tests.conftest import make_synthetic_lifelog

    data = make_synthetic_lifelog(n_subjects_per_class=(36, 18, 12), records_per_subject=6)
    outer = make_group_folds(data, n_splits=3, seed=0, prefix="outer")
    for of in outer:
        outer_train = data.take(of.train_idx)
        outer_eval_subjects = set(data.subject[of.eval_idx])
        inner = make_group_folds(outer_train, n_splits=3, seed=0, prefix="inner")
        for f in inner:
            for idx in (f.train_idx, f.eval_idx):
                assert not (set(outer_train.subject[idx]) & outer_eval_subjects)


def test_enumerate_candidates_respects_max_evals():
    from src.experiments.nested_cv import enumerate_candidates

    space = {
        "classifier": ["xgboost", "dnn", "tabnet", "wide_deep"],
        "augmentation.method": ["none", "vae", "smote"],
        "augmentation.vae.latent_dim": [50, 500],
    }
    assert len(enumerate_candidates(space, max_evals=100, seed=0)) == 24   # 전수
    got = enumerate_candidates(space, max_evals=5, seed=0)
    assert len(got) == 5 and len({tuple(sorted(c.items())) for c in got}) == 5


def test_enumerate_candidates_empty_space():
    from src.experiments.nested_cv import enumerate_candidates

    assert enumerate_candidates({}, max_evals=10, seed=0) == [{}]


def test_apply_candidate_sets_nested_keys():
    from src.experiments.nested_cv import _apply_candidate

    base = {"augmentation": {"method": "none", "vae": {"latent_dim": 500}}, "outlier": {"method": "none"}}
    cfg, model, aug = _apply_candidate(
        base,
        {"classifier": "dnn", "augmentation.method": "vae", "augmentation.vae.latent_dim": 50,
         "outlier.method": "isolation_forest"},
    )
    assert model == "dnn" and aug == "vae"
    assert cfg["augmentation"]["vae"]["latent_dim"] == 50
    assert cfg["outlier"]["method"] == "isolation_forest"
    assert base["augmentation"]["vae"]["latent_dim"] == 500, "base config가 변형되었다"


# ── base.ipynb 연동 회귀 테스트 ────────────────────────────────────────────────
def test_kernel_argv_is_detected_and_ignored(monkeypatch):
    """base.ipynb Cell 5 경로에서 Jupyter 커널 argv를 argparse에 넘기면 안 된다.

    이걸 놓치면 노트북 실행이 'unrecognized arguments: -f ...'로 즉사한다.
    """
    import run as R

    monkeypatch.setattr(
        sys, "argv",
        ["/usr/local/lib/python3/dist-packages/colab_kernel_launcher.py",
         "-f", "/root/.local/share/jupyter/runtime/kernel-abcd.json"],
    )
    monkeypatch.delenv(R.ARGS_ENV_VAR, raising=False)
    assert R._running_under_kernel() is True
    # None = "전체 파이프라인을 한 번에 실행하라"
    assert R._resolve_argv(None) is None


def test_env_var_overrides_kernel_argv(monkeypatch):
    import run as R

    monkeypatch.setattr(
        sys, "argv",
        ["/usr/local/lib/python3/dist-packages/colab_kernel_launcher.py",
         "-f", "/root/.local/share/jupyter/runtime/kernel-abcd.json"],
    )
    monkeypatch.setenv(R.ARGS_ENV_VAR, "--config configs/base.yaml --dry-run")
    assert R._resolve_argv(None) == ["--config", "configs/base.yaml", "--dry-run"]


def test_explicit_argv_always_wins(monkeypatch):
    import run as R

    monkeypatch.setenv(R.ARGS_ENV_VAR, "--inspect-data")
    assert R._resolve_argv(["--dry-run"]) == ["--dry-run"]


def test_run_all_configs_exist():
    """run_all이 참조하는 config가 실제로 존재해야 한다."""
    import run as R
    from pathlib import Path

    for kind, rel in R.RUN_ALL_CONFIGS.items():
        assert (Path(R.REPO_ROOT) / rel).exists(), f"{kind}: {rel} 없음"
    assert (Path(R.REPO_ROOT) / R.RUN_ALL_A_PRIMARY).exists()


def test_shell_argv_still_parsed_normally(monkeypatch):
    import run as R

    monkeypatch.setattr(sys, "argv", ["run.py", "--dry-run", "--seed", "7"])
    monkeypatch.delenv(R.ARGS_ENV_VAR, raising=False)
    monkeypatch.setattr(R, "_running_under_kernel", lambda: False)
    assert R._resolve_argv(None) == ["--dry-run", "--seed", "7"]
