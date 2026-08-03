"""outer test가 early stopping·모델 선택·임계값 선택에 쓰이지 않는지 검증한다.

사용자 지시 5절(실험 C)·8절·11절.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
            fold.fold_id,
            subjects=fake_data.subject[fold.eval_idx],
            row_ids=fake_data.row_id[fold.eval_idx],
        )


def test_auditor_accepts_early_stopping_inside_train(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    enforcing_auditor.record_early_stopping(
        fold.fold_id,
        subjects=fake_data.subject[fold.train_idx][:10],
        row_ids=fake_data.row_id[fold.train_idx][:10],
    )
    assert enforcing_auditor.violations == []


def test_explicit_validation_row_scope_rejects_outer_eval_even_when_subject_matches(fake_data):
    """행 split처럼 subject가 겹쳐도 test 행은 early stopping에 쓸 수 없다."""
    from src.audit.leakage import LeakageAuditor

    train_idx = np.arange(0, fake_data.n, 3)
    valid_idx = np.arange(1, fake_data.n, 3)
    eval_idx = np.arange(2, fake_data.n, 3)
    auditor = LeakageAuditor(mode="enforce", name="explicit-validation-row-scope")
    auditor.register_split(
        "row_split",
        train_subjects=fake_data.subject[train_idx],
        eval_subjects=fake_data.subject[eval_idx],
        train_row_ids=fake_data.row_id[train_idx],
        eval_row_ids=fake_data.row_id[eval_idx],
        validation_subjects=fake_data.subject[valid_idx],
        validation_row_ids=fake_data.row_id[valid_idx],
        require_disjoint_subjects=False,
    )
    auditor.record_early_stopping(
        "row_split",
        subjects=fake_data.subject[valid_idx],
        row_ids=fake_data.row_id[valid_idx],
    )
    with pytest.raises(LeakageError, match="EARLY_STOPPING_ROW_SCOPE"):
        auditor.record_early_stopping(
            "row_split",
            subjects=fake_data.subject[eval_idx[:1]],
            row_ids=fake_data.row_id[eval_idx[:1]],
        )


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


def test_model_without_early_stopping_uses_every_training_row(
    fake_data, enforcing_auditor, monkeypatch
):
    """XGBoost 같은 모델은 validation 명목으로 train 20%를 버리면 안 된다."""
    from src.models import registry

    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    _register(enforcing_auditor, fake_data, fold)
    train = fake_data.take(fold.train_idx)
    captured = {}

    class FakeModel:
        name = "fake-no-es"
        uses_early_stopping = False
        fit_log = {}

        def fit(self, X, y, *, eval_set=None):
            captured.update(n=len(y), eval_set=eval_set)

    monkeypatch.setattr(registry, "make_model", lambda *_args, **_kwargs: FakeModel())
    registry.fit_classifier(
        "xgboost", train, {}, auditor=enforcing_auditor, fold_id=fold.fold_id
    )
    assert captured == {"n": train.n, "eval_set": None}


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
    # none/smote에서 VAE latent 축은 비활성이므로 24개 명목 조합이 아니라
    # 16개 유효 파이프라인만 남는다.
    all_candidates = enumerate_candidates(space, max_evals=100, seed=0)
    assert len(all_candidates) == 16
    assert all(
        "augmentation.vae.latent_dim" not in candidate
        for candidate in all_candidates
        if candidate["augmentation.method"] != "vae"
    )
    got = enumerate_candidates(space, max_evals=5, seed=0)
    assert len(got) == 5 and len({tuple(sorted(c.items())) for c in got}) == 5
    assert {c["augmentation.method"] for c in got} == {"none", "vae", "smote"}


def test_enumerate_candidates_balances_augmentation_arms():
    from src.experiments.nested_cv import enumerate_candidates

    space = {
        "classifier": ["xgboost", "dnn", "tabnet", "wide_deep"],
        "augmentation.method": ["none", "vae", "class_weight", "smote"],
        "augmentation.vae.latent_dim": [50, 500],
        "augmentation.vae.ratio_to_real": [1.0, 3.0, 10.0],
    }
    got = enumerate_candidates(space, max_evals=8, seed=0)
    counts = {
        arm: sum(c["augmentation.method"] == arm for c in got)
        for arm in ("none", "vae", "class_weight", "smote")
    }
    assert counts == {"none": 2, "vae": 2, "class_weight": 2, "smote": 2}
    classifier_counts = {
        classifier: sum(c["classifier"] == classifier for c in got)
        for classifier in ("xgboost", "dnn", "tabnet", "wide_deep")
    }
    assert classifier_counts == {"xgboost": 2, "dnn": 2, "tabnet": 2, "wide_deep": 2}

    covered = enumerate_candidates(space, max_evals=16, seed=0)
    assert {
        (candidate["classifier"], candidate["augmentation.method"])
        for candidate in covered
    } == {
        (classifier, augmentation)
        for classifier in ("xgboost", "dnn", "tabnet", "wide_deep")
        for augmentation in ("none", "vae", "class_weight", "smote")
    }

    shipped_budget = enumerate_candidates(space, max_evals=24, seed=0)
    per_classifier = [
        sum(c["classifier"] == classifier for c in shipped_budget)
        for classifier in ("xgboost", "dnn", "tabnet", "wide_deep")
    ]
    per_augmentation = [
        sum(c["augmentation.method"] == augmentation for c in shipped_budget)
        for augmentation in ("none", "vae", "class_weight", "smote")
    ]
    assert max(per_classifier) - min(per_classifier) <= 1
    assert max(per_augmentation) - min(per_augmentation) <= 1


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


# ── 산출물 영속성 회귀 테스트 ─────────────────────────────────────────────────
def test_output_root_prefers_mydrive_when_mounted(tmp_path, monkeypatch):
    """Colab 런타임(/content)은 세션 종료 시 삭제되므로 Drive를 우선해야 한다."""
    import run as R

    fake_drive = tmp_path / "MyDrive"
    fake_drive.mkdir()
    monkeypatch.setattr(R, "COLAB_MYDRIVE", fake_drive)
    monkeypatch.delenv(R.OUTPUT_ROOT_ENV_VAR, raising=False)

    out = R._resolve_output_root({"PROJECT_ROOT": "/content/Google-Ajou-AICapstone"}, None, None)
    assert out == fake_drive / R.DRIVE_OUTPUT_SUBDIR
    assert str(out).startswith(str(fake_drive))


def test_output_root_falls_back_to_repo_when_no_drive(tmp_path, monkeypatch):
    import run as R

    monkeypatch.setattr(R, "COLAB_MYDRIVE", tmp_path / "nonexistent_drive")
    monkeypatch.delenv(R.OUTPUT_ROOT_ENV_VAR, raising=False)
    out = R._resolve_output_root({}, None, None)
    assert out == (Path(R.REPO_ROOT) / "outputs").resolve()


def test_output_root_explicit_and_env_take_priority(tmp_path, monkeypatch):
    import run as R

    fake_drive = tmp_path / "MyDrive"
    fake_drive.mkdir()
    monkeypatch.setattr(R, "COLAB_MYDRIVE", fake_drive)

    monkeypatch.setenv(R.OUTPUT_ROOT_ENV_VAR, str(tmp_path / "from_env"))
    assert R._resolve_output_root({}, None, None) == tmp_path / "from_env"
    # --out-root가 환경변수보다 우선
    assert R._resolve_output_root({}, str(tmp_path / "explicit"), None) == tmp_path / "explicit"


def test_session_dirs_never_overwrite_each_other(tmp_path, monkeypatch):
    """전체 실행을 두 번 해도 이전 스윕 결과가 남아 있어야 한다."""
    import run as R

    base = tmp_path / "base"
    seen = iter(["20260101_000000", "20260101_000001"])
    monkeypatch.setattr(R.time, "strftime", lambda *_a, **_k: next(seen))

    first = R._make_session_dir(base, tag="full")
    second = R._make_session_dir(base, tag="full")
    assert first != second
    assert first.exists() and second.exists()
    assert (base / "LATEST.txt").read_text().strip() == second.name
