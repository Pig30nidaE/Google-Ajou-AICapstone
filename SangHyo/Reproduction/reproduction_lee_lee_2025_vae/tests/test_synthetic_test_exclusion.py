"""합성행이 평가에 절대 들어가지 않고, 독립 피험자로 집계되지 않는지 검증한다.

사용자 지시 11·15·16 / synthetic_data_risk.md §2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.audit.checks import (
    check_no_synthetic_in_eval,
    check_subject_aggregation_excludes_synthetic,
)
from src.audit.leakage import LeakageError
from src.augmentation.provenance import SyntheticProvenance
from src.data.schema import SYNTHETIC_SUBJECT_SENTINEL
from src.evaluation.aggregate import aggregate_to_subject
from src.splits.group_cv import make_group_folds


def _append_fake_synthetic(data, k=20, label=2):
    prov = SyntheticProvenance.create(
        source_class="Dem", generator="vae", generator_seed=0,
        generator_config_hash="x", source_subjects=data.subject[:3], n_source_rows=3,
    )
    X = data.X.iloc[:k].reset_index(drop=True)
    return data.append_synthetic(X, label, prov.to_frame(k))


def test_synthetic_in_eval_detected():
    v = check_no_synthetic_in_eval([False, True, False])
    assert len(v) == 1 and v[0].code == "SYNTHETIC_IN_EVAL"
    assert v[0].detail["n_synthetic"] == 1


def test_clean_eval_passes():
    assert check_no_synthetic_in_eval([False, False]) == []


def test_appended_synthetic_gets_sentinel_subject_and_negative_row_id(fake_data):
    aug = _append_fake_synthetic(fake_data, k=20)
    assert aug.n == fake_data.n + 20
    syn = aug.take(aug.is_synthetic)
    assert (syn.subject == SYNTHETIC_SUBJECT_SENTINEL).all()
    assert (syn.row_id < 0).all()
    # 실제 row_id와 절대 충돌하지 않는다.
    assert not set(syn.row_id.tolist()) & set(fake_data.row_id.tolist())


def test_synthetic_rows_are_not_counted_as_subjects(fake_data):
    aug = _append_fake_synthetic(fake_data, k=50)
    assert len(aug.subjects()) == len(fake_data.subjects())
    assert SYNTHETIC_SUBJECT_SENTINEL not in set(aug.subjects())


def test_subject_aggregation_rejects_sentinel():
    v = check_subject_aggregation_excludes_synthetic(["a", SYNTHETIC_SUBJECT_SENTINEL])
    assert len(v) == 1 and v[0].code == "SYNTHETIC_AS_SUBJECT"


def test_aggregate_drops_synthetic_rows(fake_data):
    aug = _append_fake_synthetic(fake_data, k=30)
    proba = np.full((aug.n, 3), 1 / 3)
    out = aggregate_to_subject(aug.subject, aug.y, proba, is_synthetic=aug.is_synthetic)
    assert len(out.subject) == len(fake_data.subjects())
    assert SYNTHETIC_SUBJECT_SENTINEL not in set(out.subject.tolist())
    assert out.n_records.sum() == fake_data.n


def test_aggregate_raises_if_synthetic_not_flagged(fake_data):
    aug = _append_fake_synthetic(fake_data, k=10)
    proba = np.full((aug.n, 3), 1 / 3)
    with pytest.raises(ValueError, match="SYNTHETIC_AS_SUBJECT"):
        aggregate_to_subject(aug.subject, aug.y, proba, is_synthetic=None)


def test_aggregate_mean_matches_manual(fake_data):
    n = fake_data.n
    rng = np.random.default_rng(0)
    proba = rng.dirichlet([1, 1, 1], size=n)
    out = aggregate_to_subject(fake_data.subject, fake_data.y, proba)
    df = pd.DataFrame(proba)
    df["s"] = fake_data.subject
    expected = df.groupby("s", sort=True)[[0, 1, 2]].mean().to_numpy()
    assert np.allclose(out.proba, expected / expected.sum(axis=1, keepdims=True))


def test_auditor_raises_when_eval_contains_synthetic(fake_data, enforcing_auditor):
    fold = make_group_folds(fake_data, n_splits=3, seed=0)[0]
    enforcing_auditor.register_split(
        fold.fold_id,
        train_subjects=fake_data.subject[fold.train_idx],
        eval_subjects=fake_data.subject[fold.eval_idx],
        train_row_ids=fake_data.row_id[fold.train_idx],
        eval_row_ids=fake_data.row_id[fold.eval_idx],
    )
    with pytest.raises(LeakageError, match="SYNTHETIC_IN_EVAL"):
        enforcing_auditor.record_eval(
            fold.fold_id, is_synthetic=np.array([False, True, False])
        )


def test_postprocess_enforces_nonnegative_and_range():
    from src.augmentation.generators import postprocess_synthetic
    from src.data.schema import PAPER_FEATURES

    ref = pd.DataFrame(
        np.tile(np.arange(1.0, 11.0)[:, None], (1, len(PAPER_FEATURES))),
        columns=list(PAPER_FEATURES),
    )
    bad = pd.DataFrame(
        np.tile(np.array([-5.0, 999.0])[:, None], (1, len(PAPER_FEATURES))),
        columns=list(PAPER_FEATURES),
    )
    out, diag = postprocess_synthetic(
        bad, ref, {"enforce_nonnegative": True, "clip_to_train_range": True}
    )
    assert diag["n_negative_values_clipped"] > 0
    assert diag["n_above_train_max"] > 0
    assert (out <= ref.max()).all().all()
    # sleep_midpoint_at_delta 등 부호 허용 변수는 음수 clip 대상이 아니다.
    assert "sleep_temperature_delta" in out.columns
