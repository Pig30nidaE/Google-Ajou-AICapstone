"""Padding must never be confused with real data.

Regression tests for a bug that silently blanked every real observation: the
builder defaults to ``padding='pre'`` (valid steps at the *end*), while every
downstream consumer derived its mask as ``arange(T) < lengths`` (valid steps at
the *front*).  That selects exactly the padding, so the deep models in experiment
A trained on all-zero inputs and collapsed to predicting the majority class.

The fix routes everything through ``Representation.valid_mask()`` /
``left_aligned()``.  These tests pin that down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.representations import (
    FoldPreprocessor,
    build_temporal_sequence,
    fit_transform_pair,
    zero_padding,
)

FEATURE = "f1"
BASE_VALUE = 100.0


def _daily(day_counts=(3, 5, 8)) -> pd.DataFrame:
    rows = []
    for i, n_days in enumerate(day_counts):
        for day in range(n_days):
            rows.append(
                {
                    "subject_id": f"s{i}",
                    "date": pd.Timestamp("2020-10-17") + pd.Timedelta(days=int(day)),
                    "label": i % 2,
                    FEATURE: BASE_VALUE + 10 * i + day,
                    "row_id": len(rows),
                }
            )
    return pd.DataFrame(rows)


def _build(side: str, daily: pd.DataFrame | None = None):
    return build_temporal_sequence(
        daily if daily is not None else _daily(),
        [FEATURE],
        sequence_length="max",
        padding=side,
    )


@pytest.mark.parametrize("side", ["pre", "post"])
def test_valid_mask_selects_the_real_observations(side: str) -> None:
    rep = _build(side)
    mask = rep.valid_mask()
    assert mask is not None and mask.shape == rep.X.shape[:2]
    for i, length in enumerate(rep.lengths):
        assert int(mask[i].sum()) == int(length)
        selected = rep.X[i, mask[i], 0]
        assert (selected != 0).all(), "mask must not select padding"
        assert np.allclose(selected, np.sort(selected)), "date order must survive"


@pytest.mark.parametrize("side", ["pre", "post"])
def test_zero_padding_preserves_every_real_observation(side: str) -> None:
    """The exact bug: zero_padding used to blank the real data for pre-padding."""
    rep = _build(side)
    before = rep.X.copy()
    mask = rep.valid_mask()
    zero_padding(rep)
    assert np.allclose(rep.X[mask], before[mask]), "real observations were altered"
    assert np.allclose(rep.X[~mask], 0.0), "padding must be zero"
    assert not np.allclose(rep.X, 0.0), "the tensor must not be entirely blank"


@pytest.mark.parametrize("side", ["pre", "post"])
def test_left_aligned_puts_valid_steps_first(side: str) -> None:
    rep = _build(side)
    aligned, lengths = rep.left_aligned()
    for i, length in enumerate(lengths):
        assert (aligned[i, :length, 0] != 0).all(), "valid steps must lead"
        assert np.allclose(aligned[i, length:, 0], 0.0), "padding must trail"


def test_left_aligned_is_identical_for_pre_and_post_padding() -> None:
    """Masked models must be padding-invariant; this is what makes them so."""
    pre_X, pre_len = _build("pre").left_aligned()
    post_X, post_len = _build("post").left_aligned()
    assert np.allclose(pre_X, post_X)
    assert np.array_equal(pre_len, post_len)


@pytest.mark.parametrize("side", ["pre", "post"])
def test_preprocessor_statistics_ignore_padding(side: str) -> None:
    """Padding is not an observation and must not drag the mean toward zero."""
    rep = _build(side)
    pre = FoldPreprocessor(standardize=True)
    pre.fit(rep.X, subjects=rep.subjects, feature_names=rep.feature_names,
            mask=rep.valid_mask())

    observed = _daily()[FEATURE].to_numpy()
    assert pre.mean_[0] == pytest.approx(observed.mean(), rel=1e-9)
    assert pre.mean_[0] > BASE_VALUE, "a padding-contaminated mean would collapse"


def test_preprocessor_mean_is_the_same_for_both_padding_sides() -> None:
    means = []
    for side in ("pre", "post"):
        rep = _build(side)
        pre = FoldPreprocessor(standardize=True)
        pre.fit(rep.X, subjects=rep.subjects, feature_names=rep.feature_names,
                mask=rep.valid_mask())
        means.append(float(pre.mean_[0]))
    assert means[0] == pytest.approx(means[1])


def test_preprocessor_rejects_a_mismatched_mask() -> None:
    from src.features.representations import PreprocessingScopeError

    rep = _build("pre")
    pre = FoldPreprocessor()
    with pytest.raises(PreprocessingScopeError, match="mask shape"):
        pre.fit(rep.X, subjects=rep.subjects, feature_names=rep.feature_names,
                mask=np.ones((rep.X.shape[0], rep.X.shape[1] + 1), dtype=bool))


@pytest.mark.parametrize("side", ["pre", "post"])
def test_full_pipeline_leaves_real_signal_in_the_tensor(side: str) -> None:
    """End-to-end: build -> fit/transform -> zero padding -> left align."""
    daily = _daily()
    train = build_temporal_sequence(
        daily, [FEATURE], subjects=["s0", "s1"], sequence_length="max", padding=side
    )
    test = build_temporal_sequence(
        daily, [FEATURE], subjects=["s2"],
        sequence_length=int(train.meta["sequence_length"]), padding=side,
    )
    fit_transform_pair(FoldPreprocessor(standardize=True), train, test)

    for rep in (train, test):
        mask = rep.valid_mask()
        assert np.isfinite(rep.X).all()
        assert np.allclose(rep.X[~mask], 0.0), "padding must stay zero after scaling"
        assert np.abs(rep.X[mask]).sum() > 0, "real observations were blanked"
        aligned, lengths = rep.left_aligned()
        for i, length in enumerate(lengths):
            assert np.abs(aligned[i, :length]).sum() > 0
