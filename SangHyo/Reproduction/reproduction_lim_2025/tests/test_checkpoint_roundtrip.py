"""The checkpoint audit must separate a broken restore from float-kernel noise.

A CUDA-trained LSTM reloaded onto CPU differed by 1.5e-4 and the audit rejected
the whole run.  That is cuDNN vs CPU float32 arithmetic, not a checkpoint that
lost state.  The audit now asks two different questions with two different bars:

* same device -- did every piece of state come back?  Near-exact.
* CPU         -- is the checkpoint portable?  Numeric drift is tolerated up to
                 ``CROSS_DEVICE_PROBABILITY_TOLERANCE``, but every thresholded
                 decision must be identical.

Loosening the bar without keeping the strict same-device leg would hide exactly
the failure AGENTS.md 2-10 was written for, so both legs are asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engine import CROSS_DEVICE_PROBABILITY_TOLERANCE


def test_cross_device_tolerance_is_wide_enough_for_observed_kernel_drift() -> None:
    """The Colab failure measured 1.478e-4 on a 122-step LSTM."""
    observed_cuda_to_cpu_drift = 1.478642225265503e-4
    assert observed_cuda_to_cpu_drift <= CROSS_DEVICE_PROBABILITY_TOLERANCE


def test_cross_device_tolerance_stays_tight_enough_to_catch_a_broken_restore() -> None:
    """A checkpoint that lost state moves probabilities far more than 1e-3."""
    assert CROSS_DEVICE_PROBABILITY_TOLERANCE <= 1e-2
    # A restore bug typically returns near-random or constant scores; even a
    # modest 0.05 shift must still fail.
    assert 0.05 > CROSS_DEVICE_PROBABILITY_TOLERANCE


def _decisions_identical(a: np.ndarray, b: np.ndarray, threshold: float) -> bool:
    return bool(np.array_equal(a >= threshold, b >= threshold))


def test_kernel_drift_preserves_decisions_but_a_broken_restore_does_not() -> None:
    threshold = 0.5
    trained = np.array([0.10, 0.45, 0.55, 0.90])

    drifted = trained + np.array([1e-4, -1e-4, 1e-4, -1e-4])
    assert _decisions_identical(trained, drifted, threshold)
    assert float(np.max(np.abs(drifted - trained))) <= CROSS_DEVICE_PROBABILITY_TOLERANCE

    broken = np.full_like(trained, 0.5)          # collapsed / lost state
    assert not _decisions_identical(trained, broken, threshold)
    assert float(np.max(np.abs(broken - trained))) > CROSS_DEVICE_PROBABILITY_TOLERANCE


def test_drift_that_flips_a_decision_is_still_rejected() -> None:
    """Tiny drift across the threshold changes the reported confusion matrix."""
    threshold = 0.5
    trained = np.array([0.4999])
    drifted = np.array([0.5001])                 # within tolerance, but flips
    assert float(np.max(np.abs(drifted - trained))) <= CROSS_DEVICE_PROBABILITY_TOLERANCE
    assert not _decisions_identical(trained, drifted, threshold)


@pytest.mark.parametrize("name", ["lstm", "bilstm", "cnn1d"])
def test_sequence_checkpoint_reloads_identically_on_the_same_device(
    name: str, tmp_path
) -> None:
    pytest.importorskip("torch")
    from src.models.registry import build_model, load_model

    rng = np.random.default_rng(0)
    n, timesteps, n_features = 12, 16, 3
    X = rng.normal(size=(n, timesteps, n_features))
    y = np.array([0, 1] * (n // 2))
    lengths = np.full(n, timesteps)

    model = build_model(name, seed=0, device="cpu", training={"max_epochs": 2})
    model.fit(X, y, lengths=lengths, seed=0)
    before = model.predict_proba(X, lengths=lengths)

    path = tmp_path / f"{name}.pt"
    model.save(path)
    after = load_model(name, path, device="cpu").predict_proba(X, lengths=lengths)

    assert np.allclose(after, before, rtol=1e-5, atol=1e-7), (
        "a same-device reload must reproduce the scores near-exactly"
    )


def test_tabular_checkpoint_reloads_identically(tmp_path) -> None:
    from src.models.registry import build_model, load_model

    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    y = np.array([0, 1] * 20)

    model = build_model("random_forest", seed=0, device="cpu")
    model.fit(X, y)
    before = model.predict_proba(X)

    path = tmp_path / "rf.joblib"
    model.save(path)
    after = load_model("random_forest", path, device="cpu").predict_proba(X)

    assert np.allclose(after, before, rtol=1e-5, atol=1e-7)
