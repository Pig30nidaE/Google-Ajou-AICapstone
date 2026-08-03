"""Early stopping must not restore an untrained network.

Regression tests for the second failure in experiment A: the monitor split was
unstratified and the monitored quantity was BCE loss on ~28 subjects.  That loss
rises as soon as the net grows confident, so the best epoch was 0 and early
stopping restored the initial weights -- the reported LSTM/Bi-LSTM numbers came
from an essentially untrained model.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.sequence import (
    TRAINING_DEFAULTS,
    _safe_auc,
    _stratified_monitor_split,
)


# --- monitor split ------------------------------------------------------------

def test_monitor_split_preserves_class_balance() -> None:
    y = np.array([1] * 56 + [0] * 85)          # the real Training balance
    monitor, fit = _stratified_monitor_split(y, fraction=0.2, seed=0)

    assert len(monitor) + len(fit) == len(y)
    assert set(monitor).isdisjoint(set(fit))
    monitor_rate = y[monitor].mean()
    assert monitor_rate == pytest.approx(y.mean(), abs=0.02), (
        "the monitored positive rate must track the training one"
    )


def test_monitor_split_is_stable_across_seeds() -> None:
    """Stratification is what makes the monitored positive count reproducible."""
    y = np.array([1] * 56 + [0] * 85)
    counts = {
        int(y[_stratified_monitor_split(y, fraction=0.2, seed=s)[0]].sum())
        for s in range(10)
    }
    assert len(counts) == 1, f"positive count drifted across seeds: {counts}"


def test_monitor_split_falls_back_when_a_side_would_be_single_class() -> None:
    y = np.array([1, 0, 0, 0])                  # only one positive to go around
    monitor, fit = _stratified_monitor_split(y, fraction=0.2, seed=0)
    assert len(monitor) == 0, "no usable monitor split"
    assert len(fit) == len(y), "all rows must still be trained on"


def test_monitor_split_both_sides_have_both_classes() -> None:
    y = np.array([1] * 20 + [0] * 30)
    monitor, fit = _stratified_monitor_split(y, fraction=0.2, seed=3)
    assert len(np.unique(y[monitor])) == 2
    assert len(np.unique(y[fit])) == 2


# --- monitored metric ---------------------------------------------------------

def test_default_monitor_metric_is_auc() -> None:
    """BCE loss on a tiny monitor split stopped training at epoch 0."""
    assert TRAINING_DEFAULTS["early_stopping_metric"] == "auc"


def test_safe_auc_handles_a_single_class_slice() -> None:
    assert _safe_auc(np.array([1, 1, 1]), np.array([0.2, 0.5, 0.9])) == 0.5
    assert _safe_auc(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0
    assert _safe_auc(np.array([0, 1]), np.array([0.9, 0.1])) == 0.0


def test_unknown_monitor_metric_is_rejected() -> None:
    torch = pytest.importorskip("torch")  # noqa: F841
    from src.models.registry import build_model

    model = build_model("cnn1d", seed=0, device="cpu",
                        training={"early_stopping_metric": "f1"})
    X = np.random.default_rng(0).normal(size=(8, 5, 3))
    y = np.array([0, 1] * 4)
    with pytest.raises(ValueError, match="early_stopping_metric"):
        model.fit(X, y, lengths=np.full(8, 5), seed=0)


# --- training actually happens ------------------------------------------------

@pytest.mark.parametrize("name", ["lstm", "bilstm", "cnn1d"])
def test_model_trains_past_initialisation_on_a_learnable_signal(name: str) -> None:
    pytest.importorskip("torch")
    from src.models.registry import build_model

    rng = np.random.default_rng(0)
    n, timesteps, n_features = 40, 12, 3
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, timesteps, n_features))
    X[:, :, 0] += 3.0 * y[:, None]            # an obvious planted signal

    model = build_model(name, seed=0, device="cpu",
                        training={"max_epochs": 40, "early_stopping_patience": 10})
    model.fit(X, y, lengths=np.full(n, timesteps), seed=0)

    history = model.history
    assert history["best_epoch"] > 0, "restored the untrained initial weights"
    assert not history["degenerate_training"]
    probabilities = model.predict_proba(X, lengths=np.full(n, timesteps))
    assert probabilities.max() - probabilities.min() > 0.2, (
        "probabilities stayed collapsed near the initial output"
    )


def test_history_records_the_diagnostics_the_report_needs() -> None:
    pytest.importorskip("torch")
    from src.models.registry import build_model

    rng = np.random.default_rng(1)
    n, timesteps = 24, 6
    y = np.array([0, 1] * (n // 2))
    X = rng.normal(size=(n, timesteps, 2))
    model = build_model("cnn1d", seed=0, device="cpu", training={"max_epochs": 5})
    model.fit(X, y, lengths=np.full(n, timesteps), seed=0)

    for key in ("best_epoch", "monitor_metric", "monitor_split_positives",
                "degenerate_training", "epochs_run"):
        assert key in model.history, f"history is missing {key}"

    diagnostics = model.summary()["training_diagnostics"]
    assert diagnostics["monitor_metric"] == "auc"
    assert isinstance(diagnostics["degenerate_training"], bool)
