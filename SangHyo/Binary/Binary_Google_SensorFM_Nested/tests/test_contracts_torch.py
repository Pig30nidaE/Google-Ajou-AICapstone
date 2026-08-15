"""Torch-dependent contract tests (tiny synthetic tensors, no real training).

Skipped automatically when torch is unavailable; the torch-free contracts
live in ``test_contracts.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from sensorfmnested import config as C                                  # noqa: E402
from test_contracts import _make_bank                                   # noqa: E402

from sensorfmnested.mae import (                                        # noqa: E402
    SensorFMMae, cyclic_features, patchify, sample_artificial_masks,
    token_observed_mask, unpatchify,
)
from sensorfmnested.pretrain import embed_all_subjects                  # noqa: E402


def test_patchify_roundtrip_channel_major():
    minutes = torch.arange(C.MINUTES_PER_DAY * C.N_CHANNELS, dtype=torch.float32)
    minutes = minutes.reshape(1, C.MINUTES_PER_DAY, C.N_CHANNELS)
    tokens = patchify(minutes)
    assert tokens.shape == (1, C.TOKENS_PER_DAY, C.PATCH_MINUTES)
    # Token c*T + t must hold channel c, minutes [20t, 20t+20).
    c, t = 3, 5
    expected = minutes[0, t * 20: t * 20 + 20, c]
    torch.testing.assert_close(tokens[0, c * C.TOKENS_PER_CHANNEL + t], expected)
    torch.testing.assert_close(unpatchify(tokens), minutes)


def test_aim_masks_respect_inherited_missingness_and_ratios():
    generator = torch.Generator().manual_seed(0)
    observed = torch.ones(64, C.TOKENS_PER_DAY, dtype=torch.bool)
    observed[:, : C.TOKENS_PER_CHANNEL] = False  # channel 0 fully missing
    artificial = sample_artificial_masks(observed, generator)
    assert not (artificial & ~observed).any()  # never masks missing tokens
    per_sample = artificial.sum(dim=1).float() / observed.sum(dim=1).float()
    assert per_sample.min() > 0.05 and per_sample.max() <= 0.85
    visible = observed & ~artificial
    assert (visible.sum(dim=1) >= 1).all()


def test_mae_forward_loss_only_on_artificial_observed_cells():
    torch.manual_seed(0)
    model = SensorFMMae(C.MODEL_VARIANTS["XXS"])
    batch = 2
    minutes = torch.randn(batch, C.MINUTES_PER_DAY, C.N_CHANNELS)
    element_mask = torch.rand(batch, C.MINUTES_PER_DAY, C.N_CHANNELS) > 0.3
    minutes = torch.where(element_mask, minutes, torch.zeros_like(minutes))
    meta = torch.tensor([[2, 100, 2020], [5, 200, 2020]])
    observed = token_observed_mask(element_mask)
    generator = torch.Generator().manual_seed(1)
    artificial = sample_artificial_masks(observed, generator)
    out = model(minutes, element_mask, meta, artificial)
    assert torch.isfinite(out["loss"])
    expected_cells = (
        patchify(element_mask.float()) * artificial.unsqueeze(-1).float()
    ).sum()
    torch.testing.assert_close(out["n_loss_cells"], expected_cells)


def test_embedding_ignores_unobserved_tokens():
    torch.manual_seed(0)
    model = SensorFMMae(C.MODEL_VARIANTS["XXS"]).eval()
    minutes = torch.randn(1, C.MINUTES_PER_DAY, C.N_CHANNELS)
    element_mask = torch.zeros(1, C.MINUTES_PER_DAY, C.N_CHANNELS, dtype=torch.bool)
    element_mask[:, :200, :] = True
    zero_filled = torch.where(element_mask, minutes, torch.zeros_like(minutes))
    base = model.embed_days(zero_filled, element_mask, torch.tensor([[0, 0, 2020]]))
    # Garbage placed DIRECTLY in unobserved minutes (deliberately not
    # zero-filled) must not change the embedding: those tokens are dropped
    # before the encoder, not merely zeroed.
    tampered = zero_filled.clone()
    tampered[:, 1000:, :] = 123.0
    after = model.embed_days(tampered, element_mask, torch.tensor([[0, 0, 2020]]))
    torch.testing.assert_close(base, after)


def test_cyclic_features_shape_and_range():
    meta = torch.tensor([[0, 0, 2020], [6, 364, 2021]])
    feats = cyclic_features(meta)
    assert feats.shape == (2, C.TOKENS_PER_CHANNEL, C.N_CYCLIC_DIMS)
    assert float(feats.abs().max()) <= 1.0 + 1e-6


def test_embed_all_subjects_aggregates_mean_and_std():
    bank = _make_bank()
    model = SensorFMMae(C.MODEL_VARIANTS["XXS"]).eval()
    mean, std = bank.fold_channel_stats(np.arange(len(bank.subject_ids)))
    frame = embed_all_subjects(bank, model, mean, std, device="cpu", batch_size=4)
    dim = C.MODEL_VARIANTS["XXS"].enc_dim
    assert frame.shape == (2, 2 * dim)
    assert list(frame.index) == bank.subject_ids
    assert np.isfinite(frame.to_numpy()).all()
