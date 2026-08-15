"""SensorFM-recipe ViT-1D masked autoencoder (PyTorch re-implementation).

Faithful pieces (SensorFM M.3.1 / M.3.3, Table ED.4):

* patch size [20 minutes x 1 channel] -> 72 x 8 = 576 tokens per day;
* ViT-1D encoder/decoder with the exact Table ED.4 dimensions (XXS/XS/S);
* 2D additive positional encodings: the feature-dimension half is a learned
  per-channel embedding, the temporal half is 1D sinusoidal except for eight
  dims driven by cyclic datetime features (minute-of-hour, hour-of-day,
  day-of-week, day-of-year) through a learned projection;
* AIM two-stage masking (LSM-2): the full mask is the union of the inherited
  missingness mask and one artificial mask drawn per sample from
  {80% random patches, 50% temporal block, 50% modality block}; the encoder
  drops masked tokens (token dropout + padding attention mask), the decoder
  reconstructs from latents + learnable mask tokens;
* MSE loss computed ONLY on artificially-masked cells that were originally
  observed (inherited missingness never contributes loss).

Documented deviations (README_KO.md section D): D1 input schema (8 Oura
channels, not 34 Fitbit features), D2 from-scratch pretraining inside each
outer fold (no external cohort), D3 base LR rescaled for small batches,
D4 the 8 cyclic dims use fixed sin/cos pairs passed through one learned
linear layer (the paper cites Spathis et al. 2021 without exact shapes).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from .config import (
    ACTIVITY_DAY_START_HOUR,
    AIM_MODALITY_BLOCK_RATIO,
    AIM_RANDOM_MASK_RATIO,
    AIM_TEMPORAL_BLOCK_RATIO,
    MINUTES_PER_DAY,
    ModelVariant,
    N_CHANNELS,
    N_CYCLIC_DIMS,
    PATCH_MINUTES,
    TOKENS_PER_CHANNEL,
    TOKENS_PER_DAY,
    TOKEN_OBSERVED_MIN_FRACTION,
)


# ------------------------------------------------------------- patch utils ---
def patchify(minutes: torch.Tensor) -> torch.Tensor:
    """(B, 1440, C) -> (B, T*C, P) token values, channel-major token order.

    Token index t*C + c? No: channel-major means token = c*T + t so that a
    "modality block" is a contiguous token range.  Kept consistent everywhere.
    """

    batch, n_minutes, n_channels = minutes.shape
    assert n_minutes == MINUTES_PER_DAY and n_channels == N_CHANNELS
    tokens = minutes.reshape(batch, TOKENS_PER_CHANNEL, PATCH_MINUTES, n_channels)
    tokens = tokens.permute(0, 3, 1, 2)  # (B, C, T, P)
    return tokens.reshape(batch, TOKENS_PER_DAY, PATCH_MINUTES)


def unpatchify(tokens: torch.Tensor) -> torch.Tensor:
    """(B, T*C, P) -> (B, 1440, C); inverse of :func:`patchify`."""

    batch = tokens.shape[0]
    tokens = tokens.reshape(batch, N_CHANNELS, TOKENS_PER_CHANNEL, PATCH_MINUTES)
    tokens = tokens.permute(0, 2, 3, 1)
    return tokens.reshape(batch, MINUTES_PER_DAY, N_CHANNELS)


def token_observed_mask(element_mask: torch.Tensor) -> torch.Tensor:
    """(B, 1440, C) bool -> (B, T*C) bool: token has enough observed minutes."""

    fraction = patchify(element_mask.float()).mean(dim=-1)
    return fraction >= TOKEN_OBSERVED_MIN_FRACTION


# ------------------------------------------------------- artificial masking --
def sample_artificial_masks(observed: torch.Tensor, generator: torch.Generator
                            ) -> torch.Tensor:
    """Per-sample AIM artificial mask over tokens (True = artificially masked).

    Applies one mode per sample, restricted to observed tokens (masking an
    already-missing token adds nothing to the loss).  Guarantees at least one
    visible observed token and at least one masked observed token whenever the
    sample has >= 2 observed tokens.
    """

    batch, n_tokens = observed.shape
    device = observed.device
    artificial = torch.zeros_like(observed)
    modes = torch.randint(0, 3, (batch,), generator=generator, device=device)

    for row in range(batch):
        obs = observed[row]
        n_obs = int(obs.sum())
        if n_obs < 2:
            continue
        mode = int(modes[row])
        if mode == 0:  # 80% random patch masking
            candidates = torch.nonzero(obs, as_tuple=False).squeeze(1)
            n_mask = max(1, min(n_obs - 1, int(round(AIM_RANDOM_MASK_RATIO * n_obs))))
            order = torch.randperm(n_obs, generator=generator, device=device)
            artificial[row, candidates[order[:n_mask]]] = True
        elif mode == 1:  # 50% contiguous temporal block, all channels
            block = int(round(AIM_TEMPORAL_BLOCK_RATIO * TOKENS_PER_CHANNEL))
            start = int(torch.randint(0, TOKENS_PER_CHANNEL - block + 1, (1,),
                                      generator=generator, device=device))
            time_index = torch.arange(TOKENS_PER_CHANNEL, device=device)
            in_block = (time_index >= start) & (time_index < start + block)
            token_block = in_block.repeat(N_CHANNELS)
            artificial[row] = token_block & obs
        else:  # 50% modality (channel) block masking
            n_drop = max(1, int(round(AIM_MODALITY_BLOCK_RATIO * N_CHANNELS)))
            channels = torch.randperm(N_CHANNELS, generator=generator, device=device)[:n_drop]
            token_channel = (torch.arange(n_tokens, device=device)
                             // TOKENS_PER_CHANNEL)
            drop = torch.isin(token_channel, channels)
            artificial[row] = drop & obs

        # Fallbacks: never mask everything, never mask nothing.
        row_mask = artificial[row]
        if bool(row_mask.all() | (~row_mask & obs).sum().eq(0)):
            visible_candidates = torch.nonzero(row_mask, as_tuple=False).squeeze(1)
            keep = visible_candidates[int(torch.randint(0, len(visible_candidates), (1,),
                                                        generator=generator, device=device))]
            artificial[row, keep] = False
        if not bool(row_mask.any()):
            candidates = torch.nonzero(obs, as_tuple=False).squeeze(1)
            pick = candidates[int(torch.randint(0, len(candidates), (1,),
                                                generator=generator, device=device))]
            artificial[row, pick] = True
    return artificial


# ------------------------------------------------------ positional encoding --
def _sinusoidal_table(n_positions: int, dim: int) -> torch.Tensor:
    table = torch.zeros(n_positions, dim)
    if dim == 0:
        return table
    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    half = (dim + 1) // 2
    frequency = torch.exp(
        torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / max(1, half))
    )
    angles = position * frequency
    table[:, 0::2] = torch.sin(angles[:, : (dim + 1) // 2])
    table[:, 1::2] = torch.cos(angles[:, : dim // 2])
    return table


def cyclic_features(meta: torch.Tensor) -> torch.Tensor:
    """(B, 3) [dow, doy, year] -> (B, T_per_channel, 8) sin/cos datetime feats.

    Time-of-day terms vary per patch (patch start clock time); day-of-week and
    day-of-year vary per sample.  Order: [sin/cos minute-of-hour, sin/cos
    hour-of-day, sin/cos day-of-week, sin/cos day-of-year].
    """

    device = meta.device
    batch = meta.shape[0]
    patch_start = torch.arange(TOKENS_PER_CHANNEL, device=device) * PATCH_MINUTES
    clock_minute = (ACTIVITY_DAY_START_HOUR * 60 + patch_start) % MINUTES_PER_DAY
    minute_of_hour = (clock_minute % 60).float()
    hour_of_day = torch.div(clock_minute, 60, rounding_mode="floor").float()

    angle_minute = 2 * math.pi * minute_of_hour / 60.0
    angle_hour = 2 * math.pi * hour_of_day / 24.0
    per_time = torch.stack(
        [torch.sin(angle_minute), torch.cos(angle_minute),
         torch.sin(angle_hour), torch.cos(angle_hour)], dim=-1
    )  # (T, 4)

    angle_dow = 2 * math.pi * meta[:, 0].float() / 7.0
    angle_doy = 2 * math.pi * meta[:, 1].float() / 365.0
    per_sample = torch.stack(
        [torch.sin(angle_dow), torch.cos(angle_dow),
         torch.sin(angle_doy), torch.cos(angle_doy)], dim=-1
    )  # (B, 4)

    tiled_time = per_time.unsqueeze(0).expand(batch, -1, -1)
    tiled_sample = per_sample.unsqueeze(1).expand(-1, TOKENS_PER_CHANNEL, -1)
    return torch.cat([tiled_time, tiled_sample], dim=-1)  # (B, T, 8)


class PositionalEncoding2D(nn.Module):
    """Additive 2D positional encoding per SensorFM M.3.1 (see module doc)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        feature_half = dim // 2
        temporal_half = dim - feature_half
        if temporal_half <= N_CYCLIC_DIMS:
            raise ValueError(f"dim={dim} leaves no room for sinusoidal encodings")
        self.feature_half = feature_half
        self.temporal_sin_dims = temporal_half - N_CYCLIC_DIMS
        self.channel_embedding = nn.Parameter(
            torch.zeros(N_CHANNELS, feature_half)
        )
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)
        self.register_buffer(
            "temporal_table",
            _sinusoidal_table(TOKENS_PER_CHANNEL, self.temporal_sin_dims),
            persistent=False,
        )
        self.cyclic_projection = nn.Linear(N_CYCLIC_DIMS, N_CYCLIC_DIMS)

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        """(B, 3) -> (B, TOKENS_PER_DAY, dim) additive encoding."""

        batch = meta.shape[0]
        cyclic = self.cyclic_projection(cyclic_features(meta))       # (B, T, 8)
        sinusoid = self.temporal_table.unsqueeze(0).expand(batch, -1, -1)
        temporal = torch.cat([sinusoid, cyclic], dim=-1)             # (B, T, th)
        temporal = temporal.repeat(1, N_CHANNELS, 1)                 # (B, C*T, th)
        feature = self.channel_embedding.repeat_interleave(
            TOKENS_PER_CHANNEL, dim=0
        ).unsqueeze(0).expand(batch, -1, -1)                         # (B, C*T, fh)
        return torch.cat([feature, temporal], dim=-1)


# ------------------------------------------------------------- transformer ---
def _encoder_stack(dim: int, mlp: int, heads: int, layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=mlp, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True,
    )
    # nested-tensor fast path is disabled: its interaction with
    # src_key_padding_mask varies across torch versions, and determinism
    # matters more here than the small speedup.
    return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(dim),
                                 enable_nested_tensor=False)


def _gather_tokens(tokens: torch.Tensor, keep: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack per-sample kept tokens into a padded batch (vectorized).

    Returns (packed (B, L_max, D), padding_mask (B, L_max) True = pad,
    scatter_index (B, L_max) original token positions).  ``scatter_index`` is a
    per-row slice of a permutation, so pad slots still hold UNIQUE (non-kept)
    token positions -- callers exploit that to scatter safely without loops.
    """

    batch, _, dim = tokens.shape
    lengths = keep.sum(dim=1)
    max_length = int(lengths.max())
    # Stable argsort of (not kept): kept tokens first, original order preserved.
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    scatter = order[:, :max_length]
    packed = torch.gather(tokens, 1, scatter.unsqueeze(-1).expand(-1, -1, dim))
    padding = (
        torch.arange(max_length, device=tokens.device).unsqueeze(0)
        >= lengths.unsqueeze(1)
    )
    return packed, padding, scatter


class SensorFMMae(nn.Module):
    """MAE with AIM masking on 576 day tokens (Table ED.4 dimensions)."""

    def __init__(self, variant: ModelVariant) -> None:
        super().__init__()
        self.variant = variant
        self.patch_embed = nn.Linear(PATCH_MINUTES, variant.enc_dim)
        self.encoder_pos = PositionalEncoding2D(variant.enc_dim)
        self.encoder = _encoder_stack(
            variant.enc_dim, variant.enc_mlp, variant.enc_heads, variant.enc_layers
        )
        self.decoder_embed = nn.Linear(variant.enc_dim, variant.dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, variant.dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder_pos = PositionalEncoding2D(variant.dec_dim)
        self.decoder = _encoder_stack(
            variant.dec_dim, variant.dec_mlp, variant.dec_heads, variant.dec_layers
        )
        self.reconstruction_head = nn.Linear(variant.dec_dim, PATCH_MINUTES)

    # ------------------------------------------------------------ encoding ---
    def encode_visible(self, minutes: torch.Tensor, meta: torch.Tensor,
                       visible: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode only ``visible`` tokens (True = fed to the encoder)."""

        tokens = self.patch_embed(patchify(minutes))
        tokens = tokens + self.encoder_pos(meta)
        packed, padding, scatter = _gather_tokens(tokens, visible)
        encoded = self.encoder(packed, src_key_padding_mask=padding)
        return encoded, padding, scatter

    def forward(self, minutes: torch.Tensor, element_mask: torch.Tensor,
                meta: torch.Tensor, artificial: torch.Tensor) -> dict:
        """Training step tensor flow; returns loss plus diagnostics.

        minutes       (B, 1440, C) normalized values, zero-filled where missing
        element_mask  (B, 1440, C) bool, True = originally observed
        meta          (B, 3) int day metadata
        artificial    (B, T*C) bool, True = artificially masked token
        """

        artificial = artificial.clone()  # never mutate the caller's mask
        observed_tokens = token_observed_mask(element_mask)
        visible = observed_tokens & ~artificial
        # Degenerate guard: a sample must keep >= 1 visible token.
        no_visible = visible.sum(dim=1) == 0
        if bool(no_visible.any()):
            first_observed = observed_tokens.float().argmax(dim=1)
            visible[no_visible, first_observed[no_visible]] = True
            artificial[no_visible, first_observed[no_visible]] = False

        encoded, padding, scatter = self.encode_visible(minutes, meta, visible)

        # Scatter encoded tokens back to the full grid; masked slots get the
        # learnable mask token.  (Two-stage AIM: inherited-missing tokens are
        # ALSO represented by mask tokens, but never contribute loss.)  Pad
        # slots of ``scatter`` hold unique non-kept positions, so scattering
        # the mask token through them is a no-op by value.
        batch = minutes.shape[0]
        decoder_tokens = self.mask_token.expand(
            batch, TOKENS_PER_DAY, self.variant.dec_dim
        ).clone()
        projected = self.decoder_embed(encoded)
        source = torch.where(
            padding.unsqueeze(-1), self.mask_token.expand_as(projected), projected
        )
        decoder_tokens.scatter_(
            1, scatter.unsqueeze(-1).expand(-1, -1, self.variant.dec_dim), source
        )
        decoder_tokens = decoder_tokens + self.decoder_pos(meta)
        decoded = self.decoder(decoder_tokens)
        reconstruction = self.reconstruction_head(decoded)  # (B, T*C, P)

        target = patchify(minutes)
        element_observed = patchify(element_mask.float())
        loss_weight = element_observed * artificial.unsqueeze(-1).float()
        denominator = loss_weight.sum().clamp_min(1.0)
        loss = (((reconstruction - target) ** 2) * loss_weight).sum() / denominator
        return {
            "loss": loss,
            "n_loss_cells": denominator.detach(),
            "n_visible_tokens": visible.sum().detach(),
        }

    @torch.no_grad()
    def embed_days(self, minutes: torch.Tensor, element_mask: torch.Tensor,
                   meta: torch.Tensor) -> torch.Tensor:
        """Frozen-encoder day embeddings: mean over observed encoded tokens.

        Mirrors M.3.4 ("aggregated the embeddings per person across all
        non-masked (no inherited missingness) tokens"); the per-day mean is
        this module's output, the across-days mean/std lives in pretrain.py.
        """

        observed_tokens = token_observed_mask(element_mask)
        empty = observed_tokens.sum(dim=1) == 0
        if bool(empty.any()):  # grid admission should prevent this
            observed_tokens[empty, 0] = True
        encoded, padding, _ = self.encode_visible(minutes, meta, observed_tokens)
        real = (~padding).unsqueeze(-1).float()
        summed = (encoded * real).sum(dim=1)
        return summed / real.sum(dim=1).clamp_min(1.0)

    def parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())


def normalize_minutes(values: np.ndarray, mask: np.ndarray, mean: np.ndarray,
                      std: np.ndarray, clip_sigma: float) -> np.ndarray:
    """Fold-local z-score + clip + zero-fill of missing cells (paper M.3.2)."""

    normalized = (values.astype(np.float32) - mean[None, None, :]) / std[None, None, :]
    normalized = np.clip(normalized, -clip_sigma, clip_sigma)
    return np.where(mask, normalized, 0.0).astype(np.float32)


__all__ = [
    "PositionalEncoding2D", "SensorFMMae", "cyclic_features", "normalize_minutes",
    "patchify", "sample_artificial_masks", "token_observed_mask", "unpatchify",
]
