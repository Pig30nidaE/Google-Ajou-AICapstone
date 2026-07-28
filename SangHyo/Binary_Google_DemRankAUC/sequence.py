"""TSMixer-style daily-sequence encoder (optional arm).

Is a sequence model justified here?
-----------------------------------
Yes on the data side, doubtfully on the label side, so it is implemented and
switched off by default rather than skipped or assumed.

*Data*: subjects have 35-120 consecutive daily records across ~26 numeric
channels (median 66 days, coverage 0.88).  That is a genuine multivariate
time series, not an aggregate, so the "sequence is too short, do not force it"
escape clause in the brief does not apply.

*Labels*: there are **12 positives in the entire dataset**.  A model with even a
few thousand parameters has orders of magnitude more capacity than 12 positive
examples can constrain, and this repository has already measured what that
produces on the neighbouring task -- ``Binary_Wearable_ConvBiLSTM_NoMMSE`` at
OOF AUC 0.515-0.526 and ``Binary_Wearable_SequenceFusion_Google`` at 0.566
ensemble (0.625 for the standalone Transformer, which did not reproduce).

So the arm exists, is honest about its size (a deliberately small TSMixer: two
mixing blocks, ~16 hidden units), and is enabled only under the ``max`` profile
or ``--sequence-arm``.  If it loses, that is a *measured* answer to the brief's
question rather than an opinion.

Architecture
------------
TSMixer's premise is that alternating MLPs over the time axis and the channel
axis match attention on tabular-ish multivariate series at a fraction of the
parameters -- which is the right trade at this sample size.

    input  (B, T, C) + mask (B, T)
      -> per-channel fold-local standardisation
      -> [time-mixing MLP over T] + [channel-mixing MLP over C]  x n_blocks
      -> masked attention pooling over T
      -> concat static covariates (the tabular feature vector)
      -> linear -> logit

Masking is applied at every stage: padded days are excluded from the time-mixing
MLP's input, from the pooling weights, and from the normalisation statistics, so
a subject with 35 days is not treated as a subject with 64 days of zeros.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LOCAL_TZ, RICH_ACTIVITY, RICH_SLEEP, SOURCE_FILES, SPLIT_DIRS
from .features import read_csv
from .models import Model, torch_device

SEQUENCE_CHANNELS = tuple(RICH_ACTIVITY) + tuple(RICH_SLEEP)
DEFAULT_MAX_DAYS = 64


@dataclass(frozen=True)
class SequenceBundle:
    """Right-aligned daily tensor for the whole cohort (values are *raw*).

    Normalisation is deliberately not applied here: it is a fitted statistic and
    therefore belongs inside the fold, in :class:`TSMixerModel`.
    """

    subject_ids: np.ndarray          # (N,)
    values: np.ndarray               # (N, T, C), NaN where padded/missing
    mask: np.ndarray                 # (N, T) bool, True = real observed day
    channels: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        return len(self.channels)


def build_sequences(data_root: str | Path, *, channels: tuple[str, ...] = SEQUENCE_CHANNELS,
                    max_days: int = DEFAULT_MAX_DAYS) -> SequenceBundle:
    """Daily multivariate series per subject, right-aligned to the last ``max_days``.

    Right alignment keeps the most recent window, which is the part closest in
    time to the diagnosis.  Every value comes from that subject's own rows.
    """

    data_root = Path(data_root)
    frames: list[pd.DataFrame] = []
    for split in ("train", "val"):
        activity = read_csv(data_root / SPLIT_DIRS[split] / SOURCE_FILES[split]["activity"])
        sleep = read_csv(data_root / SPLIT_DIRS[split] / SOURCE_FILES[split]["sleep"])
        activity = activity.copy()
        sleep = sleep.copy()
        activity["_sid"] = activity["EMAIL"].astype(str).str.strip()
        sleep["_sid"] = sleep["EMAIL"].astype(str).str.strip()
        activity["_day"] = (
            pd.to_datetime(activity["activity_day_start"], errors="coerce", utc=True)
            .dt.tz_convert(LOCAL_TZ).dt.normalize()
        )
        sleep["_day"] = (
            pd.to_datetime(sleep["sleep_bedtime_end"], errors="coerce", utc=True)
            .dt.tz_convert(LOCAL_TZ).dt.normalize()
        )
        activity_columns = [c for c in channels if c in activity.columns]
        sleep_columns = [c for c in channels if c in sleep.columns]
        merged = pd.merge(
            activity[["_sid", "_day", *activity_columns]],
            sleep[["_sid", "_day", *sleep_columns]],
            on=["_sid", "_day"], how="outer",
        )
        frames.append(merged)

    daily = pd.concat(frames, axis=0, ignore_index=True)
    present = [c for c in channels if c in daily.columns]
    for column in present:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")

    subjects = sorted(daily["_sid"].dropna().unique().tolist())
    n, t, c = len(subjects), int(max_days), len(present)
    values = np.full((n, t, c), np.nan, dtype=np.float32)
    mask = np.zeros((n, t), dtype=bool)

    for position, sid in enumerate(subjects):
        group = daily.loc[daily["_sid"] == sid].sort_values("_day")
        block = group[present].to_numpy(dtype=np.float32)
        if block.shape[0] == 0:
            continue
        block = block[-t:]
        length = block.shape[0]
        values[position, t - length:, :] = block
        mask[position, t - length:] = np.isfinite(block).any(axis=1)

    return SequenceBundle(subject_ids=np.asarray(subjects, dtype=str), values=values,
                          mask=mask, channels=tuple(present))


class TSMixerModel(Model):
    """Small masked TSMixer classifier over daily sequences + static covariates.

    ``rows`` (from ``engine.fold_fit_predict``) selects this fold's subjects out
    of the shared cohort tensor.  Channel standardisation statistics come from
    the **training rows of the fold only**, which is the whole reason the tensor
    is stored un-normalised.
    """

    name = "tsmixer"

    def __init__(self, bundle: SequenceBundle, params: dict | None = None, *, seed: int = 0) -> None:
        super().__init__(params, seed=seed)
        self.bundle = bundle

    # ---------------------------------------------------------------- torch --
    def _build_network(self, n_channels: int, n_static: int):
        import torch
        from torch import nn

        hidden = int(self.params.get("hidden", 16))
        n_blocks = int(self.params.get("n_blocks", 2))
        dropout = float(self.params.get("dropout", 0.3))
        max_days = self.bundle.values.shape[1]

        class MixerBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.time_norm = nn.LayerNorm(n_channels)
                self.time_mlp = nn.Sequential(
                    nn.Linear(max_days, hidden), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden, max_days),
                )
                self.channel_norm = nn.LayerNorm(n_channels)
                self.channel_mlp = nn.Sequential(
                    nn.Linear(n_channels, hidden), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(hidden, n_channels),
                )

            def forward(self, x, mask):
                # x: (B, T, C); mask: (B, T, 1) with 1.0 on observed days.
                residual = x
                z = self.time_norm(x) * mask
                z = self.time_mlp(z.transpose(1, 2)).transpose(1, 2)
                x = residual + z * mask
                x = x + self.channel_mlp(self.channel_norm(x)) * mask
                return x

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([MixerBlock() for _ in range(n_blocks)])
                self.pool_score = nn.Linear(n_channels, 1)
                self.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(n_channels + n_static, 1),
                )

            def forward(self, x, mask, static):
                mask3 = mask.unsqueeze(-1)
                for block in self.blocks:
                    x = block(x, mask3)
                # Masked attention pooling: padded days get -inf logits, so they
                # contribute exactly zero weight.
                logits = self.pool_score(x).squeeze(-1)
                logits = logits.masked_fill(mask <= 0, float("-inf"))
                weights = torch.softmax(logits, dim=1).unsqueeze(-1)
                weights = torch.nan_to_num(weights, nan=0.0)
                pooled = (x * weights).sum(dim=1)
                return self.head(torch.cat([pooled, static], dim=1)).squeeze(-1)

        return Net()

    def _tensor_for(self, rows: np.ndarray):
        values = self.bundle.values[np.asarray(rows, dtype=np.int64)]
        mask = self.bundle.mask[np.asarray(rows, dtype=np.int64)]
        return values, mask

    def _fit(self, X: np.ndarray, y: np.ndarray) -> None:
        import torch
        from torch import nn

        if self.rows_ is None:
            raise ValueError(
                "tsmixer needs cohort row indices; it cannot run under a synthetic "
                "resampler that invents rows"
            )
        torch.manual_seed(self.seed)
        device = torch.device(torch_device())

        values, mask = self._tensor_for(self.rows_)
        # Fold-local channel statistics over observed days of training subjects.
        observed = values[mask]
        self.center_ = np.nanmean(observed, axis=0)
        scale = np.nanstd(observed, axis=0)
        self.scale_ = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        self.center_ = np.nan_to_num(self.center_, nan=0.0)

        network = self._build_network(self.bundle.n_channels, X.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            network.parameters(), lr=float(self.params.get("lr", 3e-3)),
            weight_decay=float(self.params.get("weight_decay", 1e-2)),
        )
        epochs = int(self.params.get("epochs", 120))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        x_t, m_t = self._to_torch(values, mask, device)
        static_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32), device=device)
        positive_weight = float((y == 0).sum()) / max(1.0, float((y == 1).sum()))
        gamma = float(self.params.get("focal_gamma", 0.0))

        network.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logit = network(x_t, m_t, static_t)
            loss = self._loss(logit, y_t, positive_weight, gamma, nn)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        network.eval()
        self.network_ = network
        self.device_ = device

    @staticmethod
    def _loss(logit, target, positive_weight: float, gamma: float, nn_module):
        import torch

        if gamma <= 0:
            return nn_module.functional.binary_cross_entropy_with_logits(
                logit, target, pos_weight=torch.tensor(positive_weight, device=logit.device)
            )
        # Focal loss: down-weights the 162 easy negatives so the 12 positives are
        # not drowned out by an already-solved majority.
        probability = torch.sigmoid(logit)
        p_t = target * probability + (1 - target) * (1 - probability)
        alpha = target * positive_weight + (1 - target) * 1.0
        return (alpha * (1 - p_t).pow(gamma) * -torch.log(p_t.clamp_min(1e-7))).mean()

    def _to_torch(self, values: np.ndarray, mask: np.ndarray, device):
        import torch

        normalized = (values - self.center_) / self.scale_
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.tensor(normalized.astype(np.float32), device=device),
            torch.tensor(mask.astype(np.float32), device=device),
        )

    def _score(self, X: np.ndarray) -> np.ndarray:
        import torch

        rows = getattr(self, "score_rows_", None)
        if rows is None:
            raise ValueError("tsmixer needs cohort row indices at score time")
        values, mask = self._tensor_for(rows)
        x_t, m_t = self._to_torch(values, mask, self.device_)
        static_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device_)
        with torch.no_grad():
            return self.network_(x_t, m_t, static_t).detach().cpu().numpy()


def align_bundle(bundle: SequenceBundle, subject_ids) -> SequenceBundle:
    """Reorder the tensor so row *i* is ``subject_ids[i]``.

    ``TSMixerModel`` addresses the tensor with **cohort** row indices, so the
    bundle must share the cohort's row order.  The quality-filtered arm drops
    subjects, which is exactly the case where a silent misalignment would pair
    one subject's sequence with another's label.
    """

    position = {str(sid): index for index, sid in enumerate(bundle.subject_ids)}
    missing = [str(s) for s in subject_ids if str(s) not in position]
    if missing:
        raise KeyError(f"{len(missing)} cohort subjects have no daily sequence "
                       f"(e.g. {missing[:3]})")
    order = np.array([position[str(s)] for s in subject_ids], dtype=np.int64)
    return SequenceBundle(
        subject_ids=np.asarray([str(s) for s in subject_ids], dtype=str),
        values=bundle.values[order],
        mask=bundle.mask[order],
        channels=bundle.channels,
    )


def make_sequence_spec(bundle: SequenceBundle, params: dict | None = None):
    """A ``ModelSpec``-compatible factory bound to one cohort tensor."""

    from .models import ModelSpec

    class _BoundSpec(ModelSpec):
        def build(self, seed: int) -> Model:  # type: ignore[override]
            return TSMixerModel(bundle, self.params, seed=seed)

    return _BoundSpec(name="tsmixer", params=dict(params or {}))


__all__ = ["DEFAULT_MAX_DAYS", "SEQUENCE_CHANNELS", "SequenceBundle", "TSMixerModel",
           "build_sequences", "make_sequence_spec"]
