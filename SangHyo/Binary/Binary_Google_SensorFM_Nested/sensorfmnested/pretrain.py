"""Fold-local SSL pretraining and frozen-encoder embedding extraction.

Leakage contract (stricter than the paper, see config docstring):

* ``pretrain_encoder`` receives ONLY the outer-training subjects' day indices;
  outer-test subjects contribute zero minutes to pretraining;
* normalization stats are computed from those same subjects' precomputed
  moment sums (``DayBank.fold_channel_stats``);
* early stopping watches reconstruction MSE on a held-out 10% subject subset
  OF THE PRETRAIN SUBJECTS (never outer-test), with a fixed artificial mask
  per validation day so epochs are comparable;
* ``embed_all_subjects`` then runs the frozen encoder over everyone's days --
  inference only -- and aggregates per subject as mean+std across days
  (paper M.3.4).
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .config import CLIP_SIGMA, MODEL_VARIANTS, PretrainBudget
from .grids import DayBank
from .mae import SensorFMMae, normalize_minutes, sample_artificial_masks, token_observed_mask


def resolve_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator | None
             ) -> list[np.ndarray]:
    if rng is not None:
        indices = rng.permutation(indices)
    return [indices[i: i + batch_size] for i in range(0, len(indices), batch_size)]


def _to_device(bank: DayBank, day_index: np.ndarray, mean: np.ndarray,
               std: np.ndarray, device: str
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = normalize_minutes(
        bank.values[day_index], bank.mask[day_index], mean, std, CLIP_SIGMA
    )
    minutes = torch.from_numpy(values).to(device)
    element_mask = torch.from_numpy(bank.mask[day_index]).to(device)
    meta = torch.from_numpy(bank.meta[day_index].astype(np.int64)).to(device)
    return minutes, element_mask, meta


def _cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if total_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def pretrain_encoder(
    bank: DayBank,
    pretrain_subject_indices: np.ndarray,
    variant_name: str,
    budget: PretrainBudget,
    seed: int,
    device: str,
    log: Callable[[str], None] = print,
    log_every_epochs: int = 10,
    heartbeat: Callable[[], None] | None = None,
) -> dict:
    """Pretrain one MAE on the given subjects' days; return model + diagnostics."""

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)

    subject_indices = np.asarray(pretrain_subject_indices, dtype=int)
    n_val_subjects = max(1, int(round(len(subject_indices) * budget.val_subject_fraction)))
    shuffled = rng.permutation(subject_indices)
    val_subjects = shuffled[:n_val_subjects]
    train_subjects = shuffled[n_val_subjects:]
    if len(train_subjects) == 0:  # smoke-size cohorts
        train_subjects, val_subjects = shuffled, shuffled[:1]

    train_days = bank.days_of_subjects(train_subjects)
    val_days = bank.days_of_subjects(val_subjects)
    mean, std = bank.fold_channel_stats(train_subjects)

    variant = MODEL_VARIANTS[variant_name]
    model = SensorFMMae(variant).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=budget.base_lr, weight_decay=budget.weight_decay
    )

    steps_per_epoch = max(1, math.ceil(len(train_days) / budget.batch_size))
    total_steps = steps_per_epoch * budget.epochs
    warmup_steps = max(1, int(round(total_steps * budget.warmup_fraction)))

    # Fixed validation masks: comparable epochs, honest early stopping.
    val_artificial: list[torch.Tensor] = []
    val_batches = _batches(val_days, budget.batch_size, rng=None)
    for batch in val_batches:
        observed = token_observed_mask(torch.from_numpy(bank.mask[batch]))
        val_artificial.append(
            sample_artificial_masks(observed, torch.Generator().manual_seed(seed + 7))
        )

    history: list[dict] = []
    best = {"epoch": -1, "val_loss": float("inf"), "state": None}
    step = 0
    started = time.monotonic()
    for epoch in range(budget.epochs):
        model.train()
        epoch_loss, epoch_cells = 0.0, 0.0
        for batch in _batches(train_days, budget.batch_size, rng):
            lr = _cosine_lr(step, total_steps, warmup_steps, budget.base_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            minutes, element_mask, meta = _to_device(bank, batch, mean, std, device)
            observed = token_observed_mask(element_mask.cpu())
            artificial = sample_artificial_masks(observed, generator).to(device)
            out = model(minutes, element_mask, meta, artificial)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cells = float(out["n_loss_cells"])
            epoch_loss += float(out["loss"]) * cells
            epoch_cells += cells
            step += 1
            if heartbeat is not None:
                heartbeat()

        model.eval()
        val_loss, val_cells = 0.0, 0.0
        with torch.no_grad():
            for batch, artificial in zip(val_batches, val_artificial):
                minutes, element_mask, meta = _to_device(bank, batch, mean, std, device)
                out = model(minutes, element_mask, meta, artificial.to(device))
                cells = float(out["n_loss_cells"])
                val_loss += float(out["loss"]) * cells
                val_cells += cells
        train_mse = epoch_loss / max(1.0, epoch_cells)
        val_mse = val_loss / max(1.0, val_cells)
        history.append({"epoch": epoch, "train_mse": round(train_mse, 6),
                        "val_mse": round(val_mse, 6), "lr": round(lr, 8)})

        if val_mse < best["val_loss"] - 1e-6:
            best = {
                "epoch": epoch, "val_loss": val_mse,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        if (epoch + 1) % log_every_epochs == 0 or epoch == budget.epochs - 1:
            log(
                f"      [pretrain] epoch {epoch + 1:3d}/{budget.epochs} | "
                f"train MSE {train_mse:.4f} | val MSE {val_mse:.4f} | "
                f"best {best['val_loss']:.4f}@{best['epoch'] + 1} | "
                f"{time.monotonic() - started:6.1f}s"
            )
        if (epoch + 1 >= budget.min_epochs
                and epoch - best["epoch"] >= budget.patience):
            log(f"      [pretrain] early stop at epoch {epoch + 1} "
                f"(no val improvement for {budget.patience} epochs)")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    return {
        "model": model,
        "channel_mean": mean,
        "channel_std": std,
        "history": history,
        "best_epoch": int(best["epoch"]),
        "best_val_mse": float(best["val_loss"]),
        "epochs_ran": len(history),
        "n_pretrain_days": int(len(train_days)),
        "n_val_days": int(len(val_days)),
        "n_parameters": model.parameter_count(),
        "seconds": round(time.monotonic() - started, 1),
    }


@torch.no_grad()
def embed_all_subjects(
    bank: DayBank,
    model: SensorFMMae,
    channel_mean: np.ndarray,
    channel_std: np.ndarray,
    device: str,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Frozen-encoder inference for EVERY subject; mean+std across days.

    Running inference on outer-test subjects with an encoder that never saw
    them is legitimate scoring, not leakage.  Columns: emb_mean_*, emb_std_*.
    """

    model.eval()
    dim = model.variant.enc_dim
    all_days = np.arange(bank.day_subject.size)
    embeddings = np.zeros((bank.day_subject.size, dim), dtype=np.float64)
    for batch in _batches(all_days, batch_size, rng=None):
        minutes, element_mask, meta = _to_device(
            bank, batch, channel_mean, channel_std, device
        )
        embeddings[batch] = model.embed_days(minutes, element_mask, meta).cpu().numpy()

    n_subjects = len(bank.subject_ids)
    features = np.zeros((n_subjects, 2 * dim), dtype=np.float64)
    for index in range(n_subjects):
        rows = embeddings[bank.day_subject == index]
        features[index, :dim] = rows.mean(axis=0)
        features[index, dim:] = rows.std(axis=0, ddof=0) if len(rows) > 1 else 0.0

    columns = [f"emb_mean_{i:03d}" for i in range(dim)] + [
        f"emb_std_{i:03d}" for i in range(dim)
    ]
    return pd.DataFrame(features, index=pd.Index(bank.subject_ids, name="subject_id"),
                        columns=columns)


__all__ = ["embed_all_subjects", "pretrain_encoder", "resolve_device"]
