"""Centralised seeding and device selection.

The original paper does not report a random seed (``unresolved_questions.md`` Q15),
so every seed used here is ours.  Keeping them in one place makes the reported
seed-to-seed variation meaningful.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed python, numpy and (if installed) torch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fold_seed(base_seed: int, repeat: int = 0, fold: int = 0) -> int:
    """Derive a reproducible per-(repeat, fold) seed from the base seed."""
    return int(base_seed) + 1000 * int(repeat) + int(fold)


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device string. Returns ``'cpu'`` when torch is unavailable."""
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"device must be auto/cpu/cuda, got {requested!r}")
    try:
        import torch
    except ImportError:
        return "cpu"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"
