"""Deterministic seeding and device resolution."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> int:
    """Seed python, numpy and torch (if installed).  Returns the seed used."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # cuDNN autotuning picks different kernels per run; determinism beats the
        # few percent of throughput on a cohort this small.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed


def fold_seed(base_seed: int, *parts: int) -> int:
    """A reproducible per-(repeat, fold, length) seed derived from *base_seed*."""
    value = int(base_seed)
    for part in parts:
        value = (value * 1_000_003 + int(part) * 31 + 17) % (2**31 - 1)
    return value


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
