"""전역 난수 seed 설정."""

from __future__ import annotations

import logging
import os
import random

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["set_global_seed"]


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """random / numpy / torch의 seed를 설정한다 (torch는 설치된 경우에만)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        log.debug("torch 미설치 — torch seed 생략")
