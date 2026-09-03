"""Training reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA RNGs for one process."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
