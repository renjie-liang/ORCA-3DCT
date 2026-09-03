"""
Random seed utilities for reproducibility

Set seed for all random number generators to ensure reproducible results
"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Set random seed for all libraries

    Args:
        seed: Random seed value

    Sets seed for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDNN (deterministic mode)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to: {seed}")
