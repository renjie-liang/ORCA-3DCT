"""
Simplified optimizer and scheduler builders

Fail-fast principles:
- No default values except function parameters
- Direct dict access
- Explicit errors
"""
import math
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR

from .logger import get_module_logger


# ============================================================================
# Optimizer Builders
# ============================================================================

def separate_weight_decay_params(model: nn.Module):
    """
    Separate parameters into weight decay and no weight decay groups

    Weight decay is applied to:
    - Parameters with ndim >= 2 (weights)

    No weight decay for:
    - Parameters with ndim < 2 (biases, layer norms)

    Args:
        model: PyTorch model

    Returns:
        (wd_params, no_wd_params) lists
    """
    wd_params = []
    no_wd_params = []

    for param in model.parameters():
        if not param.requires_grad:
            continue

        if param.ndim < 2:
            no_wd_params.append(param)
        else:
            wd_params.append(param)

    return wd_params, no_wd_params


def build_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Build optimizer from config

    Args:
        model: PyTorch model
        config: Optimizer config dict

    Config format:
        optimizer:
          type: "adamw"  # or "adam"
          lr: 1.0e-4
          weight_decay: 1.0e-4
          betas: [0.9, 0.999]
          eps: 1.0e-8
          separate_wd_groups: true  # Separate weight decay groups

    Returns:
        PyTorch optimizer
    """
    optimizer_type = config['type'].lower()
    lr = config['lr']
    weight_decay = config['weight_decay']
    betas = tuple(config['betas'])
    eps = config['eps']
    separate_wd_groups = config.get('separate_wd_groups', True)

    # Get trainable parameters
    params = list(model.parameters())
    params = [p for p in params if p.requires_grad]

    # Separate weight decay groups if needed
    if weight_decay > 0 and separate_wd_groups:
        wd_params, no_wd_params = separate_weight_decay_params(model)

        param_groups = [
            {'params': wd_params, 'weight_decay': weight_decay},
            {'params': no_wd_params, 'weight_decay': 0.0}
        ]

        logger = get_module_logger()
        logger.info(f"Optimizer: {len(wd_params)} params with WD, {len(no_wd_params)} params without WD")
    else:
        param_groups = params

    # Create optimizer
    if optimizer_type == 'adamw':
        optimizer = AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    elif optimizer_type == 'adam':
        optimizer = Adam(param_groups, lr=lr, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    logger = get_module_logger()
    logger.info(f"Built {optimizer_type.upper()} optimizer: lr={lr}, wd={weight_decay}")

    return optimizer


# ============================================================================
# Scheduler Builders
# ============================================================================

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Build learning rate scheduler from config

    Args:
        optimizer: PyTorch optimizer
        config: Scheduler config dict

    Config format:
        scheduler:
          type: "warmup_cosine"  # or "warmup_constant"
          warmup_steps: 1000
          total_steps: 100000
          min_lr_ratio: 0.01  # For cosine only

    Returns:
        PyTorch LR scheduler
    """
    scheduler_type = config['type'].lower()
    warmup_steps = config['warmup_steps']

    if scheduler_type == 'warmup_cosine':
        total_steps = config['total_steps']
        min_lr_ratio = config.get('min_lr_ratio', 0.01)

        scheduler = get_warmup_cosine_scheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )
        logger = get_module_logger()
        logger.info(f"Built warmup+cosine scheduler: warmup={warmup_steps}, total={total_steps}, min_ratio={min_lr_ratio}")

    elif scheduler_type == 'warmup_constant':
        scheduler = get_warmup_constant_scheduler(
            optimizer,
            warmup_steps=warmup_steps
        )
        logger = get_module_logger()
        logger.info(f"Built warmup+constant scheduler: warmup={warmup_steps}")

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def get_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01
) -> LambdaLR:
    """
    Linear warmup + cosine decay scheduler

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum lr as ratio of initial lr

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay phase
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def get_warmup_constant_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int
) -> LambdaLR:
    """
    Linear warmup + constant lr scheduler

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)
