"""Small distributed-runtime helpers shared by training entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedState:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int


def distributed_state() -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistributedState(
        distributed=world_size > 1,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def init_distributed_runtime(args: Any, *, backend: str = "nccl") -> DistributedState:
    """Initialize CUDA/DDP runtime and update args.device for local-rank runs."""
    runtime = distributed_state()
    if runtime.distributed:
        args.device = f"cuda:{runtime.local_rank}"
        torch.cuda.set_device(runtime.local_rank)
        dist.init_process_group(backend=backend, device_id=torch.device(args.device))
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    return runtime


def distributed_barrier(local_rank: int) -> None:
    dist.barrier(device_ids=[local_rank])


def is_rank0(rank: int) -> bool:
    return int(rank) == 0


def rank0_print(rank: int, *args: Any, **kwargs: Any) -> None:
    if is_rank0(rank):
        print(*args, **kwargs)
