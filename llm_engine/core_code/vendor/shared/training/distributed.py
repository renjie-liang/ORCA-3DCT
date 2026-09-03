"""
Distributed training utilities - Simplified version

Supports both single-GPU and multi-GPU training.
Fail-fast: missing config or wrong environment → immediate error.
"""
import os
import datetime
import torch
import torch.distributed as dist


def is_distributed():
    """Check if running in distributed mode (cached)"""
    # Check torchrun environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        return True
    # Check SLURM environment
    if 'SLURM_PROCID' in os.environ:
        return True
    return False


def setup_distributed():
    """
    Initialize distributed training

    Returns:
        rank: Process rank (0 for single GPU, 0-N for multi-GPU)

    Supports three modes:
    1. Single GPU:
        python train.py --config xxx.yaml
        → rank = 0

    2. Multi-GPU (torchrun):
        torchrun --nproc_per_node=4 train.py --config xxx.yaml
        → rank = 0, 1, 2, 3

    3. Multi-node (SLURM):
        srun python train.py --config xxx.yaml
        → rank = 0, 1, 2, 3, ...
    """
    # Check torchrun environment
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Launched by torchrun
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])

        # Debug info
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        print(f"[Rank {rank}] Initializing NCCL: MASTER={master_addr}:{master_port}, "
              f"WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")

        # Use explicit timeout to prevent infinite hanging
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(minutes=2)
        )
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print(f"Distributed training (torchrun): {world_size} GPUs")

        return rank

    # Check SLURM environment
    elif 'SLURM_PROCID' in os.environ:
        # Launched by SLURM srun
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])

        # Set environment variables for PyTorch DDP
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)

        # Debug info for NCCL troubleshooting
        master_addr = os.environ.get('MASTER_ADDR', 'NOT_SET')
        master_port = os.environ.get('MASTER_PORT', 'NOT_SET')
        print(f"[Rank {rank}] Initializing NCCL: MASTER={master_addr}:{master_port}, "
              f"WORLD_SIZE={world_size}, LOCAL_RANK={local_rank}")

        # CRITICAL: Use explicit parameters to avoid hanging
        # - init_method='env://' uses MASTER_ADDR and MASTER_PORT
        # - timeout prevents infinite waiting (default is 30 min, we use 10 min)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=2)
        )
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print(f"Distributed training (SLURM): {world_size} GPUs")
            print(f"  MASTER_ADDR: {master_addr}")
            print(f"  MASTER_PORT: {master_port}")

        return rank

    else:
        # Single GPU mode
        return 0


def get_rank():
    """Get current process rank"""
    if not is_distributed():
        return 0
    return dist.get_rank()


def get_world_size():
    """Get total number of processes"""
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_local_rank():
    """Get local rank within node"""
    if not is_distributed():
        return 0
    # Try LOCAL_RANK first (torchrun), then SLURM_LOCALID
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    elif 'SLURM_LOCALID' in os.environ:
        return int(os.environ['SLURM_LOCALID'])
    return 0


def cleanup_distributed():
    """Clean up distributed training process group"""
    if is_distributed() and dist.is_initialized():
        # Synchronize all processes before destroying
        # This ensures all ranks finish their work before cleanup
        try:
            dist.barrier()
        except Exception:
            # Barrier may fail if some ranks already exited, ignore
            pass

        # Destroy the process group
        dist.destroy_process_group()


class AllGatherWithGradient(torch.autograd.Function):
    """
    All-gather with gradient support

    Forward: Gather tensors from all GPUs
    Backward: Each GPU receives its own gradient slice

    Single GPU: Returns input unchanged
    Multi-GPU: Gathers and concatenates
    """

    @staticmethod
    def forward(ctx, tensor):
        if not is_distributed():
            return tensor

        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()

        tensor_list = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(tensor_list, tensor)

        gathered = torch.cat(tensor_list, dim=0)

        # Clear temporary list
        del tensor_list

        return gathered

    @staticmethod
    def backward(ctx, grad_output):
        if not is_distributed():
            return grad_output

        grad_list = grad_output.chunk(ctx.world_size, dim=0)
        return grad_list[ctx.rank]


# Convenient function
all_gather = AllGatherWithGradient.apply
