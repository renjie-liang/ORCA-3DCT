"""
Time and memory profiling utilities

Simplified profiling for training loops.
Fail-fast principles: No try-except, explicit errors.
"""
import time
from typing import Dict, Optional
from collections import deque

import torch
import psutil


class StepTimer:
    """
    Simple step timer with ETA calculation

    Uses moving average for smoothing.
    """

    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of recent steps to average
        """
        self.window_size = window_size
        self.step_times = deque(maxlen=window_size)
        self.start_time = time.time()
        self.last_time = time.time()

    def update(self) -> float:
        """
        Record time for current step

        Returns:
            Step duration in seconds
        """
        current_time = time.time()
        step_duration = current_time - self.last_time
        self.step_times.append(step_duration)
        self.last_time = current_time
        return step_duration

    def get_avg_step_time(self) -> float:
        """
        Get average step time (moving average)

        Returns:
            Average step duration in seconds
        """
        if len(self.step_times) == 0:
            return 0.0
        return sum(self.step_times) / len(self.step_times)

    def get_eta(self, current_step: int, total_steps: int) -> float:
        """
        Calculate ETA (Estimated Time to Arrival)

        Args:
            current_step: Current training step
            total_steps: Total training steps

        Returns:
            ETA in seconds
        """
        if current_step >= total_steps:
            return 0.0

        remaining_steps = total_steps - current_step
        avg_step_time = self.get_avg_step_time()

        if avg_step_time == 0.0:
            return 0.0

        eta = remaining_steps * avg_step_time
        return eta

    def get_elapsed_time(self) -> float:
        """
        Get total elapsed time since timer initialization

        Returns:
            Elapsed time in seconds
        """
        return time.time() - self.start_time


def get_gpu_memory() -> Dict[str, float]:
    """
    Get GPU memory usage

    Returns:
        {
            'allocated_gb': float,
            'reserved_gb': float,
            'max_allocated_gb': float
        }

    Returns empty dict if CUDA not available
    """
    if not torch.cuda.is_available():
        return {}

    return {
        'allocated_gb': torch.cuda.memory_allocated() / 1e9,
        'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9
    }


def get_process_memory() -> Dict[str, float]:
    """
    Get process memory usage

    Returns:
        {
            'rss_gb': float,  # Resident Set Size
            'vms_gb': float   # Virtual Memory Size
        }
    """
    process = psutil.Process()
    mem_info = process.memory_info()

    return {
        'rss_gb': mem_info.rss / 1e9,
        'vms_gb': mem_info.vms / 1e9
    }


def get_memory_summary() -> Dict[str, Dict[str, float]]:
    """
    Get complete memory summary

    Returns:
        {
            'gpu': {...},
            'process': {...}
        }
    """
    return {
        'gpu': get_gpu_memory(),
        'process': get_process_memory()
    }


def format_eta(seconds: float) -> str:
    """
    Format ETA as human-readable string

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2h 30m" or "45m 12s")
    """
    if seconds < 0:
        return "0s"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
