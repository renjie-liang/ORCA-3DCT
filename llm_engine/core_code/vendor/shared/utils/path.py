"""
Path utilities for experiment management

Provides utilities for adding timestamps to experiment paths
"""
from datetime import datetime
from pathlib import Path
import re


def add_timestamp_to_stage_path(path: str, timestamp: str = None) -> str:
    """
    Add timestamp to stage name in experiment path

    Transforms paths like:
        ./results/stage1_ctclip/baseline/checkpoints
        -> ./results/stage1_ctclip_20251222_143025/baseline/checkpoints

        ./results/stage3_generation/baseline/predictions
        -> ./results/stage3_generation_20251222_150830/baseline/predictions

    Args:
        path: Original path with stage name
        timestamp: Timestamp string (YYYYMMDD_HHMMSS). If None, generates current timestamp.

    Returns:
        Path with timestamp appended to stage name

    Examples:
        >>> add_timestamp_to_stage_path("./results/stage1_ctclip/baseline/checkpoints", "20251222_143025")
        './results/stage1_ctclip_20251222_143025/baseline/checkpoints'

        >>> add_timestamp_to_stage_path("./results/stage3_generation/inference/predictions")
        './results/stage3_generation_20251222_143025/inference/predictions'
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to Path for easier manipulation
    path_obj = Path(path)
    parts = list(path_obj.parts)

    # Find and modify the stage name (stage1_ctclip, stage3_generation, etc.)
    stage_pattern = re.compile(r'^stage\d+_\w+$')

    for i, part in enumerate(parts):
        if stage_pattern.match(part):
            # Append timestamp to stage name
            parts[i] = f"{part}_{timestamp}"
            break

    # Reconstruct path
    new_path = str(Path(*parts))

    return new_path


def get_current_timestamp() -> str:
    """
    Get current timestamp in standard format

    Returns:
        Timestamp string in format YYYYMMDD_HHMMSS

    Example:
        >>> get_current_timestamp()
        '20251222_143025'
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")
