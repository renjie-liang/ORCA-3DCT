"""Rank-aware logging helpers for training runs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_rank0_logger(name: str, out_dir: Path, rank: int, filename: str = "train.log") -> logging.Logger:
    """Create a logger that writes to stdout and file on rank 0 only."""
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)

    if int(rank) != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    out_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(out_dir / filename)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
