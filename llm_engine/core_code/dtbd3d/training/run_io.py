"""Run output and artifact validation helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from dtbd3d.training.distributed import distributed_barrier


def prepare_run_output(
    args: Any,
    *,
    rank: int,
    distributed: bool,
    local_rank: int,
    write_config_fn: Callable[[Path, Any], None],
) -> Path:
    """Write rank-0 run metadata and synchronize distributed workers."""
    out_dir = Path(args.out_dir)
    if rank == 0:
        write_config_fn(out_dir, args)
    if distributed:
        distributed_barrier(local_rank)
    return out_dir


def require_existing_paths(paths: Iterable[Path], *, label: str) -> None:
    """Fail fast if any expected input path is missing."""
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return
    examples = "\n".join(str(path) for path in missing[:10])
    extra = "" if len(missing) <= 10 else f"\n... and {len(missing) - 10} more"
    raise FileNotFoundError(f"{label}: missing {len(missing)} path(s)\n{examples}{extra}")
