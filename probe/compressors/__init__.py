"""compressors: ONE FILE PER compression method, assembled into an EXPLICIT literal REGISTRY below.
Each method is `fn(grid[T,H,W,C], budget, score, **params) -> tokens[N', C']` (numpy; fixed / no training).

  apply(name, grid, budget=None, score=None, organ=None, **params) -> [N', C'] float32

This release implements the two methods that form the core comparison in the paper:
  * Grid average (`avgpack`) -- the anatomy-blind pooling baseline.
  * ORCA -- Ward region merge (`agglo_merge`, the base) plus a soft organ prior (`agglo_organ`, the full method).
"""
import inspect
import numpy as np

from .primitives import foreground_score, resize_score_to_grid, organ_vec, attn_score  # noqa: F401  (re-exported for run.py)
from .avgpack import avgpack
from .branch_c.agglo_merge import agglo_merge
from .branch_c.agglo_organ import agglo_organ

_SKIP = {"grid", "budget", "score", "organ", "labels"}


def _params(fn):   # tunable params = keyword args with a default (excludes grid/budget/score)
    return {k: v.default for k, v in inspect.signature(fn).parameters.items()
            if k not in _SKIP and v.default is not inspect.Parameter.empty}


def _m(fn, branch, label, needs_score=False, needs_budget=True, needs_organ=False, variable_len=False,
       score_source="foreground"):
    return {"fn": fn, "branch": branch, "label": label, "params": _params(fn),
            "needs_score": needs_score, "needs_budget": needs_budget, "needs_organ": needs_organ,
            "variable_len": variable_len, "score_source": score_source}


# ---- the registry: one row per method; params auto-read from the method file ----------------------
REGISTRY = {
    "avgpack":     _m(avgpack,     "A", "Grid average: avg-pool r^3 block -> 1 token (channel-preserving)", needs_budget=False),   # param: r
    "agglo_merge": _m(agglo_merge, "C", "ORCA base: Ward region merge (variable-size, budget-reallocating)"),
    "agglo_organ": _m(agglo_organ, "C", "ORCA: Ward merge + soft organ prior (embedding-dominant, organ-guided)", needs_organ=True),   # param: lam
}


def apply(name, grid, budget=None, score=None, organ=None, **params):
    m = REGISTRY[name]
    kw = {k: params[k] for k in m["params"] if k in params}      # pass only the params the YAML overrides
    if m["needs_organ"]:
        kw["organ"] = organ
    return m["fn"](grid, budget, score, **kw).astype(np.float32)
