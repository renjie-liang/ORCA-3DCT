"""probe_families: ONE FILE PER PROBE FAMILY, assembled into an EXPLICIT literal FAMILY_REGISTRY below
(no decorator / no auto-scan -- every family + its spec is visible and greppable in one place). Mirrors
compress_operators/ ("add a method = add a file + one row"). Add a family = add a file + one import + one row.

Each `{name}.py` exports a `SPEC` dict: task, metric, readout, role, train, valid [, log1p_cols, log1p_scale].
Consumed as `config.FAMILIES` (config.py re-exports FAMILY_REGISTRY). Entry point: `get(name) -> SPEC`.

Design decision (2026-07): every family is probed in its OWN separate run (no multi-task mixing) -- so a single
place enumerating them, one file each, keeps runs and specs decoupled exactly like the compression operators."""
from .disease import SPEC as _disease
from .disease_merlin import SPEC as _disease_merlin
from .density_merlin import SPEC as _density_merlin
from .size_merlin import SPEC as _size_merlin
from .location_merlin import SPEC as _location_merlin
from .size import SPEC as _size
from .density import SPEC as _density
from .location import SPEC as _location
from .radiomics import SPEC as _radiomics

# ---- the registry: one row per family; all specs explicit ----------------------------------------
FAMILY_REGISTRY = {
    "disease":        _disease,
    "disease_merlin": _disease_merlin,   # Merlin dataset; CT-RATE families do not apply to it and vice versa
    "density_merlin": _density_merlin,   # Merlin HU density: vert/muscle/liver-spleen
    "size_merlin":    _size_merlin,      # Merlin organ size: spleen/kidney/aorta, absolute AND vert-ratio form
    "location_merlin": _location_merlin, # Merlin landmark z-position: kidney_z/kidney_lr_asym/bladder_z
    "size":           _size,
    "density":        _density,
    "location":       _location,
    "radiomics":      _radiomics,
}


def get(name):
    return FAMILY_REGISTRY[name]
