"""Pydantic-validated experiment config. One YAML per experiment (an exp_id + a compression method+params + a
probe setup). Load with `load_config(path)`; the runner (run.py) consumes the validated object. No Hydra.

Validation catches typos early: unknown encoder / family / compression method, or params not accepted by the
chosen method, fail at load time (not 3h into a job)."""
import sys
from pathlib import Path
import yaml
from pydantic import BaseModel, field_validator, model_validator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import ENCODERS, FAMILIES                # noqa: E402
import compressors as co                             # noqa: E402


class CompressionCfg(BaseModel):
    method: str
    budget: int | None = None                        # token budget (None for pack / count-invariant methods)
    params: dict = {}                                # method-specific overrides (validated against the method's params)

    @field_validator("method")
    @classmethod
    def _known_method(cls, v):
        if v not in co.REGISTRY:
            raise ValueError(f"unknown compression method {v!r}; known: {list(co.REGISTRY)}")
        return v

    @model_validator(mode="after")
    def _check(self):
        m = co.REGISTRY[self.method]
        bad = set(self.params) - set(m["params"])
        if bad:
            raise ValueError(f"method {self.method!r} does not accept params {bad}; accepts {list(m['params'])}")
        if m["needs_budget"] and self.budget is None:
            raise ValueError(f"method {self.method!r} needs a budget")
        return self


class ProbeCfg(BaseModel):
    families: list[str]                              # families to probe (each its OWN shared-multi-head model)
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    bs: int = 48
    dropout: float = 0.1
    readout_dim: int = 256
    readout_heads: int = 4
    seeds: list[int] = [2026]
    proj: int | None = None                          # optional Linear(token_dim -> proj) BEFORE the readout; default None
    #                                                  (= no proj, numbers identical to prior runs). Set (e.g. 512) to
    #                                                  stabilize high-dim compressions like pack4 (diag: pack4 mean R2 0.47->0.65).

    @field_validator("families")
    @classmethod
    def _known_families(cls, v):
        bad = [f for f in v if f not in FAMILIES]
        if bad:
            raise ValueError(f"unknown families {bad}; known: {list(FAMILIES)}")
        return v


class DataCfg(BaseModel):
    limit: int = 0                                   # 0 = full; limits TRAIN only
    valid_limit: int = 0                             # 0 = full valid (default, matches all prior runs); set >0 ONLY for
    #                                                  cheap-anchor runs where valid R2 need not be exact (e.g. btb3d Ward),
    #                                                  since capping valid CHANGES the R2 estimate -- never set on paper-grade runs.
    num_workers: int = 12
    prefetch: int = 2


class ExperimentCfg(BaseModel):
    exp_id: str
    desc: str = ""
    encoder: str
    compression: CompressionCfg
    probe: ProbeCfg
    data: DataCfg = DataCfg()

    @field_validator("encoder")
    @classmethod
    def _known_encoder(cls, v):
        if v not in ENCODERS:
            raise ValueError(f"unknown encoder {v!r}; known: {list(ENCODERS)}")
        return v


def load_config(path):
    cfg = ExperimentCfg(**yaml.safe_load(open(path)))
    return cfg


if __name__ == "__main__":   # quick validate: python schema.py experiments/foo.yaml
    c = load_config(sys.argv[1])
    print("OK:", c.model_dump())
