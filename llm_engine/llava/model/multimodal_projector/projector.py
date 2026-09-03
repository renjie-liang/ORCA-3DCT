"""The visual projector this study trains.

Replaces BTB3D's `coca_attentional_pooler.py`. That file was named for CoCa's attentional pooler, but
the pooler, its LayerNorm and its 3D positional encoding were all commented out upstream; what actually
ran was a flatten followed by the MLP that `build_vision_projector` passes in. This is that, written out.

The module holds no parameters of its own. All 20,029,440 trainable weights live in `self.proj`, so that
attribute name is load-bearing: it is what makes the state-dict keys
`...mm_projector.proj.{0,2}.{weight,bias}`, and checkpoints written by the original module load unchanged.
"""
import torch
from torch import nn


class FlattenProjector(nn.Module):
    """[B, X, Y, Z, D] token grid -> [B, X*Y*Z, D] -> the language model's width.

    The grid axes carry no meaning for ORCA, whose regions are unordered and stored as [budget, 1, 1];
    grid average keeps a real cube. Either way the language model receives X*Y*Z visual tokens, which is
    exactly the budget being compared, so the two arms stay budget-matched.
    """

    def __init__(self, projector: nn.Module):
        super().__init__()
        self.proj = projector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected a [B, X, Y, Z, D] visual token grid, got {tuple(x.shape)}")
        return self.proj(x.flatten(1, 3))
