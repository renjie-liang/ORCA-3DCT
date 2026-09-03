from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dtbd3d.training.constants import IMAGE_TOKEN_INDEX
from dtbd3d.training.models import build_tiny_btb3d_model_for_tests


def _first_lora_param(model: torch.nn.Module) -> torch.nn.Parameter:
    for name, parameter in model.named_parameters():
        if "lora_" in name and parameter.requires_grad:
            return parameter
    raise AssertionError("no trainable LoRA parameter found")


def _first_frozen_q_proj_param(model: torch.nn.Module) -> torch.nn.Parameter:
    for name, parameter in model.named_parameters():
        if "q_proj" in name and "lora_" not in name:
            return parameter
    raise AssertionError("no frozen q_proj parameter found")


def test_model_forward_backward_updates_lora_only():
    _, model = build_tiny_btb3d_model_for_tests(vocab_size=128, hidden_size=32)
    input_ids = torch.tensor([[1, IMAGE_TOKEN_INDEX, 9, 10, 11, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    labels = input_ids.clone()
    labels[:, :3] = -100
    labels[input_ids == IMAGE_TOKEN_INDEX] = -100
    image_features = torch.randn(1, 2, 2, 2, 18)

    lora_param = _first_lora_param(model)
    frozen_q_proj = _first_frozen_q_proj_param(model)
    lora_before = lora_param.detach().clone()
    frozen_before = frozen_q_proj.detach().clone()

    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, image_features=image_features)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    optimizer.step()

    assert not torch.equal(lora_before, lora_param.detach())
    assert torch.equal(frozen_before, frozen_q_proj.detach())

