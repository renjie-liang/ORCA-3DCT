"""Build the BTB3D-style LLaVA report-generation model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from dtbd3d.training.constants import (
    BTB3D_SPECIAL_TOKENS,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    LORA_TARGET_MODULES,
    PAD_TOKEN,
)


@dataclass(frozen=True)
class BTB3DModelConfig:
    """Configuration for the trainable BTB3D report-generation wrapper."""

    model_name_or_path: str
    mm_projector_type: str
    mm_hidden_size: int
    mm_context_size: int
    hidden_size: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_bias: str
    image_context_max_length: int
    max_visual_tokens: int
    freeze_base_model: bool


class MLP2xGELUProjector(nn.Module):
    """The released BTB3D `attn_pool+mlp2x_gelu` path flattens 3D tokens then projects."""

    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        if image_features.ndim != 5:
            raise ValueError(f"image_features must be B,T,H,W,C, got {tuple(image_features.shape)}")
        flat = image_features.flatten(1, 3)
        return self.proj(flat)


def build_mm_projector(projector_type: str, input_dim: int, hidden_size: int) -> nn.Module:
    """Build the BTB3D multimodal projector."""

    if projector_type == "linear":
        return nn.Sequential(nn.Flatten(start_dim=1, end_dim=3), nn.Linear(input_dim, hidden_size))
    if projector_type == "attn_pool+mlp2x_gelu":
        return MLP2xGELUProjector(input_dim, hidden_size)
    if projector_type == "mlp2x_gelu":
        return MLP2xGELUProjector(input_dim, hidden_size)
    raise ValueError(f"unknown mm_projector_type: {projector_type}")


def _clone_tensor_for_save(tensor: torch.Tensor, name: str) -> torch.Tensor:
    cloned = tensor.detach().cpu().contiguous().clone()
    if cloned.numel() == 0:
        raise RuntimeError(f"refusing to save empty tensor for {name}")
    return cloned


def _module_state_for_save(module: nn.Module, prefix: str = "") -> dict[str, torch.Tensor]:
    return {
        f"{prefix}{name}": _clone_tensor_for_save(value, f"{prefix}{name}")
        for name, value in module.state_dict().items()
    }


def smart_tokenizer_and_embedding_resize(tokenizer: Any, model: nn.Module, pad_token: str) -> int:
    """Add a pad token and initialize new embeddings from the old mean."""

    before = len(tokenizer)
    num_new_tokens = tokenizer.add_special_tokens({"pad_token": pad_token})
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data
        input_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_avg
        output_embeddings[-num_new_tokens:] = output_avg
    return before


def prepare_btb3d_tokenizer(tokenizer: Any, model: nn.Module) -> int:
    """Add BTB3D special tokens in the author-preserved order."""

    base_vocab_size = len(tokenizer)
    if tokenizer.pad_token is None:
        base_vocab_size = smart_tokenizer_and_embedding_resize(tokenizer, model, PAD_TOKEN)
    added = tokenizer.add_tokens(BTB3D_SPECIAL_TOKENS, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    if added != len(BTB3D_SPECIAL_TOKENS):
        raise ValueError(f"expected to add {len(BTB3D_SPECIAL_TOKENS)} task tokens, added {added}")
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return base_vocab_size


def _zero_original_token_grads(grad: torch.Tensor, base_vocab_size: int) -> torch.Tensor:
    grad = grad.clone()
    grad[:base_vocab_size] = 0
    return grad


class BTB3DReportGenerator(nn.Module):
    """A minimal LLaVA-style wrapper: LFQ visual prefix + LLaMA causal LM."""

    def __init__(
        self,
        language_model: nn.Module,
        mm_projector: nn.Module,
        config: BTB3DModelConfig,
        base_vocab_size: int,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.mm_projector = mm_projector
        self.config = config
        self.base_vocab_size = base_vocab_size

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _merge_text_and_image(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor | None,
        projected_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.LongTensor | None]:
        embed_tokens = self.language_model.get_input_embeddings()
        merged_embeds: list[torch.Tensor] = []
        merged_masks: list[torch.Tensor] = []
        merged_labels: list[torch.Tensor] = []

        for batch_idx in range(input_ids.shape[0]):
            valid_len = int(attention_mask[batch_idx].sum().item())
            cur_ids = input_ids[batch_idx, :valid_len]
            image_positions = torch.where(cur_ids == IMAGE_TOKEN_INDEX)[0]
            if image_positions.numel() != 1:
                raise ValueError(f"sample {batch_idx} must contain exactly one image token")
            image_pos = int(image_positions[0].item())

            pre_ids = cur_ids[:image_pos]
            post_ids = cur_ids[image_pos + 1 :]
            text_ids = torch.cat([pre_ids, post_ids])
            text_embeds = embed_tokens(text_ids)
            pre_embeds = text_embeds[: pre_ids.numel()]
            post_embeds = text_embeds[pre_ids.numel() :]

            cur_image = projected_images[batch_idx]
            cur_embeds = torch.cat([pre_embeds, cur_image, post_embeds], dim=0)
            cur_mask = torch.ones(cur_embeds.shape[0], dtype=attention_mask.dtype, device=attention_mask.device)
            if labels is not None:
                cur_labels = labels[batch_idx, :valid_len]
                pre_labels = cur_labels[:image_pos]
                post_labels = cur_labels[image_pos + 1 :]
                image_labels = torch.full(
                    (cur_image.shape[0],),
                    IGNORE_INDEX,
                    dtype=cur_labels.dtype,
                    device=cur_labels.device,
                )
                merged_labels.append(torch.cat([pre_labels, image_labels, post_labels], dim=0))

            if cur_embeds.shape[0] > self.config.image_context_max_length:
                cur_embeds = cur_embeds[: self.config.image_context_max_length]
                cur_mask = cur_mask[: self.config.image_context_max_length]
                if labels is not None:
                    merged_labels[-1] = merged_labels[-1][: self.config.image_context_max_length]

            merged_embeds.append(cur_embeds)
            merged_masks.append(cur_mask)

        max_len = max(item.shape[0] for item in merged_embeds)
        hidden = merged_embeds[0].shape[1]
        batch = len(merged_embeds)
        padded_embeds = torch.zeros(
            (batch, max_len, hidden),
            dtype=merged_embeds[0].dtype,
            device=merged_embeds[0].device,
        )
        padded_masks = torch.zeros((batch, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        padded_labels = None
        if labels is not None:
            padded_labels = torch.full(
                (batch, max_len),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )

        for idx, cur_embeds in enumerate(merged_embeds):
            cur_len = cur_embeds.shape[0]
            padded_embeds[idx, :cur_len] = cur_embeds
            padded_masks[idx, :cur_len] = merged_masks[idx]
            if padded_labels is not None:
                padded_labels[idx, :cur_len] = merged_labels[idx]

        return padded_embeds, padded_masks, padded_labels

    def _compress_visual_prefix(self, projected_images: torch.Tensor) -> torch.Tensor:
        """Bound LLaMA sequence length by deterministic strided token subsampling."""

        if self.config.max_visual_tokens <= 0:
            raise ValueError(f"max_visual_tokens must be positive, got {self.config.max_visual_tokens}")
        if projected_images.shape[1] <= self.config.max_visual_tokens:
            return projected_images
        indices = torch.linspace(
            0,
            projected_images.shape[1] - 1,
            steps=self.config.max_visual_tokens,
            device=projected_images.device,
        ).round().long()
        return projected_images.index_select(1, indices)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.LongTensor,
        image_features: torch.Tensor,
    ) -> Any:
        image_features = image_features.to(device=input_ids.device, dtype=self.mm_projector_dtype)
        projected_images = self.mm_projector(image_features)
        projected_images = self._compress_visual_prefix(projected_images)
        inputs_embeds, merged_attention_mask, merged_labels = self._merge_text_and_image(
            input_ids,
            attention_mask,
            labels,
            projected_images,
        )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=merged_attention_mask,
            labels=merged_labels,
            use_cache=False,
        )

    @property
    def mm_projector_dtype(self) -> torch.dtype:
        return next(self.mm_projector.parameters()).dtype

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        image_features: torch.Tensor,
        generation_kwargs: dict[str, Any],
    ) -> torch.LongTensor:
        image_features = image_features.to(device=input_ids.device, dtype=self.mm_projector_dtype)
        projected_images = self.mm_projector(image_features)
        projected_images = self._compress_visual_prefix(projected_images)
        inputs_embeds, merged_attention_mask, _ = self._merge_text_and_image(
            input_ids,
            attention_mask,
            None,
            projected_images,
        )
        pad_token_id = getattr(self.language_model.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.language_model.config, "eos_token_id", 0)
        dummy_input_ids = torch.full(
            (inputs_embeds.shape[0], inputs_embeds.shape[1]),
            int(pad_token_id),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        output_ids = self.language_model.generate(
            input_ids=dummy_input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=merged_attention_mask,
            **generation_kwargs,
        )
        return output_ids[:, dummy_input_ids.shape[1] :]

    def save_pretrained(self, output_dir: Path, tokenizer: Any | None) -> None:
        """Save LoRA adapters, projector weights, and trainable token rows in HF-style files."""

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_model_config.json").write_text(json.dumps(asdict(self.config), indent=2))
        (output_dir / "base_vocab_size.json").write_text(json.dumps({"base_vocab_size": self.base_vocab_size}))

        projector_state = _module_state_for_save(self.mm_projector, prefix="mm_projector.")
        save_file(projector_state, output_dir / "mm_projector.safetensors")

        input_embeddings = self.language_model.get_input_embeddings().weight
        new_rows = _clone_tensor_for_save(
            input_embeddings[self.base_vocab_size :],
            "model.embed_tokens.new_rows",
        )
        new_token_state = {"model.embed_tokens.new_rows": new_rows}
        save_file(new_token_state, output_dir / "new_token_embeddings.safetensors")

        self.language_model.config.save_pretrained(output_dir)
        if hasattr(self.language_model, "save_pretrained"):
            self.language_model.save_pretrained(output_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: Path,
        base_model_name_or_path: str,
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> "BTB3DReportGenerator":
        """Load a checkpoint saved by `save_pretrained`."""

        cfg = BTB3DModelConfig(**json.loads((checkpoint_dir / "training_model_config.json").read_text()))
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False, trust_remote_code=True)
        language_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        base_vocab_size = prepare_btb3d_tokenizer(tokenizer, language_model)
        language_model = PeftModel.from_pretrained(language_model, str(checkpoint_dir), is_trainable=True)
        mm_projector = build_mm_projector(cfg.mm_projector_type, cfg.mm_hidden_size, cfg.hidden_size)
        mm_projector = mm_projector.to(dtype=torch_dtype)
        model = cls(language_model, mm_projector, cfg, base_vocab_size)
        projector_state = load_file(checkpoint_dir / "mm_projector.safetensors")
        model.mm_projector.load_state_dict({k.removeprefix("mm_projector."): v for k, v in projector_state.items()})
        new_rows = load_file(checkpoint_dir / "new_token_embeddings.safetensors")["model.embed_tokens.new_rows"]
        with torch.no_grad():
            model.language_model.get_input_embeddings().weight[base_vocab_size:] = new_rows
        model.to(device)
        return model


def configure_trainable_parameters(model: BTB3DReportGenerator) -> None:
    """Freeze base weights while leaving LoRA, projector, and new token rows trainable."""

    if model.config.freeze_base_model:
        for parameter in model.language_model.parameters():
            parameter.requires_grad = False
    model.language_model = get_peft_model(
        model.language_model,
        LoraConfig(
            r=model.config.lora_r,
            lora_alpha=model.config.lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=model.config.lora_dropout,
            bias=model.config.lora_bias,
            task_type="CAUSAL_LM",
        ),
    )
    for parameter in model.mm_projector.parameters():
        parameter.requires_grad = True

    embedding_weight = model.language_model.get_input_embeddings().weight
    embedding_weight.requires_grad = True
    embedding_weight.register_hook(lambda grad: _zero_original_token_grads(grad, model.base_vocab_size))


def build_llava_btb3d_model_with_lora(config: BTB3DModelConfig) -> tuple[Any, BTB3DReportGenerator]:
    """Load LLaMA, add BTB3D tokens, attach projector, and apply LoRA."""

    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=False, trust_remote_code=True)
    language_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_vocab_size = prepare_btb3d_tokenizer(tokenizer, language_model)
    mm_projector = build_mm_projector(config.mm_projector_type, config.mm_hidden_size, config.hidden_size)
    mm_projector = mm_projector.to(dtype=torch.bfloat16)
    model = BTB3DReportGenerator(language_model, mm_projector, config, base_vocab_size)
    configure_trainable_parameters(model)
    return tokenizer, model


def build_tiny_btb3d_model_for_tests(vocab_size: int, hidden_size: int) -> tuple[Any, BTB3DReportGenerator]:
    """Build a tiny random LLaMA-backed model for CPU unit tests."""

    class TinyTokenizer:
        bos_token_id = 1
        pad_token_id = 0
        pad_token = "<pad>"
        model_max_length = 128

        def __len__(self) -> int:
            return vocab_size

        def __call__(self, text: str) -> Any:
            ids = [self.bos_token_id]
            ids.extend([(ord(ch) % (vocab_size - 8)) + 8 for ch in text])
            return type("Tokenized", (), {"input_ids": ids})

    llama_config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    language_model = LlamaForCausalLM(llama_config)
    cfg = BTB3DModelConfig(
        model_name_or_path="tiny-random",
        mm_projector_type="attn_pool+mlp2x_gelu",
        mm_hidden_size=18,
        mm_context_size=18,
        hidden_size=hidden_size,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_bias="none",
        image_context_max_length=512,
        max_visual_tokens=8,
        freeze_base_model=True,
    )
    tokenizer = TinyTokenizer()
    model = BTB3DReportGenerator(
        language_model=language_model,
        mm_projector=build_mm_projector(cfg.mm_projector_type, cfg.mm_hidden_size, cfg.hidden_size),
        config=cfg,
        base_vocab_size=vocab_size - 4,
    )
    configure_trainable_parameters(model)
    return tokenizer, model
