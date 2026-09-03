"""Prefix-token pre-alignment against a frozen causal Llama objective."""

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dtbd3d.diagnostic_token_learning.config import LlamaPrefixPrealignmentConfig
from dtbd3d.diagnostic_token_learning.llama_space_alignment import merge_tokens_spatial_2x2
from dtbd3d.diagnostic_token_learning.text_pair_sampling import finding_text_from_pair, normalize_finding_text


def _rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _torch_dtype(name: str) -> torch.dtype:
    normalized = str(name).strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported llama_prefix_prealignment.dtype={name!r}")


class PrefixTokenProjector(nn.Module):
    """Compress CT token grid into K vectors in the target Llama hidden space.

    Expected input after reportgen merge is [B,D,H,W,C]. It is flattened to
    [B,N,C], adaptively pooled to K prefix slots, then projected to hidden_dim.
    """

    def __init__(self, visual_dim: int, hidden_dim: int, num_prefix_tokens: int) -> None:
        super().__init__()
        self.num_prefix_tokens = int(num_prefix_tokens)
        if self.num_prefix_tokens <= 0:
            raise ValueError(f"num_prefix_tokens must be positive, got {num_prefix_tokens}")
        self.projector = nn.Sequential(
            nn.LayerNorm(int(visual_dim)),
            nn.Linear(int(visual_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
        )

    def forward(self, tokens_bdhwc: torch.Tensor) -> torch.Tensor:
        if tokens_bdhwc.ndim != 5:
            raise ValueError(f"expected [B,D,H,W,C] tokens, got {tuple(tokens_bdhwc.shape)}")
        bsz, depth, height, width, channels = tokens_bdhwc.shape
        flat = tokens_bdhwc.reshape(bsz, depth * height * width, channels).float()
        pooled = F.adaptive_avg_pool1d(flat.transpose(1, 2), self.num_prefix_tokens).transpose(1, 2)
        return self.projector(pooled)


class LlamaPrefixPrealignment(nn.Module):
    """Train CT prefix tokens by predicting RadGenome finding text with frozen Llama."""

    def __init__(self, config: LlamaPrefixPrealignmentConfig, visual_dim: int) -> None:
        super().__init__()
        self.config = config
        self.visual_dim = int(visual_dim)
        self.hidden_dim = int(config.hidden_dim)
        self.target_dhw = tuple(int(x) for x in config.reportgen_grid_dhw)
        self.num_prefix_tokens = int(config.num_prefix_tokens)
        self.max_pairs_per_batch = int(config.max_pairs_per_batch)
        if not config.freeze_llama:
            raise ValueError("llama_prefix_prealignment currently supports frozen Llama only")
        self.prefix_projector = PrefixTokenProjector(self.visual_dim, self.hidden_dim, self.num_prefix_tokens)
        # Keep the frozen Llama out of state_dict/optimizer; only the prefix
        # projector is trainable and checkpointed in this encoder-training stage.
        self._tokenizer_holder: list[Any | None] = [None]
        self._llm_holder: list[nn.Module | None] = [None]

    def _ensure_llama(self) -> tuple[Any, nn.Module]:
        if self._llm_holder[0] is not None and self._tokenizer_holder[0] is not None:
            return self._tokenizer_holder[0], self._llm_holder[0]
        if not self.config.model_name_or_path:
            raise ValueError("llama_prefix_prealignment.model_name_or_path is required when enabled")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name_or_path,
            local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        llm = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            dtype=_torch_dtype(self.config.dtype),
            local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code,
        )
        for param in llm.parameters():
            param.requires_grad_(False)
        llm.eval()
        self._tokenizer_holder[0] = tokenizer
        self._llm_holder[0] = llm
        return tokenizer, llm

    def _prefix_tokens(self, z_quantized: torch.Tensor) -> torch.Tensor:
        target_dhw = (int(z_quantized.shape[2]), int(self.target_dhw[1]), int(self.target_dhw[2]))
        merged = merge_tokens_spatial_2x2(z_quantized, target_dhw)
        return self.prefix_projector(merged.to(dtype=next(self.prefix_projector.parameters()).dtype))

    def _pair_text(self, pair: dict[str, Any]) -> tuple[str, str]:
        finding = finding_text_from_pair(pair).strip()
        if not finding:
            finding = str(pair.get("text", "")).strip()
        finding = finding.strip(" .")
        if not finding:
            raise ValueError(f"text pair has empty finding text: keys={sorted(pair.keys())}")
        region = str(pair.get("organ_group", pair.get("matched_key", "region"))).replace("_", " ")
        prompt = self.config.prompt_template.format(region=region, organ_group=region, finding=finding)
        target = self.config.target_template.format(region=region, organ_group=region, finding=finding)
        return prompt, target

    def _collect_examples(self, text_batch: Any | None, volume_ids: list[str]) -> list[tuple[int, str, dict[str, Any], str, str]]:
        examples: list[tuple[int, str, dict[str, Any], str, str]] = []
        if text_batch is None:
            return examples
        for batch_index, volume_id in enumerate(volume_ids):
            sample_pairs = text_batch[batch_index] if batch_index < len(text_batch) else []
            for pair in sample_pairs:
                prompt, target = self._pair_text(pair)
                examples.append((batch_index, str(volume_id), pair, prompt, target))
                if self.max_pairs_per_batch > 0 and len(examples) >= self.max_pairs_per_batch:
                    return examples
        return examples

    def _build_inputs(
        self,
        *,
        tokenizer: Any,
        llm: nn.Module,
        prefix_tokens: torch.Tensor,
        examples: list[tuple[int, str, dict[str, Any], str, str]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        embed = llm.get_input_embeddings()
        device = prefix_tokens.device
        dtype = embed.weight.dtype
        sequences: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        target_token_count = 0
        bos_id = tokenizer.bos_token_id
        for batch_index, _volume_id, _pair, prompt, target in examples:
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            if not target_ids:
                continue
            input_ids = ([] if bos_id is None else [int(bos_id)]) + list(prompt_ids) + list(target_ids)
            prompt_label_len = (0 if bos_id is None else 1) + len(prompt_ids)
            ids = torch.tensor(input_ids, device=device, dtype=torch.long)
            text_embeds = embed(ids)
            prefix = prefix_tokens[batch_index].to(device=device, dtype=dtype)
            sequences.append(torch.cat([prefix, text_embeds], dim=0))
            label = torch.full((self.num_prefix_tokens + len(input_ids),), -100, device=device, dtype=torch.long)
            label[self.num_prefix_tokens + prompt_label_len :] = torch.tensor(target_ids, device=device, dtype=torch.long)
            labels.append(label)
            target_token_count += len(target_ids)
        if not sequences:
            raise RuntimeError("llama_prefix_prealignment found examples but all target token lists were empty")

        max_len = max(int(seq.shape[0]) for seq in sequences)
        hidden_dim = int(sequences[0].shape[-1])
        inputs_embeds = torch.zeros((len(sequences), max_len, hidden_dim), device=device, dtype=dtype)
        attention_mask = torch.zeros((len(sequences), max_len), device=device, dtype=torch.long)
        padded_labels = torch.full((len(sequences), max_len), -100, device=device, dtype=torch.long)
        for row, (seq, label) in enumerate(zip(sequences, labels, strict=True)):
            length = int(seq.shape[0])
            inputs_embeds[row, :length] = seq
            attention_mask[row, :length] = 1
            padded_labels[row, :length] = label
        return inputs_embeds, attention_mask, padded_labels, target_token_count

    def forward(
        self,
        *,
        z_quantized: torch.Tensor,
        volume_ids: list[str],
        text_batch: Any | None = None,
        step: int = 0,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if not self.config.enabled:
            zero = z_quantized.new_zeros(())
            return zero, {"llama_prefix_loss": 0.0, "llama_prefix_pairs": 0}

        examples = self._collect_examples(text_batch, volume_ids)
        if not examples:
            zero = z_quantized.new_zeros(())
            return zero, {
                "llama_prefix_loss": 0.0,
                "llama_prefix_pairs": 0,
                "llama_prefix_target_tokens": 0,
            }

        tokenizer, llm = self._ensure_llama()
        if next(llm.parameters()).device != z_quantized.device:
            llm.to(z_quantized.device)
        llm.eval()
        prefix_tokens = self._prefix_tokens(z_quantized)
        inputs_embeds, attention_mask, labels, target_token_count = self._build_inputs(
            tokenizer=tokenizer,
            llm=llm,
            prefix_tokens=prefix_tokens,
            examples=examples,
        )
        output = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False)
        loss = output.loss
        if step == 1 and _rank0():
            print(
                "[debug:llama_prefix_prealignment] "
                + json.dumps(
                    {
                        "z_quantized_shape": list(z_quantized.shape),
                        "prefix_tokens_shape": list(prefix_tokens.shape),
                        "inputs_embeds_shape": list(inputs_embeds.shape),
                        "pairs": len(examples),
                        "target_tokens": target_token_count,
                        "example_targets": [normalize_finding_text(item[4]) for item in examples[:4]],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        return loss, {
            "llama_prefix_loss": float(loss.detach().cpu()),
            "llama_prefix_pairs": len(examples),
            "llama_prefix_target_tokens": int(target_token_count),
            "llama_prefix_prefix_tokens": int(self.num_prefix_tokens),
        }
