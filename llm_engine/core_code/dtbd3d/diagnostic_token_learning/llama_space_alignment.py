"""Llama-space region-text alignment for diagnosis-aware token learning."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dtbd3d.diagnostic_token_learning.config import TextAlignmentConfig, TextEmbeddingSourceConfig
from dtbd3d.diagnostic_token_learning.text_pair_sampling import select_abnormal_first_pairs
from dtbd3d.diagnostic_token_learning.volume_id import normalize_volume_id


SMOKE_DEBUG_PRINTS_HARDCODED = True
SMOKE_DEBUG_BREAKPOINT_HARDCODED = True
SMOKE_DEBUG_BREAKPOINT_STEP = 1


def _rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _breakpoints_enabled() -> bool:
    return "PYTEST_CURRENT_TEST" not in os.environ


@dataclass(frozen=True)
class _TextPair:
    source_index: int
    embedding_row: int
    organ_group: str
    mask_name: str
    text_hash: str
    text: str
    row: dict[str, Any]


class TokenWiseLlamaProjector(nn.Module):
    """Project reportgen-grid CT token features into Llama hidden space."""

    def __init__(self, visual_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(visual_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, tokens_bdhwc: torch.Tensor) -> torch.Tensor:
        if tokens_bdhwc.ndim != 5:
            raise ValueError(f"expected B,D,H,W,C tokens, got {tuple(tokens_bdhwc.shape)}")
        return self.proj(tokens_bdhwc)


def merge_tokens_spatial_2x2(z_quantized: torch.Tensor, target_dhw: tuple[int, int, int]) -> torch.Tensor:
    """Convert encoder tokens B,C,D,16,16 to reportgen-grid tokens B,D,8,8,C."""
    if z_quantized.ndim != 5:
        raise ValueError(f"expected B,C,D,H,W latent tokens, got {tuple(z_quantized.shape)}")
    bsz, channels, depth, height, width = z_quantized.shape
    target_d, target_h, target_w = target_dhw
    if depth != target_d:
        raise ValueError(f"depth mismatch: latent depth={depth}, target depth={target_d}")
    if height % target_h != 0 or width % target_w != 0:
        raise ValueError(f"cannot merge latent grid {(depth, height, width)} to {target_dhw}")
    kernel = (height // target_h, width // target_w)
    x = z_quantized.float().permute(0, 2, 1, 3, 4).reshape(bsz * depth, channels, height, width)
    x = F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
    return x.reshape(bsz, depth, channels, target_h, target_w).permute(0, 1, 3, 4, 2).contiguous()


def downsample_mask_64_to_grid(mask_dhw: np.ndarray, target_dhw: tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Average-pool a saved [D,64,64] mask to the reportgen grid [D,8,8]."""
    target_d, target_h, target_w = target_dhw
    mask = torch.as_tensor(mask_dhw, device=device, dtype=torch.float32)
    if mask.ndim != 3:
        raise ValueError(f"expected mask D,H,W, got {tuple(mask.shape)}")
    if mask.shape[0] != target_d:
        mask = F.interpolate(mask[None, None], size=target_dhw, mode="trilinear", align_corners=False)[0, 0]
        return mask.to(dtype=dtype).clamp_min(0.0)
    height, width = int(mask.shape[1]), int(mask.shape[2])
    if height % target_h != 0 or width % target_w != 0:
        mask = F.interpolate(mask[None, None], size=target_dhw, mode="trilinear", align_corners=False)[0, 0]
        return mask.to(dtype=dtype).clamp_min(0.0)
    kernel = (height // target_h, width // target_w)
    pooled = F.avg_pool2d(mask.reshape(target_d, 1, height, width), kernel_size=kernel, stride=kernel)
    return pooled.reshape(target_d, target_h, target_w).to(dtype=dtype).clamp_min(0.0)


class _EmbeddingSource:
    def __init__(self, config: TextEmbeddingSourceConfig, source_index: int) -> None:
        self.config = config
        self.source_index = int(source_index)
        split_dir = Path(config.artifact_dir) / config.split
        self.index_path = split_dir / "index.jsonl"
        self.embedding_path = split_dir / "embeddings.npy"
        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)
        if not self.embedding_path.exists():
            raise FileNotFoundError(self.embedding_path)
        self.embeddings = np.load(self.embedding_path, mmap_mode="r")
        encoded_mask_path = split_dir / "encoded_mask.npy"
        if not encoded_mask_path.exists():
            raise FileNotFoundError(f"{encoded_mask_path} is required; formal training requires complete text artifacts")
        encoded_mask = np.asarray(np.load(encoded_mask_path, mmap_mode="r"), dtype=bool)
        if encoded_mask.shape != (int(self.embeddings.shape[0]),):
            raise ValueError(f"{encoded_mask_path} shape={encoded_mask.shape}, expected {(int(self.embeddings.shape[0]),)}")
        encoded_rows = int(encoded_mask.sum())
        if encoded_rows != int(encoded_mask.shape[0]):
            raise ValueError(f"{encoded_mask_path} is incomplete: encoded_rows={encoded_rows}/{encoded_mask.shape[0]}")
        self.pairs_by_volume = self._load_pairs()

    def _load_pairs(self) -> dict[str, list[_TextPair]]:
        pairs: dict[str, list[_TextPair]] = defaultdict(list)
        row_count = 0
        with self.index_path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row_count += 1
                row = json.loads(line)
                volume_id = normalize_volume_id(str(row.get("volume_id", "")))
                mask_name = str(row.get("mask_name", ""))
                organ_group = str(row.get("organ_group", ""))
                if not volume_id or not mask_name or not organ_group:
                    raise ValueError(f"invalid text embedding index row at {self.index_path}:{line_no}")
                embedding_row = int(row.get("embedding_row", row.get("row_id", -1)))
                if embedding_row < 0 or embedding_row >= int(self.embeddings.shape[0]):
                    raise ValueError(f"embedding_row out of range at {self.index_path}:{line_no}")
                pairs[volume_id].append(
                    _TextPair(
                        source_index=self.source_index,
                        embedding_row=embedding_row,
                        organ_group=organ_group,
                        mask_name=mask_name,
                        text_hash=str(row.get("text_hash", "")),
                        text=str(row.get("text", row.get("target_text", row.get("sentence", row.get("prompt", ""))))),
                        row=row,
                    )
                )
        if row_count != int(self.embeddings.shape[0]):
            raise ValueError(f"{self.index_path} rows={row_count}, embeddings rows={int(self.embeddings.shape[0])}")
        return dict(pairs)

    def select_pairs(self, volume_id: str, step: int) -> list[_TextPair]:
        del step
        norm_volume_id = normalize_volume_id(volume_id)
        pairs = self.pairs_by_volume.get(norm_volume_id, [])
        max_pairs = int(self.config.max_pairs_per_volume)
        return select_abnormal_first_pairs(
            pairs,
            max_pairs=max_pairs,
            volume_id=norm_volume_id,
            source_name=str(self.config.name),
        )


class LlamaSpaceTextAlignment(nn.Module):
    """Align region-pooled projected CT tokens to frozen Llama hidden embeddings."""

    def __init__(self, config: TextAlignmentConfig, visual_dim: int) -> None:
        super().__init__()
        self.config = config
        self.visual_dim = int(visual_dim)
        self.hidden_dim = int(config.hidden_dim)
        self.target_dhw = tuple(int(x) for x in config.reportgen_grid_dhw)
        self.projector = TokenWiseLlamaProjector(self.visual_dim, self.hidden_dim)
        self.sources: list[_EmbeddingSource] | None = None

    def _ensure_sources(self) -> list[_EmbeddingSource]:
        if self.sources is None:
            self.sources = [_EmbeddingSource(source, index) for index, source in enumerate(self.config.embedding_sources)]
        for source in self.sources:
            if int(source.embeddings.shape[1]) != self.hidden_dim:
                raise ValueError(
                    f"{source.embedding_path} has dim={source.embeddings.shape[1]}, expected {self.hidden_dim}"
                )
        return self.sources

    def _load_mask_dir(
        self,
        mask_root: Path,
        split: str,
        volume_id: str,
        cache_key: tuple[int, int, str, tuple[int, int, int]],
        cache: dict[tuple[int, int, str, tuple[int, int, int]], dict[str, torch.Tensor]],
        device: torch.device,
        target_dhw: tuple[int, int, int],
    ) -> dict[str, torch.Tensor]:
        if cache_key in cache:
            return cache[cache_key]
        path = mask_root / split / f"{volume_id}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        npz = np.load(path, allow_pickle=False)
        names = [str(x) for x in npz["mask_names"]]
        groups = [str(x) for x in npz["mask_groups"]] if "mask_groups" in npz.files else names
        lookup: dict[str, torch.Tensor] = {}
        grouped: dict[str, list[torch.Tensor]] = defaultdict(list)
        for idx, name in enumerate(names):
            mask = downsample_mask_64_to_grid(npz["mask_token"][idx], target_dhw, device, torch.float32)
            lookup[name] = mask
            grouped[groups[idx]].append(mask)
        for group_name, masks in grouped.items():
            lookup[group_name] = torch.stack(masks, dim=0).amax(dim=0) if len(masks) > 1 else masks[0]
        cache[cache_key] = lookup
        return lookup

    def _load_mask(
        self,
        pair: _TextPair,
        volume_id: str,
        cache: dict[tuple[int, int, str, tuple[int, int, int]], dict[str, torch.Tensor]],
        device: torch.device,
        target_dhw: tuple[int, int, int],
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        sources = self._ensure_sources()
        source = sources[pair.source_index]
        norm_volume_id = normalize_volume_id(volume_id)
        lookup = self._load_mask_dir(
            Path(source.config.mask_artifact_dir),
            source.config.split,
            norm_volume_id,
            (pair.source_index, 0, norm_volume_id, target_dhw),
            cache,
            device,
            target_dhw,
        )
        alignment_key = pair.organ_group
        mask = lookup.get(alignment_key)
        if mask is None:
            raise KeyError(
                f"text alignment mask key {alignment_key!r} missing for volume={norm_volume_id} "
                f"mask_name={pair.mask_name!r} source={source.config.name}; "
                f"available={sorted(lookup.keys())[:32]}"
            )
        return mask, {
            "mask_root": source.config.mask_artifact_dir,
            "matched_key": alignment_key,
            "requested_mask_name": pair.mask_name,
            "requested_organ_group": pair.organ_group,
        }

    def _text_embedding(self, pair: _TextPair, device: torch.device) -> torch.Tensor:
        sources = self._ensure_sources()
        source = sources[pair.source_index]
        vector = np.asarray(source.embeddings[pair.embedding_row]).copy()
        return torch.as_tensor(vector, device=device, dtype=torch.float32)

    @staticmethod
    def _multi_positive_contrastive(
        visual: torch.Tensor,
        text: torch.Tensor,
        positive_keys: list[tuple[str, str]],
        temperature: float,
    ) -> torch.Tensor:
        visual = F.normalize(visual.float(), dim=-1)
        text = F.normalize(text.float(), dim=-1)
        logits = visual @ text.t() / max(float(temperature), 1e-6)
        key_count = len(positive_keys)
        positive = torch.eye(key_count, dtype=torch.bool, device=logits.device)
        for i, key_i in enumerate(positive_keys):
            for j, key_j in enumerate(positive_keys):
                if key_i == key_j:
                    positive[i, j] = True
        log_prob_v = F.log_softmax(logits, dim=1)
        log_prob_t = F.log_softmax(logits.t(), dim=1)
        pos_f = positive.float()
        loss_v = -((log_prob_v * pos_f).sum(dim=1) / pos_f.sum(dim=1).clamp_min(1.0)).mean()
        loss_t = -((log_prob_t * pos_f.t()).sum(dim=1) / pos_f.t().sum(dim=1).clamp_min(1.0)).mean()
        return 0.5 * (loss_v + loss_t)

    def forward(
        self,
        *,
        z_quantized: torch.Tensor,
        volume_ids: list[str],
        full_shape_dhw: tuple[int, int, int],
        crop_depth_value: int,
        step: int,
        text_batch: Any | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del full_shape_dhw, crop_depth_value
        if not self.config.enabled:
            zero = z_quantized.new_zeros(())
            return zero, {"text_loss": 0.0, "text_pairs": 0}

        target_dhw = (int(z_quantized.shape[2]), int(self.target_dhw[1]), int(self.target_dhw[2]))
        merged = merge_tokens_spatial_2x2(z_quantized, target_dhw)
        projected = self.projector(merged.to(dtype=next(self.projector.parameters()).dtype))
        visual_vectors: list[torch.Tensor] = []
        text_vectors: list[torch.Tensor] = []
        positive_keys: list[tuple[str, str]] = []
        mask_cache: dict[tuple[int, int, str, tuple[int, int, int]], dict[str, torch.Tensor]] = {}
        device = z_quantized.device
        selected_pair_count = 0
        supervised_volume_count = 0
        missing_mask_count = 0
        empty_mask_count = 0
        debug_matches: list[dict[str, Any]] = []

        if text_batch is not None:
            for batch_index, volume_id in enumerate(volume_ids):
                sample_pairs = text_batch[batch_index] if batch_index < len(text_batch) else []
                if sample_pairs:
                    supervised_volume_count += 1
                for pair in sample_pairs:
                    selected_pair_count += 1
                    raw_mask = pair.get("mask")
                    mask_debug = {
                        "mask_root": "",
                        "matched_key": str(pair.get("matched_key", "")),
                        "requested_mask_name": str(pair.get("mask_name", "")),
                        "requested_organ_group": str(pair.get("organ_group", "")),
                    }
                    if raw_mask is None:
                        raise RuntimeError(f"text alignment missing preloaded mask: volume_id={volume_id} {mask_debug}")
                    mask = downsample_mask_64_to_grid(raw_mask, target_dhw, device, torch.float32)
                    denom = mask.sum().clamp_min(float(self.config.min_mask_sum))
                    if float(denom.detach().cpu()) <= float(self.config.min_mask_sum):
                        raise RuntimeError(f"text alignment empty preloaded mask: volume_id={volume_id} {mask_debug}")
                    region_visual = (projected[batch_index].float() * mask.unsqueeze(-1)).sum(dim=(0, 1, 2)) / denom
                    visual_vectors.append(region_visual)
                    text_vectors.append(pair["embedding"].to(device=device, dtype=torch.float32))
                    positive_keys.append((str(pair.get("organ_group", "")), str(pair.get("text_hash", ""))))
                    if len(debug_matches) < 12:
                        debug_matches.append(
                            {
                                "volume_id": volume_id,
                                "status": "matched",
                                "denom": float(denom.detach().cpu()),
                                **mask_debug,
                            }
                        )
        else:
            sources = self._ensure_sources()
            if not sources:
                zero = z_quantized.new_zeros(())
                return zero, {"text_loss": 0.0, "text_pairs": 0}
            for batch_index, volume_id in enumerate(volume_ids):
                volume_pair_count = 0
                for source in sources:
                    for pair in source.select_pairs(volume_id, step):
                        selected_pair_count += 1
                        volume_pair_count += 1
                        mask, mask_debug = self._load_mask(pair, volume_id, mask_cache, device, target_dhw)
                        denom = mask.sum().clamp_min(float(self.config.min_mask_sum))
                        if float(denom.detach().cpu()) <= float(self.config.min_mask_sum):
                            raise RuntimeError(f"text alignment empty mask: volume_id={volume_id} {mask_debug}")
                        region_visual = (projected[batch_index].float() * mask.unsqueeze(-1)).sum(dim=(0, 1, 2)) / denom
                        visual_vectors.append(region_visual)
                        text_vectors.append(self._text_embedding(pair, device))
                        positive_keys.append((pair.organ_group, pair.text_hash))
                        if len(debug_matches) < 12:
                            debug_matches.append(
                                {
                                    "volume_id": volume_id,
                                    "status": "matched",
                                    "denom": float(denom.detach().cpu()),
                                    **mask_debug,
                                }
                            )
                if volume_pair_count > 0:
                    supervised_volume_count += 1

        no_text_pairs = selected_pair_count == 0
        if SMOKE_DEBUG_PRINTS_HARDCODED and step == SMOKE_DEBUG_BREAKPOINT_STEP and _rank0():
            print(
                "[debug:text_alignment] "
                + json.dumps(
                    {
                        "volume_ids": volume_ids,
                        "z_quantized_shape": list(z_quantized.shape),
                        "merged_shape": list(merged.shape),
                        "projected_shape": list(projected.shape),
                        "selected_pairs": selected_pair_count,
                        "used_pairs": len(visual_vectors),
                        "supervised_volumes": supervised_volume_count,
                        "no_text_pairs": no_text_pairs,
                        "missing_masks": missing_mask_count,
                        "empty_masks": empty_mask_count,
                        "configured_target_dhw": list(self.target_dhw),
                        "target_dhw": list(target_dhw),
                        "examples": debug_matches,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if (
            SMOKE_DEBUG_BREAKPOINT_HARDCODED
            and step == SMOKE_DEBUG_BREAKPOINT_STEP
            and _rank0()
            and _breakpoints_enabled()
        ):
            print("[debug:breakpoint] text alignment collected pairs; inspect visual_vectors/text_vectors/debug_matches, then type c", flush=True)
            breakpoint()

        if not visual_vectors:
            if no_text_pairs:
                zero = z_quantized.new_zeros(())
                return zero, {
                    "text_loss": 0.0,
                    "text_pairs": 0,
                    "text_candidate_pairs": 0,
                    "text_supervised_volumes": 0,
                }
            raise RuntimeError(
                "text_alignment produced zero usable pairs; "
                f"selected_pairs={selected_pair_count} missing_masks={missing_mask_count} empty_masks={empty_mask_count} "
                f"volume_ids={volume_ids}"
            )
        visual = torch.stack(visual_vectors)
        text = torch.stack(text_vectors)
        if self.config.mode in {"cosine"} or visual.shape[0] == 1:
            loss = 1.0 - (F.normalize(visual.float(), dim=-1) * F.normalize(text.float(), dim=-1)).sum(dim=-1).mean()
        elif self.config.mode in {"infonce", "multi_positive_infonce"}:
            loss = self._multi_positive_contrastive(visual, text, positive_keys, self.config.temperature)
        else:
            raise ValueError(f"unsupported Llama-space text_alignment.mode={self.config.mode!r}")
        return loss, {
            "text_loss": float(loss.detach().cpu()),
            "text_pairs": len(visual_vectors),
            "text_candidate_pairs": selected_pair_count,
            "text_supervised_volumes": supervised_volume_count,
        }
