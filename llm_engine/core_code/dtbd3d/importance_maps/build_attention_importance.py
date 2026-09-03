"""Flavor C: attention-based (FastV-style) per-token importance.

Runs a TRAINED full-token pack2x2x2 ReportGen model forward over each volume and
reads the LLM's attention to each visual token at an early decoder layer (FastV,
Chen et al. 2024). Tokens the model attends to most = most important. Output is a
per-volume importance map at the base grid (31x32x32), so it plugs into the
existing selection machinery (`--token-selection attention_mask`).

This is TASK-DERIVED (the model's own attention) and deployable (no external
masks). Strongest legitimate importance signal — try after the panel gate.

NOTE: written without GPU; the grid-mapping is CPU-self-tested, but the model
forward + attention extraction MUST be verified on one GPU volume before a full
run (attn_implementation must be 'eager' to return attention weights; flash/sdpa
return None). See `--smoke` for a 2-volume check.

Usage (on 1 GPU):
  python -m dtbd3d.importance_maps.build_attention_importance \
    --resume <full_pack2x2x2 step_011786/training_state.pt> \
    --manifest results/reportgen_visual_tokens/e000_base_merged/manifest.json \
    --split valid --layer 2 --smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT_ROOT = Path("./data/token_importance")
BASE_GRID = (31, 32, 32)  # stored LFQ grid (D,H,W); selection resizes down to packed grid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--resume", required=True, help="training_state.pt of a TRAINED full-token pack2x2x2 model")
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", choices=["train", "valid"], required=True)
    p.add_argument("--layer", type=int, default=2, help="decoder layer to read attention from (FastV uses an early layer)")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="run 2 volumes and print, do not write")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--config-model-path", default="./llm_engine/base/llava_config")
    p.add_argument("--model-base", default="./checkpoints/Llama-3.1-8B-Instruct")
    p.add_argument("--projector-input-dim", type=int, default=144, help="pack2x2x2 = 18*8 = 144")
    return p.parse_args()


def packed_grid_from_image(image: torch.Tensor) -> tuple[int, int, int]:
    """image is (T,H,W,C) packed features -> (T,H,W)."""
    return tuple(int(x) for x in image.shape[:3])


def attention_to_grid(score_flat: np.ndarray, t: int, h: int, w: int) -> np.ndarray:
    """per-visual-token attention [T*H*W] -> base grid (31,32,32) for parity with organ maps."""
    g = torch.as_tensor(score_flat.reshape(t, h, w), dtype=torch.float32)[None, None]
    up = F.adaptive_avg_pool3d(g, output_size=BASE_GRID)[0, 0]
    return up.numpy().astype(np.float16)


@torch.no_grad()
def extract_one(model, tokenizer, batch, image_token_index: int, layer: int, device: str) -> np.ndarray:
    """Forward one sample with output_attentions; return per-visual-token importance [n_visual]."""
    input_ids = batch["input_ids"].to(device)
    images = batch["images"].to(device)
    attn_mask = batch.get("attention_mask")
    attn_mask = attn_mask.to(device) if attn_mask is not None else None

    # n_visual = number of projected visual tokens = T*H*W of the (1,T,H,W,C) image
    img = images[0] if images.ndim == 5 else images
    t, h, w = packed_grid_from_image(img)
    n_visual = t * h * w

    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        images=images,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    # attentions: tuple(num_layers) of (B, heads, q, k). LLaVA has expanded <image>
    # to n_visual positions in-place where image_token_index was.
    att = out.attentions[layer][0]  # (heads, q, k)
    # locate visual block: the contiguous run of n_visual key positions starting at
    # the (single) image token slot in the ORIGINAL input_ids, after expansion.
    ids = input_ids[0].tolist()
    try:
        img_pos = ids.index(image_token_index)
    except ValueError:
        # already expanded / not found -> assume visual block is the first n_visual
        img_pos = 0
    v0, v1 = img_pos, img_pos + n_visual
    # importance of each visual key = mean attention it RECEIVES from all query positions
    # (averaged over heads), restricted to queries after the visual block (text/report).
    att_mean = att.mean(0)  # (q, k)
    q_text = att_mean[v1:, v0:v1] if att_mean.shape[0] > v1 else att_mean[:, v0:v1]
    score = q_text.mean(0).float().cpu().numpy()  # (n_visual,)
    if score.shape[0] != n_visual:
        raise RuntimeError(f"visual block size {score.shape[0]} != n_visual {n_visual} (check img_pos/expansion)")
    return score, (t, h, w)


def main() -> int:
    import sys
    sys.path.insert(0, "Experiment/core_code")
    from dtbd3d.training.scripts.train_author_reportgen_artifact import (
        ArtifactReportgenDataset, ArtifactCollator, load_author_architecture_from_scratch,
    )
    from llava.constants import IMAGE_TOKEN_INDEX

    args = parse_args()
    # --- self-test the grid mapping on CPU (always runs, no GPU needed) ---
    demo = np.random.rand(15 * 16 * 16).astype(np.float32)
    g = attention_to_grid(demo, 15, 16, 16)
    assert g.shape == BASE_GRID, g.shape
    print(f"[selftest] attention_to_grid OK -> {g.shape}", flush=True)

    print("[load] author architecture + trained weights...", flush=True)
    tokenizer, model = load_author_architecture_from_scratch(
        config_model_path=Path(args.config_model_path),
        checkpoint_path=None,                 # fresh init; trained weights loaded below
        model_base=Path(args.model_base),
        device=torch.device(args.device),
        projector_input_dim=args.projector_input_dim,
        reinit_lora=None,
    )
    # VERIFY-ON-GPU #1: attention weights require eager attention. flash/sdpa return
    # None for out.attentions. Force eager (some HF versions need rebuild, not just a
    # config flip -> verify out.attentions is not None in --smoke before a full run).
    model.config._attn_implementation = "eager"
    if hasattr(model, "model"):
        model.model.config._attn_implementation = "eager"
    state = torch.load(args.resume, map_location="cpu")
    sd = state.get("model", state.get("model_state_dict", state))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] loaded trained sd (missing={len(missing)} unexpected={len(unexpected)})", flush=True)
    model.to(args.device).eval()

    ds = ArtifactReportgenDataset(
        Path(args.manifest), args.split, Path("/dev/null"), tokenizer,
        "16x16x8", "pack2x2x2", args.limit, "none", 0, "",
    )
    collate = ArtifactCollator(tokenizer)
    out_dir = OUT_ROOT / "attention" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    n = 2 if args.smoke else len(ds)
    for i in range(n):
        if i % args.num_shards != args.shard_id and not args.smoke:
            continue
        rec = ds.records[i]
        batch = collate([ds[i]])
        score, (t, h, w) = extract_one(model, tokenizer, batch, IMAGE_TOKEN_INDEX, args.layer, args.device)
        if args.smoke:
            print(f"[smoke] {rec.volume_id}: grid=({t},{h},{w}) n={score.size} "
                  f"top32-mean={np.sort(score)[-32:].mean():.4g} bot32-mean={np.sort(score)[:32].mean():.4g}", flush=True)
            continue
        imp = attention_to_grid(score, t, h, w)
        out_path = out_dir / f"{rec.volume_id}.npy"
        tmp = out_dir / f"{rec.volume_id}.tmp.npy"
        np.save(tmp, imp); tmp.rename(out_path)
    print("DONE" if not args.smoke else "SMOKE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
