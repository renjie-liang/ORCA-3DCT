#!/usr/bin/env python3
"""Debug whether author ReportGen generation is invariant to eval batch size."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from dtbd3d.training.scripts.train_author_reportgen_artifact import (
    ArtifactCollator,
    ArtifactReportgenDataset,
    DEFAULT_REPRO_CONFIG,
    CORE_CODE_ROOT,
    PROJECT_ROOT,
    _ensure_python_paths,
    _model_config,
    _move_batch,
    _path,
    image_output_name,
    load_author_model_trainable,
)


def _load_repro_defaults(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text())
    paths = raw.get("paths", {})
    return {
        "btb3d_repo": str(_path(paths["btb3d_repo"])),
        "ctclip_repo": str(_path(paths["ctclip_repo"])),
        "model_path": str(_path(paths["model_path"])),
        "model_base": str(_path(paths["model_base"])),
        "valid_vqa_json": str(_path(paths["vqa_json"])),
    }


def parse_args() -> argparse.Namespace:
    defaults = _load_repro_defaults(DEFAULT_REPRO_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reportgen-artifact-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--compression", choices=["16x16x8", "8x8x8"], default="16x16x8")
    parser.add_argument("--model-path", default=defaults["model_path"])
    parser.add_argument("--model-base", default=defaults["model_base"])
    parser.add_argument("--btb3d-repo", default=defaults["btb3d_repo"])
    parser.add_argument("--ctclip-repo", default=defaults["ctclip_repo"])
    parser.add_argument("--valid-vqa-json", default=defaults["valid_vqa_json"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--valid-limit", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--padding-side", choices=["left", "right"], default="left")
    parser.add_argument(
        "--repeat-same-sample",
        action="store_true",
        help="Build the batch from repeated copies of --single-index. Separates padding bugs from batch-kernel numeric drift.",
    )
    parser.add_argument(
        "--no-generate-cache",
        action="store_true",
        help="Call generate(..., use_cache=False). Slow, but helps isolate KV-cache effects.",
    )
    parser.add_argument(
        "--model-float32",
        action="store_true",
        help="Convert model and image features to float32 for a small diagnostic run. Uses much more memory.",
    )
    parser.add_argument(
        "--single-index",
        type=int,
        default=-1,
        help="Only compare one dataset row. Default compares the first --batch-size rows.",
    )
    return parser.parse_args()


def _unwrap_peft(model: torch.nn.Module) -> torch.nn.Module:
    candidate = getattr(model, "base_model", model)
    if hasattr(candidate, "prepare_inputs_labels_for_multimodal"):
        return candidate
    candidate_model = getattr(candidate, "model", None)
    if candidate_model is not None and hasattr(candidate_model, "prepare_inputs_labels_for_multimodal"):
        return candidate_model
    return candidate


def _position_ids_from_attention(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def _prepare_prefill(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    image_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    batch = _move_batch(batch, device)
    batch["images"] = batch["images"].to(dtype=image_dtype)
    base_model = _unwrap_peft(model)
    with torch.inference_mode():
        (
            _,
            prepared_position_ids,
            prepared_attention_mask,
            _,
            inputs_embeds,
            _,
        ) = base_model.prepare_inputs_labels_for_multimodal(
            batch["prompt_input_ids"],
            None,
            batch["prompt_attention_mask"],
            None,
            None,
            batch["images"],
            image_sizes=[1 for _ in batch["metadata"]],
        )
    if inputs_embeds is None or prepared_attention_mask is None:
        raise RuntimeError("multimodal prefill did not produce inputs_embeds/attention_mask")
    explicit_position_ids = (
        prepared_position_ids
        if prepared_position_ids is not None
        else _position_ids_from_attention(prepared_attention_mask)
    )
    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": prepared_attention_mask,
        "position_ids": explicit_position_ids,
    }


def _last_valid_logits(
    *,
    model: torch.nn.Module,
    prepared: dict[str, torch.Tensor],
) -> torch.Tensor:
    attention_mask = prepared["attention_mask"]
    last_indices = attention_mask.long().sum(dim=1) - 1
    if torch.all(attention_mask[:, -1] == 1):
        last_indices = torch.full_like(last_indices, attention_mask.shape[1] - 1)
    with torch.inference_mode():
        outputs = model(
            input_ids=None,
            inputs_embeds=prepared["inputs_embeds"],
            attention_mask=attention_mask,
            position_ids=prepared["position_ids"],
            use_cache=False,
            return_dict=True,
        )
    return outputs.logits[torch.arange(outputs.logits.shape[0], device=outputs.logits.device), last_indices]


def _tensor_compare(single: torch.Tensor, batched: torch.Tensor) -> dict[str, float]:
    single_f = single.detach().float().cpu()
    batched_f = batched.detach().float().cpu()
    diff = (single_f - batched_f).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "single_norm": float(single_f.norm().item()),
        "batched_norm": float(batched_f.norm().item()),
    }


def _generate(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    batch: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
    repetition_penalty: float,
    image_dtype: torch.dtype,
    use_cache: bool,
) -> torch.Tensor:
    batch = _move_batch(batch, device)
    batch["images"] = batch["images"].to(dtype=image_dtype)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.inference_mode():
        return model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            images=batch["images"],
            image_sizes=[1 for _ in batch["metadata"]],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )


def _first_token_diff(a: list[int], b: list[int]) -> int | None:
    for idx, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return idx
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True).replace("<|eot_id|>", "").replace(tokenizer.pad_token or "", "").strip()


def main() -> int:
    args = parse_args()
    if args.batch_size < 2:
        raise ValueError("--batch-size should be >= 2 for invariance debugging")
    torch.manual_seed(args.seed)

    btb3d_repo = _path(args.btb3d_repo)
    ctclip_repo = _path(args.ctclip_repo)
    _ensure_python_paths(btb3d_repo, ctclip_repo)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)

    tokenizer, model = load_author_model_trainable(
        _path(args.model_path),
        _path(args.model_base),
        device,
    )
    image_dtype = torch.float32 if args.model_float32 else torch.bfloat16
    if args.model_float32:
        model = model.to(dtype=torch.float32)
    model.eval()
    tokenizer.model_max_length = int(getattr(_model_config(model), "tokenizer_model_max_length", 128000))
    tokenizer.padding_side = args.padding_side
    _model_config(model).tokenizer_padding_side = args.padding_side

    dataset = ArtifactReportgenDataset(
        _path(args.reportgen_artifact_manifest),
        "valid",
        _path(args.valid_vqa_json),
        tokenizer,
        args.compression,
        args.valid_limit,
    )
    collator = ArtifactCollator(tokenizer)
    if args.repeat_same_sample:
        repeated_index = args.single_index if args.single_index >= 0 else 0
        if repeated_index >= len(dataset):
            raise IndexError(f"--single-index {repeated_index} out of range for dataset length {len(dataset)}")
        batch_indices = [repeated_index for _ in range(args.batch_size)]
        compare_pairs = [(repeated_index, row) for row in range(args.batch_size)]
    elif args.single_index >= 0:
        batch_indices = list(range(min(args.batch_size, len(dataset))))
        if args.single_index >= len(dataset):
            raise IndexError(f"--single-index {args.single_index} out of range for dataset length {len(dataset)}")
        if args.single_index not in batch_indices:
            batch_indices[-1] = args.single_index
        compare_pairs = [(args.single_index, batch_indices.index(args.single_index))]
    else:
        batch_indices = list(range(min(args.batch_size, len(dataset))))
        compare_pairs = [(index, row) for row, index in enumerate(batch_indices)]
    batch_items = [dataset[index] for index in batch_indices]
    batch = collator(batch_items)

    prepared_batch = _prepare_prefill(model=model, batch=batch, device=device, image_dtype=image_dtype)
    batch_logits = _last_valid_logits(model=model, prepared=prepared_batch)
    batch_generated = _generate(
        model=model,
        tokenizer=tokenizer,
        batch=batch,
        device=device,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        image_dtype=image_dtype,
        use_cache=not args.no_generate_cache,
    )

    reports = []
    for index, batch_row in compare_pairs:
        single = collator([dataset[index]])
        prepared_single = _prepare_prefill(model=model, batch=single, device=device, image_dtype=image_dtype)
        single_logits = _last_valid_logits(model=model, prepared=prepared_single)
        single_generated = _generate(
            model=model,
            tokenizer=tokenizer,
            batch=single,
            device=device,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            image_dtype=image_dtype,
            use_cache=not args.no_generate_cache,
        )

        single_tokens = [int(x) for x in single_generated[0].detach().cpu().tolist()]
        batch_tokens = [int(x) for x in batch_generated[batch_row].detach().cpu().tolist()]
        single_top = torch.topk(single_logits[0].detach().float().cpu(), k=5)
        batch_top = torch.topk(batch_logits[batch_row].detach().float().cpu(), k=5)

        single_valid = prepared_single["attention_mask"][0].bool()
        batch_valid = prepared_batch["attention_mask"][batch_row].bool()
        single_embeds = prepared_single["inputs_embeds"][0, single_valid]
        batch_embeds = prepared_batch["inputs_embeds"][batch_row, batch_valid]
        single_positions = prepared_single["position_ids"][0, single_valid]
        batch_positions = prepared_batch["position_ids"][batch_row, batch_valid]
        single_mask_valid = prepared_single["attention_mask"][0, single_valid]
        batch_mask_valid = prepared_batch["attention_mask"][batch_row, batch_valid]
        if single_embeds.shape != batch_embeds.shape:
            embed_compare: dict[str, Any] = {
                "shape_match": False,
                "single_shape": list(single_embeds.shape),
                "batched_shape": list(batch_embeds.shape),
            }
        else:
            embed_compare = {"shape_match": True, **_tensor_compare(single_embeds, batch_embeds)}

        record = {
            "dataset_index": index,
            "batch_row": batch_row,
            "image": image_output_name(single["metadata"][0]["image"]),
            "prompt": single["metadata"][0]["question"],
            "prompt_token_shape_single": list(single["prompt_input_ids"].shape),
            "prompt_token_shape_batch": list(batch["prompt_input_ids"].shape),
            "prefill_shape_single": list(prepared_single["inputs_embeds"].shape),
            "prefill_shape_batch": list(prepared_batch["inputs_embeds"].shape),
            "valid_prefill_tokens_single": int(prepared_single["attention_mask"][0].sum().item()),
            "valid_prefill_tokens_batch": int(prepared_batch["attention_mask"][batch_row].sum().item()),
            "embeds": embed_compare,
            "attention_mask": {
                "valid_shape_match": tuple(single_mask_valid.shape) == tuple(batch_mask_valid.shape),
                "valid_equal": bool(
                    tuple(single_mask_valid.shape) == tuple(batch_mask_valid.shape)
                    and torch.equal(single_mask_valid.detach().cpu(), batch_mask_valid.detach().cpu())
                ),
                "single_sum": int(prepared_single["attention_mask"][0].sum().item()),
                "batch_sum": int(prepared_batch["attention_mask"][batch_row].sum().item()),
                "single_full_len": int(prepared_single["attention_mask"].shape[1]),
                "batch_full_len": int(prepared_batch["attention_mask"].shape[1]),
            },
            "position_ids": {
                "valid_shape_match": tuple(single_positions.shape) == tuple(batch_positions.shape),
                "valid_equal": bool(
                    tuple(single_positions.shape) == tuple(batch_positions.shape)
                    and torch.equal(single_positions.detach().cpu(), batch_positions.detach().cpu())
                ),
                "single_valid_first": int(single_positions[0].item()) if single_positions.numel() else None,
                "single_valid_last": int(single_positions[-1].item()) if single_positions.numel() else None,
                "batch_valid_first": int(batch_positions[0].item()) if batch_positions.numel() else None,
                "batch_valid_last": int(batch_positions[-1].item()) if batch_positions.numel() else None,
            },
            "prefill_logits": {
                **_tensor_compare(single_logits[0], batch_logits[batch_row]),
                "top1_equal": int(single_top.indices[0].item()) == int(batch_top.indices[0].item()),
                "single_top5": [
                    {"token_id": int(tok), "logit": float(val)}
                    for tok, val in zip(single_top.indices.tolist(), single_top.values.tolist())
                ],
                "batched_top5": [
                    {"token_id": int(tok), "logit": float(val)}
                    for tok, val in zip(batch_top.indices.tolist(), batch_top.values.tolist())
                ],
            },
            "generation": {
                "tokens_equal": single_tokens == batch_tokens,
                "first_diff_index": _first_token_diff(single_tokens, batch_tokens),
                "single_token_count": len(single_tokens),
                "batch_token_count": len(batch_tokens),
                "single_prefix_token_ids": single_tokens[:32],
                "batch_prefix_token_ids": batch_tokens[:32],
                "single_text": _decode(tokenizer, single_tokens),
                "batch_text": _decode(tokenizer, batch_tokens),
            },
        }
        reports.append(record)
        print(
            json.dumps(
                {
                    "image": record["image"],
                    "embed_max_abs": record["embeds"].get("max_abs"),
                    "logit_max_abs": record["prefill_logits"]["max_abs"],
                    "top1_equal": record["prefill_logits"]["top1_equal"],
                    "tokens_equal": record["generation"]["tokens_equal"],
                    "first_diff_index": record["generation"]["first_diff_index"],
                },
                sort_keys=True,
            )
        )

    summary = {
        "batch_size": args.batch_size,
        "padding_side": args.padding_side,
        "repeat_same_sample": args.repeat_same_sample,
        "generate_use_cache": not args.no_generate_cache,
        "model_float32": args.model_float32,
        "samples": len(reports),
        "embedding_all_equalish": all(
            rec["embeds"].get("shape_match") and rec["embeds"].get("max_abs", 1.0) < 1e-5
            for rec in reports
        ),
        "prefill_top1_all_equal": all(rec["prefill_logits"]["top1_equal"] for rec in reports),
        "generation_all_equal": all(rec["generation"]["tokens_equal"] for rec in reports),
    }
    payload = {"summary": summary, "records": reports}
    out_path = _path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote_author_reportgen_batch_debug=" + str(out_path))
    print(json.dumps({"event": "summary", **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
