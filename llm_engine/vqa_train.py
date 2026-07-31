#!/usr/bin/env python3
"""Continue BTB3D author ReportGen LoRA training from token artifacts.

This entrypoint intentionally uses the frozen BTB3D/LLaVA model stack, not the
new soft-prefix ReportGen module. It loads the released author LoRA checkpoint,
feeds either author binary tokens or DTBD3D learned-codebook token artifacts,
and saves checkpoints compatible with ``dtbd3d.eval.run_report_generation``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
import yaml
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dtbd3d.core.visual_embedding import (
    REPORTGEN_CHANNEL_ORDER,
    load_reportgen_codebook,
    materialize_reportgen_features_from_codes,
    resolve_reportgen_artifact_manifest,
)
from dtbd3d.core.token_codec import CODEBOOK_DIM
from dtbd3d.eval.make_reportgen_vqa_subset import volume_id_from_image
from dtbd3d.eval.btb3d_to_metrics_jsonl import convert_btb3d_jsonl
from dtbd3d.training.checkpoint import optimizer_state_to_cpu
from dtbd3d.training.logging_utils import setup_rank0_logger


# SELF-CONTAINED copy (2026-07-15): dtbd3d engine vendored into CompressToken/llm_engine (no DTBD3D dependency).
# PROJECT_ROOT only backs _path() defaults for report-gen-only fields (btb3d_repo/radbert/reports_csv) which our
# MCQ cells override via --model-path/--train-vqa-json/--reportgen-artifact-manifest + --skip-metrics. CORE_CODE_ROOT
# points at the vendored package root so DEFAULT_REPRO_CONFIG resolves to the vendored repro yaml.
PROJECT_ROOT = Path(".")
CORE_CODE_ROOT = Path("./llm_engine/core_code")
DEFAULT_REPRO_CONFIG = CORE_CODE_ROOT / "dtbd3d" / "configs" / "repro_16x16x8.yaml"


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_repro_defaults(path: Path) -> dict[str, str]:
    raw = yaml.safe_load(path.read_text())
    paths = raw.get("paths", {})
    return {
        "btb3d_repo": str(_path(paths["btb3d_repo"])),
        "ctclip_repo": str(_path(paths["ctclip_repo"])),
        "model_path": str(_path(paths["model_path"])),
        "model_base": str(_path(paths["model_base"])),
        "train_vqa_json": str(_path("Experiment/core_code/data_links/ct_rate/dataset/vqa/train_vqa.json")),
        "valid_vqa_json": str(_path(paths["vqa_json"])),
        "valid_reports_csv": str(_path(paths["reports_csv"])),
        "valid_labels_csv": str(_path(paths["labels_csv"])),
        "radbert_checkpoint": str(_path(paths["radbert_ckpt"])),
    }


def _short_path(value: Any) -> str:
    if value is None:
        return "none"
    text = str(value)
    try:
        path = Path(text)
    except TypeError:
        return text
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return text


def _format_params(count: int | float | None) -> str:
    if count is None:
        return "unknown"
    value = float(count)
    if value >= 1e9:
        return f"{value / 1e9:.2f}B"
    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.1f}K"
    return str(int(value))


def _log_startup_summary(logger: Any, config: dict[str, Any]) -> None:
    trainable = config.get("trainable_parameters", {})
    logger.info(
        "[startup] mode=%s run=%s rank=%s/%s device=%s",
        "eval_only" if config.get("eval_only") else "train",
        config.get("run_name"),
        config.get("rank"),
        config.get("world_size"),
        config.get("device"),
    )
    logger.info(
        "[paths] out_dir=%s manifest=%s resume=%s init_weights=%s",
        _short_path(config.get("run_dir")),
        _short_path(config.get("reportgen_artifact_manifest")),
        _short_path(config.get("resume_from_checkpoint") or config.get("resume_checkpoint") or ""),
        _short_path(config.get("init_weights_from_checkpoint_resolved") or config.get("init_weights_from_checkpoint") or ""),
    )
    logger.info(
        "[model] init=%s author_checkpoint=%s base=%s trainable=%s total=%s",
        "scratch" if config.get("init_from_scratch") else "author_checkpoint",
        _short_path(config.get("model_path")),
        _short_path(config.get("model_base")),
        _format_params(trainable.get("trainable")),
        _format_params(trainable.get("total")),
    )
    logger.info(
        "[visual] projector_input_dim=%s token_selection=%s token_budget=%s",
        config.get("projector_input_dim"),
        config.get("token_selection"),
        config.get("token_budget"),
    )
    logger.info(
        "[eval] valid_samples=%s limit=%s max_new_tokens=%s repetition_penalty=%s eval_batch_size=%s shard=%s/%s",
        config.get("valid_samples"),
        config.get("valid_limit"),
        config.get("max_new_tokens"),
        config.get("repetition_penalty"),
        config.get("eval_batch_size"),
        int(config.get("valid_shard_index", 0)) + 1,
        config.get("valid_num_shards"),
    )
    if not config.get("eval_only"):
        logger.info(
            "[train] start_step=%s steps=%s train_samples=%s micro_batch_size=%s grad_accum=%s effective_batch_size=%s save_every=%s eval_every=%s",
            config.get("start_step"),
            config.get("steps"),
            config.get("train_samples"),
            config.get("batch_size"),
            config.get("gradient_accumulation_steps"),
            config.get("effective_batch_size"),
            config.get("save_every"),
            config.get("eval_every"),
        )
        logger.info(
            "[optimization] lr=%s mm_projector_lr=%s deepspeed=%s keep_deepspeed_checkpoints=%s gradient_checkpointing=%s projector_only=%s",
            config.get("lr"),
            config.get("mm_projector_lr"),
            _short_path(config.get("deepspeed_config") or ""),
            config.get("keep_deepspeed_checkpoints"),
            config.get("gradient_checkpointing"),
            config.get("projector_only"),
        )


def parse_args() -> argparse.Namespace:
    defaults = _load_repro_defaults(DEFAULT_REPRO_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reportgen-artifact-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="reportgen_artifact_train")
    parser.add_argument(
        "--token-selection",
        choices=["none", "uniform_pool"],
        default="none",
        help="Visual token count reduction: none=keep the full grid; "
        "uniform_pool=MERGE (avg-pool) the grid to --token-budget tokens, per-token dim preserved.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=0,
        help="Number of visual tokens to keep when --token-selection=uniform_pool (0 = keep all).",
    )
    parser.add_argument("--model-path", default=defaults["model_path"])
    parser.add_argument("--model-base", default=defaults["model_base"])
    parser.add_argument(
        "--init-from-scratch",
        action="store_true",
        help=(
            "Use the author LLaVA config but initialize LoRA/mm_projector from scratch. "
            "By default the released author ReportGen checkpoint is loaded."
        ),
    )
    parser.add_argument(
        "--projector-only",
        action="store_true",
        help="Freeze all LoRA weights and train only the multimodal projector.",
    )
    parser.add_argument("--btb3d-repo", default=defaults["btb3d_repo"])
    parser.add_argument("--ctclip-repo", default=defaults["ctclip_repo"])
    parser.add_argument("--train-vqa-json", default=defaults["train_vqa_json"])
    parser.add_argument("--valid-vqa-json", default=defaults["valid_vqa_json"])
    # Paper-2 image-grounded VQA: filter records by this id-prefix (report-gen default unchanged); and allow
    # MULTIPLE QA per volume (multi-task) instead of the one-report-per-volume dedup.
    parser.add_argument("--record-type", default="report_generation")
    parser.add_argument("--multi-qa-per-volume", action="store_true")
    parser.add_argument("--valid-reports-csv", default=defaults["valid_reports_csv"])
    parser.add_argument("--valid-labels-csv", default=defaults["valid_labels_csv"])
    parser.add_argument("--radbert-checkpoint", default=defaults["radbert_checkpoint"])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--valid-limit", type=int, default=8)
    parser.add_argument(
        "--valid-num-shards",
        type=int,
        default=1,
        help="Split validation records into this many deterministic shards for parallel eval-only generation.",
    )
    parser.add_argument(
        "--valid-shard-index",
        type=int,
        default=0,
        help="Validation shard index in [0, --valid-num-shards).",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of micro-batches accumulated per optimizer update. --steps counts optimizer updates.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--mm-projector-lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run the author checkpoint generation/evaluation on the valid split and exit without training.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="During generation eval, write raw/prediction JSONL files but skip eval_fast.py metrics.",
    )
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam-search width during generation eval. 1 = greedy (default, unchanged behavior).",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Beam-search length penalty (>1 favors longer outputs). Only active when --num-beams>1.",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=0,
        help="Minimum new tokens to generate; raise to counter the BLEU brevity penalty.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="Override LoRA rank. With --init-weights-from-checkpoint, loads only the warm "
        "projector and builds a FRESH adapter at this rank (instead of loading the "
        "checkpoint's adapter). Ignored on --resume-checkpoint.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Override LoRA alpha; used with --lora-r. Defaults to 2*lora_r if unset.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable LLaMA gradient checkpointing; recommended for full-token ReportGen training.",
    )
    parser.add_argument(
        "--deepspeed-config",
        default="",
        help="Optional DeepSpeed config JSON. When set, launch with deepspeed/torchrun for ZeRO training.",
    )
    parser.add_argument(
        "--keep-deepspeed-checkpoints",
        type=int,
        default=1,
        help=(
            "Number of newest checkpoints that keep full DeepSpeed optimizer/ZeRO state. "
            "Older checkpoints keep compact model weights for eval and weight-only init. "
            "Set to -1 to keep DeepSpeed state for every checkpoint."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        "--resume-from-checkpoint",
        default="",
        help="Resume full training state from training_state_latest.pt, a step checkpoint dir, checkpoints dir, or run dir.",
    )
    parser.add_argument(
        "--init-weights-from-checkpoint",
        default="",
        help=(
            "Initialize model weights from a training checkpoint but start a fresh training run. "
            "Optimizer, scheduler, sampler state, and step are not restored. "
            "Use this for transitions such as projector-only stage 1 to LoRA+projector stage 2."
        ),
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)
    parser.add_argument(
        "--skip-final-save",
        action="store_true",
        help="Do not write the final PEFT checkpoint; useful for short ZeRO memory probes.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _ensure_python_paths(btb3d_repo: Path, ctclip_repo: Path) -> None:
    for path in (CORE_CODE_ROOT, btb3d_repo, ctclip_repo):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _distributed_runtime(args: argparse.Namespace) -> tuple[torch.device, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank if args.local_rank >= 0 else 0)))
    use_deepspeed = bool(args.deepspeed_config)
    if use_deepspeed:
        if not torch.cuda.is_available():
            raise RuntimeError("--deepspeed-config requires CUDA GPUs")
        import deepspeed

        deepspeed.init_distributed()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", str(local_rank)))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        args.device = f"cuda:{local_rank}"
        return device, rank, local_rank, world_size, True

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    return device, rank, local_rank, world_size, False


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def _run_forward_diagnostic(model, train_loader, device, args) -> None:
    """Live forward diagnostic (env DIAG_FORWARD=1). Runs ONE real forward through the
    fully-built model + real dataloader, with hooks on the mm_projector and a wrap of
    prepare_inputs_labels_for_multimodal, to verify: (1) grad flow (which params train),
    (2) projector in/out shapes+stats, (3) #visual tokens spliced into the LLM sequence,
    (4) visual vs text embed norm distributions. Prints, then the caller returns."""
    from llava.constants import IMAGE_TOKEN_INDEX
    raw = _unwrap_model(model)
    bar = "=" * 78
    print(f"\n{bar}\n[DIAG] live forward diagnostic (one real batch)\n{bar}")

    # (1) gradient flow — which parameters are trainable
    tot = tr = 0
    groups: dict[str, int] = {}
    tnames: list[str] = []
    for n, p in raw.named_parameters():
        tot += p.numel()
        if p.requires_grad:
            tr += p.numel()
            tnames.append(n)
            key = "mm_projector" if "mm_projector" in n else ("lora" if "lora_" in n.lower() else "other")
            groups[key] = groups.get(key, 0) + p.numel()
    print(f"[DIAG] projector_only={args.projector_only}  trainable={tr:,}/{tot:,}")
    print(f"[DIAG] trainable by group: { {k: f'{v:,}' for k,v in groups.items()} }")
    print(f"[DIAG] #trainable tensors={len(tnames)}; first/last names:")
    for nm in tnames[:4] + (["..."] if len(tnames) > 8 else []) + tnames[-4:]:
        print(f"         {nm}")

    # (2) hook the mm_projector to capture in/out
    proj = raw.get_model().mm_projector
    print(f"[DIAG] mm_projector = {proj.__class__.__name__}")
    cap: dict[str, object] = {}
    def _hook(_m, inp, out):
        x = inp[0]
        cap["pin"] = tuple(x.shape); cap["pout"] = tuple(out.shape)
        cap["pin_stat"] = (float(x.float().mean()), float(x.float().std()))
        cap["pout_stat"] = (float(out.float().mean()), float(out.float().std()))
        cap["pout_tok_norm"] = float(out.float().reshape(-1, out.shape[-1]).norm(dim=-1).mean())
    h = proj.register_forward_hook(_hook)

    # (3) wrap prepare_inputs_labels_for_multimodal to capture splice geometry.
    # The method lives on LlavaLlamaForCausalLM, nested inside PeftModel.base_model.model;
    # find the module whose class actually defines it.
    host = next((m for m in raw.modules() if hasattr(type(m), "prepare_inputs_labels_for_multimodal")), None)
    if host is None:
        raise RuntimeError("could not locate prepare_inputs_labels_for_multimodal in model tree")
    print(f"[DIAG] prepare_inputs host = {host.__class__.__name__}")
    Cls = type(host)
    orig_prep = Cls.prepare_inputs_labels_for_multimodal
    def _patched(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None):
        cap["n_image_placeholders"] = int((input_ids == IMAGE_TOKEN_INDEX).sum())
        cap["in_seq"] = tuple(input_ids.shape)
        res = orig_prep(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes)
        emb = res[4]
        if emb is not None:
            cap["out_embeds_seq"] = tuple(emb.shape)
            cap["text_like_norm"] = None
        return res
    Cls.prepare_inputs_labels_for_multimodal = _patched

    try:
        batch = _move_batch(next(iter(train_loader)), device)
        print(f"[DIAG] batch images {tuple(batch['images'].shape)} dtype={batch['images'].dtype}  "
              f"input_ids {tuple(batch['input_ids'].shape)}")
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["labels"], images=batch["images"])
    finally:
        h.remove(); Cls.prepare_inputs_labels_for_multimodal = orig_prep

    print(f"[DIAG] loss = {float(out.loss):.4f}")
    print(f"[DIAG] projector IN  {cap['pin']}  mean/std={cap['pin_stat'][0]:+.4f}/{cap['pin_stat'][1]:.4f}")
    print(f"[DIAG] projector OUT {cap['pout']}  mean/std={cap['pout_stat'][0]:+.4f}/{cap['pout_stat'][1]:.4f}  "
          f"mean token L2={cap['pout_tok_norm']:.3f}")
    n_img = cap.get("n_image_placeholders")
    in_seq = cap.get("in_seq"); out_seq = cap.get("out_embeds_seq")
    vis_tokens = cap["pout"][1] if len(cap["pout"]) == 3 else None
    print(f"[DIAG] image placeholders in input_ids = {n_img} (one <image> token per sample)")
    print(f"[DIAG] visual tokens produced per image = {vis_tokens}  (expect pack4 grid 6*6*6=216)")
    if in_seq and out_seq:
        delta = out_seq[1] - in_seq[1]
        print(f"[DIAG] LLM sequence: text input_ids seq={in_seq[1]} -> spliced embeds seq={out_seq[1]} "
              f"(+{delta} = {vis_tokens} visual − {n_img} placeholder)")
    print(f"{bar}\n[DIAG] done.\n{bar}")


def _enable_gradient_checkpointing(model: torch.nn.Module) -> None:
    target = _unwrap_model(model)
    if hasattr(target, "enable_input_require_grads"):
        target.enable_input_require_grads()
    if hasattr(target, "gradient_checkpointing_enable"):
        target.gradient_checkpointing_enable()
    elif hasattr(target, "base_model") and hasattr(target.base_model, "gradient_checkpointing_enable"):
        target.base_model.gradient_checkpointing_enable()
    else:
        raise RuntimeError("model does not expose gradient_checkpointing_enable")
    _model_config(target).use_cache = False


def _load_deepspeed_config(path: Path, args: argparse.Namespace, world_size: int) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text())
    config["train_micro_batch_size_per_gpu"] = int(args.batch_size)
    config["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    config["train_batch_size"] = int(args.batch_size) * int(args.gradient_accumulation_steps) * int(world_size)
    config["gradient_clipping"] = float(args.max_grad_norm)
    config.setdefault("bf16", {})["enabled"] = True
    config.setdefault("fp16", {})["enabled"] = False
    config.pop("num_processes", None)
    return config


def _normalise_non_lora_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalised = {(key[11:] if key.startswith("base_model.") else key): value for key, value in state.items()}
    if any(key.startswith("model.model.") for key in normalised):
        normalised = {(key[6:] if key.startswith("model.") else key): value for key, value in normalised.items()}
    return normalised


def _load_author_configured_base(
    config_path: Path,
    model_base: Path,
    *,
    mm_hidden_size_override: int | None = None,
) -> tuple[Any, torch.nn.Module]:
    from llava.constants import (
        TOKEN_FOR_LONG_ANSWER,
        TOKEN_FOR_MULTIPLE_CHOICE,
        TOKEN_FOR_REPORT_GENERATION,
        TOKEN_FOR_SHORT_ANSWER,
    )
    from llava.model.language_model.llava_llama import LlavaConfig, LlavaLlamaForCausalLM
    from llava.train.train import smart_tokenizer_and_embedding_resize

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False, trust_remote_code=True)
    cfg = LlavaConfig.from_pretrained(config_path)
    if mm_hidden_size_override is not None:
        cfg.mm_hidden_size = int(mm_hidden_size_override)
    cfg.pad_token_id = None
    cfg.vocab_size = cfg.vocab_size - 1 - 4
    base_model = LlavaLlamaForCausalLM.from_pretrained(
        model_base,
        low_cpu_mem_usage=True,
        config=cfg,
        torch_dtype=torch.bfloat16,
    )
    smart_tokenizer_and_embedding_resize({"pad_token": "<pad>"}, tokenizer=tokenizer, model=base_model)
    tokenizer.add_tokens(TOKEN_FOR_MULTIPLE_CHOICE, special_tokens=True)
    tokenizer.add_tokens(TOKEN_FOR_LONG_ANSWER, special_tokens=True)
    tokenizer.add_tokens(TOKEN_FOR_SHORT_ANSWER, special_tokens=True)
    tokenizer.add_tokens(TOKEN_FOR_REPORT_GENERATION, special_tokens=True)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False
    return tokenizer, base_model


def _load_author_adapter_lora_config(model_path: Path) -> LoraConfig:
    adapter_config_path = model_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(adapter_config_path)
    raw = json.loads(adapter_config_path.read_text())
    return LoraConfig(
        r=int(raw["r"]),
        lora_alpha=int(raw["lora_alpha"]),
        target_modules=list(raw["target_modules"]),
        lora_dropout=float(raw.get("lora_dropout", 0.0)),
        bias=str(raw.get("bias", "none")),
        task_type=str(raw.get("task_type", "CAUSAL_LM")),
    )


def load_author_model_trainable(
    model_path: Path,
    model_base: Path,
    device: torch.device,
    *,
    base_author_model_path: Path | None = None,
    projector_input_dim: int | None = None,
) -> tuple[Any, torch.nn.Module]:
    """Load an author-compatible LoRA checkpoint without merging it."""

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not (model_path / "adapter_model.safetensors").exists():
        raise FileNotFoundError(model_path / "adapter_model.safetensors")
    if not (model_path / "non_lora_trainables.bin").exists():
        raise FileNotFoundError(model_path / "non_lora_trainables.bin")
    if base_author_model_path is None:
        base_author_model_path = model_path
    if not base_author_model_path.exists():
        raise FileNotFoundError(base_author_model_path)
    if not (base_author_model_path / "non_lora_trainables.bin").exists():
        raise FileNotFoundError(base_author_model_path / "non_lora_trainables.bin")

    tokenizer, base_model = _load_author_configured_base(
        base_author_model_path,
        model_base,
        mm_hidden_size_override=projector_input_dim,
    )

    base_non_lora = torch.load(
        base_author_model_path / "non_lora_trainables.bin",
        map_location="cpu",
        weights_only=False,
    )
    non_lora = dict(base_non_lora)
    if base_author_model_path != model_path:
        checkpoint_non_lora = torch.load(
            model_path / "non_lora_trainables.bin",
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_has_embeddings = any("embed_tokens" in key for key in checkpoint_non_lora)
        if not checkpoint_has_embeddings:
            print(
                "[author-load] resume checkpoint non-LoRA has no special-token embeddings; "
                "using frozen embeddings from base author checkpoint"
            )
        non_lora.update(checkpoint_non_lora)
    missing, unexpected = base_model.load_state_dict(_normalise_non_lora_state(non_lora), strict=False)
    unexpected_real = [key for key in unexpected if "rotary_emb.inv_freq" not in key]
    if unexpected_real:
        raise RuntimeError(f"unexpected non-LoRA keys while loading author checkpoint: {unexpected_real[:20]}")
    if not any("mm_projector" in key for key in non_lora):
        raise RuntimeError("author checkpoint non_lora_trainables.bin does not contain mm_projector weights")
    if not any("embed_tokens" in key for key in non_lora):
        raise RuntimeError("author checkpoint non_lora_trainables.bin does not contain special-token embeddings")
    relevant_missing = [key for key in missing if "mm_projector" in key or "embed_tokens" in key]
    if relevant_missing:
        raise RuntimeError(
            "author checkpoint did not populate required non-LoRA tensors; "
            f"missing={relevant_missing[:20]}"
        )
    if missing:
        print(
            "[author-load] non-LoRA checkpoint contains only author trainables; "
            f"ignored_expected_base_missing={len(missing)}"
        )

    model = PeftModel.from_pretrained(base_model, model_path, is_trainable=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "lora_" in name or "mm_projector" in name:
            parameter.requires_grad_(True)
    trainable_embeddings = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "embed_tokens" in name]
    if trainable_embeddings:
        raise RuntimeError(f"LLaMA text embeddings must stay frozen; trainable={trainable_embeddings[:5]}")
    model.to(device)
    return tokenizer, model


def load_author_architecture_from_scratch(
    *,
    config_model_path: Path,
    checkpoint_path: Path | None,
    model_base: Path,
    device: torch.device,
    projector_input_dim: int | None = None,
    reinit_lora: tuple[int, int] | None = None,
) -> tuple[Any, torch.nn.Module]:
    """Load the author full-token architecture while random-initializing ReportGen trainables.

    When ``reinit_lora=(r, alpha)`` is set together with ``checkpoint_path`` (an
    init-weights/stage-1 checkpoint), the warm projector (non-LoRA trainables) is
    loaded from the checkpoint but a FRESH LoRA adapter is built at the requested
    rank instead of loading the checkpoint's adapter. This enables changing the
    LoRA rank between stages while keeping the stage-1 projector warmup.
    """

    if not config_model_path.exists():
        raise FileNotFoundError(config_model_path)
    tokenizer, base_model = _load_author_configured_base(
        config_model_path,
        model_base,
        mm_hidden_size_override=projector_input_dim,
    )

    if checkpoint_path is not None:
        non_lora_path = checkpoint_path / "non_lora_trainables.bin"
        if not non_lora_path.exists():
            raise FileNotFoundError(non_lora_path)
        non_lora = torch.load(non_lora_path, map_location="cpu", weights_only=False)
        missing, unexpected = base_model.load_state_dict(_normalise_non_lora_state(non_lora), strict=False)
        unexpected_real = [key for key in unexpected if "rotary_emb.inv_freq" not in key]
        if unexpected_real:
            raise RuntimeError(f"unexpected scratch non-LoRA keys while resuming: {unexpected_real[:20]}")
        if not any("mm_projector" in key for key in non_lora):
            raise RuntimeError("scratch resume checkpoint non_lora_trainables.bin has no mm_projector weights")
        if missing:
            print(
                "[scratch-load] non-LoRA checkpoint contains only scratch trainables; "
                f"ignored_expected_base_missing={len(missing)}"
            )
        if reinit_lora is not None:
            override_r, override_alpha = reinit_lora
            template = _load_author_adapter_lora_config(config_model_path)
            fresh_lora_config = LoraConfig(
                r=int(override_r),
                lora_alpha=int(override_alpha),
                target_modules=list(template.target_modules),
                lora_dropout=template.lora_dropout,
                bias=template.bias,
                task_type=template.task_type,
            )
            print(
                f"[reinit-lora] warm projector loaded from {checkpoint_path.name}; "
                f"building FRESH LoRA r={override_r} alpha={override_alpha}"
            )
            model = get_peft_model(base_model, fresh_lora_config)
        else:
            model = PeftModel.from_pretrained(base_model, checkpoint_path, is_trainable=True)
    else:
        lora_config = _load_author_adapter_lora_config(config_model_path)
        model = get_peft_model(base_model, lora_config)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if "lora_" in name or "mm_projector" in name:
            parameter.requires_grad_(True)
    trainable_embeddings = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "embed_tokens" in name]
    if trainable_embeddings:
        raise RuntimeError(f"LLaMA text embeddings must stay frozen; trainable={trainable_embeddings[:5]}")
    model.to(device)
    return tokenizer, model


def _reportgen_base_dim(args: Any) -> int:
    """Per-token visual dim. Encoder-agnostic: npy_grid encoders read it from the manifest
    (512 CT-CLIP / 768 CoLiPri); BTB3D uses the LFQ codebook dim."""
    mani = json.loads(Path(args.reportgen_artifact_manifest).read_text())
    if mani.get("artifact_type") == "npy_grid":
        return int(mani["visual_dim"])
    return CODEBOOK_DIM


def _effective_projector_input_dim(args: Any) -> int:
    """mm_projector input dim = per-token visual dim (uniform_pool preserves it).
    Single source of truth so the model build and the saved checkpoint metadata agree."""
    return _reportgen_base_dim(args)


def _factor3(n: int) -> tuple[int, int, int]:
    """Most-cubic factorization t*h*w == n (avoids degenerate [N,1,1] grids)."""
    best = None
    for t in range(1, n + 1):
        if n % t:
            continue
        rem = n // t
        for h in range(1, rem + 1):
            if rem % h:
                continue
            w = rem // h
            triple = tuple(sorted((t, h, w), reverse=True))
            score = triple[0] - triple[2]
            if best is None or score < best[0]:
                best = (score, triple)
    return best[1]


def _select_visual_tokens(features: np.ndarray, mode: str, budget: int) -> np.ndarray:
    """Reduce a `(1,T,H,W,C)` grid to ~`budget` tokens, per-token channel dim C preserved.

    mode: none -> keep the full grid; uniform_pool -> avg-pool the grid to ~budget tokens
    via a near-cube target shape (the projector input dim is unchanged).
    """
    if mode == "none" or budget <= 0:
        return features
    if features.ndim != 5 or features.shape[0] != 1:
        raise ValueError(f"_select_visual_tokens expects (1,T,H,W,C), got {features.shape}")
    _, t, h, w, c = features.shape
    if budget >= t * h * w:
        return features
    # MERGE (not select): avg-pool the (1,T,H,W,C) grid to ~budget tokens; per-token dim C preserved.
    tn, hn, wn = _factor3(budget)
    x = torch.as_tensor(features[0], dtype=torch.float32).permute(3, 0, 1, 2)[None]  # (1,C,T,H,W)
    pooled = F.adaptive_avg_pool3d(x, (tn, hn, wn))[0].permute(1, 2, 3, 0)  # (Tn,Hn,Wn,C)
    return pooled.numpy()[None].astype(np.float32, copy=False)


@dataclass(frozen=True)
class ReportgenRecord:
    sample_id: str
    volume_id: str
    image: str
    conversations: list[dict[str, Any]]


def _load_records(vqa_json: Path, ids: list[str], limit: int, type_filter: str = "report_generation",
                  multi_qa: bool = False) -> list[ReportgenRecord]:
    wanted_order = ids[:limit] if limit > 0 else ids
    wanted = set(wanted_order)
    with vqa_json.open() as f:
        raw_records = json.load(f)
    if multi_qa:
        # MULTI-TASK VQA (Paper-2 image-grounded): keep EVERY matching QA (multiple per volume), NOT
        # deduped by volume_id; each record loads its own conversation + that volume's tokens via __getitem__.
        recs: list[ReportgenRecord] = []
        for raw in raw_records:
            if not str(raw.get("id", "")).startswith(type_filter):
                continue
            volume_id = volume_id_from_image(raw["image"])
            if volume_id not in wanted:
                continue
            conversations = raw["conversations"]
            if len(conversations) != 2:
                raise ValueError(f"{raw['id']} must contain exactly one human/gpt turn pair")
            recs.append(ReportgenRecord(sample_id=raw["id"], volume_id=volume_id,
                                        image=raw["image"], conversations=conversations))
        if not recs:
            raise RuntimeError(f"no {type_filter!r} records from {vqa_json} matched artifact ids")
        return recs
    records_by_volume_id: dict[str, ReportgenRecord] = {}
    for raw in raw_records:
        if not str(raw.get("id", "")).startswith(type_filter):
            continue
        volume_id = volume_id_from_image(raw["image"])
        if volume_id not in wanted:
            continue
        conversations = raw["conversations"]
        if len(conversations) != 2:
            raise ValueError(f"{raw['id']} must contain exactly one human/gpt turn pair")
        records_by_volume_id[volume_id] = ReportgenRecord(
            sample_id=raw["id"],
            volume_id=volume_id,
            image=raw["image"],
            conversations=conversations,
        )
        if len(records_by_volume_id) >= len(wanted):
            break
    records = [records_by_volume_id[volume_id] for volume_id in wanted_order if volume_id in records_by_volume_id]
    if not records:
        raise RuntimeError(f"no report_generation records from {vqa_json} matched artifact ids")
    if len(records) != len(wanted_order):
        found = {record.volume_id for record in records}
        missing = [volume_id for volume_id in wanted_order if volume_id not in found]
        raise RuntimeError(
            f"requested {len(wanted_order)} {type_filter!r} record(s) from {vqa_json}, "
            f"but only found {len(records)}; first missing={missing[:10]}"
        )
    return records


class StepIndexedBatchSampler(Sampler[list[int]]):
    """Yield deterministic per-rank batches so resume does not repeat data."""

    def __init__(
        self,
        *,
        dataset_size: int,
        local_batch_size: int,
        world_size: int,
        rank: int,
        start_step: int,
        end_step: int,
        seed: int,
        shuffle: bool,
        drop_last: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be positive, got {dataset_size}")
        if local_batch_size <= 0:
            raise ValueError(f"local_batch_size must be positive, got {local_batch_size}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank={rank} is outside world_size={world_size}")
        if end_step < start_step:
            raise ValueError(f"end_step={end_step} must be >= start_step={start_step}")
        self.dataset_size = int(dataset_size)
        self.local_batch_size = int(local_batch_size)
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.global_batch_size = self.local_batch_size * self.world_size
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.steps_per_epoch = (
            self.dataset_size // self.global_batch_size
            if self.drop_last
            else math.ceil(self.dataset_size / self.global_batch_size)
        )
        if self.steps_per_epoch <= 0:
            raise ValueError(
                f"global_batch_size={self.global_batch_size} is larger than dataset_size={self.dataset_size} "
                "with drop_last=true"
            )
        self._perm_cache: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return self.end_step - self.start_step + 1

    def _epoch_indices(self, epoch: int) -> list[int]:
        if epoch in self._perm_cache:
            return self._perm_cache[epoch]
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(epoch))
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))
        self._perm_cache[epoch] = indices
        return indices

    def _batch_for_step(self, step: int) -> list[int]:
        zero_based_step = int(step) - 1
        epoch = zero_based_step // self.steps_per_epoch
        batch_offset = zero_based_step % self.steps_per_epoch
        global_start = batch_offset * self.global_batch_size
        local_start = global_start + self.rank * self.local_batch_size
        local_end = local_start + self.local_batch_size
        batch = self._epoch_indices(epoch)[local_start:local_end]
        if len(batch) != self.local_batch_size and self.drop_last:
            raise RuntimeError(f"step={step} rank={self.rank} produced incomplete batch: {batch}")
        return batch

    def __iter__(self) -> Iterator[list[int]]:
        for step in range(self.start_step, self.end_step + 1):
            yield self._batch_for_step(step)

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "step_indexed_batch_sampler",
            "dataset_size": self.dataset_size,
            "local_batch_size": self.local_batch_size,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "rank": self.rank,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "steps_per_epoch": self.steps_per_epoch,
        }


class ArtifactReportgenDataset(Dataset[dict[str, Any]]):
    """BTB3D ReportGen dataset that materializes image tensors from token artifacts."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        vqa_json: Path,
        tokenizer: Any,
        max_samples: int,
        token_selection: str = "none",
        token_budget: int = 0,
        record_type: str = "report_generation",
        multi_qa: bool = False,
    ) -> None:
        super().__init__()
        self.split = split
        self.record_type = record_type
        self.multi_qa = bool(multi_qa)
        self.token_selection = token_selection
        self.token_budget = int(token_budget)
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("artifact_type") == "npy_grid":
            self.backend = "npy_grid"
        elif "token_artifacts" in self.manifest:
            self.backend = "final_eval_token_artifact"
        else:
            self.backend = "visual_token_shards"
        self.tokens: np.ndarray | None = None
        self.codebook: np.ndarray | None = None
        self.visual_rows: dict[str, dict[str, Any]] = {}
        if self.backend == "final_eval_token_artifact":
            artifact = resolve_reportgen_artifact_manifest(self.manifest_path, split)
            self.tokens = np.load(artifact.tokens_int_path, mmap_mode="r")
            self.ids = [line.strip() for line in artifact.ids_path.read_text().splitlines() if line.strip()]
            if len(self.ids) != self.tokens.shape[0]:
                raise ValueError(f"{artifact.ids_path} has {len(self.ids)} ids but tokens has {self.tokens.shape[0]} rows")
            self.id_to_row = {volume_id: row for row, volume_id in enumerate(self.ids)}
            self.codebook, _ = load_reportgen_codebook(artifact.codebook_path, artifact.codebook_metadata_path)
        elif self.backend == "npy_grid":
            # Encoder-agnostic per-volume npy grids ([T,H,W,C], or [C,T,H,W] with axis_transpose).
            self.npy_transpose = self.manifest.get("axis_transpose")  # e.g. [1,2,3,0] for (C,T,H,W)->(T,H,W,C)
            split_entries = self.manifest.get("splits")
            if not isinstance(split_entries, dict) or split not in split_entries:
                available = sorted(split_entries) if isinstance(split_entries, dict) else []
                raise KeyError(f"split {split!r} not found in {self.manifest_path}; available={available}")
            se = split_entries[split]
            self.npy_dir = Path(se["dir"])
            ids_path = Path(se["ids"])
            if not ids_path.exists():
                raise FileNotFoundError(ids_path)
            self.ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
            self.id_to_row = {volume_id: row for row, volume_id in enumerate(self.ids)}
        else:
            split_entries = self.manifest.get("splits")
            if not isinstance(split_entries, dict) or split not in split_entries:
                available = sorted(split_entries) if isinstance(split_entries, dict) else []
                raise KeyError(f"split {split!r} not found in {self.manifest_path}; available={available}")
            split_entry = split_entries[split]
            index_path = self.manifest_path.parent / split_entry["index"]
            if not index_path.exists():
                raise FileNotFoundError(index_path)
            self.ids = []
            with index_path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    volume_id = str(row["volume_id"])
                    self.ids.append(volume_id)
                    self.visual_rows[volume_id] = row
            self.id_to_row = {volume_id: row for row, volume_id in enumerate(self.ids)}
        self.records = _load_records(vqa_json, self.ids, max_samples,
                                     type_filter=self.record_type, multi_qa=self.multi_qa)
        self.tokenizer = tokenizer

        from llava import conversation as conversation_lib
        from llava.train.train import DataArguments

        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
        self.data_args = DataArguments()
        self.data_args.is_multimodal = True
        self.data_args.mm_use_im_start_end = False

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from llava import conversation as conversation_lib
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token
        from llava.train.train import preprocess, preprocess_multimodal

        record = self.records[index]
        if self.backend == "final_eval_token_artifact":
            if self.tokens is None or self.codebook is None:
                raise RuntimeError("final-eval token backend was not initialized")
            row = self.id_to_row[record.volume_id]
            codes = np.asarray(self.tokens[row], dtype=np.uint32)
            features = materialize_reportgen_features_from_codes(
                codes,
                "16x16x8",
                REPORTGEN_CHANNEL_ORDER,
                self.codebook,
            )
        elif self.backend == "npy_grid":
            arr = np.load(self.npy_dir / f"{record.volume_id}.npy")
            if self.npy_transpose is not None:
                arr = np.transpose(arr, self.npy_transpose)
            if arr.ndim != 4:
                raise ValueError(f"{record.volume_id} npy grid must be [T,H,W,C] (after transpose), got {arr.shape}")
            features = np.ascontiguousarray(arr, dtype=np.float32)[None]  # [1,T,H,W,C]
            if os.environ.get("VQA_SMOKE_DEBUG"):                          # smoke only: trace token flow
                print(f"[SMOKE npy_grid] {record.volume_id}: loaded {arr.shape} -> features {features.shape} "
                      f"(T*H*W={arr.shape[0]*arr.shape[1]*arr.shape[2]} tokens, dim={arr.shape[3]}), "
                      f"finite={np.isfinite(features).all()}", flush=True)
        else:
            row = self.visual_rows[record.volume_id]
            shard_path = self.manifest_path.parent / record_split_relative_path(row["split"], row["shard"])
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                z_quantized = handle.get_tensor(row["z_quantized_key"])
            if z_quantized.ndim != 4:
                raise ValueError(f"{record.volume_id} z_quantized must be [C,D,H,W], got {tuple(z_quantized.shape)}")
            features_cdhw = z_quantized.float().numpy()[None]
            features = features_cdhw.transpose(0, 2, 3, 4, 1)
            if features.shape[-1] != CODEBOOK_DIM:
                raise ValueError(
                    f"{record.volume_id} materialized channel dim is {features.shape[-1]}; "
                    f"the released 16x16x8 author checkpoint expects {CODEBOOK_DIM}-dim visual tokens"
                )
        if self.token_selection != "none" and self.token_budget > 0:
            features = _select_visual_tokens(features, self.token_selection, self.token_budget)
        if os.environ.get("VQA_NOISE_INPUT"):  # FLOOR CONTROL: replace visual tokens with fresh unit-Gaussian noise (same shape) -> VQA accuracy should collapse to the language prior. Mirrors report-gen --noise-input.
            if index == 0:
                print(f"[NOISE-FLOOR] VQA_NOISE_INPUT active: replacing visual features {features.shape} with fresh N(0,1) noise", flush=True)
            features = np.random.randn(*features.shape).astype(np.float32)
        image = torch.from_numpy(np.asarray(features[0], dtype=np.float32))

        sources = preprocess_multimodal(copy.deepcopy([record.conversations]), self.data_args)
        data = preprocess(sources, self.tokenizer, has_image=True)
        human_turn = record.conversations[0]
        conv = conversation_lib.conv_templates["llama3"].copy()
        conv.append_message(conv.roles[0], human_turn["value"])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt_input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        return {
            "input_ids": data["input_ids"][0],
            "labels": data["labels"][0],
            "prompt_input_ids": prompt_input_ids,
            "image": image,
            "metadata": {
                "sample_id": record.sample_id,
                "volume_id": record.volume_id,
                "image": record.image,
                "question": human_turn["value"],
            },
        }


@dataclass(frozen=True)
class ArtifactCollator:
    tokenizer: Any

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        from llava.constants import IGNORE_INDEX

        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"] for item in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [item["labels"] for item in instances],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["prompt_input_ids"] for item in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        prompt_input_ids = prompt_input_ids[:, : self.tokenizer.model_max_length]
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_input_ids.ne(self.tokenizer.pad_token_id),
            "images": torch.stack([item["image"] for item in instances]),
            "metadata": [item["metadata"] for item in instances],
        }


def record_split_relative_path(split: str, shard: str) -> Path:
    return Path(split) / shard


def image_output_name(image: str) -> str:
    return f"{volume_id_from_image(image)}.nii_embedded.npz"


def _write_subset_json(records: list[ReportgenRecord], out_path: Path) -> None:
    out = [
        {
            "id": record.sample_id,
            "image": record.image,
            "conversations": record.conversations,
        }
        for record in records
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")


def _apply_valid_shard(dataset: ArtifactReportgenDataset, num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("--valid-num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--valid-shard-index must satisfy 0 <= index < --valid-num-shards")
    if num_shards == 1:
        return
    dataset.records = dataset.records[shard_index::num_shards]
    if not dataset.records:
        raise RuntimeError(
            f"validation shard {shard_index}/{num_shards} is empty; "
            f"reduce --valid-num-shards or increase --valid-limit"
        )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            target_dtype = torch.bfloat16 if key == "images" else None
            moved[key] = value.to(device=device, dtype=target_dtype) if target_dtype is not None else value.to(device)
        else:
            moved[key] = value
    return moved


def _expected_eval_images(valid_loader: DataLoader[dict[str, Any]]) -> set[str]:
    dataset = getattr(valid_loader, "dataset", None)
    records = getattr(dataset, "records", None)
    if records is None:
        raise RuntimeError("validation loader dataset does not expose records; cannot validate eval resume state")
    # multi_qa (Paper-2 VQA): many QA share one image -> key by unique sample_id so EVERY QA is generated
    # (not deduped to one-per-volume). report-gen keys by image name (one report per volume) -> unchanged.
    if getattr(dataset, "multi_qa", False):
        return {record.sample_id for record in records}
    return {image_output_name(record.image) for record in records}


def _load_completed_eval_images(raw_jsonl: Path, expected_images: set[str], multi_qa: bool = False) -> set[str]:
    completed: set[str] = set()
    if not raw_jsonl.exists():
        return completed
    with raw_jsonl.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise RuntimeError(f"{raw_jsonl} contains invalid JSON at line {line_no}; cannot resume safely")
            if multi_qa:  # resume key = per-QA sample id (many QA per image)
                conv = record.get("conversations_out") or [{}]
                key = conv[0].get("id")
            else:
                key = record.get("image")
            if not isinstance(key, str) or not key:
                raise RuntimeError(f"{raw_jsonl} line {line_no} has no resume key; cannot resume safely")
            if key in completed:
                raise RuntimeError(f"{raw_jsonl} contains duplicate eval key {key!r}; cannot resume safely")
            completed.add(key)
    unexpected = sorted(completed - expected_images)
    if unexpected:
        examples = ", ".join(unexpected[:5])
        raise RuntimeError(
            f"{raw_jsonl} contains {len(unexpected)} image(s) outside this validation dataset "
            f"(examples: {examples}); use a fresh --out-dir or remove the stale evaluation directory"
        )
    return completed


def _select_batch_rows(batch: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    if len(indices) == len(batch["metadata"]):
        return batch
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    selected: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "metadata":
            selected[key] = [value[i] for i in indices]
        elif isinstance(value, torch.Tensor) and value.shape[:1] == (len(batch["metadata"]),):
            selected[key] = value.index_select(0, index_tensor)
        else:
            selected[key] = value
    return selected


def _trainable_summary(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    lora = sum(parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" in name)
    mm = sum(parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad and "mm_projector" in name)
    embed = sum(parameter.numel() for name, parameter in model.named_parameters() if parameter.requires_grad and "embed_tokens" in name)
    return {"total": total, "trainable": trainable, "lora": lora, "mm_projector": mm, "embed_tokens": embed}


def _build_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    projector_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "mm_projector" in name:
            projector_params.append(parameter)
        else:
            other_params.append(parameter)
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if projector_params:
        param_groups.append({"params": projector_params, "lr": args.mm_projector_lr, "weight_decay": args.weight_decay})
    if not param_groups:
        raise RuntimeError("optimizer has no trainable parameters")
    if args.projector_only and other_params:
        raise RuntimeError("projector-only mode unexpectedly has trainable non-projector parameters")
    if args.projector_only and not projector_params:
        raise RuntimeError("projector-only mode has no trainable mm_projector parameters")
    return torch.optim.AdamW(param_groups)


def _checkpoint_step(checkpoint_dir: Path) -> int:
    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        return int(payload["step"])
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        return int(json.loads(metadata_path.read_text())["step"])
    match = re.fullmatch(r"(?:checkpoint-|step_)(\d+)", checkpoint_dir.name)
    if not match:
        raise ValueError(
            f"resume checkpoint must contain training_state.pt/metadata.json or be named step_<step>, "
            f"got {checkpoint_dir}"
        )
    return int(match.group(1))


def _checkpoint_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        return dict(metadata) if isinstance(metadata, dict) else {}
    metadata_path = checkpoint_dir / "metadata.json"
    if metadata_path.exists():
        return dict(json.loads(metadata_path.read_text()))
    return {}


def _validate_checkpoint_model_compatibility(
    checkpoint_dir: Path,
    *,
    projector_input_dim: int,
) -> dict[str, Any]:
    metadata = _checkpoint_metadata(checkpoint_dir)
    saved_projector_dim = metadata.get("projector_input_dim")
    if saved_projector_dim is not None and int(saved_projector_dim) != int(projector_input_dim):
        raise ValueError(
            f"checkpoint projector_input_dim={saved_projector_dim} does not match current {projector_input_dim}"
        )
    return metadata


def _resolve_resume_checkpoint(path: Path) -> Path:
    if path.is_file():
        if path.name == "latest":
            checkpoint_name = path.read_text().strip()
            if not checkpoint_name:
                raise RuntimeError(f"empty latest checkpoint file: {path}")
            return _resolve_resume_checkpoint(path.parent / checkpoint_name)
        if path.name == "training_state.pt" or path.name.startswith("training_state_latest"):
            return path.resolve(strict=True).parent
        raise ValueError(f"unsupported resume checkpoint file: {path}")
    if path.is_dir():
        if (path / "training_state.pt").exists():
            return path
        if (path / "training_state_latest.pt").exists():
            return _resolve_resume_checkpoint(path / "training_state_latest.pt")
        if (path / "checkpoints" / "training_state_latest.pt").exists():
            return _resolve_resume_checkpoint(path / "checkpoints" / "training_state_latest.pt")
        if (path / "latest").exists():
            return _resolve_resume_checkpoint(path / "latest")
        raise ValueError(
            "resume checkpoint directory must be a step checkpoint, checkpoints dir, or run dir "
            f"with training_state_latest.pt; got {path}"
        )
    raise FileNotFoundError(path)


def _update_latest_checkpoint_link(target_path: Path, latest_path: Path) -> None:
    if not target_path.exists():
        raise FileNotFoundError(target_path)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = latest_path.with_name(f".{latest_path.name}.{os.getpid()}.tmp")
    if tmp_path.exists() or tmp_path.is_symlink():
        tmp_path.unlink()
    relative_target = os.path.relpath(target_path, start=latest_path.parent)
    tmp_path.symlink_to(relative_target)
    tmp_path.replace(latest_path)


def _validate_resume_sampler_metadata(checkpoint_dir: Path, expected: dict[str, Any]) -> None:
    metadata = _checkpoint_metadata(checkpoint_dir)
    actual = metadata.get("sampler")
    if not isinstance(actual, dict):
        raise ValueError(f"resume checkpoint is missing sampler metadata: {checkpoint_dir}")
    keys = (
        "kind",
        "dataset_size",
        "local_batch_size",
        "world_size",
        "global_batch_size",
        "seed",
        "shuffle",
        "drop_last",
        "steps_per_epoch",
    )
    mismatches = {
        key: {"checkpoint": actual.get(key), "current": expected.get(key)}
        for key in keys
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError("resume sampler metadata mismatch: " + json.dumps(mismatches, sort_keys=True))


def _non_lora_state_for_save(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model = _unwrap_model(model)
    return {
        key: value.detach().cpu().clone()
        for key, value in model.named_parameters()
        if "lora_" not in key and value.requires_grad
    }


def _checkpoint_training_state_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "training_state.pt"


def _checkpoint_deepspeed_dir(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "deepspeed"


def _prune_old_deepspeed_checkpoints(out_dir: Path, keep: int) -> list[Path]:
    if keep < 0:
        return []
    checkpoint_root = out_dir / "checkpoints"
    if not checkpoint_root.exists():
        return []
    checkpoint_dirs = sorted(
        (path for path in checkpoint_root.glob("step_*") if path.is_dir()),
        key=_checkpoint_step,
    )
    checkpoint_dirs = [path for path in checkpoint_dirs if _checkpoint_deepspeed_dir(path).is_dir()]
    if keep > 0:
        checkpoint_dirs = checkpoint_dirs[:-keep]
    removed: list[Path] = []
    for checkpoint_dir in checkpoint_dirs:
        deepspeed_dir = _checkpoint_deepspeed_dir(checkpoint_dir)
        if not deepspeed_dir.is_dir():
            continue
        shutil.rmtree(deepspeed_dir)
        removed.append(deepspeed_dir)
    return removed


def save_checkpoint(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    out_dir: Path,
    step: int,
    args: argparse.Namespace,
    sampler_metadata: dict[str, Any],
    rank: int,
    use_deepspeed: bool,
) -> Path:
    unwrapped = _unwrap_model(model)
    checkpoint_dir = out_dir / "checkpoints" / f"step_{step:06d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "checkpoint_kind": "author_reportgen_training_state",
        "step": int(step),
        "run_name": args.run_name,
        "reportgen_artifact_manifest": str(args.reportgen_artifact_manifest),
        "compression": "16x16x8",
        "projector_input_dim": _effective_projector_input_dim(args),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "effective_batch_size": int(args.batch_size) * int(args.gradient_accumulation_steps),
        "init_from_scratch": bool(args.init_from_scratch),
        "projector_only": bool(args.projector_only),
        "sampler": sampler_metadata,
        "trainable_parameters": _trainable_summary(unwrapped),
        "deepspeed": bool(use_deepspeed),
    }
    if use_deepspeed:
        model.save_checkpoint(
            str(_checkpoint_deepspeed_dir(checkpoint_dir)),
            tag="ds",
            client_state={"step": int(step), "metadata": metadata},
        )
    if rank == 0:
        _model_config(unwrapped).save_pretrained(checkpoint_dir)
        unwrapped.save_pretrained(checkpoint_dir, safe_serialization=True)
        torch.save(_non_lora_state_for_save(unwrapped), checkpoint_dir / "non_lora_trainables.bin")
        tokenizer.save_pretrained(checkpoint_dir)
        training_state = {
            "step": int(step),
            "metadata": metadata,
        }
        if not use_deepspeed:
            training_state["optimizer"] = optimizer_state_to_cpu(optimizer.state_dict())
            training_state["scheduler"] = optimizer_state_to_cpu(scheduler.state_dict())
        else:
            training_state["deepspeed_checkpoint"] = str(_checkpoint_deepspeed_dir(checkpoint_dir))
        torch.save(training_state, _checkpoint_training_state_path(checkpoint_dir))
        (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        _update_latest_checkpoint_link(
            _checkpoint_training_state_path(checkpoint_dir),
            out_dir / "checkpoints" / "training_state_latest.pt",
        )
        _update_latest_checkpoint_link(checkpoint_dir / "metadata.json", out_dir / "checkpoints" / "metadata_latest.json")
        if use_deepspeed:
            removed = _prune_old_deepspeed_checkpoints(out_dir, int(args.keep_deepspeed_checkpoints))
            if removed:
                print(
                    json.dumps(
                        {
                            "event": "pruned_old_deepspeed_checkpoints",
                            "keep_deepspeed_checkpoints": int(args.keep_deepspeed_checkpoints),
                            "removed": [str(path) for path in removed],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return checkpoint_dir


def load_training_state(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    checkpoint_dir: Path,
    use_deepspeed: bool,
) -> int:
    if use_deepspeed:
        ds_dir = _checkpoint_deepspeed_dir(checkpoint_dir)
        if not ds_dir.exists():
            raise FileNotFoundError(f"DeepSpeed resume checkpoint not found: {ds_dir}")
        load_path, client_state = model.load_checkpoint(str(ds_dir), tag="ds")
        if load_path is None:
            raise RuntimeError(f"DeepSpeed failed to load checkpoint from {ds_dir}")
        return int((client_state or {}).get("step", _checkpoint_step(checkpoint_dir)))
    state_path = _checkpoint_training_state_path(checkpoint_dir)
    if not state_path.exists():
        raise FileNotFoundError(f"training state not found for resume: {state_path}")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return int(payload["step"])


def _model_config(model: torch.nn.Module) -> Any:
    model = _unwrap_model(model)
    config_owner = getattr(model, "base_model", model)
    if hasattr(config_owner, "model") and hasattr(config_owner.model, "config"):
        return config_owner.model.config
    if hasattr(model, "config"):
        return model.config
    raise AttributeError("model does not expose a Hugging Face config")


def run_generation_eval(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    valid_loader: DataLoader[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    step: int,
) -> None:
    eval_dir = run_dir / "evaluations" / f"step_{step:06d}"
    raw_jsonl = eval_dir / "raw_btb3d_output.jsonl"
    pred_jsonl = eval_dir / "predictions.jsonl"
    metrics_json = eval_dir / "metrics_fast.json"
    eval_dir.mkdir(parents=True, exist_ok=True)
    valid_loss_metrics = None if args.skip_metrics else run_validation_loss_eval(
        model=model,
        valid_loader=valid_loader,
        args=args,
        run_dir=run_dir,
        step=step,
    )
    was_training = model.training
    model.eval()
    model_config = _model_config(model)
    old_tokenizer_padding_side = getattr(tokenizer, "padding_side", "right")
    old_model_padding_side = getattr(model_config, "tokenizer_padding_side", "right")
    tokenizer.padding_side = "left"
    model_config.tokenizer_padding_side = "left"
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer must define pad_token_id or eos_token_id for generation")
    device = torch.device(args.device)
    eval_multi_qa = getattr(getattr(valid_loader, "dataset", None), "multi_qa", False)

    def _eval_key(metadata: dict[str, Any]) -> str:
        return metadata["sample_id"] if eval_multi_qa else image_output_name(metadata["image"])

    expected_images = _expected_eval_images(valid_loader)
    completed_images = _load_completed_eval_images(raw_jsonl, expected_images, eval_multi_qa)
    if completed_images:
        print(f"[eval-resume] found {len(completed_images)} completed report(s) in {raw_jsonl}; skipping them")
    generated_count = len(completed_images)
    write_mode = "a" if completed_images else "w"
    try:
        with raw_jsonl.open(write_mode) as f:
            for batch in tqdm(valid_loader, desc=f"eval-generation step {step}", unit="batch"):
                keep_indices = [
                    i
                    for i, metadata in enumerate(batch["metadata"])
                    if _eval_key(metadata) not in completed_images
                ]
                if not keep_indices:
                    continue
                batch = _select_batch_rows(batch, keep_indices)
                batch = _move_batch(batch, device)
                with torch.no_grad():
                    output_ids = model.generate(
                        batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        images=batch["images"],
                        image_sizes=[1 for _ in batch["metadata"]],
                        do_sample=False,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        min_new_tokens=args.min_new_tokens,
                        max_new_tokens=args.max_new_tokens,
                        repetition_penalty=args.repetition_penalty,
                        pad_token_id=pad_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )
                decoded = []
                for output in output_ids:
                    text = tokenizer.decode(output, skip_special_tokens=True)
                    decoded.append(text.replace("<|eot_id|>", "").replace(tokenizer.pad_token or "", "").strip())
                for metadata, answer in zip(batch["metadata"], decoded):
                    f.write(
                        json.dumps(
                            {
                                "image": image_output_name(metadata["image"]),
                                "conversations_out": [
                                    {
                                        "id": metadata["sample_id"],
                                        "question": metadata["question"],
                                        "answer": answer,
                                    }
                                ],
                            }
                        )
                        + "\n"
                    )
                    completed_images.add(_eval_key(metadata))
                    generated_count += 1
                f.flush()
        if generated_count == 0:
            raise RuntimeError(f"step {step} generated zero validation reports")
    finally:
        tokenizer.padding_side = old_tokenizer_padding_side
        model_config.tokenizer_padding_side = old_model_padding_side
        if was_training:
            model.train()

    if args.skip_metrics:
        # VQA / non-report tasks: keep raw_btb3d_output.jsonl (scored by score_vqa.py) but skip the
        # report-specific metric conversion (which filters on "report" in the question -> 0 records ->
        # SystemExit). report-gen default leaves skip_metrics=False so convert+eval_fast run unchanged.
        print(f"skipped_metrics={metrics_json} (raw inference kept for external scorer)")
        return
    convert_btb3d_jsonl(
        btb3d_jsonl=str(raw_jsonl),
        reports_csv=str(_path(args.valid_reports_csv)),
        out_jsonl=str(pred_jsonl),
        reference_fields=["Findings_EN", "Impressions_EN"],
        report_task_only=True,
        keep_empty_answers=False,
    )
    env = os.environ.copy()
    py_paths = [str(CORE_CODE_ROOT), str(_path(args.btb3d_repo)), str(_path(args.ctclip_repo))]
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(py_paths)
    subprocess.run(
        [
            sys.executable,
            str(CORE_CODE_ROOT / "dtbd3d" / "eval" / "eval_fast.py"),
            "--pred",
            str(pred_jsonl),
            "--labels-csv",
            str(_path(args.valid_labels_csv)),
            "--radbert",
            str(_path(args.radbert_checkpoint)),
            "--device",
            args.device,
            "--out",
            str(metrics_json),
        ],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    _append_valid_loss_to_metrics(metrics_json, valid_loss_metrics)


def run_validation_loss_eval(
    *,
    model: torch.nn.Module,
    valid_loader: DataLoader[dict[str, Any]],
    args: argparse.Namespace,
    run_dir: Path,
    step: int,
) -> dict[str, Any]:
    from llava.constants import IGNORE_INDEX

    eval_dir = run_dir / "evaluations" / f"step_{step:06d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = eval_dir / "valid_loss.json"
    metrics_csv = eval_dir / "valid_loss_batches.csv"
    was_training = model.training
    model.eval()
    device = torch.device(args.device)

    loss_sum = 0.0
    target_tokens = 0
    samples = 0
    batch_rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(
                tqdm(valid_loader, desc=f"valid-loss step {step}", unit="batch"),
                start=1,
            ):
                batch = _move_batch(batch, device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    images=batch["images"],
                )
                loss = outputs.loss.detach()
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite validation loss at step {step}, batch {batch_index}: {loss}")
                batch_target_tokens = int(batch["labels"].ne(IGNORE_INDEX).sum().item())
                batch_samples = len(batch["metadata"])
                loss_float = float(loss.cpu())
                loss_sum += loss_float * batch_target_tokens
                target_tokens += batch_target_tokens
                samples += batch_samples
                batch_rows.append(
                    {
                        "batch": batch_index,
                        "samples": batch_samples,
                        "target_tokens": batch_target_tokens,
                        "loss": loss_float,
                        "volume_ids": " ".join(item["volume_id"] for item in batch["metadata"]),
                    }
                )
    finally:
        if was_training:
            model.train()

    if samples == 0 or target_tokens == 0:
        raise RuntimeError(f"validation loss at step {step} saw samples={samples}, target_tokens={target_tokens}")
    mean_loss = loss_sum / target_tokens
    result = {
        "step": int(step),
        "valid_samples": int(samples),
        "valid_target_tokens": int(target_tokens),
        "valid_loss": float(mean_loss),
        "valid_ppl": float(math.exp(mean_loss)) if mean_loss < 100 else float("inf"),
        "valid_batches": len(batch_rows),
        "valid_limit": int(args.valid_limit),
        "eval_batch_size": int(args.eval_batch_size),
        "reportgen_artifact_manifest": str(args.reportgen_artifact_manifest),
        "resume_checkpoint": str(args.resume_checkpoint or ""),
    }
    metrics_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with metrics_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["batch", "samples", "target_tokens", "loss", "volume_ids"])
        writer.writeheader()
        writer.writerows(batch_rows)
    print(
        "VALID_LOSS_SUMMARY\t"
        f"step={step}\t"
        f"samples={samples}\t"
        f"target_tokens={target_tokens}\t"
        f"valid_loss={mean_loss:.6f}\t"
        f"valid_ppl={result['valid_ppl']:.4f}\t"
        f"path={metrics_json}"
    )
    return result


def _append_valid_loss_to_metrics(metrics_json: Path, valid_loss_metrics: dict[str, Any] | None) -> None:
    if valid_loss_metrics is None:
        return
    metrics = json.loads(metrics_json.read_text())
    metrics.update(
        {
            "valid_loss": valid_loss_metrics["valid_loss"],
            "valid_ppl": valid_loss_metrics["valid_ppl"],
            "valid_target_tokens": valid_loss_metrics["valid_target_tokens"],
            "valid_loss_samples": valid_loss_metrics["valid_samples"],
            "valid_loss_batches": valid_loss_metrics["valid_batches"],
            "valid_loss_path": str(metrics_json.with_name("valid_loss.json")),
        }
    )
    metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if args.steps < 1 and not args.eval_only:
        raise ValueError("--steps must be >= 1")
    if args.eval_only and args.deepspeed_config:
        raise ValueError("--eval-only should run without --deepspeed-config")
    if args.batch_size < 1 or args.eval_batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("--batch-size, --eval-batch-size, and --gradient-accumulation-steps must be >= 1")
    if args.valid_num_shards < 1:
        raise ValueError("--valid-num-shards must be >= 1")
    if args.valid_shard_index < 0 or args.valid_shard_index >= args.valid_num_shards:
        raise ValueError("--valid-shard-index must satisfy 0 <= index < --valid-num-shards")
    # VQA is multiple-choice (answer = one letter): max-new-tokens 16 over the full valid set is correct.
    # (The report-gen guard "full validation requires >=512" does not apply to VQA and is removed.)
    projector_input_dim = _effective_projector_input_dim(args)

    btb3d_repo = _path(args.btb3d_repo)
    ctclip_repo = _path(args.ctclip_repo)
    _ensure_python_paths(btb3d_repo, ctclip_repo)

    author_model_path = _path(args.model_path)
    if args.resume_checkpoint and args.init_weights_from_checkpoint:
        raise ValueError("--resume-checkpoint and --init-weights-from-checkpoint are mutually exclusive")
    resume_checkpoint = _resolve_resume_checkpoint(_path(args.resume_checkpoint)) if args.resume_checkpoint else None
    init_weights_checkpoint = (
        _resolve_resume_checkpoint(_path(args.init_weights_from_checkpoint))
        if args.init_weights_from_checkpoint
        else None
    )
    start_step = 0
    model_path = author_model_path
    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(resume_checkpoint)
        start_step = _checkpoint_step(resume_checkpoint)
        if not args.eval_only and start_step >= args.steps:
            raise ValueError(f"resume checkpoint step {start_step} must be lower than --steps {args.steps}")
        resume_metadata = _validate_checkpoint_model_compatibility(
            resume_checkpoint,
            projector_input_dim=projector_input_dim,
        )
        saved_grad_accum = resume_metadata.get("gradient_accumulation_steps")
        if saved_grad_accum is not None and int(saved_grad_accum) != int(args.gradient_accumulation_steps):
            raise ValueError(
                f"resume checkpoint gradient_accumulation_steps={saved_grad_accum} does not match "
                f"current {args.gradient_accumulation_steps}"
            )
        model_path = resume_checkpoint
    if init_weights_checkpoint is not None:
        if not init_weights_checkpoint.exists():
            raise FileNotFoundError(init_weights_checkpoint)
        _validate_checkpoint_model_compatibility(
            init_weights_checkpoint,
            projector_input_dim=projector_input_dim,
        )
        model_path = init_weights_checkpoint

    torch.manual_seed(args.seed)
    device, rank, local_rank, world_size, distributed = _distributed_runtime(args)
    is_main = _is_main_process(rank)

    run_dir = _path(args.out_dir)
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        (run_dir / "evaluations").mkdir(exist_ok=True)
        (run_dir / "subsets").mkdir(exist_ok=True)
    _distributed_barrier()
    logger = setup_rank0_logger(
        "dtbd3d.training.scripts.train_author_reportgen_artifact",
        run_dir,
        rank,
        filename="run.log",
    )

    if args.init_from_scratch:
        reinit_lora = None
        if (
            init_weights_checkpoint is not None
            and resume_checkpoint is None
            and args.lora_r is not None
        ):
            override_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r
            reinit_lora = (args.lora_r, override_alpha)
        tokenizer, model = load_author_architecture_from_scratch(
            config_model_path=author_model_path,
            checkpoint_path=resume_checkpoint or init_weights_checkpoint,
            model_base=_path(args.model_base),
            device=device,
            projector_input_dim=projector_input_dim,
            reinit_lora=reinit_lora,
        )
    else:
        tokenizer, model = load_author_model_trainable(
            model_path,
            _path(args.model_base),
            device,
            base_author_model_path=author_model_path,
            projector_input_dim=projector_input_dim,
        )
    if args.projector_only:
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(False)
            elif "mm_projector" in name:
                parameter.requires_grad_(True)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    if tokenizer.pad_token_id is None:
        raise RuntimeError("tokenizer.pad_token_id must be set")
    tokenizer.model_max_length = int(getattr(_model_config(model), "tokenizer_model_max_length", 128000))

    valid_dataset = ArtifactReportgenDataset(
        _path(args.reportgen_artifact_manifest),
        "valid",
        _path(args.valid_vqa_json),
        tokenizer,
        args.valid_limit,
        args.token_selection,
        args.token_budget,
        record_type=args.record_type,
        multi_qa=args.multi_qa_per_volume,
    )
    _apply_valid_shard(valid_dataset, args.valid_num_shards, args.valid_shard_index)
    shard_suffix = (
        f"_shard{args.valid_shard_index:03d}of{args.valid_num_shards:03d}"
        if args.valid_num_shards > 1
        else ""
    )
    valid_subset_json = run_dir / "subsets" / f"valid_report_generation_{len(valid_dataset)}{shard_suffix}.json"
    if is_main:
        _write_subset_json(valid_dataset.records, valid_subset_json)
    _distributed_barrier()

    collator = ArtifactCollator(tokenizer)
    train_dataset = None
    train_sampler = None
    train_loader = None
    if not args.eval_only:
        train_dataset = ArtifactReportgenDataset(
            _path(args.reportgen_artifact_manifest),
            "train",
            _path(args.train_vqa_json),
            tokenizer,
            args.train_limit,
            args.token_selection,
            args.token_budget,
            record_type=args.record_type,
            multi_qa=args.multi_qa_per_volume,
        )
        train_sampler = StepIndexedBatchSampler(
            dataset_size=len(train_dataset),
            local_batch_size=args.batch_size,
            world_size=world_size,
            rank=rank,
            start_step=start_step * args.gradient_accumulation_steps + 1,
            end_step=args.steps * args.gradient_accumulation_steps,
            seed=args.seed,
            shuffle=True,
            drop_last=True,
        )
        if resume_checkpoint is not None:
            _validate_resume_sampler_metadata(resume_checkpoint, train_sampler.metadata())
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collator,
        )
        if len(train_loader) == 0:
            raise RuntimeError("train DataLoader is empty; increase --train-limit or lower --batch-size")
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        drop_last=False,
    )

    config_dump = vars(args).copy()
    config_dump.update(
        {
            "run_dir": str(run_dir),
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "distributed": distributed,
            "train_samples": len(train_dataset) if train_dataset is not None else None,
            "valid_samples": len(valid_dataset),
            "trainable_parameters": _trainable_summary(model),
            "projector_input_dim": projector_input_dim,
            "effective_batch_size": int(args.batch_size) * int(args.gradient_accumulation_steps) * int(world_size),
            "start_step": start_step,
            "resume_from_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else "",
            "init_weights_from_checkpoint_resolved": (
                str(init_weights_checkpoint) if init_weights_checkpoint is not None else ""
            ),
            "train_sampler": train_sampler.metadata() if train_sampler is not None else None,
        }
    )
    if is_main:
        (run_dir / "run_config.json").write_text(json.dumps(config_dump, indent=2) + "\n")
        _log_startup_summary(logger, config_dump)

    if args.eval_only:
        if is_main:
            run_generation_eval(
                model=model,
                tokenizer=tokenizer,
                valid_loader=valid_loader,
                args=args,
                run_dir=run_dir,
                step=start_step,
            )
            print(f"eval_only_run_dir={run_dir}")
        return 0

    optimizer = _build_optimizer(model, args)
    warmup_steps = max(1, int(args.steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.steps)
    if args.deepspeed_config:
        import deepspeed

        ds_config = _load_deepspeed_config(_path(args.deepspeed_config), args, world_size)
        if is_main:
            (run_dir / "deepspeed_config_resolved.json").write_text(json.dumps(ds_config, indent=2) + "\n")
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )
    if resume_checkpoint is not None:
        loaded_step = load_training_state(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=resume_checkpoint,
            use_deepspeed=bool(args.deepspeed_config),
        )
        if loaded_step != start_step:
            raise RuntimeError(f"resume step mismatch: checkpoint_dir={start_step} training_state={loaded_step}")
        if is_main:
            print(f"resumed_checkpoint={resume_checkpoint} start_step={start_step}")
    model.train()
    if os.environ.get("DIAG_FORWARD") == "1":
        if is_main:
            _run_forward_diagnostic(model, train_loader, device, args)
        return 0
    train_csv = run_dir / "train_metrics.csv"
    if is_main and (start_step == 0 or not train_csv.exists()):
        with train_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "lr", "time_sec", "peak_mem_gb", "batch_volume_ids"])

    train_micro_iter = iter(train_loader)
    progress = tqdm(
        range(start_step + 1, args.steps + 1),
        total=args.steps - start_step,
        desc="train",
        unit="step",
        disable=not is_main,
    )
    last_checkpoint: Path | None = resume_checkpoint
    for step in progress:
        start = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if not args.deepspeed_config:
            optimizer.zero_grad(set_to_none=True)
        loss_values: list[float] = []
        volume_ids: list[str] = []
        for micro_index in range(args.gradient_accumulation_steps):
            try:
                batch = next(train_micro_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"train loader exhausted at optimizer step={step}, micro_index={micro_index}; "
                    "this indicates a sampler/gradient-accumulation length mismatch"
                ) from exc
            batch = _move_batch(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                images=batch["images"],
            )
            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}, micro_index={micro_index}: {loss}")
            loss_values.append(float(loss.detach().cpu()))
            volume_ids.extend(item["volume_id"] for item in batch["metadata"])
            if args.deepspeed_config:
                model.backward(loss)
                model.step()
            else:
                (loss / args.gradient_accumulation_steps).backward()
        mean_loss = sum(loss_values) / len(loss_values)
        if not args.deepspeed_config:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        elapsed = time.perf_counter() - start
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        lr = float(scheduler.get_last_lr()[0]) if hasattr(scheduler, "get_last_lr") else float(optimizer.param_groups[0]["lr"])
        if is_main:
            progress.set_postfix(loss=f"{mean_loss:.4f}", lr=f"{lr:.2e}", mem=f"{peak_mem:.1f}G")
            with train_csv.open("a", newline="") as f:
                csv.writer(f).writerow([step, f"{mean_loss:.8f}", f"{lr:.10f}", f"{elapsed:.4f}", f"{peak_mem:.4f}", " ".join(volume_ids)])

        should_save = args.save_every > 0 and step % args.save_every == 0
        should_eval = args.eval_every > 0 and step % args.eval_every == 0
        if should_save or should_eval:
            _distributed_barrier()
            saved_checkpoint = save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                out_dir=run_dir,
                step=step,
                args=args,
                sampler_metadata=train_sampler.metadata(),
                rank=rank,
                use_deepspeed=bool(args.deepspeed_config),
            )
            if is_main:
                last_checkpoint = saved_checkpoint
            _distributed_barrier()
        if should_eval:
            if is_main:
                run_generation_eval(
                    model=_unwrap_model(model),
                    tokenizer=tokenizer,
                    valid_loader=valid_loader,
                    args=args,
                    run_dir=run_dir,
                    step=step,
                )
            _distributed_barrier()
            model.train()

    if not args.skip_final_save and (last_checkpoint is None or _checkpoint_step(last_checkpoint) != args.steps):
        _distributed_barrier()
        saved_checkpoint = save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            out_dir=run_dir,
            step=args.steps,
            args=args,
            sampler_metadata=train_sampler.metadata(),
            rank=rank,
            use_deepspeed=bool(args.deepspeed_config),
        )
        if is_main:
            last_checkpoint = saved_checkpoint
        _distributed_barrier()
    if not args.skip_final_eval and (args.eval_every <= 0 or args.steps % args.eval_every != 0):
        if is_main:
            run_generation_eval(
                model=_unwrap_model(model),
                tokenizer=tokenizer,
                valid_loader=valid_loader,
                args=args,
                run_dir=run_dir,
                step=args.steps,
            )
        _distributed_barrier()
    if is_main:
        print(f"run_dir={run_dir}")
        if last_checkpoint is not None:
            print(f"latest_checkpoint={last_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
