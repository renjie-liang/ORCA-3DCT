"""CLI entry point for Phase 1 BTB3D report-generation training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from dtbd3d.training.constants import CORE_CODE_ROOT, RUN_ROOT
from dtbd3d.core.visual_embedding import resolve_reportgen_artifact_manifest
from dtbd3d.training.data import BTB3DDataCollator, BTB3DReportGenDataset
from dtbd3d.training.models import BTB3DModelConfig, build_llava_btb3d_model_with_lora
from dtbd3d.training.train.eval_callback import PeriodicEvalRunner
from dtbd3d.training.train.trainer import BTB3DTrainer, TrainLoopConfig, configure_logging, create_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--max-samples", type=int, help="Override train max_samples for debugging")
    parser.add_argument("--disable-eval", action="store_true", help="Skip periodic eval generation/eval_fast")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        raise ValueError(f"unrecognized CLI arguments: {unknown_args}")
    return args


def _path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def build_model_config(raw: dict[str, Any]) -> BTB3DModelConfig:
    model = raw["model"]
    return BTB3DModelConfig(
        model_name_or_path=model["model_name_or_path"],
        mm_projector_type=model["mm_projector_type"],
        mm_hidden_size=int(model["mm_hidden_size"]),
        mm_context_size=int(model["mm_context_size"]),
        hidden_size=int(model["hidden_size"]),
        lora_r=int(model["lora_r"]),
        lora_alpha=int(model["lora_alpha"]),
        lora_dropout=float(model["lora_dropout"]),
        lora_bias=model["lora_bias"],
        image_context_max_length=int(model["image_context_max_length"]),
        max_visual_tokens=int(model["max_visual_tokens"]),
        freeze_base_model=bool(model["freeze_base_model"]),
    )


def build_train_config(raw: dict[str, Any], args: argparse.Namespace) -> TrainLoopConfig:
    train = raw["training"]
    phase = raw["phase"]
    hp_string = (
        f"lora_r{raw['model']['lora_r']}_a{raw['model']['lora_alpha']}_"
        f"dropout{str(raw['model']['lora_dropout']).replace('.', '')}_"
        f"lr{str(train['lr']).replace('-', 'm').replace('.', 'e')}_"
        f"bs{train['per_device_train_batch_size']}_gradacc{train['gradient_accumulation_steps']}_"
        f"seed{train['seed']}"
    )
    return TrainLoopConfig(
        phase=phase,
        run_root=RUN_ROOT,
        n_steps=int(train["n_steps"]),
        save_every=int(train["save_every"]),
        eval_every=int(train["eval_every"]),
        lr=float(train["lr"]),
        mm_projector_lr=float(train["mm_projector_lr"]),
        weight_decay=float(train["weight_decay"]),
        warmup_ratio=float(train["warmup_ratio"]),
        per_device_train_batch_size=int(train["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train["gradient_accumulation_steps"]),
        max_grad_norm=float(train["max_grad_norm"]),
        seed=int(train["seed"]),
        num_workers=int(train["num_workers"]),
        hp_string=hp_string,
    )


def build_dataset(raw: dict[str, Any], tokenizer: Any, split: str, max_samples_override: int | None) -> BTB3DReportGenDataset:
    data = raw["data"]
    split_cfg = data[split]
    artifact = resolve_reportgen_artifact_manifest(_path(data["reportgen_artifact_manifest"]), split)
    compression = data.get("compression", raw.get("compression"))
    if compression is None:
        raise KeyError("missing compression; set either top-level `compression` or `data.compression` in the config")
    max_samples = split_cfg["max_samples"]
    if split == "train" and max_samples_override is not None:
        max_samples = max_samples_override
    return BTB3DReportGenDataset(
        tokens_path=artifact.tokens_int_path,
        ids_path=artifact.ids_path,
        vqa_json_path=_path(split_cfg["vqa_json_path"]),
        tokenizer=tokenizer,
        token_shape=tuple(data["token_shape"]),
        token_dim=int(data["token_dim"]),
        max_text_length=int(data["max_text_length"]),
        max_samples=max_samples,
        compression=str(compression),
        codebook_path=artifact.codebook_path,
        codebook_metadata_path=artifact.codebook_metadata_path,
    )


def main() -> None:
    args = parse_args()
    config_path = _path(args.config)
    raw = yaml.safe_load(config_path.read_text())
    model_config = build_model_config(raw)
    train_config = build_train_config(raw, args)
    run_dir = create_run_dir(train_config, raw)
    logger = configure_logging(run_dir)
    logger.info("run_dir=%s", run_dir)
    logger.info("Phase 1 uses text max_length=%s; image prefix context=%s", raw["data"]["max_text_length"], model_config.image_context_max_length)

    tokenizer, model = build_llava_btb3d_model_with_lora(model_config)
    train_dataset = build_dataset(raw, tokenizer, "train", args.max_samples)
    collator = BTB3DDataCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.per_device_train_batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collator,
    )

    eval_runner = None
    if not args.disable_eval:
        valid_dataset = build_dataset(raw, tokenizer, "valid", None)
        eval_cfg = raw["evaluation"]
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=int(eval_cfg.get("batch_size", 1)),
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        eval_runner = PeriodicEvalRunner(
            eval_loader=valid_loader,
            eval_fast_path=CORE_CODE_ROOT / "dtbd3d" / "eval" / "eval_fast.py",
            converter_path=CORE_CODE_ROOT / "dtbd3d" / "eval" / "btb3d_to_metrics_jsonl.py",
            reports_csv=_path(raw["data"]["valid"]["reports_csv_path"]),
            labels_csv=_path(eval_cfg["labels_csv"]),
            radbert_checkpoint=_path(eval_cfg["radbert_checkpoint"]),
            device=eval_cfg["device"],
            max_new_tokens=int(eval_cfg["max_new_tokens"]),
            repetition_penalty=float(eval_cfg["repetition_penalty"]),
        )

    trainer = BTB3DTrainer(model, tokenizer, train_loader, train_config, run_dir, logger, eval_runner)
    trainer.train()


if __name__ == "__main__":
    main()
