"""Manual-file-logged training loop for BTB3D report generation."""

from __future__ import annotations

import csv
import itertools
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from dtbd3d.training.train.eval_callback import PeriodicEvalRunner


@dataclass(frozen=True)
class TrainLoopConfig:
    """Runtime settings for the Phase 1 training loop."""

    phase: str
    run_root: Path
    n_steps: int
    save_every: int
    eval_every: int
    lr: float
    mm_projector_lr: float
    weight_decay: float
    warmup_ratio: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    seed: int
    num_workers: int
    hp_string: str


def create_run_dir(config: TrainLoopConfig, frozen_config: dict[str, Any]) -> Path:
    """Create the mandatory manual run directory and freeze config.yaml."""

    now = datetime.now()
    phase_dir = config.run_root / f"{config.phase}_{now:%Y-%m-%d}"
    run_dir = phase_dir / f"{config.hp_string}_{config.seed}_{now:%m%d}_{now:%H%M}"
    for child in ["checkpoints", "predictions", "evaluations"]:
        (run_dir / child).mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(frozen_config, f, sort_keys=False)
    return run_dir


def configure_logging(run_dir: Path) -> logging.Logger:
    """Configure DEBUG logging to both stdout and train_log.txt."""

    logger = logging.getLogger("dtbd3d.training")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "train_log.txt")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_optimizer(model: torch.nn.Module, config: TrainLoopConfig) -> torch.optim.Optimizer:
    """Create AdamW with a dedicated mm_projector group."""

    projector_params = []
    other_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("mm_projector"):
            projector_params.append(parameter)
        else:
            other_params.append(parameter)
    if not projector_params:
        raise ValueError("mm_projector has no trainable parameters")
    if not other_params:
        raise ValueError("LoRA/new-token parameter group is empty")
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": config.lr, "weight_decay": config.weight_decay},
            {"params": projector_params, "lr": config.mm_projector_lr, "weight_decay": config.weight_decay},
        ]
    )


def trainable_parameter_summary(model: torch.nn.Module) -> dict[str, int]:
    """Return total/trainable parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": total, "trainable": trainable}


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


class BTB3DTrainer:
    """Small training harness with the exact run-directory contract."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        train_loader: DataLoader[dict[str, Any]],
        config: TrainLoopConfig,
        run_dir: Path,
        logger: logging.Logger,
        eval_runner: PeriodicEvalRunner | None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.config = config
        self.run_dir = run_dir
        self.logger = logger
        self.eval_runner = eval_runner

    def _write_headers(self) -> None:
        with (self.run_dir / "train_metrics.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "lr", "grad_norm", "step_time_sec", "peak_gpu_mem_gb"])
        with (self.run_dir / "eval_metrics.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "clinical_f1", "precision", "recall", "crg", "bleu_1", "bleu_4"])

    def _append_train_metrics(
        self,
        step: int,
        loss: float,
        lr: float,
        grad_norm: float,
        step_time_sec: float,
        peak_gpu_mem_gb: float,
    ) -> None:
        with (self.run_dir / "train_metrics.csv").open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    step,
                    f"{loss:.8f}",
                    f"{lr:.10f}",
                    f"{grad_norm:.8f}",
                    f"{step_time_sec:.4f}",
                    f"{peak_gpu_mem_gb:.4f}",
                ]
            )

    def _save_checkpoint(self, step: int) -> None:
        checkpoint_dir = self.run_dir / "checkpoints" / f"step_{step}"
        self.logger.info("saving checkpoint: %s", checkpoint_dir)
        self.model.save_pretrained(checkpoint_dir, self.tokenizer)

    def _run_eval(self, step: int) -> None:
        if self.eval_runner is None:
            self.logger.info("eval_runner not configured; skipping eval at step %d", step)
            return
        metrics = self.eval_runner.run(step, self.model, self.tokenizer, self.run_dir)
        clinical = metrics["clinical"]
        text = metrics["text"]
        with (self.run_dir / "eval_metrics.csv").open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    step,
                    clinical["f1"],
                    clinical["precision"],
                    clinical["recall"],
                    clinical["crg"],
                    text["bleu_1"],
                    text["bleu_4"],
                ]
            )

    def train(self) -> dict[str, Any]:
        """Run Phase 1 training and return final metrics metadata."""

        self._write_headers()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.config.seed)
        optimizer = build_optimizer(self.model, self.config)
        warmup_steps = max(1, math.ceil(self.config.n_steps * self.config.warmup_ratio))
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, self.config.n_steps)
        trainable = trainable_parameter_summary(self.model)
        self.logger.info("trainable parameters: %s", trainable)
        self.logger.info("effective batch size: %d", self.config.gradient_accumulation_steps * self.config.per_device_train_batch_size)

        self.model.to(device)

        data_iter = itertools.cycle(self.train_loader)
        optimizer.zero_grad(set_to_none=True)
        optimizer_step = 0
        micro_losses: list[float] = []
        final_loss = float("nan")
        step_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)

        while optimizer_step < self.config.n_steps:
            batch = _move_batch_to_device(next(data_iter), device)
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                image_features=batch["image_features"],
            )
            raw_loss = outputs.loss
            micro_losses.append(float(raw_loss.detach().cpu()))

            loss = raw_loss / self.config.gradient_accumulation_steps
            loss.backward()
            boundary = len(micro_losses) == self.config.gradient_accumulation_steps
            grad_norm = float("nan")
            if boundary:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                grad_norm = float(grad_norm_tensor.detach().cpu())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not boundary:
                continue

            optimizer_step += 1
            final_loss = sum(micro_losses) / len(micro_losses)
            micro_losses = []
            lr = scheduler.get_last_lr()[0]
            step_time_sec = time.perf_counter() - step_start
            peak_gpu_mem_gb = (
                torch.cuda.max_memory_allocated(device=device) / (1024**3) if torch.cuda.is_available() else 0.0
            )
            self._append_train_metrics(optimizer_step, final_loss, lr, grad_norm, step_time_sec, peak_gpu_mem_gb)
            step_start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device=device)
            if optimizer_step % 10 == 0 or optimizer_step == 1:
                self.logger.info(
                    "step=%d loss=%.6f lr=%.8g time=%.2fs peak_mem=%.2fGB",
                    optimizer_step,
                    final_loss,
                    lr,
                    step_time_sec,
                    peak_gpu_mem_gb,
                )

            if optimizer_step == 200 and final_loss > 5.0:
                raise RuntimeError(f"hard early stop: step 200 train_loss {final_loss:.4f} > 5")
            if optimizer_step == 500 and final_loss > 2.0:
                raise RuntimeError(f"hard early stop: step 500 train_loss {final_loss:.4f} > 2")
            if optimizer_step % self.config.save_every == 0:
                self._save_checkpoint(optimizer_step)
            if optimizer_step % self.config.eval_every == 0:
                self._run_eval(optimizer_step)

        self._save_checkpoint(self.config.n_steps)
        final_metrics = {"final_step": self.config.n_steps, "final_train_loss": final_loss, "trainable_parameters": trainable}
        with (self.run_dir / "final_metrics.json").open("w") as f:
            json.dump(final_metrics, f, indent=2)
        self._write_phase_summary(final_metrics)
        return final_metrics

    def _write_phase_summary(self, final_metrics: dict[str, Any]) -> None:
        passed_loss = final_metrics["final_train_loss"] < 1.0
        text = (
            "# Phase 1 Summary\n\n"
            f"- B-bar loss criterion passed: {'Yes' if passed_loss else 'No'}\n"
            f"- Final train_loss: {final_metrics['final_train_loss']:.6f}\n"
            "- Final eval F1: see `eval_metrics.csv` if periodic eval was enabled.\n"
            "- Deviations: Phase 1 code records text `max_length` separately from visual-prefix context length; "
            "visual LFQ tokens are inserted through `inputs_embeds`.\n"
            "- LFQ convention: report-generation defaults to `msb_reverse_channels`, per sub-task 1 H1/H4 audit.\n"
        )
        (self.run_dir / "phase1_summary.md").write_text(text)
