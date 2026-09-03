"""Periodic 100-volume validation callback for Phase 1 training."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


class PeriodicEvalRunner:
    """Generate BTB3D-format predictions, convert, then call `eval_fast.py`."""

    def __init__(
        self,
        eval_loader: DataLoader[dict[str, Any]],
        eval_fast_path: Path,
        converter_path: Path,
        reports_csv: Path,
        labels_csv: Path,
        radbert_checkpoint: Path,
        device: str,
        max_new_tokens: int,
        repetition_penalty: float,
    ) -> None:
        self.eval_loader = eval_loader
        self.eval_fast_path = eval_fast_path
        self.converter_path = converter_path
        self.reports_csv = reports_csv
        self.labels_csv = labels_csv
        self.radbert_checkpoint = radbert_checkpoint
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty

    def run(self, step: int, model: torch.nn.Module, tokenizer: Any, run_dir: Path) -> dict[str, Any]:
        prediction_path = run_dir / "predictions" / f"step_{step}.jsonl"
        metrics_input_path = run_dir / "evaluations" / f"step_{step}_metrics_input.jsonl"
        metrics_path = run_dir / "evaluations" / f"step_{step}_metrics.json"

        was_training = model.training
        model.eval()
        num_predictions = 0
        with prediction_path.open("w") as f:
            for batch in self.eval_loader:
                batch = {
                    key: value.to(model.device) if isinstance(value, torch.Tensor) and hasattr(model, "device") else value
                    for key, value in batch.items()
                }
                if not hasattr(model, "device"):
                    device = next(model.parameters()).device
                    for key, value in list(batch.items()):
                        if isinstance(value, torch.Tensor):
                            batch[key] = value.to(device)
                output_ids = model.generate(
                    input_ids=batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    image_features=batch["image_features"],
                    generation_kwargs={
                        "do_sample": False,
                        "max_new_tokens": self.max_new_tokens,
                        "repetition_penalty": self.repetition_penalty,
                        "use_cache": True,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    },
                )
                decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                for metadata, answer in zip(batch["metadata"], decoded):
                    record = {
                        "image": metadata["image_name"],
                        "conversations_out": [
                            {
                                "id": metadata["sample_id"],
                                "question": metadata["question"],
                                "answer": answer.strip(),
                            }
                        ],
                    }
                    f.write(json.dumps(record) + "\n")
                    num_predictions += 1
        if was_training:
            model.train()
        if num_predictions == 0:
            raise RuntimeError(f"eval at step {step} produced zero predictions")

        subprocess.run(
            [
                "python",
                str(self.converter_path),
                "--btb3d-jsonl",
                str(prediction_path),
                "--reports-csv",
                str(self.reports_csv),
                "--out",
                str(metrics_input_path),
            ],
            check=True,
        )
        subprocess.run(
            [
                "python",
                str(self.eval_fast_path),
                "--pred",
                str(metrics_input_path),
                "--labels-csv",
                str(self.labels_csv),
                "--radbert",
                str(self.radbert_checkpoint),
                "--device",
                self.device,
                "--out",
                str(metrics_path),
            ],
            check=True,
        )
        return json.loads(metrics_path.read_text())
