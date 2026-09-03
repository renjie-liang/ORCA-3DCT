"""
Simplified checkpoint manager for research code

Fail-fast principles:
- No try-except blocks
- No default values (except function parameters)
- Direct dict access with []
- Explicit error messages
"""
import os
import random
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import numpy as np


class CheckpointManager:
    """
    Simplified checkpoint manager

    Features:
    - Save complete training state (model, optimizer, scheduler, random seeds)
    - Keep only last N checkpoints
    - Track best checkpoint
    - Support training resumption

    Usage:
        manager = CheckpointManager(
            save_dir="./checkpoints",
            keep_last_n=3,
            best_metric="macro_auroc",
            best_metric_mode="max"
        )

        # Save
        manager.save_checkpoint(
            step=1000,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics={'macro_auroc': 0.85}
        )

        # Load
        checkpoint = manager.load_checkpoint(
            "checkpoints/step_1000.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
    """

    def __init__(
        self,
        save_dir: str,
        keep_last_n: int,
        best_metric: str,
        best_metric_mode: str
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            keep_last_n: Keep only last N checkpoints
            best_metric: Metric name for best checkpoint
            best_metric_mode: 'max' or 'min'
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.best_metric = best_metric

        # Validate mode
        if best_metric_mode not in ['max', 'min']:
            raise ValueError(f"best_metric_mode must be 'max' or 'min', got: {best_metric_mode}")
        self.best_metric_mode = best_metric_mode

        # Initialize best value
        self.best_value = -float('inf') if best_metric_mode == 'max' else float('inf')

        # Track checkpoint history (list of paths)
        self.checkpoint_history = []

    def save_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
        wandb_run_id: Optional[str] = None,
        model_state_dict: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
    ) -> str:
        """
        Save checkpoint

        Args:
            step: Training step
            model: PyTorch model
            optimizer: Optimizer
            scheduler: LR scheduler
            metrics: Validation metrics dict
            config: Config dict (optional)
            wandb_run_id: WandB run ID for resuming (optional)
            model_state_dict: If set, save this instead of model.state_dict() (e.g. LoRA-only)
            tokenizer: Tokenizer to save info for consistency validation (optional)

        Returns:
            Path to saved checkpoint
        """
        state_to_save = model_state_dict if model_state_dict is not None else model.state_dict()
        # Build checkpoint dict
        checkpoint = {
            'step': step,
            'model_state_dict': state_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': metrics,
            'random_states': self._get_random_states()
        }

        # Add optional fields
        if config is not None:
            checkpoint['config'] = config
        if wandb_run_id is not None:
            checkpoint['wandb_run_id'] = wandb_run_id

        # Save tokenizer info for consistency validation
        if tokenizer is not None:
            checkpoint['tokenizer_info'] = {
                'vocab_size': len(tokenizer),
                'eos_token': tokenizer.eos_token,
                'eos_token_id': tokenizer.eos_token_id,
                'pad_token': tokenizer.pad_token,
                'pad_token_id': tokenizer.pad_token_id,
                'special_tokens': list(tokenizer.all_special_tokens),
            }
            print(f"Saved tokenizer info: vocab_size={len(tokenizer)}, eos_token_id={tokenizer.eos_token_id}")

        # Save checkpoint
        checkpoint_path = self.save_dir / f"step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")

        # Update history
        self.checkpoint_history.append(checkpoint_path)

        # Clean old checkpoints
        self._cleanup_old_checkpoints()

        # Save best checkpoint if needed
        self._save_best_if_needed(checkpoint, metrics, step)

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        load_optimizer: bool = True,
        tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint and restore state

        Args:
            checkpoint_path: Path to checkpoint file
            model: PyTorch model
            optimizer: Optimizer
            scheduler: LR scheduler
            load_optimizer: Whether to load optimizer state
            tokenizer: Tokenizer to validate consistency (optional)

        Returns:
            Checkpoint dict
        """
        checkpoint_path = Path(checkpoint_path)

        # Check file exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Validate tokenizer consistency if both saved info and current tokenizer are available
        if tokenizer is not None and 'tokenizer_info' in checkpoint:
            saved_info = checkpoint['tokenizer_info']
            current_vocab_size = len(tokenizer)
            saved_vocab_size = saved_info['vocab_size']

            if current_vocab_size != saved_vocab_size:
                print(f"⚠ WARNING: Tokenizer vocab size mismatch!")
                print(f"  Saved: {saved_vocab_size}, Current: {current_vocab_size}")
                print(f"  This may cause issues. Ensure special tokens are added consistently.")

            if tokenizer.eos_token_id != saved_info['eos_token_id']:
                print(f"⚠ WARNING: EOS token ID mismatch!")
                print(f"  Saved: {saved_info['eos_token_id']}, Current: {tokenizer.eos_token_id}")

            print(f"✓ Tokenizer info: vocab_size={current_vocab_size}, eos_token_id={tokenizer.eos_token_id}")

        # Restore model (fail fast if keys mismatch)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model state restored")

        # Restore optimizer
        if load_optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✓ Optimizer state restored")
        else:
            print(f"⚠ Optimizer state NOT loaded (load_optimizer=False)")

        # Restore scheduler
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✓ Scheduler state restored")

        # Restore random states
        self._set_random_states(checkpoint['random_states'])
        print(f"✓ Random states restored")

        # Print info
        print(f"Resumed from step {checkpoint['step']}")
        if 'metrics' in checkpoint:
            print(f"Checkpoint metrics: {checkpoint['metrics']}")

        return checkpoint

    def _save_best_if_needed(
        self,
        checkpoint: Dict,
        metrics: Dict[str, float],
        step: int
    ):
        """Save best checkpoint if metric improved"""
        # Skip if no best_metric configured
        if self.best_metric is None:
            return

        # Skip if best_metric not in metrics (e.g., regular checkpoint without validation)
        if self.best_metric not in metrics:
            return

        # Get metric value
        metric_value = metrics[self.best_metric]

        # Check if better
        is_better = False
        if self.best_metric_mode == 'max':
            is_better = metric_value > self.best_value
        else:
            is_better = metric_value < self.best_value

        if is_better:
            self.best_value = metric_value
            best_path = self.save_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"★ New best {self.best_metric}: {metric_value:.4f} (step {step})")

    def _cleanup_old_checkpoints(self):
        """Delete old checkpoints, keep only last N"""
        if len(self.checkpoint_history) <= self.keep_last_n:
            return

        # Delete oldest checkpoints
        to_delete = self.checkpoint_history[:-self.keep_last_n]

        for path in to_delete:
            if path.exists():
                path.unlink()
                print(f"Deleted old checkpoint: {path.name}")

        # Update history
        self.checkpoint_history = self.checkpoint_history[-self.keep_last_n:]

    def _get_random_states(self) -> Dict[str, Any]:
        """Get all random states for reproducibility"""
        states = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
        }

        # CUDA random states
        if torch.cuda.is_available():
            states['cuda'] = torch.cuda.get_rng_state_all()

        return states

    def _set_random_states(self, states: Dict[str, Any]):
        """Restore all random states"""
        random.setstate(states['python'])
        np.random.set_state(states['numpy'])
        torch.set_rng_state(states['torch'])

        if torch.cuda.is_available() and 'cuda' in states:
            torch.cuda.set_rng_state_all(states['cuda'])
