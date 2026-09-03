"""
Path utilities for checkpoint-based output directory management.

This module provides functions to parse checkpoint paths and generate
corresponding output directories for inference and metrics.
"""

import re
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from copy import deepcopy


def parse_checkpoint_path(checkpoint_path: str) -> Tuple[str, int, str]:
    """
    Parse checkpoint path to extract base path, step number, and experiment name.
    
    Args:
        checkpoint_path: Full path to checkpoint file
        
    Returns:
        Tuple of (base_path, step_number, experiment_name)
        
    Raises:
        ValueError: If checkpoint path format is invalid
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Check if file exists and has correct extension
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
    
    if checkpoint_path.suffix != '.pt':
        raise ValueError(f"Checkpoint file must have .pt extension, got: {checkpoint_path.suffix}")
    
    # Extract step number from filename
    step_match = re.match(r'step_(\d+)', checkpoint_path.stem)
    if not step_match:
        raise ValueError(
            f"Invalid checkpoint filename format. Expected: step_<number>.pt, "
            f"got: {checkpoint_path.name}"
        )
    
    step_number = int(step_match.group(1))
    
    # Get base path (directory before checkpoints/)
    if 'checkpoints' not in checkpoint_path.parts:
        raise ValueError(
            f"Checkpoint path must contain 'checkpoints' directory. "
            f"Got: {checkpoint_path}"
        )
    
    checkpoints_idx = checkpoint_path.parts.index('checkpoints')
    base_path_parts = checkpoint_path.parts[:checkpoints_idx]
    base_path = Path(*base_path_parts)
    
    # Extract experiment name (last directory before checkpoints)
    experiment_name = checkpoint_path.parts[checkpoints_idx - 1]
    
    return str(base_path), step_number, experiment_name


def generate_output_path(base_path: str, step_number: int, file_type: str = "inference") -> Path:
    """
    Generate output path based on base path, step number, and file type.
    
    Args:
        base_path: Base directory path
        step_number: Step number from checkpoint
        file_type: Type of output ("inference" or "metrics")
        
    Returns:
        Path object for output file
        
    Raises:
        ValueError: If file_type is invalid
    """
    base_path = Path(base_path)
    
    if file_type == "inference":
        output_dir = base_path / "infer"
        filename = f"infer_step_{step_number}.jsonl"
    elif file_type == "metrics":
        output_dir = base_path / "metrics"
        filename = f"metrics_step_{step_number}.json"
    else:
        raise ValueError(f"Invalid file_type: {file_type}. Must be 'inference' or 'metrics'")
    
    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir / filename


def get_checkpoint_output_paths(checkpoint_path: str) -> Tuple[Path, Path]:
    """
    Get both inference and metrics output paths for a checkpoint.
    
    Args:
        checkpoint_path: Full path to checkpoint file
        
    Returns:
        Tuple of (inference_output_path, metrics_output_path)
    """
    base_path, step_number, _ = parse_checkpoint_path(checkpoint_path)
    
    inference_path = generate_output_path(base_path, step_number, "inference")
    metrics_path = generate_output_path(base_path, step_number, "metrics")
    
    return inference_path, metrics_path


def validate_checkpoint_path(checkpoint_path: str) -> bool:
    """
    Validate if checkpoint path has correct format.
    
    Args:
        checkpoint_path: Path to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parse_checkpoint_path(checkpoint_path)
        return True
    except ValueError:
        return False


def parse_predictions_path(predictions_path: str) -> Tuple[str, int]:
    """
    Parse predictions path to extract base path and step number.
    
    Args:
        predictions_path: Full path to predictions JSONL file
        
    Returns:
        Tuple of (base_path, step_number)
        
    Raises:
        ValueError: If predictions path format is invalid
    """
    predictions_path = Path(predictions_path)
    
    # Check if file exists and has correct extension
    if not predictions_path.exists():
        raise ValueError(f"Predictions file does not exist: {predictions_path}")
    
    if predictions_path.suffix not in ['.jsonl', '.json']:
        raise ValueError(f"Predictions file must have .jsonl or .json extension, got: {predictions_path.suffix}")
    

    # Extract step number from filename (any format containing step_<number>)
    step_match = re.search(r'step_(\d+)', predictions_path.stem)
    if not step_match:
        raise ValueError(
            f"Invalid predictions filename format. Expected filename containing step_<number>, "
            f"got: {predictions_path.name}"
        )

    step_number = int(step_match.group(1))
    
    # Base path = parent of the directory containing the predictions file
    base_path = predictions_path.parent.parent
    
    return str(base_path), step_number


def generate_metrics_path_from_predictions(predictions_path: str) -> Path:
    """
    Generate metrics output path based on predictions file path.
    
    Args:
        predictions_path: Full path to predictions file
        
    Returns:
        Path object for metrics output file
        
    Raises:
        ValueError: If predictions path format is invalid
    """
    base_path, step_number = parse_predictions_path(predictions_path)
    
    # Map infer dir name to metrics dir name: infer/ → metrics/, infer_oracle_rag/ → metrics_oracle_rag/
    infer_dir_name = Path(predictions_path).parent.name
    if infer_dir_name == "infer":
        metrics_dir_name = "metrics"
    elif infer_dir_name.startswith("infer_"):
        metrics_dir_name = "metrics_" + infer_dir_name[len("infer_"):]
    else:
        metrics_dir_name = "metrics"
    
    metrics_dir = Path(base_path) / metrics_dir_name
    filename = f"metrics_step_{step_number}.json"
    
    # Create directory if it doesn't exist
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    return metrics_dir / filename


# Default inference configuration
DEFAULT_INFERENCE_CONFIG = {
    "generation": {
        "temperature": 0.3,
        "repetition_penalty": 1.2,
        "top_p": 0.9,
        "top_k": 50
    },
    "data": {
        "batch_size": 1,
        "sample_types": ["image", "paper"],
        "max_samples": None
    },
    "validation": {
        "max_new_tokens_by_task": {
            "report_generation": 800,
            "long_answer": 400,
            "short_answer": 200,
            "multiple_choice": 50
        }
    }
}


def load_config_from_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load training config from checkpoint directory.
    
    Args:
        checkpoint_path: Full path to checkpoint file
        
    Returns:
        Training configuration dictionary
        
    Raises:
        ValueError: If config file cannot be found or loaded
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Get base path (directory before checkpoints/)
    if 'checkpoints' not in checkpoint_path.parts:
        raise ValueError(
            f"Checkpoint path must contain 'checkpoints' directory. "
            f"Got: {checkpoint_path}"
        )
    
    checkpoints_idx = checkpoint_path.parts.index('checkpoints')
    base_path_parts = checkpoint_path.parts[:checkpoints_idx]
    base_path = Path(*base_path_parts)
    
    # Look for config.json in the base directory
    config_path = base_path / "config.json"
    
    if not config_path.exists():
        raise ValueError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise ValueError(f"Failed to load config from {config_path}: {e}")


def apply_default_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default inference values to configuration.
    
    Args:
        config: Base configuration (from training or empty)
        
    Returns:
        Configuration with all required fields filled
    """
    config = deepcopy(config)
    
    # Apply generation defaults
    config.setdefault("generation", {})
    for key, value in DEFAULT_INFERENCE_CONFIG["generation"].items():
        config["generation"].setdefault(key, value)
    
    # Apply data defaults
    config.setdefault("data", {})
    for key, value in DEFAULT_INFERENCE_CONFIG["data"].items():
        config["data"].setdefault(key, value)
    
    # Apply validation defaults
    config.setdefault("validation", {})
    if "max_new_tokens_by_task" not in config["validation"]:
        config["validation"]["max_new_tokens_by_task"] = DEFAULT_INFERENCE_CONFIG["validation"]["max_new_tokens_by_task"].copy()
    
    return config


def apply_cli_overrides(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Apply command line argument overrides to configuration.
    
    Args:
        config: Base configuration
        args: Parsed command line arguments
        
    Returns:
        Configuration with CLI overrides applied
    """
    config = deepcopy(config)
    
    # Generation overrides
    if hasattr(args, 'temperature') and args.temperature is not None:
        config["generation"]["temperature"] = args.temperature
    if hasattr(args, 'repetition_penalty') and args.repetition_penalty is not None:
        config["generation"]["repetition_penalty"] = args.repetition_penalty
    if hasattr(args, 'top_p') and args.top_p is not None:
        config["generation"]["top_p"] = args.top_p
    if hasattr(args, 'top_k') and args.top_k is not None:
        config["generation"]["top_k"] = args.top_k
    
    # Data overrides
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if hasattr(args, 'max_samples') and args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if hasattr(args, 'sample_types') and args.sample_types is not None:
        import json
        try:
            config["data"]["sample_types"] = json.loads(args.sample_types)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON for sample_types: {args.sample_types}")
    
    # Validation overrides
    if hasattr(args, 'max_new_tokens_report_generation') and args.max_new_tokens_report_generation is not None:
        config.setdefault("validation", {}).setdefault("max_new_tokens_by_task", {})
        config["validation"]["max_new_tokens_by_task"]["report_generation"] = args.max_new_tokens_report_generation
    if hasattr(args, 'max_new_tokens_long_answer') and args.max_new_tokens_long_answer is not None:
        config.setdefault("validation", {}).setdefault("max_new_tokens_by_task", {})
        config["validation"]["max_new_tokens_by_task"]["long_answer"] = args.max_new_tokens_long_answer
    if hasattr(args, 'max_new_tokens_short_answer') and args.max_new_tokens_short_answer is not None:
        config.setdefault("validation", {}).setdefault("max_new_tokens_by_task", {})
        config["validation"]["max_new_tokens_by_task"]["short_answer"] = args.max_new_tokens_short_answer
    if hasattr(args, 'max_new_tokens_multiple_choice') and args.max_new_tokens_multiple_choice is not None:
        config.setdefault("validation", {}).setdefault("max_new_tokens_by_task", {})
        config["validation"]["max_new_tokens_by_task"]["multiple_choice"] = args.max_new_tokens_multiple_choice
    
    return config


def build_inference_config(checkpoint_path: str, args) -> Dict[str, Any]:
    """
    Build complete inference configuration from checkpoint and CLI args.
    
    Args:
        checkpoint_path: Path to checkpoint file
        args: Parsed command line arguments
        
    Returns:
        Complete inference configuration
    """
    # 1. Load base config from checkpoint
    base_config = load_config_from_checkpoint(checkpoint_path)
    
    # 2. Apply default values
    config = apply_default_values(base_config)
    
    # 3. Apply CLI overrides
    config = apply_cli_overrides(config, args)
    
    return config
