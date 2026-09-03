"""
Config loading with fail-fast validation

Features:
- Load YAML config files
- Config inheritance (_base_ field)
- Environment variable expansion
- Strict validation (fail fast)

Fail-fast principles:
- No try-except
- Direct dict access with []
- Explicit validation errors
"""
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


def expand_env_vars(config: Any) -> Any:
    """
    Recursively expand environment variables in config

    Format: ${VAR_NAME} or ${VAR_NAME:default_value}

    Args:
        config: Config value (dict, list, str, or other)

    Returns:
        Config with expanded environment variables
    """
    if isinstance(config, dict):
        return {k: expand_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [expand_env_vars(item) for item in config]
    elif isinstance(config, str):
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

        def replace_env(match):
            var_name = match.group(1)
            default_value = match.group(2)
            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ValueError(
                    f"Environment variable '{var_name}' not set and no default provided"
                )

        return re.sub(pattern, replace_env, config)
    else:
        return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two config dicts

    Args:
        base: Base config
        override: Override config

    Returns:
        Merged config
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def load_config(config_path: str) -> Dict:
    """
    Load config file with inheritance and validation

    Supports:
    - YAML format
    - Config inheritance via _base_ field
    - Environment variable expansion ${VAR}
    - Strict validation

    Args:
        config_path: Path to config YAML file

    Returns:
        Complete validated config dict

    Example:
        config = load_config("configs/train.yaml")
        print(config['experiment']['name'])
    """
    config_path = Path(config_path)

    # Check file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    # Handle config inheritance
    if '_base_' in config:
        base_path = config.pop('_base_')

        # Resolve relative path
        if not Path(base_path).is_absolute():
            base_path = config_path.parent / base_path

        # Load base config recursively
        base_config = load_config(str(base_path))

        # Merge base and current
        config = merge_configs(base_config, config)

    # Expand environment variables
    config = expand_env_vars(config)

    # Validate
    validate_config(config)

    return config


def validate_config(config: Dict):
    """
    Validate config structure (fail fast)

    Args:
        config: Config dict

    Raises:
        ValueError: If any required field is missing or invalid
    """
    # Required top-level sections (skip training for inference configs)
    exp_name = config.get('experiment', {}).get('name', '').lower()
    if 'infer' in exp_name:
        required_sections = ['experiment', 'device', 'paths', 'data', 'model']
    else:
        required_sections = ['experiment', 'device', 'paths', 'data', 'model', 'training']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Validate experiment
    exp = config['experiment']
    if 'name' not in exp:
        raise ValueError("experiment.name is required")
    if 'seed' not in exp:
        raise ValueError("experiment.seed is required")

    # Validate device
    device = config['device']
    if 'type' not in device:
        raise ValueError("device.type is required")

    # Validate paths
    paths = config['paths']
    # Paths can vary by stage, so just check it exists

    # Validate data
    data = config['data']
    if 'batch_size' not in data:
        raise ValueError("data.batch_size is required")

    # Validate training (skip for inference configs)
    if 'training' in config:
        training = config['training']
        if 'max_steps' not in training:
            raise ValueError("training.max_steps is required")
        if training['max_steps'] <= 0:
            raise ValueError(f"training.max_steps must be positive, got {training['max_steps']}")

        # Validate optimizer
        if 'optimizer' not in training:
            raise ValueError("training.optimizer is required")
        opt = training['optimizer']
        if 'type' not in opt:
            raise ValueError("training.optimizer.type is required")
        if 'lr' not in opt:
            raise ValueError("training.optimizer.lr is required")

        # Validate scheduler
        if 'scheduler' not in training:
            raise ValueError("training.scheduler is required")
        sched = training['scheduler']
        if 'type' not in sched:
            raise ValueError("training.scheduler.type is required")

    # Validate checkpoint (if present)
    if 'checkpoint' in config:
        checkpoint = config['checkpoint']
        if 'keep_last_n' not in checkpoint:
            raise ValueError("checkpoint.keep_last_n is required")
        # best_metric and best_metric_mode are optional (only needed when validation is enabled)

    # Validate logging (if present)
    if 'logging' in config:
        logging = config['logging']
        # Must have at least one logger enabled
        has_logger = logging.get('use_console', False) or logging.get('use_wandb', False)
        if not has_logger:
            raise ValueError("At least one logger must be enabled (use_console or use_wandb)")
