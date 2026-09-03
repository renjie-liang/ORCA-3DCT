"""
Simplified logging system for research code

Supports:
- Console logging
- WandB logging
- Multiple loggers simultaneously

Fail-fast principles:
- No try-except for logging operations
- Direct dict access
- Explicit errors
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

# Import format_eta for time formatting
from shared.utils.profiling import format_eta


# ============================================================================
# Base Logger Interface
# ============================================================================

class BaseLogger(ABC):
    """Abstract base class for loggers"""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "", max_steps: Optional[int] = None):
        """
        Log metrics

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
            max_steps: Optional total steps for progress display
        """
        pass

    @abstractmethod
    def log_config(self, config: Dict[str, Any]):
        """Log configuration"""
        pass

    @abstractmethod
    def info(self, message: str):
        """Log informational message"""
        pass

    @abstractmethod
    def finish(self):
        """Finish logging (cleanup)"""
        pass


# ============================================================================
# Dummy Logger (for non-master ranks)
# ============================================================================

class DummyLogger(BaseLogger):
    """Logger that does nothing (for non-master ranks in distributed training)"""

    def info(self, message: str):
        """Do nothing"""
        pass

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "", max_steps: Optional[int] = None):
        """Do nothing"""
        pass

    def log_config(self, config: Dict[str, Any]):
        """Do nothing"""
        pass

    def finish(self):
        """Do nothing"""
        pass


# ============================================================================
# File Logger
# ============================================================================

class FileLogger(BaseLogger):
    """Logger that writes to a file"""

    def __init__(self, log_file: str, print_timestamp: bool = True):
        """
        Args:
            log_file: Path to log file
            print_timestamp: Whether to include timestamps
        """
        from pathlib import Path
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.print_timestamp = print_timestamp

        # Open file in append mode with line buffering
        self.file_handle = open(self.log_file, 'a', buffering=1)

    def info(self, message: str):
        """Write message to file"""
        timestamp = ""
        if self.print_timestamp:
            timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        self.file_handle.write(f"{timestamp}{message}\n")

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "", max_steps: Optional[int] = None):
        """Write metrics to file (same format as console)"""
        timestamp = ""
        if self.print_timestamp:
            timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "

        # Format step with progress
        if max_steps:
            step_str = f"Step {step}/{max_steps}"
        else:
            step_str = f"Step {step}"

        # Smart formatting for different metric types
        def format_value(k: str, v: float) -> str:
            # Learning rate: scientific notation if very small
            if k in ['lr', 'learning_rate'] and v < 0.001:
                return f"{prefix}{k}: {v:.2e}"
            # Time metrics: readable format
            elif k == 'step_time':
                return f"time: {v:.2f}s/step"
            elif k == 'eta_hours':
                eta_seconds = v * 3600
                return f"ETA: {format_eta(eta_seconds)}"
            elif k == 'elapsed_hours':
                elapsed_seconds = v * 3600
                return f"elapsed: {format_eta(elapsed_seconds)}"
            # Regular metrics: 4 decimal places
            else:
                return f"{prefix}{k}: {v:.4f}"

        metrics_str = " | ".join([format_value(k, v) for k, v in metrics.items()])
        self.file_handle.write(f"{timestamp}{step_str} | {metrics_str}\n")

    def log_config(self, config: Dict[str, Any]):
        """Write config to file"""
        self.file_handle.write("=" * 80 + "\n")
        self.file_handle.write("Configuration:\n")
        self.file_handle.write("=" * 80 + "\n")
        self._write_dict(config, indent=0)
        self.file_handle.write("=" * 80 + "\n")

    def _write_dict(self, d: Dict, indent: int):
        """Recursively write dict"""
        for key, value in d.items():
            if isinstance(value, dict):
                self.file_handle.write("  " * indent + f"{key}:\n")
                self._write_dict(value, indent + 1)
            else:
                self.file_handle.write("  " * indent + f"{key}: {value}\n")

    def finish(self):
        """Close file handle"""
        if hasattr(self, 'file_handle') and self.file_handle:
            self.file_handle.close()


# ============================================================================
# Console Logger
# ============================================================================

class ConsoleLogger(BaseLogger):
    """Simple console logger with timestamps"""

    def __init__(self, print_timestamp: bool = True):
        """
        Args:
            print_timestamp: Whether to print timestamps
        """
        self.print_timestamp = print_timestamp

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "", max_steps: Optional[int] = None):
        """
        Print metrics to console with intelligent formatting

        Args:
            metrics: Dict of metric name -> value
            step: Current step number
            prefix: Prefix to add to metric names
            max_steps: Optional total steps for progress display
        """
        timestamp = ""
        if self.print_timestamp:
            timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "

        # Format step with progress
        if max_steps:
            step_str = f"Step {step}/{max_steps}"
        else:
            step_str = f"Step {step}"

        # Smart formatting for different metric types
        def format_value(k: str, v: float) -> str:
            # Learning rate: scientific notation if very small
            if k in ['lr', 'learning_rate'] and v < 0.001:
                return f"{prefix}{k}: {v:.2e}"
            # Time metrics: readable format
            elif k == 'step_time':
                return f"time: {v:.2f}s/step"
            elif k == 'eta_hours':
                eta_seconds = v * 3600
                return f"ETA: {format_eta(eta_seconds)}"
            elif k == 'elapsed_hours':
                elapsed_seconds = v * 3600
                return f"elapsed: {format_eta(elapsed_seconds)}"
            # Regular metrics: 4 decimal places
            else:
                return f"{prefix}{k}: {v:.4f}"

        metrics_str = " | ".join([format_value(k, v) for k, v in metrics.items()])
        print(f"{timestamp}{step_str} | {metrics_str}")

    def log_config(self, config: Dict[str, Any]):
        """Print config to console"""
        print("=" * 80)
        print("Configuration:")
        print("=" * 80)
        self._print_dict(config, indent=0)
        print("=" * 80)

    def _print_dict(self, d: Dict, indent: int):
        """Recursively print dict"""
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self._print_dict(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")

    def info(self, message: str):
        """Print message to console"""
        timestamp = ""
        if self.print_timestamp:
            timestamp = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
        print(f"{timestamp}{message}")

    def finish(self):
        """Nothing to cleanup for console logger"""
        pass


# ============================================================================
# WandB Logger
# ============================================================================

class WandBLogger(BaseLogger):
    """WandB logger with resume support"""

    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        entity: Optional[str] = None,
        group: Optional[str] = None,
        resume_id: Optional[str] = None
    ):
        """
        Args:
            project: WandB project name
            name: Run name
            config: Config dict to log
            entity: WandB entity (username/team)
            group: WandB group
            resume_id: WandB run ID to resume (for continuing interrupted runs)
        """
        import wandb

        # Resume or create new run
        if resume_id is not None:
            print(f"Resuming WandB run: {resume_id}")
            self.run = wandb.init(
                project=project,
                entity=entity,
                group=group,
                id=resume_id,
                resume="must",
                config=config
            )
        else:
            print(f"Creating new WandB run: {name}")
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=name,
                group=group,
                config=config
            )

        print(f"WandB run ID: {self.run.id}")
        print(f"WandB run URL: {self.run.url}")

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "", max_steps: Optional[int] = None):
        """Log metrics to WandB"""
        import wandb

        # Add prefix to all metrics
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}

        wandb.log(metrics, step=step)

    def log_config(self, config: Dict[str, Any]):
        """Update WandB config"""
        import wandb
        wandb.config.update(config)

    def info(self, message: str):
        """Log message to WandB (no-op, WandB is for metrics)"""
        pass

    def finish(self):
        """Finish WandB run"""
        import wandb
        wandb.finish()
        print("WandB run finished")

    def get_run_id(self) -> str:
        """Get WandB run ID for resuming"""
        return self.run.id


# ============================================================================
# Multi Logger
# ============================================================================

class MultiLogger(BaseLogger):
    """Combine multiple loggers"""

    def __init__(self, loggers: List[BaseLogger]):
        """
        Args:
            loggers: List of logger instances
        """
        if len(loggers) == 0:
            raise ValueError("Must provide at least one logger")

        self.loggers = loggers

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "", max_steps: Optional[int] = None):
        """Log to all loggers"""
        for logger in self.loggers:
            logger.log_metrics(metrics, step, prefix, max_steps)

    def log_config(self, config: Dict[str, Any]):
        """Log config to all loggers"""
        for logger in self.loggers:
            logger.log_config(config)

    def info(self, message: str):
        """Log to all loggers"""
        for logger in self.loggers:
            logger.info(message)

    def finish(self):
        """Finish all loggers"""
        for logger in self.loggers:
            logger.finish()

    def get_run_id(self) -> Optional[str]:
        """Get WandB run ID if available"""
        for logger in self.loggers:
            if hasattr(logger, 'get_run_id'):
                return logger.get_run_id()
        return None


# ============================================================================
# Logger Factory
# ============================================================================

def create_logger(
    config: Dict[str, Any],
    resume_wandb_id: Optional[str] = None,
    rank: int = 0
) -> BaseLogger:
    """
    Create logger based on config and rank

    Args:
        config: Config dict with 'logging' section
        resume_wandb_id: WandB run ID to resume (optional)
        rank: Process rank (only rank 0 creates real loggers, others get DummyLogger)

    Returns:
        Logger instance (DummyLogger for non-master ranks, MultiLogger for rank 0)

    Config format:
        logging:
          use_wandb: true
          wandb:
            project: "ct-clip"
            entity: "my-team"  # optional
            group: "experiment-1"  # optional

    Usage:
        config = load_config("config.yaml")
        logger = create_logger(config, rank=rank)

        logger.info("Starting training")
        logger.log_metrics({'loss': 0.5}, step=100, prefix='train/')
        logger.finish()
    """
    # Non-master ranks get DummyLogger
    if rank != 0:
        return DummyLogger()

    logging_config = config['logging']
    loggers = []

    # Console logger (always enabled)
    console_logger = ConsoleLogger(print_timestamp=True)
    loggers.append(console_logger)
    print("✓ Console logger enabled")

    # File logger (always enabled)
    from pathlib import Path
    output_dir = config['paths']['output_dir']
    log_file = Path(output_dir) / "train.log"  # Log at experiment root
    file_logger = FileLogger(str(log_file), print_timestamp=True)
    loggers.append(file_logger)
    print(f"✓ File logger enabled: {log_file}")

    # WandB logger (based on config)
    if logging_config.get('use_wandb', False):
        wandb_config = logging_config['wandb']
        # Use run_name if available (includes timestamp), otherwise use experiment name
        run_name = config['experiment'].get('run_name', config['experiment']['name'])
        wandb_logger = WandBLogger(
            project=wandb_config['project'],
            name=run_name,
            config=config,
            entity=wandb_config.get('entity'),
            group=wandb_config.get('group'),
            resume_id=resume_wandb_id
        )
        loggers.append(wandb_logger)
        print("✓ WandB logger enabled")

    # Return MultiLogger (always has at least console + file)
    return MultiLogger(loggers)


# ============================================================================
# Module-level Logger (for shared modules)
# ============================================================================

_global_logger: Optional[BaseLogger] = None


def get_module_logger() -> BaseLogger:
    """
    Get global logger for shared modules

    Returns DummyLogger if not initialized (safe for imports).
    This allows modules to import and use logger before main script initializes it.

    Returns:
        BaseLogger: Global logger instance (DummyLogger if not yet initialized)

    Usage:
        # In any shared module
        from shared.utils.logger import get_module_logger

        logger = get_module_logger()
        logger.info("Loading dataset...")
    """
    global _global_logger
    if _global_logger is None:
        # Logger not yet initialized, return DummyLogger
        # This ensures modules can be imported before logger setup
        return DummyLogger()
    return _global_logger


def set_module_logger(logger: BaseLogger):
    """
    Set global logger for shared modules

    Should be called once in main training script after creating logger.

    Args:
        logger: Logger instance to use globally

    Usage:
        # In main training script
        from shared.utils.logger import create_logger, set_module_logger

        rank = setup_distributed()
        config = load_config(args.config)
        logger = create_logger(config, rank=rank)
        set_module_logger(logger)  # Make logger available to all modules
    """
    global _global_logger
    _global_logger = logger
