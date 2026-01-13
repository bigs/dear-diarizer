"""
Wandb logging for WavLeJEPA training.

Basic integration: metrics, loss curves, config tracking.
"""

from typing import Optional, Any

import wandb

from .config import LoggingConfig, TrainingConfig


class WandBLogger:
    """
    Wandb logger for training metrics.

    Handles:
    - Run initialization with config tracking
    - Step-based metric logging
    - Evaluation metric logging
    - Run resumption
    """

    def __init__(
        self,
        config: LoggingConfig,
        training_config: TrainingConfig,
        run_name: Optional[str] = None,
        resume_run_id: Optional[str] = None,
    ):
        """
        Initialize wandb run.

        Args:
            config: Logging configuration
            training_config: Full training config (logged to wandb)
            run_name: Optional run name
            resume_run_id: Optional run ID for resuming
        """
        self.config = config
        self._step_buffer: dict[str, list] = {}

        # Initialize wandb
        wandb.init(
            project=config.project,
            entity=config.entity,
            config=training_config.to_dict(),
            name=run_name,
            resume="allow" if resume_run_id else None,
            id=resume_run_id,
        )

        # Store run ID for potential resume
        self.run_id = wandb.run.id if wandb.run else None

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        """
        Log training metrics for a step.

        Only logs every log_every_n_steps.

        Args:
            step: Current training step
            metrics: Dict of metric name -> value (can be JAX arrays)
        """
        if step % self.config.log_every_n_steps != 0:
            return

        # Convert JAX arrays to Python scalars
        logged = {}
        for k, v in metrics.items():
            try:
                logged[k] = float(v)
            except (TypeError, ValueError):
                logged[k] = v

        wandb.log(logged, step=step)

    def log_eval(
        self,
        step: int,
        metrics: dict[str, Any],
        prefix: str = "eval/",
    ) -> None:
        """
        Log evaluation metrics.

        Args:
            step: Current training step
            metrics: Dict of metric name -> value
            prefix: Prefix for metric names (default "eval/")
        """
        logged = {}
        for k, v in metrics.items():
            try:
                logged[f"{prefix}{k}"] = float(v)
            except (TypeError, ValueError):
                logged[f"{prefix}{k}"] = v

        wandb.log(logged, step=step)

    def log_lr(self, step: int, lr: float) -> None:
        """Log learning rate."""
        wandb.log({"lr": lr}, step=step)

    def log_config_update(self, key: str, value: Any) -> None:
        """Update a config value (useful for dynamic configs)."""
        if wandb.run:
            wandb.run.config.update({key: value}, allow_val_change=True)

    def finish(self) -> None:
        """Finish the wandb run."""
        wandb.finish()

    @property
    def is_active(self) -> bool:
        """Check if wandb run is active."""
        return wandb.run is not None


class NoOpLogger:
    """No-op logger for when wandb is disabled."""

    def __init__(self, *args, **kwargs):
        self.run_id = None

    def log_step(self, step: int, metrics: dict[str, Any]) -> None:
        pass

    def log_eval(
        self, step: int, metrics: dict[str, Any], prefix: str = "eval/"
    ) -> None:
        pass

    def log_lr(self, step: int, lr: float) -> None:
        pass

    def log_config_update(self, key: str, value: Any) -> None:
        pass

    def finish(self) -> None:
        pass

    @property
    def is_active(self) -> bool:
        return False
