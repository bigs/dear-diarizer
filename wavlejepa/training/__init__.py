"""
WavLeJEPA training module.

Provides training infrastructure:
- Configuration management
- Training state and optimizer
- Train/eval step functions with multi-GPU support
- Orbax checkpointing
- Wandb logging
"""

from .config import (
    TrainingConfig,
    OptimizerConfig,
    LossConfig,
    CheckpointConfig,
    LoggingConfig,
    DataConfig,
)
from .state import TrainState, create_train_state, create_optimizer, get_lr_at_step
from .step import (
    init_sharding,
    shard_batch,
    shard_state,
    make_train_step,
    make_eval_step,
)
from .checkpoint import WavLeJEPACheckpointer
from .logging import WandBLogger, NoOpLogger

__all__ = [
    # Config
    "TrainingConfig",
    "OptimizerConfig",
    "LossConfig",
    "CheckpointConfig",
    "LoggingConfig",
    "DataConfig",
    # State
    "TrainState",
    "create_train_state",
    "create_optimizer",
    "get_lr_at_step",
    # Step
    "init_sharding",
    "shard_batch",
    "shard_state",
    "make_train_step",
    "make_eval_step",
    # Checkpoint
    "WavLeJEPACheckpointer",
    # Logging
    "WandBLogger",
    "NoOpLogger",
]
